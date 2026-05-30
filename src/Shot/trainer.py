from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SHOT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = SHOT_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Pass.data_utils import REPO_ROOT

from .features import ShotFeatureBuilder
from .models import ShotEPVNet


@dataclass
class ShotTrainingConfig:
    hidden_dim: int = 10
    epochs: int = 40
    device: str = "cpu"
    learning_rates: Tuple[float, ...] = (1e-3, 1e-4, 1e-5, 1e-6)
    batch_sizes: Tuple[int, ...] = (16, 32)
    weight_decay: float = 0.0
    early_stopping_delta: float = 1e-5
    patience: int = 6


class ShotTrainer:
    def __init__(self, train_config: Optional[ShotTrainingConfig] = None):
        self.train_config = train_config or ShotTrainingConfig(
            device=("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = torch.device(self.train_config.device)
        self.feature_builder = ShotFeatureBuilder()

    @staticmethod
    def _ensure_non_empty_split(df: pd.DataFrame, split_name: str) -> None:
        if df.empty:
            raise ValueError(f"Split '{split_name}' is empty. Check shot dataset split or data availability.")

    @staticmethod
    def _as_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).view(-1),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def _fit_one(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        learning_rate: float,
        batch_size: int,
    ) -> Tuple[ShotEPVNet, float]:
        model = ShotEPVNet(n_features=x_train.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(learning_rate),
            betas=(0.9, 0.999),
            weight_decay=float(self.train_config.weight_decay),
        )
        criterion = nn.MSELoss()

        train_loader = self._as_loader(x_train, y_train, int(batch_size), shuffle=True)
        val_loader = self._as_loader(x_val, y_val, int(batch_size), shuffle=False)

        best_state = None
        best_loss = float("inf")
        patience_left = int(self.train_config.patience)

        for _ in range(int(self.train_config.epochs)):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses: List[float] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    val_losses.append(float(criterion(model(xb), yb).item()))

            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            if (best_loss - val_loss) > float(self.train_config.early_stopping_delta):
                best_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience_left = int(self.train_config.patience)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, best_loss

    def run(
        self,
        dataset: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, object]:
        if dataset.empty:
            raise ValueError("Shot dataset is empty.")
        if "split" not in dataset.columns:
            raise ValueError("Shot dataset must contain a 'split' column.")

        required_columns = list(self.feature_builder.feature_columns) + ["reward_norm"]
        clean = dataset.dropna(subset=required_columns).copy()
        clean["reward_norm"] = pd.to_numeric(clean["reward_norm"], errors="coerce")
        clean = clean[clean["reward_norm"].notna()].copy()

        splits = {
            "train": clean[clean["split"] == "train"].copy(),
            "val": clean[clean["split"] == "val"].copy(),
            "test": clean[clean["split"] == "test"].copy(),
        }

        self._ensure_non_empty_split(splits["train"], "train")
        self._ensure_non_empty_split(splits["val"], "val")
        self._ensure_non_empty_split(splits["test"], "test")

        x_train = splits["train"][self.feature_builder.feature_columns].to_numpy(dtype=float)
        y_train = splits["train"]["reward_norm"].to_numpy(dtype=float)
        x_val = splits["val"][self.feature_builder.feature_columns].to_numpy(dtype=float)
        y_val = splits["val"]["reward_norm"].to_numpy(dtype=float)
        x_test = splits["test"][self.feature_builder.feature_columns].to_numpy(dtype=float)
        y_test = splits["test"]["reward_norm"].to_numpy(dtype=float)

        best_model = None
        best_info: Dict[str, float] = {"val_loss": float("inf"), "lr": 0.0, "batch_size": 0.0}
        for learning_rate in self.train_config.learning_rates:
            for batch_size in self.train_config.batch_sizes:
                model, val_loss = self._fit_one(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    learning_rate=float(learning_rate),
                    batch_size=int(batch_size),
                )
                if val_loss < float(best_info["val_loss"]):
                    best_model = model
                    best_info = {
                        "val_loss": float(val_loss),
                        "lr": float(learning_rate),
                        "batch_size": float(batch_size),
                    }

        if best_model is None:
            raise RuntimeError("Could not train Shot EPV model.")

        best_model.eval()
        with torch.no_grad():
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=self.device)
            test_pred = best_model(x_test_tensor).cpu().numpy().reshape(-1)
            test_mse = float(np.mean((test_pred - y_test) ** 2))

        resolved_dir = output_dir or (REPO_ROOT / "results" / "models" / "shot")
        resolved_dir.mkdir(parents=True, exist_ok=True)
        model_path = resolved_dir / "shot_epv.pt"
        metadata_path = resolved_dir / "shot_epv_metadata.json"

        torch.save(best_model.state_dict(), model_path)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "train_config": {
                        "hidden_dim": self.train_config.hidden_dim,
                        "epochs": self.train_config.epochs,
                        "learning_rates": list(self.train_config.learning_rates),
                        "batch_sizes": list(self.train_config.batch_sizes),
                        "weight_decay": self.train_config.weight_decay,
                        "early_stopping_delta": self.train_config.early_stopping_delta,
                        "patience": self.train_config.patience,
                    },
                    "best_params": best_info,
                    "test_mse": test_mse,
                    "feature_columns": list(self.feature_builder.feature_columns),
                },
                handle,
                indent=2,
                sort_keys=True,
            )

        return {
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "best_params": best_info,
            "test_mse": test_mse,
            "train_rows": int(len(splits["train"])),
            "val_rows": int(len(splits["val"])),
            "test_rows": int(len(splits["test"])),
        }
