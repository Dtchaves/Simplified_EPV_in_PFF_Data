from __future__ import annotations

import hashlib
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

BALL_DRIVE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = BALL_DRIVE_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Pass.data_utils import REPO_ROOT

try:
    from .data import (
        BallDriveDataConfig,
        apply_match_splits,
        attach_reward_labels,
        build_ball_drive_canonical_dataset,
        build_ball_drive_split_manifest,
        get_ball_drive_cache_root,
        segment_ball_drives,
    )
    from .evaluate import evaluate_de, evaluate_dp
    from .features import BallDriveFeatureBuilder
    from .models import BallDriveDEModel, BallDriveDPModel
except ImportError:
    from data import (  # type: ignore
        BallDriveDataConfig,
        apply_match_splits,
        attach_reward_labels,
        build_ball_drive_canonical_dataset,
        build_ball_drive_split_manifest,
        get_ball_drive_cache_root,
        segment_ball_drives,
    )
    from evaluate import evaluate_de, evaluate_dp  # type: ignore
    from features import BallDriveFeatureBuilder  # type: ignore
    from models import BallDriveDEModel, BallDriveDPModel  # type: ignore


@dataclass
class BallDriveTrainingConfig:
    hidden_dim: int = 64
    epochs: int = 20
    device: str = "cpu"
    learning_rates: Tuple[float, ...] = (1e-3, 1e-4, 1e-5, 1e-6)
    batch_sizes: Tuple[int, ...] = (16, 32)
    weight_decay: float = 0.0


class BallDriveTrainer:
    def __init__(
        self,
        data_config: Optional[BallDriveDataConfig] = None,
        train_config: Optional[BallDriveTrainingConfig] = None,
    ):
        self.data_config = data_config or BallDriveDataConfig()
        self.train_config = train_config or BallDriveTrainingConfig(
            device=("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.device = torch.device(self.train_config.device)
        self.feature_builder = BallDriveFeatureBuilder()

    @staticmethod
    def _ensure_non_empty_split(df: pd.DataFrame, split_name: str) -> None:
        if df.empty:
            raise ValueError(f"Split '{split_name}' is empty. Check split ratios or data availability.")

    @staticmethod
    def _as_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).view(-1, 1),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def _train_dp_with_grid(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[BallDriveDPModel, Dict[str, float]]:
        best_model = None
        best_info: Dict[str, float] = {"val_loss": float("inf"), "lr": 0.0, "batch_size": 0.0}

        for lr in self.train_config.learning_rates:
            for batch_size in self.train_config.batch_sizes:
                model = BallDriveDPModel(input_dim=x_train.shape[1], hidden_dim=self.train_config.hidden_dim).to(self.device)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=float(lr),
                    betas=(0.9, 0.999),
                    weight_decay=float(self.train_config.weight_decay),
                )
                criterion = nn.BCELoss()

                train_loader = self._as_loader(x_train, y_train, int(batch_size), shuffle=True)
                val_loader = self._as_loader(x_val, y_val, int(batch_size), shuffle=False)

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
                        pred = model(xb)
                        val_losses.append(float(criterion(pred, yb).item()))

                val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
                if val_loss < float(best_info["val_loss"]):
                    best_info = {"val_loss": val_loss, "lr": float(lr), "batch_size": float(batch_size)}
                    best_model = model

        if best_model is None:
            raise RuntimeError("Could not train DP model.")

        return best_model, best_info

    def _train_de_model(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        hidden_dim: int,
    ) -> Tuple[BallDriveDEModel, Dict[str, float]]:
        criterion = nn.MSELoss()
        best_model = None
        best_info: Dict[str, float] = {"val_loss": float("inf"), "lr": 0.0, "batch_size": 0.0}

        for lr in self.train_config.learning_rates:
            for batch_size in self.train_config.batch_sizes:
                model = BallDriveDEModel(input_dim=x_train.shape[1], hidden_dim=hidden_dim).to(self.device)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=float(lr),
                    betas=(0.9, 0.999),
                    weight_decay=float(self.train_config.weight_decay),
                )

                train_loader = self._as_loader(x_train, y_train, batch_size=int(batch_size), shuffle=True)
                val_loader = self._as_loader(x_val, y_val, batch_size=int(batch_size), shuffle=False)

                best_state = None
                best_loss = float("inf")

                for _ in range(max(5, int(self.train_config.epochs // 2))):
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
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                if best_state is not None:
                    model.load_state_dict(best_state)

                if best_loss < float(best_info["val_loss"]):
                    best_info = {"val_loss": best_loss, "lr": float(lr), "batch_size": float(batch_size)}
                    best_model = model

        if best_model is None:
            raise RuntimeError("Could not train DE model.")

        return best_model, best_info

    @staticmethod
    def _collect_tracking_data(segmented_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        tracking_paths = sorted({str(path) for path in segmented_df.get("source_tracking_path", pd.Series(dtype=str)).dropna().unique().tolist() if str(path)})
        if not tracking_paths:
            return None

        chunks: List[pd.DataFrame] = []
        columns = ["match_id", "frame_id", "period", "elapsed_seconds", "team_side", "player_id", "x", "y", "ball_x", "ball_y"]
        for path in tracking_paths:
            try:
                if str(path).lower().endswith(".csv"):
                    df = pd.read_csv(path)
                else:
                    df = pd.read_parquet(path)
            except Exception:
                continue

            rename_map = {}
            if "game_id" in df.columns and "match_id" not in df.columns:
                rename_map["game_id"] = "match_id"
            if "frame_num" in df.columns and "frame_id" not in df.columns:
                rename_map["frame_num"] = "frame_id"
            if "team" in df.columns and "team_side" not in df.columns:
                rename_map["team"] = "team_side"
            if rename_map:
                df = df.rename(columns=rename_map)

            keep = [col for col in columns if col in df.columns]
            if {"match_id", "frame_id", "team_side", "player_id", "x", "y"}.issubset(set(keep)):
                chunks.append(df[keep].copy())

        if not chunks:
            return None

        tracking = pd.concat(chunks, ignore_index=True)
        tracking = tracking.drop_duplicates(subset=["match_id", "frame_id", "team_side", "player_id"], keep="first")
        return tracking

    @staticmethod
    def _build_fingerprint(paths: List[Path]) -> str:
        parts = []
        for path in paths:
            if not path.exists():
                continue
            stat = path.stat()
            parts.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]

    def run(self) -> Dict[str, object]:
        canonical_df, canonical_summary = build_ball_drive_canonical_dataset(self.data_config)
        segmented_df, segmentation_summary = segment_ball_drives(canonical_df)
        labeled_df, reward_summary = attach_reward_labels(
            segmented_df,
            include_open_play_null=self.data_config.include_open_play_null,
        )

        manifest, manifest_path = build_ball_drive_split_manifest(labeled_df, self.data_config)
        dataset = apply_match_splits(labeled_df, manifest)

        dataset = dataset[dataset["y_success"].notna()].copy()
        dataset["y_success"] = pd.to_numeric(dataset["y_success"], errors="coerce")
        dataset = dataset[dataset["y_success"].isin([0, 1])].copy()

        splits = {
            "train": dataset[dataset["split"] == "train"].copy(),
            "val": dataset[dataset["split"] == "val"].copy(),
            "test": dataset[dataset["split"] == "test"].copy(),
        }

        split_fallback_used = False
        if splits["train"].empty or splits["val"].empty or splits["test"].empty:
            if len(dataset) < 3:
                raise ValueError("Insufficient rows to build non-empty train/val/test splits for BallDrive training.")

            split_fallback_used = True
            shuffled = dataset.sample(frac=1.0, random_state=int(self.data_config.split_seed))
            n_total = len(shuffled)
            n_train = int(round(self.data_config.train_ratio * n_total))
            n_val = int(round(self.data_config.val_ratio * n_total))
            n_train = max(1, min(n_train, n_total - 2))
            n_val = max(1, min(n_val, n_total - n_train - 1))
            n_test = max(1, n_total - n_train - n_val)

            splits = {
                "train": shuffled.iloc[:n_train].copy(),
                "val": shuffled.iloc[n_train : n_train + n_val].copy(),
                "test": shuffled.iloc[n_train + n_val : n_train + n_val + n_test].copy(),
            }

        self._ensure_non_empty_split(splits["train"], "train")
        self._ensure_non_empty_split(splits["val"], "val")
        self._ensure_non_empty_split(splits["test"], "test")

        tracking_df = self._collect_tracking_data(dataset)
        full_features = self.feature_builder.build_feature_frame(dataset, tracking_df=tracking_df)

        train_feature_df = full_features.loc[splits["train"].index]
        self.feature_builder.fit(train_feature_df)

        x_train = self.feature_builder.transform(train_feature_df)
        x_val = self.feature_builder.transform(full_features.loc[splits["val"].index])
        x_test = self.feature_builder.transform(full_features.loc[splits["test"].index])

        y_train = splits["train"]["y_success"].to_numpy(dtype=float)
        y_val = splits["val"]["y_success"].to_numpy(dtype=float)
        y_test = splits["test"]["y_success"].to_numpy(dtype=float)

        dp_model, dp_grid = self._train_dp_with_grid(x_train, y_train, x_val, y_val)

        def _predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
            model.eval()
            with torch.no_grad():
                tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
                return model(tensor).detach().cpu().numpy().reshape(-1)

        p_train = _predict(dp_model, x_train)
        p_val = _predict(dp_model, x_val)
        p_test = _predict(dp_model, x_test)

        dp_metrics = {
            "train": evaluate_dp(y_train, p_train),
            "val": evaluate_dp(y_val, p_val),
            "test": evaluate_dp(y_test, p_test),
            "grid": dp_grid,
        }

        split_predictions = []
        split_payload = [("train", splits["train"], p_train), ("val", splits["val"], p_val), ("test", splits["test"], p_test)]
        for split_name, split_df, split_pred in split_payload:
            local = split_df.copy()
            local["split"] = split_name
            local["p_ball_drive_success"] = split_pred
            split_predictions.append(local)

        predictions_df = pd.concat(split_predictions, ignore_index=True)

        model_dir = REPO_ROOT / "results" / "models" / "ball_drive"
        model_dir.mkdir(parents=True, exist_ok=True)
        dp_path = model_dir / "ball_drive_dp.pt"
        scaler_path = model_dir / "ball_drive_scaler.pkl"
        torch.save(dp_model.state_dict(), dp_path)
        self.feature_builder.save(scaler_path)

        cache_root = get_ball_drive_cache_root()
        cache_root.mkdir(parents=True, exist_ok=True)
        fingerprint = self._build_fingerprint([dp_path, scaler_path])
        pred_path = cache_root / f"p_ball_drive_success_{fingerprint}.parquet"
        predictions_df.to_parquet(pred_path, index=False)

        features_with_p = full_features.copy()
        features_with_p["p_ball_drive_success"] = np.nan
        for split_name, split_df, split_pred in split_payload:
            features_with_p.loc[split_df.index, "p_ball_drive_success"] = split_pred

        de_dataset = dataset.copy()
        for col in self.feature_builder.feature_columns:
            de_dataset[f"f_{col}"] = self.feature_builder.transform(full_features[[*self.feature_builder.feature_columns]].loc[de_dataset.index])[:, self.feature_builder.feature_columns.index(col)]
        de_dataset["p_ball_drive_success"] = features_with_p.loc[de_dataset.index, "p_ball_drive_success"]
        de_dataset["reward_G"] = pd.to_numeric(de_dataset.get("reward_G"), errors="coerce")

        de_metrics: Dict[str, object] = {}
        de_models: Dict[str, Optional[BallDriveDEModel]] = {"success": None, "failed": None}

        for tag, target_class in (("success", 1), ("failed", 0)):
            train_subset = de_dataset[(de_dataset["split"] == "train") & (de_dataset["y_success"] == target_class) & (de_dataset["reward_G"].notna())]
            val_subset = de_dataset[(de_dataset["split"] == "val") & (de_dataset["y_success"] == target_class) & (de_dataset["reward_G"].notna())]
            test_subset = de_dataset[(de_dataset["split"] == "test") & (de_dataset["y_success"] == target_class) & (de_dataset["reward_G"].notna())]

            x_cols = [f"f_{col}" for col in self.feature_builder.feature_columns] + ["p_ball_drive_success"]

            if train_subset.empty or val_subset.empty:
                de_metrics[tag] = {"status": "skipped", "reason": "insufficient subset rows"}
                continue

            x_de_train = train_subset[x_cols].to_numpy(dtype=float)
            y_de_train = train_subset["reward_G"].to_numpy(dtype=float)
            x_de_val = val_subset[x_cols].to_numpy(dtype=float)
            y_de_val = val_subset["reward_G"].to_numpy(dtype=float)

            model, de_grid = self._train_de_model(
                x_train=x_de_train,
                y_train=y_de_train,
                x_val=x_de_val,
                y_val=y_de_val,
                hidden_dim=self.train_config.hidden_dim,
            )
            de_models[tag] = model

            def _pred_de(x: np.ndarray) -> np.ndarray:
                model.eval()
                with torch.no_grad():
                    t = torch.tensor(x, dtype=torch.float32).to(self.device)
                    return model(t).detach().cpu().numpy().reshape(-1)

            val_pred = _pred_de(x_de_val)
            test_metrics = {"status": "no_test_rows"}
            if not test_subset.empty:
                x_de_test = test_subset[x_cols].to_numpy(dtype=float)
                y_de_test = test_subset["reward_G"].to_numpy(dtype=float)
                test_pred = _pred_de(x_de_test)
                test_metrics = evaluate_de(y_de_test, test_pred)

            de_metrics[tag] = {
                "status": "trained",
                "val": evaluate_de(y_de_val, val_pred),
                "test": test_metrics,
                "grid": de_grid,
            }

            torch.save(model.state_dict(), model_dir / f"ball_drive_de_{tag}.pt")

        report = {
            "canonical_summary": canonical_summary,
            "segmentation_summary": segmentation_summary,
            "reward_summary": reward_summary,
            "train_config": {
                "hidden_dim": self.train_config.hidden_dim,
                "epochs": self.train_config.epochs,
                "device": str(self.device),
                "learning_rates": list(self.train_config.learning_rates),
                "batch_sizes": list(self.train_config.batch_sizes),
                "weight_decay": self.train_config.weight_decay,
            },
            "split_fallback_used": split_fallback_used,
            "split_manifest_path": str(manifest_path),
            "dp_metrics": dp_metrics,
            "de_metrics": de_metrics,
            "dp_checkpoint": str(dp_path),
            "scaler_path": str(scaler_path),
            "dp_prediction_cache": str(pred_path),
        }

        report_path = REPO_ROOT / "results" / "metrics" / "ball_drive_training_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        return report


def _prepare_ball_drive_dataset(data_config: Optional[BallDriveDataConfig] = None) -> Dict[str, object]:
    data_config = data_config or BallDriveDataConfig()
    canonical_df, canonical_summary = build_ball_drive_canonical_dataset(data_config)
    segmented_df, segmentation_summary = segment_ball_drives(canonical_df)
    labeled_df, reward_summary = attach_reward_labels(
        segmented_df,
        include_open_play_null=data_config.include_open_play_null,
    )

    manifest, manifest_path = build_ball_drive_split_manifest(labeled_df, data_config)
    dataset = apply_match_splits(labeled_df, manifest)
    dataset = dataset[dataset["y_success"].notna()].copy()
    dataset["y_success"] = pd.to_numeric(dataset["y_success"], errors="coerce")
    dataset = dataset[dataset["y_success"].isin([0, 1])].copy()

    splits = {
        "train": dataset[dataset["split"] == "train"].copy(),
        "val": dataset[dataset["split"] == "val"].copy(),
        "test": dataset[dataset["split"] == "test"].copy(),
    }

    split_fallback_used = False
    if splits["train"].empty or splits["val"].empty or splits["test"].empty:
        if len(dataset) < 3:
            raise ValueError("Insufficient rows to build non-empty train/val/test splits for BallDrive evaluation.")

        split_fallback_used = True
        shuffled = dataset.sample(frac=1.0, random_state=int(data_config.split_seed))
        n_total = len(shuffled)
        n_train = int(round(data_config.train_ratio * n_total))
        n_val = int(round(data_config.val_ratio * n_total))
        n_train = max(1, min(n_train, n_total - 2))
        n_val = max(1, min(n_val, n_total - n_train - 1))
        n_test = max(1, n_total - n_train - n_val)

        splits = {
            "train": shuffled.iloc[:n_train].copy(),
            "val": shuffled.iloc[n_train : n_train + n_val].copy(),
            "test": shuffled.iloc[n_train + n_val : n_train + n_val + n_test].copy(),
        }

    BallDriveTrainer._ensure_non_empty_split(splits["train"], "train")
    BallDriveTrainer._ensure_non_empty_split(splits["val"], "val")
    BallDriveTrainer._ensure_non_empty_split(splits["test"], "test")
    tracking_df = BallDriveTrainer._collect_tracking_data(dataset)

    return {
        "dataset": dataset,
        "splits": splits,
        "tracking_df": tracking_df,
        "canonical_summary": canonical_summary,
        "segmentation_summary": segmentation_summary,
        "reward_summary": reward_summary,
        "split_manifest_path": str(manifest_path),
        "split_fallback_used": split_fallback_used,
    }


def test_ball_drive_models(
    data_config: Optional[BallDriveDataConfig] = None,
    model_dir: Optional[Path] = None,
) -> Dict[str, object]:
    prepared = _prepare_ball_drive_dataset(data_config)
    dataset = prepared["dataset"]
    splits = prepared["splits"]
    tracking_df = prepared["tracking_df"]

    resolved_dir = Path(model_dir or (REPO_ROOT / "results" / "models" / "ball_drive"))
    dp_path = resolved_dir / "ball_drive_dp.pt"
    scaler_path = resolved_dir / "ball_drive_scaler.pkl"
    if not dp_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("BallDrive artifacts not found. Train the model before testing.")

    hidden_dim = 64
    training_report_path = REPO_ROOT / "results" / "metrics" / "ball_drive_training_report.json"
    if training_report_path.exists():
        with training_report_path.open("r", encoding="utf-8") as handle:
            training_report = json.load(handle)
        hidden_dim = int(training_report.get("train_config", {}).get("hidden_dim", hidden_dim))

    feature_builder = BallDriveFeatureBuilder.load(scaler_path)
    full_features = feature_builder.build_feature_frame(dataset, tracking_df=tracking_df)
    scaled_all = feature_builder.transform(full_features.loc[dataset.index])
    scaled_feature_columns = [f"f_{column}" for column in feature_builder.feature_columns]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp_model = BallDriveDPModel(input_dim=scaled_all.shape[1], hidden_dim=hidden_dim).to(device)
    dp_model.load_state_dict(torch.load(dp_path, map_location=device, weights_only=False))
    dp_model.eval()

    with torch.no_grad():
        tensor = torch.tensor(scaled_all, dtype=torch.float32, device=device)
        p_all = dp_model(tensor).detach().cpu().numpy().reshape(-1)

    dp_metrics = {}
    for split_name, split_df in splits.items():
        mask = dataset.index.isin(split_df.index)
        dp_metrics[split_name] = evaluate_dp(
            split_df["y_success"].to_numpy(dtype=float),
            p_all[mask],
        )

    de_dataset = dataset.copy()
    for idx, column in enumerate(scaled_feature_columns):
        de_dataset[column] = scaled_all[:, idx]
    de_dataset["p_ball_drive_success"] = p_all
    de_dataset["reward_G"] = pd.to_numeric(de_dataset.get("reward_G"), errors="coerce")

    de_metrics: Dict[str, object] = {}
    x_cols = scaled_feature_columns + ["p_ball_drive_success"]
    for tag, target_class in (("success", 1), ("failed", 0)):
        de_path = resolved_dir / f"ball_drive_de_{tag}.pt"
        if not de_path.exists():
            de_metrics[tag] = {"status": "missing_checkpoint"}
            continue

        test_subset = de_dataset[
            (de_dataset["split"] == "test")
            & (de_dataset["y_success"] == target_class)
            & (de_dataset["reward_G"].notna())
        ].copy()
        if test_subset.empty:
            de_metrics[tag] = {"status": "no_test_rows"}
            continue

        de_model = BallDriveDEModel(input_dim=len(x_cols), hidden_dim=hidden_dim).to(device)
        de_model.load_state_dict(torch.load(de_path, map_location=device, weights_only=False))
        de_model.eval()
        with torch.no_grad():
            tensor = torch.tensor(test_subset[x_cols].to_numpy(dtype=float), dtype=torch.float32, device=device)
            predictions = de_model(tensor).detach().cpu().numpy().reshape(-1)
        de_metrics[tag] = {"status": "evaluated", "test": evaluate_de(test_subset["reward_G"].to_numpy(dtype=float), predictions)}

    report = {
        "canonical_summary": prepared["canonical_summary"],
        "segmentation_summary": prepared["segmentation_summary"],
        "reward_summary": prepared["reward_summary"],
        "split_manifest_path": prepared["split_manifest_path"],
        "split_fallback_used": prepared["split_fallback_used"],
        "dp_metrics": dp_metrics,
        "de_metrics": de_metrics,
        "dp_checkpoint": str(dp_path),
        "scaler_path": str(scaler_path),
    }

    report_path = REPO_ROOT / "results" / "metrics" / "ball_drive_test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report
