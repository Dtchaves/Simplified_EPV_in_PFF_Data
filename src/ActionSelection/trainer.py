from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pickle
import json

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from Pass.data_utils import REPO_ROOT
from Shot.baseline_xg import BaselineXGArtifacts

from .models import ActionSelectionNet
from .data import (
    ActionSelectionDataConfig,
    apply_action_selection_splits,
    build_action_selection_dataset,
    build_action_selection_source_dataset,
    build_action_selection_split_manifest,
    load_default_baseline_xg_artifacts,
)


@dataclass
class ActionSelectionTrainConfig:
    epochs: int = 30
    batch_size: int = 64
    batch_sizes: tuple[int, ...] = (16, 32)
    lr: float = 1e-3
    learning_rates: tuple[float, ...] = (1e-3, 1e-4, 1e-5, 1e-6)
    patience: int = 5
    split_seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


def _evaluate_multiclass(y_true: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    predictions = np.argmax(probabilities, axis=1) if probabilities.size else np.asarray([], dtype=int)
    return {
        "count": int(y_true.size),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])) if y_true.size > 0 else float("nan"),
        "accuracy": float((predictions == y_true).mean()) if y_true.size > 0 else float("nan"),
    }


def _as_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).long(),
    )
    return DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=shuffle)


def _ensure_non_empty_split(df: pd.DataFrame, split_name: str) -> None:
    if df.empty:
        raise ValueError(f"ActionSelection split '{split_name}' is empty. Check split manifest or source data.")


def _train_on_dataset(
    dataset: pd.DataFrame,
    output_dir: Path,
    config: ActionSelectionTrainConfig,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if dataset.empty:
        raise ValueError("ActionSelection dataset is empty.")
    if "split" not in dataset.columns:
        raise ValueError("ActionSelection dataset must contain a 'split' column.")

    feature_columns = [column for column in dataset.columns if column not in {"label", "split"}]
    clean = dataset.dropna(subset=feature_columns + ["label", "split"]).copy()
    clean["label"] = pd.to_numeric(clean["label"], errors="coerce")
    clean = clean[clean["label"].notna()].copy()
    clean["label"] = clean["label"].astype(int)

    splits = {
        "train": clean[clean["split"] == "train"].copy(),
        "val": clean[clean["split"] == "val"].copy(),
        "test": clean[clean["split"] == "test"].copy(),
    }
    _ensure_non_empty_split(splits["train"], "train")
    _ensure_non_empty_split(splits["val"], "val")
    _ensure_non_empty_split(splits["test"], "test")

    x_train = splits["train"][feature_columns].to_numpy(dtype=float)
    y_train = splits["train"]["label"].to_numpy(dtype=int)
    x_val = splits["val"][feature_columns].to_numpy(dtype=float)
    y_val = splits["val"]["label"].to_numpy(dtype=int)
    x_test = splits["test"][feature_columns].to_numpy(dtype=float)
    y_test = splits["test"]["label"].to_numpy(dtype=int)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = x_train_scaled.shape[1]
    n_actions = int(max(clean["label"].max(), 2) + 1)

    def _fit_one(learning_rate: float, batch_size: int) -> tuple[ActionSelectionNet, float]:
        model = ActionSelectionNet(n_features=n_features, n_actions=n_actions).to(device)
        optimizer = optim.Adam(model.parameters(), lr=float(learning_rate), betas=(0.9, 0.999))
        criterion = nn.CrossEntropyLoss()

        train_loader = _as_loader(x_train_scaled, y_train, batch_size, shuffle=True)
        val_loader = _as_loader(x_val_scaled, y_val, batch_size, shuffle=False)

        best_state = None
        best_val_loss = float("inf")
        patience_left = int(config.patience)

        for _ in range(1, config.epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    val_losses.append(float(criterion(model(xb), yb).item()))

            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_left = int(config.patience)
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is None:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(best_state)
        return model, best_val_loss

    best_model = None
    best_val_loss = float("inf")
    best_params: Dict[str, float] = {"lr": 0.0, "batch_size": 0.0}
    for learning_rate in config.learning_rates:
        for batch_size in config.batch_sizes:
            model, val_loss = _fit_one(float(learning_rate), int(batch_size))
            if val_loss < best_val_loss:
                best_model = model
                best_val_loss = val_loss
                best_params = {"lr": float(learning_rate), "batch_size": float(batch_size)}

    if best_model is None:
        raise RuntimeError("Could not train ActionSelection model.")

    model = best_model

    def _predict_probabilities(matrix: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(matrix).float().to(device)
            logits = model(tensor)
            return torch.softmax(logits, dim=-1).detach().cpu().numpy()

    train_probs = _predict_probabilities(x_train_scaled)
    val_probs = _predict_probabilities(x_val_scaled)
    test_probs = _predict_probabilities(x_test_scaled)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "action_selection_model.pt"
    compatibility_model_path = output_dir / "action_selection_net.pt"
    scaler_path = output_dir / "action_selection_feature_scaler.pkl"
    metadata_path = output_dir / "action_selection_metadata.json"

    torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), compatibility_model_path)
    with scaler_path.open("wb") as fh:
        pickle.dump(scaler, fh)

    split_counts = {split_name: int(len(frame)) for split_name, frame in splits.items()}
    metrics = {
        "train": _evaluate_multiclass(y_train, train_probs),
        "val": _evaluate_multiclass(y_val, val_probs),
        "test": _evaluate_multiclass(y_test, test_probs),
    }
    metadata_payload: Dict[str, Any] = {
        "feature_columns": feature_columns,
        "n_actions": n_actions,
        "split_counts": split_counts,
        "metrics": metrics,
        "train_config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "batch_sizes": list(config.batch_sizes),
            "learning_rates": list(config.learning_rates),
            "patience": config.patience,
            "split_seed": config.split_seed,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
        },
        "best_params": best_params,
    }
    if extra_metadata:
        metadata_payload.update(extra_metadata)

    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata_payload, fh, indent=2)

    report = {
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "scaler_path": str(scaler_path),
        "split_counts": split_counts,
        "metrics": metrics,
        "best_params": best_params,
    }

    report_path = REPO_ROOT / "results" / "metrics" / "action_selection_training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return report


def evaluate_action_selection_dataset(
    dataset: pd.DataFrame,
    model_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if dataset.empty:
        raise ValueError("ActionSelection dataset is empty.")
    if "split" not in dataset.columns:
        raise ValueError("ActionSelection dataset must contain a 'split' column.")

    resolved_dir = model_dir or (REPO_ROOT / "results" / "models" / "action_selection")
    model_path = resolved_dir / "action_selection_model.pt"
    scaler_path = resolved_dir / "action_selection_feature_scaler.pkl"
    metadata_path = resolved_dir / "action_selection_metadata.json"
    if not model_path.exists() or not scaler_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing ActionSelection artifacts under {resolved_dir}")

    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    feature_columns = list(metadata.get("feature_columns") or [])
    n_actions = int(metadata.get("n_actions") or 3)

    clean = dataset.dropna(subset=feature_columns + ["label", "split"]).copy()
    clean["label"] = pd.to_numeric(clean["label"], errors="coerce")
    clean = clean[clean["label"].notna()].copy()
    clean["label"] = clean["label"].astype(int)

    val_df = clean[clean["split"] == "val"].copy()
    test_df = clean[clean["split"] == "test"].copy()
    _ensure_non_empty_split(val_df, "val")
    _ensure_non_empty_split(test_df, "test")

    with scaler_path.open("rb") as fh:
        scaler = pickle.load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionSelectionNet(n_features=len(feature_columns), n_actions=n_actions).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        raise TypeError(f"Unsupported ActionSelection checkpoint payload: {type(state)!r}")
    model.eval()

    def _predict(frame: pd.DataFrame) -> np.ndarray:
        matrix = scaler.transform(frame[feature_columns].to_numpy(dtype=float))
        with torch.no_grad():
            tensor = torch.from_numpy(matrix).float().to(device)
            logits = model(tensor)
            return torch.softmax(logits, dim=-1).detach().cpu().numpy()

    val_probs = _predict(val_df)
    test_probs = _predict(test_df)
    report = {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "split_counts": {"val": int(len(val_df)), "test": int(len(test_df))},
        "metrics": {
            "val": _evaluate_multiclass(val_df["label"].to_numpy(dtype=int), val_probs),
            "test": _evaluate_multiclass(test_df["label"].to_numpy(dtype=int), test_probs),
        },
    }

    report_path = REPO_ROOT / "results" / "metrics" / "action_selection_test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return report


def train_action_selection(
    actions_df: pd.DataFrame,
    tracking_df: Optional[pd.DataFrame] = None,
    baseline_xg_artifacts: Optional[BaselineXGArtifacts] = None,
    output_dir: Optional[Path] = None,
    config: Optional[ActionSelectionTrainConfig] = None,
) -> Path:
    config = config or ActionSelectionTrainConfig()
    dataset = build_action_selection_dataset(actions_df, tracking_df, baseline_xg_artifacts)
    split_config = ActionSelectionDataConfig(
        split_seed=config.split_seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    manifest, _ = build_action_selection_split_manifest(actions_df, split_config)
    split_frame = apply_action_selection_splits(actions_df, manifest)
    dataset["split"] = split_frame["split"].values

    resolved_dir = output_dir or (Path.cwd() / "results" / "models" / "action_selection")
    report = _train_on_dataset(dataset, resolved_dir, config)
    return Path(report["output_dir"])


def train_action_selection_from_sources(
    data_config: Optional[ActionSelectionDataConfig] = None,
    output_dir: Optional[Path] = None,
    config: Optional[ActionSelectionTrainConfig] = None,
    baseline_xg_artifacts: Optional[BaselineXGArtifacts] = None,
) -> Dict[str, Any]:
    config = config or ActionSelectionTrainConfig()
    baseline_artifacts = baseline_xg_artifacts or load_default_baseline_xg_artifacts()
    dataset, data_summary = build_action_selection_source_dataset(
        config=data_config,
        baseline_xg_artifacts=baseline_artifacts,
    )
    return _train_on_dataset(
        dataset,
        output_dir or (REPO_ROOT / "results" / "models" / "action_selection"),
        config,
        extra_metadata={
            "data_summary": data_summary,
            "baseline_xg_available": bool(baseline_artifacts is not None),
        },
    )


def test_action_selection_from_sources(
    data_config: Optional[ActionSelectionDataConfig] = None,
    model_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    baseline_xg_artifacts: Optional[BaselineXGArtifacts] = None,
) -> Dict[str, Any]:
    baseline_artifacts = baseline_xg_artifacts or load_default_baseline_xg_artifacts()
    dataset, data_summary = build_action_selection_source_dataset(
        config=data_config,
        baseline_xg_artifacts=baseline_artifacts,
    )
    report = evaluate_action_selection_dataset(
        dataset,
        model_dir=model_dir or output_dir or (REPO_ROOT / "results" / "models" / "action_selection"),
    )
    report["data_summary"] = data_summary
    report["baseline_xg_available"] = bool(baseline_artifacts is not None)
    return report
