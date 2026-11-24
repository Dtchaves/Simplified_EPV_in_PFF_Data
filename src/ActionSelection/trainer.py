from __future__ import annotations

from dataclasses import dataclass
import logging
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
from tqdm.auto import tqdm

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
from .inference import _load_action_selection_artifacts


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
    calibrate_temperature: bool = True
    att_weight_clip: float = 50.0


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


def _ece_binary(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    if y_true.size == 0:
        return float("nan")
    p = np.clip(p, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        if high == 1.0:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)
        if not np.any(mask):
            continue
        weight = float(mask.mean())
        ece += weight * abs(float(y_true[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def _shot_binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.size == 0 or probabilities.size == 0:
        return {"count": int(y_true.size), "shot_rate": float("nan"), "binary_log_loss": float("nan"), "brier": float("nan"), "ece_15": float("nan"), "mean_p_shot": float("nan")}
    y_shot = (y_true == 2).astype(int)
    p_shot = np.clip(probabilities[:, 2], 1e-6, 1.0 - 1e-6)
    return {
        "count": int(y_true.size),
        "shot_rate": float(y_shot.mean()),
        "binary_log_loss": float(log_loss(y_shot, p_shot, labels=[0, 1])),
        "brier": float(np.mean((p_shot - y_shot) ** 2)),
        "ece_15": _ece_binary(y_shot, p_shot, n_bins=15),
        "mean_p_shot": float(np.mean(p_shot)),
    }


def _predict_logits(model: ActionSelectionNet, matrix: np.ndarray, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    outputs = []
    loader = DataLoader(torch.from_numpy(matrix).float(), batch_size=max(1, int(batch_size)), shuffle=False)
    with torch.no_grad():
        for xb in loader:
            logits = model(xb.to(device))
            outputs.append(logits.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 3), dtype=float)


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    if logits.size == 0:
        return logits
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _fit_temperature_from_logits(logits_val: np.ndarray, y_val: np.ndarray, enabled: bool = True) -> float:
    if not enabled or len(y_val) == 0:
        return 1.0
    logits_tensor = torch.tensor(logits_val, dtype=torch.float32)
    y_tensor = torch.tensor(y_val, dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 100.0)
        loss = criterion(logits_tensor / temperature, y_tensor)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        temperature = float(torch.exp(log_temperature).detach().cpu().clamp(0.05, 100.0).item())
    except Exception:
        temperature = 1.0
    return temperature if np.isfinite(temperature) and temperature > 0 else 1.0


def _smd(treated_values: np.ndarray, control_values: np.ndarray, control_weights: Optional[np.ndarray] = None) -> float:
    treated_values = np.asarray(treated_values, dtype=float)
    control_values = np.asarray(control_values, dtype=float)
    treated_values = treated_values[np.isfinite(treated_values)]
    finite_control = np.isfinite(control_values)
    control_values = control_values[finite_control]
    if control_weights is not None:
        control_weights = np.asarray(control_weights, dtype=float)[finite_control]
        control_weights = np.where(np.isfinite(control_weights), control_weights, 0.0)
    if treated_values.size == 0 or control_values.size == 0:
        return float("nan")
    mt = float(np.mean(treated_values))
    vt = float(np.var(treated_values))
    if control_weights is None:
        mc = float(np.mean(control_values))
        vc = float(np.var(control_values))
    else:
        weight_sum = float(np.sum(control_weights))
        if weight_sum <= 0:
            return float("nan")
        norm_w = control_weights / weight_sum
        mc = float(np.sum(norm_w * control_values))
        vc = float(np.sum(norm_w * (control_values - mc) ** 2))
    denom = float(np.sqrt((vt + vc) / 2.0))
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float((mt - mc) / denom)


def _causal_propensity_diagnostics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    feature_columns: list[str],
    att_weight_clip: float = 50.0,
) -> Dict[str, Any]:
    if frame.empty or probabilities.size == 0:
        return {}
    labels = frame["label"].to_numpy(dtype=int)
    treated = labels == 2
    control = ~treated
    p_shot = np.clip(probabilities[:, 2], 1e-6, 1.0 - 1e-6)
    p_continue = np.clip(probabilities[:, 0] + probabilities[:, 1], 1e-6, 1.0)
    weights = p_shot[control] / (1.0 - p_shot[control])
    weights_clipped = np.minimum(weights, float(att_weight_clip))
    ess = float((weights_clipped.sum() ** 2) / np.sum(weights_clipped ** 2)) if weights_clipped.size and np.sum(weights_clipped ** 2) > 0 else float("nan")
    support = {
        "continue_tau_002_shot_share": float(np.mean(p_continue[treated] > 0.02)) if treated.any() else float("nan"),
        "continue_tau_005_shot_share": float(np.mean(p_continue[treated] > 0.05)) if treated.any() else float("nan"),
        "continue_tau_010_shot_share": float(np.mean(p_continue[treated] > 0.10)) if treated.any() else float("nan"),
        "common_support_005_095_shot_share": float(np.mean((p_shot[treated] > 0.05) & (p_shot[treated] < 0.95))) if treated.any() else float("nan"),
    }
    smd = {}
    for column in feature_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        raw = abs(_smd(values[treated], values[control])) if treated.any() and control.any() else float("nan")
        weighted = abs(_smd(values[treated], values[control], weights_clipped)) if treated.any() and control.any() else float("nan")
        smd[column] = {"raw_abs_smd": raw, "att_weighted_abs_smd": weighted}
    worst_weighted = sorted(
        ((column, vals["att_weighted_abs_smd"]) for column, vals in smd.items() if np.isfinite(vals["att_weighted_abs_smd"])),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    return {
        "count": int(len(frame)),
        "n_shots": int(treated.sum()),
        "n_continue": int(control.sum()),
        "support": support,
        "control_weight_summary": {
            "clip": float(att_weight_clip),
            "ess_clipped": ess,
            "ess_share_of_controls": float(ess / max(1, int(control.sum()))) if np.isfinite(ess) else float("nan"),
            "mean": float(np.mean(weights)) if weights.size else float("nan"),
            "p95": float(np.quantile(weights, 0.95)) if weights.size else float("nan"),
            "p99": float(np.quantile(weights, 0.99)) if weights.size else float("nan"),
            "max": float(np.max(weights)) if weights.size else float("nan"),
        },
        "smd_by_feature": smd,
        "worst_weighted_smd": [{"feature": column, "att_weighted_abs_smd": float(value)} for column, value in worst_weighted],
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

        for _ in tqdm(range(config.epochs), desc=f"LR={learning_rate}, Batch={batch_size}", unit="epoch", dynamic_ncols=True, colour="BLUE"):
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

    train_logits = _predict_logits(model, x_train_scaled, device)
    val_logits = _predict_logits(model, x_val_scaled, device)
    test_logits = _predict_logits(model, x_test_scaled, device)
    temperature = _fit_temperature_from_logits(val_logits, y_val, enabled=bool(config.calibrate_temperature))

    def _predict_probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
        return _softmax_numpy(logits / max(float(temperature), 1e-6))

    train_probs = _predict_probabilities_from_logits(train_logits)
    val_probs = _predict_probabilities_from_logits(val_logits)
    test_probs = _predict_probabilities_from_logits(test_logits)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "action_selection_model.pt"
    compatibility_model_path = output_dir / "action_selection_net.pt"
    scaler_path = output_dir / "action_selection_feature_scaler.pkl"
    metadata_path = output_dir / "action_selection_metadata.json"

    split_counts = {split_name: int(len(frame)) for split_name, frame in splits.items()}
    metrics = {
        "train": _evaluate_multiclass(y_train, train_probs),
        "val": _evaluate_multiclass(y_val, val_probs),
        "test": _evaluate_multiclass(y_test, test_probs),
    }
    shot_binary_metrics = {
        "train": _shot_binary_metrics(y_train, train_probs),
        "val": _shot_binary_metrics(y_val, val_probs),
        "test": _shot_binary_metrics(y_test, test_probs),
    }
    causal_diagnostics = {
        "train": _causal_propensity_diagnostics(splits["train"], train_probs, feature_columns, config.att_weight_clip),
        "val": _causal_propensity_diagnostics(splits["val"], val_probs, feature_columns, config.att_weight_clip),
        "test": _causal_propensity_diagnostics(splits["test"], test_probs, feature_columns, config.att_weight_clip),
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
            "calibrate_temperature": config.calibrate_temperature,
            "att_weight_clip": config.att_weight_clip,
        },
        "best_params": best_params,
        "temperature": float(temperature),
        "calibration": {"method": "temperature_scaling", "temperature": float(temperature), "fit_split": "val"},
        "shot_binary_metrics": shot_binary_metrics,
        "causal_propensity_diagnostics": causal_diagnostics,
    }
    if extra_metadata:
        metadata_payload.update(extra_metadata)

    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "model_config": {
            "n_features": int(n_features),
            "n_actions": int(n_actions),
        },
        "feature_columns": list(feature_columns),
        "split_counts": split_counts,
        "best_params": best_params,
        "temperature": float(temperature),
        "train_config": metadata_payload["train_config"],
    }

    torch.save(checkpoint_payload, model_path)
    torch.save(model.state_dict(), compatibility_model_path)
    with scaler_path.open("wb") as fh:
        pickle.dump(scaler, fh)

    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata_payload, fh, indent=2)

    report = {
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "scaler_path": str(scaler_path),
        "checkpoint_path": str(model_path),
        "split_counts": split_counts,
        "metrics": metrics,
        "shot_binary_metrics": shot_binary_metrics,
        "causal_propensity_diagnostics": causal_diagnostics,
        "temperature": float(temperature),
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

    model, scaler, metadata = _load_action_selection_artifacts(
        model_path=model_path,
        metadata_path=metadata_path,
        scaler_path=scaler_path,
    )
    feature_columns = list(metadata.get("feature_columns") or feature_columns)

    temperature = float(metadata.get("temperature") or (metadata.get("calibration") or {}).get("temperature") or 1.0)

    def _predict(frame: pd.DataFrame) -> np.ndarray:
        matrix = scaler.transform(frame[feature_columns].to_numpy(dtype=float))
        with torch.no_grad():
            tensor = torch.from_numpy(matrix).float()
            logits = model(tensor) / max(float(temperature), 1e-6)
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
        "shot_binary_metrics": {
            "val": _shot_binary_metrics(val_df["label"].to_numpy(dtype=int), val_probs),
            "test": _shot_binary_metrics(test_df["label"].to_numpy(dtype=int), test_probs),
        },
        "causal_propensity_diagnostics": {
            "val": _causal_propensity_diagnostics(val_df, val_probs, feature_columns),
            "test": _causal_propensity_diagnostics(test_df, test_probs, feature_columns),
        },
        "temperature": temperature,
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
    data_config = ActionSelectionDataConfig(
        split_seed=config.split_seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    from .data import _filter_action_rows

    filtered_actions, _ = _filter_action_rows(actions_df, config=data_config)
    dataset, _ = build_action_selection_dataset(
        filtered_actions,
        tracking_df,
        baseline_xg_artifacts,
        data_config=data_config,
        return_audit=True,
    )
    split_config = ActionSelectionDataConfig(
        split_seed=config.split_seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    manifest, manifest_path = build_action_selection_split_manifest(filtered_actions, split_config)
    split_frame = apply_action_selection_splits(filtered_actions, manifest)
    dataset["split"] = split_frame["split"].values

    resolved_dir = output_dir or (Path.cwd() / "results" / "models" / "action_selection")
    report = _train_on_dataset(
        dataset,
        resolved_dir,
        config,
        extra_metadata={"split_manifest_path": str(manifest_path)},
    )
    return Path(report["output_dir"])


def train_action_selection_from_sources(
    data_config: Optional[ActionSelectionDataConfig] = None,
    output_dir: Optional[Path] = None,
    config: Optional[ActionSelectionTrainConfig] = None,
    baseline_xg_artifacts: Optional[BaselineXGArtifacts] = None,
) -> Dict[str, Any]:
    config = config or ActionSelectionTrainConfig()
    if data_config is None:
        data_config = ActionSelectionDataConfig(
            split_seed=config.split_seed,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
        )
    baseline_artifacts = baseline_xg_artifacts or load_default_baseline_xg_artifacts()
    logging.info("Loaded baseline XG artifacts: %s", bool(baseline_artifacts))
    logging.info("Building ActionSelection dataset from source files")
    dataset, data_summary = build_action_selection_source_dataset(
        config=data_config,
        baseline_xg_artifacts=baseline_artifacts,
    )

    logging.info("Built ActionSelection dataset from sources with %d samples", len(dataset))
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
