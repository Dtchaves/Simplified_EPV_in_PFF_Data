from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from Pass.data_utils import REPO_ROOT

from Shot.baseline_xg import BaselineXGArtifacts, load_baseline_xg

from .models import ActionSelectionNet


def _resolve_action_selection_artifact_paths(
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
) -> tuple[Path, Path, Path]:
    resolved_model_path = model_path or (REPO_ROOT / "results" / "models" / "action_selection" / "action_selection_model.pt")
    resolved_model_dir = resolved_model_path.parent
    resolved_metadata_path = metadata_path or (resolved_model_dir / "action_selection_metadata.json")
    resolved_scaler_path = scaler_path or (resolved_model_dir / "action_selection_feature_scaler.pkl")
    return resolved_model_path, resolved_metadata_path, resolved_scaler_path


def _load_action_selection_artifacts(
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
) -> tuple[ActionSelectionNet, Any, Dict[str, Any]]:
    resolved_model_path, resolved_metadata_path, resolved_scaler_path = _resolve_action_selection_artifact_paths(
        model_path=model_path,
        metadata_path=metadata_path,
        scaler_path=scaler_path,
    )
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Action-selection model not found: {resolved_model_path}")
    if not resolved_metadata_path.exists():
        raise FileNotFoundError(f"Action-selection metadata not found: {resolved_metadata_path}")
    if not resolved_scaler_path.exists():
        raise FileNotFoundError(f"Action-selection scaler not found: {resolved_scaler_path}")

    with resolved_metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    feature_columns = list(metadata.get("feature_columns") or [])
    n_actions = int(metadata.get("n_actions") or 3)
    if not feature_columns:
        raise ValueError(f"Action-selection metadata missing feature_columns: {resolved_metadata_path}")

    with resolved_scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)

    payload = torch.load(resolved_model_path, map_location="cpu", weights_only=False)
    if isinstance(payload, ActionSelectionNet):
        model = payload
        model.eval()
        return model, scaler, metadata

    if isinstance(payload, dict) and "model_state_dict" in payload:
        model_config = payload.get("model_config") or {}
        n_features = int(model_config.get("n_features") or len(feature_columns))
        n_actions = int(model_config.get("n_actions") or n_actions)
        model = ActionSelectionNet(n_features=n_features, n_actions=n_actions)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model, scaler, metadata

    if isinstance(payload, dict):
        model = ActionSelectionNet(n_features=len(feature_columns), n_actions=n_actions)
        model.load_state_dict(payload)
        model.eval()
        return model, scaler, metadata

    raise TypeError(f"Unsupported action-selection model payload type: {type(payload)!r}")


def _normalize_open_play_mask(frame: pd.DataFrame) -> pd.Series:
    if "set_piece" not in frame.columns:
        return pd.Series(True, index=frame.index)
    normalized = (
        frame["set_piece"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
        .replace(
            {
                "openplay": "open_play",
                "corner_kick": "corner",
                "freekick": "free_kick",
                "throwin": "throw_in",
                "goalkick": "goal_kick",
                "kickoff": "kick_off",
            }
        )
    )
    return normalized.eq("open_play") | normalized.isna()


def score_action_selection_for_causal_decisions(
    decision_points_df: pd.DataFrame,
    tracking_window: Optional[pd.DataFrame] = None,
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
    baseline_xg_artifacts: Optional[BaselineXGArtifacts] = None,
    open_play_only: bool = True,
) -> pd.DataFrame:
    if decision_points_df is None:
        raise ValueError("decision_points_df is None")

    resolved_model_path, resolved_metadata_path, resolved_scaler_path = _resolve_action_selection_artifact_paths(
        model_path=model_path,
        metadata_path=metadata_path,
        scaler_path=scaler_path,
    )
    model, scaler, metadata = _load_action_selection_artifacts(
        model_path=resolved_model_path,
        metadata_path=resolved_metadata_path,
        scaler_path=resolved_scaler_path,
    )
    feature_columns = list(metadata.get("feature_columns") or [])

    scored = decision_points_df.copy()
    if open_play_only:
        keep_mask = _normalize_open_play_mask(scored)
        scored = scored[keep_mask].copy()

    if scored.empty:
        empty = scored.copy()
        for column in (
            "p_action_pass",
            "p_action_ball_drive",
            "p_action_shot",
            "p_continue",
            "p_shot",
            "p_pass",
            "p_drive",
            "support_continue_tau_002",
            "support_continue_tau_005",
            "support_continue_tau_010",
            "common_support_shot",
            "as_feature_missing_count",
            "as_feature_missing_rate",
        ):
            empty[column] = pd.Series(dtype=float)
        return empty

    baseline_artifacts = baseline_xg_artifacts
    if baseline_artifacts is None:
        baseline_model_path = REPO_ROOT / "results" / "models" / "shot" / "baseline_xg_model.pkl"
        baseline_scaler_path = REPO_ROOT / "results" / "models" / "shot" / "baseline_xg_feature_scaler.pkl"
        if baseline_model_path.exists() and baseline_scaler_path.exists():
            baseline_artifacts = load_baseline_xg(baseline_model_path, baseline_scaler_path)

    from .features import ActionSelectionFeatureBuilder, ActionSelectionFeatureConfig

    feature_builder = ActionSelectionFeatureBuilder(
        config=ActionSelectionFeatureConfig(),
        baseline_xg_artifacts=baseline_artifacts,
    )
    feature_df, audit_df = feature_builder.build_feature_frame(
        scored,
        tracking_df=tracking_window,
        baseline_xg_artifacts=baseline_artifacts,
        return_audit=True,
    )

    if feature_columns:
        missing_columns = [column for column in feature_columns if column not in feature_df.columns]
        if missing_columns:
            raise ValueError(f"Action-selection feature frame missing columns: {missing_columns}")
        ordered_feature_df = feature_df.loc[:, feature_columns]
    else:
        ordered_feature_df = feature_df
    matrix = scaler.transform(ordered_feature_df.to_numpy(dtype=float))
    tensor = torch.tensor(matrix, dtype=torch.float32)
    temperature = float(metadata.get("temperature") or (metadata.get("calibration") or {}).get("temperature") or 1.0)
    with torch.no_grad():
        logits = model(tensor) / max(float(temperature), 1e-6)
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

    result = scored.copy()
    result["p_action_pass"] = probabilities[:, 0]
    result["p_action_ball_drive"] = probabilities[:, 1]
    result["p_action_shot"] = probabilities[:, 2]
    result["p_continue"] = result["p_action_pass"] + result["p_action_ball_drive"]
    result["p_shot"] = result["p_action_shot"]
    result["p_pass"] = result["p_action_pass"]
    result["p_drive"] = result["p_action_ball_drive"]
    result["support_continue_tau_002"] = (result["p_continue"] > 0.02).astype(int)
    result["support_continue_tau_005"] = (result["p_continue"] > 0.05).astype(int)
    result["support_continue_tau_010"] = (result["p_continue"] > 0.10).astype(int)
    result["common_support_shot"] = ((result["p_shot"] > 0.05) & (result["p_shot"] < 0.95)).astype(int)
    result["as_feature_missing_count"] = audit_df["feature_missing_count"].to_numpy(dtype=int)
    result["as_feature_missing_rate"] = audit_df["feature_missing_rate"].to_numpy(dtype=float)
    return result


def predict_action_selection_probs(
    state: Dict[str, Any],
    tracking_window: Optional[pd.DataFrame] = None,
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
    baseline_xg_artifacts: Optional[BaselineXGArtifacts] = None,
) -> Dict[str, float]:
    scored = score_action_selection_for_causal_decisions(
        pd.DataFrame([state]),
        tracking_window=tracking_window,
        model_path=model_path,
        metadata_path=metadata_path,
        scaler_path=scaler_path,
        baseline_xg_artifacts=baseline_xg_artifacts,
        open_play_only=False,
    )
    if scored.empty:
        raise ValueError("Action-selection scoring returned no rows.")

    row = scored.iloc[0]
    return {
        "p_pass": float(row["p_pass"]),
        "p_drive": float(row["p_drive"]),
        "p_shot": float(row["p_shot"]),
        "p_action_pass": float(row["p_action_pass"]),
        "p_action_ball_drive": float(row["p_action_ball_drive"]),
        "p_action_shot": float(row["p_action_shot"]),
        "p_continue": float(row["p_continue"]),
        "common_support_shot": int(row["common_support_shot"]),
    }
