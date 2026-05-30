from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from Pass.data_utils import REPO_ROOT

from Shot.baseline_xg import BaselineXGArtifacts, load_baseline_xg

from .features import ActionSelectionFeatureBuilder
from .models import ActionSelectionNet


def _load_action_selection_model(model_path: Path, n_features: int, n_actions: int = 3) -> ActionSelectionNet:
    loaded = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(loaded, ActionSelectionNet):
        loaded.eval()
        return loaded

    model = ActionSelectionNet(n_features=n_features, n_actions=n_actions)
    if isinstance(loaded, dict):
        model.load_state_dict(loaded)
    else:
        raise TypeError(f"Unsupported action-selection model payload type: {type(loaded)!r}")
    model.eval()
    return model


def predict_action_selection_probs(
    state: Dict[str, Any],
    tracking_window: Optional[pd.DataFrame] = None,
    model_path: Optional[Path] = None,
    baseline_xg_artifacts: Optional[BaselineXGArtifacts] = None,
) -> Dict[str, float]:
    resolved_model_path = model_path or (REPO_ROOT / "results" / "models" / "action_selection" / "action_selection_net.pt")
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Action-selection model not found: {resolved_model_path}")

    baseline_artifacts = baseline_xg_artifacts
    if baseline_artifacts is None:
        baseline_model_path = REPO_ROOT / "results" / "models" / "shot" / "baseline_xg_model.pkl"
        baseline_scaler_path = REPO_ROOT / "results" / "models" / "shot" / "baseline_xg_feature_scaler.pkl"
        if baseline_model_path.exists() and baseline_scaler_path.exists():
            baseline_artifacts = load_baseline_xg(baseline_model_path, baseline_scaler_path)

    builder = ActionSelectionFeatureBuilder(baseline_xg_artifacts=baseline_artifacts)
    state_df = pd.DataFrame([state])
    feature_df = builder.build_feature_frame(state_df, tracking_df=tracking_window)

    model = _load_action_selection_model(resolved_model_path, n_features=feature_df.shape[1])
    x = torch.tensor(feature_df.to_numpy(dtype=np.float32), dtype=torch.float32)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1)

    return {
        "p_pass": float(probabilities[0]),
        "p_drive": float(probabilities[1]),
        "p_shot": float(probabilities[2]),
    }
