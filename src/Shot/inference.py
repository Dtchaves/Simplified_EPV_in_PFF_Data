from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from Pass.data_utils import REPO_ROOT

from .baseline_xg import BaselineXGArtifacts, load_baseline_xg
from .features import ShotFeatureBuilder
from .models import ShotEPVNet


def _load_shot_model(model_path: Path, n_features: int) -> ShotEPVNet:
    loaded = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(loaded, ShotEPVNet):
        loaded.eval()
        return loaded

    model = ShotEPVNet(n_features=n_features)
    if isinstance(loaded, dict):
        model.load_state_dict(loaded)
    else:
        raise TypeError(f"Unsupported shot model payload type: {type(loaded)!r}")
    model.eval()
    return model


def predict_shot_epv(
    state: Dict[str, Any],
    tracking_window: Optional[pd.DataFrame] = None,
    model_path: Optional[Path] = None,
    baseline_xg_artifacts: Optional[BaselineXGArtifacts] = None,
) -> Dict[str, float]:
    resolved_model_path = model_path or (REPO_ROOT / "results" / "models" / "shot" / "shot_epv.pt")
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Shot EPV model not found: {resolved_model_path}")

    baseline_artifacts = baseline_xg_artifacts
    if baseline_artifacts is None:
        baseline_model_path = REPO_ROOT / "results" / "models" / "shot" / "baseline_xg_model.pkl"
        baseline_scaler_path = REPO_ROOT / "results" / "models" / "shot" / "baseline_xg_feature_scaler.pkl"
        if baseline_model_path.exists() and baseline_scaler_path.exists():
            baseline_artifacts = load_baseline_xg(baseline_model_path, baseline_scaler_path)

    builder = ShotFeatureBuilder()
    state_df = pd.DataFrame([state])
    feature_df = builder.build_feature_frame(state_df, tracking_df=tracking_window, baseline_xg_artifacts=baseline_artifacts)

    model = _load_shot_model(resolved_model_path, n_features=feature_df.shape[1])
    x = torch.tensor(feature_df.to_numpy(dtype=np.float32), dtype=torch.float32)

    with torch.no_grad():
        y_hat_norm = float(model(x).cpu().numpy().reshape(-1)[0])

    return {
        "y_hat_norm": float(y_hat_norm),
        "shot_epv": float((2.0 * y_hat_norm) - 1.0),
        "baseline_xg": float(feature_df.iloc[0].get("baseline_xg", 0.0)),
    }
