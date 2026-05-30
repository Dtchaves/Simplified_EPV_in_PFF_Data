from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

BALL_DRIVE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = BALL_DRIVE_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Pass.data_utils import REPO_ROOT

try:
    from .features import BallDriveFeatureBuilder
    from .models import BallDriveDEModel, BallDriveDPModel
except ImportError:
    from features import BallDriveFeatureBuilder  # type: ignore
    from models import BallDriveDEModel, BallDriveDPModel  # type: ignore


def _latest_checkpoint(pattern: str) -> Optional[Path]:
    candidates = sorted((REPO_ROOT / "results" / "models" / "ball_drive").glob(pattern))
    return candidates[-1] if candidates else None


def _default_paths() -> Dict[str, Optional[Path]]:
    model_root = REPO_ROOT / "results" / "models" / "ball_drive"
    return {
        "dp": model_root / "ball_drive_dp.pt",
        "de_success": model_root / "ball_drive_de_success.pt",
        "de_failed": model_root / "ball_drive_de_failed.pt",
        "scaler": model_root / "ball_drive_scaler.pkl",
    }


def predict_ball_drive_epv(
    state: Dict[str, Any],
    tracking_window: Optional[pd.DataFrame] = None,
    paths: Optional[Dict[str, Path]] = None,
    hidden_dim: int = 64,
) -> Dict[str, float]:
    resolved = _default_paths()
    if paths:
        resolved.update(paths)

    scaler_path = resolved.get("scaler")
    if scaler_path is None or not scaler_path.exists():
        raise FileNotFoundError("Missing BallDrive scaler bundle. Train model first.")

    builder = BallDriveFeatureBuilder.load(scaler_path)
    state_df = pd.DataFrame([state])

    feature_df = builder.build_feature_frame(state_df, tracking_df=tracking_window)
    x_dp = builder.transform(feature_df)

    dp_model = BallDriveDPModel(input_dim=x_dp.shape[1], hidden_dim=hidden_dim)
    dp_path = resolved.get("dp")
    if dp_path is None or not dp_path.exists():
        raise FileNotFoundError("Missing DP checkpoint. Train model first.")
    dp_model.load_state_dict(torch.load(dp_path, map_location="cpu"))
    dp_model.eval()

    with torch.no_grad():
        p_success = float(dp_model(torch.tensor(x_dp, dtype=torch.float32)).cpu().numpy().reshape(-1)[0])

    feature_df_with_p = feature_df.copy()
    feature_df_with_p["p_drive_success"] = p_success
    x_de = builder.transform(feature_df_with_p, include_p_drive_success=True)

    def _predict_de(path_key: str) -> float:
        path = resolved.get(path_key)
        if path is None or not path.exists():
            return float("nan")
        model = BallDriveDEModel(input_dim=x_de.shape[1], hidden_dim=hidden_dim)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            return float(model(torch.tensor(x_de, dtype=torch.float32)).cpu().numpy().reshape(-1)[0])

    v_success = _predict_de("de_success")
    v_failed = _predict_de("de_failed")

    if np.isnan(v_success) or np.isnan(v_failed):
        drive_epv = float("nan")
    else:
        drive_epv = float((p_success * v_success) + ((1.0 - p_success) * v_failed))

    return {
        "p_drive_success": float(p_success),
        "v_drive_success": float(v_success),
        "v_drive_failed": float(v_failed),
        "drive_epv": float(drive_epv),
    }
