from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from inference import predict_ball_drive_epv


def main() -> None:
    sample_state = {
        "game_id": 0,
        "game_event_id": 0,
        "possession_event_id": 0,
        "team_side": "home",
        "ball_x_start": 0.0,
        "ball_y_start": 0.0,
        "segment_start_frame_id": 0,
        "start_frame_id": 0,
    }
    empty_tracking = pd.DataFrame(columns=["match_id", "frame_id", "team_side", "player_id", "x", "y", "ball_x", "ball_y", "elapsed_seconds"])
    pred = predict_ball_drive_epv(sample_state, tracking_window=empty_tracking)

    expected_keys = ["p_drive_success", "v_drive_success", "v_drive_failed", "drive_epv"]
    missing_keys = [key for key in expected_keys if key not in pred]
    finite_keys = {
        key: bool(isinstance(pred.get(key), (int, float)) and math.isfinite(float(pred.get(key))))
        for key in expected_keys
        if key in pred
    }
    pred["expected_keys_present"] = len(missing_keys) == 0
    pred["missing_keys"] = missing_keys
    pred["finite_keys"] = finite_keys

    out_path = Path(__file__).resolve().parents[2] / "results" / "metrics" / "ball_drive_inference_smoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(pred, handle, indent=2)

    print(json.dumps(pred, indent=2))


if __name__ == "__main__":
    main()
