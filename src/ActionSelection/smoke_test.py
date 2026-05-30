from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ACTION_ROOT = Path(__file__).resolve().parent
SRC_ROOT = ACTION_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ActionSelection.data import build_action_selection_dataset
from ActionSelection.features import ActionSelectionFeatureBuilder
from ActionSelection.trainer import train_action_selection


def _make_tracking_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"match_id": 1, "frame_id": 100, "team_side": "home", "player_id": 1, "team_id": 10, "x": 30.0, "y": 1.0, "position_name": "midfielder"},
            {"match_id": 1, "frame_id": 100, "team_side": "away", "player_id": 2, "team_id": 20, "x": 40.0, "y": 2.0, "position_name": "defender"},
            {"match_id": 1, "frame_id": 101, "team_side": "home", "player_id": 3, "team_id": 10, "x": 42.0, "y": -4.0, "position_name": "midfielder"},
            {"match_id": 1, "frame_id": 101, "team_side": "away", "player_id": 4, "team_id": 20, "x": 45.0, "y": -3.0, "position_name": "defender"},
            {"match_id": 1, "frame_id": 102, "team_side": "away", "player_id": 5, "team_id": 20, "x": 36.0, "y": 1.0, "position_name": "defender"},
            {"match_id": 1, "frame_id": 102, "team_side": "home", "player_id": 6, "team_id": 10, "x": 38.0, "y": 0.5, "position_name": "midfielder"},
        ]
    )


def _make_actions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_id": 1, "match_id": 1, "frame_id": 100, "team_side": "home", "ball_x": 32.0, "ball_y": 0.0, "action_label": "pass", "set_piece": "open_play"},
            {"game_id": 1, "match_id": 1, "frame_id": 101, "team_side": "home", "ball_x": 40.0, "ball_y": -2.0, "action_label": "ball_drive", "set_piece": "open_play"},
            {"game_id": 1, "match_id": 1, "frame_id": 102, "team_side": "away", "ball_x": 34.0, "ball_y": 1.0, "action_label": "shot", "set_piece": "corner"},
            {"game_id": 1, "match_id": 1, "frame_id": 103, "team_side": "away", "ball_x": 38.0, "ball_y": 2.0, "action_label": "pass", "set_piece": "open_play"},
            {"game_id": 1, "match_id": 1, "frame_id": 104, "team_side": "home", "ball_x": 44.0, "ball_y": -1.5, "action_label": "carry", "set_piece": "open_play"},
            {"game_id": 1, "match_id": 1, "frame_id": 105, "team_side": "home", "ball_x": 47.0, "ball_y": 3.0, "action_label": "shot", "set_piece": "free_kick"},
        ]
    )


def main() -> None:
    actions = _make_actions_frame()
    tracking = _make_tracking_frame()

    builder = ActionSelectionFeatureBuilder()
    features = builder.build_feature_frame(actions, tracking_df=tracking)
    if list(features.columns) != builder.feature_columns:
        raise AssertionError("ActionSelection feature columns mismatch")
    if not features.replace([float("inf"), float("-inf")], pd.NA).notna().all().all():
        raise AssertionError("ActionSelection feature frame has invalid values")

    dataset = build_action_selection_dataset(actions, tracking_df=tracking)
    if "label" not in dataset.columns:
        raise AssertionError("ActionSelection dataset missing label column")

    output_dir = Path(__file__).resolve().parents[2] / "results" / "models" / "action_selection_smoke"
    report_path = train_action_selection(actions, tracking_df=tracking, output_dir=output_dir)

    payload = {
        "status": "ok",
        "rows": int(len(dataset)),
        "output_dir": str(report_path),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
