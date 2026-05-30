from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

SHOT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = SHOT_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Shot.features import ShotFeatureBuilder
from Shot.trainer import ShotTrainer


def _make_tracking_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"match_id": 1, "frame_id": 10, "team_side": "home", "player_id": 1, "team_id": 11, "x": 48.0, "y": 0.0, "position_name": "goalkeeper"},
            {"match_id": 1, "frame_id": 10, "team_side": "away", "player_id": 2, "team_id": 22, "x": 38.0, "y": 4.0, "position_name": "defender"},
            {"match_id": 1, "frame_id": 11, "team_side": "away", "player_id": 3, "team_id": 22, "x": 40.0, "y": -3.0, "position_name": "goalkeeper"},
            {"match_id": 1, "frame_id": 11, "team_side": "home", "player_id": 4, "team_id": 11, "x": 42.0, "y": 1.5, "position_name": "defender"},
            {"match_id": 1, "frame_id": 12, "team_side": "home", "player_id": 5, "team_id": 11, "x": 30.0, "y": 0.0, "position_name": "goalkeeper"},
            {"match_id": 1, "frame_id": 12, "team_side": "away", "player_id": 6, "team_id": 22, "x": 41.0, "y": 2.0, "position_name": "defender"},
        ]
    )


def _make_shot_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": 1,
                "match_id": 1,
                "frame_id": 10,
                "team_id": 11,
                "player_id": 101,
                "team_side": "home",
                "ball_x": 36.0,
                "ball_y": 0.0,
                "body_part": "head",
                "set_piece": "open_play",
                "shot_outcome": "goal",
                "possession_type": "shot",
                "reward_norm": 1.0,
                "split": "train",
            },
            {
                "game_id": 1,
                "match_id": 1,
                "frame_id": 11,
                "team_id": 11,
                "player_id": 102,
                "team_side": "home",
                "ball_x": 40.0,
                "ball_y": 3.0,
                "body_part": "foot",
                "set_piece": "corner",
                "shot_outcome": "saved",
                "possession_type": "shot",
                "reward_norm": 0.5,
                "split": "val",
            },
            {
                "game_id": 1,
                "match_id": 1,
                "frame_id": 12,
                "team_id": 22,
                "player_id": 103,
                "team_side": "away",
                "ball_x": 34.0,
                "ball_y": -2.0,
                "body_part": "foot",
                "set_piece": "free_kick",
                "shot_outcome": "miss",
                "possession_type": "shot",
                "reward_norm": 0.0,
                "split": "test",
            },
        ]
    )


def main() -> None:
    shot_events = _make_shot_events()
    tracking = _make_tracking_frame()

    builder = ShotFeatureBuilder()
    features = builder.build_feature_frame(shot_events, tracking_df=tracking)
    if list(features.columns) != builder.feature_columns:
        raise AssertionError("Shot feature columns mismatch")
    if not features.replace([float("inf"), float("-inf")], pd.NA).notna().all().all():
        raise AssertionError("Shot feature frame has invalid values")

    # Smoke test training on a synthetic dataset instead of file-backed inputs.
    train_frame = pd.concat([shot_events.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    train_frame["reward_norm"] = shot_events["reward_norm"].astype(float)
    train_frame["split"] = shot_events["split"]

    trainer = ShotTrainer()
    output_dir = Path(__file__).resolve().parents[2] / "results" / "models" / "shot_smoke"
    report = trainer.run(train_frame, output_dir=output_dir)

    payload = {
        "status": "ok",
        "rows": int(len(train_frame)),
        "report": report,
        "output_dir": str(output_dir),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
