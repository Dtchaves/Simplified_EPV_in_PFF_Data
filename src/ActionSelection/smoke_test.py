from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ACTION_ROOT = Path(__file__).resolve().parent
SRC_ROOT = ACTION_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ActionSelection.data import build_action_selection_dataset, load_default_baseline_xg_artifacts
from ActionSelection.features import ActionSelectionFeatureBuilder
from ActionSelection.inference import predict_action_selection_probs, score_action_selection_for_causal_decisions
from ActionSelection.data import ActionSelectionDataConfig, build_action_selection_frames
from ActionSelection.trainer import train_action_selection


def _make_tracking_frame() -> pd.DataFrame:
    frame_specs = {
        98: {"elapsed_seconds": 0.0, "ball_x": 28.0, "ball_y": 0.0},
        99: {"elapsed_seconds": 0.5, "ball_x": 30.0, "ball_y": 0.0},
        100: {"elapsed_seconds": 1.0, "ball_x": 32.0, "ball_y": 0.0},
        101: {"elapsed_seconds": 1.5, "ball_x": 40.0, "ball_y": -2.0},
        102: {"elapsed_seconds": 2.0, "ball_x": 34.0, "ball_y": 1.0},
        103: {"elapsed_seconds": 2.5, "ball_x": 38.0, "ball_y": 2.0},
        104: {"elapsed_seconds": 3.0, "ball_x": 44.0, "ball_y": -1.5},
        105: {"elapsed_seconds": 3.5, "ball_x": 47.0, "ball_y": 3.0},
    }
    rows = []
    for frame_id, spec in frame_specs.items():
        rows.extend(
            [
                {
                    "match_id": 1,
                    "frame_id": frame_id,
                    "period": 1,
                    "elapsed_seconds": spec["elapsed_seconds"],
                    "ball_x": spec["ball_x"],
                    "ball_y": spec["ball_y"],
                    "team_side": "home",
                    "player_id": 1 + frame_id,
                    "team_id": 10,
                    "x": spec["ball_x"] - 2.0,
                    "y": spec["ball_y"] + 1.0,
                    "position_name": "midfielder",
                },
                {
                    "match_id": 1,
                    "frame_id": frame_id,
                    "period": 1,
                    "elapsed_seconds": spec["elapsed_seconds"],
                    "ball_x": spec["ball_x"],
                    "ball_y": spec["ball_y"],
                    "team_side": "home",
                    "player_id": 101 + frame_id,
                    "team_id": 10,
                    "x": spec["ball_x"] - 6.0,
                    "y": spec["ball_y"] - 3.0,
                    "position_name": "defender",
                },
                {
                    "match_id": 1,
                    "frame_id": frame_id,
                    "period": 1,
                    "elapsed_seconds": spec["elapsed_seconds"],
                    "ball_x": spec["ball_x"],
                    "ball_y": spec["ball_y"],
                    "team_side": "away",
                    "player_id": 201 + frame_id,
                    "team_id": 20,
                    "x": spec["ball_x"] + 3.0,
                    "y": spec["ball_y"] + 2.0,
                    "position_name": "midfielder",
                },
                {
                    "match_id": 1,
                    "frame_id": frame_id,
                    "period": 1,
                    "elapsed_seconds": spec["elapsed_seconds"],
                    "ball_x": spec["ball_x"],
                    "ball_y": spec["ball_y"],
                    "team_side": "away",
                    "player_id": 301 + frame_id,
                    "team_id": 20,
                    "x": spec["ball_x"] + 8.0,
                    "y": spec["ball_y"] - 1.5,
                    "position_name": "defender",
                },
            ]
        )
    return pd.DataFrame(rows)


def _make_actions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_id": 1, "match_id": 1, "frame_id": 100, "team_side": "home", "ball_x": 32.0, "ball_y": 0.0, "action_label": "pass", "set_piece": "open_play"},
            {"game_id": 1, "match_id": 1, "frame_id": 101, "team_side": "home", "ball_x": 40.0, "ball_y": -2.0, "action_label": "ball_drive", "set_piece": "open_play"},
            {"game_id": 1, "match_id": 1, "frame_id": 102, "team_side": "away", "ball_x": 34.0, "ball_y": 1.0, "action_label": "shot", "set_piece": "open_play"},
            {"game_id": 1, "match_id": 1, "frame_id": 103, "team_side": "away", "ball_x": 38.0, "ball_y": 2.0, "action_label": "pass", "set_piece": "open_play"},
            {"game_id": 1, "match_id": 1, "frame_id": 104, "team_side": "home", "ball_x": 44.0, "ball_y": -1.5, "action_label": "carry", "set_piece": "open_play"},
            {"game_id": 1, "match_id": 1, "frame_id": 105, "team_side": "home", "ball_x": 47.0, "ball_y": 3.0, "action_label": "shot", "set_piece": "free_kick"},
        ]
    )


def main() -> None:
    data_config = ActionSelectionDataConfig(max_sources=1)
    actions, tracking, frame_summary = build_action_selection_frames(data_config)
    baseline_xg_artifacts = load_default_baseline_xg_artifacts()
    if baseline_xg_artifacts is None:
        raise AssertionError("Baseline xG artifacts are required for the ActionSelection smoke test")

    builder = ActionSelectionFeatureBuilder()
    features = builder.build_feature_frame(actions, tracking_df=tracking, baseline_xg_artifacts=baseline_xg_artifacts)
    if list(features.columns) != builder.feature_columns:
        raise AssertionError("ActionSelection feature columns mismatch")
    if not features.replace([float("inf"), float("-inf")], pd.NA).notna().all().all():
        raise AssertionError("ActionSelection feature frame has invalid values")
    if float(features["pitch_control_attacking_team_at_ball"].std(ddof=0)) <= 0.0:
        raise AssertionError("ActionSelection pitch-control feature collapsed in smoke test")

    dataset = build_action_selection_dataset(actions, tracking_df=tracking, baseline_xg_artifacts=baseline_xg_artifacts)
    if "label" not in dataset.columns:
        raise AssertionError("ActionSelection dataset missing label column")
    if len(dataset) != len(actions):
        raise AssertionError("ActionSelection dataset row count mismatch")

    scored = score_action_selection_for_causal_decisions(
        actions,
        tracking_window=tracking,
        baseline_xg_artifacts=baseline_xg_artifacts,
        open_play_only=True,
    )
    expected_probability_columns = {
        "p_action_pass",
        "p_action_ball_drive",
        "p_action_shot",
        "p_continue",
        "p_shot",
        "common_support_shot",
        "support_continue_tau_002",
        "support_continue_tau_005",
        "support_continue_tau_010",
        "as_feature_missing_count",
        "as_feature_missing_rate",
    }
    if not expected_probability_columns.issubset(set(scored.columns)):
        raise AssertionError(f"ActionSelection scorer missing expected columns: {expected_probability_columns - set(scored.columns)}")
    if len(scored) != len(actions):
        raise AssertionError(f"Expected scorer to keep all open-play rows, got {len(scored)} of {len(actions)}")
    if not scored["p_continue"].between(0.0, 1.0).all():
        raise AssertionError("ActionSelection p_continue outside [0, 1]")
    if float(scored["common_support_shot"].mean()) <= 0.0:
        raise AssertionError("ActionSelection common support collapsed in smoke test")

    compat = predict_action_selection_probs(
        actions.iloc[0].to_dict(),
        tracking_window=tracking,
        baseline_xg_artifacts=baseline_xg_artifacts,
    )
    for key in ("p_pass", "p_drive", "p_shot", "p_action_pass", "p_action_ball_drive", "p_action_shot", "p_continue", "common_support_shot"):
        if key not in compat:
            raise AssertionError(f"ActionSelection compatibility prediction missing {key}")

    output_dir = Path(__file__).resolve().parents[2] / "results" / "models" / "action_selection_smoke"
    report_path = train_action_selection(actions, tracking_df=tracking, baseline_xg_artifacts=baseline_xg_artifacts, output_dir=output_dir)

    payload = {
        "status": "ok",
        "rows": int(len(dataset)),
        "output_dir": str(report_path),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
