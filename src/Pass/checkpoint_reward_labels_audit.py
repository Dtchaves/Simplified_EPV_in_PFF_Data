from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd

from data_utils import discover_pass_sources, load_or_build_canonical_pass_cache
from reward_labels import GameEventIndex, PassRewardLabeler


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = [
    "game_id",
    "game_event_id",
    "possession_event_id",
    "player_id",
    "ball_x_start",
    "ball_y_start",
    "ball_x_end",
    "ball_y_end",
    "team_id",
]


def _build_synthetic_pass_event(
    game_event_id: int,
    possession_event_id: int,
    event_time: float,
    team_id: int,
    home_team: bool,
    setpiece_type,
) -> Dict:
    return {
        "GAME_EVENT_ID": game_event_id,
        "POSSESSION_EVENT_ID": float(possession_event_id),
        "EVENT_TIME": float(event_time),
        "GAME_EVENTS": {
            "GAME_EVENT_TYPE": "OTB",
            "OUT_TYPE": None,
            "TEAM_ID": int(team_id),
            "HOME_TEAM": bool(home_team),
            "SETPIECE_TYPE": setpiece_type,
            "START_GAME_CLOCK": int(event_time),
        },
        "POSSESSION_EVENTS": {
            "POSSESSION_EVENT_TYPE": "PA",
            "EVENT_GAME_CLOCK": int(event_time),
        },
    }


def _build_synthetic_goal_event(game_event_id: int, event_time: float, out_type: str) -> Dict:
    return {
        "GAME_EVENT_ID": game_event_id,
        "POSSESSION_EVENT_ID": None,
        "EVENT_TIME": float(event_time),
        "GAME_EVENTS": {
            "GAME_EVENT_TYPE": "OUT",
            "OUT_TYPE": out_type,
            "TEAM_ID": None,
            "HOME_TEAM": None,
            "SETPIECE_TYPE": None,
            "START_GAME_CLOCK": int(event_time),
        },
        "POSSESSION_EVENTS": {
            "POSSESSION_EVENT_TYPE": None,
            "EVENT_GAME_CLOCK": None,
        },
    }


def run_boundary_and_open_play_checks() -> None:
    records = [
        _build_synthetic_pass_event(1, 1, 100.0, team_id=10, home_team=True, setpiece_type="O"),
        _build_synthetic_pass_event(2, 2, 90.0, team_id=20, home_team=False, setpiece_type="O"),
        _build_synthetic_goal_event(11, 100.0, "H"),
        _build_synthetic_goal_event(12, 100.0001, "H"),
        _build_synthetic_goal_event(13, 115.0001, "A"),
    ]
    index = GameEventIndex(records, horizon_seconds=15.0)

    pass_row = pd.Series(
        {
            "game_event_id": 1,
            "possession_event_id": 1,
            "team_id": 10,
        }
    )
    label, metadata = index.label_pass_row(pass_row, include_open_play_null=True)
    assert label == 1, f"Expected +1 boundary label, got {label} with metadata={metadata}"
    assert metadata["first_goal_time"] == 100.0001, (
        f"Expected strict lower boundary behavior, got metadata={metadata}"
    )

    records_upper = [
        _build_synthetic_pass_event(3, 3, 200.0, team_id=10, home_team=True, setpiece_type="O"),
        _build_synthetic_pass_event(4, 4, 210.0, team_id=20, home_team=False, setpiece_type="O"),
        _build_synthetic_goal_event(14, 215.0, "H"),
    ]
    index_upper = GameEventIndex(records_upper, horizon_seconds=15.0)
    label_upper, metadata_upper = index_upper.label_pass_row(
        pd.Series({"game_event_id": 3, "possession_event_id": 3, "team_id": 10}),
        include_open_play_null=True,
    )
    assert label_upper == 1, f"Expected inclusive t+15 boundary, got {label_upper} {metadata_upper}"

    records_open_play = [
        _build_synthetic_pass_event(5, 5, 300.0, team_id=10, home_team=True, setpiece_type=None),
        _build_synthetic_pass_event(6, 6, 300.0, team_id=10, home_team=True, setpiece_type="F"),
        _build_synthetic_pass_event(7, 7, 305.0, team_id=20, home_team=False, setpiece_type="O"),
        _build_synthetic_goal_event(15, 304.0, "H"),
    ]
    index_open_play = GameEventIndex(records_open_play, horizon_seconds=15.0)

    label_null, metadata_null = index_open_play.label_pass_row(
        pd.Series({"game_event_id": 5, "possession_event_id": 5, "team_id": 10}),
        include_open_play_null=True,
    )
    assert label_null == 1, f"Expected NULL setpiece treated as open play, got {label_null} {metadata_null}"

    label_setpiece, metadata_setpiece = index_open_play.label_pass_row(
        pd.Series({"game_event_id": 6, "possession_event_id": 6, "team_id": 10}),
        include_open_play_null=True,
    )
    assert label_setpiece is None and metadata_setpiece["status"] == "setpiece_filtered", (
        f"Expected non-open-play filtering, got {label_setpiece} {metadata_setpiece}"
    )


def run_reward_checkpoint() -> None:
    run_boundary_and_open_play_checks()

    sources = discover_pass_sources("data/passes", source_format="auto")
    if not sources:
        raise FileNotFoundError("No canonical pass sources found under data/passes.")

    labeler = PassRewardLabeler(
        event_root=ROOT / "data/raw/event",
        horizon_seconds=15.0,
        include_open_play_null=True,
    )

    label_counter: Counter = Counter()
    status_counter: Counter = Counter()
    join_counter: Counter = Counter()
    per_file_rows: List[Dict] = []
    sample_rows: List[Dict] = []

    for source in sources:
        source_path = Path(source["source_path"])
        source_df, _ = load_or_build_canonical_pass_cache(
            source=source,
            required_columns=REQUIRED_COLUMNS,
            source_filename=source.get("source_name"),
            event_root="data/raw/event",
        )
        source_df = source_df[source_df["pass_outcome_type"].notna()].copy()

        labeled_df, summary = labeler.label_pass_dataframe(
            source_df,
            source_filename=source.get("source_name") or source_path.name,
            drop_unlabeled=False,
            source_kind=str(source.get("source_kind", "legacy_wide")),
            processed_events_path=source.get("events_path"),
            processed_tracking_path=source.get("tracking_path"),
        )

        valid_labels = labeled_df["reward_label"].dropna().astype(int)
        unique_values = set(valid_labels.unique().tolist())
        assert unique_values.issubset({-1, 0, 1}), (
            f"Unexpected reward labels in {source.get('source_name') or source_path.name}: {sorted(unique_values)}"
        )

        label_counter.update(valid_labels.tolist())
        status_counter.update(labeled_df["reward_status"].tolist())
        join_counter.update(labeled_df["reward_join_strategy"].tolist())

        per_file_rows.append(
            {
                "file": source.get("source_name") or source_path.name,
                "game_id": summary["game_id"],
                "rows_total": summary["rows_total"],
                "rows_labeled": summary["rows_labeled"],
                "rows_dropped": summary["rows_dropped"],
                "label_pos_1": int((valid_labels == 1).sum()),
                "label_neg_1": int((valid_labels == -1).sum()),
                "label_zero": int((valid_labels == 0).sum()),
            }
        )

        labeled_examples = labeled_df[labeled_df["reward_label"].notna()].head(5).copy()
        if not labeled_examples.empty:
            labeled_examples["source_file"] = source.get("source_name") or source_path.name
            sample_rows.extend(
                labeled_examples[
                    [
                        "source_file",
                        "game_id",
                        "team_id",
                        "game_event_id",
                        "possession_event_id",
                        "reward_label",
                        "reward_status",
                        "reward_join_strategy",
                        "reward_pass_event_time",
                        "reward_first_goal_time",
                        "reward_first_goal_out_type",
                        "reward_pass_setpiece_type",
                    ]
                ].to_dict("records")
            )

    total_labeled = int(sum(label_counter.values()))
    assert total_labeled > 0, "No labeled rows were produced by reward labeling."

    output_dir = ROOT / "results/metrics/reward_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_file_df = pd.DataFrame(per_file_rows).sort_values(by=["file"])
    per_file_df.to_csv(output_dir / "reward_label_file_summary.csv", index=False)

    pd.DataFrame(
        {
            "label": [-1, 0, 1],
            "count": [int(label_counter.get(-1, 0)), int(label_counter.get(0, 0)), int(label_counter.get(1, 0))],
        }
    ).to_csv(output_dir / "reward_label_distribution.csv", index=False)

    if sample_rows:
        pd.DataFrame(sample_rows).to_csv(output_dir / "reward_label_examples.csv", index=False)

    summary_payload = {
        "files_scanned": len(sources),
        "rows_labeled_total": total_labeled,
        "label_counts": {str(k): int(v) for k, v in sorted(label_counter.items())},
        "status_counts": {str(k): int(v) for k, v in status_counter.items()},
        "join_strategy_counts": {str(k): int(v) for k, v in join_counter.items()},
        "outputs": {
            "file_summary_csv": str(output_dir / "reward_label_file_summary.csv"),
            "distribution_csv": str(output_dir / "reward_label_distribution.csv"),
            "examples_csv": str(output_dir / "reward_label_examples.csv"),
        },
    }

    with (output_dir / "reward_label_summary.json").open("w", encoding="utf-8") as out_file:
        json.dump(summary_payload, out_file, indent=2)

    print("REWARD_CHECKPOINT_AUDIT_OK")
    print(f"files_scanned={len(sources)}")
    print(f"rows_labeled_total={total_labeled}")
    print(f"label_counts={summary_payload['label_counts']}")
    print(f"status_counts={summary_payload['status_counts']}")
    print(f"join_strategy_counts={summary_payload['join_strategy_counts']}")
    print(f"artifacts={output_dir}")


if __name__ == "__main__":
    run_reward_checkpoint()
