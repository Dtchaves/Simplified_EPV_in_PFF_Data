from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pandas as pd

from data_utils import discover_pass_sources, load_or_build_canonical_pass_cache
from reward_labels import PassRewardLabeler

try:
    from Pass_sucess_probability.utils import ToSoccerMapTensor
except ImportError:
    from .Pass_sucess_probability.utils import ToSoccerMapTensor


def run_audit() -> None:
    required_cols = [
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

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        match_dir = root / "4436"
        match_dir.mkdir(parents=True, exist_ok=True)

        events = pd.DataFrame(
            [
                {
                    "match_id": 4436,
                    "event_id": 101,
                    "possession_id": 1001,
                    "team_id": 10,
                    "player_id": 100,
                    "possession_type": "pass",
                    "set_piece": "open_play",
                    "frame_id": 500,
                    "ball_x": -5.0,
                    "ball_y": 3.0,
                    "pass_outcome": "completed",
                    "elapsed_seconds": 100.0,
                },
                {
                    "match_id": 4436,
                    "event_id": 102,
                    "possession_id": 1002,
                    "team_id": 20,
                    "player_id": 200,
                    "possession_type": "shot",
                    "set_piece": "open_play",
                    "frame_id": 501,
                    "shot_outcome": "goal",
                    "elapsed_seconds": 110.0,
                },
            ]
        )
        events.to_parquet(match_dir / "events.parquet", index=False)

        tracking = pd.DataFrame(
            [
                {
                    "match_id": 4436,
                    "frame_id": 500,
                    "team_id": 10,
                    "player_id": 100,
                    "x": -6.0,
                    "y": 2.0,
                    "vx": 0.2,
                    "vy": 0.1,
                    "ball_x": -5.0,
                    "ball_y": 3.0,
                    "elapsed_seconds": 100.0,
                },
                {
                    "match_id": 4436,
                    "frame_id": 500,
                    "team_id": 20,
                    "player_id": 200,
                    "x": -1.0,
                    "y": -2.0,
                    "vx": -0.1,
                    "vy": 0.0,
                    "ball_x": -5.0,
                    "ball_y": 3.0,
                    "elapsed_seconds": 100.0,
                },
            ]
        )
        tracking.to_parquet(match_dir / "tracking.parquet", index=False)

        players = pd.DataFrame([{"player_id": 100}, {"player_id": 200}])
        players.to_parquet(match_dir / "players.parquet", index=False)

        sources = discover_pass_sources(root, source_format="auto")
        assert len(sources) == 1, f"Expected one source, got {len(sources)}"
        assert sources[0]["source_kind"] == "pff_match_triplets", sources[0]["source_kind"]

        canonical_df, canonical_summary = load_or_build_canonical_pass_cache(
            source=sources[0],
            required_columns=required_cols,
            source_filename=sources[0]["source_name"],
        )
        assert canonical_summary.get("cache_hit") is False

        cached_df, cached_summary = load_or_build_canonical_pass_cache(
            source=sources[0],
            required_columns=required_cols,
            source_filename=sources[0]["source_name"],
        )
        assert cached_summary.get("cache_hit") is True
        assert len(cached_df) == len(canonical_df)

        assert len(canonical_df) == 1, f"Expected one canonical pass row, got {len(canonical_df)}"
        for col in required_cols + ["pass_outcome_type", "x_player_1", "original_pId_player_1"]:
            assert col in canonical_df.columns, f"Missing canonical column: {col}"

        labeler = PassRewardLabeler(event_root="data/raw/event", horizon_seconds=15.0, include_open_play_null=True)
        labeled_df, label_summary = labeler.label_pass_dataframe(
            canonical_df,
            source_filename="4436",
            source_kind="pff_match_triplets",
            processed_events_path=sources[0]["events_path"],
            processed_tracking_path=sources[0]["tracking_path"],
            drop_unlabeled=True,
        )

        assert len(labeled_df) == 1, "Expected one labeled row"
        assert int(labeled_df.iloc[0]["reward_label"]) == -1, labeled_df.iloc[0]["reward_label"]

        row = labeled_df.iloc[0]
        tensorizer = ToSoccerMapTensor()
        frame = labeled_df.loc[[labeled_df.index[0]]].copy()
        sample = {
            "ball_x_start": float(row["ball_x_start"]),
            "ball_y_start": float(row["ball_y_start"]),
            "ball_x_end": float(row["ball_x_end"]),
            "ball_y_end": float(row["ball_y_end"]),
            "pass_outcome_type": row["pass_outcome_type"],
            "team_id": int(row["team_id"]),
            "vx_carrier": 0.2,
            "vy_carrier": 0.1,
            "carrier_velocity": (0.2 ** 2 + 0.1 ** 2) ** 0.5,
            "frame": frame,
        }
        matrix, mask, _ = tensorizer(sample)
        assert tuple(matrix.shape) == (13, 68, 104), matrix.shape
        assert tuple(mask.shape) == (1, 68, 104), mask.shape

        # Cache invalidation check: touching events.parquet must force rebuild.
        events.loc[events["event_id"] == 101, "pass_outcome"] = "intercepted"
        time.sleep(0.01)
        events.to_parquet(match_dir / "events.parquet", index=False)

        rebuilt_df, rebuilt_summary = load_or_build_canonical_pass_cache(
            source=sources[0],
            required_columns=required_cols,
            source_filename=sources[0]["source_name"],
        )
        assert rebuilt_summary.get("cache_hit") is False
        assert set(rebuilt_df["pass_outcome_type"].dropna().unique()) == {"D"}

        # Missing outcome columns should not raise; adapter should skip rows with clear summary.
        missing_dir = root / "4437"
        missing_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "match_id": 4437,
                    "event_id": 201,
                    "possession_id": 2001,
                    "team_id": 30,
                    "player_id": 300,
                    "possession_type": "pass",
                    "set_piece": "open_play",
                    "frame_id": 900,
                    "ball_x": 0.0,
                    "ball_y": 0.0,
                    "elapsed_seconds": 200.0,
                }
            ]
        ).to_parquet(missing_dir / "events.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "match_id": 4437,
                    "frame_id": 900,
                    "team_id": 30,
                    "player_id": 300,
                    "x": 0.0,
                    "y": 0.0,
                    "ball_x": 0.0,
                    "ball_y": 0.0,
                    "elapsed_seconds": 200.0,
                }
            ]
        ).to_parquet(missing_dir / "tracking.parquet", index=False)
        pd.DataFrame([{"player_id": 300}]).to_parquet(missing_dir / "players.parquet", index=False)

        missing_sources = discover_pass_sources(root, source_format="pff_match_triplets")
        missing_source = [source for source in missing_sources if source.get("source_name") == "4437"][0]
        missing_df, missing_summary = load_or_build_canonical_pass_cache(
            source=missing_source,
            required_columns=required_cols,
            source_filename=missing_source["source_name"],
            source_format="pff_match_triplets",
        )
        assert missing_df.empty
        assert int(missing_summary["merge_summary"].get("rows_missing_outcome", -1)) == 1

    print("CHECKPOINT_PFF_TRIPLET_ADAPTER_OK")
    print(f"canonical_rows={int(canonical_summary.get('rows_total', 0))}")
    print(f"reward_source_kind={label_summary.get('source_kind')}")


if __name__ == "__main__":
    run_audit()