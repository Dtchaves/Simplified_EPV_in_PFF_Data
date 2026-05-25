from __future__ import annotations

from pathlib import Path

import numpy as np

from data_utils import discover_pass_sources, load_or_build_canonical_pass_cache


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


def _outside_count(series, low: float, high: float) -> int:
    values = series.to_numpy(dtype=float)
    mask = np.isfinite(values) & ((values < low) | (values > high))
    return int(mask.sum())


def run_audit(data_root: str = "data/processed") -> None:
    try:
        sources = discover_pass_sources(data_root, source_format="pff_match_triplets")
    except FileNotFoundError:
        print("CHECKPOINT_PFF_COORDINATE_AUDIT_SKIPPED")
        print("reason=no_pff_sources")
        return

    total_rows = 0
    total_outside = 0
    forward_samples = 0
    forward_non_negative = 0

    for source in sources:
        df, _ = load_or_build_canonical_pass_cache(
            source=source,
            required_columns=REQUIRED_COLUMNS,
            source_filename=source.get("source_name"),
            source_format="pff_match_triplets",
        )
        if df.empty:
            continue

        total_rows += int(len(df))

        total_outside += _outside_count(df["ball_x_start"], -52.5, 52.5)
        total_outside += _outside_count(df["ball_x_end"], -52.5, 52.5)
        total_outside += _outside_count(df["ball_y_start"], -34.0, 34.0)
        total_outside += _outside_count(df["ball_y_end"], -34.0, 34.0)

        player_x_columns = [col for col in df.columns if col.startswith("x_player_")]
        player_y_columns = [col for col in df.columns if col.startswith("y_player_")]
        for column in player_x_columns:
            total_outside += _outside_count(df[column], -52.5, 52.5)
        for column in player_y_columns:
            total_outside += _outside_count(df[column], -34.0, 34.0)

        delta_x = (df["ball_x_end"].astype(float) - df["ball_x_start"].astype(float)).to_numpy()
        valid_delta = np.isfinite(delta_x)
        forward_samples += int(valid_delta.sum())
        forward_non_negative += int((delta_x[valid_delta] >= 0.0).sum())

    if total_rows == 0:
        print("CHECKPOINT_PFF_COORDINATE_AUDIT_SKIPPED")
        print("reason=no_canonical_rows")
        return

    if total_outside > 0:
        raise AssertionError(
            f"Found {total_outside} coordinate values outside centered pitch bounds."
        )

    forward_rate = (forward_non_negative / forward_samples) if forward_samples > 0 else 0.0
    if forward_samples > 0 and forward_rate < 0.45:
        raise AssertionError(
            f"Forward-direction proxy below threshold: forward_rate={forward_rate:.4f}"
        )

    print("CHECKPOINT_PFF_COORDINATE_AUDIT_OK")
    print(f"sources={len(sources)}")
    print(f"rows={total_rows}")
    print(f"forward_rate={forward_rate:.4f}")
    print("goal_x_reference=52.5")


if __name__ == "__main__":
    run_audit()