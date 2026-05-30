from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data import BallDriveDataConfig, build_ball_drive_canonical_dataset, segment_ball_drives
from features import BallDriveFeatureBuilder
from trainer import BallDriveTrainer


def main() -> None:
    config = BallDriveDataConfig()
    canonical_df, canonical_summary = build_ball_drive_canonical_dataset(config)
    segmented_df, segmentation_summary = segment_ball_drives(canonical_df)

    trainer = BallDriveTrainer(data_config=config)
    tracking_df = trainer._collect_tracking_data(segmented_df)

    report = {
        "rows_canonical": int(len(canonical_df)),
        "rows_segmented": int(len(segmented_df)),
        "canonical_summary": canonical_summary,
        "segmentation_summary": segmentation_summary,
        "window_found": False,
        "window_frame_count": 0,
        "pitch_control_value": float("nan"),
        "pitch_control_finite": False,
        "ball_grid_lookup_in_bounds": False,
        "status": "no_data",
    }

    if segmented_df.empty or tracking_df is None or tracking_df.empty:
        report["status"] = "skipped_no_segment_or_tracking_data"
    else:
        sample_row = segmented_df.iloc[0]
        builder = BallDriveFeatureBuilder()

        window = builder._build_pitch_control_window(sample_row, tracking_df)
        if window is not None and not window.empty:
            report["window_found"] = True
            report["window_frame_count"] = int(window["frame_id"].nunique())

            pc_value = builder._pitch_control_attacking_team_at_ball(sample_row, tracking_df)
            report["pitch_control_value"] = float(pc_value)
            report["pitch_control_finite"] = bool(np.isfinite(pc_value))

            ball_x_bl = float(sample_row.get("ball_x_start", 0.0)) + 52.5
            ball_y_bl = float(sample_row.get("ball_y_start", 0.0)) + 34.0
            report["ball_grid_lookup_in_bounds"] = bool(0.0 <= ball_x_bl <= 105.0 and 0.0 <= ball_y_bl <= 68.0)

            report["status"] = "ok" if report["pitch_control_finite"] else "window_found_non_finite_pitch_control"
        else:
            report["status"] = "skipped_no_3frame_window"

    out_path = Path(__file__).resolve().parents[2] / "results" / "metrics" / "ball_drive_gandula_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
