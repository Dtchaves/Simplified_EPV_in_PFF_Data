from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data import BallDriveDataConfig, apply_match_splits, build_ball_drive_canonical_dataset, build_ball_drive_split_manifest, segment_ball_drives
from features import BallDriveFeatureBuilder
from trainer import BallDriveTrainer


def main() -> None:
    config = BallDriveDataConfig()
    canonical_df, _ = build_ball_drive_canonical_dataset(config)
    segmented_df, _ = segment_ball_drives(canonical_df)
    manifest, _ = build_ball_drive_split_manifest(segmented_df, config)
    split_df = apply_match_splits(segmented_df, manifest)

    trainer = BallDriveTrainer(data_config=config)
    tracking_df = trainer._collect_tracking_data(split_df)

    builder = BallDriveFeatureBuilder()
    feature_df = builder.build_feature_frame(split_df.head(200), tracking_df=tracking_df)

    finite_mask = np.isfinite(feature_df.to_numpy(dtype=float))
    pitch_col = "pitch_control_attacking_team_at_ball"
    pitch_min = float(feature_df[pitch_col].min()) if pitch_col in feature_df.columns and not feature_df.empty else float("nan")
    pitch_max = float(feature_df[pitch_col].max()) if pitch_col in feature_df.columns and not feature_df.empty else float("nan")
    pressure_att_col = "nearest_attacking_pressure_line_distance"
    pressure_def_col = "nearest_defending_pressure_line_distance"

    report = {
        "rows_checked": int(len(feature_df)),
        "cols_checked": int(feature_df.shape[1]),
        "all_finite": bool(finite_mask.all()),
        "non_finite_count": int((~finite_mask).sum()),
        "feature_columns": list(feature_df.columns),
        "pitch_control_min": pitch_min,
        "pitch_control_max": pitch_max,
        "pitch_control_in_unit_interval": bool(0.0 <= pitch_min <= 1.0 and 0.0 <= pitch_max <= 1.0) if np.isfinite(pitch_min) and np.isfinite(pitch_max) else False,
        "pressure_attacking_min": float(feature_df[pressure_att_col].min()) if pressure_att_col in feature_df.columns and not feature_df.empty else float("nan"),
        "pressure_attacking_max": float(feature_df[pressure_att_col].max()) if pressure_att_col in feature_df.columns and not feature_df.empty else float("nan"),
        "pressure_defending_min": float(feature_df[pressure_def_col].min()) if pressure_def_col in feature_df.columns and not feature_df.empty else float("nan"),
        "pressure_defending_max": float(feature_df[pressure_def_col].max()) if pressure_def_col in feature_df.columns and not feature_df.empty else float("nan"),
    }

    out_path = Path(__file__).resolve().parents[2] / "results" / "metrics" / "ball_drive_feature_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
