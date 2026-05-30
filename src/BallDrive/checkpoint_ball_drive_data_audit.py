from __future__ import annotations

import json
from pathlib import Path

from data import (
    BallDriveDataConfig,
    apply_match_splits,
    attach_reward_labels,
    build_ball_drive_canonical_dataset,
    build_ball_drive_split_manifest,
    segment_ball_drives,
)


def main() -> None:
    config = BallDriveDataConfig()
    canonical_df, canonical_summary = build_ball_drive_canonical_dataset(config)
    segmented_df, segment_summary = segment_ball_drives(canonical_df)
    labeled_df, reward_summary = attach_reward_labels(segmented_df, include_open_play_null=config.include_open_play_null)
    manifest, manifest_path = build_ball_drive_split_manifest(labeled_df, config)
    split_df = apply_match_splits(labeled_df, manifest)

    split_leakage_count = 0
    split_leakage_examples = {}
    if not split_df.empty and {"game_id", "split"}.issubset(set(split_df.columns)):
        per_match_split_count = split_df.groupby("game_id", dropna=True)["split"].nunique(dropna=True)
        leaking_matches = per_match_split_count[per_match_split_count > 1]
        split_leakage_count = int(leaking_matches.shape[0])
        split_leakage_examples = {str(int(idx)): int(value) for idx, value in leaking_matches.head(10).to_dict().items()}

    report = {
        "canonical_summary": canonical_summary,
        "segmentation_summary": segment_summary,
        "reward_summary": reward_summary,
        "success_label_counts": split_df.get("y_success", []).value_counts(dropna=False).to_dict() if "y_success" in split_df.columns else {},
        "carry_outcome_counts": split_df.get("carry_outcome", []).value_counts(dropna=False).to_dict() if "carry_outcome" in split_df.columns else {},
        "success_source_counts": split_df.get("success_source", []).value_counts(dropna=False).to_dict() if "success_source" in split_df.columns else {},
        "open_play_count": int((split_df.get("set_piece_normalized") == "open_play").sum()) if "set_piece_normalized" in split_df.columns else 0,
        "null_set_piece_count": int(split_df.get("set_piece_normalized").isna().sum()) if "set_piece_normalized" in split_df.columns else 0,
        "split_counts": split_df["split"].value_counts(dropna=False).to_dict() if "split" in split_df.columns else {},
        "split_leakage_match_count": split_leakage_count,
        "split_leakage_examples": split_leakage_examples,
        "rows_total": int(len(split_df)),
        "manifest_path": str(manifest_path),
    }

    out_path = Path(__file__).resolve().parents[2] / "results" / "metrics" / "ball_drive_data_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
