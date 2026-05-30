from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from Pass.data_utils import REPO_ROOT, discover_action_sources
from Pass.reward_labels import PassRewardLabeler

from .baseline_xg import BaselineXGArtifacts, load_baseline_xg
from .features import ShotFeatureBuilder


@dataclass
class ShotDataConfig:
    source_root: str = "data/processed"
    source_format: str = "pff_match_triplets"
    include_open_play_null: bool = True
    split_seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


def _normalize_tracking_path(source: Dict[str, Any]) -> Optional[Path]:
    tracking_path = source.get("tracking_path")
    if not tracking_path:
        return None
    return Path(tracking_path)


def build_shot_split_manifest(
    shot_df: pd.DataFrame,
    config: Optional[ShotDataConfig] = None,
    manifest_relpath: str = "data/processed/cache/splits/shot_split_manifest.json",
) -> Tuple[Dict[str, Any], Path]:
    config = config or ShotDataConfig()

    if shot_df.empty:
        manifest = {
            "split_mode": "match",
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "split_seed": config.split_seed,
            "match_assignments": {},
        }
    else:
        match_ids = sorted({int(value) for value in pd.to_numeric(shot_df["game_id"], errors="coerce").dropna().astype(int).tolist()})
        rng = random.Random(int(config.split_seed))
        shuffled = list(match_ids)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(round(config.train_ratio * n_total))
        n_val = int(round(config.val_ratio * n_total))
        if n_total >= 3:
            n_train = max(1, min(n_train, n_total - 2))
            n_val = max(1, min(n_val, n_total - n_train - 1))
        n_test = max(0, n_total - n_train - n_val)

        assignments = {}
        train_ids = set(shuffled[:n_train])
        val_ids = set(shuffled[n_train : n_train + n_val])
        test_ids = set(shuffled[n_train + n_val : n_train + n_val + n_test])
        for match_id in shuffled:
            if match_id in train_ids:
                assignments[str(match_id)] = "train"
            elif match_id in val_ids:
                assignments[str(match_id)] = "val"
            else:
                assignments[str(match_id)] = "test"

        manifest = {
            "split_mode": "match",
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "split_seed": config.split_seed,
            "match_assignments": assignments,
        }

    manifest_path = REPO_ROOT / manifest_relpath
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    return manifest, manifest_path


def apply_match_splits(shot_df: pd.DataFrame, manifest: Dict[str, Any]) -> pd.DataFrame:
    if shot_df.empty:
        output = shot_df.copy()
        output["split"] = pd.Series(dtype=str)
        return output

    assignments = {str(key): str(value) for key, value in (manifest.get("match_assignments") or {}).items()}
    result = shot_df.copy()
    result["split"] = result["game_id"].map(lambda value: assignments.get(str(int(value)), "train") if pd.notna(value) else "train")
    return result


def build_shot_epv_dataset(
    config: Optional[ShotDataConfig] = None,
    baseline_xg_artifacts: Optional[BaselineXGArtifacts] = None,
    output_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    config = config or ShotDataConfig()
    sources = discover_action_sources(config.source_root, source_format=config.source_format, prefer_parquet=True)
    sources = [source for source in sources if str(source.get("source_kind")) == "pff_match_triplets"]

    if baseline_xg_artifacts is None:
        baseline_model_path = REPO_ROOT / "results" / "models" / "shot" / "baseline_xg_model.pkl"
        baseline_scaler_path = REPO_ROOT / "results" / "models" / "shot" / "baseline_xg_feature_scaler.pkl"
        if baseline_model_path.exists() and baseline_scaler_path.exists():
            baseline_xg_artifacts = load_baseline_xg(baseline_model_path, baseline_scaler_path)

    labeler = PassRewardLabeler(
        event_root=REPO_ROOT / "data" / "raw" / "event",
        horizon_seconds=15.0,
        include_open_play_null=config.include_open_play_null,
    )
    feature_builder = ShotFeatureBuilder()

    parts: List[pd.DataFrame] = []
    source_summaries: List[Dict[str, Any]] = []

    for source in sources:
        events_path = source.get("events_path")
        tracking_path = _normalize_tracking_path(source)
        if not events_path:
            continue

        events_df = pd.read_parquet(events_path)
        if "possession_type" not in events_df.columns:
            continue

        shot_rows = events_df[events_df["possession_type"].astype(str).str.lower().str.strip() == "shot"].copy()
        if shot_rows.empty:
            continue

        labeled_df, summary = labeler.label_action_dataframe(
            action_df=shot_rows,
            action_type="shot",
            source_kind="pff_match_triplets",
            processed_events_path=Path(events_path),
            processed_tracking_path=tracking_path,
            drop_unlabeled=True,
        )
        if labeled_df.empty:
            source_summaries.append(summary)
            continue

        tracking_df = pd.read_parquet(tracking_path) if tracking_path is not None and tracking_path.exists() else None
        feature_df = feature_builder.build_feature_frame(
            labeled_df,
            tracking_df=tracking_df,
            baseline_xg_artifacts=baseline_xg_artifacts,
        )

        shot_df = pd.concat([labeled_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
        shot_df["reward_G"] = pd.to_numeric(shot_df.get("reward_label"), errors="coerce")
        shot_df["reward_norm"] = (shot_df["reward_G"] + 1.0) / 2.0
        parts.append(shot_df)
        source_summaries.append(summary)

    dataset = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    manifest, manifest_path = build_shot_split_manifest(dataset, config)
    dataset = apply_match_splits(dataset, manifest)

    if output_path is None:
        output_path = REPO_ROOT / "data" / "processed" / "cache" / "shot" / "shot_epv_dataset.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not dataset.empty:
        dataset.to_parquet(output_path, index=False)

    summary = {
        "rows_total": int(len(dataset)),
        "rows_labeled": int(dataset.get("reward_label", pd.Series(dtype=float)).notna().sum()) if not dataset.empty else 0,
        "rows_open_play": int((dataset.get("reward_status", pd.Series(dtype=str)) == "ok").sum()) if not dataset.empty else 0,
        "source_count": int(len(sources)),
        "source_summaries": source_summaries,
        "manifest_path": str(manifest_path),
        "output_path": str(output_path),
    }

    return dataset, summary
