from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Pass.data_utils import REPO_ROOT, discover_action_sources, sample_sources_by_season

from .features import ActionSelectionFeatureBuilder
from Shot.baseline_xg import BaselineXGArtifacts, load_baseline_xg


ACTION_LABEL_MAP = {
    "pass": "pass",
    "cross": "pass",
    "through_ball": "pass",
    "carry": "ball_drive",
    "dribble": "ball_drive",
    "shot": "shot",
}
TRACKING_COLUMNS: Tuple[str, ...] = (
    "match_id",
    "frame_id",
    "period",
    "elapsed_seconds",
    "team_side",
    "team_id",
    "player_id",
    "x",
    "y",
    "ball_x",
    "ball_y",
)


@dataclass
class ActionSelectionDataConfig:
    source_root: str = "data/processed"
    source_format: str = "pff_match_triplets"
    split_seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    max_sources: Optional[int] = None
    season_sample_ratio: Optional[float] = None


def load_default_baseline_xg_artifacts() -> Optional[BaselineXGArtifacts]:
    model_path = REPO_ROOT / "results" / "models" / "shot" / "baseline_xg_model.pkl"
    scaler_path = REPO_ROOT / "results" / "models" / "shot" / "baseline_xg_feature_scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        return None
    return load_baseline_xg(model_path, scaler_path)


def _normalize_tracking_frame(tracking_df: pd.DataFrame) -> pd.DataFrame:
    if tracking_df.empty:
        return tracking_df.copy()

    frame = tracking_df.copy()
    rename_map = {}
    if "game_id" in frame.columns and "match_id" not in frame.columns:
        rename_map["game_id"] = "match_id"
    if "frame_num" in frame.columns and "frame_id" not in frame.columns:
        rename_map["frame_num"] = "frame_id"
    if "team" in frame.columns and "team_side" not in frame.columns:
        rename_map["team"] = "team_side"
    if rename_map:
        frame = frame.rename(columns=rename_map)

    for column in ("match_id", "frame_id", "team_id", "player_id", "x", "y", "ball_x", "ball_y"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if column in {"match_id", "frame_id", "team_id", "player_id"}:
                frame[column] = frame[column].astype("Int64")

    keep_columns = [column for column in TRACKING_COLUMNS if column in frame.columns]
    return frame[keep_columns].copy() if keep_columns else frame


def _build_action_rows(events_df: pd.DataFrame, source: Dict[str, Any]) -> pd.DataFrame:
    if events_df.empty or "possession_type" not in events_df.columns:
        return pd.DataFrame()

    actions = events_df.copy()
    actions["possession_type_normalized"] = actions["possession_type"].astype(str).str.lower().str.strip()
    actions = actions[actions["possession_type_normalized"].isin(ACTION_LABEL_MAP)].copy()
    if actions.empty:
        return actions

    actions["action_label"] = actions["possession_type_normalized"].map(ACTION_LABEL_MAP)

    match_values = pd.to_numeric(actions["match_id"], errors="coerce") if "match_id" in actions.columns else pd.Series(index=actions.index, dtype=float)
    source_match_id = source.get("match_id")
    if source_match_id is not None:
        match_values = match_values.fillna(float(source_match_id))
    actions["match_id"] = match_values.astype("Int64")

    game_values = pd.to_numeric(actions["game_id"], errors="coerce") if "game_id" in actions.columns else pd.Series(index=actions.index, dtype=float)
    actions["game_id"] = game_values.fillna(match_values).astype("Int64")

    frame_values = pd.to_numeric(actions["frame_id"], errors="coerce") if "frame_id" in actions.columns else pd.Series(index=actions.index, dtype=float)
    if "start_frame_id" in actions.columns:
        frame_values = frame_values.fillna(pd.to_numeric(actions["start_frame_id"], errors="coerce"))
    actions["frame_id"] = frame_values.astype("Int64")

    for column in ("team_id", "player_id", "start_frame_id", "end_frame_id"):
        if column in actions.columns:
            actions[column] = pd.to_numeric(actions[column], errors="coerce").astype("Int64")

    actions["source_kind"] = source.get("source_kind")
    actions["source_events_path"] = source.get("events_path")
    actions["source_tracking_path"] = source.get("tracking_path")
    event_values = pd.to_numeric(actions["event_id"], errors="coerce") if "event_id" in actions.columns else pd.Series(index=actions.index, dtype=float)
    possession_values = pd.to_numeric(actions["possession_id"], errors="coerce") if "possession_id" in actions.columns else pd.Series(index=actions.index, dtype=float)
    frame_tokens = actions["frame_id"].astype("string").fillna("na")
    match_tokens = actions["match_id"].astype("string").fillna("na")
    event_tokens = event_values.astype("Int64").astype("string").fillna("na")
    possession_tokens = possession_values.astype("Int64").astype("string").fillna("na")
    ordinal_tokens = pd.Series(range(len(actions)), index=actions.index, dtype=int).astype(str)
    actions["action_uid"] = (
        match_tokens
        + ":"
        + event_tokens
        + ":"
        + possession_tokens
        + ":"
        + frame_tokens
        + ":"
        + actions["action_label"].astype("string").fillna("na")
        + ":"
        + ordinal_tokens
    )
    return actions.drop(columns=["possession_type_normalized"])


def _resolve_split_sizes(total: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0

    n_train = int(round(train_ratio * total))
    n_val = int(round(val_ratio * total))
    if total >= 3:
        n_train = max(1, min(n_train, total - 2))
        n_val = max(1, min(n_val, total - n_train - 1))
    n_test = max(0, total - n_train - n_val)

    if total >= 3 and n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val = max(1, n_val - 1)

    return n_train, n_val, n_test


def build_action_selection_split_manifest(
    actions_df: pd.DataFrame,
    config: Optional[ActionSelectionDataConfig] = None,
    manifest_relpath: str = "data/processed/cache/splits/action_selection_split_manifest.json",
) -> Tuple[Dict[str, Any], Path]:
    config = config or ActionSelectionDataConfig()

    if actions_df.empty:
        manifest = {
            "split_mode": "match",
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "split_seed": config.split_seed,
            "match_assignments": {},
            "row_assignments": {},
        }
    else:
        match_series = pd.to_numeric(actions_df.get("match_id", actions_df.get("game_id")), errors="coerce")
        match_ids = sorted({int(value) for value in match_series.dropna().astype(int).tolist()})
        rng = random.Random(int(config.split_seed))

        if len(match_ids) >= 3:
            shuffled = list(match_ids)
            rng.shuffle(shuffled)
            n_train, n_val, n_test = _resolve_split_sizes(len(shuffled), config.train_ratio, config.val_ratio)
            train_ids = set(shuffled[:n_train])
            val_ids = set(shuffled[n_train : n_train + n_val])
            test_ids = set(shuffled[n_train + n_val : n_train + n_val + n_test])
            assignments: Dict[str, str] = {}
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
                "row_assignments": {},
            }
        else:
            row_keys = actions_df.get("action_uid", pd.Series(actions_df.index.astype(str), index=actions_df.index)).astype(str).tolist()
            shuffled = list(row_keys)
            rng.shuffle(shuffled)
            n_train, n_val, n_test = _resolve_split_sizes(len(shuffled), config.train_ratio, config.val_ratio)
            train_keys = set(shuffled[:n_train])
            val_keys = set(shuffled[n_train : n_train + n_val])
            test_keys = set(shuffled[n_train + n_val : n_train + n_val + n_test])
            assignments = {}
            for key in shuffled:
                if key in train_keys:
                    assignments[str(key)] = "train"
                elif key in val_keys:
                    assignments[str(key)] = "val"
                else:
                    assignments[str(key)] = "test"

            manifest = {
                "split_mode": "row",
                "train_ratio": config.train_ratio,
                "val_ratio": config.val_ratio,
                "test_ratio": config.test_ratio,
                "split_seed": config.split_seed,
                "match_assignments": {},
                "row_assignments": assignments,
            }

    manifest_path = REPO_ROOT / manifest_relpath
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    return manifest, manifest_path


def apply_action_selection_splits(actions_df: pd.DataFrame, manifest: Dict[str, Any]) -> pd.DataFrame:
    if actions_df.empty:
        output = actions_df.copy()
        output["split"] = pd.Series(dtype=str)
        return output

    split_mode = str(manifest.get("split_mode") or "match").strip().lower()
    result = actions_df.copy()

    if split_mode == "row":
        row_assignments = {str(key): str(value) for key, value in (manifest.get("row_assignments") or {}).items()}
        keys = result.get("action_uid", pd.Series(result.index.astype(str), index=result.index)).astype(str)
        result["split"] = keys.map(lambda value: row_assignments.get(str(value), "train"))
        return result

    match_assignments = {str(key): str(value) for key, value in (manifest.get("match_assignments") or {}).items()}
    match_series = pd.to_numeric(result.get("match_id", result.get("game_id")), errors="coerce")
    result["split"] = match_series.map(lambda value: match_assignments.get(str(int(value)), "train") if pd.notna(value) else "train")
    return result


def build_action_selection_frames(
    config: Optional[ActionSelectionDataConfig] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    config = config or ActionSelectionDataConfig()
    sources = discover_action_sources(config.source_root, source_format=config.source_format, prefer_parquet=True)
    sources = [source for source in sources if str(source.get("source_kind")) == "pff_match_triplets"]
    sources, season_sampling_summary = sample_sources_by_season(
        sources,
        season_sample_ratio=config.season_sample_ratio,
        split_seed=config.split_seed,
    )
    if config.max_sources is not None:
        sources = sources[: int(config.max_sources)]

    action_parts: List[pd.DataFrame] = []
    tracking_parts: List[pd.DataFrame] = []
    source_summaries: List[Dict[str, Any]] = []

    for source in sources:
        events_path = source.get("events_path")
        if not events_path:
            continue

        events_df = pd.read_parquet(events_path)
        actions = _build_action_rows(events_df, source)
        if actions.empty:
            source_summaries.append(
                {
                    "match_id": int(source["match_id"]) if source.get("match_id") is not None else None,
                    "rows": 0,
                    "action_counts": {},
                }
            )
            continue

        action_parts.append(actions)
        action_counts = {str(key): int(value) for key, value in actions["action_label"].value_counts().to_dict().items()}
        source_summaries.append(
            {
                "match_id": int(source["match_id"]) if source.get("match_id") is not None else None,
                "rows": int(len(actions)),
                "action_counts": action_counts,
            }
        )

        tracking_path = source.get("tracking_path")
        if tracking_path and Path(tracking_path).exists():
            tracking_df = _normalize_tracking_frame(pd.read_parquet(tracking_path))
            if not tracking_df.empty and "frame_id" in tracking_df.columns:
                frame_ids = pd.to_numeric(actions["frame_id"], errors="coerce").dropna().astype(int).tolist()
                if frame_ids:
                    tracking_df = tracking_df[tracking_df["frame_id"].isin(frame_ids)].copy()
            if not tracking_df.empty:
                tracking_parts.append(tracking_df)

    actions_df = pd.concat(action_parts, ignore_index=True) if action_parts else pd.DataFrame()
    tracking_df: Optional[pd.DataFrame]
    if tracking_parts:
        tracking_df = pd.concat(tracking_parts, ignore_index=True)
        dedupe_columns = [column for column in ("match_id", "frame_id", "team_side", "player_id") if column in tracking_df.columns]
        if dedupe_columns:
            tracking_df = tracking_df.drop_duplicates(subset=dedupe_columns, keep="first")
    else:
        tracking_df = None

    summary = {
        "source_count": int(len(sources)),
        "season_sampling": season_sampling_summary,
        "rows_total": int(len(actions_df)),
        "tracking_rows": int(len(tracking_df)) if tracking_df is not None else 0,
        "action_counts": {str(key): int(value) for key, value in actions_df.get("action_label", pd.Series(dtype=str)).value_counts().to_dict().items()},
        "source_summaries": source_summaries,
    }
    return actions_df, tracking_df, summary


def build_action_selection_dataset(
    actions_df: pd.DataFrame,
    tracking_df: Optional[pd.DataFrame] = None,
    baseline_xg_artifacts: Optional[BaselineXGArtifacts] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build dataset for action-selection model.

    Expects `actions_df` to contain an `action_label` column with values
    in {"pass", "ball_drive", "shot"} (case-insensitive). Returns a
    DataFrame with feature columns and integer `label` column.
    """
    if actions_df is None or actions_df.empty:
        raise ValueError("actions_df is empty")

    builder = ActionSelectionFeatureBuilder()
    features = builder.build_feature_frame(actions_df, tracking_df, baseline_xg_artifacts)

    # Normalize label column
    if "action_label" not in actions_df.columns:
        raise ValueError("actions_df must contain 'action_label' column")

    label_map = {"pass": 0, "ball_drive": 1, "shot": 2}

    def _map_label(v: Any) -> int:
        if pd.isna(v):
            raise ValueError("Found NaN in action_label column")
        token = str(v).strip().lower()
        if token in label_map:
            return label_map[token]
        # allow common synonyms
        if token in {"carry", "dribble"}:
            return label_map["ball_drive"]
        if token in {"cross", "through_ball"}:
            return label_map["pass"]
        raise ValueError(f"Unknown action label: {v}")

    labels = actions_df["action_label"].apply(_map_label).astype(int)

    dataset = features.copy()
    dataset["label"] = labels.values

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(output_path, index=False)

    return dataset


def build_action_selection_source_dataset(
    config: Optional[ActionSelectionDataConfig] = None,
    baseline_xg_artifacts: Optional[BaselineXGArtifacts] = None,
    output_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    config = config or ActionSelectionDataConfig()
    actions_df, tracking_df, frame_summary = build_action_selection_frames(config)
    dataset = build_action_selection_dataset(actions_df, tracking_df=tracking_df, baseline_xg_artifacts=baseline_xg_artifacts)
    manifest, manifest_path = build_action_selection_split_manifest(actions_df, config)
    split_frame = apply_action_selection_splits(actions_df[[column for column in ("match_id", "game_id", "action_uid") if column in actions_df.columns]].copy(), manifest)

    dataset = dataset.copy()
    if not split_frame.empty and "split" in split_frame.columns:
        dataset["split"] = split_frame["split"].values
    else:
        dataset["split"] = pd.Series(dtype=str)

    if output_path is None:
        output_path = REPO_ROOT / "data" / "processed" / "cache" / "action_selection" / "action_selection_dataset.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not dataset.empty:
        dataset.to_parquet(output_path, index=False)

    summary = {
        **frame_summary,
        "split_manifest_path": str(manifest_path),
        "output_path": str(output_path),
        "split_counts": {str(key): int(value) for key, value in dataset.get("split", pd.Series(dtype=str)).value_counts().to_dict().items()},
    }
    return dataset, summary
