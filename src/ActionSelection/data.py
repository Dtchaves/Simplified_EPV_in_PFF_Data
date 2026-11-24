from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional runtime dependency
    pq = None

from Pass.data_utils import (
    REPO_ROOT,
    discover_action_sources,
    get_cache_artifact_paths,
    get_optional_fingerprint,
    get_source_fingerprint,
    is_cache_manifest_current,
    load_cache_manifest,
    sample_sources_by_season,
    write_cache_manifest,
)

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
ACTION_SELECTION_SOURCE_CACHE_VERSION = "v3_action_selection_validated_cache"
ACTION_SELECTION_AGGREGATE_CACHE_VERSION = "v1_action_selection_aggregate_dataset_cache"
ACTION_EVENT_COLUMNS: Tuple[str, ...] = (
    "match_id",
    "game_id",
    "event_id",
    "possession_id",
    "possession_type",
    "frame_id",
    "start_frame_id",
    "end_frame_id",
    "team_id",
    "player_id",
    "team_side",
    "ball_x",
    "ball_y",
    "ball_x_start",
    "ball_y_start",
    "shot_x",
    "shot_y",
    "set_piece",
    "body_part",
    "shot_body_part",
    "shot_body",
    "header",
    "is_header",
)
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
    open_play_only: bool = True
    include_open_play_null: bool = True
    require_baseline_xg: bool = True
    cache_enabled: bool = True
    reuse_existing_dataset: bool = True
    force_rebuild_dataset: bool = False
    # Guard against accidentally reusing a tiny/stale aggregate dataset from debug runs.
    # Set to 0 to disable.
    min_existing_dataset_rows: int = 10_000


def load_default_baseline_xg_artifacts() -> Optional[BaselineXGArtifacts]:
    logging.info("Loading default baseline XG artifacts from expected paths.")
    model_path = REPO_ROOT / "results" / "models" / "shot" / "baseline_xg_model.pkl"
    scaler_path = REPO_ROOT / "results" / "models" / "shot" / "baseline_xg_feature_scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        return None
    return load_baseline_xg(model_path, scaler_path)


def _default_baseline_xg_dependency_state() -> Dict[str, Optional[str]]:
    model_path = REPO_ROOT / "results" / "models" / "shot" / "baseline_xg_model.pkl"
    scaler_path = REPO_ROOT / "results" / "models" / "shot" / "baseline_xg_feature_scaler.pkl"
    return {
        "baseline_xg_model_path": str(model_path),
        "baseline_xg_model_fingerprint": get_optional_fingerprint(model_path),
        "baseline_xg_scaler_path": str(scaler_path),
        "baseline_xg_scaler_fingerprint": get_optional_fingerprint(scaler_path),
    }


@lru_cache(maxsize=None)
def _parquet_columns(path_str: str) -> Tuple[str, ...]:
    if pq is None:
        return tuple()
    return tuple(pq.ParquetFile(path_str).schema.names)


def _read_parquet_subset(
    path: str,
    requested_columns: Tuple[str, ...],
    filters: Optional[List[Tuple[str, str, List[int]]]] = None,
) -> pd.DataFrame:
    if pq is None:
        return pd.read_parquet(path)

    available = set(_parquet_columns(str(path)))
    columns = [column for column in requested_columns if column in available]
    kwargs: Dict[str, Any] = {}
    if columns:
        kwargs["columns"] = columns
    if filters:
        kwargs["filters"] = filters
    try:
        return pd.read_parquet(path, **kwargs)
    except Exception:
        # Fall back to an unfiltered read when parquet engine filtering is unavailable.
        fallback_kwargs = {"columns": columns} if columns else {}
        return pd.read_parquet(path, **fallback_kwargs)


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


def _load_action_selection_source_frames(
    source: Dict[str, Any],
    config: Optional[ActionSelectionDataConfig] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    config = config or ActionSelectionDataConfig()
    events_path = source.get("events_path")
    if not events_path:
        return pd.DataFrame(), None, {
            "match_id": int(source["match_id"]) if source.get("match_id") is not None else None,
            "rows": 0,
            "tracking_rows": 0,
            "action_counts": {},
            "filter_summary": {
                "rows_before_filter": 0,
                "rows_after_filter": 0,
                "rows_filtered_out": 0,
                "open_play_only": bool(config.open_play_only),
                "include_open_play_null": bool(config.include_open_play_null),
            },
        }

    events_df = _read_parquet_subset(str(events_path), ACTION_EVENT_COLUMNS)
    actions, filter_summary = _build_action_rows(events_df, source, config=config)
    if actions.empty:
        return actions, None, {
            "match_id": int(source["match_id"]) if source.get("match_id") is not None else None,
            "rows": 0,
            "tracking_rows": 0,
            "action_counts": {},
            "filter_summary": filter_summary,
        }

    tracking_df: Optional[pd.DataFrame] = None
    tracking_path = source.get("tracking_path")
    if tracking_path and Path(tracking_path).exists():
        frame_ids = pd.to_numeric(actions["frame_id"], errors="coerce").dropna().astype(int).tolist()
        if frame_ids:
            expanded_frame_ids = sorted({frame_id + offset for frame_id in frame_ids for offset in (-2, -1, 0)})
            tracking_df = _read_parquet_subset(
                str(tracking_path),
                TRACKING_COLUMNS,
                filters=[("frame_id", "in", expanded_frame_ids)],
            )
        else:
            tracking_df = _read_parquet_subset(str(tracking_path), TRACKING_COLUMNS)
        tracking_df = _normalize_tracking_frame(tracking_df)
        if tracking_df is not None and not tracking_df.empty and "frame_id" in tracking_df.columns and frame_ids:
            valid_frame_ids = set(expanded_frame_ids)
            tracking_df = tracking_df[tracking_df["frame_id"].isin(valid_frame_ids)].copy()

    action_counts = {str(key): int(value) for key, value in actions["action_label"].value_counts().to_dict().items()}
    return actions, tracking_df, {
        "match_id": int(source["match_id"]) if source.get("match_id") is not None else None,
        "rows": int(len(actions)),
        "tracking_rows": int(len(tracking_df)) if tracking_df is not None else 0,
        "action_counts": action_counts,
        "filter_summary": filter_summary,
    }


def _filter_action_rows(
    actions_df: pd.DataFrame,
    config: Optional[ActionSelectionDataConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    config = config or ActionSelectionDataConfig()
    summary: Dict[str, Any] = {
        "open_play_only": bool(config.open_play_only),
        "include_open_play_null": bool(config.include_open_play_null),
        "set_piece_column_present": "set_piece" in actions_df.columns,
    }

    if actions_df.empty:
        summary.update(
            {
                "rows_before_filter": 0,
                "rows_after_filter": 0,
                "rows_filtered_out": 0,
                "rows_with_explicit_non_open_play_set_piece": 0,
                "rows_with_open_play_or_null_set_piece": 0,
            }
        )
        return actions_df.copy(), summary

    filtered = actions_df.copy()
    if "set_piece" in filtered.columns:
        filtered["set_piece_normalized"] = filtered["set_piece"].map(ActionSelectionFeatureBuilder._normalize_set_piece)
        open_play_mask = filtered["set_piece_normalized"].eq("open_play")
        if config.include_open_play_null:
            open_play_mask = open_play_mask | filtered["set_piece_normalized"].isna()
        filtered["open_play_eligible"] = open_play_mask.astype(int)
        summary["rows_with_explicit_non_open_play_set_piece"] = int(
            filtered["set_piece_normalized"].notna().sum() - int((filtered["set_piece_normalized"] == "open_play").sum())
        )
        summary["rows_with_open_play_or_null_set_piece"] = int(open_play_mask.sum())
    else:
        filtered["set_piece_normalized"] = pd.Series([None] * len(filtered), index=filtered.index, dtype=object)
        filtered["open_play_eligible"] = 1
        open_play_mask = pd.Series(True, index=filtered.index)

    if config.open_play_only and "set_piece" in filtered.columns:
        filtered = filtered[open_play_mask].copy()

    summary.update(
        {
            "rows_before_filter": int(len(actions_df)),
            "rows_after_filter": int(len(filtered)),
            "rows_filtered_out": int(len(actions_df) - len(filtered)),
        }
    )
    return filtered, summary


def _validate_action_selection_feature_health(feature_frame: pd.DataFrame) -> None:
    if feature_frame.empty:
        raise ValueError("ActionSelection feature frame is empty.")

    checks = {
        "pitch_control_attacking_team_at_ball": {
            "min_std": 0.0,
            "max_zero_rate": 0.80,
        },
        "defending_team_influence_at_ball": {
            "min_std": 0.0,
            "max_zero_rate": 0.95,
        },
    }

    problems: List[str] = []
    for column, thresholds in checks.items():
        if column not in feature_frame.columns:
            problems.append(f"{column} missing from feature frame")
            continue

        values = pd.to_numeric(feature_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = values.dropna()
        if finite.empty:
            problems.append(f"{column} has no finite values")
            continue

        std = float(finite.std(ddof=0))
        zero_rate = float((finite == 0).mean())
        if not np.isfinite(std) or std <= thresholds["min_std"]:
            problems.append(f"{column} collapsed to constant; std={std:.6f}")
        if zero_rate > thresholds["max_zero_rate"]:
            problems.append(f"{column} has suspiciously high zero rate; zero_rate={zero_rate:.3f}")

    if problems:
        raise ValueError("ActionSelection feature health check failed: " + "; ".join(problems))


def _build_action_rows(events_df: pd.DataFrame, source: Dict[str, Any], config: Optional[ActionSelectionDataConfig] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if events_df.empty or "possession_type" not in events_df.columns:
        return pd.DataFrame(), {
            "rows_before_filter": 0,
            "rows_after_filter": 0,
            "rows_filtered_out": 0,
            "open_play_only": bool((config or ActionSelectionDataConfig()).open_play_only),
            "include_open_play_null": bool((config or ActionSelectionDataConfig()).include_open_play_null),
            "set_piece_column_present": "set_piece" in events_df.columns,
            "rows_with_explicit_non_open_play_set_piece": 0,
            "rows_with_open_play_or_null_set_piece": 0,
        }

    actions = events_df.copy()
    actions["possession_type_normalized"] = actions["possession_type"].astype(str).str.lower().str.strip()
    actions = actions[actions["possession_type_normalized"].isin(ACTION_LABEL_MAP)].copy()
    if actions.empty:
        return actions, {
            "rows_before_filter": 0,
            "rows_after_filter": 0,
            "rows_filtered_out": 0,
            "open_play_only": bool((config or ActionSelectionDataConfig()).open_play_only),
            "include_open_play_null": bool((config or ActionSelectionDataConfig()).include_open_play_null),
            "set_piece_column_present": "set_piece" in events_df.columns,
            "rows_with_explicit_non_open_play_set_piece": 0,
            "rows_with_open_play_or_null_set_piece": 0,
        }

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

    actions, filter_summary = _filter_action_rows(actions, config=config)
    if actions.empty:
        return actions.drop(columns=["possession_type_normalized"], errors="ignore"), filter_summary

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
    actions = actions.drop(columns=["possession_type_normalized"])
    return actions, filter_summary


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


def _action_selection_source_cache_dependencies(
    source: Dict[str, Any],
    config: ActionSelectionDataConfig,
) -> Dict[str, Any]:
    dependency_state: Dict[str, Any] = {
        "source_kind": str(source.get("source_kind")),
        "source_format": str(source.get("source_format")),
        "open_play_only": bool(config.open_play_only),
        "include_open_play_null": bool(config.include_open_play_null),
        "require_baseline_xg": bool(config.require_baseline_xg),
        "cache_version": ACTION_SELECTION_SOURCE_CACHE_VERSION,
        "tracking_path": source.get("tracking_path"),
        "tracking_fingerprint": get_optional_fingerprint(source.get("tracking_path")),
        "players_path": source.get("players_path"),
        "players_fingerprint": get_optional_fingerprint(source.get("players_path")),
    }
    dependency_state.update(_default_baseline_xg_dependency_state())
    return dependency_state


def _load_cached_action_selection_source_payload(
    source: Dict[str, Any],
    config: ActionSelectionDataConfig,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    if not config.cache_enabled:
        return None, None

    events_path = source.get("events_path")
    if not events_path:
        return None, None

    data_path, manifest_path = get_cache_artifact_paths(
        source_path=str(events_path),
        cache_family="action_selection",
        artifact_stem="source_dataset",
        data_suffix=".parquet",
    )
    manifest = load_cache_manifest(manifest_path)
    if not data_path.exists() or not is_cache_manifest_current(
        manifest,
        source_path=str(events_path),
        cache_version=ACTION_SELECTION_SOURCE_CACHE_VERSION,
        dependencies=_action_selection_source_cache_dependencies(source, config),
    ):
        return None, None

    cached = pd.read_parquet(data_path)
    summary = dict(manifest.get("summary", {}))
    summary["cache_hit"] = True
    summary["cache_data_path"] = str(data_path)
    summary["cache_manifest_path"] = str(manifest_path)
    return cached, summary


def _save_action_selection_source_payload(
    source: Dict[str, Any],
    config: ActionSelectionDataConfig,
    payload: pd.DataFrame,
    summary: Dict[str, Any],
) -> None:
    if not config.cache_enabled:
        return

    events_path = source.get("events_path")
    if not events_path:
        return

    data_path, manifest_path = get_cache_artifact_paths(
        source_path=str(events_path),
        cache_family="action_selection",
        artifact_stem="source_dataset",
        data_suffix=".parquet",
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    payload.to_parquet(data_path, index=False)
    manifest_payload = {
        "cache_version": ACTION_SELECTION_SOURCE_CACHE_VERSION,
        "source_path": str(events_path),
        "source_fingerprint": get_source_fingerprint(str(events_path)),
        "dependencies": _action_selection_source_cache_dependencies(source, config),
        "summary": summary,
        "cache_data_path": str(data_path),
        "rows_total": int(len(payload)),
    }
    write_cache_manifest(manifest_path, manifest_payload)


def _build_action_selection_source_payload(
    source: Dict[str, Any],
    config: ActionSelectionDataConfig,
    baseline_xg_artifacts: Optional[BaselineXGArtifacts],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cached_payload, cached_summary = _load_cached_action_selection_source_payload(source, config)
    if cached_payload is not None and cached_summary is not None:
        return cached_payload, cached_summary

    actions_df, tracking_df, source_summary = _load_action_selection_source_frames(source, config)
    if actions_df.empty:
        empty_summary = dict(source_summary)
        empty_summary["cache_hit"] = False
        return pd.DataFrame(), empty_summary

    dataset, feature_summary = build_action_selection_dataset(
        actions_df,
        tracking_df=tracking_df,
        baseline_xg_artifacts=baseline_xg_artifacts,
        data_config=config,
        return_audit=True,
    )

    payload = pd.concat(
        [
            actions_df[[column for column in ("match_id", "game_id", "action_uid") if column in actions_df.columns]].reset_index(drop=True),
            dataset.reset_index(drop=True),
        ],
        axis=1,
    )
    merged_summary = {
        **source_summary,
        **feature_summary,
        "cache_hit": False,
        "dataset_rows": int(len(dataset)),
    }
    _save_action_selection_source_payload(source, config, payload, merged_summary)
    return payload, merged_summary


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

    source_progress = tqdm(
        sources,
        desc="ActionSelection sources",
        unit="source",
        dynamic_ncols=True,
    ) if sources else None

    source_iter = source_progress if source_progress is not None else []

    for source in source_iter:
        actions, tracking_df, source_summary = _load_action_selection_source_frames(source, config)
        if not actions.empty:
            action_parts.append(actions)
        if tracking_df is not None and not tracking_df.empty:
            tracking_parts.append(tracking_df)
        source_summaries.append(source_summary)

        if source_progress is not None:
            source_progress.set_postfix(
                match_id=source.get("match_id"),
                rows=source_summary.get("rows", 0),
            )

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
        "source_match_ids": [int(source["match_id"]) if source.get("match_id") is not None else None for source in sources],
        "source_identities": [_source_identity(source) for source in sources],
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
    data_config: Optional[ActionSelectionDataConfig] = None,
    return_audit: bool = False,
) -> pd.DataFrame:
    """Build dataset for action-selection model.

    Expects `actions_df` to contain an `action_label` column with values
    in {"pass", "ball_drive", "shot"} (case-insensitive). Returns a
    DataFrame with feature columns and integer `label` column.
    """
    if actions_df is None or actions_df.empty:
        raise ValueError("actions_df is empty")

    config = data_config or ActionSelectionDataConfig()
    filtered_actions, filter_summary = _filter_action_rows(actions_df, config=config)
    if filtered_actions.empty:
        raise ValueError("No action-selection rows remain after filtering.")

    resolved_baseline_xg_artifacts = baseline_xg_artifacts or load_default_baseline_xg_artifacts()
    if config.require_baseline_xg and resolved_baseline_xg_artifacts is None and "baseline_xg" not in filtered_actions.columns:
        raise FileNotFoundError("Baseline xG artifacts are required for action-selection feature building but were not found.")

    builder = ActionSelectionFeatureBuilder()
    features, feature_audit = builder.build_feature_frame(
        filtered_actions,
        tracking_df,
        resolved_baseline_xg_artifacts,
        return_audit=True,
    )
    _validate_action_selection_feature_health(features)

    # Normalize label column
    if "action_label" not in filtered_actions.columns:
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

    labels = filtered_actions["action_label"].apply(_map_label).astype(int)

    dataset = features.copy()
    dataset["label"] = labels.values

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(output_path, index=False)

    if not return_audit:
        return dataset

    audit_summary = {
        "rows_before_filter": int(filter_summary.get("rows_before_filter", len(actions_df))),
        "rows_after_filter": int(filter_summary.get("rows_after_filter", len(filtered_actions))),
        "rows_filtered_out": int(filter_summary.get("rows_filtered_out", len(actions_df) - len(filtered_actions))),
        "feature_missing_count_mean": float(feature_audit["feature_missing_count"].mean()) if not feature_audit.empty else 0.0,
        "feature_missing_rate_mean": float(feature_audit["feature_missing_rate"].mean()) if not feature_audit.empty else 0.0,
        "feature_missing_rates": {
            column.removeprefix("missing__"): float(feature_audit[column].mean())
            for column in feature_audit.columns
            if column.startswith("missing__")
        },
        "filter_summary": filter_summary,
    }
    return dataset, audit_summary



def _dataset_cache_config_signature(config: ActionSelectionDataConfig) -> Dict[str, Any]:
    """Configuration fields that define the aggregate ActionSelection dataset."""
    return {
        "source_root": str(config.source_root),
        "source_format": str(config.source_format),
        "split_seed": int(config.split_seed),
        "train_ratio": float(config.train_ratio),
        "val_ratio": float(config.val_ratio),
        "test_ratio": float(config.test_ratio),
        "max_sources": config.max_sources if config.max_sources is None else int(config.max_sources),
        "season_sample_ratio": config.season_sample_ratio if config.season_sample_ratio is None else float(config.season_sample_ratio),
        "open_play_only": bool(config.open_play_only),
        "include_open_play_null": bool(config.include_open_play_null),
        "require_baseline_xg": bool(config.require_baseline_xg),
    }


def _source_identity(source: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_kind": str(source.get("source_kind")),
        "source_format": str(source.get("source_format")),
        "match_id": int(source["match_id"]) if source.get("match_id") is not None else None,
        "events_path": str(source.get("events_path")),
        "events_fingerprint": get_optional_fingerprint(source.get("events_path")),
        "tracking_path": str(source.get("tracking_path")),
        "tracking_fingerprint": get_optional_fingerprint(source.get("tracking_path")),
    }


def _aggregate_cache_context(
    sources: List[Dict[str, Any]],
    season_sampling_summary: Dict[str, Any],
    config: ActionSelectionDataConfig,
) -> Dict[str, Any]:
    source_identities = [_source_identity(source) for source in sources]
    return {
        "cache_version": ACTION_SELECTION_AGGREGATE_CACHE_VERSION,
        "source_count": int(len(sources)),
        "source_match_ids": [identity.get("match_id") for identity in source_identities],
        "source_identities": source_identities,
        "season_sampling": season_sampling_summary,
        "config": _dataset_cache_config_signature(config),
        "baseline_xg_dependencies": _default_baseline_xg_dependency_state(),
    }


def _manifest_matches_expected_context(
    manifest: Dict[str, Any],
    expected_context: Dict[str, Any],
) -> Tuple[bool, str]:
    if not manifest:
        return False, "missing aggregate dataset manifest"
    if manifest.get("cache_version") != expected_context.get("cache_version"):
        return False, "aggregate cache version mismatch"
    for key in ("config", "source_count", "source_match_ids", "baseline_xg_dependencies"):
        if manifest.get(key) != expected_context.get(key):
            return False, f"aggregate cache {key} mismatch"
    # Source fingerprints catch stale caches after processed files change.
    if manifest.get("source_identities") != expected_context.get("source_identities"):
        return False, "aggregate cache source identity/fingerprint mismatch"
    return True, "current"


def _label_action_counts_from_dataset(dataset: pd.DataFrame) -> Dict[str, int]:
    if dataset.empty or "label" not in dataset.columns:
        return {}
    inverse_label_map = {0: "pass", 1: "ball_drive", 2: "shot"}
    labels = pd.to_numeric(dataset["label"], errors="coerce").dropna().astype(int)
    counts = labels.map(lambda value: inverse_label_map.get(int(value), str(int(value)))).value_counts()
    return {str(key): int(value) for key, value in counts.to_dict().items()}


def _summarize_existing_action_selection_dataset(
    dataset: pd.DataFrame,
    output_path: Path,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    split_counts = (
        {str(key): int(value) for key, value in dataset["split"].value_counts().to_dict().items()}
        if "split" in dataset.columns
        else {}
    )
    summary: Dict[str, Any] = {
        "source_count": None,
        "season_sampling": {},
        "rows_total": int(len(dataset)),
        "tracking_rows": None,
        "action_counts": _label_action_counts_from_dataset(dataset),
        "source_summaries": [],
        "rows_before_filter": None,
        "rows_after_filter": int(len(dataset)),
        "rows_filtered_out": None,
        "feature_missing_count_mean": None,
        "feature_missing_rate_mean": None,
        "feature_missing_rates": {},
        "filter_summary": {},
        "split_manifest_path": None,
        "output_path": str(output_path),
        "split_counts": split_counts,
        "aggregate_cache_hit": True,
    }
    if manifest_path is not None and manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            summary["aggregate_cache_manifest_path"] = str(manifest_path)
            summary["aggregate_cache_manifest"] = manifest
            for key, value in (manifest.get("summary") or {}).items():
                if key not in {"rows_total", "output_path", "split_counts", "action_counts"}:
                    summary[key] = value
        except Exception as exc:  # pragma: no cover - defensive metadata path
            summary["aggregate_cache_manifest_error"] = str(exc)
    return summary


def _load_existing_action_selection_dataset_if_available(
    output_path: Path,
    config: ActionSelectionDataConfig,
    expected_cache_context: Dict[str, Any],
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    if not config.cache_enabled or not config.reuse_existing_dataset or config.force_rebuild_dataset:
        return None, None
    if not output_path.exists():
        return None, None

    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if not manifest_path.exists():
        logging.warning(
            "ActionSelection: aggregate dataset exists at %s but has no manifest; rebuilding to avoid stale/debug-cache reuse.",
            output_path,
        )
        return None, None

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:
        logging.warning("ActionSelection: could not read aggregate dataset manifest %s (%s); rebuilding.", manifest_path, exc)
        return None, None

    matches, reason = _manifest_matches_expected_context(manifest, expected_cache_context)
    if not matches:
        logging.info("ActionSelection: aggregate dataset cache is not current (%s); rebuilding.", reason)
        return None, None

    dataset = pd.read_parquet(output_path)
    required_columns = {"label", "split"}
    missing = sorted(required_columns - set(dataset.columns))
    if missing:
        raise ValueError(
            f"Existing ActionSelection dataset at {output_path} is missing required columns: {missing}. "
            "Delete it or set force_rebuild_dataset=True to rebuild."
        )
    if dataset.empty:
        raise ValueError(
            f"Existing ActionSelection dataset at {output_path} is empty. "
            "Delete it or set force_rebuild_dataset=True to rebuild."
        )
    if int(config.min_existing_dataset_rows) > 0 and len(dataset) < int(config.min_existing_dataset_rows):
        logging.warning(
            "ActionSelection: existing aggregate dataset at %s has only %d rows (< min_existing_dataset_rows=%d); rebuilding to avoid tiny debug-cache reuse.",
            output_path,
            len(dataset),
            int(config.min_existing_dataset_rows),
        )
        return None, None

    summary = _summarize_existing_action_selection_dataset(dataset, output_path, manifest_path)
    summary["aggregate_cache_reason"] = reason
    logging.info(
        "ActionSelection: reusing current aggregate dataset at %s with %d rows",
        output_path,
        len(dataset),
    )
    return dataset, summary


def _write_action_selection_dataset_manifest(
    output_path: Path,
    config: ActionSelectionDataConfig,
    summary: Dict[str, Any],
) -> None:
    if not config.cache_enabled:
        return
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_payload = {
        "cache_version": ACTION_SELECTION_AGGREGATE_CACHE_VERSION,
        "dataset_path": str(output_path),
        "config": _dataset_cache_config_signature(config),
        "baseline_xg_dependencies": _default_baseline_xg_dependency_state(),
        "source_count": summary.get("source_count"),
        "source_match_ids": summary.get("source_match_ids"),
        "source_identities": summary.get("source_identities"),
        "season_sampling": summary.get("season_sampling"),
        "summary": {
            "source_count": summary.get("source_count"),
            "rows_total": summary.get("rows_total"),
            "split_counts": summary.get("split_counts"),
            "action_counts": summary.get("action_counts"),
            "rows_before_filter": summary.get("rows_before_filter"),
            "rows_after_filter": summary.get("rows_after_filter"),
            "rows_filtered_out": summary.get("rows_filtered_out"),
            "feature_missing_count_mean": summary.get("feature_missing_count_mean"),
            "feature_missing_rate_mean": summary.get("feature_missing_rate_mean"),
            "filter_summary": summary.get("filter_summary"),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2, sort_keys=True)

def build_action_selection_source_dataset(
    config: Optional[ActionSelectionDataConfig] = None,
    baseline_xg_artifacts: Optional[BaselineXGArtifacts] = None,
    output_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    config = config or ActionSelectionDataConfig()
    if output_path is None:
        output_path = REPO_ROOT / "data" / "processed" / "cache" / "action_selection" / "action_selection_dataset.parquet"

    baseline_artifacts = baseline_xg_artifacts or load_default_baseline_xg_artifacts()
    sources = discover_action_sources(config.source_root, source_format=config.source_format, prefer_parquet=True)
    sources = [source for source in sources if str(source.get("source_kind")) == "pff_match_triplets"]
    sources, season_sampling_summary = sample_sources_by_season(
        sources,
        season_sample_ratio=config.season_sample_ratio,
        split_seed=config.split_seed,
    )
    if config.max_sources is not None:
        sources = sources[: int(config.max_sources)]

    expected_cache_context = _aggregate_cache_context(sources, season_sampling_summary, config)
    cached_dataset, cached_summary = _load_existing_action_selection_dataset_if_available(
        output_path,
        config,
        expected_cache_context,
    )
    if cached_dataset is not None and cached_summary is not None:
        return cached_dataset, cached_summary

    payload_parts: List[pd.DataFrame] = []
    source_summaries: List[Dict[str, Any]] = []
    source_progress = tqdm(
        sources,
        desc="ActionSelection sources",
        unit="source",
        dynamic_ncols=True,
    ) if sources else None
    source_iter = source_progress if source_progress is not None else []

    logging.info("ActionSelection: building feature payloads for %d sources", len(sources))
    for source in source_iter:
        payload, source_summary = _build_action_selection_source_payload(source, config, baseline_artifacts)
        source_summaries.append(source_summary)
        if not payload.empty:
            payload_parts.append(payload)
        if source_progress is not None:
            source_progress.set_postfix(
                match_id=source.get("match_id"),
                rows=source_summary.get("dataset_rows", source_summary.get("rows", 0)),
                cache="hit" if source_summary.get("cache_hit") else "miss",
            )

    payload_df = pd.concat(payload_parts, ignore_index=True) if payload_parts else pd.DataFrame()
    split_keys = payload_df[[column for column in ("match_id", "game_id", "action_uid") if column in payload_df.columns]].copy()
    manifest, manifest_path = build_action_selection_split_manifest(split_keys, config)
    split_frame = apply_action_selection_splits(split_keys, manifest)

    dataset = payload_df.drop(columns=["match_id", "game_id", "action_uid"], errors="ignore").copy()
    if not split_frame.empty and "split" in split_frame.columns:
        dataset["split"] = split_frame["split"].values
    else:
        dataset["split"] = pd.Series(dtype=str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not dataset.empty:
        dataset.to_parquet(output_path, index=False)

    rows_before_filter = int(sum(summary.get("rows_before_filter", 0) for summary in source_summaries))
    rows_after_filter = int(sum(summary.get("rows_after_filter", summary.get("rows", 0)) for summary in source_summaries))
    rows_filtered_out = int(sum(summary.get("rows_filtered_out", 0) for summary in source_summaries))
    weighted_rows = max(1, int(sum(summary.get("dataset_rows", 0) for summary in source_summaries)))
    feature_missing_count_mean = float(
        sum(float(summary.get("feature_missing_count_mean", 0.0)) * int(summary.get("dataset_rows", 0)) for summary in source_summaries)
        / float(weighted_rows)
    )
    feature_missing_rate_mean = float(
        sum(float(summary.get("feature_missing_rate_mean", 0.0)) * int(summary.get("dataset_rows", 0)) for summary in source_summaries)
        / float(weighted_rows)
    )

    feature_missing_rates: Dict[str, float] = {}
    for summary in source_summaries:
        rows = int(summary.get("dataset_rows", 0))
        for key, value in (summary.get("feature_missing_rates") or {}).items():
            feature_missing_rates[key] = feature_missing_rates.get(key, 0.0) + float(value) * rows
    if weighted_rows > 0:
        feature_missing_rates = {
            key: float(value / float(weighted_rows))
            for key, value in feature_missing_rates.items()
        }

    action_counts: Dict[str, int] = {}
    for summary in source_summaries:
        for key, value in (summary.get("action_counts") or {}).items():
            action_counts[str(key)] = action_counts.get(str(key), 0) + int(value)

    summary = {
        "source_count": int(len(sources)),
        "source_match_ids": [int(source["match_id"]) if source.get("match_id") is not None else None for source in sources],
        "source_identities": [_source_identity(source) for source in sources],
        "season_sampling": season_sampling_summary,
        "rows_total": int(len(dataset)),
        "tracking_rows": int(sum(summary.get("tracking_rows", 0) for summary in source_summaries)),
        "action_counts": action_counts,
        "source_summaries": source_summaries,
        "rows_before_filter": rows_before_filter,
        "rows_after_filter": rows_after_filter,
        "rows_filtered_out": rows_filtered_out,
        "feature_missing_count_mean": feature_missing_count_mean,
        "feature_missing_rate_mean": feature_missing_rate_mean,
        "feature_missing_rates": feature_missing_rates,
        "filter_summary": {
            "open_play_only": bool(config.open_play_only),
            "include_open_play_null": bool(config.include_open_play_null),
            "rows_before_filter": rows_before_filter,
            "rows_after_filter": rows_after_filter,
            "rows_filtered_out": rows_filtered_out,
        },
        "split_manifest_path": str(manifest_path),
        "output_path": str(output_path),
        "split_counts": {str(key): int(value) for key, value in dataset.get("split", pd.Series(dtype=str)).value_counts().to_dict().items()},
        "aggregate_cache_hit": False,
    }
    _write_action_selection_dataset_manifest(output_path, config, summary)
    return dataset, summary