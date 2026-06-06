from __future__ import annotations

import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BALL_DRIVE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = BALL_DRIVE_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Pass.data_utils import REPO_ROOT, discover_action_sources, get_cache_root, sample_sources_by_season
from Pass.reward_labels import PassRewardLabeler


OUTCOME_SUCCESS_MAP = {
    "retained": 1,
    "lost": 0,
}
OUTCOME_EXCLUDED = {"challenged", "stoppage"}
LEGACY_TOUCH_OUTCOME_MAP = {
    # Conservative mapping for legacy processed exports.
    "P": "retained",
    "O": "lost",
    "C": "challenged",
}


@dataclass
class BallDriveDataConfig:
    source_root: str = "data/processed"
    source_format: str = "pff_match_triplets"
    include_open_play_null: bool = True
    split_seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    season_sample_ratio: Optional[float] = None


def _extract_match_id_from_text(value: Any) -> Optional[int]:
    if value is None:
        return None
    matches = re.findall(r"(\d+)", str(value))
    if not matches:
        return None
    return int(matches[-1])


def _normalize_token(value: Any) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip().lower()
    if not token:
        return None
    if token in {"nan", "none", "null", "na", "n/a", "nat"}:
        return None
    return token


def _normalize_set_piece(value: Any) -> Optional[str]:
    token = _normalize_token(value)
    if token is None:
        return None
    mapping = {
        "open_play": "open_play",
        "corner": "corner",
        "free_kick": "free_kick",
        "throw_in": "throw_in",
        "penalty": "penalty",
        "goal_kick": "goal_kick",
        "kick_off": "kick_off",
        "o": "open_play",
        "c": "corner",
        "f": "free_kick",
        "t": "throw_in",
        "p": "penalty",
        "g": "goal_kick",
        "k": "kick_off",
    }
    return mapping.get(token, token)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return int(round(number))


def _resolve_success_label(row: pd.Series) -> Tuple[Optional[int], str]:
    carry_success = row.get("carry_success")
    parsed_success = _safe_int(carry_success)
    if parsed_success in (0, 1):
        return parsed_success, "carry_success"

    outcome = _normalize_token(row.get("carry_outcome"))
    if outcome in OUTCOME_SUCCESS_MAP:
        return OUTCOME_SUCCESS_MAP[outcome], "carry_outcome"
    if outcome in OUTCOME_EXCLUDED:
        return None, "excluded_outcome"
    return None, "unknown_outcome"


def discover_ball_drive_sources(config: BallDriveDataConfig) -> List[Dict[str, Any]]:
    try:
        sources = discover_action_sources(
            directory=config.source_root,
            source_format=config.source_format,
            prefer_parquet=True,
        )
        if sources:
            return sources
    except FileNotFoundError:
        pass

    processed_root = (REPO_ROOT / "data" / "processed").resolve()
    event_root = processed_root / "event"
    tracking_root = processed_root / "tracking"

    if not event_root.exists() or not tracking_root.exists():
        return []

    event_files = sorted(event_root.glob("Ball_drive_event*.csv"))
    action_files = sorted(
        path
        for path in tracking_root.glob("final_ball_drive_*.csv")
        if "_track_" not in path.name
    )
    track_files = sorted(tracking_root.glob("final_ball_drive_track_*.csv"))

    event_by_match = {
        _extract_match_id_from_text(path.name): path
        for path in event_files
        if _extract_match_id_from_text(path.name) is not None
    }
    track_by_match = {
        _extract_match_id_from_text(path.name): path
        for path in track_files
        if _extract_match_id_from_text(path.name) is not None
    }

    sources: List[Dict[str, Any]] = []
    for action_path in action_files:
        match_id = _extract_match_id_from_text(action_path.name)
        if match_id is None:
            continue
        source = {
            "source_kind": "legacy_ball_drive_csv",
            "source_format": "legacy_ball_drive_csv",
            "source_name": action_path.name,
            "source_path": str(action_path.resolve()),
            "match_id": int(match_id),
            "legacy_event_path": str(event_by_match[match_id].resolve()) if match_id in event_by_match else "",
            "legacy_tracking_path": str(track_by_match[match_id].resolve()) if match_id in track_by_match else "",
        }
        sources.append(source)

    return sources


def _read_source_data(source: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    source_kind = str(source.get("source_kind"))
    if source_kind == "pff_match_triplets":
        events_df = pd.read_parquet(source["events_path"])
        tracking_df = pd.read_parquet(source["tracking_path"])
        return events_df, tracking_df

    if source_kind == "legacy_ball_drive_csv":
        events_df = pd.read_csv(source["source_path"])
        tracking_path = str(source.get("legacy_tracking_path") or "")
        tracking_df = pd.read_csv(tracking_path) if tracking_path else pd.DataFrame()
        return events_df, tracking_df

    raise ValueError(f"Unsupported BallDrive source_kind: {source_kind}")


def _carry_rows_from_legacy_source(source: Dict[str, Any]) -> pd.DataFrame:
    actions_df = pd.read_csv(source["source_path"])
    if actions_df.empty:
        return pd.DataFrame()

    actions = actions_df.copy()
    actions["game_id"] = pd.to_numeric(actions.get("game_id"), errors="coerce")
    actions["game_event_id"] = pd.to_numeric(actions.get("game_event_id"), errors="coerce")
    actions["possession_event_id"] = pd.to_numeric(actions.get("possession_event_id"), errors="coerce")
    actions["player_id"] = pd.to_numeric(actions.get("player_id"), errors="coerce")
    actions["team_id"] = pd.to_numeric(actions.get("team_id"), errors="coerce")
    actions["start_frame_id"] = pd.to_numeric(actions.get("start_frame"), errors="coerce")
    actions["end_frame_id"] = pd.to_numeric(actions.get("end_frame"), errors="coerce")
    actions["frame_num"] = pd.to_numeric(actions.get("frame_num"), errors="coerce")
    actions["ball_x"] = pd.to_numeric(actions.get("ball_x"), errors="coerce")
    actions["ball_y"] = pd.to_numeric(actions.get("ball_y"), errors="coerce")
    actions["duration"] = pd.to_numeric(actions.get("duration"), errors="coerce")

    def _to_team_side(home_value: Any) -> str:
        if isinstance(home_value, bool):
            return "home" if home_value else "away"
        token = str(home_value).strip().lower()
        return "home" if token in {"1", "true", "t", "yes", "y"} else "away"

    def _frame_to_seconds(series: pd.Series) -> float:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return float("nan")
        return float(values.iloc[0]) / 10.0

    grouped = actions.sort_values(["game_event_id", "possession_event_id", "frame_num"]).groupby(
        ["game_id", "game_event_id", "possession_event_id", "player_id", "team_id", "start_frame_id", "end_frame_id"],
        as_index=False,
    )

    carries = grouped.agg(
        team_side=("home_team", lambda s: _to_team_side(s.dropna().iloc[0]) if not s.dropna().empty else "away"),
        elapsed_seconds_start=("start_frame_id", _frame_to_seconds),
        elapsed_seconds_end=("end_frame_id", _frame_to_seconds),
        ball_x_start=("ball_x", "first"),
        ball_y_start=("ball_y", "first"),
        ball_x_end=("ball_x", "last"),
        ball_y_end=("ball_y", "last"),
    )

    legacy_event_path = str(source.get("legacy_event_path") or "")
    if legacy_event_path:
        outcome_df = pd.read_csv(legacy_event_path)
        outcome_df["game_event_id"] = pd.to_numeric(outcome_df.get("game_event_id"), errors="coerce")
        outcome_df["possession_event_id"] = pd.to_numeric(outcome_df.get("possession_event_id"), errors="coerce")
        outcome_df["carry_outcome"] = outcome_df.get("touchOutcomeType").map(LEGACY_TOUCH_OUTCOME_MAP)

        carries = carries.merge(
            outcome_df[["game_event_id", "possession_event_id", "carry_outcome"]],
            on=["game_event_id", "possession_event_id"],
            how="left",
        )
    else:
        carries["carry_outcome"] = np.nan

    labels = carries.apply(_resolve_success_label, axis=1, result_type="expand")
    carries["y_success_provider"] = labels[0]
    carries["success_source"] = labels[1]
    carries["set_piece_normalized"] = np.nan
    carries["carry_success"] = np.nan

    carries["source_name"] = str(source.get("source_name", "unknown"))
    carries["source_path"] = str(source.get("source_path", ""))
    carries["source_events_path"] = ""
    carries["source_tracking_path"] = str(source.get("legacy_tracking_path", ""))
    carries["source_players_path"] = ""
    carries["source_kind"] = "legacy_ball_drive_csv"

    return carries[
        [
            "game_id",
            "game_event_id",
            "possession_event_id",
            "player_id",
            "team_id",
            "team_side",
            "start_frame_id",
            "end_frame_id",
            "elapsed_seconds_start",
            "elapsed_seconds_end",
            "ball_x_start",
            "ball_y_start",
            "ball_x_end",
            "ball_y_end",
            "set_piece_normalized",
            "carry_success",
            "carry_outcome",
            "y_success_provider",
            "success_source",
            "source_name",
            "source_path",
            "source_events_path",
            "source_tracking_path",
            "source_players_path",
            "source_kind",
        ]
    ].copy()
    return events_df, tracking_df


def _carry_rows_from_source(source: Dict[str, Any]) -> pd.DataFrame:
    if str(source.get("source_kind")) == "legacy_ball_drive_csv":
        return _carry_rows_from_legacy_source(source)

    events_df, _ = _read_source_data(source)

    events = events_df.copy()
    events["possession_type_norm"] = events.get("possession_type", pd.Series(index=events.index)).astype(str).str.lower().str.strip()
    carries = events[events["possession_type_norm"] == "carry"].copy()
    if carries.empty:
        return pd.DataFrame()

    carries["game_id"] = pd.to_numeric(carries.get("match_id"), errors="coerce")
    carries["game_event_id"] = pd.to_numeric(carries.get("event_id"), errors="coerce")
    carries["possession_event_id"] = pd.to_numeric(carries.get("possession_id"), errors="coerce")
    carries["player_id"] = pd.to_numeric(carries.get("player_id"), errors="coerce")
    carries["team_id"] = pd.to_numeric(carries.get("team_id"), errors="coerce")
    frame_id_series = pd.to_numeric(carries.get("frame_id"), errors="coerce")
    elapsed_seconds_series = pd.to_numeric(carries.get("elapsed_seconds"), errors="coerce")
    event_x_series = pd.to_numeric(carries.get("x"), errors="coerce")
    event_y_series = pd.to_numeric(carries.get("y"), errors="coerce")
    ball_x_series = pd.to_numeric(carries.get("ball_x"), errors="coerce").fillna(event_x_series)
    ball_y_series = pd.to_numeric(carries.get("ball_y"), errors="coerce").fillna(event_y_series)

    def _coerce_with_fallback(column_name: str, fallback_series: pd.Series) -> pd.Series:
        if column_name not in carries.columns:
            return fallback_series.copy()
        return pd.to_numeric(carries[column_name], errors="coerce").fillna(fallback_series)

    carries["start_frame_id"] = _coerce_with_fallback("start_frame_id", frame_id_series)
    carries["end_frame_id"] = _coerce_with_fallback("end_frame_id", frame_id_series)

    carries["elapsed_seconds_start"] = _coerce_with_fallback("elapsed_seconds_start", elapsed_seconds_series)
    carries["elapsed_seconds_end"] = _coerce_with_fallback("elapsed_seconds_end", elapsed_seconds_series)

    carries["ball_x_start"] = _coerce_with_fallback("ball_x_start", ball_x_series)
    carries["ball_y_start"] = _coerce_with_fallback("ball_y_start", ball_y_series)
    carries["ball_x_end"] = _coerce_with_fallback("ball_x_end", ball_x_series)
    carries["ball_y_end"] = _coerce_with_fallback("ball_y_end", ball_y_series)

    carries["set_piece_normalized"] = carries.get("set_piece", pd.Series(index=carries.index)).map(_normalize_set_piece)

    labels = carries.apply(_resolve_success_label, axis=1, result_type="expand")
    carries["y_success_provider"] = labels[0]
    carries["success_source"] = labels[1]

    carries = carries[
        [
            "game_id",
            "game_event_id",
            "possession_event_id",
            "player_id",
            "team_id",
            "team_side",
            "start_frame_id",
            "end_frame_id",
            "elapsed_seconds_start",
            "elapsed_seconds_end",
            "ball_x_start",
            "ball_y_start",
            "ball_x_end",
            "ball_y_end",
            "set_piece_normalized",
            "carry_success",
            "carry_outcome",
            "y_success_provider",
            "success_source",
        ]
    ].copy()

    carries["source_name"] = str(source.get("source_name", "unknown"))
    carries["source_path"] = str(source.get("source_path", ""))
    carries["source_events_path"] = str(source.get("events_path", ""))
    carries["source_tracking_path"] = str(source.get("tracking_path", ""))
    carries["source_players_path"] = str(source.get("players_path", ""))
    carries["source_kind"] = str(source.get("source_kind", "pff_match_triplets"))

    return carries


def build_ball_drive_canonical_dataset(config: Optional[BallDriveDataConfig] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    config = config or BallDriveDataConfig()
    sources = discover_ball_drive_sources(config)
    sources, season_sampling_summary = sample_sources_by_season(
        sources,
        season_sample_ratio=config.season_sample_ratio,
        split_seed=config.split_seed,
    )

    chunks: List[pd.DataFrame] = []
    for source in sources:
        source_rows = _carry_rows_from_source(source)
        if not source_rows.empty:
            chunks.append(source_rows)

    if not chunks:
        return pd.DataFrame(), {
            "rows_total": 0,
            "rows_labeled": 0,
            "rows_excluded": 0,
            "source_count": len(sources),
            "season_sampling": season_sampling_summary,
        }

    canonical = pd.concat(chunks, ignore_index=True)
    canonical = canonical.dropna(
        subset=[
            "game_id",
            "game_event_id",
            "possession_event_id",
            "player_id",
            "team_id",
            "start_frame_id",
            "elapsed_seconds_start",
            "ball_x_start",
            "ball_y_start",
        ]
    ).copy()

    canonical["game_id"] = pd.to_numeric(canonical["game_id"], errors="coerce").astype(int)
    canonical["game_event_id"] = pd.to_numeric(canonical["game_event_id"], errors="coerce").astype(int)
    canonical["possession_event_id"] = pd.to_numeric(canonical["possession_event_id"], errors="coerce").astype(int)
    canonical["player_id"] = pd.to_numeric(canonical["player_id"], errors="coerce").astype(int)
    canonical["team_id"] = pd.to_numeric(canonical["team_id"], errors="coerce").astype(int)
    canonical["start_frame_id"] = pd.to_numeric(canonical["start_frame_id"], errors="coerce").astype(int)
    canonical["end_frame_id"] = pd.to_numeric(canonical["end_frame_id"], errors="coerce").astype(int)

    summary = {
        "rows_total": int(len(canonical)),
        "rows_labeled": int(canonical["y_success_provider"].notna().sum()),
        "rows_excluded": int(canonical["y_success_provider"].isna().sum()),
        "excluded_outcome_count": int((canonical["success_source"] == "excluded_outcome").sum()),
        "source_count": int(len(sources)),
        "season_sampling": season_sampling_summary,
    }

    return canonical, summary


def segment_ball_drives(canonical_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    discarded_remainders = 0
    lost_terminal_failures = 0

    for _, row in canonical_df.iterrows():
        start_t = _safe_float(row.get("elapsed_seconds_start"))
        end_t = _safe_float(row.get("elapsed_seconds_end"))
        start_frame = _safe_int(row.get("start_frame_id"))
        end_frame = _safe_int(row.get("end_frame_id"))
        provider_label = row.get("y_success_provider")
        provider_label = int(provider_label) if pd.notna(provider_label) else None

        if start_t is None:
            continue
        if end_t is None:
            end_t = start_t

        duration = max(0.0, end_t - start_t)

        if duration <= 1.0:
            sample = row.to_dict()
            sample.update(
                {
                    "segment_index": 0,
                    "segment_start_seconds": start_t,
                    "segment_end_seconds": end_t,
                    "segment_start_frame_id": start_frame,
                    "segment_end_frame_id": end_frame,
                    "segment_reaches_terminal": True,
                    "segment_duration_seconds": duration,
                    "y_success": provider_label,
                }
            )
            rows.append(sample)
            continue

        full_chunks = int(duration // 1.0)
        remainder = duration - float(full_chunks)
        if remainder > 1e-9:
            discarded_remainders += 1

        terminal_assigned = False
        for idx in range(full_chunks):
            seg_start_t = start_t + float(idx)
            seg_end_t = start_t + float(idx + 1)
            reaches_terminal = seg_end_t >= (end_t - 1e-9)

            seg_start_frame = start_frame + idx * 10 if start_frame is not None else None
            seg_end_frame = min(end_frame, seg_start_frame + 10) if (end_frame is not None and seg_start_frame is not None) else end_frame

            sample = row.to_dict()
            sample.update(
                {
                    "segment_index": idx,
                    "segment_start_seconds": seg_start_t,
                    "segment_end_seconds": seg_end_t,
                    "segment_start_frame_id": seg_start_frame,
                    "segment_end_frame_id": seg_end_frame,
                    "segment_reaches_terminal": reaches_terminal,
                    "segment_duration_seconds": 1.0,
                }
            )

            if reaches_terminal:
                sample["y_success"] = provider_label
                terminal_assigned = True
            else:
                sample["y_success"] = 1
            rows.append(sample)

        if (not terminal_assigned) and provider_label == 0:
            lost_terminal_failures += 1

    segmented = pd.DataFrame(rows)
    summary = {
        "rows_total": int(len(segmented)),
        "discarded_remainders": int(discarded_remainders),
        "lost_terminal_failures": int(lost_terminal_failures),
        "label_counts": segmented.get("y_success", pd.Series(dtype=float)).value_counts(dropna=False).to_dict(),
    }
    return segmented, summary


def build_ball_drive_split_manifest(
    segmented_df: pd.DataFrame,
    config: Optional[BallDriveDataConfig] = None,
    manifest_relpath: str = "data/processed/cache/splits/ball_drive_split_manifest.json",
) -> Tuple[Dict[str, Any], Path]:
    config = config or BallDriveDataConfig()
    if segmented_df.empty:
        manifest = {
            "split_mode": "match",
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "split_seed": config.split_seed,
            "match_assignments": {},
        }
    else:
        match_ids = sorted({int(x) for x in segmented_df["game_id"].dropna().astype(int).tolist()})
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

        train_ids = set(shuffled[:n_train])
        val_ids = set(shuffled[n_train : n_train + n_val])
        test_ids = set(shuffled[n_train + n_val : n_train + n_val + n_test])

        assignments = {}
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


def apply_match_splits(segmented_df: pd.DataFrame, manifest: Dict[str, Any]) -> pd.DataFrame:
    if segmented_df.empty:
        output = segmented_df.copy()
        output["split"] = pd.Series(dtype=str)
        return output

    assignments = {str(k): str(v) for k, v in (manifest.get("match_assignments") or {}).items()}
    result = segmented_df.copy()
    result["split"] = result["game_id"].map(lambda value: assignments.get(str(int(value)), "train") if pd.notna(value) else "train")
    return result


def attach_reward_labels(segmented_df: pd.DataFrame, include_open_play_null: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if segmented_df.empty:
        return segmented_df.copy(), {"rows_total": 0, "rows_labeled": 0}

    labeler = PassRewardLabeler(
        event_root=REPO_ROOT / "data" / "raw" / "event",
        horizon_seconds=15.0,
        include_open_play_null=include_open_play_null,
    )

    parts: List[pd.DataFrame] = []
    source_status: List[Dict[str, Any]] = []

    grouped = segmented_df.groupby(["source_kind", "source_events_path", "source_tracking_path", "source_name"], dropna=False)
    for (source_kind, events_path, tracking_path, source_name), group in grouped:
        normalized_kind = str(source_kind or "").strip().lower()

        if normalized_kind == "legacy_ball_drive_csv":
            labeled, summary = labeler.label_action_dataframe(
                action_df=group,
                action_type="carry",
                source_kind="legacy_wide",
                source_filename=str(source_name) if source_name else None,
                drop_unlabeled=False,
            )
            parts.append(labeled)
            source_status.append(summary)
            continue

        if not events_path:
            copy_group = group.copy()
            copy_group["reward_label"] = np.nan
            parts.append(copy_group)
            continue

        labeled, summary = labeler.label_action_dataframe(
            action_df=group,
            action_type="carry",
            source_kind="pff_match_triplets",
            processed_events_path=Path(events_path),
            processed_tracking_path=Path(tracking_path) if tracking_path else None,
            drop_unlabeled=False,
        )
        parts.append(labeled)
        source_status.append(summary)

    concat_parts = [part.dropna(axis=1, how="all") for part in parts if not part.empty]
    merged = pd.concat(concat_parts, ignore_index=False).sort_index().reset_index(drop=True) if concat_parts else pd.DataFrame()
    merged["reward_G"] = pd.to_numeric(merged.get("reward_label"), errors="coerce")

    summary = {
        "rows_total": int(len(merged)),
        "rows_labeled": int(merged["reward_G"].notna().sum()),
        "source_summaries": source_status,
    }
    return merged, summary


def get_ball_drive_cache_root() -> Path:
    return get_cache_root() / "ball_drive"
