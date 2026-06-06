from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


NULL_EQUIVALENT_TOKENS = {"", "nan", "none", "null", "na", "n/a", "nat"}
GOAL_OUT_TYPES = {"H", "A"}
PROCESSED_GOAL_TOKENS = {"goal", "g"}


def _normalize_processed_set_piece(value: Any) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip().lower()
    if not token or token in NULL_EQUIVALENT_TOKENS:
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


def _normalize_token(value: Any) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip()
    if not token:
        return None
    if token.lower() in NULL_EQUIVALENT_TOKENS:
        return None
    return token.upper()


def _normalize_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    if number.is_integer():
        return int(number)
    return int(round(number))


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _extract_event_time(record: Dict[str, Any]) -> Optional[float]:
    game_events = record.get("GAME_EVENTS") or {}
    possession_events = record.get("POSSESSION_EVENTS") or {}

    for candidate in (
        record.get("EVENT_TIME"),
        possession_events.get("EVENT_GAME_CLOCK"),
        game_events.get("START_GAME_CLOCK"),
    ):
        normalized = _as_float(candidate)
        if normalized is not None:
            return normalized
    return None


def _is_open_play(setpiece_type: Optional[str], include_null_setpiece: bool) -> bool:
    normalized = _normalize_token(setpiece_type)
    if normalized == "O":
        return True
    if normalized is None and include_null_setpiece:
        return True
    return False


@dataclass(frozen=True)
class ParsedEvent:
    game_event_id: Optional[int]
    possession_event_id: Optional[int]
    team_id: Optional[int]
    home_team: Optional[bool]
    event_time: Optional[float]
    setpiece_type: Optional[str]
    game_event_type: Optional[str]
    out_type: Optional[str]
    possession_event_type: Optional[str]


class GameEventIndex:
    def __init__(self, records: List[Dict[str, Any]], horizon_seconds: float = 15.0):
        self.horizon_seconds = float(horizon_seconds)

        self._by_pair: Dict[Tuple[int, int], List[ParsedEvent]] = {}
        self._by_game_event_id: Dict[int, List[ParsedEvent]] = {}
        self._by_possession_event_id: Dict[int, List[ParsedEvent]] = {}

        self._goal_events: List[ParsedEvent] = []
        self._home_team_id: Optional[int] = None
        self._away_team_id: Optional[int] = None

        for record in records:
            parsed = self._parse_event(record)
            self._store_event(parsed)

        self._goal_events.sort(key=lambda event: event.event_time if event.event_time is not None else float("inf"))

    def _parse_event(self, record: Dict[str, Any]) -> ParsedEvent:
        game_events = record.get("GAME_EVENTS") or {}
        possession_events = record.get("POSSESSION_EVENTS") or {}

        home_team_value = game_events.get("HOME_TEAM")
        home_team: Optional[bool]
        if isinstance(home_team_value, bool):
            home_team = home_team_value
        else:
            home_team = None

        return ParsedEvent(
            game_event_id=_normalize_id(record.get("GAME_EVENT_ID")),
            possession_event_id=_normalize_id(record.get("POSSESSION_EVENT_ID")),
            team_id=_normalize_id(game_events.get("TEAM_ID")),
            home_team=home_team,
            event_time=_extract_event_time(record),
            setpiece_type=_normalize_token(game_events.get("SETPIECE_TYPE")),
            game_event_type=_normalize_token(game_events.get("GAME_EVENT_TYPE")),
            out_type=_normalize_token(game_events.get("OUT_TYPE")),
            possession_event_type=_normalize_token(possession_events.get("POSSESSION_EVENT_TYPE")),
        )

    def _store_event(self, event: ParsedEvent) -> None:
        if event.game_event_id is not None and event.possession_event_id is not None:
            key = (event.game_event_id, event.possession_event_id)
            self._by_pair.setdefault(key, []).append(event)

        if event.game_event_id is not None:
            self._by_game_event_id.setdefault(event.game_event_id, []).append(event)

        if event.possession_event_id is not None:
            self._by_possession_event_id.setdefault(event.possession_event_id, []).append(event)

        if event.team_id is not None and event.home_team is not None:
            if event.home_team and self._home_team_id is None:
                self._home_team_id = event.team_id
            if (not event.home_team) and self._away_team_id is None:
                self._away_team_id = event.team_id

        if (
            event.game_event_type == "OUT"
            and event.out_type in GOAL_OUT_TYPES
            and event.event_time is not None
        ):
            self._goal_events.append(event)

    @staticmethod
    def _select_best_candidate(candidates: List[ParsedEvent], team_id: Optional[int]) -> Optional[ParsedEvent]:
        if not candidates:
            return None

        subset = [candidate for candidate in candidates if candidate.possession_event_type == "PA"]
        if not subset:
            subset = list(candidates)

        if team_id is not None:
            team_subset = [candidate for candidate in subset if candidate.team_id == team_id]
            if team_subset:
                subset = team_subset

        timed_subset = [candidate for candidate in subset if candidate.event_time is not None]
        if timed_subset:
            subset = timed_subset

        return subset[0]

    def resolve_pass_event(self, pass_row: pd.Series) -> Tuple[Optional[ParsedEvent], str]:
        game_event_id = _normalize_id(pass_row.get("game_event_id"))
        possession_event_id = _normalize_id(pass_row.get("possession_event_id"))
        team_id = _normalize_id(pass_row.get("team_id"))

        if game_event_id is not None and possession_event_id is not None:
            from_pair = self._select_best_candidate(
                self._by_pair.get((game_event_id, possession_event_id), []),
                team_id,
            )
            if from_pair is not None:
                return from_pair, "pair"

        if game_event_id is not None:
            from_game_event = self._select_best_candidate(
                self._by_game_event_id.get(game_event_id, []),
                team_id,
            )
            if from_game_event is not None:
                return from_game_event, "game_event_id"

        if possession_event_id is not None:
            from_possession_event = self._select_best_candidate(
                self._by_possession_event_id.get(possession_event_id, []),
                team_id,
            )
            if from_possession_event is not None:
                return from_possession_event, "possession_event_id"

        return None, "unmatched"

    def _scoring_team_id(self, goal_event: ParsedEvent) -> Optional[int]:
        if goal_event.out_type == "H":
            return self._home_team_id
        if goal_event.out_type == "A":
            return self._away_team_id
        return None

    def _find_first_goal_after(self, event_time: float) -> Optional[ParsedEvent]:
        horizon = event_time + self.horizon_seconds
        for goal_event in self._goal_events:
            goal_time = goal_event.event_time
            if goal_time is None:
                continue
            if goal_time <= event_time:
                continue
            if goal_time > horizon:
                break
            return goal_event
        return None

    def label_pass_row(
        self,
        pass_row: pd.Series,
        include_open_play_null: bool = True,
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        resolved_event, join_strategy = self.resolve_pass_event(pass_row)

        metadata: Dict[str, Any] = {
            "join_strategy": join_strategy,
            "status": "ok",
            "pass_event_time": None,
            "pass_setpiece_type": None,
            "first_goal_time": None,
            "first_goal_out_type": None,
        }

        if resolved_event is None:
            metadata["status"] = "join_missing"
            return None, metadata

        metadata["pass_event_time"] = resolved_event.event_time
        metadata["pass_setpiece_type"] = resolved_event.setpiece_type

        if not _is_open_play(resolved_event.setpiece_type, include_open_play_null):
            metadata["status"] = "setpiece_filtered"
            return None, metadata

        if resolved_event.event_time is None:
            metadata["status"] = "missing_pass_time"
            return None, metadata

        first_goal = self._find_first_goal_after(resolved_event.event_time)
        if first_goal is None:
            metadata["status"] = "no_goal_window"
            return 0, metadata

        metadata["first_goal_time"] = first_goal.event_time
        metadata["first_goal_out_type"] = first_goal.out_type

        scoring_team_id = self._scoring_team_id(first_goal)
        pass_team_id = _normalize_id(pass_row.get("team_id"))
        if pass_team_id is None:
            pass_team_id = resolved_event.team_id

        if scoring_team_id is None or pass_team_id is None:
            metadata["status"] = "unknown_goal_team"
            return 0, metadata

        label = 1 if scoring_team_id == pass_team_id else -1
        metadata["status"] = "goal_scored" if label == 1 else "goal_conceded"
        return label, metadata

    def label_action_row(
        self,
        action_row: pd.Series,
        action_type: str = "pass",
        include_open_play_null: bool = True,
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        label, metadata = self.label_pass_row(
            action_row,
            include_open_play_null=include_open_play_null,
        )
        metadata["action_type"] = str(action_type).strip().lower() or "action"
        return label, metadata


@dataclass(frozen=True)
class ProcessedActionEvent:
    game_event_id: Optional[int]
    possession_event_id: Optional[int]
    team_id: Optional[int]
    event_time: Optional[float]
    set_piece: Optional[str]
    possession_type: Optional[str]


@dataclass(frozen=True)
class ProcessedGoalEvent:
    team_id: Optional[int]
    event_time: Optional[float]


class ProcessedEventIndex:
    def __init__(self, events_df: pd.DataFrame, tracking_df: Optional[pd.DataFrame], horizon_seconds: float = 15.0):
        self.horizon_seconds = float(horizon_seconds)

        self._action_by_pair: Dict[str, Dict[Tuple[int, int], List[ProcessedActionEvent]]] = {}
        self._action_by_game_event_id: Dict[str, Dict[int, List[ProcessedActionEvent]]] = {}
        self._action_by_possession_event_id: Dict[str, Dict[int, List[ProcessedActionEvent]]] = {}
        self._goal_events: List[ProcessedGoalEvent] = []

        normalized_events = self._prepare_events(events_df, tracking_df)
        action_df = normalized_events[normalized_events["possession_type_norm"].notna()].copy()

        for _, row in action_df.iterrows():
            possession_type = str(row.get("possession_type_norm") or "").strip().lower()
            if not possession_type:
                continue

            event = ProcessedActionEvent(
                game_event_id=_normalize_id(row.get("event_id")),
                possession_event_id=_normalize_id(row.get("possession_id")),
                team_id=_normalize_id(row.get("team_id")),
                event_time=_as_float(row.get("elapsed_seconds")),
                set_piece=_normalize_processed_set_piece(row.get("set_piece")),
                possession_type=possession_type,
            )

            self._store_action_event(event)

        shot_df = normalized_events[normalized_events["possession_type_norm"] == "shot"].copy()
        shot_df["shot_outcome_norm"] = shot_df.get("shot_outcome", pd.Series(index=shot_df.index)).astype(str).str.lower().str.strip()
        goals_df = shot_df[shot_df["shot_outcome_norm"].isin(PROCESSED_GOAL_TOKENS)]

        for _, row in goals_df.iterrows():
            goal_event = ProcessedGoalEvent(
                team_id=_normalize_id(row.get("team_id")),
                event_time=_as_float(row.get("elapsed_seconds")),
            )
            if goal_event.event_time is not None:
                self._goal_events.append(goal_event)

        self._goal_events.sort(key=lambda event: event.event_time if event.event_time is not None else float("inf"))

    @staticmethod
    def _prepare_events(events_df: pd.DataFrame, tracking_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        normalized = events_df.copy()
        normalized["possession_type_norm"] = normalized.get("possession_type", pd.Series(index=normalized.index)).astype(str).str.lower().str.strip()

        has_elapsed = "elapsed_seconds" in normalized.columns and normalized["elapsed_seconds"].notna().any()
        if has_elapsed:
            normalized["elapsed_seconds"] = pd.to_numeric(normalized["elapsed_seconds"], errors="coerce")
            return normalized

        if tracking_df is None:
            return normalized

        if "match_id" not in normalized.columns or "frame_id" not in normalized.columns:
            return normalized
        if "match_id" not in tracking_df.columns or "frame_id" not in tracking_df.columns or "elapsed_seconds" not in tracking_df.columns:
            return normalized

        tracking = tracking_df[["match_id", "frame_id", "elapsed_seconds"]].copy()
        tracking["match_id"] = pd.to_numeric(tracking["match_id"], errors="coerce").astype("Int64")
        tracking["frame_id"] = pd.to_numeric(tracking["frame_id"], errors="coerce").astype("Int64")
        tracking["elapsed_seconds"] = pd.to_numeric(tracking["elapsed_seconds"], errors="coerce")
        tracking = tracking.dropna(subset=["match_id", "frame_id", "elapsed_seconds"])
        tracking = tracking.drop_duplicates(subset=["match_id", "frame_id"], keep="first")

        normalized["match_id"] = pd.to_numeric(normalized["match_id"], errors="coerce").astype("Int64")
        normalized["frame_id"] = pd.to_numeric(normalized["frame_id"], errors="coerce").astype("Int64")

        merged = normalized.merge(
            tracking,
            on=["match_id", "frame_id"],
            how="left",
            suffixes=("", "_tracking"),
        )
        merged["elapsed_seconds"] = merged["elapsed_seconds"].where(
            merged["elapsed_seconds"].notna(),
            merged["elapsed_seconds_tracking"],
        )
        return merged.drop(columns=[col for col in ["elapsed_seconds_tracking"] if col in merged.columns])

    @staticmethod
    def _select_best_candidate(candidates: List[ProcessedActionEvent], team_id: Optional[int]) -> Optional[ProcessedActionEvent]:
        if not candidates:
            return None

        subset = list(candidates)
        if team_id is not None:
            team_subset = [candidate for candidate in subset if candidate.team_id == team_id]
            if team_subset:
                subset = team_subset

        timed_subset = [candidate for candidate in subset if candidate.event_time is not None]
        if timed_subset:
            subset = timed_subset
        return subset[0]

    def _store_action_event(self, event: ProcessedActionEvent) -> None:
        action_type = str(event.possession_type or "").strip().lower()
        if not action_type:
            return

        by_pair = self._action_by_pair.setdefault(action_type, {})
        by_game_event_id = self._action_by_game_event_id.setdefault(action_type, {})
        by_possession_event_id = self._action_by_possession_event_id.setdefault(action_type, {})

        if event.game_event_id is not None and event.possession_event_id is not None:
            by_pair.setdefault((event.game_event_id, event.possession_event_id), []).append(event)
        if event.game_event_id is not None:
            by_game_event_id.setdefault(event.game_event_id, []).append(event)
        if event.possession_event_id is not None:
            by_possession_event_id.setdefault(event.possession_event_id, []).append(event)

    def resolve_action_event(
        self,
        action_row: pd.Series,
        action_type: str = "pass",
    ) -> Tuple[Optional[ProcessedActionEvent], str]:
        normalized_action_type = str(action_type).strip().lower()
        game_event_id = _normalize_id(action_row.get("game_event_id"))
        if game_event_id is None:
            game_event_id = _normalize_id(action_row.get("event_id"))
        possession_event_id = _normalize_id(action_row.get("possession_event_id"))
        if possession_event_id is None:
            possession_event_id = _normalize_id(action_row.get("possession_id"))
        team_id = _normalize_id(action_row.get("team_id"))

        by_pair = self._action_by_pair.get(normalized_action_type, {})
        by_game_event_id = self._action_by_game_event_id.get(normalized_action_type, {})
        by_possession_event_id = self._action_by_possession_event_id.get(normalized_action_type, {})

        if game_event_id is not None and possession_event_id is not None:
            from_pair = self._select_best_candidate(
                by_pair.get((game_event_id, possession_event_id), []),
                team_id,
            )
            if from_pair is not None:
                return from_pair, "pair"

        if game_event_id is not None:
            from_game_event = self._select_best_candidate(
                by_game_event_id.get(game_event_id, []),
                team_id,
            )
            if from_game_event is not None:
                return from_game_event, "game_event_id"

        if possession_event_id is not None:
            from_possession_event = self._select_best_candidate(
                by_possession_event_id.get(possession_event_id, []),
                team_id,
            )
            if from_possession_event is not None:
                return from_possession_event, "possession_event_id"

        return None, "unmatched"

    def resolve_pass_event(self, pass_row: pd.Series) -> Tuple[Optional[ProcessedActionEvent], str]:
        return self.resolve_action_event(pass_row, action_type="pass")

    def _find_first_goal_after(self, event_time: float) -> Optional[ProcessedGoalEvent]:
        horizon = event_time + self.horizon_seconds
        for goal_event in self._goal_events:
            goal_time = goal_event.event_time
            if goal_time is None:
                continue
            if goal_time <= event_time:
                continue
            if goal_time > horizon:
                break
            return goal_event
        return None

    def label_action_row(
        self,
        action_row: pd.Series,
        action_type: str = "pass",
        include_open_play_null: bool = True,
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        resolved_event, join_strategy = self.resolve_action_event(action_row, action_type=action_type)

        normalized_action_type = str(action_type).strip().lower() or "action"
        action_prefix = normalized_action_type

        metadata: Dict[str, Any] = {
            "join_strategy": join_strategy,
            "status": "ok",
            "action_type": normalized_action_type,
            "action_event_time": None,
            "action_setpiece_type": None,
            "first_goal_time": None,
            "first_goal_out_type": None,
        }

        if resolved_event is None:
            metadata["status"] = "join_missing"
            return None, metadata

        metadata["action_event_time"] = resolved_event.event_time
        metadata["action_setpiece_type"] = resolved_event.set_piece
        metadata[f"{action_prefix}_event_time"] = resolved_event.event_time
        metadata[f"{action_prefix}_setpiece_type"] = resolved_event.set_piece

        if normalized_action_type == "pass":
            metadata["pass_event_time"] = resolved_event.event_time
            metadata["pass_setpiece_type"] = resolved_event.set_piece

        if resolved_event.set_piece != "open_play":
            if not (resolved_event.set_piece is None and include_open_play_null):
                metadata["status"] = "setpiece_filtered"
                return None, metadata

        if resolved_event.event_time is None:
            metadata["status"] = "missing_pass_time"
            return None, metadata

        first_goal = self._find_first_goal_after(resolved_event.event_time)
        if first_goal is None:
            metadata["status"] = "no_goal_window"
            return 0, metadata

        metadata["first_goal_time"] = first_goal.event_time
        metadata["first_goal_out_type"] = "GOAL"

        scoring_team_id = first_goal.team_id
        action_team_id = _normalize_id(action_row.get("team_id"))
        if action_team_id is None:
            action_team_id = resolved_event.team_id

        if scoring_team_id is None or action_team_id is None:
            metadata["status"] = "unknown_goal_team"
            return 0, metadata

        label = 1 if scoring_team_id == action_team_id else -1
        metadata["status"] = "goal_scored" if label == 1 else "goal_conceded"
        return label, metadata

    def label_pass_row(
        self,
        pass_row: pd.Series,
        include_open_play_null: bool = True,
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        return self.label_action_row(
            pass_row,
            action_type="pass",
            include_open_play_null=include_open_play_null,
        )


class PassRewardLabeler:
    def __init__(
        self,
        event_root: str | Path,
        horizon_seconds: float = 15.0,
        include_open_play_null: bool = True,
    ):
        self.event_root = Path(event_root)
        self.horizon_seconds = float(horizon_seconds)
        self.include_open_play_null = bool(include_open_play_null)
        self._cache: Dict[int, GameEventIndex] = {}
        self._processed_cache: Dict[Tuple[str, Optional[str]], ProcessedEventIndex] = {}

    @staticmethod
    def _extract_game_id(pass_df: pd.DataFrame, source_filename: Optional[str]) -> Optional[int]:
        if "game_id" in pass_df.columns:
            values = pd.to_numeric(pass_df["game_id"], errors="coerce").dropna()
            if not values.empty:
                return int(values.iloc[0])

        if source_filename:
            match = re.search(r"(\d+)", source_filename)
            if match:
                return int(match.group(1))

        return None

    def _find_event_file(self, game_id: int) -> Path:
        candidates = sorted(self.event_root.glob(f"**/{game_id}.json"))
        if not candidates:
            raise FileNotFoundError(
                f"Could not find raw event JSON for game_id={game_id} under {self.event_root}."
            )
        return candidates[0]

    def _get_game_index(self, game_id: int) -> GameEventIndex:
        if game_id in self._cache:
            return self._cache[game_id]

        event_file = self._find_event_file(game_id)
        with event_file.open("r", encoding="utf-8") as input_file:
            records = json.load(input_file)

        game_index = GameEventIndex(records=records, horizon_seconds=self.horizon_seconds)
        self._cache[game_id] = game_index
        return game_index

    def _get_processed_index(
        self,
        events_path: Path,
        tracking_path: Optional[Path],
    ) -> ProcessedEventIndex:
        cache_key = (str(events_path.resolve()), str(tracking_path.resolve()) if tracking_path else None)
        if cache_key in self._processed_cache:
            return self._processed_cache[cache_key]

        events_df = pd.read_parquet(events_path)
        tracking_df = pd.read_parquet(tracking_path) if tracking_path is not None and tracking_path.exists() else None

        index = ProcessedEventIndex(
            events_df=events_df,
            tracking_df=tracking_df,
            horizon_seconds=self.horizon_seconds,
        )
        self._processed_cache[cache_key] = index
        return index

    def label_action_dataframe(
        self,
        action_df: pd.DataFrame,
        action_type: str = "pass",
        source_filename: Optional[str] = None,
        drop_unlabeled: bool = True,
        source_kind: str = "legacy_wide",
        processed_events_path: Optional[str | Path] = None,
        processed_tracking_path: Optional[str | Path] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        game_id = self._extract_game_id(action_df, source_filename)
        normalized_action_type = str(action_type).strip().lower() or "pass"

        if source_kind == "pff_match_triplets":
            if processed_events_path is None:
                raise ValueError("processed_events_path is required for pff_match_triplets reward labeling")

            events_path = Path(processed_events_path)
            tracking_path = Path(processed_tracking_path) if processed_tracking_path is not None else None
            game_index: Any = self._get_processed_index(events_path, tracking_path)
        else:
            if game_id is None:
                raise ValueError(
                    "Could not resolve game_id from pass dataframe. "
                    "Expected column 'game_id' or a filename containing a numeric game id."
                )
            game_index = self._get_game_index(game_id)

        labeled_df = action_df.copy()
        labels: List[Optional[int]] = []
        statuses: List[str] = []
        join_strategies: List[str] = []
        action_event_times: List[Optional[float]] = []
        action_setpiece_types: List[Optional[str]] = []
        first_goal_times: List[Optional[float]] = []
        first_goal_out_types: List[Optional[str]] = []

        for _, row in labeled_df.iterrows():
            label, metadata = game_index.label_action_row(
                row,
                action_type=normalized_action_type,
                include_open_play_null=self.include_open_play_null,
            )
            labels.append(label)
            statuses.append(str(metadata["status"]))
            join_strategies.append(str(metadata["join_strategy"]))
            action_event_times.append(metadata.get("action_event_time"))
            action_setpiece_types.append(metadata.get("action_setpiece_type"))
            first_goal_times.append(metadata["first_goal_time"])
            first_goal_out_types.append(metadata["first_goal_out_type"])

        labeled_df["reward_label"] = pd.Series(labels, index=labeled_df.index, dtype="float")
        labeled_df["reward_status"] = statuses
        labeled_df["reward_join_strategy"] = join_strategies
        labeled_df["reward_action_event_time"] = action_event_times
        labeled_df["reward_action_setpiece_type"] = action_setpiece_types
        labeled_df[f"reward_{normalized_action_type}_event_time"] = action_event_times
        labeled_df[f"reward_{normalized_action_type}_setpiece_type"] = action_setpiece_types
        if normalized_action_type == "pass":
            labeled_df["reward_pass_event_time"] = action_event_times
            labeled_df["reward_pass_setpiece_type"] = action_setpiece_types
        labeled_df["reward_first_goal_time"] = first_goal_times
        labeled_df["reward_first_goal_out_type"] = first_goal_out_types

        summary: Dict[str, Any] = {
            "game_id": game_id,
            "action_type": normalized_action_type,
            "source_kind": source_kind,
            "rows_total": int(len(labeled_df)),
            "rows_labeled": int(labeled_df["reward_label"].notna().sum()),
            "rows_dropped": int(labeled_df["reward_label"].isna().sum()),
            "status_counts": labeled_df["reward_status"].value_counts(dropna=False).to_dict(),
            "join_strategy_counts": labeled_df["reward_join_strategy"].value_counts(dropna=False).to_dict(),
            "label_counts": labeled_df["reward_label"].value_counts(dropna=False).to_dict(),
        }

        if drop_unlabeled:
            labeled_df = labeled_df[labeled_df["reward_label"].notna()].copy()
            labeled_df["reward_label"] = labeled_df["reward_label"].astype(int)

        return labeled_df, summary

    def filter_pass_dataframe_open_play(
        self,
        pass_df: pd.DataFrame,
        source_filename: Optional[str] = None,
        drop_filtered: bool = True,
        source_kind: str = "legacy_wide",
        processed_events_path: Optional[str | Path] = None,
        processed_tracking_path: Optional[str | Path] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        game_id = self._extract_game_id(pass_df, source_filename)

        if source_kind == "pff_match_triplets":
            if processed_events_path is None:
                raise ValueError("processed_events_path is required for pff_match_triplets open-play filtering")

            events_path = Path(processed_events_path)
            tracking_path = Path(processed_tracking_path) if processed_tracking_path is not None else None
            game_index: Any = self._get_processed_index(events_path, tracking_path)
        else:
            if game_id is None:
                raise ValueError(
                    "Could not resolve game_id from pass dataframe. "
                    "Expected column 'game_id' or a filename containing a numeric game id."
                )
            game_index = self._get_game_index(game_id)

        filtered_df = pass_df.copy()
        statuses: List[str] = []
        join_strategies: List[str] = []
        pass_event_times: List[Optional[float]] = []
        pass_setpiece_types: List[Optional[str]] = []

        for _, row in filtered_df.iterrows():
            resolved_event, join_strategy = game_index.resolve_pass_event(row)
            join_strategies.append(str(join_strategy))

            if resolved_event is None:
                statuses.append("join_missing")
                pass_event_times.append(None)
                pass_setpiece_types.append(None)
                continue

            event_time = resolved_event.event_time
            setpiece_type = getattr(resolved_event, "set_piece", getattr(resolved_event, "setpiece_type", None))

            if source_kind == "pff_match_triplets":
                is_open_play = setpiece_type == "open_play" or (
                    setpiece_type is None and self.include_open_play_null
                )
            else:
                is_open_play = _is_open_play(setpiece_type, self.include_open_play_null)

            statuses.append("open_play" if is_open_play else "setpiece_filtered")
            pass_event_times.append(event_time)
            pass_setpiece_types.append(setpiece_type)

        filtered_df["open_play_status"] = statuses
        filtered_df["open_play_join_strategy"] = join_strategies
        filtered_df["open_play_pass_event_time"] = pass_event_times
        filtered_df["open_play_pass_setpiece_type"] = pass_setpiece_types

        summary: Dict[str, Any] = {
            "game_id": game_id,
            "source_kind": source_kind,
            "rows_total": int(len(filtered_df)),
            "rows_open_play": int(sum(status == "open_play" for status in statuses)),
            "rows_filtered": int(sum(status != "open_play" for status in statuses)),
            "status_counts": filtered_df["open_play_status"].value_counts(dropna=False).to_dict(),
            "join_strategy_counts": filtered_df["open_play_join_strategy"].value_counts(dropna=False).to_dict(),
        }

        if drop_filtered:
            filtered_df = filtered_df[filtered_df["open_play_status"] == "open_play"].copy()

        return filtered_df, summary

    def label_pass_dataframe(
        self,
        pass_df: pd.DataFrame,
        source_filename: Optional[str] = None,
        drop_unlabeled: bool = True,
        source_kind: str = "legacy_wide",
        processed_events_path: Optional[str | Path] = None,
        processed_tracking_path: Optional[str | Path] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        return self.label_action_dataframe(
            action_df=pass_df,
            action_type="pass",
            source_filename=source_filename,
            drop_unlabeled=drop_unlabeled,
            source_kind=source_kind,
            processed_events_path=processed_events_path,
            processed_tracking_path=processed_tracking_path,
        )
