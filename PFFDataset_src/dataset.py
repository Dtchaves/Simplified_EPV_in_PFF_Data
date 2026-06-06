"""Dataset module for loading and processing PFF data."""

import bz2
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import gandula
import numpy as np
import orjson
import pandas as pd
from tqdm import tqdm

from PFFDataset_src.events import parse_names, parse_events
from PFFDataset_src.tracking import pff_frames_to_dataframe, change_events_side

# Constants
ROOT_DIR = Path(__file__).parent.parent
ALLOWED_COMPETITIONS = {"PL", "UCL", "WC", "BR"}
ALLOWED_SEASONS = {"23", "24", "22-23", "23-24", "24-25"}
CACHE_FILENAMES = ("tracking.parquet", "events.parquet", "players.parquet")
NULL_EQUIVALENT_TOKENS = {"", "none", "null", "nan", "na", "n/a", "nat", "<na>"}



def _add_players_speed_fast(frames_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized replacement for the slower pandas groupby-apply speed pipeline."""
    if frames_df.empty:
        return frames_df

    group_columns = ["period", "team", "shirt"]
    df = frames_df.copy()
    for column in ["elapsed_seconds", "x", "y"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["_row_order"] = np.arange(len(df))
    df = df.sort_values(group_columns + ["frame_id"]).reset_index(drop=True)

    grouped = df.groupby(group_columns, sort=False)
    time_diff = grouped["elapsed_seconds"].diff()
    time_diff = time_diff.where(time_diff > 0)

    x_diff = grouped["x"].diff()
    y_diff = grouped["y"].diff()

    df["vx"] = x_diff / time_diff
    df["vy"] = y_diff / time_diff

    vx_diff = grouped["vx"].diff()
    vy_diff = grouped["vy"].diff()
    df["ax"] = vx_diff / time_diff
    df["ay"] = vy_diff / time_diff
    df["speed"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)

    return df.sort_values("_row_order").drop(columns=["_row_order"]).reset_index(drop=True)


def _add_ball_speed_fast(frames_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized ball speed/acceleration calculation merged back to player rows."""
    if frames_df.empty:
        return frames_df

    ball_df = (
        frames_df.drop_duplicates(subset=["period", "frame_id"], keep="first")
        [["period", "frame_id", "elapsed_seconds", "ball_x", "ball_y", "ball_z"]]
        .copy()
        .sort_values(["period", "elapsed_seconds", "frame_id"])
        .reset_index(drop=True)
    )
    for column in ["elapsed_seconds", "ball_x", "ball_y", "ball_z"]:
        ball_df[column] = pd.to_numeric(ball_df[column], errors="coerce")

    grouped = ball_df.groupby("period", sort=False)
    time_diff = grouped["elapsed_seconds"].diff()
    time_diff = time_diff.where(time_diff > 0)

    ball_df["ball_vx"] = grouped["ball_x"].diff() / time_diff
    ball_df["ball_vy"] = grouped["ball_y"].diff() / time_diff
    ball_df["ball_vz"] = grouped["ball_z"].diff() / time_diff
    ball_df["ball_ax"] = grouped["ball_vx"].diff() / time_diff
    ball_df["ball_ay"] = grouped["ball_vy"].diff() / time_diff
    ball_df["ball_az"] = grouped["ball_vz"].diff() / time_diff
    ball_df["ball_speed"] = np.sqrt(
        ball_df["ball_vx"] ** 2 + ball_df["ball_vy"] ** 2 + ball_df["ball_vz"] ** 2
    )

    return frames_df.merge(
        ball_df[
            [
                "period",
                "frame_id",
                "ball_vx",
                "ball_vy",
                "ball_vz",
                "ball_speed",
                "ball_ax",
                "ball_ay",
                "ball_az",
            ]
        ],
        on=["period", "frame_id"],
        how="left",
    )


def _process_match_job(job: dict[str, Any]) -> dict[str, Any]:
    """Worker entrypoint for parallel cache generation."""
    dataset = PFFDataset(job["competition"], job["season"])
    tracking_df, events_df, players_info = dataset._process_match(
        match_id=job["match_id"],
        add_velocity=job["add_velocity"],
        event_type=job["event_type"],
        save=job["save"],
        filter_tracking_to_event_frames=job["filter_tracking_to_event_frames"],
    )
    return {
        "match_id": job["match_id"],
        "tracking_rows": int(len(tracking_df)),
        "event_rows": int(len(events_df)),
        "player_rows": int(len(players_info)),
        "cache_hit": False,
    }

class PFFDataset:
    """
    Dataset class for loading and processing PFF data.

    Handles loading of tracking and event data, processing, and storing
    intermediate results for efficient reuse.

    Attributes:
        competition: Competition identifier (e.g., 'PL', 'UCL')
        season: Season identifier (e.g., '23-24', '24-25')
        match_ids: List of available match IDs
        players: List of player DataFrames for each loaded match
        tracking: List of tracking DataFrames for each loaded match
        events: List of event DataFrames for each loaded match
    """

    def __init__(self, competition: str, season: str):
        """
        Initialize the PFFDataset.

        Args:
            competition: Competition code (PL, UCL, WC, BR)
            season: Season code (e.g., '23-24', '24-25')

        Raises:
            ValueError: If competition or season is not valid
        """
        self._validate_inputs(competition, season)

        self.competition = competition
        self.season = season

        # Setup directory structure
        self.data_path = ROOT_DIR / "data"
        self.tracking_path = self.data_path / "raw" / competition / season / "tracking"
        self.events_path = self.data_path / "raw" / competition / season / "events"
        self.legacy_save_path = self.data_path / "interim" / competition / season
        self.save_path = self.data_path / "processed" / "pff_match_triplets" / competition / season

        self._create_directories()
        self.match_ids = self._discover_match_ids()

        # Data storage
        self.players: list[pd.DataFrame] = []
        self.tracking: list[pd.DataFrame] = []
        self.events: list[pd.DataFrame] = []
        self.last_load_summary: list[dict[str, Any]] = []

        self.data_version = 'v4_velocity_orientation_fix'

    def _validate_inputs(self, competition: str, season: str) -> None:
        """Validate competition and season inputs."""
        if competition not in ALLOWED_COMPETITIONS:
            raise ValueError(
                f"Competition must be one of {ALLOWED_COMPETITIONS}, got '{competition}'"
            )
        if season not in ALLOWED_SEASONS:
            raise ValueError(
                f"Season must be one of {ALLOWED_SEASONS}, got '{season}'"
            )

    def _create_directories(self) -> None:
        """Create necessary directory structure."""
        for path in [
            self.data_path,
            self.tracking_path,
            self.events_path,
            self.legacy_save_path,
            self.save_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _match_cache_dirs(self, match_id: str) -> list[Path]:
        return [self.save_path / match_id, self.legacy_save_path / match_id]

    @staticmethod
    def _has_cached_triplet(match_dir: Path) -> bool:
        return all((match_dir / filename).exists() for filename in CACHE_FILENAMES)

    def _resolve_cache_dir(self, match_id: str) -> Optional[Path]:
        for match_dir in self._match_cache_dirs(match_id):
            if self._has_cached_triplet(match_dir):
                return match_dir
        return None

    @staticmethod
    def _resolve_worker_count(n_jobs: Optional[int], match_count: int) -> int:
        if match_count <= 1:
            return 1
        if n_jobs is None:
            cpu_count = os.cpu_count() or 1
            return max(1, min(match_count, cpu_count - 1, 8))
        return max(1, min(int(n_jobs), match_count))

    def _discover_match_ids(self) -> list[str]:
        """Discover available match IDs from tracking or events directories."""
        tracking_ids: list[str] = []
        if self.tracking_path.exists():
            tracking_ids = sorted(
                {
                    match.group(1)
                    for path in self.tracking_path.iterdir()
                    if path.is_file()
                    and path.name != "players.json"
                    and (match := re.match(r"^(\d+)", path.stem)) is not None
                },
                key=int,
            )
        if tracking_ids:
            return tracking_ids

        if self.events_path.exists():
            event_ids = sorted(
                {
                    match.group(1)
                    for path in self.events_path.iterdir()
                    if path.is_file()
                    and (match := re.match(r"^(\d+)_events$", path.stem)) is not None
                },
                key=int,
            )
            if event_ids:
                return event_ids

        return []

    @staticmethod
    def _is_null_like(value: Any) -> bool:
        """Return True for real nulls and string tokens such as 'None' or ''."""
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() in NULL_EQUIVALENT_TOKENS
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _clean_scalar(cls, value: Any) -> Any:
        if cls._is_null_like(value):
            return None
        if hasattr(value, "name"):
            return value.name
        return value

    @staticmethod
    def _case_insensitive_get(mapping: Any, *keys: str, default: Any = None) -> Any:
        """Read a key from a dict regardless of source casing."""
        if not isinstance(mapping, dict):
            return default

        lookup = {str(key).lower(): value for key, value in mapping.items()}
        for key in keys:
            lowered = str(key).lower()
            if lowered in lookup:
                return lookup[lowered]
        return default

    @classmethod
    def _coerce_int(cls, value: Any) -> Optional[int]:
        if cls._is_null_like(value):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _coerce_float(cls, value: Any) -> Optional[float]:
        if cls._is_null_like(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _coerce_bool(cls, value: Any) -> Optional[bool]:
        if cls._is_null_like(value):
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "t", "1", "yes", "y"}:
                return True
            if lowered in {"false", "f", "0", "no", "n"}:
                return False
            return None
        return bool(value)

    @staticmethod
    def _event_key(mapping: Any, *keys: str) -> Any:
        return PFFDataset._case_insensitive_get(mapping, *keys)

    def _extract_ball_entry(self, frame: dict[str, Any], smoothed: bool = True) -> Optional[dict[str, Any]]:
        """Extract a ball location from Gradient tracking frames.

        The current data spec stores ball location in a BALL/ball dict with X/Y/Z,
        but older/alternate exports may use balls, ballsSmoothed, ballWithKalman,
        or list-valued variants. This helper accepts all of them.
        """
        preferred_keys = [
            "ballSmoothed",
            "ball_smoothed",
            "ballWithKalman",
            "ball_with_kalman",
            "ballsSmoothed",
            "balls_smoothed",
        ] if smoothed else []
        fallback_keys = ["BALL", "ball", "balls", "ballRaw", "ball_raw"]
        for key in preferred_keys + fallback_keys:
            value = self._case_insensitive_get(frame, key)
            if value is None:
                continue
            if isinstance(value, list):
                return value[0] if value and isinstance(value[0], dict) else None
            if isinstance(value, dict):
                return value
        return None

    def _extract_players_for_side(self, frame: dict[str, Any], team_side: str, smoothed: bool = True) -> list[dict[str, Any]]:
        """Extract home/away player lists from Gradient tracking frames."""
        if team_side == "home":
            preferred = ["homePlayersSmoothed", "home_players_smoothed", "homePlayersWithKalman", "home_players_with_kalman"]
            fallback = ["HOME_PLAYERS", "homePlayers", "home_players"]
        else:
            preferred = ["awayPlayersSmoothed", "away_players_smoothed", "awayPlayersWithKalman", "away_players_with_kalman"]
            fallback = ["AWAY_PLAYERS", "awayPlayers", "away_players"]
        keys = preferred + fallback if smoothed else fallback + preferred
        for key in keys:
            players = self._case_insensitive_get(frame, key)
            if isinstance(players, list):
                return [p for p in players if isinstance(p, dict)]
        return []


    def _load_gradient_tracking(
        self,
        match_id: str,
        smoothed: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load tracking data from the Gradient JSONL/BZ2 format described in the spec."""
        tracking_file = self.tracking_path / f"{match_id}.jsonl.bz2"
        if not tracking_file.exists():
            raise FileNotFoundError(f"Tracking file not found: {tracking_file}")

        metadata_rows: list[dict[str, Any]] = []
        tracking_rows: list[dict[str, Any]] = []

        with bz2.open(tracking_file, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                frame = orjson.loads(line)
                game_event = self._case_insensitive_get(frame, "game_event", "gameEvent", "GAME_EVENTS", default={}) or {}
                possession_event = self._case_insensitive_get(frame, "possession_event", "possessionEvent", "POSSESSION_EVENTS", default={}) or {}

                match_id_value = self._coerce_int(
                    self._case_insensitive_get(frame, "gameRefId", "game_id", "GAME_ID", default=match_id)
                )
                frame_id = self._coerce_int(self._case_insensitive_get(frame, "frameNum", "frame_id", "FRAME_ID"))
                period = self._coerce_int(self._case_insensitive_get(frame, "period", "PERIOD"))
                elapsed_seconds = self._coerce_float(
                    self._case_insensitive_get(frame, "periodElapsedTime", "elapsed_seconds", "ELAPSED_SECONDS")
                )

                metadata_rows.append(
                    {
                        "match_id": match_id_value,
                        "event_id": self._coerce_int(
                            self._case_insensitive_get(
                                frame,
                                "game_event_id",
                                "GAME_EVENT_ID",
                                default=self._case_insensitive_get(game_event, "GAME_EVENT_ID", "game_event_id", "id"),
                            )
                        ),
                        "possession_id": self._coerce_int(
                            self._case_insensitive_get(
                                frame,
                                "possession_event_id",
                                "POSSESSION_EVENT_ID",
                                default=self._case_insensitive_get(possession_event, "POSSESSION_EVENT_ID", "possession_event_id", "id"),
                            )
                        ),
                        "frame_id": frame_id,
                        "period": period,
                        "possession_type": self._case_insensitive_get(
                            possession_event,
                            "POSSESSION_EVENT_TYPE",
                            "possession_event_type",
                            "possessionEventType",
                            default=self._case_insensitive_get(game_event, "GAME_EVENT_TYPE", "game_event_type", "gameEventType"),
                        ),
                        "elapsed_seconds": elapsed_seconds,
                    }
                )

                ball_entry = self._extract_ball_entry(frame, smoothed=smoothed)
                ball_x = self._coerce_float(self._case_insensitive_get(ball_entry, "X", "x"))
                ball_y = self._coerce_float(self._case_insensitive_get(ball_entry, "Y", "y"))
                ball_z = self._coerce_float(self._case_insensitive_get(ball_entry, "Z", "z"))

                # This is the team in possession, not the team of every tracked player.
                # Keep it separate so downstream code can still inspect possession context
                # without corrupting player/team identity.
                possession_team_id = self._coerce_int(self._case_insensitive_get(game_event, "TEAM_ID", "team_id"))

                for team_side in ("home", "away"):
                    players = self._extract_players_for_side(frame, team_side=team_side, smoothed=smoothed)
                    for player in players:
                        shirt_value = self._case_insensitive_get(player, "JERSEY_NUM", "jerseyNum", "jersey_num", "shirt")
                        tracking_rows.append(
                            {
                                "match_id": match_id_value,
                                "frame_id": frame_id,
                                "period": period,
                                "elapsed_seconds": elapsed_seconds,
                                "team": team_side,
                                "possession_team_id": possession_team_id,
                                "shirt": self._clean_scalar(shirt_value),
                                "x": self._coerce_float(self._case_insensitive_get(player, "X", "x")),
                                "y": self._coerce_float(self._case_insensitive_get(player, "Y", "y")),
                                "speed": self._coerce_float(self._case_insensitive_get(player, "SPEED", "speed")),
                                "confidence": self._clean_scalar(self._case_insensitive_get(player, "CONFIDENCE", "confidence")),
                                "visibility": self._clean_scalar(self._case_insensitive_get(player, "VISIBILITY", "visibility")),
                                "ball_x": ball_x,
                                "ball_y": ball_y,
                                "ball_z": ball_z,
                            }
                        )

        metadata_df = pd.DataFrame(metadata_rows)
        tracking_df = pd.DataFrame(tracking_rows)
        return metadata_df, tracking_df

    def _extract_team_id_by_side(self, events_df: pd.DataFrame) -> dict[str, Optional[int]]:
        """Infer actual home/away team IDs from GAME_EVENTS.HOME_TEAM and TEAM_ID."""
        rows: list[tuple[str, int]] = []
        for _, row in events_df.iterrows():
            game_events = row.get("GAME_EVENTS") or {}
            home_team = self._coerce_bool(self._case_insensitive_get(game_events, "HOME_TEAM"))
            team_id = self._coerce_int(self._case_insensitive_get(game_events, "TEAM_ID"))
            if home_team is None or team_id is None:
                continue
            rows.append(("home" if home_team else "away", team_id))

        if not rows:
            return {"home": None, "away": None}

        tmp = pd.DataFrame(rows, columns=["team_side", "team_id"])
        result: dict[str, Optional[int]] = {"home": None, "away": None}
        for side, group in tmp.groupby("team_side"):
            if not group.empty:
                result[side] = int(group["team_id"].mode().iloc[0])
        return result

    def _extract_gradient_player_info(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Build player metadata from Gradient event files."""
        player_rows: list[dict[str, Any]] = []
        team_id_by_side = self._extract_team_id_by_side(events_df)

        for _, row in events_df.iterrows():
            for team_side, key in (("home", "HOME_PLAYERS"), ("away", "AWAY_PLAYERS")):
                players = row.get(key) or []
                for player in players if isinstance(players, list) else []:
                    player_rows.append(
                        {
                            "player_id": self._coerce_int(
                                self._case_insensitive_get(player, "PLAYER_ID", "player_id")
                            ),
                            "shirt_number": self._clean_scalar(
                                self._case_insensitive_get(player, "JERSEY_NUM", "jerseyNum", "shirt_number")
                            ),
                            "team_id": team_id_by_side.get(team_side),
                            "team_side": team_side,
                            "position_name": self._clean_scalar(self._case_insensitive_get(
                                player,
                                "POSITION",
                                "position",
                                "POSITION_GROUP_TYPE",
                                "positionGroupType",
                            )),
                        }
                    )

        players_info = pd.DataFrame(player_rows)
        if not players_info.empty:
            players_info = players_info.dropna(subset=["player_id", "shirt_number", "team_side", "team_id"])
            for column in ["player_id", "team_id"]:
                players_info[column] = pd.to_numeric(players_info[column], errors="coerce").astype("Int64")
            players_info["shirt_number"] = players_info["shirt_number"].astype(str)
            players_info = players_info.drop_duplicates(subset=["player_id", "shirt_number", "team_side", "team_id"]).reset_index(drop=True)
        return players_info

    def _build_gradient_event_row(self, row: pd.Series) -> dict[str, Any]:
        """Flatten one Gradient event record into the canonical event schema."""
        game_events = row.get("GAME_EVENTS") or {}
        possession_events = row.get("POSSESSION_EVENTS") or {}
        stadium_metadata = row.get("STADIUM_METADATA") or {}

        possession_type = self._case_insensitive_get(possession_events, "POSSESSION_EVENT_TYPE")
        home_team = self._coerce_bool(self._case_insensitive_get(game_events, "HOME_TEAM"))

        parsed_row: dict[str, Any] = {
            "match_id": self._coerce_int(row.get("GAME_ID")),
            "event_id": self._coerce_int(row.get("GAME_EVENT_ID")),
            "possession_id": self._coerce_int(row.get("POSSESSION_EVENT_ID")),
            "possession_type": possession_type,
            "player_id": self._coerce_int(
                self._case_insensitive_get(
                    game_events,
                    "PLAYER_ID",
                    default=self._case_insensitive_get(possession_events, "PASSER_PLAYER_ID", "CROSSER_PLAYER_ID", "SHOOTER_PLAYER_ID", "BALL_CARRIER_PLAYER_ID", "CARRIER_PLAYER_ID", "TOUCH_PLAYER_ID"),
                )
            ),
            "team_id": self._coerce_int(self._case_insensitive_get(game_events, "TEAM_ID")),
            "set_piece": self._case_insensitive_get(game_events, "SETPIECE_TYPE"),
            "video_url": row.get("VIDEO_URL"),
            "home_team_start_left": self._coerce_bool(
                self._case_insensitive_get(stadium_metadata, "HOME_TEAM_START_LEFT")
            ),
            "team_side": "home" if home_team else "away" if home_team is not None else None,
            "period": self._coerce_int(self._case_insensitive_get(game_events, "PERIOD")),
            # EVENT_TIME is when the possession event takes place; START_TIME is the
            # start of the broader possession/game event. Use EVENT_TIME first.
            "elapsed_seconds": self._coerce_float(
                self._case_insensitive_get(row, "EVENT_TIME", "START_TIME")
            ),
            "ball_moving": self._coerce_bool(self._case_insensitive_get(possession_events, "BALL_MOVING")),
            "ball_height": self._case_insensitive_get(possession_events, "BALL_HEIGHT_TYPE"),
            "body_part": self._case_insensitive_get(possession_events, "BODY_TYPE"),
            "pressure_type": self._case_insensitive_get(possession_events, "PRESSURE_TYPE"),
            "creates_space": self._coerce_bool(self._case_insensitive_get(possession_events, "CREATES_SPACE")),
            "no_look": self._coerce_bool(self._case_insensitive_get(possession_events, "NO_LOOK")),
            "lines_broken_type": self._case_insensitive_get(possession_events, "LINES_BROKEN_TYPE"),
        }

        if possession_type in {"PA", "CR"}:
            is_cross = possession_type == "CR"
            parsed_row.update(
                {
                    "pass_type": "cross" if is_cross else self._case_insensitive_get(possession_events, "PASS_TYPE"),
                    "pass_accuracy_type": self._case_insensitive_get(possession_events, "ACCURACY_TYPE", "PASS_ACCURACY_TYPE"),
                    "pass_outcome": self._case_insensitive_get(possession_events, "PASS_OUTCOME_TYPE"),
                    "cross_outcome": self._case_insensitive_get(possession_events, "CROSS_OUTCOME_TYPE"),
                    "cross_type": self._case_insensitive_get(possession_events, "CROSS_TYPE"),
                    "cross_zone_type": self._case_insensitive_get(possession_events, "CROSS_ZONE", "CROSS_ZONE_TYPE"),
                    "pass_height": self._case_insensitive_get(possession_events, "RECEIVER_HEIGHT_TYPE", "PASS_HEIGHT_TYPE"),
                    "receiver_player_id": self._coerce_int(self._case_insensitive_get(possession_events, "RECEIVER_PLAYER_ID")),
                    "target_player_id": self._coerce_int(self._case_insensitive_get(possession_events, "TARGET_PLAYER_ID")),
                }
            )

        if possession_type == "SH":
            parsed_row.update(
                {
                    "shot_type": self._case_insensitive_get(possession_events, "SHOT_TYPE"),
                    "shot_outcome": self._case_insensitive_get(possession_events, "SHOT_OUTCOME_TYPE", "SHOT_OUTCOME"),
                    "nature_type": self._case_insensitive_get(possession_events, "SHOT_NATURE_TYPE", "NATURE_TYPE"),
                    "body_movement_type": self._case_insensitive_get(possession_events, "BODY_MOVEMENT_TYPE"),
                    "shot_initial_height_type": self._case_insensitive_get(possession_events, "SHOT_INITIAL_HEIGHT_TYPE"),
                }
            )

        if possession_type == "BC":
            parsed_row.update(
                {
                    "carry_outcome": self._case_insensitive_get(possession_events, "BALL_CARRY_OUTCOME", "CARRY_OUTCOME"),
                    "carry_type": self._case_insensitive_get(possession_events, "CARRY_TYPE"),
                    "carry_intent": self._case_insensitive_get(possession_events, "CARRY_INTENT"),
                    "carry_success": self._coerce_bool(self._case_insensitive_get(possession_events, "CARRY_SUCCESSFUL", "CARRY_SUCCESS")),
                    "dribble_type": self._case_insensitive_get(possession_events, "DRIBBLE_TYPE"),
                    "dribble_outcome": self._case_insensitive_get(possession_events, "DRIBBLE_OUTCOME", "DRIBBLE_OUTCOME_TYPE"),
                }
            )

        return parsed_row

    def _process_gradient_events(
        self,
        events_df: pd.DataFrame,
        event_type: str | list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Process Gradient event rows from the wide JSON format described in the spec."""
        parsed_events = pd.DataFrame([self._build_gradient_event_row(row) for _, row in events_df.iterrows()])
        parsed_events = parse_names(parsed_events)

        if isinstance(event_type, str) and event_type in ["shot", "pass", "carry", "all"]:
            if event_type == "all":
                filtered_events = parsed_events
            else:
                filtered_events = parsed_events[parsed_events["possession_type"] == event_type].reset_index(drop=True)
        elif isinstance(event_type, list):
            filtered_events = parsed_events[parsed_events["possession_type"].isin(event_type)].reset_index(drop=True)
        else:
            raise NotImplementedError(f"Event type '{event_type}' is not yet supported")

        players_info = self._extract_gradient_player_info(events_df)
        return filtered_events, players_info

    def load_data(
        self,
        n_matches: Optional[int] = None,
        match_ids: Optional[list[str]] = None,
        add_velocity: bool = False,
        event_type: str = 'all',
        save: bool = True,
        overwrite: bool = False,
        filter_tracking_to_event_frames: bool = False,
        n_jobs: Optional[int] = None,
        store_in_memory: bool = True,
    ) -> None:
        """
        Load and process data for specified matches.

        Args:
            n_matches: Number of matches to load (from the start of match_ids list)
            match_ids: Specific match IDs to load (overrides n_matches)
            add_velocity: Whether to calculate velocity features
            event_type: Type of events to filter ('shot', 'pass', 'carry', or 'all')
            save: Whether to save processed data
            overwrite: Whether to overwrite existing processed data
            n_jobs: Number of worker processes to use when generating cached triplets.
                Parallel processing is only used when store_in_memory=False.
            store_in_memory: Whether to retain loaded match DataFrames on this instance.
                Disable this when the goal is only to generate processed triplet files.
        """
        matches_to_load = self._determine_matches_to_load(n_matches, match_ids)

        if not save and not store_in_memory:
            raise ValueError("save=False requires store_in_memory=True because no output would be retained")

        self.last_load_summary = []
        if not store_in_memory:
            self.players = []
            self.tracking = []
            self.events = []

        self._load_data(
            matches_to_load,
            add_velocity,
            event_type,
            save,
            overwrite,
            filter_tracking_to_event_frames,
            n_jobs,
            store_in_memory,
        )

    def _load_data(
        self,
        matches_to_load: list[str],
        add_velocity: bool,
        event_type: str,
        save: bool,
        overwrite: bool,
        filter_tracking_to_event_frames: bool,
        n_jobs: Optional[int],
        store_in_memory: bool,
    ) -> None:
        """Load data sequentially (original implementation)."""
        if not store_in_memory:
            cached_matches = []
            pending_matches = []
            for match_id in matches_to_load:
                if not overwrite and self._can_load_cached(match_id):
                    cached_matches.append(match_id)
                else:
                    pending_matches.append(match_id)

            self.last_load_summary.extend(
                {
                    "match_id": match_id,
                    "tracking_rows": None,
                    "event_rows": None,
                    "player_rows": None,
                    "cache_hit": True,
                }
                for match_id in cached_matches
            )

            worker_count = self._resolve_worker_count(n_jobs, len(pending_matches))
            if worker_count > 1:
                jobs = [
                    {
                        "competition": self.competition,
                        "season": self.season,
                        "match_id": match_id,
                        "add_velocity": add_velocity,
                        "event_type": event_type,
                        "save": save,
                        "filter_tracking_to_event_frames": filter_tracking_to_event_frames,
                    }
                    for match_id in pending_matches
                ]
                progress = tqdm(total=len(jobs), desc="Processing matches", unit="match")
                try:
                    with ProcessPoolExecutor(max_workers=worker_count) as executor:
                        future_map = {
                            executor.submit(_process_match_job, job): job["match_id"] for job in jobs
                        }
                        for future in as_completed(future_map):
                            match_id = future_map[future]
                            try:
                                self.last_load_summary.append(future.result())
                            except Exception as e:
                                import traceback
                                print(f"Error processing match {match_id}: {e}")
                                print("Full traceback:")
                                traceback.print_exc()
                            finally:
                                progress.update(1)
                finally:
                    progress.close()
                self.last_load_summary.sort(key=lambda item: str(item.get("match_id")))
                return

            matches_to_load = pending_matches

        for match_id in tqdm(matches_to_load, desc="Loading matches"):
            try:
                cache_hit = False
                if self._can_load_cached(match_id) and not overwrite:
                    cache_hit = True
                    try:
                        tracking_df, events_df, players_info = self._load_cached_data(match_id)
                    except ValueError:
                        tracking_df, events_df, players_info = self._process_match(
                            match_id,
                            add_velocity,
                            event_type,
                            save,
                            filter_tracking_to_event_frames,
                        )
                else:
                    tracking_df, events_df, players_info = self._process_match(
                        match_id,
                        add_velocity,
                        event_type,
                        save,
                        filter_tracking_to_event_frames,
                    )

                self.last_load_summary.append(
                    {
                        "match_id": match_id,
                        "tracking_rows": int(len(tracking_df)),
                        "event_rows": int(len(events_df)),
                        "player_rows": int(len(players_info)),
                        "cache_hit": cache_hit,
                    }
                )

                if store_in_memory:
                    self.players.append(players_info)
                    self.tracking.append(tracking_df)
                    self.events.append(events_df)
            except Exception as e:
                import traceback
                print(f"Error processing match {match_id}: {e}")
                print(f"Full traceback:")
                traceback.print_exc()
                continue

    def _determine_matches_to_load(
        self,
        n_matches: Optional[int],
        match_ids: Optional[list[str]]
    ) -> list[str]:
        """Determine which matches to load based on parameters."""
        if match_ids is not None:
            return match_ids
        if n_matches is not None:
            return self.match_ids[:n_matches]
        return self.match_ids

    def _can_load_cached(self, match_id: str) -> bool:
        """Check if cached data exists for a match."""
        return self._resolve_cache_dir(match_id) is not None

    def _load_cached_data(self, match_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load cached processed data for a match."""
        match_dir = self._resolve_cache_dir(match_id)
        if match_dir is None:
            raise FileNotFoundError(f"No cached parquet triplet found for match {match_id}")
        tracking_df = pd.read_parquet(match_dir / "tracking.parquet")
        events_df = pd.read_parquet(match_dir / "events.parquet")
        players_info = pd.read_parquet(match_dir / "players.parquet")

        if 'data_version' not in events_df.columns or not (events_df['data_version'] == self.data_version).all():
            raise ValueError(f"Cached events version mismatch for match {match_id}")

        return tracking_df, events_df, players_info

    def _process_match(
        self,
        match_id: str,
        add_velocity: bool,
        event_type: str,
        save: bool,
        filter_tracking_to_event_frames: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process a single match from raw data."""
        # Load raw data. Velocity is computed on continuous raw coordinates before
        # event-perspective normalization; orientation transforms then flip vectors
        # together with positions.
        metadata_df, tracking_df = self.load_tracking(match_id, add_velocity=False)
        events_df = self.load_events(match_id)

        # Process events and get player info
        events_df, players_info = self._process_events(events_df, event_type=event_type)

        # Merge metadata with events and standardize IDs before any frame-level joins.
        events_df = self._merge_event_metadata(events_df, metadata_df)
        events_df = self._standardize_id_columns(events_df)

        # Merge true tracked-player team IDs. Do not use possession team_id as player team_id.
        tracking_df = self._merge_player_info(tracking_df, players_info)

        if add_velocity:
            tracking_df = _add_players_speed_fast(tracking_df)
            tracking_df = _add_ball_speed_fast(tracking_df)

        # Normalize coordinates from the acting team's perspective where event intervals are known.
        home_team_start_left = events_df['home_team_start_left'].dropna().iloc[0] if events_df['home_team_start_left'].notna().any() else True
        tracking_df = change_events_side(tracking_df, events_df, bool(home_team_start_left))

        # Keep continuous tracking by default. Event-frame filtering is optional and is
        # applied after velocity computation to avoid differencing sparse event frames.
        if filter_tracking_to_event_frames:
            tracking_df = self._filter_tracking_frames(tracking_df, events_df)

        # Add player/start and ball/start-end locations to events.
        events_df = self._add_event_location(tracking_df, events_df)

        events_df['data_version'] = self.data_version

        tracking_df = self._make_serializable(tracking_df)
        events_df = self._make_serializable(events_df)

        # Get columns for tracking and events data
        tracking_columns = self._get_tracking_columns(tracking_df, add_velocity)
        event_columns = self._get_event_columns(events_df)

        # Filter dataframes to selected columns
        clean_tracking_df = tracking_df[tracking_columns]
        events_df = events_df[event_columns]

        if save:
            self.save_data(match_id, clean_tracking_df, events_df, players_info)

        return clean_tracking_df, events_df, players_info

    def _merge_event_metadata(
        self,
        events_df: pd.DataFrame,
        metadata_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge event data with tracking metadata."""
        metadata_subset = metadata_df[["match_id", "event_id", "possession_id", "frame_id"]].copy()
        metadata_subset["match_id"] = metadata_subset["match_id"].astype(int)
        metadata_subset["possession_id"] = metadata_subset["possession_id"].astype(int)
        metadata_subset["frame_id"] = metadata_subset["frame_id"].astype(int)

        frame_bounds = (
            metadata_subset
            .groupby(["match_id", "possession_id"], as_index=False)
            .agg(start_frame_id=("frame_id", "min"), end_frame_id=("frame_id", "max"))
        )

        merged = events_df.merge(
            frame_bounds,
            on=['match_id', 'possession_id'],
            how='inner',
        )
        # Keep frame_id as start-frame alias for backward compatibility.
        merged["frame_id"] = merged["start_frame_id"]
        return merged

    def _filter_tracking_frames(
        self,
        tracking_df: pd.DataFrame,
        events_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter tracking data to only include frames with events."""
        return tracking_df[tracking_df['frame_id'].isin(events_df['frame_id'])]

    def _standardize_id_columns(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Standardize ID columns to integer type."""

        EVENT_ID_COLUMNS = [
            "match_id",
            "event_id",
            "possession_id",
            "frame_id",
            "start_frame_id",
            "end_frame_id",
        ]

        events_df = events_df.dropna(subset=EVENT_ID_COLUMNS).reset_index(drop=True)

        for col in EVENT_ID_COLUMNS:
            events_df[col] = events_df[col].astype(int)
        return events_df

    def _merge_player_info(
        self,
        tracking_df: pd.DataFrame,
        players_info: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge true player metadata into tracking rows.

        The raw tracking side column is named ``team`` (home/away). The output keeps
        ``team_side`` as the home/away side and ``team_id`` as the actual team ID.
        """
        tracking_df = tracking_df.copy()
        if tracking_df.empty or players_info.empty:
            return tracking_df

        # Remove any stale/corrupt team_id created by older loaders.
        tracking_df = tracking_df.drop(columns=[c for c in ["team_id", "team_side", "shirt_number", "position_name", "player_id"] if c in tracking_df.columns], errors="ignore")

        player_columns = [
            column
            for column in ["player_id", "team_id", "team_side", "shirt_number", "position_name"]
            if column in players_info.columns
        ]

        tracking_df['shirt'] = tracking_df['shirt'].map(self._clean_scalar).astype(str)
        players_info = players_info.copy()
        players_info['shirt_number'] = players_info['shirt_number'].map(self._clean_scalar).astype(str)

        merged = tracking_df.merge(
            players_info[player_columns],
            left_on=['shirt', 'team'],
            right_on=['shirt_number', 'team_side'],
            how='left'
        )
        if 'team_id' in merged.columns:
            merged['team_id'] = pd.to_numeric(merged['team_id'], errors='coerce').astype('Int64')
        if 'player_id' in merged.columns:
            merged['player_id'] = pd.to_numeric(merged['player_id'], errors='coerce').astype('Int64')
        return merged

    def _add_event_location(
        self,
        tracking_df: pd.DataFrame,
        events_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add event start/end locations from normalized tracking data.

        ``x``/``y`` are the acting player's start-frame location. For passes/crosses,
        ``ball_x_end``/``ball_y_end`` are kept from ball tracking when available, but
        fall back to the intended target/receiver player's location rather than to the
        acting player's start location. This prevents the pass-selection label from
        being identical to the distance-to-ball minimum.
        """
        tracking_df = tracking_df.copy()
        for col in ["match_id", "frame_id", "player_id", "x", "y", "ball_x", "ball_y", "ball_z"]:
            if col in tracking_df.columns:
                tracking_df[col] = pd.to_numeric(tracking_df[col], errors="coerce")

        ball_frame_df = (
            tracking_df[['match_id', 'frame_id', 'period', 'elapsed_seconds', 'ball_x', 'ball_y', 'ball_z']]
            .drop_duplicates(subset=['match_id', 'frame_id'])
            .reset_index(drop=True)
        )

        player_xy_df = tracking_df[['match_id', 'frame_id', 'player_id', 'x', 'y']].dropna(
            subset=['match_id', 'frame_id', 'player_id', 'x', 'y']
        ).copy()

        # Acting/player-in-possession start location.
        player_start_df = player_xy_df.rename(columns={
            'frame_id': 'start_frame_id',
            'x': 'x',
            'y': 'y',
        })
        events_df = events_df.merge(
            player_start_df,
            on=['match_id', 'start_frame_id', 'player_id'],
            how='left',
        )

        ball_start_df = ball_frame_df.rename(columns={
            'frame_id': 'start_frame_id',
            'period': 'period_start',
            'elapsed_seconds': 'elapsed_seconds_start',
            'ball_x': 'ball_x_start',
            'ball_y': 'ball_y_start',
            'ball_z': 'ball_z_start',
        })
        events_df = events_df.merge(ball_start_df, on=['match_id', 'start_frame_id'], how='left')

        ball_end_df = ball_frame_df.rename(columns={
            'frame_id': 'end_frame_id',
            'period': 'period_end',
            'elapsed_seconds': 'elapsed_seconds_end',
            'ball_x': 'ball_x_end',
            'ball_y': 'ball_y_end',
            'ball_z': 'ball_z_end',
        })
        events_df = events_df.merge(ball_end_df, on=['match_id', 'end_frame_id'], how='left')

        # Intended pass/cross destination from target/receiver player location.
        if {'target_player_id', 'receiver_player_id'}.intersection(events_df.columns):
            destination_player_id = pd.to_numeric(
                events_df.get('target_player_id'), errors='coerce'
            ) if 'target_player_id' in events_df.columns else pd.Series(index=events_df.index, dtype='float64')
            if 'receiver_player_id' in events_df.columns:
                destination_player_id = destination_player_id.fillna(pd.to_numeric(events_df['receiver_player_id'], errors='coerce'))
            events_df['_destination_player_id'] = destination_player_id.astype('Int64')

            dest_end_df = player_xy_df.rename(columns={
                'frame_id': 'end_frame_id',
                'player_id': '_destination_player_id',
                'x': 'target_x_end',
                'y': 'target_y_end',
            })
            events_df = events_df.merge(dest_end_df, on=['match_id', 'end_frame_id', '_destination_player_id'], how='left')

            dest_start_df = player_xy_df.rename(columns={
                'frame_id': 'start_frame_id',
                'player_id': '_destination_player_id',
                'x': 'target_x_start',
                'y': 'target_y_start',
            })
            events_df = events_df.merge(dest_start_df, on=['match_id', 'start_frame_id', '_destination_player_id'], how='left')

            events_df['target_x'] = pd.to_numeric(events_df.get('target_x_end'), errors='coerce').fillna(
                pd.to_numeric(events_df.get('target_x_start'), errors='coerce')
            )
            events_df['target_y'] = pd.to_numeric(events_df.get('target_y_end'), errors='coerce').fillna(
                pd.to_numeric(events_df.get('target_y_start'), errors='coerce')
            )

            is_pass = events_df.get('possession_type').astype(str).str.lower().eq('pass') if 'possession_type' in events_df.columns else pd.Series(False, index=events_df.index)
            start_end_dist = (
                (pd.to_numeric(events_df.get('ball_x_end'), errors='coerce') - pd.to_numeric(events_df.get('ball_x_start'), errors='coerce')) ** 2
                + (pd.to_numeric(events_df.get('ball_y_end'), errors='coerce') - pd.to_numeric(events_df.get('ball_y_start'), errors='coerce')) ** 2
            ) ** 0.5
            replace_end = is_pass & events_df['target_x'].notna() & events_df['target_y'].notna() & (
                pd.to_numeric(events_df.get('ball_x_end'), errors='coerce').isna()
                | pd.to_numeric(events_df.get('ball_y_end'), errors='coerce').isna()
                | start_end_dist.isna()
                | (start_end_dist <= 0.5)
            )
            events_df.loc[replace_end, 'ball_x_end'] = events_df.loc[replace_end, 'target_x']
            events_df.loc[replace_end, 'ball_y_end'] = events_df.loc[replace_end, 'target_y']

        # Preserve previous fields as aliases to start-frame values.
        events_df['period'] = events_df.get('period_start')
        events_df['elapsed_seconds'] = events_df.get('elapsed_seconds_start')
        events_df['ball_x'] = events_df.get('ball_x_start')
        events_df['ball_y'] = events_df.get('ball_y_start')
        events_df['ball_z'] = events_df.get('ball_z_start')

        return events_df

    def _get_tracking_columns(
        self,
        tracking_df: pd.DataFrame,
        add_velocity: bool
    ) -> list[str]:
        """
        Get the columns to include in tracking data based on available columns and velocity settings.

        Args:
            tracking_df: Tracking DataFrame to check for available columns
            add_velocity: Whether to include velocity columns

        Returns:
            List of column names to include in tracking data
        """
        base_columns = [
            'match_id', 'frame_id', 'period', 'elapsed_seconds',
            'team_id', 'player_id', 'x', 'y', 'ball_x', 'ball_y', 'ball_z',
            'shirt', 'team_side', 'team_phase'
        ]

        velocity_cols = [
            'vx', 'vy', 'ax', 'ay', 'speed',
            'ball_vx', 'ball_vy', 'ball_vz', 'ball_speed',
            'ball_ax', 'ball_ay', 'ball_az'
        ] if add_velocity else []

        # Combine base and velocity columns, filtering for available columns
        all_columns = base_columns + velocity_cols
        return [col for col in all_columns if col in tracking_df.columns]

    def _get_event_columns(self, events_df: pd.DataFrame) -> list[str]:
        """
        Get the columns to include in event data based on event type and available columns.

        Args:
            events_df: Event DataFrame to check for available columns

        Returns:
            List of column names to include in event data
        """
        # Define base columns that are always included
        base_columns = [
            'match_id', 'event_id', 'possession_id', 'frame_id', 'start_frame_id', 'end_frame_id',
            'period', 'elapsed_seconds',
            'team_id', 'player_id', 'possession_type', 'x', 'y',
            'ball_x', 'ball_y', 'ball_z', 'ball_height', 'set_piece',
            'ball_x_start', 'ball_y_start', 'ball_z_start',
            'ball_x_end', 'ball_y_end', 'ball_z_end',
            'video_url', 'team_side', 'home_team_start_left', 'target_x', 'target_y', 'data_version'
        ]

        # Define event-specific column groups
        event_specific_columns = {
            'shot': ['shot_outcome', 'shot_type', 'body_part', 'body_movement_type', 'ball_moving', 'nature_type'],
            'pass': ['pass_type', 'pass_accuracy_type', 'receiver_player_id', 'target_player_id', 'no_look', 'creates_space', 'pressure_type', 'lines_broken_type', 'cross_type', 'cross_zone_type', 'cross_outcome', 'pass_outcome'],
            'carry': ['carry_outcome', 'carry_type', 'carry_intent', 'carry_success', 'dribble_type', 'dribble_outcome']
        }

        # Collect all available event-specific columns
        available_event_columns = []
        for _, columns in event_specific_columns.items():
            available_event_columns.extend([col for col in columns if col in events_df.columns])

        # Combine all columns and filter for available ones
        all_columns = base_columns + available_event_columns
        return [col for col in all_columns if col in events_df.columns]

    def save_data(
        self,
        match_id: str,
        tracking_df: pd.DataFrame,
        events_df: pd.DataFrame,
        players_df: pd.DataFrame
    ) -> None:
        """
        Save processed data to parquet files.

        Args:
            match_id: Match identifier
            tracking_df: Tracking data
            events_df: Event data
            players_df: Player information
        """
        match_dir = self.save_path / match_id
        match_dir.mkdir(parents=True, exist_ok=True)

        players_df.to_parquet(match_dir / "players.parquet", engine='pyarrow')
        events_df.to_parquet(match_dir / "events.parquet", engine='pyarrow')
        tracking_df.to_parquet(match_dir / "tracking.parquet", engine='pyarrow')

    def load_events(self, match_id: str) -> pd.DataFrame:
        """
        Load raw event data for a match.

        Args:
            match_id: Match identifier

        Returns:
            DataFrame with raw event data
        """
        events_file = self.events_path / f"{match_id}_events.json"
        try:
            with events_file.open("rb") as handle:
                records = orjson.loads(handle.read())
            return pd.DataFrame(records)
        except Exception as e:
            print(f"Error loading events for match {match_id}: {e}")
            print(f"Events file: {events_file}")
            raise

    def load_tracking(
        self,
        match_id: str,
        add_velocity: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and process tracking data for a match.

        Args:
            match_id: Match identifier
            add_velocity: Whether to calculate velocity features

        Returns:
            Tuple of (metadata_df, tracking_df)
        """
        gradient_tracking_file = self.tracking_path / f"{match_id}.jsonl.bz2"
        if gradient_tracking_file.exists():
            metadata_df, tracking_df = self._load_gradient_tracking(match_id, smoothed=True)
        else:
            metadata_df, tracking_df = pff_frames_to_dataframe(
                gandula.get_frames(str(self.tracking_path), match_id),
                smoothed=True
            )

        metadata_df, tracking_df = self._process_tracking(metadata_df, tracking_df)

        if add_velocity:
            tracking_df = _add_players_speed_fast(tracking_df)
            tracking_df = _add_ball_speed_fast(tracking_df)

        return metadata_df, tracking_df

    def _process_events(
        self,
        events_df: pd.DataFrame,
        event_type: str | list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process event data and extract player information.

        Args:
            events_df: Raw event data
            event_type: Type of events to filter

        Returns:
            Tuple of (filtered_events, players_info)
        """
        if {"GAME_EVENTS", "POSSESSION_EVENTS"}.issubset(events_df.columns):
            return self._process_gradient_events(events_df, event_type)

        try:
            game_data = events_df['game'].iloc[0]
            match_events = pd.DataFrame(game_data['gameEvents'])
            home_team_id = int(game_data['homeTeam']['id']) if game_data['homeTeam'] else None
            home_team_start_left = game_data['homeTeamStartLeft'] if game_data['homeTeamStartLeft'] else None
            match_id = int(game_data['id']) if game_data['id'] else None
        except Exception as e:
            print(f"Error processing game data: {e}")
            print(f"Events_df shape: {events_df.shape}")
            print(f"Events_df columns: {events_df.columns.tolist()}")
            if 'game' in events_df.columns:
                print(f"Game column type: {type(events_df['game'].iloc[0])}")
            raise

        # Extract player information
        players_info = self._extract_player_info(game_data['rosters'], home_team_id)

        match_events['match_id'] = match_id
        match_events['team_id'] = match_events['team'].apply(
            lambda x: int(x.get('id')) if x else None
        )
        match_events['player_id'] = match_events['player'].apply(
            lambda x: int(x.get('id')) if x else None
        )

        # Process possession events
        possession_events = (
            match_events[['match_id','player_id','team_id', 'possessionEvents', 'setpieceType', 'videoUrl']]
            .explode('possessionEvents')
            .dropna(subset=['match_id','player_id','team_id', 'possessionEvents'])
            .reset_index(drop=True)
        )


        # Filter by event type
        filtered_events = self._filter_by_event_type(
            possession_events, event_type, home_team_id
        )
        filtered_events['home_team_start_left'] = home_team_start_left

        return filtered_events, players_info

    def _extract_player_info(
        self,
        rosters: list,
        home_team_id: int
    ) -> pd.DataFrame:
        """Extract and format player information from rosters."""

        try:
            columns = [
                'player.id', 'player.nickname', 'positionGroupType', 'shirtNumber',
                'team.id', 'team.name', 'player.preferredFoot', 'player.height', 'player.weight'
            ]
            players_info = pd.json_normalize(rosters)[columns]

            # Rename columns for consistency
            players_info.rename(columns={
                'player.id': 'player_id',
                'player.nickname': 'player_name',
                'team.id': 'team_id',
                'team.name': 'team_name',
                'player.preferredFoot': 'preferred_foot',
                'positionGroupType': 'position_name',
                'shirtNumber': 'shirt_number',
                'player.height': 'height',
                'player.weight': 'weight'
            }, inplace=True)

            players_info['team_id'] = players_info['team_id'].astype(int)
            players_info['shirt_number'] = players_info['shirt_number'].astype(int)
            players_info['player_id'] = players_info['player_id'].astype(int)

            # Add team side indicator
            players_info['team_side'] = players_info['team_id'].apply(
                lambda x: 'home' if int(x) == home_team_id else 'away'
            )

            return players_info
        except Exception as e:
            print(f"Error extracting player info: {e}")
            print(f"Rosters: {rosters}")
            print(f"Home team ID: {home_team_id}")
            raise

    def _filter_by_event_type(
        self,
        possession_events: pd.DataFrame,
        event_type: str | list[str],
        home_team_id: int
    ) -> pd.DataFrame:
        """Filter events by type and add team side information."""

        try:
            parsed_events = pd.json_normalize(
                possession_events.apply(parse_events, axis=1).dropna()
            )

            parsed_events = parse_names(parsed_events)

            if isinstance(event_type, str) and event_type in ['shot', 'pass', 'carry', 'all']:
                if event_type == 'all':
                    filtered_events = parsed_events
                else:
                    filtered_events = parsed_events[parsed_events['possession_type'] == event_type].reset_index(drop=True)
            elif isinstance(event_type, list):
                filtered_events = parsed_events[parsed_events['possession_type'].isin(event_type)].reset_index(drop=True)
            else:
                # Placeholder for other event types
                raise NotImplementedError(f"Event type '{event_type}' is not yet supported")

            filtered_events['team_side'] = filtered_events['team_id'].apply(
                lambda x: 'home' if x is not None and int(x) == home_team_id else 'away'
            )
            return filtered_events
        except Exception as e:
            print(f"Error filtering events: {e}")
            print(f"Possession events: {possession_events.shape}")
            print(f"Event type: {event_type}")
            print(f"Home team ID: {home_team_id}")
            raise

    def _process_tracking(
        self,
        metadata_df: pd.DataFrame,
        tracking_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process and clean tracking data.

        Args:
            metadata_df: Metadata DataFrame
            tracking_df: Players tracking DataFrame

        Returns:
            Tuple of processed (metadata_df, tracking_df)
        """
        # Process metadata
        metadata_df = self._make_serializable(metadata_df)
        metadata_df['possession_type'] = metadata_df['possession_type'].map(self._clean_scalar).astype(str)
        metadata_df['match_id'] = pd.to_numeric(metadata_df['match_id'], errors='coerce').ffill()
        metadata_df['possession_id'] = pd.to_numeric(metadata_df['possession_id'], errors='coerce')
        metadata_df['frame_id'] = pd.to_numeric(metadata_df['frame_id'], errors='coerce')
        metadata_df = metadata_df.dropna(subset=['possession_id', 'frame_id', 'match_id'])
        metadata_df = metadata_df.drop_duplicates(subset=['frame_id', 'match_id']).reset_index(drop=True)
        metadata_df['possession_id'] = metadata_df['possession_id'].astype(int)
        metadata_df['frame_id'] = metadata_df['frame_id'].astype(int)
        metadata_df['match_id'] = metadata_df['match_id'].astype(int)

        # Process player tracking
        tracking_df = self._make_serializable(tracking_df)
        tracking_df['match_id'] = pd.to_numeric(tracking_df['match_id'], errors='coerce').ffill()
        tracking_df['frame_id'] = pd.to_numeric(tracking_df['frame_id'], errors='coerce')
        tracking_df = tracking_df.dropna(subset=['frame_id', 'match_id'], how='any')
        tracking_df = tracking_df.drop_duplicates(subset=['frame_id', 'match_id', 'team', 'shirt']).reset_index(drop=True)
        tracking_df['frame_id'] = tracking_df['frame_id'].astype(int)
        tracking_df['match_id'] = tracking_df['match_id'].astype(int)
        tracking_df['shirt'] = pd.to_numeric(tracking_df['shirt'], errors='coerce').astype('Int64')

        return metadata_df, tracking_df

    def _make_serializable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert object columns safely for parquet storage without stringifying nulls."""
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].map(self._clean_scalar)
        return df

    def __len__(self) -> int:
        """Return the number of available matches."""
        return len(self.match_ids)

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        return (
            f"PFFDataset(competition='{self.competition}', "
            f"season='{self.season}', matches={len(self)}, "
            f"version='{self.data_version}')"
        )
