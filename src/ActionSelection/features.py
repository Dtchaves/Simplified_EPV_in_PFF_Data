from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
try:
    from gandula.utils.pitch_control.config import PitchControlConfig as GandulaPitchControlConfig
except Exception:  # pragma: no cover - runtime optional dependency
    GandulaPitchControlConfig = None

ACTION_ROOT = Path(__file__).resolve().parent
SRC_ROOT = ACTION_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from BallDrive.features import BallDriveFeatureBuilder
from Shot.baseline_xg import BaselineXGArtifacts


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_LENGTH = PITCH_LENGTH / 2.0
HALF_WIDTH = PITCH_WIDTH / 2.0
GOAL_X = HALF_LENGTH
GOAL_Y = 0.0


@dataclass
class _TrackingFrameContext:
    elapsed_seconds: Optional[float]
    ball_xy_bl: Optional[Tuple[float, float]]
    player_positions_bl: Dict[str, Dict[str, Tuple[float, float]]]
    team_side_by_player: Dict[int, str]
    team_side_by_team: Dict[int, str]
    xy_by_side: Dict[str, np.ndarray]
    pressure_lines_by_side: Dict[str, np.ndarray]


@dataclass
class ActionSelectionFeatureConfig:
    orientation_mode: str = "attack_right"
    pressure_radius_m: float = 8.0
    require_baseline_xg: bool = True


class ActionSelectionFeatureBuilder:
    def __init__(
        self,
        config: Optional[ActionSelectionFeatureConfig] = None,
        baseline_xg_artifacts: Optional[BaselineXGArtifacts] = None,
    ):
        self.config = config or ActionSelectionFeatureConfig()
        self.baseline_xg_artifacts = baseline_xg_artifacts
        self.feature_columns = [
            "ball_x",
            "angle_to_goal",
            "distance_to_goal",
            "pitch_control_attacking_team_at_ball",
            "defending_team_influence_at_ball",
            "closest_attacking_pressure_line_index",
            "closest_defending_pressure_line_index",
            "baseline_xg",
        ]
        self._orientation_helper = BallDriveFeatureBuilder()
        pitch_defaults = GandulaPitchControlConfig() if GandulaPitchControlConfig is not None else None
        self._pitch_grid_x = int(getattr(pitch_defaults, "grid_x", 50))
        self._pitch_grid_y = int(getattr(pitch_defaults, "grid_y", 30))
        self._pitch_reaction_time = float(getattr(pitch_defaults, "reaction_time", 0.7))
        self._pitch_max_player_speed = float(getattr(pitch_defaults, "max_player_speed", 5.0))
        self._pitch_average_ball_speed = float(getattr(pitch_defaults, "average_ball_speed", 15.0))
        self._pitch_sigma = float(getattr(pitch_defaults, "sigma", 4.029665766610985))
        self._pitch_lambda = float(getattr(pitch_defaults, "lamb", 4.3))
        self._pitch_fps_hint = float(getattr(pitch_defaults, "fps_hint", 10.0))
        self._pitch_x_grid = np.linspace(0.0, 105.0, self._pitch_grid_x, dtype=float)
        self._pitch_y_grid = np.linspace(0.0, 68.0, self._pitch_grid_y, dtype=float)
        gauss_points = int(getattr(pitch_defaults, "gauss_points", 50))
        self._pitch_ti, self._pitch_wi = np.polynomial.legendre.leggauss(gauss_points)

    @staticmethod
    def _normalize_token(value: Any) -> Optional[str]:
        if value is None:
            return None
        token = str(value).strip().lower()
        if not token or token in {"nan", "none", "null", "na", "n/a", "nat"}:
            return None
        return token

    @staticmethod
    def _normalize_set_piece(value: Any) -> Optional[str]:
        token = ActionSelectionFeatureBuilder._normalize_token(value)
        if token is None:
            return None
        token = token.replace("-", "_").replace(" ", "_")
        mapping = {
            "open_play": "open_play",
            "openplay": "open_play",
            "corner": "corner",
            "corner_kick": "corner",
            "free_kick": "free_kick",
            "freekick": "free_kick",
            "throw_in": "throw_in",
            "throwin": "throw_in",
            "penalty": "penalty",
            "goal_kick": "goal_kick",
            "goalkick": "goal_kick",
            "kick_off": "kick_off",
            "kickoff": "kick_off",
            "o": "open_play",
            "c": "corner",
            "f": "free_kick",
            "t": "throw_in",
            "p": "penalty",
            "g": "goal_kick",
            "k": "kick_off",
        }
        return mapping.get(token, token)

    @staticmethod
    def _normalize_tracking_df(tracking_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if tracking_df is None or tracking_df.empty:
            return None

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

        if {"match_id", "frame_id"}.issubset(frame.columns):
            frame = frame.sort_values(["match_id", "frame_id"]).set_index(["match_id", "frame_id"], drop=False)

        return frame

    def _normalize_row(self, row: pd.Series) -> pd.Series:
        normalized = row.copy()
        if "game_id" not in normalized.index and "match_id" in normalized.index:
            normalized["game_id"] = normalized.get("match_id")
        if "match_id" not in normalized.index and "game_id" in normalized.index:
            normalized["match_id"] = normalized.get("game_id")
        if "start_frame_id" not in normalized.index and "frame_id" in normalized.index:
            normalized["start_frame_id"] = normalized.get("frame_id")
        if "ball_x_start" not in normalized.index and "ball_x" in normalized.index:
            normalized["ball_x_start"] = normalized.get("ball_x")
        if "ball_y_start" not in normalized.index and "ball_y" in normalized.index:
            normalized["ball_y_start"] = normalized.get("ball_y")
        if "set_piece" not in normalized.index and "set_piece_normalized" in normalized.index:
            normalized["set_piece"] = normalized.get("set_piece_normalized")
        return normalized

    @staticmethod
    def _resolve_ball_xy(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
        for x_key, y_key in (
            ("ball_x", "ball_y"),
            ("ball_x_start", "ball_y_start"),
            ("shot_x", "shot_y"),
        ):
            if x_key in row.index and y_key in row.index:
                x_val = pd.to_numeric(row.get(x_key), errors="coerce")
                y_val = pd.to_numeric(row.get(y_key), errors="coerce")
                if pd.notna(x_val) and pd.notna(y_val):
                    return float(x_val), float(y_val)
        return None, None

    def _resolve_team_side(self, row: pd.Series, frame_players: Optional[pd.DataFrame]) -> str:
        token = self._normalize_token(row.get("team_side"))
        if token in {"home", "away"}:
            return token

        if frame_players is not None and not frame_players.empty:
            player_id = pd.to_numeric(row.get("player_id"), errors="coerce")
            if pd.notna(player_id) and "player_id" in frame_players.columns and "team_side" in frame_players.columns:
                player_rows = frame_players[frame_players["player_id"] == int(player_id)]
                if not player_rows.empty:
                    candidate = self._normalize_token(player_rows.iloc[0].get("team_side"))
                    if candidate in {"home", "away"}:
                        return candidate

            team_id = pd.to_numeric(row.get("team_id"), errors="coerce")
            if pd.notna(team_id) and "team_id" in frame_players.columns and "team_side" in frame_players.columns:
                team_rows = frame_players[frame_players["team_id"] == int(team_id)]
                if not team_rows.empty:
                    candidate = self._normalize_token(team_rows.iloc[0].get("team_side"))
                    if candidate in {"home", "away"}:
                        return candidate

        return "home"

    def _tracking_slice(self, row: pd.Series, tracking_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if tracking_df is None or tracking_df.empty:
            return None

        match_id = pd.to_numeric(row.get("game_id", row.get("match_id")), errors="coerce")
        frame_id = pd.to_numeric(row.get("frame_id", row.get("start_frame_id")), errors="coerce")
        if pd.isna(match_id) or pd.isna(frame_id):
            return None

        if isinstance(tracking_df.index, pd.MultiIndex) and list(tracking_df.index.names[:2]) == ["match_id", "frame_id"]:
            key = (int(match_id), int(frame_id))
            try:
                frame_slice = tracking_df.loc[key]
            except KeyError:
                return None
            if isinstance(frame_slice, pd.Series):
                return frame_slice.to_frame().T.copy()
            return frame_slice.copy()

        match_col = "match_id" if "match_id" in tracking_df.columns else "game_id"
        frame_col = "frame_id"
        if match_col not in tracking_df.columns or frame_col not in tracking_df.columns:
            return None

        return tracking_df[(tracking_df[match_col] == int(match_id)) & (tracking_df[frame_col] == int(frame_id))].copy()

    def _baseline_xg_row(self, row: pd.Series) -> Dict[str, float]:
        ball_x, ball_y = self._resolve_ball_xy(row)
        if ball_x is None or ball_y is None:
            ball_x, ball_y = 0.0, 0.0

        team_side = self._normalize_token(row.get("team_side")) or "home"
        ball_x_oriented, ball_y_oriented = self._orientation_helper._orient_attack_right(
            ball_x,
            ball_y,
            team_side,
        )
        dx = GOAL_X - ball_x_oriented
        dy = GOAL_Y - ball_y_oriented

        set_piece = self._normalize_set_piece(row.get("set_piece"))
        is_open_play = 1 if set_piece in (None, "open_play") else 0
        is_set_piece = 1 if set_piece not in (None, "open_play") else 0

        return {
            "shot_x": float(ball_x_oriented),
            "shot_y": float(ball_y_oriented),
            "distance_to_goal": float(math.hypot(dx, dy)),
            "angle_to_goal": float(abs(math.atan2(dy, dx))),
            "is_open_play": float(is_open_play),
            "is_set_piece": float(is_set_piece),
            "is_free_kick": 1.0 if set_piece == "free_kick" else 0.0,
            "is_corner": 1.0 if set_piece == "corner" else 0.0,
            "is_penalty": 1.0 if set_piece == "penalty" else 0.0,
            "is_header": float(self._header_flag(row)),
        }

    @staticmethod
    def _header_flag(row: pd.Series) -> int:
        for key in ("body_part", "shot_body_part", "shot_body", "header", "is_header"):
            if key not in row.index:
                continue
            value = row.get(key)
            if isinstance(value, (bool, np.bool_)):
                return int(bool(value))
            token = ActionSelectionFeatureBuilder._normalize_token(value)
            if token in {"head", "header", "he"}:
                return 1
        return 0

    def _compute_baseline_xg(
        self,
        row: pd.Series,
        baseline_xg_artifacts: Optional[BaselineXGArtifacts],
    ) -> float:
        if baseline_xg_artifacts is None:
            value = row.get("baseline_xg")
            if pd.notna(value):
                return float(value)
            if self.config.require_baseline_xg:
                raise ValueError("Missing baseline_xg feature and no baseline-xg artifacts were provided.")
            return 0.0

        baseline_row = self._baseline_xg_row(row)
        feature_df = pd.DataFrame([baseline_row])
        feature_df = feature_df[list(baseline_xg_artifacts.feature_columns)]
        probability = baseline_xg_artifacts.predict_proba(feature_df)
        return float(probability[0, 1])

    def _pressure_line_indices(
        self,
        frame_players: Optional[pd.DataFrame],
        ball_x_oriented: float,
        team_side: str,
    ) -> Tuple[float, float]:
        if frame_players is None or frame_players.empty:
            return float("nan"), float("nan")

        if "team_side" not in frame_players.columns or "x" not in frame_players.columns:
            return float("nan"), float("nan")

        defenders = frame_players.copy()
        defenders["team_side"] = defenders["team_side"].astype(str).str.lower().str.strip()
        attackers = defenders[defenders["team_side"] == team_side]
        defenders = defenders[defenders["team_side"] != team_side]

        def _nearest_index(group: pd.DataFrame) -> float:
            if group.empty:
                return float("nan")
            oriented_x = []
            for _, player in group.iterrows():
                x_oriented, _ = self._orientation_helper._orient_attack_right(
                    float(player["x"]),
                    0.0,
                    player.get("team_side"),
                )
                oriented_x.append(x_oriented)
            lines = self._orientation_helper._cluster_complete_linkage_1d(np.asarray(oriented_x, dtype=float), k=3)
            if lines.size == 0:
                return float("nan")
            return float(int(np.argmin(np.abs(lines - float(ball_x_oriented)))))

        return _nearest_index(attackers), _nearest_index(defenders)

    def _build_frame_context(
        self,
        frame: pd.DataFrame,
    ) -> Optional[_TrackingFrameContext]:
        clean = frame.dropna(subset=["x", "y", "team_side"]).copy()
        if clean.empty:
            return None

        clean["team_side"] = clean["team_side"].astype(str).str.lower().str.strip()
        home = clean[clean["team_side"] == "home"].copy()
        away = clean[clean["team_side"] == "away"].copy()

        player_positions_bl = {"home": {}, "away": {}}
        team_side_by_player: Dict[int, str] = {}
        team_side_by_team: Dict[int, str] = {}
        for side_name, side_frame in (("home", home), ("away", away)):
            for player in side_frame.itertuples(index=False):
                player_id = pd.to_numeric(getattr(player, "player_id", None), errors="coerce")
                team_id = pd.to_numeric(getattr(player, "team_id", None), errors="coerce")
                x = pd.to_numeric(getattr(player, "x", None), errors="coerce")
                y = pd.to_numeric(getattr(player, "y", None), errors="coerce")
                if pd.notna(player_id):
                    player_positions_bl[side_name][str(int(player_id))] = self._orientation_helper._to_gandula_bottom_left(
                        float(x),
                        float(y),
                    )
                    team_side_by_player[int(player_id)] = side_name
                if pd.notna(team_id) and int(team_id) not in team_side_by_team:
                    team_side_by_team[int(team_id)] = side_name

        home_xy = home[["x", "y"]].to_numpy(dtype=float) if not home.empty else np.empty((0, 2), dtype=float)
        away_xy = away[["x", "y"]].to_numpy(dtype=float) if not away.empty else np.empty((0, 2), dtype=float)
        home_lines = self._orientation_helper._cluster_complete_linkage_1d(
            home["x"].to_numpy(dtype=float),
            k=3,
        ) if not home.empty else np.array([], dtype=float)
        away_oriented_x = np.array(
            [
                self._orientation_helper._orient_attack_right(float(x), 0.0, "away")[0]
                for x in away["x"].to_numpy(dtype=float)
            ],
            dtype=float,
        ) if not away.empty else np.array([], dtype=float)
        away_lines = self._orientation_helper._cluster_complete_linkage_1d(
            away_oriented_x,
            k=3,
        ) if away_oriented_x.size else np.array([], dtype=float)

        elapsed_seconds = pd.to_numeric(clean.get("elapsed_seconds"), errors="coerce")
        ball_x = pd.to_numeric(clean.get("ball_x"), errors="coerce")
        ball_y = pd.to_numeric(clean.get("ball_y"), errors="coerce")
        elapsed_value = float(elapsed_seconds.dropna().iloc[0]) if elapsed_seconds is not None and not elapsed_seconds.dropna().empty else None
        if ball_x is not None and ball_y is not None and not ball_x.dropna().empty and not ball_y.dropna().empty:
            ball_xy_bl: Optional[Tuple[float, float]] = self._orientation_helper._to_gandula_bottom_left(
                float(ball_x.dropna().iloc[0]),
                float(ball_y.dropna().iloc[0]),
            )
        else:
            ball_xy_bl = None

        return _TrackingFrameContext(
            elapsed_seconds=elapsed_value,
            ball_xy_bl=ball_xy_bl,
            player_positions_bl=player_positions_bl,
            team_side_by_player=team_side_by_player,
            team_side_by_team=team_side_by_team,
            xy_by_side={"home": home_xy, "away": away_xy},
            pressure_lines_by_side={"home": home_lines, "away": away_lines},
        )

    def _get_frame_context(
        self,
        tracking_df: Optional[pd.DataFrame],
        frame_cache: Dict[Tuple[int, int], Optional[_TrackingFrameContext]],
        match_id: Optional[int],
        frame_id: Optional[int],
    ) -> Optional[_TrackingFrameContext]:
        if tracking_df is None or tracking_df.empty or match_id is None or frame_id is None:
            return None

        key = (int(match_id), int(frame_id))
        if key in frame_cache:
            return frame_cache[key]

        if isinstance(tracking_df.index, pd.MultiIndex) and list(tracking_df.index.names[:2]) == ["match_id", "frame_id"]:
            try:
                frame_slice = tracking_df.loc[key]
            except KeyError:
                frame_cache[key] = None
                return None
            if isinstance(frame_slice, pd.Series):
                frame_slice = frame_slice.to_frame().T
            else:
                frame_slice = frame_slice.copy()
        else:
            match_col = "match_id" if "match_id" in tracking_df.columns else "game_id"
            frame_slice = tracking_df[
                (tracking_df[match_col] == int(match_id)) & (tracking_df["frame_id"] == int(frame_id))
            ].copy()
            if frame_slice.empty:
                frame_cache[key] = None
                return None

        context = self._build_frame_context(frame_slice)
        frame_cache[key] = context
        return context

    def _resolve_team_side_from_context(
        self,
        row: pd.Series,
        frame_context: Optional[_TrackingFrameContext],
    ) -> str:
        token = self._normalize_token(row.get("team_side"))
        if token in {"home", "away"}:
            return token

        if frame_context is not None:
            player_id = pd.to_numeric(row.get("player_id"), errors="coerce")
            if pd.notna(player_id):
                resolved = frame_context.team_side_by_player.get(int(player_id))
                if resolved in {"home", "away"}:
                    return resolved

            team_id = pd.to_numeric(row.get("team_id"), errors="coerce")
            if pd.notna(team_id):
                resolved = frame_context.team_side_by_team.get(int(team_id))
                if resolved in {"home", "away"}:
                    return resolved

        return "home"

    def _defending_influence_from_context(
        self,
        frame_context: Optional[_TrackingFrameContext],
        team_side: str,
        ball_x: float,
        ball_y: float,
    ) -> float:
        if frame_context is None:
            return float("nan")

        defending_side = "away" if team_side == "home" else "home"
        defenders = frame_context.xy_by_side.get(defending_side)
        if defenders is None or defenders.size == 0:
            return 0.0

        sigma = float(self.config.pressure_radius_m)
        dx = defenders[:, 0] - float(ball_x)
        dy = defenders[:, 1] - float(ball_y)
        d2 = dx * dx + dy * dy
        return float(np.mean(np.exp(-d2 / (2.0 * sigma * sigma))))

    def _pressure_line_indices_from_context(
        self,
        frame_context: Optional[_TrackingFrameContext],
        team_side: str,
        ball_x_oriented: float,
    ) -> Tuple[float, float]:
        if frame_context is None:
            return float("nan"), float("nan")

        attacking_lines = frame_context.pressure_lines_by_side.get(team_side, np.array([], dtype=float))
        defending_side = "away" if team_side == "home" else "home"
        defending_lines = frame_context.pressure_lines_by_side.get(defending_side, np.array([], dtype=float))

        def _nearest(lines: np.ndarray) -> float:
            if lines.size == 0:
                return float("nan")
            return float(int(np.argmin(np.abs(lines - float(ball_x_oriented)))))

        return _nearest(attacking_lines), _nearest(defending_lines)

    def _build_tracking_frame_lookup(
        self,
        tracking_df: Optional[pd.DataFrame],
    ) -> Dict[Tuple[int, int], _TrackingFrameContext]:
        if tracking_df is None or tracking_df.empty:
            return {}

        working = tracking_df.reset_index(drop=True) if isinstance(tracking_df.index, pd.MultiIndex) else tracking_df.copy()
        lookup: Dict[Tuple[int, int], _TrackingFrameContext] = {}

        current_key: Optional[Tuple[int, int]] = None
        elapsed_value: Optional[float] = None
        ball_xy_bl: Optional[Tuple[float, float]] = None
        player_positions_bl: Dict[str, Dict[str, Tuple[float, float]]] = {"home": {}, "away": {}}
        team_side_by_player: Dict[int, str] = {}
        team_side_by_team: Dict[int, str] = {}
        home_xy: List[Tuple[float, float]] = []
        away_xy: List[Tuple[float, float]] = []
        home_x: List[float] = []
        away_oriented_x: List[float] = []

        def _flush() -> None:
            nonlocal current_key, elapsed_value, ball_xy_bl, player_positions_bl
            nonlocal team_side_by_player, team_side_by_team, home_xy, away_xy, home_x, away_oriented_x
            if current_key is None:
                return

            lookup[current_key] = _TrackingFrameContext(
                elapsed_seconds=elapsed_value,
                ball_xy_bl=ball_xy_bl,
                player_positions_bl={
                    "home": dict(player_positions_bl["home"]),
                    "away": dict(player_positions_bl["away"]),
                },
                team_side_by_player=dict(team_side_by_player),
                team_side_by_team=dict(team_side_by_team),
                xy_by_side={
                    "home": np.asarray(home_xy, dtype=float) if home_xy else np.empty((0, 2), dtype=float),
                    "away": np.asarray(away_xy, dtype=float) if away_xy else np.empty((0, 2), dtype=float),
                },
                pressure_lines_by_side={
                    "home": self._orientation_helper._cluster_complete_linkage_1d(np.asarray(home_x, dtype=float), k=3)
                    if home_x else np.array([], dtype=float),
                    "away": self._orientation_helper._cluster_complete_linkage_1d(np.asarray(away_oriented_x, dtype=float), k=3)
                    if away_oriented_x else np.array([], dtype=float),
                },
            )

            current_key = None
            elapsed_value = None
            ball_xy_bl = None
            player_positions_bl = {"home": {}, "away": {}}
            team_side_by_player = {}
            team_side_by_team = {}
            home_xy = []
            away_xy = []
            home_x = []
            away_oriented_x = []

        for row in working.itertuples(index=False):
            match_id = pd.to_numeric(getattr(row, "match_id", None), errors="coerce")
            frame_id = pd.to_numeric(getattr(row, "frame_id", None), errors="coerce")
            if pd.isna(match_id) or pd.isna(frame_id):
                continue

            key = (int(match_id), int(frame_id))
            if current_key != key:
                _flush()
                current_key = key

            elapsed = pd.to_numeric(getattr(row, "elapsed_seconds", None), errors="coerce")
            if elapsed_value is None and pd.notna(elapsed):
                elapsed_value = float(elapsed)

            raw_ball_x = pd.to_numeric(getattr(row, "ball_x", None), errors="coerce")
            raw_ball_y = pd.to_numeric(getattr(row, "ball_y", None), errors="coerce")
            if ball_xy_bl is None and pd.notna(raw_ball_x) and pd.notna(raw_ball_y):
                ball_xy_bl = self._orientation_helper._to_gandula_bottom_left(float(raw_ball_x), float(raw_ball_y))

            team_side = str(getattr(row, "team_side", "")).strip().lower()
            if team_side not in {"home", "away"}:
                continue

            x = pd.to_numeric(getattr(row, "x", None), errors="coerce")
            y = pd.to_numeric(getattr(row, "y", None), errors="coerce")
            if pd.isna(x) or pd.isna(y):
                continue

            player_id = pd.to_numeric(getattr(row, "player_id", None), errors="coerce")
            team_id = pd.to_numeric(getattr(row, "team_id", None), errors="coerce")
            if pd.notna(player_id):
                player_positions_bl[team_side][str(int(player_id))] = self._orientation_helper._to_gandula_bottom_left(
                    float(x),
                    float(y),
                )
                team_side_by_player[int(player_id)] = team_side
            if pd.notna(team_id) and int(team_id) not in team_side_by_team:
                team_side_by_team[int(team_id)] = team_side

            if team_side == "home":
                home_xy.append((float(x), float(y)))
                home_x.append(float(x))
            else:
                away_xy.append((float(x), float(y)))
                away_oriented_x.append(self._orientation_helper._orient_attack_right(float(x), 0.0, "away")[0])

        _flush()
        return lookup

    def _grid_target_from_ball(self, ball_x: float, ball_y: float) -> Tuple[float, float]:
        ball_x_bl, ball_y_bl = self._orientation_helper._to_gandula_bottom_left(ball_x, ball_y)
        x_idx = int(np.clip(round(ball_x_bl), 0, self._pitch_grid_x - 1))
        y_idx = int(np.clip(round(ball_y_bl), 0, self._pitch_grid_y - 1))
        return float(self._pitch_x_grid[x_idx]), float(self._pitch_y_grid[y_idx])

    def _pitch_control_at_ball(
        self,
        frame_lookup: Dict[Tuple[int, int], _TrackingFrameContext],
        match_id: Optional[int],
        frame_id: Optional[int],
        team_side: str,
        ball_x: float,
        ball_y: float,
    ) -> float:
        if match_id is None or frame_id is None:
            return float("nan")

        frames = [
            frame_lookup.get((int(match_id), int(frame_id) + offset))
            for offset in (-2, -1, 0)
        ]
        if any(frame is None for frame in frames):
            return float("nan")

        window_frames = [frame for frame in frames if frame is not None]
        elapsed = np.asarray(
            [frame.elapsed_seconds if frame.elapsed_seconds is not None else np.nan for frame in window_frames],
            dtype=float,
        )
        if elapsed.size < 3:
            return float("nan")

        delta_t = np.diff(elapsed).astype(float, copy=False)
        delta_t[delta_t <= 0] = np.nan
        fill = np.nanmedian(delta_t)
        if not np.isfinite(fill):
            fill = 1.0 / self._pitch_fps_hint
        delta_t = np.nan_to_num(delta_t, nan=fill)

        def _stack_players(side_name: str) -> Tuple[np.ndarray, List[str]]:
            player_ids = sorted(
                {
                    player_id
                    for frame in window_frames
                    for player_id in frame.player_positions_bl.get(side_name, {}).keys()
                }
            )
            if not player_ids:
                return np.empty((0, len(window_frames), 2), dtype=float), []

            coords = np.full((len(player_ids), len(window_frames), 2), -1000.0, dtype=float)
            player_index = {player_id: idx for idx, player_id in enumerate(player_ids)}
            for time_idx, frame in enumerate(window_frames):
                for player_id, position in frame.player_positions_bl.get(side_name, {}).items():
                    coords[player_index[player_id], time_idx, :] = position
            return coords, player_ids

        attacking_side = team_side if team_side in {"home", "away"} else "home"
        defending_side = "away" if attacking_side == "home" else "home"
        att_pos_np, _ = _stack_players(attacking_side)
        def_pos_np, _ = _stack_players(defending_side)
        if att_pos_np.size == 0 or def_pos_np.size == 0:
            return float("nan")

        ball_positions = np.asarray(
            [frame.ball_xy_bl if frame.ball_xy_bl is not None else (np.nan, np.nan) for frame in window_frames],
            dtype=float,
        )
        if np.isnan(ball_positions).all():
            return float("nan")
        for axis in range(2):
            axis_values = ball_positions[:, axis]
            if np.isnan(axis_values).all():
                return float("nan")
            series = pd.Series(axis_values).ffill().bfill()
            ball_positions[:, axis] = series.to_numpy(dtype=float)

        target = np.asarray(self._grid_target_from_ball(ball_x, ball_y), dtype=float)
        att_pos = att_pos_np[:, 1, :]
        def_pos = def_pos_np[:, 1, :]
        att_v = (att_pos_np[:, 1, :] - att_pos_np[:, 0, :]) / float(delta_t[0])
        def_v = (def_pos_np[:, 1, :] - def_pos_np[:, 0, :]) / float(delta_t[0])
        ball_now = ball_positions[1]

        ball_tt = float(np.linalg.norm(target - ball_now) / self._pitch_average_ball_speed)
        att_r = (att_pos + att_v * self._pitch_reaction_time) - target
        def_r = (def_pos + def_v * self._pitch_reaction_time) - target
        tti_att = (np.linalg.norm(att_r, axis=1) + self._pitch_reaction_time) / self._pitch_max_player_speed
        tti_def = (np.linalg.norm(def_r, axis=1) + self._pitch_reaction_time) / self._pitch_max_player_speed
        tti = np.concatenate([tti_att, tti_def], axis=0)

        tmp2 = self._pitch_sigma * (ball_tt - tti)[:, None]
        tmp1 = self._pitch_sigma * 0.5 * (self._pitch_ti + 1.0) * 10.0 + tmp2
        hh_full = np.exp(-np.logaddexp(0.0, -tmp1)) * self._pitch_lambda
        h_team = hh_full[: len(tti_att)].sum(axis=0)
        attenuation = np.exp(
            -self._pitch_lambda
            * (np.logaddexp(0.0, tmp1) - np.logaddexp(0.0, tmp2)).sum(axis=0)
            / self._pitch_sigma
        )
        value = float(np.dot(attenuation * h_team, self._pitch_wi) * 5.0)
        return value if np.isfinite(value) else float("nan")

    def _compute_baseline_xg_batch(
        self,
        actions_df: pd.DataFrame,
        resolved_team_sides: List[str],
        ball_x: np.ndarray,
        ball_y: np.ndarray,
        baseline_xg_artifacts: Optional[BaselineXGArtifacts],
    ) -> np.ndarray:
        if baseline_xg_artifacts is None:
            existing = pd.to_numeric(actions_df.get("baseline_xg"), errors="coerce") if "baseline_xg" in actions_df.columns else None
            if existing is not None:
                return existing.fillna(0.0).to_numpy(dtype=float)
            if self.config.require_baseline_xg:
                raise ValueError("Missing baseline_xg feature and no baseline-xg artifacts were provided.")
            return np.zeros(len(actions_df), dtype=float)

        team_side_array = np.asarray(
            [side if side in {"home", "away"} else "home" for side in resolved_team_sides],
            dtype=object,
        )
        away_mask = team_side_array == "away"
        oriented_x = np.where(away_mask, -ball_x, ball_x)
        oriented_y = np.where(away_mask, -ball_y, ball_y)
        dx = GOAL_X - oriented_x
        dy = GOAL_Y - oriented_y

        if "set_piece" in actions_df.columns:
            set_piece = actions_df["set_piece"].map(self._normalize_set_piece)
        else:
            set_piece = pd.Series([None] * len(actions_df), index=actions_df.index, dtype=object)

        is_open_play = set_piece.isin([None, "open_play"]).astype(float)
        is_set_piece = (~set_piece.isin([None, "open_play"])).astype(float)
        is_free_kick = set_piece.eq("free_kick").astype(float)
        is_corner = set_piece.eq("corner").astype(float)
        is_penalty = set_piece.eq("penalty").astype(float)

        is_header = np.zeros(len(actions_df), dtype=float)
        for key in ("body_part", "shot_body_part", "shot_body", "header", "is_header"):
            if key not in actions_df.columns:
                continue
            column = actions_df[key]
            bool_mask = column.map(lambda value: int(bool(value)) if isinstance(value, (bool, np.bool_)) else None)
            if bool_mask.notna().any():
                is_header = np.where(bool_mask.fillna(0).to_numpy(dtype=int) == 1, 1.0, is_header)
            token_mask = column.astype("string").str.strip().str.lower().isin({"head", "header", "he"})
            is_header = np.where(token_mask.fillna(False).to_numpy(dtype=bool), 1.0, is_header)

        feature_df = pd.DataFrame(
            {
                "shot_x": oriented_x,
                "shot_y": oriented_y,
                "distance_to_goal": np.hypot(dx, dy),
                "angle_to_goal": np.abs(np.arctan2(dy, dx)),
                "is_open_play": is_open_play.to_numpy(dtype=float),
                "is_set_piece": is_set_piece.to_numpy(dtype=float),
                "is_free_kick": is_free_kick.to_numpy(dtype=float),
                "is_corner": is_corner.to_numpy(dtype=float),
                "is_penalty": is_penalty.to_numpy(dtype=float),
                "is_header": is_header,
            },
            index=actions_df.index,
        )
        probabilities = baseline_xg_artifacts.predict_proba(feature_df)
        return probabilities[:, 1].astype(float, copy=False)

    def _extract_single_row_features(
        self,
        row: pd.Series,
        tracking_df: Optional[pd.DataFrame],
        baseline_xg_artifacts: Optional[BaselineXGArtifacts],
    ) -> Dict[str, float]:
        normalized_row = self._normalize_row(row)
        ball_x, ball_y = self._resolve_ball_xy(normalized_row)
        if ball_x is None or ball_y is None:
            ball_x, ball_y = 0.0, 0.0

        normalized_tracking = tracking_df
        if normalized_tracking is not None and not (
            isinstance(normalized_tracking.index, pd.MultiIndex)
            and list(normalized_tracking.index.names[:2]) == ["match_id", "frame_id"]
        ):
            normalized_tracking = self._normalize_tracking_df(tracking_df)
        frame_players = self._tracking_slice(normalized_row, normalized_tracking)
        if frame_players is not None and not frame_players.empty:
            if "team_side" in frame_players.columns:
                frame_players["team_side"] = frame_players["team_side"].astype(str).str.lower().str.strip()

        team_side = self._resolve_team_side(normalized_row, frame_players)
        normalized_row["team_side"] = team_side
        normalized_row["ball_x_start"] = ball_x
        normalized_row["ball_y_start"] = ball_y

        ball_x_oriented, ball_y_oriented = self._orientation_helper._orient_attack_right(ball_x, ball_y, team_side)
        dx = GOAL_X - ball_x_oriented
        dy = GOAL_Y - ball_y_oriented
        distance_to_goal = float(math.hypot(dx, dy))
        angle_to_goal = float(abs(math.atan2(dy, dx)))

        pitch_control = self._orientation_helper._pitch_control_attacking_team_at_ball(normalized_row, normalized_tracking)
        defending_influence = self._orientation_helper.defending_team_influence_at_ball(
            frame_players,
            ball_x,
            ball_y,
            team_side,
        )
        att_idx, def_idx = self._pressure_line_indices(frame_players, ball_x_oriented, team_side)
        baseline_xg = self._compute_baseline_xg(normalized_row, baseline_xg_artifacts)

        return {
            "ball_x": float(ball_x_oriented),
            "angle_to_goal": float(angle_to_goal),
            "distance_to_goal": float(distance_to_goal),
            "pitch_control_attacking_team_at_ball": float(pitch_control) if pd.notna(pitch_control) else 0.0,
            "defending_team_influence_at_ball": float(defending_influence) if pd.notna(defending_influence) else 0.0,
            "closest_attacking_pressure_line_index": float(att_idx) if pd.notna(att_idx) else 0.0,
            "closest_defending_pressure_line_index": float(def_idx) if pd.notna(def_idx) else 0.0,
            "baseline_xg": float(baseline_xg),
        }

    def build_feature_frame(
        self,
        actions_df: pd.DataFrame,
        tracking_df: Optional[pd.DataFrame] = None,
        baseline_xg_artifacts: Optional[BaselineXGArtifacts] = None,
        return_audit: bool = False,
    ) -> pd.DataFrame:
        self._orientation_helper._flip_away_team_coordinates = self._orientation_helper._resolve_orientation_mode(actions_df)
        normalized_tracking = self._normalize_tracking_df(tracking_df)
        frame_lookup = self._build_tracking_frame_lookup(normalized_tracking)
        rows: List[Dict[str, float]] = []
        resolved_team_sides: List[str] = []
        raw_ball_x: List[float] = []
        raw_ball_y: List[float] = []
        row_iter = tqdm(
            actions_df.iterrows(),
            total=len(actions_df),
            desc="ActionSelection features",
            unit="row",
            dynamic_ncols=True,
        ) if len(actions_df) else []

        for _, row in row_iter:
            normalized_row = self._normalize_row(row)
            ball_x, ball_y = self._resolve_ball_xy(normalized_row)
            if ball_x is None or ball_y is None:
                ball_x, ball_y = 0.0, 0.0

            match_id = pd.to_numeric(normalized_row.get("game_id", normalized_row.get("match_id")), errors="coerce")
            frame_id = pd.to_numeric(normalized_row.get("frame_id", normalized_row.get("start_frame_id")), errors="coerce")
            frame_context = None
            match_key: Optional[int] = None
            frame_key: Optional[int] = None
            if pd.notna(match_id) and pd.notna(frame_id):
                match_key = int(match_id)
                frame_key = int(frame_id)
                frame_context = frame_lookup.get((match_key, frame_key))

            team_side = self._resolve_team_side_from_context(normalized_row, frame_context)
            ball_x_oriented, ball_y_oriented = self._orientation_helper._orient_attack_right(ball_x, ball_y, team_side)
            dx = GOAL_X - ball_x_oriented
            dy = GOAL_Y - ball_y_oriented
            att_idx, def_idx = self._pressure_line_indices_from_context(frame_context, team_side, ball_x_oriented)

            rows.append(
                {
                    "ball_x": float(ball_x_oriented),
                    "angle_to_goal": float(abs(math.atan2(dy, dx))),
                    "distance_to_goal": float(math.hypot(dx, dy)),
                    "pitch_control_attacking_team_at_ball": float(
                        self._pitch_control_at_ball(frame_lookup, match_key, frame_key, team_side, ball_x, ball_y)
                    ),
                    "defending_team_influence_at_ball": float(
                        self._defending_influence_from_context(frame_context, team_side, ball_x, ball_y)
                    ),
                    "closest_attacking_pressure_line_index": float(att_idx),
                    "closest_defending_pressure_line_index": float(def_idx),
                }
            )
            resolved_team_sides.append(team_side)
            raw_ball_x.append(float(ball_x))
            raw_ball_y.append(float(ball_y))

        raw_frame = pd.DataFrame(rows, index=actions_df.index)
        if not raw_frame.empty:
            raw_frame["baseline_xg"] = self._compute_baseline_xg_batch(
                actions_df,
                resolved_team_sides,
                np.asarray(raw_ball_x, dtype=float),
                np.asarray(raw_ball_y, dtype=float),
                baseline_xg_artifacts or self.baseline_xg_artifacts,
            )
        else:
            raw_frame["baseline_xg"] = pd.Series(dtype=float)
        raw_frame = raw_frame.replace([np.inf, -np.inf], np.nan)
        missing_mask = raw_frame.isna()
        frame = raw_frame.fillna(0.0)
        if not return_audit:
            return frame

        audit = missing_mask.astype(int)
        audit.columns = [f"missing__{column}" for column in audit.columns]
        audit["feature_missing_count"] = missing_mask.sum(axis=1).astype(int)
        audit["feature_missing_rate"] = audit["feature_missing_count"] / float(len(self.feature_columns)) if self.feature_columns else 0.0
        return frame, audit
