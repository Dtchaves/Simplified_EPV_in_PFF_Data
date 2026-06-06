from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
class ActionSelectionFeatureConfig:
    orientation_mode: str = "attack_right"
    pressure_radius_m: float = 8.0


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
            return float(value) if pd.notna(value) else 0.0

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
    ) -> pd.DataFrame:
        self._orientation_helper._flip_away_team_coordinates = self._orientation_helper._resolve_orientation_mode(actions_df)

        normalized_tracking = self._normalize_tracking_df(tracking_df)
        rows: List[Dict[str, float]] = []
        for _, row in actions_df.iterrows():
            rows.append(self._extract_single_row_features(row, normalized_tracking, baseline_xg_artifacts or self.baseline_xg_artifacts))

        frame = pd.DataFrame(rows, index=actions_df.index)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.fillna(0.0)
        return frame
