from __future__ import annotations

import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from gandula.utils.pitch_control.service import compute_pitch_control_from_dataframe
    from gandula.utils.pitch_control.config import PitchControlConfig
except Exception:  # pragma: no cover - runtime optional dependency
    compute_pitch_control_from_dataframe = None
    PitchControlConfig = None

BALL_DRIVE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = BALL_DRIVE_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Pass.data_utils import REPO_ROOT


GOAL_X = 52.5
GOAL_Y = 0.0


@dataclass
class BallDriveFeatureConfig:
    sigma_def_influence: float = 8.0
    pitch_control_grid_x: int = 105
    pitch_control_grid_y: int = 68
    orientation_mode: str = "attack_right"
    orientation_progress_threshold: float = 0.25
    orientation_min_samples: int = 50


class BallDriveFeatureBuilder:
    def __init__(self, config: Optional[BallDriveFeatureConfig] = None):
        self.config = config or BallDriveFeatureConfig()
        self.scaler = StandardScaler()
        self.feature_columns = [
            "ball_x",
            "ball_y",
            "distance_to_goal",
            "angle_to_goal",
            "pitch_control_attacking_team_at_ball",
            "defending_team_influence_at_ball",
            "nearest_attacking_pressure_line_distance",
            "nearest_defending_pressure_line_distance",
        ]
        self._fitted = False
        self._flip_away_team_coordinates = True

    @staticmethod
    def _as_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        token = str(value).strip().lower()
        if token in {"true", "1", "yes", "y", "t"}:
            return True
        if token in {"false", "0", "no", "n", "f"}:
            return False
        return None

    def _infer_flip_away_from_metadata(self, actions_df: pd.DataFrame) -> Optional[bool]:
        metadata_columns = [
            "attack_right_normalized",
            "coordinates_attack_right",
            "is_attack_right_normalized",
            "already_attack_right",
        ]

        for col in metadata_columns:
            if col not in actions_df.columns:
                continue
            values = actions_df[col].dropna().tolist()
            if not values:
                continue
            parsed = [self._as_bool(value) for value in values]
            parsed = [value for value in parsed if value is not None]
            if not parsed:
                continue
            normalized_true = float(sum(1 for value in parsed if value)) / float(len(parsed))
            # True means coordinates already attack-right, so no away flip.
            return not (normalized_true >= 0.5)

        return None

    def _infer_flip_away_from_progress(self, actions_df: pd.DataFrame) -> Optional[bool]:
        required = {"team_side", "ball_x_start", "ball_x_end"}
        if not required.issubset(set(actions_df.columns)):
            return None

        probe = actions_df[list(required)].copy()
        probe["team_side"] = probe["team_side"].astype(str).str.lower().str.strip()
        probe["ball_x_start"] = pd.to_numeric(probe["ball_x_start"], errors="coerce")
        probe["ball_x_end"] = pd.to_numeric(probe["ball_x_end"], errors="coerce")
        probe = probe.dropna(subset=["team_side", "ball_x_start", "ball_x_end"])
        if probe.empty:
            return None

        min_samples = int(max(1, self.config.orientation_min_samples))
        home = probe[probe["team_side"] == "home"]
        away = probe[probe["team_side"] == "away"]
        if len(home) < min_samples or len(away) < min_samples:
            return None

        home_progress = (home["ball_x_end"] - home["ball_x_start"]).to_numpy(dtype=float)
        away_progress = (away["ball_x_end"] - away["ball_x_start"]).to_numpy(dtype=float)
        home_med = float(np.nanmedian(home_progress))
        away_med = float(np.nanmedian(away_progress))
        threshold = float(max(0.0, self.config.orientation_progress_threshold))

        if (abs(home_med) < threshold) or (abs(away_med) < threshold):
            return None

        # Opposite median direction -> likely raw team-side coordinates, so flip away.
        if np.sign(home_med) != np.sign(away_med):
            return True

        # Same median direction -> likely already attack-right normalized.
        return False

    def _resolve_orientation_mode(self, actions_df: pd.DataFrame) -> bool:
        mode = str(self.config.orientation_mode).strip().lower()
        if mode in {"attack_right", "normalized", "already_attack_right"}:
            return False
        if mode in {"raw", "team_side", "flip_away"}:
            return True

        from_metadata = self._infer_flip_away_from_metadata(actions_df)
        if from_metadata is not None:
            return from_metadata

        from_progress = self._infer_flip_away_from_progress(actions_df)
        if from_progress is not None:
            return from_progress

        # Backward compatible fallback: raw team-side assumption.
        return True

    def _orient_attack_right(self, x: float, y: float, team_side: Any) -> Tuple[float, float]:
        side = str(team_side).strip().lower()
        if self._flip_away_team_coordinates and side == "away":
            return -float(x), -float(y)
        return float(x), float(y)

    @staticmethod
    def _to_gandula_bottom_left(x: float, y: float) -> Tuple[float, float]:
        return float(x + 52.5), float(y + 34.0)

    @staticmethod
    def _cluster_complete_linkage_1d(values: np.ndarray, k: int = 3) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.array([], dtype=float)
        clusters = [[value] for value in values]

        def dist(cluster_a: List[float], cluster_b: List[float]) -> float:
            return max(abs(a - b) for a in cluster_a for b in cluster_b)

        while len(clusters) > k:
            best_i, best_j, best_dist = 0, 1, float("inf")
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    candidate = dist(clusters[i], clusters[j])
                    if candidate < best_dist:
                        best_i, best_j, best_dist = i, j, candidate
            merged = clusters[best_i] + clusters[best_j]
            clusters = [clusters[idx] for idx in range(len(clusters)) if idx not in (best_i, best_j)]
            clusters.append(merged)

        centroids = np.array([np.mean(cluster) for cluster in clusters], dtype=float)
        centroids.sort()
        return centroids

    def _pressure_line_distances(
        self,
        tracking_frame: pd.DataFrame,
        ball_x_oriented: float,
        action_team_side: Any,
    ) -> Tuple[float, float]:
        if tracking_frame is None or tracking_frame.empty:
            return float("nan"), float("nan")

        frame = tracking_frame.copy()
        frame = frame.dropna(subset=["x", "team_side"])
        if frame.empty:
            return float("nan"), float("nan")

        oriented_x = []
        for _, row in frame.iterrows():
            x_oriented, _ = self._orient_attack_right(float(row["x"]), 0.0, row.get("team_side"))
            oriented_x.append(x_oriented)
        frame["x_oriented"] = oriented_x

        action_side = str(action_team_side).strip().lower()
        attackers = frame[frame["team_side"].astype(str).str.lower() == action_side]
        defenders = frame[frame["team_side"].astype(str).str.lower() != action_side]

        def _nearest_line_distance(group: pd.DataFrame) -> float:
            if group.empty:
                return float("nan")
            lines = self._cluster_complete_linkage_1d(group["x_oriented"].to_numpy(dtype=float), k=3)
            if lines.size == 0:
                return float("nan")
            return float(np.min(np.abs(lines - float(ball_x_oriented))))

        return _nearest_line_distance(attackers), _nearest_line_distance(defenders)

    def defending_team_influence_at_ball(
        self,
        tracking_frame: pd.DataFrame,
        ball_x: float,
        ball_y: float,
        action_team_side: Any,
    ) -> float:
        if tracking_frame is None or tracking_frame.empty:
            return float("nan")

        sigma = float(self.config.sigma_def_influence)
        frame = tracking_frame.copy()
        frame = frame.dropna(subset=["x", "y", "team_side"])
        if frame.empty:
            return float("nan")

        action_side = str(action_team_side).strip().lower()
        defenders = frame[frame["team_side"].astype(str).str.lower() != action_side]
        if defenders.empty:
            return 0.0

        dx = defenders["x"].to_numpy(dtype=float) - float(ball_x)
        dy = defenders["y"].to_numpy(dtype=float) - float(ball_y)
        d2 = dx * dx + dy * dy
        influence = np.exp(-d2 / (2.0 * sigma * sigma))
        return float(np.mean(influence))

    def _build_pitch_control_window(
        self,
        row: pd.Series,
        tracking_df: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        if tracking_df is None or tracking_df.empty:
            return None

        required = {"match_id", "frame_id", "elapsed_seconds", "team_side", "player_id", "x", "y", "ball_x", "ball_y"}
        if not required.issubset(set(tracking_df.columns)):
            return None

        match_id = row.get("game_id")
        frame_id = row.get("segment_start_frame_id", row.get("start_frame_id"))
        if pd.isna(match_id) or pd.isna(frame_id):
            return None

        match_id = int(match_id)
        frame_id = int(frame_id)
        frames = [frame_id - 2, frame_id - 1, frame_id]

        window = tracking_df[(tracking_df["match_id"] == match_id) & (tracking_df["frame_id"].isin(frames))].copy()
        if window.empty:
            return None

        unique_frames = sorted(window["frame_id"].dropna().astype(int).unique().tolist())
        if len(unique_frames) < 3:
            return None

        period_col = "period" if "period" in window.columns else None
        period_values = window[period_col] if period_col is not None else 1

        payload = pd.DataFrame(
            {
                "frame_id": window["frame_id"].astype(int),
                "period": period_values,
                "elapsed_seconds": pd.to_numeric(window["elapsed_seconds"], errors="coerce"),
                "team": window["team_side"].astype(str),
                "player_uid": window["player_id"].astype(str),
                "x": pd.to_numeric(window["x"], errors="coerce"),
                "y": pd.to_numeric(window["y"], errors="coerce"),
                "ball_x": pd.to_numeric(window["ball_x"], errors="coerce"),
                "ball_y": pd.to_numeric(window["ball_y"], errors="coerce"),
            }
        )

        payload = payload.dropna(subset=["elapsed_seconds", "x", "y", "ball_x", "ball_y"])
        if payload.empty:
            return None

        payload["x"] = payload["x"] + 52.5
        payload["y"] = payload["y"] + 34.0
        payload["ball_x"] = payload["ball_x"] + 52.5
        payload["ball_y"] = payload["ball_y"] + 34.0
        return payload

    def _pitch_control_attacking_team_at_ball(
        self,
        row: pd.Series,
        tracking_df: Optional[pd.DataFrame],
    ) -> float:
        if compute_pitch_control_from_dataframe is None or PitchControlConfig is None:
            return float("nan")

        payload = self._build_pitch_control_window(row, tracking_df)
        if payload is None or payload["frame_id"].nunique() < 3:
            return float("nan")

        try:
            config = PitchControlConfig(
                x_grids=int(self.config.pitch_control_grid_x),
                y_grids=int(self.config.pitch_control_grid_y),
            )
        except TypeError:
            config = PitchControlConfig()

        attack_team = str(row.get("team_side", "home")).strip().lower()

        try:
            pc_surface = compute_pitch_control_from_dataframe(
                payload,
                attacking_team=attack_team,
                config=config,
            )
            arr = pc_surface.team_control_numpy()
        except Exception:
            return float("nan")

        if not isinstance(arr, np.ndarray) or arr.ndim != 3:
            return float("nan")

        last_frame_grid = arr[-1]
        ball_x_bl, ball_y_bl = self._to_gandula_bottom_left(float(row.get("ball_x_start", 0.0)), float(row.get("ball_y_start", 0.0)))

        x_idx = int(np.clip(round(ball_x_bl), 0, last_frame_grid.shape[0] - 1))
        y_idx = int(np.clip(round(ball_y_bl), 0, last_frame_grid.shape[1] - 1))

        try:
            value = float(last_frame_grid[x_idx, y_idx])
        except Exception:
            try:
                value = float(last_frame_grid[y_idx, x_idx])
            except Exception:
                return float("nan")
        if not np.isfinite(value):
            return float("nan")
        return value

    def _extract_single_row_features(
        self,
        row: pd.Series,
        tracking_df: Optional[pd.DataFrame],
    ) -> Dict[str, float]:
        ball_x = float(row.get("ball_x_start", 0.0))
        ball_y = float(row.get("ball_y_start", 0.0))
        team_side = row.get("team_side")

        ball_x_oriented, ball_y_oriented = self._orient_attack_right(ball_x, ball_y, team_side)
        dx = GOAL_X - ball_x_oriented
        dy = GOAL_Y - ball_y_oriented

        frame_slice = None
        if tracking_df is not None and not tracking_df.empty:
            match_id = row.get("game_id")
            frame_id = row.get("segment_start_frame_id", row.get("start_frame_id"))
            if pd.notna(match_id) and pd.notna(frame_id):
                frame_slice = tracking_df[
                    (tracking_df["match_id"] == int(match_id))
                    & (tracking_df["frame_id"] == int(frame_id))
                ]

        nearest_att_dist, nearest_def_dist = self._pressure_line_distances(frame_slice, ball_x_oriented, team_side)

        return {
            "ball_x": float(ball_x_oriented),
            "ball_y": float(ball_y_oriented),
            "distance_to_goal": float(np.hypot(dx, dy)),
            "angle_to_goal": float(abs(np.arctan2(dy, dx))),
            "pitch_control_attacking_team_at_ball": self._pitch_control_attacking_team_at_ball(row, tracking_df),
            "defending_team_influence_at_ball": self.defending_team_influence_at_ball(frame_slice, ball_x, ball_y, team_side),
            "nearest_attacking_pressure_line_distance": nearest_att_dist,
            "nearest_defending_pressure_line_distance": nearest_def_dist,
        }

    def build_feature_frame(
        self,
        actions_df: pd.DataFrame,
        tracking_df: Optional[pd.DataFrame] = None,
        include_p_drive_success: bool = False,
    ) -> pd.DataFrame:
        self._flip_away_team_coordinates = self._resolve_orientation_mode(actions_df)

        rows: List[Dict[str, float]] = []
        for _, row in actions_df.iterrows():
            features = self._extract_single_row_features(row, tracking_df)
            if include_p_drive_success:
                features["p_drive_success"] = float(row.get("p_ball_drive_success", row.get("p_drive_success", np.nan)))
            rows.append(features)

        frame = pd.DataFrame(rows, index=actions_df.index)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.fillna(0.0)
        return frame

    def fit(self, feature_df: pd.DataFrame) -> None:
        self.scaler.fit(feature_df[self.feature_columns].to_numpy(dtype=float))
        self._fitted = True

    def transform(self, feature_df: pd.DataFrame, include_p_drive_success: bool = False) -> np.ndarray:
        cols = list(self.feature_columns)
        if include_p_drive_success:
            cols = cols + ["p_drive_success"]
        matrix = feature_df[cols].to_numpy(dtype=float)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        if include_p_drive_success:
            if not self._fitted:
                raise RuntimeError("Feature scaler must be fitted before transform.")
            base_scaled = self.scaler.transform(matrix[:, : len(self.feature_columns)])
            return np.concatenate([base_scaled, matrix[:, -1:]], axis=1)

        if not self._fitted:
            raise RuntimeError("Feature scaler must be fitted before transform.")
        return self.scaler.transform(matrix)

    def fit_transform(self, feature_df: pd.DataFrame) -> np.ndarray:
        self.fit(feature_df)
        return self.transform(feature_df)

    def save(self, path: Path) -> Path:
        payload = {
            "config": self.config,
            "feature_columns": self.feature_columns,
            "scaler": self.scaler,
            "fitted": self._fitted,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        return path

    @classmethod
    def load(cls, path: Path) -> "BallDriveFeatureBuilder":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        builder = cls(config=payload["config"])
        builder.feature_columns = list(payload["feature_columns"])
        builder.scaler = payload["scaler"]
        builder._fitted = bool(payload["fitted"])
        return builder
