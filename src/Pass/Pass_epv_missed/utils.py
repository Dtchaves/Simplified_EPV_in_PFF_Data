from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


THIS_DIR = Path(__file__).resolve().parent
PP_SOCCERMAP_PATH = THIS_DIR.parent / "Pass_sucess_probability" / "soccermap.py"


def _load_pp_model_class():
    spec = importlib.util.spec_from_file_location("pp_soccermap", str(PP_SOCCERMAP_PATH))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.SoccerMapPassSucess


class ToSoccerMapTensor:
    """Build 16-channel PE-missed tensors aligned to Table 5/6 contracts."""

    def __init__(self, dim: Tuple[int, int] = (68, 104), pp_model_path: Optional[Path] = None):
        assert len(dim) == 2
        self.y_bins, self.x_bins = dim
        self.pp_model = self._load_or_init_pp_model(pp_model_path)
        self._last_att_vertical_lines: Optional[np.ndarray] = None
        self._last_att_horizontal_lines: Optional[np.ndarray] = None
        self._last_def_vertical_lines: Optional[np.ndarray] = None
        self._last_def_horizontal_lines: Optional[np.ndarray] = None

    @staticmethod
    def _safe_float(value):
        if isinstance(value, pd.Series):
            if value.empty:
                return np.nan
            return float(value.iloc[0])
        return float(value)

    def _load_or_init_pp_model(self, pp_model_path: Optional[Path]) -> torch.nn.Module:
        model: Optional[torch.nn.Module] = None
        if pp_model_path is not None and pp_model_path.exists():
            try:
                loaded = torch.load(pp_model_path, map_location="cpu")
                if isinstance(loaded, torch.nn.Module):
                    model = loaded
            except Exception:
                model = None

        if model is None:
            pp_class = _load_pp_model_class()
            model = pp_class(in_channels=13)

        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _get_cell_indexes(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_bin = np.clip((x + 52.5) / 105 * self.x_bins, 0, self.x_bins - 1).astype(np.uint8)
        y_bin = np.clip((y + 34) / 68 * self.y_bins, 0, self.y_bins - 1).astype(np.uint8)
        return x_bin, y_bin

    def _cluster_complete_linkage_1d(self, values: np.ndarray, k: int = 3) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.array([], dtype=float)

        clusters = [[value] for value in values]

        def cluster_distance(cluster_a: List[float], cluster_b: List[float]) -> float:
            return max(abs(a - b) for a in cluster_a for b in cluster_b)

        while len(clusters) > k:
            best_i, best_j, best_dist = 0, 1, float("inf")
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = cluster_distance(clusters[i], clusters[j])
                    if dist < best_dist:
                        best_i, best_j, best_dist = i, j, dist
            merged = clusters[best_i] + clusters[best_j]
            clusters = [clusters[idx] for idx in range(len(clusters)) if idx not in (best_i, best_j)]
            clusters.append(merged)

        centroids = np.array([np.mean(cluster) for cluster in clusters], dtype=float)
        centroids.sort()
        return centroids

    def _default_lines(self, axis: str, k: int = 3) -> np.ndarray:
        axis_size = self.x_bins if axis == "x" else self.y_bins
        return np.linspace(0.2 * (axis_size - 1), 0.8 * (axis_size - 1), k, dtype=float)

    def _resolve_dynamic_lines(
        self,
        values: np.ndarray,
        last_lines: Optional[np.ndarray],
        axis: str,
        k: int = 3,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=float)

        if values.size >= k:
            lines = self._cluster_complete_linkage_1d(values, k=k)
        elif last_lines is not None:
            lines = np.asarray(last_lines, dtype=float)
        elif values.size > 0:
            partial = self._cluster_complete_linkage_1d(values, k=int(values.size))
            lines = self._default_lines(axis, k=k)
            lines[: partial.size] = partial
            lines.sort()
        else:
            lines = self._default_lines(axis, k=k)

        if lines.size != k:
            fixed = self._default_lines(axis, k=k)
            sorted_lines = np.sort(np.asarray(lines, dtype=float))
            keep = min(k, sorted_lines.size)
            fixed[:keep] = sorted_lines[:keep]
            fixed.sort()
            lines = fixed

        return np.asarray(lines, dtype=float)

    def _drop_goalkeeper_like_outlier(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        min_players: int = 8,
        min_gap_m: float = 6.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove a single extreme x-axis outlier when it creates a lone pressure line."""
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)

        if x_values.size != y_values.size or x_values.size < min_players:
            return x_values, y_values

        centroids = self._cluster_complete_linkage_1d(x_values, k=3)
        if centroids.size != 3:
            return x_values, y_values

        nearest_cluster = np.argmin(np.abs(x_values[:, None] - centroids[None, :]), axis=1)
        cluster_sizes = np.bincount(nearest_cluster, minlength=3)
        singleton_cluster = int(np.argmin(cluster_sizes))

        if cluster_sizes[singleton_cluster] != 1 or singleton_cluster not in (0, 2):
            return x_values, y_values

        singleton_idx = int(np.where(nearest_cluster == singleton_cluster)[0][0])
        sorted_idx = np.argsort(x_values)
        position = int(np.where(sorted_idx == singleton_idx)[0][0])

        min_gap_bins = float(min_gap_m * self.x_bins / 105.0)
        if position == 0:
            separation_gap = float(x_values[sorted_idx[1]] - x_values[sorted_idx[0]])
        elif position == x_values.size - 1:
            separation_gap = float(x_values[sorted_idx[-1]] - x_values[sorted_idx[-2]])
        else:
            return x_values, y_values

        if separation_gap < min_gap_bins:
            return x_values, y_values

        keep_mask = np.ones(x_values.size, dtype=bool)
        keep_mask[singleton_idx] = False
        return x_values[keep_mask], y_values[keep_mask]

    def _signed_nearest_distance_map(self, line_positions: np.ndarray, axis: str) -> np.ndarray:
        if axis == "x":
            axis_values = np.arange(self.x_bins, dtype=float)[None, :]
            signed_distances = axis_values - line_positions[:, None]
            nearest_idx = np.argmin(np.abs(signed_distances), axis=0)
            nearest_signed = signed_distances[nearest_idx, np.arange(self.x_bins)]
            normalized = np.clip(nearest_signed / (0.5 * self.x_bins), -1.0, 1.0)
            return np.tile(normalized[None, :], (self.y_bins, 1)).astype(float)

        axis_values = np.arange(self.y_bins, dtype=float)[:, None]
        signed_distances = axis_values - line_positions[None, :]
        nearest_idx = np.argmin(np.abs(signed_distances), axis=1)
        nearest_signed = signed_distances[np.arange(self.y_bins), nearest_idx]
        normalized = np.clip(nearest_signed / (0.5 * self.y_bins), -1.0, 1.0)
        return np.tile(normalized[:, None], (1, self.x_bins)).astype(float)

    def _build_pressure_map(
        self,
        x_bins: np.ndarray,
        y_bins: np.ndarray,
        last_vertical_lines: Optional[np.ndarray],
        last_horizontal_lines: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vertical_lines = self._resolve_dynamic_lines(x_bins, last_vertical_lines, axis="x", k=3)
        horizontal_lines = self._resolve_dynamic_lines(y_bins, last_horizontal_lines, axis="y", k=3)

        vertical_component = self._signed_nearest_distance_map(vertical_lines, axis="x")
        horizontal_component = self._signed_nearest_distance_map(horizontal_lines, axis="y")
        combined = np.clip(0.5 * (vertical_component + horizontal_component), -1.0, 1.0)

        return combined, vertical_lines, horizontal_lines

    def _count_players_between_x(self, player_x: np.ndarray, x_start: np.ndarray, x_end: np.ndarray) -> np.ndarray:
        lower = np.minimum(x_start, x_end)
        upper = np.maximum(x_start, x_end)
        if player_x.size == 0:
            return np.zeros((self.y_bins, self.x_bins), dtype=float)

        player_x_exp = player_x[None, None, :]
        lower_exp = lower[:, :, None]
        upper_exp = upper[:, :, None]
        between = (player_x_exp > lower_exp) & (player_x_exp < upper_exp)
        return between.sum(axis=2).astype(float)

    def _build_pp_input(self, sample: Dict) -> torch.Tensor:
        start_x, start_y = sample["ball_x_start"], sample["ball_y_start"]
        carrier_vx = float(sample.get("vx_carrier", 0.0) or 0.0)
        carrier_vy = float(sample.get("vy_carrier", 0.0) or 0.0)
        frame = sample["frame"]
        team_id = int(sample["team_id"])

        ball_coo = np.array([[start_x, start_y]], dtype=float)
        goal_coo = np.array([[52.5, 0]], dtype=float)

        matrix = np.zeros((13, self.y_bins, self.x_bins), dtype=float)
        x0_ball, y0_ball = self._get_cell_indexes(ball_coo[:, 0], ball_coo[:, 1])
        x0_goal, y0_goal = self._get_cell_indexes(goal_coo[:, 0], goal_coo[:, 1])

        x_ball = float(x0_ball[0])
        y_ball = float(y0_ball[0])
        x_goal = float(x0_goal[0])
        y_goal = float(y0_goal[0])

        yy = np.arange(self.y_bins, dtype=float)[:, None]
        xx = np.arange(self.x_bins, dtype=float)[None, :]

        dx_goal = x_goal - xx
        dy_goal = y_goal - yy
        distance_to_goal = np.sqrt(dx_goal ** 2 + dy_goal ** 2)
        angle_to_goal = np.abs(np.arctan2(dy_goal, dx_goal))

        dx_ball = x_ball - xx
        dy_ball = y_ball - yy
        distance_to_ball = np.sqrt(dx_ball ** 2 + dy_ball ** 2)
        angle_to_ball = np.arctan2(dy_ball, dx_ball)
        sin_angle_to_ball = np.sin(angle_to_ball)
        cos_angle_to_ball = np.cos(angle_to_ball)

        pass_dx = np.broadcast_to(xx - x_ball, (self.y_bins, self.x_bins))
        pass_dy = np.broadcast_to(yy - y_ball, (self.y_bins, self.x_bins))
        pass_norm = np.sqrt(pass_dx ** 2 + pass_dy ** 2)
        carrier_speed = float(np.hypot(carrier_vx, carrier_vy))
        sin_angle_to_carrier_velocity = np.zeros((self.y_bins, self.x_bins), dtype=float)
        cos_angle_to_carrier_velocity = np.zeros((self.y_bins, self.x_bins), dtype=float)
        if carrier_speed > 0:
            valid = pass_norm > 0
            denom = pass_norm[valid] * carrier_speed
            cos_vals = (pass_dx[valid] * carrier_vx + pass_dy[valid] * carrier_vy) / denom
            sin_vals = (pass_dx[valid] * carrier_vy - pass_dy[valid] * carrier_vx) / denom
            cos_angle_to_carrier_velocity[valid] = np.clip(cos_vals, -1.0, 1.0)
            sin_angle_to_carrier_velocity[valid] = np.clip(sin_vals, -1.0, 1.0)

        player_columns = [col for col in frame.columns if col.startswith("x_player_")]
        for player_col in player_columns:
            player_suffix = player_col.split("_")[-1]
            x_col = f"x_player_{player_suffix}"
            y_col = f"y_player_{player_suffix}"
            team_col = f"team_id_player_{player_suffix}"

            if team_col not in frame.columns:
                continue
            team_value = frame[team_col]
            if pd.isna(team_value).all():
                continue

            x_val = self._safe_float(frame[x_col])
            y_val = self._safe_float(frame[y_col])
            if np.isnan(x_val) or np.isnan(y_val):
                continue

            vx_col = f"vx_player_{player_suffix}"
            vy_col = f"vy_player_{player_suffix}"
            vx_val = self._safe_float(frame[vx_col]) if vx_col in frame.columns else 0.0
            vy_val = self._safe_float(frame[vy_col]) if vy_col in frame.columns else 0.0
            if np.isnan(vx_val):
                vx_val = 0.0
            if np.isnan(vy_val):
                vy_val = 0.0

            x_bin, y_bin = self._get_cell_indexes(np.array([x_val]), np.array([y_val]))
            x_idx = int(x_bin[0])
            y_idx = int(y_bin[0])

            if int(team_value.iloc[0]) == team_id:
                matrix[0, y_idx, x_idx] = 1.0
                matrix[2, y_idx, x_idx] += vx_val
                matrix[3, y_idx, x_idx] += vy_val
            else:
                matrix[1, y_idx, x_idx] = 1.0
                matrix[4, y_idx, x_idx] += vx_val
                matrix[5, y_idx, x_idx] += vy_val

        matrix[6, :, :] = angle_to_goal
        matrix[7, :, :] = sin_angle_to_ball
        matrix[8, :, :] = cos_angle_to_ball
        matrix[9, :, :] = sin_angle_to_carrier_velocity
        matrix[10, :, :] = cos_angle_to_carrier_velocity
        matrix[11, :, :] = distance_to_goal
        matrix[12, :, :] = distance_to_ball

        return torch.from_numpy(matrix).float().unsqueeze(0)

    def _infer_pp_surface(self, sample: Dict) -> np.ndarray:
        pp_input = self._build_pp_input(sample)
        with torch.no_grad():
            pp_surface = self.pp_model(pp_input)
        return pp_surface[0, 0].cpu().numpy().astype(float)

    def __call__(self, sample: Dict):
        pass_outcome_mapping = {
            "C": 1,
            "D": 0,
            "B": 0,
            "O": 0,
            "S": 0,
            "G": 0,
            "I": 0,
        }

        start_x, start_y, end_x, end_y = (
            sample["ball_x_start"],
            sample["ball_y_start"],
            sample["ball_x_end"],
            sample["ball_y_end"],
        )
        frame = sample["frame"]
        team_id = int(sample["team_id"])

        pass_outcome_type = sample.get("pass_outcome_type")
        pass_outcome_type_mapped = pass_outcome_mapping.get(pass_outcome_type, None)
        if pass_outcome_type_mapped is None:
            raise ValueError(f"Invalid pass_outcome_type: {pass_outcome_type}")
        target = int(pass_outcome_type_mapped)

        ball_coo = np.array([[start_x, start_y]], dtype=float)
        goal_coo = np.array([[52.5, 0]], dtype=float)

        matrix = np.zeros((16, self.y_bins, self.x_bins), dtype=float)
        x0_ball, y0_ball = self._get_cell_indexes(ball_coo[:, 0], ball_coo[:, 1])
        x0_goal, y0_goal = self._get_cell_indexes(goal_coo[:, 0], goal_coo[:, 1])

        x_ball = float(x0_ball[0])
        y_ball = float(y0_ball[0])
        x_goal = float(x0_goal[0])
        y_goal = float(y0_goal[0])

        yy = np.arange(self.y_bins, dtype=float)[:, None]
        xx = np.arange(self.x_bins, dtype=float)[None, :]

        dx_goal = x_goal - xx
        dy_goal = y_goal - yy
        angle_to_goal = np.abs(np.arctan2(dy_goal, dx_goal))
        distance_to_goal = np.sqrt(dx_goal ** 2 + dy_goal ** 2)

        dx_ball = x_ball - xx
        dy_ball = y_ball - yy
        distance_to_ball = np.sqrt(dx_ball ** 2 + dy_ball ** 2)

        attacking_x_bins: List[int] = []
        attacking_y_bins: List[int] = []
        defending_x_bins: List[int] = []
        defending_y_bins: List[int] = []

        player_columns = [col for col in frame.columns if col.startswith("x_player_")]
        for player_col in player_columns:
            player_suffix = player_col.split("_")[-1]
            x_col = f"x_player_{player_suffix}"
            y_col = f"y_player_{player_suffix}"
            team_col = f"team_id_player_{player_suffix}"

            if team_col not in frame.columns:
                continue
            team_value = frame[team_col]
            if pd.isna(team_value).all():
                continue

            x_val = self._safe_float(frame[x_col])
            y_val = self._safe_float(frame[y_col])
            if np.isnan(x_val) or np.isnan(y_val):
                continue

            vx_col = f"vx_player_{player_suffix}"
            vy_col = f"vy_player_{player_suffix}"
            vx_val = self._safe_float(frame[vx_col]) if vx_col in frame.columns else 0.0
            vy_val = self._safe_float(frame[vy_col]) if vy_col in frame.columns else 0.0
            if np.isnan(vx_val):
                vx_val = 0.0
            if np.isnan(vy_val):
                vy_val = 0.0

            x_bin, y_bin = self._get_cell_indexes(np.array([x_val]), np.array([y_val]))
            x_idx = int(x_bin[0])
            y_idx = int(y_bin[0])

            if int(team_value.iloc[0]) == team_id:
                matrix[0, y_idx, x_idx] = 1.0
                matrix[2, y_idx, x_idx] += vx_val
                matrix[3, y_idx, x_idx] += vy_val
                attacking_x_bins.append(x_idx)
                attacking_y_bins.append(y_idx)
            else:
                matrix[1, y_idx, x_idx] = 1.0
                matrix[4, y_idx, x_idx] += vx_val
                matrix[5, y_idx, x_idx] += vy_val
                defending_x_bins.append(x_idx)
                defending_y_bins.append(y_idx)

        matrix[6, :, :] = angle_to_goal
        matrix[7, :, :] = distance_to_goal
        matrix[8, :, :] = distance_to_ball

        att_x_arr = np.asarray(attacking_x_bins, dtype=float)
        att_y_arr = np.asarray(attacking_y_bins, dtype=float)
        def_x_arr = np.asarray(defending_x_bins, dtype=float)
        def_y_arr = np.asarray(defending_y_bins, dtype=float)

        att_pressure_x, att_pressure_y = self._drop_goalkeeper_like_outlier(att_x_arr, att_y_arr)
        def_pressure_x, def_pressure_y = self._drop_goalkeeper_like_outlier(def_x_arr, def_y_arr)

        att_pressure_map, att_vertical_lines, att_horizontal_lines = self._build_pressure_map(
            att_pressure_x,
            att_pressure_y,
            self._last_att_vertical_lines,
            self._last_att_horizontal_lines,
        )
        self._last_att_vertical_lines = att_vertical_lines
        self._last_att_horizontal_lines = att_horizontal_lines

        def_pressure_map, def_vertical_lines, def_horizontal_lines = self._build_pressure_map(
            def_pressure_x,
            def_pressure_y,
            self._last_def_vertical_lines,
            self._last_def_horizontal_lines,
        )
        self._last_def_vertical_lines = def_vertical_lines
        self._last_def_horizontal_lines = def_horizontal_lines

        matrix[9, :, :] = att_pressure_map
        matrix[10, :, :] = def_pressure_map

        x_ball_surface = np.full((self.y_bins, self.x_bins), x_ball, dtype=float)
        x_goal_surface = np.full((self.y_bins, self.x_bins), x_goal, dtype=float)
        matrix[11, :, :] = self._count_players_between_x(att_x_arr, x_ball_surface, xx)
        matrix[12, :, :] = self._count_players_between_x(def_x_arr, x_ball_surface, xx)
        matrix[13, :, :] = self._count_players_between_x(att_x_arr, x_goal_surface, xx)
        matrix[14, :, :] = self._count_players_between_x(def_x_arr, x_goal_surface, xx)

        matrix[15, :, :] = self._infer_pp_surface(sample)

        mask = np.zeros((1, self.y_bins, self.x_bins), dtype=float)
        end_ball_coo = np.array([[end_x, end_y]], dtype=float)
        if np.isnan(end_ball_coo).any():
            raise ValueError("End coordinates not known.")
        x0_ball_end, y0_ball_end = self._get_cell_indexes(end_ball_coo[:, 0], end_ball_coo[:, 1])
        mask[0, y0_ball_end, x0_ball_end] = 1.0

        return (
            torch.from_numpy(matrix).float(),
            torch.from_numpy(mask).float(),
            torch.tensor([target]).float(),
        )
