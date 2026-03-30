import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt
import os

def plot_loss(train_losses, val_losses,epoch,model_name,path_save_plot):
    fig = plt.figure(figsize=(13,5))
    ax = fig.gca()
    plt.ion()
    ax.plot(train_losses, label="Train loss", color = "tab:blue")
    ax.plot(val_losses, label="Validation loss", color = "tab:orange")
    ax.legend(fontsize="16")
    ax.set_xlabel("Epochs", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title(f"Training and Validation Loss", fontsize="16")
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)

    plt.grid(axis="y", linestyle="--", alpha=0.7)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)

    ax.tick_params(width=0.5)

    ax.set_facecolor("whitesmoke")
    model = model_name + ".png" 
    save_path = os.path.join(path_save_plot, model)
    plt.savefig(save_path, dpi=300)
    plt.close()
    




class ToSoccerMapTensor:
    """Convert inputs to a spatial representation.

    Parameters
    ----------
    dim : tuple(int), default=(68, 104)
        The dimensions of the pitch in the spatial representation.
    """

    def __init__(self, dim=(68, 104)):
        assert len(dim) == 2
        self.y_bins, self.x_bins = dim
        self._last_att_lines = None
        self._last_def_lines = None

    @staticmethod
    def _safe_float(value):
        if isinstance(value, pd.Series):
            if value.empty:
                return np.nan
            return float(value.iloc[0])
        return float(value)

    def _cluster_complete_linkage_1d(self, values, k=3):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return np.array([], dtype=float)

        clusters = [[v] for v in values]

        def cluster_distance(a, b):
            return max(abs(x - y) for x in a for y in b)

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

        centroids = np.array([np.mean(c) for c in clusters], dtype=float)
        centroids.sort()
        return centroids

    def _resolve_vertical_lines(self, x_bin_values, last_lines):
        if len(x_bin_values) >= 3:
            return self._cluster_complete_linkage_1d(x_bin_values, k=3)
        if last_lines is not None:
            return last_lines
        return np.linspace(0.2 * (self.x_bins - 1), 0.8 * (self.x_bins - 1), 3)

    def _signed_nearest_vertical_map(self, line_x_bins):
        x_axis = np.arange(self.x_bins, dtype=float)[None, :]
        signed_distances = x_axis - line_x_bins[:, None]
        nearest_idx = np.argmin(np.abs(signed_distances), axis=0)
        nearest_signed = signed_distances[nearest_idx, np.arange(self.x_bins)]
        normalized = np.clip(nearest_signed / (0.5 * self.x_bins), -1.0, 1.0)
        return np.tile(normalized, (self.y_bins, 1))

    def _normalize_surface(self, surface):
        surface = np.asarray(surface, dtype=float)
        if surface.shape == (1, self.y_bins, self.x_bins):
            surface = surface[0]
        if surface.shape != (self.y_bins, self.x_bins):
            return np.full((self.y_bins, self.x_bins), 1.0 / (self.y_bins * self.x_bins), dtype=float)
        clipped = np.clip(surface, 0.0, None)
        total = clipped.sum()
        if total <= 0:
            return np.full((self.y_bins, self.x_bins), 1.0 / (self.y_bins * self.x_bins), dtype=float)
        return clipped / total

    def _get_cell_indexes(self, x, y):
        x_bin = np.clip((x + 52.5) / 105 * self.x_bins, 0, self.x_bins - 1).astype(np.uint8)
        y_bin = np.clip((y + 34) / 68 * self.y_bins, 0, self.y_bins - 1).astype(np.uint8)
        return x_bin, y_bin

    def __call__(self, sample):
        pass_outcome_mapping = {
            'C': 1,
            'D': 0,
            'B': 0,
            'O': 0,
            'S': 0,
            'G': 0,
            'I': 0,
            
        }
        start_x, start_y, end_x, end_y = (
            sample["ball_x_start"],
            sample["ball_y_start"],
            sample["ball_x_end"],
            sample["ball_y_end"],
        )
        carrier_vx = float(sample.get("vx_carrier", 0.0) or 0.0)
        carrier_vy = float(sample.get("vy_carrier", 0.0) or 0.0)
        frame = sample["frame"]

        pass_outcome_type = sample.get("pass_outcome_type")
        pass_outcome_type_mapped = pass_outcome_mapping.get(pass_outcome_type, None)
        if pass_outcome_type_mapped is None:
            raise ValueError(f"Invalid pass_outcome_type: {pass_outcome_type}")
        target = int(pass_outcome_type_mapped)

        ball_coo = np.array([[start_x, start_y]], dtype=float)
        goal_coo = np.array([[52.5, 0]], dtype=float)
        team_id = sample['team_id']
        player_columns = [col for col in frame.columns if col.startswith('x_player_')]

        matrix = np.zeros((13, self.y_bins, self.x_bins), dtype=float)
        x0_ball, y0_ball = self._get_cell_indexes(ball_coo[:, 0], ball_coo[:, 1])
        x0_goal, y0_goal = self._get_cell_indexes(goal_coo[:, 0], goal_coo[:, 1])

        x_ball = float(x0_ball[0])
        y_ball = float(y0_ball[0])
        x_goal = float(x0_goal[0])
        y_goal = float(y0_goal[0])

        yy = np.arange(self.y_bins, dtype=float)[:, None]
        xx = np.arange(self.x_bins, dtype=float)[None, :]

        # Geometric channels over all field locations.
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

        for player_col in player_columns:
            player_id = player_col.split('_')[-1]
            x_col = f'x_player_{player_id}'
            y_col = f'y_player_{player_id}'

            team_col = f'team_id_player_{player_id}'
            if team_col not in frame.columns:
                continue

            team_id_col = frame[team_col]
            if pd.isna(team_id_col).all():
                continue

            x_val = self._safe_float(frame[x_col])
            y_val = self._safe_float(frame[y_col])
            if np.isnan(x_val) or np.isnan(y_val):
                continue

            vx_col = f'vx_player_{player_id}'
            vy_col = f'vy_player_{player_id}'
            vx_val = self._safe_float(frame[vx_col]) if vx_col in frame.columns else 0.0
            vy_val = self._safe_float(frame[vy_col]) if vy_col in frame.columns else 0.0
            if np.isnan(vx_val):
                vx_val = 0.0
            if np.isnan(vy_val):
                vy_val = 0.0

            x_bin, y_bin = self._get_cell_indexes(np.array([x_val]), np.array([y_val]))
            x_idx = int(x_bin[0])
            y_idx = int(y_bin[0])

            if int(team_id_col.iloc[0]) == int(team_id):
                matrix[0, y_idx, x_idx] = 1.0
                matrix[2, y_idx, x_idx] += vx_val
                matrix[3, y_idx, x_idx] += vy_val
            else:
                matrix[1, y_idx, x_idx] = 1.0
                matrix[4, y_idx, x_idx] += vx_val
                matrix[5, y_idx, x_idx] += vy_val

        # Table-5 PP/PS channels (shared):
        # 1 att loc, 2 def loc, 3 att vx, 4 att vy, 5 def vx, 6 def vy,
        # 7 angle-to-goal, 8 sin(angle-to-ball), 9 cos(angle-to-ball),
        # 10 sin(angle-to-carrier-velocity), 11 cos(angle-to-carrier-velocity),
        # 12 distance-to-goal, 13 distance-to-ball.
        matrix[6, :, :] = angle_to_goal
        matrix[7, :, :] = sin_angle_to_ball
        matrix[8, :, :] = cos_angle_to_ball
        matrix[9, :, :] = sin_angle_to_carrier_velocity
        matrix[10, :, :] = cos_angle_to_carrier_velocity
        matrix[11, :, :] = distance_to_goal
        matrix[12, :, :] = distance_to_ball

        mask = np.zeros((1, self.y_bins, self.x_bins))
        end_ball_coo = np.array([[end_x, end_y]])
        if np.isnan(end_ball_coo).any():
            raise ValueError("End coordinates not known.")
        x0_ball_end, y0_ball_end = self._get_cell_indexes(end_ball_coo[:, 0], end_ball_coo[:, 1])
        mask[0, y0_ball_end, x0_ball_end] = 1

        return (
            torch.from_numpy(matrix).float(),
            torch.from_numpy(mask).float(),
            torch.tensor([target]).float()
        )