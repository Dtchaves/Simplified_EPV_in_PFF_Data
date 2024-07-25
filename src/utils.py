import pandas as pd
import numpy as np
import torch


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

    def _get_cell_indexes(self, x, y):
        x_bin = np.clip((x + 52.5) / 105 * self.x_bins, 0, self.x_bins - 1).astype(np.uint8)
        y_bin = np.clip((y + 34) / 68 * self.y_bins, 0, self.y_bins - 1).astype(np.uint8)
        return x_bin, y_bin

    def __call__(self, sample):
        pass_outcome_mapping = {
            'C': 0,
            'D': 1,
            'B': 2,
            'O': 3,
            'S': 4
        }
        start_x, start_y, end_x, end_y = (
            sample["ball_x_start"],
            sample["ball_y_start"],
            sample["ball_x_end"],
            sample["ball_y_end"],
        )
        speed_x, speed_y = sample["vx_player_201"], sample["vy_player_201"]
        frame = sample["frame"]

        pass_outcome_type = sample.get("pass_outcome_type")
        pass_outcome_type_mapped = pass_outcome_mapping.get(pass_outcome_type, None)
        if pass_outcome_type_mapped is None:
            raise ValueError(f"Invalid pass_outcome_type: {pass_outcome_type}")
        target = int(pass_outcome_type_mapped)

        ball_coo = np.array([[start_x, start_y]])
        goal_coo = np.array([[52.5, 0]])

        players_att_coo = frame.loc[frame["team_id"] == sample["team_id"], ["x_player_201", "y_player_201"]].values.reshape(-1, 2)
        players_def_coo = frame.loc[frame["team_id"] != sample["team_id"], ["x_player_201", "y_player_201"]].values.reshape(-1, 2)

        matrix = np.zeros((15, self.y_bins, self.x_bins))

        # Channel 1: Locations of attacking team
        x_bin_att, y_bin_att = self._get_cell_indexes(players_att_coo[:, 0], players_att_coo[:, 1])
        matrix[0, y_bin_att, x_bin_att] = 1

        # Channel 2: Locations of defending team
        x_bin_def, y_bin_def = self._get_cell_indexes(players_def_coo[:, 0], players_def_coo[:, 1])
        matrix[1, y_bin_def, x_bin_def] = 1

        # Channel 3: Distance to ball
        yy, xx = np.ogrid[0.5: self.y_bins, 0.5: self.x_bins]
        x0_ball, y0_ball = self._get_cell_indexes(ball_coo[:, 0], ball_coo[:, 1])
        matrix[2, :, :] = np.sqrt((xx - x0_ball) * 2 + (yy - y0_ball) * 2)

        # Channel 4: Distance to goal
        x0_goal, y0_goal = self._get_cell_indexes(goal_coo[:, 0], goal_coo[:, 1])
        matrix[3, :, :] = np.sqrt((xx - x0_goal) * 2 + (yy - y0_goal) * 2)

        # Channel 5: Cosine of the angle between the ball and goal
        coords = np.dstack(np.meshgrid(xx, yy))
        goal_coo_bin = np.concatenate((x0_goal, y0_goal))
        ball_coo_bin = np.concatenate((x0_ball, y0_ball))
        a = goal_coo_bin - coords
        b = ball_coo_bin - coords
        matrix[4, :, :] = np.clip(np.sum(a * b, axis=2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)), -1, 1)

        # Channel 6: Sine of the angle between the ball and goal
        matrix[5, :, :] = np.sqrt(1 - matrix[4, :, :] ** 2)

        # Channel 7: Angle to the goal location
        matrix[6, :, :] = np.abs(np.arctan2((y0_goal - coords[:, :, 1]), (x0_goal - coords[:, :, 0])))

        # Channels 8-9: Ball speed
        matrix[7, y0_ball, x0_ball] = speed_x
        matrix[8, y0_ball, x0_ball] = speed_y

        # Channel 10: Number of possession team’s players between the ball and every other location
        dist_att_goal = matrix[0, :, :] * matrix[3, :, :]
        dist_att_goal[dist_att_goal == 0] = np.nan
        dist_ball_goal = matrix[3, y0_ball, x0_ball]
        player_in_front_of_ball = dist_att_goal <= dist_ball_goal

        outplayed1 = lambda x: np.sum(player_in_front_of_ball & (x <= dist_ball_goal) & (dist_att_goal >= x))
        matrix[9, :, :] = np.vectorize(outplayed1)(matrix[3, :, :])

        # Channel 11: Number of opponent team’s players between the ball and every other location
        dist_def_goal = matrix[1, :, :] * matrix[3, :, :]
        dist_def_goal[dist_def_goal == 0] = np.nan
        dist_ball_goal = matrix[3, y0_ball, x0_ball]
        player_in_front_of_ball = dist_def_goal <= dist_ball_goal

        outplayed2 = lambda x: np.sum(player_in_front_of_ball & (x <= dist_ball_goal) & (dist_def_goal >= x))
        matrix[10, :, :] = np.vectorize(outplayed2)(matrix[3, :, :])

        # Channel 12: Carrier's velocity
        carrier_velocity = sample["carrier_velocity"]
        matrix[11, y0_ball, x0_ball] = carrier_velocity

        # Channels 13-14: Possession team players' velocity (x and y)
        players_vel_coo = frame.loc[frame["team_id"] == sample["team_id"], ["vx_player_201", "vy_player_201"]].values
        x_bin_att_vel, y_bin_att_vel = self._get_cell_indexes(players_att_coo[:, 0], players_att_coo[:, 1])
        matrix[12, y_bin_att_vel, x_bin_att_vel] = players_vel_coo[:, 0]
        matrix[13, y_bin_att_vel, x_bin_att_vel] = players_vel_coo[:, 1]

        # Channel 15: Distance to event's origin location
        x0_start, y0_start = self._get_cell_indexes(np.array([start_x]), np.array([start_y]))
        matrix[14, :, :] = np.sqrt((xx - x0_start) * 2 + (yy - y0_start) * 2)

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