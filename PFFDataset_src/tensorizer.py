"""Tensorizer utilities for converting tracking data to spatial representations."""

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd


def filter_frames_with_players(
    df: pd.DataFrame,
    n_players: int = 22
) -> pd.DataFrame:
    """Filter frames that have exactly n_players.

    Args:
        df: DataFrame with tracking data containing match_id, frame_id columns
        n_players: Number of players required per frame (default: 22)

    Returns:
        Filtered DataFrame containing only frames with exactly n_players
    """
    return df.groupby(['match_id', 'frame_id']).filter(lambda x: len(x) == n_players)


def aggregate_frame(group: pd.DataFrame) -> pd.Series:
    """Aggregate player and ball data for a single frame.

    Args:
        group: DataFrame group for a single frame

    Returns:
        Series with aggregated frame data including balls, atkPlayers, defPlayers

    Note:
        When used with groupby.apply(include_groups=False), match_id and frame_id
        will be restored from the group index.
    """
    frame_data = {}

    # Get ball data (assuming it's the same for all players in the frame)
    frame_data['balls'] = [{'x': group['ball_x'].iloc[0], 'y': group['ball_y'].iloc[0]}]

    # Separate attacking and defending players
    atk_players = group[group['team_phase'] == 'attacking']
    def_players = group[group['team_phase'] == 'defending']

    # Create lists of dictionaries for attacking and defending players
    frame_data['atkPlayers'] = [
        {'jerseyNum': row['shirt'], 'x': row['x'], 'y': row['y']}
        for _, row in atk_players.iterrows()
    ]

    frame_data['defPlayers'] = [
        {'jerseyNum': row['shirt'], 'x': row['x'], 'y': row['y']}
        for _, row in def_players.iterrows()
    ]

    return pd.Series(frame_data)


def get_default_bins(
    pitch_length: float = 120.0,
    pitch_width: float = 80.0
) -> Tuple[List[float], List[float]]:
    """Get default soccer zone bins (5x6 grid).

    Args:
        pitch_length: Length of pitch (default: 120 for StatsBomb)
        pitch_width: Width of pitch (default: 80 for StatsBomb)

    Returns:
        Tuple of (x_bins, y_bins) for the 5x6 default grid
    """
    # Default zones based on standard soccer pitch divisions
    x_bins = [0, 18, (60+18)/2, pitch_length/2, pitch_length - (60+18)/2, pitch_length-18, pitch_length]
    y_bins = [0, 18, 30, pitch_width-30, pitch_width-18, pitch_width]
    return x_bins, y_bins


def get_uniform_bins(
    grid_shape: Tuple[int, int],
    pitch_length: float = 120.0,
    pitch_width: float = 80.0
) -> Tuple[List[float], List[float]]:
    """Generate uniform grid bins.

    Args:
        grid_shape: Tuple of (rows, cols) for the grid
        pitch_length: Length of pitch
        pitch_width: Width of pitch

    Returns:
        Tuple of (x_bins, y_bins) for uniform grid division
    """
    rows, cols = grid_shape
    x_bins = np.linspace(0, pitch_length, cols + 1).tolist()
    y_bins = np.linspace(0, pitch_width, rows + 1).tolist()
    return x_bins, y_bins


def tensorize(
    frame: pd.Series,
    grid_shape: Optional[Tuple[int, int]] = None,
    x_bins: Optional[List[float]] = None,
    y_bins: Optional[List[float]] = None,
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
    input_range: Tuple[Tuple[float, float], Tuple[float, float]] = ((-50, 50), (-35, 35))
) -> np.ndarray:
    """Convert frame to tensor representation with region counts.

    The tensor has 3 channels:
    - Channel 0: Attacking team player counts
    - Channel 1: Defending team player counts
    - Channel 2: Ball location

    Args:
        frame: Series containing atkPlayers, defPlayers, and balls data
        grid_shape: Tuple of (rows, cols) for uniform grid division.
            If provided, overrides x_bins/y_bins with uniform grid.
        x_bins: Custom x-axis bin boundaries. If None, uses default zones.
        y_bins: Custom y-axis bin boundaries. If None, uses default zones.
        pitch_length: Length of the output pitch coordinates (default: 120)
        pitch_width: Width of the output pitch coordinates (default: 80)
        input_range: Input coordinate ranges ((x_min, x_max), (y_min, y_max))
            Default: ((-50, 50), (-35, 35)) for the raw data format

    Returns:
        3D numpy array of shape (3, n_rows, n_cols) with player/ball counts per region

    Examples:
        # Default 5x6 grid with standard zones
        tensor = tensorize(frame)

        # Custom uniform 8x10 grid
        tensor = tensorize(frame, grid_shape=(8, 10))

        # Custom bins
        tensor = tensorize(frame, x_bins=[0, 40, 80, 120], y_bins=[0, 40, 80])
    """
    # Calculate coordinate transformation parameters
    x_in_min, x_in_max = input_range[0]
    y_in_min, y_in_max = input_range[1]

    x_scale = pitch_length / (x_in_max - x_in_min)
    y_scale = pitch_width / (y_in_max - y_in_min)

    # Transform coordinates to pitch coordinates
    home_x = np.array([x_scale * (p['x'] - x_in_min) for p in frame['atkPlayers']])
    home_y = np.array([y_scale * (p['y'] - y_in_min) for p in frame['atkPlayers']])

    away_x = np.array([x_scale * (p['x'] - x_in_min) for p in frame['defPlayers']])
    away_y = np.array([y_scale * (p['y'] - y_in_min) for p in frame['defPlayers']])

    ball_x = np.array([x_scale * (frame['balls'][0]['x'] - x_in_min)])
    ball_y = np.array([y_scale * (frame['balls'][0]['y'] - y_in_min)])

    # Determine bins
    if grid_shape is not None:
        x_bins, y_bins = get_uniform_bins(grid_shape, pitch_length, pitch_width)
    elif x_bins is None or y_bins is None:
        x_bins, y_bins = get_default_bins(pitch_length, pitch_width)

    n_rows = len(y_bins) - 1
    n_cols = len(x_bins) - 1

    # Count players per region
    home_region_counts = np.zeros((n_rows, n_cols), dtype=int)
    away_region_counts = np.zeros((n_rows, n_cols), dtype=int)
    ball_region_counts = np.zeros((n_rows, n_cols), dtype=int)

    for i in range(n_cols):
        for j in range(n_rows):
            x_min, x_max = x_bins[i], x_bins[i+1]
            y_min, y_max = y_bins[j], y_bins[j+1]

            count_home = np.sum(
                (home_x >= x_min) & (home_x < x_max) &
                (home_y >= y_min) & (home_y < y_max)
            )
            home_region_counts[j, i] = count_home

            count_away = np.sum(
                (away_x >= x_min) & (away_x < x_max) &
                (away_y >= y_min) & (away_y < y_max)
            )
            away_region_counts[j, i] = count_away

            count_ball = np.sum(
                (ball_x >= x_min) & (ball_x < x_max) &
                (ball_y >= y_min) & (ball_y < y_max)
            )
            ball_region_counts[j, i] = count_ball

    tensor = np.stack([home_region_counts, away_region_counts, ball_region_counts])
    return tensor


def process_tracking_to_tensors(
    df: pd.DataFrame,
    grid_shape: Optional[Tuple[int, int]] = None,
    x_bins: Optional[List[float]] = None,
    y_bins: Optional[List[float]] = None,
    n_players: int = 22
) -> pd.DataFrame:
    """Process tracking DataFrame to aggregated frames with tensors.

    Args:
        df: Raw tracking DataFrame
        grid_shape: Tuple of (rows, cols) for uniform grid
        x_bins: Custom x-axis bin boundaries
        y_bins: Custom y-axis bin boundaries
        n_players: Number of players required per frame

    Returns:
        DataFrame with aggregated frames and tensor representations
    """
    # Filter frames
    df_filtered = filter_frames_with_players(df, n_players)

    # Aggregate frames
    df_agg = df_filtered.groupby(
        ['match_id', 'frame_id']
    ).apply(aggregate_frame, include_groups=True).reset_index()

    print(df_agg.columns)

    # Create tensors
    df_agg['tensor'] = df_agg.apply(
        lambda row: tensorize(row, grid_shape=grid_shape, x_bins=x_bins, y_bins=y_bins),
        axis=1
    )

    return df_agg
