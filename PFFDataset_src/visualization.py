"""Visualization utilities for xG data using mplsoccer."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mplsoccer import Pitch
from typing import Optional, List, Tuple


def plot_shot_frame(
    tracking_df: pd.DataFrame,
    event_row: pd.Series,
    figsize: Tuple[int, int] = (12, 8),
    pitch_style: str = 'impect',
    show_players: bool = True,
    show_ball: bool = True,
    show_velocity: bool = False,
    team_colors: Optional[dict] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot a single shot frame showing player positions and ball location.
    
    Args:
        tracking_df: DataFrame with tracking data for the frame
        event_row: Series containing shot event information
        figsize: Figure size (width, height)
        pitch_style: Style of the pitch ('impect', 'statsbomb', 'uefa', 'wyscout', 'opta')
        show_players: Whether to show player positions
        show_ball: Whether to show ball position
        show_velocity: Whether to show velocity vectors
        team_colors: Dict with team colors {'attacking': 'blue', 'defending': 'red'}
        title: Optional title for the plot
        
    Returns:
        matplotlib Figure object
    """
    # Create pitch using Impect coordinate system (matches your data: -52.5 to 52.5, -34 to 34)
    pitch = Pitch(pitch_type=pitch_style, line_zorder=2)
    fig, ax = pitch.draw(figsize=figsize)
    
    # Get frame data
    frame_id = event_row['frame_id']
    match_id = event_row['match_id']
    frame_data = tracking_df[(tracking_df['frame_id'] == frame_id) & (tracking_df['match_id'] == match_id)]
    
    if frame_data.empty:
        ax.text(0.5, 0.5, 'No tracking data for this frame', 
                transform=ax.transAxes, ha='center', va='center')
        return fig
    
    # Default team colors
    if team_colors is None:
        team_colors = {'attacking': '#1f77b4', 'defending': '#ff7f0e'}
    
    shooter_id = event_row.get('player_id')

    # Plot players
    if show_players:
        # Home team
        attacking_players = frame_data[frame_data['team_phase'] == 'attacking']
        if not attacking_players.empty:

            ax.scatter(attacking_players['x'], attacking_players['y'], 
                      c=team_colors['attacking'], s=150, alpha=0.7, 
                      edgecolors='black', linewidth=0.5, label='Attacking')
            
            # Add player numbers
            for _, player in attacking_players.iterrows():
                if player.get('player_id') == shooter_id:
                    ax.scatter(player['x'], player['y'], 
                              c=team_colors['attacking'], s=150, alpha=1, 
                              edgecolors='black', linewidth=1, label='Shooter')
                if pd.notna(player.get('shirt_number')):
                    ax.text(player['x'], player['y'], str(int(player['shirt_number'])), 
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           color='white' if team_colors['attacking'] != 'white' else 'black')
        
        # Away team
        defending_players = frame_data[frame_data['team_phase'] == 'defending']
        if not defending_players.empty:
            ax.scatter(defending_players['x'], defending_players['y'], 
                      c=team_colors['defending'], s=150, alpha=0.7, 
                      edgecolors='black', linewidth=0.5, label='Defending')
            
            # Add player numbers
            for _, player in defending_players.iterrows():
                if pd.notna(player.get('shirt_number')):
                    ax.text(player['x'], player['y'], str(int(player['shirt_number'])), 
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           color='white' if team_colors['defending'] != 'white' else 'black')
    
    # Plot ball
    if show_ball:
        ball_data = frame_data[frame_data['ball_x'].notna()]
        if not ball_data.empty:
            ax.scatter(ball_data['ball_x'].iloc[0], ball_data['ball_y'].iloc[0], 
                      c='black', s=50, edgecolors='black', linewidth=1, 
                      marker='o', label='Ball', zorder=5)
    
    # Plot velocity vectors
    if show_velocity and 'vx' in frame_data.columns and 'vy' in frame_data.columns:
        # Home team velocity
        attacking_players = frame_data[frame_data['team_phase'] == 'attacking']
        if not attacking_players.empty:
            ax.quiver(attacking_players['x'], attacking_players['y'], 
                     attacking_players['vx'], attacking_players['vy'], 
                     color=team_colors['attacking'], alpha=0.6, scale=20, width=0.003)
        
        # Away team velocity
        defending_players = frame_data[frame_data['team_phase'] == 'defending']
        if not defending_players.empty:
            ax.quiver(defending_players['x'], defending_players['y'], 
                     defending_players['vx'], defending_players['vy'], 
                     color=team_colors['defending'], alpha=0.6, scale=20, width=0.003)
    
    
    # Add legend
    if show_players or show_ball:
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
    
    plt.tight_layout()
    return fig


def plot_shot_heatmap(
    events_df: pd.DataFrame,
    pitch_style: str = 'statsbomb',
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create a heatmap of shot locations.
    
    Args:
        events_df: DataFrame with shot events
        pitch_style: Style of the pitch
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    # Filter shots
    shots = events_df[events_df['possession_type'] == 'shot'].copy()
    
    if shots.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No shots found', transform=ax.transAxes, 
                ha='center', va='center')
        return fig
    
    # Create pitch using Impect coordinate system
    pitch = Pitch(pitch_type='impect', line_zorder=2)
    fig, ax = pitch.draw(figsize=figsize)
    
    # Create hexbin plot
    hb = ax.hexbin(shots['x'], shots['y'], gridsize=20, alpha=0.7, cmap='Reds')
    
    # Add colorbar
    plt.colorbar(hb, ax=ax, label='Number of shots')
    
    # Add shot points
    ax.scatter(shots['x'], shots['y'], c='white', s=30, alpha=0.8, 
               edgecolors='black', linewidth=0.5)
    
    # Highlight goals
    goals = shots[shots['outcome'] == 'goal']
    if not goals.empty:
        ax.scatter(goals['x'], goals['y'], c='gold', s=100, marker='*', 
                   edgecolors='black', linewidth=1, label='Goals', zorder=5)
    
    ax.set_title('Shot Location Heatmap', fontsize=14, fontweight='bold')
    
    if not goals.empty:
        ax.legend()

    plt.tight_layout()
    return fig


def plot_tensorized_frames(
    df: pd.DataFrame,
    frame1: int,
    frame2: Optional[int] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    x_bins: Optional[List[float]] = None,
    y_bins: Optional[List[float]] = None,
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
    input_range: Tuple[Tuple[float, float], Tuple[float, float]] = ((-50, 50), (-35, 35)),
    figsize_width: float = 5.0
) -> plt.Figure:
    """Plot one or two soccer frames with player positions and grid overlay.

    Args:
        df: DataFrame containing frame data with columns: frame_id, atkPlayers, defPlayers, balls
        frame1: Frame number for the first frame
        frame2: Frame number for the second frame (optional)
        grid_shape: Tuple of (rows, cols) for uniform grid. If None, uses default zones.
        x_bins: Custom x-axis bin boundaries
        y_bins: Custom y-axis bin boundaries
        pitch_length: Length of pitch (default: 120)
        pitch_width: Width of pitch (default: 80)
        input_range: Input coordinate ranges ((x_min, x_max), (y_min, y_max))
        figsize_width: Width of each subplot

    Returns:
        matplotlib Figure object
    """
    # Import here to avoid circular imports
    from src.tensorizer import get_default_bins, get_uniform_bins

    # Determine bins for grid overlay
    if grid_shape is not None:
        x_bins_plot, y_bins_plot = get_uniform_bins(grid_shape, pitch_length, pitch_width)
    elif x_bins is not None and y_bins is not None:
        x_bins_plot, y_bins_plot = x_bins, y_bins
    else:
        x_bins_plot, y_bins_plot = get_default_bins(pitch_length, pitch_width)

    # Calculate coordinate transformation parameters
    x_in_min, x_in_max = input_range[0]
    y_in_min, y_in_max = input_range[1]
    x_scale = pitch_length / (x_in_max - x_in_min)
    y_scale = pitch_width / (y_in_max - y_in_min)

    # Select frames
    frames = [frame1] if frame2 is None else [frame1, frame2]
    frames_data = [df[df['frame_id'] == f].iloc[0] for f in frames]

    # Prepare figure
    n_cols = len(frames)
    pitch = Pitch(pitch_type='statsbomb', goal_type='box', linewidth=2, line_color='black')
    depth = 1.54 * figsize_width
    fig, axes = pitch.draw(nrows=1, ncols=n_cols, figsize=(depth * n_cols, figsize_width))

    if n_cols == 1:
        axes = [axes]  # Ensure axes is iterable

    for ax, frame, num in zip(axes, frames_data, frames):
        # Extract and transform coordinates
        home_x = [x_scale * (p['x'] - x_in_min) for p in frame.atkPlayers]
        home_y = [y_scale * (p['y'] - y_in_min) for p in frame.atkPlayers]
        home_jerseys = [p['jerseyNum'] for p in frame.atkPlayers]

        away_x = [x_scale * (p['x'] - x_in_min) for p in frame.defPlayers]
        away_y = [y_scale * (p['y'] - y_in_min) for p in frame.defPlayers]
        away_jerseys = [p['jerseyNum'] for p in frame.defPlayers]

        ball_x = [x_scale * (frame.balls[0]['x'] - x_in_min)]
        ball_y = [y_scale * (frame.balls[0]['y'] - y_in_min)]

        # Plot elements
        pitch.scatter(home_x, home_y, ax=ax, color='blue', s=100, label='Home Players')
        pitch.scatter(away_x, away_y, ax=ax, color='red', s=100, label='Away Players')
        pitch.scatter(ball_x, ball_y, ax=ax, color='k', s=120, label='Ball', alpha=0.5)

        # Annotate jerseys
        for j, x, y in zip(home_jerseys, home_x, home_y):
            pitch.annotate(j, (x, y), ax=ax, color='k', ha='center', va='bottom', fontsize=9)
        for j, x, y in zip(away_jerseys, away_x, away_y):
            pitch.annotate(j, (x, y), ax=ax, color='k', ha='center', va='bottom', fontsize=9)

        # Grid lines based on bins
        for y_line in y_bins_plot[1:-1]:  # Skip first and last (pitch boundaries)
            ax.axhline(y_line, color='grey', linestyle='--', linewidth=1)
        for x_line in x_bins_plot[1:-1]:  # Skip first and last (pitch boundaries)
            ax.axvline(x_line, color='grey', linestyle='--', linewidth=1)

        # Titles
        ax.set_title(f'Frame {num}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')

    plt.tight_layout()
    return fig
