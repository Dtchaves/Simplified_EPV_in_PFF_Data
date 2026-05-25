from typing import Any

import pandas as pd
from pandas import json_normalize, merge
from gandula.providers.pff.schema.tracking import PFF_Frame
from gandula.export.dataframe import _build_metadata_df, _build_df

def _extract_players_ball_from_frames(
    frames: list[PFF_Frame], *, smoothed: bool = False
) -> list[dict[str, Any]]:
    """Extract player and ball data from tracking frames.
    
    Args:
        frames: List of PFF_Frame objects
        smoothed: Whether to use smoothed (kalman) data or raw data
        
    Returns:
        List of dictionaries containing frame data with players and ball information
    """
    # Define keys based on whether we want smoothed or raw data
    ball_key = 'ball_with_kalman' if smoothed else 'ball'
    home_key = 'home_players_with_kalman' if smoothed else 'home_players'
    away_key = 'away_players_with_kalman' if smoothed else 'away_players'
    
    # Fallback keys for when smoothed data is not available
    fallback_ball_key = 'ball'
    fallback_home_key = 'home_players'
    fallback_away_key = 'away_players'

    # Options for serializing player and ball data
    player_dump_options = {'exclude': {'shirt_confidence', 'visibility'}}
    ball_dump_options = {'exclude': {'visibility'}}

    extracted_frames = []

    for frame in frames:
        # Get player data with fallback to non-smoothed data if needed
        home_players = getattr(frame, home_key, None) or getattr(frame, fallback_home_key, []) or []
        away_players = getattr(frame, away_key, None) or getattr(frame, fallback_away_key, []) or []

        # Process home players
        home = [
            {**player.model_dump(**player_dump_options), 'team': 'home'}
            for player in home_players
        ]
        
        # Process away players
        away = [
            {**player.model_dump(**player_dump_options), 'team': 'away'}
            for player in away_players
        ]

        # Get ball data with fallback to non-smoothed data if needed
        ball = getattr(frame, ball_key, None) or getattr(frame, fallback_ball_key, None)
        
        # Process ball data
        if ball:
            # Handle case where ball might be a list
            if isinstance(ball, list):
                ball = ball[0] if ball else None
            
            if ball:  # Double check ball is not None after list processing
                ball_data = [ball.model_dump(**ball_dump_options)]
            else:
                ball_data = []
        else:
            ball_data = []

        # Combine all data for this frame
        players = home + away
        frame_id = frame.frame_id

        extracted_frames.append({
            'frame_id': frame_id, 
            'players': players, 
            'ball': ball_data
        })

    return extracted_frames

def _build_players_ball_df(
    frames: list[PFF_Frame], *, smoothed: bool = False
) -> pd.DataFrame:
    dump_options = {'include': {'match_id', 'frame_id', 'period', 'elapsed_seconds'}}

    frame_df = _build_df(frames, dump_options)

    coordinates = _extract_players_ball_from_frames(frames, smoothed=smoothed)
    players_df = json_normalize(
        data=coordinates,
        record_path='players',
        meta=['frame_id'],
        sep='_',
    )
    ball_df = json_normalize(
        data=coordinates,
        record_path='ball',
        record_prefix='ball_',
        meta=['frame_id'],
        sep='_',
    )

    players_ball_df = merge(players_df, ball_df, on='frame_id')
    return frame_df.merge(players_ball_df, on='frame_id')

def pff_frames_to_dataframe(frames, smoothed=True, **kwargs):
    """Patched version that supports smoothed parameter."""
    pitch_size = None
    pitch_center = None
    
    # Handle GandulaFrame objects (same logic as original)
    if hasattr(frames[0], 'frame'):  # GandulaFrame
        pitch_size = frames[0].pitch_size
        pitch_center = frames[0].pitch_center
        frames = [frame.frame for frame in frames]
    
    # Build metadata (same as original)
    metadata_df = _build_metadata_df(frames)
    
    # Build players_ball_df with smoothed parameter
    players_ball_df = _build_players_ball_df(frames, smoothed=smoothed)
    
    # Add pitch info if available (same as original)
    if pitch_size is not None and pitch_center is not None:
        metadata_df['pitch_size'] = [pitch_size] * len(metadata_df)
        metadata_df['pitch_center'] = [pitch_center] * len(metadata_df)
    
    return metadata_df, players_ball_df

def change_events_side(players_df: pd.DataFrame, events_df: pd.DataFrame, homeTeamStartLeft: bool):

    home_events = events_df[events_df['team_side'] == 'home']['frame_id'].unique()
    away_events = events_df[events_df['team_side'] == 'away']['frame_id'].unique()

    if homeTeamStartLeft:
        mask_flip = (
            ((players_df['period'] == 2) & (players_df['frame_id'].isin(home_events))) |
            ((players_df['period'] == 1) & (players_df['frame_id'].isin(away_events)))
        )
    else:
        mask_flip = (
            ((players_df['period'] == 1) & (players_df['frame_id'].isin(home_events))) |
            ((players_df['period'] == 2) & (players_df['frame_id'].isin(away_events)))
        )
    
    players_df.loc[mask_flip, ['x', 'y', 'ball_x', 'ball_y']] *= -1

    players_df['team_phase'] = None
    players_df.loc[(players_df['team']=='home') & (players_df['frame_id'].isin(home_events)), 'team_phase'] = 'attacking'
    players_df.loc[(players_df['team']=='away') & (players_df['frame_id'].isin(away_events)), 'team_phase'] = 'attacking'
    players_df.loc[(players_df['team']=='home') & (players_df['frame_id'].isin(away_events)), 'team_phase'] = 'defending'
    players_df.loc[(players_df['team']=='away') & (players_df['frame_id'].isin(home_events)), 'team_phase'] = 'defending'

    return players_df
