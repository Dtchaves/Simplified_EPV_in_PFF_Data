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
    """Normalize tracking coordinates from the acting team's perspective.

    Older code only normalized frames whose frame_id was exactly an event start frame,
    leaving continuous tracking mostly unnormalized and team_phase mostly null. This
    version assigns an acting side to frames covered by event intervals and flips the
    whole frame whenever the acting team is attacking left-to-right only after flipping.
    Velocity and acceleration columns are transformed with the same sign change as
    their coordinate axes; vertical ball components are intentionally left unchanged.
    """
    players_df = players_df.copy()
    if players_df.empty or events_df.empty:
        players_df['team_phase'] = None
        return players_df

    for col in ['match_id', 'frame_id', 'period']:
        if col in players_df.columns:
            players_df[col] = pd.to_numeric(players_df[col], errors='coerce')
    for col in ['match_id', 'start_frame_id', 'end_frame_id', 'frame_id', 'period']:
        if col in events_df.columns:
            events_df[col] = pd.to_numeric(events_df[col], errors='coerce')

    frame_state_rows = []
    unique_frames = players_df[['match_id', 'period', 'frame_id']].drop_duplicates().dropna()
    for _, event in events_df.dropna(subset=['match_id', 'period', 'team_side']).iterrows():
        match_id = event.get('match_id')
        period = event.get('period')
        acting_side = event.get('team_side')
        start_frame = event.get('start_frame_id', event.get('frame_id'))
        end_frame = event.get('end_frame_id', start_frame)
        if pd.isna(start_frame):
            continue
        if pd.isna(end_frame):
            end_frame = start_frame
        lo, hi = sorted((int(start_frame), int(end_frame)))
        frame_subset = unique_frames[
            (unique_frames['match_id'] == match_id)
            & (unique_frames['period'] == period)
            & (unique_frames['frame_id'] >= lo)
            & (unique_frames['frame_id'] <= hi)
        ]
        if frame_subset.empty:
            # Fall back to the event frame only if no interval frames are present.
            frame_subset = unique_frames[
                (unique_frames['match_id'] == match_id)
                & (unique_frames['period'] == period)
                & (unique_frames['frame_id'] == int(start_frame))
            ]
        for _, frame in frame_subset.iterrows():
            frame_state_rows.append({
                'match_id': int(frame['match_id']),
                'period': int(frame['period']),
                'frame_id': int(frame['frame_id']),
                'acting_team_side': acting_side,
            })

    if frame_state_rows:
        frame_state = pd.DataFrame(frame_state_rows).drop_duplicates(
            subset=['match_id', 'period', 'frame_id'], keep='last'
        )
        players_df = players_df.merge(frame_state, on=['match_id', 'period', 'frame_id'], how='left')
    else:
        players_df['acting_team_side'] = None

    acting = players_df['acting_team_side']
    if homeTeamStartLeft:
        mask_flip = ((players_df['period'] == 2) & acting.eq('home')) | ((players_df['period'] == 1) & acting.eq('away'))
    else:
        mask_flip = ((players_df['period'] == 1) & acting.eq('home')) | ((players_df['period'] == 2) & acting.eq('away'))

    flip_cols = [
        col
        for col in [
            'x',
            'y',
            'ball_x',
            'ball_y',
            'vx',
            'vy',
            'ax',
            'ay',
            'ball_vx',
            'ball_vy',
            'ball_ax',
            'ball_ay',
        ]
        if col in players_df.columns
    ]
    if flip_cols:
        players_df.loc[mask_flip, flip_cols] = players_df.loc[mask_flip, flip_cols] * -1

    players_df['team_phase'] = None
    players_df.loc[players_df['team'].eq(players_df['acting_team_side']), 'team_phase'] = 'attacking'
    players_df.loc[players_df['acting_team_side'].notna() & ~players_df['team'].eq(players_df['acting_team_side']), 'team_phase'] = 'defending'

    return players_df.drop(columns=['acting_team_side'])
