import pandas as pd

def parse_names(df: pd.DataFrame):

    df['possession_type'] = df['possession_type'].replace({
        'CH': 'challenge',
        'PA': 'pass',
        'CR': 'pass', # cross as pass
        'SH': 'shot',
        'BC': 'carry',
        'FO': 'foul',
        'IT': 'touch', # initial touch as touch
        'TC': 'touch',
        'RE': 'rebound',
        'CL': 'clearance'
    })

    if 'body_movement_type' in df.columns:
        df['body_movement_type'] = df['body_movement_type'].replace({
            'AG': 'away_from_goal',
            'LA': 'lateral',
            'ST': 'static',
            'TG': 'towards_goal'
        })

    if 'shot_type' in df.columns:
        df['shot_type'] = df['shot_type'].replace({
            'B': 'bycicle',
            'D': 'diving',
            'F': 'side_shot',
            'I': 'sliding',
            'L': 'low',
            'O': 'outside_foot',
            'S': 'shot',
            'V': 'volley'
        })

    if 'shot_outcome' in df.columns:
        df['shot_outcome'] = df['shot_outcome'].replace({
            'B': 'blocked',
            'C': 'blocked',
            'F': 'saved',
            'G': 'goal',
            'L': 'cleared',
            'O': 'off_target',
            'S': 'saved'
        })

    if 'nature_type' in df.columns:
        df['nature_type'] = df['nature_type'].replace({
            'A': 'placement',
            'F': 'flick',
            'P': 'power',
            'T': 'toe_punt'
        })

    if 'ball_height' in df.columns:
        df['ball_height'] = df['ball_height'].replace({
            'A': 'above_head',
            'G': 'ground',
            'H': 'above_waist',
            'L': 'bellow_waist',
            'M': None,
            'V': 'half_volley'
        })

    if 'pass_height' in df.columns:
        df['pass_height'] = df['pass_height'].replace({
            'A': 'above_head',
            'G': 'ground',
            'H': 'above_waist',
            'L': 'bellow_waist',
            'M': None,
            'V': 'half_volley'
        })

    if 'body_part' in df.columns:
        df['body_part'] = df['body_part'].replace({
            'R': 'right_foot',
            'L': 'left_foot',
            'HE': 'head',
            'CH': 'chest'
        })

    # Pass-specific field mappings
    if 'pass_type' in df.columns:
        df['pass_type'] = df['pass_type'].replace({
            'C': 'cross',
            'F': 'forward',
            'B': 'cutback',
            'T': 'through',
            'S': 'standard',
            'H': 'long_throw_in',
            'O': 'high_through_pass',
            'W': 'switch'
        })

    if 'pass_accuracy_type' in df.columns:
        df['pass_accuracy_type'] = df['pass_accuracy_type'].replace({
            'A': 'accurate',
            'I': 'inaccurate',
            'P': 'partially_accurate'
        })

    if 'cross_type' in df.columns:
        df['cross_type'] = df['cross_type'].replace({
            'C': 'cross',
            'I': 'inswing',
            'O': 'outswing',
            'D': 'deep',
            'S': 'short'
        })

    if 'cross_zone_type' in df.columns:
        df['cross_zone_type'] = df['cross_zone_type'].replace({
            'G': 'goal_area',
            'P': 'penalty_area',
            'B': 'box',
            'O': 'outside_box'
        })

    if 'pass_outcome' in df.columns:
        df['pass_outcome'] = df['pass_outcome'].replace({
            'B': 'blocked',
            'C': 'completed',
            'D': 'intercepted',
            'G': 'shot_at_own_goal',
            'I': 'shot_at_goal',
            'S': 'stoppage',
            'O': 'out_of_play'
        })

    if 'cross_outcome' in df.columns:
        df['cross_outcome'] = df['cross_outcome'].replace({
            'B': 'blocked',
            'C': 'completed',
            'D': 'intercepted',
            'G': 'shot_at_own_goal',
            'I': 'shot_at_goal',
            'S': 'stoppage',
            'O': 'out_of_play'
        })

    # Ball carry-specific field mappings
    if 'carry_outcome' in df.columns:
        df['carry_outcome'] = df['carry_outcome'].replace({
            'C': 'challenged',
            'L': 'lost',
            'R': 'retained',
            'S': 'stoppage'
        })

    if 'carry_type' in df.columns:
        df['carry_type'] = df['carry_type'].replace({
            'B': 'line_break',
            'C': 'change_direction',
            'D': 'drive'
        })

    if 'carry_intent' in df.columns:
        df['carry_intent'] = df['carry_intent'].replace({
            'B': 'break_line',
            'C': 'create_space',
            'E': 'escape_pressure'
        })

    if 'dribble_type' in df.columns:
        df['dribble_type'] = df['dribble_type'].replace({
            'B': 'between_defenders',
            'I': 'inside',
            'K': 'knocks_in_front',
            'O': 'outside',
            'T': 'trick'
        })

    if 'pressure_type' in df.columns:
        df['pressure_type'] = df['pressure_type'].replace({
            'A': 'attempted',
            'L': 'passing_lane',
            'N': 'no_pressure',
            'P': 'pressured'
        })

    if 'lines_broken_type' in df.columns:
        df['lines_broken_type'] = df['lines_broken_type'].replace({
            'A': 'attack',
            'AD': 'attack_and_defense',
            'AM': 'attack_and_midfield',
            'AMD': 'attack_and_midfield_and_defense',
            'D': 'defense',
            'M': 'midfield',
            'MD': 'midfield_and_defense'
        })

    if 'cross_zone_type' in df.columns:
        df['cross_zone_type'] = df['cross_zone_type'].replace({
            'C': 'central',
            'N': 'near_post',
            'S': 'six_yard_box',
            'F': 'far_post'
        })

    if 'cross_type' in df.columns:
        df['cross_type'] = df['cross_type'].replace({
            'D': 'drilled',
            'F': 'floated',
            'I': 'inswing',
            'O': 'outswing',
            'P': 'placed'
        })

    if 'set_piece' in df.columns:
        df['set_piece'] = df['set_piece'].replace({
            'P': 'penalty',
            'C': 'corner',
            'F': 'free_kick',
            'T': 'throw_in',
            'R': 'restart',
            'D': 'drop_ball',
            'G': 'goal_kick',
            'O': 'open_play',
            'K': 'kick_off'
        })

    return df

def parse_events(row):
    """
    Parse shooting events from possession event data.

    Args:
        row: Series or dict containing 'possessionEvents' and 'team_id' keys

    Returns:
        Dictionary with parsed shooting event data, or None if no shooting event
    """
    # Convert Series to dict if needed (when called via DataFrame.apply)
    if hasattr(row, 'to_dict'):
        row = row.to_dict()

    if not isinstance(row, dict):
        return None

    # Extract the possession event and team_id from the row
    event_row = row.get('possessionEvents')
    team_id = row.get('team_id')
    match_id = row.get('match_id')
    player_id = row.get('player_id')
    set_piece = row.get('setpieceType')
    video_url = row.get('videoUrl')

    # Check if event_row is valid (not None, NaN, or non-dict)
    if not event_row or not isinstance(event_row, dict):
        return None

    parsed_row = {}

    parsed_row['match_id'] = int(match_id)

    parsed_row['event_id'] = int(event_row['gameEvent'].get('id'))
    parsed_row['possession_id'] = int(event_row['id'])
    parsed_row['possession_type'] = event_row['possessionEventType']
    parsed_row['player_id'] = int(player_id)
    parsed_row['team_id'] = int(team_id)

    parsed_row['set_piece'] = set_piece

    parsed_row['video_url'] = video_url

    if event_row.get('shootingEvent'):
        # handle shootingEvent
        shooting_event = event_row['shootingEvent']

        # Shot characteristics
        parsed_row['body_movement_type'] = shooting_event.get('bodyMovementType', None)
        parsed_row['ball_moving'] = shooting_event.get('ballMoving', None)
        parsed_row['shot_type'] = shooting_event.get('shotType', None)
        parsed_row['body_part'] = shooting_event.get('shotBodyType', None)
        parsed_row['shot_outcome'] = shooting_event.get('shotOutcomeType', None)
        parsed_row['nature_type'] = shooting_event.get('shotNatureType', None)
        parsed_row['ball_height'] = shooting_event.get('ballHeightType', None)

        parsed_row['pressure_type'] = shooting_event.get('pressureType', None)

    if event_row.get('passingEvent'):
        # Handle passing events
        passing_event = event_row['passingEvent']

        # Pass characteristics
        parsed_row['ball_moving'] = passing_event.get('ballMoving', None)
        parsed_row['pass_type'] = passing_event.get('passType', None)
        parsed_row['body_part'] = passing_event.get('passBodyType', None)
        parsed_row['pass_outcome'] = passing_event.get('passOutcomeType', None)
        parsed_row['ball_height'] = passing_event.get('ballHeightType', None)
        parsed_row['pass_height'] = passing_event.get('receiverHeightType', None)

        # Receiver information
        receiver_player = passing_event.get('receiverPlayer')
        if receiver_player and receiver_player.get('id') is not None:
            parsed_row['receiver_player_id'] = int(receiver_player.get('id'))
        else:
            parsed_row['receiver_player_id'] = None

        target_player = passing_event.get('targetPlayer')
        if target_player and target_player.get('id') is not None:
            parsed_row['target_player_id'] = int(target_player.get('id'))
        else:
            parsed_row['target_player_id'] = None

        # Additional pass characteristics
        parsed_row['no_look'] = passing_event.get('noLook', None)
        parsed_row['creates_space'] = passing_event.get('createsSpace', None)
        parsed_row['pressure_type'] = passing_event.get('pressureType', None)
        parsed_row['pass_accuracy_type'] = passing_event.get('passAccuracyType', None)
        parsed_row['lines_broken_type'] = passing_event.get('linesBrokenType', None)

    if event_row.get('crossEvent'):
        # Handle crossing events (unified as pass events with cross type)
        cross_event = event_row['crossEvent']

        # Cross characteristics (unified as pass with cross type)
        parsed_row['ball_moving'] = cross_event.get('ballMoving', None)
        parsed_row['pass_type'] = 'cross'  # Always 'cross' for cross events
        parsed_row['body_part'] = cross_event.get('crosserBodyType', None)
        parsed_row['cross_outcome'] = cross_event.get('crossOutcomeType', None)
        parsed_row['ball_height'] = cross_event.get('ballHeightType', None)
        parsed_row['pass_height'] = cross_event.get('receiverHeightType', None)


        # Receiver information
        receiver_player = cross_event.get('receiverPlayer')
        if receiver_player and receiver_player.get('id') is not None:
            parsed_row['receiver_player_id'] = int(receiver_player.get('id'))
        else:
            parsed_row['receiver_player_id'] = None

        parsed_row['cross_zone_type'] = cross_event.get('crossZoneType', None)
        parsed_row['cross_type'] = cross_event.get('crossType', None)

        # Additional cross characteristics
        parsed_row['no_look'] = cross_event.get('noLook', None)
        parsed_row['creates_space'] = cross_event.get('createsSpace', None)
        parsed_row['pressure_type'] = cross_event.get('pressureType', None)
        parsed_row['pass_accuracy_type'] = cross_event.get('crossAccuracyType', None)
        parsed_row['lines_broken_type'] = cross_event.get('linesBrokenType', None)

    if event_row.get('ballCarryEvent'):
        # Handle ball carry events
        ball_carry_event = event_row['ballCarryEvent']

        # Ball carry characteristics
        parsed_row['carry_outcome'] = ball_carry_event.get('carryOutcome', None)
        parsed_row['carry_intent'] = ball_carry_event.get('carryIntent', None)
        parsed_row['carry_type'] = ball_carry_event.get('carryType', None)
        parsed_row['carry_success'] = ball_carry_event.get('carrySuccess', None)

        parsed_row['creates_space'] = ball_carry_event.get('createsSpace', None)
        parsed_row['pressure_type'] = ball_carry_event.get('pressureType', None)

        parsed_row['dribble_type'] = ball_carry_event.get('dribbleType', None)
        parsed_row['dribble_outcome'] = ball_carry_event.get('dribbleOutcome', None)

        parsed_row['ball_height'] = ball_carry_event.get('ballHeightType', None)


    # No shooting, passing, or ball carry event found
    return parsed_row