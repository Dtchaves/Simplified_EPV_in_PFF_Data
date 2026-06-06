import pandas as pd

NULL_EQUIVALENT_TOKENS = {"", "none", "null", "nan", "na", "n/a", "nat", "<na>"}


def _clean_null_tokens(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(
                lambda value: None
                if value is None or (isinstance(value, str) and value.strip().lower() in NULL_EQUIVALENT_TOKENS)
                else value
            )
    return df


def parse_names(df: pd.DataFrame):
    """Map Gradient/PFF coded event values to readable canonical tokens.

    Crosses are intentionally mapped to possession_type='pass' because the pass models
    treat passes and crosses as distributions to a teammate. The original possession
    event type remains recoverable only upstream, so cross-specific columns are kept.
    """
    df = _clean_null_tokens(df)

    if 'possession_type' in df.columns:
        df['possession_type'] = df['possession_type'].replace({
            'CH': 'challenge',
            'PA': 'pass',
            'CR': 'pass',
            'SH': 'shot',
            'BC': 'carry',
            'FO': 'foul',
            'IT': 'touch',
            'TC': 'touch',
            'RE': 'rebound',
            'CL': 'clearance',
        })

    if 'body_movement_type' in df.columns:
        df['body_movement_type'] = df['body_movement_type'].replace({
            'AG': 'away_from_goal',
            'LA': 'lateral',
            'ST': 'static',
            'TG': 'towards_goal',
        })

    if 'shot_type' in df.columns:
        df['shot_type'] = df['shot_type'].replace({
            'B': 'bicycle',
            'D': 'diving',
            'F': 'side_foot',
            'I': 'sliding',
            'L': 'lob',
            'O': 'outside_foot',
            'V': 'volley',
        })

    if 'shot_outcome' in df.columns:
        df['shot_outcome'] = df['shot_outcome'].replace({
            'B': 'blocked_on_target',
            'C': 'blocked_off_target',
            'F': 'saved_off_target',
            'G': 'goal',
            'L': 'goalline_clearance',
            'O': 'off_target',
            'S': 'saved_on_target',
        })

    if 'nature_type' in df.columns:
        df['nature_type'] = df['nature_type'].replace({
            'A': 'placement',
            'F': 'flick',
            'P': 'power',
            'T': 'toe_punt',
        })

    height_map = {
        'A': 'above_head',
        'G': 'ground',
        'H': 'between_waist_and_head',
        'L': 'off_ground_below_waist',
        'M': None,
        'V': 'half_volley',
    }
    for column in ['ball_height', 'pass_height']:
        if column in df.columns:
            df[column] = df[column].replace(height_map)

    if 'body_part' in df.columns:
        df['body_part'] = df['body_part'].replace({
            'RF': 'right_foot', 'LF': 'left_foot',
            'RB': 'right_back_heel', 'LB': 'left_back_heel',
            'HE': 'head', 'CH': 'chest',
            'RA': 'right_arm', 'LA': 'left_arm',
            'RH': 'right_hand', 'LH': 'left_hand',
            'RK': 'right_knee', 'LK': 'left_knee',
            'RS': 'right_shin', 'LS': 'left_shin',
            'RT': 'right_thigh', 'LT': 'left_thigh',
            'RC': 'right_shoulder', 'LC': 'left_shoulder',
            'BA': 'back', 'BO': 'bottom',
            'CA': 'two_hand_catch', 'PA': 'two_hand_palm', 'PU': 'two_hand_punch',
            '2H': 'two_hands', 'VM': None,
            # legacy shortcuts
            'R': 'right_foot', 'L': 'left_foot',
        })

    if 'pass_type' in df.columns:
        df['pass_type'] = df['pass_type'].replace({
            'B': 'cutback',
            'C': 'creates_contest',
            'F': 'flick_on',
            'H': 'long_throw_to_box',
            'O': 'over_the_top',
            'S': 'standard',
            'T': 'through_ball',
            'W': 'switch',
            'cross': 'cross',
        })

    if 'pass_accuracy_type' in df.columns:
        df['pass_accuracy_type'] = df['pass_accuracy_type'].replace({
            'A': 'away_from_defender',
            'C': 'checks_movement',
            'E': 'leads_into_challenge',
            'H': 'heavy',
            'I': 'in_stride',
            'L': 'light',
            'P': 'precise',
            'R': 'redirects',
            'S': 'standard',
        })

    if 'cross_type' in df.columns:
        df['cross_type'] = df['cross_type'].replace({
            'D': 'drilled',
            'F': 'floated',
            'I': 'inswinger',
            'O': 'outswinger',
            'P': 'placed',
        })

    if 'cross_zone_type' in df.columns:
        df['cross_zone_type'] = df['cross_zone_type'].replace({
            'C': 'central',
            'N': 'near_post',
            'S': 'six_yard_box',
            'F': 'far_post',
        })

    outcome_map = {
        'B': 'blocked',
        'C': 'completed',
        'D': 'intercepted',
        'G': 'shot_at_own_goal',
        'I': 'shot_at_goal',
        'S': 'stoppage',
        'O': 'out_of_play',
    }
    for column in ['pass_outcome', 'cross_outcome']:
        if column in df.columns:
            df[column] = df[column].replace(outcome_map)

    if 'carry_outcome' in df.columns:
        df['carry_outcome'] = df['carry_outcome'].replace({
            'C': 'leads_into_challenge',
            'L': 'lost',
            'R': 'retained',
            'S': 'stoppage',
        })

    if 'carry_type' in df.columns:
        df['carry_type'] = df['carry_type'].replace({
            'B': 'line_break',
            'C': 'change_direction',
            'D': 'drive_with_intent',
        })

    if 'carry_intent' in df.columns:
        df['carry_intent'] = df['carry_intent'].replace({
            'B': 'break_line',
            'C': 'create_space',
            'E': 'escape_pressure',
        })

    if 'dribble_type' in df.columns:
        df['dribble_type'] = df['dribble_type'].replace({
            'B': 'between_defenders',
            'I': 'inside',
            'K': 'knocks_in_front',
            'O': 'outside',
            'T': 'trick',
        })

    if 'pressure_type' in df.columns:
        df['pressure_type'] = df['pressure_type'].replace({
            'A': 'attempted',
            'L': 'passing_lane',
            'N': 'no_pressure',
            'P': 'pressured',
        })

    if 'lines_broken_type' in df.columns:
        df['lines_broken_type'] = df['lines_broken_type'].replace({
            'A': 'attack',
            'AD': 'attack_and_defense',
            'AM': 'attack_and_midfield',
            'AMD': 'attack_and_midfield_and_defense',
            'D': 'defense',
            'M': 'midfield',
            'MD': 'midfield_and_defense',
        })

    if 'set_piece' in df.columns:
        df['set_piece'] = df['set_piece'].replace({
            'P': 'penalty',
            'C': 'corner',
            'F': 'free_kick',
            'T': 'throw_in',
            'D': 'drop_ball',
            'G': 'goal_kick',
            'O': 'open_play',
            'K': 'kick_off',
        })

    return _clean_null_tokens(df)

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