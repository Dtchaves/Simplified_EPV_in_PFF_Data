"""Post-hoc Tier 2 context features for public StatsBomb 360 decisions (version ctx-v1).

Separate from ``data.py`` so canonical decision tables and their cache keys stay
unchanged. Rows are keyed by (original_match_id, event_id) of the canonical table.

Group A is strictly historical: only same-period events ordered before the
decision group by (time, index) are read, never the decision's own events or
anything later. Group B reads only the decision's own 360 frame; quantities that
the frame cannot see are NaN (unknown), never zero. Definitions are prespecified
in docs/public_360_tier2_prespecification.md.

Not used by rule (documented here, checked by tests below the marker): StatsBomb
possession ids/possession team, play pattern, pressure and counterpress flags,
shot attributes, key-pass links and pass assist annotations, best open teammate.
# BANNED-FIELDS-END
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .data import _segment_distance, disk_visible, metric_xy, time_seconds, visible_polygon

FEATURE_VERSION = 'ctx-v1'
GROUP_A = ['ctx_possession_age_s', 'ctx_possession_passes', 'ctx_possession_start_x',
           'ctx_restart_corner', 'ctx_restart_free_kick', 'ctx_restart_throw_in', 'ctx_restart_other',
           'ctx_set_piece_within_20s', 'ctx_in_linked', 'ctx_in_unresolved',
           'ctx_in_high', 'ctx_in_low', 'ctx_in_cross', 'ctx_in_through', 'ctx_in_cutback',
           'ctx_in_length_m', 'ctx_in_from_wide', 'ctx_actor_on_ball_s', 'ctx_actor_prior_actions']
GROUP_B = ['geo_triangle_defenders', 'geo_keeper_goal_dist', 'geo_keeper_line_offset']
OBSERVATION = ['geo_triangle_visible', 'geo_keeper_observed']

OPPONENT_CONTROL = {'Pass', 'Carry', 'Dribble', 'Shot', 'Clearance', 'Miscontrol', 'Dispossessed'}
INTERCEPTION_WON = {'Won', 'Success', 'Success In Play', 'Success Out'}
KEEPER_CONTROL = {'Collected', 'Keeper Sweeper', 'Smother'}
TEAM_ON_BALL = {'Pass', 'Ball Receipt*', 'Carry', 'Dribble', 'Ball Recovery', 'Shot', 'Interception',
                'Goal Keeper', 'Clearance', 'Miscontrol'}
RESTART_KIND = {'Corner': 'corner', 'Free Kick': 'free_kick', 'Throw-in': 'throw_in', 'Throw In': 'throw_in',
                'Goal Kick': 'other', 'Kick Off': 'other', 'Kickoff': 'other'}
SET_PIECES = {'Corner', 'Free Kick', 'Throw-in', 'Throw In'}
POSTS = ((52.5, 3.66), (52.5, -3.66))
GOAL_CENTRE = (52.5, 0.)
BOX_HALF_WIDTH = 20.16


def _name(event, *path):
    value = event
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def _team(event):
    return _name(event, 'team', 'id')


def _player(event):
    return _name(event, 'player', 'id')


def _restart_kind(event):
    if event['type']['name'] != 'Pass':
        return None
    return RESTART_KIND.get(_name(event, 'pass', 'type', 'name'))


def opponent_control(event, team):
    """Opponent event that ends the acting team's possession (turnover reset)."""
    if str(_team(event)) == str(team):
        return False
    typ = event['type']['name']
    if typ in OPPONENT_CONTROL:
        return True
    if typ == 'Ball Recovery':
        return not _name(event, 'ball_recovery', 'recovery_failure')
    if typ == 'Ball Receipt*':
        return _name(event, 'ball_receipt', 'outcome', 'name') != 'Incomplete'
    if typ == 'Interception':
        return _name(event, 'interception', 'outcome', 'name') in INTERCEPTION_WON
    if typ == 'Goal Keeper':
        return _name(event, 'goalkeeper', 'type', 'name') in KEEPER_CONTROL
    return False


def _x(event):
    try:
        return metric_xy(event.get('location'))[0]
    except (TypeError, ValueError):
        return math.nan


def group_a(history, decision):
    """Group A from ``history`` (same period, strictly before the decision group, chronological)."""
    team, actor, stamp = str(decision['team_id']), decision['player_id'], float(decision['time_seconds'])
    possession, origin = [], 'open_play'
    for event in reversed(history):
        if opponent_control(event, team):
            break
        if str(_team(event)) == team:
            possession.append(event)
            kind = _restart_kind(event)
            if kind:
                origin = kind
                break
    possession.reverse()
    on_ball = [e for e in possession if e['type']['name'] in TEAM_ON_BALL]
    start = on_ball[0] if on_ball else None
    start_x = _x(start) if start is not None else math.nan
    out = dict(ctx_possession_age_s=max(0., stamp - time_seconds(start['timestamp'])) if start else 0.,
               ctx_possession_passes=sum(e['type']['name'] == 'Pass' and 'outcome' not in (e.get('pass') or {})
                                         for e in possession),
               ctx_possession_start_x=start_x if math.isfinite(start_x) else float(decision['x']),
               ctx_restart_corner=int(origin == 'corner'), ctx_restart_free_kick=int(origin == 'free_kick'),
               ctx_restart_throw_in=int(origin == 'throw_in'), ctx_restart_other=int(origin == 'other'),
               ctx_restart_origin=origin)
    out['ctx_set_piece_within_20s'] = 0
    for event in reversed(history):
        if stamp - time_seconds(event['timestamp']) > 20.:
            break
        if event['type']['name'] == 'Pass' and _name(event, 'pass', 'type', 'name') in SET_PIECES:
            out['ctx_set_piece_within_20s'] = 1
            break

    passes = [e for e in possession if e['type']['name'] == 'Pass']
    status, last = 'no_incoming_pass', None
    if passes:
        last = passes[-1]
        body = last.get('pass') or {}
        after = history[history.index(last) + 1:]
        received = any(e['type']['name'] == 'Ball Receipt*' and _player(e) == actor
                       and _name(e, 'ball_receipt', 'outcome', 'name') != 'Incomplete' for e in after)
        linked = 'outcome' not in body and _name(body, 'recipient', 'id') == actor and received
        status = 'linked' if linked else 'unresolved'
    out.update(ctx_in_status=status, ctx_in_linked=int(status == 'linked'), ctx_in_unresolved=int(status == 'unresolved'),
               ctx_in_high=0, ctx_in_low=0, ctx_in_cross=0, ctx_in_through=0, ctx_in_cutback=0,
               ctx_in_length_m=0., ctx_in_from_wide=0)
    if status == 'linked':
        body = last['pass']
        height = _name(body, 'height', 'name')
        out.update(ctx_in_high=int(height == 'High Pass'), ctx_in_low=int(height == 'Low Pass'),
                   ctx_in_cross=int(bool(body.get('cross'))),
                   ctx_in_through=int(bool(body.get('through_ball')) or _name(body, 'technique', 'name') == 'Through Ball'),
                   ctx_in_cutback=int(bool(body.get('cut_back'))))
        try:
            a, b = metric_xy(last.get('location')), metric_xy(body.get('end_location'))
            out['ctx_in_length_m'] = math.dist(a, b)
            out['ctx_in_from_wide'] = int(abs(a[1]) >= BOX_HALF_WIDTH)
        except (TypeError, ValueError):
            out['ctx_in_length_m'] = math.nan

    run = []
    for event in reversed(possession):
        if _player(event) != actor:
            break
        run.append(event)
    out['ctx_actor_on_ball_s'] = max(0., stamp - time_seconds(run[-1]['timestamp'])) if run else 0.
    out['ctx_actor_prior_actions'] = sum(e['type']['name'] in {'Carry', 'Dribble'} for e in run)
    return out


def _inside(point, triangle):
    a, b, c = triangle
    d1 = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
    d2 = (c[0] - b[0]) * (point[1] - b[1]) - (c[1] - b[1]) * (point[0] - b[0])
    d3 = (a[0] - c[0]) * (point[1] - c[1]) - (a[1] - c[1]) * (point[0] - c[0])
    return (d1 > 0 and d2 > 0 and d3 > 0) or (d1 < 0 and d2 < 0 and d3 < 0)


def _proper_cross(p, q, r, s):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    o1, o2, o3, o4 = orient(p, q, r), orient(p, q, s), orient(r, s, p), orient(r, s, q)
    return o1 * o2 < 0 and o3 * o4 < 0


def triangle_visible(triangle, polygon, shrink=.05):
    cx = sum(p[0] for p in triangle) / 3
    cy = sum(p[1] for p in triangle) / 3
    inner = []
    for x, y in triangle:
        length = math.hypot(cx - x, cy - y)
        inner.append((x + (cx - x) * shrink / length, y + (cy - y) * shrink / length) if length > shrink else (cx, cy))
    if not all(disk_visible(p, polygon, 0.) for p in inner):
        return False
    edges = list(zip(polygon, polygon[1:] + polygon[:1]))
    for a, b in zip(inner, inner[1:] + inner[:1]):
        if any(_proper_cross(a, b, c, d) for c, d in edges):
            return False
    return not any(_inside(v, inner) for v in polygon)


def group_b(frame, decision):
    out = dict(geo_triangle_visible=False, geo_keeper_observed=False,
               geo_triangle_defenders=math.nan, geo_keeper_goal_dist=math.nan, geo_keeper_line_offset=math.nan)
    if frame is None:
        return out
    ball = (float(decision['x']), float(decision['y']))
    triangle = (ball,) + POSTS
    try:
        polygon = visible_polygon(frame.get('visible_area'))
        players = [(q, metric_xy(q['location'])) for q in frame['freeze_frame']]
    except (TypeError, ValueError, KeyError):
        return out
    opponents = [(q, p) for q, p in players if q.get('teammate') is False]
    out['geo_triangle_visible'] = bool(triangle_visible(triangle, polygon))
    if out['geo_triangle_visible']:
        out['geo_triangle_defenders'] = float(sum(_inside(p, triangle) for q, p in opponents if not q.get('keeper')))
    keepers = [p for q, p in opponents if q.get('keeper')]
    if keepers:
        keeper = keepers[0]
        out.update(geo_keeper_observed=True, geo_keeper_goal_dist=math.dist(keeper, GOAL_CENTRE),
                   geo_keeper_line_offset=_segment_distance(keeper, ball, GOAL_CENTRE))
    return out


def match_context_features(events, frames, decisions):
    """Feature rows for the given decisions of one match (pure; fixture-testable)."""
    ordered = sorted(events, key=lambda e: (e['period'], time_seconds(e['timestamp']), e['index']))
    position = {e['id']: i for i, e in enumerate(ordered)}
    period_start = {}
    for i, e in enumerate(ordered):
        period_start.setdefault(e['period'], i)
    by_frame = {f['event_uuid']: f for f in frames}
    rows = []
    for decision in decisions:
        group = set(decision['source_event_ids']) | {decision['event_id']}
        first = min(position[g] for g in group)
        period = ordered[first]['period']
        history = [e for e in ordered[period_start[period]:first] if e['id'] not in group]
        row = dict(original_match_id=decision['original_match_id'], event_id=decision['event_id'])
        row.update(group_a(history, decision))
        row.update(group_b(by_frame.get(decision['event_id']), decision))
        rows.append(row)
    return rows


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def table_key(canonical_key, raw_fingerprint):
    code = _sha(Path(__file__).read_bytes())
    return _sha(json.dumps(dict(version=FEATURE_VERSION, code=code, canonical=canonical_key, raw=raw_fingerprint),
                           sort_keys=True).encode())


def build_context_table(data_root, decisions, canonical_key, logger=None):
    """Build the feature table for every candidate decision of a canonical table.

    Returns (frame, meta). Raw files are hashed into the fingerprint, so a changed raw
    file, canonical table or feature code yields a new key (downstream invalidation).
    """
    import pandas as pd
    root = Path(data_root)
    rows, hashes = [], {}
    for match_id, part in decisions.groupby('original_match_id', sort=True):
        event_path = root / 'raw' / 'data' / 'events' / f'{match_id}.json'
        frame_path = root / 'raw' / 'data' / 'three-sixty' / f'{match_id}.json'
        event_bytes, frame_bytes = event_path.read_bytes(), frame_path.read_bytes()
        hashes[str(match_id)] = [_sha(event_bytes), _sha(frame_bytes)]
        records = part[['original_match_id', 'event_id', 'source_event_ids', 'player_id', 'team_id', 'period',
                        'time_seconds', 'x', 'y']].to_dict('records')
        rows.extend(match_context_features(json.loads(event_bytes), json.loads(frame_bytes), records))
        if logger:
            logger('context_match', match_id=str(match_id), decisions=len(records))
    fingerprint = _sha(json.dumps(hashes, sort_keys=True).encode())
    key = table_key(canonical_key, fingerprint)
    frame = pd.DataFrame(rows)
    meta = dict(version=FEATURE_VERSION, key=key, canonical_key=canonical_key, raw_fingerprint=fingerprint,
                code_sha256=_sha(Path(__file__).read_bytes()), rows=len(frame), matches=len(hashes),
                group_a=GROUP_A, group_b=GROUP_B, observation=OBSERVATION)
    return frame, meta
