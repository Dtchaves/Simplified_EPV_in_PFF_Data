"""Group A (strictly historical event context) and Group B (limited 360 geometry) contracts."""
from pathlib import Path
import copy, math, sys, unittest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from PublicCausal import context_features as cf

TEAM, OPP, ACTOR, MATE = 1, 2, 10, 11
FULL = [0, 0, 120, 0, 120, 80, 0, 80]


def ts(seconds):
    return f'00:{int(seconds // 60):02d}:{seconds % 60:06.3f}'


class Builder:
    def __init__(self):
        self.events, self.index = [], 0

    def add(self, seconds, typ, team=TEAM, player=MATE, location=(90, 40), **extra):
        self.index += 1
        event = dict(id=f'e{self.index}', index=self.index, period=1, timestamp=ts(seconds),
                     type=dict(name=typ), team=dict(id=team), player=dict(id=player),
                     location=list(location), **extra)
        self.events.append(event)
        return event

    def pass_(self, seconds, recipient=ACTOR, team=TEAM, player=MATE, outcome=None, height='Ground Pass',
              kind=None, end=(100, 40), **flags):
        body = dict(recipient=dict(id=recipient), height=dict(name=height), end_location=list(end), **flags)
        if outcome:
            body['outcome'] = dict(name=outcome)
        if kind:
            body['type'] = dict(name=kind)
        return self.add(seconds, 'Pass', team=team, player=player, **{'pass': body})


def decision_for(event, x=None, y=None):
    from PublicCausal.data import metric_xy, time_seconds
    mx, my = metric_xy(event['location'])
    return dict(original_match_id=1, event_id=event['id'], source_event_ids=[event['id']],
                player_id=event['player']['id'], team_id=str(event['team']['id']), period=event['period'],
                time_seconds=time_seconds(event['timestamp']), x=mx if x is None else x, y=my if y is None else y)


def features(builder, decision_event, frames=()):
    rows = cf.match_context_features(builder.events, list(frames), [decision_for(decision_event)])
    return rows[0]


def standard_sequence():
    b = Builder()
    b.add(0, 'Ball Recovery', player=MATE, location=(60, 40))
    b.pass_(2, recipient=ACTOR, player=MATE, height='High Pass', cross=True, end=(108, 40))
    b.add(3, 'Ball Receipt*', player=ACTOR, location=(108, 40))
    b.add(3.5, 'Pressure', team=OPP, player=99, location=(108, 41))
    b.add(3.6, 'Carry', player=ACTOR, location=(108, 40))
    decision = b.add(5, 'Shot', player=ACTOR, location=(110, 40), shot=dict(key_pass_id='e2'))
    return b, decision


class GroupATests(unittest.TestCase):
    def test_standard_linked_sequence(self):
        b, d = standard_sequence()
        f = features(b, d)
        self.assertEqual(f['ctx_in_status'], 'linked')
        self.assertEqual((f['ctx_in_linked'], f['ctx_in_unresolved']), (1, 0))
        self.assertEqual((f['ctx_in_high'], f['ctx_in_low'], f['ctx_in_cross']), (1, 0, 1))
        self.assertAlmostEqual(f['ctx_possession_age_s'], 5.)
        self.assertEqual(f['ctx_possession_passes'], 1)
        self.assertAlmostEqual(f['ctx_possession_start_x'], 105 * 60 / 120 - 52.5)
        self.assertAlmostEqual(f['ctx_actor_on_ball_s'], 2.)  # receipt at 3s, opponent pressure does not break run
        self.assertEqual(f['ctx_actor_prior_actions'], 1)
        self.assertEqual(f['ctx_restart_corner'] + f['ctx_restart_free_kick'] + f['ctx_restart_throw_in'] + f['ctx_restart_other'], 0)
        self.assertAlmostEqual(f['ctx_in_length_m'], 105 * 18 / 120)

    def test_future_events_cannot_change_features(self):
        b, d = standard_sequence()
        before = features(b, d)
        b.add(5, 'Pressure', team=OPP, player=98)  # same clock, later index
        b.add(6, 'Ball Recovery', team=OPP, player=97)
        b.pass_(7, recipient=ACTOR, player=MATE, height='Low Pass')
        b.add(8, 'Ball Receipt*', player=ACTOR)
        self.assertEqual(features(b, d), before)

    def test_current_group_and_assist_annotations_are_ignored(self):
        b, d = standard_sequence()
        before = features(b, d)
        b2 = copy.deepcopy(b)
        b2.events[1]['pass'].update(shot_assist=True, goal_assist=True, assisted_shot_id=d['id'])
        self.assertEqual(features(b2, b2.events[-1]), before)
        b3 = copy.deepcopy(b)
        extra = b3.add(5, 'Carry', player=ACTOR, location=(110, 40))
        decision = decision_for(b3.events[5])
        decision['source_event_ids'] = [b3.events[5]['id'], extra['id']]
        self.assertEqual(cf.match_context_features(b3.events, [], [decision])[0], before)

    def test_opponent_control_resets_possession(self):
        b = Builder()
        b.pass_(0, recipient=ACTOR, player=MATE)
        b.add(1, 'Ball Receipt*', player=ACTOR)
        b.add(2, 'Ball Recovery', team=OPP, player=99)
        b.add(4, 'Ball Recovery', player=ACTOR, location=(80, 40))
        d = b.add(6, 'Pass', player=ACTOR, location=(82, 40), **{'pass': dict(recipient=dict(id=MATE))})
        f = features(b, d)
        self.assertEqual(f['ctx_in_status'], 'no_incoming_pass')
        self.assertEqual(f['ctx_possession_passes'], 0)
        self.assertAlmostEqual(f['ctx_possession_age_s'], 2.)
        self.assertAlmostEqual(f['ctx_actor_on_ball_s'], 2.)

    def test_failed_opponent_actions_do_not_reset(self):
        b, d = standard_sequence()
        b.events.insert(2, dict(id='x1', index=2.5, period=1, timestamp=ts(2.5), type=dict(name='Ball Receipt*'),
                                team=dict(id=OPP), player=dict(id=99), location=[100, 40],
                                ball_receipt=dict(outcome=dict(name='Incomplete'))))
        self.assertEqual(features(b, d)['ctx_possession_passes'], 1)

    def test_restart_origin_and_linkage(self):
        b = Builder()
        b.pass_(0, recipient=ACTOR, player=MATE, kind='Corner', height='High Pass', end=(112, 40))
        b.add(2, 'Ball Receipt*', player=ACTOR, location=(112, 40))
        d = b.add(2.4, 'Shot', player=ACTOR, location=(112, 40))
        f = features(b, d)
        self.assertEqual(f['ctx_restart_corner'], 1)
        self.assertEqual(f['ctx_set_piece_within_20s'], 1)
        self.assertEqual(f['ctx_in_status'], 'linked')

    def test_unresolved_when_last_pass_not_to_actor_or_no_receipt(self):
        b = Builder()
        b.pass_(0, recipient=MATE + 1, player=MATE)
        d = b.add(2, 'Shot', player=ACTOR, location=(110, 40))
        f = features(b, d)
        self.assertEqual(f['ctx_in_status'], 'unresolved')
        self.assertEqual((f['ctx_in_high'], f['ctx_in_cross'], f['ctx_in_length_m']), (0, 0, 0.))
        b2 = Builder()
        b2.pass_(0, recipient=ACTOR, player=MATE)
        d2 = b2.add(2, 'Shot', player=ACTOR, location=(110, 40))
        self.assertEqual(features(b2, d2)['ctx_in_status'], 'unresolved')

    def test_set_piece_window(self):
        b = Builder()
        b.pass_(0, team=OPP, player=99, recipient=98, kind='Free Kick')
        b.add(1, 'Clearance', player=MATE)
        b.add(3, 'Ball Recovery', player=ACTOR)
        near = b.add(15, 'Carry', player=ACTOR)
        self.assertEqual(features(b, near)['ctx_set_piece_within_20s'], 1)
        far = b.add(25, 'Pass', player=ACTOR, **{'pass': dict(recipient=dict(id=MATE))})
        self.assertEqual(features(b, far)['ctx_set_piece_within_20s'], 0)

    def test_history_is_limited_to_the_decision_period(self):
        b = Builder()
        b.pass_(0, recipient=ACTOR, player=MATE, kind='Corner', height='High Pass', end=(112, 40))
        b.add(1, 'Ball Receipt*', player=ACTOR, location=(112, 40))
        b.add(1.5, 'Carry', player=ACTOR, location=(112, 40))
        recovery = b.add(1, 'Ball Recovery', player=ACTOR, location=(80, 40))
        d = b.add(2, 'Shot', player=ACTOR, location=(110, 40))
        for event in (recovery, d):
            event['period'] = 2  # earlier-period events must not reach the second-period decision
        f = features(b, d)
        self.assertEqual(f['ctx_in_status'], 'no_incoming_pass')
        self.assertEqual(f['ctx_possession_passes'], 0)
        self.assertEqual(f['ctx_restart_corner'], 0)
        self.assertEqual(f['ctx_set_piece_within_20s'], 0)
        self.assertAlmostEqual(f['ctx_possession_age_s'], 1.)
        self.assertAlmostEqual(f['ctx_actor_on_ball_s'], 1.)
        self.assertEqual(f['ctx_actor_prior_actions'], 0)

    def test_incoming_pass_with_outcome_is_never_linked(self):
        for outcome in ('Incomplete', 'Out', 'Injury Clearance'):
            with self.subTest(outcome=outcome):
                b = Builder()
                b.pass_(0, recipient=ACTOR, player=MATE, outcome=outcome)
                b.add(1, 'Ball Receipt*', player=ACTOR)  # a receipt by the named recipient does not complete the pass
                d = b.add(2, 'Shot', player=ACTOR, location=(110, 40))
                f = features(b, d)
                self.assertEqual(f['ctx_in_status'], 'unresolved')
                self.assertEqual((f['ctx_in_linked'], f['ctx_in_unresolved']), (0, 1))
                self.assertEqual((f['ctx_in_high'], f['ctx_in_cross'], f['ctx_in_length_m']), (0, 0, 0.))
                self.assertEqual(f['ctx_possession_passes'], 0)  # only completed passes count

    def test_linked_incoming_pass_is_the_same_for_every_continuation_arm(self):
        b, shot = standard_sequence()
        reference = features(b, shot)
        self.assertEqual(reference['ctx_in_status'], 'linked')
        linkage = [k for k in reference if k.startswith('ctx_in_')]
        for typ in ('Pass', 'Carry', 'Dribble'):
            with self.subTest(arm=typ):
                c, _ = standard_sequence()
                c.events.pop()  # drop the shot, decide on the continuation action instead
                extra = {'pass': dict(recipient=dict(id=MATE))} if typ == 'Pass' else {}
                d = c.add(5, typ, player=ACTOR, location=(110, 40), **extra)
                f = features(c, d)
                self.assertEqual(f['ctx_in_status'], 'linked')
                self.assertEqual({k: f[k] for k in linkage}, {k: reference[k] for k in linkage})
                self.assertEqual(f['ctx_possession_passes'], 1)

    def test_won_interception_resets_possession(self):
        for outcome, resets in (('Won', True), ('Success', True), ('Success In Play', True), ('Success Out', True),
                                ('Lost In Play', False), ('Lost Out', False)):
            with self.subTest(outcome=outcome):
                b = Builder()
                b.pass_(0, recipient=ACTOR, player=MATE)
                b.add(1, 'Ball Receipt*', player=ACTOR)
                b.add(2, 'Interception', team=OPP, player=99, interception=dict(outcome=dict(name=outcome)))
                b.add(4, 'Ball Recovery', player=ACTOR, location=(80, 40))
                d = b.add(6, 'Pass', player=ACTOR, location=(82, 40), **{'pass': dict(recipient=dict(id=MATE))})
                f = features(b, d)
                self.assertEqual(f['ctx_in_status'], 'no_incoming_pass' if resets else 'linked')
                self.assertEqual(f['ctx_possession_passes'], 0 if resets else 1)

    def test_keeper_control_resets_possession(self):
        for kind, resets in (('Collected', True), ('Keeper Sweeper', True), ('Smother', True),
                             ('Shot Saved', False), ('Punch', False)):
            with self.subTest(keeper_event=kind):
                b = Builder()
                b.pass_(0, recipient=ACTOR, player=MATE)
                b.add(1, 'Ball Receipt*', player=ACTOR)
                b.add(2, 'Goal Keeper', team=OPP, player=99, goalkeeper=dict(type=dict(name=kind)))
                b.add(4, 'Ball Recovery', player=ACTOR, location=(80, 40))
                d = b.add(6, 'Pass', player=ACTOR, location=(82, 40), **{'pass': dict(recipient=dict(id=MATE))})
                f = features(b, d)
                self.assertEqual(f['ctx_in_status'], 'no_incoming_pass' if resets else 'linked')
                self.assertEqual(f['ctx_possession_passes'], 0 if resets else 1)


def frame_for(event, visible=FULL, players=()):
    ff = [dict(teammate=True, actor=True, keeper=False, location=event['location'])] + list(players)
    return dict(event_uuid=event['id'], visible_area=list(visible), freeze_frame=ff)


class GroupBTests(unittest.TestCase):
    def setUp(self):
        self.b = Builder()
        self.d = self.b.add(0, 'Shot', player=ACTOR, location=(100, 40))

    def test_full_visibility_counts_triangle_defenders_and_keeper(self):
        players = [dict(teammate=False, actor=False, keeper=False, location=[110, 40]),    # inside
                   dict(teammate=False, actor=False, keeper=False, location=[110, 60]),    # outside
                   dict(teammate=True, actor=False, keeper=False, location=[112, 40]),     # teammate
                   dict(teammate=False, actor=False, keeper=True, location=[118, 42])]
        f = features(self.b, self.d, [frame_for(self.d, players=players)])
        self.assertTrue(f['geo_triangle_visible'])
        self.assertEqual(f['geo_triangle_defenders'], 1)
        self.assertTrue(f['geo_keeper_observed'])
        kx, ky = 105 * 118 / 120 - 52.5, 34 - 68 * 42 / 80
        self.assertAlmostEqual(f['geo_keeper_goal_dist'], math.hypot(52.5 - kx, ky))
        self.assertAlmostEqual(f['geo_keeper_line_offset'], abs(ky))

    def test_unseen_triangle_is_unknown_not_zero(self):
        partial = [0, 0, 105, 0, 105, 80, 0, 80]
        players = [dict(teammate=False, actor=False, keeper=True, location=[104, 40])]
        f = features(self.b, self.d, [frame_for(self.d, visible=partial, players=players)])
        self.assertFalse(f['geo_triangle_visible'])
        self.assertTrue(math.isnan(f['geo_triangle_defenders']))
        self.assertTrue(f['geo_keeper_observed'])

    def test_missing_keeper_and_frame_are_unknown(self):
        f = features(self.b, self.d, [frame_for(self.d)])
        self.assertFalse(f['geo_keeper_observed'])
        self.assertTrue(math.isnan(f['geo_keeper_goal_dist']))
        g = features(self.b, self.d, [])
        self.assertFalse(g['geo_triangle_visible'])
        self.assertTrue(math.isnan(g['geo_triangle_defenders']))

    def test_polygon_notch_inside_triangle_breaks_visibility(self):
        notch = [0, 0, 120, 0, 120, 39, 110, 40, 120, 41, 120, 80, 0, 80]
        f = features(self.b, self.d, [frame_for(self.d, visible=notch)])
        self.assertFalse(f['geo_triangle_visible'])


class TableTests(unittest.TestCase):
    def test_feature_lists_and_exclusions(self):
        self.assertEqual(len(cf.GROUP_A), 16 + 3)  # A4 is four flags
        self.assertEqual(cf.GROUP_B, ['geo_triangle_defenders', 'geo_keeper_goal_dist', 'geo_keeper_line_offset'])
        source = (ROOT / 'src/PublicCausal/context_features.py').read_text()
        for banned in ['play_pattern', 'under_pressure', 'key_pass_id', 'shot_assist', 'goal_assist',
                       "'possession'", 'possession_team', 'counterpress']:
            self.assertNotIn(banned, source.split('# BANNED-FIELDS-END')[-1])

    def test_table_key_changes_with_inputs(self):
        a = cf.table_key('ck1', 'raw1')
        self.assertNotEqual(a, cf.table_key('ck2', 'raw1'))
        self.assertNotEqual(a, cf.table_key('ck1', 'raw2'))
        self.assertEqual(a, cf.table_key('ck1', 'raw1'))


if __name__ == '__main__':
    unittest.main()
