"""Saved-score dependence inference: cluster SEs, CGM two-way, wild cluster bootstrap, paired differences."""
from pathlib import Path
import sys, unittest
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from PublicCausal import dependence as dep


def _rows(seed=0, matches=24, per=30, effect=.05):
    rng = np.random.default_rng(seed)
    teams = [f't{i}' for i in range(8)]
    frames = []
    for m in range(matches):
        team, opp = rng.choice(teams, 2, replace=False)
        t = (rng.random(per) < .15).astype(int)
        e = np.clip(.1 + .1 * rng.random(per), .01, .9)
        m0 = rng.normal(0, .05, per)
        y = m0 + effect * t + rng.normal(0, .3, per)
        frames.append(pd.DataFrame(dict(original_match_id=f'm{m}', team_id=team, opponent_id=opp,
                                        T=t, e=e, m0=m0, Y=y, support=True, eligible=True,
                                        decision_row_id=[f'm{m}:{i}' for i in range(per)])))
    return pd.concat(frames, ignore_index=True)


def _numerator(rows):
    w = np.where(rows['T'] == 0, rows.e / (1 - rows.e), 0.)
    r = rows.Y - rows.m0
    return np.where(rows['T'] == 1, r, -w * r)


class ClusterSETests(unittest.TestCase):
    def test_match_clustering_equals_hand_formula(self):
        rows = _rows()
        rows['att_numerator'] = _numerator(rows)
        n1 = rows['T'].sum(); theta = rows.att_numerator.sum() / n1
        u = (rows.att_numerator - theta * rows['T']).groupby(rows.original_match_id).sum()
        g = len(u); expected = np.sqrt(g / (g - 1) * (u ** 2).sum()) / n1
        out = dep.cluster_se(rows, 'original_match_id')
        self.assertAlmostEqual(out['att'], theta, places=12)
        self.assertAlmostEqual(out['se'], expected, places=12)
        self.assertEqual(out['clusters'], g)
        self.assertEqual(out['df'], g - 1)

    def test_registry_zero_clusters_count_in_G(self):
        rows = _rows()
        rows['att_numerator'] = _numerator(rows)
        base = dep.cluster_se(rows, 'original_match_id')
        more = dep.cluster_se(rows, 'original_match_id', registry=list(rows.original_match_id.unique()) + ['extra'])
        self.assertEqual(more['clusters'], base['clusters'] + 1)
        g = base['clusters']
        self.assertAlmostEqual(more['se'] / base['se'], np.sqrt(((g + 1) / g) / (g / (g - 1))), places=12)

    def test_cgm_two_way_identity(self):
        rows = _rows(1)
        rows['att_numerator'] = _numerator(rows)
        a = dep.cluster_se(rows, 'team_id')['se'] ** 2
        b = dep.cluster_se(rows, 'opponent_id')['se'] ** 2
        rows['pair'] = rows.team_id + '|' + rows.opponent_id
        c = dep.cluster_se(rows, 'pair')['se'] ** 2
        out = dep.two_way_cgm(rows)
        self.assertAlmostEqual(out['variance_raw'], a + b - c, places=14)
        if a + b - c > 0:
            self.assertAlmostEqual(out['se'], np.sqrt(a + b - c), places=12)
            self.assertFalse(out['fallback_used'])

    def test_cgm_negative_variance_falls_back_to_max(self):
        out = dep._cgm_combine(1e-4, 2e-4, 5e-4, g_a=10, g_b=12)
        self.assertTrue(out['fallback_used'])
        self.assertAlmostEqual(out['se'], np.sqrt(2e-4))
        self.assertEqual(out['df'], 9)


class WildBootstrapTests(unittest.TestCase):
    def test_webb_weights_moments(self):
        w = dep.webb_weights(np.random.default_rng(0), 200000, 1).ravel()
        self.assertEqual(set(np.round(np.abs(w) ** 2, 6)), {.5, 1., 1.5})
        self.assertAlmostEqual(w.mean(), 0, delta=.01)
        self.assertAlmostEqual((w ** 2).mean(), 1, delta=.01)

    def test_pvalue_at_estimate_is_one_and_ci_contains_estimate(self):
        rows = _rows(2, effect=.1)
        rows['att_numerator'] = _numerator(rows)
        out = dep.wild_cluster_bootstrap(rows, 'original_match_id', draws=999, seed=3)
        self.assertGreater(out['pvalue_at_estimate'], .99)
        low, high = out['interval']
        self.assertLess(low, out['att']); self.assertGreater(high, out['att'])
        self.assertTrue(0 <= out['pvalue_zero'] <= 1)

    def test_null_size_is_reasonable(self):
        rejections = 0
        for seed in range(40):
            rows = _rows(100 + seed, effect=0.)
            rows['att_numerator'] = _numerator(rows)
            rejections += dep.wild_cluster_bootstrap(rows, 'original_match_id', draws=199, seed=seed)['pvalue_zero'] < .05
        self.assertLessEqual(rejections, 7)

    def test_deterministic_given_seed(self):
        rows = _rows(4)
        rows['att_numerator'] = _numerator(rows)
        a = dep.wild_cluster_bootstrap(rows, 'team_id', draws=299, seed=5)
        b = dep.wild_cluster_bootstrap(rows, 'team_id', draws=299, seed=5)
        self.assertEqual(a['interval'], b['interval'])

    def test_wild_bootstrap_rejects_clusters_outside_registry(self):
        rows = _rows(7)
        rows['att_numerator'] = _numerator(rows)
        registry = [m for m in rows.original_match_id.unique() if m != 'm0']
        with self.assertRaisesRegex(ValueError, 'outside the registry'):
            dep.wild_cluster_bootstrap(rows, 'original_match_id', registry=registry, draws=99, seed=1)

    def test_wild_bootstrap_full_registry_matches_no_registry_plus_zero_clusters(self):
        rows = _rows(7)
        rows['att_numerator'] = _numerator(rows)
        full = sorted(rows.original_match_id.unique())  # sorted = groupby order, so the weight draws line up
        base = dep.wild_cluster_bootstrap(rows, 'original_match_id', draws=99, seed=1)
        same = dep.wild_cluster_bootstrap(rows, 'original_match_id', registry=full, draws=99, seed=1)
        self.assertEqual(base['interval'], same['interval'])
        self.assertEqual(base['clusters'], same['clusters'])
        more = dep.wild_cluster_bootstrap(rows, 'original_match_id', registry=full + ['extra'], draws=99, seed=1)
        self.assertEqual(more['clusters'], base['clusters'] + 1)


class PairedDifferenceTests(unittest.TestCase):
    def test_paired_difference_matches_hand_formula(self):
        a = _rows(5); b = a.copy(); b['m0'] = b.m0 + .01 * np.sin(np.arange(len(b)))
        out = dep.paired_difference(a, b)
        na, nb = _numerator(a), _numerator(b)
        n1 = a['T'].sum(); ta, tb = na.sum() / n1, nb.sum() / n1
        psi = (nb - tb * a['T']) - (na - ta * a['T'])
        u = pd.Series(psi).groupby(a.original_match_id.values).sum()
        g = len(u)
        self.assertAlmostEqual(out['difference'], tb - ta, places=12)
        self.assertAlmostEqual(out['se'], np.sqrt(g / (g - 1) * (u ** 2).sum()) / n1, places=12)

    def test_paired_difference_rejects_different_samples(self):
        a = _rows(6); b = a.iloc[1:].copy()
        with self.assertRaises(ValueError):
            dep.paired_difference(a, b)
        c = a.copy(); c.loc[c.index[0], 'T'] = 1 - c.loc[c.index[0], 'T']
        with self.assertRaises(ValueError):
            dep.paired_difference(a, c)

    def test_paired_difference_rejects_clusters_outside_registry(self):
        a = _rows(8); b = a.copy(); b['m0'] = b.m0 + .01
        registry = [m for m in a.original_match_id.unique() if m != 'm0']
        with self.assertRaisesRegex(ValueError, 'outside the registry'):
            dep.paired_difference(a, b, registry=registry)

    def test_paired_difference_full_registry_unchanged(self):
        a = _rows(8); b = a.copy(); b['m0'] = b.m0 + .01
        full = list(a.original_match_id.unique())
        base = dep.paired_difference(a, b)
        same = dep.paired_difference(a, b, registry=full)
        self.assertEqual(base['se'], same['se'])
        self.assertEqual(dep.paired_difference(a, b, registry=full + ['extra'])['clusters'], base['clusters'] + 1)


class NullOffsetTests(unittest.TestCase):
    def test_match_heterogeneity_plugin(self):
        rows = pd.DataFrame(dict(original_match_id=['a'] * 4 + ['b'] * 4, T=[1, 1, 0, 0, 1, 0, 0, 0],
                                 Y=[1, 0, 1, 0, 0, 0, 0, 0], m0=0., support=True))
        out = dep.match_heterogeneity_plugin(rows)
        # match a: n1=2,n0=2, mean residual .5; match b: n1=1,n0=3, mean 0. N1=3,N0=5
        self.assertAlmostEqual(out['residual_plugin'], .5 * (2 / 3 - 2 / 5), places=12)

    def test_match_stratified_att_hand_example(self):
        rows = pd.DataFrame(dict(original_match_id=['a'] * 3 + ['b'] * 3, T=[1, 0, 0, 1, 0, 0],
                                 Y=[1., 0., 1., 0., 1., 0.], m0=0., e=[.5, .5, .2, .5, .5, .5], support=True))
        out = dep.match_stratified_att(rows)
        # a: treated r=1, controls w=1 (r=0), .25 (r=1) -> 1 - .25/1.25 = .8 ; b: 0 - (1*1+1*0)/2 = -.5
        self.assertAlmostEqual(out['att'], (.8 - .5) / 2, places=12)
        self.assertEqual(out['dropped_treated'], 0)

    def test_paired_null_summary(self):
        recs = [dict(status='ok', att=.003, known_assignment_att=.001), dict(status='ok', att=.001, known_assignment_att=.0),
                dict(status='failed', att=None, known_assignment_att=None)]
        out = dep.paired_null_summary(recs)
        self.assertEqual(out['n'], 2)
        self.assertAlmostEqual(out['mean_difference'], .0015)


if __name__ == '__main__':
    unittest.main()
