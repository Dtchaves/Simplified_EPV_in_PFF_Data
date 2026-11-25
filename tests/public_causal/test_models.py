"""Tests execute the notebook's exported definitions, not a parallel estimator."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from PublicCausal.notebook_runtime import load_notebook_core


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.core = load_notebook_core(ROOT / 'results/models/public_statsbomb_causal_study.ipynb', cls.temporary.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def fixture(self):
        c = self.core
        rng = c.np.random.default_rng(41)
        n = 480
        frame = c.pd.DataFrame({name: rng.normal(size=n) for name in c.NUMERIC_FEATURES})
        frame['x'] = rng.uniform(20, 45, n)
        frame['y'] = rng.uniform(-8, 8, n)
        frame['distance'] = rng.uniform(10, 17, n)
        frame['angle'] = rng.uniform(5, 50, n)
        frame['defenders_3m'] = 1
        frame['nearest_defender'] = rng.uniform(.1, 3, n)
        frame['second_defender'] = 3.
        frame['period'] = 1
        frame['time_seconds'] = rng.uniform(0, 2700, n)
        frame['score_difference'] = 0
        frame['original_match_id'] = [str(i // 40) for i in range(n)]
        frame['match_id'] = frame.original_match_id
        frame['event_id'] = [f'uuid-{i}' for i in range(n)]
        frame['T'] = [int(i % 5 == 0) for i in range(n)]
        frame['Y'] = rng.choice([-1, 0, 1], n, p=[.01, .94, .05])
        for name in c.CATEGORICAL_FEATURES:
            frame[name] = 'test'
        return frame

    def config(self):
        return self.core.scientific_config(dict(outer_folds=3, inner_folds=2,
            e_candidates=[{'kind': 'logistic', 'C': .2}],
            outcome_candidates=[{'kind': 'ridge', 'alpha': 1}],
            min_shots=5, min_matches=2, min_ess=5, donor_min=2, donor_match_min=2,
            key_smd_max=10, other_smd_max=10, max_weight_share=1, max_match_weight_share=1))

    def test_all_predictions_and_training_provenance(self):
        c = self.core
        result = c.fit_study(self.fixture(), self.config())
        rows = result['rows']
        self.assertTrue(c.np.isfinite(rows[['e', 'm0', 'm1', 'portable_e']]).all().all())
        self.assertEqual(set(rows.outer_fold), {0, 1, 2})
        for record in result['provenance']:
            self.assertFalse(set(record['train_match_ids']) & set(record['test_match_ids']))
        for fold in range(3):
            expected = set(rows.loc[rows.outer_fold == fold, 'original_match_id'])
            for name in ['e_outer', 'm0_outer', 'm1_outer']:
                record = next(r for r in result['provenance'] if r['stage'] == name and r['fold'] == fold)
                self.assertEqual(set(record['test_match_ids']), expected)
                self.assertFalse(expected & set(record['train_match_ids']))

    def test_bootstrap_copies_stay_in_original_match_fold(self):
        c = self.core
        frame = c.prepare_analysis(self.fixture(), self.config())
        repeated = c.pd.concat([frame, frame.loc[frame.original_match_id == '0']], ignore_index=True)
        folds = c.grouped_folds(repeated, 3, 41)
        for i, j in folds:
            self.assertFalse(set(repeated.iloc[i].original_match_id) & set(repeated.iloc[j].original_match_id))

    def test_fixed_domain_does_not_use_evaluation_treatment(self):
        c = self.core
        frame = c.prepare_analysis(self.fixture(), self.config())
        probability = c.np.full(len(frame), .2)
        domain = c.learn_domain(frame, probability, .05, self.config())
        before = c.domain_mask(frame, probability, domain)
        frame['Treatment'] = 1 - frame.Treatment
        after = c.domain_mask(frame, probability, domain)
        c.np.testing.assert_array_equal(before, after)

    def test_distinct_donors_not_inflated_by_draws(self):
        c = self.core
        frame = c.prepare_analysis(self.fixture(), self.config())
        before = c.learn_domain(frame, c.np.full(len(frame), .2), .05, self.config())
        copies = c.pd.concat([frame] * 3, ignore_index=True)
        after = c.learn_domain(copies, c.np.full(len(copies), .2), .05, self.config())
        self.assertEqual(before['donor_counts'], after['donor_counts'])

    def test_calibration_and_nonlinear_model_are_grouped(self):
        c = self.core
        frame = c.prepare_analysis(self.fixture(), self.config())
        spec = {'kind': 'hgb', 'leaves': 7, 'iterations': 10}
        bundle = c.fit_propensity(frame, self.config(), c.NUMERIC_FEATURES, frozen_spec=spec)
        self.assertTrue(c.np.isfinite(bundle['oof']).all())
        self.assertIsNotNone(bundle['calibration'])
        self.assertTrue(any('calibration_prediction' in item['stage'] for item in bundle['provenance']))
        for item in bundle['provenance']:
            self.assertFalse(set(item['train_match_ids']) & set(item['test_match_ids']))
        model = c.feature_pipeline(frame, c.NUMERIC_FEATURES, {'kind': 'hgb', 'leaves': 7, 'iterations': 10})
        self.assertFalse(model.named_steps['model'].early_stopping)

    def test_missing_features_and_small_groups_fail_closed(self):
        c = self.core
        with self.assertRaises(ValueError):
            c.prepare_analysis(self.fixture().drop(columns='distance'), self.config())
        frame = c.prepare_analysis(self.fixture().iloc[:40], self.config())
        with self.assertRaises(ValueError):
            c.grouped_folds(frame, 3, 1)


if __name__ == '__main__':
    unittest.main()
