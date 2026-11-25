"""Submission tests execute the notebook-owned scientific definitions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from PublicCausal.notebook_runtime import load_notebook_core


class SubmissionModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.c = load_notebook_core(ROOT / 'results/models/public_statsbomb_causal_study.ipynb', cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fixture(self):
        c = self.c
        n = 120
        rows = c.pd.DataFrame({name: c.np.linspace(0, 1, n) for name in c.NUMERIC_FEATURES})
        rows['distance'], rows['y'], rows['defenders_3m'] = 12., 0., 1
        rows['original_match_id'] = [str(i // 10) for i in range(n)]
        rows['match_id'], rows['event_id'] = rows.original_match_id, [str(i) for i in range(n)]
        rows['T'], rows['Y'] = [i % 2 for i in range(n)], 0.
        for name in c.CATEGORICAL_FEATURES:
            rows[name] = 'test'
        return rows

    def config(self):
        return self.c.scientific_config(dict(profile='submission', outer_folds=3, inner_folds=3,
            min_shots=1, min_matches=2, min_ess=1, donor_min=1, donor_match_min=2,
            key_smd_max=100, other_smd_max=100, max_weight_share=1, max_match_weight_share=1))

    def test_profile_explicit_and_bounded(self):
        cfg = self.config()
        self.assertEqual(cfg['bootstrap_target'], 200)
        self.assertEqual(cfg['bootstrap_max_target'], 500)
        self.assertEqual(cfg['null_reps'], 50)
        self.assertTrue(cfg['att_only'])
        self.assertEqual(len(cfg['e_candidates']), 3)
        self.assertEqual({s['kind'] for s in cfg['e_candidates']}, {'logistic', 'hgb', 'xgb'})
        with self.assertRaises(ValueError):
            self.c.scientific_config({'profile': 'typo'})
        self.assertEqual(self.c.scientific_config()['profile'], 'extended')

    def test_fixed_propensity_and_outcome_fit_counts(self):
        c = self.c
        cfg = self.config()
        rows = c.prepare_analysis(self.fixture(), cfg)
        original = c.feature_pipeline
        with patch.object(c, 'feature_pipeline', wraps=original) as factory:
            bundle = c.fit_propensity(rows, cfg, cfg['feature_names'], {'kind': 'hgb', 'leaves': 3, 'iterations': 3})
            self.assertEqual(factory.call_count, 4)
            self.assertIsNone(bundle['oof'])
            self.assertEqual(bundle['tuning'], [])
            self.assertEqual(len(bundle['provenance']), 3)
            self.assertIsNotNone(bundle['calibration'])
        repeated = c.pd.concat([rows, rows[rows.original_match_id == '0']], ignore_index=True)
        bundle = c.fit_propensity(repeated, cfg, cfg['feature_names'], {'kind': 'hgb', 'leaves': 3, 'iterations': 3})
        for p in bundle['provenance']:
            self.assertFalse(set(p['train_match_ids']) & set(p['test_match_ids']))
        with patch.object(c, 'feature_pipeline', wraps=original) as factory:
            m = c.fit_outcome(rows, cfg, 0, None, {'kind': 'ridge', 'alpha': 10})
            self.assertEqual(factory.call_count, 1)
            self.assertIsNone(m['oof'])
            self.assertEqual(m['tuning'], [])

    def test_fixed_study_skips_m1_and_reuses_portable(self):
        c, cfg = self.c, self.config()
        rows = c.prepare_analysis(self.fixture(), cfg)
        portable = c.fit_propensity(rows, cfg, cfg['support_features'], {'kind': 'logistic', 'C': .2})
        frozen = dict(portable_model=portable, domain=c.learn_domain(rows, c.np.full(len(rows), .5), .05, cfg),
            model_specs={'e': {'kind': 'logistic', 'C': .2}, 'm0': {'kind': 'ridge', 'alpha': 10}, 'm1': {'kind': 'ridge', 'alpha': 10}})
        with patch.object(c, 'fit_outcome', wraps=c.fit_outcome) as fit:
            result = c.fit_study(self.fixture(), cfg, frozen)
            self.assertEqual(fit.call_count, 3)
            self.assertEqual({call.args[2] for call in fit.call_args_list}, {0})
            self.assertTrue(result['rows'].m1.isna().all())
            self.assertEqual(result['m1_status'], 'omitted_att_only')
        self.assertFalse(any(p['stage'].startswith('support') for p in result['provenance']))
        with patch.object(c, 'fit_outcome', wraps=c.fit_outcome) as fit:
            c.fit_study(self.fixture(), dict(cfg, diagnostic_m1=True), frozen)
            self.assertEqual(fit.call_count, 6)

    def score_rows(self):
        return self.c.pd.DataFrame(dict(T=[1,1,0,0,0], Treatment=[1,1,0,0,0],
            Y=[1.,0.,0.,-1.,0.], e=[.2,.4,.2,.4,.5], m0=[.1,.2,.1,.2,0.],
            support=[True,True,True,True,False], original_match_id=['a','b','a','b','c']))

    def test_full_registry_t_interval_hand_calculation(self):
        from scipy.stats import t
        c, rows = self.c, self.score_rows()
        answer = c.att_estimate(rows)
        theta = (.9 - .2 + .025 + .8) / 2
        score_a, score_b = .925-theta, .6-theta
        se = ((3/2)*(score_a**2+score_b**2))**.5 / 2
        self.assertAlmostEqual(answer['att'], theta)
        self.assertEqual(answer['n_matches'], 3)
        self.assertEqual(answer['cluster_scores'].loc['c', 'score'], 0)
        self.assertAlmostEqual(answer['cluster_score_se'], se)
        self.assertEqual(answer['cluster_score_df'], 2)
        self.assertAlmostEqual(answer['score_interval'][0], theta-t.ppf(.975,2)*se)
        diagnostic = c.cluster_score_interval(rows)
        self.assertEqual(diagnostic['clusters'], 3)
        self.assertAlmostEqual(diagnostic['diagnostic_lower'], answer['score_interval'][0])

    def test_score_rejects_degenerate_registry_or_targets(self):
        rows = self.score_rows()
        rows['original_match_id'] = 'one'
        with self.assertRaises(ValueError):
            self.c.att_estimate(rows)
        rows['T'], rows['Treatment'] = 0, 0
        with self.assertRaises(ValueError):
            self.c.att_estimate(rows)

    def test_xgb_explicit_backend_and_fallback_detection(self):
        c = self.c
        rows = c.prepare_analysis(self.fixture(), self.config())
        spec = {'kind': 'xgb', 'device': 'cpu', 'leaves': 3, 'iterations': 3}
        model = c.feature_pipeline(rows, ['distance'], spec, 'propensity')
        self.assertEqual(model.named_steps['model'].get_params()['device'], 'cpu')
        fitted = c.fit_checked(model, rows[['distance']], rows.Treatment, spec)
        self.assertEqual(fitted.named_steps['model'].get_params()['tree_method'], 'hist')
        self.assertIsNone(fitted.named_steps['model'].get_params().get('early_stopping_rounds'))
        class Booster:
            def save_config(self):
                return json.dumps({'learner': {'generic_param': {'device': 'cpu'}}})
        class FakeModel:
            def get_booster(self): return Booster()
        class FakePipeline:
            named_steps = {'model': FakeModel()}
            def fit(self, *args): return self
        with self.assertRaisesRegex(RuntimeError, 'CUDA|cuda'):
            c.fit_checked(FakePipeline(), rows[['distance']], rows.Treatment, dict(spec, device='cuda'))


    def test_xgb_grouped_calibration_and_backend_provenance(self):
        c, cfg = self.c, self.config()
        rows = c.prepare_analysis(self.fixture(), cfg)
        spec = {'kind': 'xgb', 'device': 'cpu', 'leaves': 3, 'iterations': 3}
        with patch.object(c, 'feature_pipeline', wraps=c.feature_pipeline) as factory:
            bundle = c.fit_propensity(rows, cfg, ['distance'], spec)
            self.assertEqual(factory.call_count, 4)
        self.assertIsNotNone(bundle['calibration'])
        self.assertEqual(len(bundle['provenance']), 3)
        self.assertEqual(bundle['model'].backend_manifest['actual_device'], 'cpu')
        self.assertIn('booster_config', bundle['model'].backend_manifest)
        self.assertIn('xgboost_version', bundle['model'].backend_manifest)
        for p in bundle['provenance']:
            self.assertFalse(set(p['train_match_ids']) & set(p['test_match_ids']))

    def test_registry_reports_eligible_supported_treated_matches(self):
        c, rows = self.c, self.score_rows()
        rows['eligible'] = True
        excluded = rows.iloc[[-1]].copy()
        excluded['original_match_id'], excluded['eligible'] = 'ineligible', False
        result = c.att_estimate(c.pd.concat([rows, excluded], ignore_index=True))
        self.assertEqual(result['eligible_matches'], 3)
        self.assertEqual(result['support_bearing_matches'], 2)
        self.assertEqual(result['treated_bearing_matches'], 2)
        self.assertNotIn('ineligible', result['cluster_scores'].index)

    def test_fixed_outcome_sensitivity_preserves_e_and_population(self):
        c, cfg = self.c, self.config()
        rows = c.prepare_analysis(self.fixture(), cfg)
        rows['e'], rows['support'], rows['m0'] = .5, True, 0.
        rows['outer_fold'] = rows.original_match_id.astype(int) % 3
        with patch.object(c, 'fit_propensity', side_effect=AssertionError('Outcome-only sensitivity must reuse e')):
            result, provenance = c.refit_on_fixed_population(rows, cfg, {'kind': 'ridge', 'alpha': 10})
        c.np.testing.assert_array_equal(result.e, rows.e)
        self.assertEqual(result.loc[result.support, 'decision_row_id'].tolist(), rows.decision_row_id.tolist())
        self.assertEqual(len(provenance), 3)


    def test_cuda_dmatrix_prediction_warning_is_recorded_and_visible(self):
        import warnings
        c = self.c
        class FakePipeline:
            backend_manifest = dict(requested_device='cuda', actual_device='cuda:0')
            def predict_proba(self, frame):
                warnings.warn('Falling back to prediction using DMatrix due to mismatched devices', UserWarning)
                return c.np.array([[.7, .3]])
        class Logger:
            def __init__(self): self.events = []
            def event(self, name, **fields): self.events.append((name, fields))
        model, logger = FakePipeline(), Logger()
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            result = c.predict_checked(model, c.pd.DataFrame({'x': [0]}), method='predict_proba',
                                       logger=logger, stage='e_prediction')
        c.np.testing.assert_allclose(result, [[.7, .3]])
        self.assertEqual(model.backend_manifest['actual_device'], 'cuda:0')
        self.assertIn('DMatrix', model.backend_manifest['prediction_warnings'][0])
        self.assertEqual(model.backend_manifest['prediction_calls'], 1)
        self.assertTrue(any('DMatrix' in str(item.message) for item in observed))
        self.assertEqual(logger.events[0][0], 'xgb_prediction')
        self.assertEqual(logger.events[0][1]['actual_training_device'], 'cuda:0')
        self.assertIn('DMatrix', logger.events[0][1]['warnings'][0])

if __name__ == '__main__':
    unittest.main()


class PropensityBackendTieBreakTests(unittest.TestCase):
    """Predeclared one-SE rule: prefer the explicit CUDA candidate only when it is statistically tied."""
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.c = load_notebook_core(ROOT / 'results/models/public_statsbomb_causal_study.ipynb', cls.temp.name)
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def results(self, xgb_folds):
        hgb = [.110, .118, .125]
        return [dict(spec={'kind': 'logistic', 'C': .2}, loss=.1275, fold_losses=[.12, .13, .133]),
                dict(spec={'kind': 'hgb', 'leaves': 7, 'iterations': 100}, loss=sum(hgb) / 3, fold_losses=hgb),
                dict(spec={'kind': 'xgb', 'device': 'cuda', 'leaves': 7, 'iterations': 100},
                     loss=sum(xgb_folds) / 3, fold_losses=xgb_folds)]
    def test_tied_cuda_candidate_is_selected_with_recorded_rule(self):
        c = self.c
        cfg = c.scientific_config({'profile': 'submission', 'candidate_device': 'cuda'})
        chosen, rule = c.select_propensity_candidate(self.results([.1102, .1182, .1251]), cfg)
        self.assertEqual(chosen['spec']['kind'], 'xgb')
        self.assertEqual(rule['applied'], 'one_se_prefer_cuda')
        self.assertLessEqual(chosen['loss'], rule['lowest_loss'] + rule['lowest_loss_se'])
    def test_clearly_worse_cuda_candidate_is_not_selected(self):
        c = self.c
        cfg = c.scientific_config({'profile': 'submission', 'candidate_device': 'cuda'})
        chosen, rule = c.select_propensity_candidate(self.results([.130, .138, .145]), cfg)
        self.assertEqual(chosen['spec']['kind'], 'hgb')
        self.assertEqual(rule['applied'], 'lowest_loss')
    def test_rule_off_for_cpu_device_and_extended_profile(self):
        c = self.c
        for cfg in [c.scientific_config({'profile': 'submission', 'candidate_device': 'cpu'}),
                    c.scientific_config({'profile': 'extended'})]:
            chosen, rule = c.select_propensity_candidate(self.results([.1102, .1182, .1251]), cfg)
            self.assertEqual(chosen['spec']['kind'], 'hgb')
            self.assertEqual(rule['applied'], 'lowest_loss')
