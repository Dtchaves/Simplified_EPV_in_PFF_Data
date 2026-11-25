"""Independent production-fitting simulation and null contracts."""
from pathlib import Path
import sys, tempfile, unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from PublicCausal.notebook_runtime import load_notebook_core

class SubmissionScienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.c = load_notebook_core(ROOT/'results/models/public_statsbomb_causal_study.ipynb', cls.temp.name)
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def config(self):
        return self.c.scientific_config(dict(profile='submission', outer_folds=3, inner_folds=2,
            min_shots=1,min_matches=1,min_ess=1,donor_min=1,donor_match_min=1,
            key_smd_max=100,other_smd_max=100,max_weight_share=1,max_match_weight_share=1))
    def frozen(self):
        return dict(model_specs=dict(e={'kind':'logistic','C':1},m0={'kind':'ridge','alpha':10}))
    def test_independent_task_uses_production_fit_and_actual_interval(self):
        c,cfg=self.c,self.config()
        spec=dict(kind='simulation',scenario=c.submission_scenarios()[1],seed=19,replicate=0,
                  n_matches=12,decisions_per_match=40)
        with patch.object(c,'fit_study',wraps=c.fit_study) as fit, patch.object(c,'fit_outcome',wraps=c.fit_outcome) as outcome:
            result=c.submission_simulation_task(spec,cfg,self.frozen())
        self.assertEqual(result['status'],'ok',result)
        self.assertEqual(fit.call_count,1)
        call=fit.call_args
        design=call.kwargs['frozen_design']
        self.assertTrue(set(design['training_match_ids']).isdisjoint(call.args[0].original_match_id))
        self.assertEqual(outcome.call_count,3)
        self.assertEqual({v.args[2] for v in outcome.call_args_list},{0})
        self.assertEqual(result['ci_kind'],'conditional_approximate_match_cluster_t')
        self.assertTrue(result['independent_design_confirmation'])
        self.assertEqual(len(result['score_interval']),2)
        self.assertTrue(c.np.isfinite([result['att'],result['truth']]).all())
        self.assertEqual(result['covered'],result['score_interval'][0]<=result['truth']<=result['score_interval'][1])
    def baseline(self):
        c,cfg=self.c,self.config()
        frame=c.simulate_causal_matches(c.submission_scenarios()[1],27,12,40)
        rows=c.prepare_analysis(frame,cfg)
        portable=c.fit_propensity(rows,cfg,cfg['support_features'],{'kind':'logistic','C':.2})
        domain=c.learn_domain(rows,c.np.full(len(rows),.2),.05,cfg)
        frozen=dict(self.frozen(),portable_model=portable,domain=domain,training_match_ids=['independentEuro'])
        return c.fit_study(frame,cfg,frozen)['rows'],frozen
    def test_dummy_null_reuses_e_support_and_refits_only_m0(self):
        c,cfg=self.c,self.config(); rows,frozen=self.baseline()
        with (patch.object(c,'fit_propensity',side_effect=AssertionError('dummy null must reuse e')),
              patch.object(c,'refit_on_fixed_population',wraps=c.refit_on_fixed_population) as fit):
            result=c.submission_null_task(dict(kind='null',null_kind='independent_dummy_y',seed=4,replicate=0),rows,cfg,frozen)
        self.assertEqual(result['status'],'ok',result)
        self.assertEqual(fit.call_count,1)
        self.assertEqual(fit.call_args.kwargs['outcome_spec'],frozen['model_specs']['m0'])
        self.assertTrue(fit.call_args.args[0].e.equals(rows.e))
        self.assertTrue(fit.call_args.args[0].support.equals(rows.support))
    def test_treatment_null_refits_e_and_records_known_assignment(self):
        c,cfg=self.c,self.config();rows,frozen=self.baseline()
        with patch.object(c,'fit_study',wraps=c.fit_study) as fit:
            result=c.submission_null_task(dict(kind='null',null_kind='within_match_treatment',seed=5,replicate=0),rows,cfg,frozen)
        self.assertEqual(result['status'],'ok',result)
        self.assertEqual(fit.call_count,1)
        self.assertNotIn('e',fit.call_args.args[0])
        self.assertIn('known_assignment_att',result)
    def test_summary_retains_diagnostic_poor_finite_draw_and_failures(self):
        records=[dict(status='ok',att=.01,truth=0.,covered=True,reject_zero=False,diagnostics={'passed':False}),
                 dict(status='ok',att=-.01,truth=0.,covered=True,reject_zero=False,diagnostics={'passed':True}),
                 dict(status='failed',failure='structural')]
        summary=self.c.submission_science_summary(records,requested=3,kind='simulation')
        self.assertEqual(summary['successful'],2)
        self.assertEqual(summary['failed'],1)
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['diagnostic_poor_successes'],1)
        self.assertAlmostEqual(summary['bias'],0.)
        self.assertAlmostEqual(summary['rmse'],.01)
        self.assertIn('coverage_mc_interval',summary)
        self.assertEqual(summary['requested'],3)

if __name__=='__main__':unittest.main()
