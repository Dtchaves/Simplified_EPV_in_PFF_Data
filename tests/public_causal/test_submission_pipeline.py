"""Submission tests exercise the notebook's exported source, with tiny fixtures."""
from pathlib import Path
import sys, tempfile, unittest, json
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from PublicCausal.notebook_runtime import load_notebook_core

class SubmissionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.c = load_notebook_core(ROOT / 'results/models/public_statsbomb_causal_study.ipynb', cls.temp.name)
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def test_submission_contract_is_present(self):
        required = ['execute_submission_study', 'verify_development_import', 'SubmissionTaskPool',
                    'submission_scenarios', 'submission_simulation_pair', 'submission_null_frame',
                    'submission_manifest', 'submission_selected_truth', 'benchmark_submission_workload']
        self.assertEqual([name for name in required if not hasattr(self.c, name)], [])
    def test_independent_pairs_and_selected_truth(self):
        c = self.c
        scenarios = c.submission_scenarios()
        self.assertEqual(len(scenarios), 8)
        self.assertEqual(len({s['name'] for s in scenarios}), 8)
        design, confirm = c.submission_simulation_pair(scenarios[-1], 4, n_matches=4, decisions_per_match=20)
        self.assertTrue(set(design.original_match_id).isdisjoint(confirm.original_match_id))
        rows = confirm.copy(); rows['Treatment'] = rows['T']; rows['support'] = rows.eligible
        expected = (rows.loc[rows.support & rows.Treatment.eq(1), 'true_y1'] -
                    rows.loc[rows.support & rows.Treatment.eq(1), 'true_y0']).mean()
        self.assertAlmostEqual(c.submission_selected_truth(rows), expected)
    def fixture(self):
        return self.c.pd.DataFrame({'T':[1,0,1,0,0,0], 'Treatment':[1,0,1,0,0,0],
            'Y':[1,0,0,-1,0,1], 'original_match_id':['a','a','b','b','b','a'],
            'match_id':['a','a','b','b','b','a'], 'decision_row_id':list('abcdef'),
            'e':[.2]*6, 'portable_e':[.3]*6,'m0':[.1]*6,'m1':[.1]*6,
            'support':[True]*6,'outer_fold':[0,0,1,1,1,0], 'x':[1.,2.,3.,4.,5.,6.]})
    def test_null_reuse_only_for_dummy_outcome(self):
        c = self.c; rows = self.fixture()
        randomized = c.submission_null_frame(rows, 'within_match_treatment', 12)
        self.assertEqual(randomized.groupby('original_match_id').T.sum().to_dict(), rows.groupby('original_match_id').T.sum().to_dict())
        self.assertNotIn('e', randomized)
        dummy = c.submission_null_frame(rows, 'independent_dummy_y', 13)
        self.assertTrue(dummy.e.equals(rows.e)); self.assertTrue(dummy.support.equals(rows.support))
        self.assertNotIn('m0', dummy)
        self.assertTrue(set(dummy.Y.unique()).issubset({-1,0,1}))
    def test_checkpoint_resume_strict_and_per_completion(self):
        c = self.c; calls = []
        def task(spec):
            calls.append(spec['seed']); return {'status':'ok','att':spec['seed'], 'diagnostics':{'passed':False}}
        jobs = [{'task_id':str(i),'seed':i} for i in range(4)]
        with tempfile.TemporaryDirectory() as d:
            with c.SubmissionTaskPool(d, {'source':'a','config':'b'}, workers=2, executor_kind='thread') as pool:
                records = pool.run('probe', jobs, task)
            self.assertEqual(len(records), 4); self.assertEqual(len(list((Path(d)/'probe').glob('*.json'))), 4)
            with c.SubmissionTaskPool(d, {'source':'a','config':'b'}, workers=2, executor_kind='thread') as pool:
                self.assertEqual(len(pool.run('probe', jobs, task)), 4)
            self.assertEqual(len(calls), 4)
            with self.assertRaisesRegex(ValueError, 'identity|provenance|incompatible'):
                c.SubmissionTaskPool(d, {'source':'changed','config':'b'}, workers=2, executor_kind='thread')
            p = next((Path(d)/'probe').glob('*.json')); value=json.loads(p.read_text()); value['result']['att']=999; p.write_text(json.dumps(value))
            with c.SubmissionTaskPool(d, {'source':'a','config':'b'}, workers=2, executor_kind='thread') as pool:
                with self.assertRaisesRegex(ValueError, 'checksum|corrupt'):
                    pool.run('probe', jobs, task)
    def test_worker_budget_and_stop_drains_admitted(self):
        c=self.c
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): c.SubmissionTaskPool(d, {'source':'a'}, workers=0)
            stop=Path(d)/'STOP'; calls=[]
            def task(spec):
                calls.append(spec['seed']); stop.write_text('stop'); return {'status':'ok'}
            with c.SubmissionTaskPool(d, {'source':'a'}, workers=1, executor_kind='thread', stop_path=stop) as pool:
                got=pool.run('probe', [{'task_id':str(i),'seed':i} for i in range(5)], task)
            self.assertEqual(len(calls),1); self.assertEqual(len(got),1)
    def test_manifest_counts_and_engineering_lock_refusal(self):
        c=self.c
        frozen={'model_specs':{'e':{'kind':'logistic','C':.2},'m0':{'kind':'ridge','alpha':1}},
                'domain':{'threshold':.05,'cells':[1]},'training_match_ids':['e1']}
        cfg=c.scientific_config({'profile':'submission','workers':2})
        manifest=c.submission_manifest(cfg,frozen)
        for name in ['horizon','model','backend','support','subgroup','inference','simulation','budget']:
            self.assertIn(name,manifest)
        self.assertEqual(manifest['simulation']['scenarios'],[s['name'] for s in c.submission_scenarios()])
        self.assertEqual(manifest['budget']['bootstrap_successes'],200)
        self.assertEqual(manifest['budget']['null_per_kind'],50)
        self.assertEqual(manifest['budget']['simulation_per_scenario'],50)
        with self.assertRaisesRegex(PermissionError,'engineering|smoke|pilot'):
            c.submission_lock_payload({'phase':'pilot'}, cfg, frozen, {k:True for k in ['timing_audit','leakage_tests','development_diagnostics','weighted_validation','design_sanity']}, {})
    def test_bounded_analysis_allocation_and_outcome_reuse(self):
        c=self.c; cfg=c.scientific_config({'profile':'submission'})
        specs=c.submission_analysis_specs(cfg)
        kinds=[s['kind'] for s in specs]
        self.assertEqual(kinds.count('outcome_model'),3)
        self.assertEqual(kinds.count('horizon'),3)
        self.assertEqual(kinds.count('population'),2)
        self.assertEqual(kinds.count('ablation'),2)
        baseline={'rows':self.fixture()}; frozen={'model_specs':{'m0':{'kind':'ridge','alpha':1}}}
        with patch.object(c,'refit_on_fixed_population',return_value=(self.fixture(),[])) as fitter:
            c.submission_outcome_refit(baseline['rows'],cfg,frozen,new_outcome=[0]*6)
        self.assertEqual(fitter.call_args.kwargs['outcome_spec'],frozen['model_specs']['m0'])
        self.assertTrue(fitter.call_args.args[0].e.equals(baseline['rows'].e))
        self.assertTrue(fitter.call_args.args[0].support.equals(baseline['rows'].support))
    def test_verified_import_rejects_prediction_and_checksum_corruption(self):
        c=self.c
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'exports').mkdir(); rows=self.fixture()
            fit={'rows':rows,'provenance':[],'design':{'config':{'feature_names':['x']}},'diagnostics':{'passed':True},'model_validation':[]}
            c.atomic_joblib(fit,p/'fit.joblib'); provenance={'core_hash':'a'*64,'helper_hashes':{},'environment_hash':'b'*64,'config':{'feature_names':['x']},'data_hash':'c'*64}
            c.atomic_write_json(p/'provenance.json',provenance)
            c.atomic_write_json(p/'fit.sha256.json',{'key':c.stable_hash(provenance),'sha256':c.file_hash(p/'fit.joblib')})
            c.atomic_parquet(rows,p/'exports/nuisance_rows.parquet'); c.atomic_parquet(rows,p/'candidate_decisions.parquet')
            fit2,manifest=c.verify_development_import(p, original_source_inventory={'core_hash':'a'*64,'helper_hashes':{}})
            self.assertEqual(len(fit2['rows']),6); self.assertEqual(manifest['purpose'],'baseline_development_evidence')
            changed=rows.copy(); changed.loc[0,'m0']=.9; c.atomic_parquet(changed,p/'exports/nuisance_rows.parquet')
            with self.assertRaisesRegex(ValueError,'prediction|scalar|identity'):
                c.verify_development_import(p,original_source_inventory={'core_hash':'a'*64,'helper_hashes':{}})
            with open(p/'fit.joblib','ab') as f: f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                c.verify_development_import(p,original_source_inventory={'core_hash':'a'*64,'helper_hashes':{}})


    def test_bootstrap_summary_retains_poor_finite_and_precision_is_sign_independent(self):
        c=self.c
        values=c.np.random.default_rng(11).normal(0,.02,200)
        records=[{'status':'ok','att':float(x),'diagnostics':{'passed':False}} for x in values]
        summary=c.submission_bootstrap_summary(records,requested=200,seed=2)
        self.assertEqual(summary['successful'],200)
        self.assertEqual(summary['diagnostic_poor'],200)
        shifted=[dict(r,att=r['att']+1) for r in records]
        other=c.submission_bootstrap_summary(shifted,requested=200,seed=2)
        self.assertAlmostEqual(summary['endpoint_mcse'][0],other['endpoint_mcse'][0])
        self.assertEqual(summary['precision_passed'],other['precision_passed'])
    def test_bounded_real_tasks_fit_only_changed_components(self):
        c=self.c; rows=self.fixture(); rows['Y5']=rows.Y; rows['contains_carry']=[True,False]*3
        cfg=c.scientific_config({'profile':'submission','feature_names':['x'],'support_features':['x']})
        frozen={'model_specs':{'e':{'kind':'logistic','C':.2},'m0':{'kind':'ridge','alpha':1}}}
        spec={'task_id':'h','kind':'horizon','outcome':'Y5','name':'Y5','seed':1}
        with patch.object(c,'submission_outcome_refit',return_value=(rows,[])) as out, patch.object(c,'support_diagnostics',return_value={'passed':True}):
            record=c.submission_analysis_task(spec,rows,cfg,frozen)
        self.assertEqual(record['status'],'ok'); self.assertTrue(out.called)
        spec={'task_id':'p','kind':'population','name':'carry_excluded','seed':1}
        with patch.object(c,'fit_study',return_value={'rows':rows,'diagnostics':{'passed':True}}) as fit:
            record=c.submission_analysis_task(spec,rows,cfg,frozen)
        self.assertEqual(len(fit.call_args.args[0]),3)
        self.assertEqual(record['population'],'changed_selected_population')
    def test_subgroup_insufficiency_never_promoted_to_interval(self):
        c=self.c; rows=self.fixture(); rows['distance']=[10,20]*3; rows['angle']=[20,40]*3; rows['nearest_defender']=[1,2]*3
        cfg=c.scientific_config({'profile':'submission','feature_names':['x'],'support_features':['x'],'key_balance_features':['x']})
        result=c.submission_subgroups(rows,cfg,numerical_draws=100)
        self.assertEqual(len(result['strata']),6)
        self.assertTrue(all(not r['sufficient'] for r in result['strata']))
        self.assertTrue(all(r.get('simultaneous_interval') is None for r in result['strata']))
    def test_claim_completion_requires_all_declared_checks(self):
        c=self.c
        status=c.submission_claim_status({'bootstrap':False,'nulls':True,'simulations':True,'analyses':True,'support':True,'precision':True,'inference_agreement':True})
        self.assertNotEqual(status['status'],'completed')
        self.assertFalse(status['primary_claim'])

if __name__=='__main__': unittest.main()
