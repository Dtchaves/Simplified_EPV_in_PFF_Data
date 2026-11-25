"""Submission confirmation orchestration on tiny synthetic cohorts (no WC data)."""
from pathlib import Path
import sys, tempfile, unittest, json
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from PublicCausal.notebook_runtime import load_notebook_core, RunLogger


class SubmissionConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.c = load_notebook_core(ROOT / 'results/models/public_statsbomb_causal_study.ipynb', cls.temp.name)
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def config(self, **extra):
        return self.c.scientific_config(dict(profile='submission', outer_folds=3, inner_folds=2,
            min_shots=1, min_matches=1, min_ess=1, donor_min=1, donor_match_min=1,
            key_smd_max=100, other_smd_max=100, max_weight_share=1, max_match_weight_share=1,
            e_candidates=[{'kind': 'logistic', 'C': 1}], outcome_candidates=[{'kind': 'ridge', 'alpha': 10}],
            **extra))

    def cohort(self, seed, n_matches=12, decisions=40):
        c = self.c
        frame = c.simulate_causal_matches(c.submission_scenarios()[1], seed, n_matches, decisions)
        rng = c.np.random.default_rng(seed)
        frame['Y15'] = frame.Y
        for name in ['Y5', 'Y10', 'Y30', 'Y15_shot_start']:
            frame[name] = frame.Y
        frame['contains_carry'] = rng.uniform(size=len(frame)) < .3
        frame['visible_disk_5m'] = rng.uniform(size=len(frame)) < .8
        frame['drop_reason'] = None
        return frame

    def frozen(self, config):
        c = self.c
        design = c.fit_frozen_design(self.cohort(101), config)
        design['model_specs'] = dict(e={'kind': 'logistic', 'C': 1}, m0={'kind': 'ridge', 'alpha': 10})
        return design

    def test_score_influence_equals_exact_fixed_nuisance_removal(self):
        c = self.c; cfg = self.config()
        rows = c.fit_study(self.cohort(7), cfg, frozen_design=self.frozen(cfg))['rows']
        screen = c.submission_score_influence(rows, top_matches=5, top_teams=2)
        base = c.att_estimate(rows)['att']
        for record in screen['matches'][:2]:
            kept = rows.loc[rows.original_match_id != record['original_match_id']]
            self.assertAlmostEqual(record['att_without'], c.att_estimate(kept)['att'], places=12)
            self.assertAlmostEqual(record['delta'], record['att_without'] - base, places=12)
        team = screen['teams'][0]
        kept = rows.loc[(rows.team_id.astype(str) != team['team']) & (rows.opponent_id.astype(str) != team['team'])]
        self.assertAlmostEqual(team['att_without'], c.att_estimate(kept)['att'], places=12)
        deltas = [abs(r['delta']) for r in screen['matches']]
        self.assertEqual(deltas, sorted(deltas, reverse=True))
        self.assertEqual(len(screen['targeted_matches']), 5)
        self.assertEqual(len(screen['targeted_teams']), 2)
        self.assertIn('not complete', screen['interpretation'])

    def test_score_influence_with_integer_match_ids(self):
        # Real StatsBomb match IDs are integers; the screen must still remove them.
        c = self.c; cfg = self.config()
        rows = c.fit_study(self.cohort(8), cfg, frozen_design=self.frozen(cfg))['rows']
        rows['original_match_id'] = rows.original_match_id.str.replace('sim_', '').astype(int) + 3788000
        screen = c.submission_score_influence(rows, top_matches=5, top_teams=2)
        top = screen['matches'][0]
        self.assertGreater(top['removed_treated'], 0)
        kept = rows.loc[rows.original_match_id.astype(str) != top['original_match_id']]
        self.assertAlmostEqual(top['att_without'], c.att_estimate(kept)['att'], places=12)
        self.assertGreater(abs(top['delta']), 1e-9)

    def test_tasks_with_live_worker_logging_keep_results(self):
        # Production sets log_run_dir, so every task closes a real WorkerProgressLogger.
        c = self.c
        with tempfile.TemporaryDirectory() as d:
            cfg = self.config(log_run_dir=d, run_id='logging')
            frozen = self.frozen(cfg)
            rows = c.fit_study(self.cohort(9), cfg, frozen_design=frozen)['rows']
            records = [
                c.submission_null_task(dict(kind='null', null_kind='independent_dummy_y', seed=3, replicate=0), rows, cfg, frozen),
                c.submission_null_task(dict(kind='null', null_kind='within_match_treatment', seed=4, replicate=0), rows, cfg, frozen),
                c.submission_simulation_task(dict(kind='simulation', scenario=c.submission_scenarios()[0], seed=5, replicate=0,
                                                  n_matches=12, decisions_per_match=40), cfg, frozen),
                c.submission_analysis_task(dict(task_id='h5', kind='horizon', name='Y5', outcome='Y5', seed=6), rows, cfg, frozen)]
            self.assertEqual([r['status'] for r in records], ['ok'] * 4, [r.get('failure') for r in records])
            ends = [json.loads(p.read_text()) for p in (Path(d) / 'worker_status').glob('*.json')]
            self.assertTrue(ends and all(e['event'] == 'worker_end' for e in ends))

    def test_bootstrap_success_prefix_is_resume_invariant(self):
        c = self.c
        records = [{'status': 'ok' if i % 4 else 'failed', 'att': float(i)} for i in range(300)]
        prefix = c._submission_success_prefix(records, 200)
        self.assertEqual(sum(r['status'] == 'ok' for r in prefix), 200)
        self.assertEqual(prefix, records[:len(prefix)])
        # A resumed stage that cached extra draws judges precision on the identical prefix.
        self.assertEqual(c._submission_success_prefix(records + records[:5], 200), prefix)

    def test_refit_failure_is_categorized(self):
        c = self.c; cfg = self.config()
        with patch.object(c, 'fit_study', side_effect=ValueError('insufficient arm')):
            record = c.submission_refit_task(dict(task_id='r', seed=1), self.cohort(3), cfg, {})
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['failure_category'], 'structural')

    def test_worker_crash_is_computational_and_not_persisted(self):
        c = self.c
        with tempfile.TemporaryDirectory() as d:
            with patch.object(c, '_submission_task_call', side_effect=MemoryError('killed')):
                with c.SubmissionTaskPool(d, {'s': 2}, workers=1, executor_kind='thread') as pool:
                    got = pool.run('crash', [{'task_id': '0', 'seed': 0}], lambda s: {'status': 'ok'})
            self.assertEqual(got[0]['failure_category'], 'computational')
            self.assertEqual(list((Path(d) / 'crash').glob('*.json')), [])

    def test_failed_records_do_not_block_eta_and_empty_scenario_fails_claim(self):
        c = self.c
        with tempfile.TemporaryDirectory() as d:
            progress = Path(d) / 'p.json'
            with c.SubmissionTaskPool(Path(d) / 't', {'s': 3}, workers=1, executor_kind='thread', progress_path=progress) as pool:
                pool.plan('nulls', 2, 'null')
                pool.run('nulls', [{'task_id': str(i), 'seed': i} for i in range(2)], lambda s: {'status': 'failed', 'failure_category': 'structural'})
            self.assertEqual(json.loads(progress.read_text())['remaining_tasks'], 0)
        summary = dict(attempted=2, requested=2, failure_categories=dict(structural=1, computational=0),
                       by_scenario=dict(a=dict(successful=1), b=dict(successful=0)))
        self.assertFalse(c._submission_simulation_check(summary))
        summary['by_scenario']['b']['successful'] = 1
        self.assertTrue(c._submission_simulation_check(summary))

    def test_inference_agreement_is_predeclared_ratio(self):
        c = self.c
        self.assertTrue(c.submission_inference_agreement(.01, dict(sd=.012, successful=200))['passed'])
        self.assertFalse(c.submission_inference_agreement(.01, dict(sd=.03, successful=200))['passed'])
        self.assertFalse(c.submission_inference_agreement(.01, dict(sd=None, successful=0))['passed'])

    def test_overall_progress_reports_remaining_work_and_measured_eta(self):
        c = self.c
        with tempfile.TemporaryDirectory() as d:
            progress = Path(d) / 'overall_progress.json'
            with c.SubmissionTaskPool(Path(d) / 'tasks', {'s': 1}, workers=2, executor_kind='thread',
                                      progress_path=progress) as pool:
                pool.plan('first', 2, 'refit'); pool.plan('second', 2, 'refit')
                pool.run('first', [{'task_id': str(i), 'seed': i} for i in range(2)], lambda s: {'status': 'ok'})
                middle = json.loads(progress.read_text())
                self.assertEqual(middle['remaining_tasks'], 2)
                self.assertIsNotNone(middle['eta_seconds'])
                pool.run('second', [{'task_id': str(i), 'seed': i} for i in range(2)], lambda s: {'status': 'ok'})
            final = json.loads(progress.read_text())
            self.assertEqual(final['remaining_tasks'], 0)
            self.assertEqual(final['stages']['second']['completed'], 2)
            self.assertIn('method', final)

    def runtime(self, d, **extra):
        return dict(dict(run_dir=str(Path(d) / 'confirmation'), run_id='unit-confirmation', phase='confirmation',
                         executor_kind='thread', workers=2), **extra)

    def run_confirmation(self, d, **extra):
        c = self.c
        cfg = self.config(workers=2, bootstrap_target=3, bootstrap_max_attempts=5, null_reps=1,
                          simulation_reps=1, simulation_matches=12, simulation_decisions_per_match=40,
                          bootstrap_precision_extension=False, study_root=str(d))
        frozen = self.frozen(cfg)
        candidates = self.cohort(55)
        frame = c.prepare_analysis(candidates, cfg)
        runtime = self.runtime(d, **extra)
        Path(runtime['run_dir']).mkdir(parents=True, exist_ok=True)
        logger = RunLogger(runtime['run_dir'], runtime['run_id'])
        with patch.object(c, 'check_disk_budget', return_value=None):
            return c._execute_submission_confirmation(runtime, cfg, frozen, frame, candidates, {}, {}, 'core', logger), runtime

    def test_confirmation_runs_declared_bounded_workload_and_honest_claim(self):
        with tempfile.TemporaryDirectory() as d:
            outcome, runtime = self.run_confirmation(d)
            run = Path(runtime['run_dir'])
            self.assertTrue(outcome['confirmation_opened'])
            self.assertIn(outcome['status'], {'completed', 'incomplete', 'inconclusive_uncertainty', 'unsupported'})
            self.assertTrue((run / 'exports/att.json').is_file())
            results = json.loads((run / 'submission_results.json').read_text())
            self.assertEqual(results['bootstrap']['successful'], 3)
            self.assertEqual(results['nulls']['within_match_treatment']['attempted'], 1)
            self.assertEqual(results['nulls']['independent_dummy_y']['attempted'], 1)
            self.assertEqual(results['simulations']['attempted'], 8)
            kinds = [r['kind'] for r in results['analyses']['records']]
            self.assertEqual(kinds.count('horizon'), 3)
            self.assertEqual(kinds.count('targeted_match'), 5)
            self.assertEqual(kinds.count('targeted_team'), 2)
            self.assertEqual(len(results['subgroups']['strata']), 6)
            self.assertIn('bounds', results['sensitivity'])
            claim = results['claim']
            self.assertEqual(claim['primary_claim'], all(claim['checks'].values()))
            self.assertEqual(outcome['claim_status'], claim['status'])
            progress = json.loads((run / 'overall_progress.json').read_text())
            self.assertEqual(progress['remaining_tasks'], 0)
            self.assertTrue((run / 'phase_result.json').is_file())

    def test_confirmation_stop_token_stops_after_primary_exports(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / 'confirmation'; run.mkdir()
            (run / 'STOP_REQUESTED').write_text('{}')
            outcome, runtime = self.run_confirmation(d)
            self.assertEqual(outcome['status'], 'stopped')
            self.assertTrue((run / 'exports/att.json').is_file())
            self.assertFalse((run / 'submission_results.json').exists())


if __name__ == '__main__':
    unittest.main()
