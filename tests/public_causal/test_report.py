"""Results-report cells read saved artifacts only (synthetic run directory, no WC data)."""
from pathlib import Path
import copy, hashlib, json, os, subprocess, sys, tempfile, unittest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd
from PublicCausal.notebook_runtime import export_notebook_core

NOTEBOOK = ROOT / 'results/models/public_statsbomb_causal_study.ipynb'
REPORT_TAG = 'public-causal-report'


def report_cells(notebook):
    return [cell for cell in notebook['cells'] if REPORT_TAG in cell.get('metadata', {}).get('tags', [])]


def load_report_namespace():
    notebook = json.loads(NOTEBOOK.read_text())
    definitions = [cell for cell in report_cells(notebook) if 'report-definitions' in cell['metadata']['tags']]
    assert len(definitions) == 1, 'exactly one report-definitions cell'
    namespace = {}
    exec(compile(''.join(definitions[0]['source']), 'report-definitions', 'exec'), namespace)
    return namespace


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_run(base):
    """Small artifact tree with the saved-result schema of a completed full run."""
    rng = np.random.default_rng(3)
    full = base / 'full'
    dev, conf = full / 'development', full / 'confirmation'
    science = dict(min_shots=200, min_matches=30, min_ess=200, ess_shot_fraction=.25, key_smd_max=.10,
                   other_smd_max=.15, max_weight_share=.01, max_match_weight_share=.10,
                   key_balance_features=['distance', 'angle'], support_features=['distance', 'angle'],
                   feature_names=['distance', 'angle', 'team_id'])
    for phase in (dev, conf):
        write_json(phase / 'config.json', dict(runtime=dict(run_id='fixture-run'), science=science))
        write_json(phase / 'timing_audit.json', dict(eligible=900, matches=12, passed=True, limitation='Start-labelled snapshot timing',
                   exclusions={'': 900, 'outside_region': 400, 'outside_region|missing_360': 50, 'direct_restart': 30}))
    write_json(dev / 'development_gates.json', dict(timing_audit=True, leakage_tests=True, development_diagnostics=True,
                                                    weighted_validation=True, design_sanity=True))
    write_json(dev / 'design_lock.json', dict(status='frozen', confirmation_allowed=True, profile='submission',
               core_source_hash='c' * 64, frozen_at='2026-09-30T21:51:00Z', development_gates=dict(timing_audit=True),
               specification_manifest=dict(
                   model=dict(specifications=dict(e={'kind': 'xgb', 'device': 'cuda'}, m0={'kind': 'ridge', 'alpha': 10}),
                              propensity_selection='lowest grouped log-loss', outer_folds=5),
                   horizon=dict(primary='Y15', seconds=15), backend=dict(implementation='xgb'),
                   inference=dict(primary='conditional approximate match-cluster t score CI'),
                   support=dict(domain=dict(threshold=.1)), subgroup=dict(strata=[['distance', 18]]),
                   simulation=dict(n_matches=48), budget=dict(bootstrap_target=200))))
    write_json(dev / 'weighted_validation.json', dict(passed=True, calibration=[dict(bin='(0, 1]', n=10, residual=.01, se=.02, passed=True)]))
    write_json(dev / 'phase_result.json', dict(phase='development', status='completed', claim_status='not_started'))
    write_json(conf / 'phase_result.json', dict(phase='confirmation', status='completed', claim_status='completed'))
    point = dict(att=.03, att_per_100_supported_decisions=3., cluster_score_se=.012, cluster_score_df=11,
                 score_interval=[.005, .055], n_shots=120, n_controls=780, n_matches=12, control_ess=300.,
                 hajek_aipw_att=.0301, max_donor_weight_share=.005, max_match_weight_share=.04)
    write_json(conf / 'exports/att.json', point)
    checks = dict(support=True, bootstrap=True, precision=True, inference_agreement=True, nulls=True, simulations=True, analyses=True)
    write_json(conf / 'submission_results.json', dict(
        claim=dict(status='completed', checks=checks, failed_or_missing=[], qualification='Conditional approximate inference'),
        point=point, units='signed first-goal reward per supported decision', provenance_hash='p' * 64,
        inference_agreement=dict(passed=True, ratio=1.1, rule='0.667<=ratio<=1.5')))
    write_json(conf / 'exports/support_diagnostics.json', dict(
        passed=True, gates=dict(shot_count=True, key_balance=True), n_shots=120, n_controls=780, control_ess=300.,
        key_max_smd=.05, other_max_smd=.08, max_weight_share=.005, max_match_weight_share=.04, shot_retention=.95,
        balance={'distance': .04, 'angle': .05, 'team_id=1': .02}))
    n = 900
    T = (rng.uniform(size=n) < .13).astype(int)
    e = np.clip(.05 + .5 * T * rng.uniform(size=n), .001, .95)
    nuisance = pd.DataFrame(dict(original_match_id=rng.integers(0, 12, n).astype(str), Treatment=T, T=T, e=e,
                                 portable_e=e, m0=rng.normal(.02, .03, n), Y15=rng.choice([-1, 0, 1], n, p=[.05, .85, .1]),
                                 distance=rng.uniform(5, 30, n), angle=rng.uniform(5, 80, n), team_id=rng.integers(0, 3, n),
                                 support=rng.uniform(size=n) < .97, eligible=True))
    (conf / 'exports/att_contributions').mkdir(parents=True, exist_ok=True)
    nuisance.to_parquet(conf / 'exports/nuisance_rows.parquet')
    rows = nuisance.loc[nuisance.support].copy()
    rows['att_weight'] = np.where(rows.Treatment == 1, 1., rows.e / (1 - rows.e))
    rows['att_contribution'] = rng.normal(0, .01, len(rows))
    rows.to_parquet(conf / 'exports/att_contributions/rows.parquet')
    pd.DataFrame(dict(original_match_id=[str(i) for i in range(12)], score=rng.normal(0, .01, 12))).to_parquet(
        conf / 'exports/att_contributions/cluster_scores.parquet')
    pd.DataFrame(dict(estimator=['raw_difference', 'AIPW_count_ATT'], att=[.08, .03], n_shots=[120, 120], n_controls=[780, 780],
                      same_treated_ids=[True, True], inference=['cluster_inference'] * 2)).to_parquet(conf / 'exports/comparator_summary.parquet')
    (conf / 'exports/outcome_validation.csv').write_text('arm,population,n,mse,weighted_mse,residual_mean,weighted_residual\n0,supported,780,.02,.07,-.003,.003\n')
    (conf / 'exports/weighted_calibration.csv').write_text('arm,bin,n,predicted,observed\n0,"(0, .1]",400,.01,.012\n0,"(.1, .3]",380,.05,.06\n')
    draws = [dict(task_id=f'confirmation_bootstrap_{i}', status='ok', att=float(.03 + .013 * rng.normal()), n_matches=8,
                  diagnostics=dict(passed=bool(i % 7 == 0), control_ess=250., n_shots=110, key_max_smd=.12 if i % 2 else .05,
                                   other_max_smd=.1, max_weight_share=.02 if i % 3 else .005, max_match_weight_share=.08,
                                   fold_support_selection_passed=True))
             for i in range(40)]
    write_json(conf / 'exports/bootstrap_summary.json', dict(requested=20, successful=40, attempted=40, failed=0,
               extension_executed=True, maximum_successes=40, precision_passed=True, endpoint_mcse=[.002, .0018],
               percentile_interval=[.004, .056], mean=.031, sd=.0134, diagnostic_poor=34, draws=draws,
               precision_rule='max endpoint MCSE <= 0.05 x percentile width'))
    null_records = lambda kind: [dict(att=float(rng.normal(0, .005)), reject_zero=bool(i == 0), status='ok', null_kind=kind) for i in range(10)]
    write_json(conf / 'exports/nulls.json', {kind: dict(requested=10, successful=10, failed=0, mean_att=.001, mean_att_mcse=.001,
               reject_zero_rate=.1, reject_zero_mc_interval=[.02, .4], records=null_records(kind))
               for kind in ['within_match_treatment', 'independent_dummy_y']})
    scenario = lambda mean, bias: dict(requested=5, successful=5, failed=0, mean_att=mean, bias=bias, bias_mcse=.001, rmse=.01,
                                       coverage=.9, coverage_mc_interval=[.6, .98], reject_zero_rate=.2, diagnostic_poor_successes=1)
    write_json(conf / 'exports/simulations.json', dict(requested=10, successful=10, coverage=.9, bias=0.,
               by_scenario=dict(healthy_zero=scenario(.001, .001), healthy_positive=scenario(.02, -.002))))
    (conf / 'exports/sensitivity').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(gamma=[1., 1.25, 1.5, 2.], att_lower=[.03, .01, -.01, -.06], att_upper=[.03, .045, .056, .07])).to_parquet(
        conf / 'exports/sensitivity/bounds.parquet')
    write_json(conf / 'exports/sensitivity/summary.json', dict(primary_att=.03, tipping=dict(status='bracketed', gamma_low=1.35, gamma_high=1.36),
               limitations='General DVDS identification'))
    write_json(conf / 'exports/analyses.json', dict(requested=3, successful=3, records=[
        dict(task_id='analysis_0_outcome_model_ridge', kind='outcome_model', name='ridge', status='ok', att=.03, score_interval=[.005, .055], n_shots=120),
        dict(task_id='analysis_3_horizon_Y5', kind='horizon', name='Y5', status='ok', att=.039, score_interval=[.016, .062], n_shots=120),
        dict(task_id='targeted_team_1', kind='targeted_team', name='787', status='ok', att=.023, score_interval=[-.001, .047], n_shots=100)]))
    write_json(conf / 'exports/score_influence.json', dict(att=.03, interpretation='Fixed-nuisance score screen',
               matches=[dict(original_match_id='1', att_without=.036, delta=.006, removed_treated=19)],
               teams=[dict(team='799', att_without=.04, delta=.01, removed_matches=3, removed_treated=63)],
               targeted_matches=['1'], targeted_teams=['799']))
    write_json(conf / 'exports/subgroups_joint.json', dict(critical_value=1.94, family=['angle_low'], interpretation='contrasts',
               strata=[dict(subgroup='angle_low', feature='angle', boundary=30., n_shots=60, att=.016, marginal_interval=[-.002, .033],
                            simultaneous_interval=[-.0015, .0327], sufficient=True),
                       dict(subgroup='distance_low', feature='distance', boundary=18., n_shots=40, att=.059, marginal_interval=[.02, .1],
                            simultaneous_interval=None, sufficient=False)],
               contrasts=[dict(contrast='distance_low-distance_high', estimate=None, simultaneous_interval=None, sufficient=False)]))
    (conf / 'exports/cases').mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2, 2)); ax.plot([0, 1], [0, 1]); fig.savefig(conf / 'exports/cases/case_1.png'); plt.close(fig)
    return full


def tree_digest(root, exclude):
    digest = {}
    for path in sorted(Path(root).rglob('*')):
        if path.is_file() and exclude not in path.parents:
            digest[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = load_report_namespace()

    def test_report_cells_never_change_scientific_core_hash(self):
        notebook = json.loads(NOTEBOOK.read_text())
        cells = report_cells(notebook)
        self.assertGreaterEqual(len(cells), 10)
        for cell in cells:
            self.assertNotIn('public-causal-core', cell['metadata'].get('tags', []))
        stripped = copy.deepcopy(notebook)
        stripped['cells'] = [cell for cell in stripped['cells'] if REPORT_TAG not in cell.get('metadata', {}).get('tags', [])]
        with tempfile.TemporaryDirectory() as temp:
            copy_path = Path(temp) / 'results/models/nb.ipynb'
            copy_path.parent.mkdir(parents=True)
            (Path(temp) / 'src').mkdir()
            copy_path.write_text(json.dumps(stripped))
            _, stripped_hash = export_notebook_core(copy_path, Path(temp) / 'a')
            _, full_hash = export_notebook_core(NOTEBOOK, Path(temp) / 'b')
        self.assertEqual(stripped_hash, full_hash)

    def test_pitch_image_is_warped_onto_true_markings(self):
        image = ROOT / 'results/models/pass_selection/campo_futebol.png'
        if not image.is_file():
            self.skipTest('campo_futebol.png not available')
        os.environ['PUBLIC_CAUSAL_PITCH_IMAGE'] = str(image)
        self.r['_PITCH_LAYER'].clear()
        layer = self.r['_pitch_layer']()
        self.assertEqual(layer.shape, (680, 1050, 4))
        alpha = layer[..., 3]
        cell = lambda x, y: (int(round((34 - y) * 10)), int(round((x + 52.5) * 10)))
        def near(x, y, r=4):
            row, col = cell(x, y)
            return alpha[row - r:row + r + 1, col - r:col + r + 1].max()
        for x, y in [(36, 10), (47, -5), (41.5, 0), (-36, -10), (-41.5, 0)]:   # box line, goal-area line, spots
            self.assertGreater(near(x, y), .5, (x, y))
        for x, y in [(44, 12), (30, 25), (-20, -20)]:                          # open grass: checkerboard removed
            self.assertLess(near(x, y), .05, (x, y))

    def test_case_figures_are_redrawn_from_saved_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            full = fixture_run(Path(temp))
            exports = full / 'confirmation/exports'
            cases = pd.DataFrame(dict(match_id=[1, 2], event_id=['a', 'b'], decision_row_id=['1:a', '2:b'],
                                      x=[30., 48.], y=[10., -2.], distance=[25., 5.], support=[True, False],
                                      defenders_3m=[1., 3.], support_reason=['supported_X_domain', 'insufficient_cross_match_cell_donors']))
            cases.to_parquet(exports / 'cases/cases.parquet')
            pd.DataFrame(dict(shot_id=['1:a'], donor_id=['3:c'])).to_parquet(exports / 'cases/case_donor_pairs.parquet')
            pd.DataFrame(dict(decision_row_id=['3:c'], x=[29.], y=[11.])).to_parquet(exports / 'nuisance_rows.parquet')
            items = self.r['section_cases'](self.r['report_paths'](full))
            figures = [item for item in items if item[0] == 'figure']
            self.assertEqual([title for _, title, _ in figures], ['case 1', 'case 2'])
            import matplotlib.pyplot as plt
            for _, _, fig in figures:
                plt.close(fig)

    def test_every_section_renders_from_saved_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            full = fixture_run(Path(temp))
            out = full / 'report'
            before = tree_digest(full, out)
            report = self.r['build_report'](full, out)
            self.assertEqual(report['errors'], [])
            names = [section['name'] for section in report['sections']]
            self.assertEqual(names, [name for name, _, _ in self.r['REPORT_SECTIONS']])
            for section in report['sections']:
                self.assertTrue(section['items'], section['name'])
            self.assertEqual(before, tree_digest(full, out), 'report must not modify run artifacts')
            self.assertTrue(list(out.glob('*.png')))
            self.assertTrue(list(out.glob('*.csv')))

    def test_tables_reproduce_saved_numbers(self):
        with tempfile.TemporaryDirectory() as temp:
            full = fixture_run(Path(temp))
            paths = self.r['report_paths'](full)
            tables = {item[1]: item[2] for item in self.r['section_estimate'](paths) if item[0] == 'table'}
            headline = next(table for title, table in tables.items() if 'Primary' in title)
            self.assertAlmostEqual(float(headline.loc['ATT', 'value']), .03)
            boot = {item[1]: item[2] for item in self.r['section_bootstrap'](paths) if item[0] == 'table'}
            summary = next(table for title, table in boot.items() if 'summary' in title.lower())
            self.assertEqual(int(summary.loc['successful draws', 'value']), 40)
            gates = next(table for title, table in boot.items() if 'gate' in title.lower())
            self.assertEqual(int(gates.loc['key_balance', 'failing draws']), 20)
            claim = {item[1]: item[2] for item in self.r['section_overview'](paths) if item[0] == 'table'}
            checks = next(table for title, table in claim.items() if 'check' in title.lower())
            self.assertTrue(bool(checks['passed'].all()))
            sens = [item for item in self.r['section_sensitivity'](paths) if item[0] == 'md']
            self.assertTrue(any('1.35' in item[1] for item in sens))

    def test_missing_artifacts_render_not_available(self):
        with tempfile.TemporaryDirectory() as temp:
            pilot = Path(temp) / 'pilot'
            pilot.mkdir()
            write_json(pilot / 'phase_result.json', dict(phase='pilot', status='incomplete'))
            report = self.r['build_report'](pilot, None)
            self.assertEqual(report['errors'], [])
            text = json.dumps([[item[1] for item in s['items'] if item[0] == 'md'] for s in report['sections']])
            self.assertIn('not available', text)

    def test_standalone_script_executes_report_notebook(self):
        with tempfile.TemporaryDirectory() as temp:
            full = fixture_run(Path(temp))
            output = full / 'report.ipynb'
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/report_public_causal.py'), '--phase-dir', str(full),
                                     '--output', str(output)], capture_output=True, text=True, timeout=600)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            notebook = json.loads(output.read_text())
            kinds = [kind for cell in notebook['cells'] for out in cell.get('outputs', []) for kind in out.get('data', {})]
            self.assertIn('image/png', kinds)
            self.assertIn('text/html', kinds)
            errors = [out for cell in notebook['cells'] for out in cell.get('outputs', []) if out.get('output_type') == 'error']
            self.assertEqual(errors, [])
            self.assertTrue(notebook['metadata']['public_causal_report']['phase_dir'].endswith('full'))


if __name__ == '__main__':
    unittest.main()
