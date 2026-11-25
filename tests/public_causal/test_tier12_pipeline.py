"""Tier 1/2 orchestration guards and the identical-sample comparison contract."""
from pathlib import Path
import importlib.util, json, sys, tempfile, time, unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from PublicCausal.notebook_runtime import load_notebook_core

spec = importlib.util.spec_from_file_location('tier12', ROOT / 'scripts' / 'tier12_public_causal.py')
tier12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tier12)


def synthetic(core, matches=12, per=80, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for m in range(matches):
        x1, x2 = rng.normal(size=per), rng.normal(size=per)
        p = 1 / (1 + np.exp(-(-1.8 + .8 * x1 + .6 * x2)))
        t = (rng.random(per) < p).astype(int)
        y = rng.choice([-1, 0, 1], per, p=[.02, .9, .08])
        frames.append(pd.DataFrame(dict(original_match_id=f'm{m}', match_id=f'm{m}', event_id=[f'e{i}' for i in range(per)],
                                        decision_row_id=[f'm{m}:{i}' for i in range(per)], team_id=f't{m % 6}',
                                        opponent_id=f't{(m + 1) % 6}', cohort='euro2020', T=t, Treatment=t, Y=y,
                                        x1=x1, x2=x2, eligible=True, support=True, e=.2, m0=0., m0_clipped=False)))
    rows = pd.concat(frames, ignore_index=True)
    rows['outer_fold'] = -1
    for fold, (_, test) in enumerate(core.grouped_folds(rows, 3, seed=1)):
        rows.loc[rows.index[test], 'outer_fold'] = fold
    return rows


class GuardTests(unittest.TestCase):
    def test_run_root_must_be_disjoint_from_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / 'orig'; original.mkdir()
            with self.assertRaises(PermissionError):
                tier12.check_separate(original / 'inside', original)
            with self.assertRaises(PermissionError):
                tier12.check_separate(original, original)
            tier12.check_separate(Path(tmp) / 'new', original)

    def test_prespec_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = Path(tmp) / 'prespec.md'; doc.write_text('v1')
            log = lambda *a, **k: None
            with patch.object(tier12, 'PRESPEC', doc):
                with self.assertRaises(PermissionError):
                    tier12.require_prespec(tmp)
                tier12.phase_prespec(SimpleNamespace(run_root=tmp), log)
                tier12.require_prespec(tmp)
                doc.write_text('v2')
                with self.assertRaises(PermissionError):
                    tier12.require_prespec(tmp)
                with self.assertRaises(SystemExit):
                    tier12.phase_prespec(SimpleNamespace(run_root=tmp), log)

    def test_confirmation_requires_development_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = Path(tmp) / 'prespec.md'; doc.write_text('v1')
            (Path(tmp) / 'features').mkdir()
            (Path(tmp) / 'features' / 'confirmation.current.json').write_text(json.dumps(dict(table='x.parquet', key='k' * 64)))
            with patch.object(tier12, 'PRESPEC', doc):
                tier12.phase_prespec(SimpleNamespace(run_root=tmp), lambda *a, **k: None)
                args = SimpleNamespace(run_root=tmp, cohort='confirmation', original=tmp)
                with self.assertRaises(PermissionError):
                    tier12.phase_compare(args, lambda *a, **k: None)

    def test_join_rejects_missing_features(self):
        rows = pd.DataFrame(dict(original_match_id=[1, 1], event_id=['a', 'b']))
        table = pd.DataFrame(dict(original_match_id=[1], event_id=['a'], f=[1.]))
        with self.assertRaises(ValueError):
            tier12.join_features(rows, table, ['f'])
        joined = tier12.join_features(rows.iloc[:1], table, ['f'])
        self.assertEqual(joined.f.tolist(), [1.])


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.core = load_notebook_core(ROOT / 'results/models/public_statsbomb_causal_study.ipynb', cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_fixed_population_arms_share_rows_and_report_paired_difference(self):
        core = self.core
        rows = synthetic(core)
        cfg = core.scientific_config(dict(profile='submission', outer_folds=3, inner_folds=2, min_shots=1, min_matches=1,
                                          min_ess=1, donor_min=1, donor_match_min=1, key_smd_max=100, other_smd_max=100,
                                          max_weight_share=1, max_match_weight_share=1,
                                          feature_names=['x1', 'team_id'], support_features=['x1'],
                                          key_balance_features=['x1']))
        specs = dict(e={'kind': 'logistic', 'C': 1}, m0={'kind': 'ridge', 'alpha': 10})
        registries = tier12._registries(rows)
        records, differences, fitted = tier12.fixed_population_comparison(
            core, rows, cfg, specs, [('baseline', ['x1', 'team_id']), ('plus', ['x1', 'x2', 'team_id'])],
            registries, int(rows['T'].sum()))
        self.assertEqual({r['status'] for r in records.values()}, {'ok'}, records)
        self.assertEqual(fitted['baseline'].decision_row_id.tolist(), fitted['plus'].decision_row_id.tolist())
        diff = differences['plus_minus_baseline']['original_match_id']
        self.assertAlmostEqual(diff['difference'], records['plus']['att'] - records['baseline']['att'], places=12)
        self.assertGreater(diff['se'], 0)
        self.assertEqual(records['plus']['n_shots'], records['baseline']['n_shots'])
        self.assertIn('opponent', records['plus']['dependence'])
        self.assertIsNotNone(records['plus']['oof_auc'])
        self.assertAlmostEqual(records['plus']['shot_retention'], 1.)


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(Path(root).rglob('*')) if p.is_file()}


def record_prespec(run_root):
    doc = Path(run_root) / 'prespec_source.md'
    doc.write_text('v1')
    return doc


class WriteOnceTests(unittest.TestCase):
    def test_refuse_existing_lists_every_existing_path_and_names_run_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a.json', Path(tmp) / 'b.json'
            a.write_text('x')
            tier12.refuse_existing([b])
            with self.assertRaises(FileExistsError) as ctx:
                tier12.refuse_existing([a, b])
            self.assertIn(str(a), str(ctx.exception))
            self.assertNotIn(str(b), str(ctx.exception))
            self.assertIn('--run-root', str(ctx.exception))

    def test_tier1_refuses_before_reading_or_writing_when_any_output_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root, original = Path(tmp) / 'run', Path(tmp) / 'orig'
            (run_root / 'tier1').mkdir(parents=True)
            (run_root / 'tier1' / 'null_offset.json').write_text('{"keep": 1}')
            before = snapshot(run_root)
            with self.assertRaises(FileExistsError):
                tier12.phase_tier1(SimpleNamespace(run_root=run_root, original=original), lambda *a, **k: None)
            self.assertEqual(snapshot(run_root), before)

    def test_report_refuses_when_any_output_exists_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'report').mkdir()
            (Path(tmp) / 'report' / 'tier2_estimates.md').write_text('keep')
            before = snapshot(tmp)
            with self.assertRaises(FileExistsError):
                tier12.phase_report(SimpleNamespace(run_root=tmp), lambda *a, **k: None)
            self.assertEqual(snapshot(tmp), before)

    def test_compare_refuses_when_summary_partial_or_design_exists(self):
        for cohort, name in [('development', 'summary.json'), ('development', 'partial_F1.json'),
                             ('development', 'partial_F2.json'), ('development', 'design_A.joblib'),
                             ('confirmation', 'summary.json'), ('confirmation', 'partial_F1.json'),
                             ('confirmation', 'partial_F2.json')]:
            with self.subTest(cohort=cohort, file=name), tempfile.TemporaryDirectory() as tmp:
                key = 'a' * 64
                doc = Path(tmp) / 'prespec.md'; doc.write_text('v1')
                features = Path(tmp) / 'features'; features.mkdir()
                (features / f'{cohort}.current.json').write_text(json.dumps(dict(table='x.parquet', key=key)))
                (features / 'development.current.json').write_text(json.dumps(dict(table='x.parquet', key=key)))
                out = Path(tmp) / cohort / key[:12]; out.mkdir(parents=True)
                if cohort == 'confirmation':
                    (Path(tmp) / 'development' / key[:12]).mkdir(parents=True)
                    (Path(tmp) / 'development' / key[:12] / 'summary.json').write_text('{}')
                (out / name).write_text('keep')
                with patch.object(tier12, 'PRESPEC', doc):
                    tier12.phase_prespec(SimpleNamespace(run_root=tmp), lambda *a, **k: None)
                    before = snapshot(tmp)
                    with self.assertRaises(FileExistsError):
                        tier12.phase_compare(SimpleNamespace(run_root=tmp, cohort=cohort, original=tmp), lambda *a, **k: None)
                self.assertEqual(snapshot(tmp), before)

    def test_write_once_json_refuses_to_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sub' / 'x.json'
            tier12.write_once_json(path, dict(a=1))
            with self.assertRaises(FileExistsError):
                tier12.write_once_json(path, dict(a=2))
            self.assertEqual(json.loads(path.read_text()), dict(a=1))


class FeaturePointerTests(unittest.TestCase):
    def run_features(self, tmp, ctx_status, key='k' * 64):
        from PublicCausal import context_features as cf
        eligible = pd.DataFrame(dict(original_match_id=['m1', 'm1'], event_id=['a', 'b'], T=[0, 1], eligible=[True, True]))
        table = pd.DataFrame(dict(original_match_id=['m1', 'm1'], event_id=['a', 'b'], ctx_in_status=ctx_status))
        meta = dict(key=key, version=cf.FEATURE_VERSION)
        args = SimpleNamespace(run_root=tmp, cohort='development', original=tmp, data_root=tmp)
        with patch.object(tier12, 'canonical_rows', return_value=(eligible, 'canon')), \
                patch.object(tier12, 'data_root', return_value=Path(tmp)), \
                patch.object(cf, 'build_context_table', return_value=(table, dict(meta))):
            tier12.phase_features(args, lambda *a, **k: None)

    def test_identical_pointer_is_a_noop_and_different_pointer_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_features(tmp, ['linked', 'linked'])
            pointer = Path(tmp) / 'features' / 'development.current.json'
            first = snapshot(tmp)
            self.run_features(tmp, ['linked', 'linked'])
            self.assertEqual(snapshot(tmp), first)
            with self.assertRaises(FileExistsError):
                self.run_features(tmp, ['linked', 'unresolved'])
            self.assertEqual(snapshot(tmp), first)
            self.assertTrue(pointer.exists())

    def test_versioned_table_with_another_key_is_never_replaced(self):
        from PublicCausal import context_features as cf
        with tempfile.TemporaryDirectory() as tmp:
            features = Path(tmp) / 'features'; features.mkdir()
            name = f'development-{cf.FEATURE_VERSION}-{"k" * 16}.parquet'
            (features / name).write_bytes(b'old')
            (features / name).with_suffix('.meta.json').write_text(json.dumps(dict(key='z' * 64)))
            before = snapshot(tmp)
            with self.assertRaises(FileExistsError):
                self.run_features(tmp, ['linked', 'linked'])
            self.assertEqual(snapshot(tmp), before)


class GuardVerifyTests(unittest.TestCase):
    def test_verify_writes_new_timestamped_file_and_keeps_existing_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, run_root = Path(tmp) / 'orig', Path(tmp) / 'run'
            original.mkdir(); run_root.mkdir()
            (original / 'f.txt').write_text('1')
            log = lambda *a, **k: None
            tier12.phase_guard(SimpleNamespace(run_root=run_root, original=original, verify=False), log)
            legacy = run_root / 'original_manifest_verify.json'
            legacy.write_text('{"identical": true, "legacy": 1}')
            args = SimpleNamespace(run_root=run_root, original=original, verify=True)
            with patch.object(tier12.time, 'gmtime', return_value=time.gmtime(1_700_000_000)):
                tier12.phase_guard(args, log)
            first = run_root / 'original_manifest_verify-20231114T221320Z.json'
            self.assertTrue(first.exists())
            self.assertEqual(legacy.read_text(), '{"identical": true, "legacy": 1}')
            self.assertTrue(json.loads(first.read_text())['identical'])
            with patch.object(tier12.time, 'gmtime', return_value=time.gmtime(1_700_000_001)):
                tier12.phase_guard(args, log)
            self.assertTrue((run_root / 'original_manifest_verify-20231114T221321Z.json').exists())
            self.assertTrue(first.exists())
            with patch.object(tier12.time, 'gmtime', return_value=time.gmtime(1_700_000_001)):
                with self.assertRaises(FileExistsError):
                    tier12.phase_guard(args, log)

    def test_verify_reports_a_changed_original_and_still_records_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, run_root = Path(tmp) / 'orig', Path(tmp) / 'run'
            original.mkdir(); run_root.mkdir()
            (original / 'f.txt').write_text('1')
            log = lambda *a, **k: None
            tier12.phase_guard(SimpleNamespace(run_root=run_root, original=original, verify=False), log)
            (original / 'f.txt').write_text('2')
            with self.assertRaises(SystemExit):
                tier12.phase_guard(SimpleNamespace(run_root=run_root, original=original, verify=True), log)
            written = sorted(run_root.glob('original_manifest_verify-*.json'))
            self.assertEqual(len(written), 1)
            self.assertFalse(json.loads(written[0].read_text())['identical'])


class LabelTests(unittest.TestCase):
    def test_world_cup_label_is_post_hoc_exploratory(self):
        self.assertEqual(tier12.label_for('confirmation'), 'post_hoc_exploratory')
        self.assertEqual(tier12.label_for('development'), 'development_post_hoc')


class F3BasisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.core = load_notebook_core(ROOT / 'results/models/public_statsbomb_causal_study.ipynb', cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_shot_overlap_counts_supported_treated_ids(self):
        primary = pd.DataFrame(dict(decision_row_id=['a', 'b', 'c', 'd'], T=[1, 1, 1, 0], support=[True, True, False, True]))
        selected = pd.DataFrame(dict(decision_row_id=['b', 'c', 'e', 'f'], T=[1, 1, 1, 0], support=[True, True, True, True]))
        self.assertEqual(tier12.shot_overlap(primary, selected),
                         dict(primary_shots=2, selected_shots=3, both=1, jaccard=1 / 4))

    def test_shot_retention_uses_reference_and_handles_missing_reference(self):
        self.assertAlmostEqual(tier12.shot_retention(2186, 2195), 2186 / 2195)
        self.assertIsNone(tier12.shot_retention(5, 0))
        self.assertIsNone(tier12.shot_retention(5, None))

    def test_f3_record_retention_is_vs_primary_supported_not_eligible_and_has_overlap(self):
        core = self.core
        primary = synthetic(core)
        primary.loc[primary.index[primary['T'] == 1][:20], 'support'] = False
        selected = primary.copy()
        dropped = selected.index[(selected['T'] == 1) & selected.support][:7]
        selected.loc[dropped, 'support'] = False
        cfg = core.scientific_config(dict(profile='submission', outer_folds=3, inner_folds=2, min_shots=1, min_matches=1,
                                          min_ess=1, donor_min=1, donor_match_min=1, key_smd_max=100, other_smd_max=100,
                                          max_weight_share=1, max_match_weight_share=1,
                                          feature_names=['x1', 'x2'], support_features=['x1'], key_balance_features=['x1']))
        reference = int((primary.support & (primary['T'] == 1)).sum())
        eligible_treated = int((selected['T'] == 1).sum())
        record = tier12.f3_record(core, selected, primary, cfg, ['x1', 'x2'], tier12._registries(selected), reference)
        est = record['estimate']
        self.assertEqual(est['n_shots'], reference - 7)
        self.assertAlmostEqual(est['shot_retention'], (reference - 7) / reference)
        self.assertGreater(eligible_treated, reference)
        self.assertNotAlmostEqual(est['shot_retention'], est['n_shots'] / eligible_treated)
        self.assertEqual(record['overlap'], dict(primary_shots=reference, selected_shots=reference - 7, both=reference - 7,
                                                 jaccard=(reference - 7) / reference))


class F3SupplementTests(unittest.TestCase):
    KEY = 'c' * 64

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.core = load_notebook_core(ROOT / 'results/models/public_statsbomb_causal_study.ipynb', cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def build(self, tmp, att_shift=0.):
        import joblib
        core = self.core
        primary = synthetic(core)
        selected = primary.copy()
        selected.loc[selected.index[selected['T'] == 1][:9], 'support'] = False
        run_root, original = Path(tmp) / 'run', Path(tmp) / 'orig'
        (original / 'development').mkdir(parents=True)
        joblib.dump(dict(rows=primary), original / 'development' / 'fit.joblib')
        out = run_root / 'development' / self.KEY[:12]
        out.mkdir(parents=True)
        joblib.dump(dict(development_selection=dict(rows=selected)), out / 'design_A.joblib')
        est = core.att_estimate(selected)
        reference = int((primary.support & (primary['T'] == 1)).sum())
        (run_root / 'features').mkdir()
        (run_root / 'features' / 'development.current.json').write_text(json.dumps(dict(table='t.parquet', key=self.KEY)))
        summary = dict(label='development_post_hoc', reference_supported_shots=reference,
                       F3=dict(status='ok', estimate=dict(att=est['att'] + att_shift, n_shots=est['n_shots'],
                                                          shot_retention=est['n_shots'] / int((selected['T'] == 1).sum()))))
        (out / 'summary.json').write_text(json.dumps(summary, sort_keys=True))
        doc = Path(tmp) / 'prespec.md'; doc.write_text('v1')
        with patch.object(tier12, 'PRESPEC', doc):
            tier12.phase_prespec(SimpleNamespace(run_root=run_root), lambda *a, **k: None)
        return run_root, original, out, doc, reference, est

    def run_phase(self, run_root, original, doc, cohort='development'):
        with patch.object(tier12, 'PRESPEC', doc):
            tier12.phase_f3_supplement(SimpleNamespace(run_root=run_root, original=original, cohort=cohort),
                                       lambda *a, **k: None)

    def test_writes_one_new_file_from_saved_artifacts_without_touching_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root, original, out, doc, reference, est = self.build(tmp)
            before = snapshot(run_root)
            self.run_phase(run_root, original, doc)
            after = snapshot(run_root)
            self.assertEqual(set(after) - set(before), {f'development/{self.KEY[:12]}/f3_supplement.json'})
            self.assertEqual({k: v for k, v in after.items() if k in before}, before)
            payload = json.loads((out / 'f3_supplement.json').read_text())
            self.assertEqual(payload['label'], 'development_post_hoc')
            self.assertEqual(payload['path'], 'saved_artifacts_no_refit')
            self.assertEqual(payload['retention']['reference_supported_shots'], reference)
            self.assertEqual(payload['retention']['n_shots'], est['n_shots'])
            self.assertAlmostEqual(payload['retention']['shot_retention'], est['n_shots'] / reference)
            self.assertEqual(payload['overlap']['primary_shots'], reference)
            self.assertEqual(payload['overlap']['selected_shots'], reference - 9)
            self.assertEqual(payload['overlap']['both'], reference - 9)
            self.assertEqual(payload['script_sha256'], tier12.sha256_file(ROOT / 'scripts' / 'tier12_public_causal.py'))
            self.assertLessEqual(payload['checks']['att_abs_diff'], 1e-9)
            self.assertEqual(set(payload['inputs']), {'summary', 'design_A', 'primary_fit'})
            self.assertTrue(all('sha256' in v and 'path' in v for v in payload['inputs'].values()))
            self.assertIn('utc', payload)

    def test_second_run_is_refused_and_leaves_the_file_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root, original, out, doc, *_ = self.build(tmp)
            self.run_phase(run_root, original, doc)
            before = snapshot(run_root)
            with self.assertRaises(FileExistsError):
                self.run_phase(run_root, original, doc)
            self.assertEqual(snapshot(run_root), before)

    def test_refuses_when_saved_att_is_not_reproduced(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root, original, out, doc, *_ = self.build(tmp, att_shift=1e-4)
            before = snapshot(run_root)
            with self.assertRaises(ValueError):
                self.run_phase(run_root, original, doc)
            self.assertEqual(snapshot(run_root), before)

    def test_requires_prespec_and_development_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root, original, out, doc, *_ = self.build(tmp)
            with self.assertRaises(ValueError):
                self.run_phase(run_root, original, doc, cohort='confirmation')
            doc.write_text('changed')
            with self.assertRaises(PermissionError):
                self.run_phase(run_root, original, doc)
            self.assertFalse((out / 'f3_supplement.json').exists())


if __name__ == '__main__':
    unittest.main()
