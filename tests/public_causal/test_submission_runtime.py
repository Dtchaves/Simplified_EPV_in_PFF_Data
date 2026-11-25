import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PublicCausal import notebook_runtime as rt

ROOT = Path(__file__).resolve().parents[2]
def script(name):
    spec = importlib.util.spec_from_file_location('submission_' + name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class SubmissionRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'source'
        (self.root / 'src').mkdir(parents=True)
        notebook = self.root / 'results/models/public_statsbomb_causal_study.ipynb'
        notebook.parent.mkdir(parents=True)
        notebook.write_text(json.dumps({'cells': [
            {'cell_type':'code','metadata':{'tags':[rt.PARAMETERS_TAG]},'source':'STUDY_CONFIG = {}'},
            {'cell_type':'code','metadata':{'tags':[rt.CORE_TAG]},'source':'VALUE = 1'}]}))
        self.study = Path(self.temp.name) / 'public_statsbomb_causal'
        self.study.mkdir()
        self.runner = script('run_public_causal')

    def args(self, *extra):
        return self.runner.make_parser().parse_args(['--root', str(self.root), '--study-root', str(self.study), '--profile','submission','--run-id','test','--free-space-floor-gb','0', *extra])

    def test_profile_defaults_and_data_binding(self):
        for phase, counts in [('development',(200,300,50,0,50)),('pilot',(5,8,2,0,2)),('smoke',(3,5,1,0,1))]:
            _, config, directory = self.runner.prepare_run(self.args('--phase',phase))
            self.assertEqual(tuple(config[k] for k in ['bootstrap_target','bootstrap_max_attempts','simulation_datasets','simulation_bootstrap_target','null_reps']),counts)
            self.assertEqual(config['root'], str(self.root))
            self.assertEqual(config['study_root'], str(self.study))
            self.assertEqual(config['profile'],'submission')
            self.assertEqual(config['candidate_device'],'cpu')
            self.assertTrue(directory.is_relative_to(self.study))
            self.assertTrue(Path(config['implementation_tests_path']).is_relative_to(self.study))
        self.assertFalse((self.root/'data').exists())

    def test_existing_shared_path_required_and_import_explicit(self):
        with self.assertRaises(ValueError):
            self.runner.prepare_run(self.args('--study-root',str(self.study/'absent')))
        old = self.study/'runs/old/development'; old.mkdir(parents=True)
        _, config, _ = self.runner.prepare_run(self.args('--reuse-development',str(old)))
        self.assertEqual(config['reuse_development'],str(old))
        self.assertFalse(config['resume'])

    def test_resume_config_rejects_device_profile_and_counts_changes(self):
        self.runner.prepare_run(self.args())
        self.runner.prepare_run(self.args('--resume'))
        for extra in [('--device','cuda'),('--profile','extended'),('--bootstrap-target','4')]:
            with self.assertRaises(ValueError):
                self.runner.prepare_run(self.args('--resume',*extra))

    def payload(self, profile='submission'):
        gates=['timing_audit','leakage_tests','development_diagnostics','weighted_validation','design_sanity']
        spec={k:{'chosen':'fixed'} for k in ['horizon','model','backend','support','subgroup','inference','simulation','budget']}
        return dict(profile=profile, core_source_hash=rt.stable_hash('core'),data_manifest_hash=rt.stable_hash('data'),config_hash=rt.stable_hash('config'),development_gates={k:True for k in gates},specification_manifest=spec,specification_manifest_hash=rt.stable_hash(spec))

    def test_submission_lock_automatic_adapter_boundary(self):
        path=self.study/'lock.json'
        rt.freeze_design_lock(path,self.payload())
        lock=rt.validate_design_lock(path,expected_profile='submission')
        self.assertEqual(lock['profile'],'submission')
        with self.assertRaises(PermissionError): rt.validate_design_lock(path,expected_profile='extended')
        with self.assertRaises(PermissionError): rt.validate_design_lock(path,required_gates=('timing_audit',))
        lock['specification_manifest']['model']['chosen']='changed'
        rt.atomic_write_json(path,lock)
        rt.atomic_write_bytes(path.with_suffix('.json.sha256'),(rt.file_hash(path)+'\n').encode())
        with self.assertRaises(PermissionError): rt.validate_design_lock(path)

    def test_lock_missing_unknown_or_weakened_gates(self):
        for name, mutate in [('unknown',lambda p:p.update(profile='other')),('missing',lambda p:p['development_gates'].pop('design_sanity')),('manifest',lambda p:p['specification_manifest'].pop('backend'))]:
            payload=self.payload(); mutate(payload)
            with self.assertRaises((PermissionError,ValueError)):
                rt.freeze_design_lock(self.study/(name+'.json'),payload)
        with self.assertRaises(PermissionError):
            rt.freeze_design_lock(self.study/'weakened.json',self.payload(),required_gates=('timing_audit',))

    def test_resume_eta_counts_only_new_work(self):
        logger=rt.RunLogger(self.study/'logging','test',heartbeat_seconds=0)
        self.addCleanup(logger.close)
        with patch.object(rt.time,'monotonic',side_effect=[100,100,110,110]):
            first=logger.progress('bootstrap',100,200,cached_completed=100)
            second=logger.progress('bootstrap',102,200,cached_completed=100)
        self.assertIsNone(first['eta_seconds'])
        self.assertEqual(second['fresh_completed'],2)
        self.assertEqual(second['eta_seconds'],490)

    def test_validation_explicit_destination_preserves_baseline(self):
        validator=script('validate_public_causal')
        baseline=self.root/'data/public_statsbomb_causal/manifests/implementation_tests.json'
        baseline.parent.mkdir(parents=True); baseline.write_text('historical')
        target=self.study/'runs/new/manifests'
        suite=unittest.TestSuite([unittest.FunctionTestCase(lambda: None)])
        with patch.object(validator,'ROOT',self.root), patch.object(validator.unittest.defaultTestLoader,'discover',return_value=suite):
            self.assertEqual(validator.main(['--output-dir',str(target)]),0)
        self.assertEqual(baseline.read_text(),'historical')
        self.assertTrue(json.loads((target/'implementation_tests.json').read_text())['passed'])
        self.assertIn('xgboost',rt.environment_provenance()['packages'])
        self.assertEqual(json.loads((target/'implementation_tests.json').read_text())['tests_run'],1)

    def test_validation_rejects_empty_suite(self):
        validator=script('validate_public_causal')
        target=self.study/'runs/empty/manifests'
        with patch.object(validator,'ROOT',self.root), patch.object(validator.unittest.defaultTestLoader,'discover',return_value=unittest.TestSuite()):
            self.assertEqual(validator.main(['--output-dir',str(target)]),1)
        record=json.loads((target/'implementation_tests.json').read_text())
        self.assertFalse(record['passed'])
        self.assertEqual(record['tests_run'],0)
        self.assertIn('No implementation tests discovered',record['validation_error'])
