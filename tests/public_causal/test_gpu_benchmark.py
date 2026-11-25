"""GPU benchmark contracts; kernel/device execution is a separate live check."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('gpu_benchmark_test', ROOT / 'scripts/benchmark_public_causal_gpu.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class GPUBenchmarkTests(unittest.TestCase):
    def test_cpu_cuda_only_change_device(self):
        spec = {'leaves': 7, 'iterations': 100}
        for task in ['propensity', 'outcome']:
            cpu = benchmark.xgb_parameters(spec, task, 'cpu')
            gpu = benchmark.xgb_parameters(spec, task, 'cuda:0')
            self.assertEqual({k:v for k,v in cpu.items() if k != 'device'},
                             {k:v for k,v in gpu.items() if k != 'device'})
            self.assertEqual(cpu['n_jobs'], 1)
            self.assertNotIn('early_stopping_rounds', cpu)

    def test_cpu_fallback_is_not_a_gpu_result(self):
        class FakeModel:
            def __init__(self, device): self.device = device
            def get_booster(self): return self
            def save_config(self):
                return json.dumps({'learner': {'generic_param': {'device': self.device}}})
        self.assertEqual(benchmark.assert_cuda_model(FakeModel('cuda:0')), 'cuda:0')
        with self.assertRaisesRegex(RuntimeError, 'fallback'):
            benchmark.assert_cuda_model(FakeModel('cpu'))

    def test_backend_restores_original_factory_on_error(self):
        class Core: pass
        core = Core()
        core.feature_pipeline = lambda *a, **k: None
        original = core.feature_pipeline
        with self.assertRaisesRegex(ValueError, 'fixture'):
            with benchmark.backend_factory(core, 'sklearn_cpu', []):
                raise ValueError('fixture')
        self.assertIs(core.feature_pipeline, original)
