#!/usr/bin/env python3
"""Isolated GPU feasibility/timing benchmark; never modifies the primary study.

Reads only saved Euro decisions and the notebook's scientific definitions.
Temporarily replaces boosted estimators inside this process, preserving grouped
preprocessing/calibration/tuning. XGBoost is not statistically identical to HGB.
CPU/GPU XGBoost use identical settings except device. No confirmation access,
bootstrap checkpoint reuse, production lock, package or driver modifications.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback
import warnings

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from PublicCausal.notebook_runtime import (
    RunLogger, atomic_write_json, check_disk_budget, environment_provenance,
    file_hash, load_notebook_core, utc_now,
)


def xgb_parameters(spec, task, device):
    return dict(n_estimators=spec['iterations'], max_leaves=spec['leaves'],
                max_depth=0, grow_policy='lossguide', learning_rate=.05,
                reg_lambda=1, min_child_weight=5 if task == 'propensity' else 20,
                max_bin=255, subsample=1, colsample_bytree=1,
                tree_method='hist', device=device, n_jobs=1, random_state=20260930,
                objective='binary:logistic' if task == 'propensity' else 'reg:squarederror')


def booster_device(model):
    return json.loads(model.get_booster().save_config())['learner']['generic_param']['device']


def assert_cuda_model(model):
    device = booster_device(model)
    if not device.startswith('cuda'):
        raise RuntimeError(f'CUDA requested but XGBoost used {device}; fallback is not a GPU benchmark')
    return device


@contextlib.contextmanager
def backend_factory(core, backend, fitted_devices):
    """Replace just HGB candidates, including nested calibration, in this process."""
    original = core.feature_pipeline
    if backend == 'sklearn_cpu':
        yield
        return
    import xgboost as xgb
    device = 'cuda:0' if backend == 'xgboost_cuda' else 'cpu'

    def factory(frame, features, spec, task='outcome'):
        pipeline = original(frame, features, spec, task)
        if spec['kind'] != 'hgb':
            return pipeline
        base = xgb.XGBClassifier if task == 'propensity' else xgb.XGBRegressor

        class CheckedEstimator(base):
            # Every nested boosted fit must actually use CUDA; no silent fallback.
            def fit(self, *args, **kwargs):
                result = super().fit(*args, **kwargs)
                actual = assert_cuda_model(self) if device.startswith('cuda') else booster_device(self)
                fitted_devices.append(actual)
                return result

        pipeline.set_params(model=CheckedEstimator(**xgb_parameters(spec, task, device)))
        return pipeline

    core.feature_pipeline = factory
    try:
        yield
    finally:
        core.feature_pipeline = original


class GPUSampler:
    """Read GPU metrics once per second, capped to a small benchmark trace."""
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                output = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4, check=True)
                memory, utilization = map(float, output.stdout.strip().splitlines()[0].split(','))
                if len(self.samples) < 20000:
                    self.samples.append(dict(utc=utc_now(), memory_mib=memory, utilization_pct=utilization))
            except Exception as exc:
                self.samples.append(dict(utc=utc_now(), error=str(exc)))
                break
            self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)


def verify_provenance(bundle, outer_test):
    held_out = set(outer_test.original_match_id.astype(str))
    records = bundle.get('provenance', [])
    if not records:
        raise AssertionError('Missing grouped training provenance')
    for record in records:
        trained, evaluated = set(record['train_match_ids']), set(record['test_match_ids'])
        if not trained.isdisjoint(evaluated) or not trained.isdisjoint(held_out) or not evaluated.isdisjoint(held_out):
            raise AssertionError(f'Benchmark match leakage in {record["stage"]}')
    return len(records)


def task_fit(core, train, test, config, task, weights, logger):
    if task == 'propensity':
        bundle = core.fit_propensity(train, config, config['feature_names'], logger=logger)
        predict = lambda: core.predict_propensity(bundle, test, config['feature_names'])
    else:
        arm = int(task[-1])
        bundle = core.fit_outcome(train, config, arm, weights, logger=logger)
        predict = lambda: core.np.clip(bundle['model'].predict(test[config['feature_names']]), -1, 1)
    return bundle, predict


def measure_task(core, backend, task, train, test, config, weights, logger):
    devices = []
    with warnings.catch_warnings(record=True) as messages, backend_factory(core, backend, devices):
        warnings.simplefilter('always')
        started = time.perf_counter()
        bundle, predict = task_fit(core, train, test, config, task, weights, logger)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prediction = predict()
        prediction_seconds = time.perf_counter() - started
    if not core.np.isfinite(prediction).all():
        raise AssertionError('Nonfinite benchmark predictions')
    provenance_count = verify_provenance(bundle, test)
    metrics = {}
    if task == 'propensity':
        metrics.update(log_loss=float(core.log_loss(test.Treatment, prediction, labels=[0, 1])),
                       brier=float(core.brier_score_loss(test.Treatment, prediction)))
    else:
        mask = test.Treatment.to_numpy() == int(task[-1])
        residual = test.Y.to_numpy()[mask] - prediction[mask]
        metrics.update(observed_arm_mse=float(core.np.mean(residual**2)),
                       observed_arm_residual=float(core.np.mean(residual)))
    record = dict(backend=backend, task=task, fit_seconds=fit_seconds,
                  prediction_seconds=prediction_seconds, total_seconds=fit_seconds + prediction_seconds,
                  train_rows=len(train), test_rows=len(test), provenance_records=provenance_count,
                  trained_cuda_models=sum(d.startswith('cuda') for d in devices),
                  trained_xgboost_models=len(devices), selected_spec=bundle['spec'], metrics=metrics,
                  warnings=sorted({str(message.message) for message in messages}))
    return record, bundle, prediction


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--baseline-run', default='public-study-20260930')
    parser.add_argument('--run-id', default='gpu-benchmark-20260930')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--full-suite', action='store_true', help='Also benchmark one complete outer-fold candidate-selection workload per backend')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,120}', args.run_id) or args.repeats < 1:
        parser.error('Safe run ID and positive repetitions required')
    study = args.root.resolve() / 'data/public_statsbomb_causal'
    check_disk_budget(study)
    source = study / 'runs' / args.baseline_run / 'full/development'
    output = study / 'benchmarks' / args.run_id
    if output.exists():
        raise FileExistsError('Use a new benchmark run ID; existing results are not overwritten')
    output.mkdir(parents=True)
    logger = RunLogger(output, args.run_id)
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from threadpoolctl import threadpool_limits
    notebook = args.root / 'results/models/public_statsbomb_causal_study.ipynb'
    core = load_notebook_core(notebook, output / 'runtime')
    provenance = json.loads((source / 'provenance.json').read_text())
    if provenance['core_hash'] != core.CORE_SOURCE_HASH:
        raise ValueError('Notebook changed from the baseline; this benchmark must compare the exact baseline source')
    data_path = source / 'candidate_decisions.parquet'
    config = core.scientific_config()
    frame = core.prepare_analysis(pd.read_parquet(data_path), config)
    outer_train, outer_test = core.grouped_folds(frame, 5, config['seed'])[0]
    train, test = frame.iloc[outer_train].copy(), frame.iloc[outer_test].copy()
    assert set(train.original_match_id).isdisjoint(test.original_match_id)
    atomic_write_json(output / 'provenance.json', dict(
        environment=environment_provenance(), core_source_hash=core.CORE_SOURCE_HASH,
        baseline_run=args.baseline_run, candidates_sha256=file_hash(data_path),
        benchmark_script_sha256=file_hash(__file__), xgboost_version=xgb.__version__,
        xgboost_build=xgb.build_info(), repeat_count=args.repeats,
        outer_train_match_ids=sorted(train.original_match_id.astype(str).unique()),
        outer_test_match_ids=sorted(test.original_match_id.astype(str).unique()),
        cpu_threads=1, concurrent_baseline_workers=4,
        caveats=['CPU timings measured alongside the four-worker baseline, not an idle machine.',
                 'XGBoost and sklearn HGB differ statistically; only XGBoost CPU versus CUDA isolates hardware.',
                 'Timing includes preprocessing, nested grouped calibration/validation and prediction transfers.',
                 'No ATT, support selection or confirmation validation is performed by this benchmark.']))
    records, prediction_records = [], []
    try:
        with GPUSampler() as telemetry, threadpool_limits(limits=1):
            with logger.stage('cuda_probe'):
                warm_start = time.perf_counter()
                model = xgb.XGBRegressor(**xgb_parameters({'iterations': 4, 'leaves': 7}, 'outcome', 'cuda:0'))
                model.fit(np.arange(400, dtype=np.float32).reshape(200, 2), np.arange(200) % 3)
                actual_device = assert_cuda_model(model)
                atomic_write_json(output / 'cuda_probe.json', dict(passed=True, actual_device=actual_device,
                    first_fit_seconds=time.perf_counter()-warm_start))
            compact = copy.deepcopy(config)
            compact['e_candidates'] = [{'kind': 'hgb', 'leaves': 7, 'iterations': 100}]
            compact['outcome_candidates'] = [{'kind': 'hgb', 'leaves': 7, 'iterations': 100}]
            with logger.stage('reference_weights'):
                reference = core.fit_propensity(train, compact, config['feature_names'], logger=logger)
                verify_provenance(reference, test)
                weights = reference['oof']
            backends = ['sklearn_cpu', 'xgboost_cpu', 'xgboost_cuda']
            for repeat in range(args.repeats):
                # Rotate order to avoid always giving one backend cold caches.
                order = backends[repeat % 3:] + backends[:repeat % 3]
                for backend in order:
                    for task in ['propensity', 'm0', 'm1']:
                        with logger.stage(f'compact_{repeat}_{backend}_{task}'):
                            record, _, prediction = measure_task(core, backend, task, train, test, compact, weights, logger)
                            record.update(profile='compact_grouped_fit', repeat=repeat)
                            records.append(record)
                            logger.event('benchmark_result', **record)
                            if repeat == 0:
                                prediction_records.extend(dict(backend=backend, task=task, decision_row_id=row,
                                    prediction=float(value)) for row, value in zip(test.decision_row_id, prediction))
                            atomic_write_json(output / 'records.json', records)
            if args.full_suite:
                for backend in backends:
                    with logger.stage(f'full_selection_{backend}'):
                        started = time.perf_counter()
                        e_record, propensity, _ = measure_task(core, backend, 'propensity', train, test, config, weights, logger)
                        suite = [e_record]
                        for task in ['m0', 'm1']:
                            record, _, _ = measure_task(core, backend, task, train, test, config, propensity['oof'], logger)
                            suite.append(record)
                        record = dict(backend=backend, task='complete_nuisance_selection', profile='full_candidate_outer_fold',
                            repeat=0, total_seconds=time.perf_counter()-started, components=suite,
                            trained_cuda_models=sum(s['trained_cuda_models'] for s in suite))
                        records.append(record)
                        logger.event('benchmark_result', **record)
                        atomic_write_json(output / 'records.json', records)
            atomic_write_json(output / 'gpu_telemetry.json', telemetry.samples)
        table = pd.DataFrame([{k:v for k,v in r.items() if k not in ['components', 'metrics', 'warnings']} for r in records])
        table.to_csv(output / 'timings.csv', index=False)
        predictions = pd.DataFrame(prediction_records)
        predictions.to_parquet(output / 'held_out_predictions.parquet', index=False)
        comparisons = []
        for task, group in predictions.groupby('task'):
            matrix = group.pivot(index='decision_row_id', columns='backend', values='prediction')
            for reference_backend in ['xgboost_cpu', 'sklearn_cpu']:
                delta = matrix.xgboost_cuda - matrix[reference_backend]
                comparisons.append(dict(task=task, reference=reference_backend, max_absolute_difference=float(delta.abs().max()),
                    mean_absolute_difference=float(delta.abs().mean())))
        medians = table.groupby(['profile', 'task', 'backend']).total_seconds.median().unstack('backend')
        medians['cuda_speedup_vs_xgb_cpu'] = medians.xgboost_cpu / medians.xgboost_cuda
        medians['cuda_speedup_vs_sklearn_cpu'] = medians.sklearn_cpu / medians.xgboost_cuda
        medians.reset_index().to_csv(output / 'summary.csv', index=False)
        observed = [s for s in telemetry.samples if 'memory_mib' in s]
        atomic_write_json(output / 'benchmark_result.json', dict(status='completed', cuda_probe_passed=True,
            comparisons=comparisons, summary=medians.reset_index().to_dict('records'),
            maximum_gpu_memory_mib=max(s['memory_mib'] for s in observed) if observed else None,
            maximum_gpu_utilization_pct=max(s['utilization_pct'] for s in observed) if observed else None,
            production_changed=False, causal_validation_passed=False))
        logger.close('completed', production_changed=False)
        return 0
    except Exception as exc:
        atomic_write_json(output / 'benchmark_result.json', dict(status='failed', error=str(exc),
            traceback=traceback.format_exc(), production_changed=False,
            qualification='No package/driver changes or fallback to CPU are authorized by this benchmark.'))
        logger.close('failed', error=str(exc), traceback=traceback.format_exc())
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
