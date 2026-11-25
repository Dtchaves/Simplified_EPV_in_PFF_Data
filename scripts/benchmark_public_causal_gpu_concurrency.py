#!/usr/bin/env python3
"""Benchmark concurrent independent grouped fits, without changing production.

Four identical representative propensity workloads (not bootstrap draws) are
run with CUDA worker counts 1, 2 and 4, then XGBoost CPU with four workers.
Each child has one numerical thread and its own GPU context/logs. Throughput,
latency, actual devices and GPU telemetry are saved separately from the study.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark_public_causal_gpu as bench


def worker(args):
    import pandas as pd
    from threadpoolctl import threadpool_limits
    out = args.output
    logger = bench.RunLogger(out, out.name)
    try:
        core = bench.load_notebook_core(ROOT / 'results/models/public_statsbomb_causal_study.ipynb', out / 'runtime')
        source = ROOT / 'data/public_statsbomb_causal/runs/public-study-20260930/full/development'
        prior = json.loads((source / 'provenance.json').read_text())
        assert prior['core_hash'] == core.CORE_SOURCE_HASH
        config = core.scientific_config()
        config['e_candidates'] = [{'kind':'hgb', 'leaves':7, 'iterations':100}]
        data = core.prepare_analysis(pd.read_parquet(source / 'candidate_decisions.parquet'), config)
        a,b = core.grouped_folds(data, 5, config['seed'])[0]
        train,test = data.iloc[a].copy(),data.iloc[b].copy()
        with threadpool_limits(limits=1):
            record,_,_ = bench.measure_task(core,args.backend,'propensity',train,test,config,None,logger)
        bench.atomic_write_json(out / 'result.json',dict(record,status='completed'))
        logger.close('completed')
        return 0
    except Exception as exc:
        bench.atomic_write_json(out / 'result.json',dict(status='failed',error=str(exc),traceback=traceback.format_exc()))
        logger.close('failed',error=str(exc))
        return 1


def launch_child(output, backend, job):
    out=output/f'job_{job}'
    out.mkdir(parents=True)
    environment=dict(os.environ)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        environment[name]='1'
    started=time.perf_counter()
    with (out/'process.log').open('w') as stream:
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--backend',backend,
            '--output',str(out)],stdout=stream,stderr=subprocess.STDOUT,env=environment)
    record=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else dict(status='failed')
    record.update(job=job,return_code=result.returncode,process_seconds=time.perf_counter()-started)
    return record


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--backend',choices=['xgboost_cpu','xgboost_cuda'],default='xgboost_cuda')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if args.worker:
        if args.output is None: parser.error('Worker output required')
        return worker(args)
    study=ROOT/'data/public_statsbomb_causal'
    bench.check_disk_budget(study)
    out=study/'benchmarks/gpu-concurrency-20260930'
    if out.exists(): raise FileExistsError('Concurrency benchmark already exists')
    out.mkdir(parents=True)
    logger=bench.RunLogger(out,out.name)
    notebook=ROOT/'results/models/public_statsbomb_causal_study.ipynb'
    core=bench.load_notebook_core(notebook,out/'runtime')
    bench.atomic_write_json(out/'provenance.json',dict(core_source_hash=core.CORE_SOURCE_HASH,
        script_sha256=bench.file_hash(__file__),factory_sha256=bench.file_hash(bench.__file__),
        candidate_sha256=bench.file_hash(study/'runs/public-study-20260930/full/development/candidate_decisions.parquet'),
        environment=bench.environment_provenance(),jobs_per_condition=4,threads_per_child=1,
        caveats=['Four identical grouped-fit workloads, not bootstrap draws or causal validation.',
                 'Single timing per condition; includes process startup/data preparation and concurrent baseline load.',
                 'No CUDA MPS or driver changes. Each child uses a separate CUDA context.',
                 'No production changes.']))
    results=[]
    try:
        with bench.GPUSampler() as telemetry:
            for backend,workers in [('xgboost_cuda',1),('xgboost_cuda',2),('xgboost_cuda',4),('xgboost_cpu',4)]:
                condition=f'{backend}_{workers}'
                folder=out/condition
                with logger.stage(condition,total=4):
                    started=time.perf_counter()
                    records=[]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                        pending=[executor.submit(launch_child,folder,backend,j) for j in range(4)]
                        for future in concurrent.futures.as_completed(pending):
                            record=future.result()
                            records.append(record)
                            logger.progress(condition,len(records),4,job=record['job'],job_status=record['status'])
                    wall=time.perf_counter()-started
                    if not all(r['status']=='completed' and r['return_code']==0 for r in records):
                        raise RuntimeError(f'Incomplete condition {condition}: {records}')
                    result=dict(backend=backend,workers=workers,jobs=4,wall_seconds=wall,
                        jobs_per_hour=4*3600/wall,
                        mean_task_fit_seconds=sum(r['total_seconds'] for r in records)/4,
                        cuda_models=sum(r['trained_cuda_models'] for r in records),records=records)
                    results.append(result)
                    bench.atomic_write_json(out/'conditions.json',results)
                    logger.event('condition_result',**{k:v for k,v in result.items() if k!='records'})
            bench.atomic_write_json(out/'gpu_telemetry.json',telemetry.samples)
        observed=[s for s in telemetry.samples if 'memory_mib' in s]
        bench.atomic_write_json(out/'benchmark_result.json',dict(status='completed',conditions=results,
            maximum_gpu_memory_mib=max(s['memory_mib'] for s in observed) if observed else None,
            maximum_gpu_utilization_pct=max(s['utilization_pct'] for s in observed) if observed else None,
            production_changed=False))
        logger.close('completed',production_changed=False)
        return 0
    except Exception as exc:
        bench.atomic_write_json(out/'benchmark_result.json',dict(status='failed',error=str(exc),
            traceback=traceback.format_exc(),conditions=results,production_changed=False))
        logger.close('failed',error=str(exc))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
