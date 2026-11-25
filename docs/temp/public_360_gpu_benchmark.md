# GPU benchmark (separate from the causal study)

The baseline notebook/experiment remains unchanged. This benchmark reads only saved Euro decisions and tests one match-held-out outer fold. No World Cup files, primary bootstrap checkpoints, frozen support, packages or drivers are changed.

Three backends: existing scikit-learn CPU models, XGBoost CPU, and XGBoost CUDA. Only CPU-versus-CUDA XGBoost isolates hardware: replacing HGB with XGBoost is a modeling revision and predictions can differ. The experiment retains the original feature preprocessing, training-match grouping, calibration and validation functions through an isolated process-local factory. Every fitted CUDA model is checked against its actual booster device; silent CPU fallback fails the benchmark.

Compact timings include three repetitions of grouped propensity and both outcome fits, including preprocessing, nested calibration and held-out prediction/transfer costs. All compact outcome comparisons use the same match-held-out reference donor weights. A full candidate-selection workload is also timed once per backend; it is a pilot, not a repeated benchmark or full supported ATT. Prediction differences and held-out losses are saved, but this is not causal validation. CPU timings occur alongside the four-worker baseline, so they are not idle-machine measurements.

Launch in a separate tmux session, one numerical thread / one GPU task at a time:

```sh
tmux new-session -d -s public-causal-gpu-benchmark -c /home/joaomarcostomaz/causal/Simplified_EPV_in_PFF_Data '.venv/bin/python -u scripts/benchmark_public_causal_gpu.py --run-id gpu-benchmark-20260930 --repeats 3 --full-suite'
```

Results/logs: `data/public_statsbomb_causal/benchmarks/gpu-benchmark-20260930/`. Read `benchmark_result.json`, `summary.csv`, `timings.csv`, grouped provenance, GPU telemetry and warnings before recommending a switch. A CUDA/driver compatibility failure is reported, never hidden by upgrading the machine or CPU fallback. A new backend would require new source/config identifiers and all scientific gates; this benchmark cannot authorize confirmation or mixing bootstrap draws from different pipelines.

Execution milestone: benchmark contract tests passed locally and on Vulcano. The live CUDA probe passed at 2026-09-30 11:34 UTC using the existing driver and installed XGBoost build, without upgrades. The first benchmark completed at 11:43 UTC.

Measured compact-fit medians (three repetitions): propensity sklearn28.60s / XGBoostCPU22.18s / CUDA20.45s; continuation outcome7.20s /5.43s /5.27s; shot outcome4.90s /4.75s /4.79s. One full candidate-selection workload: sklearn71.94s / XGBoostCPU54.78s / CUDA53.09s. CUDA-only benefit in the full pilot was approximately3%, not the approximately26% saving against a different library. Peak sampled GPU memory524MiB/utilization37%. CPU/GPU predictions differed (mean absolute propensity difference0.00169), so outputs are not interchangeable or mixable in a production run. No primary estimate or sensitivity-stage speedup is established by this benchmark.

The exact saved v6 workflow notebook contains no explicit CUDA/PyTorch/XGBoost/LightGBM code. Its propensity stabilization helper and imported reward helper also have no GPU backend references. CPU logistic/Ridge/random-forest estimators are configured. This does not identify which earlier notebook/session produced remembered GPU utilization.

A second throughput pilot, `scripts/benchmark_public_causal_gpu_concurrency.py`, runs four identical grouped propensity workloads at CUDA concurrency1/2/4 and compares XGBoost CPU at concurrency4. Each child is an independent process with one numerical thread, its own logs and verified devices; no production or driver changes. This tests useful concurrency rather than allocating unused VRAM. Results: `data/public_statsbomb_causal/benchmarks/gpu-concurrency-20260930/`. One measurement per condition is exploratory; it includes process startup and concurrent baseline load. It does not benchmark complete bootstrap/sensitivity throughput.

## Concurrent throughput results

Completed at approximately 2026-09-30 11:53 UTC; all conditions passed. Every CUDA condition verified 64 CUDA model fits (four grouped propensity workloads, including nested calibration).

| Backend | Concurrent jobs | Time for four workloads | Workloads/hour |
|---|---:|---:|---:|
| XGBoost CUDA | 1 | 106.37s | 135.38 |
| XGBoost CUDA | 2 | 57.46s | 250.62 |
| XGBoost CUDA | 4 | 36.29s | 396.80 |
| XGBoost CPU | 4 | 42.78s | 336.63 |

Four CUDA workers provided 2.93x throughput versus one CUDA worker. At equal concurrency CUDA saved approximately15.2% elapsed time / increased throughput17.9% versus XGBoost CPU in this representative pilot. Peak sampled GPU usage1890MiB (approximately1.85GiB), utilization69%. Concurrent jobs can therefore use the GPU more effectively, but filling24GiB is neither necessary nor a speed objective. Each job took longer under concurrency; the benefit is aggregate throughput.

Recommendation: four independent GPU fitting workers are a promising optional backend for a separately identified, revalidated study. Keep support construction/CPU preprocessing/calibration and original-match grouping intact. Do not increase tree complexity or fabricate workload just to use memory. The baseline is unchanged, and these pilots do not establish complete bootstrap, sensitivity-stage or simulation speedups. A replacement backend changes predictions and may change support/model selection; existing CPU draws cannot be mixed with new GPU-model draws.
