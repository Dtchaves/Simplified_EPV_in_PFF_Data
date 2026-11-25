# Executing the public 360 study

Scientific source: `results/models/public_statsbomb_causal_study.ipynb`. Approved design: `docs/public_360_approved_plan.md`. Existing private notebooks/data are not overwritten. This document describes execution, not completed empirical findings.

The repository currently ignores all of `results/models/`. Before a future authorized commit/release, explicitly include this single new notebook (for example, `git add -f results/models/public_statsbomb_causal_study.ipynb`) rather than adding the whole private models directory. No staging or commit is performed by this workflow.

## Environment and storage

On Vulcano, work from `/home/joaomarcostomaz/causal/Simplified_EPV_in_PFF_Data`. The existing `.venv/bin/python` has the pinned scientific packages; no packages need to be upgraded for the initial run. `requirements_public_causal.txt` describes an isolated reproducible environment. Pure polygon primitives avoid an additional Shapely dependency. CPU only, four workers and one numerical thread per worker.

All new data/results live in `data/public_statsbomb_causal/`: pinned raw files and manifests, hash-addressed canonical decisions, and per-run folders. A 10 GiB free-space floor and 8 GiB study budget protect this nearly full disk. Large computations save summaries/checkpoints, not fitted models for every draw. Never delete existing PFF data to make room.

## Validate and launch

Run fixture tests first (writes the source-bound implementation gate):

```sh
.venv/bin/python -u scripts/validate_public_causal.py
```

Launch the engineering pilot in tmux:

```sh
tmux new-session -d -s public-causal-smoke -c /home/joaomarcostomaz/causal/Simplified_EPV_in_PFF_Data '.venv/bin/python -u scripts/run_public_causal.py --phase smoke --run-id smoke-20260930 --workers 4'
tmux attach -t public-causal-smoke
```

The pilot uses six interleaved Euro matches, ten cluster draws and small controlled simulations. Its relaxed support/sample gates are strictly engineering checks: it cannot authorize World Cup confirmation or a causal headline.

Production (after validated implementation/pilot):

```sh
tmux new-session -d -s public-causal-study -c /home/joaomarcostomaz/causal/Simplified_EPV_in_PFF_Data '.venv/bin/python -u scripts/run_public_causal.py --phase full --run-id public-study-20260930 --workers 4'
```

`full` performs development then confirmation only if a passing immutable development lock exists. No passing support rule means a scientific failure result, not an exception to the holdout gate. Simulation can also run independently:

```sh
tmux new-session -d -s public-causal-simulation -c /home/joaomarcostomaz/causal/Simplified_EPV_in_PFF_Data '.venv/bin/python -u scripts/run_public_causal.py --phase simulation --run-id simulation-20260930 --workers 4'
```

Do not run two four-worker jobs concurrently without adjusting their worker counts. Production requests 1,000 successful primary cluster draws, 200 null draws per kind/domain branch, and the approved simulation operating-characteristic study. Coverage simulations are intentionally expensive; measured progress/ETA and checkpoints determine duration, not a promised deadline.

To resume, reuse the identical run arguments plus `--resume`. A changed source, helper, environment, config, raw data or checksum requires a new run ID. A clean kernel reruns definition/state cells and restores only compatible content-addressed scientific artifacts. It does not blindly skip cells based on old outputs.

## Track progress

Each phase directory contains `events.jsonl`, `stdout.log`, `status.json`, configs/environment/provenance, hash-named generated runtime source, compact checkpoints, exports, and `phase_result.json`. The runner separately writes `execution/status.json` and saves `executed.ipynb` after cells, including partial notebooks on exceptions. Worker-specific logs avoid concurrent writes to a shared JSONL file.

```sh
tmux list-sessions
tail -f data/public_statsbomb_causal/runs/public-study-20260930/full/development/stdout.log
```

Events carry UTC time, stage/fold/draw IDs, seeds, counts, elapsed time and errors. Status/heartbeats distinguish ongoing computation from a stalled process. Failed and extreme bootstrap draws remain auditable. Read the final phase status: a launched process or a notebook that merely executed is not a completed supported causal study.

Actual first successful pilot: `smoke-v2-20260930`, tmux session `public-causal-smoke-v2`. It passed on 2026-09-30 after an array-source runner issue was fixed and regression-tested. The original failed pilot is retained for diagnosis. See `docs/public_360_execution_status.md` for the latest launched run and measured milestones.

Live per-worker progress is in `worker_status/*.json`; worker traces are sampled and capped at 8 MiB each. Full replicate outcomes remain in master checkpoint/attempt exports. A heartbeat confirms the process is alive, not that its scientific checks passed.

## Interpretation and release

The notebook exports row ATT scores, support/weight/balance diagnostics, grouped model validation, uncertainty, nulls, sensitivity bounds, simulation results, subgroup summaries, and case/rating packets. Read the support and uncertainty gates before interpreting the estimated effect. Report signed first-goal reward per100 supported decisions, with horizon and target attached; do not call it EPV or net expected goals.

The public package includes only pinned public source data under its source agreement, new code/config/environment, notebook and generated research outputs. Private PFF exports/refutation reconstructions are excluded. Nothing is automatically published; analyst ratings require actual supplied assessments, not the blank template. License/attribution are stored with the acquisition manifest.
