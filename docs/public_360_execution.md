# Executing the public 360 study

## Submission profile: isolated source and shared public data

The revised scientific specification is `docs/public_360_submission_profile_proposal.md`.
The notebook remains the scientific source. Run the revised experiment from its
isolated Vulcano source directory, using the existing project interpreter and
existing `data/public_statsbomb_causal/` directory. Source exports, imports and
helper checksums must refer to the isolated source; data and run outputs must
refer to the shared public folder. No environment upgrades, raw-data copy or
replacement of the active baseline source is required.

`--profile submission` explicitly selects the shorter experiment. `extended`
remains the runner default and retains the previous large experiment settings;
the legacy launch instructions below refer to that profile. Always specify the
profile for the revised run. Use a new run ID for each incompatible scientific
configuration. Existing baseline execution must continue until replacement
tests and a complete-path pilot pass; stop only that identified experiment at a
verified durable checkpoint before launching the replacement tmux job.

An example preparation and source-bound validation sequence on Vulcano is:

```sh
PUBLIC360_PROJECT=/home/joaomarcostomaz/causal/Simplified_EPV_in_PFF_Data
PUBLIC360_STUDY="$PUBLIC360_PROJECT/data/public_statsbomb_causal"
PUBLIC360_SOURCE="$PUBLIC360_STUDY/runs/public-submission-20260930/source"
PUBLIC360_PYTHON="$PUBLIC360_PROJECT/.venv/bin/python"
PUBLIC360_RUN=submission-pilot-example
cd "$PUBLIC360_SOURCE"
"$PUBLIC360_PYTHON" -u scripts/validate_public_causal.py --output-dir "$PUBLIC360_STUDY/runs/$PUBLIC360_RUN/manifests"
"$PUBLIC360_PYTHON" -u scripts/run_public_causal.py --profile submission --phase pilot --study-root "$PUBLIC360_STUDY" --run-id "$PUBLIC360_RUN" --workers 1 --device cpu --prepare-only
```

Validation rejects an empty discovered test suite, failures, errors and skipped
tests. Its `implementation_tests.json` is bound to the tested notebook core,
helper checksums and actual environment, including the installed XGBoost
version. The explicit destination preserves the baseline validation manifest.
The runner injects that run's `implementation_tests_path` into the notebook.
`--prepare-only` exports and configures source without acquiring data or
launching a scientific kernel. After inspecting the resolved config, launch
that prepared phase with identical arguments plus `--resume`, removing
`--prepare-only`. Pilot and smoke use Euro/synthetic data and cannot open WC.

Submission defaults are:

| Phase | Successful fixed refits | Maximum attempts | Simulation pairs per scenario | Nested simulation bootstrap | Null repetitions per kind |
|---|---:|---:|---:|---:|---:|
| smoke | 3 | 5 | 1 | 0 | 1 |
| pilot | 5 | 8 | 2 | 0 | 2 |
| other phases | 200 | 300 | 50 | 0 | 50 |

These are execution allocations, not development freeze completion gates.
Production follows the declared eight simulation scenarios, two null kinds and
200 fixed-design confirmation refits. Any predeclared precision-driven extension
to 500 must remain identified and auditable. Explicit count overrides, device,
profile, source/data roots and import path are persisted and hashed.
`--device cpu|cuda` selects the explicit development candidate backend;
actual CUDA availability, fitted booster device, warnings and model diagnostics
must be checked before choosing and freezing a backend. An available GPU alone
does not establish a complete-experiment speedup.

`--reuse-development <existing-development-directory>` requests a verified
artifact import under the shared public folder. It does not forge a resumed
run or replace original provenance. Imports retain original checksums,
source/helpers/environment/configuration/folds/domain/model lineage and an
explicit reuse purpose. Bootstrap draws from incompatible procedures or
configurations remain separate. To resume new work, use identical arguments
with `--resume`; incompatible configuration or source hashes require a new ID.

The submission design lock requires exactly `timing_audit`, `leakage_tests`,
`development_diagnostics`, `weighted_validation` and `design_sanity`. The adapter
independently recognizes and checks that profile. Unknown profiles, missing
gates, weakened gate overrides, cross-profile confirmation and checksum or
specification tampering are rejected. The immutable lock contains `profile`,
`specification_manifest` and its `specification_manifest_hash`; the manifest
records complete choices in `horizon`, `model`, `backend`, `support`, `subgroup`,
`inference`, `simulation` and `budget`. Extended/legacy gates remain the previous
strict contract. A passing design lock authorizes confirmation access; final
claim completion separately requires all declared scientific checks and draws.

Progress logs report total, cached and freshly completed work. Resumed timing
uses `cached_completed` to exclude old completions from new throughput and ETA.
Benchmark 5–10 complete fixed refits plus representative null/simulation tasks
under measured worker counts and live CPU/GPU load. Report component and wall
throughput, failures/retries, remaining allocations and an ETA range. Approximately
14 hours is a planning budget; the revised run's measured ETA is unknown until
that benchmark runs. Heartbeats and launched processes do not establish
scientific completion. Preserve provisional, failed, unsupported and incomplete
statuses in the saved notebook and phase/claim outputs.

### Submission confirmation stages and monitoring

After a passing lock, `confirmation` (or the second half of `full`) runs, in order:

1. Protected WC acquisition/canonicalization, timing audit, one primary fit (m1 fitted
   once for diagnostics only) and early exports: `exports/att.json`, row contributions,
   support/balance, comparators, cases and `primary_status.json` (provisional). An
   unsupported WC domain stops here as `unsupported`; nothing is relaxed.
2. Point analyses without refits: DVDS point bounds/tipping (`exports/sensitivity`),
   six strata with joint multiplier max-t intervals and contrasts
   (`exports/subgroups_joint.json`), exact fixed-nuisance match/team score screen
   (`exports/score_influence.json`).
3. One persistent worker pool (`tasks/<stage>/<task-hash>.json`, durable per task):
   200 successful fixed-spec refits (≤300 attempts; one predeclared precision
   extension to 500), 11 bounded analyses plus 5 targeted match and 2 targeted team
   refits, 50+50 nulls, 8 scenarios × 50 independent simulation pairs.
4. `submission_results.json` and `phase_result.json` with the claim status:
   `completed` only when support, bootstrap count, Monte Carlo precision, score/refit
   agreement (SD ratio in [2/3, 3/2]), nulls, simulations and analyses all pass;
   otherwise `incomplete`, `inconclusive_uncertainty` or `unsupported`.

Monitor:

```sh
RUN="$PUBLIC360_STUDY/runs/<run-id>/full"
cat "$RUN/confirmation/overall_progress.json"        # remaining tasks, measured concurrency, ETA
tail -f "$RUN/confirmation/stdout.log" | grep -v heartbeat
ls "$RUN/confirmation/worker_status/"                 # live per-worker stage/fold
```

`overall_progress.json` (also written during development benchmarking) sums remaining
required records × measured mean fresh seconds per task kind, divided by measured
concurrency. Cached completions are excluded; failures/retries are not forecast.

Stop cleanly: `touch "$RUN/confirmation/STOP_REQUESTED"` (or SIGINT/SIGTERM to the
kernel). Admitted tasks finish and are saved; no new tasks start; the phase ends as
`stopped`. Remove the token and relaunch with identical arguments plus `--resume`.

### Results report

The notebook ends with a `public-causal-report` section (Query → Model → Identify → Estimate → Refute → Interpretation) that only reads saved artifacts: cohort flow, frozen design, held-out nuisance fit, support/overlap/balance, ATT and comparators, bootstrap, nulls, simulations, sensitivity, robustness, influence, subgroups, cases and limitations. Report cells are not core-tagged, so they never change `CORE_SOURCE_HASH`. A runner execution renders them after the study cell; for an existing run, render without re-running the study:

```sh
"$PUBLIC360_PYTHON" scripts/report_public_causal.py --phase-dir "$PUBLIC360_STUDY/runs/<run-id>/full"
```

This writes `<phase-dir>/report.ipynb` and `<phase-dir>/report/` (PNG/CSV) and never modifies `executed.ipynb` or run artifacts. Missing artifacts (pilot, stopped, unsupported runs) are shown as not available. `exports/model_validation.json` lists hyper-parameter search results only and is empty for frozen-spec fits; held-out fit quality is in the report's model-fit section.

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
