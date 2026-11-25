# Public 360 implementation status

Approved design: [public_360_approved_plan.md](public_360_approved_plan.md).
Context compacted before implementation on 2026-09-30.

- Completed: private-code audit, public schema/coverage investigation, design review and user approval.
- Implemented: notebook-owned grouped nuisance models, shared support, row-level ATT, refit bootstrap, sensitivity bounds, null checks, simulations, robustness, case exports and guarded confirmation.
- Implemented: selective pinned public acquisition, timing/visibility adapter, source-bound checkpoints, notebook runner and bounded worker logging.
- Verified: 52 remote tests passed, no skips, including a real Jupyter runner regression (2026-09-30 11:01 UTC).
- Verified: `smoke-v2-20260930` completed in tmux at 11:02 UTC: six public matches, ten successful refit cluster draws, 52 small simulation datasets without computational failures, saved notebook/exports. Relaxed smoke gates are engineering checks, not causal validation.
- Corrected during smoke: raw JSON cell-source arrays require `nbformat.reads`, rather than `from_dict`, before nbclient execution. The initial failed attempt is preserved under `smoke-20260930`.
- Superseded: extended-profile run `public-study-20260930` (tmux `public-causal-study`) was stopped at a durable bootstrap checkpoint on 2026-09-30 (`cancelled_after_durable_checkpoint`); its partial notebook/logs are retained. Its draws are never pooled with the submission run.
- Current: submission-profile run `public-submission-full-gpu-20260930` completed; see below.

## Submission run `public-submission-full-gpu-20260930` (completed 2026-09-30 23:05 UTC)

Isolated source `runs/public-submission-20260930/source`, notebook core `0959a70c7605b8fe…`, 103 source-bound tests.
Four workers, propensity XGBoost CUDA selected by the predeclared one-SE rule (Euro log-loss 0.11766 vs HGB 0.11757, SE 0.00224).
Wall time 77 min (development 4 min including benchmark; WC confirmation 73 min).

- Development: all five lock gates passed; `design_lock.json` frozen, `confirmation_allowed: true`, checksum verified before WC access.
- WC 2022 (pinned open-data commit `4b73468f`): 64 matches, 1,228 supported shots, 20,532 controls, control ESS 1,006, key SMD 0.075, max donor share 0.5%.
- ATT 0.0308 signed first-goal reward per supported decision (+3.08 per 100); match-cluster score 95% CI [0.0067, 0.0548] (t, 63 df).
- Bootstrap: 500 successful fixed-spec refits (predeclared precision extension from 200 triggered by lower-endpoint MCSE 0.00313 > 0.05 x width); percentile interval [0.0029, 0.0561]; precision passes at 500. 489/500 resamples fail at least one support diagnostic (duplicated matches concentrate weights); retained per plan and reported as a caveat.
- Inference agreement: bootstrap SD / score SE = 1.11 (rule 2/3–3/2).
- Nulls (50 each): within-match treatment mean 0.0022, reject 2%; dummy Y mean 0.0011, reject 4%.
- Simulations (8 x 50): zero scenarios reject 0%; coverage 0.88–1.00; action-visibility bias -0.0068 (descriptive).
- Analyses (18/18 ok): outcome model GAM/HGB 0.028/0.030; horizons Y5/Y10/Y30 0.039/0.032/0.032; visible-disk population 0.023 (CI includes 0); targeted team removal 0.023–0.037.
- Sensitivity (DVDS): lower bound crosses zero at Gamma about 1.35.
- Subgroups: only `angle_low` qualified (0.016, joint CI includes 0); contrasts insufficient.
- Claim status: `completed` (all seven checks pass). Qualification: conditional approximate match-cluster inference; repeated national teams and nuisance remainder remain limitations; no automatic publication.

Report (presentation of saved artifacts, generated 2026-10-01): `runs/public-submission-full-gpu-20260930/full/report.ipynb` and `full/report/`; 108 source-bound tests pass with the report cells (core hash unchanged, `runs/public-submission-report-20260930/manifests`).
Results: `runs/public-submission-full-gpu-20260930/full/confirmation/{submission_results.json,phase_result.json,exports/}`; notebook `full/executed.ipynb`.
Engineering pilots/rehearsals (`public-submission-pilot*`, `pilot3/rehearsal`) are not results and must not be quoted.

Remote root: `/home/joaomarcostomaz/causal/Simplified_EPV_in_PFF_Data`.
Remote data: `data/public_statsbomb_causal/`.
Notebook: `results/models/public_statsbomb_causal_study.ipynb`.

Live production logs: `data/public_statsbomb_causal/runs/public-study-20260930/full/development/{status.json,events.jsonl,stdout.log}`. Worker status: `worker_status/*.json`; sampled worker logs have bounded size. Runner logs: `.../full/execution/`. Successful smoke artifacts: `data/public_statsbomb_causal/runs/smoke-v2-20260930/smoke/`.

Scientific source hash: `2fa31ae58d4983d6490fb3f749455836e37db3548c700ce28825ac1a2e7f9f95`. Temporary construction fragments are removed after embedding; scientific edits belong in the notebook's tagged core cells. Helpers contain only data/runtime plumbing. No private PFF files were changed.

GPU follow-up: separate serial and concurrent benchmark pilots completed on 2026-09-30. CUDA compatibility passed without environment changes. Four GPU jobs reduced representative grouped propensity workload time from42.78s (four XGBoostCPU jobs) to36.29s; sampled peak1890MiB/69% utilization. These are pilot timings, not complete causal-pipeline acceleration. Baseline notebook/run remain unchanged. See `public_360_gpu_benchmark.md` and saved benchmark results.

Continue from the plan and saved audits, not by repeating the initial investigation. Update this file at implementation milestones and place run-specific progress in the structured runtime logs.
