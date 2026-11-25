# Public 360 implementation status

Approved design: [public_360_approved_plan.md](public_360_approved_plan.md).
Context compacted before implementation on 2026-09-30.

- Completed: private-code audit, public schema/coverage investigation, design review and user approval.
- Implemented: notebook-owned grouped nuisance models, shared support, row-level ATT, refit bootstrap, sensitivity bounds, null checks, simulations, robustness, case exports and guarded confirmation.
- Implemented: selective pinned public acquisition, timing/visibility adapter, source-bound checkpoints, notebook runner and bounded worker logging.
- Verified: 52 remote tests passed, no skips, including a real Jupyter runner regression (2026-09-30 11:01 UTC).
- Verified: `smoke-v2-20260930` completed in tmux at 11:02 UTC: six public matches, ten successful refit cluster draws, 52 small simulation datasets without computational failures, saved notebook/exports. Relaxed smoke gates are engineering checks, not causal validation.
- Corrected during smoke: raw JSON cell-source arrays require `nbformat.reads`, rather than `from_dict`, before nbclient execution. The initial failed attempt is preserved under `smoke-20260930`.
- Current: full production study launched in `public-causal-study`, run ID `public-study-20260930`, four workers / one numerical thread each.
- Pending: production support/development gates, 1,000 successful draws, null/sensitivity/simulation checks, immutable lock and guarded confirmation. These are not claimed complete.
- No empirical public ATT or completed bootstrap/simulation results yet.

Remote root: `/home/joaomarcostomaz/causal/Simplified_EPV_in_PFF_Data`.
Remote data: `data/public_statsbomb_causal/`.
Notebook: `results/models/public_statsbomb_causal_study.ipynb`.

Live production logs: `data/public_statsbomb_causal/runs/public-study-20260930/full/development/{status.json,events.jsonl,stdout.log}`. Worker status: `worker_status/*.json`; sampled worker logs have bounded size. Runner logs: `.../full/execution/`. Successful smoke artifacts: `data/public_statsbomb_causal/runs/smoke-v2-20260930/smoke/`.

Scientific source hash: `2fa31ae58d4983d6490fb3f749455836e37db3548c700ce28825ac1a2e7f9f95`. Temporary construction fragments are removed after embedding; scientific edits belong in the notebook's tagged core cells. Helpers contain only data/runtime plumbing. No private PFF files were changed.

GPU follow-up: separate serial and concurrent benchmark pilots completed on 2026-09-30. CUDA compatibility passed without environment changes. Four GPU jobs reduced representative grouped propensity workload time from42.78s (four XGBoostCPU jobs) to36.29s; sampled peak1890MiB/69% utilization. These are pilot timings, not complete causal-pipeline acceleration. Baseline notebook/run remain unchanged. See `public_360_gpu_benchmark.md` and saved benchmark results.

Continue from the plan and saved audits, not by repeating the initial investigation. Update this file at implementation milestones and place run-specific progress in the structured runtime logs.
