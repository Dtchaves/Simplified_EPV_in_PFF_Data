# Public causal notebook execution

The scientific source is `results/models/public_statsbomb_causal_study.ipynb`.
Tagged `public-causal-core` cells are exported verbatim to a hash-named module
for workers and tests. There is one `public-causal-parameters` cell containing
a literal `STUDY_CONFIG` dictionary. The runner preserves its defaults and
overrides explicit execution settings.

From the Vulcano project directory, after implementation validation:

```sh
rtk proxy .venv/bin/python -u scripts/run_public_causal.py --phase smoke --run-id euro-study
rtk proxy tmux new-session -d -s public-causal '.venv/bin/python -u scripts/run_public_causal.py --phase development --run-id euro-study'
rtk proxy tmux attach -t public-causal
```

Run independent long simulation work through `--phase simulation`; overrides
include `--simulation-datasets`, `--simulation-bootstrap-target`,
`--bootstrap-target`, `--bootstrap-max-attempts`, and `--null-reps`. Defaults are
four processes and one numerical thread each. Smoke defaults to ten bootstrap
targets, two simulation datasets per scenario and three null repetitions; the
full profiles default to 1,000 bootstrap targets and 200 datasets/nulls.
These commands launch jobs; they do not establish scientific completion.

Resume an interrupted phase with identical scientific settings:

```sh
rtk proxy .venv/bin/python -u scripts/run_public_causal.py --phase development --run-id euro-study --resume
```

A fresh kernel reruns state-building cells and reuses compatible scientific
checkpoints. It never skips arbitrary completed cells without restoring their
Python state. Per-cell checkpoints and `executed.ipynb` preserve partial output
and tracebacks. Content/configuration changes require a new run ID.

Artifacts live exclusively under `data/public_statsbomb_causal/runs/<run-id>/`.
Each phase has `configured.ipynb`, `executed.ipynb`, provenance and an `execution/`
directory with JSONL events, readable stdout and atomic heartbeat/status files.
Scientific stages have their own logger; workers use separate PID logs and
status files so they cannot overwrite the master's progress. Scientific final
status belongs in `phase_result.json`; absent that result, the runner reports
only `execution_complete`, not study completion. Failed/blocked/scientifically
unsupported phases return a nonzero process exit status.

Confirmation requires a development `design_lock.json` and matching `.sha256`
sidecar. All five measured gates must pass: timing audit, leakage tests,
development diagnostics, null checks and simulation validation. The lock binds
scientific code, data and configuration hashes; acquisition independently
validates it before World Cup access. Neither an executed notebook nor successful
smoke tests grant confirmation access. Existing frozen locks cannot be replaced.

```sh
rtk proxy .venv/bin/python -u scripts/run_public_causal.py --phase confirmation --run-id euro-study
```

The initial disk guard requires 10 GiB free and caps the dedicated study folder
at 8 GiB. Checkpoints retain compact metrics/seeds/status rather than per-draw
models or large row predictions. Do not launch the expensive phases until smoke
timings and validation have been reviewed.
