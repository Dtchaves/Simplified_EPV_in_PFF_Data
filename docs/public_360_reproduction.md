# Public 360 study — reproduction package

## Artifacts on Vulcano (relative to `data/public_statsbomb_causal/runs/`)

| item | path |
|---|---|
| run of record | `public-submission-full-gpu-20260930/full/` (`executed.ipynb` sha256 `a8a59c46…`, `report.ipynb` `a43794a3…`) |
| scientific source used by the run | `public-submission-20260930/source/` (notebook core hash `0959a70c7605…`) |
| design lock | `public-submission-full-gpu-20260930/full/development/design_lock.json` (sha256 `bd5d6f77…`) |
| post-hoc Tier 1/2 source + outputs | `public-tier12-20261001/` (`source/`, `source_r2/`, `tier1/`, `features/`, `development/`, `confirmation/`, `report/`) |
| provenance copy of the post-hoc source | `public-tier12-20261001/source/` (unchanged since the Tier 1/2 runs) |
| updated post-hoc source (guards, F3 supplement) | `public-tier12-20261001/source_r2/` = `source/` plus updated files; hashes in `source_r2_sha256.txt` |
| corrected Euro F3 retention and overlap | `public-tier12-20261001/development/6c15146a1e7f/f3_supplement.json` (written by `source_r2`, script `953845ec…`) |
| environment + file hashes | `public-tier12-20261001/reproduction/{reproduction_manifest.json,pip_freeze.txt}` |
| original-run integrity | `public-tier12-20261001/original_manifest.json` (1,227 files); verified twice: `original_manifest_verify.json` (04:20:11Z) and `original_manifest_verify-20261001T115942Z.json` (11:59:42Z), both identical |

## Environment

- Python 3.10.12, xgboost 3.2.0, scikit-learn 1.6.1, pandas 2.3.3, numpy 2.2.6,
  scipy 1.15.3.
- NVIDIA RTX 4090 with driver 560.35.03. XGBoost propensity runs on `cuda:0`;
  falling back to CPU raises an error.
- The full list of 196 packages is in `pip_freeze.txt`.

## Commands

Run every command from the isolated source directory:

```bash
cd data/public_statsbomb_causal/runs/public-tier12-20261001/source
export PYTHONPATH=src
```

Test suite (`test_data.py` needs `PYTHONPATH=src`):

```bash
./.venv/bin/python -m unittest discover -s tests/public_causal
```

Post-hoc analyses. Outputs are written only below `--run-root`; the original run
is opened read-only.

```bash
R=../ ; O=../../public-submission-full-gpu-20260930/full
./.venv/bin/python scripts/tier12_public_causal.py --phase guard   --run-root $R --original $O
./.venv/bin/python scripts/tier12_public_causal.py --phase tier1   --run-root $R --original $O
./.venv/bin/python scripts/tier12_public_causal.py --phase prespec --run-root $R --original $O
for c in development confirmation; do
  ./.venv/bin/python scripts/tier12_public_causal.py --phase features --cohort $c --run-root $R --original $O
  ./.venv/bin/python scripts/tier12_public_causal.py --phase compare  --cohort $c --run-root $R --original $O
done
./.venv/bin/python scripts/tier12_public_causal.py --phase report       --run-root $R --original $O
./.venv/bin/python scripts/tier12_public_causal.py --phase guard --verify --run-root $R --original $O
```

Re-running any of these on the existing run root now stops at the output guards
(see below). Corrected Euro F3 retention and overlap, from the saved artifacts
(no refit), run from `source_r2/` (same interpreter and `PYTHONPATH=src`):

```bash
cd ../source_r2
./.venv/bin/python scripts/tier12_public_causal.py --phase f3-supplement --cohort development --run-root $R --original $O
./.venv/bin/python scripts/tier12_public_causal.py --phase guard --verify --run-root $R --original $O
```

`f3-supplement` first re-derives the saved F3 ATT and shot count from the saved
design and stops if they differ (here: difference 0.0). It supports the
development cohort only.

Test suite results: `source/` 140 tests OK on Vulcano (`logs/suite_final.log`);
`source_r2/` 166 tests OK on Vulcano with xgboost (`logs/suite_r2.log`). Locally
without xgboost the 166 tests give 0 failures and 2 errors
(`ModuleNotFoundError: xgboost` in `test_submission_models`), an environment
limit.

The current script (`source_r2/`, sha256 `953845ec…`) guards against unsafe re-runs:

- Every phase refuses up front, before computing, if any of its outputs exist:
  `tier1` (all ten `tier1/` files), `report` (`tier2_estimates.csv`, `.md`),
  `compare` (`summary.json`, `partial_F1.json`, `partial_F2.json`, and
  `design_A.joblib` for the development cohort) and `f3-supplement`
  (`f3_supplement.json`). `guard` refuses if `original_manifest.json` exists.
- `features`: versioned `parquet` and `.meta.json` files are never overwritten
  (same key: reused; other key: refused). The `<cohort>.current.json` pointer is a
  no-op if its content is identical and is refused if it differs.
- `prespec`: a no-op if the recorded hash matches, an error if the file changed.
- `guard --verify` writes a new `original_manifest_verify-<UTC>.json` each time.
- `compare` refuses to run if the prespecification hash differs from the
  recorded one, and the World Cup `compare` refuses to run before the Euro
  outputs exist.
- The check-then-write is not atomic: use one runner per run root.

These guards were added after the runs of record. The outputs of record were
produced by earlier script versions without them (table below). In this run
`tier1` ran twice (03:46:58Z and 03:47:58Z) and `report` ran twice (04:19:32Z, which
failed on the missing `tabulate`, and 04:20:11Z). The first `tier1` output was moved
by hand to `tier1_v0/`, which is kept but not used.

### Script versions

`scripts/tier12_public_causal.py` sha256 by output:

| script sha256 | produced | evidence |
|---|---|---|
| not recorded (earlier version, before the `prespec`, `compare` and `report` phases) | `original_manifest.json`, `tier1/` (second run, 03:47:58Z) | `logs/events.jsonl` |
| `d0ef1cf2…` | `development/` and `confirmation/` comparison outputs (`summary.json`, partials, `design_A.joblib`); value recorded in `reproduction_manifest.json` (`tier12_source_files`) | `verification.md`; manifest |
| `df376011…` | `report/tier2_estimates.{csv,md}` (rerun 04:20:11Z) and `original_manifest_verify.json` (same second and upload, inferred from timestamps); differs from `d0ef1cf2…` only in the report fallback when `tabulate` is absent | `verification.md` |
| `953845ec…` | `development/6c15146a1e7f/f3_supplement.json` and `original_manifest_verify-20261001T115942Z.json`; adds the output guards and the `f3-supplement` phase | `f3_supplement.json` (`script_sha256`); `source_r2_sha256.txt` |

`reproduction_manifest.json` therefore lists `d0ef1cf2…` for the script, not the
shipped `953845ec…`. The manifest was not regenerated.

`source/` on Vulcano is the provenance copy and is unchanged. `source_r2/` is
`source/` plus updated files, with all hashes in `source_r2_sha256.txt`. Against the
file hashes in `reproduction_manifest.json`, five code and test files differ:
`scripts/tier12_public_causal.py` (`d0ef1cf2…` → `953845ec…`),
`src/PublicCausal/dependence.py` (`51c52688…` → `10b892e0…`),
`tests/public_causal/test_context_features.py` (`76fc9c54…` → `f0df33ad…`),
`tests/public_causal/test_dependence.py` (`67f0258e…` → `69de403c…`) and
`tests/public_causal/test_tier12_pipeline.py` (`4bec0e81…` → `13c5f871…`). The hash
list also contains three documentation files that the manifest does not list.
Two of them were uploaded with corrected wording: `public_360_tier12_results.md`
(`9ff182a9…`) and `public_360_disclosures.md` (`fd172d8e…`). The third,
`public_360_reproduction.md` (`c5602c91…`), is the copy inherited from `source/`. The
documentation in the repo is newer than the copies in `source_r2/`; the repo
versions are the current text.

The `source_r2/` changes are the registry checks in `dependence.py`, stronger tests, the
write-once guards, the P0-supported F3 retention with overlap, and the
`f3-supplement` phase. The comparison estimates were not recomputed.

## Known issue: design-lock path in `run_config.json`

`run_config.json` → `config.design_lock_path` points to
`public-submission-full-gpu-20260930/design_lock.json`, which does not exist. The
lock that was actually written and used is under `full/development/`.

If the runner is restarted with that config, it would not find the lock. It would
then redo the Euro development phase instead of reusing the frozen design. Two
safe options:

- Pass the real lock path explicitly.
- Reuse the saved `development/frozen_design.joblib` and `fit.joblib`, as the
  Tier 1/2 script does.

The run of record itself is unaffected. Its provenance (`core_hash`,
`design_lock_hash`) binds the real lock.

## Post-hoc status

Tier 1 analyses use saved scores only. Tier 2 feature tables are versioned
(`features/<cohort>-ctx-v1-<key>.parquet`, where the key hashes the feature
code, the canonical table and the raw files). Comparison outputs are stored
under `<cohort>/<key[:12]>/`, so a change to the features produces a new
directory instead of overwriting older results.
