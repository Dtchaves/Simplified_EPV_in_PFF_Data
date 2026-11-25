# Verified reuse of public 360 study artifacts

Status: **live inventory and read-only verification completed; no experiment changes**.
Verified on Vulcano on 2026-09-30, principally 14:13 UTC (11:13 BRT).
Environment verification followed at 14:14 UTC. This is a timestamped snapshot,
not a promise that files, draw counts or GPU availability remain unchanged.

## Conclusion

The revised submission study should not restart from scratch. Reuse the verified
Euro downloads, canonical decisions, development predictions and diagnostic outputs.
Preserve the existing adaptive development bootstrap as supplementary development
evidence, including failed support diagnostics. Do not pool its draws with a revised
fixed-specification bootstrap or World Cup confirmation.

Two conditional points from the earlier assessment are now resolved: **no final
`frozen_design.joblib` or `design_lock.json` exists yet**, and **no confirmation
directory/manifest exists** in the inspected study. A final all-Euro design still
needs to be fitted and locked; the cached fold-specific domains cannot replace it.

The current production run remains active and unchanged. No notebook/helper was
edited, no job was stopped or restarted, no model was fitted, and no GPU workload
was launched during this verification. Documentation writes are separate from the
running study. The proposed shorter profile remains a proposal, not an implemented
or newly approved specification; see [submission profile](public_360_submission_profile_proposal.md).

## Locations and identity

Vulcano project:
`/home/joaomarcostomaz/causal/Simplified_EPV_in_PFF_Data`

Study root: `data/public_statsbomb_causal/`.
Production phase: `runs/public-study-20260930/full/development/` beneath that root.
Scientific source: `results/models/public_statsbomb_causal_study.ipynb`.
tmux session: `public-causal-study`; four active CPU workers, one numerical thread each.

Notebook core hash:
`2fa31ae58d4983d6490fb3f749455836e37db3548c700ce28825ac1a2e7f9f95`.

The current remote core and all three PublicCausal helpers match the saved run
provenance. Local helpers also match those remote hashes. The current Python/package
environment fingerprint matches the saved fingerprint:
`c64ca963cc85369c60172ce3d1962c78e5195fe6b5f8257f27d524cb31d9c8d0`.

The detailed inventory, checksums, recorded configuration and audit results are
saved in [audit JSON](public_360_artifact_reuse_audit.json). Artifact hashes identify
the inspected files; actively written checkpoints require a fresh checksum check
when imported or resumed.

## Verified inventory and reuse decisions

Paths in the following table are relative to the study root unless marked otherwise.

| Artifact | What exists and was checked | Reuse decision |
|---|---|---|
| `raw/`, `manifests/development.json` | 311 pinned source files, 102 Euro matches, 1,102,932,145 raw bytes. Every file size/SHA-256, the manifest checksum and data fingerprint passed. | Direct reuse after verification; no Euro re-download needed. Preserve source agreement, attribution and pinned commit. |
| `canonical/development-b1e7c846453dfec59b9d61ea37c538e1476926016ba895877562321e3d028d6b.pkl` and sidecars | Full 102-match cache, 93,627,203 bytes; checksum and current full-cohort key verified. | Direct reuse with unchanged adapter, 3 m visibility radius and L1 region. |
| Other canonical development cache | The `34c17a...` cache covers only six smoke matches; checksum passed. | Engineering fixture reuse only; do not mistake it for the production cohort. |
| `runs/public-study-20260930/full/development/candidate_decisions.parquet` | 200,581 candidate rows, 37,043 eligible. Contains 5/10/15/30-second rewards, alternative goal-timing rewards, and 3/5/10 m visibility flags. | Reuse as the prepared development input; apply the intended eligibility/configuration explicitly. |
| Development `fit.joblib`, `fit.sha256.json`, `provenance.json` | Fit checksum and provenance key passed. Five outer folds; all have HGB propensity (100 iterations, 7 leaves), Ridge(alpha=10) m0/m1, support threshold 0.05. | Reuse predictions/results for the unchanged development analysis or as a documented baseline. Not a bank of trained outer-fold models. |
| Development `exports/nuisance_rows.parquet` | 37,043 rows across 102 matches; 2,317 shots, 2,181 supported shots; no missing e/m0/m1/portable-e predictions. | Reuse for calibration, weighted residuals, same-population estimators and development influence/subgroup diagnostics. |
| Development `exports/att_contributions/` | 36,863 supported row contributions and 102 match scores. Independent recomputation reproduces ATT and cluster-score SE. | Reuse for decomposition, approximate score uncertainty and influence screens; do not relabel the existing diagnostic interval as validated primary inference. |
| Comparator, subgroup and case exports | Matching pairs/donor reuse, comparator summaries, six case graphics, subgroup points, calibration CSVs and balance figures exist. | Reuse for unchanged development specifications; regenerate when support, predictions or targets change. Blank analyst-rating template is not evidence of completed ratings. |
| Development `bootstrap/bootstrap_ccd1ee006a3408414354.json` | Checksum passed; 104 records, attempts 0-103, all computational statuses and companion-statistic statuses `ok`. Summaries include DVDS endpoints and subgroup estimates, not fitted models. | Preserve as old adaptive-Euro uncertainty/sensitivity evidence. Never pool with revised fixed-model or WC draws. |
| Final design and later-stage outputs | No final frozen model/lock, phase result, completed bootstrap exports, or subsequent sensitivity/robustness/influence/null/simulation exports were present. | These are not available to import. Some sensitivity/subgroup numbers already exist inside bootstrap records, but are not completed later-stage validation. |
| World Cup confirmation | No confirmation directory or manifest; development acquisition lists Euro 2020 and Euro 2024, and raw 360 contains 102 match files. | No existing WC analysis to reuse in this study. Acquire/analyze only after the required revised design lock. |

Pinned public-data commit:
`4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`.

## Integrity and prediction checks

All 555 saved fold-provenance records have disjoint recorded training and testing
match IDs. Each original match belongs to exactly one outer prediction fold.
This verifies the saved membership records, not every possible leakage mechanism
or the causal exchangeability assumption.

The scientifically relevant scalar fields in `fit.joblib` and the Parquet export
agree exactly, including row/match IDs, treatment, outcome, covariates, predictions,
support and outer-fold membership. A strict whole-DataFrame `.equals()` initially
returned false because Parquet changes representations/dtypes for nested-list and
some boolean columns. All rows also agree after JSON normalization. This is not an
observed prediction discrepancy; import routines should normalize types explicitly.

Independently reconstructing the count-normalized ATT from saved supported rows,

`sum[T*(Y-m0) - (1-T)*e/(1-e)*(Y-m0)] / sum(T)`,

gives `0.017351150323680925`, agreeing with the saved `0.017351150323680935` to
floating-point precision. The denominator-aware match-score standard error also
matches exactly: `0.00978873572129941`. These are development diagnostics, not a
completed confirmation claim. Current exported intervals are labelled diagnostic
and use 1.96; adopting the proposed primary inference requires its own justification
and validation, not merely reusing this number.

The original development point support passes recorded gates: shot retention
94.13%, control ESS 1,774.74, maximum weight 7.826, key maximum SMD 0.0432 and
other maximum SMD 0.0925.

## Important qualification: completed draws versus support validity

At the checkpoint audit, 104/1,000 requested development draws were saved.
Mean measured draw computation was 379.57 seconds. Four workers were active on
the next batch; the checkpoint count is not the number of all computations in flight.

All 104 had computational `status=ok`, and all 104 companion-statistic calculations
completed. **Only 18 had recorded overall support diagnostics passing; 86 did not.**
There were 83 fold-support-selection failures, six key-balance failures and nine
other-balance failures; these categories overlap and must not be summed.

Consequently, "104 successful draws" means computational success, not 104
scientifically validated replicates. The implementation retains these finite draws
rather than dropping them; preserve that behavior and disclose the diagnostics.
Do not estimate a deceptively clean interval by retaining only the 18 passing draws.

This is relevant evidence of instability in the adaptive support procedure under
cluster resampling. It supports scrutinizing the inference design, but does not
prove that frozen support or a shorter profile fixes all uncertainty problems.
The revised procedure still needs diagnostic, null and known-truth checks.

## What is cached, and what is not

`fit.joblib` contains `rows`, `diagnostics`, `design`, `provenance` and
`model_validation`. Its development `design` contains fold-specific domains,
selected specifications and configuration; it does not contain trained outer-fold
propensity/outcome/portable models. Models fitted inside bootstrap draws are also
not persisted. We can avoid rerunning the completed development fit to inspect its
evidence, but cannot recover model parameters that were never saved.

The separate final `fit_frozen_design` step would save a trained all-Euro portable
support model, domain, model specifications and training match IDs. That step's
artifact does not yet exist. Its propensity/outcome model specifications must be
verified when fitted; five identical outer-fold selections are not proof of its
eventual all-Euro choices.

## Compatibility rules for the shorter study

- **Unchanged point analysis:** directly reuse existing Euro predictions/support
  for its diagnostics and comparisons. Preserve original source/configuration and
  label it development evidence.
- **New outcome horizon/timing:** prepared labels already exist. With identical
  rows, treatment and covariates, propensity/support predictions can be reused for
  a fixed-population sensitivity; m0 must be refitted for the new outcome. A
  full adaptive procedure that selects using outcome-dependent information is a
  different sensitivity and cannot silently skip those steps.
- **5 m visibility eligibility:** the flag is already present, permitting a
  restricted-population sensitivity with the same 3 m adjustment features. This
  changes the target population and requires new support diagnostics. Expanding
  local adjustment features to 5 m is a separate feature/model change.
- **Carry exclusion or feature ablation:** reuse raw/prepared input, but removing
  rows or adjustment features generally requires new nuisance fits and diagnostics.
  Do not treat original full-population predictions as an automatic fresh refit.
- **GPU candidate:** HGB-to-CUDA-XGBoost is a model change. Existing CPU predictions
  remain a useful baseline, not predictions from the new backend. Validate and freeze
  backend/settings before WC access; record XGBoost/CUDA/device versions explicitly.
- **World Cup bootstrap:** refit fixed-specification evaluation nuisances on the
  appropriate match samples and apply the properly frozen Euro domain. Old Euro
  draws cannot supply those fits, observations or inferential target.
- **Tests and benchmark infrastructure:** reuse code/cases/environment information;
  rerun affected tests and smoke validation for the new source. Earlier passing
  tests and engineering smoke results do not certify modified code.

At the read-only GPU query, the RTX 4090 had 4 MiB used of 24,564 MiB, 0% utilization
and no listed compute processes. This confirms the user's availability report at
that instant. No allocation was attempted. The current CPU run is not GPU-enabled.

## Safe import and execution policy

1. Preserve the original run and its checkpoints. Do not rename it to imply it used
   the revised procedure, change its counters, or overwrite source while it runs.
2. Use a new source/configuration/run identity for the revised profile. Approval of
   changes or stopping/replacing the existing run is separate from this audit.
3. Create an explicit import manifest for compatible artifacts: original run/path,
   SHA-256, source/helper/environment fingerprints, schema, feature/outcome/eligibility
   definitions, original folds, support recipe, model backend/specification, reuse
   purpose and validation checks. Document imports as reused development evidence.
4. Leave provenance/checksum safeguards intact. Existing fit/checkpoint guards
   intentionally reject incompatible automatic resume. Do not falsify hashes or
   loosen guards to get cached results accepted by different scientific code.
5. Keep the data adapter unchanged when scientifically appropriate. Canonical keys
   include the **whole adapter source hash**, not only its declared version; even an
   unrelated edit to `data.py` invalidates automatic reuse. Notebook-only changes do
   not inherently invalidate this independent data cache.
6. Fit and validate a final all-Euro portable domain, freeze the selected nuisance
   specifications/backend, and create a fresh valid design lock before WC access.
   Do not copy a lock from another scientific source or substitute fold-specific domains.
7. Retain every finite old draw, including extremes and support failures, under its
   original target/procedure. Further counts must be read from a fresh checkpoint.

Caching saves repeated acquisition, preparation and completed development fitting.
The much larger prospective time saving comes from removing unnecessary future
nested searches/refits, not from pretending old draws estimate the revised target.
No revised runtime guarantee follows from this audit.

## Scope of verification

Read-only checks included SSH/tmux/process status, manifest/raw/canonical/fit and
checkpoint integrity, source/helper/environment identity, prediction consistency,
recorded fold membership, and independent ATT/cluster-score recomputation. No
training, downloads, null tests, simulation, intervention or full re-audit of causal
identification was performed. Hashes establish integrity/lineage, not scientific validity.
Only new documentation files were added locally and mirrored to Vulcano's `docs/`.
