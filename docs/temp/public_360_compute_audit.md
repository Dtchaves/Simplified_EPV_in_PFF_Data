# Public 360 experiment: computational audit and shorter-study proposal

Audit date: 2026-09-30. Read-only inspection of the notebook, adapter/runtime,
approved design, legacy notebook/helpers, saved benchmarks, and live Vulcano
checkpoints. No running job, scientific configuration, environment, or GPU
allocation was changed. This document proposes revisions; it does not enact them.

Follow-up clarification: [sloan_validation_scope.md](sloan_validation_scope.md)
reviews full Sloan winner/finalist papers and explains that the 200 simulation
datasets are artificial, not additional competitions. It revises the blanket
1,000-draw recommendation below: a provisional 500-draw empirical target with
predeclared Monte Carlo precision checks is a possible compute/precision compromise,
not a guaranteed sufficient count. The approved plan and running gates are unchanged.

Scientific implementation: `results/models/public_statsbomb_causal_study.ipynb`,
tagged notebook-owned core, source hash
`2fa31ae58d4983d6490fb3f749455836e37db3548c700ce28825ac1a2e7f9f95`.
Remote root: `/home/joaomarcostomaz/causal/Simplified_EPV_in_PFF_Data`.
Existing run: `public-study-20260930`, tmux `public-causal-study`, four CPU workers,
one numerical thread each. Keep the other user's GPU job untouched.

## Conclusion

The original scientific recommendations do not require 1.6 million complete
pipeline bootstrap refits. That number comes from the particular nested simulation
design, not from public 360 data or ATT estimation. The approved plan did specify
six coverage scenarios × 200 datasets × 1,000 bootstrap draws; implementation also
added two DVDS stress scenarios with the same nested allocation. Revising this
allocation is a documented design revision, not merely a runtime flag change.

The main bottlenecks are repeated nuisance-model searches inside resampling,
unnecessary treated-outcome fits for ATT tasks, sequential influence analyses,
and the multiplication of simulation repetitions by bootstrap repetitions.
The dataset itself is modest. A larger GPU-memory allocation would not resolve
this workload design problem.

## Evidence and timing interpretation

Development comprises 102 Euro matches and 37,043 eligible decisions:
17,978/1,123 shots in Euro 2020 and 19,065/1,194 shots in Euro 2024.
The 64-match World Cup confirmation cohort remains unopened.

At **2026-09-30 12:28 UTC / 09:28 BRT**, the bootstrap checkpoint contained
40 attempts, 40 successful primary draws, 40 successful companion-statistic draws,
and zero computational failures. Mean draw wall time was **383.87 seconds**,
range **354.97–406.85 seconds**. Draw time includes the complete nuisance/support
refit and sensitivity/subgroup companion statistics. It is not an isolated timing
of AIPW arithmetic or an isolated nuisance-training time.

At four-way concurrency, 1,000 such draws require approximately **26.66 hours**
of useful work at this throughput, before overhead, further contention, failures,
or subsequent phases. This is a projection from measured empirical draws.
Synthetic datasets contain 12,000 decisions in 48 matches; their production
refit time has **not** been measured and must not be assigned the empirical
383.87-second timing.

Measured completed stages: acquisition 233.13 s; canonicalization 218.00 s;
initial nuisance/support fit 421.44 s; initial exports 14.58 s.
The point-fit log has 525 model-fit events; it does not log every final estimator
or calibration fit, so this is not the exact estimator-fit count.

## Entire execution path

### Development

| Stage | Purpose and present workload | Cost/evidence | Shortening opportunity |
|---|---|---|---|
| Tests/smoke | Leakage, timing, ATT accounting, checkpoint/runner contracts; six-match engineering smoke | 52 implementation tests passed; smoke is not scientific validation | Retain; cheap compared with production |
| Acquisition | Pinned public event/360 files and manifests, development cohort only | 3.89 min measured | Reuse hash-verified downloads; do not redownload |
| Canonicalization | Align events/360 snapshots; eligibility/visibility; pre-action geometry/history; net goal labels at fixed horizons | 3.63 min measured | Cache compatible canonical rows, keyed by source and adapter |
| Initial nuisance/support fit | Five outer match folds, three inner folds; propensity/calibration, both action outcomes, portable support propensity and donor/balance selection | 7.02 min measured; roughly 545–565 estimator fits | Smaller model menu; fold-local transform reuse; preserve match exclusion |
| Exports/comparators | ATT row contributions, raw/OR/IPW/AIPW on a shared domain, spatial donors, balance/calibration, subgroup/case exports | 14.58 s measured | Reuse saved predictions; little reason to cut |
| Primary uncertainty | 1,000 successful complete match-cluster resamples; support and model selection repeated | About 26.66 h projected from live throughput | ATT-only refits; bounded candidate menu; better scheduling; retain final uncertainty target unless deliberately revising it |
| Sensitivity/subgroup uncertainty | Seven Gamma values and six subgroups are computed inside the same primary draws | Included in 383.87 s/draw; no separate 1,000-pipeline loop | Subgroups are cheap; benchmark DVDS separately; endpoint draw targets can be distinct if transparently revised |
| Point sensitivity | DVDS bounds at Gamma 1, 1.25, 1.5, 2, 3, 5, 10 plus tipping search | Untimed; much smaller than 1,000 refits | Preserve quantitative confounding analysis; cache compatible stages within one fit |
| Robustness | 13 full pipeline fits, five e/m0 partial refits, three m0-only refits; plus prediction-only comparisons | Rough planning range 1.5–3 h, not measured | Reuse propensity/support for outcome-only changes; avoid inner selection for a single prescribed model |
| Match influence | An unnecessary original fit plus 102 leave-one-match-out fits, presently sequential | 103 × live draw timing ≈ 11 h proxy, not measured; draw timing includes work LOMO does not need | Reuse point fit; parallelize safe bounded CPU work; preserve complete refit analysis |
| Team influence | Remove each of 29 distinct team IDs, presently sequential | 29 × live draw timing ≈ 3.1 h proxy; subset sizes and omitted DVDS can make this cheaper | Parallelize/cache; screening only is a weaker alternative |
| Candidate frozen design | Fit portable support/domain and model settings on development for locked confirmation | One training/selection job, not five outer jobs; untimed, likely minutes | Cache by exact source/data/config |
| Software nulls | Three null kinds × two support branches × 200 repetitions = 1,200 pipeline calls | Rebuilt-domain 600 calls alone ≈ 16 h at empirical full-draw/four-worker proxy, plus cheaper frozen calls; unmeasured null workload | Stage pilot counts; reuse treatment-only nuisance work where appropriate; preserve rebuilding when treatment changes |
| Simulations | 5,200 restricted-class point fits, 1,600 learned-procedure point fits, and 1.6 million nested bootstrap refits | Unmeasured; overwhelmingly larger than empirical work | Redesign allocation around precision and wall-clock budget; separate required and extended profiles |
| Gates/design lock | Timing/test/support/null/simulation completion and operating-characteristic gates; immutable freeze | Negligible computation | Revise target counts explicitly; never bypass failures or label a pilot as complete validation |

This sequence is implemented by `execute_public_study`. `phase='full'` executes
development first and only opens confirmation if every development gate passes.
Consequently, the huge simulation study currently blocks the entire confirmation
stage. It is not a detached optional background task.

Robustness details in `run_robustness`:

- Four outcome-label variants: 5/10/30-second rewards and shot-start timing fallback.
  All call the full pipeline even though X, treatment, folds, and treatment-only
  support construction have not changed. Refit the required outcome model(s),
  reuse exact compatible e/support, and retain the support-ID equality assertion.
- Three prescribed outcome alternatives (Ridge, GAM, HGB), fixed primary domain.
  Each currently performs 20 m0 fits because one-candidate inner validation is
  still executed. Five final outer fits suffice for predictions when no selection
  remains; diagnostics can be evaluated separately on held-out predictions.
- Five feature ablations, each evaluated on the fixed primary population and again
  with reselected support. The latter are different estimands and are development
  only. Their portable-support feature list currently remains unchanged.
- Four eligibility/visibility subsets: exclude carries, require full 15 seconds,
  and require visible 5 m/10 m disks. These change populations, not merely features.
- Common-overlap and football-caliper comparisons reuse existing predictions;
  they are not expensive full training loops.

### Confirmation

Only after a valid revised development lock:

1. Acquire/build the untouched 64-match World Cup cohort. These stages are untimed.
2. Refit match-grouped full-adjustment nuisances using frozen development model
   choices; portable support model/domain remain frozen. This avoids the full
   development candidate search but still contains redundant one-candidate inner
   outcome validation. Initial fit time is unknown.
3. Run 1,000 match-bootstrap draws conditional on that development domain/design,
   with shared sensitivity/subgroup statistics. It should be cheaper than the
   development search, but no measured throughput or linear match-count scaling
   justifies a precise ETA.
4. Point sensitivity and robustness: nine full frozen-model fits, five e/m0
   partial refits, three m0-only fits; no five reselected-support ablations and
   no development-only overlap trimming.
5. 64 LOMO removals plus redundant point refit; team removals for the cohort's
   participating IDs. Team count/cost has not been inspected in the holdout.
6. 200 joint development/confirmation resamples: fit a development frozen design
   then fit confirmation per draw. This probes domain-learning uncertainty and
   is separate from the conditional confirmation interval. Untimed; prioritize
   or pilot independently rather than quietly equating the two intervals.
7. Exports and final notebook/manuscript tables. No repeat of the complete
   development null grid or production simulation grid in this branch.

## Why a single refit is expensive

For each of five outer folds, `fit_propensity` evaluates three logistic candidates
and one HGB candidate over three inner folds. HGB probability calibration requires
additional match-grouped nested fits. `fit_outcome` evaluates 13 candidates per
action arm: three Ridge, six GAM, four HGB. Portable support propensity adds another
fixed logistic with grouped inner predictions.

Approximate estimator-fit accounting for one complete development fit:

- Full propensity: 110–125 base fits, depending on the selected final model.
- Portable support propensity: 20 fits.
- Continuation outcome m0: 200 fits.
- Shot outcome m1: 200 fits.
- Sigmoid calibrators: 15–20 fits.
- Total: 545–565 estimator fits, before companion DVDS work. All point-fit outer
  folds selected HGB propensity and Ridge(alpha=10) outcomes.

DVDS may additionally fit five ternary-probability models and up to 60 tail models
per seven-Gamma empirical draw, with caching when quantile patterns coincide.
Six subgroup ATTs are arithmetic on the same predictions and are inexpensive.
Do not multiply subgroup counts by the full pipeline cost.

The ATT score uses e, m0, treatment and observed Y:

`[sum_T(Y-m0) - sum_C e/(1-e)*(Y-m0)] / n_treated`.

m1 does not enter this ATT, its subgroup ATTs, or the implemented ATT DVDS bounds.
An ATT-only repeated-fit mode can omit 200 m1 fits (~35% of fit count), keeping m1
for one-time outcome diagnostics or any explicitly requested ATC analysis.
This is not automatically a 35% wall-time saving: treated-arm training is smaller.
The current finite-prediction assertion requires m1; revise the mode/contracts
properly rather than filling dummy predictions.

## Exact simulation multiplication

`simulation_scenarios` has 24 sign/overlap/misspecification combinations and two
stress scenarios. Each synthetic dataset has 48 matches × 250 decisions.

| Component | Allocation | Point fits | Bootstrap pipeline refits |
|---|---|---:|---:|
| Base bias/RMSE/retention diagnostics | 26 scenarios × 200 datasets, restricted learner classes | 5,200 | 0 |
| Actual learned-procedure CI coverage | 6 sign/overlap scenarios × 200 datasets × 1,000 draws | 1,200 | 1,200,000 |
| DVDS stress endpoint/CI coverage | 2 stress scenarios × 200 datasets × 1,000 draws | 400 | 400,000 |
| Total | | 6,800 | 1,600,000 |

Thus **1,606,800 calls to the pipeline**, before any failed attempts. A pipeline
call itself includes many estimator fits. The base diagnostic branch restricts
the nuisance class menu and is cheaper (~80 fits per point dataset), but it does
not validate the actual production bootstrap CI.

Outer simulation datasets execute four at a time; each dataset's nested bootstrap
uses one worker, avoiding nested oversubscription. This limits useful aggregate
concurrency to four expensive pipeline calls at a time.

Arithmetic scenarios only, NOT measured forecasts: 1.6 million refits at four-way
concurrency take 4.6 days if each costs one second, 46 days at ten seconds, and
139 days at thirty seconds. These explain the concern, but an empirical refit
timing cannot be transferred to the smaller synthetic datasets as a forecast.

`run_production_simulations` hard-requires datasets >= 200 AND bootstrap draws
>= 1,000. Null gates hard-require >= 200 per branch. Empirical endpoint interval
gates require as many successful companion statistics as primary bootstrap draws.
Smaller CLI counts alone therefore cannot produce a passing design lock.

## Comparison with the earlier PFF workflow

Remote reference: `results/models/support_aware_causal_pipeline_demo-v6-workflow copy.ipynb`.
It configures `BOOTSTRAP_REPS=50`, uses mainly fixed logistic/Ridge estimators
(optional random forest), and permits cached exports. Compared with the new
1,000-draw study, draw count alone is 20 times larger; repeating a large nested
candidate search makes each new draw much more expensive as well. The reference
does not contain an equivalent 1.6-million-refit validation grid.

Its saved notebook and inspected causal/reward helpers contain no explicit GPU
backend. This does not disprove remembered GPU use in another stage/session.
Useful things to retain are its simple model menu, selective execution, and saved
predictions/results—not its verified cross-fitting problems. Opposite-arm model
averaging could include the observation's match, and bootstrap duplication could
break original-match exclusion. Those shortcuts must not return for speed.

## Prioritized optimizations

### A. Preserve the estimator and inferential design

1. ATT-only repeated fitting: omit m1 where mathematically unused; retain appropriate
   one-time diagnostics and separate ATC contracts.
2. Reuse unchanged propensity/support for reward-horizon/timing sensitivities.
   Match identical rows, treatment, fold registry, feature sets, seeds, and weights.
3. Skip inner model comparison when a model specification is prescribed/frozen;
   keep necessary grouped calibration/support OOF stages.
4. Cache fold-local transformed matrices within a fit. Keys must include training
   indices, bootstrap multiplicities, feature definitions and preprocessing settings.
   Never cache a transform fitted on the entire evaluation population. Avoid an
   unbounded disk cache across millions of resamples.
5. Reuse the original point result for influence; parallelize match/team removals
   within a safe CPU budget that respects other users' CPU/data-loader needs.
6. Persistent bounded work queues rather than four-task batch barriers; preserve
   all completed draws and do not discard slow or finite extreme draws.
7. Add per-variant robustness checkpoints. Bootstrap presently rewrites/fsyncs the
   entire growing JSON history each batch (each draw inside simulations), causing
   quadratic cumulative serialization. Use checksummed append/shards plus periodic
   snapshots, deterministic attempt IDs, and durable failure records.
8. Amortize recursive disk checks without removing the disk-budget safeguard.
9. Log per-component seconds (preprocessing/e/m0/support/DVDS/I/O), throughput,
   stage/total ETA, failures and cache hits. Separate per-attempt status from phase
   status. Resume ETA should use new successes per new elapsed time, not cached
   successes divided by new elapsed time.

### B. Reduce model-selection cost with an explicit scientific revision

Keep three outcome families, but reduce to one justified Ridge, one GAM and one
HGB configuration; use one logistic and one calibrated HGB propensity candidate.
With existing fold/calibration counts this is approximately 200–220 fits with
both arms, or 150–170 in an ATT-only task, versus 545–565 now. These are counts,
not measured speedups. Selection remains outside each evaluated match.

A more aggressive option freezes model choices before repeated resampling, while
refitting parameters with strict original-match cross-fitting. This estimates a
fixed-model procedure and does not automatically include selection uncertainty.
Global tuning on all development matches cannot be called entirely out-of-match
prediction merely because final coefficients are cross-fitted. Use independently
declared settings or honest separate tuning data/design. Full-reselection draws
can be a separately reported robustness analysis, not silently mixed into one CI.

### C. Stage validation rather than removing it

- Keep 1,000 successful match-bootstrap draws for final empirical primary ATT
  unless the revised inference method has its own defensible validation. A 200-draw
  pilot provides feedback, not the advertised final interval precision.
- Share subgroup statistics with those draws; do not cut inexpensive arithmetic.
- Keep point confounding bounds. Separate the endpoint-bootstrap allocation from
  primary ATT uncertainty only with explicit precision/completion statements.
- Null-check pilot: 50 per branch, fresh nuisance fitting/rebuilt support where
  treatment changes; then 200 for the defined final branches. Dummy outcomes can
  reuse compatible treatment-only models/support but require new outcome fits.
  A fixed-choice null and a fully reselected null test different algorithms.
- Keep pre-action timing, calibration among weighted donors, balance/donor quality,
  same-population estimators, complete match influence, and untouched confirmation.
- Do not substitute software nulls for exchangeability evidence or score intervals
  for adaptive-support production bootstrap validation.

### D. Make simulations an explicit bounded study

Recommended initial allocation:

1. Broad point-estimation diagnostics across all 26 scenarios, 100–200 datasets
   each, no nested bootstrap. Retain signs, overlap and both misspecifications,
   bias/RMSE, support retention and supported-treated ground truth.
2. Actual-CI engineering pilot: six cases × 20 datasets × 99 draws, plus two DVDS
   cases × 20 × 99 = **15,840 bootstrap refits**, about 101× fewer than now.
   This is exploratory validation, not precise publication coverage evidence.
3. Time a small CPU pilot of production-like simulations before choosing final
   allocations. No such new pilot is authorized/launched by this read-only audit.
4. Select final coverage scenarios/replications using an explicit wall-clock budget
   and Monte Carlo precision. An example larger allocation, six × 100 × 199 plus
   two DVDS × 50 × 199, costs **139,300** nested refits (~11.5× fewer), but can
   still be lengthy. It is not a promise of an hours-long study.
5. Keep extended validation separately executable and resumable; require a defined
   bounded core validation before the lock. Do not call optional/incomplete work
   completed, or automatically hold all empirical progress hostage to a giant grid.

At true 95% coverage, the Monte Carlo standard error is
`sqrt(.95 * .05 / R)`: 4.87 percentage points for R=20, 3.08 for 50, 2.18 for 100,
and 1.54 for 200. The bootstrap's 2.5% tail has only about 2.5 draws at B=99,
5 at B=199, 12.5 at B=499, and 25 at B=1,000. Small allocations make both
coverage estimates and interval endpoints less precise. Choose these two budgets
separately; a rounded 1,000 for every synthetic dataset is not a universal rule.
The relevant methodological guidance is to plan repetitions around Monte Carlo
precision and report that uncertainty, not claim weak pilots establish nominal
coverage: [Morris, White & Crowther (2019)](https://discovery.ucl.ac.uk/10066118/1/2019%20-%20Morris%20-%20simulation%20studies%20tutorial%20-%20stat%20med.pdf).

Score/multiplier bootstrap on saved match-level scores can be fast, but an adaptive
hard support rule plus model selection makes it non-interchangeable with the
current full-refit procedure. It requires an appropriately defined target,
regularity assumptions and validation; it is not an automatic optimization.

## GPU and realistic wall-clock goals

Prior benchmarks (already completed before the protected job): a full candidate
pilot took sklearn CPU 71.94 s, XGBoost CPU 54.78 s, CUDA 53.09 s. Switching the
library was most of that improvement; isolated CPU-to-CUDA savings were ~3%.
Four concurrent comparable XGBoost propensity workloads took CPU 42.78 s versus
CUDA 36.29 s (~15% lower wall time). These are small workload pilots, not complete
bootstrap/simulation benchmarks. They do not support an orders-of-magnitude GPU
rescue, and XGBoost changes predictions/selection rather than being a transparent
hardware-only replacement for sklearn HGB. See `public_360_gpu_benchmark.md`.

Current empirical development checks alone can occupy several CPU days under
this implementation: ~26.7 h primary bootstrap, sequential influence, substantial
nulls and robustness, before simulations. Non-bootstrap stage costs here are
proxies rather than timings. Publication-scale simulation time is unknown.

An hours-oriented experiment needs fewer fits per replicate and staged validation,
not just more VRAM. Illustration only: at 20–60 seconds per optimized empirical
draw, 1,000 draws/four workers take 1.4–4.2 hours, plus other stages. Such throughput
has not been demonstrated. Benchmark the revised CPU procedure before promising
completion dates or launching its larger grids.

Any accepted revision belongs mainly in the new notebook, with helpers limited
to data/runtime infrastructure as requested. Save a revised plan/profile and tests,
use a new source/config/run ID, reuse only compatible pinned input artifacts,
and never mix draws from different estimators. This audit leaves the existing CPU
run and the other user's GPU job unchanged.
