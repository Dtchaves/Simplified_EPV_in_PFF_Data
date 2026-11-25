# Proposed final public-study submission profile

Status: **proposal for approval, not implemented**. Written 2026-09-30 after
comparing the user's independent-agent assessment with the actual notebook,
approved plan, code audit, compute audit and GPU benchmarks. Supersedes earlier
compute-allocation recommendations if approved; does not silently amend the
approved plan or running source/configuration.

Scientific work stays primarily in `results/models/public_statsbomb_causal_study.ipynb`.
Helpers remain data/runtime infrastructure. Vulcano data stays in
`/home/joaomarcostomaz/causal/Simplified_EPV_in_PFF_Data/data/public_statsbomb_causal/`.
No existing experiment was stopped, modified or restarted for this review.

## Main decision

Use Euro 2020 + Euro 2024 as development/design, and the untouched World Cup 2022
cohort as the **primary confirmation supported ATT**. Euro ATT is development
evidence, not a second confirmatory headline requiring a full adaptive bootstrap.
This changes the emphasis and uncertainty budget, not the football question.
World Cup is held-out tournament confirmation, not future-period validation:
2022 predates Euro 2024. The historical continuation mixture may differ between
competitions, so the two population-specific ATTs need not equal one another.

Keep L1 eligibility, the 3 m fully visible local disk, pre-action/event-start proxy
features, shared covariate support domain, original-match cross-fitting and signed
first-goal reward (+1/-1/0) within 15 seconds/natural period termination. Preserve
all eligibility, visibility, alignment, attribution, own-goal and period-end audits.
Do not substitute full-roster visibility, invented tracking velocity or individual
player identities. Keep natural historical Pass/Carry/Dribble continuation and
describe its cohort-specific composition.

## What is accepted and what needs correcting in the independent assessment

| Suggestion | Decision |
|---|---|
| Confirmation as headline; fixed Euro settings | Accept. Frozen domain/settings are separate from fitted WC nuisance parameters. |
| HGB/Ridge(alpha=10) as final choices | Strong development evidence, not yet proof of final full-Euro selection. Verify the all-Euro freeze and weighted validation before lock; do not simply vote across outer folds. |
| Remove repeated m1 and search | Accept for ATT-only resampling. Keep one-time m1 diagnostics; preserve grouped calibration and preprocessing. |
| Implement denominator-aware cluster scores | Already implemented in `att_estimate` and `cluster_score_interval`. Consolidate/test/promote only with an explicit inference justification; existing outputs are labelled diagnostic. |
| Analytical CI automatically defensible once frozen | Too strong. Conditional support removes evaluation-sample support selection, not nuisance error, rare-event instability or dependence across matches/teams. A t critical value is a finite-cluster approximation, not a guarantee. |
| Analytical/refit CIs agree, therefore strong validation | Agreement is useful, not proof: both can share bias. Require diagnostics, direct nulls, known-truth checks and transparent assumptions. |
| Twenty datasets, no coverage assessment | No nested bootstrap grid. But evaluate the actual analytical interval's empirical coverage/FPR cheaply across independent simulations; report Monte Carlo uncertainty. Twenty datasets is a pilot, not precise validation. |
| Old DoWhy problem is fully known | Incorrect. A synthetic stale-score mechanism was reproduced; its complete contribution to the historical football failure is unresolved. Keep regression/private documentation, omit from public empirical compute. |
| Move all large validation before confirmation out of the gate | Accept separation of design lock from claim-completion gate; deterministic tests and small sanity checks still precede opening holdout. |
| LOMO only most influential | Accept as a clearly labelled targeted refit screen, not complete leave-one-match-out. Cheap all-match score influence remains. |
| Point DVDS robust tipping sentence | Keep with caveats: estimated identification-set tipping, not a confidence/significance threshold or proof of a specific confounder's strength. |
| Fourteen-hour finish | Planning budget only; unmeasured revised throughput and unknown live progress cannot support this guarantee. |
| Free GPU | Available according to user. Integrate/test a CUDA candidate before lock; changing HGB to XGBoost is a model change, not transparent acceleration. |

## Primary fitting and inference

### Development and final model freeze

Reuse compatible raw/canonical data and existing complete development diagnostics,
with exact source/config provenance. Keep the current HGB propensity and Ridge
outcome as baseline. The final all-Euro freeze must independently pass support
diagnostics and save its actual chosen model specifications.

Perform a **bounded** Euro-only comparison, not a new large tuning grid:

- Propensity: existing calibrated HGB, one logistic configuration, one small CUDA
  XGBoost configuration. Compare grouped proper loss, calibration, weighted support
  diagnostics and complete ATT-only throughput; never select using ATT sign/magnitude.
- m0: Ridge(alpha=10), one GAM configuration, one boosted alternative (CUDA XGBoost
  if tested). Evaluate weighted residual/calibration behavior among treated-like and
  influential continuation donors, not only overall MSE.
- Use honest match-held-out predictions for development comparisons. Freeze the
  selected specifications and backend before any WC access. Changes selected using
  all-Euro data are development choices; do not falsely claim globally chosen settings
  were trained/tuned entirely outside every Euro evaluation match.
- m1 once for diagnostic purposes; no ATC workflow in this profile.

For fixed-specification confirmation/refits, remove inner candidate-validation loops
whose only purpose was selection. Fit the outer-fold outcome model directly. For a
calibrated boosted propensity, retain three grouped calibration-training fits plus
the final outer-training model and calibrator. These all exclude the outer evaluation
matches. No extra inner propensity OOF predictions are needed when there is no
outcome-model selection or newly learned evaluation support depending on them.

For the current fixed HGB/Ridge choices, the resulting ATT-only five-fold fit can
need approximately **30 estimator fits**, versus ~545–565 in a complete development
search. This is a code-path count, not a measured runtime/speedup. The portable
Euro support model is reused, not refitted in WC resamples.

### Supported target and cluster uncertainty

Let S_i be membership in the frozen Euro X-domain and

`a_i = S_i * [T_i*(Y_i-m0_i) - (1-T_i)*e_i/(1-e_i)*(Y_i-m0_i)]`.

Then `theta = sum(a_i)/N1`, `N1 = sum(S_i*T_i)` and the match numerator score is

`U_g = sum_in_match_g[a_i - theta*S_i*T_i]`.

The existing implementation includes that denominator contribution. With G original
independent sampling clusters, the small-sample-adjusted sandwich is

`SE = sqrt((G/(G-1))*sum_g(U_g**2))/N1`.

Retain a full eligible-match registry, including zero-score clusters, and explicitly
report eligible/support-bearing/treated-bearing matches. Use a t critical value with
the documented cluster degrees of freedom instead of 1.96 as a practical approximation.
Point estimates remain decision-weighted ATT, not a match-average target.

Proposed primary interval: **approximate conditional match-cluster score CI** for
the locked supported WC ATT, after derivation/tests/validation. Cite the relevant
[ATT orthogonal-score work](https://www.aeaweb.org/articles?id=10.1257/aer.p20171038)
and [clustered DML theory](https://arxiv.org/abs/1909.03489), without claiming these
automatically prove validity for every fitted model. Assumptions include adequate
overlap, appropriately small nuisance remainder and independent clusters conditional
on the model. Repeated national teams across matches make the last assumption a
limitation; neither a t adjustment nor team-removal screening fixes all dependence.

Refit-bootstrap cross-check: **200 successful fixed-design, fixed-spec ATT draws**
initially, with an optional predeclared extension to 500 if Monte Carlo endpoint
precision is inadequate. No routine 1,000 target, no Euro adaptive bootstrap gate.
Keep original-match grouping for all duplicated match copies; refit parameters,
preprocessing and calibration; keep Euro support/model choices frozen. Retain finite
extreme or diagnostic-poor draws; disclose structural/computational failures.
Compare score/refit uncertainty, distribution shape and sensitivity to nuisance choice.
Do not pick the interval method yielding significance. If materially contradictory,
report provisional/inconclusive uncertainty and investigate before a strong causal claim.
No new endpoint DVDS fitting inside primary bootstrap draws.

Monte Carlo diagnostics can resample saved ATT numbers cheaply. This is distinct
from generating new football datasets or refitting thousands of model pipelines.
If the 200 draws are unfinished, report their status; the analytical interval can be
shown as approximate/provisional, not a substitute declaration that the planned
validation has completed.

## Bounded checks, practical findings and deferred work

| Component | Submission-profile allocation |
|---|---|
| Real cohorts | Euro design; 64-match WC cohort accessed only after lock |
| Development uncertainty | Existing compatible full-procedure draws as descriptive sensitivity; no new 1,000-draw requirement |
| Confirmation uncertainty | Score CI + 200 fixed-spec refits, predeclared precision-driven extension to 500 |
| Comparators | Raw, OR, IPW, actual spatial matching, AIPW, point estimates on identical support/target IDs |
| Outcome-model robustness | Fixed-population Ridge/GAM/boosting checks, point estimates and donor-weighted validation |
| Outcome horizons/timing | 5, 10, 30 s plus goal-occurrence timing fallback where relevant; reuse e/support/folds, refit m0 per new Y |
| Measurement sensitivities | Carry exclusion and visible 5 m disk; population changes explicitly labelled |
| Feature ablations | Two targeted fixed-population groups: local pressure and recent history; no exhaustive reselected-support branches |
| Influence | All-match and team score screens; actual nuisance refits for a prespecified most-influential set (e.g. five WC matches plus two high-ranked team removals if budget permits). Label as targeted, not complete LOMO |
| Nulls | 50 within-match randomized-treatment runs + 50 independent signed dummy outcomes, fixed locked domain/specs; deterministic rebuild-support tests and a small development-only rebuilt-domain smoke |
| Simulations | Eight targeted scenarios × 50 independent confirmation datasets/design pairs; no nested full-model bootstrap |
| Hidden confounding | Seven existing point Gamma values, tipping grid/search, omitted-covariate benchmarking and oracle/identity tests; no full endpoint-refit bootstrap |
| Tactical results | Six existing prespecified distance/angle/pressure strata with counts, matches, goals, ESS, balance and support qualification |
| Cases | A few supported/unsupported states and actual donors; analyst-rating template only, no unperformed assessment implied |
| Deferred | ATC, 1.6-million simulation grid, 200 joint-domain refits, exhaustive team/LOMO refits, all reselected support ablations, matching inference, extensive DVDS endpoint coverage |

Nulls test software/design behavior, not exchangeability. For randomized treatment,
fit fresh propensities/outcomes and check known-assignment scores where positivity
allows; within-match assignment probabilities may vary by match. Do not diagnose
nonzero null scores as software bugs without separating nuisance approximation.
For independent dummy Y, reuse compatible treatment-only models/support but fit
fresh m0. Fixed primary domain is the relevant conditional estimator; rebuilt-domain
tests are a separate design-code exercise, not six obligatory 200-run branches.

Simulation scenarios: healthy-overlap zero/positive/negative effects; zero effect
with weak overlap; propensity misspecification only; outcome misspecification only;
both misspecified; action-dependent visibility. Each uses an independent synthetic
design/confirmation split, frozen design/configurations and the **actual proposed
production fitting/CI functions**. Ground truth is the selected supported treated
population's effect. Report bias, RMSE, retention, ESS, concentration, failures and
empirical score-CI coverage/FPR with Monte Carlo uncertainty. At R=50, coverage
near .95 has MCSE ~3.1 percentage points: useful stress evidence, not precise proof
of nominal coverage. Both-misspecified/visibility failures are expected descriptive
risks, not gates that force unbiased recovery. R=20 is engineering pilot depth only.
Gamma=1/ternary-atom/quantile-boundary/support counterexamples remain deterministic
tests rather than expensive additional simulation branches.

Subgroup inference reuses saved match scores. Use inexpensive joint match-multiplier
max-statistic intervals or another predeclared multiplicity adjustment for the six
strata; these numerical draws do **not** refit models. Save heterogeneity contrasts
and do not infer a difference from 'significant here, not significant there'. Apply
existing count/ESS/match/balance/concentration thresholds; inadequate groups stay
insufficient. This does not add a new individualized CATE architecture.

DVDS point bounds/tipping must be distinguished from sampling uncertainty. If the
primary CI includes zero, there is no positive-effect inferential robustness claim
even if the point bounds exclude zero for some Gamma >1. Point tipping means the
estimated identification region permits zero under the specified marginal-odds
departure, not that a particular hidden factor must have exactly that strength.
Report monotonicity, endpoint crossings and ternary-quantile boundary diagnostics.
The [DVDS paper](https://jacobdorn.info/files/DVDS.pdf) provides ATT identification
and inference results under its assumptions; it is not automatic clustered ternary
coverage certification for our implementation.

Report net signed first-goal reward per 100 supported decisions. Current exports
give observed positive/negative fractions, not separate causal scoring/conceding
effects. Do not relabel them as causal decomposition. If a causal decomposition is
needed, fit the two outcome-component models with the same e/support and verify
the signed-score identity; otherwise omit that additional analysis.

## Gate changes and protection of confirmation

Create distinct **design-lock** and **claim-completion** statuses.

Before WC: source-bound deterministic tests, ordinary smoke, timing/visibility audit,
proper Euro cross-fitting/weighted outcome validation, passing final frozen-domain
diagnostics, minimal null/known-truth sanity checks, and an immutable manifest of
all model/backend/support/horizon/subgroup/inference/simulation/budget choices.
No hard-coded huge bootstrap/simulation counts in this gate.

After lock, bounded null/simulation validation can run alongside confirmation
within the **same global resource budget**. They can invalidate/qualify claims,
but cannot authorize effect-driven changes to locked WC choices. The primary WC
support gates must pass unchanged. Unsupported confirmation remains unsupported;
do not retreat to Euro as an interchangeable confirmatory success.

Before calling the submission profile complete: all declared required checks/draws
executed, failure accounting and Monte Carlo precision reviewed, compatible artifacts
exported, confirmation diagnostics and inferential assumptions reviewed. Incomplete
checks stay incomplete. Post-WC software/design revisions are disclosed, and a rerun
is not newly untouched confirmation. No automatic public submission/release.

## GPU, runtime and implementation order

The user reports the 4090 is free. SSH recheck in this review failed because the old
authenticated connection expired; current run progress/occupancy is unverified.
Reconnect interactively before execution; never obtain passwords through logs/chat.

Use existing GPU benchmarks as pilots: ~3% serial CUDA saving versus XGBoost CPU,
~15% lower elapsed time/~18% higher throughput at four concurrent representative
propensity jobs. No complete fixed-ATT/simulation speedup has been measured.

Make CUDA XGBoost an explicit notebook model candidate/backend, not the benchmark's
process-local monkeypatch as production science. Retain fold-local preprocessing,
grouped calibration, fixed tree limits and disabled implicit row early stopping.
Verify actual booster device, package versions and warnings. CPU Ridge/logistic/GAM
and support bookkeeping remain CPU. Use two-to-four independent GPU tasks only if
complete-workflow pilot shows benefit; never enlarge models to fill VRAM. Choose
the backend on Euro metrics/diagnostics/runtime before lock. If baseline HGB is
retained, the GPU can still run nonlinear alternatives; GPU activity is not a goal.

Implementation order:

1. Add explicit submission/extended profiles and ATT-only fixed-spec paths, keeping
   scientific logic in the notebook; unify/test the cluster score implementation.
2. Fold-safe caching, outcome-only reuse, bounded persistent workers, per-task resume,
   efficient durable checkpoints and timings/device telemetry. Preserve disk guards.
3. Tests/smoke and small complete fixed-refit/GPU/null/simulation timing pilots.
4. Review Euro diagnostics and small model comparison, freeze/lock final choices.
5. Confirmation point/score CI and exports early; run bounded refits/checks under a
   global worker budget, then produce qualified tactical results and case packets.
6. Consolidate final uncertainty/validation status, saved notebook, tables/figures,
   configurations, environment and authorized-data release manifest.

Record source/config/data hashes, stage/fold/replicate/attempt seeds, success/failure,
elapsed component timings, worker count, actual devices, progress/ETA and completion
status. Use tmux; no per-replicate model or large prediction dumps. Reuse only
hash-compatible inputs; never pool old adaptive sklearn draws with fixed/GPU draws.

A same-day/roughly-half-day empirical run is a planning objective, **not a promised
14-hour completion**. Budget only after 5–10 complete revised refits and representative
simulations are timed. Publish early point/approximate-CI artifacts as provisional;
reserve contingency time instead of scheduling required work through the last minute.
There is no verified user deadline in the independent agent's 'tomorrow' wording.

Old run handling requires approval of this proposal and a live status check: request
a graceful stop at a completed batch/checkpoint boundary if it is still running;
preserve all artifacts as an exploratory adaptive-development run. Do not restart
it under a new source/config or overwrite its history. Use a new run ID for this
profile. No job-stop authorization is inferred from the request to assess suggestions.
