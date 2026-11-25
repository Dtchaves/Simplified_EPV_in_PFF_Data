# Public 360 study — disclosures and limitations (for the paper)

Run of record: `public-submission-full-gpu-20260930` (core hash `0959a70c…`). All
numbers below come from that run's saved artifacts or from the saved-score
analyses in `runs/public-tier12-20261001/tier1/`. None of them required refitting
any model.

## 1. What is estimated

- **Estimand.** The supported ATT of choosing **Shot** instead of the historical
  continuation mix at live-play event starts in the attacking region (x ≥ 17.5 m),
  with the 3 m disk visible.
  - **Continuation mix.** The weighted mix is Carry 50.3%, Pass 46.1%, Dribble 3.6%.
  - **Outcome.** The signed first goal (+1/−1/0) within 15 s or before the
    natural half end. This is not EPV.
- **Supported population.**
  - Supported: 1,228 of 1,294 WC shots (94.9%) and 20,532 controls.
  - Trimmed: 66 shots, all close-range (median 5.2 m, mean angle 63°, goal rate
    0.30), removed for `insufficient_cross_match_cell_donors`.
  - The estimate therefore excludes the highest-value shots.
- **What drives the outcome.** It is almost entirely scoring.
  - Among all 21,760 supported rows (shots and controls), 611 rows led to a goal
    for the acting team (144 distinct goal events) and 5 to a goal against (4
    distinct events).
  - Among the 1,228 supported shots, 132 led to a goal for (124 distinct goal
    events) and 0 to a goal against.
- **Holdout.**
  - The World Cup 2022 holdout is a different competition, not a later period of
    the same one.
  - The WC nuisances (propensity e and outcome model m0) were refit on WC data
    under the frozen Euro design.
  - Nothing from the WC was used before the lock: lock 21:51:38.025Z, first WC
    download 21:51:38.620Z.

## 2. Inference

- **Primary interval.** Match-cluster score interval, t(63): ATT 0.0308,
  CI [0.0067, 0.0548]. It treats the fitted nuisances as fixed.
- **Dependence across matches.** National teams recur across matches (32 teams,
  64 matches), so clustering choice matters. Under team and opponent clustering
  (`tier1/dependence.json`):

  | clustering | SE | 95% CI | p (H0: ATT = 0) |
  |---|---|---|---|
  | match (primary) | 0.0120 | [0.0067, 0.0548] | 0.013 |
  | team (G = 32) | 0.0131 | [0.0041, 0.0574] | 0.025 |
  | opponent (G = 32) | 0.0165 | [−0.0028, 0.0643] | 0.071 |
  | two-way team × opponent (CGM) | 0.0162 | [−0.0022, 0.0637] | 0.066 |
  | wild cluster bootstrap, opponent (Webb, B = 9,999) | — | [−0.0020, 0.0634] | 0.068 |
  | wild cluster bootstrap, team | — | [0.0042, 0.0576] | 0.023 |

  Under opponent-level or two-way clustering, the interval includes zero.

- **Bootstrap.**
  - All 500/500 fixed-design refits succeeded. Percentile interval
    [0.0029, 0.0561]; SD/score-SE ratio 1.11.
  - 489 of the 500 draws fail at least one support/balance gate (median key
    SMD 0.125). They are retained, per the plan.
- **Development vs confirmation.** The Euro development estimate was 0.0177,
  CI [−0.0022, 0.0376], with 102 matches and 2,195 shots. The World Cup estimate
  is larger. The two intervals overlap.
- **Monte Carlo precision.** The bootstrap was extended from 200 to 500 draws by
  the pre-declared endpoint MCSE rule. The rule does not depend on significance.

## 3. Design choices made after looking at Euro results

- The rule "prefer CUDA XGBoost if within one SE" was added between 20:46Z and
  21:19Z.
  - That was after the pilot Euro ATTs had been seen, but before any WC access.
  - It switched the propensity model from HGB to XGBoost. Their grouped
    log-losses were 0.117572 and 0.117655 respectively (the selection SE came
    from 3 folds).
  - The Euro ATT was 0.0169 with HGB (pilot) and 0.0177 with XGBoost.

## 4. Measurement and timing limitations

- **Pre-action timing.** Covariates come from the 360 frame and event data at the
  event start. The timing gate (`development_timing_gate`) only checks that the
  outcome is non-missing and that there are at least 30 matches. It does not
  prove that the covariates are measured before the decision.
- **3 m caps.** Nearest and second-nearest defender distances are capped at 3 m:
  54.5% and 91.9% of eligible controls sit at the cap.
- **Visibility depends on the action.**
  - The opposing keeper is located in 94% of shots vs 59% of controls.
  - The 5 m visible-disk population gives 0.0226, CI [−0.0047, 0.050].
- **Remaining imbalance on unused 360 covariates (weighted SMD).**
  - Teammates ahead of the actor: +0.19 (2.47 vs 2.10).
  - Defenders within 5 m: +0.12.
  - Keeper x: −0.10.
- **Outcome model calibration.** On the WC, m0 under-predicts by 2–6 percentage
  points in the middle bins. Weighted validation passed on Euro (decile
  residuals +0.02 to +0.04, within the `.03 + 4SE` rule).

## 5. Software checks and their limits

- **Within-match treatment-permutation null.**
  - The fitted mean is +0.0022 (MCSE 0.0007). The known-assignment estimator
    gives +0.0003. The paired offset is +0.0019 (replicate SD 0.0002).
  - A plug-in approximation of the between-match component predicts +0.0021
    (observed +0.0019). The plug-in uses the original baseline m0 and support,
    not the refitted nuisances.
  - Mechanism: within-match permutation preserves each match's shot share, and
    the nuisances have no match term (`tier1/null_offset.json`).
  - On the real data, a within-match (match-stratified Hajek) reweighting of the
    saved nuisances gives 0.0307, CI [0.0072, 0.0542]
    (`tier1/match_stratified.json`; exploratory). Between-match comparisons
    therefore do not drive the estimate.
- **Rejection rates.** The within-match null rejects at 2%, the dummy-Y null at
  4%. These are software checks, not tests of exchangeability.
- **Simulations.**
  - Pooled coverage is 0.9325 (8 scenarios × 50 datasets). By scenario, in the
    three zero-effect scenarios: 1.00 in healthy_zero and weak_overlap_zero,
    and 0.92 in both_nonlinear. In the five non-zero-effect scenarios: 0.98 in
    healthy_negative, 0.90 in healthy_positive and nonlinear_propensity, and
    0.88 in action_visibility and nonlinear_outcome.
  - Power at an effect of about 0.02 is 56–64% (healthy_positive 56%,
    nonlinear_propensity 62%, nonlinear_outcome 64%).
  - The action-visibility scenario has bias −0.0068.
  - The simulated covariates are synthetic, so these results are not calibrated
    to football data.
- **Sensitivity (DVDS).** The tipping point is Γ ≈ 1.35. The lower bound at
  Γ = 1.25 is 0.0095; at Γ = 1.5 it is −0.0138.

## 6. Status vocabulary

The run's claim status `completed` means **all pre-declared computations
completed**. It is not a validation of identification. See
`tier1/status_relabel.json` for the meaning of each check.

`confirmation/primary_status.json` in the run is a stale provisional record. It
is superseded by `submission_results.json` and has not been edited.

## 7. Robustness table

`tier1/robustness_with_diagnostics.csv` adds a diagnostics-passed column.

Analyses that **fail** the support/balance diagnostics:

- carry-excluded population: 0.0358
- local-pressure ablation: 0.0311
- removal of match 3857297: 0.0386
- removal of team 799: 0.0373

The interval includes zero after removing team 787 (0.0230, CI [−0.0009, 0.047])
and in the 5 m visible-disk population.

## 8. Tier 2 (post hoc)

Richer-covariate analyses are prespecified in
`docs/public_360_tier2_prespecification.md` (hash recorded before any fit).
Results are in `docs/public_360_tier12_results.md`. World Cup Tier 2 numbers are
post-hoc exploratory.
