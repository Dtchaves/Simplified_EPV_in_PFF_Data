# Independent evaluation — public StatsBomb 360 submission run

Run evaluated: `public-submission-full-gpu-20260930` (phase `full`), Vulcano
`<root>/data/public_statsbomb_causal/runs/`. Evaluated 2026-09-30/10-01 from a read-only
snapshot of the run outputs (`*.joblib` excluded) plus read-only Python on Vulcano. No run,
notebook, data or git state was modified; no new World Cup fits were made.

Estimand: supported ATT of shooting vs the historical Pass/Carry/Dribble continuation mixture at
live-play event starts; signed first-goal reward within 15 s (or natural half end). Not EPV.

## Verdict

The headline is **computed correctly and the World Cup holdout was respected**. It supports a
**modest, narrowly scoped** statement: within the supported domain, shooting increased net
signed first-goal reward by about 3 per 100 decisions (WC 2022: 0.0308, score CI
[0.0067, 0.0548]; bootstrap [0.0029, 0.0561]).

The evidence is **moderate and fragile, not strong**:

- The Euro development cohort gives about half the effect, with a CI that includes 0.
- Unmeasured confounding with an odds ratio of about 1.35 is enough to tip the estimate to zero,
  and plausible unadjusted confounders exist that would bias it upward.
- Two of the robustness checks include 0.
- The fitted-model null shows a small positive bias.
- The claim status `completed` (7/7) mostly confirms that the work ran to the end. Its checks
  could not have failed on bad results, so it should not be read as validation.

## Headline re-derivation (holds)

| Quantity | Reported | Re-derived from `att_contributions/rows.parquet` |
|---|---|---|
| ATT | 0.0307738 | 0.0307738 (Σa/N1, N1 = 1,228 treated) |
| Hajek AIPW | 0.0307798 | 0.0307798 |
| Match-cluster SE | 0.0120398 | 0.0120398 (G = 64, G/(G−1) adjustment, denominator term included) |
| t(63) critical value / CI | 1.9983 / [0.00671, 0.05483] | identical |
| Cluster scores | 64, sum ≈ 0 | identical to `cluster_scores.parquet` (ratio 1.000) |

Further facts from the same rows:

- All 64 matches carry both shots and supported rows, so no match has a zero score.
- Control weights are raw e/(1−e). They sum to 1,230.2, a ratio of 1.002 to the treated count.
- Largest control weight is 6.19; control ESS is 1,006.
- The top 1% of controls hold 35% of the control weight.
- Propensity clipping at [1e-6, 1−1e-6] has no practical effect; trimming comes from the frozen
  support domain.
- Leaving out one match at a time moves the ATT between 0.0279 and 0.0367.
- Raw outcomes:
  - shots: 132 of 1,228 scored within 15 s (10.7%), 0 conceded;
  - continuations: 479 scored (2.3%), 5 conceded.
- Comparators on the same treated IDs:

  | Estimator | ATT |
  |---|---|
  | Raw difference | 0.084 |
  | Outcome regression | 0.034 |
  | IPW | 0.031 |
  | Spatial nearest-neighbour matching | 0.030 |
  | Nearest-neighbour matching | 0.047 |

`att.json` has two intervals. `score_interval` (t-based) is the headline. The other,
`diagnostic_interval_only`, is a legacy ±1.96·SE interval used only by the extended profile.

## Findings by severity

### High

**H1. `completed` is a completion status, not a validation status.**
Code: `cell_24` `_submission_null_check`, `_submission_simulation_check`, `analyses=...`, and
`precision=...`; `cell_26` `_submission_science_failure`. I verified each:

- **simulations** passes if every task was attempted, there were no "computational" failures,
  and each scenario has at least one success. Bias and coverage are never checked.
- **analyses** passes on completion only. It passed even though visible_disk5
  [−0.0047, 0.050] and removing team 787 [−0.0009, 0.047] include 0.
- **nulls** requires |mean| ≤ max(0.005, 4·MCSE). It ignores the rejection rate, and it passes
  automatically when the MCSE is undefined.
- **failure classification**: every `ValueError` is classed as "structural", and structural
  failures never block a check. 49/50 structural failures would still pass. The test
  `test_submission_confirmation.py:115` asserts this behaviour.
- **precision** passes automatically once 500 draws are reached. In this run it passed on its
  own merits.

Effect on this run: none of these paths were triggered (0 failures). But the report's
"7/7 passed" table overstates what was validated.

Fix: rename the status (e.g. `execution_complete`). Gate on the null rejection rate,
simulation coverage and bias, and structural failures. State in the report that the checks are
completion checks.

**H2. Unmeasured confounding is plausible and points upward; the robustness margin is small.**

- **Covariates used:** location, distance, angle, defenders within 3 m, nearest and second
  defender, score, time, recent history, plus role/team/opponent.
- **Not adjusted:**
  - goalkeeper position;
  - defenders in the shooting lane;
  - teammates ahead;
  - defenders within 5 m and 10 m.

  The columns exist in the data but are unused. They drive both the decision to shoot and the
  chance of scoring.
- **Balance of unused columns under the saved weights** (descriptive, no refit):
  - `observed_teammates_ahead`: weighted SMD **+0.19**, above the study's own 0.15 threshold for
    secondary covariates. Shots have more teammates ahead (2.47) than weighted continuations
    (2.10), which suggests the continuation counterfactual for shot situations is understated.
    This would bias the ATT **upward**.
  - Defenders within 5 m: +0.12.
  - Keeper x: −0.10.
  - The other unused columns balance to |SMD| ≤ 0.09.
  - Keeper position is missing for 41% of controls vs 6% of shots, so visibility depends on the
    action taken.
- **DVDS:** the point identification bounds include 0 at Γ ≈ 1.35 (bracketed between 1.352 and
  1.359), a modest odds departure.

Recommendation: state the result as conditional on the measured local state. Report the
teammates-ahead imbalance. Do not describe the effect as robust to hidden confounding.

### Medium

**M1. The WC result is not replicated on Euro.**

- Euro development ATT: 0.0177, CI [−0.0022, 0.0376], from 102 matches and 2,195 shots.
- WC: 0.0308.
- The two are statistically consistent: the difference is 0.013 with SE ≈ 0.016.
- An exploratory inverse-variance pool gives ≈ 0.023 [0.008, 0.038]. Euro was used for design
  selection, so it is not independent confirmation.
- The handoff and status documents quote only WC. Report the Euro estimate alongside it.

**M2. Researcher degree of freedom: the "one-SE prefer CUDA" propensity rule (`cell_05`
`select_propensity_candidate`).**

- **Losses:** HGB 0.117572 (SE 0.00224, from 3 inner-fold losses); XGB-CUDA 0.117655;
  logistic 0.1275.
- **The rule changed the selection.** A plain lowest-loss rule picks HGB; the event
  `one_se_prefer_cuda` shows XGB was chosen.
- **Timeline:** the rule was added between about 20:46 and 21:19 UTC, after pilots 1–3 had
  shown the Euro in-sample ATT under HGB (0.0169, interval including 0), but before any WC
  access.
- **Did it matter on Euro?** No: under XGB the Euro ATT was 0.0177, still with an interval
  including 0, so the switch did not make the Euro result look better.
- **Verdict:** acceptable given the tie (a loss difference of about 0.04 SE). It must be
  disclosed as a choice made after the pilots, for the user's GPU preference, and not
  pre-registered. It is not "set in advance" in any strict sense.

**M3. The fitted-model null shows a small positive bias.**

- Within-match randomized-treatment null: mean ATT **+0.0022** (MCSE 0.0007, t ≈ 3.1) with
  fitted propensities, vs +0.0003 with the known assignment.
- So the nuisance models alone add about 7% of the headline, in the same direction.
- It passes only because of the 0.005 floor. Disclose it.
- Rejection rates were 2/50 and 2/50. With 50 runs, a 5% error rate cannot be told apart from
  10%. These are software checks, not identification evidence; the outputs label them correctly.

**M4. The simulations are weak evidence of coverage in the relevant regime.**

- Coverage:
  - zero-effect scenarios: 100/100;
  - non-zero-effect scenarios (healthy_positive, nonlinear_propensity, nonlinear_outcome):
    0.88–0.90, each with a Monte Carlo interval of about ±0.08.
- The pooled 0.93 hides this split. It hints at about 5 points of under-coverage when the effect
  is non-zero, which is the case that matches reality.
- **How realistic the generated data are** (`cell_13`):
  - covariates are synthetic;
  - the outcome depends on the covariates by only about ±0.005, so outcome-model
    misspecification barely matters;
  - there are no team effects, so dependence from repeated teams is untested.
- Effect sizes are realistic: 0.02 vs the observed 0.031.
- Power at a true effect of 0.02 is only 56–64%.
- 178/400 simulated datasets failed support diagnostics but were kept. The real pipeline would
  have stopped on those.
- action_visibility scenario: bias −0.0068 (about 60% of its truth), coverage 0.88. This shows
  that action-dependent visibility can bias the ATT materially. The direction depends on the
  generated data; it is not evidence about the real direction.

**M5. The WC support pass does not survive resampling.**

- 489/500 bootstrap draws fail at least one support gate.
- **Weight-share failures** (median 0.0122 vs 0.01 limit) are largely mechanical: duplicated
  matches multiply donor shares.
- **Key-balance failures** are substantive: median key SMD 0.125 across draws, vs 0.075 on the
  point sample and a 0.10 limit.
- Draw ATT correlates +0.34 with key SMD: worse balance goes with a larger ATT.
- The 11 draws that pass every gate average about 0.025.
- **Is the percentile interval still meaningful?** Yes. It correctly keeps every draw (none
  failed), its precision rule reproduces, and bootstrap SD / score SE = 1.11.
- But the single-sample support pass sits within resampling noise.
- **Precision rule reproduced:** at 200 draws, the lower-endpoint MCSE was 0.0032 > 0.0028 (the
  limit), so the extension triggered. At 500 draws, 0.0020 < 0.0027, so it passed. P(draw ≤ 0)
  = 1%.

**M6. The supported population excludes the most valuable shots, and the effect varies strongly
by distance.**

- **Excluded shots:** 66 of 1,294 eligible shots (5.1%) were dropped for
  `insufficient_cross_match_cell_donors`:
  - median distance 5.2 m, mean angle 63°, 30% conversion;
  - kept shots average 16.5 m and 10.7% conversion.

  The estimand therefore excludes close-range shots where passing alternatives almost never
  occur. Relative to an all-shots effect the ATT is probably conservative. Label the population.
- **Distance strata** (both "insufficient" because secondary SMD was 0.16–0.18 > 0.15):

  | Distance | ATT | Marginal CI |
  |---|---|---|
  | < 18 m | 0.059 | [0.022, 0.097] |
  | ≥ 18 m | −0.013 | [−0.031, 0.005] |

  This is the most interpretable pattern in the study, but it is exploratory only.
- **Qualifying stratum:** only angle_low qualified: 0.0156, simultaneous interval
  [−0.0015, 0.0327].
- **Contrasts:** all insufficient.

**M7. Two lock gates are almost impossible to fail.**

- `design_sanity` allows |att − truth| ≤ 0.05 + about 4·SE, for a 0.02 effect.
- `weighted_validation` allows weighted MSE ≤ 0.25 (typical values 0.07–0.10), and decile
  residuals within 0.03 + 4·SE.
- Euro m0 weighted-decile residuals are +0.02 to +0.04 in 9 of 10 bins, a systematic
  under-prediction that still passed.
- WC m0 control-arm calibration also under-predicts by 2–6 percentage points in the middle bins.
  AIPW corrects this through the weights (residual correction +0.0034), but double robustness
  then rests on the propensity model.

### Low

- **L1. Subgroup "simultaneous" interval is narrower than the marginal one.** It uses a Gaussian
  critical value with no df correction: 1.937 vs t(63) 1.998.
- **L2. Wrong lock path in the run config.** `run_config.design_lock_path` points to a path that
  does not exist (`runs/<id>/design_lock.json`; the real lock is in `full/development/`). A
  rerun of `full` would re-enter development and re-lock after WC was opened. This was not
  triggered here: there was a single development pass. Fix before any rerun.
- **L3. Bootstrap attempt cap and summary label.** The extension's attempt cap is hard-coded at
  750, while the lock records `bootstrap_max_attempts=300`. The summary also reports
  `requested: 200` after extending to 500.
- **L4. The source notebook was edited after the run.** The isolated copy changed at
  2026-10-01 02:16Z (report cells added). The core hash is unchanged (`0959a70c…`), but the file
  is not byte-identical to what ran. Archive `executed.ipynb` / `configured.ipynb` as the record.
- **L5. Single cross-fitting split for the point estimate.** The bootstrap mean is 0.0317.
- **L6. `exports/model_validation.json` contains 5 empty fold stubs by construction** (the
  fixed-spec path records no tuning). Rename it or point it to `weighted_validation.json`,
  `outcome_validation.csv` and `weighted_calibration.csv`.
- **L7. Test gaps.** Nothing tests the precision-extension trigger, the automatic pass at the
  maximum, the null thresholds, or claim-status combinations beyond one negative case.

### Holds (info)

- **Holdout integrity:**
  - Lock `frozen_at` 21:51:38.025Z.
  - First WC event 21:51:38.327Z; first download 21:51:38.620Z.
  - All 193 WC files were freshly downloaded, with remote mtimes 21:51:38–21:53:54Z.
  - No earlier run (pilots, preflight, smoke, baseline, benchmarks) references WC. The pilot
    "confirmation" rehearsals ran on Euro in-sample data.
  - `raw/data/competitions.json` (11:02Z) lists WC in the catalogue only.
- **No leakage from imported artifacts:**
  - The `import_manifest.json` artifacts match their recomputed sha256.
  - The imported fold registry covers 102 Euro matches only, all part of the lock's
    development matches.
  - The support model was trained on development matches only.
- **Nothing changed after WC access:** the core and helper hashes are identical across lock,
  confirmation and execution provenance; the confirmation is bound to the lock hash; one process
  ran from 21:51:38 to 23:05:23Z. The 200→500 extension was pre-declared, independent of
  significance, and resume-invariant.
- **Confirmation used the frozen spec:** it re-validates the lock from disk and raises an error
  on any manifest mismatch.
- **XGBoost on CUDA:**
  - Boosters trained on `cuda:0` (`grow_gpu_hist`).
  - The "mismatched devices" warning comes from passing CPU (numpy) input to a GPU booster. It
    costs speed only; predictions are unchanged.
  - Confirmation `xgb_prediction` events record no warnings.
- **Report cells read the exports faithfully,** say "Not EPV", and disclose the diagnostic-poor
  draws and the Γ tipping point.

## Recommended framing for the write-up

1. **Headline:** "In the held-out World Cup 2022 cohort, within the supported domain (95% of
   shots; close-range shots without comparable continuations excluded), shooting increased net
   signed first-goal reward within 15 s by 3.1 per 100 decisions (approximate match-cluster 95%
   CI 0.7–5.5). Conditional on measured local state; not EPV."
2. **Report next to it:**
   - the Euro development estimate (1.8, CI −0.2 to 3.8);
   - DVDS tipping at Γ ≈ 1.35;
   - the two robustness analyses that include 0;
   - the +0.0022 fitted-model null bias;
   - the teammates-ahead imbalance.
3. Describe the claim checks as execution/completeness checks.
4. Disclose that the CUDA propensity rule was added after the Euro pilots.
5. Present the distance heterogeneity as exploratory and gated out.
6. **Before any rerun:**
   - fix the lock path (L2);
   - strengthen the claim gates (H1);
   - record a paired-comparison rule for backend selection.

   Any rerun is no longer untouched confirmation.

## Addendum (2026-10-01): comparison with a second independent assessment

### Claims from the second assessment that I checked against the saved WC rows (all hold)

- Weighted continuation mix: 50.3% Carry, 46.1% Pass, 3.6% Dribble.
- Defender distances are capped at 3 m. Among eligible controls, 54.5% have the nearest-defender
  distance at the cap, and 91.9% have the second-defender distance at the cap.
- The 132 positive shot rows correspond to 124 distinct goals (`goal_event_15`).
- The 5 negative control rows correspond to 4 distinct goals.
- `confirmation/primary_status.json` is a stale provisional snapshot (`primary_claim=false`).
- `development_timing_gate` (cell 21) only checks that outcomes are present and that there are at
  least 30 matches. Its own docstring says it is not a pre-action timing check.
- Paired null offset: fitted vs known assignment probabilities. My figures, +0.0022 vs +0.0003,
  match theirs, +0.218 vs +0.030 per 100.

### New: inference under dependence across repeated national teams

This uses the saved scores only, with no refits. It is a post-hoc sensitivity analysis.

| Clustering | Clusters | SE | 95% CI | p |
|---|---|---|---|---|
| Match (primary) | 64 | 0.01204 | [0.0067, 0.0548] | 0.013 |
| Attacking team | 32 | 0.01306 | [0.0041, 0.0574] | 0.025 |
| Opponent | 32 | 0.01645 | [−0.0028, 0.0643] | 0.071 |
| Two-way team + opponent (Cameron–Gelbach–Miller) | 32 | 0.01616 | [−0.0022, 0.0637] | 0.066 |

All rows use t(G−1).

Conclusion: whether the result is significant at 5% depends on matches being independent. Once
opponent-level (defensive) dependence is allowed, the interval includes 0. Two caveats apply: two-way
clustering with 32 clusters is itself imprecise, and opponent clustering is not the pre-registered
analysis. Both must still be reported.
