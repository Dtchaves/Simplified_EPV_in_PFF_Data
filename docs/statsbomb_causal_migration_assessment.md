# Public-data migration assessment: shooting versus continuing possession

Reviewed September 30, 2026. Remote project: `/home/joaomarcostomaz/causal/Simplified_EPV_in_PFF_Data` on Vulcano. This is an assessment, not an implemented migration or a fresh execution of the experiments. Remote source files were inspected and copied to temporary local storage; they were not edited.

## Conclusion

Hudl/StatsBomb events **plus 360** are suitable in principle for a public-data version of the current signed goal-reward causal experiment. Feasibility of a sufficiently supported, precise causal result must still be established in the pilot. The research question and 15-second outcome can remain, but provider adaptation alone is insufficient: verified support, cross-fitting, bootstrap and refuter issues must also be addressed. This reproduces the research framework in a different population, rather than reproducing the original Premier League 2024/25 dataset or its numerical results.

The final continuous-state EPV experiment requires a separate adaptation. Public 360 supplies event snapshots, not full tracking trajectories, velocities or arbitrary-time states. An event-based value model would be a new state/outcome definition.

## What the latest notebook is doing

The newest non-checkpoint notebook by modification time is `results/models/support_aware_causal_pipeline_demo-v6-workflow copy.ipynb`, modified June 15, 2026, 02:22:49 in the remote filesystem's displayed time. Also reviewed: the original v6 workflow notebook's plan, `dowhy_placebo_diagnostics.ipynb`, its library, current exported configuration and relevant source modules.

The question is the average effect of **shooting now versus continuing possession now**, among actual shot situations where continuing is plausible. Treatment is shot; control combines pass, carry and dribble. The latter represents the observed mixture of continuation actions, not the best available pass or a particular prescribed continuation strategy.

The primary estimand is `E[Y(shot) - Y(continue) | actual shot, E0=1, S_ATT=1]`. Adjustment uses pre-action spatial state, team/player-position/match context and recent possession history. Shot outcome, pass completion, save, rebound and future state are excluded from adjustment.

The latest copy's saved run has:

| Item | Saved value |
|---|---:|
| Input | 240 Premier League 2024/25 matches |
| Raw event/decision candidates | 313,302 |
| Eligible candidates under L2 | 55,787 |
| Primary supported shots / continuation donors | 3,484 / 51,389 |
| Primary outcome | `Y_reward_15` |
| Primary AIPW ATT | +0.043582 |
| Common-overlap ATT / ATC | +0.047004 / +0.017345 |

`Y_reward_h` is the first goal's signed reward within h seconds: +1 if the acting team scores, -1 if the opponent scores, otherwise 0. The working horizons are 5, 10 and 15 seconds. It is not EPV. The intended final outcome adds future state value and subtracts current state value, with a terminal-state convention.

The input uses centered, attack-right, normalized 105×68 coordinates: x in [-52.5,52.5], y in [-34,34]. L1 covers the final third. **L2 is active in the latest copy**, despite prose elsewhere saying L1: x≥17.5 and either |y|≥9.16 or x≤36. This excludes some close, central shooting positions and materially changes the target population.

Primary support requires continuation probability >0.05 at treated shot rows and at least 10 continuation donors in the same distance-to-goal × field-zone × defender-density cell. These are coarse tactical cells; this is not a nearest-neighbor caliper guarantee. The common-overlap sensitivity/ATC branch additionally requires 0.05<shot propensity<0.95. Primary continuation donors have no corresponding upper propensity bound, which permits large ATT weights.

## Current reproducibility and diagnostics

Treat the latest numbers as saved exploratory results. The latest copy contains edited cells and retained old outputs; its bootstrap and final export cells are unexecuted. Its prose and eligibility definitions are not fully synchronized.

Three distinct configurations coexist:

| Evidence | Configuration | Finding |
|---|---|---|
| Latest June 15 notebook copy | L2, primary continuation support | ATT +0.043582; donor ESS 541.61; max donor weight 108.86 |
| Current v6 export configuration, June 14 | L1, primary continuation support | ATT +0.443859; donor ESS 63.65; very unstable exported bootstrap |
| Saved adjacent placebo analysis / June 13 summary | Older L1 common-overlap sample | 17,339 supported rows; manually refitted placebo effects generally near zero |

The latest copy still has substantial weighted imbalance, including SMD 0.258 for defenders within 10m. Its DoWhy fixed-support placebo is -0.044266 with reported p=0; EconML ATT is -0.330541 versus custom AIPW +0.043582. The older manual placebo results suggest the original DoWhy anomaly depends on diagnostic implementation/configuration, but do not validate the latest L2 analysis. Rerun those diagnostics against matching inputs and support rules.

Two inference details should be addressed during a clean rerun. Opposite-arm nuisance predictions currently average all fold models, including models trained on other observations from the predicted row's match. Use shared match folds and make both-arm predictions only from models excluding that match. Bootstrap draw IDs currently allow duplicate copies of one original match into separate cross-validation folds; retain the original match ID as the fold grouping key.

The follow-up executable audit reproduced these issues and reconstructed the saved L1 ATT **0.4438589886** from every row's saved predictions. Only **82 of 52,317 donors**, with e≥.95 and therefore outside the shot propensity gate, account for **68.84% of donor weight** and **+0.35978** of that ATT score. This is not a corrected effect estimate. The asymmetric target/donor domain also invalidates the ordinary propensity-correct double-robustness argument without an additional correction. The ATT arithmetic itself agrees with its residual decomposition.

A synthetic probe of installed DoWhy 0.14 reproduced a stale-score placebo artifact: randomizing treatment can retain propensity scores for the original treatment. It explains a possible failure mechanism but does not resolve the football placebo result without a corrected rerun. Full evidence, source fingerprints, normalization details, comparator caveats and runnable probes are in [the code audit](causal_code_audit.md), with [saved results](causal_code_audit_results.json).

## Public data and cohort selection

Audited Hudl repository revision: `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`. Its tree contains 4,235 event files and 426 360 files. Count actual files, rather than assuming competition-level 360 metadata guarantees availability. Events and lineups exist much more broadly than 360. These counts are derived from the [repository tree](https://api.github.com/repos/hudl/open-data/git/trees/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb?recursive=1) and [competition metadata](https://raw.githubusercontent.com/hudl/open-data/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb/data/competitions.json).

| Cohort | Competition / season IDs | Matches with events and 360 | Scope |
|---|---|---:|---|
| Euro 2024 | 55 / 282 | 51 | Complete tournament |
| World Cup 2022 | 43 / 106 | 64 | Complete tournament |
| Euro 2020 | 55 / 43 | 51 | Complete tournament |
| Bundesliga 2023/24 | 9 / 281 | 34 | Leverkusen fixtures, not the full league |

Start with Euro 2024, then use the other men's tournaments as separate replication cohorts or pool the 166 matches with explicit competition context and separate cohort results. League subsets centered on Barcelona, PSG or Leverkusen create team-selection problems for a general-league estimand.

The available Premier League seasons are 2003/04 and 2015/16; neither supplies the needed 360 cohort. Premier League 2024/25 is absent from this snapshot. These are different data populations from the Vulcano study. [Competition metadata](https://raw.githubusercontent.com/hudl/open-data/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb/data/competitions.json).

The [360 specification](https://github.com/hudl/open-data/blob/master/doc/Open%20Data%20360%20Frames%20v1.0.0%20(1).pdf) describes an event UUID, visibility polygon and participant positions with teammate/actor/keeper flags. Anonymous nonactors do not have persistent player identities. The [event specification](https://github.com/hudl/open-data/blob/master/doc/Open%20Data%20Events%20v4.0.0.pdf) supplies action, time, player/team, possession and outcome fields. Public shot-only freeze frames cannot provide equivalent control-arm spatial context; use 360 for both arms.

## Acquisition

Prefer selective raw JSON downloads at the pinned revision for reproducibility:

1. Fetch `data/competitions.json`, `data/matches/55/282.json` and the documentation/license.
2. For every chosen match ID, fetch `data/events/{match_id}.json`, `data/lineups/{match_id}.json` and `data/three-sixty/{match_id}.json`.
3. Confirm each requested file exists and that UUID joins are unique and valid. Preserve the entire event stream, including events without 360, for history, score and reward labels.
4. Save a manifest with revision, exact match IDs, file hashes and acquisition time. Convert normalized tables to Parquet match by match.

The raw URL prefix is `https://raw.githubusercontent.com/hudl/open-data/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb/`. Euro 2024's 153 event/lineup/360 files total **560,556,509 bytes**, approximately 561 MB before tiny metadata files; this is calculated from tree blob sizes. A full repository download is unnecessary.

For interactive exploration, [official statsbombpy](https://github.com/hudl/statsbombpy) offers unauthenticated `sb.competitions()`, `sb.matches(...)`, `sb.events(...)`, `sb.lineups(...)` and `sb.frames(...)`. Fetch frames separately: `events(include_360_metrics=True)` is not the public-data route and the implementation requires commercial credentials for those metrics. [Official implementation](https://github.com/hudl/statsbombpy/blob/master/statsbombpy/sb.py).

Publication should credit StatsBomb and use its logo as requested in the [repository README](https://github.com/hudl/open-data). Read the bundled [research-use agreement](https://github.com/hudl/open-data/blob/master/LICENSE.pdf); public accessibility is not an unrestricted redistribution license. Share acquisition/processing scripts and result artifacts with appropriate attribution.

## Required code changes

All source paths below refer to the newer Vulcano project, not the older local checkout.

| Location | Required change |
|---|---|
| New provider adapter; notebook cells 10/12 | Load events, lineups and UUID-linked snapshots instead of PFF event/tracking/player triplets. Separate event data from snapshot observations. |
| Cell 19: `build_decision_points`, sorting/history/score helpers | Preserve event UUID, native event index, possession, snapshot quality and explicit period clock. Build context before filtering candidate actions. |
| Cell 21: `classify_action`; restart mapping | Map Shot→1 and Pass/Carry/Dribble→0; distinguish restarts and open-play actions using pass/shot types and play-pattern context. |
| Cells 24/26: eligibility and tracking features | Replace full-tracking player counts and player-ID joins with snapshot rows, actor/teammate/keeper flags and visibility checks. |
| `src/Pass/reward_labels.py` | Add a provider-neutral or StatsBomb goal ledger/labeling branch, preserve UUID identity, account for own goals and goal occurrence timing. |
| Cell 38; `src/CausalSupport/analysis.py` and `propensity.py` | Refit nuisance models and recompute support, balance, sensitivity and calipers on the new feature schema. |
| `score_action_selection_propensity_source` | Disable the optional PFF-trained comparator initially; make its unavailability explicit and use separate provider artifacts. |
| Cells 4/66/80 | Separate output/cache namespace, fingerprint input/schema/policies, fix grouping in inference, execute/export the complete run. Disable synthetic fallback. |

### Identity and geometry

Use `(match_id,event_uuid)` as the snapshot join key. Existing `_normalize_id` in `reward_labels.py` converts IDs to numbers, so raw UUIDs become missing. Either generalize event identity handling or retain the native integer event index as an internal `event_id` while preserving the UUID for joins and provenance. Never interpret an event index as a video frame or a uniform time step.

Normalize event locations, snapshot locations and visibility polygons together. An explicit centered convention is `x=105*x_sb/120-52.5`, `y=34-68*y_sb/80`, with attacks toward x=52.5. A consistently mirrored y convention is also possible. This assumes a normalized 105×68 pitch, not known physical dimensions for every venue. Preserve StatsBomb's event-team attacking orientation; avoid applying a second home/away flip. Use explicit geometry rather than inferring it from sample minima.

Use event `position` for actor context, with lineup position intervals when needed. Native possession groups also contain opponent defensive events, so construct acting-team history carefully. Provide actual match minute rather than adding a period offset twice.

### Spatial features and visibility

`tracking_features_for_event` currently drops participants lacking player IDs and uses distinct IDs to count them. Both assumptions fail for anonymous 360 participants. Count snapshot rows, use `teammate` for team relation, explicitly exclude `actor` from nearest-teammate calculations, and use `keeper & not teammate` for the opposing goalkeeper. Snapshot-local participant indices are permissible row keys, not identities across events.

Distinguish **observed** spatial counts from whole-team counts. A missing defender outside the camera view is not evidence of no defender. Nearest-defender distance can be valid if the relevant neighborhood is observed; defender counts within 3/5/10m require corresponding local coverage. Ahead-of-ball, penalty-box and ball-to-goal features require their own relevant-area checks. Manually included players outside a polygon do not make its entire surrounding region observed. A missing keeper is missing, not located at the goal center.

An actual-file audit of Euro 2024 Germany–Scotland, match 3930158, found:

| Candidate action | Total events | With 360 |
|---|---:|---:|
| Shot | 21 | 21 |
| Pass | 998 | 907 |
| Carry | 774 | 735 |
| Dribble | 20 | 17 |

Within the L2 region, a snapshot analogue of the current ≥6 attackers/≥6 defenders/≥14 players plus observed opposing-keeper gate retains **9/12 shots versus 169/444 continuation candidates**. These counts precede restart exclusions and other eligibility checks; they are not a complete causal sample or corpus-wide rates. They demonstrate why keeping the PFF observation gate would select the arms differently. Sources: [events](https://raw.githubusercontent.com/hudl/open-data/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb/data/events/3930158.json), [360 snapshots](https://raw.githubusercontent.com/hudl/open-data/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb/data/three-sixty/3930158.json).

Start by auditing spatial missingness by action, location and cohort. Define a common observable region/feature rule and report the population lost through it. Consider a primary model with reliable local geometry and a richer-feature sensitivity analysis over narrower visibility. Do not simply lower player-count thresholds until estimates run. This restriction changes the target to supported, observable shot situations; it does not establish exchangeability or represent all shots.

In particular, a common gate does not automatically cure selection bias if snapshot availability/camera framing responds to the action or unmeasured tactical state. Establish the observation timing/mechanism as far as possible, document treatment-differential coverage, and evaluate visibility-policy sensitivity. If this cannot be justified, restrict the causal claim accordingly; unmeasured-confounding sensitivity conditional on observed rows does not by itself recover unobserved situations.

### Decision occasions and rewards

Define the index moment at the start of the selected action. Carry snapshots in the audited match match the carry start/receipt context, not the endpoint. Sequential carry followed by pass or shot can produce multiple decision rows; retain an explicit event-level interpretation and perform carry-excluded sensitivity. Do not use carry endpoints, duration or success as adjustment. Carry `under_pressure` can reflect pressure during or at its end, so derive baseline pressure from the start snapshot instead. [Event specification](https://github.com/hudl/open-data/blob/master/doc/Open%20Data%20Events%20v4.0.0.pdf).

Construct a scoring ledger from the **full event stream**. Recognize ordinary scored shots and own goals, deduplicating linked Own Goal For/Against records. Attribute the reward to the event actor's team, which is not always the same as `possession_team` around a turnover. Reconstruct pre-event score using only goals already scored.

A scored shot timestamp records the attempt's start. Prefer reliable related Goal Keeper/Goal Conceded timing for actual goal occurrence; use a documented fallback and assess horizon-boundary sensitivity. `shot timestamp + duration` is not universally exact. Preserve reward for the current scoring shot and distinguish equal-clock event order. Group windows by match and period, exclude penalty shootouts, and explicitly define terminal period/censoring rules rather than automatically giving unobserved trailing windows a zero.

### What can remain

The observational DAG, exclusion of mediators, supported ATT/ATC distinction, logistic/richer binary propensity models, custom AIPW/IPW/outcome-regression/matching estimators, balance diagnostics and refutation workflow can remain structurally. Their fitted models, support populations and results must be recomputed. Keep primary ATT and common-overlap diagnostics labeled separately; insufficient overlap remains a substantive result.

The notebook's actual imported dependencies are `PassRewardLabeler`/`ProcessedEventIndex` and `CausalSupport.run_propensity_stabilization_suite`. It does not compute EPV by calling the original neural inference stack. Consequently the first public signed-reward run does **not** require retraining the four pass models, BallDrive or Shot.

## Optional work for full EPV and action selection

The optional ActionSelection comparator uses consecutive tracking frames and persistent identities for pitch-control/velocity features. Event-adjacent snapshots cannot replace those frames. Build and retrain a snapshot-only feature model, or omit the comparator. Avoid silently filling missing pitch control with zero or loading the existing PFF artifact bundle. `shot.statsbomb_xg` is available only for chosen shots; putting it into the propensity covariates would reveal treatment. An xG-like covariate must be computed consistently for every candidate state from admissible pre-action inputs.

For a later EPV version, migrate `Pass/data_utils.py`, all pass tensorizers/dataloaders, `BallDrive` and `Shot` feature builders. Remove/redefine unavailable velocity channels and dynamic pitch-control inputs, add observation masks, and retrain in dependency order. Completed StatsBomb passes generally omit an outcome field; adapt semantics rather than dropping all rows with missing pass outcome. Carry is not a success-labeled take-on; do not invent its success label from Dribble semantics.

Public 360 cannot supply exact `V_t_minus_5` or `V_t_plus_h` states at arbitrary seconds. A next-event state value or event-based possession value is feasible but changes the outcome's temporal definition. State that change explicitly; per-shot xG is not interchangeable with team-perspective EPV.

## Incorporating the 12 improvement suggestions

**All 12 remain relevant.** None should be discarded because the data become public. Items 1–6, 8 and 12 are essential to the empirical claim; 7 and 9–11 strengthen validation and usefulness. Public data add an essential prerequisite: a defensible observable cohort and enough supported shots, goal events and independent matches. Snapshot coverage, tactical support, information loss and inference precision are separate checks.

The contribution should be: **which shooting-versus-historical-continuation comparisons are credible, how naive comparisons differ on that same population, and what supported tactical patterns remain after adjustment and sensitivity analysis**. A positive overall average or a complex architecture alone is insufficient.

### 1. Stable primary supported ATT — essential; strengthen the support requirement

Retain the user's recommendation, but fix the target/donor score compatibility before choosing among support candidates. Ordinary common-overlap support, applied to both arms as a common pre-action covariate domain, is a useful candidate. Empirical donor rules must be learned honestly and applied consistently to the target and residual correction. Football calipers remain useful diagnostics, but the current union-of-donors implementation is not a caliper-matched ATT and needs redesigned weighting/theory before it can become primary.

Select from a declared small set using balance, effective donor support, weight concentration, match-level donor availability, geometry and shot retention—not estimated effect. Report maximum/upper-tail weights, ESS, top-donor and top-match concentration, important covariate balance, retained shots/matches and visibility attrition. Predeclare diagnostic adequacy criteria using development data and disclose the prior result-driven methodological revision. If none pass, the result is insufficient support, not a forced headline estimate.

The old L1 common-overlap retention of 96.02% and ESS 1,728.23 are verified, but cannot be promised on the 51-match Euro cohort. L1/L2, cohort, visibility gate and learned support together define a new target. Keep unstable historical analyses as diagnostic evidence, not competing interchangeable estimates of one effect.

### 2. End-to-end cross-fitting and exact ATT accounting — essential; confirmed fixes needed

Use a shared outer registry keyed by original match ID for propensity, both action outcome models and any learned preprocessing/features. Fit each decision's observed and counterfactual predictions without any data from its match. Use nested grouped validation for calibration and tuning; prohibit silent row-split/in-sample fallbacks. DoWhy/EconML and any optional snapshot ActionSelection comparator need equivalent provenance, not merely an OOF flag.

Learn data-driven support in a separate honest design sample or outside each validation match. Current support counting is outcome-free, but that does not make it held out. Freeze support rules before confirmation; applying a frozen learned rule to a new match is distinct from estimating that rule using the same match.

Export a contribution table keyed by `(match_id,event_uuid)`: outer fold, training provenance, observation/eligibility/support reason, Z, outcome, e, mu0/mu1, odds weight, residual and ATT contribution. Save numerator/denominator totals and numerical assertions that row sums, estimator output and manuscript decomposition agree. The count-normalized AIPW already passes arithmetic checks; a weight-sum-normalized alternative is a separately labeled sensitivity, not a presumed bug fix.

### 3. Outcome models and influential-donor validation — essential; adapt to sparse ternary reward

Retain Ridge as a benchmark; compare a small prespecified set such as gradient boosting and a spline/additive model. A calibrated three-class model can provide `mu(X)=P(Y=+1|X)-P(Y=-1|X)` and respect outcome bounds. Its probabilities must be coherent. Do not use a binary scoring model alone as a replacement for the signed reward without separately addressing conceding.

Tune through nested match validation without selecting the ATT sign or magnitude. Evaluate held-out shot-like controls and influential donors, not only all controls. Report prediction ranges, calibration of reward components, weighted residual means/quantiles, weighted prediction error, influential-row/match contributions and ATT sensitivity at fixed support. For counterfactual shot predictions, treated outcomes do not directly validate mu0; use observed comparable controls and simulation. Report rare positive/negative outcome counts by fold/cohort and handling of absent classes.

Use only available pre-action inputs. Full velocities, orientation and balance remain unmeasured; extra architecture cannot recover them. Public 360 geometry can help but its reliability depends on visibility.

### 4. Uncertainty for the chosen primary ATT — essential; original-match grouping first

After fixing support and nuisance prediction, target approximately **1,000 successful match-cluster bootstrap replications**, with attempted/successful/failed counts, reasons, seeds, Monte Carlo stability and saved draws. Do not discard informative structural failures quietly. Report primary ATT uncertainty, not the different ATC interval. Add fully refitted leave-one-match-out influence checks; fixed-score contribution deletion can be a fast preliminary diagnostic but is not the same analysis.

Declare which preprocessing, tuning, calibration, nuisance fits and support components are refitted. Keep original matches together when bootstrap duplicates are split for fitting. Distinguish fixed learned-domain inference from recalculated-support pipeline inference; refitted discontinuous support can create nonregular behavior and needs simulation/alternative inference checks. Examine team/cohort dependence and report the small number of independent matches. More rows or repetitions do not compensate for sparse rewards, low ESS or influential matches. Euro 2024 has 51 matches, not the old 240.

### 5. Quantitative unmeasured-confounding sensitivity — essential; ATT and signed outcome explicitly

Add bounds for the **same supported ATT**, varying an explicit marginal-sensitivity odds factor such as Lambda. Report the smallest departure for which bounds include zero or permit reversal, with uncertainty. [Dorn, Guo and Kallus](https://arxiv.org/abs/2112.11449) provide a marginal sensitivity framework whose [Appendix A](https://jacobdorn.info/files/DVDS.pdf) explicitly extends to ATT. This is a candidate method, not an existing implemented feature of this project.

Use the appropriate general/signed discrete-outcome construction, not binary-outcome shortcuts or a mechanically applied scoring-risk E-value. Validate the bound implementation against synthetic truth, including the exchangeable limit, ties at zero, selected target and match dependence. Benchmark against influential measured covariates through a clearly explained omission/odds comparison; such benchmarks are suggestive, not proof that omitted confounding is weaker. Visibility selection needs its own treatment. Phrase the causal conclusion in light of both sensitivity and remaining observation assumptions.

### 6. Null checks and negative controls — essential; reproduce and repair before claiming resolution

Rerun the original football DoWhy failure with matching data/configuration, then with fresh clean inputs and refitted propensity. The synthetic stale-score reproduction is an explanation to test, not a certificate that the real-data result passes. Audit subset/random-common-cause refuters too. Independently test the custom score on dummy outcomes and randomized treatment; rebuild treatment-dependent support and nuisance fits when the null design requires it. Report frozen-domain and full-pipeline checks separately and explain the null being tested.

Choose any pre-action negative-control outcome on substantive causal grounds, with event timing verified and a specified expected relationship. Public 360 cannot supply arbitrary-time `V(t−5)`; use a justified available event-derived pre-action outcome, or explain why none is defensible. A random dummy outcome is not a substantive negative control. Prior football outcomes can affect current decisions, so an arbitrary lagged outcome need not have zero adjusted association. Null software checks cannot prove exchangeability.

### 7. Controlled simulation — strongly recommended; add public-data observation selection

Simulate match clusters, imbalanced treatment, sparse +1/0/−1 rewards, nonlinear game state and varying overlap, with known positive, zero and negative effects. Vary outcome/propensity misspecification individually and jointly. Add incomplete snapshot visibility and action-dependent observation scenarios to expose failures specific to the new data; do not assume the visibility restriction is harmless by construction.

Run the actual support-selection, nuisance, ATT and uncertainty workflow on repeated synthetic datasets. Compute conditional potential-outcome truth for each design's selected treated target using the generating model, not full-population truth; report selection-induced distortions separately where conditioning invalidates identification. Assess bias, RMSE, interval coverage/width, false-positive rate under zero, support retention, ESS and replication failures, with simulation Monte Carlo error. Include the reproduced support mismatch as a regression scenario and test whether diagnostics signal failure.

### 8. Timing, outcome and continuation — essential; provider adaptation is substantive

Keep 15 seconds primary, 5/10 seconds as existing sensitivities, and optionally predeclare 30 seconds. Report full and component first-goal effects across horizons; choose no horizon from the strongest estimate. A longer reward window does not create full-tracking EPV.

For each action, audit the start snapshot and event ordering. No carry end coordinates/duration/success, post-start carry pressure, shot xG/outcome, future score/history, fabricated velocity or next-event state may enter adjustment. Validate UUID joins, duplicates, history before action, native period clocks, goal occurrence timing, own goals, rebounds, equal-clock events, stoppages and end-of-half policy. Keep full streams even when decision rows lack 360. Boundary fixtures and hand-reviewed examples should make each label traceable.

Define continuation as a comparable **historical mixture of admissible Pass/Carry/Dribble actions with subsequent play unfolding naturally**, not best continuation or a universal deterministic action. Report mixture proportions and their changes by support, cohort and subgroup. Specify event-start occasions, sequential receipt/carry/pass/shot handling, exclusions and a carry-excluded sensitivity. Freezing the statistical procedure across tournaments does not freeze their historical action mixture; different cohorts can therefore answer slightly different policy questions.

### 9. Controlled comparisons and ablations — strongly recommended; snapshot versus tracking terminology

Compare raw differences, spatial matching, outcome regression, IPW and AIPW on the same supported shot IDs and donor domain with declared estimator weights. Use actual matching pair/reuse weights. Report nuisance-model changes with fixed support separately from analyses that rebuild support and change population; similarly separate propensity-source differences from sample selection.

Public data allow ablations of snapshot spatial state, context, recent event history, empirical support and football calipers—not unavailable continuous tracking. Holding the observable cohort/support fixed isolates adjustment-information differences. Expanding an events-only cohort or using a stricter visibility gate is a separate population/coverage analysis. Do not claim the missing full-tracking contribution is measured by a public-data feature ablation. L1 and L2 remain different target definitions.

### 10. Untouched confirmation — strongly recommended; reserve matches before public development

Reserve an untouched match/competition/season cohort now, before fitting public outcomes or selecting specifications. A practical design is Euro 2024 development followed by a frozen World Cup 2022 replication; this is a different-tournament confirmation/transport check, **not** prospective temporal validation. If development uses pooled tournaments, reserve independent match groups instead and record all prior exposure. Metadata inspection alone is not fitted-outcome validation, but any inspected pilot outcomes/matches should be labeled development.

Freeze eligibility, visibility, continuation, horizon, support-learning rules, features, model-selection algorithm and uncertainty procedure. State whether confirmation uses development-trained nuisance models or a frozen grouped refitting algorithm within the new cohort; both can be honest if declared, but they answer different generalization questions. Evaluate ATT, support/coverage, balance, donor quality and uncertainty, and match/team influence. If all matches have informed development, call resampling retrospective robustness. The public cohort may lack enough supported shots/goals for a strong confirmation; report that limitation rather than promise a decisive result.

### 11. Supported tactical findings — important; restrict subgroups to observable state

Prespecify a small set such as distance, angle, locally observed defensive pressure, adequately observed shooting lane and score/time context. Reuse the corrected subgroup target/donor construction; averaging selected subgroup shots against unrestricted controls with original weights is not automatically a subgroup ATT. Anonymous snapshots do not support player-specific trajectories or full-pressure claims outside the visible area.

For each subgroup report shot/donor/match counts, positive/negative reward counts, donor ESS, retention/visibility coverage and intervals; handle multiplicity and label exploratory work. Include supported/unsupported visual cases with the visible polygon, available defenders/keeper, comparable continuations and reasons for exclusion. A blinded analyst assessment can establish usefulness/interpretability but not causal identification. Never turn an average subgroup estimate into an assertion that a particular player's choice was a mistake.

### 12. Manuscript and reproducible package — essential; correct the effect units

Lead with the concrete decision comparison, one supported observable target, one primary specification and what analysts learn. Move alternatives and extensive diagnostics to supplements. Distinguish prior exploratory PFF findings, revised design and new public confirmation. Generate tables/figures from versioned saved artifacts, not manually edited notebook outputs.

One adaptation is necessary to the suggested wording: the current Y is the signed **first goal**, not net goal counts or EPV. `100×ATT` means additional favorable-minus-unfavorable first-goal outcomes per 100 supported decisions within 15 seconds. Do **not** call it “net expected goals per 100” unless the reward is deliberately redefined to count all goals, which would change the maintained outcome. Show scoring/conceding components before any scoring-probability approximation. An xG-like or EPV interpretation is not justified by the present label.

Release authorized code, pinned acquisition manifest/scripts, schema/observation policies, fold registry, locked environment, configs/seeds, simulations, contribution/diagnostic artifacts where permitted, and figure-generation workflow. Follow the StatsBomb agreement and attribution requirements; do not assume raw redistribution is unrestricted. Clearly state which public empirical outputs are reproducible and that private PFF results are not reproduced without authorized source data. Disable synthetic fallback, fingerprint inputs/configuration/source, and require clean end-to-end notebook/script execution with no stale outputs.

## Revised implementation sequence and acceptance gates

1. **Design/cohort gate:** create a separate public namespace and reserve confirmation matches; record the 15-second first-goal reward, L1/L2 candidates, action occasions, continuation mixture, visibility policy and development/confirmation exposure.
2. **Data gate:** acquire a pinned Euro pilot and implement UUID adapter, normalized geometry, history and goal ledger. Tests must pass for both-arm joins/attrition, own goals, equal clocks, horizon boundaries and terminal observation. Audit feature timing and coverage before fitting effects.
3. **Estimator gate:** correct shared-domain augmentation, common match folds, preprocessing/calibration/tuning, original-match bootstrap groups and clean refuter inputs. Convert the reproducible red audit probes into passing regression tests; reconcile row contributions with formulas. Optional PFF artifacts remain disabled.
4. **Primary-design gate:** on development matches, compare the small support/model set using prespecified diagnostics. Freeze a defensible target and model-selection procedure; save same-population estimator comparisons and influential-donor diagnostics. Stop the headline claim if support or observed reward information is inadequate.
5. **Validity gate:** run corrected null checks, justified negative controls if available, ATT-confounding sensitivity and controlled public-like simulations. Investigate remaining failures; do not replace them with agreement among other estimators.
6. **Inference gate:** obtain stable primary-ATT uncertainty with approximately 1,000 successful clustered draws after the earlier fixes, failure accounting and refitted match/team influence analyses. Validate coverage and distinguish fixed/reselected support.
7. **Confirmation/usefulness gate:** execute the frozen pipeline on reserved matches/cohort, then report prespecified supported subgroups and visual cases. Keep cohort targets/mixtures explicit; do not pool automatically to gain significance.
8. **Release gate:** build manuscript tables/figures directly from saved versioned results and release the authorized reproducibility package. A small/inconclusive supported effect is an acceptable outcome. Snapshot ActionSelection or separately redefined EPV can follow; neither is a prerequisite for this paper.

This plan incorporates the suggestions without treating them all as existing software bugs. Verified defects have executable evidence in the audit; stronger sensitivity, simulation, confirmation and tactical reporting are additional research requirements. No revised empirical ATT or public-data effect has yet been computed.
