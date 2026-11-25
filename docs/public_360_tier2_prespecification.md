# Public 360 study — Tier 2 prespecification (richer covariates, post hoc)

Status: written 2026-10-01, before any Tier 2 fit. Its SHA-256 is recorded in
`runs/public-tier12-20261001/prespec.sha256`. The `compare` phase refuses to run
unless the hash of this file matches the recorded one.

## 1. Standing of these analyses

- The primary result stays as reported by run `public-submission-full-gpu-20260930`:
  the supported World Cup 2022 ATT of 0.0308, CI [0.0067, 0.0548]. Nothing below
  replaces, re-selects or re-tunes it.
- All Tier 2 analyses are **post hoc**. They were designed after the World Cup result
  was seen. They run on Euro 2020/2024 (development) first. World Cup 2022 analyses
  are labelled **post-hoc exploratory** in every output.
- Every analysis listed here is reported whatever its result. No analysis is added,
  dropped or modified after seeing World Cup Tier 2 output. A deviation must be
  listed as a deviation, with its reason.

## 2. Feature definitions (module `src/PublicCausal/context_features.py`, version `ctx-v1`)

**Ordering and history.** Events of a match are ordered by (period, time, index).
The history of a decision is the set of same-period events ordered before the
decision group by (time, index). The decision group's own events are excluded.
Group A reads nothing at or after the decision. Group B reads only the decision's
own 360 frame.

### Possession reconstruction

Walk the history backwards and stop at the first of these:

- **An opponent control event.** This stop is exclusive. Control events are:
  Pass, Carry, Dribble, Shot, Clearance, Miscontrol, Dispossessed, a Ball Recovery
  that is not a failure, a Ball Receipt* that is not Incomplete, an Interception
  with a won/success outcome, and a Goal Keeper event of type Collected, Keeper
  Sweeper or Smother.
- **The period start.**
- **A team restart pass** (Corner, Free Kick, Throw-in, Goal Kick, Kick Off). This
  stop is inclusive.

The included team events form the possession. The possession starts at the first
on-ball team event.

### Group A: event context (19 numeric features)

| # | name | definition |
|---|---|---|
| A1 | `ctx_possession_age_s` | decision time − time of the possession start (0 if there is none) |
| A2 | `ctx_possession_passes` | completed team passes in the possession |
| A3 | `ctx_possession_start_x` | metric x of the possession start (the decision x if there is none) |
| A4 | `ctx_restart_{corner,free_kick,throw_in,other}` | the possession began with this team restart; all four are 0 when it began in open play |
| A5 | `ctx_set_piece_within_20s` | a corner, free kick or throw-in pass by either team in the history within 20 s |
| A6 | `ctx_in_linked` | the latest team pass in the possession is complete, its recipient is the actor, and the actor has a later non-Incomplete Ball Receipt* in the history |
| A7 | `ctx_in_unresolved` | there is a team pass in the possession but it is not linked as in A6 (the third state, no incoming pass, is A6 = A7 = 0) |
| A8–A12 | `ctx_in_high`, `ctx_in_low`, `ctx_in_cross`, `ctx_in_through`, `ctx_in_cutback` | attributes of the linked incoming pass (all 0 unless linked) |
| A13 | `ctx_in_length_m` | metric length of the linked pass (0 unless linked) |
| A14 | `ctx_in_from_wide` | the linked pass started with \|y\| ≥ 20.16 m |
| A15 | `ctx_actor_on_ball_s` | decision time − first event of the actor's uninterrupted run of possession events (0 if empty). Opponent non-control events do not interrupt the run. This replaces any "first-time" flag. |
| A16 | `ctx_actor_prior_actions` | the actor's Carry/Dribble events in that run, all completed before the decision |

### Group B: limited 360 geometry (3 features plus 2 observation flags)

Coordinates are metric; the posts are at (52.5, ±3.66).

| # | name | definition |
|---|---|---|
| — | `geo_triangle_visible` | the ball–posts triangle (shrunk 5 cm toward its centroid) lies inside the visible polygon |
| B1 | `geo_triangle_defenders` | opposing outfield players strictly inside the triangle; **NaN unless the triangle is visible** |
| — | `geo_keeper_observed` | an opposing keeper is present in the freeze frame |
| B2 | `geo_keeper_goal_dist` | keeper distance to (52.5, 0); NaN unless the keeper is observed |
| B3 | `geo_keeper_line_offset` | keeper distance to the ball–goal-centre segment; NaN unless the keeper is observed |

**Excluded by rule:**

- StatsBomb possession id and possession team
- `play_pattern`
- the `under_pressure` and `counterpress` flags
- all shot attributes
- `shot.key_pass_id`
- `pass.shot_assist`, `pass.goal_assist` and `pass.assisted_shot_id`
- best-open-teammate features

Unseen defenders are unknown, never absent.

**Known limitations, stated in advance:**

- The 360 frame is captured at the event. For shots, Group B may reflect
  defenders' and the keeper's reaction to the shooting motion, which is a
  post-treatment risk. Group B is therefore a sensitivity analysis, not the main
  Tier 2 analysis.
- Visibility depends on the action (the keeper is visible far more often for
  shots). Restricting to an observable sample changes the population and does not
  remove visibility-driven selection.
- Incoming-pass linkage is "unresolved" more often for shots (Euro probe: 21% vs
  10%), so it enters as an explicit category.

## 3. Samples and comparisons

**Notation:**

- P0 = the eligible rows of the saved fit (Euro: `development/fit.joblib`; World
  Cup: `confirmation/fit.joblib`).
- S0 = their saved support.
- V = the rows of P0 with `geo_triangle_visible` and `geo_keeper_observed` both
  true.

**Fitting rules for F1 and F2.** All nuisances are refit with the frozen
specifications (e = XGBoost CUDA, 7 leaves, 100 iterations, with grouped sigmoid
calibration; m0 = ridge α = 10) on the stored outer folds, via
`refit_on_fixed_population`. The support is held fixed.

| id | sample | arms (feature sets) | role |
|---|---|---|---|
| F1 | P0, target S0 | baseline; baseline + A | adjustment change on an identical sample |
| F2 | V ∩ P0, target V ∩ S0, nuisances trained on V only | baseline; baseline + A; baseline + A + B | adjustment change on an identical observable sample |
| F3 | full re-selection: the Euro design is rebuilt by `fit_frozen_design` with `feature_names` and `support_features` augmented by A; the World Cup is fit by `fit_study(frozen_design)` | baseline + A | **newly selected support population**, reported separately |

"Baseline" means the original 13 numeric and 4 categorical features. In F1 and F2
every arm, including the baseline, is refit, so that refit noise is not mistaken
for an adjustment change.

**Reported for every estimate:**

- ATT and the match-cluster t interval
- standard errors under team, opponent and two-way (CGM) clustering
- N1 and the number of controls
- shot retention relative to the P0-supported shots
- control ESS, maximum weight, maximum weight share, maximum match weight share,
  and the top-1% control weight share
- key and other maximum SMD on the estimate's own features and on the union
  feature list, with the gate flags
- out-of-fold propensity log-loss and AUC on supported rows
- positive and negative shot goals

**Also reported:**

- For each within-sample contrast: Δ, the paired match-cluster SE, and the CI.
- For V: membership counts by arm.
- For F3: the overlap of its supported shots with the primary supported shots.

## 4. Interpretation rules (fixed in advance)

- Every number is descriptive. No threshold turns a Tier 2 result into a new claim
  or into a rejection of the primary claim.
- A shrinking ATT may reflect additional adjustment, a different selected
  population, or measurement problems in the new features (timing, linkage,
  visibility). It is not by itself evidence of confounding of a given size.
- A stable ATT strengthens robustness **only to the tested additions**. It says
  nothing about other unmeasured confounders.
- Group B results carry the post-treatment and visibility caveats above.
- The World Cup analyses are post-hoc exploratory and reuse the holdout.

## 5. Uncertainty

Intervals use match-cluster t(G − 1) scores with nuisances treated as fixed, as in
the primary analysis. Within-sample differences use paired scores on identical
rows. No bootstrap is run for Tier 2.
