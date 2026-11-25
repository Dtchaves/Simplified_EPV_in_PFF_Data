# Approved public 360 causal study

Status: approved by the user; implementation authorized. Updated 2026-09-30.

## Deliverables and scope

Create `results/models/public_statsbomb_causal_study.ipynb`, inspired by the existing support-aware causal notebooks. The notebook owns the scientific functions, analysis, configuration, narrative, diagnostics and exports. Small helpers in `src/PublicCausal/` handle acquisition, canonicalization, logging, checkpoints and execution. Do not replace the private PFF notebooks or alter unrelated worktree changes.

On Vulcano, store new data exclusively in `/home/joaomarcostomaz/causal/Simplified_EPV_in_PFF_Data/data/public_statsbomb_causal/`, with `manifests/`, `raw/`, `canonical/` and `runs/`. Keep notebook-tagged scientific definitions executable by workers/tests through a hash-named generated module, not a second maintained implementation.

Reference audits: `docs/statsbomb_causal_migration_assessment.md`, `docs/causal_code_audit.md`, and the reproducible probes beside them. No manuscript draft requested: produce the notebook and research exports.

## Public source and protected confirmation

Pin `https://github.com/hudl/open-data` at commit `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`. Download competition/match metadata, every event file, lineup and 360 file for Euro 2020 (55/43; 51 matches) and Euro 2024 (55/282; 51 matches), then World Cup 2022 (43/106; 64 matches) only after a valid immutable development design lock. Full selective raw acquisition is about 1.76 GB. Retain events without 360 for goals/history. Record URL, commit, checksum, bytes, license and acquisition status. Do not use cached misleadingly named WC samples: those are Euro 2024 data.

Development: 102 Euro matches. Confirmation: 64 WC matches, unopened until the development gates pass. Freeze support geometry/rules and model configurations before confirmation. Confirmation refits nuisances within match-grouped WC folds with frozen settings, but retains the Euro-trained support-domain model and does not retrim against WC propensity. Failed confirmation diagnostics mean unsupported confirmation, not retuning.

## Decisions, timing, features and outcome

Primary question: shooting versus a continuation drawn from the observed historical mixture in comparable contexts, followed by natural play. L1: centered field x >= 17.5 m. Convert coordinates x=105*x_SB/120-52.5, y=34-68*y_SB/80, consistently for events, actors and visible polygons. No second team flip.

Use Shot versus Pass/Carry/Dribble on live-play decisions, with a matched UUID 360 frame, actor within 1 m of event location, finite coordinates and a valid visible polygon covering the entire 3 m disk around the ball. Exclude direct restart decisions (penalty/direct-free-kick shots; corner/free-kick/throw-in/goal-kick/kickoff passes); retain subsequent live play. Anonymous 360 players cannot supply identities, velocity, continuous tracking or complete-pitch counts. Restrict primary local pressure to the fully visible disk: counts within 3 m and first/second opponent distance capped at 3 m. No goalkeeper/full roster requirements; keeper counts as opponent when local. Rich visibility/keeper features and 5/10 m disks are separate sensitivities.

Collapse same actor/team/period/timestamp equivalent continuation records, preserving source UUIDs; flag/exclude ambiguous shot-plus-continuation collisions. Keep distinct-time decisions. Carry-excluded sensitivity required. Adjustment uses only start/context/strictly earlier event history: no current duration/end location/xG/action consequence. Public event-start snapshots remain proxies, not proof of pre-action measurement; action-dependent visibility is a selection limitation.

Primary reward: signed first goal (actor-team +1, opposing-team -1, otherwise 0) within 15 s or natural period termination, whichever first. Periods 1-4 included, shootouts excluded. Normal Half End is absorbing, not a future-dependent baseline eligibility filter; collapse duplicate end records. Missing/abnormal ends or incomplete observation are unknown, never silently zero. Deduplicate ordinary and own goals, attribute to actor team, preserve event-index ordering at equal clocks. Prefer linked Goal Conceded/Penalty Conceded occurrence time, then valid shot duration, then shot start; export timing provenance and sensitivity. History/score crosses periods, windows do not. Keep horizon sensitivities including 30 s, and full-horizon-before-end as an explicitly different selected population.

## Cross-fitting and model selection

Five outer and three inner folds, grouped by original match, balanced using cohort/treatment only. Both potential-outcome predictions, propensity, preprocessing, tuning, calibration and learned support must exclude the evaluated match. Never fall back to row splits or in-sample OOF. Bootstrap duplicate match copies keep original-match groups; draw IDs do not become folds. Export fold/train provenance and test actual memberships.

Propensity candidates: logistic C={0.05,0.2,1}; histogram boosting with sigmoid calibration learned from grouped held-out predictions. Choose by grouped proper loss/calibration. Outcome candidates: Ridge alpha={0.1,1,10}; additive SplineTransformer/Ridge knots={3,5} and the same alphas; histogram gradient boosting max_leaves={7,15}, iterations={100,200}, learning_rate=.05, L2=1, early_stopping=False. Choose m0 using treated-resembling donor-weighted held-out MSE and a one-SE preference for Ridge, additive, then boosting; m1 grouped arm-specific validation. Report calibration, clipping to [-1,1], ranges and influential-donor weighted residuals. All data-adaptive choices for development OOF rows happen outside their match. Refit/freeze final choices on all development data for confirmation.

## Supported ATT and estimator

Portable support score excludes team/opponent/player/cohort; full adjustment propensity may include cohort/team/opponent/role. Apply the same X-domain to both arms. Test continuation-probability thresholds {.05,.10,.15,.20}. Football cells: goal-distance cuts 6/12/18/24/30 m, left/central/right y cuts +/-9.16 m, local defender counts 0/1/2+. Require at least 10 distinct continuation occasions from at least three training matches after the common gate.

Select the broadest diagnostics-passing domain, never on estimated effect: >=200 shots, >=30 matches; control ESS >=max(200,.25*n_shots); weighted SMD <=.10 on key features and <=.15 on other comparable covariates; maximum donor weight share <=1%; maximum match weight share <=10%. These are study tolerances, not guarantees of causal identification. A portable-score gate does not cap full-propensity weights; diagnose actual estimator weights.

Count-normalized ATT AIPW: [sum treated(Y-m0) - sum controls e/(1-e)*(Y-m0)]/n_treated, on the common supported domain. Save every row contribution and residual decomposition; assert equality. Keep weight-sum normalization explicitly separate. The legacy asymmetric support and opposite-arm leakage are regression tests, not reused primary code.

Compare raw differences, outcome regression, IPW, AIPW and actual nearest-neighbor/spatial matching on identical treated IDs/support; save match pairs and donor reuse weights. Do not call union-of-donors IPW a matched ATT or give invalid ordinary bootstrap inference for nonsmooth NN matching. Caliper/common-overlap sensitivities may change population, labeled accordingly. Feature ablations first fix support; rebuilt-support ablations explicitly change estimand.

## Uncertainty, null checks and influence

Target 1,000 successful development match-cluster bootstrap replications (stratified Euro year, repeat the learned procedure) and 1,000 confirmation replications (fixed development domain/settings, grouped nuisance refits). Save every attempt, seed, status and failure. Keep finite extreme/diagnostics-poor draws; only computational/structural failures are failures. Report percentile 95% intervals and cluster influence diagnostics. LOMO uses actual refitting, with fixed-domain confirmation; add team-removal sensitivity. Separate 200 joint development/confirmation domain-refit draws from fixed-domain inference.

Reproduce the legacy DoWhy failure against exact artifacts where available, private appendix only. Version 0.14 preserves a propensity_score column when randomizing treatment; rebuild fresh inputs/nuisances. Compare direct custom estimator nulls, global and within-match randomized treatment, and independent random signed dummy outcomes, 200 replications each. Distinguish fixed-frozen-domain from rebuilt treatment-dependent support nulls. These are software checks, not evidence of exchangeability. Use a pre-action negative-control outcome only with an explicit substantive justification; otherwise record its absence.

## Hidden-confounding sensitivity

Implement ATT general-outcome DVDS (Dorn/Guo/Kallus, Appendix A), not a binary shortcut for {-1,0,+1}. Gamma={1,1.25,1.5,2,3,5,10}, tipping search tolerance .01. beta=1/(Gamma+1); q+ at 1-beta, q- at beta from three-class grouped models. h+=q+ + max(y-q+,0)/beta; h-=q- + min(y-q-,0)/beta. Fit tail conditional expectations from actual observed training labels with quantile-stage separation. t=y/Gamma+(1-1/Gamma)h; rho=primary m0/Gamma+(1-1/Gamma)E[h|X,Z=0]. theta0=[sum treated rho + sum controls e/(1-e)*(t-rho)]/n_treated; lower ATT=treated mean-theta0+, upper=treated mean-theta0-. Gamma=1 must exactly reproduce primary AIPW. Do not clip transformed tail responses. Test against oracle three-point linear programs with density ratios in [1/Gamma,Gamma], including ties/degenerate atoms/fractional boundaries. Endpoint cluster uncertainty is separate from identification bounds. Benchmark omitted measured groups cautiously. The published iid theory does not automatically prove clustered ternary coverage; state and simulate that limitation.

## Controlled simulations, practical findings and exports

Use actual notebook functions, realistic imbalance/rare signed goals, nonlinear contexts, match clustering, variable overlap/visibility. Twenty-four base scenarios: three effect signs x two overlaps x four nuisance correctness settings, plus visibility/quantile-boundary stresses. Target 200 datasets/scenario, ground truth for the selected treated population. Report bias, RMSE, coverage, false-positive rates, retention, ESS, failures and Monte Carlo error. Six prespecified learned-procedure sign/overlap cases x200 datasets x1,000 bootstrap draws evaluate the actual CI procedure as long resumable jobs. Do not substitute analytic-only toy code for production functions or claim unrun results.

Subgroups: distance <=18/>18 m; goal opening angle <=30/>30 degrees; nearest defender <=1.5/>1.5 m. Six strata with simultaneous bootstrap intervals, counts/goals/matches/ESS/balance/coverage. Minimum 50 shots, ESS50, ten matches; otherwise insufficient. Context/visible lane exploratory. Supported/unsupported visual cases with comparable donors, plus a blinded rating template (no analyst contacting without authority). Avoid individual decision mistake claims.

Generate tables/figures from saved results, cohort flow, features, support, ATT contribution/uncertainty, nulls, sensitivity, simulations, subgroup/case outputs, environment/config/license/provenance and public/private release manifest. State signed first-goal reward per100 supported decisions, not EPV or an unconditional scoring-probability change. No automatic publication.

## Logging, execution and acceptance

Execution phases: fixture/unit tests; smoke six development matches with ten bootstraps and small simulations; full development; null/influence/sensitivity/simulation validation; immutable design lock; confirmation; exports. Protect confirmation with explicit programmatic gates. Default four workers with one numerical thread each, no nested parallelism. Start/run remotely in tmux after implementation validation. Save an executed notebook matching its parameters and output provenance.

Structured JSONL events include UTC time, run/stage/fold/replicate/attempt IDs, seed, counts, elapsed time, success/failure, exception traceback. Human stdout log, atomic status.json/heartbeats, stage progress/ETA after measured pilot, per-worker logs or master aggregation, atomic resumable checkpoints and explicit completion/blocked/scientific-failure states. Never call a launched job complete.

Initial study budget 8 GB, free-space floor 10 GB. Avoid duplicate raw caches, per-replicate fitted models/large row predictions and millions of tiny files. Preserve existing data. Freeze compatible environment without blindly upgrading the existing venv.

Acceptance tests cover UUID/coordinate geometry; dedup/goals/own-goals/equal-time/period end/unknown windows; strict outer/inner/preprocess/calibration/both-arm provenance; bootstrap original IDs and distinct donors; fixed-domain invariance to evaluation treatment; row ATT decomposition/asymmetric-support counterexample; fresh placebo propensity; Gamma=1/oracle bounds; identical comparator population; hashes/cache invalidation/resume seeds; locked confirmation access; baseline simulations and expected old-code failures. Tests import the exact notebook-generated scientific module.
