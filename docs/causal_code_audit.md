# Audit of the current causal workflow

Reviewed September 30, 2026. This is a read-only diagnosis, not a corrected empirical run. The remote experiment, data, notebooks and installed packages were not changed. Three agents reviewed inference/support, calibration/library behavior, and public-data migration. Small probes ran in Vulcano's existing environment using scripts sent through SSH stdin; local audit files are the only additions.

## Evidence and configuration

Latest remote notebook: `results/models/support_aware_causal_pipeline_demo-v6-workflow copy.ipynb`, SHA-256 `7516c5a59f9459e04dcb7ad618c05da6d23b1c163105414195897c72561fff97`. Relevant source fingerprints:

- `src/CausalSupport/analysis.py`: `9c7f0c4a0b0234d7e0f6645f6c2cdbeb7fd550ea4059a042b0d4194ca49eaca4`.
- `src/CausalSupport/propensity.py`: `1f6dcce3495d4ba7a3586ebb824ea5c919d1e30be21170b3e3b4db9280fe0587`.
- `src/Pass/reward_labels.py`: `0720c0993afe60158aa76a86b055715e57a98ee4fb71a9529f72d9a79a5b0d3d`.

Environment: Python 3.10, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.6.1, DoWhy 0.14. Source fingerprints were checked against Vulcano during this audit. Notebook cells below mean JSON cell indices, not execution counts.

The user's quoted +0.4439 ATT, ESS 63.7 and maximum weight 1,228.7 belong to the saved **L1** export under `results/models/demo_outputs/v6_workflow_refactor/`. The latest notebook copy is an **L2** run with ATT +0.043582, ESS 541.61 and maximum weight 108.86. They must not be merged into one specification. The saved L1 common-overlap branch has 5,575 versus 5,806 target shots: **96.02% retention**, ESS **1,728.23**, and maximum weight **18.82**. These figures were verified in the exported weight table, not inferred from the prose report. They are not guaranteed for a new public cohort.

## 1. Primary support is incompatible with the usual propensity-correct ATT argument

Location: notebook `apply_att_continuation_support_filter` (cell 38), with downstream `aipw_att` (cell 52).

Shot targets must have continuation probability >0.05, equivalent to shot propensity <0.95. Control donors need only finite propensity, mechanical eligibility and a target tactical cell. The propensity gate does **not** restrict their covariate states. Thus an e=0.99 control can enter with odds weight 99, although a shot in that same state would fail the target gate.

This is more than weight concentration. Standard supported ATT augmentation needs control residuals representing the same covariate-defined target domain, or a derived selection/transport correction. An accurate outcome model could still make extra donor residuals vanish in expectation, but a correct original propensity alone does not establish ordinary double robustness for this asymmetric selected target. This inference is based on the actual support and score algebra; it is not a measurement of the true empirical bias. ATT-specific AIPW theory is reviewed by [Shu and Tan](https://arxiv.org/abs/1808.01408).

Executable counterexample in `audit_current_causal_workflow.py`: two covariate strata have e=.2 and .99; both have exactly zero treatment effect, with respective constant potential outcomes 0 and 1. Using expected treatment counts and a deliberately incorrect constant-zero outcome model, the existing support/ATT functions return **−4.95**, rather than the known supported ATT of zero. Applying the target propensity domain to both arms returns zero. This isolates failure of the propensity-correct branch; it is not a real-data estimate.

Required change: define an honest common covariate domain for supported treated targets and augmented donor residuals, or derive and validate explicit selection/transport weights. A weight cap or donor subsample cannot silently be substituted while preserving the original estimand and double-robustness claim.

### Reconstruction of the actual +0.443859 estimate

`audit_saved_att.py` reads the saved L1 nuisance table without fitting anything. It reconstructs all 58,123 row contributions and reproduces the exported ATT:

| Component | Value |
|---|---:|
| Treated count / donor count | 5,806 / 52,317 |
| Observed treated reward mean | 0.1298656562 |
| Mean predicted continuation reward at treated states | 0.0594593167 |
| Donor weighted residual sum | −2,168.2660808 |
| Residual correction divided by treated count | −0.3734526491 |
| AIPW ATT | 0.4438589886 |
| Donor weight sum / ESS / maximum | 23,244.91 / 63.65 / 1,228.67 |
| Donors with e≥.95 | 82 |
| Their fraction of donor weight | 68.84% |
| Their ATT contribution, using existing saved predictions | +0.3597778378 |

The 82 rows are about 0.16% of donors. Their contribution is approximately 81% of the reported ATT, but it is a decomposition of the current score, **not** the estimated change after a proper support/model refit. Outcome predictions at these controls deserve particular scrutiny.

## 2. Opposite-action outcome predictions are not fully match-held out

Locations: notebook `_cross_fit_arm_predictions`, `cross_fit_mu0_predictions`, `cross_fit_mu1_predictions` (cell 52); duplicated routines in `src/CausalSupport/analysis.py` near lines 108 and 167.

Each model fits one treatment arm. Own-arm predictions hold out a match, but opposite-arm predictions average every fold model, including models trained on other rows from that opposite-arm decision's match. In the four-match fixture, each arm produces 16 opposite-arm model predictions; **12 include the predicted decision's match**, affecting all four opposite-arm decisions. Own-arm predictions pass this membership check.

Required change: establish one outer match-fold registry across both arms. For each held-out fold, fit both outcome models and propensity models outside its matches, then predict both potential-action outcomes only for that fold. Pipeline preprocessing, inner tuning, calibration and upstream learned features need the same exclusion guarantees. Export training-match provenance. Reduce grouped folds or fail explicitly when groups/arms are insufficient; do not fall back silently to row CV or full-sample predictions.

## 3. Bootstrap copies leak across nuisance folds

Location: notebook `make_bootstrap_cluster_sample` and `full_pipeline_cluster_bootstrap_aipw` (cell 66).

Each bootstrap draw receives a distinct match label. Copies of the same original match can therefore enter different cross-fitting groups. With draws `[0,0,1,2]`, original match 0 occurs in both train and validation in two folds. This is reproduced using the actual sample-construction helper.

Required change: keep draw IDs for resampling bookkeeping, but group all copies by `original_match_id` for nuisance splitting. Bootstrap refitting currently **does** recompute the primary support rule; the issue is not a proven support-rule mismatch. Declare whether inference targets a fixed supported domain or the data-adaptive support-selection procedure. Hard threshold/caliper selection can be nonregular; investigate coverage in simulation rather than assuming 1,000 draws guarantees validity. The bootstrap point estimate also references global `analysis_df` instead of exclusively the supplied `base_df`; make data and target explicit.

The saved L1 export has only 50 successful draws, ATT SE 16.34 and interval approximately [−0.0062,52.09]. The stable ATC interval targets another population and cannot replace a primary ATT interval.

## 4. ATT arithmetic passes; normalization must be documented

The implemented score and residual decomposition agree:

```text
n1 = sum(Z)
ATT = sum[Z*(Y-mu0) - (1-Z)*e/(1-e)*(Y-mu0)] / n1
```

On a four-row fixture, per-row contributions sum to 1.05, equal to both helper outputs. The saved empirical reconstruction also matches. Dividing the donor residual sum by its weight sum gives a different finite-sample, normalized estimator; it is not an automatic arithmetic repair. Current IPW uses normalized weighted means, while AIPW's residual correction uses treated count. State both definitions and export numerator, denominator and per-row contributions. The nuisance diagnostic's normalized residual mean −0.09328 and AIPW correction −0.37345 are compatible because total donor weight is about four times treated count.

The manuscript formula was not supplied during this review, so manuscript-to-code consistency remains a required verification step. Arithmetic consistency does not cure support mismatch, misspecification or leakage.

## 5. A DoWhy placebo failure mechanism is reproduced

Installed source chain:

1. `CausalModel.estimate_effect` lets the weighting estimator add `propensity_score` to its mutable data.
2. `refute_estimate` reuses that data.
3. `placebo_treatment_refuter._refute_once` assigns new placebo treatment but preserves the original score column.
4. `PropensityScoreEstimator.fit` and the weighting estimator's prediction path skip score fitting/prediction if the column already exists.

`audit_dowhy_placebo.py` calls the installed helper on deterministic synthetic data. With true effect 2, original ATT is **2.130396**. A random placebo with retained original scores gives **−2.042934**; the identical placebo draw after removing scores and refitting gives **0.071862**. This confirms the mechanism, not its complete contribution to the football placebo failure. The script prints exact installed source paths and lines.

Required change: construct fresh clean estimator inputs/objects for every perturbation; refit the propensity for the perturbed treatment and outcome models where appropriate. Random-common-cause and subset refuters also preserve the score column, so audit their claimed refitting. DoWhy estimates its own propensity on the selected sample; it is not testing the custom cross-fitted score automatically. Distinguish a frozen-domain estimator check from an end-to-end null experiment rebuilding treatment-dependent support. Run corrected null checks on the current matching configuration and public implementation. The earlier near-zero L1/common-overlap manual placebos do not validate the latest L2 or revised public analysis.

## 6. Other inference and comparator gaps

| Path | Verified behavior | Required adaptation |
|---|---|---|
| Main logistic propensity | Fold pipeline fits preprocessing inside the ordinary grouped outer fold: correct. Small-group fallback uses row splits/in-sample prediction. | Fail closed or reduce valid grouped folds; test every fallback. |
| Richer HGB propensity | Ordinary outer folds are grouped. Calibration helper can fall back to row splitting or in-sample calibration. Automatic early stopping above 10,000 rows uses internal row splitting in installed sklearn. | Group inner calibration/tuning. Disable automatic row-based early stopping and select iteration count through grouped validation. This is not evidence of outer-test leakage. |
| EconML DRLearner | Input preprocessing is fitted on the entire selected sample; integer CV is row-based, with no match groups passed. | Put preprocessing into nuisance pipelines and supply explicit shared match splits. Its fitted-effect SD is not an ATT standard error. |
| ActionSelection comparator | Validation labels inform model selection/temperature calibration and binary calibration, but validation and test are both marked strict OOF. Scores also cover training rows. | For causal use, regenerate complete outer-fold predictions. Test-only claims require full upstream provenance. Omit PFF artifacts from the first public run. |
| Upstream optional xG | Scaler fits before row-based GridSearchCV; artifact metadata lacks training-match lineage. | Repair nesting/provenance if comparator retained. This dependency is not needed for the primary logistic/Ridge reward experiment. |
| Model/estimator comparisons | Different support masks and, in some ATT/ATC comparisons, different populations. | Compare identical target/row IDs first; report reselected support separately. |
| Football calipers | Keeps qualifying shots and union of controls, then applies original propensity weights. Reuse counts do not enter estimation; the matching comparator performs different nearest-neighbor matching. | Call this a caliper-restricted sensitivity branch, not an implemented caliper-matched ATT. Use actual pair/reuse weights for matching, or derive a support-compatible augmented estimator. |
| Empirical support counts | Use the full sample, including same-match donors. | Learn donor rules in an honest design/training process and audit external-match donor quality. This is a design gap, not proof of outcome leakage. |

## 7. Outcome and public-data timing gaps

`src/Pass/reward_labels.py` implements a **signed first-goal** reward, not EPV or a goal-count outcome. Its processed branch uses same-period search, allows the current scoring shot, excludes equal-time goals for other actions, and returns zero when scoring-team identity is unresolved. The public adapter should mark unresolved attribution missing, deduplicate own-goal records, resolve equal clocks using native event order, and validate real goal timing and end-of-period observation.

UUIDs cannot pass through the current numeric `_normalize_id` unchanged. Public carry endpoints/duration and carry `under_pressure` may be post-action. Anonymous 360 snapshots do not justify fabricated velocities, persistent player identities, complete team counts or arbitrary-time EPV states. Visibility selection must be audited separately from treatment overlap and unmeasured confounding. See the [migration assessment](statsbomb_causal_migration_assessment.md) for provider details and all 12 adapted requirements.

## Reproducing these checks

The scripts only read data/source and print results. They do not execute the whole notebook. Spy regressors in the notebook probe expose actual fit/predict membership without fitting expensive nuisance models; support and ATT helpers are executed as extracted. Expected failing design checks make the first script exit 1. This is a deliberately red diagnostic baseline for the future implementation, not a failed audit run.

From this local project, with an interactive SSH connection configured:

```sh
rtk proxy ssh -F /home/dtidigital/.ssh/config -o ClearAllForwardings=yes Vulcano 'cd causal/Simplified_EPV_in_PFF_Data && .venv/bin/python -B - "results/models/support_aware_causal_pipeline_demo-v6-workflow copy.ipynb"' < docs/audit_current_causal_workflow.py
rtk proxy ssh -F /home/dtidigital/.ssh/config -o ClearAllForwardings=yes Vulcano 'cd causal/Simplified_EPV_in_PFF_Data && .venv/bin/python -B - results/models/demo_outputs/v6_workflow_refactor/outcome_nuisance_rows.parquet 0.44385898859270523' < docs/audit_saved_att.py
rtk proxy ssh -F /home/dtidigital/.ssh/config -o ClearAllForwardings=yes Vulcano 'cd causal/Simplified_EPV_in_PFF_Data && .venv/bin/python -B -' < docs/audit_dowhy_placebo.py
```

The saved [audit results](causal_code_audit_results.json) contain concise observed outputs and source fingerprint. Future corrected tests should add common-fold preprocessing/calibration provenance, bootstrap membership, shared target domain, null checks, reward boundary fixtures and simulation coverage. These probes do not quantify the corrected football effect; a clean complete rerun remains necessary.
