"""Read-only, small-fixture probes of functions extracted from the current notebook.

Run with numpy, pandas and sklearn installed:
    python -B audit_current_causal_workflow.py /path/to/notebook.ipynb

Or send this script to the existing Vulcano Python environment over SSH stdin.
No notebook cell is executed wholesale. Estimator spies record the actual helper's
fit/predict membership; they intentionally replace regression fitting, not folds.
Exit 1 means a requested design requirement is violated, not that the audit crashed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


def extract_functions(notebook: dict) -> dict[str, str]:
    functions = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = ast.get_source_segment(source, node)
    return functions


def main(notebook_path: str) -> int:
    notebook_bytes = Path(notebook_path).read_bytes()
    functions = extract_functions(json.loads(notebook_bytes))
    ns = {
        "np": np, "pd": pd, "GroupKFold": GroupKFold, "KFold": KFold,
        "RANDOM_SEED": 42,
        "MODEL_NUMERIC_FEATURES": ["audit_match_id", "x"],
        "MODEL_CATEGORICAL_FEATURES": [],
        "PRIMARY_ATT_SUPPORT_COL": "supported_att_continuation",
        "PRIMARY_SUPPORT_VARIANT": "att_continuation",
        "PRIMARY_OUTCOME": "Y_reward_15",
        "MECHANICAL_ELIGIBILITY_LEVEL": "L2",
        "MECHANICAL_ELIGIBILITY_LEVELS": {"L2": {"label": "audit_fixture"}},
        "SUPPORT_TAU": 0.05,
        "EMPIRICAL_SUPPORT_COLS": ["tactical_cell"],
        "MIN_CONTINUATION_DONORS_FOR_ATT": 1,
        "ATT_CONTINUATION_SUPPORT_COL": "supported_att_continuation",
        "ATT_CONTINUATION_TARGET_COL": "target_att_shot_att_continuation",
        "ATT_CONTINUATION_DONOR_COL": "control_donor_att_continuation",
        "ATT_CONTINUATION_DROP_REASON_COL": "support_drop_reason_att_continuation",
    }
    for name in ["existing_columns", "_cell_key", "implied_support_max_e"]:
        exec(functions[name], ns)

    prediction_log = []

    class MembershipSpy:
        def fit(self, X, y):
            self.training_matches = set(X["audit_match_id"])
            return self

        def predict(self, X):
            for row_id, row in X.iterrows():
                prediction_log.append({
                    "row_id": int(row_id),
                    "match_id": int(row["audit_match_id"]),
                    "training_matches": sorted(self.training_matches),
                    "same_match_in_training": bool(
                        row["audit_match_id"] in self.training_matches
                    ),
                })
            return np.zeros(len(X))

    ns["make_mu0_pipeline"] = lambda *args, **kwargs: MembershipSpy()
    ns["clone"] = lambda estimator: MembershipSpy()
    exec(functions["_cross_fit_arm_predictions"], ns)
    rows = [
        {"match_id": m, "audit_match_id": m, "x": float(m),
         "Z": arm, "Y_reward_15": 0.0}
        for m in range(4) for arm in [0, 1]
    ]
    fixture = pd.DataFrame(rows)
    crossfit_details = []
    for fit_arm, predict_arm in [(0, 1), (1, 0)]:
        prediction_log.clear()
        ns["_cross_fit_arm_predictions"](
            fixture, "Y_reward_15", fit_arm, predict_arm, max_splits=4
        )
        opposite = [r for r in prediction_log if fixture.at[r["row_id"], "Z"] == predict_arm]
        own = [r for r in prediction_log if fixture.at[r["row_id"], "Z"] == fit_arm]
        crossfit_details.append({
            "fit_arm": fit_arm,
            "opposite_arm_model_predictions": len(opposite),
            "opposite_arm_same_match_predictions": sum(r["same_match_in_training"] for r in opposite),
            "affected_opposite_arm_decisions": len({r["row_id"] for r in opposite if r["same_match_in_training"]}),
            "own_arm_same_match_predictions": sum(r["same_match_in_training"] for r in own),
            "prediction_membership": list(prediction_log),
        })

    exec(functions["make_bootstrap_cluster_sample"], ns)
    boot = ns["make_bootstrap_cluster_sample"](
        fixture, sampled_clusters=np.array([0, 0, 1, 2])
    )
    bootstrap_leaks = []
    for fold, (train, test) in enumerate(GroupKFold(4).split(boot, groups=boot["match_id"])):
        overlap = sorted(set(boot.iloc[train]["original_match_id"]) & set(boot.iloc[test]["original_match_id"]))
        if overlap:
            bootstrap_leaks.append({"fold": fold, "original_matches_in_both": overlap})

    # Prebuilt tactical cells isolate the actual support rule from bin construction.
    ns["add_support_audit_bins"] = lambda df: df.copy()
    ns["ensure_common_overlap_aliases"] = lambda df: df.copy()
    exec(functions["apply_att_continuation_support_filter"], ns)
    support_fixture = pd.DataFrame({
        "Z": [1, 0], "E0": [1, 1], "e_hat": [0.2, 0.99],
        "p_continue_hat": [0.8, 0.01], "tactical_cell": ["same", "same"],
    })
    support = ns["apply_att_continuation_support_filter"](support_fixture)
    control_retained = bool(support.at[1, "control_donor_att_continuation"])
    support_detail = {
        "treated_propensity": 0.2, "donor_propensity": 0.99,
        "donor_retained": control_retained,
        "donor_odds_weight": float(0.99 / (1 - 0.99)),
        "threshold_implied_weight_bound_if_applied_to_donors": 19.0,
    }

    exec(functions["get_analysis_sample"], ns)
    arithmetic_fixture = pd.DataFrame({
        "Z": [1, 1, 0, 0], "E0": [1] * 4,
        "supported_att_continuation": [1] * 4,
        "e_hat": [0.5, 0.5, 0.8, 0.8],
        "Y_reward_15": [1.0, 0.0, 0.0, 0.0],
    })
    mu0 = pd.Series([0.25, 0.25, 0.2, 0.2], index=arithmetic_fixture.index)
    ns["cross_fit_mu0_predictions"] = lambda sample, *args, **kwargs: (
        mu0.loc[sample.index], {"cv_folds": 4, "outcome_model": "fixed_audit_predictions"}
    )
    exec(functions["aipw_att"], ns)
    exec(functions["aipw_att_decomposition"], ns)
    computed = float(ns["aipw_att"](arithmetic_fixture, "Y_reward_15"))
    decomposition = ns["aipw_att_decomposition"](arithmetic_fixture).iloc[0]
    n1 = int(arithmetic_fixture["Z"].sum())
    contributions = []
    for i, row in arithmetic_fixture.iterrows():
        weight = float(row["e_hat"] / (1 - row["e_hat"]))
        value = (
            (row["Y_reward_15"] - mu0.at[i]) / n1
            if row["Z"] == 1 else
            -weight * (row["Y_reward_15"] - mu0.at[i]) / n1
        )
        contributions.append({"row_id": i, "Z": int(row["Z"]), "ATT_contribution": float(value)})
    normalized_residual_att = float(0.5 - 0.25 - (-0.2))

    # Same tactical cell remains supported even if nearest donor is the same match.
    self_donor_fixture = support_fixture.copy()
    self_donor_fixture["e_hat"] = [0.2, 0.3]
    self_donor_fixture["p_continue_hat"] = [0.8, 0.7]
    self_donor_fixture["match_id"] = ["same_match", "same_match"]
    self_donor = ns["apply_att_continuation_support_filter"](self_donor_fixture)

    # Correct assignment probabilities do not rescue asymmetric target/donor
    # restrictions. Exact expected cell counts make sampling noise irrelevant:
    # low stratum: e=.2, Y(0)=Y(1)=0; high: e=.99, Y(0)=Y(1)=1.
    # The target contains only low-stratum shots, but the high-stratum control
    # remains a donor. Deliberately misspecify m0=0 to test the propensity branch
    # of the ordinary double-robustness argument, not outcome-model accuracy.
    target_fixture = pd.DataFrame([
        {"Z": arm, "E0": 1, "e_hat": propensity,
         "p_continue_hat": 1 - propensity, "tactical_cell": "same",
         "Y_reward_15": outcome}
        for propensity, outcome, treated, controls in [
            (0.2, 0.0, 20, 80), (0.99, 1.0, 99, 1)
        ]
        for arm, count in [(1, treated), (0, controls)]
        for _ in range(count)
    ])
    target_fixture = ns["apply_att_continuation_support_filter"](target_fixture)
    ns["cross_fit_mu0_predictions"] = lambda sample, *args, **kwargs: (
        pd.Series(0.0, index=sample.index),
        {"cv_folds": 4, "outcome_model": "misspecified_zero_fixture"},
    )
    asymmetric_att = float(ns["aipw_att"](target_fixture, "Y_reward_15"))
    shared_fixture = target_fixture.copy()
    shared_fixture.loc[shared_fixture["e_hat"] >= 0.95,
                       "supported_att_continuation"] = 0
    shared_att = float(ns["aipw_att"](shared_fixture, "Y_reward_15"))

    results = {
        "notebook_sha256": hashlib.sha256(notebook_bytes).hexdigest(),
        "crossfit_match_exclusion": {
            "status": "FAIL" if any(d["opposite_arm_same_match_predictions"] for d in crossfit_details) else "PASS",
            "details": crossfit_details,
        },
        "bootstrap_original_match_exclusion": {
            "status": "FAIL" if bootstrap_leaks else "PASS", "leaking_folds": bootstrap_leaks,
        },
        "primary_donor_weight_bound": {
            "status": "FAIL" if control_retained else "PASS", **support_detail,
            "interpretation": "Current design permits this; a bounded supported primary must change it.",
        },
        "empirical_support_outside_match": {
            "status": "FAIL" if bool(self_donor.at[0, "target_att_shot_att_continuation"]) else "PASS",
            "interpretation": "Support counts include same-match donors; this is not cross-fitted support.",
        },
        "support_compatible_propensity_correction": {
            "status": "PASS" if np.isclose(asymmetric_att, 0.0) else "FAIL",
            "known_supported_target_ATT": 0.0,
            "current_asymmetric_support_ATT": asymmetric_att,
            "shared_domain_support_ATT": shared_att,
            "propensity_model": "exact assignment probabilities",
            "outcome_model": "deliberately misspecified constant zero",
            "interpretation": "A zero-effect expected-count fixture demonstrates that out-of-target donor residuals invalidate the ordinary propensity-correct branch for this supported target. This is a structural counterexample, not an estimate of empirical bias.",
        },
        "aipw_formula_and_decomposition": {
            "status": "PASS" if np.isclose(computed, 1.05) and np.isclose(computed, decomposition["AIPW_ATT"]) and np.isclose(computed, sum(r["ATT_contribution"] for r in contributions)) else "FAIL",
            "ATT": computed, "decomposition_ATT": float(decomposition["AIPW_ATT"]),
            "control_weight_sum": 8.0, "treated_count": n1,
            "residual_denominator": "treated_count",
            "normalized_residual_alternative_ATT": normalized_residual_att,
            "row_contributions": contributions,
            "interpretation": "Arithmetic agrees. Count and weight-sum normalization differ in finite samples; neither discrepancy alone establishes a coding error.",
        },
    }
    print(json.dumps(results, indent=2))
    return int(any(v.get("status") == "FAIL" for v in results.values() if isinstance(v, dict)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: audit_current_causal_workflow.py NOTEBOOK.ipynb")
    raise SystemExit(main(sys.argv[1]))
