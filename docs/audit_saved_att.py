"""Read-only row-contribution reconstruction from a saved nuisance table.

Usage: python -B audit_saved_att.py outcome_nuisance_rows.parquet [EXPECTED_ATT]
Needs pandas/numpy and a Parquet engine. Does not fit models or write any files.
This verifies saved-score arithmetic, not causal identification or cross-fitting.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd


def contributions(rows: pd.DataFrame, outcome: str = "Y_reward_15") -> pd.DataFrame:
    rows = rows.copy()
    required = ["Z", outcome, "e_hat", "mu0_hat"]
    if rows[required].isna().any().any():
        raise ValueError("Saved analysis rows contain missing estimator inputs")
    if not rows["Z"].isin([0, 1]).all():
        raise ValueError("Treatment is not binary")
    if not rows["e_hat"].between(0, 1, inclusive="neither").all():
        raise ValueError("Propensities must be strictly between zero and one")
    n1 = int(rows["Z"].sum())
    if n1 == 0:
        raise ValueError("No treated rows")
    rows["audit_residual"] = rows[outcome] - rows["mu0_hat"]
    rows["audit_odds_weight"] = rows["e_hat"] / (1 - rows["e_hat"])
    rows["audit_ATT_numerator"] = np.where(
        rows["Z"].eq(1), rows["audit_residual"],
        -rows["audit_odds_weight"] * rows["audit_residual"],
    )
    rows["audit_ATT_contribution"] = rows["audit_ATT_numerator"] / n1
    return rows


def main(path: str, expected: float | None = None) -> int:
    rows = contributions(pd.read_parquet(path))
    shots = rows.loc[rows["Z"].eq(1)]
    controls = rows.loc[rows["Z"].eq(0)]
    weights = controls["audit_odds_weight"]
    n1 = len(shots)
    residual_sum = float((weights * controls["audit_residual"]).sum())
    shot_mean = float(shots["Y_reward_15"].mean())
    predicted_continuation_mean = float(shots["mu0_hat"].mean())
    att = float(rows["audit_ATT_contribution"].sum())
    high_propensity = controls.loc[controls["e_hat"].ge(0.95)]
    result = {
        "interpretation": "Reconstructs the existing L1 export from saved predictions; no nuisance refit or causal-validity claim.",
        "n_rows": len(rows), "n_treated": n1, "n_controls": len(controls),
        "ATT": att, "expected_ATT": expected,
        "matches_expected": None if expected is None else bool(np.isclose(att, expected)),
        "observed_treated_mean": shot_mean,
        "mu0_on_treated_mean": predicted_continuation_mean,
        "weighted_control_residual_sum": residual_sum,
        "control_residual_correction_divided_by_treated_count": residual_sum / n1,
        "control_weight_sum": float(weights.sum()),
        "normalized_weighted_control_residual_mean": residual_sum / float(weights.sum()),
        "control_ESS": float(weights.sum() ** 2 / weights.pow(2).sum()),
        "maximum_control_weight": float(weights.max()),
        "saved_w_att_matches_odds": bool(np.allclose(controls["w_att"], weights)) if "w_att" in rows else None,
        "saved_mu0_residual_matches": bool(np.allclose(controls["mu0_residual"], controls["audit_residual"])) if "mu0_residual" in rows else None,
        "controls_outside_shot_propensity_gate": {
            "threshold": 0.95, "n": len(high_propensity),
            "ATT_contribution_sum": float(high_propensity["audit_ATT_contribution"].sum()),
            "weight_sum_fraction": float(high_propensity["audit_odds_weight"].sum() / weights.sum()),
        },
        "prediction_ranges": {
            "mu0_controls_min": float(controls["mu0_hat"].min()),
            "mu0_controls_max": float(controls["mu0_hat"].max()),
            "mu0_treated_min": float(shots["mu0_hat"].min()),
            "mu0_treated_max": float(shots["mu0_hat"].max()),
        },
        "formula": "sum[Z*(Y-mu0) - (1-Z)*e/(1-e)*(Y-mu0)] / sum(Z)",
        "row_calculation": "contributions() returns every saved row with audit_residual, audit_odds_weight, audit_ATT_numerator and audit_ATT_contribution; no rows are silently omitted.",
    }
    print(json.dumps(result, indent=2))
    return int(expected is not None and not np.isclose(att, expected))


if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        raise SystemExit("Usage: audit_saved_att.py SAVED_PARQUET [EXPECTED_ATT]")
    raise SystemExit(main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) == 3 else None))
