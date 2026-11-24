"""Read-only reproduction of stale propensity scores in DoWhy's placebo refuter.

Uses synthetic data only. Prints diagnostics to stdout; writes no files and does
not change installed packages. This reproduces an implementation mechanism,
not the cause or magnitude of any real-data placebo failure.

On Vulcano, run with the project's .venv/bin/python. It also runs from stdin,
so there is no need to copy this script to the remote machine.
"""

from __future__ import annotations

import copy
import inspect
import json

import dowhy
import networkx as nx
import numpy as np
import pandas as pd
from dowhy import CausalModel
from dowhy.causal_estimators.propensity_score_estimator import PropensityScoreEstimator
from dowhy.causal_estimators.propensity_score_weighting_estimator import (
    PropensityScoreWeightingEstimator,
)
from dowhy.causal_refuters.placebo_treatment_refuter import _refute_once


DATA_SEED = 12
PLACEBO_SEED = 17
N_ROWS = 2000
TRUE_EFFECT = 2.0


def source_evidence(function, tokens: tuple[str, ...]) -> dict:
    """Report exact installed source path, start line, and relevant statements."""
    source_lines, first_line = inspect.getsourcelines(function)
    return {
        "function": function.__qualname__,
        "installed_path": inspect.getfile(function),
        "first_line": first_line,
        "selected_lines": [
            {"line": first_line + offset, "text": line.strip()}
            for offset, line in enumerate(source_lines)
            if any(token in line for token in tokens)
        ],
    }


def main() -> None:
    print("SYNTHETIC MECHANISM REPRODUCTION ONLY; no real-data result is tested.")
    print(json.dumps({
        "dowhy_version": dowhy.__version__,
        "data_seed": DATA_SEED,
        "placebo_seed": PLACEBO_SEED,
        "n_rows": N_ROWS,
        "true_effect": TRUE_EFFECT,
    }, indent=2))

    evidence = [
        source_evidence(CausalModel.estimate_effect, ("self._data,",)),
        source_evidence(CausalModel.refute_estimate, ("refuter_class(self._data",)),
        source_evidence(PropensityScoreEstimator.fit, (
            "self.propensity_score_column not in data", "propensity_score_model.fit",
        )),
        source_evidence(PropensityScoreWeightingEstimator.estimate_effect, (
            "self.propensity_score_column not in data",
            "self.estimate_propensity_score_column(data)",
        )),
        source_evidence(_refute_once, (
            "np.random.randint(", "data.assign(placebo=", "new_estimator.fit(",
        )),
    ]
    print("Installed source provenance:")
    print(json.dumps(evidence, indent=2))

    rng = np.random.default_rng(DATA_SEED)
    x = rng.normal(size=N_ROWS)
    treatment = rng.binomial(1, 1 / (1 + np.exp(-2 * x)))
    outcome = x + TRUE_EFFECT * treatment + rng.normal(scale=0.2, size=N_ROWS)
    data = pd.DataFrame({"x": x, "Z": treatment, "Y": outcome})
    graph = nx.DiGraph([("x", "Z"), ("x", "Y"), ("Z", "Y")])
    model = CausalModel(data=data, treatment="Z", outcome="Y", graph=graph)
    estimand = model.identify_effect()
    original = model.estimate_effect(
        estimand,
        method_name="backdoor.propensity_score_weighting",
        target_units="att",
    )
    score_column = original.estimator.propensity_score_column
    retained_scores = score_column in model._data.columns
    print(json.dumps({
        "generated_score_column": score_column,
        "model_data_retains_generated_scores": retained_scores,
    }, indent=2))
    if not retained_scores:
        raise RuntimeError("Installed DoWhy does not retain scores: mechanism changed.")

    placebo_estimand = copy.deepcopy(estimand)
    placebo_estimand.treatment_variable = ["placebo"]

    # DoWhy 0.14's default integer-treatment branch uses NumPy's global RNG.
    # Calling the installed refuter helper tests its actual data-preserving path.
    np.random.seed(PLACEBO_SEED)
    stale_effect = _refute_once(
        model._data,
        placebo_estimand,
        original,
        ["Z"],
        dict(model._data.dtypes),
    )

    # Recreate exactly the same placebo draw, removing the generated score column
    # so the installed estimator must fit scores for the new treatment.
    np.random.seed(PLACEBO_SEED)
    placebo = np.random.randint(
        low=int(model._data["Z"].min()),
        high=int(model._data["Z"].max()) + 1,
        size=N_ROWS,
    )
    clean_data = model._data.drop(columns=[score_column]).assign(placebo=placebo)
    clean_estimator = original.estimator.get_new_estimator_object(placebo_estimand)
    clean_estimator.fit(
        clean_data,
        effect_modifier_names=original.estimator._effect_modifier_names,
    )
    clean_estimate = clean_estimator.estimate_effect(
        clean_data,
        control_value=original.control_value,
        treatment_value=original.treatment_value,
        target_units="att",
    )
    print("Synthetic estimates, using the identical random placebo draw:")
    print(json.dumps({
        "original_att": float(original.value),
        "placebo_with_retained_original_scores": float(stale_effect),
        "placebo_after_score_removal_and_refit": float(clean_estimate.value),
    }, indent=2))
    print(
        "A large stale-score placebo effect demonstrates this mechanism only. "
        "Diagnosing the real-data failure requires its own corrected rerun."
    )


if __name__ == "__main__":
    main()
