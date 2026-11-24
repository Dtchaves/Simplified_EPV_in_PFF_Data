from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .propensity import (
    SHOT_REGION_SCOPES,
    add_baseline_propensity_source_columns,
    add_support_audit_bins,
    apply_support_filter_for_source,
    calibration_bin_table,
    cross_fit_richer_binary_propensity_model,
    existing_columns,
    high_propensity_region_table,
    implied_support_max_e,
    make_preprocessor,
    propensity_calibration_summary,
    score_action_selection_propensity_source,
    set_default_support_aliases,
)


def get_analysis_sample(
    df: pd.DataFrame,
    outcome: str,
    propensity_col: str = "e_hat",
    support_col: str = "supported",
    supported_only: bool = True,
) -> pd.DataFrame:
    mask = df[support_col].eq(1) if supported_only else df["E0"].eq(1)
    sample = df.loc[mask].copy()
    propensity = pd.to_numeric(sample.get(propensity_col), errors="coerce")
    sample = sample.loc[
        sample[outcome].notna()
        & sample["Z"].isin([0, 1])
        & propensity.notna()
        & np.isfinite(propensity)
    ].copy()
    sample["Z"] = sample["Z"].astype(int)
    sample[propensity_col] = propensity.loc[sample.index].astype(float)
    return sample


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w))


def weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    mean_value = weighted_mean(x, w)
    return float(np.sum(w * (x - mean_value) ** 2) / np.sum(w))


def effective_sample_size(weights: Any) -> float:
    arr = np.asarray(weights, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0 or np.sum(arr ** 2) == 0:
        return np.nan
    return float(arr.sum() ** 2 / np.sum(arr ** 2))


def att_weights_from_propensity(
    sample: pd.DataFrame,
    propensity_col: str = "e_hat",
    weight_cap: Optional[float] = None,
) -> pd.Series:
    controls = sample.loc[sample["Z"].eq(0)].copy()
    weights = pd.to_numeric(controls[propensity_col], errors="coerce") / (1 - pd.to_numeric(controls[propensity_col], errors="coerce"))
    weights = weights.replace([np.inf, -np.inf], np.nan)
    if weight_cap is not None:
        weights = np.minimum(weights, float(weight_cap))
    return pd.Series(weights, index=controls.index, dtype=float)


def atc_weights_from_propensity(
    sample: pd.DataFrame,
    propensity_col: str = "e_hat",
    weight_cap: Optional[float] = None,
) -> pd.Series:
    treated = sample.loc[sample["Z"].eq(1)].copy()
    propensity = pd.to_numeric(treated[propensity_col], errors="coerce")
    weights = (1 - propensity) / propensity
    weights = weights.replace([np.inf, -np.inf], np.nan)
    if weight_cap is not None:
        weights = np.minimum(weights, float(weight_cap))
    return pd.Series(weights, index=treated.index, dtype=float)


def make_mu0_pipeline(numeric_features: list[str], categorical_features: list[str], alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", make_preprocessor(numeric_features, categorical_features)),
            ("reg", Ridge(alpha=alpha)),
        ]
    )


def cross_fit_mu0_predictions(
    sample: pd.DataFrame,
    outcome: str,
    numeric_features: list[str],
    categorical_features: list[str],
    random_seed: int = 42,
    max_splits: int = 5,
) -> tuple[pd.Series, dict[str, Any]]:
    sample = sample.copy()
    numeric_features = existing_columns(sample, numeric_features)
    categorical_features = existing_columns(sample, categorical_features)
    feature_cols = numeric_features + categorical_features
    controls = sample[sample["Z"] == 0].dropna(subset=[outcome]).copy()
    treated = sample[sample["Z"] == 1].copy()
    mu0_hat = pd.Series(index=sample.index, dtype=float)

    if len(controls) == 0 or not feature_cols:
        return mu0_hat, {"cv_folds": 0, "n_controls": int(len(controls)), "outcome_model": "unavailable"}

    cv = int(min(max_splits, len(controls)))
    base_model = make_mu0_pipeline(numeric_features, categorical_features)

    if cv >= 2:
        treated_fold_predictions: list[np.ndarray] = []
        if "match_id" in controls.columns and controls["match_id"].nunique(dropna=True) >= cv:
            splitter = GroupKFold(n_splits=cv)
            split_iter = splitter.split(controls, groups=controls["match_id"])
            model_kind = "cross_fit_oof_controls_match_grouped_foldavg_treated"
        else:
            splitter = KFold(n_splits=cv, shuffle=True, random_state=random_seed)
            split_iter = splitter.split(controls)
            model_kind = "cross_fit_oof_controls_foldavg_treated"
        for train_pos, test_pos in split_iter:
            train = controls.iloc[train_pos]
            test = controls.iloc[test_pos]
            model = clone(base_model)
            model.fit(train[feature_cols].replace([np.inf, -np.inf], np.nan), train[outcome])
            mu0_hat.loc[test.index] = model.predict(test[feature_cols].replace([np.inf, -np.inf], np.nan))
            if len(treated) > 0:
                treated_fold_predictions.append(model.predict(treated[feature_cols].replace([np.inf, -np.inf], np.nan)))
        if treated_fold_predictions:
            mu0_hat.loc[treated.index] = np.vstack(treated_fold_predictions).mean(axis=0)
    else:
        model_kind = "in_sample_fallback"

    full_model = clone(base_model)
    full_model.fit(controls[feature_cols].replace([np.inf, -np.inf], np.nan), controls[outcome])
    missing = mu0_hat.isna()
    if missing.any():
        mu0_hat.loc[missing] = full_model.predict(sample.loc[missing, feature_cols].replace([np.inf, -np.inf], np.nan))

    return mu0_hat, {
        "cv_folds": int(cv),
        "n_controls": int(len(controls)),
        "n_treated": int(len(treated)),
        "outcome_model": model_kind,
    }


def cross_fit_mu1_predictions(
    sample: pd.DataFrame,
    outcome: str,
    numeric_features: list[str],
    categorical_features: list[str],
    random_seed: int = 42,
    max_splits: int = 5,
) -> tuple[pd.Series, dict[str, Any]]:
    sample = sample.copy()
    numeric_features = existing_columns(sample, numeric_features)
    categorical_features = existing_columns(sample, categorical_features)
    feature_cols = numeric_features + categorical_features
    treated = sample[sample["Z"] == 1].dropna(subset=[outcome]).copy()
    controls = sample[sample["Z"] == 0].copy()
    mu1_hat = pd.Series(index=sample.index, dtype=float)

    if len(treated) == 0 or not feature_cols:
        return mu1_hat, {"cv_folds": 0, "n_treated": int(len(treated)), "outcome_model": "unavailable"}

    cv = int(min(max_splits, len(treated)))
    base_model = make_mu0_pipeline(numeric_features, categorical_features)

    if cv >= 2:
        control_fold_predictions: list[np.ndarray] = []
        if "match_id" in treated.columns and treated["match_id"].nunique(dropna=True) >= cv:
            splitter = GroupKFold(n_splits=cv)
            split_iter = splitter.split(treated, groups=treated["match_id"])
            model_kind = "cross_fit_oof_shots_match_grouped_foldavg_controls"
        else:
            splitter = KFold(n_splits=cv, shuffle=True, random_state=random_seed)
            split_iter = splitter.split(treated)
            model_kind = "cross_fit_oof_shots_foldavg_controls"
        for train_pos, test_pos in split_iter:
            train = treated.iloc[train_pos]
            test = treated.iloc[test_pos]
            model = clone(base_model)
            model.fit(train[feature_cols].replace([np.inf, -np.inf], np.nan), train[outcome])
            mu1_hat.loc[test.index] = model.predict(test[feature_cols].replace([np.inf, -np.inf], np.nan))
            if len(controls) > 0:
                control_fold_predictions.append(model.predict(controls[feature_cols].replace([np.inf, -np.inf], np.nan)))
        if control_fold_predictions:
            mu1_hat.loc[controls.index] = np.vstack(control_fold_predictions).mean(axis=0)
    else:
        model_kind = "in_sample_fallback"

    full_model = clone(base_model)
    full_model.fit(treated[feature_cols].replace([np.inf, -np.inf], np.nan), treated[outcome])
    missing = mu1_hat.isna()
    if missing.any():
        mu1_hat.loc[missing] = full_model.predict(sample.loc[missing, feature_cols].replace([np.inf, -np.inf], np.nan))

    return mu1_hat, {
        "cv_folds": int(cv),
        "n_controls": int(len(controls)),
        "n_treated": int(len(treated)),
        "outcome_model": model_kind,
    }


def raw_difference_att_generic(
    df: pd.DataFrame,
    outcome: str,
    propensity_col: str = "e_hat",
    support_col: str = "supported",
) -> float:
    sample = get_analysis_sample(df, outcome, propensity_col=propensity_col, support_col=support_col)
    treated = sample[sample["Z"] == 1]
    controls = sample[sample["Z"] == 0]
    if len(treated) == 0 or len(controls) == 0:
        return np.nan
    return float(treated[outcome].mean() - controls[outcome].mean())


def raw_difference_atc_generic(
    df: pd.DataFrame,
    outcome: str,
    propensity_col: str = "e_hat",
    support_col: str = "supported",
) -> float:
    return raw_difference_att_generic(df, outcome, propensity_col=propensity_col, support_col=support_col)


def ipw_att_with_weights(
    df: pd.DataFrame,
    outcome: str,
    propensity_col: str = "e_hat",
    support_col: str = "supported",
    weight_cap: Optional[float] = None,
) -> float:
    sample = get_analysis_sample(df, outcome, propensity_col=propensity_col, support_col=support_col)
    treated = sample[sample["Z"] == 1]
    controls = sample[sample["Z"] == 0].copy()
    if len(treated) == 0 or len(controls) == 0:
        return np.nan
    weights = att_weights_from_propensity(sample, propensity_col=propensity_col, weight_cap=weight_cap).loc[controls.index]
    valid = weights.notna() & np.isfinite(weights)
    if valid.sum() == 0:
        return np.nan
    y0_weighted = np.average(controls.loc[valid, outcome], weights=weights.loc[valid])
    return float(treated[outcome].mean() - y0_weighted)


def ipw_atc_with_weights(
    df: pd.DataFrame,
    outcome: str,
    propensity_col: str = "e_hat",
    support_col: str = "supported",
    weight_cap: Optional[float] = None,
) -> float:
    sample = get_analysis_sample(df, outcome, propensity_col=propensity_col, support_col=support_col)
    treated = sample[sample["Z"] == 1].copy()
    controls = sample[sample["Z"] == 0]
    if len(treated) == 0 or len(controls) == 0:
        return np.nan
    weights = atc_weights_from_propensity(sample, propensity_col=propensity_col, weight_cap=weight_cap).loc[treated.index]
    valid = weights.notna() & np.isfinite(weights)
    if valid.sum() == 0:
        return np.nan
    y1_weighted = np.average(treated.loc[valid, outcome], weights=weights.loc[valid])
    return float(y1_weighted - controls[outcome].mean())


def outcome_regression_atc_with_weights(
    df: pd.DataFrame,
    outcome: str,
    numeric_features: list[str],
    categorical_features: list[str],
    propensity_col: str = "e_hat",
    support_col: str = "supported",
    random_seed: int = 42,
) -> float:
    sample = get_analysis_sample(df, outcome, propensity_col=propensity_col, support_col=support_col)
    treated = sample[sample["Z"] == 1]
    controls = sample[sample["Z"] == 0].copy()
    if len(treated) == 0 or len(controls) == 0:
        return np.nan
    mu1_hat, _ = cross_fit_mu1_predictions(
        sample,
        outcome=outcome,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_seed=random_seed,
    )
    controls["mu1_hat"] = mu1_hat.loc[controls.index]
    return float(controls["mu1_hat"].mean() - controls[outcome].mean())


def aipw_att_with_weights(
    df: pd.DataFrame,
    outcome: str,
    numeric_features: list[str],
    categorical_features: list[str],
    propensity_col: str = "e_hat",
    support_col: str = "supported",
    weight_cap: Optional[float] = None,
    random_seed: int = 42,
) -> float:
    sample = get_analysis_sample(df, outcome, propensity_col=propensity_col, support_col=support_col)
    treated = sample[sample["Z"] == 1].copy()
    controls = sample[sample["Z"] == 0].copy()
    n_treated = len(treated)
    if n_treated == 0 or len(controls) == 0:
        return np.nan

    mu0_hat, _ = cross_fit_mu0_predictions(
        sample,
        outcome=outcome,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_seed=random_seed,
    )
    treated["mu0_hat"] = mu0_hat.loc[treated.index]
    controls["mu0_hat"] = mu0_hat.loc[controls.index]
    weights = att_weights_from_propensity(sample, propensity_col=propensity_col, weight_cap=weight_cap).loc[controls.index]
    valid = weights.notna() & np.isfinite(weights)
    if valid.sum() == 0:
        return np.nan
    y1_hat = float(treated[outcome].mean())
    y0_hat_for_treated = (
        treated["mu0_hat"].sum()
        + (weights.loc[valid] * (controls.loc[valid, outcome] - controls.loc[valid, "mu0_hat"])).sum()
    ) / n_treated
    return float(y1_hat - y0_hat_for_treated)


def aipw_atc_with_weights(
    df: pd.DataFrame,
    outcome: str,
    numeric_features: list[str],
    categorical_features: list[str],
    propensity_col: str = "e_hat",
    support_col: str = "supported",
    weight_cap: Optional[float] = None,
    random_seed: int = 42,
) -> float:
    sample = get_analysis_sample(df, outcome, propensity_col=propensity_col, support_col=support_col)
    treated = sample[sample["Z"] == 1].copy()
    controls = sample[sample["Z"] == 0].copy()
    n_controls = len(controls)
    if len(treated) == 0 or n_controls == 0:
        return np.nan

    mu1_hat, _ = cross_fit_mu1_predictions(
        sample,
        outcome=outcome,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_seed=random_seed,
    )
    treated["mu1_hat"] = mu1_hat.loc[treated.index]
    controls["mu1_hat"] = mu1_hat.loc[controls.index]
    weights = atc_weights_from_propensity(sample, propensity_col=propensity_col, weight_cap=weight_cap).loc[treated.index]
    valid = weights.notna() & np.isfinite(weights)
    if valid.sum() == 0:
        return np.nan
    y1_hat_for_controls = (
        controls["mu1_hat"].sum()
        + (weights.loc[valid] * (treated.loc[valid, outcome] - treated.loc[valid, "mu1_hat"])).sum()
    ) / n_controls
    y0_hat = float(controls[outcome].mean())
    return float(y1_hat_for_controls - y0_hat)


def smd_table_with_weights(
    df: pd.DataFrame,
    features: list[str],
    outcome: str,
    propensity_col: str = "e_hat",
    support_col: str = "supported",
    weight_cap: Optional[float] = None,
) -> pd.DataFrame:
    sample = get_analysis_sample(df, outcome, propensity_col=propensity_col, support_col=support_col)
    treated = sample[sample["Z"] == 1]
    controls = sample[sample["Z"] == 0].copy()
    if treated.empty or controls.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "mean_shot",
                "mean_control",
                "mean_control_ATT_weighted",
                "SMD_unweighted",
                "SMD_ATT_weighted",
                "abs_SMD_unweighted",
                "abs_SMD_ATT_weighted",
                "weight_cap",
            ]
        )

    weights = att_weights_from_propensity(sample, propensity_col=propensity_col, weight_cap=weight_cap).loc[controls.index]
    rows = []
    for feature in existing_columns(sample, features):
        x1 = pd.to_numeric(treated[feature], errors="coerce").astype(float).to_numpy()
        x0 = pd.to_numeric(controls[feature], errors="coerce").astype(float).to_numpy()
        w0 = weights.to_numpy(dtype=float)
        ok1 = np.isfinite(x1)
        ok0 = np.isfinite(x0) & np.isfinite(w0)
        x1 = x1[ok1]
        x0 = x0[ok0]
        w0 = w0[ok0]
        if len(x1) < 2 or len(x0) < 2 or np.sum(w0) <= 0:
            continue
        m1 = float(x1.mean())
        m0 = float(x0.mean())
        v1 = float(x1.var(ddof=1))
        v0 = float(x0.var(ddof=1))
        pooled = np.sqrt((v1 + v0) / 2)
        smd_unweighted = (m1 - m0) / pooled if pooled > 0 else np.nan
        m0w = weighted_mean(x0, w0)
        v0w = weighted_var(x0, w0)
        pooled_w = np.sqrt((v1 + v0w) / 2)
        smd_weighted = (m1 - m0w) / pooled_w if pooled_w > 0 else np.nan
        rows.append(
            {
                "feature": feature,
                "mean_shot": m1,
                "mean_control": m0,
                "mean_control_ATT_weighted": m0w,
                "SMD_unweighted": smd_unweighted,
                "SMD_ATT_weighted": smd_weighted,
                "abs_SMD_unweighted": abs(smd_unweighted),
                "abs_SMD_ATT_weighted": abs(smd_weighted),
                "weight_cap": weight_cap,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("abs_SMD_ATT_weighted", ascending=False).reset_index(drop=True)


def smd_table_atc_with_weights(
    df: pd.DataFrame,
    features: list[str],
    outcome: str,
    propensity_col: str = "e_hat",
    support_col: str = "supported",
    weight_cap: Optional[float] = None,
) -> pd.DataFrame:
    sample = get_analysis_sample(df, outcome, propensity_col=propensity_col, support_col=support_col)
    treated = sample[sample["Z"] == 1].copy()
    controls = sample[sample["Z"] == 0]
    if treated.empty or controls.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "mean_shot",
                "mean_control",
                "mean_shot_ATC_weighted",
                "SMD_unweighted",
                "SMD_ATC_weighted",
                "abs_SMD_unweighted",
                "abs_SMD_ATC_weighted",
                "weight_cap",
            ]
        )

    weights = atc_weights_from_propensity(sample, propensity_col=propensity_col, weight_cap=weight_cap).loc[treated.index]
    rows = []
    for feature in existing_columns(sample, features):
        x1 = pd.to_numeric(treated[feature], errors="coerce").astype(float).to_numpy()
        x0 = pd.to_numeric(controls[feature], errors="coerce").astype(float).to_numpy()
        w1 = weights.to_numpy(dtype=float)
        ok1 = np.isfinite(x1) & np.isfinite(w1)
        ok0 = np.isfinite(x0)
        x1w = x1[ok1]
        w1 = w1[ok1]
        x1 = x1[np.isfinite(x1)]
        x0 = x0[ok0]
        if len(x1) < 2 or len(x0) < 2 or len(x1w) < 2 or np.sum(w1) <= 0:
            continue
        m1 = float(x1.mean())
        m0 = float(x0.mean())
        v1 = float(x1.var(ddof=1))
        v0 = float(x0.var(ddof=1))
        pooled = np.sqrt((v1 + v0) / 2)
        smd_unweighted = (m1 - m0) / pooled if pooled > 0 else np.nan
        m1w = weighted_mean(x1w, w1)
        v1w = weighted_var(x1w, w1)
        pooled_w = np.sqrt((v1w + v0) / 2)
        smd_weighted = (m1w - m0) / pooled_w if pooled_w > 0 else np.nan
        rows.append(
            {
                "feature": feature,
                "mean_shot": m1,
                "mean_control": m0,
                "mean_shot_ATC_weighted": m1w,
                "SMD_unweighted": smd_unweighted,
                "SMD_ATC_weighted": smd_weighted,
                "abs_SMD_unweighted": abs(smd_unweighted),
                "abs_SMD_ATC_weighted": abs(smd_weighted),
                "weight_cap": weight_cap,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("abs_SMD_ATC_weighted", ascending=False).reset_index(drop=True)


def weight_diagnostics_with_weights(
    df: pd.DataFrame,
    outcome: str,
    propensity_col: str = "e_hat",
    support_col: str = "supported",
    weight_cap: Optional[float] = None,
    run_label: str = "primary",
    tau_support: Optional[float] = None,
    min_e_support: Optional[float] = None,
    propensity_source: Optional[str] = None,
) -> pd.DataFrame:
    sample = get_analysis_sample(df, outcome, propensity_col=propensity_col, support_col=support_col)
    treated = sample[sample["Z"] == 1]
    controls = sample[sample["Z"] == 0].copy()
    weights = att_weights_from_propensity(sample, propensity_col=propensity_col, weight_cap=weight_cap).to_numpy(dtype=float) if len(controls) else np.array([], dtype=float)
    weights = weights[np.isfinite(weights)]
    atc_weights = atc_weights_from_propensity(sample, propensity_col=propensity_col, weight_cap=weight_cap).to_numpy(dtype=float) if len(treated) else np.array([], dtype=float)
    atc_weights = atc_weights[np.isfinite(atc_weights)]
    quantiles = np.quantile(weights, [0.90, 0.95, 0.99]) if len(weights) else [np.nan, np.nan, np.nan]
    atc_quantiles = np.quantile(atc_weights, [0.90, 0.95, 0.99]) if len(atc_weights) else [np.nan, np.nan, np.nan]
    return pd.DataFrame(
        [
            {
                "run_label": run_label,
                "propensity_source": propensity_source,
                "outcome": outcome,
                "tau_support": tau_support,
                "min_e_support": min_e_support,
                "weight_cap": weight_cap,
                "n_supported": int(len(sample)),
                "n_supported_shots": int(len(treated)),
                "n_supported_controls": int(len(controls)),
                "sum_weights_controls": float(np.sum(weights)) if len(weights) else np.nan,
                "mean_weight_controls": float(np.mean(weights)) if len(weights) else np.nan,
                "max_weight_controls": float(np.max(weights)) if len(weights) else np.nan,
                "p90_weight_controls": float(quantiles[0]),
                "p95_weight_controls": float(quantiles[1]),
                "p99_weight_controls": float(quantiles[2]),
                "ess_controls": effective_sample_size(weights),
                "sum_weights_shots_atc": float(np.sum(atc_weights)) if len(atc_weights) else np.nan,
                "mean_weight_shots_atc": float(np.mean(atc_weights)) if len(atc_weights) else np.nan,
                "max_weight_shots_atc": float(np.max(atc_weights)) if len(atc_weights) else np.nan,
                "p90_weight_shots_atc": float(atc_quantiles[0]),
                "p95_weight_shots_atc": float(atc_quantiles[1]),
                "p99_weight_shots_atc": float(atc_quantiles[2]),
                "ess_shots_atc": effective_sample_size(atc_weights),
            }
        ]
    )


def matching_att_generic(
    df: pd.DataFrame,
    outcome: str,
    numeric_features: list[str],
    propensity_col: str = "e_hat",
    support_col: str = "supported",
    k: int = 3,
) -> float:
    numeric_features = existing_columns(df, numeric_features)
    sample = get_analysis_sample(df, outcome, propensity_col=propensity_col, support_col=support_col)
    treated = sample[sample["Z"] == 1].copy()
    controls = sample[sample["Z"] == 0].copy()
    if len(treated) == 0 or len(controls) == 0 or not numeric_features:
        return np.nan

    x_controls_raw = controls[numeric_features].replace([np.inf, -np.inf], np.nan)
    x_treated_raw = treated[numeric_features].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_controls = scaler.fit_transform(imputer.fit_transform(x_controls_raw))
    x_treated = scaler.transform(imputer.transform(x_treated_raw))
    nn = NearestNeighbors(n_neighbors=min(k, len(controls)), metric="euclidean")
    nn.fit(x_controls)
    _, idx = nn.kneighbors(x_treated)
    matched_control_y = controls.iloc[idx.ravel()][outcome].to_numpy().reshape(len(treated), -1).mean(axis=1)
    return float(np.mean(treated[outcome].to_numpy() - matched_control_y))


def matching_atc_generic(
    df: pd.DataFrame,
    outcome: str,
    numeric_features: list[str],
    propensity_col: str = "e_hat",
    support_col: str = "supported",
    k: int = 3,
) -> float:
    numeric_features = existing_columns(df, numeric_features)
    sample = get_analysis_sample(df, outcome, propensity_col=propensity_col, support_col=support_col)
    treated = sample[sample["Z"] == 1].copy()
    controls = sample[sample["Z"] == 0].copy()
    if len(treated) == 0 or len(controls) == 0 or not numeric_features:
        return np.nan

    x_treated_raw = treated[numeric_features].replace([np.inf, -np.inf], np.nan)
    x_controls_raw = controls[numeric_features].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_treated = scaler.fit_transform(imputer.fit_transform(x_treated_raw))
    x_controls = scaler.transform(imputer.transform(x_controls_raw))
    nn = NearestNeighbors(n_neighbors=min(k, len(treated)), metric="euclidean")
    nn.fit(x_treated)
    _, idx = nn.kneighbors(x_controls)
    matched_treated_y = treated.iloc[idx.ravel()][outcome].to_numpy().reshape(len(controls), -1).mean(axis=1)
    return float(np.mean(matched_treated_y - controls[outcome].to_numpy()))


def _resolve_horizon_outcomes(
    df: pd.DataFrame,
    primary_outcome: str,
    horizons: tuple[int, ...] = (5, 10, 15),
) -> list[tuple[Optional[int], str]]:
    rows: list[tuple[Optional[int], str]] = []
    for horizon in horizons:
        for candidate in (f"Y_epv_{horizon}", f"Y_reward_{horizon}"):
            if candidate in df.columns and df[candidate].notna().any():
                rows.append((horizon, candidate))
                break
    if not rows:
        rows = [(None, primary_outcome)]
    elif primary_outcome not in [outcome for _, outcome in rows] and primary_outcome in df.columns and df[primary_outcome].notna().any():
        rows.append((None, primary_outcome))
    return rows


def _support_variant_label(source_name: str, min_e: float, tau: float, default_min_e: float, default_tau: float) -> str:
    if float(min_e) == float(default_min_e) and float(tau) == float(default_tau):
        return f"{source_name}|default"
    return f"{source_name}|min_e={min_e:.2f}|tau={tau:.2f}"


def build_propensity_diagnostics_outputs(
    df: pd.DataFrame,
    source_registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    bin_frames: list[pd.DataFrame] = []
    high_frames: list[pd.DataFrame] = []

    for _, source_row in source_registry.iterrows():
        source_name = str(source_row["propensity_source"])
        propensity_col = str(source_row["active_propensity_col"])
        supported_col = f"supported_{source_name}"
        drop_reason_col = f"support_drop_reason_{source_name}"
        base_mask = df["E0"].eq(1) & df["Z"].isin([0, 1]) & pd.to_numeric(df.get(propensity_col), errors="coerce").notna()
        supported_mask = base_mask & df.get(supported_col, pd.Series(0, index=df.index)).eq(1)

        summary_rows.append(
            propensity_calibration_summary(
                df,
                y_col="Z",
                p_col=propensity_col,
                source_name=source_name,
                scope_mask=base_mask,
                scope_label="E0_all",
            )
        )
        summary_rows.append(
            propensity_calibration_summary(
                df,
                y_col="Z",
                p_col=propensity_col,
                source_name=source_name,
                scope_mask=supported_mask,
                scope_label="supported_default",
            )
        )
        high_frame = high_propensity_region_table(
            df,
            y_col="Z",
            p_col=propensity_col,
            source_name=source_name,
            scope_mask=base_mask,
        )
        if not high_frame.empty:
            high_frames.append(high_frame)
            summary_rows.extend(high_frame.to_dict(orient="records"))

        for scope_label, scope_mask in {"E0_all": base_mask, "supported_default": supported_mask}.items():
            scoped_df = df.loc[scope_mask].copy()
            for strategy in ("uniform", "quantile"):
                bins = calibration_bin_table(
                    scoped_df,
                    y_col="Z",
                    p_col=propensity_col,
                    scope_label=scope_label,
                    source_name=source_name,
                    n_bins=15,
                    strategy=strategy,
                    supported_col=supported_col,
                    drop_reason_col=drop_reason_col,
                )
                if not bins.empty:
                    bin_frames.append(bins)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.drop_duplicates(subset=["propensity_source", "scope"], keep="last").reset_index(drop=True)
    bins_df = pd.concat(bin_frames, ignore_index=True) if bin_frames else pd.DataFrame()
    high_df = pd.concat(high_frames, ignore_index=True) if high_frames else pd.DataFrame()
    return summary_df, bins_df, high_df


def build_support_grid_sensitivity(
    df: pd.DataFrame,
    source_registry: pd.DataFrame,
    primary_outcome: str,
    numeric_features: list[str],
    categorical_features: list[str],
    empirical_support_cols: list[str],
    min_empirical_support_per_arm: int,
    add_outcomes_fn: Callable[[pd.DataFrame], pd.DataFrame],
    recompute_epv_fn: Callable[[pd.DataFrame], pd.DataFrame],
    default_min_e: float,
    default_tau: float,
    min_e_grid: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10),
    tau_grid: tuple[float, ...] = (0.05, 0.10, 0.15),
    random_seed: int = 42,
) -> pd.DataFrame:
    preview = recompute_epv_fn(add_outcomes_fn(df.copy()))
    outcomes = _resolve_horizon_outcomes(preview, primary_outcome=primary_outcome)
    rows: list[dict[str, Any]] = []
    e0_mask = df["E0"].eq(1) & df["Z"].isin([0, 1])

    for _, source_row in source_registry.iterrows():
        source_name = str(source_row["propensity_source"])
        propensity_col = str(source_row["active_propensity_col"])
        if propensity_col not in df.columns:
            continue
        n_e0_shots = int((e0_mask & df["Z"].eq(1)).sum())
        n_e0_controls = int((e0_mask & df["Z"].eq(0)).sum())

        for min_e in min_e_grid:
            for tau in tau_grid:
                variant = apply_support_filter_for_source(
                    df,
                    propensity_col=propensity_col,
                    source_name=source_name,
                    tau=tau,
                    min_e=min_e,
                    empirical_support_cols=empirical_support_cols,
                    min_empirical_support_per_arm=min_empirical_support_per_arm,
                )
                variant = recompute_epv_fn(add_outcomes_fn(variant))
                support_col = f"supported_{source_name}"
                drop_col = f"support_drop_reason_{source_name}"
                variant_label = _support_variant_label(source_name, min_e, tau, default_min_e=default_min_e, default_tau=default_tau)

                for horizon_seconds, outcome in outcomes:
                    sample = get_analysis_sample(variant, outcome, propensity_col=propensity_col, support_col=support_col)
                    n_treated = int((sample["Z"] == 1).sum())
                    n_controls = int((sample["Z"] == 0).sum())
                    ipw = ipw_att_with_weights(
                        variant,
                        outcome=outcome,
                        propensity_col=propensity_col,
                        support_col=support_col,
                    )
                    aipw = aipw_att_with_weights(
                        variant,
                        outcome=outcome,
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                        propensity_col=propensity_col,
                        support_col=support_col,
                        random_seed=random_seed,
                    )
                    ipw_atc = ipw_atc_with_weights(
                        variant,
                        outcome=outcome,
                        propensity_col=propensity_col,
                        support_col=support_col,
                    )
                    atc_or = outcome_regression_atc_with_weights(
                        variant,
                        outcome=outcome,
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                        propensity_col=propensity_col,
                        support_col=support_col,
                        random_seed=random_seed,
                    )
                    aipw_atc = aipw_atc_with_weights(
                        variant,
                        outcome=outcome,
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                        propensity_col=propensity_col,
                        support_col=support_col,
                        random_seed=random_seed,
                    )
                    diag = weight_diagnostics_with_weights(
                        variant,
                        outcome=outcome,
                        propensity_col=propensity_col,
                        support_col=support_col,
                        run_label=variant_label,
                        tau_support=tau,
                        min_e_support=min_e,
                        propensity_source=source_name,
                    )
                    smds = smd_table_with_weights(
                        variant,
                        features=numeric_features,
                        outcome=outcome,
                        propensity_col=propensity_col,
                        support_col=support_col,
                    )
                    smds_atc = smd_table_atc_with_weights(
                        variant,
                        features=numeric_features,
                        outcome=outcome,
                        propensity_col=propensity_col,
                        support_col=support_col,
                    )
                    rows.append(
                        {
                            "propensity_source": source_name,
                            "support_variant": variant_label,
                            "outcome": outcome,
                            "horizon_seconds": horizon_seconds,
                            "min_e": float(min_e),
                            "tau": float(tau),
                            "max_e_effective": implied_support_max_e(tau),
                            "n_E0_shots": n_e0_shots,
                            "n_E0_controls": n_e0_controls,
                            "n_supported_shots": n_treated,
                            "n_supported_controls": n_controls,
                            "treated_retention_pct": (100.0 * n_treated / n_e0_shots) if n_e0_shots else np.nan,
                            "control_retention_pct": (100.0 * n_controls / n_e0_controls) if n_e0_controls else np.nan,
                            "IPW_ATT": ipw,
                            "AIPW_ATT": aipw,
                            "IPW_ATC": ipw_atc,
                            "outcome_regression_ATC": atc_or,
                            "AIPW_ATC": aipw_atc,
                            "ess_controls": float(diag.loc[0, "ess_controls"]) if len(diag) else np.nan,
                            "ess_shots_atc": float(diag.loc[0, "ess_shots_atc"]) if len(diag) else np.nan,
                            "max_weight_controls": float(diag.loc[0, "max_weight_controls"]) if len(diag) else np.nan,
                            "max_weight_shots_atc": float(diag.loc[0, "max_weight_shots_atc"]) if len(diag) else np.nan,
                            "p95_weight_controls": float(diag.loc[0, "p95_weight_controls"]) if len(diag) else np.nan,
                            "p99_weight_controls": float(diag.loc[0, "p99_weight_controls"]) if len(diag) else np.nan,
                            "top_abs_weighted_smd": float(smds["abs_SMD_ATT_weighted"].max()) if not smds.empty else np.nan,
                            "top_abs_weighted_smd_atc": float(smds_atc["abs_SMD_ATC_weighted"].max()) if not smds_atc.empty else np.nan,
                            "support_drop_counts_json": json.dumps(
                                variant.loc[e0_mask, drop_col].value_counts(dropna=False).to_dict(),
                                sort_keys=True,
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def build_weight_cap_sensitivity(
    df: pd.DataFrame,
    source_registry: pd.DataFrame,
    primary_outcome: str,
    numeric_features: list[str],
    categorical_features: list[str],
    empirical_support_cols: list[str],
    min_empirical_support_per_arm: int,
    add_outcomes_fn: Callable[[pd.DataFrame], pd.DataFrame],
    recompute_epv_fn: Callable[[pd.DataFrame], pd.DataFrame],
    default_min_e: float,
    default_tau: float,
    min_e_grid: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10),
    tau_grid: tuple[float, ...] = (0.05, 0.10, 0.15),
    weight_cap_grid: tuple[Optional[float], ...] = (None, 10.0, 5.0),
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preview = recompute_epv_fn(add_outcomes_fn(df.copy()))
    outcomes = _resolve_horizon_outcomes(preview, primary_outcome=primary_outcome)
    rows: list[dict[str, Any]] = []
    top_smd_rows: list[dict[str, Any]] = []

    for _, source_row in source_registry.iterrows():
        source_name = str(source_row["propensity_source"])
        propensity_col = str(source_row["active_propensity_col"])
        if propensity_col not in df.columns:
            continue
        support_variants = list(dict.fromkeys(
            [(default_min_e, default_tau)] + [(min_e, tau) for min_e in min_e_grid for tau in tau_grid]
        ))

        for min_e, tau in support_variants:
            variant = apply_support_filter_for_source(
                df,
                propensity_col=propensity_col,
                source_name=source_name,
                tau=tau,
                min_e=min_e,
                empirical_support_cols=empirical_support_cols,
                min_empirical_support_per_arm=min_empirical_support_per_arm,
            )
            variant = recompute_epv_fn(add_outcomes_fn(variant))
            support_col = f"supported_{source_name}"
            variant_label = _support_variant_label(source_name, min_e, tau, default_min_e=default_min_e, default_tau=default_tau)
            for horizon_seconds, outcome in outcomes:
                for weight_cap in weight_cap_grid:
                    diag = weight_diagnostics_with_weights(
                        variant,
                        outcome=outcome,
                        propensity_col=propensity_col,
                        support_col=support_col,
                        weight_cap=weight_cap,
                        run_label=variant_label,
                        tau_support=tau,
                        min_e_support=min_e,
                        propensity_source=source_name,
                    )
                    smds = smd_table_with_weights(
                        variant,
                        features=numeric_features,
                        outcome=outcome,
                        propensity_col=propensity_col,
                        support_col=support_col,
                        weight_cap=weight_cap,
                    )
                    if not smds.empty:
                        top_slice = smds.head(10).copy()
                        top_slice["estimand"] = "ATT"
                        top_slice["propensity_source"] = source_name
                        top_slice["support_variant"] = variant_label
                        top_slice["outcome"] = outcome
                        top_slice["horizon_seconds"] = horizon_seconds
                        top_slice["weight_cap"] = weight_cap
                        top_smd_rows.extend(top_slice.to_dict(orient="records"))
                    smds_atc = smd_table_atc_with_weights(
                        variant,
                        features=numeric_features,
                        outcome=outcome,
                        propensity_col=propensity_col,
                        support_col=support_col,
                        weight_cap=weight_cap,
                    )
                    if not smds_atc.empty:
                        top_slice = smds_atc.head(10).copy()
                        top_slice["estimand"] = "ATC"
                        top_slice["propensity_source"] = source_name
                        top_slice["support_variant"] = variant_label
                        top_slice["outcome"] = outcome
                        top_slice["horizon_seconds"] = horizon_seconds
                        top_slice["weight_cap"] = weight_cap
                        top_smd_rows.extend(top_slice.to_dict(orient="records"))
                    rows.append(
                        {
                            "propensity_source": source_name,
                            "support_variant": variant_label,
                            "outcome": outcome,
                            "horizon_seconds": horizon_seconds,
                            "min_e": float(min_e),
                            "tau": float(tau),
                            "max_e_effective": implied_support_max_e(tau),
                            "weight_cap": weight_cap,
                            "IPW_ATT": ipw_att_with_weights(
                                variant,
                                outcome=outcome,
                                propensity_col=propensity_col,
                                support_col=support_col,
                                weight_cap=weight_cap,
                            ),
                            "AIPW_ATT": aipw_att_with_weights(
                                variant,
                                outcome=outcome,
                                numeric_features=numeric_features,
                                categorical_features=categorical_features,
                                propensity_col=propensity_col,
                                support_col=support_col,
                                weight_cap=weight_cap,
                                random_seed=random_seed,
                            ),
                            "IPW_ATC": ipw_atc_with_weights(
                                variant,
                                outcome=outcome,
                                propensity_col=propensity_col,
                                support_col=support_col,
                                weight_cap=weight_cap,
                            ),
                            "outcome_regression_ATC": outcome_regression_atc_with_weights(
                                variant,
                                outcome=outcome,
                                numeric_features=numeric_features,
                                categorical_features=categorical_features,
                                propensity_col=propensity_col,
                                support_col=support_col,
                                random_seed=random_seed,
                            ),
                            "AIPW_ATC": aipw_atc_with_weights(
                                variant,
                                outcome=outcome,
                                numeric_features=numeric_features,
                                categorical_features=categorical_features,
                                propensity_col=propensity_col,
                                support_col=support_col,
                                weight_cap=weight_cap,
                                random_seed=random_seed,
                            ),
                            "ess_controls": float(diag.loc[0, "ess_controls"]) if len(diag) else np.nan,
                            "ess_shots_atc": float(diag.loc[0, "ess_shots_atc"]) if len(diag) else np.nan,
                            "max_weight_controls": float(diag.loc[0, "max_weight_controls"]) if len(diag) else np.nan,
                            "max_weight_shots_atc": float(diag.loc[0, "max_weight_shots_atc"]) if len(diag) else np.nan,
                            "p95_weight_controls": float(diag.loc[0, "p95_weight_controls"]) if len(diag) else np.nan,
                            "p99_weight_controls": float(diag.loc[0, "p99_weight_controls"]) if len(diag) else np.nan,
                            "top_abs_weighted_smd": float(smds["abs_SMD_ATT_weighted"].max()) if not smds.empty else np.nan,
                            "top_abs_weighted_smd_atc": float(smds_atc["abs_SMD_ATC_weighted"].max()) if not smds_atc.empty else np.nan,
                            "n_supported_shots": int(diag.loc[0, "n_supported_shots"]) if len(diag) else 0,
                            "n_supported_controls": int(diag.loc[0, "n_supported_controls"]) if len(diag) else 0,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(top_smd_rows)


DISTANCE_BIN_ORDER = {"0-12": 0, "12-18": 1, "18-24": 2, "24-30": 3, "30+": 4, "missing": -1}


def build_football_caliper_branch(
    df: pd.DataFrame,
    primary_outcome: str,
    numeric_features: list[str],
    categorical_features: list[str],
    propensity_col: str = "e_hat_binary_logistic",
    support_col: str = "supported_binary_logistic",
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audited = add_support_audit_bins(df.copy())
    audited["football_caliper_supported"] = 0
    audited["football_caliper_match_count"] = 0
    audited["football_caliper_reason"] = np.where(audited[support_col].eq(1), "no_caliper_match", "not_in_default_support")

    sample = audited.loc[
        audited[support_col].eq(1)
        & audited["Z"].isin([0, 1])
        & pd.to_numeric(audited[propensity_col], errors="coerce").notna()
    ].copy()
    if sample.empty:
        return pd.DataFrame(), audited

    sample["distance_bin_ord"] = sample["distance_to_goal_bin"].map(DISTANCE_BIN_ORDER).fillna(-1).astype(int)
    sample["logit_propensity"] = np.log(
        np.clip(pd.to_numeric(sample[propensity_col], errors="coerce").to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        / (1 - np.clip(pd.to_numeric(sample[propensity_col], errors="coerce").to_numpy(dtype=float), 1e-6, 1 - 1e-6))
    )
    logit_sd = float(np.nanstd(sample["logit_propensity"], ddof=1))
    if not np.isfinite(logit_sd) or logit_sd <= 0:
        logit_sd = 1.0
    logit_caliper = 0.2 * logit_sd

    shots = sample.loc[sample["Z"].eq(1)].copy()
    controls = sample.loc[sample["Z"].eq(0)].copy()
    control_groups = {
        key: group.copy()
        for key, group in controls.groupby(["field_zone", "defender_density_bin"], dropna=False)
    }
    retained_shots: list[int] = []
    used_controls: list[int] = []
    control_use_count: dict[int, int] = {}
    shot_match_count: dict[int, int] = {}

    for shot_index, shot_row in shots.iterrows():
        key = (shot_row["field_zone"], shot_row["defender_density_bin"])
        group = control_groups.get(key)
        if group is None or group.empty:
            shot_match_count[shot_index] = 0
            continue
        eligible = group.loc[
            group["distance_bin_ord"].between(int(shot_row["distance_bin_ord"]) - 1, int(shot_row["distance_bin_ord"]) + 1)
            & (pd.to_numeric(group["distance_to_goal"], errors="coerce") - float(shot_row["distance_to_goal"])).abs().le(3.0)
            & (pd.to_numeric(group["logit_propensity"], errors="coerce") - float(shot_row["logit_propensity"])).abs().le(logit_caliper)
        ].copy()
        match_count = int(len(eligible))
        shot_match_count[shot_index] = match_count
        if match_count <= 0:
            continue
        retained_shots.append(shot_index)
        for control_index in eligible.index.tolist():
            used_controls.append(control_index)
            control_use_count[control_index] = control_use_count.get(control_index, 0) + 1

    if retained_shots:
        audited.loc[retained_shots, "football_caliper_supported"] = 1
        audited.loc[retained_shots, "football_caliper_reason"] = "retained_treated"
        audited.loc[retained_shots, "football_caliper_match_count"] = [shot_match_count.get(index, 0) for index in retained_shots]
    used_controls_unique = list(dict.fromkeys(used_controls))
    if used_controls_unique:
        audited.loc[used_controls_unique, "football_caliper_supported"] = 1
        audited.loc[used_controls_unique, "football_caliper_reason"] = "retained_control"
        audited.loc[used_controls_unique, "football_caliper_match_count"] = [control_use_count.get(index, 0) for index in used_controls_unique]

    outcomes = _resolve_horizon_outcomes(audited, primary_outcome=primary_outcome)
    smd_features = [
        "distance_to_goal",
        "ball_x",
        "in_box",
        "centrality",
        "angle_to_goal",
        "nearest_defender_distance",
        "goalkeeper_distance_to_goal_center",
        "teammates_in_box",
        "defenders_within_10m",
    ]
    result_rows: list[dict[str, Any]] = []
    for horizon_seconds, outcome in outcomes:
        diag = weight_diagnostics_with_weights(
            audited,
            outcome=outcome,
            propensity_col=propensity_col,
            support_col="football_caliper_supported",
            run_label="football_caliper",
            propensity_source="binary_logistic",
        )
        smds = smd_table_with_weights(
            audited,
            features=smd_features,
            outcome=outcome,
            propensity_col=propensity_col,
            support_col="football_caliper_supported",
        )
        smds_atc = smd_table_atc_with_weights(
            audited,
            features=smd_features,
            outcome=outcome,
            propensity_col=propensity_col,
            support_col="football_caliper_supported",
        )
        result_rows.append(
            {
                "propensity_source": "binary_logistic",
                "support_variant": "football_caliper",
                "outcome": outcome,
                "horizon_seconds": horizon_seconds,
                "raw_difference": raw_difference_att_generic(
                    audited,
                    outcome=outcome,
                    propensity_col=propensity_col,
                    support_col="football_caliper_supported",
                ),
                "IPW_ATT": ipw_att_with_weights(
                    audited,
                    outcome=outcome,
                    propensity_col=propensity_col,
                    support_col="football_caliper_supported",
                ),
                "AIPW_ATT": aipw_att_with_weights(
                    audited,
                    outcome=outcome,
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    propensity_col=propensity_col,
                    support_col="football_caliper_supported",
                    random_seed=random_seed,
                ),
                "matching_ATC": matching_atc_generic(
                    audited,
                    outcome=outcome,
                    numeric_features=numeric_features,
                    propensity_col=propensity_col,
                    support_col="football_caliper_supported",
                ),
                "IPW_ATC": ipw_atc_with_weights(
                    audited,
                    outcome=outcome,
                    propensity_col=propensity_col,
                    support_col="football_caliper_supported",
                ),
                "outcome_regression_ATC": outcome_regression_atc_with_weights(
                    audited,
                    outcome=outcome,
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    propensity_col=propensity_col,
                    support_col="football_caliper_supported",
                    random_seed=random_seed,
                ),
                "AIPW_ATC": aipw_atc_with_weights(
                    audited,
                    outcome=outcome,
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    propensity_col=propensity_col,
                    support_col="football_caliper_supported",
                    random_seed=random_seed,
                ),
                "ess_controls": float(diag.loc[0, "ess_controls"]) if len(diag) else np.nan,
                "ess_shots_atc": float(diag.loc[0, "ess_shots_atc"]) if len(diag) else np.nan,
                "max_weight_controls": float(diag.loc[0, "max_weight_controls"]) if len(diag) else np.nan,
                "max_weight_shots_atc": float(diag.loc[0, "max_weight_shots_atc"]) if len(diag) else np.nan,
                "p95_weight_controls": float(diag.loc[0, "p95_weight_controls"]) if len(diag) else np.nan,
                "p99_weight_controls": float(diag.loc[0, "p99_weight_controls"]) if len(diag) else np.nan,
                "top_abs_weighted_smd": float(smds["abs_SMD_ATT_weighted"].max()) if not smds.empty else np.nan,
                "top_abs_weighted_smd_atc": float(smds_atc["abs_SMD_ATC_weighted"].max()) if not smds_atc.empty else np.nan,
                "n_supported_shots": int(diag.loc[0, "n_supported_shots"]) if len(diag) else 0,
                "n_supported_controls": int(diag.loc[0, "n_supported_controls"]) if len(diag) else 0,
            }
        )
    row_columns = existing_columns(
        audited,
        [
            "decision_row_id",
            "match_id",
            "event_id",
            "Z",
            propensity_col,
            support_col,
            "football_caliper_supported",
            "football_caliper_match_count",
            "football_caliper_reason",
            "field_zone",
            "defender_density_bin",
            "distance_to_goal_bin",
            "distance_to_goal",
            "ball_x",
            "angle_to_goal",
            "nearest_defender_distance",
            "goalkeeper_distance_to_goal_center",
            "teammates_in_box",
            "defenders_within_10m",
        ],
    )
    rows_df = audited[row_columns].copy() if row_columns else pd.DataFrame()
    return pd.DataFrame(result_rows), rows_df


def build_support_loss_profile_by_source(
    df: pd.DataFrame,
    source_registry: pd.DataFrame,
    empirical_support_cols: list[str],
    min_empirical_support_per_arm: int,
    default_min_e: float,
    default_tau: float,
    min_e_grid: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10),
    tau_grid: tuple[float, ...] = (0.05, 0.10, 0.15),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["field_zone", "distance_to_goal_bin", "defender_density_bin"]

    for _, source_row in source_registry.iterrows():
        source_name = str(source_row["propensity_source"])
        propensity_col = str(source_row["active_propensity_col"])
        if propensity_col not in df.columns:
            continue
        support_variants = list(dict.fromkeys(
            [(default_min_e, default_tau)] + [(min_e, tau) for min_e in min_e_grid for tau in tau_grid]
        ))
        for min_e, tau in support_variants:
            variant = apply_support_filter_for_source(
                df,
                propensity_col=propensity_col,
                source_name=source_name,
                tau=tau,
                min_e=min_e,
                empirical_support_cols=empirical_support_cols,
                min_empirical_support_per_arm=min_empirical_support_per_arm,
            )
            support_col = f"supported_{source_name}"
            reason_col = f"support_drop_reason_{source_name}"
            eligible = variant.loc[variant["E0"].eq(1) & variant["Z"].isin([0, 1])].copy()
            if eligible.empty:
                continue
            for values, group in eligible.groupby(group_cols, dropna=False):
                if not isinstance(values, tuple):
                    values = (values,)
                reason_counts = group.loc[group[support_col].ne(1), reason_col].value_counts(dropna=False)
                rows.append(
                    {
                        "propensity_source": source_name,
                        "support_variant": _support_variant_label(source_name, min_e, tau, default_min_e=default_min_e, default_tau=default_tau),
                        "min_e": float(min_e),
                        "tau": float(tau),
                        "max_e_effective": implied_support_max_e(tau),
                        "field_zone": values[0],
                        "distance_to_goal_bin": values[1],
                        "defender_density_bin": values[2],
                        "eligible_shots": int(group["Z"].eq(1).sum()),
                        "supported_shots": int((group["Z"].eq(1) & group[support_col].eq(1)).sum()),
                        "dropped_shots": int((group["Z"].eq(1) & group[support_col].ne(1)).sum()),
                        "shot_retention_pct": 100.0 * float((group["Z"].eq(1) & group[support_col].eq(1)).sum()) / float(group["Z"].eq(1).sum()) if int(group["Z"].eq(1).sum()) else np.nan,
                        "eligible_controls": int(group["Z"].eq(0).sum()),
                        "supported_controls": int((group["Z"].eq(0) & group[support_col].eq(1)).sum()),
                        "dropped_controls": int((group["Z"].eq(0) & group[support_col].ne(1)).sum()),
                        "control_retention_pct": 100.0 * float((group["Z"].eq(0) & group[support_col].eq(1)).sum()) / float(group["Z"].eq(0).sum()) if int(group["Z"].eq(0).sum()) else np.nan,
                        "mean_propensity": float(pd.to_numeric(group[propensity_col], errors="coerce").mean()),
                        "mean_p_continue": float((1 - pd.to_numeric(group[propensity_col], errors="coerce")).mean()),
                        "most_common_drop_reason": str(reason_counts.index[0]) if len(reason_counts) else "kept_supported",
                    }
                )
    return pd.DataFrame(rows)


def build_propensity_source_comparison(
    df: pd.DataFrame,
    source_registry: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    support_loss_profile_by_source: pd.DataFrame,
    primary_outcome: str,
    numeric_features: list[str],
    categorical_features: list[str],
    random_seed: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    calibration_lookup = calibration_summary.set_index(["propensity_source", "scope"]) if not calibration_summary.empty else pd.DataFrame()
    e0_mask = df["E0"].eq(1) & df["Z"].isin([0, 1])

    for _, source_row in source_registry.iterrows():
        source_name = str(source_row["propensity_source"])
        propensity_col = str(source_row["active_propensity_col"])
        support_col = f"supported_{source_name}"
        default_scope = (source_name, "supported_default")
        base_scope = (source_name, "E0_all")
        default_cal = calibration_lookup.loc[default_scope] if default_scope in calibration_lookup.index else pd.Series(dtype=object)
        base_cal = calibration_lookup.loc[base_scope] if base_scope in calibration_lookup.index else pd.Series(dtype=object)
        diag = weight_diagnostics_with_weights(
            df,
            outcome=primary_outcome,
            propensity_col=propensity_col,
            support_col=support_col,
            propensity_source=source_name,
            run_label=f"{source_name}|default",
        )
        smds = smd_table_with_weights(
            df,
            features=numeric_features,
            outcome=primary_outcome,
            propensity_col=propensity_col,
            support_col=support_col,
        )
        smds_atc = smd_table_atc_with_weights(
            df,
            features=numeric_features,
            outcome=primary_outcome,
            propensity_col=propensity_col,
            support_col=support_col,
        )
        rows.append(
            {
                "propensity_source": source_name,
                "role": source_row.get("role"),
                "prediction_type": source_row.get("prediction_type"),
                "calibration_status": source_row.get("calibration_status"),
                "is_main_estimator": bool(source_row.get("is_main_estimator")),
                "OOF_or_OOS_scope": source_row.get("OOF_or_OOS_scope"),
                "n_E0_rows": int(e0_mask.sum()),
                "n_E0_shots": int((e0_mask & df["Z"].eq(1)).sum()),
                "n_E0_controls": int((e0_mask & df["Z"].eq(0)).sum()),
                "auc": float(base_cal.get("auc", np.nan)) if len(base_cal) else np.nan,
                "log_loss": float(base_cal.get("log_loss", np.nan)) if len(base_cal) else np.nan,
                "brier": float(base_cal.get("brier", np.nan)) if len(base_cal) else np.nan,
                "ece_15_uniform": float(base_cal.get("ece_15_uniform", np.nan)) if len(base_cal) else np.nan,
                "ece_15_quantile": float(base_cal.get("ece_15_quantile", np.nan)) if len(base_cal) else np.nan,
                "supported_shots": int(diag.loc[0, "n_supported_shots"]) if len(diag) else 0,
                "supported_controls": int(diag.loc[0, "n_supported_controls"]) if len(diag) else 0,
                "treated_retention_pct": 100.0 * int(diag.loc[0, "n_supported_shots"]) / int((e0_mask & df["Z"].eq(1)).sum()) if len(diag) and int((e0_mask & df["Z"].eq(1)).sum()) else np.nan,
                "control_ESS": float(diag.loc[0, "ess_controls"]) if len(diag) else np.nan,
                "shot_ESS_ATC": float(diag.loc[0, "ess_shots_atc"]) if len(diag) else np.nan,
                "max_weight": float(diag.loc[0, "max_weight_controls"]) if len(diag) else np.nan,
                "max_weight_ATC": float(diag.loc[0, "max_weight_shots_atc"]) if len(diag) else np.nan,
                "p95_weight": float(diag.loc[0, "p95_weight_controls"]) if len(diag) else np.nan,
                "p99_weight": float(diag.loc[0, "p99_weight_controls"]) if len(diag) else np.nan,
                "top_abs_weighted_smd": float(smds["abs_SMD_ATT_weighted"].max()) if not smds.empty else np.nan,
                "top_abs_weighted_smd_atc": float(smds_atc["abs_SMD_ATC_weighted"].max()) if not smds_atc.empty else np.nan,
                "AIPW_ATT": aipw_att_with_weights(
                    df,
                    outcome=primary_outcome,
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    propensity_col=propensity_col,
                    support_col=support_col,
                    random_seed=random_seed,
                ),
                "IPW_ATT": ipw_att_with_weights(
                    df,
                    outcome=primary_outcome,
                    propensity_col=propensity_col,
                    support_col=support_col,
                ),
                "matching_ATC": matching_atc_generic(
                    df,
                    outcome=primary_outcome,
                    numeric_features=numeric_features,
                    propensity_col=propensity_col,
                    support_col=support_col,
                ),
                "outcome_regression_ATC": outcome_regression_atc_with_weights(
                    df,
                    outcome=primary_outcome,
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    propensity_col=propensity_col,
                    support_col=support_col,
                    random_seed=random_seed,
                ),
                "AIPW_ATC": aipw_atc_with_weights(
                    df,
                    outcome=primary_outcome,
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    propensity_col=propensity_col,
                    support_col=support_col,
                    random_seed=random_seed,
                ),
                "IPW_ATC": ipw_atc_with_weights(
                    df,
                    outcome=primary_outcome,
                    propensity_col=propensity_col,
                    support_col=support_col,
                ),
                "support_loss_profile_available": bool(
                    not support_loss_profile_by_source.empty
                    and support_loss_profile_by_source["support_variant"].astype(str).eq(f"{source_name}|default").any()
                ),
            }
        )
    return pd.DataFrame(rows)


def run_propensity_stabilization_suite(
    base_df: pd.DataFrame,
    tracking_df: Optional[pd.DataFrame],
    repo_root: Path,
    random_seed: int,
    primary_outcome: str,
    numeric_features: list[str],
    categorical_features: list[str],
    support_tau: float,
    support_min_e: float,
    empirical_support_cols: list[str],
    min_empirical_support_per_arm: int,
    add_outcomes_fn: Callable[[pd.DataFrame], pd.DataFrame],
    recompute_epv_fn: Callable[[pd.DataFrame], pd.DataFrame],
    action_selection_artifact_dir: Optional[Path] = None,
) -> dict[str, Any]:
    working = recompute_epv_fn(add_outcomes_fn(base_df.copy()))
    working = add_baseline_propensity_source_columns(working)

    source_rows: list[dict[str, Any]] = []
    baseline_prediction_types = (
        working.loc[working["E0"].eq(1), "prediction_type_binary_logistic"].dropna().astype(str).unique().tolist()
        if "prediction_type_binary_logistic" in working.columns
        else []
    )
    source_rows.append(
        {
            "propensity_source": "binary_logistic",
            "role": "current baseline",
            "propensity_column": "e_hat_binary_logistic",
            "active_propensity_col": "e_hat_binary_logistic",
            "prediction_type": "|".join(baseline_prediction_types) if baseline_prediction_types else "unknown",
            "calibration_status": "oof_binary_logistic_native",
            "is_main_estimator": True,
            "OOF_or_OOS_scope": "E0_match_grouped_oof",
            "artifact_dir": None,
        }
    )

    working, richer_diagnostics = cross_fit_richer_binary_propensity_model(
        working,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_seed=random_seed,
    )
    source_rows.append(
        {
            "propensity_source": "richer_binary",
            "role": "short_term_upgrade",
            "propensity_column": "e_hat_richer_binary",
            "active_propensity_col": "e_hat_richer_binary",
            "prediction_type": richer_diagnostics.get("prediction_type"),
            "calibration_status": "per_fold_logistic_calibration",
            "is_main_estimator": False,
            "OOF_or_OOS_scope": "E0_match_grouped_oof",
            "artifact_dir": None,
        }
    )

    working, action_selection_metadata = score_action_selection_propensity_source(
        working,
        tracking_df=tracking_df,
        repo_root=repo_root,
        artifact_dir=action_selection_artifact_dir,
    )
    source_rows.append(
        {
            "propensity_source": "action_selection",
            "role": "football_policy_comparator",
            "propensity_column": "e_hat_action_selection_calibrated",
            "active_propensity_col": action_selection_metadata.get("active_propensity_col", "e_hat_action_selection_raw"),
            "prediction_type": action_selection_metadata.get("prediction_type"),
            "calibration_status": action_selection_metadata.get("calibration_status"),
            "is_main_estimator": False,
            "OOF_or_OOS_scope": action_selection_metadata.get("oof_or_oos_scope"),
            "artifact_dir": action_selection_metadata.get("artifact_dir"),
        }
    )

    source_registry = pd.DataFrame(source_rows)
    for _, source_row in source_registry.iterrows():
        source_name = str(source_row["propensity_source"])
        propensity_col = str(source_row["active_propensity_col"])
        if propensity_col not in working.columns:
            continue
        working = apply_support_filter_for_source(
            working,
            propensity_col=propensity_col,
            source_name=source_name,
            tau=support_tau,
            min_e=support_min_e,
            empirical_support_cols=empirical_support_cols,
            min_empirical_support_per_arm=min_empirical_support_per_arm,
        )

    working = set_default_support_aliases(working, "binary_logistic")
    analysis_df = working.copy()

    propensity_calibration_summary_df, propensity_reliability_bins, propensity_high_shot_regions = build_propensity_diagnostics_outputs(
        analysis_df,
        source_registry=source_registry,
    )
    support_grid_sensitivity = build_support_grid_sensitivity(
        analysis_df,
        source_registry=source_registry,
        primary_outcome=primary_outcome,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        empirical_support_cols=empirical_support_cols,
        min_empirical_support_per_arm=min_empirical_support_per_arm,
        add_outcomes_fn=add_outcomes_fn,
        recompute_epv_fn=recompute_epv_fn,
        default_min_e=support_min_e,
        default_tau=support_tau,
        random_seed=random_seed,
    )
    weight_cap_sensitivity, weight_cap_top_smds = build_weight_cap_sensitivity(
        analysis_df,
        source_registry=source_registry,
        primary_outcome=primary_outcome,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        empirical_support_cols=empirical_support_cols,
        min_empirical_support_per_arm=min_empirical_support_per_arm,
        add_outcomes_fn=add_outcomes_fn,
        recompute_epv_fn=recompute_epv_fn,
        default_min_e=support_min_e,
        default_tau=support_tau,
        random_seed=random_seed,
    )
    football_caliper_results, football_caliper_rows = build_football_caliper_branch(
        analysis_df,
        primary_outcome=primary_outcome,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        propensity_col="e_hat_binary_logistic",
        support_col="supported_binary_logistic",
        random_seed=random_seed,
    )
    support_loss_profile_by_source = build_support_loss_profile_by_source(
        analysis_df,
        source_registry=source_registry,
        empirical_support_cols=empirical_support_cols,
        min_empirical_support_per_arm=min_empirical_support_per_arm,
        default_min_e=support_min_e,
        default_tau=support_tau,
    )
    propensity_source_comparison = build_propensity_source_comparison(
        analysis_df,
        source_registry=source_registry,
        calibration_summary=propensity_calibration_summary_df,
        support_loss_profile_by_source=support_loss_profile_by_source,
        primary_outcome=primary_outcome,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_seed=random_seed,
    )

    return {
        "features_with_propensity_sources": working,
        "analysis_df": analysis_df,
        "propensity_source_registry": source_registry,
        "propensity_calibration_summary": propensity_calibration_summary_df,
        "propensity_reliability_bins": propensity_reliability_bins,
        "propensity_high_shot_regions": propensity_high_shot_regions,
        "propensity_source_comparison": propensity_source_comparison,
        "support_grid_sensitivity": support_grid_sensitivity,
        "weight_cap_sensitivity": weight_cap_sensitivity,
        "weight_cap_top_smds": weight_cap_top_smds,
        "football_caliper_results": football_caliper_results,
        "football_caliper_rows": football_caliper_rows,
        "support_loss_profile_by_source": support_loss_profile_by_source,
        "richer_propensity_diagnostics": richer_diagnostics,
        "action_selection_metadata": action_selection_metadata,
    }
