from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover - sklearn compatibility
    StratifiedGroupKFold = None


SHOT_REGION_SCOPES = (
    "high_shot_top_10pct",
    "high_shot_top_5pct",
    "high_shot_p_ge_050",
    "high_shot_p_ge_080",
)


def existing_columns(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in df.columns]


def implied_support_max_e(tau: float) -> float:
    return float(1 - tau)


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def _binary_cv_splits(z: pd.Series, max_splits: int = 5) -> int:
    counts = z.astype(int).value_counts()
    if len(counts) < 2:
        return 1
    return int(max(1, min(max_splits, counts.min())))


def _clip_probabilities(values: Any, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return np.clip(arr, eps, 1 - eps)


def _logit(values: Any, eps: float = 1e-6) -> np.ndarray:
    clipped = _clip_probabilities(values, eps=eps)
    return np.log(clipped / (1 - clipped))


def _dense_matrix(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def add_support_audit_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "distance_to_goal" in out.columns:
        out["distance_to_goal_bin"] = pd.cut(
            pd.to_numeric(out["distance_to_goal"], errors="coerce"),
            bins=[0, 12, 18, 24, 30, np.inf],
            labels=["0-12", "12-18", "18-24", "24-30", "30+"],
            include_lowest=True,
        ).astype(object)
    else:
        out["distance_to_goal_bin"] = "missing"

    centrality = pd.to_numeric(out.get("centrality", pd.Series(index=out.index, dtype=float)), errors="coerce")
    in_box = pd.to_numeric(out.get("in_box", pd.Series(0, index=out.index)), errors="coerce").fillna(0).eq(1)
    half_space = pd.to_numeric(out.get("half_space", pd.Series(0, index=out.index)), errors="coerce").fillna(0).eq(1)
    wide_zone = pd.to_numeric(out.get("wide_zone", pd.Series(0, index=out.index)), errors="coerce").fillna(0).eq(1)
    out["field_zone"] = np.select(
        [in_box, half_space, wide_zone, centrality.ge(0.67), centrality.lt(0.33)],
        ["box", "half_space", "wide", "central", "wide_centrality"],
        default="middle",
    )

    if "defenders_within_10m" in out.columns:
        out["defender_density_bin"] = pd.cut(
            pd.to_numeric(out["defenders_within_10m"], errors="coerce"),
            bins=[-np.inf, 0, 2, 4, np.inf],
            labels=["0", "1-2", "3-4", "5+"],
        ).astype(object)
    else:
        out["defender_density_bin"] = "missing"

    if "minute" in out.columns:
        out["minute_bin"] = pd.cut(
            pd.to_numeric(out["minute"], errors="coerce"),
            bins=[-np.inf, 15, 30, 45, 60, 75, np.inf],
            labels=["0-15", "15-30", "30-45", "45-60", "60-75", "75+"],
        ).astype(object)
    else:
        out["minute_bin"] = "missing"

    for column in ["distance_to_goal_bin", "field_zone", "defender_density_bin", "minute_bin"]:
        out[column] = out[column].where(pd.notna(out[column]), "missing")
    return out


def _cell_key(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series([()] * len(frame), index=frame.index, dtype=object)
    tmp = frame[cols].astype(object).where(pd.notna(frame[cols]), "missing")
    return tmp.apply(lambda row: tuple(row), axis=1)


def apply_support_filter_for_source(
    df: pd.DataFrame,
    propensity_col: str,
    source_name: str,
    tau: float,
    min_e: float,
    empirical_support_cols: list[str] | tuple[str, ...],
    min_empirical_support_per_arm: int,
) -> pd.DataFrame:
    out = add_support_audit_bins(df)
    support_name = str(source_name)
    support_col = f"supported_{support_name}"
    target_col = f"target_att_shot_{support_name}"
    target_atc_col = f"target_atc_continue_{support_name}"
    reason_col = f"support_drop_reason_{support_name}"
    tau_col = f"support_tau_{support_name}"
    min_e_col = f"support_min_e_{support_name}"
    max_e_col = f"support_max_e_{support_name}"
    empirical_cols_col = f"support_empirical_cols_{support_name}"
    empirical_min_col = f"min_empirical_support_per_arm_{support_name}"

    empirical_support_cols = existing_columns(out, list(empirical_support_cols or []))
    p_continue = 1 - pd.to_numeric(out.get(propensity_col), errors="coerce")
    implied_max_e = implied_support_max_e(tau)

    analysis_action = out["Z"].isin([0, 1])
    propensity = pd.to_numeric(out.get(propensity_col), errors="coerce")
    e_valid = propensity.notna() & np.isfinite(propensity)
    base_mask = (
        out["E0"].eq(1)
        & analysis_action
        & e_valid
        & propensity.gt(min_e)
        & propensity.lt(implied_max_e)
    )

    empirical_mask = pd.Series(True, index=out.index)
    if empirical_support_cols:
        eligible = out.loc[base_mask].copy()
        if not eligible.empty:
            cell_counts = (
                eligible.groupby(empirical_support_cols + ["Z"], dropna=False)
                .size()
                .unstack(fill_value=0)
            )
            if 0 not in cell_counts.columns:
                cell_counts[0] = 0
            if 1 not in cell_counts.columns:
                cell_counts[1] = 0
            good_cells = set(
                cell_counts.loc[
                    (cell_counts[0] >= int(min_empirical_support_per_arm))
                    & (cell_counts[1] >= int(min_empirical_support_per_arm))
                ].index
            )
            empirical_mask = _cell_key(out, empirical_support_cols).isin(good_cells)
        else:
            empirical_mask = pd.Series(False, index=out.index)

    supported_mask = base_mask & empirical_mask
    out[support_col] = supported_mask.astype(int)
    out[target_col] = ((out[support_col] == 1) & out["Z"].eq(1)).astype(int)
    out[target_atc_col] = ((out[support_col] == 1) & out["Z"].eq(0)).astype(int)

    reason = pd.Series("kept_supported", index=out.index, dtype=object)
    reason.loc[out["E0"].ne(1)] = "not_E0"
    reason.loc[out["E0"].eq(1) & ~analysis_action] = "non_analysis_action"
    reason.loc[out["E0"].eq(1) & analysis_action & ~e_valid] = "missing_propensity"
    reason.loc[out["E0"].eq(1) & analysis_action & e_valid & propensity.le(min_e)] = "e_below_min"
    reason.loc[out["E0"].eq(1) & analysis_action & e_valid & propensity.ge(implied_max_e)] = "continue_support_below_tau"
    reason.loc[base_mask & ~empirical_mask] = "no_empirical_cell_support"
    reason.loc[supported_mask] = "kept_supported"

    out[reason_col] = reason
    out[tau_col] = float(tau)
    out[min_e_col] = float(min_e)
    out[max_e_col] = float(implied_max_e)
    out[empirical_cols_col] = ",".join(empirical_support_cols) if empirical_support_cols else "none"
    out[empirical_min_col] = int(min_empirical_support_per_arm)
    out[f"p_continue_hat_{support_name}"] = p_continue

    default_aliases = {
        f"{support_col}_default": support_col,
        f"{target_col}_default": target_col,
        f"{target_atc_col}_default": target_atc_col,
        f"{reason_col}_default": reason_col,
        f"{tau_col}_default": tau_col,
        f"{min_e_col}_default": min_e_col,
        f"{max_e_col}_default": max_e_col,
        f"{empirical_cols_col}_default": empirical_cols_col,
    }
    for alias, original in default_aliases.items():
        out[alias] = out[original]
    return out


def set_default_support_aliases(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = df.copy()
    source_name = str(source_name)
    alias_map = {
        "supported": f"supported_{source_name}",
        "target_att_shot": f"target_att_shot_{source_name}",
        "target_atc_continue": f"target_atc_continue_{source_name}",
        "support_drop_reason": f"support_drop_reason_{source_name}",
        "support_tau": f"support_tau_{source_name}",
        "support_min_e": f"support_min_e_{source_name}",
        "support_max_e": f"support_max_e_{source_name}",
        "support_empirical_cols": f"support_empirical_cols_{source_name}",
        "p_continue_hat": f"p_continue_hat_{source_name}",
        "e_hat": f"e_hat_{source_name}",
    }
    for alias, original in alias_map.items():
        if original in out.columns:
            out[alias] = out[original]
    if f"fold_{source_name}" in out.columns:
        out["propensity_fold"] = out[f"fold_{source_name}"]
    if f"prediction_type_{source_name}" in out.columns:
        out["propensity_prediction_type"] = out[f"prediction_type_{source_name}"]
    return out


def binary_log_loss_safe(y: Any, p: Any) -> float:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    p_arr = _clip_probabilities(p)
    mask = np.isfinite(y_arr) & np.isfinite(p_arr)
    if mask.sum() == 0:
        return np.nan
    y_arr = y_arr[mask]
    p_arr = p_arr[mask]
    if not np.isin(y_arr, [0, 1]).all():
        return np.nan
    return float(-np.mean(y_arr * np.log(p_arr) + (1 - y_arr) * np.log(1 - p_arr)))


def brier_score_safe(y: Any, p: Any) -> float:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    mask = np.isfinite(y_arr) & np.isfinite(p_arr)
    if mask.sum() == 0:
        return np.nan
    y_arr = y_arr[mask]
    p_arr = p_arr[mask]
    return float(np.mean((y_arr - p_arr) ** 2))


def expected_calibration_error(
    y: Any,
    p: Any,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> float:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    mask = np.isfinite(y_arr) & np.isfinite(p_arr)
    if mask.sum() == 0:
        return np.nan
    y_arr = y_arr[mask]
    p_arr = p_arr[mask]
    if len(y_arr) == 0:
        return np.nan

    bin_ids, _, _ = _calibration_bins_from_scores(p_arr, n_bins=n_bins, strategy=strategy)
    total = float(len(y_arr))
    ece = 0.0
    for bin_id in np.unique(bin_ids):
        bin_mask = bin_ids == bin_id
        if not np.any(bin_mask):
            continue
        observed = float(np.mean(y_arr[bin_mask]))
        predicted = float(np.mean(p_arr[bin_mask]))
        ece += (bin_mask.sum() / total) * abs(observed - predicted)
    return float(ece)


def _calibration_bins_from_scores(
    scores: np.ndarray,
    n_bins: int,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    clipped = _clip_probabilities(scores)
    strategy = str(strategy).strip().lower()
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(clipped, quantiles)
        edges[0] = 0.0
        edges[-1] = 1.0
        edges = np.maximum.accumulate(edges)
        for idx in range(1, len(edges)):
            if edges[idx] <= edges[idx - 1]:
                edges[idx] = np.nextafter(edges[idx - 1], 1.0)
        if not np.isfinite(edges).all() or edges[-1] <= edges[0]:
            return _calibration_bins_from_scores(clipped, n_bins=n_bins, strategy="uniform")
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        strategy = "uniform"
    bin_ids = np.digitize(clipped, edges[1:-1], right=False)
    return bin_ids, edges, strategy


def calibration_bin_table(
    df: pd.DataFrame,
    y_col: str,
    p_col: str,
    scope_label: str,
    source_name: str,
    n_bins: int = 15,
    strategy: str = "uniform",
    supported_col: Optional[str] = None,
    drop_reason_col: Optional[str] = None,
) -> pd.DataFrame:
    subset = df.copy()
    subset = subset.loc[
        subset[y_col].isin([0, 1])
        & pd.to_numeric(subset[p_col], errors="coerce").notna()
    ].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "propensity_source",
                "scope",
                "bin_strategy",
                "bin_id",
                "bin_low",
                "bin_high",
                "n_rows",
                "n_shots",
                "observed_shot_rate",
                "mean_predicted_shot_rate",
                "calibration_gap",
                "mean_distance_to_goal",
                "share_supported",
                "share_continue_support_below_tau",
                "share_no_empirical_cell_support",
            ]
        )

    y = subset[y_col].astype(int).to_numpy()
    p = pd.to_numeric(subset[p_col], errors="coerce").to_numpy(dtype=float)
    bin_ids, edges, strategy = _calibration_bins_from_scores(p, n_bins=n_bins, strategy=strategy)
    subset = subset.assign(_bin_id=bin_ids)

    rows: list[dict[str, Any]] = []
    for bin_id in sorted(subset["_bin_id"].unique()):
        part = subset.loc[subset["_bin_id"] == bin_id].copy()
        if part.empty:
            continue
        supported_share = np.nan
        if supported_col and supported_col in part.columns:
            supported_share = float(pd.to_numeric(part[supported_col], errors="coerce").fillna(0).eq(1).mean())
        continue_drop_share = np.nan
        empirical_drop_share = np.nan
        if drop_reason_col and drop_reason_col in part.columns:
            continue_drop_share = float(part[drop_reason_col].astype(str).eq("continue_support_below_tau").mean())
            empirical_drop_share = float(part[drop_reason_col].astype(str).eq("no_empirical_cell_support").mean())
        rows.append(
            {
                "propensity_source": source_name,
                "scope": scope_label,
                "bin_strategy": strategy,
                "bin_id": int(bin_id),
                "bin_low": float(edges[int(bin_id)]),
                "bin_high": float(edges[int(bin_id) + 1]),
                "n_rows": int(len(part)),
                "n_shots": int(part[y_col].sum()),
                "observed_shot_rate": float(part[y_col].mean()),
                "mean_predicted_shot_rate": float(pd.to_numeric(part[p_col], errors="coerce").mean()),
                "calibration_gap": float(part[y_col].mean() - pd.to_numeric(part[p_col], errors="coerce").mean()),
                "mean_distance_to_goal": float(pd.to_numeric(part.get("distance_to_goal"), errors="coerce").mean()) if "distance_to_goal" in part.columns else np.nan,
                "share_supported": supported_share,
                "share_continue_support_below_tau": continue_drop_share,
                "share_no_empirical_cell_support": empirical_drop_share,
            }
        )
    return pd.DataFrame(rows)


def propensity_calibration_summary(
    df: pd.DataFrame,
    y_col: str,
    p_col: str,
    source_name: str,
    scope_mask: pd.Series,
    scope_label: str,
) -> dict[str, Any]:
    subset = df.loc[scope_mask].copy()
    subset = subset.loc[
        subset[y_col].isin([0, 1])
        & pd.to_numeric(subset[p_col], errors="coerce").notna()
    ].copy()
    if subset.empty:
        return {
            "propensity_source": source_name,
            "scope": scope_label,
            "auc": np.nan,
            "log_loss": np.nan,
            "brier": np.nan,
            "ece_15_uniform": np.nan,
            "ece_15_quantile": np.nan,
            "n_rows": 0,
            "n_shots": 0,
            "shot_rate": np.nan,
            "mean_predicted_shot_rate": np.nan,
        }

    y = subset[y_col].astype(int).to_numpy()
    p = pd.to_numeric(subset[p_col], errors="coerce").to_numpy(dtype=float)
    auc = np.nan
    if len(np.unique(y)) >= 2:
        auc = float(roc_auc_score(y, p))
    return {
        "propensity_source": source_name,
        "scope": scope_label,
        "auc": auc,
        "log_loss": binary_log_loss_safe(y, p),
        "brier": brier_score_safe(y, p),
        "ece_15_uniform": expected_calibration_error(y, p, n_bins=15, strategy="uniform"),
        "ece_15_quantile": expected_calibration_error(y, p, n_bins=15, strategy="quantile"),
        "n_rows": int(len(subset)),
        "n_shots": int(y.sum()),
        "shot_rate": float(np.mean(y)),
        "mean_predicted_shot_rate": float(np.mean(p)),
    }


def high_propensity_region_table(
    df: pd.DataFrame,
    y_col: str,
    p_col: str,
    source_name: str,
    scope_mask: pd.Series,
) -> pd.DataFrame:
    scope = df.loc[scope_mask].copy()
    scope = scope.loc[
        scope[y_col].isin([0, 1])
        & pd.to_numeric(scope[p_col], errors="coerce").notna()
    ].copy()
    if scope.empty:
        return pd.DataFrame()

    scores = pd.to_numeric(scope[p_col], errors="coerce")
    top10 = scores >= scores.quantile(0.90)
    top05 = scores >= scores.quantile(0.95)
    masks = {
        "high_shot_top_10pct": top10,
        "high_shot_top_5pct": top05,
        "high_shot_p_ge_050": scores >= 0.50,
        "high_shot_p_ge_080": scores >= 0.80,
    }
    rows = []
    for label, mask in masks.items():
        summary = propensity_calibration_summary(
            scope,
            y_col=y_col,
            p_col=p_col,
            source_name=source_name,
            scope_mask=mask.reindex(scope.index, fill_value=False),
            scope_label=label,
        )
        rows.append(summary)
    return pd.DataFrame(rows)


def _fit_logit_calibrator(raw_probabilities: np.ndarray, y: np.ndarray) -> Optional[LogisticRegression]:
    raw_probabilities = _clip_probabilities(raw_probabilities)
    y = np.asarray(y, dtype=int).reshape(-1)
    if len(raw_probabilities) < 20 or len(np.unique(y)) < 2:
        return None
    calibrator = LogisticRegression(max_iter=2000, solver="lbfgs")
    calibrator.fit(_logit(raw_probabilities).reshape(-1, 1), y)
    return calibrator


def _apply_logit_calibrator(raw_probabilities: np.ndarray, calibrator: Optional[LogisticRegression]) -> np.ndarray:
    clipped = _clip_probabilities(raw_probabilities)
    if calibrator is None:
        return clipped
    return calibrator.predict_proba(_logit(clipped).reshape(-1, 1))[:, 1]


def _calibration_train_test_indices(
    train_df: pd.DataFrame,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    z = train_df["Z"].astype(int)
    if len(train_df) < 30 or len(np.unique(z)) < 2:
        idx = np.arange(len(train_df))
        return idx, idx

    groups = train_df["match_id"] if "match_id" in train_df.columns else None
    if (
        groups is not None
        and groups.nunique(dropna=True) >= 3
        and StratifiedGroupKFold is not None
    ):
        group_values = groups.fillna(-1)
        n_splits = min(5, int(group_values.nunique(dropna=True)), int(z.value_counts().min()))
        if n_splits >= 2:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
            fit_pos, cal_pos = next(splitter.split(train_df, z, groups=group_values))
            if len(np.unique(z.iloc[fit_pos])) >= 2 and len(np.unique(z.iloc[cal_pos])) >= 2:
                return np.asarray(fit_pos), np.asarray(cal_pos)

    test_size = min(0.25, max(0.10, 5000 / max(len(train_df), 1)))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    fit_pos, cal_pos = next(splitter.split(train_df, z))
    if len(np.unique(z.iloc[fit_pos])) < 2 or len(np.unique(z.iloc[cal_pos])) < 2:
        idx = np.arange(len(train_df))
        return idx, idx
    return np.asarray(fit_pos), np.asarray(cal_pos)


def _fit_richer_binary_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    random_seed: int,
) -> np.ndarray:
    fit_pos, cal_pos = _calibration_train_test_indices(train_df, random_seed=random_seed)
    fit_df = train_df.iloc[fit_pos]
    cal_df = train_df.iloc[cal_pos]

    preprocessor = make_preprocessor(numeric_features, categorical_features)
    x_fit = _dense_matrix(preprocessor.fit_transform(fit_df[numeric_features + categorical_features]))
    x_cal = _dense_matrix(preprocessor.transform(cal_df[numeric_features + categorical_features]))
    x_test = _dense_matrix(preprocessor.transform(test_df[numeric_features + categorical_features]))

    base_model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=random_seed,
    )
    base_model.fit(x_fit, fit_df["Z"].astype(int).to_numpy())
    raw_cal = base_model.predict_proba(x_cal)[:, 1]
    calibrator = _fit_logit_calibrator(raw_cal, cal_df["Z"].astype(int).to_numpy())
    raw_test = base_model.predict_proba(x_test)[:, 1]
    return _apply_logit_calibrator(raw_test, calibrator)


def cross_fit_richer_binary_propensity_model(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    random_seed: int = 42,
    max_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    numeric_features = existing_columns(df, numeric_features)
    categorical_features = existing_columns(df, categorical_features)
    feature_cols = numeric_features + categorical_features
    if not feature_cols:
        raise ValueError("No usable richer-binary propensity features found.")

    model_df = df.query("E0 == 1").dropna(subset=["Z"]).copy()
    model_df[feature_cols] = model_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    z = model_df["Z"].astype(int)
    if z.nunique() < 2:
        raise ValueError("Richer propensity model needs both shot and continue rows after E0 filtering.")

    cv = _binary_cv_splits(z, max_splits=max_splits)
    oof_e = pd.Series(index=model_df.index, dtype=float)
    fold_id = pd.Series(index=model_df.index, dtype="Int64")

    if cv >= 2:
        groups = model_df["match_id"] if "match_id" in model_df.columns else None
        use_group_cv = (
            groups is not None
            and groups.nunique(dropna=True) >= cv
            and StratifiedGroupKFold is not None
        )
        if use_group_cv:
            splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=random_seed)
            split_iter = splitter.split(model_df, z, groups=groups)
            prediction_type = "cross_fit_oof_match_grouped_calibrated_hgb"
        else:
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_seed)
            split_iter = splitter.split(model_df, z)
            prediction_type = "cross_fit_oof_calibrated_hgb"

        for fold, (train_pos, test_pos) in enumerate(split_iter):
            train_df = model_df.iloc[train_pos].copy()
            test_df = model_df.iloc[test_pos].copy()
            oof_predictions = _fit_richer_binary_model(
                train_df=train_df,
                test_df=test_df,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                random_seed=random_seed + fold,
            )
            oof_e.loc[test_df.index] = oof_predictions
            fold_id.loc[test_df.index] = fold
    else:
        prediction_type = "in_sample_calibrated_hgb_fallback"
        oof_predictions = _fit_richer_binary_model(
            train_df=model_df,
            test_df=model_df,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_seed=random_seed,
        )
        oof_e.loc[model_df.index] = oof_predictions
        fold_id.loc[model_df.index] = 0

    full_predictions = _fit_richer_binary_model(
        train_df=model_df,
        test_df=df.assign(Z=df.get("Z")),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_seed=random_seed + 999,
    )

    out = df.copy()
    out["e_hat_richer_binary"] = _clip_probabilities(full_predictions, eps=1e-4)
    out.loc[oof_e.index, "e_hat_richer_binary"] = _clip_probabilities(oof_e.to_numpy(), eps=1e-4)
    out["p_continue_hat_richer_binary"] = 1 - out["e_hat_richer_binary"]
    out["fold_richer_binary"] = pd.Series(-1, index=out.index, dtype="Int64")
    out.loc[fold_id.index, "fold_richer_binary"] = fold_id
    out["prediction_type_richer_binary"] = "full_fit_calibrated_hgb_fallback"
    out.loc[oof_e.index, "prediction_type_richer_binary"] = prediction_type

    auc = np.nan
    if len(np.unique(z)) >= 2 and oof_e.notna().any():
        auc = float(roc_auc_score(z.loc[oof_e.index], oof_e.loc[oof_e.index]))
    diagnostics = {
        "auc_oof": auc,
        "cv_folds": int(cv),
        "n_model_rows": int(len(model_df)),
        "n_treated": int(z.sum()),
        "n_controls": int((z == 0).sum()),
        "prediction_type": prediction_type,
        "model_class": "HistGradientBoostingClassifier",
        "match_grouped_cv": bool(
            cv >= 2
            and "match_id" in model_df.columns
            and model_df["match_id"].nunique(dropna=True) >= cv
            and StratifiedGroupKFold is not None
        ),
        "calibration_method": "per_fold_logistic_on_raw_score",
    }
    return out, diagnostics


def add_baseline_propensity_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "e_hat" in out.columns:
        out["e_hat_binary_logistic"] = pd.to_numeric(out["e_hat"], errors="coerce")
    if "p_continue_hat" in out.columns:
        out["p_continue_hat_binary_logistic"] = pd.to_numeric(out["p_continue_hat"], errors="coerce")
    if "propensity_fold" in out.columns:
        out["fold_binary_logistic"] = out["propensity_fold"]
    if "propensity_prediction_type" in out.columns:
        out["prediction_type_binary_logistic"] = out["propensity_prediction_type"]
    return out


def _ensure_src_root(repo_root: Path) -> Path:
    src_root = (repo_root / "src").resolve()
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return src_root


def _resolve_action_selection_paths(repo_root: Path, artifact_dir: Optional[Path] = None) -> tuple[Path, Path, Path, Path]:
    resolved_dir = (artifact_dir or (repo_root / "results" / "models" / "action_selection")).resolve()
    return (
        resolved_dir / "action_selection_model.pt",
        resolved_dir / "action_selection_metadata.json",
        resolved_dir / "action_selection_feature_scaler.pkl",
        resolved_dir,
    )


def _action_selection_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    score_df = df.copy()
    if "frame_id" not in score_df.columns and "tracking_frame_id" in score_df.columns:
        score_df["frame_id"] = score_df["tracking_frame_id"]
    if "start_frame_id" not in score_df.columns and "frame_id" in score_df.columns:
        score_df["start_frame_id"] = score_df["frame_id"]
    if "game_id" not in score_df.columns and "match_id" in score_df.columns:
        score_df["game_id"] = score_df["match_id"]
    return score_df


def _load_action_selection_split_manifest(repo_root: Path) -> tuple[Optional[dict[str, Any]], Optional[Path]]:
    manifest_path = repo_root / "data" / "processed" / "cache" / "splits" / "action_selection_split_manifest.json"
    if not manifest_path.exists():
        return None, None
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle), manifest_path


def score_action_selection_propensity_source(
    df: pd.DataFrame,
    tracking_df: Optional[pd.DataFrame],
    repo_root: Path,
    artifact_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    out["e_hat_action_selection_raw"] = np.nan
    out["p_continue_hat_action_selection_raw"] = np.nan
    out["e_hat_action_selection_calibrated"] = np.nan
    out["p_continue_hat_action_selection_calibrated"] = np.nan
    out["action_selection_split"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["action_selection_is_strict_oof"] = np.nan
    out["action_selection_artifact_dir"] = np.nan

    model_path, metadata_path, scaler_path, resolved_dir = _resolve_action_selection_paths(repo_root, artifact_dir=artifact_dir)
    metadata: dict[str, Any] = {
        "artifact_dir": str(resolved_dir),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "scaler_path": str(scaler_path),
        "available": False,
        "calibration_status": "unavailable",
        "prediction_type": "pretrained_comparator_missing",
        "oof_or_oos_scope": "missing",
    }
    required_paths = [model_path, metadata_path, scaler_path]
    if any(not path.exists() for path in required_paths):
        return out, metadata

    _ensure_src_root(repo_root)
    from ActionSelection.data import load_default_baseline_xg_artifacts
    from ActionSelection.inference import score_action_selection_for_causal_decisions

    metadata["available"] = True
    baseline_xg_artifacts = load_default_baseline_xg_artifacts()

    score_mask = out["E0"].eq(1) & out["Z"].isin([0, 1])
    score_input = _action_selection_input_frame(out.loc[score_mask].copy())
    scored = score_action_selection_for_causal_decisions(
        score_input,
        tracking_window=tracking_df,
        model_path=model_path,
        metadata_path=metadata_path,
        scaler_path=scaler_path,
        baseline_xg_artifacts=baseline_xg_artifacts,
        open_play_only=False,
    )
    raw_shot = pd.to_numeric(scored.get("p_action_shot"), errors="coerce")
    raw_continue = pd.to_numeric(scored.get("p_continue"), errors="coerce")
    out.loc[scored.index, "e_hat_action_selection_raw"] = _clip_probabilities(raw_shot.to_numpy(), eps=1e-4)
    out.loc[scored.index, "p_continue_hat_action_selection_raw"] = _clip_probabilities(raw_continue.to_numpy(), eps=1e-4)
    out["action_selection_artifact_dir"] = str(resolved_dir)

    manifest, manifest_path = _load_action_selection_split_manifest(repo_root)
    if manifest is not None:
        match_assignments = {str(key): str(value) for key, value in (manifest.get("match_assignments") or {}).items()}
        match_ids = pd.to_numeric(out.get("match_id"), errors="coerce")
        splits = match_ids.map(
            lambda value: match_assignments.get(str(int(value))) if pd.notna(value) and str(int(value)) in match_assignments else pd.NA
        )
        out["action_selection_split"] = splits.astype("string")
        strict_oof = pd.Series(np.nan, index=out.index, dtype=float)
        strict_oof.loc[out["action_selection_split"].isin(["train"])] = 0.0
        strict_oof.loc[out["action_selection_split"].isin(["val", "test"])] = 1.0
        out["action_selection_is_strict_oof"] = strict_oof
        metadata["split_manifest_path"] = str(manifest_path)
        metadata["prediction_type"] = "pretrained_split_aware_comparator"
        metadata["oof_or_oos_scope"] = "val_test_match_oos"

    eval_mask = (
        out["E0"].eq(1)
        & out["Z"].isin([0, 1])
        & pd.to_numeric(out["e_hat_action_selection_raw"], errors="coerce").notna()
    )
    raw_scores = pd.to_numeric(out["e_hat_action_selection_raw"], errors="coerce")
    y = pd.to_numeric(out["Z"], errors="coerce")

    val_mask = eval_mask & out["action_selection_split"].eq("val")
    test_mask = eval_mask & out["action_selection_split"].eq("test")
    non_train_mask = eval_mask & out["action_selection_split"].isin(["val", "test"])

    calibrator = None
    fit_scope = "none"
    eval_scope = "none"
    if val_mask.sum() >= 20 and y.loc[val_mask].nunique(dropna=True) >= 2:
        calibrator = _fit_logit_calibrator(raw_scores.loc[val_mask].to_numpy(), y.loc[val_mask].astype(int).to_numpy())
        fit_scope = "val"
        if test_mask.sum() >= 20 and y.loc[test_mask].nunique(dropna=True) >= 2:
            eval_scope = "test"
            metadata["calibration_status"] = "split_aware_val_fit_test_eval"
            metadata["oof_or_oos_scope"] = "test_oos_eval"
        else:
            eval_scope = "val"
            metadata["calibration_status"] = "split_aware_val_fit"
            metadata["oof_or_oos_scope"] = "val_oos_fit_eval"
    elif non_train_mask.sum() >= 20 and y.loc[non_train_mask].nunique(dropna=True) >= 2:
        calibrator = _fit_logit_calibrator(raw_scores.loc[non_train_mask].to_numpy(), y.loc[non_train_mask].astype(int).to_numpy())
        fit_scope = "val_test"
        eval_scope = "val_test"
        metadata["calibration_status"] = "mixed_split_comparator"
        metadata["prediction_type"] = "pretrained_comparator_mixed_split_calibrated"
        metadata["oof_or_oos_scope"] = "val_test_mixed"
    else:
        metadata["calibration_status"] = "raw_untrusted"
        metadata["prediction_type"] = "pretrained_comparator_raw_untrusted"

    calibrated = _apply_logit_calibrator(raw_scores.loc[eval_mask].to_numpy(), calibrator)
    out.loc[eval_mask, "e_hat_action_selection_calibrated"] = _clip_probabilities(calibrated, eps=1e-4)
    out.loc[eval_mask, "p_continue_hat_action_selection_calibrated"] = 1 - out.loc[eval_mask, "e_hat_action_selection_calibrated"]

    if eval_scope != "none":
        scope_mask = val_mask if eval_scope == "val" else test_mask if eval_scope == "test" else non_train_mask
        metadata["eval_metrics"] = propensity_calibration_summary(
            out,
            y_col="Z",
            p_col="e_hat_action_selection_calibrated",
            source_name="action_selection",
            scope_mask=scope_mask,
            scope_label=f"action_selection_{eval_scope}_eval",
        )
    metadata["fit_scope"] = fit_scope
    metadata["eval_scope"] = eval_scope
    metadata["active_propensity_col"] = (
        "e_hat_action_selection_calibrated"
        if metadata["calibration_status"] != "raw_untrusted"
        else "e_hat_action_selection_raw"
    )
    return out, metadata
