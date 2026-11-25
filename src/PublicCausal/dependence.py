"""Dependence-robust inference on saved supported-ATT scores (no refits).

The supported ATT is theta = sum(a_i) / N1 with a_i the AIPW-ATT numerator
(treated: Y - m0; control: -e/(1-e) (Y - m0)). Every function here treats the
fitted nuisances as fixed, like the original match-cluster score interval, so
nuisance estimation error is not propagated. Cluster sums run over a registry
that may include clusters with no supported score (they count in G).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

WEBB_VALUES = np.array([-np.sqrt(1.5), -1., -np.sqrt(.5), np.sqrt(.5), 1., np.sqrt(1.5)])


def score_sample(rows):
    """Supported rows with an ``att_numerator`` column (computed if absent)."""
    sample = rows.copy()
    if 'support' in sample:
        sample = sample.loc[sample['support'].fillna(False).astype(bool)]
    if 'Y' in sample:
        sample = sample.loc[sample['Y'].notna()]
    sample = sample.copy()
    if 'att_numerator' not in sample:
        t = sample['T'].to_numpy(int)
        e = sample['e'].to_numpy(float)
        residual = sample['Y'].to_numpy(float) - sample['m0'].to_numpy(float)
        weight = np.where(t == 0, e / (1 - e), 0.)
        sample['att_numerator'] = np.where(t == 1, residual, -weight * residual)
    if not np.isfinite(sample['att_numerator'].to_numpy(float)).all():
        raise ValueError('Nonfinite ATT score numerators')
    n1 = int(sample['T'].sum())
    if n1 <= 0 or n1 == len(sample):
        raise ValueError('Supported ATT scores need both arms')
    return sample


def _reindex_registry(cluster, registry, *series):
    """Reindex per-cluster series to the registry; clusters outside it are an error, never dropped."""
    keys = pd.Index(pd.Series(list(registry)).astype(str).unique())
    for part in series:
        if not set(part.index).issubset(set(keys)):
            raise ValueError(f'Scores carry {cluster} values outside the registry')
    out = tuple(part.reindex(keys, fill_value=0.) for part in series)
    return out if len(out) > 1 else out[0]


def _cluster_sums(sample, cluster, theta, registry=None):
    psi = sample['att_numerator'].to_numpy(float) - theta * sample['T'].to_numpy(float)
    sums = pd.Series(psi, index=sample[cluster].astype(str).to_numpy()).groupby(level=0).sum()
    return sums if registry is None else _reindex_registry(cluster, registry, sums)


def cluster_se(rows, cluster, registry=None):
    """One-way cluster-robust score SE with G/(G-1) correction and t(G-1) interval."""
    sample = score_sample(rows)
    n1 = float(sample['T'].sum())
    theta = float(sample['att_numerator'].sum() / n1)
    sums = _cluster_sums(sample, cluster, theta, registry)
    g = len(sums)
    if g <= 1:
        raise ValueError('Cluster inference needs at least two clusters')
    se = float(np.sqrt(g / (g - 1) * np.square(sums.to_numpy()).sum()) / n1)
    critical = float(student_t.ppf(.975, g - 1))
    return dict(cluster=cluster, att=theta, se=se, clusters=g, df=g - 1, critical=critical,
                interval=[theta - critical * se, theta + critical * se],
                pvalue_zero=float(2 * student_t.sf(abs(theta / se), g - 1)) if se > 0 else None)


def _cgm_combine(var_a, var_b, var_ab, g_a, g_b):
    raw = var_a + var_b - var_ab
    fallback = raw <= 0
    variance = max(var_a, var_b) if fallback else raw
    return dict(variance_raw=float(raw), se=float(np.sqrt(variance)), fallback_used=bool(fallback),
                df=int(min(g_a, g_b) - 1))


def two_way_cgm(rows, first='team_id', second='opponent_id', registries=None):
    """Cameron-Gelbach-Miller two-way variance; intersection = directed (first, second) pair."""
    registries = registries or {}
    sample = score_sample(rows)
    sample['_pair'] = sample[first].astype(str) + '|' + sample[second].astype(str)
    a = cluster_se(sample, first, registries.get(first))
    b = cluster_se(sample, second, registries.get(second))
    ab = cluster_se(sample, '_pair', registries.get('pair'))
    out = _cgm_combine(a['se'] ** 2, b['se'] ** 2, ab['se'] ** 2, a['clusters'], b['clusters'])
    critical = float(student_t.ppf(.975, out['df']))
    theta = a['att']
    out.update(att=theta, critical=critical, interval=[theta - critical * out['se'], theta + critical * out['se']],
               pvalue_zero=float(2 * student_t.sf(abs(theta / out['se']), out['df'])),
               components=dict(first=a['clusters'], second=b['clusters'], intersection=ab['clusters']))
    return out


def webb_weights(rng, draws, clusters):
    return WEBB_VALUES[rng.integers(0, 6, size=(draws, clusters))]


def wild_cluster_bootstrap(rows, cluster, registry=None, draws=9999, seed=20261001, level=.95,
                           grid_halfwidth=None, grid_points=801):
    """Null-imposed score wild cluster bootstrap (Webb weights) with test inversion.

    For a hypothesised theta0: U_g = sum_g(a_i - theta0 T_i); U*_g = w_g U_g;
    theta* - theta0 = sum U*_g / N1; SE* from re-centred U*_g - (theta*-theta0) N1_g;
    p(theta0) = P*(|t*| >= |t(theta0)|). The same weight matrix is used for every theta0.
    """
    sample = score_sample(rows)
    n1 = float(sample['T'].sum())
    theta = float(sample['att_numerator'].sum() / n1)
    keys = sample[cluster].astype(str).to_numpy()
    numer = pd.Series(sample['att_numerator'].to_numpy(float), index=keys).groupby(level=0).sum()
    treated = pd.Series(sample['T'].to_numpy(float), index=keys).groupby(level=0).sum()
    if registry is not None:
        numer, treated = _reindex_registry(cluster, registry, numer, treated)
    a_g, n_g = numer.to_numpy(), treated.to_numpy()
    g = len(a_g)
    if g <= 1:
        raise ValueError('Wild cluster bootstrap needs at least two clusters')
    c = g / (g - 1)
    observed_u = a_g - theta * n_g
    se_hat = float(np.sqrt(c * np.square(observed_u).sum()) / n1)
    weights = webb_weights(np.random.default_rng(seed), int(draws), g)

    def pvalue(theta0):
        t_obs = (theta - theta0) / se_hat
        u0 = a_g - theta0 * n_g
        star = weights * u0
        delta = star.sum(axis=1) / n1
        centred = star - delta[:, None] * n_g[None, :]
        se_star = np.sqrt(c * np.square(centred).sum(axis=1)) / n1
        with np.errstate(divide='ignore', invalid='ignore'):
            t_star = np.where(se_star > 0, delta / se_star, 0.)
        return float(np.mean(np.abs(t_star) >= abs(t_obs) - 1e-12))

    alpha = 1 - level
    half = grid_halfwidth or 8 * se_hat
    grid = np.linspace(theta - half, theta + half, grid_points)
    values = np.array([pvalue(x) for x in grid])
    accepted = grid[values > alpha]
    if not len(accepted):
        raise ValueError('Test inversion accepted no grid value')

    def refine(inside, outside):
        for _ in range(40):
            middle = (inside + outside) / 2
            inside, outside = (middle, outside) if pvalue(middle) > alpha else (inside, middle)
        return inside

    step = grid[1] - grid[0]
    low = refine(accepted.min(), accepted.min() - step) if accepted.min() > grid[0] else None
    high = refine(accepted.max(), accepted.max() + step) if accepted.max() < grid[-1] else None
    return dict(cluster=cluster, att=theta, clusters=g, draws=int(draws), seed=int(seed), weights='webb6',
                analytic_se=se_hat, pvalue_zero=pvalue(0.), pvalue_at_estimate=pvalue(theta),
                interval=[None if low is None else float(low), None if high is None else float(high)], interval_open_ended=low is None or high is None,
                method='null-imposed score wild cluster bootstrap, test inversion; nuisances fixed')


def paired_difference(base, alternative, cluster='original_match_id', registry=None):
    """theta_alt - theta_base on identical supported decisions and treatment; paired cluster SE."""
    a, b = score_sample(base), score_sample(alternative)
    key = 'decision_row_id'
    if len(a) != len(b) or a[key].astype(str).tolist() != b[key].astype(str).tolist():
        raise ValueError('Paired difference needs identical supported decision rows in the same order')
    if not np.array_equal(a['T'].to_numpy(int), b['T'].to_numpy(int)):
        raise ValueError('Paired difference needs identical treatment')
    n1 = float(a['T'].sum())
    ta, tb = a['att_numerator'].sum() / n1, b['att_numerator'].sum() / n1
    t = a['T'].to_numpy(float)
    psi = (b['att_numerator'].to_numpy(float) - tb * t) - (a['att_numerator'].to_numpy(float) - ta * t)
    sums = pd.Series(psi, index=a[cluster].astype(str).to_numpy()).groupby(level=0).sum()
    if registry is not None:
        sums = _reindex_registry(cluster, registry, sums)
    g = len(sums)
    se = float(np.sqrt(g / (g - 1) * np.square(sums.to_numpy()).sum()) / n1)
    critical = float(student_t.ppf(.975, g - 1))
    diff = float(tb - ta)
    return dict(base=float(ta), alternative=float(tb), difference=diff, se=se, clusters=g, df=g - 1,
                interval=[diff - critical * se, diff + critical * se], cluster=cluster)


def match_heterogeneity_plugin(rows):
    """Approximate offset of a within-match treatment permutation when nuisances ignore match.

    sum_m rbar_m (N1_m/N1 - N0_m/N0), rbar_m = mean(Y - m0) in match m (supported rows).
    Inference/approximation: assumes the fitted propensity carries no match-level shot share.
    """
    sample = rows.loc[rows['support'].fillna(False).astype(bool)] if 'support' in rows else rows
    t = sample['T'].astype(int)
    n1, n0 = int(t.sum()), int((t == 0).sum())
    grouped = pd.DataFrame(dict(m=sample['original_match_id'].astype(str).to_numpy(), t=t.to_numpy(),
                                r=(sample['Y'] - sample['m0']).to_numpy(float), y=sample['Y'].to_numpy(float)))
    by = grouped.groupby('m').agg(n1=('t', 'sum'), n=('t', 'size'), r=('r', 'mean'), y=('y', 'mean'))
    share = by.n1 / n1 - (by.n - by.n1) / n0
    return dict(residual_plugin=float((by.r * share).sum()),
                outcome_plugin=float(((by.y - grouped.y.mean()) * share).sum()),
                match_share_outcome_corr=float(np.corrcoef(by.n1 / by.n, by.y)[0, 1]) if len(by) > 2 else None,
                matches=int(len(by)))


def match_stratified_att(rows, cluster='original_match_id'):
    """Within-match Hajek ATT: each match's control weights rescaled to its treated count.

    theta = sum_m N1_m (rbar1_m - sum_w r / sum_w over match-m controls) / N1, with
    r = Y - m0 and w = e/(1-e). Removes between-match comparisons; treated in matches
    without supported controls are dropped and counted. Match-cluster SE, weights fixed.
    """
    sample = score_sample(rows)
    t = sample['T'].to_numpy(int)
    e = sample['e'].to_numpy(float)
    frame = pd.DataFrame(dict(m=sample[cluster].astype(str).to_numpy(), t=t,
                              r=sample['Y'].to_numpy(float) - sample['m0'].to_numpy(float),
                              w=np.where(t == 0, e / (1 - e), 0.)))
    frame['wr'] = frame.w * frame.r
    frame['tr'] = frame.t * frame.r
    by = frame.groupby('m').agg(n1=('t', 'sum'), tr=('tr', 'sum'), w=('w', 'sum'), wr=('wr', 'sum'))
    usable = (by.n1 > 0) & (by.w > 0)
    used = by.loc[usable]
    contribution = used.tr - used.n1 * used.wr / used.w
    n1 = float(used.n1.sum())
    theta = float(contribution.sum() / n1)
    scores = (contribution - theta * used.n1).reindex(by.index, fill_value=0.)
    g = len(by)
    se = float(np.sqrt(g / (g - 1) * np.square(scores.to_numpy()).sum()) / n1)
    critical = float(student_t.ppf(.975, g - 1))
    return dict(att=theta, se=se, clusters=g, df=g - 1, interval=[theta - critical * se, theta + critical * se],
                n_shots_used=int(n1), dropped_treated=int(by.loc[~usable, 'n1'].sum()),
                method='within-match Hajek reweighting of saved nuisances (exploratory, not prespecified)')


def paired_null_summary(records):
    ok = [r for r in records if r.get('status') == 'ok' and r.get('att') is not None
          and r.get('known_assignment_att') is not None]
    d = np.array([r['att'] - r['known_assignment_att'] for r in ok], float)
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else None
    mcse = sd / np.sqrt(n) if sd is not None else None
    return dict(n=n, mean_difference=float(d.mean()) if n else None, sd=sd, mcse=mcse,
                t=float(d.mean() / mcse) if mcse else None,
                mean_fitted=float(np.mean([r['att'] for r in ok])) if n else None,
                mean_known=float(np.mean([r['known_assignment_att'] for r in ok])) if n else None)
