#!/usr/bin/env python
"""Post-hoc Tier 1 / Tier 2 analyses of the public 360 submission run.

Reads the original run read-only and writes only below ``--run-root``. Phases:
  guard     sha256 manifest of the original run (``--verify`` re-checks it)
  tier1     saved-score analyses: headline check, team/opponent/two-way inference,
            wild cluster bootstrap, status relabel, robustness diagnostic column,
            null-offset diagnosis (no refits)
  features  versioned Group A/B context table for one cohort
  compare   prespecified adjustment / re-selected population comparisons
  report    markdown + CSV from saved outputs only
  f3-supplement  Euro F3 shot retention (P0-supported basis) and primary overlap from saved artifacts
Every phase refuses to overwrite an existing output; use a new --run-root to repeat a phase.
World Cup comparisons are post-hoc exploratory and are labelled as such.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from PublicCausal import dependence as dep  # noqa: E402
from PublicCausal.notebook_runtime import atomic_write_json, json_safe  # noqa: E402

NOTEBOOK = ROOT / 'results/models/public_statsbomb_causal_study.ipynb'
DATA_ROOT_DEFAULT = ROOT.parents[2] if (ROOT.parents[2] / 'raw').exists() else None
POST_HOC_LABEL = 'post_hoc_exploratory'
WCB_SEED = 20261001
WCB_DRAWS = 9999


# ----------------------------------------------------------------------------- utilities

def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def check_separate(run_root, original):
    run_root, original = Path(run_root).resolve(), Path(original).resolve()
    if run_root == original or original in run_root.parents or run_root in original.parents:
        raise PermissionError(f'Run root {run_root} must be disjoint from the original run {original}')
    return run_root, original


class EventLog:
    def __init__(self, run_root):
        self.path = Path(run_root) / 'logs' / 'events.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, name, **fields):
        record = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), event=name, **fields)
        with open(self.path, 'a') as handle:
            handle.write(json.dumps(json_safe(record), sort_keys=True) + '\n')
        print(json.dumps(json_safe(record), sort_keys=True), flush=True)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, json_safe(payload))
    return path


def refuse_existing(paths):
    """Results are never deleted or overwritten: stop before computing when any output exists."""
    found = [str(p) for p in paths if Path(p).exists()]
    if found:
        raise FileExistsError('Outputs already exist and are never overwritten: ' + ', '.join(found)
                              + '. Use a new --run-root (or move the files by hand).')


def write_once_json(path, payload):
    # pragmatic: check-then-write is not atomic across processes; one runner per run root is the contract.
    refuse_existing([path])
    return write_json(path, payload)


def utc_stamp():
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())


def label_for(cohort):
    return POST_HOC_LABEL if cohort == 'confirmation' else 'development_post_hoc'


def load_core(run_root):
    from PublicCausal.notebook_runtime import load_notebook_core
    return load_notebook_core(NOTEBOOK, Path(run_root) / 'runtime')


# ----------------------------------------------------------------------------- guard

def original_manifest(original):
    original = Path(original)
    files = {}
    for path in sorted(p for p in original.rglob('*') if p.is_file() and '__pycache__' not in p.parts):
        files[str(path.relative_to(original))] = dict(sha256=sha256_file(path), bytes=path.stat().st_size)
    return dict(original=str(original), files=files, count=len(files))


def phase_guard(args, log):
    target = Path(args.run_root) / 'original_manifest.json'
    current = original_manifest(args.original)
    if args.verify:
        saved = json.loads(target.read_text())
        same = saved['files'] == current['files']
        diff = sorted(set(saved['files']) ^ set(current['files'])) + sorted(
            k for k in set(saved['files']) & set(current['files']) if saved['files'][k] != current['files'][k])
        write_once_json(Path(args.run_root) / f'original_manifest_verify-{utc_stamp()}.json',
                        dict(identical=same, differences=diff[:200], checked_files=current['count']))
        log('guard_verify', identical=same, files=current['count'], differences=len(diff))
        if not same:
            raise SystemExit('Original run changed: ' + ', '.join(diff[:20]))
        return
    if target.exists():
        raise FileExistsError(f'{target} exists; use --verify')
    write_once_json(target, current)
    log('guard_written', files=current['count'])


# ----------------------------------------------------------------------------- tier 1

CHECK_LABELS = {
    'support': 'Support, balance and concentration gates passed on the WC supported sample',
    'inference_agreement': 'Score interval and bootstrap percentile interval agreed under the pre-declared rule',
    'bootstrap': 'Requested fixed-design refits completed (diagnostic-poor draws retained)',
    'precision': 'Monte Carlo precision rule for bootstrap endpoints met',
    'nulls': 'Software null experiments ran; fitted mean within tolerance (software check, not identification evidence)',
    'simulations': 'Every simulation scenario produced successful replicates (no coverage criterion)',
    'analyses': 'All pre-declared robustness analyses ran without computational failure (not that they passed diagnostics)',
}


def _registries(eligible):
    return dict(original_match_id=sorted(eligible.original_match_id.astype(str).unique()),
                team_id=sorted(eligible.team_id.astype(str).unique()),
                opponent_id=sorted(eligible.opponent_id.astype(str).unique()))


def dependence_table(rows, registries, draws=WCB_DRAWS, seed=WCB_SEED, bootstrap=True):
    out = []
    for name, column in [('match', 'original_match_id'), ('team', 'team_id'), ('opponent', 'opponent_id')]:
        record = dep.cluster_se(rows, column, registries.get(column))
        record['clustering'] = name
        if bootstrap:
            wcb = dep.wild_cluster_bootstrap(rows, column, registries.get(column), draws=draws, seed=seed)
            record.update(wcb_interval=wcb['interval'], wcb_pvalue_zero=wcb['pvalue_zero'],
                          wcb_open_ended=wcb['interval_open_ended'])
        out.append(record)
    cgm = dep.two_way_cgm(rows, registries={'team_id': registries.get('team_id'),
                                            'opponent_id': registries.get('opponent_id')})
    cgm.update(clustering='two_way_team_opponent_cgm', clusters=cgm['components']['first'])
    out.append(cgm)
    return out


TIER1_OUTPUTS = ('headline_check.json', 'dependence.json', 'dependence.csv', 'status_relabel.json',
                 'robustness_with_diagnostics.csv', 'robustness_with_diagnostics.json', 'null_offset_replicates.csv',
                 'null_offset.json', 'match_stratified.json', 'euro_vs_wc.json')


def phase_tier1(args, log):
    import numpy as np
    import pandas as pd
    import joblib
    original, out = Path(args.original), Path(args.run_root) / 'tier1'
    refuse_existing(out / name for name in TIER1_OUTPUTS)
    exports = original / 'confirmation' / 'exports'
    rows = pd.read_parquet(exports / 'att_contributions' / 'rows.parquet')
    eligible = pd.read_parquet(exports / 'nuisance_rows.parquet', columns=['original_match_id', 'team_id', 'opponent_id', 'eligible'])
    eligible = eligible.loc[eligible.eligible.astype(bool)]
    registries = _registries(eligible)
    att = json.loads((exports / 'att.json').read_text())
    results = json.loads((original / 'confirmation' / 'submission_results.json').read_text())
    saved_att = att.get('att', results.get('point', {}).get('att'))
    saved_se = att.get('cluster_score_se', results.get('point', {}).get('score_se'))

    match = dep.cluster_se(rows, 'original_match_id', registries['original_match_id'])
    headline = dict(recomputed_att=match['att'], recomputed_se=match['se'], saved_att=saved_att, saved_se=saved_se,
                    att_abs_diff=abs(match['att'] - saved_att), se_abs_diff=abs(match['se'] - saved_se),
                    clusters=match['clusters'])
    headline['passed'] = bool(headline['att_abs_diff'] <= 1e-9 and headline['se_abs_diff'] <= 1e-9)
    write_once_json(out / 'headline_check.json', headline)
    log('tier1_headline', **headline)
    if not headline['passed']:
        raise SystemExit('Saved scores do not reproduce the headline estimate')

    table = dependence_table(rows, registries)
    write_once_json(out / 'dependence.json', dict(rows=table, nuisance_fixed=True, wcb_draws=WCB_DRAWS, wcb_seed=WCB_SEED,
                                             note='All intervals treat fitted nuisances as fixed (as the primary score CI).'))
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in table]).to_csv(out / 'dependence.csv', index=False)
    log('tier1_dependence', clusterings=[(r['clustering'], r['se'], r['interval']) for r in table])

    claim = results['claim']
    boot = results['bootstrap']
    status = [dict(check=k, original_flag=v, meaning=CHECK_LABELS.get(k, k)) for k, v in claim['checks'].items()]
    relabel = dict(original_status=claim['status'],
                   relabelled_status='all pre-declared computations completed; this is not a validation of identification',
                   checks=status,
                   bootstrap=dict(successful=boot.get('attempted'), diagnostic_poor=boot.get('diagnostic_poor'),
                                  percentile_interval=boot.get('percentile_interval')),
                   primary_status_note='confirmation/primary_status.json in the original run is a stale provisional '
                                       'record superseded by submission_results.json (not edited).')
    write_once_json(out / 'status_relabel.json', relabel)

    analyses = json.loads((exports / 'analyses.json').read_text())['records']
    robust = []
    for r in analyses:
        diag = r.get('diagnostics') or {}
        interval = r.get('score_interval') or [None, None]
        robust.append(dict(kind=r.get('kind'), name=r.get('name'), status=r.get('status'), att=r.get('att'),
                           ci_low=interval[0], ci_high=interval[1], n_shots=r.get('n_shots'),
                           control_ess=r.get('control_ess'), diagnostics_passed=diag.get('passed'),
                           key_max_smd=diag.get('key_max_smd'), other_max_smd=diag.get('other_max_smd'),
                           shot_retention=diag.get('shot_retention'), population=r.get('population'),
                           interval_excludes_zero=(interval[0] is not None and (interval[0] > 0 or interval[1] < 0))))
    pd.DataFrame(robust).to_csv(out / 'robustness_with_diagnostics.csv', index=False)
    write_once_json(out / 'robustness_with_diagnostics.json', robust)

    nulls = json.loads((exports / 'nulls.json').read_text())
    within = nulls['within_match_treatment']['records']
    paired = dep.paired_null_summary(within)
    core = load_core(args.run_root)
    fit = joblib.load(original / 'confirmation' / 'fit.joblib')
    baseline = fit['rows']
    per_rep = []
    for record in within:
        if record.get('status') != 'ok':
            continue
        permuted = core.submission_null_frame(baseline, 'within_match_treatment', record['seed'])
        permuted['support'] = baseline['support'].to_numpy()
        permuted['m0'] = baseline['m0'].to_numpy()
        plugin = dep.match_heterogeneity_plugin(permuted)
        per_rep.append(dict(replicate=record['replicate'], seed=record['seed'], fitted=record['att'],
                            known=record['known_assignment_att'], difference=record['att'] - record['known_assignment_att'],
                            residual_plugin=plugin['residual_plugin'], outcome_plugin=plugin['outcome_plugin']))
    frame = pd.DataFrame(per_rep)
    corr = (float(np.corrcoef(frame.difference, frame.residual_plugin)[0, 1]) if len(frame) > 2 else None)
    corr_fitted = (float(np.corrcoef(frame.fitted, frame.residual_plugin)[0, 1]) if len(frame) > 2 else None)
    frame.to_csv(out / 'null_offset_replicates.csv', index=False)
    write_once_json(out / 'null_offset.json', dict(
        paired=paired, replicates=len(frame), corr_difference_vs_plugin=corr, corr_fitted_vs_plugin=corr_fitted,
        mean_residual_plugin=float(frame.residual_plugin.mean()) if len(frame) else None,
        mechanism='Within-match permutation keeps each match shot share; fitted propensity/outcome models have no '
                  'match term, so match-level outcome differences can enter the estimate. Plug-in uses the original '
                  'baseline m0 and support (approximation; not the refitted nuisances).',
        dummy_y=dict(mean_att=nulls['independent_dummy_y'].get('mean_att'),
                     mcse=nulls['independent_dummy_y'].get('mean_att_mcse'))))
    log('tier1_null_offset', paired=paired, corr=corr, corr_fitted=corr_fitted)
    stratified = dep.match_stratified_att(rows)
    write_once_json(out / 'match_stratified.json', dict(stratified, label='exploratory saved-score sensitivity motivated by '
                                                   'the null offset diagnosis; not prespecified'))
    log('tier1_match_stratified', **stratified)

    dev_att = json.loads((original / 'development' / 'exports' / 'att.json').read_text())
    write_once_json(out / 'euro_vs_wc.json', dict(
        euro_development=dict(att=dev_att.get('att'), interval=dev_att.get('score_interval'), se=dev_att.get('cluster_score_se'),
                              n_shots=dev_att.get('n_shots'), matches=dev_att.get('n_matches')),
        world_cup=dict(att=saved_att, interval=att.get('score_interval'), se=saved_se, n_shots=att.get('n_shots'),
                       matches=att.get('n_matches'))))
    log('tier1_done')


# ----------------------------------------------------------------------------- tier 2

PRESPEC = ROOT / 'docs' / 'public_360_tier2_prespecification.md'


def data_root(args):
    root = Path(args.data_root) if args.data_root else Path(args.original).parents[2]
    if not (root / 'raw').exists():
        raise FileNotFoundError(f'No raw data below {root}')
    return root


def phase_prespec(args, log):
    target = Path(args.run_root) / 'prespec.sha256'
    digest = sha256_file(PRESPEC)
    if target.exists():
        saved = json.loads(target.read_text())
        if saved['sha256'] != digest:
            raise SystemExit('Prespecification changed after it was recorded; record a deviation instead')
        log('prespec_unchanged', sha256=digest)
        return
    write_json(target, dict(sha256=digest, path=str(PRESPEC), recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    log('prespec_recorded', sha256=digest)


def require_prespec(run_root):
    target = Path(run_root) / 'prespec.sha256'
    if not target.exists():
        raise PermissionError('Record the prespecification first (--phase prespec)')
    if json.loads(target.read_text())['sha256'] != sha256_file(PRESPEC):
        raise PermissionError('Prespecification file differs from the recorded hash')


def canonical_rows(args, cohort):
    from PublicCausal.data import build_decisions
    lock = Path(args.original) / 'development' / 'design_lock.json' if cohort == 'confirmation' else None
    frame = build_decisions(data_root(args), cohort, design_lock=lock)
    return frame, frame.attrs['canonical_key']


def feature_paths(run_root, cohort):
    pointer = Path(run_root) / 'features' / f'{cohort}.current.json'
    if not pointer.exists():
        raise FileNotFoundError(f'Build features first: {pointer}')
    meta = json.loads(pointer.read_text())
    return Path(run_root) / 'features' / meta['table'], meta


def phase_features(args, log):
    from PublicCausal import context_features as cf
    frame, canonical_key = canonical_rows(args, args.cohort)
    eligible = frame.loc[frame.eligible.astype(bool)].copy()
    table, meta = cf.build_context_table(data_root(args), eligible, canonical_key)
    out = Path(args.run_root) / 'features'
    name = f'{args.cohort}-{cf.FEATURE_VERSION}-{meta["key"][:16]}.parquet'
    path = out / name
    summary = dict(rows=len(table), status_by_arm=table.merge(eligible[['original_match_id', 'event_id', 'T']],
                   on=['original_match_id', 'event_id']).groupby(['T', 'ctx_in_status']).size().to_dict())
    pointer = out / f'{args.cohort}.current.json'
    pointer_payload = json.loads(json.dumps(json_safe(dict(
        table=name, key=meta['key'], canonical_key=canonical_key,
        summary={str(k): v for k, v in summary['status_by_arm'].items()}, rows=summary['rows'])), sort_keys=True))
    pointer_same = pointer.exists() and json.loads(pointer.read_text()) == pointer_payload
    if pointer.exists() and not pointer_same:
        raise FileExistsError(f'{pointer} exists with different content; outputs are never overwritten. '
                              'Use a new --run-root (or move the files by hand).')
    if path.exists():
        saved = json.loads(path.with_suffix('.meta.json').read_text())
        if saved['key'] != meta['key']:
            raise FileExistsError(f'{path} exists with another key')
        log('features_reused', cohort=args.cohort, key=meta['key'])
    else:
        out.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        table.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        meta['sha256'] = sha256_file(path)
        write_once_json(path.with_suffix('.meta.json'), meta)
    if not pointer_same:
        write_once_json(pointer, pointer_payload)
    log('features_done', cohort=args.cohort, key=meta['key'], rows=len(table))


def join_features(rows, table, columns):
    key = rows['original_match_id'].astype(str) + '|' + rows['event_id'].astype(str)
    indexed = table.assign(_k=table['original_match_id'].astype(str) + '|' + table['event_id'].astype(str)).set_index('_k')
    if indexed.index.duplicated().any():
        raise ValueError('Duplicate feature keys')
    missing = ~key.isin(indexed.index)
    if missing.any():
        raise ValueError(f'{int(missing.sum())} decision rows have no context features')
    out = rows.copy()
    for column in columns:
        out[column] = indexed.loc[key, column].to_numpy()
    return out


def _top_share(weights, fraction=.01):
    import numpy as np
    w = np.sort(np.asarray(weights, float))[::-1]
    k = max(1, int(round(fraction * len(w))))
    return float(w[:k].sum() / w.sum()) if w.sum() > 0 else None


def shot_retention(n_shots, reference_shots):
    """Shots kept relative to the P0-supported shots of the cohort (the prespecified denominator)."""
    return (n_shots / reference_shots) if reference_shots else None


def supported_shot_ids(rows):
    return set(rows.loc[rows.support.astype(bool) & (rows['T'] == 1), 'decision_row_id'].astype(str))


def shot_overlap(primary_rows, selected_rows):
    primary, selected = supported_shot_ids(primary_rows), supported_shot_ids(selected_rows)
    return dict(primary_shots=len(primary), selected_shots=len(selected), both=len(primary & selected),
                jaccard=len(primary & selected) / max(1, len(primary | selected)))


def f3_record(core, selected_rows, primary_rows, cfg, features, registries, reference_shots):
    """F3 estimate and its overlap with the primary supported shots; one path for both cohorts."""
    return dict(estimate=estimate_record(core, selected_rows, cfg, features, features, registries, reference_shots),
                overlap=shot_overlap(primary_rows, selected_rows))


def estimate_record(core, rows, cfg, features, union, registries, reference_shots=None):
    import numpy as np
    from sklearn.metrics import log_loss, roc_auc_score
    est = core.att_estimate(rows)
    own = core.support_diagnostics(rows, dict(cfg, feature_names=list(features)))
    joint = core.support_diagnostics(rows, dict(cfg, feature_names=list(union)))
    sample = est['rows']
    controls = sample.loc[sample['T'] == 0]
    shots = sample.loc[sample['T'] == 1]
    supported = rows.loc[rows.support.astype(bool)]
    dependence = {r['clustering']: dict(se=r['se'], interval=r['interval'])
                  for r in dependence_table(sample.assign(eligible=True), registries, bootstrap=False)}
    record = dict(att=est['att'], interval=est['score_interval'], se=est['cluster_score_se'], df=est['cluster_score_df'],
                  dependence=dependence, n_shots=est['n_shots'], n_controls=est['n_controls'],
                  shot_retention=shot_retention(est['n_shots'], reference_shots),
                  control_ess=est['control_ess'], max_weight=float(controls.att_weight.max()),
                  max_weight_share=est['max_donor_weight_share'], max_match_weight_share=est['max_match_weight_share'],
                  top1pct_control_weight_share=_top_share(controls.att_weight),
                  key_max_smd_own=own['key_max_smd'], other_max_smd_own=own['other_max_smd'],
                  key_max_smd_union=joint['key_max_smd'], other_max_smd_union=joint['other_max_smd'],
                  gates_own=own.get('gates'), passed_own=own.get('passed'), passed_union=joint.get('passed'),
                  positive_shot_goals=int((shots.Y == 1).sum()), negative_shot_goals=int((shots.Y == -1).sum()),
                  features=list(features))
    try:
        record.update(oof_logloss=float(log_loss(supported.Treatment, supported.e.clip(1e-6, 1 - 1e-6))),
                      oof_auc=float(roc_auc_score(supported.Treatment, supported.e)))
    except ValueError as exc:
        record['oof_metric_error'] = repr(exc)
    return record


def fixed_population_comparison(core, rows, cfg, frozen_specs, arms, registries, reference_shots, log=None, label=None):
    """Refit e and m0 for every arm on identical rows/support/folds; paired differences vs the first arm."""
    union = list(dict.fromkeys(name for _, features in arms for name in features))
    fitted, records, differences = {}, {}, {}
    for name, features in arms:
        try:
            answer, _ = core.refit_on_fixed_population(rows, dict(cfg, feature_names=list(features)),
                                                       frozen_specs=frozen_specs)
            fitted[name] = answer
            records[name] = dict(status='ok', **estimate_record(core, answer, cfg, features, union, registries, reference_shots))
        except Exception as exc:  # recorded, never retried with other specs
            records[name] = dict(status='failed', error=repr(exc), category='structural' if isinstance(exc, ValueError) else 'computational')
        if log:
            log('arm_done', comparison=label, arm=name, status=records[name]['status'], att=records[name].get('att'))
    base = arms[0][0]
    for name, _ in arms[1:]:
        if base in fitted and name in fitted:
            out = {}
            for cluster in ['original_match_id', 'team_id', 'opponent_id']:
                out[cluster] = dep.paired_difference(fitted[base], fitted[name], cluster=cluster,
                                                     registry=registries.get(cluster))
            differences[f'{name}_minus_{base}'] = out
    return records, differences, fitted


def _saved_config(original):
    path = Path(original) / 'confirmation' / 'config.json'
    config = json.loads(path.read_text()) if path.exists() else {}
    config = config.get('config', config)
    return {k: v for k, v in config.items() if k in {
        'profile', 'seed', 'outer_folds', 'inner_folds', 'support_thresholds', 'donor_min', 'donor_match_min',
        'min_shots', 'min_matches', 'min_ess', 'ess_shot_fraction', 'key_smd_max', 'other_smd_max',
        'max_weight_share', 'max_match_weight_share', 'candidate_device', 'outcome_column'}}


COMPARE_OUTPUTS = {'development': ('summary.json', 'partial_F1.json', 'partial_F2.json', 'design_A.joblib'),
                   'confirmation': ('summary.json', 'partial_F1.json', 'partial_F2.json')}


def phase_compare(args, log):
    import joblib
    import numpy as np
    from PublicCausal import context_features as cf
    require_prespec(args.run_root)
    cohort = args.cohort
    table_path, meta = feature_paths(args.run_root, cohort)
    out = Path(args.run_root) / cohort / meta['key'][:12]
    refuse_existing([out / name for name in COMPARE_OUTPUTS[cohort]])
    label = label_for(cohort)
    if cohort == 'confirmation':
        dev_pointer = Path(args.run_root) / 'features' / 'development.current.json'
        dev_key = json.loads(dev_pointer.read_text())['key'] if dev_pointer.exists() else None
        if not dev_key or not (Path(args.run_root) / 'development' / dev_key[:12] / 'summary.json').exists():
            raise PermissionError('Run the Euro development comparison before the World Cup')
    import pandas as pd
    table = pd.read_parquet(table_path)
    if sha256_file(table_path) != json.loads(table_path.with_suffix('.meta.json').read_text())['sha256']:
        raise ValueError('Feature table checksum mismatch')
    core = load_core(args.run_root)
    phase_dir = 'development' if cohort == 'development' else 'confirmation'
    fit = joblib.load(Path(args.original) / phase_dir / 'fit.joblib')
    frozen = joblib.load(Path(args.original) / 'development' / 'frozen_design.joblib')
    cfg = core.scientific_config(dict(_saved_config(args.original), profile='submission', att_only=True,
                                      diagnostic_m1=False, workers=1))
    base = list(cfg['feature_names'])
    group_a, group_b = list(cf.GROUP_A), list(cf.GROUP_B)
    extra = group_a + group_b + cf.OBSERVATION + ['ctx_in_status', 'ctx_restart_origin']
    rows = join_features(fit['rows'], table, extra)
    rows['geo_triangle_visible'] = rows['geo_triangle_visible'].astype(bool)
    rows['geo_keeper_observed'] = rows['geo_keeper_observed'].astype(bool)
    eligible = rows.loc[rows.eligible.astype(bool)] if 'eligible' in rows else rows
    registries = _registries(eligible)
    reference = int((rows.support.astype(bool) & (rows['T'] == 1)).sum())
    log('compare_start', cohort=cohort, label=label, rows=len(rows), reference_supported_shots=reference,
        feature_key=meta['key'])
    summary = dict(cohort=cohort, label=label, feature_key=meta['key'], feature_table_sha256=sha256_file(table_path),
                   prespec_sha256=sha256_file(PRESPEC), frozen_specs=frozen['model_specs'], config=cfg,
                   reference_supported_shots=reference)
    summary['linkage_by_arm'] = {f'T={k[0]}|{k[1]}': int(v) for k, v in rows.groupby(['T', 'ctx_in_status']).size().items()}

    f1, d1, fitted1 = fixed_population_comparison(core, rows, cfg, frozen['model_specs'],
                                                  [('baseline', base), ('plus_A', base + group_a)],
                                                  registries, reference, log, 'F1')
    summary['F1'] = dict(sample='P0 eligible rows, target S0', arms=f1, differences=d1)
    if cohort == 'confirmation' and 'baseline' in fitted1:
        saved = core.att_estimate(fit['rows'])['att']
        summary['baseline_refit_check'] = dict(saved_att=saved, refit_att=f1['baseline'].get('att'),
                                               abs_diff=abs(saved - f1['baseline']['att']),
                                               within_tolerance=bool(abs(saved - f1['baseline']['att']) <= 1e-4))
    write_once_json(out / 'partial_F1.json', summary)

    visible = rows['geo_triangle_visible'] & rows['geo_keeper_observed']
    summary['V_membership'] = {f'T={t}|supported={s}': dict(rows=int(((rows['T'] == t) & (rows.support.astype(bool) == s)).sum()),
                                                           in_V=int(((rows['T'] == t) & (rows.support.astype(bool) == s) & visible).sum()))
                               for t in [0, 1] for s in [True, False]}
    subset = rows.loc[visible].copy().reset_index(drop=True)
    reference_v = int((subset.support.astype(bool) & (subset['T'] == 1)).sum())
    f2, d2, _ = fixed_population_comparison(core, subset, cfg, frozen['model_specs'],
                                            [('baseline', base), ('plus_A', base + group_a),
                                             ('plus_A_B', base + group_a + group_b)],
                                            registries, reference_v, log, 'F2')
    summary['F2'] = dict(sample='V (triangle visible and keeper observed) within P0; nuisances trained on V only',
                         reference_supported_shots_V=reference_v, arms=f2, differences=d2)
    write_once_json(out / 'partial_F2.json', summary)

    cfg_a = dict(cfg, feature_names=base + group_a, support_features=list(cfg['support_features']) + group_a)
    design_path = out / 'design_A.joblib'
    try:
        if cohort == 'development':
            canonical, _ = canonical_rows(args, 'development')
            canonical = join_features(canonical.loc[canonical.eligible.astype(bool)], table, group_a)
            design = core.fit_frozen_design(canonical, cfg_a)
            design_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(design, design_path)
            selection = design['development_selection']
            sel_rows = selection['rows']
            f3 = f3_record(core, sel_rows, fit['rows'], cfg_a, base + group_a, _registries(sel_rows), reference)
            summary['F3'] = dict(status='ok', role='Euro re-selected design (all-Euro OOF; not held out)',
                                 model_specs=design['model_specs'], domain_threshold=design['domain']['threshold'],
                                 domain_cells=len(design['domain']['cells']),
                                 selection_passed=design['domain'].get('selection_passed'), **f3)
        else:
            dev_key = json.loads((Path(args.run_root) / 'features' / 'development.current.json').read_text())['key']
            design = joblib.load(Path(args.run_root) / 'development' / dev_key[:12] / 'design_A.joblib')
            canonical, _ = canonical_rows(args, 'confirmation')
            canonical = join_features(canonical.loc[canonical.eligible.astype(bool)], table, group_a)
            fitted = core.fit_study(canonical, dict(cfg_a, att_only=True, diagnostic_m1=False), frozen_design=design)
            new_rows = fitted['rows']
            f3 = f3_record(core, new_rows, fit['rows'], cfg_a, base + group_a, registries, reference)
            summary['F3'] = dict(status='ok', role='WC with Euro re-selected design (post-hoc exploratory)',
                                 model_specs=design['model_specs'], **f3,
                                 fit_diagnostics_passed=fitted['diagnostics'].get('passed'))
    except Exception as exc:
        summary['F3'] = dict(status='failed', error=repr(exc))
    log('F3_done', cohort=cohort, status=summary['F3']['status'], att=summary['F3'].get('estimate', {}).get('att'))
    summary['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    write_once_json(out / 'summary.json', summary)
    log('compare_done', cohort=cohort, out=str(out))


def _fmt(value, digits=4):
    return '' if value is None else (f'{value:.{digits}f}' if isinstance(value, float) else str(value))


REPORT_OUTPUTS = ('tier2_estimates.csv', 'tier2_estimates.md')


def phase_report(args, log):
    import pandas as pd
    run_root = Path(args.run_root)
    refuse_existing(run_root / 'report' / name for name in REPORT_OUTPUTS)
    lines, rows = ['# Tier 1 / Tier 2 results (generated from saved outputs)', ''], []
    for cohort in ['development', 'confirmation']:
        pointer = run_root / 'features' / f'{cohort}.current.json'
        if not pointer.exists():
            continue
        key = json.loads(pointer.read_text())['key'][:12]
        path = run_root / cohort / key / 'summary.json'
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        for block in ['F1', 'F2']:
            for arm, record in summary.get(block, {}).get('arms', {}).items():
                rows.append(dict(cohort=cohort, label=summary['label'], comparison=block, arm=arm, status=record.get('status'),
                                 att=record.get('att'), ci_low=(record.get('interval') or [None, None])[0],
                                 ci_high=(record.get('interval') or [None, None])[1], n_shots=record.get('n_shots'),
                                 shot_retention=record.get('shot_retention'), control_ess=record.get('control_ess'),
                                 top1pct=record.get('top1pct_control_weight_share'),
                                 key_smd_union=record.get('key_max_smd_union'), other_smd_union=record.get('other_max_smd_union'),
                                 passed_own=record.get('passed_own'), oof_auc=record.get('oof_auc'),
                                 se_opponent=record.get('dependence', {}).get('opponent', {}).get('se'),
                                 se_two_way=record.get('dependence', {}).get('two_way_team_opponent_cgm', {}).get('se')))
            for name, diff in summary.get(block, {}).get('differences', {}).items():
                d = diff['original_match_id']
                rows.append(dict(cohort=cohort, label=summary['label'], comparison=block, arm=name, status='difference',
                                 att=d['difference'], ci_low=d['interval'][0], ci_high=d['interval'][1]))
        f3 = summary.get('F3', {})
        est = f3.get('estimate', {})
        rows.append(dict(cohort=cohort, label=summary['label'], comparison='F3', arm='reselected_plus_A', status=f3.get('status'),
                         att=est.get('att'), ci_low=(est.get('interval') or [None, None])[0],
                         ci_high=(est.get('interval') or [None, None])[1], n_shots=est.get('n_shots'),
                         shot_retention=est.get('shot_retention'), control_ess=est.get('control_ess'),
                         top1pct=est.get('top1pct_control_weight_share'), key_smd_union=est.get('key_max_smd_union'),
                         other_smd_union=est.get('other_max_smd_union'), passed_own=est.get('passed_own'),
                         oof_auc=est.get('oof_auc')))
    frame = pd.DataFrame(rows)
    out = run_root / 'report'
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'tier2_estimates.csv', index=False)
    if len(frame):
        try:
            lines.append(frame.to_markdown(index=False, floatfmt='.4f'))
        except ImportError:  # tabulate is optional and absent from the pinned environment
            lines.append('```\n' + frame.to_string(index=False, float_format=lambda v: f'{v:.4f}') + '\n```')
    (out / 'tier2_estimates.md').write_text('\n'.join(lines) + '\n')
    log('report_done', rows=len(frame))


def phase_f3_supplement(args, log):
    """Corrected F3 shot retention and primary overlap for a completed development comparison.

    Reads only saved artifacts (no refit): the saved Euro design carries the F3 selection rows, including
    the support flags, so the selected supported shots are recovered exactly. The saved F3 ATT and shot count
    must be reproduced from those rows before the single new file is written.
    """
    import joblib
    import tempfile
    require_prespec(args.run_root)
    cohort = args.cohort
    if cohort != 'development':
        raise ValueError('f3-supplement supports the development cohort only: the World Cup F3 rows are not saved '
                         'and its record already uses the P0-supported basis and overlap')
    _, meta = feature_paths(args.run_root, cohort)
    out = Path(args.run_root) / cohort / meta['key'][:12]
    target = out / 'f3_supplement.json'
    refuse_existing([target])
    summary_path, design_path = out / 'summary.json', out / 'design_A.joblib'
    fit_path = Path(args.original) / 'development' / 'fit.joblib'
    summary = json.loads(summary_path.read_text())
    saved = summary['F3']
    if saved.get('status') != 'ok':
        raise ValueError('Saved F3 did not complete; nothing to supplement')
    with tempfile.TemporaryDirectory() as runtime:  # the core export must not write into the run root
        from PublicCausal.notebook_runtime import load_notebook_core
        core = load_notebook_core(NOTEBOOK, runtime)
        selected = joblib.load(design_path)['development_selection']['rows']
        primary = joblib.load(fit_path)['rows']
        est = core.att_estimate(selected)
    reference = len(supported_shot_ids(primary))
    checks = dict(saved_att=saved['estimate']['att'], recomputed_att=est['att'],
                  att_abs_diff=abs(est['att'] - saved['estimate']['att']),
                  saved_n_shots=saved['estimate']['n_shots'], recomputed_n_shots=est['n_shots'],
                  saved_reference_supported_shots=summary['reference_supported_shots'],
                  recomputed_reference_supported_shots=reference)
    checks['passed'] = bool(checks['att_abs_diff'] <= 1e-9 and checks['saved_n_shots'] == checks['recomputed_n_shots']
                            and checks['saved_reference_supported_shots'] == reference)
    if not checks['passed']:
        raise ValueError(f'Saved artifacts do not reproduce the saved F3 record: {checks}')
    payload = dict(
        kind='f3_supplement', cohort=cohort, label=label_for(cohort), feature_key=meta['key'],
        path='saved_artifacts_no_refit',
        retention=dict(basis='P0-supported treated rows of development/fit.joblib (prespecified denominator)',
                       n_shots=est['n_shots'], reference_supported_shots=reference,
                       shot_retention=shot_retention(est['n_shots'], reference),
                       previously_reported=saved['estimate'].get('shot_retention'),
                       previous_basis='all eligible treated rows of the re-selected sample'),
        overlap=shot_overlap(primary, selected), checks=checks,
        inputs={name: dict(path=str(path), sha256=sha256_file(path))
                for name, path in [('summary', summary_path), ('design_A', design_path), ('primary_fit', fit_path)]},
        script_sha256=sha256_file(Path(__file__)), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write_once_json(target, payload)
    log('f3_supplement_done', cohort=cohort, out=str(target), shot_retention=payload['retention']['shot_retention'],
        jaccard=payload['overlap']['jaccard'])


# ----------------------------------------------------------------------------- main

PHASES = {'guard': phase_guard, 'tier1': phase_tier1, 'prespec': phase_prespec, 'features': phase_features,
          'compare': phase_compare, 'report': phase_report, 'f3-supplement': phase_f3_supplement}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', required=True, choices=sorted(PHASES))
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--original', required=True, help='original run phase dir (…/public-submission-full-gpu-20260930/full)')
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--cohort', choices=['development', 'confirmation'])
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    args.run_root, args.original = check_separate(args.run_root, args.original)
    args.run_root.mkdir(parents=True, exist_ok=True)
    log = EventLog(args.run_root)
    log('phase_start', phase=args.phase, cohort=args.cohort, pid=os.getpid())
    PHASES[args.phase](args, log)
    log('phase_end', phase=args.phase, cohort=args.cohort)


if __name__ == '__main__':
    main()
