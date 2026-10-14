"""Cost-sensitive fallback gate diagnostics on nested-clean OOF predictions."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import common
import rescore_mixture
from common import RUNS

CATS = ['APT', 'airline', 'apt_stand', 'apt_rwy', 'apt_pier', 'ades_region']
NUMS = [
    'p_fallback', 'atot_minus_sched', 'atot_minus_aobt3', 'hour_local', 'dow', 'doy',
    'flight_is_callsign', 'fb_rate_stand', 'fb_rate_airline_apt',
    'fb_rate_airline_apt_unm', 'fb_rate_stand_unm', 'fb_rate_apt_hour_unm',
    'arr_fb_rate_airline_apt', 'arr_fb_rate_airline_apt_month',
    'dep_prev15', 'dep_prev60', 'dep_same_rwy_prev30', 'n_out_at_aobt3',
    'disruption_z', 'normal_pred', 'expert_gap', 'abs_expert_gap', 'log_abs_expert_gap',
]
FEATURES = CATS + NUMS


def prepare_frame(oof, features, base_p_scale=1.0):
    d = oof.merge(features, on='MVT_ID_mvt', how='left', validate='one_to_one', suffixes=('', '_feature'))
    p_used = np.clip(d.p_fallback.values * float(base_p_scale), 0, 1)
    if 'normal_pred' in d and np.isfinite(d.normal_pred.values).all():
        expert = d.normal_pred.values.astype(float)
        valid = np.isfinite(d.atot_minus_sched.values) & (d.is_heli.values == 0)
    else:
        expert, valid = rescore_mixture.recover_expert(
            d.pred.values, p_used, d.atot_minus_sched.values, d.is_heli.values,
        )
    d['normal_pred'] = expert
    d['expert_gap'] = d.atot_minus_sched.values - expert
    d['abs_expert_gap'] = np.abs(d.expert_gap.values)
    d['log_abs_expert_gap'] = np.log1p(d.abs_expert_gap.values)
    d['gate_valid'] = valid & (d.fpl_matched.values == 0) & (d.abs_expert_gap.values >= 60)
    raw = np.divide(d.y.values - expert, d.expert_gap.values,
                    out=np.zeros(len(d)), where=np.abs(d.expert_gap.values) >= 60)
    d['optimal_weight'] = np.clip(raw, 0, 1)
    return d


def replace_clean_expert(d, normal_predictions):
    """Replace algebraic recovery with exact persisted-model predictions."""
    normal = pd.DataFrame(normal_predictions)
    if 'MVT_ID_mvt' not in normal or 'normal_pred' not in normal:
        raise ValueError('normal predictions require MVT_ID_mvt and normal_pred')
    aligned = d[['MVT_ID_mvt']].merge(
        normal[['MVT_ID_mvt', 'normal_pred']], on='MVT_ID_mvt', how='left', validate='one_to_one',
    )
    expert = aligned.normal_pred.values.astype(float)
    if not np.isfinite(expert).all():
        raise ValueError(f'non-finite exact clean expert predictions: {int((~np.isfinite(expert)).sum())}')
    d = d.copy()
    d['normal_pred'] = expert
    d['expert_gap'] = d.atot_minus_sched.values - expert
    d['abs_expert_gap'] = np.abs(d.expert_gap.values)
    d['log_abs_expert_gap'] = np.log1p(d.abs_expert_gap.values)
    valid = np.isfinite(d.atot_minus_sched.values) & (d.is_heli.values == 0)
    d['gate_valid'] = valid & (d.fpl_matched.values == 0) & (d.abs_expert_gap.values >= 60)
    return d


def _prep(train, score, features=FEATURES, cats=CATS):
    a = train[features].copy(); b = score[features].copy()
    for c in cats:
        a[c] = pd.Categorical(a[c].astype(str))
        b[c] = pd.Categorical(b[c].astype(str), categories=a[c].cat.categories)
    return a, b


def fit_predict_gate(train, score, weight_power=2.0, weight_quantile=0.995,
                     rounds=250, num_leaves=31, min_data_in_leaf=300,
                     features=FEATURES, cats=CATS, seed=42):
    fit = train[train.gate_valid].copy()
    X, Xt = _prep(fit, score, features, cats)
    raw_w = np.power(fit.abs_expert_gap.values, weight_power)
    cap = float(np.quantile(raw_w, weight_quantile)); weights = np.minimum(raw_w, cap)
    weights = weights / np.mean(weights)
    params = dict(objective='regression_l2', learning_rate=0.05, num_leaves=num_leaves,
                  min_data_in_leaf=min_data_in_leaf, feature_fraction=0.85,
                  bagging_fraction=0.85, bagging_freq=1, lambda_l2=20,
                  cat_smooth=80, max_cat_to_onehot=4, verbose=-1, seed=seed)
    model = lgb.train(params, lgb.Dataset(X, fit.optimal_weight.values, weight=weights,
                                         categorical_feature=cats), rounds)
    return np.clip(model.predict(Xt), 0, 1), model, dict(
        n_fit=len(fit), weight_cap=cap, params=params, rounds=rounds,
        features=list(features), categoricals=list(cats),
    )


def apply_gate(d, gate_weight, shrink=1.0, max_delta=None):
    """Apply gate weights, optionally capping movement from the base prediction.

    ``max_delta`` is an absolute per-row safety bound in seconds.  It is applied to
    the gate's proposed prediction relative to ``d.pred`` before the 30-second floor.
    """
    w = (1.0 - shrink) * d.p_fallback.values + shrink * np.asarray(gate_weight)
    w = np.clip(w, 0, 1); out = d.pred.values.copy(); use = d.gate_valid.values
    candidate = d.normal_pred.values + w * d.expert_gap.values
    if max_delta is not None:
        cap = float(max_delta)
        if not np.isfinite(cap) or cap <= 0:
            raise ValueError('max_delta must be a positive finite number or None')
        candidate = np.clip(candidate, d.pred.values - cap, d.pred.values + cap)
    out[use] = candidate[use]
    return np.maximum(out, 30.0)


def _metric_row(d, pred):
    m = common.metrics(pred, d.y.values, d.APT.values, d.month.values, d.fpl_matched.values)
    return {k: v for k, v in m.items() if k != 'rmse_by_apt_json'}


def cross_pair_diagnostic(run_id, feature_version=3, output=None):
    tr = pd.read_parquet(common.DATA / f'features_v{feature_version}_train.parquet')
    frames = {f: prepare_frame(pd.read_parquet(RUNS / 'oof' / f'{run_id}_fold{f}.parquet'), tr) for f in ('A', 'B')}
    report = {'run_id': run_id, 'feature_version': feature_version, 'directions': {}}
    for train_fold, test_fold in (('A', 'B'), ('B', 'A')):
        pred_w, _, meta = fit_predict_gate(frames[train_fold], frames[test_fold])
        rows = {}
        for shrink in (0.25, 0.5, 0.75, 1.0):
            rows[str(shrink)] = _metric_row(frames[test_fold], apply_gate(frames[test_fold], pred_w, shrink))
        report['directions'][f'{train_fold}_to_{test_fold}'] = {'fit': meta, 'metrics': rows}
    output = Path(output) if output else RUNS / f'tail_gate_{run_id}.json'
    output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    return report


def cross_pair_predictions(run_id, feature_version=3, shrink=1.0):
    """Return gate predictions for A and B, each fitted only on the other pair."""
    features = pd.read_parquet(common.DATA / f'features_v{feature_version}_train.parquet')
    frames = {f: prepare_frame(pd.read_parquet(RUNS / 'oof' / f'{run_id}_fold{f}.parquet'), features) for f in ('A', 'B')}
    predictions = {}
    for train_fold, test_fold in (('A', 'B'), ('B', 'A')):
        weight, _, _ = fit_predict_gate(frames[train_fold], frames[test_fold])
        predictions[test_fold] = apply_gate(frames[test_fold], weight, shrink)
    return frames, predictions


def multi_pair_diagnostic(fold_runs, feature_version=3, output=None,
                          shrinks=(0.25, 0.5, 0.75, 1.0)):
    """Leave one seasonal pair out, fitting each gate on the other five pairs."""
    folds = tuple(sorted(fold_runs))
    if folds != tuple('ABCDEF'):
        raise ValueError(f'fold_runs must contain A-F, got {folds}')
    features = pd.read_parquet(common.DATA / f'features_v{feature_version}_train.parquet')
    frames = {
        fold: prepare_frame(
            pd.read_parquet(RUNS / 'oof' / f'{fold_runs[fold]}_fold{fold}.parquet'), features,
        )
        for fold in folds
    }
    predictions = {str(s): {} for s in shrinks}
    report = {'feature_version': feature_version, 'fold_runs': fold_runs, 'folds': {}}
    for test_fold in folds:
        gate_train = pd.concat([frames[f] for f in folds if f != test_fold], ignore_index=True)
        weight, _, meta = fit_predict_gate(gate_train, frames[test_fold])
        fold_report = {'fit': meta, 'baseline': _metric_row(frames[test_fold], frames[test_fold].pred.values),
                       'metrics': {}}
        for shrink in shrinks:
            pred = apply_gate(frames[test_fold], weight, shrink)
            predictions[str(shrink)][test_fold] = pred
            fold_report['metrics'][str(shrink)] = _metric_row(frames[test_fold], pred)
        report['folds'][test_fold] = fold_report

    pooled = pd.concat([frames[f] for f in folds], ignore_index=True)
    report['pooled'] = {'baseline': _metric_row(pooled, pooled.pred.values), 'metrics': {}}
    for shrink in shrinks:
        pred = np.concatenate([predictions[str(shrink)][f] for f in folds])
        report['pooled']['metrics'][str(shrink)] = _metric_row(pooled, pred)
    output = Path(output) if output else RUNS / 'tail_gate_six_pair.json'
    output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    return report


def materialize_cross_pair_run(run_id, feature_version=3, shrink=1.0, approach_id='A14'):
    """Persist cross-pair OOF predictions and result rows for a gate approach."""
    frames, predictions = cross_pair_predictions(run_id, feature_version, shrink)
    new_id = common.new_run_id(f'tail_gate_s{str(shrink).replace(".", "p")}')
    source = pd.read_csv(RUNS / 'results.csv')
    for fold in ('A', 'B'):
        d = frames[fold]; pred = predictions[fold]
        m = common.metrics(pred, d.y.values, d.APT.values, d.month.values, d.fpl_matched.values)
        src = source[(source.run_id == run_id) & (source.fold == fold)].iloc[-1]
        params = {'source_run': run_id, 'shrink': float(shrink), 'gate_train_fold': 'B' if fold == 'A' else 'A'}
        row = dict(run_id=new_id, ts=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), feature_version=feature_version,
                   groups_on=src.groups_on, model='tail_gate', params_json=json.dumps(params), fold=fold,
                   rmse_jan=m.pop('rmse_m1'), rmse_jul=m.pop('rmse_m2'), n_train=int(src.n_train), n_test=len(d),
                   fit_seconds=0.0, notes=f'{approach_id}; cross-pair cost-weighted gate', seed=42, **m)
        common.append_result(row)
        d[['MVT_ID_mvt', 'APT', 'month', 'y', 'fpl_matched']].assign(pred=pred).to_parquet(RUNS / 'oof' / f'{new_id}_fold{fold}.parquet')
    return new_id, frames, predictions


def fit_full_gate(run_id, feature_version, base_submission, base_p_scale=1.4,
                  normal_predictions=None):
    """Fit one gate on cross-fitted A+B rows and predict weights for ranking rows."""
    train_features = pd.read_parquet(common.DATA / f'features_v{feature_version}_train.parquet')
    rank_features = pd.read_parquet(common.DATA / f'features_v{feature_version}_rank.parquet')
    train_frames = [prepare_frame(pd.read_parquet(RUNS / 'oof' / f'{run_id}_fold{f}.parquet'), train_features) for f in ('A', 'B')]
    gate_train = pd.concat(train_frames, ignore_index=True)
    base = pd.read_parquet(base_submission)[['MVT_ID_mvt', 'TAXITIME_SEC_mvt']]
    skeleton = rank_features[['MVT_ID_mvt', 'APT', 'month', 'fpl_matched']].merge(base, on='MVT_ID_mvt', validate='one_to_one')
    skeleton = skeleton.rename(columns={'TAXITIME_SEC_mvt': 'pred'}); skeleton['y'] = np.nan
    gate_rank = prepare_frame(skeleton, rank_features, base_p_scale=base_p_scale)
    if normal_predictions is not None:
        exact = pd.read_parquet(normal_predictions) if isinstance(normal_predictions, (str, Path)) else normal_predictions
        gate_rank = replace_clean_expert(gate_rank, exact)
    weight, model, meta = fit_predict_gate(gate_train, gate_rank)
    return gate_rank, weight, model, meta


def fit_full_gate_multi(fold_runs, feature_version, base_submission,
                        normal_predictions, base_p_scale=1.4):
    """Fit the ranking gate on cross-fitted OOF rows from all six seasonal pairs."""
    if tuple(sorted(fold_runs)) != tuple('ABCDEF'):
        raise ValueError('fold_runs must contain A-F')
    train_features = pd.read_parquet(common.DATA / f'features_v{feature_version}_train.parquet')
    rank_features = pd.read_parquet(common.DATA / f'features_v{feature_version}_rank.parquet')
    gate_train = pd.concat([
        prepare_frame(
            pd.read_parquet(RUNS / 'oof' / f'{fold_runs[fold]}_fold{fold}.parquet'),
            train_features,
        )
        for fold in sorted(fold_runs)
    ], ignore_index=True)
    base = pd.read_parquet(base_submission)[['MVT_ID_mvt', 'TAXITIME_SEC_mvt']]
    skeleton = rank_features[['MVT_ID_mvt', 'APT', 'month', 'fpl_matched']].merge(
        base, on='MVT_ID_mvt', validate='one_to_one',
    )
    skeleton = skeleton.rename(columns={'TAXITIME_SEC_mvt': 'pred'})
    skeleton['y'] = np.nan
    gate_rank = prepare_frame(skeleton, rank_features, base_p_scale=base_p_scale)
    exact = pd.read_parquet(normal_predictions) if isinstance(normal_predictions, (str, Path)) else normal_predictions
    gate_rank = replace_clean_expert(gate_rank, exact)
    weight, model, meta = fit_predict_gate(gate_train, gate_rank)
    meta['fold_runs'] = fold_runs
    return gate_rank, weight, model, meta
