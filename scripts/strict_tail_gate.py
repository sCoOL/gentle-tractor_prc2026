"""Blocked-pair gate diagnostic with no target-derived predictor crossing the block.

For each block (A/B, C/D, E/F), the base model and fallback detector train on the other
four seasonal pairs. They score both blocked pairs. One blocked pair calibrates the gate and
the other evaluates it, then direction is reversed. This sacrifices training volume to obtain
an auditable test in which neither calibration nor test targets influenced the base predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import common
import detectors
import tail_gate
import train_eval
from common import RUNS

BLOCKS = (('A', 'B'), ('C', 'D'), ('E', 'F'))
TARGET_ROUTE = {'pair_p10_period', 'route_period_id', 'pair_n_periods'}
TARGET_G9 = {'p_fallback', 'disruption_z', 'is_disruption', 'disruption_weather_explained'}
STRICT_GATE_FEATURES = [
    c for c in tail_gate.FEATURES
    if not c.startswith('fb_rate_') and not c.startswith('arr_fb_rate_') and c != 'disruption_z'
]
STRICT_GATE_CATS = [c for c in tail_gate.CATS if c in STRICT_GATE_FEATURES]


def _heli_const(train):
    rows = train[train.is_heli.values == 1]
    return float(rows.TAXITIME_SEC_mvt.mean()) if len(rows) else 120.0


def _metric(d, pred):
    m = common.metrics(pred, d.y.values, d.APT.values, d.month.values, d.fpl_matched.values)
    return {k: v for k, v in m.items() if k != 'rmse_by_apt_json'}


def build_block_predictions(left, right, feature_version=3, rounds=660,
                            fallback_rounds=400):
    data, _, manifest = common.load_features(feature_version)
    blocked_months = common.FOLDS[left] + common.FOLDS[right]
    base_train = data[~data.month.isin(blocked_months)].copy()
    score = data[data.month.isin(blocked_months)].copy()
    score['p_fallback'] = detectors.score_fallback(
        base_train, score, rounds=fallback_rounds,
    )
    target_cols = set(manifest['groups'].get('G8', [])) | TARGET_ROUTE | TARGET_G9
    cols = [c for c in train_eval.feature_columns(manifest, list(manifest['groups']))
            if c not in target_cols]
    cats = train_eval._cats(manifest, cols)
    pred, normal = train_eval.predict_model(
        'M3rpa', base_train, score, cols, cats, {}, rounds, _heli_const(base_train),
        return_normal=True,
    )
    out = {}
    for fold in (left, right):
        mask = score.month.isin(common.FOLDS[fold]).values
        raw = pd.DataFrame({
            'MVT_ID_mvt': score.loc[mask, 'MVT_ID_mvt'].values,
            'APT': score.loc[mask, 'APT'].values,
            'month': score.loc[mask, 'month'].values,
            'y': score.loc[mask, 'TAXITIME_SEC_mvt'].values,
            'pred': pred[mask],
            'normal_pred': normal[mask],
            'fpl_matched': score.loc[mask, 'fpl_matched'].values,
            'p_fallback': score.loc[mask, 'p_fallback'].values,
        })
        out[fold] = tail_gate.prepare_frame(raw, score.loc[mask])
        raw.to_parquet(RUNS / 'oof' / f'strict_block_{left}{right}_fold{fold}.parquet')
    return out, {'n_train': len(base_train), 'n_score': len(score), 'n_model_features': len(cols),
                 'dropped_target_features': sorted(target_cols)}


def diagnostic(feature_version=3, rounds=660, fallback_rounds=400, output=None,
               shrinks=(0.25, 0.5, 0.75, 1.0)):
    report = {'feature_version': feature_version, 'protocol': 'blocked_pair_strict',
              'gate_features': STRICT_GATE_FEATURES, 'blocks': {}}
    pooled = {str(s): [] for s in shrinks}; pooled_base = []; pooled_frames = []
    for left, right in BLOCKS:
        frames, meta = build_block_predictions(left, right, feature_version, rounds, fallback_rounds)
        block = {'base_fit': meta, 'directions': {}}
        for train_fold, test_fold in ((left, right), (right, left)):
            weight, _, gate_meta = tail_gate.fit_predict_gate(
                frames[train_fold], frames[test_fold], features=STRICT_GATE_FEATURES,
                cats=STRICT_GATE_CATS,
            )
            item = {'fit': gate_meta, 'baseline': _metric(frames[test_fold], frames[test_fold].pred.values),
                    'metrics': {}}
            for shrink in shrinks:
                pred = tail_gate.apply_gate(frames[test_fold], weight, shrink)
                item['metrics'][str(shrink)] = _metric(frames[test_fold], pred)
                pooled[str(shrink)].append(pred)
            block['directions'][f'{train_fold}_to_{test_fold}'] = item
            pooled_base.append(frames[test_fold].pred.values)
            pooled_frames.append(frames[test_fold])
        report['blocks'][f'{left}{right}'] = block
    frame = pd.concat(pooled_frames, ignore_index=True)
    report['pooled'] = {'baseline': _metric(frame, np.concatenate(pooled_base)), 'metrics': {}}
    for shrink in shrinks:
        report['pooled']['metrics'][str(shrink)] = _metric(frame, np.concatenate(pooled[str(shrink)]))
    output = Path(output) if output else RUNS / 'tail_gate_strict_blocked_pair.json'
    output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    return report


def load_saved_frames(feature_version=3):
    """Load the six already-audited blocked-pair OOF frames."""
    train_features = pd.read_parquet(common.DATA / f'features_v{feature_version}_train.parquet')
    frames = {}
    fold_blocks = {}
    for left, right in BLOCKS:
        for fold in (left, right):
            path = RUNS / 'oof' / f'strict_block_{left}{right}_fold{fold}.parquet'
            raw = pd.read_parquet(path)
            if 'p_fallback' not in raw:
                raise ValueError(f'strict OOF probabilities missing from {path.name}')
            frames[fold] = tail_gate.prepare_frame(raw, train_features)
            fold_blocks[fold] = f'{left}{right}'
    return frames, fold_blocks


def _guardrail_key(shrink, max_delta):
    cap = 'none' if max_delta is None else str(int(max_delta))
    return f'shrink={float(shrink):g}|max_delta={cap}'


def guardrail_diagnostic(feature_version=3, output=None,
                         shrinks=(0.5, 0.75, 1.0),
                         max_deltas=(300, 600, 900, 1200, 1800, None)):
    """Robustness-check cross-block gate calibration and prediction caps.

    For each nominally held-out block, the gate is fitted on the other four
    strict OOF frames. Those frames' base predictions were trained with the
    nominally held-out block available, so this is not a cleaner outer test
    than v18's paired protocol. It is a shrink/cap sensitivity check only.
    """
    frames, fold_blocks = load_saved_frames(feature_version)
    configs = [(float(s), cap) for s in shrinks for cap in max_deltas]
    report = {
        'feature_version': feature_version,
        'protocol': 'strict_cross_block_guardrail',
        'validation_note': (
            'Robustness/cap sensitivity only: calibration-frame base predictions were trained '
            'with nominally held-block labels available; not cleaner than paired v18 validation.'
        ),
        'gate_features': STRICT_GATE_FEATURES,
        'configs': [_guardrail_key(s, cap) for s, cap in configs],
        'folds': {},
    }
    pooled_frames = []
    pooled_base = []
    pooled_pred = {key: [] for key in report['configs']}

    for left, right in BLOCKS:
        held_block = f'{left}{right}'
        train_folds = [f for f in frames if fold_blocks[f] != held_block]
        gate_train = pd.concat([frames[f] for f in train_folds], ignore_index=True)
        for test_fold in (left, right):
            d = frames[test_fold]
            weight, _, meta = tail_gate.fit_predict_gate(
                gate_train, d, features=STRICT_GATE_FEATURES, cats=STRICT_GATE_CATS,
            )
            baseline = _metric(d, d.pred.values)
            item = {
                'held_block': held_block,
                'gate_train_folds': train_folds,
                'fit': meta,
                'baseline': baseline,
                'metrics': {},
            }
            for shrink, cap in configs:
                key = _guardrail_key(shrink, cap)
                pred = tail_gate.apply_gate(d, weight, shrink=shrink, max_delta=cap)
                metric = _metric(d, pred)
                item['metrics'][key] = metric
                pooled_pred[key].append(pred)
            report['folds'][test_fold] = item
            pooled_frames.append(d)
            pooled_base.append(d.pred.values)

    pooled_frame = pd.concat(pooled_frames, ignore_index=True)
    pooled_baseline = _metric(pooled_frame, np.concatenate(pooled_base))
    report['pooled'] = {'baseline': pooled_baseline, 'metrics': {}}
    candidates = []
    for key in report['configs']:
        metric = _metric(pooled_frame, np.concatenate(pooled_pred[key]))
        fold_deltas = {
            fold: report['folds'][fold]['metrics'][key]['rmse_raw']
                  - report['folds'][fold]['baseline']['rmse_raw']
            for fold in sorted(frames)
        }
        metric['delta_vs_base'] = metric['rmse_raw'] - pooled_baseline['rmse_raw']
        metric['fold_deltas'] = fold_deltas
        metric['improves_all_folds'] = all(delta < 0 for delta in fold_deltas.values())
        report['pooled']['metrics'][key] = metric
        if metric['improves_all_folds']:
            candidates.append((metric['rmse_raw'], key))
    report['selection_rule'] = 'lowest pooled RMSE among configurations improving all six folds'
    report['champion'] = min(candidates)[1] if candidates else None
    output = Path(output) if output else RUNS / 'tail_gate_strict_guardrails.json'
    output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    return report


def refresh_saved_probabilities(feature_version=3, fallback_rounds=400, atol=1e-6):
    """Refit only the strict fallback models and add their probabilities to saved block OOF."""
    data, _, _ = common.load_features(feature_version)
    checks = {}
    for left, right in BLOCKS:
        blocked_months = common.FOLDS[left] + common.FOLDS[right]
        base_train = data[~data.month.isin(blocked_months)].copy()
        score = data[data.month.isin(blocked_months)].copy()
        score['p_fallback'] = detectors.score_fallback(base_train, score, rounds=fallback_rounds)
        indexed = score.set_index('MVT_ID_mvt')
        heli_const = _heli_const(base_train)
        for fold in (left, right):
            path = RUNS / 'oof' / f'strict_block_{left}{right}_fold{fold}.parquet'
            raw = pd.read_parquet(path)
            aligned = indexed.reindex(raw.MVT_ID_mvt)
            raw['p_fallback'] = aligned.p_fallback.values
            p = np.clip(raw.p_fallback.values, 0, 1)
            schedule = aligned.atot_minus_sched.values
            valid = np.isfinite(schedule) & (schedule > 0) & np.isfinite(p)
            expected = np.where(
                valid, p * schedule + (1 - p) * raw.normal_pred.values,
                raw.normal_pred.values,
            )
            expected = np.where(aligned.is_heli.values == 1, heli_const, expected)
            expected = np.maximum(expected, 30.0)
            delta = np.abs(expected - raw.pred.values)
            if not np.isfinite(delta).all() or float(delta.max()) > atol:
                raise ValueError(f'{fold} fallback reproduction delta {float(np.nanmax(delta))}')
            raw.to_parquet(path, index=False)
            checks[fold] = float(delta.max())
    return checks


def fit_full_strict_gate(feature_version, base_submission, normal_predictions,
                         fallback_rounds=400, base_p_scale=1.4):
    """Fit a deployment gate on all six blocked-pair OOF frames using safe gate features."""
    train_features = pd.read_parquet(common.DATA / f'features_v{feature_version}_train.parquet')
    rank_features = pd.read_parquet(common.DATA / f'features_v{feature_version}_rank.parquet')
    frames = []
    for left, right in BLOCKS:
        for fold in (left, right):
            raw = pd.read_parquet(RUNS / 'oof' / f'strict_block_{left}{right}_fold{fold}.parquet')
            if 'p_fallback' not in raw:
                raise ValueError('strict OOF probabilities missing; run refresh_saved_probabilities')
            frames.append(tail_gate.prepare_frame(raw, train_features))
    gate_train = pd.concat(frames, ignore_index=True)
    base = pd.read_parquet(base_submission)[['MVT_ID_mvt', 'TAXITIME_SEC_mvt']]
    skeleton = rank_features[['MVT_ID_mvt', 'APT', 'month', 'fpl_matched']].merge(
        base, on='MVT_ID_mvt', validate='one_to_one',
    ).rename(columns={'TAXITIME_SEC_mvt': 'pred'})
    skeleton['y'] = np.nan
    gate_rank = tail_gate.prepare_frame(skeleton, rank_features, base_p_scale=base_p_scale)
    exact = pd.read_parquet(normal_predictions) if isinstance(normal_predictions, (str, Path)) else normal_predictions
    gate_rank = tail_gate.replace_clean_expert(gate_rank, exact)
    weight, model, meta = tail_gate.fit_predict_gate(
        gate_train, gate_rank, features=STRICT_GATE_FEATURES, cats=STRICT_GATE_CATS,
    )
    return gate_rank, weight, model, meta
