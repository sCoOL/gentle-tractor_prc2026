"""Fully nested rich-base diagnostic against the strict v18 protocol.

The existing strict diagnostic removes target-derived G8 and route-history columns from
the base learner.  This module rebuilds those columns inside every outer AB/CD/EF block:

* outer score rows use statistics fitted only on the other eight months;
* outer training rows receive an additional seasonal-pair inner-OOF build;
* fallback probabilities for outer score rows are fitted only on the outer training rows.

The strict target-free gate is then calibrated and tested exactly as for v18.  This makes
the resulting six-fold comparison directly comparable with the v18 diagnostic while
allowing the base learner to use the safe rich features.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import common
import detectors
import nested_features
import strict_tail_gate
import tail_gate
import train_eval
from common import RUNS


RICH_COLS = nested_features.TARGET_ENCODING_COLS + nested_features.ROUTE_COLS


def _aligned_assign(frame, values, columns):
    """Return ``frame`` with ID-aligned rich columns replaced."""
    if frame.MVT_ID_mvt.duplicated().any() or values.MVT_ID_mvt.duplicated().any():
        raise ValueError('rich-feature IDs must be unique')
    if not frame.MVT_ID_mvt.isin(values.MVT_ID_mvt).all():
        raise ValueError(f'missing rich-feature IDs for {columns}')
    aligned = values.set_index('MVT_ID_mvt')[columns].reindex(frame.MVT_ID_mvt)
    out = frame.copy()
    for col in columns:
        out[col] = aligned[col].values
    return out


def _safe_rich_frames(base_train, score):
    """Build inner-OOF train features and disjoint-history score features."""
    inner_folds = nested_features.enc.seasonal_pair_folds(base_train.month.values)
    enc_train, enc_score = nested_features.enc.encode_frames(
        base_train, score, fold_ids=inner_folds,
    )
    route_train, route_score = detectors.route_features_oof(
        base_train, score, inner_folds,
    )
    route_train = route_train.copy()
    route_score = route_score.copy()
    route_train.insert(0, 'MVT_ID_mvt', base_train.MVT_ID_mvt.values)
    route_score.insert(0, 'MVT_ID_mvt', score.MVT_ID_mvt.values)
    train = _aligned_assign(base_train, enc_train, nested_features.TARGET_ENCODING_COLS)
    train = _aligned_assign(train, route_train, nested_features.ROUTE_COLS)
    scored = _aligned_assign(score, enc_score, nested_features.TARGET_ENCODING_COLS)
    scored = _aligned_assign(scored, route_score, nested_features.ROUTE_COLS)
    return train, scored


def _model_columns(manifest, rich_groups=('G4', 'G8')):
    """Use v18-safe columns plus selected fully nested rich families."""
    all_cols = train_eval.feature_columns(manifest, list(manifest['groups']))
    strict_dropped = set(manifest['groups'].get('G8', [])) | strict_tail_gate.TARGET_ROUTE | strict_tail_gate.TARGET_G9
    allowed_rich = {
        c for group in rich_groups
        for c in manifest['groups'].get(group, [])
        if c in RICH_COLS
    }
    return [c for c in all_cols if c not in strict_dropped or c in allowed_rich]


def build_block_predictions(left, right, feature_version=3, rounds=660,
                            fallback_rounds=400, rich_groups=('G4', 'G8'),
                            persist=True):
    data, _, manifest = common.load_features(feature_version)
    blocked_months = common.FOLDS[left] + common.FOLDS[right]
    base_train = data[~data.month.isin(blocked_months)].copy()
    score = data[data.month.isin(blocked_months)].copy()
    base_train, score = _safe_rich_frames(base_train, score)
    score['p_fallback'] = detectors.score_fallback(
        base_train, score, rounds=fallback_rounds,
    )
    cols = _model_columns(manifest, rich_groups)
    cats = train_eval._cats(manifest, cols)
    pred, normal = train_eval.predict_model(
        'M3rpa', base_train, score, cols, cats, {}, rounds,
        strict_tail_gate._heli_const(base_train), return_normal=True,
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
        rows = score.loc[mask]
        out[fold] = tail_gate.prepare_frame(raw, rows)
        if persist:
            suffix = ''.join(sorted(rich_groups)).lower()
            raw.to_parquet(
                RUNS / 'oof' / f'nested_rich_base_{suffix}_{left}{right}_fold{fold}.parquet',
                index=False,
            )
    return out, {
        'n_train': len(base_train), 'n_score': len(score),
        'n_model_features': len(cols), 'rich_groups': list(rich_groups),
        'rich_columns': [c for c in cols if c in RICH_COLS],
        'source_months': sorted(int(x) for x in base_train.month.unique()),
        'blocked_months': blocked_months,
    }


def diagnostic(feature_version=3, rounds=660, fallback_rounds=400,
               rich_groups=('G4', 'G8'), output=None,
               shrinks=(0.5, 0.75, 1.0)):
    report = {
        'feature_version': feature_version,
        'protocol': 'strict_paired_block_fully_nested_rich_base',
        'rich_groups': list(rich_groups),
        'gate_features': strict_tail_gate.STRICT_GATE_FEATURES,
        'blocks': {},
    }
    pooled_frames = []
    pooled_base = []
    pooled = {str(float(s)): [] for s in shrinks}
    for left, right in strict_tail_gate.BLOCKS:
        frames, meta = build_block_predictions(
            left, right, feature_version, rounds, fallback_rounds, rich_groups,
        )
        block = {'base_fit': meta, 'directions': {}}
        for calibration, test in ((left, right), (right, left)):
            weight, _, gate_meta = tail_gate.fit_predict_gate(
                frames[calibration], frames[test],
                features=strict_tail_gate.STRICT_GATE_FEATURES,
                cats=strict_tail_gate.STRICT_GATE_CATS,
            )
            item = {
                'fit': gate_meta,
                'baseline': strict_tail_gate._metric(frames[test], frames[test].pred.values),
                'metrics': {},
            }
            for shrink in shrinks:
                pred = tail_gate.apply_gate(frames[test], weight, shrink)
                item['metrics'][str(float(shrink))] = strict_tail_gate._metric(frames[test], pred)
                pooled[str(float(shrink))].append(pred)
            block['directions'][f'{calibration}_to_{test}'] = item
            pooled_frames.append(frames[test])
            pooled_base.append(frames[test].pred.values)
        report['blocks'][f'{left}{right}'] = block

    frame = pd.concat(pooled_frames, ignore_index=True)
    base_metric = strict_tail_gate._metric(frame, np.concatenate(pooled_base))
    report['pooled'] = {'baseline': base_metric, 'metrics': {}}
    old = json.loads(
        (RUNS / 'tail_gate_strict_blocked_pair_20260915.json').read_text(encoding='utf-8')
    )
    old_by_fold = {}
    for left, right in strict_tail_gate.BLOCKS:
        old_by_fold[left] = old['blocks'][f'{left}{right}']['directions'][f'{right}_to_{left}']['metrics']['1.0']['rmse_raw']
        old_by_fold[right] = old['blocks'][f'{left}{right}']['directions'][f'{left}_to_{right}']['metrics']['1.0']['rmse_raw']

    candidates = []
    for shrink in shrinks:
        key = str(float(shrink))
        metric = strict_tail_gate._metric(frame, np.concatenate(pooled[key]))
        fold_delta = {}
        for left, right in strict_tail_gate.BLOCKS:
            for fold in (left, right):
                direction = f'{right}_to_{left}' if fold == left else f'{left}_to_{right}'
                score = report['blocks'][f'{left}{right}']['directions'][direction]['metrics'][key]['rmse_raw']
                fold_delta[fold] = score - old_by_fold[fold]
        metric['delta_vs_base'] = metric['rmse_raw'] - base_metric['rmse_raw']
        metric['delta_vs_v18'] = metric['rmse_raw'] - old['pooled']['metrics']['1.0']['rmse_raw']
        metric['fold_delta_vs_v18'] = fold_delta
        metric['max_fold_regression'] = max(fold_delta.values())
        metric['accepted'] = (
            metric['delta_vs_v18'] <= -1.0 and metric['max_fold_regression'] <= 1.0
        )
        report['pooled']['metrics'][key] = metric
        if metric['accepted']:
            candidates.append((metric['rmse_raw'], key))
    report['selection_rule'] = 'pooled >=1s better than v18 and no fold >1s worse'
    report['champion'] = min(candidates)[1] if candidates else None
    output = Path(output) if output else RUNS / 'tail_gate_nested_rich_base.json'
    output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    return report
