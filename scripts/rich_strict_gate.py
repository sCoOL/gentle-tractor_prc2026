"""Outer-safe rich features layered on the strict blocked-pair gate.

For each AB/CD/EF block, all target-derived rich features are fitted only on the
other eight months. The same safe features are used for the calibration and test
pair, preserving the clean paired-block boundary of v18.
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
from common import RUNS


enc = nested_features.enc
EXTRA_FEATURES = nested_features.TARGET_ENCODING_COLS + nested_features.ROUTE_COLS
FEATURES = list(dict.fromkeys(strict_tail_gate.STRICT_GATE_FEATURES + EXTRA_FEATURES))
CATS = [c for c in tail_gate.CATS if c in FEATURES]


def _safe_block_features(base_train, score):
    encoded = enc.encode_score(base_train, score).set_index('MVT_ID_mvt')
    stand = detectors.detect_route_changes(base_train, 'apt_stand')
    pier = detectors.detect_route_changes(base_train, 'apt_pier')
    route = detectors.route_features(score, stand, pier, is_rank=True)
    route.index = score.MVT_ID_mvt.values
    out = encoded.join(route[nested_features.ROUTE_COLS], how='left')
    out.index.name = 'MVT_ID_mvt'
    return out.reset_index()


def build_frames(feature_version=3, persist=True, rebuild=False):
    data, _, _ = common.load_features(feature_version)
    frames = {}
    metadata = {}
    for left, right in strict_tail_gate.BLOCKS:
        blocked = common.FOLDS[left] + common.FOLDS[right]
        base_train = data[~data.month.isin(blocked)].copy()
        score = data[data.month.isin(blocked)].copy()
        cache = RUNS / 'oof' / f'outer_safe_rich_block_{left}{right}.parquet'
        safe = (pd.read_parquet(cache) if cache.exists() and not rebuild
                else _safe_block_features(base_train, score))
        if persist:
            safe.to_parquet(cache, index=False)
        safe_indexed = safe.set_index('MVT_ID_mvt')
        for fold in (left, right):
            raw = pd.read_parquet(RUNS / 'oof' / f'strict_block_{left}{right}_fold{fold}.parquet')
            rows = score[score.month.isin(common.FOLDS[fold])].copy()
            aligned = safe_indexed.reindex(rows.MVT_ID_mvt)
            if aligned[EXTRA_FEATURES].isna().all(axis=None):
                raise ValueError(f'failed safe-rich alignment for fold {fold}')
            for col in EXTRA_FEATURES:
                rows[col] = aligned[col].values
            frames[fold] = tail_gate.prepare_frame(raw, rows)
        metadata[f'{left}{right}'] = {
            'source_months': sorted(int(x) for x in base_train.month.unique()),
            'blocked_months': blocked,
            'n_source': len(base_train),
            'n_score': len(score),
        }
    return frames, metadata


def diagnostic(feature_version=3, output=None, shrinks=(0.25, 0.5, 0.75, 1.0),
               extra_features=None, frames=None, metadata=None):
    if frames is None or metadata is None:
        frames, metadata = build_frames(feature_version)
    extras = EXTRA_FEATURES if extra_features is None else list(extra_features)
    features = list(dict.fromkeys(strict_tail_gate.STRICT_GATE_FEATURES + extras))
    report = {
        'feature_version': feature_version,
        'protocol': 'strict_paired_block_outer_safe_rich_gate',
        'features': features,
        'extra_features': extras,
        'blocks': metadata,
        'folds': {},
    }
    pooled_frames = []
    pooled_base = []
    pooled = {str(float(s)): [] for s in shrinks}
    for left, right in strict_tail_gate.BLOCKS:
        for calibration, test in ((left, right), (right, left)):
            weight, _, meta = tail_gate.fit_predict_gate(
                frames[calibration], frames[test], features=features,
                cats=[c for c in CATS if c in features],
            )
            item = {
                'calibration_fold': calibration,
                'fit': meta,
                'baseline': strict_tail_gate._metric(frames[test], frames[test].pred.values),
                'metrics': {},
            }
            for shrink in shrinks:
                pred = tail_gate.apply_gate(frames[test], weight, shrink=shrink)
                item['metrics'][str(float(shrink))] = strict_tail_gate._metric(frames[test], pred)
                pooled[str(float(shrink))].append(pred)
            report['folds'][test] = item
            pooled_frames.append(frames[test])
            pooled_base.append(frames[test].pred.values)

    frame = pd.concat(pooled_frames, ignore_index=True)
    base_metric = strict_tail_gate._metric(frame, np.concatenate(pooled_base))
    report['pooled'] = {'baseline': base_metric, 'metrics': {}}
    old = json.loads((RUNS / 'tail_gate_strict_blocked_pair_20260915.json').read_text(encoding='utf-8'))
    old_by_fold = {}
    for left, right in strict_tail_gate.BLOCKS:
        old_by_fold[left] = old['blocks'][f'{left}{right}']['directions'][f'{right}_to_{left}']['metrics']['1.0']['rmse_raw']
        old_by_fold[right] = old['blocks'][f'{left}{right}']['directions'][f'{left}_to_{right}']['metrics']['1.0']['rmse_raw']
    candidates = []
    for shrink in shrinks:
        key = str(float(shrink))
        metric = strict_tail_gate._metric(frame, np.concatenate(pooled[key]))
        versus_base = {}
        versus_v18 = {}
        for fold in sorted(frames):
            score = report['folds'][fold]['metrics'][key]['rmse_raw']
            versus_base[fold] = score - report['folds'][fold]['baseline']['rmse_raw']
            versus_v18[fold] = score - old_by_fold[fold]
        metric['fold_delta_vs_base'] = versus_base
        metric['fold_delta_vs_v18'] = versus_v18
        metric['delta_vs_base'] = metric['rmse_raw'] - base_metric['rmse_raw']
        metric['delta_vs_v18'] = metric['rmse_raw'] - old['pooled']['metrics']['1.0']['rmse_raw']
        metric['improves_all_base_folds'] = all(x < 0 for x in versus_base.values())
        metric['v18_max_fold_regression'] = max(versus_v18.values())
        metric['accepted_vs_v18'] = (
            metric['delta_vs_v18'] <= -1.0 and metric['v18_max_fold_regression'] <= 1.0
        )
        report['pooled']['metrics'][key] = metric
        if metric['accepted_vs_v18']:
            candidates.append((metric['rmse_raw'], key))
    report['selection_rule'] = 'pooled >=1s better than v18 and no fold >1s worse'
    report['champion'] = min(candidates)[1] if candidates else None
    output = Path(output) if output else RUNS / 'tail_gate_outer_safe_rich.json'
    output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    return report, frames


def fit_full(frames, feature_version, base_submission, normal_predictions,
             extra_features=None):
    extras = EXTRA_FEATURES if extra_features is None else list(extra_features)
    features = list(dict.fromkeys(strict_tail_gate.STRICT_GATE_FEATURES + extras))
    train = pd.concat([frames[f] for f in sorted(frames)], ignore_index=True)
    rank_features = pd.read_parquet(common.DATA / f'features_v{feature_version}_rank.parquet')
    base = pd.read_parquet(base_submission)[['MVT_ID_mvt', 'TAXITIME_SEC_mvt']]
    skeleton = rank_features[['MVT_ID_mvt', 'APT', 'month', 'fpl_matched']].merge(
        base, on='MVT_ID_mvt', validate='one_to_one',
    ).rename(columns={'TAXITIME_SEC_mvt': 'pred'})
    skeleton['y'] = np.nan
    rank = tail_gate.prepare_frame(skeleton, rank_features, base_p_scale=1.4)
    exact = pd.read_parquet(normal_predictions)
    rank = tail_gate.replace_clean_expert(rank, exact)
    weight, model, meta = tail_gate.fit_predict_gate(
        train, rank, features=features, cats=[c for c in CATS if c in features],
    )
    return rank, weight, model, meta
