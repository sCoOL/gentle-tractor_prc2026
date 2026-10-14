"""Hierarchical airport-specific gate overrides evaluated on strict paired blocks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import strict_tail_gate
import tail_gate
from common import RUNS


LOCAL_FEATURES = [c for c in strict_tail_gate.STRICT_GATE_FEATURES if c != 'APT']
LOCAL_CATS = [c for c in strict_tail_gate.STRICT_GATE_CATS if c != 'APT']


LOCAL_CONFIGS = {
    'small': {'rounds': 80, 'num_leaves': 3, 'min_data_in_leaf': 50},
    'medium': {'rounds': 120, 'num_leaves': 7, 'min_data_in_leaf': 30},
    'flex': {'rounds': 150, 'num_leaves': 15, 'min_data_in_leaf': 20},
}


def fit_weights(calibration, score, min_eligible=100, local_config=None):
    local_config = dict(local_config or LOCAL_CONFIGS['medium'])
    global_weight, _, global_meta = tail_gate.fit_predict_gate(
        calibration, score,
        features=strict_tail_gate.STRICT_GATE_FEATURES,
        cats=strict_tail_gate.STRICT_GATE_CATS,
    )
    local_weight = global_weight.copy()
    counts = {}
    fitted = []
    for apt in sorted(score.APT.unique()):
        train_apt = calibration[calibration.APT == apt]
        score_mask = score.APT.values == apt
        n = int(train_apt.gate_valid.sum())
        counts[apt] = n
        if n < min_eligible or not score_mask.any():
            continue
        weight, _, _ = tail_gate.fit_predict_gate(
            train_apt, score[score_mask], features=LOCAL_FEATURES, cats=LOCAL_CATS,
            **local_config,
        )
        local_weight[score_mask] = weight
        fitted.append(apt)
    return global_weight, local_weight, {
        'global': global_meta,
        'local_features': LOCAL_FEATURES,
        'local_categoricals': LOCAL_CATS,
        'min_eligible': min_eligible,
        'local_config': local_config,
        'eligible_by_airport': counts,
        'fitted_airports': fitted,
    }


def diagnostic(output=None, alphas=(0.25, 0.5, 0.75, 1.0), min_eligible=100,
               local_configs=None):
    frames, _ = strict_tail_gate.load_saved_frames()
    local_configs = dict(local_configs or LOCAL_CONFIGS)
    configs = [
        f'{capacity}|{scope}:{a:g}'
        for capacity in local_configs for scope in ('all', 'LIRF') for a in alphas
    ]
    report = {
        'protocol': 'strict_paired_block_hierarchical_airport_gate',
        'configs': configs,
        'folds': {},
    }
    pooled_frames = []
    pooled_pred = {key: [] for key in configs}
    for left, right in strict_tail_gate.BLOCKS:
        for calibration, test in ((left, right), (right, left)):
            d = frames[test]
            global_w, _, global_meta = fit_weights(
                frames[calibration], d, min_eligible, next(iter(local_configs.values())),
            )
            item = {
                'calibration_fold': calibration,
                'global_fit': global_meta['global'],
                'local_fits': {},
                'baseline_global_gate': {},
                'metrics': {},
            }
            global_pred = tail_gate.apply_gate(d, global_w, shrink=1.0)
            item['baseline_global_gate'] = strict_tail_gate._metric(d, global_pred)
            for capacity, local_config in local_configs.items():
                _, local_w, meta = fit_weights(
                    frames[calibration], d, min_eligible, local_config,
                )
                item['local_fits'][capacity] = meta
                for alpha in alphas:
                    blended = (1.0 - alpha) * global_w + alpha * local_w
                    key = f'{capacity}|all:{alpha:g}'
                    pred = tail_gate.apply_gate(d, blended, shrink=1.0)
                    item['metrics'][key] = strict_tail_gate._metric(d, pred)
                    pooled_pred[key].append(pred)

                    lirf_weight = global_w.copy()
                    mask = d.APT.values == 'LIRF'
                    lirf_weight[mask] = blended[mask]
                    key = f'{capacity}|LIRF:{alpha:g}'
                    pred = tail_gate.apply_gate(d, lirf_weight, shrink=1.0)
                    item['metrics'][key] = strict_tail_gate._metric(d, pred)
                    pooled_pred[key].append(pred)
            report['folds'][test] = item
            pooled_frames.append(d)

    frame = pd.concat(pooled_frames, ignore_index=True)
    old = json.loads((RUNS / 'tail_gate_strict_blocked_pair_20260915.json').read_text(encoding='utf-8'))
    old_by_fold = {}
    for left, right in strict_tail_gate.BLOCKS:
        old_by_fold[left] = old['blocks'][f'{left}{right}']['directions'][f'{right}_to_{left}']['metrics']['1.0']['rmse_raw']
        old_by_fold[right] = old['blocks'][f'{left}{right}']['directions'][f'{left}_to_{right}']['metrics']['1.0']['rmse_raw']
    report['pooled'] = {'metrics': {}}
    candidates = []
    for key in configs:
        metric = strict_tail_gate._metric(frame, np.concatenate(pooled_pred[key]))
        fold_delta = {
            fold: report['folds'][fold]['metrics'][key]['rmse_raw'] - old_by_fold[fold]
            for fold in sorted(frames)
        }
        metric['delta_vs_v18'] = metric['rmse_raw'] - old['pooled']['metrics']['1.0']['rmse_raw']
        metric['fold_delta_vs_v18'] = fold_delta
        metric['max_fold_regression'] = max(fold_delta.values())
        metric['accepted'] = metric['delta_vs_v18'] <= -1.0 and metric['max_fold_regression'] <= 1.0
        report['pooled']['metrics'][key] = metric
        if metric['accepted']:
            candidates.append((metric['rmse_raw'], key))
    report['selection_rule'] = 'pooled >=1s better than v18 and no fold >1s worse'
    report['champion'] = min(candidates)[1] if candidates else None
    output = Path(output) if output else RUNS / 'tail_gate_airport_hierarchical.json'
    output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    return report
