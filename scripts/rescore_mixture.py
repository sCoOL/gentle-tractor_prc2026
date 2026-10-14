"""Re-score an existing two-stage OOF run under deterministic p_fallback scaling.

The clean-regressor expert is recovered algebraically from the saved mixture, so no booster
is refitted. Full submission fits use the same `p_scale` parameter in train_eval._mixture.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import common
from common import RUNS


def recover_expert(base_pred, p, schedule, is_heli):
    base_pred = np.asarray(base_pred, float); p = np.asarray(p, float)
    schedule = np.asarray(schedule, float); is_heli = np.asarray(is_heli).astype(bool)
    valid = np.isfinite(schedule) & np.isfinite(p) & ((1.0 - p) > 1e-8) & ~is_heli
    expert = base_pred.copy()
    expert[valid] = (base_pred[valid] - p[valid] * schedule[valid]) / (1.0 - p[valid])
    return expert, valid


def scaled_prediction(base_pred, p, schedule, expert, valid, scale):
    ps = np.clip(np.asarray(p, float) * float(scale), 0.0, 1.0)
    out = np.asarray(base_pred, float).copy()
    out[valid] = ps[valid] * np.asarray(schedule, float)[valid] + (1.0 - ps[valid]) * np.asarray(expert, float)[valid]
    return np.maximum(out, 30.0)


def rescore(run_id, feature_version, scales, tag='p_scale'):
    results = pd.read_csv(RUNS / 'results.csv'); src = results[results.run_id == run_id]
    if set(src.fold) < {'A', 'B'}: raise ValueError(f'{run_id} lacks Fold A/B results')
    tr = pd.read_parquet(common.DATA / f'features_v{feature_version}_train.parquet',
                         columns=['MVT_ID_mvt', 'p_fallback', 'atot_minus_sched', 'is_heli'])
    made = []
    for scale in scales:
        new_id = common.new_run_id(f'{tag}_{str(scale).replace(".", "p")}')
        for fold in ('A', 'B'):
            oof = pd.read_parquet(RUNS / 'oof' / f'{run_id}_fold{fold}.parquet').merge(tr, on='MVT_ID_mvt', how='left', validate='one_to_one')
            expert, valid = recover_expert(oof.pred.values, oof.p_fallback.values, oof.atot_minus_sched.values, oof.is_heli.values)
            pred = scaled_prediction(oof.pred.values, oof.p_fallback.values, oof.atot_minus_sched.values, expert, valid, scale)
            y = oof.y.values; m = common.metrics(pred, y, oof.APT.values, oof.month.values, oof.fpl_matched.values)
            source = src[src.fold == fold].iloc[-1]; params = json.loads(source.params_json); params['p_scale'] = float(scale)
            row = dict(run_id=new_id, ts=time.strftime('%Y-%m-%d %H:%M:%S'), feature_version=feature_version,
                       groups_on=source.groups_on, model=source.model, params_json=json.dumps(params), fold=fold,
                       rmse_jan=m.pop('rmse_m1'), rmse_jul=m.pop('rmse_m2'), n_train=int(source.n_train), n_test=len(y),
                       fit_seconds=0.0, notes=f'{tag}; algebraic rescore of {run_id}', seed=int(getattr(source, 'seed', 0)), **m)
            common.append_result(row)
            oof[['MVT_ID_mvt', 'APT', 'month', 'y', 'fpl_matched']].assign(pred=pred).to_parquet(RUNS / 'oof' / f'{new_id}_fold{fold}.parquet')
            print(f'{new_id} fold {fold}: RMSE {row["rmse_raw"]:.2f} months {row["rmse_jan"]:.2f}/{row["rmse_jul"]:.2f}', flush=True)
        made.append(new_id)
        time.sleep(1.1)  # keep run ids unique at one-second resolution
    return made
