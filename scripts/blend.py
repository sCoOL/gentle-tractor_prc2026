"""Non-negative least-squares blending of OOF predictions, per airport with a global fallback; optional equal weights."""
from __future__ import annotations
import json, time
import numpy as np, pandas as pd
from scipy.optimize import nnls
import common
from common import RUNS


def _nnls(P, y):
    w, _ = nnls(P, y); s = w.sum()
    return w / s if s > 0 else np.full(P.shape[1], 1 / P.shape[1])


def fit_weights(oofs, min_rows=5000, equal=False):
    ids = list(oofs)
    if equal: return {'global': {k: 1.0 / len(ids) for k in ids}, 'by_apt': {}}
    base = oofs[ids[0]][['MVT_ID_mvt', 'APT', 'y']].copy()
    for k in ids: base = base.merge(oofs[k][['MVT_ID_mvt', 'pred']].rename(columns={'pred': k}), on='MVT_ID_mvt')
    P = base[ids].values; y = base.y.values; g = _nnls(P, y); out = {'global': dict(zip(ids, map(float, g))), 'by_apt': {}}
    for apt in sorted(base.APT.unique()):
        m = base.APT.values == apt
        if m.sum() >= min_rows:
            w = _nnls(P[m], y[m])
            if common.rmse(P[m] @ w, y[m]) <= common.rmse(P[m] @ g, y[m]): out['by_apt'][apt] = dict(zip(ids, map(float, w))); continue
        out['by_apt'][apt] = out['global']
    return out


def apply_weights(weights, preds, apt):
    ids = list(weights['global']); out = np.zeros(len(apt))
    for a in np.unique(apt):
        w = weights['by_apt'].get(a, weights['global']); m = apt == a
        out[m] = sum(w[k] * np.asarray(preds[k])[m] for k in ids)
    return out


def blend_runs(run_ids, tag='', equal=False):
    oofs = {f: {r: pd.read_parquet(RUNS / 'oof' / f'{r}_fold{f}.parquet') for r in run_ids} for f in ('A', 'B')}
    w = fit_weights(oofs['A'], equal=equal); blend_id = common.new_run_id(('blend_equal' if equal else 'blend') + (('_' + tag) if tag else ''))
    for f in ('A', 'B'):
        first = oofs[f][run_ids[0]]; keep = ['MVT_ID_mvt', 'APT', 'month', 'y'] + (['fpl_matched'] if 'fpl_matched' in first.columns else [])
        base = first[keep].copy()
        if 'fpl_matched' not in base.columns: base = base.merge(common.load_features(1)[0][['MVT_ID_mvt', 'fpl_matched']], on='MVT_ID_mvt', how='left')
        for r in run_ids: base = base.merge(oofs[f][r][['MVT_ID_mvt', 'pred']].rename(columns={'pred': r}), on='MVT_ID_mvt')
        p = apply_weights(w, {r: base[r].values for r in run_ids}, base.APT.values); y = base.y.values
        m = common.metrics(p, y, base.APT.values, base.month.values, base.fpl_matched.values)
        common.append_result(dict(run_id=blend_id, ts=time.strftime('%Y-%m-%d %H:%M:%S'), feature_version=-1, groups_on='', model='blend_equal' if equal else 'blend', params_json=json.dumps(run_ids), fold=f,
                                  rmse_jan=m.pop('rmse_m1'), rmse_jul=m.pop('rmse_m2'), n_train=0, n_test=len(y), fit_seconds=0, notes=tag, **m))
        pd.DataFrame({'MVT_ID_mvt': base.MVT_ID_mvt.values, 'APT': base.APT.values, 'month': base.month.values, 'y': y, 'pred': p, 'fpl_matched': base.fpl_matched.values}).to_parquet(RUNS / 'oof' / f'{blend_id}_fold{f}.parquet')
        print(f'{blend_id} fold {f}: RMSE {m["rmse_raw"]:.1f} weights(global) {w["global"]}', flush=True)
    json.dump({'blend_id': blend_id, 'run_ids': run_ids, 'weights': w, 'equal': equal}, open(RUNS / f'blend_{blend_id}.json', 'w'), indent=1)
    return blend_id, w
