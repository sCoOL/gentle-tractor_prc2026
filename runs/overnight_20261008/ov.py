"""Test harness of the overnight layers: the 2025 v66 proxy + the pass rule for residual layers on it.

  py -3.14 runs/overnight_20261008/ov.py proxy     # -> runs/v66_bundle/rows.parquet (both years)

Proxy: v63 proxy + 0.75 x L1 six-fold OOF correction (six airports) + L2 rule on 2025
F_range rows + 0.5 x L3 six-fold OOF correction on July rows, each through 267.rules_v55. 2026 rows carry the v66 file.

judge(): pass = six-fold total better with >= 4/6 folds better AND both forward splits better AND January not worse by
more than noise. Body share = share of the six-fold dSSE on y <= 3600 s rows.
"""
import importlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import common  # noqa: E402

b53 = importlib.import_module('263_v53_build')
v55 = importlib.import_module('267_v55_build')
b66 = importlib.import_module('278_v66_bundle')
OUT = common.RUNS / 'overnight_20261008'   # frames and results (data); this script lives in the repository
ROWS66 = common.RUNS / 'v66_bundle' / 'rows.parquet'
V66 = common.SUBS / 'gentle-tractor_v66.parquet'
FOLDS = list('ABCDEF')
_FR = None


def frame() -> pd.DataFrame:
    """b53.frame() + status, cached (W/F/R/B inputs for both years)."""
    global _FR
    if _FR is None:
        p = OUT / 'frame_cache.parquet'
        if p.exists():
            _FR = pd.read_parquet(p)
        else:
            f = b53.frame()
            f['st'] = b53.status(f)
            f.to_parquet(p, index=False)
            _FR = f
    return _FR


def rules(ids, p) -> np.ndarray:
    f = frame().set_index('MVT_ID_mvt').loc[np.asarray(ids)].reset_index()
    return v55.rules_v55(f, np.asarray(p, float), f.st.to_numpy())[0]


def build_proxy():
    r = pd.read_parquet(b66.ROWS63, columns=['MVT_ID_mvt', 'split', 'fold', 'y', 'v63_pred_s'])
    fr = frame()[['MVT_ID_mvt', 'apt', 'month', 'st', 't', 't_sched', 'is_heli']]
    r = r.merge(fr, on='MVT_ID_mvt', how='left', validate='one_to_one')
    sub = pd.read_parquet(V66).rename(columns={'TAXITIME_SEC_mvt': 'v66_file'})
    r = r.merge(sub, on='MVT_ID_mvt', how='left')
    tr = (r.split == 'train').to_numpy()
    p = r.v63_pred_s.to_numpy().copy()
    pos = pd.Series(np.arange(len(r)), index=r.MVT_ID_mvt.to_numpy())
    layer = np.full(len(r), '', dtype=object)
    # L1
    l1 = pd.read_parquet(common.RUNS / 'v66_bundle' / 'l1_oof_sixfold.parquet')
    i1 = pos[l1.MVT_ID_mvt].to_numpy()
    assert tr[i1].all()
    p[i1] = rules(l1.MVT_ID_mvt, p[i1] + b66.ALPHA * l1['corr'].to_numpy())
    layer[i1] = 'L1'
    # L2: 2025 F_range rows 1-8 h late at LIRF -> D
    m2 = tr & (r.apt.eq('LIRF') & ~r.is_heli.astype(bool) & r.st.eq('F_range')).to_numpy()
    D = (r.t - r.t_sched).to_numpy()
    m2 &= (D > b66.F_LO) & (D <= b66.F_HI)
    assert not (layer[m2] != '').any()
    p[m2] = rules(r.MVT_ID_mvt[m2], D[m2])
    layer[m2] = 'L2'
    # L3: July rows only
    l3 = pd.read_parquet(common.RUNS / 'v66_bundle' / 'l3_oof.parquet')
    l3 = l3[l3.month.eq(7)]
    i3 = pos[l3.MVT_ID_mvt].to_numpy()
    assert not (layer[i3] != '').any()
    p[i3] = rules(l3.MVT_ID_mvt, p[i3] + b66.L3_ALPHA * l3.corr_sixfold.to_numpy())
    layer[i3] = 'L3'
    r['v66_pred_s'] = np.where(tr, p, r.v66_file)
    r['layer66'] = layer
    assert r.v66_pred_s.notna().all()
    out = r[['MVT_ID_mvt', 'split', 'fold', 'month', 'apt', 'y', 'v63_pred_s', 'v66_pred_s', 'layer66']]
    out.to_parquet(ROWS66, index=False)
    t = out[out.split == 'train']
    e63, e66 = (t.y - t.v63_pred_s) ** 2, (t.y - t.v66_pred_s) ** 2
    res = {'rows': out.split.value_counts().to_dict(), 'by_layer': t.layer66.value_counts().to_dict(),
           'rmse_v63_proxy': round(float(np.sqrt(e63.mean())), 4), 'rmse_v66_proxy': round(float(np.sqrt(e66.mean())), 4),
           'dsse_M_by_layer': ((e66 - e63).groupby(t.layer66).sum() / 1e6).round(1).to_dict(),
           'dsse_M_by_fold': ((e66 - e63).groupby(t.fold).sum() / 1e6).round(1).to_dict()}
    (OUT / 'proxy66.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def load_proxy() -> pd.DataFrame:
    return pd.read_parquet(ROWS66)


# ------------------------------------------------------------------------------------------------------------- judge
def judge(d: pd.DataFrame, p6: np.ndarray, pf: np.ndarray, name: str, n2026: int | None = None,
          extra: dict | None = None) -> dict:
    """d: 2025 rows with y, base (v66 proxy), fold, month. p6 / pf: candidate predictions out of fold (six folds) and
    forward (H1<->H2). n2026: number of 2026 rows the layer would move (recorded)."""
    y, b = d.y.to_numpy(), d.base.to_numpy()
    e6 = (y - p6) ** 2 - (y - b) ** 2
    ef = (y - pf) ** 2 - (y - b) ** 2
    h1 = (d.month <= 6).to_numpy()
    fold = (pd.Series(e6).groupby(d.fold.to_numpy()).sum() / 1e6).round(2)
    body = float(e6[y <= 3600].sum())
    res = {'name': name, 'rows_2025': int(len(d)), 'rows_moved_2025': int((np.abs(p6 - b) > 1e-6).sum()),
           'sixfold_dsse_M': round(e6.sum() / 1e6, 2), 'by_fold_M': fold.to_dict(),
           'folds_better': int((fold < 0).sum()),
           'forward_H1toH2_M': round(ef[~h1].sum() / 1e6, 2), 'forward_H2toH1_M': round(ef[h1].sum() / 1e6, 2),
           'jan_sixfold_M': round(e6[(d.month == 1).to_numpy()].sum() / 1e6, 2),
           'jan_forward_M': round(ef[(d.month == 1).to_numpy()].sum() / 1e6, 2),
           'jul_sixfold_M': round(e6[(d.month == 7).to_numpy()].sum() / 1e6, 2),
           'body_share': round(body / e6.sum(), 3) if e6.sum() != 0 else None,
           'rms_move_s': round(float(np.sqrt(np.mean((p6 - b) ** 2))), 1)}
    jan_noise = 0.05 * abs(res['sixfold_dsse_M']) + 0.5
    res['pass'] = bool(res['sixfold_dsse_M'] < 0 and res['folds_better'] >= 4 and res['forward_H1toH2_M'] < 0
                       and res['forward_H2toH1_M'] < 0 and res['jan_sixfold_M'] <= jan_noise
                       and res['jan_forward_M'] <= jan_noise)
    if n2026:
        res['rows_2026'] = int(n2026)
    if extra:
        res.update(extra)
    return res


PARAMS = dict(objective='regression', learning_rate=0.03, num_leaves=7, min_data_in_leaf=100, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=12, deterministic=True,
              force_row_wise=True)


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, X, rounds=200, clip=1800.0, seeds=(7, 8, 9), params=None):
    out = np.zeros(len(te))
    y = tr.r.clip(-clip, clip)
    for s in seeds:
        m = lgb.train({**(params or PARAMS), 'seed': s}, lgb.Dataset(tr[X], y), rounds)
        out += m.predict(te[X]) / len(seeds)
    return out


def residual_layer(d: pd.DataFrame, X, alphas=(0.25, 0.5, 0.75, 1.0), by=None, **kw) -> tuple[dict, dict]:
    """Six-fold + forward residual LightGBM on d (2025 rows: y, base, fold, month, r = y - base, X).
    by: optional column; one model per value. Returns ({alpha: (p6, pf)}, corrections)."""
    groups = [None] if by is None else sorted(d[by].dropna().unique())
    c6, cf = np.zeros(len(d)), np.zeros(len(d))
    h1 = (d.month <= 6).to_numpy()
    for g in groups:
        gm = np.ones(len(d), bool) if g is None else (d[by] == g).to_numpy()
        for f in FOLDS:
            trm, tem = gm & d.fold.ne(f).to_numpy(), gm & d.fold.eq(f).to_numpy()
            if tem.any() and trm.sum() > 50:
                c6[tem] = fit_predict(d[trm], d[tem], X, **kw)
        for trm, tem in ((gm & h1, gm & ~h1), (gm & ~h1, gm & h1)):
            if tem.any() and trm.sum() > 50:
                cf[tem] = fit_predict(d[trm], d[tem], X, **kw)
    out = {}
    for al in alphas:
        out[al] = (rules(d.MVT_ID_mvt, d.base.to_numpy() + al * c6), rules(d.MVT_ID_mvt, d.base.to_numpy() + al * cf))
    return out, {'c6': c6, 'cf': cf}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def save(name: str, res):
    (OUT / f'{name}.json').write_text(json.dumps(_clean(res), indent=1, default=str))


if __name__ == '__main__':
    if sys.argv[1] == 'proxy':
        build_proxy()
