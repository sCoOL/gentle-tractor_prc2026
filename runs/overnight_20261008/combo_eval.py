"""The four overnight layers (279 items 5, 4, 2, 11; A74-A77) combined into one v67 (A78). Check on 2025 that the four
layers stacked on the v66 proxy (moves added where two layers overlap, rules_v55 once on the sum) still pass: six folds + both forward splits.
Each layer's out-of-fold correction is recomputed with exactly the build configuration of scripts/279.
-> combo_eval.json, combo_oof.parquet
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ov  # noqa: E402

m279 = importlib.import_module('279_overnight_layers')
v66 = pd.read_parquet(m279.V66).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)
FO = ov.FOLDS


def oof(item):
    cfg = m279.ITEMS[item]
    d = m279.frame(item, v66)
    tr = d[d.split == 'train'].reset_index(drop=True)
    clip = cfg.get('clip', 1800.0)
    body = (tr.y <= 3600).to_numpy() if cfg.get('fit_body') else np.ones(len(tr), bool)
    groups = sorted(tr[cfg['by']].unique()) if cfg.get('by') else [None]
    h1 = (tr.month <= 6).to_numpy()
    c6, cf = np.zeros(len(tr)), np.zeros(len(tr))
    for splits, c in (([(tr.fold.ne(k).to_numpy(), tr.fold.eq(k).to_numpy()) for k in FO], c6), ([(h1, ~h1), (~h1, h1)], cf)):
        for trm, tem in splits:
            for g in groups:
                ga = np.ones(len(tr), bool) if g is None else (tr[cfg['by']] == g).to_numpy()
                fit = trm & ga & body
                if fit.sum() > (200 if g is not None else 50) and (tem & ga).any():
                    c[tem & ga] = ov.fit_predict(tr[fit], tr[tem & ga], cfg['X'], rounds=cfg['rounds'], clip=clip, params=cfg['params'])
    a = cfg['alpha']
    m6 = ov.rules(tr.MVT_ID_mvt, tr.base + a * c6) - tr.base.to_numpy()
    mf = ov.rules(tr.MVT_ID_mvt, tr.base + a * cf) - tr.base.to_numpy()
    return pd.DataFrame({'MVT_ID_mvt': tr.MVT_ID_mvt, f'm6_{item}': m6, f'mf_{item}': mf})


p = ov.load_proxy()
p = p[p.split == 'train'][['MVT_ID_mvt', 'fold', 'month', 'apt', 'y', 'v66_pred_s']].rename(columns={'v66_pred_s': 'base'})
res = {}
for item in (5, 4, 2, 11):
    o = oof(item)
    p = p.merge(o, on='MVT_ID_mvt', how='left')
    sub = p[p[f'm6_{item}'].notna()].reset_index(drop=True)
    res[f'item{item}_alone'] = ov.judge(sub, sub.base + sub[f'm6_{item}'], sub.base + sub[f'mf_{item}'], f'item {item} alone')
    r = res[f'item{item}_alone']
    print(item, {k: r[k] for k in ('rows_2025', 'sixfold_dsse_M', 'folds_better', 'forward_H1toH2_M', 'forward_H2toH1_M', 'jan_sixfold_M', 'pass')}, flush=True)
m6c = [c for c in p.columns if c.startswith('m6_')]
mfc = [c for c in p.columns if c.startswith('mf_')]
U = p[p[m6c].notna().any(axis=1)].reset_index(drop=True)
U['n_layers'] = U[m6c].notna().sum(axis=1)
s6 = ov.rules(U.MVT_ID_mvt, U.base + U[m6c].fillna(0).sum(axis=1))
sf = ov.rules(U.MVT_ID_mvt, U.base + U[mfc].fillna(0).sum(axis=1))
res['union'] = ov.judge(U, s6, sf, 'v67 combined = items 5+4+2+11 stacked', n2026=None)
r = res['union']
print('UNION', {k: r[k] for k in ('rows_2025', 'sixfold_dsse_M', 'by_fold_M', 'folds_better', 'forward_H1toH2_M', 'forward_H2toH1_M',
                                 'jan_sixfold_M', 'jul_sixfold_M', 'body_share', 'pass')}, flush=True)
ov_rows = (U.n_layers > 1).to_numpy()
O = U[ov_rows].reset_index(drop=True)
res['overlap_rows'] = ov.judge(O, s6[ov_rows], sf[ov_rows], 'overlap rows (items 4 + 11), stacked')
alone4 = ov.rules(O.MVT_ID_mvt, O.base + O.m6_4)
res['overlap_rows_item4_only'] = ov.judge(O, alone4, ov.rules(O.MVT_ID_mvt, O.base + O.mf_4), 'overlap rows, item 4 move only')
for k in ('overlap_rows', 'overlap_rows_item4_only'):
    r = res[k]
    print(k, {x: r[x] for x in ('rows_2025', 'sixfold_dsse_M', 'folds_better', 'forward_H1toH2_M', 'forward_H2toH1_M')})
e = (U.y - s6) ** 2 - (U.y - U.base) ** 2
res['union_by_apt_M'] = (e.groupby(U.apt).sum() / 1e6).round(2).to_dict()
res['union_by_month_M'] = {int(k): round(v / 1e6, 2) for k, v in e.groupby(U.month).sum().items()}
print('by apt', res['union_by_apt_M'])
print('by month', res['union_by_month_M'])
U.assign(p6=s6, pf=sf).to_parquet(ov.OUT / 'combo_oof.parquet', index=False)
ov.save('combo_eval', res)
