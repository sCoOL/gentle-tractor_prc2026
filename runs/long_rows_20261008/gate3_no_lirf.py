"""Step 4: four-way gate on the late-vs-schedule population (take-off >= 50 min after schedule), exact copy algebra.

Classes: short (y <= 3600) | copy (|y - D| < 120, y > 3600) | long_other (y > 3600, not copy, y <= 72000) | datebug (y > 72000).
Values: short -> S = E[y | short]; copy -> D exactly; long_other -> Lo = E[y | long_other] (regression with D, take-off - LOBT,
take-off - AOBT_3, ...); datebug -> 86400 + S. Prediction p = sum_k q_k v_k (multiclass LightGBM q, six-fold / forward).
Also the ORACLE bounds on 2025: (i) the class known, values as above; (ii) the class known only for copies; and v66 itself. VARIANT: no LIRF, three classes.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

OV = Path(__file__).resolve().parents[1] / 'overnight_20261008'
sys.path.insert(0, str(OV))
import ov  # noqa: E402

OUT = ov.common.RUNS / 'long_rows_20261008'
g = pd.read_parquet(OUT / 'gate_oof.parquet')
g = g[g.APT != 'LIRF'].reset_index(drop=True)          # LIRF keeps its dedicated gates (A50/A51 matched, v57/v63 unmatched)
X = ['apt_c', 'st_c', 'D', 'L', 'E', 'I', 'A3', 'a3_lobt', 'adsb_taxi', 'adsb_seen', 'fs', 'at_stand', 'hour', 'dow', 'al_c', 'op_rate', 'base']
CAT = ['apt_c', 'st_c', 'al_c']
y, D = g.y.to_numpy(), g.D.to_numpy()
cls = np.select([y <= 3600, np.abs(y - D) < 120], [0, 1], 2)   # no date-bug branch outside LIRF (2 rows)
g['cls'] = cls
print('class counts', pd.Series(cls).value_counts().sort_index().to_dict(), flush=True)
PM = dict(objective='multiclass', num_class=3, learning_rate=0.03, num_leaves=31, min_data_in_leaf=30, feature_fraction=0.8,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=12, deterministic=True, force_row_wise=True)
PR = {**PM, 'objective': 'regression'}
PR.pop('num_class')


def fit(a, z):
    q = np.zeros((len(z), 3))
    S = Lo = 0.0
    ca = a.cls.to_numpy()
    for sd in (7, 8, 9):
        q += lgb.train({**PM, 'seed': sd}, lgb.Dataset(a[X], ca, categorical_feature=CAT), 400).predict(z[X]) / 3
        S = S + lgb.train({**PR, 'seed': sd}, lgb.Dataset(a[X][ca == 0], a.y[ca == 0], categorical_feature=CAT), 400).predict(z[X]) / 3
        Lo = Lo + lgb.train({**PR, 'seed': sd, 'min_data_in_leaf': 20}, lgb.Dataset(a[X][ca == 2], a.y[ca == 2], categorical_feature=CAT),
                            300).predict(z[X]) / 3
    return q, S, Lo


h1 = (g.month <= 6).to_numpy()
res = {}
P = {}
for mode, splits in (('sixfold', [(g.fold.ne(k).to_numpy(), g.fold.eq(k).to_numpy()) for k in ov.FOLDS]), ('forward', [(h1, ~h1), (~h1, h1)])):
    q, S, Lo = np.zeros((len(g), 3)), np.zeros(len(g)), np.zeros(len(g))
    for trm, tem in splits:
        q[tem], S[tem], Lo[tem] = fit(g[trm], g[tem])
    V = np.c_[S, D, Lo]
    P[mode] = ((q * V).sum(axis=1), q, V)
    print(mode, 'done', flush=True)
p6, q6, V6 = P['sixfold']
g['S3'], g['Lo3'] = V6[:, 0], V6[:, 2]
g['p_gate6'], g['p_gatef'] = p6, P['forward'][0]
for k in range(3):
    g[f'q{k}'] = q6[:, k]
base = g.base.to_numpy()
sse = lambda p: float(((y - p) ** 2).sum() / 1e6)  # noqa: E731
orc_all = V6[np.arange(len(g)), cls]
orc_copy = np.where(cls == 1, D, base)
res['sse_M'] = {'v66': round(sse(base), 0), 'gate4_sixfold': round(sse(p6), 0), 'oracle_class_known': round(sse(orc_all), 0),
                'oracle_copies_known_only': round(sse(orc_copy), 0), 'population_rows': int(len(g))}
print(res['sse_M'], flush=True)
# catching: rows per class by argmax
am = q6.argmax(axis=1)
res['confusion_argmax'] = pd.crosstab(pd.Series(cls, name='true'), pd.Series(am, name='argmax')).to_dict()
print(pd.crosstab(pd.Series(cls, name='true'), pd.Series(am, name='argmax')))
for al in (0.1, 0.25, 0.5, 1.0):
    a6 = ov.rules(g.MVT_ID_mvt, base + al * (p6 - base))
    af = ov.rules(g.MVT_ID_mvt, base + al * (P['forward'][0] - base))
    r = ov.judge(g, a6, af, f'gate4 a={al}')
    e = (y - a6) ** 2 - (y - base) ** 2
    r['by_class_M'] = {int(k): round(float(e[cls == k].sum() / 1e6), 1) for k in range(3)}
    r['by_apt_M'] = (pd.Series(e).groupby(g.APT.to_numpy()).sum() / 1e6).round(1).to_dict()
    res[f'a{al}'] = r
    print(al, {k: r[k] for k in ('sixfold_dsse_M', 'by_class_M', 'by_fold_M', 'folds_better', 'forward_H1toH2_M', 'forward_H2toH1_M', 'pass')})
    print('    by apt', r['by_apt_M'], flush=True)
# where does v66 lose vs the oracle? per airport x class
e_v = (y - base) ** 2 - (y - orc_all) ** 2
res['v66_minus_oracle_M_by_apt_class'] = (pd.Series(e_v).groupby([g.APT.to_numpy(), cls]).sum() / 1e6).round(1).unstack().fillna(0).to_dict()
print('v66 SSE above the class-known oracle, M s^2 (airport x class 0 short / 1 copy / 2 long_other / 3 datebug):')
print((pd.Series(e_v).groupby([g.APT.to_numpy(), cls]).sum() / 1e6).round(1).unstack().fillna(0))
g.to_parquet(OUT / 'gate3x_oof.parquet', index=False)
ov.save('../long_rows_20261008/gate3_no_lirf', res)
