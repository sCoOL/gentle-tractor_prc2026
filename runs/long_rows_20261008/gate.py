"""Step 3: a separate "long-label gate" on the population that can contain long labels: departures whose take-off is
>= 50 min after schedule (98 % of all 2025 long labels; 286k rows, 1.4 % long).

Hurdle mixture, all inputs label-free (no block time):
  q     = P(y > 3600)                 LightGBM classifier
  S     = E[y | y <= 3600]            LightGBM regressor on short rows
  Lg    = E[y | y > 3600]             LightGBM regressor on long rows
  p     = (1 - q) S + q Lg            (the squared-error optimum of the two-branch model)
Inputs: airport, NM status, D = take-off - schedule, take-off - LOBT / EOBT_1 / IOBT, take-off - AOBT_3 (NM actual taxi),
AOBT_3 - LOBT, ADS-B take-off - stand leave (actual taxi), ADS-B first state / at stand, hour, weekday, airline code,
airline NM rate, v66 itself. Out of fold (six seasonal folds) and forward in time; judged against v66 on the whole
population and on D <= 7 h, per airport, per month, long / short rows; plus blends v66 + a (p - v66).
Classifier quality: AUC, and caught / missed / false alarms at q >= 0.5.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OV = Path(__file__).resolve().parents[1] / 'overnight_20261008'
sys.path.insert(0, str(OV))
import ov  # noqa: E402

common = ov.common
OUT = common.RUNS / 'long_rows_20261008'
b = pd.read_parquet(OUT / 'frame.parquet')
f = ov.frame()[['MVT_ID_mvt', 'op_rate']]
b = b.merge(f, on='MVT_ID_mvt', how='left')
b = b[(b.t - b.sch) > 3000].copy()
b['D'] = b.t - b.sch
b['L'] = b.t - b.lobt
b['E'] = b.t - b.eobt
b['I'] = b.t - b.iobt
b['A3'] = b.t - b.a3
b['a3_lobt'] = b.a3 - b.lobt
b['adsb_taxi'] = b.t - b.t_start
b['adsb_seen'] = b.t - b.t_first
b['fs'] = b.first_state.map({'standing': 0, 'pushing': 1, 'forward': 2, 'moving': 3}).astype(float)
b['at_stand'] = b.at_stand.astype(float)
ts = pd.to_datetime(b.t, unit='s', utc=True)
b['hour'], b['dow'] = ts.dt.hour, ts.dt.dayofweek
b['apt_c'] = b.APT.astype('category').cat.codes.astype(float)
b['st_c'] = b.st.astype('category').cat.codes.astype(float)
al = b.FLIGHT_mvt.astype('string').str.extract(r'^([A-Z]{3}|[A-Z0-9]{2})', expand=False).fillna('')
top = al.value_counts().index[:150]
b['al_c'] = pd.Categorical(al.where(al.isin(top), 'other')).codes.astype(float)
b['base'] = b.v66_pred_s
X = ['apt_c', 'st_c', 'D', 'L', 'E', 'I', 'A3', 'a3_lobt', 'adsb_taxi', 'adsb_seen', 'fs', 'at_stand', 'hour', 'dow', 'al_c',
     'op_rate', 'base']
tr = b[b.split == 'train'].reset_index(drop=True)
rk = b[b.split == 'rank']
long_ = (tr.y > 3600).to_numpy()
print(f'population: 2025 {len(tr):,} rows ({long_.sum():,} long), 2026 {len(rk):,} rows', flush=True)
PC = dict(objective='binary', learning_rate=0.03, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=12, deterministic=True, force_row_wise=True)
PR = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=12, deterministic=True, force_row_wise=True)
CAT = ['apt_c', 'st_c', 'al_c']


def fit3(a, z):
    """q, S, Lg for rows z from training rows a (three seeds each)."""
    q = S = Lg = 0.0
    la = (a.y > 3600).to_numpy()
    for sd in (7, 8, 9):
        q = q + lgb.train({**PC, 'seed': sd}, lgb.Dataset(a[X], la.astype(int), categorical_feature=CAT), 400).predict(z[X]) / 3
        S = S + lgb.train({**PR, 'seed': sd}, lgb.Dataset(a[X][~la], a.y[~la], categorical_feature=CAT), 400).predict(z[X]) / 3
        Lg = Lg + lgb.train({**PR, 'seed': sd, 'min_data_in_leaf': 20}, lgb.Dataset(a[X][la], a.y[la].clip(upper=100000),
                                                                                    categorical_feature=CAT), 300).predict(z[X]) / 3
    return q, S, Lg


h1 = (tr.month <= 6).to_numpy()
res = {}
outs = {}
for mode, splits in (('sixfold', [(tr.fold.ne(k).to_numpy(), tr.fold.eq(k).to_numpy()) for k in ov.FOLDS]),
                     ('forward', [(h1, ~h1), (~h1, h1)])):
    q, S, Lg = np.zeros(len(tr)), np.zeros(len(tr)), np.zeros(len(tr))
    for trm, tem in splits:
        q[tem], S[tem], Lg[tem] = fit3(tr[trm], tr[tem])
        print(f'  {mode} split done', flush=True)
    outs[mode] = (q, S, Lg)
q6, S6, L6 = outs['sixfold']
qf, Sf, Lf = outs['forward']
tr['q6'], tr['S6'], tr['L6'] = q6, S6, L6
auc = roc_auc_score(long_, q6)
caught = int((long_ & (q6 >= 0.5)).sum())
false = int((~long_ & (q6 >= 0.5)).sum())
res['classifier'] = {'auc_sixfold': round(auc, 4), 'auc_forward': round(roc_auc_score(long_, qf), 4), 'long_rows': int(long_.sum()),
                     'caught_q>=0.5': caught, 'missed': int(long_.sum()) - caught, 'false_alarms': false,
                     'by_bucket': pd.DataFrame({'q': pd.cut(q6, [0, 0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]), 'long': long_})
                     .groupby('q', observed=True).long.agg(['size', 'sum', 'mean']).round(3).astype(str).to_dict()}
print(res['classifier'], flush=True)
# value branches: how good are S and Lg on their own rows?
res['branch_rmse'] = {'S_on_short': round(float(np.sqrt(np.mean((tr.y[~long_] - S6[~long_]) ** 2))), 1),
                      'v66_on_short': round(float(np.sqrt(np.mean((tr.y[~long_] - tr.base[~long_]) ** 2))), 1),
                      'Lg_on_long': round(float(np.sqrt(np.mean((tr.y[long_] - L6[long_]) ** 2))), 1),
                      'v66_on_long': round(float(np.sqrt(np.mean((tr.y[long_] - tr.base[long_]) ** 2))), 1)}
print(res['branch_rmse'], flush=True)
m7 = (tr.D <= 7 * 3600).to_numpy()
for al_ in (0.25, 0.5, 0.75, 1.0):
    p6 = ov.rules(tr.MVT_ID_mvt, tr.base + al_ * ((1 - q6) * S6 + q6 * L6 - tr.base))
    pf = ov.rules(tr.MVT_ID_mvt, tr.base + al_ * ((1 - qf) * Sf + qf * Lf - tr.base))
    r = ov.judge(tr, p6, pf, f'long gate a={al_}', n2026=len(rk))
    r7 = ov.judge(tr[m7].reset_index(drop=True), p6[m7], pf[m7], f'long gate a={al_} D<=7h')
    e = (tr.y - p6) ** 2 - (tr.y - tr.base) ** 2
    r['long_rows_M'] = round(float(e[long_].sum() / 1e6), 1)
    r['short_rows_M'] = round(float(e[~long_].sum() / 1e6), 1)
    r['by_apt_M'] = (e.groupby(tr.APT).sum() / 1e6).round(1).to_dict()
    r['by_month_M'] = {int(k): round(v / 1e6, 1) for k, v in e.groupby(tr.month).sum().items()}
    res[f'a{al_}'] = {'all': r, 'D_le_7h': r7}
    print(f'\nalpha {al_}: all', {k: r[k] for k in ('sixfold_dsse_M', 'long_rows_M', 'short_rows_M', 'by_fold_M', 'folds_better',
                                                  'forward_H1toH2_M', 'forward_H2toH1_M', 'jan_sixfold_M', 'pass')})
    print('   D<=7h', {k: r7[k] for k in ('sixfold_dsse_M', 'by_fold_M', 'folds_better', 'forward_H1toH2_M', 'forward_H2toH1_M', 'jan_sixfold_M', 'pass')})
    print('   by apt', r['by_apt_M'])
    print('   by month', r['by_month_M'], flush=True)
imp = lgb.train({**PC, 'seed': 7}, lgb.Dataset(tr[X], long_.astype(int), categorical_feature=CAT), 400).feature_importance('gain')
res['classifier_gain_share'] = dict(sorted(zip(X, (imp / imp.sum()).round(3)), key=lambda x: -x[1]))
print('classifier gain share', res['classifier_gain_share'])
tr.to_parquet(OUT / 'gate_oof.parquet', index=False)
ov.save('../long_rows_20261008/gate', res)
