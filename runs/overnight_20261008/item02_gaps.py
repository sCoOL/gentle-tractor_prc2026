"""Item 2 (lead #9): LTFM month-end cut + gap-day flags as residual layers on the v66 proxy.

LTFM rows stop at 21:00 UTC on every month end (all twelve 2025 month ends and the 2026 month ends), so forward-
looking context features see an empty airport after the cut. Layer A: LTFM rows within 3 h of the cut (minutes to the
cut, v66, hour). Layer B: label-free gap flags — gap_ratio_day (in-file departures / EUROCONTROL traffic_dep_day) and
unrec_dep_prev60 / unrec_dep_pm60 (adsb.lol take-offs with >= 10 ground points + OPDI runway-entry departures that
match no airport movement, runs/opdi_audit/unrecorded.py) — on rows where they are non-trivial (ratio < 0.9 or >= 1
unrecorded take-off in the previous 60 min).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ov  # noqa: E402

common = ov.common
d = ov.load_proxy()
fr = ov.frame()[['MVT_ID_mvt', 't']]
d = d.merge(fr, on='MVT_ID_mvt', how='left')
d = d[d.apt.notna()].copy()
ts = pd.to_datetime(d.t, unit='s', utc=True)
d['date'] = ts.dt.normalize().dt.tz_localize(None)
d['hour'] = ts.dt.hour
# --- LTFM cut
eom = (ts + pd.offsets.MonthEnd(0)).dt.normalize()
last_day = ts.dt.normalize() == eom
cut = ts.dt.normalize() + pd.Timedelta(hours=21)
d['min_to_cut'] = np.where((d.apt == 'LTFM') & last_day, (cut - ts).dt.total_seconds() / 60, np.nan)
d.loc[d.min_to_cut < 0, 'min_to_cut'] = np.nan
# --- gap_ratio_day
eu = pd.read_parquet(common.DATA / 'ext_eurocontrol_daily.parquet', columns=['APT', 'date', 'traffic_dep_day'])
eu['date'] = pd.to_datetime(eu.date)
n_in = d.groupby(['apt', 'date']).size().rename('n_in').reset_index()
n_in = n_in.merge(eu, left_on=['apt', 'date'], right_on=['APT', 'date'], how='left')
n_in['gap_ratio_day'] = n_in.n_in / n_in.traffic_dep_day
d = d.merge(n_in[['apt', 'date', 'gap_ratio_day']], on=['apt', 'date'], how='left')
# --- unrecorded take-offs
import os  # noqa: E402
ROWS_ONLY = bool(os.environ.get('PRC_ROWS_ONLY'))   # final-file change: skip layer B inputs + screens
if ROWS_ONLY:
    ua = pd.DataFrame({'cls': pd.Series(dtype=str), 'n_ground': pd.Series(dtype=float), 'APT': pd.Series(dtype=str), 't_tko': pd.Series(dtype=float)})
    uo = pd.DataFrame({'cls': pd.Series(dtype=str), 'APT': pd.Series(dtype=str), 't': pd.Series(dtype=float)})
else:
    ua = pd.read_parquet(common.RUNS / 'opdi_audit' / 'unrecorded_adsb.parquet')
ua = ua[(ua.cls == 'not_in_airport_data') & (ua.n_ground >= 10)][['APT', 't_tko']].rename(columns={'t_tko': 'tt'})
if not ROWS_ONLY:
    uo = pd.read_parquet(common.RUNS / 'opdi_audit' / 'unrecorded_opdi.parquet')
uo = uo[uo.cls == 'not_in_airport_data'][['APT', 't']].rename(columns={'t': 'tt'})
u = pd.concat([ua, uo]).sort_values('tt')
u['tr'] = (u.tt // 60).astype(int)
u = u.drop_duplicates(['APT', 'tr'])           # same take-off seen by both sources -> one
d['unrec_dep_prev60'] = 0
d['unrec_dep_pm60'] = 0
for a, g in d.groupby('apt'):
    tt = np.sort(u.tt[u.APT == a].to_numpy())
    t = g.t.to_numpy()
    d.loc[g.index, 'unrec_dep_prev60'] = np.searchsorted(tt, t) - np.searchsorted(tt, t - 3600)
    d.loc[g.index, 'unrec_dep_pm60'] = np.searchsorted(tt, t + 3600) - np.searchsorted(tt, t - 3600)
d['base'] = d.v66_pred_s
d['r'] = d.y - d.base
d['apt_code'] = d.apt.map({a: i for i, a in enumerate(sorted(d.apt.dropna().unique()))}).astype(float)
d.to_parquet(ov.OUT / 'item02_frame.parquet', index=False)
if ROWS_ONLY:   # final-file change: rows only (layer A of the build reads min_to_cut / base / hour)
    raise SystemExit(0)
tr = d[d.split == 'train']
rk = d[d.split == 'rank']
res = {}
# Raw biases
cutrows = tr[tr.min_to_cut.notna() & (tr.min_to_cut <= 180)]
ref = tr[(tr.apt == 'LTFM') & tr.min_to_cut.isna() & tr.hour.between(18, 20)]
res['ltfm_cut_raw'] = {'rows_2025': int(len(cutrows)), 'mean_r': round(cutrows.r.mean(), 1), 'median_r': round(cutrows.r.median(), 1),
                       'ref_same_hours_mean_r': round(ref.r.mean(), 1), 'ref_median_r': round(ref.r.median(), 1),
                       'by_band': cutrows.groupby(pd.cut(cutrows.min_to_cut, [0, 30, 60, 120, 180])).r.agg(['size', 'mean']).round(1).astype(str).to_dict(),
                       'rows_2026': int((rk.min_to_cut <= 180).sum())}
print(res['ltfm_cut_raw'])
for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 1.1), (1.1, 9)):
    m = tr.gap_ratio_day.between(lo, hi, inclusive='left')
    res[f'gap_ratio_{lo}_{hi}'] = {'rows': int(m.sum()), 'mean_r': round(tr.r[m].mean(), 1), 'rows_2026': int(rk.gap_ratio_day.between(lo, hi, inclusive='left').sum())}
for k in (0, 1, 2, 3):
    m = tr.unrec_dep_prev60 == k if k < 3 else tr.unrec_dep_prev60 >= 3
    res[f'unrec_prev60_{k}'] = {'rows': int(m.sum()), 'mean_r': round(tr.r[m].mean(), 1), 'body_mean_r': round(tr.r[m & (tr.y <= 3600)].mean(), 1),
                                'rows_2026': int((rk.unrec_dep_prev60 == k).sum() if k < 3 else (rk.unrec_dep_prev60 >= 3).sum())}
for k, v in res.items():
    if k != 'ltfm_cut_raw':
        print(k, v)
# Layer A: LTFM cut rows
XA = ['min_to_cut', 'base', 'hour']
A = cutrows.reset_index(drop=True)
outA, _ = ov.residual_layer(A, XA, rounds=100, params={**ov.PARAMS, 'min_data_in_leaf': 30, 'num_leaves': 4})
res['layerA'] = {al: ov.judge(A, *outA[al], f'LTFM cut a={al}', n2026=res['ltfm_cut_raw']['rows_2026']) for al in outA}
for al, r in res['layerA'].items():
    print('A', al, {k: r[k] for k in ('sixfold_dsse_M', 'by_fold_M', 'forward_H1toH2_M', 'forward_H2toH1_M', 'pass', 'body_share')})
# Layer B: gap flags
mB = ((tr.gap_ratio_day < 0.9) | (tr.unrec_dep_prev60 >= 1)) & tr.min_to_cut.isna()
B = tr[mB].reset_index(drop=True)
XB = ['gap_ratio_day', 'unrec_dep_prev60', 'unrec_dep_pm60', 'base', 'hour', 'apt_code']
outB, _ = ov.residual_layer(B, XB)
n26B = int((((rk.gap_ratio_day < 0.9) | (rk.unrec_dep_prev60 >= 1)) & rk.min_to_cut.isna()).sum())
res['layerB'] = {al: ov.judge(B, *outB[al], f'gap flags a={al}', n2026=n26B) for al in outB}
for al, r in res['layerB'].items():
    print('B', al, {k: r[k] for k in ('rows_2025', 'sixfold_dsse_M', 'by_fold_M', 'forward_H1toH2_M', 'forward_H2toH1_M',
                                      'jan_sixfold_M', 'pass', 'body_share')})
ov.save('item02_gaps', res)
