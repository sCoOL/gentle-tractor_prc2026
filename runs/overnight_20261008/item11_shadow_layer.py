"""Item 11 (A9): joint coverage shadow — arrival disappearance informs departure appearance, same period.

For every v2-matched departure first seen already moving (push_phases first_state forward / moving, or not at a stand),
the ADS-B "head" t_first - off-block is unknown. Arrivals at the same stand in the same year-month tell how long the
aircraft is invisible near that stand: tail = recorded in-block - last ADS-B ground point (item11_arr_tail_extract.py;
arrival in-block is given in both years, so this is label-free for departures and measured in the SAME period, which
is what the 2025-fitted 122 shadow model cannot know when coverage changes, e.g. EGLL / EDDF 2025 -> 2026).
Features: per (airport, stand, year, month) median tail and disappear share (tail > 60 s), shrunk to the airport-month
(k = 10 arrivals); t_mvt - t_first; v66; hour; airport. Residual layer on those departures, pass rule.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ov  # noqa: E402

common = ov.common
K = 10.0
a = pd.read_parquet(ov.OUT / 'item11_arr_tail.parquet')
a['stand'] = a.STAND_mvt.astype('string').str.strip()
a['ym'] = pd.to_datetime(a.t_land, unit='s', utc=True).dt.strftime('%Y-%m')
a['tailc'] = a['tail'].clip(-600, 1800)
a['gone'] = (a['tail'] > 60).astype(float)
am = a.groupby(['APT', 'ym']).agg(m_am=('tailc', 'median'), g_am=('gone', 'mean')).reset_index()
st = a.groupby(['APT', 'stand', 'ym']).agg(m_st=('tailc', 'median'), g_st=('gone', 'mean'), n_st=('tailc', 'size')).reset_index()
ph = pd.read_parquet(common.prc_config.ADSB_DIR / 'push_phases.parquet', columns=['MVT_ID_mvt', 'APT', 't_first', 'first_state', 'at_stand', 'd_stand_first'])
raw = ['MVT_ID_mvt', 'STAND_mvt', 'PHASE_mvt']
m = pd.concat([pd.read_parquet(f, columns=raw) for f in sorted(common.DATA.glob('training_2025-*.parquet'))]
              + [pd.read_parquet(common.DATA / 'ranking.parquet', columns=raw)], ignore_index=True)
m = m[m.PHASE_mvt == 'DEP'][['MVT_ID_mvt', 'STAND_mvt']]
fr = ov.frame()[['MVT_ID_mvt', 't']]
d = ph.merge(m, on='MVT_ID_mvt', how='left').merge(fr, on='MVT_ID_mvt')
d['stand'] = d.STAND_mvt.astype('string').str.strip()
d['ym'] = pd.to_datetime(d.t, unit='s', utc=True).dt.strftime('%Y-%m')
d = d[d.first_state.isin(['forward', 'moving']) | ~d.at_stand.astype(bool)].copy()
d = d.merge(am, on=['APT', 'ym'], how='left').merge(st, on=['APT', 'stand', 'ym'], how='left')
w = (d.n_st.fillna(0) / (d.n_st.fillna(0) + K)).to_numpy()
d['tail_stand'] = np.where(np.isfinite(d.m_st), w * d.m_st.fillna(0) + (1 - w) * d.m_am, d.m_am)
d['gone_stand'] = np.where(np.isfinite(d.g_st), w * d.g_st.fillna(0) + (1 - w) * d.g_am, d.g_am)
d['x_first'] = d.t - d.t_first
p = ov.load_proxy()[['MVT_ID_mvt', 'split', 'fold', 'month', 'y', 'v66_pred_s']]
d = d.merge(p, on='MVT_ID_mvt', how='inner')
d['base'] = d.v66_pred_s
d['r'] = d.y - d.base
d['hour'] = pd.to_datetime(d.t, unit='s', utc=True).dt.hour
d['apt_code'] = d.APT.map({x: i for i, x in enumerate(sorted(common.AIRPORT_META))}).astype(float)
d['fs'] = d.first_state.map({'standing': 0, 'pushing': 1, 'forward': 2, 'moving': 3}).astype(float)
d.to_parquet(ov.OUT / 'item11_rows.parquet', index=False)
import os  # noqa: E402
if os.environ.get('PRC_ROWS_ONLY'):   # final-file change: the final file needs the rows only, not the 2025 screen
    raise SystemExit(0)
tr = d[d.split == 'train'].reset_index(drop=True)
n26 = int((d.split == 'rank').sum())
print('rows', len(tr), '2026', n26, tr.APT.value_counts().to_dict(), flush=True)
print('2026 by apt', d[d.split == 'rank'].APT.value_counts().to_dict())
# does the residual follow the same-period arrival tail?
print(tr.groupby(pd.qcut(tr.tail_stand, 5, duplicates='drop'), observed=True).r.agg(['size', 'mean']).round(1).to_string())
res = {}
for name, X in (('with_arrival_tail', ['x_first', 'tail_stand', 'gone_stand', 'm_am', 'g_am', 'd_stand_first', 'fs', 'base', 'hour', 'apt_code']),
                ('control_no_tail', ['x_first', 'd_stand_first', 'fs', 'base', 'hour', 'apt_code'])):
    out, _ = ov.residual_layer(tr, X, by='APT', rounds=200, params={**ov.PARAMS, 'num_leaves': 15, 'min_data_in_leaf': 200})
    for al, (p6, pf) in out.items():
        r = ov.judge(tr, p6, pf, f'{name} a={al}', n2026=n26)
        r['by_apt_M'] = (pd.Series((tr.y - p6) ** 2 - tr.r ** 2).groupby(tr.APT).sum() / 1e6).round(2).to_dict()
        res[f'{name}_{al}'] = r
        print(name, al, {k: r[k] for k in ('sixfold_dsse_M', 'by_fold_M', 'folds_better', 'forward_H1toH2_M', 'forward_H2toH1_M',
                                           'jan_sixfold_M', 'body_share', 'pass')}, flush=True)
        print('   by apt', r['by_apt_M'], flush=True)
ov.save('item11_shadow_layer', res)
