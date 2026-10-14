"""The errors come mostly from rows with y > 1 h: what are they, can they be told apart beforehand?

Step 1 (this file): inventory of every 2025 departure, classified by WHAT ITS LABEL EQUALS (priority order):
  datebug     y > 20 h
  copy_sched  recorded block = schedule (+-120 s)
  copy_lobt   block = LOBT (+-60 s)            copy_eobt   block = EOBT_1 (+-60 s)
  nm_agrees   NM's own actual off-block AOBT_3 within 5 min of the block (NM independently confirms the long ground time)
  adsb_real   ADS-B saw the aircraft move off the stand within 5 min of the block (a real long taxi / hold / de-icing)
  adsb_parked ADS-B saw it still parked > 30 min after the recorded block (block recorded early)
  no_witness  none of the above can be checked (no AOBT_3, no ADS-B start)
  other       witnesses exist but disagree
Rows are split y <= 1 h / 1-2 h / 2-8 h / 8-20 h / > 20 h. For each class: rows, SSE share under the v66 proxy, mean
residual, how v66 prices them (median prediction / label), by airport and lane (NM matched / unmatched).
Labels are used only to describe 2025; nothing here is fitted.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OV = Path(__file__).resolve().parents[1] / 'overnight_20261008'
sys.path.insert(0, str(OV))
import ov  # noqa: E402

common = ov.common
OUT = common.RUNS / 'long_rows_20261008'   # frames (data); this script lives in the repository
OUT.mkdir(parents=True, exist_ok=True)
cols = ['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'EOBT_1_flt',
        'IOBT_flt', 'AOBT_3_flt', 'FLIGHT_mvt', 'STAND_mvt']
b = pd.concat([pd.read_parquet(common.DATA / f, columns=cols).assign(split=s)
               for f, s in [('base_train.parquet', 'train'), ('base_rank.parquet', 'rank')]], ignore_index=True)
for c, n in [('MVT_TIME_UTC_mvt', 't'), ('BLOCK_TIME_UTC_mvt', 'blk'), ('SCHED_TIME_UTC_mvt', 'sch'), ('LOBT_flt', 'lobt'),
             ('EOBT_1_flt', 'eobt'), ('IOBT_flt', 'iobt'), ('AOBT_3_flt', 'a3')]:
    b[n] = ov.b53._epoch(b[c])
ph = pd.read_parquet(common.prc_config.ADSB_DIR / 'push_phases.parquet', columns=['MVT_ID_mvt', 't_move', 't_push_start', 't_first', 'first_state', 'at_stand'])
ph['t_start'] = ph.t_push_start.fillna(ph.t_move)
b = b.merge(ph[['MVT_ID_mvt', 't_start', 't_first', 'first_state', 'at_stand']], on='MVT_ID_mvt', how='left')
st = ov.frame()[['MVT_ID_mvt', 'st']]
p = ov.load_proxy()[['MVT_ID_mvt', 'fold', 'month', 'y', 'v66_pred_s']]
b = b.merge(st, on='MVT_ID_mvt', how='left').merge(p, on='MVT_ID_mvt', how='left')
b['lane'] = np.where(b.st == 'matched', 'matched', 'unmatched')
b['D'] = b.t - b.sch
b['r'] = b.y - b.v66_pred_s
b.to_parquet(OUT / 'frame.parquet', index=False)
import os  # noqa: E402
if os.environ.get('PRC_ROWS_ONLY'):   # final-file change: the final file needs the frame only, not the 2025 tables
    raise SystemExit(0)
t = b[b.split == 'train'].copy()
blk = t.blk
cls = np.select(
    [t.y > 72000,
     (blk - t.sch).abs() <= 120,
     (blk - t.lobt).abs() <= 60,
     (blk - t.eobt).abs() <= 60,
     (blk - t.a3).abs() <= 300,
     t.t_start.notna() & ((t.t_start - blk).abs() <= 300),
     t.t_start.notna() & (t.t_start - blk > 1800),
     t.a3.isna() & t.t_start.isna()],
    ['datebug', 'copy_sched', 'copy_lobt', 'copy_eobt', 'nm_agrees', 'adsb_real', 'adsb_parked', 'no_witness'], 'other')
t['cls'] = cls
t['band'] = pd.cut(t.y, [-1, 3600, 7200, 28800, 72000, 1e9], labels=['<=1h', '1-2h', '2-8h', '8-20h', '>20h'])
t[['MVT_ID_mvt', 'cls', 'band']].to_parquet(OUT / 'classes_2025.parquet', index=False)
tot = (t.r ** 2).sum()
pd.set_option('display.width', 250)


def summ(g):
    return pd.Series({'rows': len(g), 'sse_share_%': 100 * (g.r ** 2).sum() / tot, 'mean_r': g.r.mean(),
                      'median_y': g.y.median(), 'median_pred': g.v66_pred_s.median(),
                      'pred_over_y_med': (g.v66_pred_s / g.y).median()})


print('=== all 2025 departures by label band ===')
print(t.groupby('band', observed=True).apply(summ, include_groups=False).round(2).to_string())
L = t[t.y > 3600]
print(f'\n=== rows with y > 1 h: {len(L):,} of {len(t):,} ({100 * len(L) / len(t):.2f} %), '
      f'{100 * (L.r ** 2).sum() / tot:.1f} % of all SSE ===')
print(L.groupby(['cls']).apply(summ, include_groups=False).sort_values('sse_share_%', ascending=False).round(2).to_string())
print('\n--- y > 1 h, class x band (rows) ---')
print(pd.crosstab(L.cls, L.band).to_string())
print('\n--- y > 1 h, class x lane (SSE share %) ---')
print((L.assign(e=L.r ** 2).pivot_table(index='cls', columns='lane', values='e', aggfunc='sum') / tot * 100).round(2).to_string())
print('\n--- y > 1 h, class x airport (rows) ---')
print(pd.crosstab(L.cls, L.APT, margins=True).to_string())
print('\n--- y > 1 h, airport: SSE share % and how v66 prices them ---')
print(L.groupby('APT').apply(summ, include_groups=False).round(2).to_string())
print('\n--- y > 1 h by month (rows, SSE share %) ---')
print(L.groupby('month').apply(lambda g: pd.Series({'rows': len(g), 'sse_%': 100 * (g.r ** 2).sum() / tot}), include_groups=False).round(2).T.to_string())
# how many long rows does v66 already put high (pred > 1 h)?
print('\n--- y > 1 h: share v66 already prices above 1 h, by class ---')
print(L.groupby('cls').apply(lambda g: pd.Series({'pred>1h': (g.v66_pred_s > 3600).mean(), 'pred>0.5y': (g.v66_pred_s > 0.5 * g.y).mean()}),
                             include_groups=False).round(3).to_string())
# the converse: rows v66 prices > 1 h whose label is <= 1 h (false alarms of the existing gates)
F = t[(t.v66_pred_s > 3600) & (t.y <= 3600)]
print(f'\nrows v66 prices > 1 h but y <= 1 h: {len(F):,}, SSE share {100 * (F.r ** 2).sum() / tot:.2f} %')
