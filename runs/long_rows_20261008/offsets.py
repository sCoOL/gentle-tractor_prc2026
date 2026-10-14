"""Step 2: for long rows not explained by a plain copy, what does the recorded block time equal?
Offsets of the block from: schedule (whole hours = local-time / timezone shifts), LOBT / EOBT_1 / AOBT_3 (+-1 h, +-2 h),
the inbound arrival's in-block / landing / schedule (scripts/231 links, any source), the ADS-B first sighting and stand
leave; plus the date-bug anatomy (y - 86400 vs D). Prints counts per hypothesis and per airport, and the leftover rows.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OV = Path(__file__).resolve().parents[1] / 'overnight_20261008'
sys.path.insert(0, str(OV))
import ov  # noqa: E402

common = ov.common
OUT = ov.common.RUNS / 'long_rows_20261008'
b = pd.read_parquet(OUT / 'frame.parquet')
c = pd.read_parquet(OUT / 'classes_2025.parquet')
t = b.merge(c, on='MVT_ID_mvt', how='inner')
raw = ['MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'PHASE_mvt']
m = pd.concat([pd.read_parquet(f, columns=raw) for f in sorted(common.DATA.glob('training_2025-*.parquet'))], ignore_index=True)
arr = m[m.PHASE_mvt == 'ARR'].set_index('MVT_ID_mvt')
lk = pd.read_parquet(common.RUNS / 'v51_turnaround' / 'links.parquet', columns=['arr_mvt', 'dep_mvt', 'link_source'])
lk = lk.drop_duplicates('dep_mvt').set_index('dep_mvt')
t['arr_id'] = lk.arr_mvt.reindex(t.MVT_ID_mvt).to_numpy()
a = arr.reindex(t.arr_id)
t['arr_blk'] = ov.b53._epoch(a.BLOCK_TIME_UTC_mvt).values if False else ov.b53._epoch(pd.Series(a.BLOCK_TIME_UTC_mvt.to_numpy()))
t['arr_land'] = ov.b53._epoch(pd.Series(a.MVT_TIME_UTC_mvt.to_numpy()))
t['arr_sch'] = ov.b53._epoch(pd.Series(a.SCHED_TIME_UTC_mvt.to_numpy()))
L = t[t.y > 3600].copy()
blk = L.blk
H = {}
for k in (1, 2, -1, -2):
    H[f'sched{k:+d}h'] = (blk - (L.sch + k * 3600)).abs() <= 120
    H[f'lobt{k:+d}h'] = (blk - (L.lobt + k * 3600)).abs() <= 120
    H[f'a3{k:+d}h'] = (blk - (L.a3 + k * 3600)).abs() <= 120
H['arr_inblock'] = (blk - L.arr_blk).abs() <= 120
H['arr_landing'] = (blk - L.arr_land).abs() <= 120
H['arr_sched'] = (blk - L.arr_sch).abs() <= 120
H['adsb_first_seen'] = (blk - L.t_first).abs() <= 120
H['block_on_00s'] = (blk % 60 == 0)
H['block_on_5min'] = (blk % 300 == 0)
hyp = pd.DataFrame(H)
pd.set_option('display.width', 250)
for cl in ('other', 'no_witness', 'adsb_parked', 'copy_sched'):
    g = hyp[L.cls == cl]
    print(f'\n=== {cl}: {len(g)} rows — share matching each hypothesis ===')
    print((g.mean() * 100).round(1).sort_values(ascending=False).head(14).to_string())
# first hypothesis that explains each 'other' / 'no_witness' row
order = ['arr_inblock', 'arr_landing', 'arr_sched', 'sched+1h', 'sched-1h', 'sched+2h', 'sched-2h', 'lobt+1h', 'lobt-1h',
         'lobt+2h', 'lobt-2h', 'a3+1h', 'a3-1h', 'a3+2h', 'a3-2h', 'adsb_first_seen']
expl = pd.Series('unexplained', index=L.index)
for h in order[::-1]:
    expl[hyp[h].fillna(False)] = h
L['expl'] = np.where(L.cls.isin(['other', 'no_witness', 'adsb_parked']), expl, L.cls)
tot = (t.r ** 2).sum()
print('\n=== long rows: refined class (rows, SSE share %, median y, median v66) ===')
g = L.groupby('expl').apply(lambda g: pd.Series({'rows': len(g), 'sse_%': 100 * (g.r ** 2).sum() / tot, 'median_y': g.y.median(),
                                                 'median_pred': g.v66_pred_s.median()}), include_groups=False)
print(g.sort_values('sse_%', ascending=False).round(2).to_string())
print('\n--- refined class x airport ---')
print(pd.crosstab(L.expl, L.APT).to_string())
# unexplained matched rows: how far is the block from AOBT_3 / schedule / ADS-B start?
U = L[L.expl == 'unexplained']
print(f'\n=== unexplained: {len(U)} rows ===')
for name, ref in (('block - AOBT_3', U.a3), ('block - schedule', U.sch), ('block - LOBT', U.lobt), ('ADS-B start - block', None)):
    x = (U.t_start - U.blk) if ref is None else (U.blk - ref)
    x = x.dropna() / 60
    print(f'{name} (min): n {len(x)}, quantiles', x.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(0).to_dict())
x = (U.y - (U.t - U.a3)).dropna() / 60
print('y - (take-off - AOBT_3) (min): ', x.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(0).to_dict())
# date bugs
d = t[t.cls == 'datebug']
print('\n=== date bugs ===')
print(d.assign(y_minus_day=d.y - 86400, D_min=d.D / 60, pred=d.v66_pred_s)[['APT', 'lane', 'y', 'y_minus_day', 'D_min', 'pred', 'FLIGHT_mvt']].round(0).to_string())
L.to_parquet(OUT / 'long_rows_2025.parquet', index=False)
