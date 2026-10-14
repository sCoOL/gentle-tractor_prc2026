"""Item 4 (lead #10): stand-error flag = ADS-B parked position of the departure vs the learned location of its RECORDED
stand; flag > 60 m. Learned location = per (airport, stand) median parked position over its departures that were seen
standing (at_stand), both years (label-free), >= 5 departures. Residual of the v66 proxy flagged vs unflagged, then a
residual layer on the flagged rows (distance, flag, v66, hour, airport) under the pass rule.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ov  # noqa: E402

common = ov.common
pk = pd.read_parquet(ov.OUT / 'item04_park.parquet')
ph = pd.read_parquet(common.prc_config.ADSB_DIR / 'push_phases.parquet', columns=['MVT_ID_mvt', 'APT', 'at_stand', 'stand_pts', 'd_stand_first',
                                                                    'push_type', 'after_landing'])
raw = ['MVT_ID_mvt', 'STAND_mvt', 'PHASE_mvt']
m = pd.concat([pd.read_parquet(f, columns=raw) for f in sorted(common.DATA.glob('training_2025-*.parquet'))]
              + [pd.read_parquet(common.DATA / 'ranking.parquet', columns=raw)], ignore_index=True)
m = m[m.PHASE_mvt == 'DEP'][['MVT_ID_mvt', 'STAND_mvt']]
d = pk.merge(ph, on='MVT_ID_mvt').merge(m, on='MVT_ID_mvt', how='left')
d['stand'] = d.STAND_mvt.astype('string').str.strip()
lat0 = {a: common.AIRPORT_META[a].get('lat', None) for a in common.AIRPORT_META}
k = np.cos(np.radians(d.lat_med.to_numpy()))
d['xm'] = d.lon_med * 111320.0 * k
d['ym'] = d.lat_med * 110540.0
seen = d[d.at_stand.astype(bool) & d.stand.notna() & (d.stand != '')]
seen = seen.merge(ov.load_proxy()[['MVT_ID_mvt', 'split']], on='MVT_ID_mvt')
loc = seen.groupby(['APT', 'stand', 'split']).agg(sx=('xm', 'median'), sy=('ym', 'median'), n=('xm', 'size'),
                                        spread=('xm', lambda s: float(np.median(np.abs(s - s.median()))))).reset_index()
loc = loc[loc.n >= 5]
d = d.merge(ov.load_proxy()[['MVT_ID_mvt', 'split']], on='MVT_ID_mvt').merge(loc, on=['APT', 'stand', 'split'], how='left')
d['dist'] = np.hypot(d.xm - d.sx, d.ym - d.sy)
d['flag'] = (d.dist > 60) & d.at_stand.astype(bool)
p = ov.load_proxy()[['MVT_ID_mvt', 'fold', 'month', 'y', 'v66_pred_s', 'layer66']]
d = d.merge(p, on='MVT_ID_mvt', how='inner')
d['base'] = d.v66_pred_s
d['r'] = d.y - d.base
d.to_parquet(ov.OUT / 'item04y_rows.parquet', index=False)
import os  # noqa: E402
if os.environ.get('PRC_ROWS_ONLY'):   # final-file change: the final file needs the rows only, not the 2025 screen
    raise SystemExit(0)
ok = d.dist.notna() & d.at_stand.astype(bool)
print('rows with a learned stand + standing:', int(ok.sum()), 'of', len(d))
t = d[ok].groupby(['APT', 'split']).agg(n=('flag', 'size'), flagged=('flag', 'sum'), share=('flag', 'mean')).unstack()
print(t.round(3).to_string())
tr = d[ok & (d.split == 'train')]
g = tr.groupby('flag').agg(n=('r', 'size'), mean_r=('r', 'mean'), rmse=('r', lambda s: np.sqrt((s ** 2).mean())),
                           body_mean_r=('r', lambda s: s[tr.loc[s.index, 'y'] <= 3600].mean()))
print(g.round(1).to_string())
bands = pd.cut(tr.dist, [0, 30, 60, 100, 200, 500, 1e9])
print(tr.groupby(bands, observed=True).r.agg(['size', 'mean', lambda s: np.sqrt((s ** 2).mean())]).round(1).to_string())
# layer on flagged rows
F = tr[tr.flag].reset_index(drop=True)
F['hour'] = pd.to_datetime(0, unit='s')  # placeholder replaced below
fr = ov.frame()[['MVT_ID_mvt', 't']]
F = F.drop(columns='hour').merge(fr, on='MVT_ID_mvt', how='left')
F['hour'] = pd.to_datetime(F.t, unit='s', utc=True).dt.hour
F['apt_code'] = F.APT.map({a: i for i, a in enumerate(sorted(common.AIRPORT_META))}).astype(float)
F['logd'] = np.log1p(F.dist)
X = ['logd', 'base', 'hour', 'apt_code', 'd_stand_first', 'spread']
n26 = int((d.flag & (d.split == 'rank')).sum())
out, _ = ov.residual_layer(F, X)
res = {'flag_table': t.reset_index().astype(str).to_dict('records'), 'resid_by_flag': g.round(1).reset_index().to_dict('records')}
for al, (p6, pf) in out.items():
    res[al] = ov.judge(F, p6, pf, f'stand error a={al}', n2026=n26)
    r = res[al]
    print(al, {k_: r[k_] for k_ in ('rows_2025', 'sixfold_dsse_M', 'by_fold_M', 'folds_better', 'forward_H1toH2_M', 'forward_H2toH1_M',
                                    'jan_sixfold_M', 'body_share', 'pass')})
ov.save('item04y_stand_error', res)
