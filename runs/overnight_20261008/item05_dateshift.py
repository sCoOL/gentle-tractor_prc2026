"""Item 5 (lead #12): date-shift / recorded-time-shift detection by ADS-B within +-24 h.

Departures with no v2 ADS-B match on their recorded take-off are searched for an UNCLAIMED adsb.lol take-off (no v2
match claimed it: push_phases holds the (icao, t_tko) of every v2-matched departure) at the same airport with the same
normalised callsign (scripts/flightnorm.py key / base key of the NM callsign, the airport flight number, the destination
arrival's callsign, and 255's learned aliases), 180 s < |recorded - ADS-B| <= 24 h. Nearest one wins.
s = recorded take-off - ADS-B take-off. Scored on 2025 labels: classes (normal y <= 3600, copy |y - D| < 120, date bug
y > 20 h, other long), and the relation y vs s (a recorded-take-off shift puts y near s + taxi).
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ov  # noqa: E402

common = ov.common
m255 = importlib.import_module('255_adsb_match_v2')
fn = m255.fn
ADSB = common.prc_config.ADSB_DIR   # 121 events + push phases (PRC_ADSB_DIR)
m255._ensure_map()

ev = pd.concat([pd.read_parquet(f, columns=['icao', 'APT', 'cs', 't_tko', 'n_ground', 'at_stand'])
                for f in sorted((ADSB / 'events').glob('events_*.parquet'))], ignore_index=True)
ph = pd.read_parquet(ADSB / 'push_phases.parquet', columns=['MVT_ID_mvt', 'icao', 't_tko'])
claimed = set(zip(ph.icao, ph.t_tko.round(1)))
ev['claimed'] = [(i, round(t, 1)) in claimed for i, t in zip(ev.icao, ev.t_tko)]
print('events', len(ev), 'claimed', int(ev.claimed.sum()), flush=True)
ev = ev[~ev.claimed & ev.cs.notna()].copy()
u = ev.cs.unique()
km = dict(zip(u, [fn.key(x) for x in u]))
bm = dict(zip(u, [fn.base_key(x) for x in u]))
evk = pd.concat([pd.DataFrame({'e': ev.index, 'ncs': ev.cs.map(km)}), pd.DataFrame({'e': ev.index, 'ncs': ev.cs.map(bm)})])
evk = evk[evk.ncs != ''].drop_duplicates()

cols = ['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'CALLSIGN_flt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'TAXITIME_SEC_mvt']
b = pd.concat([pd.read_parquet(common.DATA / f, columns=cols).assign(split=s)
               for f, s in [('base_train.parquet', 'train'), ('base_rank.parquet', 'rank')]], ignore_index=True)
b = b[~b.MVT_ID_mvt.isin(ph.MVT_ID_mvt) & b.APT.ne('LTFM')].copy()
b['t'] = ov.b53._epoch(b.MVT_TIME_UTC_mvt)
b['bf_cs'] = m255.backfill_callsigns().reindex(b.MVT_ID_mvt).to_numpy()
print('unmatched departures (no LTFM)', len(b), b.split.value_counts().to_dict(), flush=True)
al = m255.aliases()
cand = []
for col in ('CALLSIGN_flt', 'FLIGHT_mvt', 'bf_cs'):
    uu = b[col].dropna().unique()
    for f in (fn.key, fn.base_key):
        mp = dict(zip(uu, [f(x) for x in uu]))
        cand.append(pd.DataFrame({'MVT_ID_mvt': b.MVT_ID_mvt, 'ncs': b[col].map(mp)}))
alias_rows = [(m, c) for m, a, f in zip(b.MVT_ID_mvt, b.APT, b.FLIGHT_mvt) if isinstance(f, str)
              for c in al.get((a, fn.key(f)), [])]
cand.append(pd.DataFrame(alias_rows, columns=['MVT_ID_mvt', 'ncs']))
cand = pd.concat(cand, ignore_index=True).dropna()
cand = cand[cand.ncs != ''].drop_duplicates()
c = b[['MVT_ID_mvt', 'APT', 't']].merge(cand, on='MVT_ID_mvt').merge(evk, on='ncs')
c = c.merge(ev[['APT', 't_tko', 'n_ground', 'at_stand', 'icao', 'cs']], left_on='e', right_index=True, suffixes=('', '_ev'))
c = c[c.APT == c.APT_ev]
c['s'] = c.t - c.t_tko
c = c[(c.s.abs() > 180) & (c.s.abs() <= 86400)]
c = c.assign(a=c.s.abs()).sort_values('a').drop_duplicates('MVT_ID_mvt')
c = c.drop_duplicates('e')                    # one ADS-B take-off explains one departure
d = b.merge(c[['MVT_ID_mvt', 's', 'n_ground', 'at_stand', 'icao', 'cs']], on='MVT_ID_mvt', how='inner')
d['y'] = d.TAXITIME_SEC_mvt.astype(float)
d['D'] = d.t - ov.b53._epoch(d.SCHED_TIME_UTC_mvt)
d['cls'] = np.select([d.y <= 3600, (d.y - d.D).abs() < 120, d.y > 72000], ['normal', 'copy', 'datebug'], 'long_other')
d.loc[d.split == 'rank', 'cls'] = 'unknown'
d['sband'] = pd.cut(d.s, [-86401, -21600, -3600, -180, 180, 3600, 21600, 86401])
d['y_minus_s'] = d.y - d.s
p = ov.load_proxy()[['MVT_ID_mvt', 'fold', 'month', 'v66_pred_s']]
d = d.merge(p, on='MVT_ID_mvt', how='left')
d['r'] = d.y - d.v66_pred_s
d.assign(sband=d.sband.astype(str)).to_parquet(ov.OUT / 'item05_rows.parquet', index=False)
tr = d[d.split == 'train']
print(pd.crosstab(tr.sband, tr.cls, margins=True).to_string())
print('rank flagged by band:', d[d.split == 'rank'].sband.value_counts().sort_index().to_dict())
# Does y track s?  (y - s) near a taxi time => a recorded take-off shift
for band, g in tr.groupby('sband', observed=True):
    q = g.y_minus_s.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(0).to_dict()
    near = ((g.y_minus_s > 0) & (g.y_minus_s < 3600)).mean()
    print(band, 'n', len(g), 'y-s quantiles', q, 'share 0<y-s<3600', round(near, 3), 'mean r', round(g.r.mean(), 1),
          'rmse', round(np.sqrt((g.r ** 2).mean()), 1))
ov.save('item05_summary', {'crosstab': pd.crosstab(tr.sband.astype(str), tr.cls).to_dict(),
                           'rank_by_band': d[d.split == 'rank'].sband.astype(str).value_counts().to_dict()})
