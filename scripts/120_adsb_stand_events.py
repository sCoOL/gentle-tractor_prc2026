"""ADS-B stand-departure events (2026-09-19): off-block = the first real move away from the stand.

  py -3.14 scripts/120_adsb_stand_events.py 2025-07-15                # stands = X-Plane + where arrivals park
  py -3.14 scripts/120_adsb_stand_events.py 2025-07-15 --dep-stands   # also learn stands from departure starts (worse)

Reads <out>/points_<date>.parquet (118 extract), links take-offs to departures like `118 link`, and per departure
ground phase records:
  first    first sighting of the departure's ground phase
  leave    last point within MOVE m of the parked position (leave_lo) / first point beyond it (leave_hi)
  clear    first point > CLEAR m from every known stand (definitely off the stand)
Rows first seen > CLEAR m from every stand are censored: the off-block came before the first sighting (94 % of them on
2025-07-15), so take-off minus first sighting is a LOWER bound on the taxi. Writes <out>/stand_events_<date>.parquet
and, for labelled days, prints accuracy against the recorded off-block.

2025-07-15, non-copy departures at EHAM/EDDF/LEBL/LSZH first seen at a stand: leave_lo within 60 s of BLOCK 84 %
(median -13 s), first sighting 56 %, AOBT_3 25 %. Stands learned from departure starts include post-push hold spots
and drop leave_lo to 73 %, hence arrivals only by default.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('p118', ROOT / 'scripts' / '118_adsb_probe.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
from common import AIRPORT_META

DATE = next((x for x in sys.argv[1:] if not x.startswith('--')), '2025-07-15')
STILL_M, STILL_S, MOVE_M, AT_STAND_M, CLEAR_M, GAP_S = 8, 30, 12, 40, 80, 900
STAND_FROM_DEPARTURES = '--dep-stands' in sys.argv   # default: stands = X-Plane + where ARRIVALS park (true stands)


def xy(apt, lat, lon):
    m = AIRPORT_META[apt]
    return (np.asarray(lon) - m['lon']) * np.cos(np.radians(m['lat'])) * 111_195.0, (np.asarray(lat) - m['lat']) * 111_195.0


pts = pd.read_parquet(p.points_path(DATE)).sort_values('t')   # final-file change (was p.OUT / points_<date>)
phases, parked = [], {a: [] for a in AIRPORT_META}
for (icao, apt), g in pts.groupby(['icao', 'APT'], sort=False):
    t, gr = g.t.to_numpy(), g.ground.to_numpy(bool)
    x, y = xy(apt, g.lat.to_numpy(), g.lon.to_numpy())
    gs, cs = g.gs.to_numpy(float), g.cs.to_numpy(object)
    # departures: ground phase before each ground->air transition
    for i in np.where(gr[:-1] & ~gr[1:])[0] + 1:
        if t[i] - t[i - 1] > 120:
            continue
        j = i - 1
        while j > 0 and gr[j - 1] and t[j] - t[j - 1] < GAP_S:
            j -= 1
        after_landing = j > 0 and not gr[j - 1] and t[j] - t[j - 1] < GAP_S
        names = [c for c in cs[j:min(i + 30, len(cs))] if c]
        phases.append(dict(icao=icao, APT=apt, cs=max(set(names), key=names.count) if names else None, t_tko=t[i],
                           t=t[j:i], x=x[j:i], y=y[j:i], gs=gs[j:i], after_landing=after_landing))
        d = np.hypot(x[j:i] - x[j], y[j:i] - y[j])
        k = np.argmax(d > STILL_M) if (d > STILL_M).any() else i - j
        if STAND_FROM_DEPARTURES and not after_landing and t[j + k - 1] - t[j] >= STILL_S:
            parked[apt].append((np.median(x[j:j + k]), np.median(y[j:j + k])))
    # arrivals: ground phase after each air->ground transition, parked position = final still points
    for i in np.where(~gr[:-1] & gr[1:])[0] + 1:
        e = i
        while e + 1 < len(t) and gr[e + 1] and t[e + 1] - t[e] < GAP_S:
            e += 1
        d = np.hypot(x[i:e + 1] - x[e], y[i:e + 1] - y[e])
        still = np.where(d <= STILL_M)[0]
        if len(still) and t[e] - t[i + still[0]] >= STILL_S:
            parked[apt].append((np.median(x[i + still[0]:e + 1]), np.median(y[i + still[0]:e + 1])))

stands = {}
xp = pd.read_parquet(common.DATA / 'ext_stand_coords.parquet')
for apt in AIRPORT_META:
    s = xp[(xp.APT == apt) & xp.lat.notna() & xp.lon.notna()]
    sx, sy = xy(apt, s.lat.to_numpy(), s.lon.to_numpy())
    px = np.array(parked[apt]).reshape(-1, 2)
    stands[apt] = np.vstack([np.c_[sx, sy], px]) if len(px) else np.c_[sx, sy]
print('parked positions learned:', {a: len(v) for a, v in parked.items() if v})

rows = []
for ph in phases:
    S = stands[ph['APT']]
    x, y, t = ph['x'], ph['y'], ph['t']
    dmin = np.min(np.hypot(S[:, 0][None, :] - x[:, None], S[:, 1][None, :] - y[:, None]), axis=1) if len(S) else np.full(len(x), np.nan)
    d0 = np.hypot(x - x[0], y - y[0])
    k = np.argmax(d0 > MOVE_M) if (d0 > MOVE_M).any() else None
    c = np.argmax(dmin > CLEAR_M) if (dmin > CLEAR_M).any() else None
    rows.append(dict(icao=ph['icao'], APT=ph['APT'], cs=ph['cs'], t_tko=ph['t_tko'], t_first=t[0], d_stand_first=dmin[0],
                     at_stand=dmin[0] <= AT_STAND_M and not ph['after_landing'],
                     leave_lo=t[k - 1] if k else np.nan, leave_hi=t[k] if k else np.nan,
                     clear=t[c] if c is not None else np.nan, n=len(t)))
ev = pd.DataFrame(rows)

# match to labelled departures exactly like 118 (callsign + take-off within 180 s)
dep, _ = p.link(DATE)
dep = dep[dep.icao.notna()][['MVT_ID_mvt', 'APT', 'icao', 't_tko', 't_blk', 't_a3', 't_mvt', 'TAXITIME_SEC_mvt', 'SCHED_TIME_UTC_mvt']]
ev["APT"] = ev.APT.astype("string"); ev["icao"] = ev.icao.astype("string")
m = dep.merge(ev, on=["APT", "icao", "t_tko"], how="inner")
print("events", len(ev), "linked departures", len(dep), "joined", len(m))
m['copy'] = (m.t_blk - p.epoch(m.SCHED_TIME_UTC_mvt)).abs() <= 10
m = m[~m['copy']]
for c in ['t_first', 'leave_lo', 'leave_hi', 'clear', 't_a3']:
    m['e_' + c] = m[c] - m.t_blk
m['leave_mid'] = (m.leave_lo + m.leave_hi) / 2
m['e_leave_mid'] = m.leave_mid - m.t_blk
m['wait'] = m.leave_lo - m.t_first          # how long it sat at the stand after the transponder came on
m['gap'] = m.leave_hi - m.leave_lo


def summ(g, cols):
    out = {'rows': len(g)}
    for c in cols:
        e = g['e_' + c].dropna()
        out[c] = f"{(e.abs() <= 60).mean():.0%} / med {e.median():+.0f}" if len(e) else '-'
    return pd.Series(out)


pd.set_option('display.width', 250)
good = m.APT.isin(['EHAM', 'EDDF', 'LEBL', 'LSZH'])
st = m[good & m.at_stand].copy()
st['wait_bin'] = pd.cut(st.wait, [-1, 30, 120, 600, 1e9], labels=['<30s', '30s-2m', '2-10m', '>10m'])
cols = ['t_first', 'leave_lo', 'leave_mid', 'clear', 't_a3']
print(f'\n{DATE} labelled non-copy departures at EHAM/EDDF/LEBL/LSZH first seen at a stand; cells = share within 60 s / median error')
print(summ(st, cols).to_frame('ALL').T.to_string())
print(st.groupby('wait_bin', observed=True).apply(lambda g: summ(g, cols)).to_string())
print(st.groupby('APT').apply(lambda g: summ(g, cols)).to_string())
cen = m[good & ~m.at_stand & (m.d_stand_first > CLEAR_M)]
print(f'\nfirst seen > {CLEAR_M} m from every stand (censored, off-block before first sighting): {len(cen)} rows;'
      f' first sighting after the off-block in {(cen.e_t_first > 0).mean():.0%}, median {cen.e_t_first.median():+.0f} s')
m.to_parquet(p.OUT / f'stand_events_{DATE}.parquet', index=False)
