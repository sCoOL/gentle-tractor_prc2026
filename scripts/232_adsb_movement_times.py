"""ADS-B timestamps and positions per movement, for the explorer.

  py -3.14 scripts/232_adsb_movement_times.py --workers 6

Departures: the event matched by script 121 (`match_day`: callsign + take-off within 180 s, else time/position within
45 s) gives take-off (first airborne point), off-block (`leave_lo`, the last point before the first real move, 121's
definition) and first sighting; the ground phase's first and last points give the first/last seen times and positions.
Arrivals: the air-to-ground transition of the aircraft with the same callsign landing within 180 s of the recorded
landing (unique on both sides, as in script 213) gives the landing (last airborne point); first seen is the earliest
point of that aircraft in the airport box during the 30 min before landing, last seen the end of the ground phase
after landing (a gap over 15 min or a take-off ends it). Output <PRC_ADSB_DIR>/adsb_times.parquet, one row per
MVT_ID_mvt, epoch seconds. Days are cached in <PRC_ADSB_DIR>/times/.
"""
import argparse
import importlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

tf = importlib.import_module('213_turnaround_features')
TOL, APPROACH_S, GAP_S = 180.0, 1800.0, 900.0
COLS = ['MVT_ID_mvt', 'adsb_icao', 'adsb_link', 'adsb_takeoff_t', 'adsb_offblock_t', 'adsb_landing_t', 'adsb_first_t',
        'adsb_first_lat', 'adsb_first_lon', 'adsb_last_t', 'adsb_last_lat', 'adsb_last_lon']


def day(job):
    date, arr_rows = job
    e = importlib.import_module('121_adsb_events')
    f = e.OUT / 'times' / f'{date}.parquet'
    if f.exists():
        return pd.read_parquet(f)
    p = pd.read_parquet(e.P118.points_path(date), columns=['icao', 'APT', 't', 'lat', 'lon', 'ground', 'cs'])
    p = p.sort_values(['icao', 'APT', 't'], kind='stable').reset_index(drop=True)
    p['cs'] = p.cs.astype('string').str.strip().replace('', pd.NA)
    t, gr, lat, lon = p.t.to_numpy(), p.ground.to_numpy(bool), p.lat.to_numpy(), p.lon.to_numpy()
    groups = p.groupby(['icao', 'APT'], sort=False).indices
    rows = []
    # departures: 121's own match, so off-block agrees with adsb_features
    _, mt = e.match_day(date)
    for r in (mt.itertuples() if len(mt) else []):
        g = groups.get((r.icao, r.APT))
        if g is None:
            continue
        gt = t[g]
        a = g[max(np.searchsorted(gt, r.t_first, side='left'), 0)]
        b = g[min(np.searchsorted(gt, r.t_tko, side='right') - 1, len(g) - 1)]
        rows.append((r.MVT_ID_mvt, r.icao, 'callsign' if r.how == 1 else 'time/position', r.t_tko,
                     r.leave_lo if np.isfinite(r.leave_lo) else np.nan, np.nan,
                     t[a], lat[a], lon[a], t[b], lat[b], lon[b]))
    # arrivals: air -> ground transition, callsign + landing time, unique both ways
    ic, ap = p.icao.to_numpy(), p.APT.to_numpy()
    same = np.r_[False, (ic[1:] == ic[:-1]) & (ap[1:] == ap[:-1])]
    dt = np.r_[np.inf, np.diff(t)]
    land = np.flatnonzero(same & gr & ~np.r_[True, gr[:-1]] & (dt <= 120))
    ev = pd.DataFrame({'i': land, 'icao': ic[land], 'APT': ap[land], 'cs': p.cs.to_numpy()[land], 't_ev': t[land - 1]})
    ev = ev.dropna(subset=['cs']).reset_index(drop=True).assign(k=lambda z: np.arange(len(z)))
    c = arr_rows.merge(ev, on=['APT', 'cs'])
    c = c[(c.t_ev - c.t).abs() <= TOL]
    c = c[c.groupby('MVT_ID_mvt').MVT_ID_mvt.transform('size').eq(1) & c.groupby('k').k.transform('size').eq(1)]
    for r in c.itertuples():
        g = groups[(r.icao, r.APT)]
        lo, hi = g[0], g[-1]
        a = r.i
        while a > lo and t[a - 1] >= r.t_ev - APPROACH_S and t[a] - t[a - 1] <= GAP_S:
            a -= 1
        b = r.i
        while b < hi and gr[b + 1] and t[b + 1] - t[b] <= GAP_S:
            b += 1
        rows.append((r.MVT_ID_mvt, r.icao, 'callsign', np.nan, np.nan, r.t_ev, t[a], lat[a], lon[a], t[b], lat[b], lon[b]))
    out = pd.DataFrame(rows, columns=COLS)
    f.parent.mkdir(exist_ok=True)
    out.to_parquet(f, index=False)
    return out


def main(workers):
    e = importlib.import_module('121_adsb_events')
    d = tf.movements()
    arr = d.loc[(d.PHASE_mvt == 'ARR') & d.cs.notna(), ['MVT_ID_mvt', 'APT', 'cs', 't', 'date']]
    by = dict(tuple(arr.groupby('date')))
    cols = ['MVT_ID_mvt', 'APT', 'cs', 't']
    jobs = [(x, by[x][cols] if x in by else pd.DataFrame(columns=cols)) for x in e.days()]
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(day, jobs, chunksize=2))
    out = pd.concat(parts, ignore_index=True).drop_duplicates('MVT_ID_mvt', keep=False)
    out.to_parquet(e.OUT / 'adsb_times.parquet', index=False)
    print(f'{len(out):,} movements with ADS-B times; departures with take-off {out.adsb_takeoff_t.notna().sum():,}, '
          f'with off-block {out.adsb_offblock_t.notna().sum():,}; arrivals with landing {out.adsb_landing_t.notna().sum():,}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    main(ap.parse_args().workers)
