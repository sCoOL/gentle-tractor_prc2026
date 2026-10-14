"""Item 4 step 1: the parked position of every v2-matched departure = its first ADS-B ground point (push_phases t_first),
plus the median position of its first 5 ground points within 30 s. Label-free. -> item04_park.parquet
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import common  # noqa: E402
import prc_config  # noqa: E402

ADSB = prc_config.ADSB_DIR       # push phases (PRC_ADSB_DIR)
POINTS = prc_config.ADSB_POINTS  # extracted points_<date> (PRC_ADSB_POINTS, read only)
import os  # noqa: E402
SINCE = os.environ.get('PRC_SINCE', '')   # final-file change: only departures dated >= SINCE (the 2026 rank days)
WORKERS = int(os.environ.get('PRC_WORKERS', '10'))   # final-file change: was a fixed 10
OUT = common.RUNS / 'overnight_20261008'


def one(args):
    date, keys = args
    f = POINTS / f'points_{date}.parquet'
    if not f.exists():
        return None
    p = pd.read_parquet(f, columns=['icao', 't', 'lat', 'lon', 'ground'])
    p = p[p.icao.isin(set(keys.icao))]
    p = p.sort_values(['icao', 't'])
    out = []
    g = dict(tuple(p.groupby('icao')))
    for r in keys.itertuples(index=False):
        q = g.get(r.icao)
        if q is None:
            continue
        t = q.t.to_numpy()
        i = np.searchsorted(t, r.t_first - 0.5)
        if i >= len(t) or abs(t[i] - r.t_first) > 1.0:
            continue
        j = min(len(t), i + 5)
        w = (t[i:j] - t[i]) <= 30
        out.append((r.MVT_ID_mvt, q.lat.iat[i], q.lon.iat[i], float(np.median(q.lat.to_numpy()[i:j][w])),
                    float(np.median(q.lon.to_numpy()[i:j][w]))))
    return pd.DataFrame(out, columns=['MVT_ID_mvt', 'lat0', 'lon0', 'lat_med', 'lon_med'])


if __name__ == '__main__':
    ph = pd.read_parquet(ADSB / 'push_phases.parquet', columns=['MVT_ID_mvt', 'icao', 't_first', 'date'])
    ph = ph[ph.t_first.notna() & (ph.date.astype(str) >= SINCE)]   # final-file change: + date filter
    jobs = [(str(d), g[['MVT_ID_mvt', 'icao', 't_first']]) for d, g in ph.groupby('date')]
    # points files are per UTC day; a t_first just before midnight may sit in the previous day's file -> also try it
    frames = []
    with ProcessPoolExecutor(WORKERS) as ex:
        for k, r in enumerate(ex.map(one, jobs, chunksize=2)):
            if r is not None:
                frames.append(r)
            if k % 50 == 0:
                print(k, len(jobs), flush=True)
    d = pd.concat(frames, ignore_index=True).drop_duplicates('MVT_ID_mvt')
    miss = ph[~ph.MVT_ID_mvt.isin(d.MVT_ID_mvt)].copy()
    miss['date'] = (pd.to_datetime(miss.t_first, unit='s', utc=True)).dt.strftime('%Y-%m-%d')
    jobs2 = [(str(dd), g[['MVT_ID_mvt', 'icao', 't_first']]) for dd, g in miss.groupby('date')]
    with ProcessPoolExecutor(WORKERS) as ex:
        for r in ex.map(one, jobs2, chunksize=2):
            if r is not None:
                frames.append(r)
    d = pd.concat(frames, ignore_index=True).drop_duplicates('MVT_ID_mvt')
    d.to_parquet(OUT / os.environ.get('PRC_PARK_OUT', 'item04_park.parquet'), index=False)   # final-file change: output name
    print('found', len(d), 'of', len(ph), flush=True)
