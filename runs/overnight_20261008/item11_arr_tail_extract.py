"""Item 11 step 1 (A9, joint coverage shadow): for every arrival record (both years; arrival in-block is given in both),
the aircraft's last ADS-B ground point after landing -> tail = recorded in-block - last seen, and that point's position.
icao24 from the OPDI link (<PRC_OPDI_DIR>/opdi_movements.parquet, ARR rows). Label-free for departures.
-> item11_arr_tail.parquet
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import common  # noqa: E402
import prc_config  # noqa: E402

ADSB = prc_config.ADSB_POINTS   # extracted points_<date> (PRC_ADSB_POINTS, read only)
import os  # noqa: E402
RANK_ONLY = bool(os.environ.get('PRC_RANK_ONLY'))   # final-file change: only the ranking file's arrivals
WORKERS = int(os.environ.get('PRC_WORKERS', '10'))   # final-file change: was a fixed 10
OUT = common.RUNS / 'overnight_20261008'
GAP_S = 300.0


def one(args):
    date, keys = args
    f = ADSB / f'points_{date}.parquet'
    if not f.exists():
        return None
    p = pd.read_parquet(f, columns=['icao', 't', 'lat', 'lon', 'ground', 'APT'])
    p = p[p.icao.isin(set(keys.icao)) & p.ground]
    g = {k: v.sort_values('t') for k, v in p.groupby(['icao', 'APT'])}
    out = []
    for r in keys.itertuples(index=False):
        q = g.get((r.icao, r.APT))
        if q is None:
            continue
        t = q.t.to_numpy()
        i = np.searchsorted(t, r.t_land - 120)
        if i >= len(t) or t[i] > r.t_land + 600:
            continue
        j = i
        while j + 1 < len(t) and t[j + 1] - t[j] < GAP_S and t[j + 1] <= r.t_blk + 600:
            j += 1
        out.append((r.MVT_ID_mvt, t[i], t[j], q.lat.iat[j], q.lon.iat[j], j - i + 1))
    return pd.DataFrame(out, columns=['MVT_ID_mvt', 't_gfirst', 't_glast', 'lat_last', 'lon_last', 'n_gpts'])


if __name__ == '__main__':
    raw = ['MVT_ID_mvt', 'PHASE_mvt', 'ADES_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt']
    m = pd.concat([pd.read_parquet(f, columns=raw).assign(split='train') for f in sorted(common.DATA.glob('training_2025-*.parquet'))]
                  + [pd.read_parquet(common.DATA / 'ranking.parquet', columns=raw).assign(split='rank')], ignore_index=True)
    a = m[m.PHASE_mvt == 'ARR'].copy()
    if RANK_ONLY:
        a = a[a.split == 'rank']
    om = pd.read_parquet(prc_config.OPDI_DIR / 'opdi_movements.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt', 'icao24'])
    om = om[(om.PHASE_mvt == 'ARR') & om.icao24.notna()][['MVT_ID_mvt', 'icao24']].drop_duplicates('MVT_ID_mvt')
    a = a.merge(om, on='MVT_ID_mvt', how='inner')
    a['icao'] = a.icao24.astype(str).str.lower().str.strip()
    a['APT'] = a.ADES_mvt
    ep = lambda s: (pd.to_datetime(s, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')) / pd.Timedelta(seconds=1)
    a['t_land'], a['t_blk'] = ep(a.MVT_TIME_UTC_mvt), ep(a.BLOCK_TIME_UTC_mvt)
    a = a[a.t_land.notna() & a.t_blk.notna()]
    a['date'] = pd.to_datetime(a.t_land, unit='s', utc=True).dt.strftime('%Y-%m-%d')
    print('arrivals with icao24', len(a), a.split.value_counts().to_dict(), flush=True)
    jobs = [(d, g[['MVT_ID_mvt', 'icao', 'APT', 't_land', 't_blk']]) for d, g in a.groupby('date')]
    frames = []
    with ProcessPoolExecutor(WORKERS) as ex:
        for k, r in enumerate(ex.map(one, jobs, chunksize=2)):
            if r is not None:
                frames.append(r)
            if k % 100 == 0:
                print(k, len(jobs), flush=True)
    d = pd.concat(frames, ignore_index=True)
    d = d.merge(a[['MVT_ID_mvt', 'APT', 'STAND_mvt', 'split', 't_land', 't_blk']], on='MVT_ID_mvt')
    d['tail'] = d.t_blk - d.t_glast
    d.to_parquet(OUT / os.environ.get('PRC_TAIL_OUT', 'item11_arr_tail.parquet'), index=False)   # final-file change: output name
    print('found', len(d), d.split.value_counts().to_dict(), flush=True)
    print(d.groupby(['APT', 'split'])['tail'].median().unstack().round(0))
