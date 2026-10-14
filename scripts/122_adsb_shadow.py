"""Where ADS-B loses sight of the ground, and what the track shows of the taxi we DID see.

  py -3.14 scripts/122_adsb_shadow.py geometry --workers 6    # per-departure track geometry -> adsb_geom.parquet
  py -3.14 scripts/122_adsb_shadow.py shadow                  # the missing head, cross-fitted -> adsb_shadow.parquet
  py -3.14 scripts/122_adsb_shadow.py check                   # how well the head is predicted, held-out

Two things come out of one pass over the point files that are already on G: (no download, and `adsb_features.parquet`
is not touched, so the deployed members keep their caches).

**Geometry** (`adsb_gm_*`): where the aircraft was first seen, how far it actually taxied, how long it stood still and
how fast it was moving as it reached the runway. It describes the part of the taxi ADS-B DID see, so it reaches rows
whose own push-back was missed.

**The shadow** (`adsb_sh_*`): 383k matched departures were first seen already moving, so their off-block is only
bounded. The missing head is `t_first - off-block`, and 2025 labels give it to us directly, so it can be learned from
WHERE the aircraft appeared (an airport's ADS-B dead zone is fixed geography), which stand it left, which runway it
was heading for, the hour and the traffic. That is the idea (2026-09-20) and the route to EGLL, LFPG and LIRF,
which gained ~0 in A38 and hold 32 % of the matched SSE.

LEAKAGE: the head is fitted from off-block times, so on 2025 rows it is CROSS-FITTED by strict block (a row's own
block never trains its head) exactly as the G8 target encodings are; 2026 rows get the model fitted on all of 2025.
`tests/test_adsb_shadow.py` checks that no block's own rows train its head.
"""
import argparse
import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import common  # noqa: E402

_spec = importlib.util.spec_from_file_location('p121', ROOT / 'scripts' / '121_adsb_events.py')
P121 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P121)
OUT = P121.OUT

STOP_M, STOP_S = 8.0, 30.0           # a "stop" while taxiing: under 8 m of movement for at least 30 s
GEOM_COLUMNS = ['adsb_gm_x0', 'adsb_gm_y0', 'adsb_gm_r0', 'adsb_gm_bear0', 'adsb_gm_x1', 'adsb_gm_y1',
                'adsb_gm_dist', 'adsb_gm_disp', 'adsb_gm_wander', 'adsb_gm_span', 'adsb_gm_stops',
                'adsb_gm_stopped', 'adsb_gm_hold', 'adsb_gm_v_mean', 'adsb_gm_v_max', 'adsb_gm_v_last',
                'adsb_gm_npts']
SHADOW_COLUMNS = ['adsb_sh_head', 'adsb_sh_taxi', 'adsb_sh_lo', 'adsb_sh_hi', 'adsb_sh_sd', 'adsb_sh_n']


def _geom_of(t, x, y, gs0):
    """Geometry of one departure's observed ground phase (arrays are the phase's points, in time order)."""
    n = len(t)
    if n == 0:
        return None
    dx, dy = np.diff(x), np.diff(y)
    step = np.hypot(dx, dy)
    dist = float(step.sum())
    disp = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    dt = np.diff(t)
    v = np.where(dt > 0, step / np.maximum(dt, 1e-6), 0.0)
    stops = stopped = hold = 0.0
    i = 0
    while i < n - 1:                                   # a run of points that stays inside STOP_M of where it began
        j = i
        while j + 1 < n and np.hypot(x[j + 1] - x[i], y[j + 1] - y[i]) <= STOP_M:
            j += 1
        if j > i and t[j] - t[i] >= STOP_S:
            stops += 1
            stopped += float(t[j] - t[i])
            if i > 0:                                  # a stop after the aircraft had already started moving
                hold += float(t[j] - t[i])
        i = max(j, i + 1)
    return dict(adsb_gm_x0=float(x[0]), adsb_gm_y0=float(y[0]), adsb_gm_r0=float(np.hypot(x[0], y[0])),
                adsb_gm_bear0=float(np.degrees(np.arctan2(y[0], x[0]))),
                adsb_gm_x1=float(x[-1]), adsb_gm_y1=float(y[-1]),
                adsb_gm_dist=dist, adsb_gm_disp=disp, adsb_gm_wander=float(dist / disp) if disp > 1 else np.nan,
                adsb_gm_span=float(t[-1] - t[0]), adsb_gm_stops=stops, adsb_gm_stopped=stopped, adsb_gm_hold=hold,
                adsb_gm_v_mean=float(np.mean(v)) if len(v) else np.nan,
                adsb_gm_v_max=float(np.max(v)) if len(v) else np.nan,
                adsb_gm_v_last=float(v[-1]) if len(v) else float(gs0),
                adsb_gm_npts=float(n))


def day_geometry(date):
    """One day's per-departure geometry, keyed by MVT_ID_mvt through the same match `121` uses."""
    f = OUT / 'geom' / f'geom_{date}.parquet'
    if f.exists():
        return date, len(pd.read_parquet(f, columns=['MVT_ID_mvt']))
    try:
        ph = P121.phases(date)[0]
    except Exception as e:
        print(f'  {date}: {type(e).__name__} {e}', flush=True)
        return date, -1
    # one row per ground phase, in `121`'s order and NEVER skipped: `matched.k` indexes into this frame, so a
    # dropped row would silently attach one aircraft's geometry to another flight
    blank = {c: np.nan for c in GEOM_COLUMNS}
    rows = []
    for icao, apt, cs, t_tko, j, i, after_landing, t, x, y, gs0 in ph:
        g = _geom_of(np.asarray(t, float), np.asarray(x, float), np.asarray(y, float), gs0)
        rows.append(g if g is not None else dict(blank))
    if not rows:
        f.parent.mkdir(exist_ok=True)
        pd.DataFrame(columns=['MVT_ID_mvt'] + GEOM_COLUMNS).to_parquet(f, index=False)
        return date, 0
    geom = pd.DataFrame(rows)[GEOM_COLUMNS]
    ev, matched = P121.match_day(date)
    if len(ev) != len(geom):
        raise SystemExit(f'{date}: {len(geom)} phases but {len(ev)} events - the k index would be wrong')
    if len(matched) == 0:
        out = pd.DataFrame(columns=['MVT_ID_mvt'] + GEOM_COLUMNS)
    else:
        k = matched.k.to_numpy()
        out = geom.iloc[k].reset_index(drop=True)
        out.insert(0, 'MVT_ID_mvt', matched.MVT_ID_mvt.to_numpy())
    f.parent.mkdir(exist_ok=True)
    out.to_parquet(f, index=False)
    return date, len(out)


def build_geometry(workers):
    d = [x for x in P121.days() if not (OUT / 'geom' / f'geom_{x}.parquet').exists()]
    print(f'geometry for {len(d)} days ({len(P121.days()) - len(d)} cached)', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for k, (date, n) in enumerate(pool.map(day_geometry, d), 1):
            if k % 50 == 0 or n < 0:
                print(f'  {k}/{len(d)} {date}: {n} rows [{time.time() - t0:.0f}s]', flush=True)
    parts = [pd.read_parquet(OUT / 'geom' / f'geom_{x}.parquet') for x in P121.days()
             if (OUT / 'geom' / f'geom_{x}.parquet').exists()]
    g = pd.concat(parts, ignore_index=True).drop_duplicates('MVT_ID_mvt')
    g.to_parquet(OUT / 'adsb_geom.parquet', index=False)
    print(f'adsb_geom.parquet: {len(g):,} rows [{time.time() - t0:.0f}s]')


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['geometry', 'shadow', 'check'])
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args()
    OUT = Path(a.out)
    P121.OUT = OUT
    P121.P118.OUT = OUT
    if a.cmd == 'geometry':
        build_geometry(a.workers)
    else:
        raise SystemExit(f'{a.cmd}: not implemented yet (geometry first)')


if __name__ == '__main__':
    main()
