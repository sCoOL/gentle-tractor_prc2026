"""Push-back phases of every ADS-B-matched departure: push start, push end, first forward move under own power.

  py -3.14 scripts/adsb_push_phases.py day 2026-07-20 --apt EDDM --show 5   # inspect one day
  py -3.14 scripts/adsb_push_phases.py run --workers 8                      # every day (cached)
  py -3.14 scripts/adsb_push_phases.py collect                               # -> push_phases.parquet

Question (2026-10-07): besides the first move off the stand (121's leave), find when the aircraft finished its
push-back (tug disconnected, aircraft at rest) and when it first moved FORWARD under its own power. adsb.lol puts the
aircraft's true heading in the trace's track field on the ground when the transponder reports one (it is present while
the aircraft stands still), so a push-back shows as motion ~180 deg away from the nose. Without a heading the push is
found geometrically: the motion bearing flips by > 120 deg between the first run and the next one (the cusp where the
tug lets go and the aircraft taxis off the way its nose points).

Inputs: <PRC_ADSB_POINTS>/points_<date>.parquet (118 extract), events_<date>.parquet (121, row k), v2 matches
runs/adsb_match_v2/matches.parquet (MVT_ID_mvt -> k_2). Label-free: no block time is read here.
Outputs: <PRC_ADSB_DIR>/push/<date>.parquet (cache), <PRC_ADSB_DIR>/push_phases.parquet (one row per matched departure).
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import AIRPORT_META, RUNS  # noqa: E402
import prc_config  # noqa: E402

ADSB = prc_config.ADSB_DIR       # writes here; 121 events are read here (PRC_ADSB_DIR)
POINTS = prc_config.ADSB_POINTS  # extracted points, read only (PRC_ADSB_POINTS)
CACHE = ADSB / 'push'
MATCHES = RUNS / 'adsb_match_v2' / 'matches.parquet'
MOVE_KT = 0.6            # moving when the reported ground speed is above this
STOP_S = 8.0             # a stationary stretch at least this long ends a motion run
BRG_M = 3.0              # bearing from a displacement of at least this many metres
REV_DEG, FWD_DEG = 120.0, 60.0
MOVE_M = 12.0            # first real move: 12 m from the first position (121's definition)
VERSION = 3


def wrap(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def xy(apt, lat, lon):
    m = AIRPORT_META[apt]
    return ((np.asarray(lon) - m['lon']) * np.cos(np.radians(m['lat'])) * 111_195.0,
            (np.asarray(lat) - m['lat']) * 111_195.0)


def bearings(t, x, y):
    """Motion bearing at each point from the nearest neighbours at least BRG_M apart (NaN when standing)."""
    n = len(t)
    b = np.full(n, np.nan)
    for i in range(n):
        j = i + 1
        while j < n and np.hypot(x[j] - x[i], y[j] - y[i]) < BRG_M and t[j] - t[i] <= 30:
            j += 1
        if j < n and t[j] - t[i] <= 30 and np.hypot(x[j] - x[i], y[j] - y[i]) >= BRG_M:
            b[i] = np.degrees(np.arctan2(x[j] - x[i], y[j] - y[i])) % 360
    return b


def circ_mean(deg):
    deg = np.asarray(deg, float)
    deg = deg[np.isfinite(deg)]
    if not len(deg):
        return np.nan
    r = np.radians(deg)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def runs_of(t, moving):
    """Motion runs as (first, last) index pairs; a stop of >= STOP_S (or a 60 s silence) separates two runs."""
    idx = np.where(moving)[0]
    if not len(idx):
        return []
    out, a, b = [], idx[0], idx[0]
    for i in idx[1:]:
        if (i > b + 1 and t[i] - t[b] >= STOP_S) or t[i] - t[b] > 60:
            out.append((a, b))
            a = i
        b = i
    out.append((a, b))
    return out


def phase(t, x, y, gs, trk, stand_xy=None):
    """Events of one departure ground phase (points from first sighting to the last ground point)."""
    n = len(t)
    r = dict(n_pts=n, push_type='nomove', hd_mode=False)
    if n < 3:
        return r
    spd = np.r_[np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 0.5), 0.0]
    g = np.where(np.isfinite(gs), gs, spd * 1.944)
    moving = g > MOVE_KT
    brg = bearings(t, x, y)
    hd = np.asarray(trk, float)
    rel = wrap(brg - hd)
    d = np.where(np.isfinite(rel) & moving, np.where(np.abs(rel) > REV_DEG, -1, np.where(np.abs(rel) < FWD_DEG, 1, 0)), 0)
    standing = ~moving
    r['hd_first'] = float(hd[0]) if np.isfinite(hd[0]) else np.nan
    r['gs_first'] = float(g[0])
    d0 = np.hypot(x - x[0], y - y[0])
    k = int(np.argmax(d0 > MOVE_M)) if (d0 > MOVE_M).any() else -1
    if k < 0:
        return r
    # heading while parked: track on standing points before the first real move (only heading-reporting transponders
    # give a track when the aircraft does not move)
    pre = standing[:k] & np.isfinite(hd[:k])
    r['hd_stand'] = circ_mean(hd[:k][pre]) if pre.sum() >= 2 else np.nan
    r['stand_pts'] = int(standing[:k].sum())
    rs = runs_of(t, moving)
    rs = [(a, b) for a, b in rs if b >= k - 1 or (a <= k <= b)] or rs
    rs = [(a, b) for a, b in rs if np.hypot(x[b] - x[a], y[b] - y[a]) >= 2.0 or b - a >= 2]
    if not rs:
        return r
    info = []
    for a, b in rs:
        dd = d[a:b + 1]
        nz = dd[dd != 0]
        seg = np.hypot(np.diff(x[a:b + 1]), np.diff(y[a:b + 1])).sum()
        # start / end bearing of the run from its first / last 15 m of path
        cum = np.r_[0, np.cumsum(np.hypot(np.diff(x[a:b + 1]), np.diff(y[a:b + 1])))]
        s_end = a + int(np.searchsorted(cum, min(15.0, cum[-1])))
        e_start = a + int(np.searchsorted(cum, max(cum[-1] - 15.0, 0)))
        b0 = (np.degrees(np.arctan2(x[s_end] - x[a], y[s_end] - y[a])) % 360) if s_end > a else np.nan
        b1 = (np.degrees(np.arctan2(x[b] - x[e_start], y[b] - y[e_start])) % 360) if b > e_start else np.nan
        stop = b + 1 if b + 1 < n else b
        info.append(dict(a=a, b=b, t0=t[a], t1=t[b], tstop=t[stop], m=float(seg), b0=b0, b1=b1,
                         nrev=int((nz == -1).sum()), nfwd=int((nz == 1).sum()),
                         hd0=hd[a], hd1=hd[b]))
    r['hd_mode'] = bool(np.isfinite(r.get('hd_stand', np.nan)) or sum(i['nrev'] + i['nfwd'] for i in info[:2]) >= 3)
    first = info[0]
    r['t_move'] = float(first['t0'])
    r['u_move'] = float(t[first['a']] - t[first['a'] - 1]) if first['a'] > 0 else np.nan
    r['n_runs'] = len(info)

    def is_rev(i):
        tot = i['nrev'] + i['nfwd']
        return tot >= 2 and i['nrev'] / tot >= 0.6

    def is_fwd(i):
        tot = i['nrev'] + i['nfwd']
        return tot >= 2 and i['nfwd'] / tot >= 0.6

    def cusp_end():
        # the motion bearing flips between two consecutive runs early in the phase (the tug lets go and the aircraft
        # taxis off the way its nose points); pauses inside a push keep the bearing, so they do not count
        run_m = 0.0
        for j in range(min(len(info) - 1, 3)):
            run_m += info[j]['m']
            if run_m > 300:
                return None
            if np.isfinite(info[j]['b1']) and np.isfinite(info[j + 1]['b0']) \
                    and abs(wrap(info[j + 1]['b0'] - info[j]['b1'])) > REV_DEG:
                return j
        return None

    def reverse_runs():
        j, out = 0, []
        while j < len(info) and (is_rev(info[j]) or (info[j]['nrev'] + info[j]['nfwd'] < 2 and info[j]['m'] < 30)):
            out.append(info[j])
            j += 1
        return out, next((i for i in info[j:] if not is_rev(i)), None)

    # first motion relative to the PARKED nose (heading seen while standing): robust to transponders that report the
    # ground track instead of the heading, whose track simply follows the motion during a push
    hs = r.get('hd_stand', np.nan)
    rel0 = wrap(first['b0'] - hs) if np.isfinite(hs) and np.isfinite(first['b0']) else np.nan
    r['rel0'] = float(rel0) if np.isfinite(rel0) else np.nan
    r['b0_first'] = float(first['b0']) if np.isfinite(first['b0']) else np.nan
    push, nxt = [], None
    if np.isfinite(rel0) and abs(rel0) > REV_DEG:                # backwards out of the stand: a push-back
        if is_rev(first):                                         # heading-type transponder: follow the reverse runs
            push, nxt = reverse_runs()
            r['push_type'] = 'push_heading'
        else:                                                     # track-type transponder: the push ends at the cusp
            j = cusp_end()
            push, nxt = (info[:j + 1], info[j + 1]) if j is not None else ([first], info[1] if len(info) > 1 else None)
            r['push_type'] = 'push_stand'
    elif np.isfinite(rel0) and abs(rel0) < FWD_DEG:              # left the stand nose first: no push-back
        r['push_type'], nxt = 'forward_stand', first
    elif is_rev(first):                                           # first seen already moving backwards
        push, nxt = reverse_runs()
        r['push_type'] = 'push_heading'
    else:
        j = cusp_end()
        if j is not None:
            push, nxt, r['push_type'] = info[:j + 1], info[j + 1], 'cusp_push'
        elif is_fwd(first):
            r['push_type'], nxt = 'forward_seen', first           # first motion seen goes nose first (push unseen?)
        else:
            r['push_type'] = 'nocusp'
    if push:
        p0, p1 = push[0], push[-1]
        r['t_push_start'] = float(p0['t0'])
        r['t_push_end'] = float(p1['tstop'])
        r['t_push_last_move'] = float(p1['t1'])
        r['push_m'] = float(sum(i['m'] for i in push))
        r['push_s'] = float(p1['tstop'] - p0['t0'])
        r['push_disp_m'] = float(np.hypot(x[p1['b']] - x[p0['a']], y[p1['b']] - y[p0['a']]))
        h0 = r.get('hd_stand') if np.isfinite(r.get('hd_stand', np.nan)) else p0['hd0']
        r['push_turn'] = float(abs(wrap(p1['hd1'] - h0))) if np.isfinite(h0) and np.isfinite(p1['hd1']) else np.nan
        r['push_parts'] = len(push)
        r['u_push_end'] = float(p1['tstop'] - p1['t1'])
        r['gap_push'] = float(np.diff(t[p0['a']:p1['b'] + 1]).max()) if p1['b'] > p0['a'] else np.nan
    if nxt is not None:
        r['t_fwd'] = float(nxt['t0'])
        r['u_fwd'] = float(t[nxt['a']] - t[nxt['a'] - 1]) if nxt['a'] > 0 else np.nan
        r['hd_fwd'] = float(nxt['hd0']) if np.isfinite(nxt['hd0']) else np.nan
        r['brg_fwd'] = float(nxt['b0']) if np.isfinite(nxt['b0']) else np.nan
        if push:
            r['hold_s'] = float(nxt['t0'] - r['t_push_end'])
    # how the aircraft was first seen: standing / being pushed / taxiing forward
    if standing[0]:
        r['first_state'] = 'standing'
    elif d[0] == -1 or (len(info) and info[0]['a'] == 0 and r['push_type'] in ('push_heading', 'push_stand', 'cusp_push')):
        r['first_state'] = 'pushing'
    elif d[0] == 1:
        r['first_state'] = 'forward'
    else:
        r['first_state'] = 'moving'
    # largest silence between first sighting and the forward start (coverage)
    tend = r.get('t_fwd', t[-1])
    m = t <= tend + 1
    r['max_gap_s'] = float(np.diff(t[m]).max()) if m.sum() >= 2 else np.nan
    return r


def matches_of(date):
    m = pd.read_parquet(MATCHES, columns=['MVT_ID_mvt', 'k_2', 'date'], filters=[('date', '=', date)])
    return m.dropna(subset=['k_2']).astype({'k_2': int})


def day(date, apt=None, show=0):
    f = CACHE / f'{date}.parquet'
    if f.exists() and apt is None and not show:
        return date, -2
    try:
        ev = pd.read_parquet(ADSB / 'events' / f'events_{date}.parquet')
        ev['k'] = np.arange(len(ev))
        m = matches_of(date).merge(ev, left_on='k_2', right_on='k')
        if apt:
            m = m[m.APT == apt]
        if m.empty:
            out = pd.DataFrame()
        else:
            p = pd.read_parquet((ADSB if (ADSB / f'points_{date}.parquet').exists() else POINTS) / f'points_{date}.parquet', columns=['icao', 't', 'lat', 'lon', 'ground', 'gs', 'track', 'APT'])
            p = p[p.icao.isin(set(m.icao)) & p.ground]
            p = p.sort_values(['icao', 'APT', 't'], kind='stable')
            grp = p.groupby(['icao', 'APT'], sort=False).indices
            T, LAT, LON, GS, TRK = (p[c].to_numpy(float) for c in ('t', 'lat', 'lon', 'gs', 'track'))
            rows = []
            for e in m.itertuples(index=False):
                idx = grp.get((e.icao, e.APT))
                if idx is None:
                    continue
                tt = T[idx]
                sel = idx[(tt >= e.t_first - 0.5) & (tt <= e.t_tko + 0.5)]
                if len(sel) < 2:
                    continue
                t = T[sel]
                keep = np.r_[True, np.diff(t) > 0]
                sel, t = sel[keep], t[keep]
                x, y = xy(e.APT, LAT[sel], LON[sel])
                r = phase(t, x, y, GS[sel], TRK[sel])
                r.update(MVT_ID_mvt=e.MVT_ID_mvt, icao=e.icao, APT=e.APT, t_first=float(e.t_first),
                         t_tko=float(e.t_tko), leave_lo=e.leave_lo, leave_hi=e.leave_hi, at_stand=bool(e.at_stand),
                         censored=bool(e.censored), d_stand_first=e.d_stand_first, after_landing=bool(e.after_landing))
                rows.append(r)
                if show and len(rows) <= show:
                    print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})
            out = pd.DataFrame(rows)
        if apt is None and not show:
            CACHE.mkdir(exist_ok=True)
            out.assign(version=VERSION).to_parquet(f, index=False)
        return date, len(out)
    except Exception as ex:                                   # one bad day must not stop the pass
        print(f'  {date}: {type(ex).__name__}: {ex}', flush=True)
        return date, -1


def all_days():
    return sorted({p.name[7:17] for d in (ADSB, POINTS) for p in d.glob('points_*.parquet')})   # final-file change


def run(workers):
    # final-file change: only days that have an events file AND rows in matches.parquet. A day absent from
    # matches.parquet (255 compare not yet run on it, or no departures at all) would be cached EMPTY by day().
    md = set(pd.read_parquet(MATCHES, columns=['date']).date.astype(str).unique())
    d = [x for x in all_days() if not (CACHE / f'{x}.parquet').exists()
         and x in md and (ADSB / 'events' / f'events_{x}.parquet').exists()]
    print(f'push phases for {len(d)} days', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for k, (date, n) in enumerate(pool.map(day, d), 1):
            if k % 20 == 0 or n < 0:
                print(f'  {k}/{len(d)} {date}: {n} rows [{time.time() - t0:.0f}s]', flush=True)
    print(f'done [{time.time() - t0:.0f}s]', flush=True)


def collect():
    fs = sorted(CACHE.glob('*.parquet'))
    d = pd.concat([pd.read_parquet(f).assign(date=f.stem) for f in fs], ignore_index=True)
    d.to_parquet(ADSB / 'push_phases.parquet', index=False)
    print(f'push_phases.parquet: {len(d):,} rows from {len(fs)} days')
    print(d.groupby('APT').push_type.value_counts(normalize=True).unstack().round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['day', 'run', 'collect'])
    ap.add_argument('date', nargs='?')
    ap.add_argument('--apt')
    ap.add_argument('--show', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    if a.cmd == 'day':
        t0 = time.time()
        print(day(a.date, a.apt, a.show or 5), f'{time.time() - t0:.0f}s')
    elif a.cmd == 'run':
        run(a.workers)
    else:
        collect()
