"""ADS-B off-block events for every extracted day -> one feature table for the challenge departures.

  py -3.14 scripts/121_adsb_events.py stands   --workers 6              # stand map from where arrivals park
  py -3.14 scripts/121_adsb_events.py events   --workers 6              # per-day departure events (cached)
  py -3.14 scripts/121_adsb_events.py features                          # join to 2025 + 2026 departures
  py -3.14 scripts/121_adsb_events.py neighbours                        # what the OTHER aircraft did
  py -3.14 scripts/121_adsb_events.py check                             # 2025 accuracy and coverage tables

Off-block = the first real move away from the stand (validated: 84 % within 60 s of the
recorded off-block vs 56 % for the first sighting and 25 % for NM's AOBT_3). Stands are learned from where ARRIVALS come
to a stop (departure starts also contain post-push hold spots), merged on a 20 m grid over all extracted days and kept
when at least MIN_VISITS aircraft parked there; X-Plane stand coordinates are added as extra points.

Inputs  <out>/points_<date>.parquet (118 extract).
Outputs <out>/stands.parquet, <out>/events/events_<date>.parquet, <out>/adsb_features.parquet (MVT_ID_mvt + adsb_*).
Nothing here reads an official score; 2025 columns are scored against the real off-blocks by `check`.
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
from common import AIRPORT_META, DATA                                    # noqa: E402

_spec = importlib.util.spec_from_file_location('p118', ROOT / 'scripts' / '118_adsb_probe.py')
P118 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P118)

OUT = P118.OUT
STILL_M, STILL_S, MOVE_M, AT_STAND_M, CLEAR_M, GAP_S = 8, 30, 12, 40, 80, 900
PARK_S = 900.0                       # an arrival stop only counts as a stand if it lasted this long (or ended there)
GRID_M, MIN_VISITS = 20, 3
TKO_TOL, POS_TOL = 180, 45
ARR_STILL_M, ARR_PARK_S = 20, 300    # arrivals: 'stopped' radius (jitter near terminals) and how long a stop lasts
QUALITY_COLUMNS = ['adsb_q_within60', 'adsb_q_med', 'adsb_q_mad', 'adsb_q_cov', 'adsb_q_n']


def days():
    # final-file change (was OUT.glob('points_*.parquet')): P118.points_days() = complete points files, OUT + POINTS.
    # final-file change: no day before 2025-01-01 (points_2024-12-31 has no departure source: departures() raises
    # IndexError in 121 quality/features, 122, 124, 232; no events are built for it), and when PRC_ADSB_DAYS names
    # a file, only the days listed in it (a batch frozen at its start, so a day finishing extraction mid-run cannot
    # reach a later step without its events file).
    import os
    d = [x for x in P118.points_days() if x >= '2025-01-01']
    f = os.environ.get('PRC_ADSB_DAYS')
    if f:
        keep = set(Path(f).read_text().split())
        d = [x for x in d if x in keep]
    return d


def xy(apt, lat, lon):
    m = AIRPORT_META[apt]
    return ((np.asarray(lon) - m['lon']) * np.cos(np.radians(m['lat'])) * 111_195.0,
            (np.asarray(lat) - m['lat']) * 111_195.0)


def latlon(apt, x, y):
    m = AIRPORT_META[apt]
    return m['lat'] + y / 111_195.0, m['lon'] + x / (111_195.0 * np.cos(np.radians(m['lat'])))


def phases(date):
    """One day's departure ground phases, arrival parking positions, and arrival in-block events."""
    pts = pd.read_parquet(P118.points_path(date)).sort_values('t')   # final-file change (was OUT / points_<date>)
    out, parked, arrivals = [], [], []
    for (icao, apt), g in pts.groupby(['icao', 'APT'], sort=False):
        t, gr = g.t.to_numpy(), g.ground.to_numpy(bool)
        x, y = xy(apt, g.lat.to_numpy(), g.lon.to_numpy())
        gs, cs = g.gs.to_numpy(float), g.cs.to_numpy(object)
        for i in np.where(gr[:-1] & ~gr[1:])[0] + 1:                     # departures
            if t[i] - t[i - 1] > 120:
                continue
            j = i - 1
            while j > 0 and gr[j - 1] and t[j] - t[j - 1] < GAP_S:
                j -= 1
            names = [c for c in cs[j:min(i + 30, len(cs))] if c]
            out.append((icao, apt, max(set(names), key=names.count) if names else None, t[i], j, i,
                        bool(j > 0 and not gr[j - 1] and t[j] - t[j - 1] < GAP_S), t[j:i], x[j:i], y[j:i], gs[j]))
        for i in np.where(~gr[:-1] & gr[1:])[0] + 1:                     # arrivals: where they come to a stop
            e = i
            while e + 1 < len(t) and gr[e + 1] and t[e + 1] - t[e] < GAP_S:
                e += 1
            # in-block = the FIRST stop of at least ARR_PARK_S after landing. The last stop is wrong: at EGLL the
            # transponder often stays on through the turnaround, so it lands on a tow or the next push-back.
            a = b = None
            k = i
            while k <= e:
                k2 = k
                while k2 + 1 <= e and np.hypot(x[k2 + 1] - x[k], y[k2 + 1] - y[k]) <= ARR_STILL_M:
                    k2 += 1
                if t[k2] - t[k] >= ARR_PARK_S or (k2 == e and t[e] - t[k] >= STILL_S):
                    a, b = k, k2
                    break
                k = k2 + 1
            if a is None:
                continue
            dwell = float(t[b] - t[a])
            ends = bool(b == len(t) - 1 or t[b + 1] - t[b] >= GAP_S)     # transponder went off there = parked
            if dwell >= STILL_S or ends:
                parked.append((apt, float(np.median(x[a:b + 1])), float(np.median(y[a:b + 1])), dwell, ends))
            names = [c for c in cs[max(0, i - 30):b + 1] if c]
            arrivals.append((icao, apt, max(set(names), key=names.count) if names else None, float(t[i - 1]),
                             float(t[a]), dwell))                   # landing (last airborne point) and coming to rest
    return (out, pd.DataFrame(parked, columns=['APT', 'x', 'y', 'dwell', 'ends']),
            pd.DataFrame(arrivals, columns=['icao', 'APT', 'cs', 't_land', 't_inblock', 'dwell']))


def _parked(date):
    try:
        return phases(date)[1]
    except Exception as e:                                               # a truncated day must not stop the pass
        print(f'  {date}: {type(e).__name__} {e}', flush=True)
        return pd.DataFrame(columns=['APT', 'x', 'y', 'dwell', 'ends'])


def build_stands(workers):
    d = days()
    print(f'stand map from {len(d)} days', flush=True)
    t0, frames = time.time(), []
    with ProcessPoolExecutor(workers) as pool:
        for k, f in enumerate(pool.map(_parked, d), 1):
            frames.append(f)
            if k % 25 == 0:
                print(f'  {k}/{len(d)} days, {sum(map(len, frames)):,} parkings [{time.time() - t0:.0f}s]', flush=True)
    p = pd.concat(frames, ignore_index=True)
    # a real stand: the aircraft stayed put for PARK_S or its transponder went off there. Short stops are queues on a
    # taxiway waiting for a gate, and taking them for stands made EGLL/LFPG fire ~13 min late (§0x).
    before = len(p)
    p = p[(p.dwell >= PARK_S) | p.ends]
    print(f'  parkings kept {len(p):,} of {before:,} (dwell >= {PARK_S:.0f}s or transponder off there)', flush=True)
    p['gx'], p['gy'] = np.round(p.x / GRID_M), np.round(p.y / GRID_M)
    s = p.groupby(['APT', 'gx', 'gy']).agg(x=('x', 'median'), y=('y', 'median'), visits=('x', 'size')).reset_index()
    s = s[s.visits >= MIN_VISITS].drop(columns=['gx', 'gy'])
    s['lat'], s['lon'] = zip(*[latlon(a, x, y) for a, x, y in zip(s.APT, s.x, s.y)])
    s['source'] = 'adsb'
    xp = pd.read_parquet(DATA / 'ext_stand_coords.parquet')
    xp = xp[xp.lat.notna() & xp.lon.notna() & xp.APT.isin(AIRPORT_META)].copy()
    xp['x'], xp['y'] = zip(*[xy(a, la, lo) for a, la, lo in zip(xp.APT, xp.lat, xp.lon)])
    xp = xp.assign(visits=0, source='xplane')[['APT', 'x', 'y', 'lat', 'lon', 'visits', 'source']]
    out = pd.concat([s[xp.columns], xp], ignore_index=True)
    out.to_parquet(OUT / 'stands.parquet', index=False)
    print(out.groupby(['APT', 'source']).size().unstack(fill_value=0).to_string())
    print(f'stands.parquet: {len(out):,} points [{time.time() - t0:.0f}s]', flush=True)


_STANDS = {}


def stands_of(apt):
    if not _STANDS:
        s = pd.read_parquet(OUT / 'stands.parquet')
        for a, g in s.groupby('APT'):
            _STANDS[a] = np.c_[g.x.to_numpy(), g.y.to_numpy()]
    return _STANDS.get(apt, np.zeros((0, 2)))


def day_events(date):
    """One row per departure ground phase: when it was first seen and when it left its stand."""
    f = OUT / 'events' / f'events_{date}.parquet'
    if f.exists():
        return date, len(pd.read_parquet(f, columns=['icao']))
    try:
        ph = phases(date)[0]
    except Exception as e:
        print(f'  {date}: {type(e).__name__} {e}', flush=True)
        return date, -1
    rows = []
    for icao, apt, cs, t_tko, j, i, after_landing, t, x, y, gs0 in ph:
        S = stands_of(apt)
        dmin = (np.min(np.hypot(S[:, 0][None, :] - x[:, None], S[:, 1][None, :] - y[:, None]), axis=1)
                if len(S) else np.full(len(x), np.nan))
        d0 = np.hypot(x - x[0], y - y[0])
        k = int(np.argmax(d0 > MOVE_M)) if (d0 > MOVE_M).any() else 0
        c = int(np.argmax(dmin > CLEAR_M)) if (dmin > CLEAR_M).any() else -1
        rows.append(dict(icao=icao, APT=apt, cs=cs, t_tko=float(t_tko), t_first=float(t[0]),
                         leave_lo=float(t[k - 1]) if k else np.nan, leave_hi=float(t[k]) if k else np.nan,
                         clear=float(t[c]) if c >= 0 else np.nan, d_stand_first=float(dmin[0]),
                         at_stand=bool(dmin[0] <= AT_STAND_M and not after_landing),
                         censored=bool(dmin[0] > CLEAR_M), after_landing=after_landing,
                         gs0=float(gs0) if gs0 == gs0 else np.nan, n_ground=len(t)))
    d = pd.DataFrame(rows)
    f.parent.mkdir(exist_ok=True)
    d.to_parquet(f, index=False)
    return date, len(d)


def build_events(workers):
    d = [x for x in days() if not (OUT / 'events' / f'events_{x}.parquet').exists()]
    print(f'events for {len(d)} days ({len(days()) - len(d)} cached)', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for k, (date, n) in enumerate(pool.map(day_events, d), 1):
            if k % 25 == 0 or n < 0:
                print(f'  {k}/{len(d)} {date}: {n} events [{time.time() - t0:.0f}s]', flush=True)
    print(f'done [{time.time() - t0:.0f}s]', flush=True)


def day_quality(date):
    """How well ADS-B saw the ARRIVALS of one day, scored against their recorded in-block times.

    Arrival in-block is given in both years (only departure block times are masked), so this measures the local ADS-B
    regime of an airport-month the same way in 2025 and in 2026 — the feature that lets a model trust ADS-B where it is
    currently good instead of where it was good last year.
    """
    try:
        arr = phases(date)[2]
    except Exception as e:
        print(f'  {date}: {type(e).__name__} {e}', flush=True)
        return pd.DataFrame()
    if arr.empty:
        return pd.DataFrame()
    arr = arr.astype({'APT': 'string', 'cs': 'string'}).assign(k=np.arange(len(arr)))
    src = (DATA / 'ranking.parquet' if date >= '2026'
           else sorted(DATA.glob(f'training_{date[:4]}-{date[5:7]}-01_*.parquet'))[0])
    if src not in _MONTH:
        _MONTH.clear()
        _MONTH[src] = pd.read_parquet(src)
    d = _MONTH[src]
    d = d[(d.PHASE_mvt == 'ARR') & (d.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d') == date)].copy()
    d['APT'] = d.ADES_mvt.astype('string')
    d['t_land'], d['t_blk'] = P118.epoch(d.MVT_TIME_UTC_mvt), P118.epoch(d.BLOCK_TIME_UTC_mvt)
    d['cs'] = d.CALLSIGN_flt.astype('string').str.strip()
    m = d.dropna(subset=['cs']).merge(arr, on=['APT', 'cs'], suffixes=('', '_a'))
    m = m[(m.t_land_a - m.t_land).abs() <= TKO_TOL].assign(gap=lambda z: (z.t_land_a - z.t_land).abs())
    m = m.sort_values('gap').drop_duplicates('MVT_ID_mvt').drop_duplicates('k')
    m['err'] = m.t_inblock - m.t_blk
    return pd.DataFrame({'APT': m.APT, 'month': date[:7], 'err': m.err,
                         'arrivals': len(d), 'seen': len(m)}).assign(date=date)


def _quality_cached(date):
    """final-file change: day_quality(date) cached in <OUT>/quality_days/<date>.parquet. day_quality returns a frame
    WITHOUT columns when phases() raised or the day has no arrival phase; such a frame is not cached."""
    f = OUT / 'quality_days' / f'{date}.parquet'
    if f.exists():
        return pd.read_parquet(f)
    q = day_quality(date)
    if len(q.columns):
        f.parent.mkdir(exist_ok=True)
        q.to_parquet(f, index=False)
    return q


def build_quality(workers):
    d = days()
    print(f'arrival quality from {len(d)} days', flush=True)
    t0, frames = time.time(), []
    # final-file change: was a serial loop over day_quality (13 s/day on the loaded machine = 1.5 h per pass). Now a
    # pool over the same day_quality, each day cached in <OUT>/quality_days (a day that raised is not cached), so a
    # later batch only computes its new days. The aggregation below is unchanged.
    with ProcessPoolExecutor(workers) as pool:
        for k, fr in enumerate(pool.map(_quality_cached, d, chunksize=8), 1):
            frames.append(fr)
            if k % 50 == 0:
                print(f'  {k}/{len(d)} [{time.time() - t0:.0f}s]', flush=True)
    a = pd.concat([f for f in frames if len(f)], ignore_index=True)
    per_day = a.groupby(['APT', 'month', 'date']).agg(arrivals=('arrivals', 'first'), seen=('seen', 'first')).reset_index()
    cov = per_day.groupby(['APT', 'month']).apply(lambda g: g.seen.sum() / max(g.arrivals.sum(), 1), include_groups=False)
    q = a.groupby(['APT', 'month']).agg(adsb_q_n=('err', 'size'), adsb_q_within60=('err', lambda e: (e.abs() <= 60).mean()),
                                        adsb_q_med=('err', 'median'), adsb_q_mad=('err', lambda e: e.abs().median()))
    q['adsb_q_cov'] = cov
    q = q.reset_index()
    q.to_parquet(OUT / 'adsb_quality.parquet', index=False)
    print(q.pivot_table(index='month', columns='APT', values='adsb_q_within60').round(2).to_string())
    print(f'adsb_quality.parquet: {len(q)} airport-months [{time.time() - t0:.0f}s]')


_MONTH = {}


def departures(date):
    """The challenge's departures of one day: 2025 training rows (labelled) or 2026 ranking rows."""
    src = (DATA / 'ranking.parquet' if date >= '2026'
           else sorted(DATA.glob(f'training_{date[:4]}-{date[5:7]}-01_*.parquet'))[0])
    if src not in _MONTH:                       # one read per month, not per day
        _MONTH.clear()
        _MONTH[src] = pd.read_parquet(src)
    dep = _MONTH[src]
    dep = dep[(dep.PHASE_mvt == 'DEP') & (dep.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d') == date)].copy()
    dep['APT'] = dep.ADEP_mvt.astype('string')
    dep['t_mvt'] = P118.epoch(dep.MVT_TIME_UTC_mvt)
    dep['t_blk'] = P118.epoch(dep.BLOCK_TIME_UTC_mvt) if dep.BLOCK_TIME_UTC_mvt.notna().any() else np.nan
    dep['t_a3'] = P118.epoch(dep.AOBT_3_flt)
    dep['cs'] = dep.CALLSIGN_flt.astype('string').str.strip()
    return dep[['MVT_ID_mvt', 'APT', 'cs', 't_mvt', 't_blk', 't_a3', 'TAXITIME_SEC_mvt']]


def match_day(date):
    """One day's events and the day's challenge departures matched to them (callsign + take-off, else take-off alone).

    Returns (events with a ``k`` index, matched departures carrying that ``k``). ``join_day`` projects the match to the
    per-row feature columns; ``build_neighbours`` needs the ``k`` itself, to take each row's own aircraft out of the
    surface context measured around it.
    """
    ev = pd.read_parquet(OUT / 'events' / f'events_{date}.parquet')
    if ev.empty:
        return ev, pd.DataFrame()
    ev = ev.astype({'APT': 'string', 'cs': 'string'}).assign(k=np.arange(len(ev)))
    dep = departures(date)
    c = dep.dropna(subset=['cs']).merge(ev, on=['APT', 'cs'])
    c = c[(c.t_tko - c.t_mvt).abs() <= TKO_TOL].assign(gap=lambda z: (z.t_tko - z.t_mvt).abs(), how=1)
    c = c.sort_values('gap').drop_duplicates('MVT_ID_mvt').drop_duplicates('k')
    rest = dep[~dep.MVT_ID_mvt.isin(c.MVT_ID_mvt)]
    p = rest.merge(ev[~ev.k.isin(c.k)].drop(columns='cs'), on='APT')
    p = p[(p.t_tko - p.t_mvt).abs() <= POS_TOL].assign(gap=lambda z: (z.t_tko - z.t_mvt).abs(), how=2)
    p = p.sort_values('gap').drop_duplicates('MVT_ID_mvt').drop_duplicates('k')
    return ev, pd.concat([c.drop(columns='cs'), p], ignore_index=True)


def join_day(date):
    ev, d = match_day(date)
    if len(ev) == 0:
        return pd.DataFrame()
    out = pd.DataFrame({
        'MVT_ID_mvt': d.MVT_ID_mvt, 'date': date, 'APT': d.APT, 'adsb_how': d.how,
        'adsb_at_stand': d.at_stand.astype('int8'), 'adsb_censored': d.censored.astype('int8'),
        'adsb_after_landing': d.after_landing.astype('int8'),
        'adsb_taxi_leave': d.t_mvt - d.leave_lo, 'adsb_taxi_leave_hi': d.t_mvt - d.leave_hi,
        'adsb_taxi_first': d.t_mvt - d.t_first, 'adsb_taxi_clear': d.t_mvt - d.clear,
        'adsb_wait': d.leave_lo - d.t_first, 'adsb_gap': d.leave_hi - d.leave_lo,
        'adsb_dstand': d.d_stand_first, 'adsb_gs0': d.gs0, 'adsb_nground': d.n_ground,
        'adsb_leave_minus_a3': d.leave_lo - d.t_a3, 'adsb_first_minus_a3': d.t_first - d.t_a3,
        'adsb_match_gap': d.gap, 'y': d.TAXITIME_SEC_mvt, 't_blk': d.t_blk, 't_mvt': d.t_mvt,
    })
    return out


def _join_day_cached(date):
    """final-file change: join_day(date) cached in <OUT>/features_days/<date>.parquet. A day's v1 match reads only that
    day's events and departures, so the cache stays valid while the ranking file and the events do not change."""
    f = OUT / 'features_days' / f'{date}.parquet'
    if f.exists():
        return pd.read_parquet(f)
    out = join_day(date)
    if len(out.columns):
        f.parent.mkdir(exist_ok=True)
        out.to_parquet(f, index=False)
    return out


def build_features(workers=1):
    d = [x for x in days() if (OUT / 'events' / f'events_{x}.parquet').exists()]
    t0, frames = time.time(), []
    # final-file change: was a serial loop over join_day; now a pool (order kept) over the per-day cache above
    with ProcessPoolExecutor(workers) as pool:
        for k, fr in enumerate(pool.map(_join_day_cached, d, chunksize=8), 1):
            frames.append(fr)
            if k % 25 == 0:
                print(f'  {k}/{len(d)} days [{time.time() - t0:.0f}s]', flush=True)
    f = pd.concat(frames, ignore_index=True)
    qf = OUT / 'adsb_quality.parquet'
    if qf.exists():                       # how good ADS-B is at this airport in this month, measured on arrivals
        q = pd.read_parquet(qf)
        f = f.merge(q.assign(month=q.month.astype('string'))[['APT', 'month'] + QUALITY_COLUMNS],
                    left_on=[f.APT.astype('string'), f.date.str[:7].astype('string')],
                    right_on=['APT', 'month'], how='left').drop(columns=['key_0', 'key_1', 'month'], errors='ignore')
    else:
        print('  (no adsb_quality.parquet yet: run `121 quality` first for the quality columns)', flush=True)
    f.to_parquet(OUT / 'adsb_features.parquet', index=False)
    n26 = (f.date >= '2026').sum()
    print(f'adsb_features.parquet: {len(f):,} rows ({n26:,} on 2026 ranking days), '
          f'usable push-back {int((f.adsb_at_stand == 1).sum()):,} [{time.time() - t0:.0f}s]')


NBR_COLUMNS = ['adsb_nb_taxi_c60', 'adsb_nb_taxi_n', 'adsb_nb_taxi_prev60', 'adsb_nb_queue_c60',
               'adsb_nb_surface', 'adsb_nb_tko_c60', 'adsb_nb_day_n',
               'adsb_nb_bias_c60', 'adsb_nb_bias_n', 'adsb_nb_bias_prev60']
TAXI_CLIP, QUEUE_CLIP, BIAS_CLIP = (60.0, 5400.0), (0.0, 3600.0), 1800.0


def _nbr_day(date):
    """final-file change: one day of build_neighbours' former serial loop (events, departures, match), or None."""
    ev, mt = match_day(date)
    if len(ev) == 0:
        return None
    return ev.assign(date=date), departures(date).assign(date=date), mt[['MVT_ID_mvt', 'k']].assign(date=date)


def build_neighbours(workers=1):
    """What the traces show the airport's surface doing around each departure - for EVERY challenge departure.

    The off-block columns only reach a row whose own aircraft was seen at its stand (25 % of the 2026 rows); these are
    measured over the OTHER aircraft, so they also reach the rows ADS-B never saw, which still carry 43 % of the strict
    SSE after A38. Every window takes the row's own aircraft out through the match, so nothing here is its own
    off-block. Rank-available: the 2026 traces and take-off times are in hand and no block time is read.

      adsb_nb_taxi_c60/_prev60  mean MEASURED taxi (take-off minus stand-leave) of the other departures seen leaving
                                within +-30 min / in the hour before: the taxi regime an unseen row sits in
      adsb_nb_taxi_n            how many aircraft that mean rests on (0 = no ground coverage here and now)
      adsb_nb_queue_c60         mean measured time from clearing the stand area to take-off = the runway queue
      adsb_nb_surface           aircraft measured between stand-leave and take-off at our take-off instant
      adsb_nb_tko_c60           ADS-B take-offs within +-30 min (the measured departure rate)
      adsb_nb_day_n             ground phases seen at the airport that day (the day's coverage scale)
      adsb_nb_bias_c60/_n/_prev60  mean of (ADS-B off-block minus AOBT_3) over the neighbours NM also matched: how far
                                NM's anchor is running from the physical push-back at this airport right now
    """
    import surge_context
    d = [x for x in days() if (OUT / 'events' / f'events_{x}.parquet').exists()]
    t0, evs, deps, mats = time.time(), [], [], []
    # final-file change: the per-day match ran in a serial loop; now a pool (order kept, same frames)
    with ProcessPoolExecutor(workers) as pool:
        for k, r in enumerate(pool.map(_nbr_day, d, chunksize=8), 1):
            if r is None:
                continue
            evs.append(r[0]); deps.append(r[1]); mats.append(r[2])
            if k % 50 == 0:
                print(f'  {k}/{len(d)} days [{time.time() - t0:.0f}s]', flush=True)
    E = pd.concat(evs, ignore_index=True); D = pd.concat(deps, ignore_index=True); M = pd.concat(mats, ignore_index=True)
    off = E.groupby('date', sort=False).size().cumsum().shift(1, fill_value=0)       # day-local k -> row of E
    E['gid'] = np.arange(len(E), dtype='int64')
    M['gid'] = M.k.values + off.reindex(M.date.values).values
    D = D.merge(M[['MVT_ID_mvt', 'gid']], on='MVT_ID_mvt', how='left', validate='one_to_one')
    D['APT'] = D.APT.astype('string'); E['APT'] = E.APT.astype('string')
    day_n = E.groupby(['APT', 'date'], observed=True).size().rename('adsb_nb_day_n').reset_index()
    # a trustworthy ground phase: at its own stand, not a post-landing tow, seen leaving it
    good = ((E.at_stand.values == 1) & (E.censored.values == 0) & (E.after_landing.values == 0)
            & np.isfinite(E.leave_lo.values))
    E['taxi_m'] = np.where(good, np.clip(E.t_tko.values - E.leave_lo.values, *TAXI_CLIP), np.nan)
    E['queue_m'] = np.where(good & np.isfinite(E.clear.values), np.clip(E.t_tko.values - E.clear.values, *QUEUE_CLIP), np.nan)
    bias = np.full(len(D), np.nan)                                   # NM anchor bias, on the rows NM matched too
    seen = np.isfinite(D.gid.values)
    gi = D.gid.values[seen].astype('int64')
    bias[seen] = np.where(good[gi], np.clip(E.leave_lo.values[gi] - D.t_a3.values[seen], -BIAS_CLIP, BIAS_CLIP), np.nan)
    D['bias'] = bias
    out = []
    for apt in sorted(D.APT.dropna().unique()):
        dd = D[(D.APT.values == apt) & np.isfinite(D.t_mvt.values)].sort_values('t_mvt', kind='stable').reset_index(drop=True)
        ee = E[E.APT.values == apt].sort_values('t_tko', kind='stable').reset_index(drop=True)
        t = dd.t_mvt.to_numpy(); tk = ee.t_tko.to_numpy(); lv = ee.leave_lo.to_numpy()
        pos = np.full(len(E), len(ee), 'int64'); pos[ee.gid.to_numpy()] = np.arange(len(ee))
        me = pos[np.where(np.isfinite(dd.gid.values), dd.gid.values, 0).astype('int64')]
        me = np.where(np.isfinite(dd.gid.values), me, len(ee))       # len(ee) = 'this row has no event of its own'
        seen_row = me < len(ee); si = np.where(seen_row, me, 0)      # win_mean needs an index inside the array
        f = pd.DataFrame({'MVT_ID_mvt': dd.MVT_ID_mvt.to_numpy()})

        def window(v, lo, hi):
            """win_mean over the other aircraft: the row's own event is taken out when it has one."""
            if len(ee) == 0:                                          # an airport ADS-B never sees on the ground
                return np.full(len(dd), np.nan), np.zeros(len(dd))
            m_all, n_all = surge_context.win_mean(t, tk, v, lo, hi)
            m_ex, n_ex = surge_context.win_mean(t, tk, v, lo, hi, si)
            return np.where(seen_row, m_ex, m_all), np.where(seen_row, n_ex, n_all)

        f['adsb_nb_taxi_c60'], f['adsb_nb_taxi_n'] = window(ee.taxi_m.to_numpy(), -1800, 1800)
        f['adsb_nb_taxi_prev60'], _ = window(ee.taxi_m.to_numpy(), -3600, 0)
        f['adsb_nb_queue_c60'], _ = window(ee.queue_m.to_numpy(), -1800, 1800)
        _, f['adsb_nb_tko_c60'] = window(np.ones(len(ee)), -1800, 1800)
        own = ((np.r_[lv, np.inf][me] <= t) & (np.r_[tk, np.inf][me] >= t)).astype('float64')   # 'departed' is t_tko < t
        g_lv = np.isfinite(lv)                                       # only phases with a stand-leave can be counted
        surface = (np.searchsorted(np.sort(lv[g_lv]), t, 'right').astype('float64')
                   - np.searchsorted(np.sort(tk[g_lv]), t, 'left'))
        f['adsb_nb_surface'] = np.maximum(surface - own, 0.0) if len(ee) else np.full(len(dd), np.nan)
        tb = dd.bias.to_numpy(); mine = np.arange(len(dd))
        f['adsb_nb_bias_c60'], f['adsb_nb_bias_n'] = surge_context.win_mean(t, t, tb, -1800, 1800, mine)
        f['adsb_nb_bias_prev60'], _ = surge_context.win_mean(t, t, tb, -3600, 0, mine)
        f['APT'] = apt; f['date'] = dd.date.to_numpy()
        out.append(f)
    g = pd.concat(out, ignore_index=True).merge(day_n, on=['APT', 'date'], how='left')
    g['adsb_nb_day_n'] = g.adsb_nb_day_n.fillna(0.0)
    g = g[['MVT_ID_mvt'] + NBR_COLUMNS]
    g.to_parquet(OUT / 'adsb_nbr.parquet', index=False)
    print(f'adsb_nbr.parquet: {len(g):,} rows, {int(np.isfinite(g.adsb_nb_taxi_c60.values).sum()):,} with a measured '
          f'neighbour taxi [{time.time() - t0:.0f}s]')


def check():
    f = pd.read_parquet(OUT / 'adsb_features.parquet')
    f['month'] = f.date.str[:7]
    lab = f[(f.date < '2026') & f.y.notna()].copy()
    lab['err'] = (lab.t_mvt - lab.adsb_taxi_leave) - lab.t_blk           # leave_lo minus recorded off-block
    use = lab[(lab.adsb_at_stand == 1) & lab.adsb_taxi_leave.notna()]
    print('2025 labelled departures, by airport')
    print(use.groupby('APT').apply(lambda g: pd.Series(dict(
        rows=len(g), within60=(g.err.abs() <= 60).mean(), within120=(g.err.abs() <= 120).mean(),
        med=g.err.median(), miss300=(g.err.abs() > 300).mean())), include_groups=False).round(3).to_string())
    cov = f.assign(usable=(f.adsb_at_stand == 1) & f.adsb_taxi_leave.notna())
    print('\nusable push-back share by month and airport')
    print(cov.pivot_table(index='month', columns='APT', values='usable', aggfunc='mean').round(2).to_string())
    cen = lab[lab.adsb_censored == 1]
    print(f'\ncensored rows {len(cen):,}: take-off - first sighting is below the true taxi in '
          f'{(cen.adsb_taxi_first <= cen.y).mean():.1%} (it is a lower bound)')


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['stands', 'events', 'quality', 'features', 'neighbours', 'check'])
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args()
    OUT = Path(a.out)
    P118.OUT = OUT
    {'stands': lambda: build_stands(a.workers), 'events': lambda: build_events(a.workers),
     'quality': lambda: build_quality(a.workers), 'features': lambda: build_features(a.workers),   # final-file change
     'neighbours': lambda: build_neighbours(a.workers), 'check': check}[a.cmd]()


if __name__ == '__main__':
    main()
