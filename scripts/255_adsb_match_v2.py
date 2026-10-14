"""ADS-B take-off matching v2: normalized callsigns in pass 1, no cross-airline pass-2 matches (2026-09-26).

  py -3.14 scripts/255_adsb_match_v2.py compare      # v1 vs v2 match per departure -> runs/adsb_match_v2/{matches,compare}
  py -3.14 scripts/255_adsb_match_v2.py features     # <PRC_ADSB_DIR>/adsb_features_v2.parquet (121's columns, v2 match)

v1 (scripts/121_adsb_events.py match_day, unchanged and still what every saved model reads):
  pass 1  NM callsign (CALLSIGN_flt) = ADS-B callsign, same airport, take-off within 180 s
  pass 2  any unclaimed ADS-B take-off at the same airport within 45 s (no callsign, runway or type check)
Unmatched departures have no NM callsign, so v1 matches all of them by time alone; on 40 sampled days 3.7 % of those
matches were another airline's aircraft (37 of 50 at EHAM).

v2 changes exactly two things:
  1. pass 1 accepts any of the departure's callsign candidates, compared in NORMALIZED form: the NM callsign, the
     airport flight number, and the NM callsign of its destination arrival (scripts/250 backfill). Normalized =
     airline prefix mapped to its ICAO code (IATA->ICAO pairs learned from NM-matched rows, >= 20 rows and >= 80 %
     of that prefix) + the number without leading zeros, so LH002 = DLH2 = DLH002 and CZ2572 = CSG2572.
  2. a pass-2 match is dropped when both sides name an airline and the normalized airlines differ.
Label-free: no departure block time is read for matching.
"""
import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import flightnorm as fn  # noqa: E402

e = importlib.import_module('121_adsb_events')
OUT = common.RUNS / 'adsb_match_v2'
FEATURES_V2 = e.OUT / 'adsb_features_v2.parquet'
BACKFILL = common.RUNS / 'nm_backfill' / 'backfill.parquet'
PREFIX = re.compile(r'^([A-Z]{3}|[A-Z0-9]{2})')
MAP_MIN_N, MAP_MIN_SHARE = 20, 0.8
FUZZY_TOL = 60.0      # pass 1c (one-character callsign difference) needs a take-off within 60 s and a unique pair
_PMAP, _BF, _RAW = {}, None, {}


def prefix_map() -> dict:
    """Airport flight-number prefix -> ICAO callsign prefix, learned where both sides carry the same number."""
    if _PMAP:
        return _PMAP
    files = [str(p) for p in sorted(common.DATA.glob('training_2025-*.parquet'))] + [str(common.DATA / 'ranking.parquet')]
    d = duckdb.connect().execute("""
        WITH m AS (SELECT FLIGHT_mvt f, CALLSIGN_flt c FROM read_parquet(?, union_by_name=true)
                   WHERE FLIGHT_mvt IS NOT NULL AND CALLSIGN_flt IS NOT NULL),
        p AS (SELECT regexp_extract(f, '^([A-Z]{3}|[A-Z0-9]{2})', 1) a, regexp_extract(c, '^([A-Z]{3}|[A-Z0-9]{2})', 1) b,
                     ltrim(substr(f, length(regexp_extract(f, '^([A-Z]{3}|[A-Z0-9]{2})', 1)) + 1), '0') fs,
                     ltrim(substr(c, length(regexp_extract(c, '^([A-Z]{3}|[A-Z0-9]{2})', 1)) + 1), '0') cs FROM m),
        n AS (SELECT a, b, count(*) n FROM p WHERE fs = cs AND fs <> '' AND a <> '' AND b <> '' GROUP BY ALL),
        t AS (SELECT *, sum(n) OVER (PARTITION BY a) tot FROM n)
        SELECT a, b FROM t WHERE n >= ? AND n >= ? * tot""", [files, MAP_MIN_N, MAP_MIN_SHARE]).fetchall()
    _PMAP.update({a: b for a, b in d})
    fn.set_prefix_map(_PMAP)
    return _PMAP


def airline(s) -> str:
    """Normalized airline code of a flight number or callsign ('' when none): scripts/flightnorm.py."""
    _ensure_map()
    return fn.airline(s)


def norm(s) -> str:
    """Normalized callsign (scripts/flightnorm.py key): ICAO airline + number without leading zeros + suffix, LIRF doubled
    prefixes undone ('' when none). Until 2026-10-07 this was prefix map + leading zeros only."""
    _ensure_map()
    return fn.key(s)


def _ensure_map():
    if not _PMAP:
        fn.set_prefix_map(prefix_map())


_ALIAS = {}


def aliases() -> dict:
    """(airport, flight-number key) -> NM callsign keys usually flown under it (flightnorm.learn_aliases, both years,
    label-free)."""
    if not _ALIAS:
        _ensure_map()
        files = [str(p) for p in sorted(common.DATA.glob('training_2025-*.parquet'))] + [str(common.DATA / 'ranking.parquet')]
        d = duckdb.connect().execute("""SELECT ADEP_mvt a, FLIGHT_mvt f, CALLSIGN_flt c FROM read_parquet(?, union_by_name=true)
            WHERE PHASE_mvt = 'DEP' AND FLIGHT_mvt IS NOT NULL AND CALLSIGN_flt IS NOT NULL""", [files]).df()
        _ALIAS.update(fn.learn_aliases(d.a, d.f, d.c))
    return _ALIAS


def backfill_callsigns() -> pd.Series:
    global _BF
    if _BF is None:
        b = pd.read_parquet(BACKFILL, columns=['MVT_ID_mvt', 'bf_callsign'])
        _BF = b.dropna().set_index('MVT_ID_mvt').bf_callsign
    return _BF


def departures(date) -> pd.DataFrame:
    """121's departures of the day plus the airport flight number and the destination-arrival NM callsign."""
    dep = e.departures(date)
    src = (common.DATA / 'ranking.parquet' if date >= '2026'
           else sorted(common.DATA.glob(f'training_{date[:4]}-{date[5:7]}-01_*.parquet'))[0])
    if src not in _RAW:
        _RAW.clear()
        _RAW[src] = pd.read_parquet(src, columns=['MVT_ID_mvt', 'FLIGHT_mvt'])
    dep = dep.merge(_RAW[src], on='MVT_ID_mvt', how='left')
    dep['bf_cs'] = backfill_callsigns().reindex(dep.MVT_ID_mvt).to_numpy()
    return dep


def match_day_v2(date):
    """Same contract as 121.match_day: (events with k, matched departures with the event columns, how, gap)."""
    ev = pd.read_parquet(e.OUT / 'events' / f'events_{date}.parquet')
    if ev.empty:
        return ev, pd.DataFrame()
    ev = ev.astype({'APT': 'string', 'cs': 'string'}).assign(k=np.arange(len(ev)))
    _ensure_map()
    ev['nal'] = [airline(x) if isinstance(x, str) else '' for x in ev.cs]
    evk = pd.concat([pd.DataFrame({'k': ev.k, 'ncs': [fn.key(x) for x in ev.cs]}),
                     pd.DataFrame({'k': ev.k, 'ncs': [fn.base_key(x) for x in ev.cs]})]).query("ncs != ''").drop_duplicates()
    dep = departures(date)
    al = aliases()
    cand = []
    for col in ('cs', 'FLIGHT_mvt', 'bf_cs'):
        for f in (fn.key, fn.base_key):
            cand.append(pd.DataFrame({'MVT_ID_mvt': dep.MVT_ID_mvt, 'ncs': [f(x) for x in dep[col]]}))
    alias_rows = [(m, c) for m, a, f in zip(dep.MVT_ID_mvt, dep.APT, dep.FLIGHT_mvt) for c in al.get((a, fn.key(f)), [])]
    if alias_rows:
        cand.append(pd.DataFrame(alias_rows, columns=['MVT_ID_mvt', 'ncs']))
    cand = pd.concat(cand, ignore_index=True)
    cand = cand[cand.ncs != ''].drop_duplicates()
    c = dep.merge(cand, on='MVT_ID_mvt').merge(evk, on='ncs').merge(ev.drop(columns='cs'), on=['k'])
    c = c[(c.APT_x == c.APT_y)].drop(columns=['APT_y']).rename(columns={'APT_x': 'APT'}) if 'APT_x' in c.columns else c
    c = c[(c.t_tko - c.t_mvt).abs() <= e.TKO_TOL].assign(gap=lambda z: (z.t_tko - z.t_mvt).abs(), how=1)
    c = c.sort_values('gap').drop_duplicates('MVT_ID_mvt').drop_duplicates('k')
    # pass 1c, fuzzy: one-character difference after the airline (same airline), take-off within FUZZY_TOL, unique
    rest = dep[~dep.MVT_ID_mvt.isin(c.MVT_ID_mvt)]
    evf = ev[~ev.k.isin(c.k)]
    fz_e = pd.DataFrame([(k, z) for k, x in zip(evf.k, evf.cs) for z in fn.fuzzy_keys(x)], columns=['k', 'fz'])
    fz_d = pd.DataFrame([(m, z) for m, x, y in zip(rest.MVT_ID_mvt, rest.cs, rest.FLIGHT_mvt)
                         for z in set(fn.fuzzy_keys(x)) | set(fn.fuzzy_keys(y))], columns=['MVT_ID_mvt', 'fz'])
    f = rest.merge(fz_d, on='MVT_ID_mvt').merge(fz_e, on='fz').drop(columns='fz').drop_duplicates(['MVT_ID_mvt', 'k'])
    f = f.merge(evf.drop(columns='cs'), on='k', suffixes=('', '_ev'))
    f = f[(f.APT == f.APT_ev)].drop(columns=['APT_ev']) if 'APT_ev' in f.columns else f
    f = f[(f.t_tko - f.t_mvt).abs() <= FUZZY_TOL].assign(gap=lambda z: (z.t_tko - z.t_mvt).abs(), how=3)
    n_m = f.groupby('MVT_ID_mvt').k.transform('nunique')
    n_k = f.groupby('k').MVT_ID_mvt.transform('nunique')
    f = f[(n_m == 1) & (n_k == 1)]
    c = pd.concat([c, f], ignore_index=True)
    rest = dep[~dep.MVT_ID_mvt.isin(c.MVT_ID_mvt)]
    p = rest.merge(ev[~ev.k.isin(c.k)].drop(columns='cs'), on='APT')
    p = p[(p.t_tko - p.t_mvt).abs() <= e.POS_TOL].assign(gap=lambda z: (z.t_tko - z.t_mvt).abs(), how=2)
    p = p.sort_values('gap').drop_duplicates('MVT_ID_mvt').drop_duplicates('k')
    dal = pd.Series([airline(a) or airline(b) for a, b in zip(p.cs, p.FLIGHT_mvt)], index=p.index)
    p = p[~((dal != '') & (p.nal != '') & (dal != p.nal))]
    keep = [x for x in c.columns if x not in ('ncs', 'nal', 'FLIGHT_mvt', 'bf_cs', 'cs')]
    return ev.drop(columns=['nal']), pd.concat([c[keep], p[keep]], ignore_index=True)


def days():
    return [d for d in e.days() if (e.OUT / 'events' / f'events_{d}.parquet').exists()]


def _compare_day(date):
    if e.departures(date).empty:     # final-file change: an edge day with points but no departure (2026-03-01, 05-31):
        return None                  # match_day_v2 raises (empty object-dtype fuzzy frame merged on MVT_ID_mvt)
    ev, m1 = e.match_day(date)
    _, m2 = match_day_v2(date)
    if len(ev) == 0:
        return None
    a = m1[['MVT_ID_mvt', 'k', 'how', 'gap', 'leave_lo', 'at_stand', 't_mvt']].rename(columns=lambda c: c if c == 'MVT_ID_mvt' else c + '_1')
    b = m2[['MVT_ID_mvt', 'k', 'how', 'gap', 'leave_lo', 'at_stand', 't_mvt']].rename(columns=lambda c: c if c == 'MVT_ID_mvt' else c + '_2')
    x = a.merge(b, on='MVT_ID_mvt', how='outer').assign(date=date)
    x = x.merge(ev[['k', 'cs']].rename(columns={'k': 'k_1', 'cs': 'ev_cs_1'}), on='k_1', how='left')
    return x.merge(ev[['k', 'cs']].rename(columns={'k': 'k_2', 'cs': 'ev_cs_2'}), on='k_2', how='left')


def _compare_day_cached(date):
    """final-file change: _compare_day(date) cached in <OUT>/days/<date>.parquet (None = no departures / no events: not
    cached). Valid while the events, ranking file, backfill and learned prefix map / aliases do not change."""
    f = OUT / 'days' / f'{date}.parquet'
    if f.exists():
        return pd.read_parquet(f)
    x = _compare_day(date)
    if x is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        x.to_parquet(f, index=False)
    return x


def _features_day_cached(date):
    """final-file change: _features_day(date) cached in <121.OUT>/features_v2_days/<date>.parquet."""
    f = e.OUT / 'features_v2_days' / f'{date}.parquet'
    if f.exists():
        return pd.read_parquet(f)
    x = _features_day(date)
    if len(x.columns):
        f.parent.mkdir(parents=True, exist_ok=True)
        x.to_parquet(f, index=False)
    return x


def compare(workers: int = 1):
    """v1 vs v2 match per departure, every extracted day."""
    from concurrent.futures import ProcessPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0, d = [], time.time(), days()
    with ProcessPoolExecutor(workers) as pool:
        for i, x in enumerate(pool.map(_compare_day_cached, d, chunksize=4), 1):   # final-file change (was _compare_day)
            if x is not None:
                rows.append(x)
            if i % 50 == 0:
                print(f'  {i} days [{time.time() - t0:.0f}s]', flush=True)
    x = pd.concat(rows, ignore_index=True)
    x.to_parquet(OUT / 'matches.parquet', index=False)
    print(f'{len(x):,} departures with a v1 or v2 match [{time.time() - t0:.0f}s]')


def _features_day(date):
    if e.departures(date).empty:     # final-file change: same edge-day guard as _compare_day
        return pd.DataFrame()
    e.match_day = match_day_v2
    return e.join_day(date)


def features(workers: int = 1):
    """121's feature table with the v2 match (121.build_features with match_day swapped); the v1 file is untouched."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as pool:
        frames = list(pool.map(_features_day_cached, days(), chunksize=4))   # final-file change (was _features_day)
    f = pd.concat(frames, ignore_index=True)
    q = pd.read_parquet(e.OUT / 'adsb_quality.parquet')
    f = f.merge(q.assign(month=q.month.astype('string'))[['APT', 'month'] + e.QUALITY_COLUMNS],
                left_on=[f.APT.astype('string'), f.date.str[:7].astype('string')], right_on=['APT', 'month'],
                how='left').drop(columns=['key_0', 'key_1', 'month'], errors='ignore')
    f.to_parquet(FEATURES_V2, index=False)
    print(f'{FEATURES_V2}: {len(f):,} rows')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['compare', 'features'])
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    compare(a.workers) if a.mode == 'compare' else features(a.workers)
