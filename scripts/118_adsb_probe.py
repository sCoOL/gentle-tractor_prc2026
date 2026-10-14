"""ADS-B ground coverage probe: can open adsb.lol traces see the push-back? (one day, 2025 labels only)

  py -3.14 scripts/118_adsb_probe.py extract --date 2025-07-15 --keep-raw
  py -3.14 scripts/118_adsb_probe.py match   --date 2025-07-15
  py -3.14 scripts/118_adsb_probe.py coverage --date 2026-07-20 --flights ITY674 QFA7006   # any day, no labels

Source: adsb.lol `globe_history_<year>` GitHub releases (ODbL-1.0; tag v<Y.m.d>-planes-readsb-prod-0, a tar
split in ~2 GB parts). The organiser confirmed on Discord (2026-09-17 and 09-19) that open ADS-B data with
ground coverage is allowed for off-block times. No official score is involved: the probe scores ADS-B
push-back estimates against the real 2025 off-blocks and against A37's strict OOF on the same rows.

Everything adsb.lol lives under PRC_ADSB_POINTS (scripts/prc_config.py; --out overrides): raw/ holds release
parts kept with --keep-raw, points_<date>.parquet and probe_<date>.parquet sit at the top.

extract  streams the parts (saving them to <out>/raw with --keep-raw, reusing them when complete), reads every
         traces/**/trace_full_<icao>.json, and keeps points inside a ~6 km box around the ten airports that
         are on the ground or below 6,000 ft (whether or not the trace ever reports 'ground') -> <out>/points_<date>.parquet (callsign forward/back-filled
         from the trace's detail objects).
match    finds ground->air transitions (take-offs), walks back through the ground phase, estimates the
         push-back (end of the first stationary run), matches take-offs to the day's departures (callsign +
         take-off within 180 s, else nearest unmatched take-off within 45 s) and prints coverage, accuracy
         against BLOCK_TIME and the RMSE A37 would have on the covered rows -> <out>/probe_<date>.parquet.
"""
import argparse
import gzip
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from common import AIRPORT_META, APTS, DATA, RUNS
import prc_config

OUT = prc_config.ADSB_DIR         # derived tables and per-day caches (PRC_ADSB_DIR)
POINTS = prc_config.ADSB_POINTS   # extracted points_<date>.parquet, read only here (PRC_ADSB_POINTS)


def points_path(date):
    """A points file in OUT (PRC_ADSB_DIR) wins, else the one in POINTS (PRC_ADSB_POINTS)."""
    p = OUT / f'points_{date}.parquet'
    return p if p.exists() else POINTS / f'points_{date}.parquet'


def _points_complete(p):
    """final-file change: a points file still being written by `118 extract` has no parquet footer yet."""
    try:
        import pyarrow.parquet as pq
        pq.ParquetFile(p).metadata
        return True
    except Exception:
        return False


def points_days():
    """Days that have a points file in OUT or in POINTS.
    final-file change: only COMPLETE files (footer readable), so a day being extracted is never cached empty."""
    return sorted({p.name[7:17] for d in (OUT, POINTS) for p in d.glob('points_*.parquet') if _points_complete(p)})
BOX_LAT, BOX_LON, MAX_ALT_FT = 0.06, 0.09, 6000
GAP_S = 900          # a ground phase breaks at a gap longer than this
MOVE_M = 30          # the aircraft has left its first position
TKO_TOL, POS_TOL = 180, 45
# cheap byte prefilter: a trace can only matter if some latitude inside an airport box appears in it
LAT_KEYS = sorted({f',{np.floor(v * 10) / 10:.1f}'.encode() for m in AIRPORT_META.values()
                   for v in np.arange(m['lat'] - BOX_LAT, m['lat'] + BOX_LAT + 0.1, 0.1) if v <= m['lat'] + BOX_LAT})


VARIANT = 'prod-0'      # also staging-0 (backup replica) and mlatonly-0 (separate airborne-MLAT archive)
TAKEOFF_COLUMNS = ['icao', 'APT', 'cs', 't_tko', 't_first', 't_move', 'lat0', 'lon0', 'gs0',
                   'still_s', 'n_ground', 'after_landing']


def assets(date):
    y = date[:4]
    tag = f"v{date.replace('-', '.')}-planes-readsb-{VARIANT}"
    done = OUT / 'raw' / f'{tag}.done'            # written by 119_adsb_download.py: no API call needed
    if done.exists():
        return [(n, s, None) for n, s in json.loads(done.read_text())['parts']]
    with urllib.request.urlopen(f'https://api.github.com/repos/adsblol/globe_history_{y}/releases/tags/{tag}',
                                timeout=60) as r:
        rel = json.load(r)
    return sorted(((a['name'], a['size'], a['browser_download_url']) for a in rel['assets']), key=lambda a: a[0])


class PartStream(io.RawIOBase):
    """The release parts as one byte stream; tees each part to raw_dir, reads complete local parts instead."""

    def __init__(self, parts, raw_dir=None):
        self.parts, self.raw_dir, self.cur, self.keep, self.nbytes = list(parts), raw_dir, None, None, 0
        self.t0 = time.time()

    def readable(self):
        return True

    def _open_next(self):
        name, size, url = self.parts.pop(0)
        local = Path(self.raw_dir) / name if self.raw_dir else None
        if local is not None and local.exists() and local.stat().st_size == size:
            print(f'  reading local {name}', flush=True)
            self.cur, self.keep = open(local, 'rb'), None
        else:
            print(f'  downloading {name} ({size / 1e9:.2f} GB)', flush=True)
            self.cur = urllib.request.urlopen(url, timeout=120)
            self.keep = open(str(local) + '.part', 'wb') if local is not None else None
            self.keep_name = local

    def readinto(self, b):
        while True:
            if self.cur is None:
                if not self.parts:
                    return 0
                self._open_next()
            n = self.cur.readinto(b)
            if n:
                if self.keep is not None:
                    self.keep.write(memoryview(b)[:n])
                self.nbytes += n
                return n
            self.cur.close()
            self.cur = None
            if self.keep is not None:
                self.keep.close()
                Path(str(self.keep_name) + '.part').replace(self.keep_name)
                self.keep = None


def trace_points(j):
    """Rows (icao, t, lat, lon, ground, alt, gs, track, cs, APT) of one trace inside the airport boxes."""
    tr = j.get('trace') or []
    if not tr:
        return []
    t0 = float(j['timestamp'])
    cs, cur = [None] * len(tr), None
    for i, p in enumerate(tr):
        d = p[8] if len(p) > 8 else None
        if isinstance(d, dict) and d.get('flight'):
            cur = d['flight'].strip() or cur
        cs[i] = cur
    first = next((c for c in cs if c), None)
    rows = []
    for i, p in enumerate(tr):
        lat, lon, alt = p[1], p[2], p[3]
        if lat is None or lon is None:
            continue
        ground = alt == 'ground'
        altv = np.nan if ground or alt is None else float(alt)
        if not ground and not (altv < MAX_ALT_FT):
            continue
        for a in APTS:
            m = AIRPORT_META[a]
            if abs(lat - m['lat']) < BOX_LAT and abs(lon - m['lon']) < BOX_LON:
                rows.append((j['icao'], t0 + float(p[0]), lat, lon, ground, altv, p[4], p[5], cs[i] or first, a))
                break
    return rows


def extract(date, keep_raw):
    OUT.mkdir(parents=True, exist_ok=True)
    raw_dir = OUT / 'raw' if keep_raw else None
    parts = assets(date)
    print(f'{date}: {len(parts)} parts, {sum(p[1] for p in parts) / 1e9:.2f} GB', flush=True)
    if raw_dir:
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
    import tarfile
    stream = PartStream(parts, raw_dir)
    rows, seen, kept, bad, t0 = [], 0, 0, 0, time.time()
    with tarfile.open(fileobj=io.BufferedReader(stream, 1 << 20), mode='r|') as tar:
        for m in tar:
            if not m.isfile() or 'trace_full_' not in m.name:
                continue
            raw = tar.extractfile(m).read()
            try:
                if raw[:2] == b'\x1f\x8b':
                    raw = gzip.decompress(raw)
            except Exception:           # one corrupt trace must not lose the day (2025-10-15 has one)
                bad += 1
                continue
            seen += 1
            if seen % 5000 == 0:
                el = time.time() - t0
                print(f'  {seen:,} traces, {kept:,} kept, {len(rows):,} points, {stream.nbytes / 1e9:.2f} GB, '
                      f'{stream.nbytes / 1e6 / el:.0f} MB/s', flush=True)
            if not any(k in raw for k in LAT_KEYS):
                continue
            try:
                pts = trace_points(json.loads(raw))
            except Exception:
                bad += 1
                continue
            if pts:
                kept += 1
                rows.extend(pts)
    while stream.readinto(bytearray(1 << 20)):   # drain the tar padding so the last part is finalised
        pass
    if seen == 0:
        raise RuntimeError('no trace_full files found: the archive layout is not what this script expects')
    d = pd.DataFrame(rows, columns=['icao', 't', 'lat', 'lon', 'ground', 'alt', 'gs', 'track', 'cs', 'APT'])
    d.to_parquet(OUT / f'points_{date}.parquet', index=False)
    print(f'{date}: {seen:,} traces, {kept:,} with airport points, {len(d):,} points, '
          f'{d.icao.nunique():,} aircraft, {bad} unreadable [{time.time() - t0:.0f}s]', flush=True)
    print(d.groupby('APT').agg(points=('t', 'size'), aircraft=('icao', 'nunique'), ground=('ground', 'mean')).to_string())


def dist_m(lat1, lon1, lat2, lon2):
    k = np.cos(np.radians((lat1 + lat2) / 2))
    return 111_195.0 * np.hypot(lat2 - lat1, (lon2 - lon1) * k)


def takeoffs(pts):
    """One row per ground->air transition with the ground phase before it."""
    out = []
    for (icao, apt), g in pts.sort_values('t').groupby(['icao', 'APT'], sort=False):
        t, gr = g.t.to_numpy(), g.ground.to_numpy(bool)
        la, lo, gs, cs = g.lat.to_numpy(), g.lon.to_numpy(), g.gs.to_numpy(float), g.cs.to_numpy(object)
        for i in np.where(gr[:-1] & ~gr[1:])[0] + 1:
            if t[i] - t[i - 1] > 120:
                continue
            j = i - 1
            while j > 0 and gr[j - 1] and t[j] - t[j - 1] < GAP_S:
                j -= 1
            after_landing = j > 0 and not gr[j - 1] and t[j] - t[j - 1] < GAP_S
            dist = dist_m(la[j], lo[j], la[j:i], lo[j:i])
            moved = np.where(dist > MOVE_M)[0]
            t_move = t[j + moved[0]] if len(moved) else np.nan
            names = [c for c in cs[j:min(i + 30, len(cs))] if c]
            out.append(dict(icao=icao, APT=apt, cs=max(set(names), key=names.count) if names else None,
                            t_tko=t[i], t_first=t[j], t_move=t_move, lat0=la[j], lon0=lo[j], gs0=gs[j],
                            still_s=(t_move - t[j]) if len(moved) else np.nan, n_ground=i - j,
                            after_landing=after_landing))
    # Preserve the schema when a feed contains airport points but no ground->air
    # transition. The separate mlatonly archive is a real example: coverage
    # should report zero rather than fail while merging an empty frame.
    return pd.DataFrame(out, columns=TAKEOFF_COLUMNS)


def epoch(s):
    return (s - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds()


def link(date):
    """The day's departures (2025 training or 2026 ranking) with their matched ADS-B take-off and ground phase."""
    pts = pd.read_parquet(points_path(date))   # final-file change (was OUT / points_<date>)
    tk = takeoffs(pts)
    month = pd.Timestamp(date).month
    f = (DATA / 'ranking.parquet' if date >= '2026' else
         sorted(DATA.glob(f'training_{date[:4]}-{month:02d}-01_*.parquet'))[0])
    dep = pd.read_parquet(f)
    dep = dep[(dep.PHASE_mvt == 'DEP') & (dep.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d') == date)].copy()
    dep['APT'] = dep.ADEP_mvt
    dep['t_mvt'], dep['t_blk'], dep['t_a3'] = epoch(dep.MVT_TIME_UTC_mvt), epoch(dep.BLOCK_TIME_UTC_mvt), epoch(dep.AOBT_3_flt)
    dep['cs'] = dep.CALLSIGN_flt.astype('string').str.strip()

    # 1) callsign + take-off time, 2) nearest unmatched take-off at the same airport.
    # Some archive variants have no ground->air transitions at all.
    if tk.empty:
        d = dep.copy()
        for col in ['icao', 't_tko', 't_first', 't_move', 'lat0', 'lon0', 'gs0', 'still_s',
                    'n_ground', 'after_landing', 'gap']:
            d[col] = np.nan
        d['how'] = pd.NA
    else:
        tk['k'] = np.arange(len(tk))
        c = dep[['MVT_ID_mvt', 'APT', 'cs', 't_mvt']].dropna(subset=['cs']).merge(tk, on=['APT', 'cs'])
        c = c[(c.t_tko - c.t_mvt).abs() <= TKO_TOL].assign(gap=lambda x: (x.t_tko - x.t_mvt).abs())
        c = c.sort_values('gap').drop_duplicates('MVT_ID_mvt').drop_duplicates('k').assign(how='callsign')
        rest = dep[~dep.MVT_ID_mvt.isin(c.MVT_ID_mvt)][['MVT_ID_mvt', 'APT', 't_mvt']]
        p = rest.merge(tk[~tk.k.isin(c.k)].drop(columns='cs'), on='APT')
        p = p[(p.t_tko - p.t_mvt).abs() <= POS_TOL].assign(gap=lambda x: (x.t_tko - x.t_mvt).abs())
        p = p.sort_values('gap').drop_duplicates('MVT_ID_mvt').drop_duplicates('k').assign(how='position')
        m = pd.concat([c.drop(columns=['cs', 't_mvt']), p.drop(columns=['t_mvt'])], ignore_index=True)
        d = dep.merge(m.drop(columns='APT'), on='MVT_ID_mvt', how='left')

    st = pd.read_parquet(DATA / 'ext_stand_coords.parquet')[['APT', 'stand', 'lat', 'lon']]
    st = st.rename(columns={'stand': 'STAND_mvt', 'lat': 'st_lat', 'lon': 'st_lon'})
    d = d.merge(st.astype({'STAND_mvt': 'string'}), on=['APT', 'STAND_mvt'], how='left')
    d['d_stand0'] = dist_m(d.lat0, d.lon0, d.st_lat, d.st_lon)
    d['at_stand'] = (d.d_stand0 <= 150) & ~d.after_landing.fillna(False).astype(bool) & (d.still_s >= 20)
    d['err_move'] = d.t_move - d.t_blk          # push-back estimate minus the airport's off-block
    d['err_first'] = d.t_first - d.t_blk
    d['err_a3'] = d.t_a3 - d.t_blk
    # the push-back estimator that works (2025-07-15): first sighting at the stand, not the first movement
    d['usable'] = (d.how.notna() & (d.d_stand0 <= 150) & (d.still_s < 600)
                   & ~d.after_landing.fillna(False).astype(bool)).fillna(False).astype(bool)
    return d, tk


def coverage(date, flights=()):
    """Per-airport coverage for any day (no labels needed), plus details of named flights."""
    d, tk = link(date)
    tab = d.groupby('APT').agg(deps=('MVT_ID_mvt', 'size'), takeoff_matched=('how', lambda s: s.notna().mean()),
                               first_seen_at_stand=('usable', 'mean'))
    tab.loc['ALL'] = [len(d), d.how.notna().mean(), d.usable.mean()]
    print(f'{date}: take-offs seen {len(tk):,}, departures {len(d):,}')
    print(tab.round(3).to_string())
    for fl in flights:
        x = d[d.FLIGHT_mvt.astype('string').str.contains(fl, regex=False, na=False)
              | d.CALLSIGN_flt.astype('string').str.contains(fl, regex=False, na=False)]
        for _, r in x.iterrows():
            t = lambda v: 'n/a' if pd.isna(v) else pd.Timestamp(v, unit='s', tz='UTC').strftime('%H:%M:%S')
            print(f"{r.FLIGHT_mvt} {r.APT} stand {r.STAND_mvt} take-off {t(r.t_mvt)} sched "
                  f"{r.SCHED_TIME_UTC_mvt:%m-%d %H:%M} | ADS-B: {r.how or 'not seen'} icao {r.icao} first {t(r.t_first)} "
                  f"move {t(r.t_move)} still {r.still_s} s, {r.d_stand0:.0f} m from stand, ground points {r.n_ground}")
    d.to_parquet(OUT / f'link_{date}.parquet', index=False)


def match(date):
    d, tk = link(date)
    oof = pd.read_parquet(RUNS / 'oof' / 'bundle_A37_strict.parquet', columns=['MVT_ID_mvt', 'pred'])
    d = d.merge(oof, on='MVT_ID_mvt', how='left')
    d['y'] = d.TAXITIME_SEC_mvt
    d['y_adsb'] = d.t_mvt - d.t_move

    def row(g):
        cov = g[g.at_stand & g.t_move.notna()]
        e = cov.err_move
        r = dict(deps=len(g), matched=g.how.notna().mean(), by_callsign=(g.how == 'callsign').mean(),
                 at_stand=len(cov) / len(g), med_err=e.median(), within60=(e.abs() <= 60).mean(),
                 within120=(e.abs() <= 120).mean(), a3_within60=(cov.err_a3.abs() <= 60).mean(),
                 rmse_A37_cov=np.sqrt(((cov.pred - cov.y) ** 2).mean()),
                 rmse_adsb_cov=np.sqrt(((cov.y_adsb - cov.y) ** 2).mean()),
                 sse_share_cov=((cov.pred - cov.y) ** 2).sum() / ((g.pred - g.y) ** 2).sum())
        return pd.Series(r)

    tab = d.groupby('APT').apply(row)
    tab.loc['ALL'] = row(d)
    pd.set_option('display.width', 250)
    print(tab.round(3).to_string())
    # what replacing A37 by the ADS-B estimate on covered rows would do to the day's RMSE
    cov = d.at_stand & d.t_move.notna()
    base = np.sqrt(((d.pred - d.y) ** 2).mean())
    rep = np.sqrt(((np.where(cov, d.y_adsb, d.pred) - d.y) ** 2).mean())
    print(f'{date}: A37 day RMSE {base:.1f} -> {rep:.1f} with the ADS-B push-back on {cov.sum():,} covered rows '
          f'({cov.mean():.1%}); take-offs seen {len(tk):,}, departures {len(d):,}')
    keep = ['MVT_ID_mvt', 'APT', 'STAND_mvt', 'how', 'gap', 'icao', 't_first', 't_move', 'still_s', 'gs0', 'd_stand0',
            'after_landing', 'at_stand', 'err_move', 'err_first', 'err_a3', 'y', 'pred', 'y_adsb']
    d[keep].to_parquet(OUT / f'probe_{date}.parquet', index=False)


def main():
    global OUT, VARIANT
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['extract', 'match', 'coverage'])
    ap.add_argument('--date', required=True)
    ap.add_argument('--flights', nargs='*', default=[], help='coverage: print these flights (substring of FLIGHT/CALLSIGN)')
    ap.add_argument('--keep-raw', action='store_true', help='keep/reuse the release parts in <out>/raw')
    ap.add_argument('--out', default=str(OUT), help='folder for everything adsb.lol')
    ap.add_argument('--variant', default=VARIANT,
                    help='release variant: prod-0 (default), staging-0, or mlatonly-0')
    a = ap.parse_args()
    OUT, VARIANT = Path(a.out), a.variant
    {'extract': lambda: extract(a.date, a.keep_raw), 'match': lambda: match(a.date),
     'coverage': lambda: coverage(a.date, a.flights)}[a.cmd]()


if __name__ == '__main__':
    main()
