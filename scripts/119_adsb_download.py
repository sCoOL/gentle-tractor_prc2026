"""Download adsb.lol globe_history daily releases (ODbL-1.0) for date ranges into <PRC_ADSB_POINTS>/raw.

  py -3.14 scripts/119_adsb_download.py 2026-07-01:2026-07-31 2026-01-01:2026-01-31 --workers 4 --extract
  py -3.14 scripts/119_adsb_download.py recommended --workers 4 --extract
  py -3.14 scripts/119_adsb_download.py 2025-07-15 --dry-run         # list the parts and sizes only
  py -3.14 scripts/119_adsb_download.py 2025-05-28:2025-06-12 2025-10-15 2025-12-31 --preferred --replace --extract

--preferred takes each day's release from adsb.lol's PREFERRED_RELEASES.txt (root of globe_history_2025 / _2026)
instead of -prod-0: 2025-05-28..06-10 exist only as -staging-0tmp, 2025-12-31 only in the 2026 repo, and where one
replica is damaged the other is usually intact (adsb.lol Discord, read 2026-09-26). --replace re-extracts a day that
already has points and moves its old points + per-day caches (events/geom/stops/times) to <out>/replaced/<day>/, so
the next 121/122/124/232 build recomputes that day; damaged prod-0 copies on disk: 2025-06-11 (8 % of a normal
day's points), 2025-06-12 (72 %), 2025-10-15 (74 %, zlib error in its extraction).


Safe to stop and re-run at any time: finished parts are skipped, an interrupted part continues from its size with an
HTTP Range request, and a day counts as complete only when raw/<tag>.done (part names and sizes) exists, written after
every part reached its full size. No GitHub API calls (anonymous limit 60/hour): part URLs are built from the tag and
probed .tar.aa, .tar.ab, ... until 404. Windows is kept awake while it runs (closing a laptop lid can still sleep it).
With --extract each finished day is passed to 118_adsb_probe.py extract in a separate process
(<out>/points_<date>.parquet; logs in <out>/logs), while the downloads continue.

A range whose end is before its start runs backward (2025-12-31:2025-01-01).
Presets: 'recommended' = Jul 2026, Jan 2026, then all of 2025 backward from 31 Dec (order: recent
months first, because 2025 ground coverage grows over the year: EGLL/EDDM none on 2025-07-15, 79 % / 55 % on 2026-07-20).
'lirf21' = the 14 other 2025 take-off days of the 21 LIRF NM-unmatched rows delayed 13-32 h.
"""
import argparse
import ctypes
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc_config  # noqa: E402

OUT = prc_config.ADSB_POINTS   # downloads, extracted points and set-asides (PRC_ADSB_POINTS)
LIRF21 = ['2025-03-28', '2025-04-18', '2025-06-12', '2025-06-25', '2025-08-02', '2025-08-27', '2025-09-06',
          '2025-09-07', '2025-09-26', '2025-09-28', '2025-11-04', '2025-11-12', '2025-12-19', '2025-12-28']
PRESETS = {
    'recommended': ['2026-07-01:2026-07-31', '2026-01-01:2026-01-31', '2025-12-31:2025-01-01'],
    'lirf21': LIRF21,
}
CHUNK = 1 << 20
LOCK = threading.Lock()
STATS = {'bytes': 0, 't0': time.time(), 'last': 0.0}


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    with LOCK:
        print(line, flush=True)
        with open(OUT / 'download.log', 'a', encoding='utf-8') as f:
            f.write(line + '\n')


def expand(specs):
    days = []
    for s in specs:
        if s in PRESETS:
            days += expand(PRESETS[s])
            continue
        a, _, b = s.partition(':')
        d0 = dt.date.fromisoformat(a)
        d1 = dt.date.fromisoformat(b) if b else d0
        step = 1 if d1 >= d0 else -1               # END before START = backward, e.g. 2025-12-31:2025-01-01
        days += [(d0 + dt.timedelta(n)).isoformat() for n in range(0, (d1 - d0).days + step, step)]
    seen = set()
    return [d for d in days if not (d in seen or seen.add(d))]


VARIANT = 'prod-0'      # adsb.lol also publishes staging-0 (backup replica) and mlatonly-0 (separate airborne MLAT);
                        # --variant <name> --out <other folder> keeps either beside the production set


PREFERRED = {}          # --preferred: day -> (repo year, variant) from the PREFERRED_RELEASES.txt lists
REPLACE = False         # --replace: re-extract days that already have points (old files kept in <out>/replaced)


def load_preferred():
    """day -> (repo year, variant) from PREFERRED_RELEASES.txt of both yearly repos (plain files, no API). The 2026
    list also names late-2025 days; a day's own-year list wins where both name it."""
    out = {}
    for y in ('2025', '2026'):
        url = f'https://raw.githubusercontent.com/adsblol/globe_history_{y}/main/PREFERRED_RELEASES.txt'
        req = urllib.request.Request(url, headers={'User-Agent': 'prc2026-adsb'})
        text = retry(lambda: urllib.request.urlopen(req, timeout=60).read().decode(), f'{y} PREFERRED_RELEASES.txt')
        for m in re.finditer(r'globe_history_(\d{4})/releases/download/v(\d{4})\.(\d\d)\.(\d\d)-planes-readsb-([\w-]+?)/', text):
            day = f'{m[2]}-{m[3]}-{m[4]}'
            if day not in out or m[1] == day[:4]:
                out[day] = (m[1], m[5])
    return out


def variant_of(day):
    return PREFERRED[day][1] if day in PREFERRED else VARIANT


def tag_of(day):
    return f"v{day.replace('-', '.')}-planes-readsb-{variant_of(day)}"


def url_of(day, name):
    repo = PREFERRED[day][0] if day in PREFERRED else day[:4]
    return f'https://github.com/adsblol/globe_history_{repo}/releases/download/{tag_of(day)}/{name}'


def set_aside(day):
    """--replace: move the day's old points and per-day caches out of the way (kept, not deleted)."""
    keep = OUT / 'replaced' / day
    for f in [OUT / f'points_{day}.parquet', OUT / 'events' / f'events_{day}.parquet', OUT / 'geom' / f'geom_{day}.parquet',
              OUT / 'stops' / f'stops_{day}.parquet', OUT / 'times' / f'{day}.parquet']:
        if f.exists():
            keep.mkdir(parents=True, exist_ok=True)
            f.replace(keep / f.name)
    return keep


def total_size(url):
    """Size of a release file from a one-byte ranged GET; None on 404."""
    req = urllib.request.Request(url, headers={'Range': 'bytes=0-0', 'User-Agent': 'prc2026-adsb'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get('Content-Range')
            return int(cr.split('/')[-1]) if cr else int(r.headers['Content-Length'])
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def discover(day):
    tag, parts = tag_of(day), []
    for i in range(26):
        name = f'{tag}.tar.a{chr(97 + i)}'
        size = retry(lambda: total_size(url_of(day, name)), f'{name} size')
        if size is None:
            break
        parts.append((name, size))
    if not parts:
        size = retry(lambda: total_size(url_of(day, f'{tag}.tar')), f'{tag}.tar size')
        if size is not None:
            parts.append((f'{tag}.tar', size))
    return parts


def retry(fn, what, tries=8):
    for k in range(tries):
        try:
            return fn()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                raise
            wait = min(300, 5 * 2 ** k)
            log(f'  {what}: {type(e).__name__} {e}; retry {k + 1}/{tries} in {wait}s')
            time.sleep(wait)
    raise RuntimeError(f'{what}: gave up after {tries} tries')


def fetch_part(day, name, size, raw):
    final, part = raw / name, raw / (name + '.part')
    if final.exists() and final.stat().st_size == size:
        return
    if final.exists():
        final.replace(part)                       # wrong size: treat as partial
    def once():
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if have < size:
            req = urllib.request.Request(url_of(day, name), headers={'Range': f'bytes={have}-',
                                                                      'User-Agent': 'prc2026-adsb'})
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = 'ab' if r.status == 206 else 'wb'  # 200 = server ignored the range: start over
                with open(part, mode) as f:
                    while True:
                        b = r.read(CHUNK)
                        if not b:
                            break
                        f.write(b)
                        tick(len(b))
        if part.stat().st_size != size:
            raise ConnectionError(f'{name}: {part.stat().st_size} of {size} bytes')
    retry(once, name, tries=12)
    part.replace(final)


def tick(n):
    with LOCK:
        STATS['bytes'] += n
        now = time.time()
        if now - STATS['last'] >= 60:
            STATS['last'] = now
            el = now - STATS['t0']
            print(f"{time.strftime('%H:%M:%S')}  downloaded {STATS['bytes'] / 1e9:.1f} GB this run, "
                  f"{STATS['bytes'] / 1e6 / el:.1f} MB/s", flush=True)


def download_day(day, raw, dry):
    done = raw / f'{tag_of(day)}.done'
    if done.exists():
        log(f'{day}: already complete')
        return True
    try:
        parts = discover(day)
    except Exception as e:
        log(f'{day}: could not list parts ({e})')
        return False
    if not parts:
        log(f'{day}: no release on GitHub (404), skipped')
        return False
    need = sum(s for _, s in parts)
    log(f'{day}: {len(parts)} parts, {need / 1e9:.2f} GB')
    if dry:
        return False
    free = shutil.disk_usage(raw).free
    if free < need + 20e9:
        log(f'{day}: only {free / 1e9:.0f} GB free on the target drive, stopping this day')
        return False
    t0 = time.time()
    try:
        for name, size in parts:
            fetch_part(day, name, size, raw)
    except Exception as e:
        log(f'{day}: FAILED ({e}); re-run to resume')
        return False
    done.write_text(json.dumps({'day': day, 'tag': tag_of(day), 'parts': parts,
                                'finished': time.strftime('%Y-%m-%d %H:%M:%S')}))
    log(f'{day}: complete in {(time.time() - t0) / 60:.1f} min')
    return True


def extract_day(day):
    pts = OUT / f'points_{day}.parquet'
    if pts.exists() and not REPLACE:
        log(f'{day}: points already extracted')
        return
    if pts.exists():
        log(f'{day}: replacing the extracted points; old files kept in {set_aside(day)}')
    (OUT / 'logs').mkdir(exist_ok=True)
    script = Path(__file__).resolve().parent / '118_adsb_probe.py'
    t0 = time.time()
    with open(OUT / 'logs' / f'extract_{day}.log', 'w', encoding='utf-8') as f:
        rc = subprocess.run([sys.executable, '-u', str(script), 'extract', '--date', day, '--keep-raw', '--out', str(OUT),
                             '--variant', variant_of(day)],
                            cwd=script.parent, stdout=f, stderr=subprocess.STDOUT,
                            env={**__import__('os').environ, 'PYTHONIOENCODING': 'utf-8'}).returncode
    log(f'{day}: extract {"ok" if rc == 0 else f"FAILED rc={rc}, see logs/extract_{day}.log"} '
        f'({(time.time() - t0) / 60:.1f} min)')


def main():
    global OUT, VARIANT, REPLACE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dates', nargs='+', help="YYYY-MM-DD, YYYY-MM-DD:YYYY-MM-DD, or a preset (recommended, lirf21)")
    ap.add_argument('--workers', type=int, default=4, help='days downloaded in parallel')
    ap.add_argument('--extract', action='store_true', help='extract each finished day (118_adsb_probe.py)')
    ap.add_argument('--extract-workers', type=int, default=2)
    ap.add_argument('--dry-run', action='store_true', help='list parts and sizes, download nothing')
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--variant', default=VARIANT,
                    help="release variant: prod-0 (default), staging-0, or mlatonly-0")
    ap.add_argument('--preferred', action='store_true',
                    help="each day's release from adsb.lol's PREFERRED_RELEASES.txt (tag and repo), else --variant")
    ap.add_argument('--replace', action='store_true',
                    help='re-extract days that already have points (old points + per-day caches kept in <out>/replaced)')
    a = ap.parse_args()
    OUT, VARIANT, REPLACE = Path(a.out), a.variant, a.replace
    if a.preferred:
        PREFERRED.update(load_preferred())
        log(f'preferred releases: {len(PREFERRED)} days listed by adsb.lol')
    raw = OUT / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    days = expand(a.dates)
    log(f'start: {len(days)} days ({days[0]} ... {days[-1]}), workers {a.workers}, extract {a.extract}, out {OUT}')
    if sys.platform == 'win32':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # keep the system awake
    ex_pool = ThreadPoolExecutor(a.extract_workers) if a.extract else None
    ex_jobs = []

    def job(day):
        ok = download_day(day, raw, a.dry_run)
        if ok and ex_pool is not None:
            ex_jobs.append(ex_pool.submit(extract_day, day))
        return ok

    try:
        with ThreadPoolExecutor(a.workers) as pool:
            results = list(pool.map(job, days))
        if ex_pool is not None:
            ex_pool.shutdown(wait=True)
    finally:
        if sys.platform == 'win32':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    ok = sum(bool(r) for r in results)
    log(f'finished: {ok} of {len(days)} days complete, {STATS["bytes"] / 1e9:.1f} GB downloaded this run')


if __name__ == '__main__':
    main()
