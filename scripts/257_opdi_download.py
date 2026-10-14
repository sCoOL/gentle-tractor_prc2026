"""Download OPDI v002 (EUROCONTROL PRU Open Performance Data Initiative) parquet files into PRC_OPDI_RAW (scripts/prc_config.py).

  py -3.14 scripts/257_opdi_download.py                 # 2025-01 .. 2026-07, all three datasets
  py -3.14 scripts/257_opdi_download.py --dry-run       # list the files and their sizes only
  py -3.14 scripts/257_opdi_download.py --datasets flight_events --workers 3

Source (OPDI use for the challenge confirmed 2026-10-06): https://www.opdi.aero/data, files under
https://www.eurocontrol.int/performance/data/download/OPDI/v002/
  flight_list/flight_list_YYYYMM.parquet                         one per month
  flight_events/flight_events_YYYYMMDD_YYYYMMDD.parquet          10-day files from 2022-01-01 (end date excluded)
  measurements/measurements_YYYYMMDD_YYYYMMDD.parquet            same 10-day grid
Methodology and code: https://github.com/euctrl-pru/opdi (built from OpenSky Network state vectors).

Safe to stop and re-run: a file counts as done only when <name>.parquet exists, has the server's Content-Length and its
parquet footer reads; an interrupted download continues from its .part size with an HTTP Range request. Every finished
file is recorded in <out>/manifest.json (url, bytes, Last-Modified, rows, row groups). The server sometimes answers 416
to range requests (seen 2026-10-06); a .part that cannot be resumed is restarted from zero.
"""
import argparse
import ctypes
import datetime as dt
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

BASE = "https://www.eurocontrol.int/performance/data/download/OPDI/v002/"
GRID0 = dt.date(2022, 1, 1)
LOCK = threading.Lock()


def file_list(start, end, datasets):
    """(dataset, name, url) for every file overlapping [start, end]."""
    out = []
    if "flight_list" in datasets:
        y, m = start.year, start.month
        while (y, m) <= (end.year, end.month):
            name = f"flight_list_{y}{m:02d}.parquet"
            out.append(("flight_list", name, f"{BASE}flight_list/{name}"))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    for ds in ("flight_events", "measurements"):
        if ds not in datasets:
            continue
        d = GRID0
        while d <= end:
            e = d + dt.timedelta(days=10)
            if e > start:
                name = f"{ds}_{d:%Y%m%d}_{e:%Y%m%d}.parquet"
                out.append((ds, name, f"{BASE}{ds}/{name}"))
            d = e
    return out


def head(url):
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as h:
                return int(h.headers["Content-Length"]), h.headers.get("Last-Modified")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, None
            err = e
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(3 + 5 * k)
    raise IOError(f"HEAD failed {url}: {err}")


def log(msg, fh):
    line = f"{dt.datetime.now():%H:%M:%S} {msg}"
    with LOCK:
        print(line, flush=True)
        fh.write(line + "\n"); fh.flush()


def fetch(ds, name, url, out, fh, manifest):
    dest = out / ds / name
    part = dest.with_suffix(".part")
    size, lastmod = head(url)
    if size is None:
        log(f"404 {name}", fh)
        return name, "missing"
    if dest.exists() and dest.stat().st_size == size:
        return name, "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(8):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink(); have = 0
        if have < size:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    if have and r.status != 206:  # server ignored the range: start over
                        have = 0
                    with open(part, "ab" if have else "wb") as f:
                        while True:
                            buf = r.read(4 << 20)
                            if not buf:
                                break
                            f.write(buf)
            except urllib.error.HTTPError as e:
                log(f"retry {attempt} {name} at {have}: HTTP {e.code}", fh)
                if e.code == 416:
                    part.unlink(missing_ok=True)
                time.sleep(5 + 10 * attempt); continue
            except Exception as e:  # noqa: BLE001
                log(f"retry {attempt} {name} at {have}: {e}", fh)
                time.sleep(5 + 10 * attempt); continue
        if part.stat().st_size != size:
            continue
        try:
            md = pq.read_metadata(part)
        except Exception as e:  # noqa: BLE001
            log(f"bad parquet {name} ({e}); downloading again", fh)
            part.unlink(); continue
        part.replace(dest)
        with LOCK:
            manifest[name] = dict(dataset=ds, url=url, bytes=size, last_modified=lastmod, rows=md.num_rows,
                                  row_groups=md.num_row_groups, downloaded=dt.datetime.now().isoformat(timespec="seconds"))
            (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
        log(f"done {name} {size / 1e6:.0f} MB rows {md.num_rows:,}", fh)
        return name, "done"
    log(f"FAILED {name}", fh)
    return name, "failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--datasets", nargs="+", default=["flight_list", "flight_events", "measurements"])
    ap.add_argument("--out", default=str(prc_config.OPDI_RAW))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    files = file_list(dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end), a.datasets)
    if a.dry_run:
        tot = 0
        for ds, name, url in files:
            size, lm = head(url)
            tot += size or 0
            print(f"{name:52s} {(size or 0) / 1e6:9.1f} MB  {lm}")
        print(f"{len(files)} files, {tot / 1e9:.1f} GB")
        return
    out.mkdir(parents=True, exist_ok=True)
    mf = out / "manifest.json"
    manifest = json.loads(mf.read_text()) if mf.exists() else {}
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # keep Windows awake while running
    with open(out / "download.log", "a", encoding="utf-8") as fh:
        log(f"start {len(files)} files -> {out}", fh)
        res = {}
        with ThreadPoolExecutor(a.workers) as ex:
            futs = [ex.submit(fetch, ds, name, url, out, fh, manifest) for ds, name, url in files]
            for f in as_completed(futs):
                name, st = f.result()
                res[st] = res.get(st, 0) + 1
        log(f"EXIT {res}", fh)
    sys.exit(1 if res.get("failed") else 0)


if __name__ == "__main__":
    main()
