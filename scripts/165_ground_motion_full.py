"""Freeze and extract sustained-motion features from all existing cached dates."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
import os
from pathlib import Path

import pandas as pd

m = importlib.import_module('164_ground_motion_candidates')
OUT = m.common.RUNS/'v49_motion_full'
if os.environ.get('PRC_MOTION_OUT'):   # final-file change: spawned pool workers re-import this module, so --out travels by env
    OUT = Path(os.environ['PRC_MOTION_OUT'])
# final-file change: --out / --since let the final-file run write a NEW folder (the protocol digest of the old one
# names the old day set) and extract only the 2026 days (the A57 model is re-applied, never refitted, so the 2025
# training days are not needed).
SINCE = None


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def day(date):
    m.OUT = OUT
    f = m.extract(date)
    return {'date': date, 'rows': len(f), 'sha256': digest(OUT/f'features_{date}.parquet')}


def run(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    dates = [d for d in m.e.days() if (m.e.OUT/'events'/f'events_{d}.parquet').exists() and (SINCE is None or d >= SINCE)]
    paths = [Path(m.__file__), Path(m.e.__file__), Path(__file__)]
    inputs = [p for d in dates for p in [m.e.P118.points_path(d), m.e.OUT/'events'/f'events_{d}.parquet']]   # final-file change: raw points path
    protocol = {'dates': dates, 'sources': {str(p): digest(p) for p in paths},
                'raw_size_mtime_inventory': {str(p): {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in inputs},
                'features': m.COLUMNS, 'label_use': 'None. Original event linkage and observed points only.',
                'submission_qualified': False}
    path = OUT/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Source/input changed; use a new run directory'
    else:
        path.write_text(json.dumps(protocol, indent=2))
    for p in paths:
        (OUT/p.name).write_bytes(p.read_bytes())
    inventory = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(day, dates, chunksize=1):
            inventory.append(row)
            print(f'{len(inventory)}/{len(dates)} {row["date"]} {row["rows"]} rows', flush=True)
    f = pd.concat([pd.read_parquet(OUT/f'features_{d}.parquet') for d in dates], ignore_index=True)
    assert f.MVT_ID_mvt.is_unique
    f.to_parquet(OUT/'features.parquet', index=False)
    summary = {'dates': len(dates), 'rows': len(f), 'train_rows': int(f.date.str.startswith('2025').sum()),
               'rank_rows': int(f.date.str.startswith('2026').sum()), 'features_sha256': digest(OUT/'features.parquet'),
               'inventory': inventory, 'submission_qualified': False}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    print('COMPLETE', summary['train_rows'], summary['rank_rows'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--out', default=None)      # final-file change
    parser.add_argument('--since', default=None)    # final-file change: first date to extract (YYYY-MM-DD)
    a = parser.parse_args()
    if a.out:
        OUT = Path(a.out)
        os.environ['PRC_MOTION_OUT'] = str(OUT)
    SINCE = a.since
    run(a.workers)
