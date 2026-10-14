"""features v5 = features v4 + G11 operations context (scripts/ops_context.py).

  py -3.14 scripts/38_build_v5_context.py

G11 also carries ``st_turn_in_is_copy``: the inbound turnaround arrival (G10) in-blocked exactly at
its schedule (|st_turn_in_delay| <= 10 s). At LIRF that raises the departure's schedule-default
rate from 15.4 % to 38.1 % (label-artifact audit 2026-09-17).
"""
import glob
import json
import time

import numpy as np
import pandas as pd

import common
import ops_context
from common import DATA

t0 = time.time()
raw_train = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(DATA / 'training_2025-*.parquet')))], ignore_index=True)
raw_rank = common.rank_with_leadin()   # continuity rule: + Dec 2025 lead-in (context only)
g11_train = ops_context.build(raw_train)
print(f'G11 train built: {len(g11_train)} [{time.time() - t0:.0f}s]', flush=True)
g11_rank = ops_context.build(raw_rank)
print(f'G11 rank built: {len(g11_rank)} [{time.time() - t0:.0f}s]', flush=True)

manifest = common.read_manifest(4)
train = pd.read_parquet(DATA / 'features_v4_train.parquet')
rank = pd.read_parquet(DATA / 'features_v4_rank.parquet')
train = train.merge(g11_train, on='MVT_ID_mvt', how='left', validate='one_to_one')
rank = rank.merge(g11_rank, on='MVT_ID_mvt', how='left', validate='one_to_one')
for d in (train, rank):
    d['st_turn_in_is_copy'] = np.where(np.isfinite(d.st_turn_in_delay.values), (np.abs(d.st_turn_in_delay.values) <= 10).astype(float), np.nan)
assert len(train) == 2085047 and len(rank) == 670790  # final-file change: final ranking file (was 344841)
cols = list(ops_context.COLUMNS) + ['st_turn_in_is_copy']
train.to_parquet(DATA / 'features_v5_train.parquet')
rank.to_parquet(DATA / 'features_v5_rank.parquet')

groups = {k: list(v) for k, v in manifest['groups'].items()}
groups['G11'] = cols
columns = dict(manifest['columns'])
for c in cols:
    columns[c] = {'group': 'G11', 'source': 'ops_context_v5', 'rank_available': True, 'dtype': str(rank[c].dtype)}
parity = {}
for c in cols:
    a, b = train[c], rank[c]
    parity[c] = {'train_null': round(float(a.isna().mean()), 4), 'rank_null': round(float(b.isna().mean()), 4),
                 'train_p50': float(np.nanmedian(a)), 'rank_p50': float(np.nanmedian(b)),
                 'train_jan_jul_p50': float(np.nanmedian(a[train.month.isin([1, 7]).values]))}
out = {**manifest, 'version': 5, 'parent_version': 4, 'built_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
       'columns': columns, 'groups': groups,
       'notes': list(manifest.get('notes', [])) + [
           'G11 operations context uses other departures\' take-off and NM AOBT_3 times and complete arrival records '
           '(landing, in-block, schedule); a departure\'s own off-block is never read.'],
       'g11_parity': parity}
common.manifest_path(5).write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps(parity, indent=1))
print(f'features v5 written [{time.time() - t0:.0f}s]')
