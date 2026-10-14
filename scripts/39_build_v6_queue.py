"""features v6 = features v5 + G12 queue and pushback-cohort context (scripts/queue_context.py).

  py -3.14 scripts/39_build_v6_queue.py

G12 also carries ``qx_own_minus_rwy_c60``: the departure's own take-off - AOBT_3 minus the G11 runway
median around its take-off (a tree cannot subtract two columns).
"""
import glob
import json
import time

import numpy as np
import pandas as pd

import common
import queue_context
from common import DATA

t0 = time.time()
raw_train = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(DATA / 'training_2025-*.parquet')))], ignore_index=True)
raw_rank = common.rank_with_leadin()   # continuity rule: + Dec 2025 lead-in (context only)
g_train = queue_context.build(raw_train)
print(f'G12 train built: {len(g_train)} [{time.time() - t0:.0f}s]', flush=True)
g_rank = queue_context.build(raw_rank)
print(f'G12 rank built: {len(g_rank)} [{time.time() - t0:.0f}s]', flush=True)

manifest = common.read_manifest(5)
train = pd.read_parquet(DATA / 'features_v5_train.parquet')
rank = pd.read_parquet(DATA / 'features_v5_rank.parquet')
train = train.merge(g_train, on='MVT_ID_mvt', how='left', validate='one_to_one')
rank = rank.merge(g_rank, on='MVT_ID_mvt', how='left', validate='one_to_one')
for d in (train, rank):
    d['qx_own_minus_rwy_c60'] = np.clip(d.atot_minus_aobt3.values, 0, 5400) - d.cx_dep_a3taxi_med_rwy_c60.values
assert len(train) == 2085047 and len(rank) == 670790  # final-file change: final ranking file (was 344841)
cols = list(queue_context.COLUMNS) + ['qx_own_minus_rwy_c60']
train.to_parquet(DATA / 'features_v6_train.parquet')
rank.to_parquet(DATA / 'features_v6_rank.parquet')

groups = {k: list(v) for k, v in manifest['groups'].items()}
groups['G12'] = cols
columns = dict(manifest['columns'])
for c in cols:
    columns[c] = {'group': 'G12', 'source': 'queue_context_v6', 'rank_available': True, 'dtype': str(rank[c].dtype)}
parity = {}
for c in cols:
    a, b = train[c], rank[c]
    parity[c] = {'train_null': round(float(a.isna().mean()), 4), 'rank_null': round(float(b.isna().mean()), 4),
                 'train_p50': float(np.nanmedian(a)), 'rank_p50': float(np.nanmedian(b)),
                 'train_jan_jul_p50': float(np.nanmedian(a[train.month.isin([1, 7]).values]))}
out = {**manifest, 'version': 6, 'parent_version': 5, 'built_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
       'columns': columns, 'groups': groups,
       'notes': list(manifest.get('notes', [])) + [
           'G12 queue/pushback-cohort context uses other departures\' take-off and NM AOBT_3 times and arrival records; '
           'a departure\'s own off-block is never read.'],
       'g12_parity': parity}
common.manifest_path(6).write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps(parity, indent=1))
print(f'features v6 written [{time.time() - t0:.0f}s]')
