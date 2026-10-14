"""features v4 = features v3 + G10 same-stand timeline (scripts/stand_timeline.py).

  py -3.14 scripts/37_build_v4_stand.py
"""
import glob
import json
import time

import numpy as np
import pandas as pd

import common
import stand_timeline
from common import DATA

t0 = time.time()
raw_train = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(DATA / 'training_2025-*.parquet')))], ignore_index=True)
raw_rank = common.rank_with_leadin()   # continuity rule: + Dec 2025 lead-in (context only)
g10_train = stand_timeline.build(raw_train)
g10_rank = stand_timeline.build(raw_rank)
print(f'G10 built: train {len(g10_train)} rank {len(g10_rank)} [{time.time() - t0:.0f}s]', flush=True)

manifest = common.read_manifest(3)
train = pd.read_parquet(DATA / 'features_v3_train.parquet')
rank = pd.read_parquet(DATA / 'features_v3_rank.parquet')
train = train.merge(g10_train, on='MVT_ID_mvt', how='left', validate='one_to_one')
rank = rank.merge(g10_rank, on='MVT_ID_mvt', how='left', validate='one_to_one')
assert len(train) == 2085047 and len(rank) == 670790  # final-file change: final ranking file (was 344841)
train.to_parquet(DATA / 'features_v4_train.parquet')
rank.to_parquet(DATA / 'features_v4_rank.parquet')

groups = {k: list(v) for k, v in manifest['groups'].items()}
groups['G10'] = list(stand_timeline.COLUMNS)
columns = dict(manifest['columns'])
for c in stand_timeline.COLUMNS:
    columns[c] = {'group': 'G10', 'source': 'stand_timeline_v4', 'rank_available': True, 'dtype': str(rank[c].dtype)}
parity = {}
for c in stand_timeline.COLUMNS:
    a, b = train[c], rank[c]
    parity[c] = {'train_null': round(float(a.isna().mean()), 4), 'rank_null': round(float(b.isna().mean()), 4),
                 'train_p50': float(np.nanmedian(a)), 'rank_p50': float(np.nanmedian(b))}
out = {**manifest, 'version': 4, 'parent_version': 3, 'built_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
       'columns': columns, 'groups': groups,
       'notes': list(manifest.get('notes', [])) + [
           'G10 same-stand timeline uses arrival in-block times and other departures\' take-off times only; '
           'a departure\'s own off-block is never read. Gap features clipped at 12 h.'],
       'g10_parity': parity}
common.manifest_path(4).write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps(parity, indent=1))
print(f'features v4 written [{time.time() - t0:.0f}s]')
