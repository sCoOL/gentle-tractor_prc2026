"""features v7 = features v5 with NM columns backfilled for unmatched departures (scripts/nm_backfill.py).

  py -3.14 scripts/41_build_v7_backfill.py

Writes data/features_v7_{train,rank}.parquet, data/features_v7.json (new group G13 = nm_backfilled) and
the audit tables data/nm_backfill_links_{train,rank}.parquet.
"""
import glob
import json
import time

import numpy as np
import pandas as pd

import common
import nm_backfill as nb
from common import DATA

t0 = time.time()
need = ['MVT_ID_mvt', 'PHASE_mvt', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt'] \
    + nb.NM_TIMES + list(nb.NM_CATS)
raw_train = pd.concat([pd.read_parquet(f, columns=need) for f in sorted(glob.glob(str(DATA / 'training_2025-*.parquet')))], ignore_index=True)
raw_rank = common.rank_with_leadin(need)   # continuity rule: + Dec 2025 lead-in
rank_ids = set(pd.read_parquet(DATA / 'ranking.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt)
links = {'train': nb.link(raw_train), 'rank': nb.link(raw_rank)}
links['rank'] = links['rank'][links['rank'].MVT_ID_mvt.isin(rank_ids)].reset_index(drop=True)   # continuity rule: rank departures only
for k, v in links.items():
    v.to_parquet(DATA / f'nm_backfill_links_{k}.parquet', index=False)
print(f"links train {len(links['train'])} rank {len(links['rank'])} [{time.time() - t0:.0f}s]", flush=True)

manifest = common.read_manifest(5)
out = {}
for k, raw in (('train', raw_train), ('rank', raw_rank)):
    f = pd.read_parquet(DATA / f'features_v5_{k}.parquet')
    f = nb.backfill(f, raw, links[k])
    f.to_parquet(DATA / f'features_v7_{k}.parquet')
    out[k] = f
    print(f"features_v7_{k}: {len(f)} rows, backfilled {int(f.nm_backfilled.sum())} [{time.time() - t0:.0f}s]", flush=True)
assert len(out['train']) == 2085047 and len(out['rank']) == 670790  # final-file change: final ranking file (was 344841)

tr = out['train']
bf = tr[tr.nm_backfilled.values == 1]
copy = bf.fallback.values == 1
gap = np.abs(bf.TAXITIME_SEC_mvt.values - bf.atot_minus_aobt3.values)
audit = {'train_backfilled': int(len(bf)), 'rank_backfilled': int(out['rank'].nm_backfilled.sum()),
         'train_backfilled_by_apt': bf.APT.value_counts().to_dict(),
         'rank_backfilled_by_apt': out['rank'][out['rank'].nm_backfilled.values == 1].APT.value_counts().to_dict(),
         'train_copy_share': float(copy.mean()), 'train_noncopy_label_within_300s_of_backfilled_taxi': float((gap[~copy] <= 300).mean())}
groups = {k: list(v) for k, v in manifest['groups'].items()}
groups['G13'] = ['nm_backfilled']
columns = dict(manifest['columns'])
columns['nm_backfilled'] = {'group': 'G13', 'source': 'nm_backfill_v7', 'rank_available': True, 'dtype': 'int'}
m = {**manifest, 'version': 7, 'parent_version': 5, 'built_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
     'columns': columns, 'groups': groups,
     'notes': list(manifest.get('notes', [])) + [
         'v7 backfills NM take-off deltas and NM categoricals of unmatched departures from the same flight\'s arrival '
         'record at another study airport (nm_backfill.py); fpl_matched stays 0, nm_backfilled flags the rows.'],
     'nm_backfill_audit': audit}
common.manifest_path(7).write_text(json.dumps(m, indent=1, default=str), encoding='utf-8')
print(json.dumps(audit, indent=1, default=str))
print(f'features v7 written [{time.time() - t0:.0f}s]')
