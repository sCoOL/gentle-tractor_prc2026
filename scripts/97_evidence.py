"""Promotion evidence for a strict six-pair OOF candidate against a reference strict OOF.

  py -3.14 scripts/97_evidence.py --candidate runs/oof/<cand>.parquet --reference runs/oof/<ref>.parquet \
      --reference-name A20 --summary "..." --out runs/evidence_<name>.json

Both files need MVT_ID_mvt, fold, y and pred on the same 2,085,047 rows. ``passes`` is a noise guard, not an
acceptance rule: pooled RMSE at least 0.3 s better (about the refit noise of the full pipeline) and no fold more than
1.0 s worse. The earlier agent-made "at least 1.0 s pooled" gate is reported as ``passes_1s`` for reference.
"""
import argparse
import json

import numpy as np
import pandas as pd

import common

parser = argparse.ArgumentParser()
parser.add_argument('--candidate', required=True)
parser.add_argument('--reference', required=True)
parser.add_argument('--reference-name', required=True)
parser.add_argument('--summary', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--pred-col', default='pred')
args = parser.parse_args()

c = pd.read_parquet(args.candidate, columns=['MVT_ID_mvt', 'APT', 'fold', 'fpl_matched', 'y', args.pred_col]).rename(columns={args.pred_col: 'pred'})
r = pd.read_parquet(args.reference, columns=['MVT_ID_mvt', 'y', 'pred']).rename(columns={'pred': 'ref', 'y': 'y_ref'})
m = c.merge(r, on='MVT_ID_mvt', how='inner', validate='one_to_one')
if len(m) != len(c) or len(m) != len(r) or not np.allclose(m.y.values, m.y_ref.values, equal_nan=True):
    raise SystemExit('candidate and reference rows or labels differ')
rm = lambda p, y: common.rmse(p, y)
folds = {f: {'ref': rm(g.ref, g.y), 'new': rm(g.pred, g.y)} for f, g in m.groupby('fold')}
for v in folds.values():
    v['delta'] = v['new'] - v['ref']
pooled_new, pooled_ref = rm(m.pred, m.y), rm(m.ref, m.y)
worst = max(v['delta'] for v in folds.values())
ev = {'candidate': args.candidate, 'reference': args.reference_name, 'reference_oof': args.reference, 'rows': int(len(m)),
      'pooled': pooled_new, 'pooled_ref': pooled_ref, 'delta_vs_ref': pooled_new - pooled_ref, 'worst_fold_vs_ref': worst,
      'fold_A': folds['A']['new'], 'fold_B': folds['B']['new'], 'folds': folds,
      'airport_delta': {a: rm(g.pred, g.y) - rm(g.ref, g.y) for a, g in m.groupby('APT')},
      'lane_delta': {('matched' if k == 1 else 'unmatched'): rm(g.pred, g.y) - rm(g.ref, g.y) for k, g in m.groupby('fpl_matched')},
      'body_delta': rm(m.pred[m.y <= 3600], m.y[m.y <= 3600]) - rm(m.ref[m.y <= 3600], m.y[m.y <= 3600]),
      'rule': 'noise guard: pooled delta <= -0.3 s and no fold delta > +1.0 s', 'summary': args.summary}
ev['passes'] = bool(ev['delta_vs_ref'] <= -0.3 and worst <= 1.0)
ev['passes_1s'] = bool(ev['delta_vs_ref'] <= -1.0 and worst <= 1.0)
open(args.out, 'w', encoding='utf-8').write(json.dumps(ev, indent=1))
print(f"{args.summary}: pooled {pooled_ref:.4f} -> {pooled_new:.4f} ({ev['delta_vs_ref']:+.4f}); worst fold {worst:+.4f}; "
      f"body {ev['body_delta']:+.3f}; lanes { {k: round(v, 3) for k, v in ev['lane_delta'].items()}}; passes={ev['passes']}")
print('  folds', {f: round(v['delta'], 3) for f, v in folds.items()})
print('  airports', {a: round(v, 3) for a, v in ev['airport_delta'].items()})
