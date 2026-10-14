"""How much of the unmatched twins' seed noise reaches v45 (sizing for v73 = more twin seeds).

Composes v45 from the variant's member caches once per single twin seed and once with all of them, then reports the
spread of pred_v45 across the single-seed versions (= per-seed noise that reaches v45). A K-seed mean carries 1/K of
that variance, so going from 3 to K seeds removes sum(var) * (1/3 - 1/K).

  py -3.14 tools/members/twin_seed_noise.py [--base-dir runs/models/final_A] [--seeds 41 42 43]
Reads only; writes logs/members/twin_seed_noise_<variant dir>.json.
"""
from __future__ import annotations

import mcommon as mc  # (threads, sys.path)

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mcommon import REB, DATA, say

rv = importlib.import_module('recompose_v45')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', default=str(REB / 'runs/models/final_A'))
    ap.add_argument('--seeds', nargs='+', type=int, default=[41, 42, 43])
    a = ap.parse_args()
    S0 = rv.rebuild_sources('new', str(mc.prc_config.ADSB_DIR), a.base_dir)
    tw = pd.read_parquet(S0['twins'])
    preds = {}
    for s in a.seeds:
        S = dict(S0)
        S['twins'] = tw[['MVT_ID_mvt', f'blind_s{s}', f'ab_s{s}']].copy()
        full, _, missing = rv.compose_v45(S)
        assert not missing, missing
        preds[s] = full.set_index('MVT_ID_mvt').pred_v45
        say(f'seed {s} composed')
    full, _, _ = rv.compose_v45(dict(S0))
    allp = full.set_index('MVT_ID_mvt').pred_v45
    P = pd.concat(preds, axis=1)
    var = P.var(axis=1, ddof=1)
    base = pd.read_parquet(DATA / 'base_rank.parquet', columns=['MVT_ID_mvt', 'APT', 'month', 'fpl_matched']).set_index('MVT_ID_mvt')
    d = base.join(pd.DataFrame({'var': var, 'mean_single': P.mean(axis=1), 'all': allp}))
    moved = d['var'] > 0
    k = len(a.seeds)
    rep = {'seeds': a.seeds, 'rows': int(len(d)), 'rows_with_seed_spread': int(moved.sum()),
           'per_seed_sd_rms_on_moved_rows': float(np.sqrt(d.loc[moved, 'var'].mean())),
           'linearity_rms_all_minus_mean_single': float(np.sqrt(((d['all'] - d.mean_single) ** 2).mean())),
           f'noise_in_{k}seed_mean_M': float(d['var'].sum() / k / 1e6),
           'removed_M_by_K': {K: float(d['var'].sum() * (1 / k - 1 / K) / 1e6) for K in (6, 9, 12, 15, 20)},
           'jan_jul_noise_M': float(d.loc[d.month.isin([1, 7]), 'var'].sum() / k / 1e6),
           'feb_jun_noise_M': float(d.loc[d.month.isin([2, 6]), 'var'].sum() / k / 1e6),
           'by_apt_noise_M': (d.groupby('APT')['var'].sum() / k / 1e6).round(3).to_dict(),
           'by_lane_noise_M': (d.groupby('fpl_matched')['var'].sum() / k / 1e6).round(3).to_dict()}
    out = REB / 'logs' / 'members' / f'twin_seed_noise_{Path(a.base_dir).name}.json'
    out.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    say(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
