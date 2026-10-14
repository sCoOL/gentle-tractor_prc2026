"""Shared helpers of the v45 member tools (docs/METHOD.md, "v45 member composition").

Import this FIRST in every members tool: it caps the BLAS/OpenMP threads at 6 before numpy / lightgbm / torch load,
puts the repository's scripts on sys.path, and offers
  * point_adsb(dir)       redirect adsb_context's v1 tables at call time (no edit of scripts/adsb_context.py)
  * old_rank_source()     context manager: nbr_set._load('rank') reads the earlier 344,841-row Jan/Jul export (PRC_DATA_OLD)
  * compare(...)          reproduction statistics of a refitted prediction against the reference one
Nothing here writes outside PRC_WORK.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '6')

import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import prc_config  # noqa: E402

REB = prc_config.WORK                   # working root (PRC_WORK)
DATA = prc_config.DATA
OLD = prc_config.DATA_OLD              # rank-side tables of the earlier Jan/Jul export (reproduction checks only)
REF = prc_config.REF_MAIN              # frozen packages, fixed weights and reference member predictions
REFM = REF / 'models'
SNAP_ADSB = prc_config.ADSB_SNAPSHOT   # the v1 ADS-B tables A38-A51 were built with (variant A training tables)
OUT = prc_config.RUNS / 'models'
REPRO = prc_config.RUNS / 'members_repro'
N_FINAL = 670790
N_OLD = 344841

_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:
    sys.path.insert(0, _here)
assert os.environ['OMP_NUM_THREADS'] == '6', 'members tools run with 6 threads'


def stamp():
    return time.strftime('%H:%M:%S')


def say(*a):
    print(f'[{stamp()}]', *a, flush=True)


def point_adsb(folder):
    """Make adsb_context read <folder>/adsb_features.parquet etc. (call-time defaults are rebound, cache cleared)."""
    import adsb_context as ac
    folder = Path(folder)
    ac.PATH = folder / 'adsb_features.parquet'
    ac.NBR_PATH = folder / 'adsb_nbr.parquet'
    ac.GEOM_PATH = folder / 'adsb_geom.parquet'
    ac.STOPS_PATH = folder / 'adsb_stops.parquet'
    ac.table.__defaults__ = (ac.PATH,)
    ac.nbr_table.__defaults__ = (ac.NBR_PATH,)
    ac.geom_table.__defaults__ = (ac.GEOM_PATH,)
    ac.stops_table.__defaults__ = (ac.STOPS_PATH,)
    ac._CACHE.clear()
    return ac


@contextlib.contextmanager
def old_rank_source():
    """Inside the block, nbr_set._load('rank') returns the earlier OLD export (PRC_DATA_OLD); 'train' is untouched."""
    import nbr_set
    orig = nbr_set._load

    def _load(which):
        if which != 'rank':
            return orig(which)
        dep = pd.read_parquet(OLD / 'base_rank.parquet', columns=nbr_set._DEP_COLS)
        arr = pd.read_parquet(OLD / 'arr_rank.parquet', columns=nbr_set._ARR_COLS)
        rs = pd.read_parquet(OLD / 'ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt', 'SCHED_TIME_UTC_mvt', 'WK_TBL_CAT_flt'])
        rs = rs[rs.PHASE_mvt == 'ARR'].drop(columns='PHASE_mvt')
        arr = arr.merge(rs, on='MVT_ID_mvt', how='left', validate='one_to_one')
        return dep, arr
    nbr_set._load = _load
    try:
        yield
    finally:
        nbr_set._load = orig


def load_rank(version, old=False, columns=None):
    path = (OLD if old else DATA) / f'features_v{version}_rank.parquet'
    d = pd.read_parquet(path, columns=columns)
    assert len(d) == (N_OLD if old else N_FINAL), (path, len(d))
    return d


def compare(new_ids, new_vals, ref_ids, ref_vals, label=''):
    """Statistics of new - ref on the ids both have and both finite."""
    a = pd.Series(np.asarray(new_vals, float), index=np.asarray(new_ids))
    b = pd.Series(np.asarray(ref_vals, float), index=np.asarray(ref_ids))
    common_ids = b.index.intersection(a.index)
    x = a.reindex(common_ids).to_numpy(); y = b.reindex(common_ids).to_numpy()
    both = np.isfinite(x) & np.isfinite(y)
    nan_mismatch = int((np.isfinite(x) != np.isfinite(y)).sum())
    d = x[both] - y[both]
    out = {'label': label, 'n_common': int(len(common_ids)), 'n_both_finite': int(both.sum()), 'nan_mismatch': nan_mismatch}
    if both.any():
        ad = np.abs(d)
        out.update(max_abs=float(ad.max()), rms=float(np.sqrt(np.mean(d ** 2))), mean=float(d.mean()),
                   p50_abs=float(np.median(ad)), p99_abs=float(np.quantile(ad, 0.99)),
                   ref_sd=float(np.std(y[both])),
                   corr=float(np.corrcoef(x[both], y[both])[0, 1]) if np.std(x[both]) > 0 and np.std(y[both]) > 0 else float('nan'),
                   share_gt_1s=float((ad > 1).mean()), share_gt_10s=float((ad > 10).mean()))
    return out


def save_report(name, obj):
    REPRO.mkdir(parents=True, exist_ok=True)
    p = REPRO / f'{name}.json'
    p.write_text(json.dumps(obj, indent=1, default=float), encoding='utf-8')
    return p


def fmt(c):
    if 'max_abs' not in c:
        return f"{c['label']}: no finite overlap ({c['n_common']} common ids)"
    return (f"{c['label']}: n {c['n_both_finite']:,}, max {c['max_abs']:.4g}, rms {c['rms']:.4g}, p99 {c['p99_abs']:.4g}, "
            f"corr {c['corr']:.6f}, nan-mismatch {c['nan_mismatch']}")
