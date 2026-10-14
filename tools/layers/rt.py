"""Runtime helpers of the final chain (tools/final_chain.py, tools/layers/build_frames.py): thread caps, memory gate,
train-row splicing, small IO helpers.

Import this module BEFORE numpy / lightgbm / duckdb so the OpenMP caps apply:
  - env OMP / OpenBLAS / MKL / NumExpr threads = 4 (<= 4 threads per process);
  - lightgbm.train and the LGBM sklearn estimators: num_threads / n_jobs capped at 4 (the layer scripts ask for 8-18);
  - duckdb.connect(): SET threads = 4.
Thread counts never change a layer's parameters otherwise; LightGBM float summation order can differ with the thread
count, so a rerun on another machine agrees to float-rounding noise, not bit for bit.
"""
import os

THREADS = 4
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS'):
    os.environ[_k] = str(THREADS)

import ctypes  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
import prc_config  # noqa: E402

REB = prc_config.WORK                         # working root (PRC_WORK): data/, runs/, logs/
REF = prc_config.REF_LAYERS                   # frozen 2025-side frames and models (docs/FROZEN_ARTEFACTS.md)
LOGD = REB / 'logs' / 'layers'


def _cap_libraries():
    import lightgbm as lgb
    if not getattr(lgb.train, '_rebuild_capped', False):
        _train = lgb.train

        def train(params, *a, **k):
            p = dict(params)
            for key in ('num_threads', 'num_thread', 'nthread', 'nthreads', 'n_jobs'):
                if key in p:
                    n = int(p[key])
                    p[key] = THREADS if (n <= 0 or n > THREADS) else n
            if not any(key in p for key in ('num_threads', 'num_thread', 'nthread', 'nthreads', 'n_jobs')):
                p['num_threads'] = THREADS
            return _train(p, *a, **k)
        train._rebuild_capped = True
        lgb.train = train
        # The sklearn estimators (274's LGBMClassifier) pass no n_jobs: LightGBM then uses the OpenMP default, which the
        # OMP_NUM_THREADS=4 above caps.
    try:
        import duckdb
        if not getattr(duckdb.connect, '_rebuild_capped', False):
            _connect = duckdb.connect

            def connect(*a, **k):
                con = _connect(*a, **k)
                con.execute(f'SET threads = {THREADS}')
                return con
            connect._rebuild_capped = True
            duckdb.connect = connect
    except ImportError:
        pass


_cap_libraries()


class _MS(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def _meminfo_gb(key: str) -> float:
    """Linux fallback: a /proc/meminfo field in GB (inf when unavailable, so the gate never blocks)."""
    try:
        for line in open('/proc/meminfo'):
            if line.startswith(key + ':'):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return float('inf')


def free_gb() -> float:
    if not hasattr(ctypes, 'windll'):
        return _meminfo_gb('MemAvailable')
    m = _MS()
    m.dwLength = ctypes.sizeof(_MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


def commit_gb() -> tuple:
    if not hasattr(ctypes, 'windll'):
        return float('nan'), float('nan')
    m = _MS()
    m.dwLength = ctypes.sizeof(_MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return (m.ullTotalPageFile - m.ullAvailPageFile) / 2 ** 30, m.ullTotalPageFile / 2 ** 30


def wait_mem(need_gb: float = 8.0, what: str = '', poll_s: int = 60, max_wait_s: int = 6 * 3600):
    """Block until free physical memory >= need_gb (checked before every heavy step)."""
    t0 = time.time()
    while True:
        f = free_gb()
        if f >= need_gb:
            say(f'memory gate ok for {what}: {f:.1f} GB free (commit {commit_gb()[0]:.1f}/{commit_gb()[1]:.1f} GB)')
            return f
        if time.time() - t0 > max_wait_s:
            raise SystemExit(f'memory gate: still {f:.1f} GB free after {max_wait_s} s ({what})')
        say(f'memory gate waiting for {what}: {f:.1f} GB free < {need_gb} GB')
        time.sleep(poll_s)


def say(msg: str):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    LOGD.mkdir(parents=True, exist_ok=True)
    with open(LOGD / 'run.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def record(name: str, info: dict, path: Path | None = None):
    """Append/replace one entry of a json record (default logs/layers/record.json)."""
    path = path or (LOGD / 'record.json')
    d = json.loads(path.read_text()) if path.exists() else {}
    d[name] = info
    path.write_text(json.dumps(d, indent=1, default=str))


def run_script(args: list, env_extra: dict | None = None, cwd: Path | None = None, log: str | None = None) -> int:
    """Run a python script as a subprocess with the thread caps; stdout/stderr to logs/layers/<log>.log."""
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (env_extra or {}).items()})
    logf = LOGD / f'{log or Path(args[0]).stem}.log'
    say(f'run {" ".join(map(str, args))} env={env_extra or {}} -> {logf.name}')
    t0 = time.time()
    with open(logf, 'w', encoding='utf-8') as f:
        rc = subprocess.call([sys.executable, *map(str, args)], cwd=str(cwd or REB), env=env, stdout=f, stderr=subprocess.STDOUT)
    say(f'  rc {rc} in {time.time() - t0:.0f} s')
    if rc != 0:
        raise SystemExit(f'{args[0]} failed rc {rc}; see {logf}')
    return rc


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        x, y = a.to_numpy(float), b.to_numpy(float)
        return (x == y) | (np.isnan(x) & np.isnan(y))
    x, y = a.astype('string'), b.astype('string')
    return ((x == y).fillna(False) | (x.isna() & y.isna())).to_numpy()


def compare_rows(new: pd.DataFrame, ref: pd.DataFrame, key: str = 'MVT_ID_mvt', cols=None) -> dict:
    """How the rebuilt rows differ from the reference rows (ids one-sided, per-column rows differing)."""
    cols = [c for c in (cols or ref.columns) if c in new.columns and c != key]
    a, b = new.drop_duplicates(key).set_index(key), ref.drop_duplicates(key).set_index(key)
    common_ids = a.index.intersection(b.index)
    out = {'rows_new': int(len(new)), 'rows_ref': int(len(ref)), 'only_new': int(len(a.index.difference(b.index))),
           'only_ref': int(len(b.index.difference(a.index))), 'common': int(len(common_ids)), 'columns_differ': {}}
    a, b = a.loc[common_ids], b.loc[common_ids]
    anyd = np.zeros(len(common_ids), bool)
    for c in cols:
        d = ~_same(a[c], b[c])
        if d.any():
            out['columns_differ'][c] = int(d.sum())
            anyd |= d
    out['rows_any_column_differs'] = int(anyd.sum())
    missing = [c for c in ref.columns if c not in new.columns]
    if missing:
        out['columns_missing_in_new'] = missing
    return out


def splice_train(new: pd.DataFrame, ref: pd.DataFrame, split_col: str = 'split', train='train',
                 is_train=None) -> tuple[pd.DataFrame, dict]:
    """2025 rows verbatim from `ref` (the frozen reference frame, in its order and columns), then `new`'s rank rows (cast
    to the reference dtypes where possible). Returns (frame, diagnostics of new-vs-ref on the 2025 rows)."""
    rt = ref[split_col].eq(train).to_numpy() if is_train is None else is_train(ref)
    nt = new[split_col].eq(train).to_numpy() if is_train is None else is_train(new)
    diag = compare_rows(new[nt], ref[rt])
    rk = new[~nt].copy()
    for c in ref.columns:
        if c not in rk.columns:
            rk[c] = np.nan
    rk = rk[list(ref.columns)]
    for c in ref.columns:
        if rk[c].dtype != ref[c].dtype:
            try:
                if isinstance(ref[c].dtype, pd.CategoricalDtype):
                    rk[c] = rk[c].astype(ref[c].dtype)
                elif pd.api.types.is_bool_dtype(ref[c].dtype) and rk[c].notna().all():
                    rk[c] = rk[c].astype(bool)
                elif pd.api.types.is_numeric_dtype(ref[c].dtype) and pd.api.types.is_numeric_dtype(rk[c].dtype):
                    rk[c] = rk[c].astype(ref[c].dtype) if (rk[c].notna().all() or pd.api.types.is_float_dtype(ref[c].dtype)) else rk[c]
            except (TypeError, ValueError):
                pass
    out = pd.concat([ref[rt], rk], ignore_index=True)
    diag['rank_rows'] = int(len(rk))
    return out, diag


def ids(path, col='MVT_ID_mvt'):
    return pd.read_parquet(path, columns=[col])[col]
