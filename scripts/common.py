"""Shared constants and helpers for the PRC 2026 taxi-out pipeline."""
from __future__ import annotations
import json, time, os
from pathlib import Path
import numpy as np, pandas as pd

import sys as _sys
if str(Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent          # repository root (code)
WORK = prc_config.WORK                                # data / runs / submissions root (PRC_WORK)
DATA, RUNS, SUBS, EXT = prc_config.DATA, prc_config.RUNS, prc_config.SUBS, prc_config.EXTERNAL
APTS = ['LTFM', 'EHAM', 'EGLL', 'LFPG', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'LIRF', 'LSZH']
EP = pd.Timestamp('2025-01-01', tz='UTC')
HELI_TYPES = {'AS50', 'A109', 'EC20', 'B429', 'EC35', 'EC45', 'H145', 'H135', 'AS55', 'AS65', 'EC30', 'EC55',
              'R44', 'R66', 'B06', 'B407', 'H160', 'A139', 'A169', 'AW139', 'S76', 'H175', 'EC75'}
FOLDS = {
    'A': [1, 7], 'B': [2, 8], 'C': [3, 9],
    'D': [4, 10], 'E': [5, 11], 'F': [6, 12],
}
RESULT_COLUMNS = ['run_id', 'ts', 'feature_version', 'groups_on', 'model', 'params_json', 'fold', 'rmse_raw',
                  'rmse_clip3600', 'mae', 'rmse_jan', 'rmse_jul', 'rmse_by_apt_json', 'n_train', 'n_test',
                  'fit_seconds', 'notes']
RESULT_COLUMNS += ['rmse_matched', 'rmse_unmatched', 'rmse_body', 'ss_top100_share', 'n_top100_lirf', 'seed']
AIRPORT_META = {
    'LTFM': dict(tz='Europe/Istanbul', country='TR', subdiv=None, lat=41.274874, lon=28.732136),
    'EHAM': dict(tz='Europe/Amsterdam', country='NL', subdiv=None, lat=52.308601, lon=4.763890),
    'EGLL': dict(tz='Europe/London', country='GB', subdiv='ENG', lat=51.470748, lon=-0.459909),
    'LFPG': dict(tz='Europe/Paris', country='FR', subdiv=None, lat=49.008960, lon=2.554117),
    'EDDF': dict(tz='Europe/Berlin', country='DE', subdiv='HE', lat=50.026706, lon=8.558350),
    'LEMD': dict(tz='Europe/Madrid', country='ES', subdiv='MD', lat=40.493407, lon=-3.572249),
    'LEBL': dict(tz='Europe/Madrid', country='ES', subdiv='CT', lat=41.297100, lon=2.078460),
    'EDDM': dict(tz='Europe/Berlin', country='DE', subdiv='BY', lat=48.353802, lon=11.786100),
    'LIRF': dict(tz='Europe/Rome', country='IT', subdiv='RM', lat=41.804532, lon=12.251998),
    'LSZH': dict(tz='Europe/Zurich', country='CH', subdiv='ZH', lat=47.458056, lon=8.548056)}


RANK_LEADIN_FILE = 'training_2025-12-01_2026-01-01.parquet'


def rank_leadin(columns=None) -> pd.DataFrame:
    """Continuity rule: December 2025 movements from the training file, used ONLY as
    look-back context when rank-side windows / links / baselines are computed, so 1 Jan 2026 rows see 31 Dec 2025.
    Departure block / taxi times are blanked exactly as in the ranking file. These rows are never written to any output
    and never enter pooled-over-the-file statistics; the 2025 (train) side does not call this."""
    d = pd.read_parquet(DATA / RANK_LEADIN_FILE, columns=columns)
    if 'PHASE_mvt' in d:
        dep = d.PHASE_mvt.eq('DEP')
        if 'BLOCK_TIME_UTC_mvt' in d:
            d.loc[dep, 'BLOCK_TIME_UTC_mvt'] = pd.NaT
        if 'TAXITIME_SEC_mvt' in d:
            d.loc[dep, 'TAXITIME_SEC_mvt'] = np.nan
    return d


def rank_with_leadin(columns=None) -> pd.DataFrame:
    """The ranking file followed by the December 2025 lead-in rows (new RangeIndex)."""
    rk = pd.read_parquet(DATA / 'ranking.parquet', columns=columns)
    return pd.concat([rk, rank_leadin(columns)], ignore_index=True)


def ensure_dirs() -> None:
    for p in (DATA, RUNS, RUNS / 'oof', RUNS / 'models', RUNS / 'logs', SUBS):
        p.mkdir(parents=True, exist_ok=True)


def sec(s: pd.Series) -> pd.Series:
    """Seconds since 2025-01-01 UTC; NaT -> NaN."""
    return (s - EP).dt.total_seconds()


def rmse(p, y) -> float:
    p = np.asarray(p, dtype=float); y = np.asarray(y, dtype=float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def metrics(p, y, apt, month, matched) -> dict:
    """Score dictionary used by every results row: pooled, per month, per lane, body, top-100 share."""
    p = np.asarray(p, float); y = np.asarray(y, float); apt = np.asarray(apt); month = np.asarray(month); matched = np.asarray(matched)
    e2 = (p - y) ** 2; ss = e2.sum(); top = np.argsort(-e2)[:100]; ms = sorted(np.unique(month))
    def r(m): return float(np.sqrt(e2[m].mean())) if m.sum() else float('nan')
    return dict(rmse_raw=float(np.sqrt(e2.mean())), rmse_clip3600=rmse(p, np.minimum(y, 3600)), mae=float(np.mean(np.abs(p - y))),
                rmse_m1=r(month == ms[0]), rmse_m2=r(month == ms[-1]), rmse_matched=r(matched == 1), rmse_unmatched=r(matched == 0),
                rmse_body=r(y <= 3600), ss_top100_share=float(e2[top].sum() / ss) if ss > 0 else 0.0, n_top100_lirf=int((apt[top] == 'LIRF').sum()),
                rmse_by_apt_json=json.dumps({a: round(r(apt == a), 1) for a in sorted(np.unique(apt))}))


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def fold_masks(month: pd.Series, fold: str):
    test = month.isin(FOLDS[fold])
    return ~test, test


def manifest_path(version: int) -> Path:
    return DATA / f'features_v{version}.json'


def write_manifest(version: int, columns: dict, groups: dict) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    p = manifest_path(version)
    p.write_text(json.dumps({'version': version, 'built_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                             'columns': columns, 'groups': groups}, indent=1), encoding='utf-8')
    return p


def read_manifest(version: int) -> dict:
    return json.loads(manifest_path(version).read_text(encoding='utf-8'))


def load_features(version: int):
    m = read_manifest(version)
    tr = pd.read_parquet(DATA / f'features_v{version}_train.parquet')
    rk = pd.read_parquet(DATA / f'features_v{version}_rank.parquet')
    return tr, rk, m


def append_result(row: dict) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    p = RUNS / 'results.csv'
    df = pd.DataFrame([{c: row.get(c) for c in RESULT_COLUMNS} | {k: v for k, v in row.items() if k not in RESULT_COLUMNS}])
    if p.exists():
        old = pd.read_csv(p)
        if list(old.columns) != list(df.columns):
            pd.concat([old, df], ignore_index=True).to_csv(p, index=False); return
    df.to_csv(p, mode='a', index=False, header=not p.exists())


def new_run_id(prefix: str) -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{prefix}"
