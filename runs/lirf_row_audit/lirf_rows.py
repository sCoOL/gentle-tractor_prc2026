"""LIRF departures, both years, one row each with raw record fields + the v57 proxy + ADS-B link. -> lirf_rows.parquet

Input frame_lirf_v58.parquet is the LIRF row table of the layer screens (263 frame + 268 reuse signals + 264 signals + the
v57 proxy); tools/layers/build_frames.py lirfbase rebuilds the same table for the final ranking rows.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import common  # noqa: E402

S = common.RUNS / 'lirf_row_audit'
f = pd.read_parquet(common.RUNS / 'v58_layers' / 'frame_lirf_v58.parquet')
raw = pd.concat([pd.read_parquet(p) for p in sorted(common.DATA.glob('training_*.parquet'))] +
                [pd.read_parquet(common.DATA / 'ranking.parquet')], ignore_index=True)
raw = raw[raw.ADEP_mvt.eq('LIRF') & raw.PHASE_mvt.eq('DEP')]
f = f.merge(raw, on='MVT_ID_mvt', how='left', suffixes=('', '_raw'))
at = pd.read_parquet(common.prc_config.ADSB_DIR / 'adsb_times.parquet')
f = f.merge(at, on='MVT_ID_mvt', how='left')
tz = 'Europe/Rome'
f['sched'] = f.SCHED_TIME_UTC_mvt
f['tko'] = f.MVT_TIME_UTC_mvt
f['blk'] = f.BLOCK_TIME_UTC_mvt
f['off_h'] = f.sched.dt.tz_convert(tz).apply(lambda x: x.utcoffset().total_seconds() / 3600 if pd.notna(x) else np.nan)
f['sched_loc'] = f.sched.dt.tz_convert(tz).dt.tz_localize(None)
f['tko_loc'] = f.tko.dt.tz_convert(tz).dt.tz_localize(None)
f['day_gap'] = (f.tko_loc.dt.normalize() - f.sched_loc.dt.normalize()).dt.days
f['adsb_taxi'] = f.adsb_takeoff_t - f.adsb_offblock_t
f['adsb_tko_err'] = f.adsb_takeoff_t - f.t
y = f.y
f['cls'] = 'normal'
f.loc[y > 1800, 'cls'] = 'long'
f.loc[(y > 1800) & ((y - f.D).abs() < 120), 'cls'] = 'copy'
f.loc[(y > 70000) & ((y - f.D).abs() >= 120), 'cls'] = 'big'
S.mkdir(parents=True, exist_ok=True)
f.to_parquet(S / 'lirf_rows.parquet', index=False)
print(f.groupby(['split', 'nm', 'cls']).size().unstack(fill_value=0))
print(f.adsb_icao.notna().groupby(f.split).mean())
