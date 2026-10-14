"""Label-free, base-prediction-independent layer frames for the final file (670,790 rank rows), built once before the
final chain (tools/final_chain.py) fills in the base prediction.

  py -3.14 tools/layers/build_frames.py static rows66 f266 f269 ovcache item05 item04 item02 item11 longrows lirfbase
  py -3.14 tools/layers/build_frames.py motion           # 165 sustained-motion features for the 2026 days (pool of 4)

Every frame keeps its 2025 rows VERBATIM from the frozen reference frame (`rt.splice_train`, PRC_FROZEN/ref_layers): the
layers' 2025 fits are the validated ones; the rebuilt 2025 rows are compared with the reference rows (diagnostics in
logs/layers/record.json, key frames.<name>). Rank rows are rebuilt from the final base tables, the ADS-B / OPDI tables
and the extracted points. Outputs go to the relative paths the layer scripts read (runs/v55_retention,
runs/v56_neighbour, runs/overnight_20261008, runs/long_rows_20261008, runs/v66_bundle/rows.parquet), so the scripts need
no path change.
"""
import importlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt  # noqa: E402  (thread caps first)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REB, REF = rt.REB, rt.REF
RUNS = REB / 'runs'
OV = RUNS / 'overnight_20261008'
LR = RUNS / 'long_rows_20261008'
CODE_OV = rt.REPO / 'runs' / 'overnight_20261008'      # the item scripts (code) live in the repository
CODE_LR = rt.REPO / 'runs' / 'long_rows_20261008'


def _train_ids():
    import common
    return set(pd.read_parquet(common.DATA / 'base_train.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt)


def _splice_file(name, new_path, ref_path, out_path=None, **kw):
    new, ref = pd.read_parquet(new_path), pd.read_parquet(ref_path)
    out, diag = rt.splice_train(new, ref, **kw)
    out.to_parquet(out_path or new_path, index=False)
    rt.record(f'frames.{name}', diag)
    rt.say(f'{name}: 2025 rows from the reference ({int((out.index < len(out) - diag["rank_rows"]).sum())}), rank rows {diag["rank_rows"]}; '
           f'rebuilt-vs-reference 2025: only_new {diag["only_new"]}, only_ref {diag["only_ref"]}, rows differing {diag["rows_any_column_differs"]} '
           f'{dict(list(diag["columns_differ"].items())[:8])}')
    return out


# ------------------------------------------------------------------------------------------------------------ static
STATIC = ['runs/a51_final/chosen_ensemble_oof.parquet', 'runs/v52_recovered_lobt/rows.parquet',
          'runs/v53_opdi/oof_LSZH.parquet', 'runs/v53_opdi/nested_LSZH.json', 'runs/v53_opdi/oof_EDDF_stand.parquet',
          'runs/v53_opdi/nested_EDDF_stand.json', 'runs/v53_build/evidence.json',
          'runs/v54_build/rows.parquet', 'runs/v54_build/evidence.json', 'runs/v55_build/rows.parquet',
          'runs/v55_build/evidence.json', 'runs/v55_retention/nested_ret6ov.json',
          'runs/v55_retention/oof_ret6ov.parquet', 'runs/v56_neighbour/nested_nb.json', 'runs/v56_neighbour/oof_nb.parquet',
          'runs/v56_build/evidence.json', 'runs/v63_context_layer/screen.json', 'runs/v63_context_layer/rows.parquet',
          'runs/v66_bundle/l1_screen_sixfold.json', 'runs/v66_bundle/l1_screen_forward.json', 'runs/v66_bundle/l2_screen.json',
          'runs/v66_bundle/l3_screen.json', 'runs/long_rows_20261008/gate3x_oof.parquet',
          'runs/long_rows_20261008/gate3_no_lirf.json', 'runs/long_rows_20261008/stack_v67.json', 'runs/models/A57_20260924_v49/motion/model.txt',
          'runs/models/A57_20260924_v49/motion/state.json', 'runs/models/A57_20260924_v49/summary.json']


def static():
    """2025-side inputs the layer scripts read by fixed relative path: copied from PRC_FROZEN/ref_layers."""
    for rel in STATIC:
        src, dst = REF / rel, REB / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
    rt.say(f'static: {len(STATIC)} reference files in place under runs/')


def rows66():
    """ov.load_proxy() table: the reference 2025 v66 proxy rows + one row per FINAL rank departure (labels / base NaN; the layer
    builds put the v66 file on rank rows themselves)."""
    import common
    ref = pd.read_parquet(REF / 'runs/v66_bundle/rows.parquet')
    b = pd.read_parquet(common.DATA / 'base_rank.parquet', columns=['MVT_ID_mvt', 'APT', 'month'])
    rk = pd.DataFrame({'MVT_ID_mvt': b.MVT_ID_mvt.to_numpy(), 'split': 'rank', 'fold': None, 'month': b.month.to_numpy(),
                       'apt': b.APT.to_numpy(), 'y': np.nan, 'v63_pred_s': np.nan, 'v66_pred_s': np.nan, 'layer66': ''})
    out, diag = rt.splice_train(rk.assign(split='rank'), ref)
    (RUNS / 'v66_bundle').mkdir(parents=True, exist_ok=True)
    out.to_parquet(RUNS / 'v66_bundle/rows.parquet', index=False)
    rt.say(f'rows66: {int((out.split == "train").sum())} train + {int((out.split == "rank").sum())} rank rows')


def f266():
    s266 = importlib.import_module('266_gate_retention_screen')
    s266.OUT = RUNS / 'v55_retention'
    rt.wait_mem(8, '266 frame')
    s266.frame()
    _splice_file('v55_retention.frame', s266.OUT / 'frame.parquet', REF / 'runs/v55_retention/frame.parquet')
    s266.overtakes()
    tr_ids = _train_ids()
    _splice_file('v55_retention.overtakes', s266.OUT / 'overtakes.parquet', REF / 'runs/v55_retention/overtakes.parquet',
                 is_train=lambda d: d.MVT_ID_mvt.isin(tr_ids).to_numpy())


def f269():
    n269 = importlib.import_module('269_neighbour_lateness')
    n269.OUT = RUNS / 'v56_neighbour'
    rt.wait_mem(8, '269 frame')
    n269.frame()
    _splice_file('v56_neighbour.frame', n269.OUT / 'frame.parquet', REF / 'runs/v56_neighbour/frame.parquet')


def ovcache():
    """runs/overnight_20261008/frame_cache.parquet = 263 frame + status on the final base tables (ov.frame())."""
    p = OV / 'frame_cache.parquet'
    if p.exists():
        p.unlink()
    sys.path.insert(0, str(CODE_OV))
    ov = importlib.import_module('ov')
    f = ov.frame()
    ref = pd.read_parquet(REF / 'runs/overnight_20261008/frame_cache.parquet')
    rt.record('frames.ov_frame_cache', rt.compare_rows(f[f.split == 'train'], ref[ref.split == 'train']))
    rt.say(f'ov frame cache: {len(f)} rows ({int((f.split == "rank").sum())} rank); 2025 rows vs reference cache: '
           f'{rt.compare_rows(f[f.split == "train"], ref[ref.split == "train"])["columns_differ"]}')


def item05():
    rt.wait_mem(8, 'item05')
    rt.run_script([CODE_OV / 'item05_dateshift.py'], log='item05_dateshift')
    _splice_file('overnight.item05_rows', OV / 'item05_rows.parquet', REF / 'runs/overnight_20261008/item05_rows.parquet')


def item04():
    if (OV / 'item04_park_rank.parquet').exists():
        rt.say('item04 park: rank extraction already on disk (item04_park_rank.parquet), not repeated')
    else:
        rt.wait_mem(8, 'item04 park extract')
        rt.run_script([CODE_OV / 'item04_park_extract.py'], {'PRC_SINCE': '2026-01-01', 'PRC_WORKERS': rt.THREADS,
                                                     'PRC_PARK_OUT': 'item04_park_rank.parquet'}, log='item04_park_extract')
    tr_ids = _train_ids()
    rk = pd.read_parquet(OV / 'item04_park_rank.parquet')
    ref = pd.read_parquet(REF / 'runs/overnight_20261008/item04_park.parquet')
    out = pd.concat([ref[ref.MVT_ID_mvt.isin(tr_ids)], rk[~rk.MVT_ID_mvt.isin(tr_ids)]], ignore_index=True)
    out.to_parquet(OV / 'item04_park.parquet', index=False)
    rt.say(f'item04 park: {int(out.MVT_ID_mvt.isin(tr_ids).sum())} 2025 rows (reference) + {int((~out.MVT_ID_mvt.isin(tr_ids)).sum())} rank rows')
    rt.wait_mem(8, 'item04y rows')
    rt.run_script([CODE_OV / 'item04y_stand_error.py'], {'PRC_ROWS_ONLY': 1}, log='item04y_stand_error')
    _splice_file('overnight.item04y_rows', OV / 'item04y_rows.parquet', REF / 'runs/overnight_20261008/item04y_rows.parquet')


def item02():
    rt.wait_mem(8, 'item02 frame')
    rt.run_script([CODE_OV / 'item02_gaps.py'], {'PRC_ROWS_ONLY': 1}, log='item02_gaps')
    _splice_file('overnight.item02_frame', OV / 'item02_frame.parquet', REF / 'runs/overnight_20261008/item02_frame.parquet')


def item11():
    rt.wait_mem(8, 'item11 arrival tail extract')
    rt.run_script([CODE_OV / 'item11_arr_tail_extract.py'], {'PRC_RANK_ONLY': 1, 'PRC_WORKERS': rt.THREADS,
                                                         'PRC_TAIL_OUT': 'item11_arr_tail_rank.parquet'}, log='item11_arr_tail_extract')
    rk = pd.read_parquet(OV / 'item11_arr_tail_rank.parquet')
    ref = pd.read_parquet(REF / 'runs/overnight_20261008/item11_arr_tail.parquet')
    out = pd.concat([ref[ref.split == 'train'], rk[rk.split == 'rank'][list(ref.columns)]], ignore_index=True)
    out.to_parquet(OV / 'item11_arr_tail.parquet', index=False)
    rt.say(f'item11 arrival tail: {int((out.split == "train").sum())} 2025 rows (reference) + {int((out.split == "rank").sum())} rank rows')
    rt.wait_mem(8, 'item11 rows')
    rt.run_script([CODE_OV / 'item11_shadow_layer.py'], {'PRC_ROWS_ONLY': 1}, log='item11_shadow_layer')
    _splice_file('overnight.item11_rows', OV / 'item11_rows.parquet', REF / 'runs/overnight_20261008/item11_rows.parquet')


def longrows():
    rt.wait_mem(8, 'long_rows frame')
    rt.run_script([CODE_LR / 'audit.py'], {'PRC_ROWS_ONLY': 1}, log='long_rows_audit')
    _splice_file('long_rows.frame', LR / 'frame.parquet', REF / 'runs/long_rows_20261008/frame.parquet')


def lirfbase():
    """Rank rows of runs/lirf_row_audit/lirf_rows.parquet (the LIRF row table: 263 frame + 268 reuse signals + 264 signals
    + the raw record fields, ADS-B times and local clock of runs/lirf_row_audit/lirf_rows.py), everything except the two inputs that depend on the run: `valid` (the 264 lane table) and p / v57_file (cv57).
    -> runs/lirf_row_audit/lirf_rows_rank_base.parquet. tools/final_chain.py step 63 completes it."""
    import common
    b53 = importlib.import_module('263_v53_build')
    r268 = importlib.import_module('268_v56_stand_reuse')
    g264 = importlib.import_module('264_v54_unmatched_gate')
    rt.wait_mem(8, 'lirf rows base')
    f = b53.frame()
    f['status'] = b53.status(f)
    f = f[(f.apt == 'LIRF') & (f.split == 'rank')].copy()
    f['fold'], f['v57_pred_s'], f['layer'], f['v57_file'], f['p'] = None, np.nan, np.nan, np.nan, np.nan
    f['D'] = f.t - f.t_sched
    f = f.merge(r268.reuse_signals(), on='MVT_ID_mvt', how='left').merge(g264.signals(), on='MVT_ID_mvt', how='left')
    f['valid'] = np.nan
    fl = pd.read_parquet(common.DATA / 'base_rank.parquet', columns=['MVT_ID_mvt', 'FLIGHT_mvt'])
    f = f.merge(fl, on='MVT_ID_mvt', how='left')
    f['bel'] = (f.apt == 'LIRF') & (~f.nm) & f.FLIGHT_mvt.str.startswith('BEL', na=False)
    # runs/lirf_row_audit/lirf_rows.py on these rows (raw record fields, ADS-B times, local clock, label class)
    raw = pd.read_parquet(common.DATA / 'ranking.parquet')
    raw = raw[raw.ADEP_mvt.eq('LIRF') & raw.PHASE_mvt.eq('DEP')]
    f = f.merge(raw, on='MVT_ID_mvt', how='left', suffixes=('', '_raw'))
    at = pd.read_parquet(rt.prc_config.ADSB_DIR / 'adsb_times.parquet')
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
    ref = pd.read_parquet(REF / 'runs/clean_chain/lirf_row_audit/lirf_rows.parquet')   # column layout of the 2025 rows
    missing = [c for c in ref.columns if c not in f.columns]
    extra = [c for c in f.columns if c not in ref.columns]
    assert not missing, f'lirf rank rows lack {missing}'
    f = f[list(ref.columns)]
    (RUNS / 'lirf_row_audit').mkdir(parents=True, exist_ok=True)
    f.to_parquet(RUNS / 'lirf_row_audit' / 'lirf_rows_rank_base.parquet', index=False)
    rt.record('frames.lirf_rows_rank_base', {'rows': int(len(f)), 'extra_columns_dropped': extra})
    rt.say(f'lirf rank base: {len(f)} rows')


# ------------------------------------------------------------------------------------------------------------ motion
MOTION_OUT = RUNS / 'v49_motion_full_final'


def motion():
    """165 sustained-motion features for every 2026 departure day of the final file (the A57 booster is re-applied, never
    refitted, so the 2025 training days are not re-extracted)."""
    import common
    r = pd.read_parquet(common.DATA / 'base_rank.parquet', columns=['MVT_ID_mvt', 'MVT_TIME_UTC_mvt'])
    days = sorted(pd.to_datetime(r.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d').unique())
    lst = rt.LOGD / 'motion_days.txt'
    lst.write_text('\n'.join(days) + '\n')
    rt.say(f'motion: {len(days)} departure days {days[0]} .. {days[-1]}')
    rt.wait_mem(10, '165 motion (pool of 4)')
    rt.run_script(['scripts/165_ground_motion_full.py', '--workers', rt.THREADS, '--out', MOTION_OUT, '--since', '2026-01-01'],
                  {'PRC_ADSB_DAYS': lst}, log='165_motion_final')
    f = pd.read_parquet(MOTION_OUT / 'features.parquet')
    rt.record('frames.motion_features', {'rows': int(len(f)), 'by_month': f.date.str[:7].value_counts().sort_index().to_dict()})
    rt.say(f'motion features: {len(f)} rows')


STEPS = {'static': static, 'rows66': rows66, 'f266': f266, 'f269': f269, 'ovcache': ovcache, 'item05': item05,
         'item04': item04, 'item02': item02, 'item11': item11, 'longrows': longrows, 'lirfbase': lirfbase, 'motion': motion}

if __name__ == '__main__':
    for a in sys.argv[1:]:
        rt.say(f'=== build_frames {a}')
        STEPS[a]()
