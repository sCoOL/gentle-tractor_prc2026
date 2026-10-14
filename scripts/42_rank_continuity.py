"""Rank-side continuity pass (continuity rule): rebuild ONLY the rank-side tables so that

  1. every rank-side window / link / baseline that looks back across 1 Jan 2026 sees the December 2025 movements of the
     training file (common.rank_leadin: context rows only, departure block/taxi blanked, never written to an output);
  2. multi-day baselines that are missing only because of the month gap (disruption_z in early June 2026: no May in the
     file) use the same rule looking forward (detectors.detect_disruption_days(forward_from='2026-01-01')).

The 2025 (train) side is never written: no *_train table is touched, the fallback detectors are refitted on the unchanged
training frames only to score the new rank rows (detectors.score_fallback = the full fit of detect_fallback).

  py -3.14 scripts/42_rank_continuity.py

Writes data/: base_rank, arr_rank, det_fallback_rank, det_fallback_v2_rank, det_runway_config_windows_rank,
det_runway_config_window_rwy_rank, det_disruption_days (2026 rows only change), features_v1..v7_rank, nm_backfill_links_rank,
and runs/continuity/report.json. Feature manifests (features_v*.json) are unchanged: no column is added or removed.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import time

import numpy as np
import pandas as pd

import assemble
import base_features as bf
import common
import detectors as det
import nm_backfill as nb
import ops_context
import queue_context
import stand_timeline
from common import DATA, RUNS

_spec = importlib.util.spec_from_file_location('nested_', pathlib.Path(__file__).with_name('nested_features.py'))
nested = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(nested)
ID = 'MVT_ID_mvt'
T0 = time.time()


def free_gb():
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2 ** 30
    except Exception:  # noqa: BLE001
        return float('nan')


def wait_ram(need=8.0, label='', max_hours=6.0):
    """Heavy work only with >= 8 GB free RAM. Wait (checking every 20 s, logging every 5 min) otherwise."""
    t = time.time(); last = -1e9
    while True:
        g = free_gb()
        if not (g < need):
            print(f'  [{label}] free RAM {g:.1f} GB -> go [{time.time() - T0:.0f}s]', flush=True); return g
        if time.time() - last >= 300:
            print(f'  [{label}] free RAM {g:.1f} GB < {need} GB: waiting [{time.time() - T0:.0f}s]', flush=True); last = time.time()
        if time.time() - t > max_hours * 3600:
            raise SystemExit(f'free RAM stayed below {need} GB for {max_hours} h')
        time.sleep(20)


def cols_of(name):
    import pyarrow.parquet as pq
    return [c for c in pq.read_schema(DATA / name).names if not c.startswith('__index_level')]


def main():
    rep = {'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    wait_ram(8.0, 'start')
    # ---------------- 1. base_rank / arr_rank (G3 congestion with the Dec 2025 lead-in) ----------------
    bf.build_base(force=True, rank_only=True)
    base_rank = pd.read_parquet(DATA / 'base_rank.parquet')
    print(f'base_rank rebuilt: {len(base_rank):,} [{time.time() - T0:.0f}s]', flush=True)

    # ---------------- 2a. fallback detectors: full fits on the unchanged 2025 frames, scored on the new rank rows -------
    wait_ram(8.0, 'fallback v1/v2')
    need = sorted(set(det.FALLBACK_FEATS_V2) | {ID, 'b', 'sched_s', 'fpl_matched'})
    tr = pd.read_parquet(DATA / 'base_train.parquet', columns=need)
    p1 = det.score_fallback(tr, base_rank, det.FALLBACK_FEATS, det.FALLBACK_CATS)
    pd.DataFrame({ID: base_rank[ID].values, 'p_fallback': p1}).to_parquet(DATA / 'det_fallback_rank.parquet')
    print(f'  det_fallback_rank [{time.time() - T0:.0f}s]', flush=True)
    p2 = det.score_fallback(tr, base_rank, det.FALLBACK_FEATS_V2, det.FALLBACK_CATS_V2)
    tru = tr[tr.fpl_matched.values == 0]; rku = base_rank[base_rank.fpl_matched.values == 0]
    pl = det.score_fallback(tru, rku, det.FALLBACK_FEATS_V2, det.FALLBACK_CATS_V2, params={'num_leaves': 63, 'min_data_in_leaf': 40}, rounds=300)
    old2 = pd.read_parquet(DATA / 'det_fallback_v2_rank.parquet')
    use_lane = not np.allclose(old2.p_fallback_best.values, old2.p_fallback_v2.values, equal_nan=True)   # stage 1: best == v2 (lane lost)
    b2 = pd.DataFrame({ID: base_rank[ID].values, 'p_fallback_v2': p2})
    b2 = b2.merge(pd.DataFrame({ID: rku[ID].values, 'p_fallback_lane': pl}), on=ID, how='left')
    b2['p_fallback_best'] = np.where(b2.p_fallback_lane.notna() & use_lane, b2.p_fallback_lane, b2.p_fallback_v2)
    b2.to_parquet(DATA / 'det_fallback_v2_rank.parquet')
    rep['p_fallback_best_uses_lane'] = use_lane
    del tr, tru
    print(f'  det_fallback_v2_rank (best uses {"lane" if use_lane else "v2"}) [{time.time() - T0:.0f}s]', flush=True)

    # ---------------- 2b. runway configuration windows (rank + lead-in) ----------------
    ar = pd.read_parquet(DATA / 'arr_rank.parquet')
    mov = pd.concat([base_rank[['APT', 'RUNWAY_mvt', 't']].assign(PHASE_mvt='DEP'),
                     ar.assign(t=common.sec(ar.MVT_TIME_UTC_mvt))[['APT', 'RUNWAY_mvt', 't']].assign(PHASE_mvt='ARR'),
                     det.leadin_movements()])
    w, wr = det.detect_runway_config(mov)
    w.to_parquet(DATA / 'det_runway_config_windows_rank.parquet'); wr.to_parquet(DATA / 'det_runway_config_window_rwy_rank.parquet')
    print(f'  runway config rank: {len(w):,} windows (incl. Dec 2025 lead-in) [{time.time() - T0:.0f}s]', flush=True)

    # ---------------- 2c. disruption days: 2025 rows unchanged, 2026 rows may take a forward baseline ----------------
    at = pd.read_parquet(DATA / 'arr_train.parquet')
    arr_all = pd.concat([at, ar]); arr_all['date'] = arr_all.MVT_TIME_UTC_mvt.dt.floor('D').dt.tz_localize(None)
    daily = arr_all.groupby(['APT', 'date']).TAXITIME_SEC_mvt.agg(taxi_in_median='median', n='size').reset_index()
    wx = pd.read_parquet(DATA / 'ext_weather.parquet'); wx['date'] = wx.valid.dt.floor('D').dt.tz_localize(None)
    wxd = wx.groupby(['station', 'date']).agg(snow_day=('wx_snow', 'max'), lvp_share=('lvp', 'mean'), gust_max=('gust_kt', 'max')).reset_index().rename(columns={'station': 'APT'})
    dd = det.detect_disruption_days(daily, wxd, forward_from='2026-01-01')
    old_dd = pd.read_parquet(DATA / 'det_disruption_days.parquet')
    j = old_dd[old_dd.date < '2026-01-01'].merge(dd, on=['APT', 'date'], suffixes=('_o', '_n'))
    same25 = all(np.allclose(j[c + '_o'].astype(float), j[c + '_n'].astype(float), equal_nan=True, rtol=0, atol=0)
                 for c in ('taxi_in_median', 'baseline', 'disruption_z', 'is_disruption', 'weather_explained'))
    assert same25 and len(j) == int((old_dd.date < '2026-01-01').sum()), '2025 disruption rows changed'
    dd.to_parquet(DATA / 'det_disruption_days.parquet')
    rep['disruption_forward_days'] = int(dd.baseline_forward.sum())
    rep['disruption_forward_days_by_month'] = {str(k): int(v) for k, v in dd[dd.baseline_forward == 1].date.dt.strftime('%Y-%m').value_counts().sort_index().items()}
    print(f'  disruption: {int(dd.baseline_forward.sum())} airport-days took a forward baseline; 2025 rows identical [{time.time() - T0:.0f}s]', flush=True)
    del at, arr_all

    # ---------------- 3. features_v1_rank (assemble v1, rank part) ----------------
    rk = base_rank.merge(pd.read_parquet(DATA / 'det_fallback_rank.parquet'), on=ID, how='left').merge(pd.read_parquet(DATA / 'det_conflicts_rank.parquet'), on=ID, how='left')
    cols1 = [c for g in assemble.GROUPS_V1.values() for c in g]
    v1_old = pd.read_parquet(DATA / 'features_v1_rank.parquet')
    v1 = rk[assemble.ID_RANK + cols1].copy()
    for c in assemble.CAT_COLS: v1[c] = v1[c].astype(str)
    assert list(v1.columns) == list(v1_old.columns) and v1[ID].equals(v1_old[ID])
    v1.to_parquet(DATA / 'features_v1_rank.parquet'); del rk, v1_old
    print(f'  features_v1_rank [{time.time() - T0:.0f}s]', flush=True)

    # ---------------- 4. features_v2_rank: replace the context columns ----------------
    G3 = assemble.G3
    CFG = ['rwy_config_id', 'min_since_config_change', 'config_dep_share', 'n_dep_rwys_config', 'n_arr_rwys_config']
    DIS = ['disruption_z', 'is_disruption', 'disruption_weather_explained']
    FB = ['p_fallback', 'p_fallback_v2', 'p_fallback_lane', 'p_fallback_best']
    v2 = pd.read_parquet(DATA / 'features_v2_rank.parquet')
    src = base_rank[[ID, 'APT', 'RUNWAY_mvt', 't', 'date'] + G3].copy()
    cf = det.config_features(src, w, wr); src[CFG] = cf[CFG].values
    src['date_dt'] = pd.to_datetime(src.date)
    jj = src[['APT', 'date_dt']].merge(dd.rename(columns={'date': 'date_dt'})[['APT', 'date_dt', 'disruption_z', 'is_disruption', 'weather_explained']], on=['APT', 'date_dt'], how='left')
    src['disruption_z'] = jj.disruption_z.values; src['is_disruption'] = jj.is_disruption.fillna(0).values; src['disruption_weather_explained'] = jj.weather_explained.fillna(0).values
    src = src.merge(pd.read_parquet(DATA / 'det_fallback_rank.parquet'), on=ID, how='left').merge(b2, on=ID, how='left')
    src['rwy_config_id'] = src.rwy_config_id.astype(str).fillna('NA')
    v2n = v2.copy()
    s = src.set_index(ID).reindex(v2n[ID].values)
    for c in G3 + CFG + DIS + FB:
        v2n[c] = s[c].values.astype(v2[c].dtype) if c != 'rwy_config_id' else s[c].values
    assert list(v2n.columns) == list(v2.columns)
    v2n.to_parquet(DATA / 'features_v2_rank.parquet')
    print(f'  features_v2_rank [{time.time() - T0:.0f}s]', flush=True)

    # ---------------- 5. features_v3_rank: G3 (clipped), G9 context, nested p_fallback rescored ----------------
    wait_ram(8.0, 'fallback v3')
    v3 = pd.read_parquet(DATA / 'features_v3_rank.parquet')
    v3n = v3.copy()
    s = v2n.set_index(ID).reindex(v3n[ID].values)
    for c in G3 + [c for c in CFG + DIS if c in v3n.columns]:
        v3n[c] = s[c].values
    for c in nested.INTERVAL_COUNT_COLS:
        if c in v3n: v3n[c] = v3n[c].clip(lower=0)
    tr3 = pd.read_parquet(DATA / 'features_v3_train.parquet', columns=[ID, 'fallback'] + det.FALLBACK_FEATS)
    v3n['p_fallback'] = det.score_fallback(tr3, v3n, det.FALLBACK_FEATS, det.FALLBACK_CATS)
    del tr3
    assert list(v3n.columns) == list(v3.columns)
    v3n.to_parquet(DATA / 'features_v3_rank.parquet')
    print(f'  features_v3_rank [{time.time() - T0:.0f}s]', flush=True)

    # ---------------- 6-8. v4 (G10), v5 (G11), v6 (G12) on ranking + lead-in (scripts 37/38/39, rank side) ----------------
    raw = common.rank_with_leadin()
    rank_ids = set(base_rank[ID])
    g10 = stand_timeline.build(raw); g10 = g10[g10[ID].isin(rank_ids)]
    v4 = v3n.merge(g10, on=ID, how='left', validate='one_to_one')
    assert list(v4.columns) == cols_of('features_v4_rank.parquet')
    v4.to_parquet(DATA / 'features_v4_rank.parquet')
    print(f'  features_v4_rank [{time.time() - T0:.0f}s]', flush=True)
    g11 = ops_context.build(raw); g11 = g11[g11[ID].isin(rank_ids)]
    v5 = v4.merge(g11, on=ID, how='left', validate='one_to_one')
    v5['st_turn_in_is_copy'] = np.where(np.isfinite(v5.st_turn_in_delay.values), (np.abs(v5.st_turn_in_delay.values) <= 10).astype(float), np.nan)
    assert list(v5.columns) == cols_of('features_v5_rank.parquet')
    v5.to_parquet(DATA / 'features_v5_rank.parquet')
    print(f'  features_v5_rank [{time.time() - T0:.0f}s]', flush=True)
    g12 = queue_context.build(raw); g12 = g12[g12[ID].isin(rank_ids)]
    v6 = v5.merge(g12, on=ID, how='left', validate='one_to_one')
    v6['qx_own_minus_rwy_c60'] = np.clip(v6.atot_minus_aobt3.values, 0, 5400) - v6.cx_dep_a3taxi_med_rwy_c60.values
    assert list(v6.columns) == cols_of('features_v6_rank.parquet')
    v6.to_parquet(DATA / 'features_v6_rank.parquet')
    print(f'  features_v6_rank [{time.time() - T0:.0f}s]', flush=True)
    del raw, v6

    # ---------------- 9. v7 = v5 + NM backfill (script 41, rank side, links on ranking + lead-in) ----------------
    need = [ID, 'PHASE_mvt', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt'] + nb.NM_TIMES + list(nb.NM_CATS)
    rawn = common.rank_with_leadin(need)
    links = nb.link(rawn); links = links[links[ID].isin(rank_ids)].reset_index(drop=True)
    links.to_parquet(DATA / 'nm_backfill_links_rank.parquet', index=False)
    v7 = nb.backfill(v5, rawn, links)
    assert list(v7.columns) == cols_of('features_v7_rank.parquet') and len(v7) == 670790
    v7.to_parquet(DATA / 'features_v7_rank.parquet')
    rep['rank_backfill_links'] = int(len(links)); rep['rank_backfilled'] = int(v7.nm_backfilled.sum())
    print(f'  features_v7_rank: links {len(links)}, backfilled {int(v7.nm_backfilled.sum())} [{time.time() - T0:.0f}s]', flush=True)

    rep['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); rep['seconds'] = round(time.time() - T0)
    (RUNS / 'continuity').mkdir(parents=True, exist_ok=True)
    (RUNS / 'continuity' / 'report.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
