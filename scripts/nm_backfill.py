"""NM flight-plan backfill for NM-unmatched departures from their destination arrival records.

A departure whose airport record did not join to Network Manager has no ``_flt`` columns. When it
flew to another study airport, that airport's arrival record of the same flight may have joined:
its AOBT_3, LOBT, EOBT_1, IOBT and ARVT columns describe this very flight. In 2025, 897 unmatched
departures link this way; they carry 5.5 % of A25's strict squared error (RMSE 2,938 s), and for the
non-copied ones the backfilled take-off - AOBT_3 is within 300 s of the label for 67 %. Rank-available:
the ranking file's arrival rows are complete.

Link rule (both files): same FLIGHT_mvt, ADEP_mvt and ADES_mvt; arrival landing 20 min to 8 h after the
departure's take-off; the arrival's AOBT_3 present and 0-3 h before that take-off. The earliest such
arrival is used, and an arrival serves at most one departure. A departure's own block time is never read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common

LANDING_MIN, LANDING_MAX, AOBT_MAX = 1200.0, 8 * 3600.0, 3 * 3600.0
NM_TIMES = ['EOBT_1_flt', 'LOBT_flt', 'IOBT_flt', 'AOBT_3_flt', 'ARVT_1_flt', 'ARVT_3_flt']
NM_CATS = {'WK_TBL_CAT_flt': 'wake', 'MARKET_SEGMENT_flt': 'segment', 'AIRCRAFT_OPERATOR_flt': 'operator', 'FLIGHT_TYPE_flt': 'flight_type'}
DELTA_COLUMNS = ['atot_minus_eobt1', 'atot_minus_lobt', 'atot_minus_aobt3', 'aobt3_minus_lobt', 'aobt3_minus_sched', 'lobt_minus_sched',
                 'lobt_minus_iobt', 'plan_block_time', 'arr_delay', 'iobt_eq_lobt', 'sched_eq_eobt1', 'aobt3_eq_lobt']


def _utc(s):
    s = pd.to_datetime(s)
    return s if s.dt.tz is not None else s.dt.tz_localize('UTC')


def link(movements: pd.DataFrame) -> pd.DataFrame:
    """One row per linked unmatched departure: MVT_ID_mvt, arrival MVT_ID and the arrival's NM columns."""
    apts = set(common.APTS)
    dep = movements[(movements.PHASE_mvt == 'DEP') & movements.AOBT_3_flt.isna() & movements.ADES_mvt.isin(apts)]
    arr = movements[(movements.PHASE_mvt == 'ARR') & movements.AOBT_3_flt.notna()]
    d = dep[['MVT_ID_mvt', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt']]
    a = arr[['MVT_ID_mvt', 'FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt'] + NM_TIMES + list(NM_CATS)].rename(
        columns={'MVT_ID_mvt': 'arr_mvt_id', 'MVT_TIME_UTC_mvt': 'arr_landing'})
    m = d.merge(a, on=['FLIGHT_mvt', 'ADEP_mvt', 'ADES_mvt'], how='inner')
    if m.empty:
        return pd.DataFrame(columns=['MVT_ID_mvt', 'arr_mvt_id'] + NM_TIMES + list(NM_CATS))
    t = _utc(m.MVT_TIME_UTC_mvt)
    landing = (_utc(m.arr_landing) - t).dt.total_seconds()
    before = (t - _utc(m.AOBT_3_flt)).dt.total_seconds()
    m = m[(landing > LANDING_MIN) & (landing < LANDING_MAX) & (before > 0) & (before < AOBT_MAX)]
    m = m.sort_values(['MVT_ID_mvt', 'arr_landing'], kind='stable').drop_duplicates('MVT_ID_mvt')
    m = m.sort_values(['arr_mvt_id', 'arr_landing'], kind='stable').drop_duplicates('arr_mvt_id')
    return m[['MVT_ID_mvt', 'arr_mvt_id'] + NM_TIMES + list(NM_CATS)].reset_index(drop=True)


def backfill(features: pd.DataFrame, movements: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Features with G2 NM deltas and NM categoricals filled for linked departures (flag ``nm_backfilled``)."""
    out = features.copy()
    out['nm_backfilled'] = 0
    if links.empty:
        return out
    base = movements[['MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']].merge(links, on='MVT_ID_mvt', how='inner')
    sec = lambda x, y: (_utc(x) - _utc(y)).dt.total_seconds().to_numpy()
    t, sched = base.MVT_TIME_UTC_mvt, base.SCHED_TIME_UTC_mvt
    vals = {
        'atot_minus_eobt1': sec(t, base.EOBT_1_flt), 'atot_minus_lobt': sec(t, base.LOBT_flt), 'atot_minus_aobt3': sec(t, base.AOBT_3_flt),
        'aobt3_minus_lobt': sec(base.AOBT_3_flt, base.LOBT_flt), 'aobt3_minus_sched': sec(base.AOBT_3_flt, sched),
        'lobt_minus_sched': sec(base.LOBT_flt, sched), 'lobt_minus_iobt': sec(base.LOBT_flt, base.IOBT_flt),
        'plan_block_time': sec(base.ARVT_1_flt, base.EOBT_1_flt), 'arr_delay': sec(base.ARVT_3_flt, base.ARVT_1_flt),
        'iobt_eq_lobt': (_utc(base.IOBT_flt) == _utc(base.LOBT_flt)).astype(int).to_numpy(),
        'sched_eq_eobt1': (_utc(sched) == _utc(base.EOBT_1_flt)).astype(int).to_numpy(),
        'aobt3_eq_lobt': (_utc(base.AOBT_3_flt) == _utc(base.LOBT_flt)).astype(int).to_numpy(),
    }
    pos = pd.Series(np.arange(len(out)), index=out.MVT_ID_mvt.values)
    rows = pos.reindex(base.MVT_ID_mvt.values).to_numpy()
    keep = ~np.isnan(rows)
    rows = rows[keep].astype(int)
    for c, v in vals.items():
        if c in out.columns:
            col = out[c].to_numpy(dtype=float, copy=True)
            col[rows] = np.asarray(v, float)[keep]
            out[c] = col
    for src, dst in NM_CATS.items():
        if dst in out.columns:
            out.iloc[rows, out.columns.get_loc(dst)] = base[src].astype(str).to_numpy()[keep]
    flag = out['nm_backfilled'].to_numpy(copy=True)
    flag[rows] = 1
    out['nm_backfilled'] = flag
    return out
