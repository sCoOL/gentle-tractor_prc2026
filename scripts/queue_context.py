"""G12 queue and pushback-cohort context (rank-available), complementing G11's take-off windows.

For a departure with take-off t, NM off-block estimate a (AOBT_3), runway r and pier p:

  qx_rwy_prev_to_gap        t - previous take-off on runway r (clipped at 3600 s)
  qx_rwy_next_to_gap        next take-off on runway r - t (clipped at 3600 s)
  qx_cohort_a3taxi_med_rwy  median (take-off - AOBT_3) of departures on runway r whose AOBT_3 lies
                            within +-10 min of a (the pushback cohort; clipped 0-5400 s)
  qx_cohort_n_rwy           size of that cohort
  qx_cohort_a3taxi_med_apt  the same cohort median over every runway of the airport
  qx_own_minus_cohort_rwy   (t - a) - qx_cohort_a3taxi_med_rwy
  qx_pier_a3taxi_med_c60    median (take-off - AOBT_3) of departures from pier p with take-off in +-30 min
  qx_pier_arr_taxiin_c60    median taxi-in of arrivals to pier p landing in +-30 min

Why: G11 centres its windows on the take-off second, which mixes aircraft that pushed at very different
times. A departure's taxi is better predicted by the aircraft that pushed back with it, and a
single NM estimate far from its cohort is more likely wrong. Queue spacing on the runway and apron
activity at the pier describe the two ends of the taxi. A departure's own block time is never read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GAP_CLIP = 3600.0
COLUMNS = ['qx_rwy_prev_to_gap', 'qx_rwy_next_to_gap', 'qx_cohort_a3taxi_med_rwy', 'qx_cohort_n_rwy', 'qx_cohort_a3taxi_med_apt',
           'qx_own_minus_cohort_rwy', 'qx_pier_a3taxi_med_c60', 'qx_pier_arr_taxiin_c60']


def _naive_utc(s):
    s = pd.to_datetime(s)
    if s.dt.tz is None:
        return s
    return s.dt.tz_convert('UTC').dt.tz_localize(None)


def _pier(stand):
    s = stand.astype(str).fillna('NA')
    return s.str.extract(r'^([A-Za-z]+)')[0].fillna(s.str[:1])


def _roll(times, values, window):
    s = pd.Series(values, index=pd.DatetimeIndex(times))
    r = s.rolling(window, center=True, closed='both', min_periods=1)
    return r.median().to_numpy(), r.count().to_numpy()


def build(movements: pd.DataFrame) -> pd.DataFrame:
    """Return one row per DEP movement: MVT_ID_mvt + COLUMNS."""
    df = movements[['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'RUNWAY_mvt', 'STAND_mvt',
                    'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'AOBT_3_flt']].reset_index(drop=True)
    is_dep = (df.PHASE_mvt == 'DEP').to_numpy()
    apt = np.where(is_dep, df.ADEP_mvt.astype(str), df.ADES_mvt.astype(str))
    t = _naive_utc(df.MVT_TIME_UTC_mvt)
    a = _naive_utc(df.AOBT_3_flt)
    tsec = (t - t.min()).dt.total_seconds().to_numpy()
    # a departure's block time is blanked before use: only arrivals contribute taxi-in
    block = np.where(is_dep, np.nan, (_naive_utc(df.BLOCK_TIME_UTC_mvt) - t).dt.total_seconds().to_numpy())
    dep_x = np.where(is_dep, np.clip((t - a).dt.total_seconds().to_numpy(), 0, 5400), np.nan)
    arr_taxi = np.where(~is_dep, np.clip(block, 0, 3600), np.nan)
    rwy = df.RUNWAY_mvt.astype(str).fillna('NA').to_numpy()
    pier = _pier(df.STAND_mvt).to_numpy()
    tv = t.to_numpy(); av = a.to_numpy()
    n = len(df)
    out = {c: np.full(n, np.nan) for c in COLUMNS}
    dep_idx = np.where(is_dep)[0]

    frame = pd.DataFrame({'apt': apt[dep_idx], 'rwy': rwy[dep_idx]}, index=dep_idx)
    for _, rows in frame.groupby(['apt', 'rwy'], sort=False).indices.items():
        rows = dep_idx[rows]
        order = rows[np.argsort(tsec[rows], kind='stable')]
        ts = tsec[order]
        out['qx_rwy_prev_to_gap'][order] = np.minimum(np.concatenate([[np.nan], np.diff(ts)]), GAP_CLIP)
        out['qx_rwy_next_to_gap'][order] = np.minimum(np.concatenate([np.diff(ts), [np.nan]]), GAP_CLIP)
        m = order[~pd.isna(av[order])]
        if len(m):
            m = m[np.argsort(av[m], kind='stable')]
            med, cnt = _roll(av[m], dep_x[m], '20min')
            out['qx_cohort_a3taxi_med_rwy'][m] = med
            out['qx_cohort_n_rwy'][m] = cnt
    matched = dep_idx[~pd.isna(av[dep_idx])]
    for _, rows in pd.Series(apt[matched]).groupby(apt[matched], sort=False).indices.items():
        m = matched[rows]
        m = m[np.argsort(av[m], kind='stable')]
        med, _ = _roll(av[m], dep_x[m], '20min')
        out['qx_cohort_a3taxi_med_apt'][m] = med
    keys = pd.Series(apt).str.cat(pd.Series(pier), sep='|').to_numpy()
    for _, rows in pd.Series(keys).groupby(keys, sort=False).indices.items():
        if not is_dep[rows].any():
            continue
        order = rows[np.argsort(tsec[rows], kind='stable')]
        med_d, _ = _roll(tv[order], dep_x[order], '60min')
        med_a, _ = _roll(tv[order], arr_taxi[order], '60min')
        d = is_dep[order]
        out['qx_pier_a3taxi_med_c60'][order[d]] = med_d[d]
        out['qx_pier_arr_taxiin_c60'][order[d]] = med_a[d]
    out['qx_own_minus_cohort_rwy'] = dep_x - out['qx_cohort_a3taxi_med_rwy']
    res = pd.DataFrame({'MVT_ID_mvt': df.MVT_ID_mvt.to_numpy()[dep_idx]})
    for c in COLUMNS:
        res[c] = out[c][dep_idx]
    return res
