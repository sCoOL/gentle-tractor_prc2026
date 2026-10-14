"""G11 operations context: contemporaneous taxi levels and schedule-copy activity at the airport.

Every input is rank-available. For a departure that took off at t:

  cx_dep_a3taxi_med_c60      median of (take-off - NM AOBT_3) over departures at the airport with
                             take-off in [t - 30 min, t + 30 min] (matched rows only, clipped 0-5400 s)
  cx_dep_a3taxi_n_c60        number of values in that median
  cx_dep_a3taxi_med_rwy_c60  the same median restricted to the departure's runway
  cx_dep_a3taxi_med_prev60   the same median over take-offs in [t - 60 min, t)
  cx_arr_taxiin_med_c60      median arrival taxi-in (in-block - landing, clipped 0-3600 s) over
                             arrivals landing in [t - 30 min, t + 30 min]
  cx_arr_taxiin_n_c60        number of values in that median
  cx_arr_taxiin_med_c180     the same median over [t - 90 min, t + 90 min]
  cx_arr_delay_med_c180      median arrival in-block delay vs schedule (clipped +-6 h), +-90 min
  cx_arr_copy_rate_c180      share of those arrivals whose in-block equals their schedule (+-10 s)
  cx_arr_copy_rate_day       the same share over the airport-local calendar day
  cx_arr_copy_rate_al_day    the same share for the departure's airline prefix that day, shrunk
                             toward the day rate with strength 5

Why: the body error follows the day's taxi level (de-icing, snow, runway works), which other
flights' AOBT_3 gaps and the complete arrival records reveal; and the airport's habit of copying
the schedule into block times on a given day shows in the arrivals as well as the departures.
A departure's own off-block time (the label) is never read. Its own take-off and AOBT_3 enter the
pooled medians exactly as every other departure's do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common

COPY_TOL = 10.0
AL_STRENGTH = 5.0
COLUMNS = ['cx_dep_a3taxi_med_c60', 'cx_dep_a3taxi_n_c60', 'cx_dep_a3taxi_med_rwy_c60', 'cx_dep_a3taxi_med_prev60',
           'cx_arr_taxiin_med_c60', 'cx_arr_taxiin_n_c60', 'cx_arr_taxiin_med_c180', 'cx_arr_delay_med_c180',
           'cx_arr_copy_rate_c180', 'cx_arr_copy_rate_day', 'cx_arr_copy_rate_al_day']


def _airline(flight):
    return flight.astype(str).str.extract(r'^([A-Z]{2,3})')[0].fillna('NA')


def _rolling(times, values, window, center=True, closed='both'):
    """Time-window median and count of ``values`` (NaN ignored) at each of ``times`` (sorted)."""
    s = pd.Series(values, index=pd.DatetimeIndex(times))
    r = s.rolling(window, center=center, closed=closed, min_periods=1)
    return r.median().to_numpy(), r.count().to_numpy()


def build(movements: pd.DataFrame) -> pd.DataFrame:
    """Return one row per DEP movement: MVT_ID_mvt + COLUMNS."""
    df = movements[['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'RUNWAY_mvt', 'FLIGHT_mvt',
                    'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt']].copy()
    df['APT'] = np.where(df.PHASE_mvt == 'DEP', df.ADEP_mvt, df.ADES_mvt)
    t = pd.to_datetime(df.MVT_TIME_UTC_mvt)
    if t.dt.tz is None:
        t = t.dt.tz_localize('UTC')
    df['t'] = t.dt.tz_convert('UTC').dt.tz_localize(None)
    sec = lambda c: (pd.to_datetime(df[c]) - pd.to_datetime(df.MVT_TIME_UTC_mvt)).dt.total_seconds()
    is_dep = (df.PHASE_mvt == 'DEP').to_numpy()
    # never read a departure's block time: blank it before anything is derived
    block_rel = np.where(is_dep, np.nan, sec('BLOCK_TIME_UTC_mvt').to_numpy())
    sched_rel = sec('SCHED_TIME_UTC_mvt').to_numpy()
    a3_rel = sec('AOBT_3_flt').to_numpy()
    df['dep_x'] = np.where(is_dep, np.clip(-a3_rel, 0, 5400), np.nan)
    df['arr_taxi'] = np.where(~is_dep, np.clip(block_rel, 0, 3600), np.nan)
    arr_delay = block_rel - sched_rel
    df['arr_delay'] = np.where(~is_dep, np.clip(arr_delay, -21600, 21600), np.nan)
    df['arr_copy'] = np.where(~is_dep & np.isfinite(arr_delay), (np.abs(arr_delay) <= COPY_TOL).astype(float), np.nan)
    df['airline'] = _airline(df.FLIGHT_mvt)
    out = []
    for apt, g in df.groupby('APT', sort=False):
        g = g.sort_values('t', kind='stable')
        tt = g.t.to_numpy()
        r = pd.DataFrame({'MVT_ID_mvt': g.MVT_ID_mvt.to_numpy()})
        r['cx_dep_a3taxi_med_c60'], r['cx_dep_a3taxi_n_c60'] = _rolling(tt, g.dep_x.to_numpy(), '60min')
        r['cx_arr_taxiin_med_c60'], r['cx_arr_taxiin_n_c60'] = _rolling(tt, g.arr_taxi.to_numpy(), '60min')
        r['cx_arr_taxiin_med_c180'], _ = _rolling(tt, g.arr_taxi.to_numpy(), '180min')
        r['cx_arr_delay_med_c180'], _ = _rolling(tt, g.arr_delay.to_numpy(), '180min')
        s = pd.Series(g.arr_copy.to_numpy(), index=pd.DatetimeIndex(tt))
        r['cx_arr_copy_rate_c180'] = s.rolling('180min', center=True, closed='both', min_periods=1).mean().to_numpy()
        prev, _ = _rolling(tt, g.dep_x.to_numpy(), '60min', center=False, closed='left')
        r['cx_dep_a3taxi_med_prev60'] = prev
        rwy = np.full(len(g), np.nan)
        runway = g.RUNWAY_mvt.astype(str).fillna('NA').to_numpy()
        dep_rows = (g.PHASE_mvt == 'DEP').to_numpy()
        for rw in pd.unique(runway[dep_rows]):
            m = dep_rows & (runway == rw)
            rwy[m], _ = _rolling(tt[m], g.dep_x.to_numpy()[m], '60min')
        r['cx_dep_a3taxi_med_rwy_c60'] = rwy
        # airport-local calendar day copy rates (arrivals only feed them)
        local = pd.DatetimeIndex(tt).tz_localize('UTC').tz_convert(common.AIRPORT_META[apt]['tz']).tz_localize(None).floor('D')
        day = pd.Series(local.asi8, index=g.index)
        copy = g.arr_copy
        day_rate = copy.groupby(day.values).mean()
        r['cx_arr_copy_rate_day'] = day.map(day_rate).to_numpy()
        key = pd.Series(day.astype(str).values + '|' + g.airline.astype(str).values, index=g.index)
        al = copy.groupby(key.values).agg(['sum', 'count'])
        al_sum = key.map(al['sum']).astype(float).fillna(0).to_numpy()
        al_n = key.map(al['count']).astype(float).fillna(0).to_numpy()
        base = r['cx_arr_copy_rate_day'].to_numpy()
        r['cx_arr_copy_rate_al_day'] = (al_sum + AL_STRENGTH * base) / (al_n + AL_STRENGTH)
        out.append(r[dep_rows])
    res = pd.concat(out, ignore_index=True)
    res['cx_dep_a3taxi_n_c60'] = res.cx_dep_a3taxi_n_c60.astype(float)
    res['cx_arr_taxiin_n_c60'] = res.cx_arr_taxiin_n_c60.astype(float)
    order = df.loc[is_dep, ['MVT_ID_mvt']]
    return order.merge(res, on='MVT_ID_mvt', how='left', validate='one_to_one')[['MVT_ID_mvt'] + COLUMNS].reset_index(drop=True)
