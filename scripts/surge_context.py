"""Surge and gap context for tree models (2026-09-19).

For a departure that took off at t, every input is rank-available (a departure's block time is never read):

  sg_dep_c60 / _n       mean over OTHER departures with take-off in [t - 30 min, t + 30 min) of (NM taxi minus that
                        departure's stand x runway normal), clipped +-1200 s; and how many there were
  sg_dep_prev60         the same over take-offs in [t - 60 min, t)
  sg_dep_c180           the same over [t - 90 min, t + 90 min)
  sg_dep_rwy_c60        the same restricted to the departure's runway
  sg_arr_c60 / _n       mean over arrivals landing in [t - 30 min, t + 30 min) of (taxi-in minus that arrival's
                        stand x runway normal), clipped +-900 s
  sg_arr_prev120        the same over landings in [t - 120 min, t)
  self_nm_norm          the departure's own normal NM taxi (a target-free stand x runway baseline)
  self_nm_res           its own NM taxi minus that normal
  gap_prev{k}/next{k}   seconds to the k-th previous / next departure take-off at the airport (k = 1, 2, 3, 5)
  mov_pm60 / dep_pm60   movements / departures within +-60 min

Normals are nbr_set's 2025 medians per (airport, stand, runway) with pier / runway / airport fallbacks, so 2026
departures are measured against 2025 normals. ``lead_in`` lets 1 January 2026 see the end of December 2025.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import nbr_set

COLUMNS = ['sg_dep_c60', 'sg_dep_c60_n', 'sg_dep_prev60', 'sg_dep_c180', 'sg_dep_rwy_c60', 'sg_arr_c60', 'sg_arr_c60_n',
           'sg_arr_prev120', 'self_nm_norm', 'self_nm_res',
           'gap_prev1', 'gap_prev2', 'gap_prev3', 'gap_prev5', 'gap_next1', 'gap_next2', 'gap_next3', 'gap_next5',
           'mov_pm60', 'dep_pm60']
GAP_CLIP = 21600.0          # the ranking file jumps from 31 January to 1 July


def win_mean(tq, ts, v, lo, hi, self_idx=None):
    """Mean and count of v (NaN skipped) over sorted source times ts in [tq + lo, tq + hi); self_idx excludes each
    query's own source row."""
    ok = np.isfinite(v); cs = np.r_[0, np.cumsum(np.where(ok, v, 0.0))]; cn = np.r_[0, np.cumsum(ok)]
    i0 = np.searchsorted(ts, tq + lo, 'left'); i1 = np.searchsorted(ts, tq + hi, 'left')
    s = cs[i1] - cs[i0]; n = (cn[i1] - cn[i0]).astype('float64')
    if self_idx is not None:
        inside = (ts[self_idx] >= tq + lo) & (ts[self_idx] < tq + hi) & ok[self_idx]
        s = s - np.where(inside, v[self_idx], 0.0); n = n - inside
    with np.errstate(invalid='ignore', divide='ignore'):
        m = np.where(n > 0, s / n, np.nan)
    return m, n


def build(which, lead_in=0.0):
    """One row per departure of the 2025 training year or the 2026 ranking file: MVT_ID_mvt + COLUMNS."""
    dep, arr = nbr_set._load(which)
    dep = dep.assign(_ctx=False); arr = arr.assign(_ctx=False)
    if which == 'rank' and lead_in > 0:
        pdep, parr = nbr_set._load('train')
        start = pd.Timestamp('2026-01-01', tz='UTC') - pd.Timedelta(seconds=float(lead_in))
        pdep = pdep[pd.to_datetime(pdep.MVT_TIME_UTC_mvt, utc=True) >= start].assign(_ctx=True)
        parr = parr[pd.to_datetime(parr.MVT_TIME_UTC_mvt, utc=True) >= start].assign(_ctx=True)
        dep = pd.concat([pdep, dep], ignore_index=True); arr = pd.concat([parr, arr], ignore_index=True)
    nm, ti = nbr_set._taxis(dep, arr)
    dep['nm_norm'] = nbr_set._normal('dep', dep); arr['ti_norm'] = nbr_set._normal('arr', arr)
    dep['nm_res'] = np.clip(nm - dep.nm_norm.values, -1200, 1200); arr['ti_res'] = np.clip(ti - arr.ti_norm.values, -900, 900)
    dep['t'] = nbr_set._sec(dep.MVT_TIME_UTC_mvt); arr['t'] = nbr_set._sec(arr.MVT_TIME_UTC_mvt)
    out = []
    for apt in sorted(dep.APT.unique()):
        dd = dep[(dep.APT.values == apt) & np.isfinite(dep.t.values)].sort_values('t', kind='stable').reset_index(drop=True)
        aa = arr[(arr.APT.values == apt) & np.isfinite(arr.t.values)].sort_values('t', kind='stable').reset_index(drop=True)
        td, ta = dd.t.to_numpy(), aa.t.to_numpy(); me = np.arange(len(dd)); res = dd.nm_res.to_numpy()
        f = pd.DataFrame({'MVT_ID_mvt': dd.MVT_ID_mvt.to_numpy()})
        f['sg_dep_c60'], f['sg_dep_c60_n'] = win_mean(td, td, res, -1800, 1800, me)
        f['sg_dep_prev60'], _ = win_mean(td, td, res, -3600, 0, me)
        f['sg_dep_c180'], _ = win_mean(td, td, res, -5400, 5400, me)
        rw = dd.RUNWAY_mvt.astype(str).to_numpy(); sg_rwy = np.full(len(dd), np.nan)
        for r in np.unique(rw):
            m = np.where(rw == r)[0]
            sg_rwy[m], _ = win_mean(td[m], td[m], res[m], -1800, 1800, np.arange(len(m)))
        f['sg_dep_rwy_c60'] = sg_rwy
        f['sg_arr_c60'], f['sg_arr_c60_n'] = win_mean(td, ta, aa.ti_res.to_numpy(), -1800, 1800)
        f['sg_arr_prev120'], _ = win_mean(td, ta, aa.ti_res.to_numpy(), -7200, 0)
        f['self_nm_norm'] = dd.nm_norm.to_numpy(); f['self_nm_res'] = res
        for k in (1, 2, 3, 5):
            p = me - k; n = me + k
            f[f'gap_prev{k}'] = np.where(p >= 0, np.minimum(td - td[np.clip(p, 0, None)], GAP_CLIP), np.nan)
            f[f'gap_next{k}'] = np.where(n < len(td), np.minimum(td[np.clip(n, None, len(td) - 1)] - td, GAP_CLIP), np.nan)
        _, nd = win_mean(td, td, np.ones(len(td)), -3600, 3600); _, na = win_mean(td, ta, np.ones(len(ta)), -3600, 3600)
        f['mov_pm60'] = nd + na; f['dep_pm60'] = nd
        out.append(f[~dd._ctx.to_numpy(dtype=bool)])
    return pd.concat(out, ignore_index=True)[['MVT_ID_mvt'] + COLUMNS]
