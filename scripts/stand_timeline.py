"""G10 same-stand timeline features from arrival in-block times (fully known in the ranking file).

For a departure from stand S that took off at t (only rank-available times are used; the
departure's own off-block is never read):

  st_last_in_gap      t - last arrival in-block at S with in-block <= t
  st_last2_in_gap     t - second-to-last such in-block
  st_next_in_gap      first arrival in-block at S after t - t
  st_n_in_60          arrival in-blocks at S within [t - 60 min, t]
  st_last_same_type   last arrival at S before t has this departure's aircraft type
  st_last_same_al     last arrival at S before t has this departure's airline prefix
  st_turn_in_gap      t - in-block of the latest same-airline, same-type arrival at S that
                      in-blocked no later than min(t, scheduled off-block + 2 h), within 20 h
  st_turn_in_delay    that arrival's in-block - its scheduled in-block
  st_prev_dep_gap     t - previous departure take-off from S

Why: about 10 % of departures (EGLL 26 %) have the next aircraft in-blocked at the same stand
before they took off, which bounds the off-block from above; the inbound turnaround bounds it
from below. Gap features are clipped at 12 h because longer gaps carry no timing information and
the ranking file jumps from 31 January to 1 July.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GAP_CLIP = 43200.0
TURN_LOOKBACK = 72000.0
COLUMNS = ['st_last_in_gap', 'st_last2_in_gap', 'st_next_in_gap', 'st_n_in_60', 'st_last_same_type',
           'st_last_same_al', 'st_turn_in_gap', 'st_turn_in_delay', 'st_prev_dep_gap']
GAP_COLUMNS = ['st_last_in_gap', 'st_last2_in_gap', 'st_next_in_gap', 'st_turn_in_gap', 'st_prev_dep_gap']


def _secs(s):
    s = pd.to_datetime(s)
    return (s.astype('int64') // 1_000_000).astype('float64').where(s.notna()).to_numpy()


def build(movements: pd.DataFrame) -> pd.DataFrame:
    """Return one row per DEP movement: MVT_ID_mvt + COLUMNS."""
    df = movements.copy()
    df['APT'] = np.where(df.PHASE_mvt == 'DEP', df.ADEP_mvt, df.ADES_mvt)
    df['key'] = df.APT.astype(str) + '|' + df.STAND_mvt.fillna('').astype(str)
    df['al'] = df.FLIGHT_mvt.fillna('').astype(str).str.extract(r'^([A-Z]{3}|[A-Z0-9]{2})', expand=False).fillna('')
    df['t'] = _secs(df.MVT_TIME_UTC_mvt); df['ts'] = _secs(df.SCHED_TIME_UTC_mvt)
    dep = df[df.PHASE_mvt == 'DEP'].reset_index(drop=True)
    arr = df[df.PHASE_mvt == 'ARR'].copy()
    arr['tb'] = _secs(arr.BLOCK_TIME_UTC_mvt)
    arr = arr[np.isfinite(arr.tb)].sort_values(['key', 'tb'], kind='stable').reset_index(drop=True)
    n = len(dep)
    out = {c: np.full(n, np.nan) for c in COLUMNS}
    arr_idx = arr.groupby('key', sort=False).indices
    a_tb = arr.tb.to_numpy(); a_ts = arr.ts.to_numpy()
    a_type = arr.AIRCRAFT_TYPE_mvt.fillna('').astype(str).to_numpy(); a_al = arr.al.to_numpy()
    d_t = dep.t.to_numpy(); d_ts = dep.ts.to_numpy()
    d_type = dep.AIRCRAFT_TYPE_mvt.fillna('').astype(str).to_numpy(); d_al = dep.al.to_numpy()
    for key, di in dep.groupby('key', sort=False).indices.items():
        if key.endswith('|') or key.endswith('|NA'):
            continue
        order = di[np.argsort(d_t[di], kind='stable')]
        tt = d_t[order]
        out['st_prev_dep_gap'][order] = tt - np.concatenate([[np.nan], tt[:-1]])
        if key not in arr_idx:
            continue
        ai = arr_idx[key]; ab = a_tb[ai]
        pos = np.searchsorted(ab, d_t[di], side='right')
        has1, has2, hasn = pos >= 1, pos >= 2, pos < len(ab)
        out['st_last_in_gap'][di[has1]] = d_t[di[has1]] - ab[pos[has1] - 1]
        out['st_last2_in_gap'][di[has2]] = d_t[di[has2]] - ab[pos[has2] - 2]
        out['st_next_in_gap'][di[hasn]] = ab[pos[hasn]] - d_t[di[hasn]]
        out['st_n_in_60'][di] = pos - np.searchsorted(ab, d_t[di] - 3600, side='left')
        last = ai[np.maximum(pos - 1, 0)]
        out['st_last_same_type'][di[has1]] = (a_type[last[has1]] == d_type[di[has1]]).astype(float)
        out['st_last_same_al'][di[has1]] = (a_al[last[has1]] == d_al[di[has1]]).astype(float)
        for row in di:
            limit = min(d_t[row], d_ts[row] + 7200.0) if np.isfinite(d_ts[row]) else d_t[row]
            p = np.searchsorted(ab, limit, side='right') - 1
            steps = 0
            while p >= 0 and steps < 6:
                cand = ai[p]
                if d_t[row] - a_tb[cand] > TURN_LOOKBACK:
                    break
                if a_al[cand] == d_al[row] and a_type[cand] == d_type[row]:
                    out['st_turn_in_gap'][row] = d_t[row] - a_tb[cand]
                    out['st_turn_in_delay'][row] = a_tb[cand] - a_ts[cand]
                    break
                p -= 1; steps += 1
    res = pd.DataFrame({'MVT_ID_mvt': dep.MVT_ID_mvt.to_numpy(), **out})
    for c in GAP_COLUMNS:
        res[c] = res[c].clip(upper=GAP_CLIP)
    return res
