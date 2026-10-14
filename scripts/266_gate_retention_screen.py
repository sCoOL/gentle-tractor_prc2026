"""Gate-retention / runway-cadence screen on the v54 2025 proxy (idea read in team outgoing-boat's public repo, src/pista.py).

  py -3.14 scripts/266_gate_retention_screen.py frame     # runs/v55_retention/frame.parquet (both years, label-free)
  py -3.14 scripts/266_gate_retention_screen.py nested    # nested residual test, arms base / ret -> nested_<arm>.json

Columns, all at t = the row's AOBT_3 (NaN without AOBT_3), from the other departures' AOBT_3 / EOBT_1 / SCHED / take-off
(never the block time):
  ret_overdue_apt / _rwy : departures of the airport (same runway) with EOBT_1 <= t < AOBT_3 (guards: EOBT <= AOBT,
                           AOBT - EOBT < 7,200 s, take-off >= AOBT); the row itself cancels.
  ret_active_rwy         : same-runway departures with AOBT_3 <= t < take-off, the row itself removed.
  ret_gate_delay_15m     : mean AOBT_3 - SCHED of the airport's departures with AOBT_3 in [t - 15 min, t).
  ret_ewma_dep           : airport take-offs per hour before t, exponential weights, half-life 10 min.
  rw_span20              : seconds from the airport's 20th previous take-off to the row's take-off.
  rw_heavy_between       : same-runway H/J take-offs in (AOBT_3, take-off].
Test: r = y - v54 proxy on 2025 rows with AOBT_3 (status 'matched'); LightGBM Huber residual model, outer
leave-one-fold-out, shrink picked by an inner leave-one-fold-out (scripts/262 protocol). Arm 'base' = p, airport, adsb.lol
state and the existing queue columns; arm 'ret' = base + the new columns. The difference ret - base is the new columns'
value. No official score, nothing fitted on 2026.
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

OUT = common.RUNS / 'v55_retention'
ROWS = common.RUNS / 'v54_build' / 'rows.parquet'
OPDI_FRAME = common.RUNS / 'v53_opdi' / 'frame.parquet'   # final-file change: was a literal in load()/spent(); final_chain repoints it
N_2025, RMSE_V54_PROXY = 2085047, 232.4905   # rows and RMSE of the v54 2025 proxy
HALF, MAX_WAIT, WIN_DELAY = 600.0, 7200.0, 900.0
NEW = ['ret_overdue_apt', 'ret_overdue_rwy', 'ret_active_rwy', 'ret_gate_delay_15m', 'ret_ewma_dep', 'rw_span20', 'rw_heavy_between']
OLD = ['n_out_at_aobt3', 'push_prev15_at_aobt3', 'dep_same_rwy_between_aobt3_atot', 'atot_minus_aobt3', 'aobt3_minus_sched']
STATES = {'none': 0, 'censored': 1, 'seen': 2, 'pushback': 3}
BASE = ['p', 'apt', 'st'] + OLD
ALPHAS = np.array([0, .25, .5, .75, 1.0])


def _epoch(s):
    return ((pd.to_datetime(s, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def _count_le(sorted_t, q):
    return np.searchsorted(sorted_t, q, 'right').astype(float)


def _ewma(mvt, q):
    """sum over take-offs before q of 2^((mvt - q)/HALF), as a per-hour rate; 6 h chunks with a local origin."""
    out = np.zeros(len(q))
    chunk, cut = 21600.0, 12 * HALF
    key = np.floor(q / chunk)
    for c in np.unique(key[np.isfinite(key)]):
        rows = np.where(key == c)[0]
        org = c * chunk
        i0, i1 = np.searchsorted(mvt, org - cut, 'left'), np.searchsorted(mvt, org + chunk, 'left')
        ev = mvt[i0:i1]
        cs = np.concatenate([[0.0], np.cumsum(np.exp2((ev - org) / HALF))])
        tq = q[rows]
        hi, lo = np.searchsorted(ev, tq, 'left'), np.searchsorted(ev, tq - cut, 'left')
        out[rows] = np.exp2((org - tq) / HALF) * (cs[hi] - cs[lo])
    return out * (3600.0 * np.log(2.0) / HALF)


def frame() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ['MVT_ID_mvt', 'APT', 'RUNWAY_mvt', 'WK_TBL_CAT_flt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'EOBT_1_flt', 'AOBT_3_flt']
    b = pd.concat([pd.read_parquet(common.DATA / f, columns=cols).assign(split=s)
                   for f, s in [('base_train.parquet', 'train'), ('base_rank.parquet', 'rank')]], ignore_index=True)
    mvt, aobt, eobt, sch = _epoch(b.MVT_TIME_UTC_mvt), _epoch(b.AOBT_3_flt), _epoch(b.EOBT_1_flt), _epoch(b.SCHED_TIME_UTC_mvt)
    heavy = b.WK_TBL_CAT_flt.isin(['H', 'J']).to_numpy()
    rwy = (b.APT.astype(str) + '|' + b.RUNWAY_mvt.astype(str)).to_numpy()
    res = {c: np.full(len(b), np.nan) for c in NEW}
    wait = aobt - eobt
    good_q = np.isfinite(wait) & (wait >= 0) & (wait < MAX_WAIT) & (mvt >= aobt)
    act = np.isfinite(aobt) & np.isfinite(mvt) & (aobt <= mvt)
    own_active = (np.isfinite(mvt) & (mvt > aobt)).astype(float)
    for apt, pos in b.groupby('APT').indices.items():
        t = aobt[pos]
        g = pos[good_q[pos]]
        res['ret_overdue_apt'][pos] = _count_le(np.sort(eobt[g]), t) - _count_le(np.sort(aobt[g]), t)
        gd = pos[np.isfinite(aobt[pos]) & np.isfinite(sch[pos])]
        o = np.argsort(aobt[gd], kind='stable')
        when = aobt[gd][o]
        cs = np.concatenate([[0.0], np.cumsum((aobt[gd] - sch[gd])[o])])
        hi, lo = np.searchsorted(when, t, 'left'), np.searchsorted(when, t - WIN_DELAY, 'left')
        n = hi - lo
        res['ret_gate_delay_15m'][pos] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
        m_sorted = np.sort(mvt[pos][np.isfinite(mvt[pos])])
        res['ret_ewma_dep'][pos] = np.where(np.isfinite(t), _ewma(m_sorted, np.nan_to_num(t, nan=0.0)), np.nan)
        i = np.searchsorted(m_sorted, mvt[pos], 'left') - 20
        res['rw_span20'][pos] = np.where(i >= 0, mvt[pos] - m_sorted[np.clip(i, 0, None)], np.nan)
    for r, pos in pd.Series(rwy).groupby(rwy).indices.items():
        t = aobt[pos]
        g = pos[good_q[pos]]
        res['ret_overdue_rwy'][pos] = _count_le(np.sort(eobt[g]), t) - _count_le(np.sort(aobt[g]), t)
        a = pos[act[pos]]
        res['ret_active_rwy'][pos] = np.maximum(_count_le(np.sort(aobt[a]), t) - _count_le(np.sort(mvt[a]), t) - own_active[pos], 0.0)
        hv = np.sort(mvt[pos][heavy[pos] & np.isfinite(mvt[pos])])
        res['rw_heavy_between'][pos] = _count_le(hv, mvt[pos]) - _count_le(hv, t) - heavy[pos].astype(float)
    for c in NEW:
        if c.startswith('ret_') or c == 'rw_heavy_between':
            res[c][~np.isfinite(aobt)] = np.nan
    d = pd.DataFrame({'MVT_ID_mvt': b.MVT_ID_mvt, 'split': b.split, 'apt': b.APT.astype(str), **res})
    d['rw_heavy_between'] = d.rw_heavy_between.clip(lower=0)
    d.loc[d.rw_span20 > 21600, 'rw_span20'] = np.nan   # the Jan 31 -> Jul 1 2026 gap in the rank file
    d.to_parquet(OUT / 'frame.parquet', index=False)
    print(d.groupby('split')[NEW].describe().T.to_string(), flush=True)
    return d


OV = ['ov_passed_me', 'ov_i_passed']


def _pairs_count(a, m, lo, hi, want_less):
    """For each i: among sorted positions lo[i]..hi[i]-1, how many have m_j < m_i (want_less) or m_j > m_i."""
    n = hi - lo
    out = np.zeros(len(a))
    k = n > 0
    if not k.any():
        return out
    rows = np.repeat(np.where(k)[0], n[k])
    start = np.repeat(lo[k], n[k])
    offs = np.arange(len(rows)) - np.repeat(np.cumsum(n[k]) - n[k], n[k])
    j = start + offs
    hit = (m[j] < m[rows]) if want_less else (m[j] > m[rows])
    np.add.at(out, rows[hit], 1.0)
    return out


def overtakes() -> pd.DataFrame:
    """Own idea after their queue columns: same-runway order changes between push (AOBT_3) and take-off.

      ov_passed_me : departures that pushed AFTER my AOBT_3 but took off BEFORE me
      ov_i_passed  : departures that pushed BEFORE my AOBT_3 (within 2 h) but took off AFTER me
    Many 'passed me' with a normal-looking t - AOBT_3 suggests my AOBT_3 is earlier than my real off-block (held, remote
    de-icing, start-up request logged as off-block). Ties in AOBT_3 (minute resolution) count as neither.
    """
    cols = ['MVT_ID_mvt', 'APT', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt']
    b = pd.concat([pd.read_parquet(common.DATA / f, columns=cols) for f in ('base_train.parquet', 'base_rank.parquet')], ignore_index=True)
    a, m = _epoch(b.AOBT_3_flt), _epoch(b.MVT_TIME_UTC_mvt)
    ok = np.isfinite(a) & np.isfinite(m) & (m >= a) & (m - a < MAX_WAIT)
    res = {c: np.full(len(b), np.nan) for c in OV}
    key = (b.APT.astype(str) + '|' + b.RUNWAY_mvt.astype(str)).to_numpy()
    for _, pos in pd.Series(key).groupby(key).indices.items():
        pos = pos[ok[pos]]
        o = np.argsort(a[pos], kind='stable')
        pa, pm = a[pos][o], m[pos][o]
        hi1 = np.searchsorted(pa, pm, 'left')            # a_j < m_i
        lo1 = np.searchsorted(pa, pa, 'right')           # a_j > a_i
        lo2 = np.searchsorted(pa, pa - MAX_WAIT, 'left')  # a_j >= a_i - 2 h
        hi2 = np.searchsorted(pa, pa, 'left')            # a_j < a_i
        res['ov_passed_me'][pos[o]] = _pairs_count(pa, pm, lo1, np.maximum(hi1, lo1), True)
        res['ov_i_passed'][pos[o]] = _pairs_count(pa, pm, lo2, hi2, False)
    d = pd.DataFrame({'MVT_ID_mvt': b.MVT_ID_mvt, **res})
    d.to_parquet(OUT / 'overtakes.parquet', index=False)
    print(d[OV].describe().to_string(), flush=True)
    return d


def load() -> pd.DataFrame:
    rows = pd.read_parquet(ROWS, columns=['MVT_ID_mvt', 'split', 'fold', 'y', 'v54_pred_s', 'v54_status'])
    rows = rows[rows.split == 'train']
    f = pd.read_parquet(OUT / 'frame.parquet').drop(columns='split')
    st = pd.read_parquet(OPDI_FRAME, columns=['MVT_ID_mvt', 'adsb', 'month'])
    old = pd.read_parquet(common.DATA / 'features_v5_train.parquet', columns=['MVT_ID_mvt'] + OLD)
    d = rows.merge(f, on='MVT_ID_mvt').merge(st, on='MVT_ID_mvt').merge(old, on='MVT_ID_mvt')
    if (OUT / 'overtakes.parquet').exists():
        d = d.merge(pd.read_parquet(OUT / 'overtakes.parquet'), on='MVT_ID_mvt', how='left')
    assert len(d) == N_2025
    d['p'] = d.v54_pred_s
    d['st'] = d.adsb.map(STATES).astype(int)
    d['apt'] = d.apt.astype('category')
    return d


def fit(x, r, cols, seed=7):
    prm = dict(objective='huber', alpha=300.0, learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=seed, num_threads=18)
    return lgb.train(prm, lgb.Dataset(x[cols], r, categorical_feature=[c for c in ('apt', 'st') if c in cols]), num_boost_round=250)


def nested(arm: str) -> dict:
    cols = (BASE + (NEW if arm.startswith('ret') else []))
    if arm == 'ov':               # base + the two overtake columns
        cols = BASE + OV
    if arm == 'ret6ov':           # ret without the gate-delay column, plus the overtake columns
        cols = [c for c in BASE + NEW if c != 'ret_gate_delay_15m'] + OV
    if arm.startswith('add_'):    # base + one new column
        cols = BASE + [arm[4:]]
    if arm.startswith('drop_'):   # ret without one new column
        cols = [c for c in BASE + NEW if c != arm[5:]]
    if arm.endswith('_nop'):   # without p: the 2025 proxy (A51 OOF) and the 2026 base file can be different functions of the inputs
        cols = [c for c in cols if c != 'p']
    d = load()
    scope = (d.v54_status == 'matched').to_numpy() & d.ret_overdue_apt.notna().to_numpy()
    tr = d[scope].reset_index(drop=True)
    r = (tr.y - tr.p).to_numpy()
    folds = sorted(tr.fold.unique())
    corr, chosen = np.zeros(len(tr)), {}
    for k in folds:
        o = (tr.fold != k).to_numpy()
        sse = np.zeros(len(ALPHAS))
        for j in folds:
            if j == k:
                continue
            ii, io = o & (tr.fold != j).to_numpy(), (tr.fold == j).to_numpy()
            h = fit(tr[ii], r[ii], cols).predict(tr.loc[io, cols])
            sse += np.array([((r[io] - a * h) ** 2).sum() for a in ALPHAS])
        a = float(ALPHAS[sse.argmin()])
        chosen[str(k)] = a
        h = (tr.fold == k).to_numpy()
        corr[h] = a * fit(tr[o], r[o], cols).predict(tr.loc[h, cols])
        print(arm, k, a, flush=True)
    de = (r - corr) ** 2 - r ** 2
    sse0 = N_2025 * RMSE_V54_PROXY ** 2
    body = (np.abs(tr.y.to_numpy()) <= 3600)

    def drmse(mask):
        return float(np.sqrt((sse0 + de[mask].sum()) / N_2025) - RMSE_V54_PROXY)
    rep = {'arm': arm, 'features': cols, 'rows': int(len(tr)), 'alpha_by_fold': chosen,
           'd_rmse_2025': drmse(np.ones(len(tr), bool)), 'd_rmse_body_y_le_3600': drmse(body), 'd_rmse_tail': drmse(~body),
           'dsse_M_by_fold': {str(k): round(float(de[(tr.fold == k).to_numpy()].sum() / 1e6), 2) for k in folds},
           'dsse_M_by_month': {str(m): round(float(de[(tr.month == m).to_numpy()].sum() / 1e6), 2) for m in sorted(tr.month.unique())},
           'dsse_M_by_apt': {str(a): round(float(de[(tr.apt == a).to_numpy()].sum() / 1e6), 2) for a in sorted(tr.apt.unique())},
           'dsse_M_by_state': {s: round(float(de[(tr.adsb == s).to_numpy()].sum() / 1e6), 2) for s in STATES},
           'top1pct_share_of_gain': float(np.sort(de)[:max(1, len(de) // 100)].sum() / de[de < 0].sum()) if (de < 0).any() else None,
           'rows_better_100s': int((np.abs(r) - np.abs(r - corr) > 100).sum()),
           'rows_worse_100s': int((np.abs(r - corr) - np.abs(r) > 100).sum())}
    if arm.startswith('ret') or arm == 'ov':
        m = fit(tr, r, cols)
        g = m.feature_importance('gain')
        rep['gain_share'] = {c: round(float(v / g.sum()), 4) for c, v in sorted(zip(cols, g), key=lambda z: -z[1])}
    (OUT / f'nested_{arm}.json').write_text(json.dumps(rep, indent=2))
    pd.DataFrame({'MVT_ID_mvt': tr.MVT_ID_mvt, 'corr': corr}).to_parquet(OUT / f'oof_{arm}.parquet', index=False)
    print(json.dumps(rep), flush=True)
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['frame', 'overtakes', 'nested'])
    ap.add_argument('--arm', default='ret')   # base | ret | base_nop | ret_nop | add_<col> | drop_<col>
    a = ap.parse_args()
    {'frame': frame, 'overtakes': overtakes, 'nested': lambda: nested(a.arm)}[a.cmd]()
