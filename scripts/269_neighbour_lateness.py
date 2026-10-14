"""Lead #21: departure-neighbour lateness (team versatile-violin's groups L / G, arnavhm/prc_26) on the v55 2025 proxy.

  py -3.14 scripts/269_neighbour_lateness.py frame              # runs/v56_neighbour/frame.parquet (both years, label-free)
  py -3.14 scripts/269_neighbour_lateness.py nested --arm nb    # nested residual test on y - v55 proxy

Columns, from the other departures' take-off (MVT), LOBT and schedule only (never a block time), around the row's own
take-off m:
  nb_lob_{p,f}{15,60}     mean (MVT - LOBT) of the airport's other departures taking off in [m - w, m) / (m, m + w]
  nb_sch_{p,f}60          mean (MVT - SCHED), same windows
  nb_rlob_{p,f}15         runway-level nb_lob over 15 min
  nb_own_lob_c60          own (MVT - LOBT) - mean of the airport's other departures within +-60 min (NaN without own LOBT)
  nb_own_sch_c60          same with the schedule
  rw_gapmax_{p,f}60       largest gap between consecutive take-offs of the runway in [m - 60 min, m] / [m, m + 60 min]
Test (scripts/266 protocol): r = y - v55 proxy on NM-matched 2025 rows with AOBT_3; LightGBM Huber residual model, outer
leave-one-fold-out, shrink by an inner leave-one-fold-out. 'base' = 266 BASE (p = v55, airport, adsb state, five queue
columns); 'nb' = base + the columns above; nb - base = their value. 'v55cols' / 'v55cols_nb' repeat it with v55's own
correction inputs (what is left after v55). No official score, nothing fitted on 2026.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

s266 = importlib.import_module('266_gate_retention_screen')
OUT = common.RUNS / 'v56_neighbour'
ROWS55 = common.RUNS / 'v55_build' / 'rows.parquet'
N_2025, R_V55 = 2085047, 232.2726   # rows and RMSE of the v55 2025 proxy
NB = ['nb_lob_p15', 'nb_lob_f15', 'nb_lob_p60', 'nb_lob_f60', 'nb_sch_p60', 'nb_sch_f60', 'nb_rlob_p15', 'nb_rlob_f15',
      'nb_own_lob_c60', 'nb_own_sch_c60', 'rw_gapmax_p60', 'rw_gapmax_f60']
V55COLS = [c for c in s266.BASE + s266.NEW if c != 'ret_gate_delay_15m'] + s266.OV
ALPHAS = np.array([0, .25, .5, .75, 1.0])


def _win_mean(ts, val, q, w, past):
    """Mean of val over sorted take-offs ts in [q - w, q) (past) or (q, q + w] (future); NaN-valued entries skipped."""
    ok = np.isfinite(val)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, val, 0.0))])
    cn = np.concatenate([[0.0], np.cumsum(ok.astype(float))])
    if past:
        lo, hi = np.searchsorted(ts, q - w, 'left'), np.searchsorted(ts, q, 'left')
    else:
        lo, hi = np.searchsorted(ts, q, 'right'), np.searchsorted(ts, q + w, 'right')
    n = cn[hi] - cn[lo]
    return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan), cs, cn


def _gapmax(ts, w):
    """Per sorted take-off: largest gap between consecutive take-offs within [t - w, t] and within [t, t + w]."""
    s = pd.Series(np.r_[0.0, np.diff(ts)], index=pd.to_datetime(ts, unit='s'))
    past = s.rolling(f'{int(w)}s', closed='both').max().to_numpy()
    # a gap ending at t_i belongs to [t - w, t] only if its start t_{i-1} >= t - w: the rolling window covers the end points,
    # so also cap it by the window length (a gap longer than w cannot sit inside the window)
    past = np.minimum(past, w)
    r = pd.Series(np.r_[np.diff(ts), 0.0][::-1], index=pd.to_datetime(-ts[::-1], unit='s'))
    fut = np.minimum(r.rolling(f'{int(w)}s', closed='both').max().to_numpy()[::-1], w)
    return past, fut


def frame() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ['MVT_ID_mvt', 'APT', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt']
    b = pd.concat([pd.read_parquet(common.DATA / f, columns=cols).assign(split=s)
                   for f, s in [('base_train.parquet', 'train'), ('base_rank.parquet', 'rank')]], ignore_index=True)
    m, sch, lobt = s266._epoch(b.MVT_TIME_UTC_mvt), s266._epoch(b.SCHED_TIME_UTC_mvt), s266._epoch(b.LOBT_flt)
    lob, sd = m - lobt, m - sch
    lob[np.abs(lob) > 4 * 3600] = np.nan          # date-shifted LOBT / schedule records do not describe the neighbourhood
    sd[np.abs(sd) > 6 * 3600] = np.nan
    res = {c: np.full(len(b), np.nan) for c in NB}
    for _, pos in b.groupby('APT').indices.items():
        pos = pos[np.isfinite(m[pos])]
        o = pos[np.argsort(m[pos], kind='stable')]
        ts = m[o]
        for w, tag in ((900, '15'), (3600, '60')):
            res[f'nb_lob_p{tag}'][o] = _win_mean(ts, lob[o], ts, w, True)[0]
            res[f'nb_lob_f{tag}'][o] = _win_mean(ts, lob[o], ts, w, False)[0]
        res['nb_sch_p60'][o] = _win_mean(ts, sd[o], ts, 3600, True)[0]
        res['nb_sch_f60'][o] = _win_mean(ts, sd[o], ts, 3600, False)[0]
        for src, dst in ((lob, 'nb_own_lob_c60'), (sd, 'nb_own_sch_c60')):
            v = src[o]
            ok = np.isfinite(v)
            cs = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))])
            cn = np.concatenate([[0.0], np.cumsum(ok.astype(float))])
            lo, hi = np.searchsorted(ts, ts - 3600, 'left'), np.searchsorted(ts, ts + 3600, 'right')
            n = cn[hi] - cn[lo] - ok
            mu = (cs[hi] - cs[lo] - np.where(ok, v, 0.0)) / np.maximum(n, 1)
            res[dst][o] = np.where(ok & (n > 0), v - mu, np.nan)
    key = (b.APT.astype(str) + '|' + b.RUNWAY_mvt.astype(str)).to_numpy()
    for _, pos in pd.Series(key).groupby(key).indices.items():
        pos = pos[np.isfinite(m[pos])]
        o = pos[np.argsort(m[pos], kind='stable')]
        ts = m[o]
        res['nb_rlob_p15'][o] = _win_mean(ts, lob[o], ts, 900, True)[0]
        res['nb_rlob_f15'][o] = _win_mean(ts, lob[o], ts, 900, False)[0]
        res['rw_gapmax_p60'][o], res['rw_gapmax_f60'][o] = _gapmax(ts, 3600.0)
    d = pd.DataFrame({'MVT_ID_mvt': b.MVT_ID_mvt, 'split': b.split, **res})
    d.to_parquet(OUT / 'frame.parquet', index=False)
    print(d.groupby('split')[NB].agg(['count', 'mean', 'median']).T.round(1).to_string(), flush=True)
    return d


def load() -> pd.DataFrame:
    rows = pd.read_parquet(ROWS55, columns=['MVT_ID_mvt', 'split', 'fold', 'y', 'v55_pred_s', 'v55_status'])
    rows = rows[rows.split == 'train'].drop(columns='split')
    f = pd.read_parquet(s266.OUT / 'frame.parquet').drop(columns='split')
    st = pd.read_parquet(s266.OPDI_FRAME, columns=['MVT_ID_mvt', 'adsb', 'month'])   # final-file change: was a literal RUNS/v53_opdi path
    old = pd.read_parquet(common.DATA / 'features_v5_train.parquet', columns=['MVT_ID_mvt', 'is_heli'] + s266.OLD)
    nb = pd.read_parquet(OUT / 'frame.parquet').drop(columns='split')
    ov = pd.read_parquet(s266.OUT / 'overtakes.parquet')
    d = rows.merge(f, on='MVT_ID_mvt').merge(st, on='MVT_ID_mvt').merge(old, on='MVT_ID_mvt').merge(nb, on='MVT_ID_mvt') \
        .merge(ov, on='MVT_ID_mvt', how='left')
    assert len(d) == N_2025
    d['p'] = d.v55_pred_s
    d['st'] = d.adsb.map(s266.STATES).astype(int)
    d['apt'] = d.apt.astype('category')
    return d


ARMS = {'base': s266.BASE, 'nb': s266.BASE + NB, 'v55cols': V55COLS, 'v55cols_nb': V55COLS + NB,
        'nb_lob': s266.BASE + [c for c in NB if 'lob' in c], 'nb_gap': s266.BASE + ['rw_gapmax_p60', 'rw_gapmax_f60']}


def nested(arm: str) -> dict:
    cols = ARMS[arm]
    d = load()
    scope = (d.v55_status == 'matched').to_numpy() & d.ret_overdue_apt.notna().to_numpy() & (d.is_heli.fillna(0) != 1).to_numpy()
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
            h = s266.fit(tr[ii], r[ii], cols).predict(tr.loc[io, cols])
            sse += np.array([((r[io] - a * h) ** 2).sum() for a in ALPHAS])
        a = float(ALPHAS[sse.argmin()])
        chosen[str(k)] = a
        h = (tr.fold == k).to_numpy()
        corr[h] = a * s266.fit(tr[o], r[o], cols).predict(tr.loc[h, cols])
        print(arm, k, a, flush=True)
    de = (r - corr) ** 2 - r ** 2
    sse0 = N_2025 * R_V55 ** 2
    body = tr.y.to_numpy() <= 3600

    def drmse(mask):
        return float(np.sqrt((sse0 + de[mask].sum()) / N_2025) - R_V55)
    rep = {'arm': arm, 'features': cols, 'rows': int(len(tr)), 'alpha_by_fold': chosen,
           'd_rmse_2025': drmse(np.ones(len(tr), bool)), 'd_rmse_body_y_le_3600': drmse(body), 'd_rmse_tail': drmse(~body),
           'dsse_M_by_fold': {str(k): round(float(de[(tr.fold == k).to_numpy()].sum() / 1e6), 2) for k in folds},
           'dsse_M_by_month': {str(m): round(float(de[(tr.month == m).to_numpy()].sum() / 1e6), 2) for m in sorted(tr.month.unique())},
           'dsse_M_by_apt': {str(a): round(float(de[(tr.apt == a).to_numpy()].sum() / 1e6), 2) for a in sorted(tr.apt.unique())},
           'top1pct_share_of_gain': float(np.sort(de)[:max(1, len(de) // 100)].sum() / de[de < 0].sum()) if (de < 0).any() else None,
           'rows_better_100s': int((np.abs(r) - np.abs(r - corr) > 100).sum()),
           'rows_worse_100s': int((np.abs(r - corr) - np.abs(r) > 100).sum())}
    rep['folds_worse'] = sum(v > 0 for v in rep['dsse_M_by_fold'].values())
    rep['months_worse'] = sum(v > 0 for v in rep['dsse_M_by_month'].values())
    if arm != 'base':
        g = s266.fit(tr, r, cols).feature_importance('gain')
        rep['gain_share'] = {c: round(float(v / g.sum()), 4) for c, v in sorted(zip(cols, g), key=lambda z: -z[1])}
    (OUT / f'nested_{arm}.json').write_text(json.dumps(rep, indent=2))
    pd.DataFrame({'MVT_ID_mvt': tr.MVT_ID_mvt, 'corr': corr}).to_parquet(OUT / f'oof_{arm}.parquet', index=False)
    print(json.dumps({k: v for k, v in rep.items() if k not in ('features', 'dsse_M_by_month')}), flush=True)
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['frame', 'nested'])
    ap.add_argument('--arm', default='nb', choices=list(ARMS))
    a = ap.parse_args()
    frame() if a.cmd == 'frame' else nested(a.arm)
