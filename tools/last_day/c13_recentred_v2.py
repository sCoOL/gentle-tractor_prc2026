"""LIRF copy / held enumeration with the day residual re-centred per month (functions used by the post-steps).

c10_daily_v2.discrete_post enumerates, for LIRF unmatched rows with 3,600 < D <= 43,200 s (1..14 per off-block UTC
day), every copy / held assignment and scores it by the day's residual R_L = K - S_O. A month-level offset of the
day residual (drift: later flights, a recording change) would be read as "some copies were held". Re-centring: before
enumerating, R_L -= the median over the eligible days of the same airport-month of the day's model residual
K - S_all (S_all = S_O + the L rows' own sum max(0, D - p), i.e. version 1's R); eligible = the days the enumeration
uses; >= 5 days, else no shift. Label-free in both years (K and the incoming p only).
c16_stack_recentred.discrete_at is the same computation at the stack output; `discrete_post_rc` is kept here as the
reference it is checked against (step 0 of c16). `pass_judge` is the pass rule of runs/overnight_20261008/ov.py judge().
"""
import numpy as np
import pandas as pd
import ld
import c10_daily_v2 as c10

MIN_DAYS = 5
DIAG = []   # per discrete_post call: month offsets (filled when RECORD[0] is True)
RECORD = [False]


def discrete_post_rc(c, P, EA, fixed=None, recentre='off'):
    """c10.discrete_post with the eligible days' R re-centred per (year-)month. recentre='off' is c10's code path."""
    NA, ND, DAY0, LIRF, MMAX = c10.NA, c10.ND, c10.DAY0, c10.LIRF, c10.MMAX
    p = c['p0']
    s2 = c10.s2_of(c, p, P, 'v1', fixed)
    di = np.floor((c['t'] - p) / 86400).astype(np.int64) - DAY0
    gid = c['apt'] * ND + di
    N, s2N = c10.N_of(c, P), c10._lookup(c, P['S_lv'], P['S_gl'])
    D = c['D']
    L = (c['apt'] == LIRF) & (c['lane'] == 1) & (D > 3600) & (D <= 43200) & (D - N > 60)
    if fixed is not None:
        L &= ~np.asarray(fixed, bool)
    gap = D - p
    act = gap > -np.sqrt(s2)
    O = ~L
    S_O = np.bincount(gid[O], np.maximum(0, gap[O]), NA * ND)
    V_O = np.bincount(gid[O], np.where(act[O], s2[O], 0.0), NA * ND)
    n = np.bincount(gid, minlength=NA * ND)
    K, F = EA
    ok = (n > 0) & np.isfinite(K) & (np.abs(n - F) <= 0.02 * F)
    sK2 = P['sK2'][LIRF]
    w = np.clip((p - N) / np.where(D - N > 60, D - N, 1), 0.001, 0.999)
    prior = w * D + (1 - w) * N
    delta = np.zeros(len(p))
    elig = np.zeros(len(p), bool)
    idxL = np.where(L)[0]
    groups = [(g, rows.to_numpy()) for g, rows in pd.Series(idxL).groupby(gid[idxL])]
    groups = [(g, rows) for g, rows in groups if ok[g] and len(rows) <= MMAX]
    # re-centring: per year-month of the off-block day, over the eligible days
    gs = np.array([g for g, _ in groups], np.int64)
    Rg = np.array([K[g] - S_O[g] for g in gs]) if len(gs) else np.zeros(0)
    # the model residual of the day (c9's R = K - S over ALL rows at the incoming p): R_L minus the L rows' own implied
    # sum max(0, D - p) (= their prior-mean contribution D - prior when w is not clipped)
    EL = np.array([np.maximum(0, gap[rows]).sum() for _, rows in groups]) if len(gs) else np.zeros(0)
    Res = Rg - EL
    mon = pd.to_datetime((gs % ND + DAY0) * 86400, unit='s').strftime('%Y-%m').to_numpy() if len(gs) else np.array([], str)
    off = np.zeros(len(gs))
    diag = {}
    for m_ in np.unique(mon):
        sel = mon == m_
        r, e = Rg[sel], Res[sel]
        o = float(np.median(e)) if (recentre == 'median' and sel.sum() >= MIN_DAYS) else 0.0
        off[sel] = o
        diag[m_] = {'days': int(sel.sum()), 'RL_mean_min': round(float(r.mean()) / 60, 1), 'RL_median_min': round(float(np.median(r)) / 60, 1),
                    'resid_mean_min': round(float(e.mean()) / 60, 1), 'resid_median_min': round(float(np.median(e)) / 60, 1),
                    'offset_min': round(o / 60, 1)}
    if RECORD[0]:
        DIAG.append(diag)
    for (g, rows), o in zip(groups, off):
        m = len(rows)
        R, v0 = K[g] - S_O[g] - o, sK2 + V_O[g]
        B = ((np.arange(2 ** m)[:, None] >> np.arange(m)[None, :]) & 1).astype(float)   # 1 = held
        mu, sn, wr = D[rows] - N[rows], s2N[rows], w[rows]
        mean, var = B @ mu, v0 + B @ sn
        lw = B @ np.log(1 - wr) + (1 - B) @ np.log(wr) - 0.5 * np.log(var) - 0.5 * (R - mean) ** 2 / var
        pw = np.exp(lw - lw.max())
        pw /= pw.sum()
        r = R - mean
        Ey = (1 - B) * D[rows][None, :] + B * (N[rows][None, :] - sn[None, :] * (r / var)[:, None])
        delta[rows] = pw @ Ey - prior[rows]
        elig[rows] = True
    return elig, delta


def pass_judge(a, b6, bf, p6, pf, name):
    """runs/overnight_20261008/ov.py judge() (same rounding and rule), base = b6 / bf."""
    y = a.y.to_numpy()
    e6 = (y - p6) ** 2 - (y - b6) ** 2
    ef = (y - pf) ** 2 - (y - bf) ** 2
    h1 = (a.month <= 6).to_numpy()
    fold = (pd.Series(e6).groupby(a.fold.to_numpy()).sum() / 1e6).round(2)
    res = {'name': name, 'sixfold_dsse_M': round(e6.sum() / 1e6, 2), 'by_fold_M': fold.to_dict(), 'folds_better': int((fold < 0).sum()),
           'forward_H1toH2_M': round(ef[~h1].sum() / 1e6, 2), 'forward_H2toH1_M': round(ef[h1].sum() / 1e6, 2),
           'jan_sixfold_M': round(e6[(a.month == 1).to_numpy()].sum() / 1e6, 2), 'jan_forward_M': round(ef[(a.month == 1).to_numpy()].sum() / 1e6, 2),
           'feb_sixfold_M': round(e6[(a.month == 2).to_numpy()].sum() / 1e6, 2), 'feb_forward_M': round(ef[(a.month == 2).to_numpy()].sum() / 1e6, 2),
           'jul_sixfold_M': round(e6[(a.month == 7).to_numpy()].sum() / 1e6, 2),
           'body_share': round(float(e6[y <= 3600].sum() / e6.sum()), 3) if e6.sum() != 0 else None,
           'rows_changed_sixfold': int((np.abs(p6 - b6) > 1e-6).sum()), 'rows_changed_forward': int((np.abs(pf - bf) > 1e-6).sum())}
    jn = 0.05 * abs(res['sixfold_dsse_M']) + 0.5
    res['jan_noise_M'] = round(jn, 3)
    res['pass_rule'] = bool(res['sixfold_dsse_M'] < 0 and res['folds_better'] >= 4 and res['forward_H1toH2_M'] < 0
                                  and res['forward_H2toH1_M'] < 0 and res['jan_sixfold_M'] <= jn and res['jan_forward_M'] <= jn)
    return res
