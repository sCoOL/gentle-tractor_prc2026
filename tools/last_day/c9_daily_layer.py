"""EUROCONTROL daily reconciliation layer, version 1 (one pass, all rows). Own implementation; the public descriptions
of teams gracious-tornado and reliable-hamburger were read for the idea only, no code copied.

Identity (c9_step1_identity.py, 2025): 60 x DLY_ALL_PRE_2 = K = sum over the airport's departures with off-block on
that UTC day of max(0, BLOCK - SCHED) = max(0, D - y). For predictions p the implied sum is S_p = sum max(0, D - p);
R = K - S_p. Linear-Gaussian posterior: row errors e_i = y_i - p_i ~ N(0, s_i^2) independent; on the day's active rows
(D - p > -s_i, rows that are or may be delayed) the constraint is sum_active e_i = -R + noise(s_K^2). Posterior mean
  e_i = -R s_i^2 / (sum_active s_j^2 + s_K^2);   p_new = rules(p + lam * e_i),
which puts most of a day's residual on the uncertain rows (LIRF copy-vs-held mixtures) and a small shift on ordinary
delayed rows. Fitted on the training rows only: s_i^2 = mean of min(r^2, (4 h)^2) per airport x lane x D-bin x
(D - p > 0); s_K^2 = per-airport variance of K - S_y (true labels, true off-block day, whole days); lam from {0.25, 0.5, 0.75, 1}
by training SSE. Day = UTC date of t - p (the estimated off-block); days whose departure count differs from
FLT_DEP_IFR_2 by more than 2 % are skipped.

The final post-steps use the combined form (c15_combined.py); this file supplies the shared variance cells and its
2025 out-of-fold frame (runs/last_day/c9_oof_2025.parquet) is the reference the combined form is judged against.

  py -3.14 tools/last_day/c9_daily_layer.py
"""
import numpy as np
import pandas as pd
import ld
from c9_step1_identity import ec

DBINS = [-np.inf, 0, 1800, 3600, 3 * 3600, 8 * 3600, np.inf]
LAMS = (0.25, 0.5, 0.75, 1.0)
CAP2 = (4 * 3600.0) ** 2


def keys(d):
    db = np.digitize(d.D.to_numpy(), DBINS[1:-1])
    dl = (d.D.to_numpy() - d.p.to_numpy()) > 0
    return pd.Series(d.APT.astype(str).to_numpy() + '|' + d.lane.astype(str).to_numpy() + '|' + db.astype(str) + '|'
                     + dl.astype(int).astype(str), index=d.index)


def fit_var(tr):
    k = keys(tr)
    r2 = pd.Series(np.minimum((tr.y - tr.p).to_numpy() ** 2, CAP2), index=tr.index)
    v = r2.groupby(k).agg(['mean', 'size'])
    apt = r2.groupby(tr.APT.astype(str)).mean()
    return v[v['size'] >= 30]['mean'], apt


def var_of(d, vtab):
    v, apt = vtab
    s2 = keys(d).map(v)
    return s2.fillna(d.APT.astype(str).map(apt)).to_numpy(float)


def fit_sK(tr, E):
    """Per-airport variance of K - S_y with the true off-block day (training rows)."""
    day = pd.to_datetime((tr.t - tr.y).to_numpy(), unit='s', utc=True).date
    g = pd.DataFrame({'APT': tr.APT.astype(str).to_numpy(), 'day': day, 's': np.maximum(0, (tr.D - tr.y).to_numpy()), 'one': 1.0}) \
        .groupby(['APT', 'day']).agg(s=('s', 'sum'), n=('one', 'sum')).reset_index()
    m = g.merge(E, on=['APT', 'day'])
    m = m[(m.n - m.FLT_DEP_IFR_2).abs() <= 0.02 * m.FLT_DEP_IFR_2]   # whole days only (fold edges cut days)
    return (m.K - m.s).groupby(m.APT).var()


def apply(d, E, vtab, sK2, lam, return_detail=False, fixed=None):
    """d: rows of one year with p = the current prediction. Returns the new p (rules on moved rows)."""
    p = d.p.to_numpy(float)
    s2 = var_of(d, vtab)
    if fixed is not None:   # rows set by a hard rule / manual rows: known values, they count in S but never move
        s2 = np.where(np.asarray(fixed, bool), 0.0, s2)
    s = np.sqrt(s2)
    day = pd.to_datetime(d.t.to_numpy() - p, unit='s', utc=True).date
    gap = d.D.to_numpy() - p
    act = gap > -s
    f = pd.DataFrame({'APT': d.APT.astype(str).to_numpy(), 'day': day, 'c': np.maximum(0, gap), 'one': 1.0,
                      's2a': np.where(act, s2, 0.0)})
    g = f.groupby(['APT', 'day']).agg(S=('c', 'sum'), n=('one', 'sum'), V=('s2a', 'sum')).reset_index() \
        .merge(E[['APT', 'day', 'K', 'FLT_DEP_IFR_2']], on=['APT', 'day'], how='left')
    ok = g.K.notna() & ((g.n - g.FLT_DEP_IFR_2).abs() <= 0.02 * g.FLT_DEP_IFR_2)
    g['ok'] = ok
    g['R'] = np.where(ok, g.K - g.S, 0.0)
    g['den'] = g.V + g.APT.map(sK2).fillna(g.V.median()).to_numpy()
    g['z'] = np.where(ok & (g.den > 0), g.R / g.den, 0.0)
    z = f[['APT', 'day']].merge(g[['APT', 'day', 'z']], on=['APT', 'day'], how='left').z.fillna(0).to_numpy()
    e = np.where(act, -z * s2, 0.0)
    newp = p + lam * e
    mv = np.abs(newp - p) > 1e-9
    out = p.copy()
    if mv.any():
        out[mv] = ld.rules(d[mv], newp[mv])
    return (out, g) if return_detail else out


def fit_and_apply(tr, te, E):
    vtab, sK2 = fit_var(tr), fit_sK(tr, E)
    y = tr.y.to_numpy()
    sse = {lam: float(((y - apply(tr, E, vtab, sK2, lam)) ** 2).sum()) for lam in LAMS}
    bl = min(sse, key=sse.get)
    return apply(te, E, vtab, sK2, bl), bl


def main():
    E25, E26 = ec(2025), ec(2026)
    a = ld.load_2025()
    p6 = np.full(len(a), np.nan)
    lams = {}
    for f in ld.FOLDS:
        te = (a.fold == f).to_numpy()
        p6[te], lams[f] = fit_and_apply(a[~te], a[te], E25)
        print('fold', f, 'lam', lams[f], flush=True)
    pf = np.full(len(a), np.nan)
    h1 = (a.month <= 6).to_numpy()
    for trm, tem, nm in ((h1, ~h1, 'H1->H2'), (~h1, h1, 'H2->H1')):
        pf[tem], lams[nm] = fit_and_apply(a[trm], a[tem], E25)
        print(nm, 'lam', lams[nm], flush=True)
    j = ld.judge(a, p6, pf, 'ec_daily_layer')
    e6 = pd.Series((a.y - p6) ** 2 - (a.y - a.p) ** 2)
    j['by_apt_M'] = (e6.groupby(a.APT.to_numpy()).sum() / 1e6).round(2).to_dict()
    j['by_apt_lane_M'] = {f'{k[0]}|{k[1]}': v for k, v in (e6.groupby([a.APT.to_numpy(), a.lane.to_numpy()]).sum() / 1e6).round(2).items()}
    j['by_month_M'] = (e6.groupby(a.month.to_numpy()).sum() / 1e6).round(2).to_dict()
    j['lams'] = lams
    print(j, flush=True)
    a['p6'], a['pf'] = p6, pf
    a[['MVT_ID_mvt', 'fold', 'month', 'APT', 'lane', 'y', 'D', 'p', 'p6', 'pf']].to_parquet(ld.OUT / 'c9_oof_2025.parquet', index=False)
    # 2026: fit on all 2025, apply to the A15 final rows
    b = ld.load_2026()
    vtab, sK2 = fit_var(a), fit_sK(a, E25)
    y = a.y.to_numpy()
    sse = {lam: float(((y - apply(a, E25, vtab, sK2, lam)) ** 2).sum()) for lam in LAMS}
    lam26 = min(sse, key=sse.get)
    p26, g26 = apply(b, E26, vtab, sK2, lam26, return_detail=True)
    mv26 = p26 - b.p.to_numpy()
    b['p_daily'] = p26
    b[['MVT_ID_mvt', 'APT', 'month', 'lane', 'p', 'p_daily']].to_parquet(ld.OUT / 'c9_rank_2026.parquet', index=False)
    g26.to_parquet(ld.OUT / 'c9_days_2026.parquet', index=False)
    mv25 = p6 - a.p.to_numpy()
    j['lam_2026'] = lam26
    j['rms_ratio'] = ld.rms_ratio(a, mv25, b, mv26)
    j['rows_2026_moved_1s'] = int((np.abs(mv26) > 1).sum())
    j['sum_move2_2026_M'] = round(float((mv26 ** 2).sum() / 1e6), 2)
    j['move2_2026_by_apt_M'] = (pd.Series(mv26 ** 2).groupby(b.APT.to_numpy()).sum() / 1e6).round(2).to_dict()
    j['days_2026_used'] = int(g26.ok.sum())
    j['days_2026_total'] = int(len(g26))
    j['days_2026_count_mismatch'] = int((g26.K.notna() & ~g26.ok).sum())
    j['R_2026_mean_min_by_apt_month'] = {f'{k[0]}-{k[1]}': round(v / 60, 1) for k, v in
                                         g26[g26.ok].assign(m=pd.to_datetime(g26.day).dt.month).groupby(['APT', 'm']).R.mean().items()}
    print({k: j[k] for k in ('lam_2026', 'rms_ratio', 'rows_2026_moved_1s', 'sum_move2_2026_M', 'move2_2026_by_apt_M',
                             'days_2026_used', 'days_2026_total', 'days_2026_count_mismatch')})
    ld.save('c9_daily_layer', j)


if __name__ == '__main__':
    main()
