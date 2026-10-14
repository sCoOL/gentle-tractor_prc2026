"""Count-gap adjustment of the EUROCONTROL daily target and the IFR / non-helicopter row filter (functions used by the
combined layer, c15_combined.py).

EUROCONTROL's K = 60 DLY_ALL_PRE_2 sums max(0, BLOCK - SCHED) over F = FLT_DEP_IFR_2 departures, but our day holds n
rows (IFR non-heli rows whose estimated off-block t - p falls on the UTC day). For every airport-day with n != F,
K_adj = K - (F - n) mu, mu = per-airport mean of max(0, D - y) on the training rows. s_K^2 is recomputed with K_adj
(true off-block day); the > 2 % skip rule and everything else is c9_daily_layer.py with the IFR / heli filter.
With mu None the functions here equal c9.fit_sK / c9.apply. `inc_judge` is the pass rule of
runs/overnight_20261008/ov.py judge().
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ld  # noqa: E402
import c9_daily_layer as c9  # noqa: E402


def ifr_keep(df: pd.DataFrame, base_file: str) -> np.ndarray:
    """True for IFR (FLIGHT_RULE_mvt != V) non-helicopter rows: the rows EUROCONTROL's FLT_DEP_IFR_2 counts."""
    fr = pd.read_parquet(ld.DATA / base_file, columns=['MVT_ID_mvt', 'FLIGHT_RULE_mvt'])
    r = df[['MVT_ID_mvt']].merge(fr, on='MVT_ID_mvt', how='left', validate='one_to_one')
    return ((r.FLIGHT_RULE_mvt != 'V').to_numpy() & (df.is_heli.to_numpy() == 0))


def fit_mu(tr) -> pd.Series:
    """mu: per airport, mean over the training rows of max(0, D - y) (s per departure)."""
    return pd.Series(np.maximum(0.0, (tr.D - tr.y).to_numpy())).groupby(tr.APT.astype(str).to_numpy()).mean()


def kadj(g: pd.DataFrame, mu):
    """K_adj = K - (F - n) mu (mu None -> K unchanged, bit-identical)."""
    if mu is None:
        return g.K
    return g.K - (g.FLT_DEP_IFR_2 - g.n) * g.APT.map(mu).astype(float)


def fit_sK(tr, E, mu=None):
    """c9.fit_sK with K -> K_adj (true off-block day, whole days within 2 %)."""
    day = pd.to_datetime((tr.t - tr.y).to_numpy(), unit='s', utc=True).date
    g = pd.DataFrame({'APT': tr.APT.astype(str).to_numpy(), 'day': day, 's': np.maximum(0, (tr.D - tr.y).to_numpy()), 'one': 1.0}) \
        .groupby(['APT', 'day']).agg(s=('s', 'sum'), n=('one', 'sum')).reset_index()
    m = g.merge(E, on=['APT', 'day'])
    m = m[(m.n - m.FLT_DEP_IFR_2).abs() <= 0.02 * m.FLT_DEP_IFR_2]
    return (kadj(m, mu) - m.s).groupby(m.APT).var()


def apply(d, E, vtab, sK2, lam, mu=None, return_detail=False, fixed=None):
    """c9.apply with R = K_adj - S on the ok days."""
    p = d.p.to_numpy(float)
    s2 = c9.var_of(d, vtab)
    if fixed is not None:
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
    g['adj'] = np.where(ok, g.K - kadj(g, mu), 0.0)          # = (F - n) mu on ok days
    g['R'] = np.where(ok, kadj(g, mu) - g.S, 0.0)
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


def fit(tr, E, mu_on: bool):
    mu = fit_mu(tr) if mu_on else None
    vtab, sK2 = c9.fit_var(tr), fit_sK(tr, E, mu)
    y = tr.y.to_numpy()
    sse = {lam: float(((y - apply(tr, E, vtab, sK2, lam, mu)) ** 2).sum()) for lam in c9.LAMS}
    return vtab, sK2, min(sse, key=sse.get), mu, sse


def inc_judge(a, p6c, pf6, p6o, pfo, mask=None):
    """Increment of candidate (p6c, pfc) over base (p6o, pfo); pass rule = ov.py judge() verbatim
    (six-fold < 0, >= 4/6 folds better, both forward < 0, Jan six-fold and forward <= 0.05 |six-fold| + 0.5 M)."""
    m = np.ones(len(a), bool) if mask is None else mask
    y = a.y.to_numpy()
    e6 = np.where(m, (y - p6c) ** 2 - (y - p6o) ** 2, 0.0)
    ef = np.where(m, (y - pf6) ** 2 - (y - pfo) ** 2, 0.0)
    h1 = (a.month <= 6).to_numpy()
    mon = a.month.to_numpy()
    fold = (pd.Series(e6).groupby(a.fold.to_numpy()).sum() / 1e6).reindex(ld.FOLDS).fillna(0)
    r = {'sixfold_dsse_M': round(e6.sum() / 1e6, 3), 'by_fold_M': fold.round(3).to_dict(),
         'folds_better': int((fold < 0).sum()), 'folds_worse': int((fold > 0).sum()),
         'forward_H1toH2_M': round(ef[~h1].sum() / 1e6, 3), 'forward_H2toH1_M': round(ef[h1].sum() / 1e6, 3),
         'jan_sixfold_M': round(e6[mon == 1].sum() / 1e6, 3), 'feb_sixfold_M': round(e6[mon == 2].sum() / 1e6, 3),
         'jan_forward_M': round(ef[mon == 1].sum() / 1e6, 3), 'feb_forward_M': round(ef[mon == 2].sum() / 1e6, 3),
         'jul_sixfold_M': round(e6[mon == 7].sum() / 1e6, 3), 'jul_forward_M': round(ef[mon == 7].sum() / 1e6, 3),
         'body_share': round(float(e6[y <= 3600].sum() / e6.sum()), 3) if e6.sum() != 0 else None,
         'rows_changed_sixfold': int((m & (np.abs(p6c - p6o) > 1e-6)).sum()),
         'rows_changed_1s_sixfold': int((m & (np.abs(p6c - p6o) > 1)).sum()),
         'sum_change2_sixfold_M': round(float((np.where(m, p6c - p6o, 0) ** 2).sum() / 1e6), 3)}
    noise = 0.05 * abs(r['sixfold_dsse_M']) + 0.5
    r['jan_noise_M'] = round(noise, 3)
    r['pass_ov_judge'] = bool(r['sixfold_dsse_M'] < 0 and r['folds_better'] >= 4 and r['forward_H1toH2_M'] < 0
                              and r['forward_H2toH1_M'] < 0 and r['jan_sixfold_M'] <= noise and r['jan_forward_M'] <= noise)
    return r
