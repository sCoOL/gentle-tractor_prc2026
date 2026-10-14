"""Midnight-crossers machinery (functions used by the hardened step, c18_crossers_hardened.py, and its build).

Per airport x UTC day: X(d) = F(d) - n_sched(d) is the net number of flights whose recorded off-block crossed UTC
midnight away from their schedule day. For a joint hypothesis h of a group, n_h(d) = n_sched(d) + X_h(d), so a count
term on n_h(d) - F(d) = X_h(d) - X(d) compares the implied crossers with EUROCONTROL's count (non-group rows at their
current day).

Every prior comes from the row's CURRENT prediction p4 and the chain's date-bug gate pb (`priors`: pb_eff keeps the
prior mean consistent with p). Rows set by the enumeration (and on 2026 the manual / A15L rule rows) are fixed and sit
in the day sums at p4. The K residual is re-centred per airport x year-month (median over ok days of K_adj - S at p4).
`stack4` = the stack + the re-centred enumeration (c16_stack_recentred.split16) also returning the training p4.
The unhardened form of this step (all cells may take the copy / date-bug states) failed on 2025 and is not used.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ld  # noqa: E402
import c10_daily_v2 as v2  # noqa: E402
from c10_daily_v2 import APTS, NA, ND, DAY0  # noqa: E402
import c15_combined as c15  # noqa: E402
import c16_stack_recentred as c16  # noqa: E402

ALPHAS = (0.0, 0.5, 1.0, 2.0)
CVS = (1.0, 2.0, 4.0)
LAM4S = (0.0, 0.25, 0.5, 0.75, 1.0)
CFGS = [(a, c) for a in ALPHAS for c in CVS]
CMAX = 8
GCAP_ROWS, GCAP_COMB = 12, 8192
MIN_DAYS = 5
PBTAB = ld.GATE_TABLE


# ---------------------------------------------------------------- data
def pb_of(df: pd.DataFrame, split: str) -> np.ndarray:
    t = pd.read_parquet(PBTAB, columns=['MVT_ID_mvt', 'split', 'pb'])
    t = t[t.split == split].drop_duplicates('MVT_ID_mvt')
    return df[['MVT_ID_mvt']].merge(t[['MVT_ID_mvt', 'pb']], on='MVT_ID_mvt', how='left', validate='one_to_one') \
        .pb.fillna(0.0).to_numpy(float)


def mk(dk: pd.DataFrame, split: str) -> dict:
    C = c15.mk(dk)
    C['pb'] = pb_of(dk, split)
    return C


# ---------------------------------------------------------------- base = the stack + the re-centred enumeration (c16)
def stack4(tr, EA, te=None, EA_te=None, fixed_te=None):
    """c16.split16 line for line, also returning the TRAINING p4 and idea-4 eligibility (needed to fit the step)."""
    EA_te = EA if EA_te is None else EA_te
    P = c15.fit15(tr, EA, True)
    ch = c15.choose15(tr, P, EA, c16.CFG['passes'])
    ptr3 = c15.run15(tr, P, EA, ch['lam'], ch['passes'])[-1]
    el, de = c16.discrete_at(tr, ptr3, P, EA, None, 'median')
    sse4 = {l4: float(((tr['y'] - c16.apply_at(tr, ptr3, el, de, l4)) ** 2).sum()) for l4 in v2.LAMS}
    l4 = min(sse4, key=sse4.get)
    out = {'P': P, 'passes': ch['passes'], 'lam': ch['lam'], 'lam4_idea4': l4, 'ptr4': c16.apply_at(tr, ptr3, el, de, l4), 'eltr': el}
    if te is not None:
        pt3 = c15.run15(te, P, EA_te, ch['lam'], ch['passes'], fixed_te)[-1]
        elt, det = c16.discrete_at(te, pt3, P, EA_te, fixed_te, 'median')
        out.update({'pt3': pt3, 'pt4': c16.apply_at(te, pt3, elt, det, l4), 'elt': elt})
    return out


# ---------------------------------------------------------------- step
def lmask(c, p, P, A, frozen):
    dT = np.floor(c['t'] / 86400).astype(np.int64)
    dS = np.floor((c['t'] - c['D']) / 86400).astype(np.int64)
    Nh, s2h, hp = c15.held(c, P, A)
    D = c['D']
    M = dT * 86400.0
    lng = (dS < dT) & (D > 3600) & (D <= 86400) & (D - Nh > 60) & ~frozen
    amb = ~lng & (D <= 86400) & (np.abs(c['t'] - p - M) < 1800) & ~frozen
    return lng, amb, Nh, s2h, hp, dT, M


def offsets(c, q, EA, mu):
    """Re-centring offset per airport x year-month: median over ok days of K_adj - S at the current p (>= 5 days)."""
    K, F = EA
    Ka = K - (F - q['n']) * np.repeat(mu, ND)
    r = Ka - q['S']
    off = np.zeros(NA * ND)
    days = np.where(q['ok'])[0]
    mon = pd.to_datetime((days % ND + DAY0) * 86400, unit='s').strftime('%Y-%m').to_numpy()
    apt = days // ND
    tab = {}
    for (a, m_), ix in pd.Series(np.arange(len(days))).groupby([apt, mon]):
        dd = days[ix.to_numpy()]
        o = float(np.median(r[dd])) if len(dd) >= MIN_DAYS else 0.0
        lo, hi = a * ND, (a + 1) * ND
        sel = np.arange(lo, hi)
        sm = pd.to_datetime((sel % ND + DAY0) * 86400, unit='s').strftime('%Y-%m').to_numpy() == m_
        off[sel[sm]] = o
        tab[f'{APTS[a]}-{m_}'] = round(o / 60, 1)
    return off, tab


def fitA(tr, p4, P, EA, frozen, sfixed=None):
    """Training-only pieces: ADS-B push offset per airport, count pmf per airport (non-L rows at p4, L rows on true day)."""
    y = tr['y']
    A = {}
    m = np.isfinite(tr['tp'])
    o = tr['t'] - tr['tp'] - y
    gm = float(np.median(o[m]))
    gs = max(60.0, 1.4826 * float(np.median(np.abs(o[m] - gm))))
    off, sd = np.full(NA, gm), np.full(NA, gs)
    for a in range(NA):
        k = m & (tr['apt'] == a)
        if k.sum() >= 30:
            off[a] = float(np.median(o[k]))
            sd[a] = max(60.0, 1.4826 * float(np.median(np.abs(o[k] - off[a]))))
    A['off'], A['sd'] = off, sd
    lng, amb, *_ = lmask(tr, p4, P, A, frozen)
    L = lng | amb
    q = c15.day_sums(tr, p4, P, EA, sfixed)
    K, F = EA
    dtrue = tr['apt'] * ND + np.floor((tr['t'] - y) / 86400).astype(np.int64) - DAY0
    nm = np.bincount(np.where(L, dtrue, q['gid']), minlength=NA * ND)
    cinf = q['ok'] & c15.complete_days(tr) & np.isfinite(F)
    lp = np.zeros((NA, 2 * CMAX + 1))
    A['pmf_hist'] = {}
    for a in range(NA):
        dd = np.where(cinf[a * ND:(a + 1) * ND])[0] + a * ND
        cm = np.clip(np.round(nm[dd] - F[dd]).astype(int), -CMAX, CMAX)
        cnt = np.bincount(cm + CMAX, minlength=2 * CMAX + 1) + 0.5
        lp[a] = np.log(cnt / cnt.sum())
        A['pmf_hist'][APTS[a]] = {int(k - CMAX): int(v) for k, v in enumerate(cnt - 0.5) if v > 0.5}
    A['logpmf'] = lp
    return A


def priors(c, p, Nh, D):
    """Copy weight from the current p and the chain's gate pb (pb_eff keeps the prior mean consistent with p)."""
    pb = np.where(c['bugdom'] & (c['pb'] > 0), c['pb'], 0.0)
    yb = Nh + 86400.0
    g_ = np.where(D - Nh > 60, D - Nh, 1.0)
    w_try = ((p - pb * yb) / np.maximum(1 - pb, 1e-9) - Nh) / g_
    Np = 0.001 * D + 0.999 * Nh
    pmax = np.maximum(0.0, (p - Np) / (yb - Np))
    pbe = np.where(w_try < 0.001, np.minimum(pb, pmax), pb)
    w = np.clip(((p - pbe * yb) / np.maximum(1 - pbe, 1e-9) - Nh) / g_, 0.001, 0.999)
    return pbe, w, yb
