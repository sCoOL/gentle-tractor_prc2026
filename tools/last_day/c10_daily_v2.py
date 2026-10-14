"""EUROCONTROL daily reconciliation layer on integer arrays: the shared machinery of the combined layer
(c15_combined.py) and of the LIRF copy / held enumeration (c13_recentred_v2.py, c16_stack_recentred.py).

Same posterior as c9_daily_layer.py (version 1): per airport x off-block UTC day (t - p),
R = 60 * DLY_ALL_PRE_2 - sum max(0, D - p); on the active rows (D - p > -s_i) e_i = -R s_i^2 / (sum_active s_j^2 + s_K^2);
p_new = rules(p + lam e_i). Re-implemented on integer arrays (fast); with every switch off it reproduces version 1.
Switches, each fitted / chosen inside the training rows by training SSE:
  passes   1/2/3 passes, each recomputing day, active set, S and s_i^2 from the current p;
  lamcell  lam per airport x lane, shrunk to the global lam with n0 = 20,000 rows (tested, not kept);
  modes    s_i^2 of long rows (D > 1 h) from the copy-vs-held mixture: (1-w) sN^2 + w(1-w)(D-N)^2, w the row's implied
           copy weight, N / sN^2 held-taxi tables by APT x lane x fine D-bin ('raw'), or that shape rescaled to version 1's
           cell mean ('cal') (tested, not kept);
  discrete exact copy / held enumeration on LIRF long unmatched rows (`discrete_post`, "idea 4").
The public description of team gracious-tornado's daily layer was read for the idea only; no code copied.
Only the functions are used by the final post-steps (the 2025 evaluation of each switch alone is research history).
"""
import numpy as np
import pandas as pd
import ld
import c9_daily_layer as c9
from c9_step1_identity import ec

APTS = sorted(ld.TZ)
NA = len(APTS)
DAY0 = int((pd.Timestamp('2024-11-01') - pd.Timestamp('1970-01-01')).days)
ND = 800
DB = np.array(c9.DBINS[1:-1], float)
FB = np.array([3600, 5400, 7200, 10800, 14400, 21600, 28800, 43200], float)
NFB = len(FB) + 1
LAMS = c9.LAMS
CAP2 = c9.CAP2
N0 = 20000.0
NCELL = NA * 2 * 6 * 2


def prep(d: pd.DataFrame) -> dict:
    c = {}
    apt = pd.Categorical(d.APT.astype(str), categories=APTS).codes.astype(np.int64)
    assert (apt >= 0).all()
    lane = d.lane.astype(str).to_numpy()
    assert set(np.unique(lane)) <= {'matched', 'unmatched'}
    c['apt'], c['lane'] = apt, (lane == 'unmatched').astype(np.int64)
    c['D'], c['t'] = d.D.to_numpy(float), d.t.to_numpy(float)
    c['dbin'] = np.digitize(c['D'], DB).astype(np.int64)
    c['fb'] = np.digitize(c['D'], FB).astype(np.int64)
    c['L'] = (d.t - d.lobt).to_numpy(float)
    c['Lb'] = (d.t - d.bf_lobt).to_numpy(float)
    st = d.st.astype(str).to_numpy()
    c['F'], c['R'], c['SF'] = st == 'F_range', st == 'R', st == 'sched_far'
    c['al'] = c['apt'] * 2 + c['lane']
    c['p0'] = d.p.to_numpy(float)
    if 'y' in d:
        c['y'] = d.y.to_numpy(float)
    c['n'] = len(d)
    return c


def sub(c: dict, m: np.ndarray) -> dict:
    return {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in c.items()} | {'n': int(m.sum())}


def rules(c, idx, p):
    """ld.rules on arrays (identical logic): W clip, F_range, R, sched_far, floor."""
    out = np.asarray(p, float).copy()
    L, Lb = c['L'][idx], c['Lb'][idx]
    w = np.isfinite(L)
    lo, hi = w & (out < L - ld.W_CLIP), w & (out > L + ld.W_CLIP)
    out[lo], out[hi] = (L - ld.W_CLIP)[lo], (L + ld.W_CLIP)[hi]
    m = c['F'][idx] & (out < Lb + ld.EDGE)
    out[m] = (Lb + ld.EDGE)[m]
    m = c['R'][idx] & (out > Lb - ld.EDGE)
    out[m] = (Lb - ld.EDGE)[m]
    sf = c['SF'][idx]
    m = sf & (np.abs(out - Lb) < ld.EDGE)
    out[m] = (Lb - ld.EDGE)[m]
    m = sf & (out > Lb - ld.EDGE)
    out[m] = (Lb - ld.EDGE)[m]
    return np.maximum(out, ld.FLOOR)


def ec_arrays(E):
    K, F = np.full((NA, ND), np.nan), np.full((NA, ND), np.nan)
    di = (pd.to_datetime(E.day) - pd.Timestamp('1970-01-01')).dt.days.to_numpy() - DAY0
    ai = pd.Categorical(E.APT, categories=APTS).codes
    ok = (ai >= 0) & (di >= 0) & (di < ND)
    K[ai[ok], di[ok]] = E.K.to_numpy(float)[ok]
    F[ai[ok], di[ok]] = E.FLT_DEP_IFR_2.to_numpy(float)[ok]
    return K.ravel(), F.ravel()


def cell(c, p):
    return ((c['al'] * 6 + c['dbin']) * 2 + ((c['D'] - p) > 0)).astype(np.int64)


def fit(c, EA):
    """All parameters from training rows c (labels y, incoming p0)."""
    y, p = c['y'], c['p0']
    r2 = np.minimum((y - p) ** 2, CAP2)
    k = cell(c, p)
    s, n = np.bincount(k, r2, NCELL), np.bincount(k, minlength=NCELL)
    vt = np.where(n >= 30, s / np.maximum(n, 1), np.nan)
    am = np.bincount(c['apt'], r2, NA) / np.maximum(np.bincount(c['apt'], minlength=NA), 1)
    # s_K^2: per-airport variance of K - S_y over whole training days (true off-block day)
    K, F = EA
    di = np.floor((c['t'] - y) / 86400).astype(np.int64) - DAY0
    assert di.min() >= 0 and di.max() < ND
    gid = c['apt'] * ND + di
    S = np.bincount(gid, np.maximum(0, c['D'] - y), NA * ND)
    nn = np.bincount(gid, minlength=NA * ND)
    ok = (nn > 0) & np.isfinite(K) & (np.abs(nn - F) <= 0.02 * F)
    sK2 = np.full(NA, np.nan)
    for a in range(NA):
        r = (K - S)[a * ND:(a + 1) * ND][ok[a * ND:(a + 1) * ND]]
        if len(r) > 1:
            sK2[a] = r.var(ddof=1)
    # held-taxi tables for the mixture variance (long rows, |y - D| > 600 s)
    h = (c['D'] > 3600) & (np.abs(y - c['D']) > 600)
    yh = y[h]
    lv = []
    for key, size in (((c['al'] * NFB + c['fb'])[h], NA * 2 * NFB), (c['al'][h], NA * 2), (c['apt'][h], NA)):
        med, cnt = np.full(size, np.nan), np.bincount(key, minlength=size)
        srt = pd.Series(yh).groupby(key).median()
        med[srt.index.to_numpy()] = srt.to_numpy()
        med[cnt < 30] = np.nan
        lv.append(med)
    gl = float(np.median(yh))
    P = {'vt': vt, 'am': am, 'sK2': sK2, 'N_lv': lv, 'N_gl': gl}
    N = N_of(c, P)
    dev = np.minimum((y - N) ** 2, CAP2)[h]
    sl = []
    for key, size in (((c['al'] * NFB + c['fb'])[h], NA * 2 * NFB), (c['al'][h], NA * 2), (c['apt'][h], NA)):
        cnt = np.bincount(key, minlength=size)
        sl.append(np.where(cnt >= 30, np.bincount(key, dev, size) / np.maximum(cnt, 1), np.nan))
    P['S_lv'], P['S_gl'] = sl, float(dev.mean())
    # 'cal' multiplier per v1 cell on the training rows at the incoming p: v1 cell mean / mean mixture variance
    mv = mixvar(c, p, P)
    lg = c['D'] > 3600
    cs, cn = np.bincount(k[lg], mv[lg], NCELL), np.bincount(k[lg], minlength=NCELL)
    v1c = np.where(np.isfinite(vt), vt, am[(np.arange(NCELL) // 24)])
    P['cal'] = np.where((cn > 0) & (cs > 0), v1c / np.where(cs > 0, cs / np.maximum(cn, 1), 1), 1.0)
    return P


def _lookup(c, lv, gl):
    out = lv[0][c['al'] * NFB + c['fb']]
    for arr, key in ((lv[1], c['al']), (lv[2], c['apt'])):
        out = np.where(np.isfinite(out), out, arr[key])
    return np.where(np.isfinite(out), out, gl)


def N_of(c, P):
    return _lookup(c, P['N_lv'], P['N_gl'])


def mixvar(c, p, P):
    N, s2N = N_of(c, P), _lookup(c, P['S_lv'], P['S_gl'])
    gap = c['D'] - N
    w = np.where(gap > 60, np.clip((p - N) / np.where(gap > 60, gap, 1), 0, 1), 0.0)
    return np.minimum((1 - w) * s2N + w * (1 - w) * gap ** 2, CAP2)


def s2_of(c, p, P, mode, fixed=None):
    k = cell(c, p)
    s2 = P['vt'][k]
    s2 = np.where(np.isfinite(s2), s2, P['am'][c['apt']])
    if mode != 'v1':
        lg = c['D'] > 3600
        mv = mixvar(c, p, P)
        if mode == 'cal':
            mv = mv * P['cal'][k]
        s2 = np.where(lg, mv, s2)
    if fixed is not None:
        s2 = np.where(fixed, 0.0, s2)
    return s2


def one_pass(c, p, P, EA, lam_row, mode, fixed=None, detail=False):
    s2 = s2_of(c, p, P, mode, fixed)
    s = np.sqrt(s2)
    di = np.floor((c['t'] - p) / 86400).astype(np.int64) - DAY0
    assert di.min() >= 0 and di.max() < ND
    gid = c['apt'] * ND + di
    gap = c['D'] - p
    act = gap > -s
    S = np.bincount(gid, np.maximum(0, gap), NA * ND)
    n = np.bincount(gid, minlength=NA * ND)
    V = np.bincount(gid, np.where(act, s2, 0.0), NA * ND)
    K, F = EA
    pres = n > 0
    ok = pres & np.isfinite(K) & (np.abs(n - F) <= 0.02 * F)
    sK = np.repeat(P['sK2'], ND)
    den = V + np.where(np.isfinite(sK), sK, np.median(V[pres]))
    z = np.where(ok & (den > 0), (np.where(ok, K, 0) - S) / np.where(den > 0, den, 1), 0.0)
    e = np.where(act, -z[gid] * s2, 0.0)
    newp = p + lam_row * e
    mv = np.abs(newp - p) > 1e-9
    out = p.copy()
    if mv.any():
        idx = np.where(mv)[0]
        out[idx] = rules(c, idx, newp[idx])
    if detail:
        g = pd.DataFrame({'gid': np.where(pres)[0]})
        g['APT'] = np.array(APTS)[g.gid // ND]
        g['day'] = pd.to_datetime((g.gid % ND + DAY0) * 86400, unit='s').dt.date
        g['n'], g['S'], g['V'], g['K'], g['ok'] = n[pres], S[pres], V[pres], K[pres], ok[pres]
        g['R'] = np.where(g.ok, g.K - g.S, 0.0)
        return out, g
    return out


MMAX = 14
LIRF = APTS.index('LIRF')


def discrete_post(c, P, EA, fixed=None):
    """Idea 4: exact two-point (copy / held) posterior for LIRF long unmatched rows (3,600 < D <= 43,200 s) per off-block
    day with 1..MMAX such rows, everything at the incoming p0. Returns (eligible mask, post - prior)."""
    p = c['p0']
    s2 = s2_of(c, p, P, 'v1', fixed)
    di = np.floor((c['t'] - p) / 86400).astype(np.int64) - DAY0
    gid = c['apt'] * ND + di
    N, s2N = N_of(c, P), _lookup(c, P['S_lv'], P['S_gl'])
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
    for g, rows in pd.Series(idxL).groupby(gid[idxL]):
        rows = rows.to_numpy()
        m = len(rows)
        if not ok[g] or m > MMAX:
            continue
        R, v0 = K[g] - S_O[g], sK2 + V_O[g]
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


def apply_discrete(c, pbase, elig, delta, lam4):
    out = pbase.copy()
    idx = np.where(elig)[0]
    if len(idx):
        out[idx] = rules(c, idx, c['p0'][idx] + lam4 * delta[idx])
    return out


def run(c, P, EA, lam_row, mode, passes, fixed=None):
    ps, p = [], c['p0'].copy()
    for _ in range(passes):
        p = one_pass(c, p, P, EA, lam_row, mode, fixed)
        ps.append(p)
    return ps


def choose(tr, P, EA, cfg, log):
    """(mode, passes, lam) by training SSE, then optional per-cell lam. Returns lam_row function args."""
    y = tr['y']
    best, cellsse = None, {}
    nmax = max(cfg['passes'])
    for mode in cfg['modes']:
        for lam in LAMS:
            ps = run(tr, P, EA, lam, mode, nmax)
            for n in cfg['passes']:
                r2 = (y - ps[n - 1]) ** 2
                sse = float(r2.sum())
                log.append({'mode': mode, 'lam': lam, 'passes': n, 'train_sse_M': round(sse / 1e6, 3)})
                if best is None or sse < best[0]:
                    best = (sse, mode, n, lam)
                cellsse[(mode, n, lam)] = np.bincount(tr['al'], r2, NA * 2)
    _, mode, n, lam = best
    lamc = np.full(NA * 2, lam)
    if cfg['lamcell']:
        cnt = np.bincount(tr['al'], minlength=NA * 2).astype(float)
        mat = np.stack([cellsse[(mode, n, l)] for l in LAMS])
        hat = np.array(LAMS)[mat.argmin(0)]
        lamc = (cnt * hat + N0 * lam) / (cnt + N0)
    return {'mode': mode, 'passes': n, 'lam': lam, 'lamc': lamc}


def fit_apply(tr, te, EA, cfg, fixed=None, EA_te=None):
    EA_te = EA if EA_te is None else EA_te
    P = fit(tr, EA)
    log = []
    ch = choose(tr, P, EA, cfg, log)
    if cfg['lamcell']:
        ps = run(tr, P, EA, ch['lamc'][tr['al']], ch['mode'], ch['passes'])
        log.append({'lamcell_train_sse_M': round(float(((tr['y'] - ps[-1]) ** 2).sum()) / 1e6, 3)})
    pt = run(te, P, EA_te, ch['lamc'][te['al']], ch['mode'], ch['passes'], fixed)[-1]
    if cfg.get('discrete'):
        ptr = run(tr, P, EA, ch['lamc'][tr['al']], ch['mode'], ch['passes'])[-1]
        el, de = discrete_post(tr, P, EA)
        sse4 = {l4: float(((tr['y'] - apply_discrete(tr, ptr, el, de, l4)) ** 2).sum()) for l4 in LAMS}
        l4 = min(sse4, key=sse4.get)
        log.append({'lam4_train_sse_M': {k: round(v / 1e6, 3) for k, v in sse4.items()}, 'lam4': l4, 'eligible_train_rows': int(el.sum()),
                    'lirf_long_train_rows': int(((tr['apt'] == LIRF) & (tr['lane'] == 1) & (tr['D'] > 3600) & (tr['D'] <= 43200)).sum())})
        elt, det = discrete_post(te, P, EA_te, fixed)
        pt = apply_discrete(te, pt, elt, det, l4)
        log.append({'eligible_test_rows': int(elt.sum())})
        ch['lam4'] = l4
    info = {'mode': ch['mode'], 'passes': ch['passes'], 'lam': ch['lam'], 'lam4': ch.get('lam4'),
            'lamc': {f'{APTS[i // 2]}|{"unmatched" if i % 2 else "matched"}': round(float(v), 3) for i, v in enumerate(ch['lamc'])}
            if cfg['lamcell'] else None, 'log': log}
    return pt, info, P, ch


def judge_inc(a, b6, bf, p6, pf, name):
    y = a.y.to_numpy()
    e6 = (y - p6) ** 2 - (y - b6) ** 2
    ef = (y - pf) ** 2 - (y - bf) ** 2
    h1 = (a.month <= 6).to_numpy()
    mon = a.month.to_numpy()
    fold = (pd.Series(e6).groupby(a.fold.to_numpy()).sum() / 1e6).reindex(ld.FOLDS).fillna(0).round(3)
    tot = e6.sum()
    r = {'name': name, 'sixfold_dsse_M': round(tot / 1e6, 3), 'by_fold_M': fold.to_dict(), 'folds_better': int((fold < 0).sum()),
         'forward_H1toH2_M': round(ef[~h1].sum() / 1e6, 3), 'forward_H2toH1_M': round(ef[h1].sum() / 1e6, 3),
         'jan_sixfold_M': round(e6[mon == 1].sum() / 1e6, 3), 'feb_sixfold_M': round(e6[mon == 2].sum() / 1e6, 3),
         'jan_forward_M': round(ef[mon == 1].sum() / 1e6, 3), 'feb_forward_M': round(ef[mon == 2].sum() / 1e6, 3),
         'body_share': round(float(e6[y <= 3600].sum() / tot), 3) if tot != 0 else None,
         'rows_changed_sixfold': int((np.abs(p6 - b6) > 1e-6).sum()),
         'sum_change2_sixfold_M': round(float(((p6 - b6) ** 2).sum() / 1e6), 3),
         'by_apt_M': (pd.Series(e6).groupby(a.APT.to_numpy()).sum() / 1e6).round(2).to_dict(),
         'by_apt_lane_M': {f'{k[0]}|{k[1]}': v for k, v in (pd.Series(e6).groupby([a.APT.to_numpy(), a.lane.to_numpy()]).sum() / 1e6).round(2).items()},
         'by_month_M': (pd.Series(e6).groupby(mon).sum() / 1e6).round(2).to_dict()}
    r['pass'] = bool(tot < 0 and r['folds_better'] >= 5 and r['forward_H1toH2_M'] < 0 and r['forward_H2toH1_M'] < 0
                     and r['jan_sixfold_M'] <= 0 and r['feb_sixfold_M'] <= 0 and r['jan_forward_M'] <= 0 and r['feb_forward_M'] <= 0)
    return r
