"""The combined EUROCONTROL daily reconciliation layer ("stack"): version 1 (c9_daily_layer.py) + (1) up to three passes
+ (2) the IFR / non-helicopter identity + (3) the count-gap target K_adj recomputed in every pass.

  py -3.14 tools/last_day/c15_combined.py --tag x            # 2025 six folds + forward -> runs/last_day/c15_oof_x.parquet

Stack: c10's integer-array version-1 posterior on the IFR non-heli rows; passes x lam by training SSE; on ok days
R = K_adj - S, K_adj = K - (F - n) mu_apt with n from the current p in every pass; s_K^2 = var(K_adj - S_y) (true day).
VFR and helicopter rows never move. `held` and `complete_days` are shared with the midnight-crossers step (c17 / c18).
The out-of-fold frame (p6_stack: six-fold, pf_stack: forward) sizes the 2026 guard of build_A15LX.py.
"""
import argparse
import json
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ld  # noqa: E402
import c10_daily_v2 as v2  # noqa: E402
from c10_daily_v2 import APTS, NA, ND, DAY0, prep, sub, rules, ec_arrays  # noqa: E402
from c9_step1_identity import ec  # noqa: E402
import c14_count_gap as c14  # noqa: E402

LAMS = v2.LAMS
LIRF = APTS.index('LIRF')
PUSH = ld.PUSH


# ---------------------------------------------------------------- data
def extras(df: pd.DataFrame) -> dict:
    """Label-free extra arrays: ADS-B v2 push start (push_phases), LIRF date-bug gate domain."""
    pp = pd.read_parquet(PUSH, columns=['MVT_ID_mvt', 't_push_start'])
    pp = pp[pp.t_push_start.notna()].drop_duplicates('MVT_ID_mvt')
    tp = df[['MVT_ID_mvt']].merge(pp, on='MVT_ID_mvt', how='left', validate='one_to_one').t_push_start.to_numpy(float)
    dom = ((df.APT.astype(str) == "LIRF") & (df.lane.astype(str) == "unmatched") & (df.is_heli == 0)).to_numpy().copy()
    dom &= ld.local_day(df.t, df.APT) > ld.local_day(df.t_sched, df.APT)
    return {'tp': tp, 'bugdom': dom}


def load(year: int, ifr: bool, p0=None):
    d = ld.load_2025() if year == 2025 else ld.load_2026()
    keep = c14.ifr_keep(d.reset_index(drop=True), 'base_train.parquet' if year == 2025 else 'base_rank.parquet') if ifr \
        else np.ones(len(d), bool)
    return d, keep


def mk(dk: pd.DataFrame) -> dict:
    C = prep(dk)
    C.update(extras(dk))
    return C


# ---------------------------------------------------------------- stack 1+2+3
def fit15(c, EA, mu_on):
    P = v2.fit(c, EA)
    P['mu'] = None
    if mu_on:
        y = c['y']
        mu = np.bincount(c['apt'], np.maximum(0, c['D'] - y), NA) / np.maximum(np.bincount(c['apt'], minlength=NA), 1)
        K, F = EA
        di = np.floor((c['t'] - y) / 86400).astype(np.int64) - DAY0
        gid = c['apt'] * ND + di
        S = np.bincount(gid, np.maximum(0, c['D'] - y), NA * ND)
        nn = np.bincount(gid, minlength=NA * ND)
        ok = (nn > 0) & np.isfinite(K) & (np.abs(nn - F) <= 0.02 * F)
        Ka = K - (F - nn) * np.repeat(mu, ND)
        sK2 = np.full(NA, np.nan)
        for a in range(NA):
            r = (Ka - S)[a * ND:(a + 1) * ND][ok[a * ND:(a + 1) * ND]]
            if len(r) > 1:
                sK2[a] = r.var(ddof=1)
        P['sK2'], P['mu'] = sK2, mu
    return P


def day_sums(c, p, P, EA, fixed=None):
    s2 = v2.s2_of(c, p, P, 'v1', fixed)
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
    ok = (n > 0) & np.isfinite(K) & (np.abs(n - F) <= 0.02 * F)
    return dict(s2=s2, s=s, gid=gid, gap=gap, act=act, S=S, n=n, V=V, ok=ok)


def one_pass(c, p, P, EA, lam, fixed=None):
    """c10.one_pass (mode v1) with R = K_adj - S, K_adj = K - (F - n) mu, n from the current p (identical when mu None)."""
    q = day_sums(c, p, P, EA, fixed)
    K, F = EA
    ok, n, S, V = q['ok'], q['n'], q['S'], q['V']
    pres = n > 0
    sK = np.repeat(P['sK2'], ND)
    den = V + np.where(np.isfinite(sK), sK, np.median(V[pres]))
    Kt = np.where(ok, K, 0)
    if P.get('mu') is not None:
        Kt = np.where(ok, K - (F - n) * np.repeat(P['mu'], ND), 0)
    z = np.where(ok & (den > 0), (Kt - S) / np.where(den > 0, den, 1), 0.0)
    e = np.where(q['act'], -z[q['gid']] * q['s2'], 0.0)
    newp = p + lam * e
    mv = np.abs(newp - p) > 1e-9
    out = p.copy()
    if mv.any():
        idx = np.where(mv)[0]
        out[idx] = rules(c, idx, newp[idx])
    return out


def run15(c, P, EA, lam, passes, fixed=None):
    ps, p = [], c['p0'].copy()
    for _ in range(passes):
        p = one_pass(c, p, P, EA, lam, fixed)
        ps.append(p)
    return ps


def choose15(tr, P, EA, passes):
    y = tr['y']
    best, log = None, []
    for lam in LAMS:
        ps = run15(tr, P, EA, lam, max(passes))
        for n in passes:
            sse = float(((y - ps[n - 1]) ** 2).sum())
            log.append({'lam': lam, 'passes': n, 'train_sse_M': round(sse / 1e6, 3)})
            if best is None or sse < best[0]:
                best = (sse, n, lam)
    return {'passes': best[1], 'lam': best[2], 'log': log}


# ---------------------------------------------------------------- shared with the midnight-crossers step (c17 / c18)
def held(c, P, A):
    N, s2N = v2.N_of(c, P), v2._lookup(c, P['S_lv'], P['S_gl'])
    tp = c['tp']
    hp = np.isfinite(tp)
    Nh = np.where(hp, np.maximum(60.0, c['t'] - tp - A['off'][c['apt']]), N)
    s2h = np.where(hp, A['sd'][c['apt']] ** 2, s2N)
    return Nh, s2h, hp


def complete_days(c):
    """Per airport, days with take-offs on d and on d + 1 in the set (fold edges are partial)."""
    td = np.floor(c['t'] / 86400).astype(np.int64) - DAY0
    has = np.zeros(NA * (ND + 1), bool)
    has[c['apt'] * (ND + 1) + td] = True
    h = has.reshape(NA, ND + 1)
    return (h[:, :ND] & h[:, 1:ND + 1]).ravel()


def split_run(tr, te, EA, cfg, fixed=None, EA_te=None):
    """Fit the stack on the training rows tr, apply it to te. Returns (te prediction, info, fitted P)."""
    EA_te = EA if EA_te is None else EA_te
    P = fit15(tr, EA, cfg['mu'])
    ch = choose15(tr, P, EA, cfg['passes'])
    pt3 = run15(te, P, EA_te, ch['lam'], ch['passes'], fixed)[-1]
    info = {'passes': ch['passes'], 'lam': ch['lam'], 'stack_log': ch['log'],
            'mu_s': None if P['mu'] is None else dict(zip(APTS, np.round(P['mu'], 1).tolist())),
            'sK_min': dict(zip(APTS, np.round(np.sqrt(P['sK2']) / 60, 1).tolist()))}
    return pt3, info, P


# ---------------------------------------------------------------- 2025 evaluation
def side_counts(t, y, M, pold, pnew):
    tru, old, new = (t - y) < M, (t - pold) < M, (t - pnew) < M
    return {'rows': int(len(t)), 'stack_right': int((old == tru).sum()), 'new_right': int((new == tru).sum()),
            'caught': int(((old != tru) & (new == tru)).sum()), 'false_alarm': int(((old == tru) & (new != tru)).sum()),
            'missed': int(((old != tru) & (new != tru)).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--ifr', type=int, default=1)
    ap.add_argument('--passes', default='1,2,3')
    ap.add_argument('--mu', type=int, default=1)
    x = ap.parse_args()
    cfg = {'passes': [int(v) for v in x.passes.split(',')], 'mu': bool(x.mu)}
    t0 = time.time()
    EA25 = ec_arrays(ec(2025))
    a, keep = load(2025, bool(x.ifr))
    ak = a[keep].reset_index(drop=True)
    C = mk(ak)
    idx = np.where(keep)[0]
    p6s, pfs = a.p.to_numpy().copy(), a.p.to_numpy().copy()
    res = {'tag': x.tag, 'cfg': cfg, 'ifr': bool(x.ifr), 'splits': {}}
    h1 = (ak.month <= 6).to_numpy()
    splits = [(f, (ak.fold != f).to_numpy(), (ak.fold == f).to_numpy(), 'six') for f in ld.FOLDS] +              [('H1->H2', h1, ~h1, 'fwd'), ('H2->H1', ~h1, h1, 'fwd')]
    for nm, trm, tem, kind in splits:
        pt3, info, _ = split_run(sub(C, trm), sub(C, tem), EA25, cfg)
        ii = idx[tem]
        if kind == 'six':
            p6s[ii] = pt3
        else:
            pfs[ii] = pt3
        res['splits'][nm] = info
        print(x.tag, nm, 'passes', info['passes'], 'lam', info['lam'], round(time.time() - t0), 's', flush=True)
    pd.DataFrame({'MVT_ID_mvt': a.MVT_ID_mvt.to_numpy(), 'p6_stack': p6s, 'pf_stack': pfs, 'p6': p6s, 'pf': pfs})         .to_parquet(ld.OUT / f'c15_oof_{x.tag}.parquet', index=False)
    apt = a.APT.astype(str).to_numpy()
    res['stack_vs_proxy'] = ld.judge(a, p6s, pfs, 'stack')
    v1p = ld.OUT / 'c9_oof_2025.parquet'
    if v1p.exists():   # judged against version 1 (c9_daily_layer.py) when its out-of-fold frame is there
        v1 = pd.read_parquet(v1p, columns=['MVT_ID_mvt', 'p6', 'pf'])
        assert (v1.MVT_ID_mvt.to_numpy() == a.MVT_ID_mvt.to_numpy()).all()
        res['stack_vs_v1'] = c14.inc_judge(a, p6s, pfs, v1.p6.to_numpy(), v1.pf.to_numpy())
        res['stack_vs_v1_by_apt'] = {k: c14.inc_judge(a, p6s, pfs, v1.p6.to_numpy(), v1.pf.to_numpy(), apt == k)['sixfold_dsse_M'] for k in APTS}
        print('STACK vs v1', json.dumps(res['stack_vs_v1']), flush=True)
    print('STACK vs proxy', json.dumps(res['stack_vs_proxy']), flush=True)
    res['seconds'] = round(time.time() - t0)
    ld.save(f'c15_combined_{x.tag}', res)


if __name__ == '__main__':
    main()
