"""The combined stack (c15_combined.py) followed by the re-centred LIRF copy / held enumeration (c13_recentred_v2.py),
computed on the stack output.

The enumeration covers LIRF unmatched rows with 3,600 < D <= 43,200 s, 1..14 per off-block UTC day. Here it runs at the
stack output p3 (not at the incoming p0), on the stack's IFR row set, with R_L = K_adj - S_O, K_adj = K - (F - n) mu
(n at p3, mu and s_K^2 from the stack fit), and the re-centring offset per year-month = median over the eligible days
of (R_L - sum_L max(0, D - p3)) (>= 5 days else 0). p = rules(p3 + lam4 delta); lam4 by training SSE inside each split
(c10 LAMS), the training delta computed at the training stack output.

  py -3.14 tools/last_day/c16_stack_recentred.py --tag x     # 2025 six folds + forward -> runs/last_day/c16_oof_x.parquet

Step 0 checks: (a) at p0 with mu off, discrete_at equals c13.discrete_post_rc; (b) the stack part equals
c15_oof_x.parquet. The out-of-fold frame sizes the increment guard of build_A15LXR.py.
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
import c15_combined as c15  # noqa: E402
import c13_recentred_v2 as c13  # noqa: E402
from c9_step1_identity import ec  # noqa: E402

CFG = {'passes': [1, 2, 3], 'mu': True}
MIN_DAYS = c13.MIN_DAYS


def discrete_at(c, p, P, EA, fixed=None, recentre='median', diag=None):
    """c13.discrete_post_rc evaluated at an arbitrary p (the stack output) with R_L = K_adj - S_O when P['mu'] is set.
    With p = c['p0'] and P['mu'] None this is c13.discrete_post_rc line for line (step-0 check)."""
    NA, ND, DAY0, LIRF, MMAX = v2.NA, v2.ND, v2.DAY0, v2.LIRF, v2.MMAX
    s2 = v2.s2_of(c, p, P, 'v1', fixed)
    di = np.floor((c['t'] - p) / 86400).astype(np.int64) - DAY0
    assert di.min() >= 0 and di.max() < ND
    gid = c['apt'] * ND + di
    N, s2N = v2.N_of(c, P), v2._lookup(c, P['S_lv'], P['S_gl'])
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
    if P.get('mu') is not None:
        K = K - (F - n) * np.repeat(P['mu'], ND)          # count-gap K_adj, n at p
    sK2 = P['sK2'][LIRF]
    w = np.clip((p - N) / np.where(D - N > 60, D - N, 1), 0.001, 0.999)
    prior = w * D + (1 - w) * N
    delta = np.zeros(len(p))
    elig = np.zeros(len(p), bool)
    idxL = np.where(L)[0]
    groups = [(g, rows.to_numpy()) for g, rows in pd.Series(idxL).groupby(gid[idxL])]
    groups = [(g, rows) for g, rows in groups if ok[g] and len(rows) <= MMAX]
    gs = np.array([g for g, _ in groups], np.int64)
    Rg = np.array([K[g] - S_O[g] for g in gs]) if len(gs) else np.zeros(0)
    EL = np.array([np.maximum(0, gap[rows]).sum() for _, rows in groups]) if len(gs) else np.zeros(0)
    Res = Rg - EL
    mon = pd.to_datetime((gs % ND + DAY0) * 86400, unit='s').strftime('%Y-%m').to_numpy() if len(gs) else np.array([], str)
    off = np.zeros(len(gs))
    dg = {}
    for m_ in np.unique(mon):
        sel = mon == m_
        r, e = Rg[sel], Res[sel]
        o = float(np.median(e)) if (recentre == 'median' and sel.sum() >= MIN_DAYS) else 0.0
        off[sel] = o
        dg[m_] = {'days': int(sel.sum()), 'RL_mean_min': round(float(r.mean()) / 60, 1), 'RL_median_min': round(float(np.median(r)) / 60, 1),
                  'resid_mean_min': round(float(e.mean()) / 60, 1), 'resid_median_min': round(float(np.median(e)) / 60, 1),
                  'offset_min': round(o / 60, 1)}
    if diag is not None:
        diag.append(dg)
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


def apply_at(c, pbase, elig, delta, lam4):
    """c10.apply_discrete with the base p (the stack output) in place of p0."""
    out = pbase.copy()
    idx = np.where(elig)[0]
    if len(idx):
        out[idx] = v2.rules(c, idx, pbase[idx] + lam4 * delta[idx])
    return out


def split16(tr, te, EA, fixed=None, EA_te=None, do4=True, recentre='median', diag=None):
    """Stack 1+2+3 exactly as c15.split_run, then the re-centred enumeration (idea 4) on its output."""
    EA_te = EA if EA_te is None else EA_te
    P = c15.fit15(tr, EA, CFG['mu'])
    ch = c15.choose15(tr, P, EA, CFG['passes'])
    pt3 = c15.run15(te, P, EA_te, ch['lam'], ch['passes'], fixed)[-1]
    info = {'passes': ch['passes'], 'lam': ch['lam']}
    pt4, elt = pt3, np.zeros(te['n'], bool)
    if do4:
        ptr3 = c15.run15(tr, P, EA, ch['lam'], ch['passes'])[-1]
        el, de = discrete_at(tr, ptr3, P, EA, None, recentre)
        sse4 = {l4: float(((tr['y'] - apply_at(tr, ptr3, el, de, l4)) ** 2).sum()) for l4 in v2.LAMS}
        l4 = min(sse4, key=sse4.get)
        elt, det = discrete_at(te, pt3, P, EA_te, fixed, recentre, diag)
        pt4 = apply_at(te, pt3, elt, det, l4)
        info.update({'lam4': l4, 'lam4_train_sse_M': {k: round(v / 1e6, 3) for k, v in sse4.items()},
                     'eligible_train_rows': int(el.sum()), 'eligible_test_rows': int(elt.sum()),
                     'test_rows_changed_1s': int((np.abs(pt4 - pt3) > 1).sum())})
    return pt3, pt4, elt, info, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--recentre', choices=['off', 'median'], default='median')
    x = ap.parse_args()
    t0 = time.time()
    EA25 = v2.ec_arrays(ec(2025))
    res = {'tag': x.tag, 'cfg': CFG, 'recentre': x.recentre, 'splits': {}}
    a, keep = c15.load(2025, True)
    # step 0a: the port at p0 with mu off = c13.discrete_post_rc (all rows, all-2025 fit)
    Call = v2.prep(a)
    P0 = v2.fit(Call, EA25)
    e1, d1 = discrete_at(Call, Call['p0'], P0, EA25, None, 'median')
    e2, d2 = c13.discrete_post_rc(Call, P0, EA25, None, 'median')
    res['step0a_port_vs_c13'] = {'elig_equal': bool((e1 == e2).all()), 'max_abs_delta': float(np.abs(d1 - d2).max()), 'eligible': int(e1.sum())}
    print('step0a', res['step0a_port_vs_c13'], flush=True)
    assert res['step0a_port_vs_c13']['elig_equal'] and res['step0a_port_vs_c13']['max_abs_delta'] < 1e-6
    ak = a[keep].reset_index(drop=True)
    C = v2.prep(ak)
    idx = np.where(keep)[0]
    p6s, pfs, p6, pf = (a.p.to_numpy().copy() for _ in range(4))
    el6, elf = np.zeros(len(a), bool), np.zeros(len(a), bool)
    h1 = (ak.month <= 6).to_numpy()
    splits = [(f, (ak.fold != f).to_numpy(), (ak.fold == f).to_numpy(), 'six') for f in ld.FOLDS] + \
             [('H1->H2', h1, ~h1, 'fwd'), ('H2->H1', ~h1, h1, 'fwd')]
    for nm, trm, tem, kind in splits:
        dg = []
        pt3, pt4, el, info, _ = split16(v2.sub(C, trm), v2.sub(C, tem), EA25, recentre=x.recentre, diag=dg)
        info['offsets_test'] = dg[-1] if dg else None
        ii = idx[tem]
        if kind == 'six':
            p6s[ii], p6[ii], el6[ii] = pt3, pt4, el
        else:
            pfs[ii], pf[ii], elf[ii] = pt3, pt4, el
        res['splits'][nm] = info
        print(x.tag, nm, {k: info.get(k) for k in ('passes', 'lam', 'lam4', 'eligible_test_rows', 'test_rows_changed_1s')},
              round(time.time() - t0), 's', flush=True)
    pd.DataFrame({'MVT_ID_mvt': a.MVT_ID_mvt.to_numpy(), 'p6_stack': p6s, 'pf_stack': pfs, 'p6': p6, 'pf': pf,
                  'el6': el6, 'elf': elf}).to_parquet(ld.OUT / f'c16_oof_{x.tag}.parquet', index=False)
    # step 0b: the stack reproduces c15's stack OOF
    o = pd.read_parquet(ld.OUT / 'c15_oof_x.parquet')
    assert (o.MVT_ID_mvt.to_numpy() == a.MVT_ID_mvt.to_numpy()).all()
    res['step0b_stack_vs_c15_x'] = {'max_abs_p6': float(np.abs(o.p6_stack.to_numpy() - p6s).max()),
                                    'max_abs_pf': float(np.abs(o.pf_stack.to_numpy() - pfs).max())}
    print('step0b', res['step0b_stack_vs_c15_x'], flush=True)
    assert max(res['step0b_stack_vs_c15_x'].values()) < 1e-6
    J = c13.pass_judge
    res['inc_over_stack'] = J(a, p6s, pfs, p6, pf, f'{x.tag}: stack+4R - stack')
    v1p = ld.OUT / 'c9_oof_2025.parquet'
    if v1p.exists():
        v1 = pd.read_parquet(v1p, columns=['MVT_ID_mvt', 'p6', 'pf'])
        assert (v1.MVT_ID_mvt.to_numpy() == a.MVT_ID_mvt.to_numpy()).all()
        res['full_vs_v1'] = J(a, v1.p6.to_numpy(), v1.pf.to_numpy(), p6, pf, f'{x.tag}: stack+4R - v1')
        res['stack_vs_v1'] = J(a, v1.p6.to_numpy(), v1.pf.to_numpy(), p6s, pfs, 'stack - v1')
    apt = a.APT.astype(str).to_numpy()
    y = a.y.to_numpy()
    e6 = (y - p6) ** 2 - (y - p6s) ** 2
    res['inc_over_stack_by_apt_M'] = (pd.Series(e6).groupby(apt).sum() / 1e6).round(2).to_dict()
    res['inc_over_stack_by_month_M'] = (pd.Series(e6).groupby(a.month.to_numpy()).sum() / 1e6).round(2).to_dict()
    mv = np.abs(p6 - p6s) > 60
    closer = np.abs(y - p6) < np.abs(y - p6s)
    res['counts_six'] = {'eligible': int(el6.sum()), 'rows_changed_1s': int((np.abs(p6 - p6s) > 1).sum()), 'moved_60s': int(mv.sum()),
                         'closer': int((mv & closer).sum()), 'further': int((mv & ~closer).sum()),
                         'held_rows_dsse_M': round(float(e6[el6 & (np.abs(y - a.D.to_numpy()) > 600)].sum() / 1e6), 2),
                         'copy_rows_dsse_M': round(float(e6[el6 & (np.abs(y - a.D.to_numpy()) <= 600)].sum() / 1e6), 2)}
    for k in ('inc_over_stack', 'full_vs_v1', 'stack_vs_v1'):
        if k in res:
            print(k, json.dumps(res[k]), flush=True)
    print('by_apt', res['inc_over_stack_by_apt_M'], 'counts', res['counts_six'], flush=True)
    res['seconds'] = round(time.time() - t0)
    ld.save(f'c16_stack_recentred_{x.tag}', res)


if __name__ == '__main__':
    main()
