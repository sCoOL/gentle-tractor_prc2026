"""The hardened midnight-crossers step, applied after the stack + re-centred enumeration (base = A15LXR).

Groups = airport x take-off UTC day d (days d-1, d); rows whose schedule falls on an earlier UTC day than the take-off
and that are long (1 h < D <= 24 h: copy / held / LIRF date bug) or ambiguous (predicted off-block within 30 min of
0 h UTC: before / after midnight). Every joint assignment is scored by prior x Gaussian K_adj likelihood (re-centred per
airport-month) x alpha count term (n_h - F) and p += lam4 (E_post - E_prior). The unhardened form failed on 2025: held
flights with a normal-taxi label were pushed toward "copy" in cells where 2025 has no copies / date bugs. Three guards,
fixed before the 2025 evaluation:
  1. support gate: copy / date-bug hypotheses only in airport x lane x D-band cells whose TRAINING artefact rate >= tau;
     unsupported long rows may only move on the midnight side (the ambiguous form, |t - p4 - M| < 30 min);
  2. priors capped by the cell rate: copy = min(copy0, k r_c), bug = min(bug0, k r_b), k = 3;
  3. move cap 1,800 s for rows whose cell artefact rate < 20 % (LIRF unmatched never capped).
tau {2, 5, 10 %} x alpha x c_v x lam4 by training SSE (summed over all training rows) inside each split.

  py -3.14 tools/last_day/c18_crossers_hardened.py --tag h   # 2025 six folds + forward -> runs/last_day/c18_oof_h.parquet
"""
import argparse
import json
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import log_ndtr, ndtr  # noqa: E402
import ld  # noqa: E402
import c10_daily_v2 as v2  # noqa: E402
from c10_daily_v2 import APTS, NA, ND, DAY0, rules  # noqa: E402
import c13_recentred_v2 as c13  # noqa: E402
import c15_combined as c15  # noqa: E402
import c17_crossers as c17  # noqa: E402
from c9_step1_identity import ec  # noqa: E402

TAUS = (0.02, 0.05, 0.10)
K_PRIOR = 3.0
CAP_S = 1800.0
CAP_RATE = 0.20
MIN_CELL = 10
BANDS = np.array([3600.0, 3 * 3600.0, 8 * 3600.0, 13 * 3600.0, 24 * 3600.0])
BAND_NAMES = ('<=1h', '1-3h', '3-8h', '8-13h', '13-24h', '>24h')
NB = len(BAND_NAMES)
LIRF = APTS.index('LIRF')
CFGS, CVS, ALPHAS, LAM4S = c17.CFGS, c17.CVS, c17.ALPHAS, c17.LAM4S
CMAX, GCAP_ROWS, GCAP_COMB = c17.CMAX, c17.GCAP_ROWS, c17.GCAP_COMB
mk, stack4, fitA, priors = c17.mk, c17.stack4, c17.fitA, c17.priors


# ---------------------------------------------------------------- cells and rates
def cell_of(c):
    band = np.digitize(c['D'], BANDS, right=True)
    return (c['apt'] * 2 + c['lane']) * NB + band


def labels(y, D):
    cp = np.abs(y - D) <= 600
    bug = (y >= 72000) & (np.abs(y - D) > 10) & (np.mod(y, 86400) < 5400) & ~cp
    return cp, bug


def fit_rates(tr):
    """Copy / date-bug / artefact rates per airport x lane x D-band cell on TRAINING rows (b0 and < MIN_CELL rows -> 0)."""
    cell = cell_of(tr)
    cp, bug = labels(tr['y'], tr['D'])
    n = np.bincount(cell, minlength=NA * 2 * NB).astype(float)
    nc = np.bincount(cell, cp.astype(float), NA * 2 * NB)
    nb = np.bincount(cell, bug.astype(float), NA * 2 * NB)
    ok = (n >= MIN_CELL) & (np.arange(NA * 2 * NB) % NB != 0)
    rc = np.where(ok, nc / np.maximum(n, 1), 0.0)
    rb = np.where(ok, nb / np.maximum(n, 1), 0.0)
    tab = {}
    for i in np.where(n > 0)[0]:
        a, rest = divmod(i, 2 * NB)
        ln, b = divmod(rest, NB)
        if b == 0:
            continue
        tab[f"{APTS[a]}|{'unm' if ln else 'mat'}|{BAND_NAMES[b]}"] = {'n': int(n[i]), 'copy': int(nc[i]), 'bug': int(nb[i]),
                                                                    'rate': round(float(rc[i] + rb[i]), 4)}
    return {'rc': rc, 'rb': rb, 'ra': rc + rb, 'tab': tab}


def row_rates(c, R):
    ce = cell_of(c)
    rc, rb, ra = R['rc'][ce], R['rb'][ce], R['ra'][ce]
    capm = (ra < CAP_RATE) & ~((c['apt'] == LIRF) & (c['lane'] == 1))
    return rc, rb, ra, capm


# ---------------------------------------------------------------- step (c17.post17 + gate + capped priors)
def post18(c, p, P, EA, A, frozen, R, tau, sfixed=None, cfgs=CFGS):
    K, F = EA
    mu = P['mu']
    q = c15.day_sums(c, p, P, EA, sfixed)
    off, otab = c17.offsets(c, q, EA, mu)
    cinf = q['ok'] & c15.complete_days(c) & np.isfinite(F)
    lng0, _, Nh, s2h, hp, dT, M = c17.lmask(c, p, P, A, frozen)
    rc, rb, ra, capm = row_rates(c, R)
    sup = ra >= tau
    lng = lng0 & sup
    t, D, s2, s = c['t'], c['D'], q['s2'], q['s']
    amb = ~lng & (D <= 86400) & (np.abs(t - p - M) < 1800) & ~frozen
    dem = amb & lng0          # unsupported long rows kept on the midnight side only
    L = lng | amb
    n = c['n']
    HM, HD, HC, HV, HP = (np.zeros((n, 3)) for _ in range(5))
    KS = np.zeros(n, np.int64)
    pbe, w, yb = priors(c, p, Nh, D)
    pcopy = np.minimum((1 - pbe) * w, K_PRIOR * rc)
    pbug = np.minimum(pbe, K_PRIOR * rb)
    phld = 1.0 - pcopy - pbug
    reld = lambda x: np.clip(np.floor((t - x) / 86400).astype(np.int64) - (dT - 1), 0, 1)  # noqa: E731
    i = lng
    HM[i, 0], HD[i, 0], HC[i, 0], HV[i, 0], HP[i, 0] = D[i], reld(D)[i], 0.0, 0.0, pcopy[i]
    HM[i, 1], HD[i, 1], HC[i, 1], HV[i, 1], HP[i, 1] = Nh[i], reld(Nh)[i], np.maximum(0, D - Nh)[i], s2h[i], phld[i]
    HM[i, 2], HD[i, 2], HC[i, 2], HV[i, 2], HP[i, 2] = yb[i], 0, 0.0, 0.0, pbug[i]
    KS[i] = np.where(pbug[i] > 0, 3, 2)
    i = amb
    a_ = t - M
    ss = np.maximum(s, 1.0)
    z = (a_ - p) / ss
    lphi = -0.5 * z ** 2 - 0.5 * np.log(2 * np.pi)
    mb = np.maximum(p + ss * np.exp(lphi - log_ndtr(-z)), a_)
    ma = np.minimum(p - ss * np.exp(lphi - log_ndtr(z)), a_)
    pbf = np.clip(ndtr(-z), 0.001, 0.999)
    pbp = np.clip(ndtr((M - c['tp'] - A['off'][c['apt']]) / A['sd'][c['apt']]), 0.02, 0.98)
    pbf = np.where(np.isfinite(c['tp']), pbp, pbf)
    vb = np.where(D - mb > -ss, s2, 0.0)
    va = np.where(D - ma > -ss, s2, 0.0)
    HM[i, 0], HD[i, 0], HC[i, 0], HV[i, 0], HP[i, 0] = mb[i], 0, np.maximum(0, D - mb)[i], vb[i], pbf[i]
    HM[i, 1], HD[i, 1], HC[i, 1], HV[i, 1], HP[i, 1] = ma[i], 1, np.maximum(0, D - ma)[i], va[i], (1 - pbf)[i]
    KS[i] = 2
    Eprior = (HM * HP).sum(1)
    elig = np.zeros(n, bool)
    deltas = {cf: np.zeros(n) for cf in cfgs}
    stats = {'groups': 0, 'groups_skipped_uninformative': 0, 'rows_capped_group': 0, 'L_long_c17': int(lng0.sum()),
             'L_long_supported': int(lng.sum()), 'L_long_demoted_amb': int(dem.sum()), 'L_long_dropped': int((lng0 & ~lng & ~amb).sum()),
             'L_amb': int(amb.sum()), 'bug_state_rows': int((lng & (pbug > 0)).sum()),
             'copy_prior_capped_rows': int((lng & ((1 - pbe) * w > K_PRIOR * rc + 1e-12)).sum()),
             'move_capped_eligible_rows': 0}
    gkey = c['apt'] * ND + (dT - DAY0)
    Li = np.where(L)[0]
    kind = np.where(lng, 1, np.where(dem, 3, np.where(amb, 2, 0)))
    if len(Li) == 0:
        return elig, deltas, kind, capm, stats
    order = np.lexsort((np.abs(t - p - M)[Li], ~lng[Li], gkey[Li]))
    Li = Li[order]
    cuts = np.flatnonzero(np.diff(gkey[Li])) + 1
    for rows in np.split(Li, cuts):
        g = int(gkey[rows[0]])
        days = np.array([g - 1, g])
        if not q['ok'][days].any():
            stats['groups_skipped_uninformative'] += 1
            continue
        sel, comb = [], 1
        for r in rows:
            if len(sel) < GCAP_ROWS and comb * KS[r] <= GCAP_COMB:
                sel.append(r)
                comb *= KS[r]
        stats['rows_capped_group'] += len(rows) - len(sel)
        rows = np.array(sel)
        m = len(rows)
        stats['groups'] += 1
        SO, nO, VO = q['S'][days].copy(), q['n'][days].astype(float), q['V'][days].copy()
        for j in range(2):
            on = q['gid'][rows] == days[j]
            SO[j] -= np.maximum(0, q['gap'][rows][on]).sum()
            nO[j] -= on.sum()
            VO[j] -= np.where(q['act'][rows], s2[rows], 0.0)[on].sum()
        H = np.stack(np.meshgrid(*[np.arange(KS[r]) for r in rows], indexing='ij'), -1).reshape(-1, m)
        ar = np.arange(m)[None, :]
        hm, hd, hc, hv, hpr = (X[rows][ar, H] for X in (HM, HD, HC, HV, HP))
        lp = np.log(np.maximum(hpr, 1e-300)).sum(1)
        a = int(c['apt'][rows[0]])
        sK2 = P['sK2'][a] if np.isfinite(P['sK2'][a]) else float(np.nanmedian(P['sK2']))
        llK = {cv: np.zeros(len(H)) for cv in CVS}
        llc = np.zeros(len(H))
        for j in range(2):
            on = hd == j
            nh = nO[j] + on.sum(1)
            dd = days[j]
            if q['ok'][dd]:
                Rr = K[dd] - (F[dd] - nh) * mu[a] - off[dd] - SO[j] - (hc * on).sum(1)
                vh = (hv * on).sum(1)
                for cv in CVS:
                    var = cv * (sK2 + VO[j]) + vh
                    llK[cv] += -0.5 * np.log(var) - 0.5 * Rr ** 2 / var
            if cinf[dd]:
                cm = np.clip(np.round(nh - F[dd]).astype(int), -CMAX, CMAX)
                llc += A['logpmf'][a][cm + CMAX]
        elig[rows] = True
        for cf in cfgs:
            al, cv = cf
            lw = lp + llK[cv] + al * llc
            pw = np.exp(lw - lw.max())
            pw /= pw.sum()
            deltas[cf][rows] = pw @ hm - Eprior[rows]
    return elig, deltas, kind, capm, stats


def moved(c, idx, p, de, l4, capm):
    mv = l4 * de[idx]
    mv = np.where(capm[idx], np.clip(mv, -CAP_S, CAP_S), mv)
    return rules(c, idx, p[idx] + mv)


def fit_step(tr, ptr4, P, EA, frozen_tr):
    A = fitA(tr, ptr4, P, EA, frozen_tr)
    R = fit_rates(tr)
    y = tr['y']
    sse, stt = {}, {}
    for tau in TAUS:
        el, des, kd, capm, st = post18(tr, ptr4, P, EA, A, frozen_tr, R, tau)
        stt[tau] = st
        idx = np.where(el)[0]
        b0 = float(((y[idx] - ptr4[idx]) ** 2).sum())
        for cf in CFGS:
            for l4 in LAM4S:
                sse[(tau, cf[0], cf[1], l4)] = float(((y[idx] - moved(tr, idx, ptr4, des[cf], l4, capm)) ** 2).sum()) - b0
    best = min(sse, key=sse.get)        # dict order = grid order; min keeps the first on ties
    info = {'tau': best[0], 'k': K_PRIOR, 'alpha': best[1], 'c_v': best[2], 'lam4': best[3], 'train_dsse_best_M': round(sse[best] / 1e6, 3),
            'train_dsse_best_by_tau_M': {str(tu): round(min(v for k_, v in sse.items() if k_[0] == tu) / 1e6, 3) for tu in TAUS},
            'train_stats': stt[best[0]], 'rates_cells_ge_2pct': {k_: v for k_, v in R['tab'].items() if v['rate'] >= 0.02},
            'push_off_s': dict(zip(APTS, np.round(A['off'], 1).tolist()))}
    return A, R, best, info


def apply_step(te, pt4, P, EA_te, A, R, best, frozen_te, sfixed=None):
    tau, al, cv, l4 = best
    el, de, kind, capm, st = post18(te, pt4, P, EA_te, A, frozen_te, R, tau, sfixed, [(al, cv)])
    out = pt4.copy()
    it = np.where(el)[0]
    if len(it) and l4 > 0:
        out[it] = moved(te, it, pt4, de[(al, cv)], l4, capm)
        st['move_capped_eligible_rows'] = int((capm[it] & (np.abs(l4 * de[(al, cv)][it]) > CAP_S)).sum())
    return out, el, kind, st


# ---------------------------------------------------------------- 2025 evaluation
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    x = ap.parse_args()
    t0 = time.time()
    EA25 = v2.ec_arrays(ec(2025))
    a, keep = c15.load(2025, True)
    ak = a[keep].reset_index(drop=True)
    C = mk(ak, 'train')
    idx = np.where(keep)[0]
    base6, basef = a.p.to_numpy().copy(), a.p.to_numpy().copy()
    p6, pf = a.p.to_numpy().copy(), a.p.to_numpy().copy()
    el6, elf = np.zeros(len(a), bool), np.zeros(len(a), bool)
    kd6, kdf = np.zeros(len(a), np.int8), np.zeros(len(a), np.int8)
    res = {'tag': x.tag, 'taus': TAUS, 'k': K_PRIOR, 'cap_s': CAP_S, 'cap_rate': CAP_RATE, 'min_cell': MIN_CELL, 'splits': {}}
    h1 = (ak.month <= 6).to_numpy()
    splits = [(f, (ak.fold != f).to_numpy(), (ak.fold == f).to_numpy(), 'six') for f in ld.FOLDS] + \
             [('H1->H2', h1, ~h1, 'fwd'), ('H2->H1', ~h1, h1, 'fwd')]
    for nm, trm, tem, kind in splits:
        tr, te = v2.sub(C, trm), v2.sub(C, tem)
        b = stack4(tr, EA25, te)
        A, R, best, info = fit_step(tr, b['ptr4'], b['P'], EA25, b['eltr'])
        pt, el, kd, st = apply_step(te, b['pt4'], b['P'], EA25, A, R, best, b['elt'])
        info.update({'test_eligible': int(el.sum()), 'test_rows_changed_1s': int((np.abs(pt - b['pt4']) > 1).sum()), 'test_stats': st})
        ii = idx[tem]
        if kind == 'six':
            base6[ii], p6[ii], el6[ii], kd6[ii] = b['pt4'], pt, el, kd
        else:
            basef[ii], pf[ii], elf[ii], kdf[ii] = b['pt4'], pt, el, kd
        res['splits'][nm] = info
        print(x.tag, nm, {k: info.get(k) for k in ('tau', 'alpha', 'c_v', 'lam4', 'train_dsse_best_M', 'train_dsse_best_by_tau_M',
                                                   'test_eligible', 'test_rows_changed_1s')}, round(time.time() - t0), 's', flush=True)
    pd.DataFrame({'MVT_ID_mvt': a.MVT_ID_mvt.to_numpy(), 'p6_base': base6, 'pf_base': basef, 'p6': p6, 'pf': pf, 'el6': el6, 'elf': elf,
                  'kind6': kd6, 'kindf': kdf}).to_parquet(ld.OUT / f'c18_oof_{x.tag}.parquet', index=False)
    o = pd.read_parquet(ld.OUT / 'c16_oof_x.parquet')
    assert (o.MVT_ID_mvt.to_numpy() == a.MVT_ID_mvt.to_numpy()).all()
    res['step0_base_vs_c16_x'] = {'max_abs_p6': float(np.abs(o.p6.to_numpy() - base6).max()), 'max_abs_pf': float(np.abs(o.pf.to_numpy() - basef).max())}
    print('STEP0', res['step0_base_vs_c16_x'], flush=True)
    assert max(res['step0_base_vs_c16_x'].values()) < 1e-6
    res['inc_over_base'] = c13.pass_judge(a, base6, basef, p6, pf, f'{x.tag}: base + hardened crossers - base')
    apt = a.APT.astype(str).to_numpy()
    y, t, D = a.y.to_numpy(), a.t.to_numpy(), a.D.to_numpy()
    mon = a.month.to_numpy()
    e6 = (y - p6) ** 2 - (y - base6) ** 2
    ef = (y - pf) ** 2 - (y - basef) ** 2
    res['by_apt'] = {k: {'six_M': round(float(e6[apt == k].sum() / 1e6), 2), 'fwd_M': round(float(ef[apt == k].sum() / 1e6), 2),
                         'jan_six_M': round(float(e6[(apt == k) & (mon == 1)].sum() / 1e6), 2), 'feb_six_M': round(float(e6[(apt == k) & (mon == 2)].sum() / 1e6), 2),
                         'rows_changed_1s_six': int((np.abs(p6 - base6)[apt == k] > 1).sum())} for k in APTS}
    res['by_month_six_M'] = (pd.Series(e6).groupby(mon).sum() / 1e6).round(2).to_dict()
    cp, bug = labels(y, D)
    art = cp | bug
    M = np.floor(t / 86400) * 86400.0
    for col, nw, bs, el, kd in (('six', p6, base6, el6, kd6), ('fwd', pf, basef, elf, kdf)):
        mv1, mv = np.abs(nw - bs) > 1, np.abs(nw - bs) > 60
        closer = np.abs(y - nw) < np.abs(y - bs)
        cnt = {'eligible': int(el.sum()), 'eligible_long': int((el & (kd == 1)).sum()), 'eligible_amb': int((el & (kd == 2)).sum()),
               'eligible_demoted': int((el & (kd == 3)).sum()), 'rows_changed_1s': int(mv1.sum()), 'moved_60s': int(mv.sum()),
               'closer': int((mv & closer).sum()), 'further': int((mv & ~closer).sum())}
        e = np.where(el & mv1)[0]
        cnt['midnight_side_moved_1s'] = c15.side_counts(t[e], y[e], M[e], bs[e], nw[e])
        lg = el & (kd == 1)
        up = (nw - bs) > 60
        cnt['copy_dir_long'] = {'rows': int(lg.sum()), 'true_artefact': int((lg & art).sum()), 'caught': int((lg & up & art).sum()),
                                'false_alarm': int((lg & up & ~art).sum()), 'missed': int((lg & ~up & art).sum()),
                                'dsse_true_art_M': round(float(((y - nw) ** 2 - (y - bs) ** 2)[lg & art].sum() / 1e6), 2),
                                'dsse_not_art_M': round(float(((y - nw) ** 2 - (y - bs) ** 2)[lg & ~art].sum() / 1e6), 2)}
        cnt['by_apt_moved_60s'] = {k: {'closer': int((mv & closer & (apt == k)).sum()), 'further': int((mv & ~closer & (apt == k)).sum())}
                                   for k in APTS if (mv & (apt == k)).any()}
        res[f'counts_{col}'] = cnt
    wi = np.argsort(-e6)[:10]
    res['worst10_six'] = [{'MVT': int(a.MVT_ID_mvt.iat[i]), 'APT': apt[i], 'lane': str(a.lane.iat[i]), 'month': int(mon[i]), 'D': float(D[i]),
                           'y': float(y[i]), 'base': round(float(base6[i]), 1), 'new': round(float(p6[i]), 1), 'kind': int(kd6[i]),
                           'dsse_M': round(float(e6[i] / 1e6), 3)} for i in wi if e6[i] > 0]
    big = np.where(np.abs(p6 - base6) > 600)[0]
    res['rows_moved_600s_six'] = [{'MVT': int(a.MVT_ID_mvt.iat[i]), 'APT': apt[i], 'month': int(mon[i]), 'D': float(D[i]), 'y': float(y[i]),
                                   'base': round(float(base6[i]), 1), 'new': round(float(p6[i]), 1), 'kind': int(kd6[i])} for i in big[:150]]
    print('INC', json.dumps(res['inc_over_base']), flush=True)
    print('BY_APT', json.dumps(res['by_apt']), flush=True)
    print('COUNTS', json.dumps({k: {kk: vv for kk, vv in res[k].items() if kk != 'by_apt_moved_60s'} for k in ('counts_six', 'counts_fwd')}), flush=True)
    print('WORST10', json.dumps(res['worst10_six']), flush=True)
    res['seconds'] = round(time.time() - t0)
    ld.save(f'c18_crossers_hardened_{x.tag}', res)


if __name__ == '__main__':
    main()
