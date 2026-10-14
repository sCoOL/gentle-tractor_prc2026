"""v63 = v57 + a same-period context layer on LIRF NM-unmatched departures 1-8 h late.

  py -3.14 scripts/274_unmatched_context_layer.py frame    # runs/v63_context_layer/frame.parquet (both years)
  py -3.14 scripts/274_unmatched_context_layer.py screen   # 2025 evidence -> screen.json, oof.parquet
  py -3.14 scripts/274_unmatched_context_layer.py build    # submissions/gentle-tractor_v63.parquet + package A70
  py -3.14 scripts/274_unmatched_context_layer.py ledger   # 2025 v63 proxy per row for the Explorer -> rows.parquet

What v57 does on these rows: p = U + w (D - U), a mixture of the gate's normal component U (A48u) and the schedule copy
D = take-off - schedule; w_imp = (p - U)/(D - U) is v57's own copy probability. On the 2025 v57 proxy it is calibrated
(copy share = w in every bucket) with AUC 0.913.
The layer re-estimates P(copy) with a small LightGBM classifier whose init score is logit(w) (so it only learns what
v57 lacks) and whose inputs are label-free context of the SAME period, read from every LIRF departure around the row:
  fn_wk_days   days the same flight number flew within +-7 d (cut to covered dates, scaled to 14; 0 = one-off number)
  fn_wk_unm    share of those other departures that are NM-unmatched
  fn_wk_late   share of them more than 1 h late
  next_n/prev_n same-number departure on the next / previous local day (NaN when that day is not in the data)
  next_late    the next-day one is > 1 h late
  al3_late_n   the carrier's departures 1-8 h late within +-3 d (per 7 covered days)
  al3_late_unm share of those that are NM-unmatched
  bank_unm     the carrier's NM-unmatched departures scheduled within +-2 h
New prediction p' = p + (P - w)(D - U): only the mixture weight changes. Scope: LIRF unmatched, gate-valid, no
helicopters, 1 h < D <= 8 h, status no_recovered_lobt (recovered-LOBT rows are placed by rules F/R/B), BEL excluded
(v56 bet), rows where v57 is not a two-point mixture (w_imp outside [-0.02, 1.02]) excluded. rules_v55 re-applied.
2025 label history (flight / carrier copy rates) is deliberately NOT an input: it wins across folds but loses forward
in time (Jan-Jun -> Jul-Dec). No official score is used.
"""
import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import final_rank  # noqa: E402  final-file change: rank-population size (was a hard-coded 344,841)

b53 = importlib.import_module('263_v53_build')
v55 = importlib.import_module('267_v55_build')
OUT = common.RUNS / 'v63_context_layer'
AUD = common.RUNS / 'lirf_row_audit'
GATE_TABLE = common.RUNS / 'v54_gate' / 'table.parquet'   # final-file change: was a literal in frame(); final_chain repoints it
V57 = common.SUBS / 'gentle-tractor_v57.parquet'
V63 = common.SUBS / 'gentle-tractor_v63.parquet'
N_2025, R_PROXY = 2085047, 232.0555   # rows and RMSE of the v57 2025 proxy
LO, HI, EPS = 3600.0, 28800.0, 0.005
YEAR = ['fn_wk_days', 'fn_wk_unm', 'fn_wk_late', 'next_n', 'prev_n', 'next_late', 'al3_late_n', 'al3_late_unm', 'bank_unm']
X = ['lw'] + YEAR
PRM = dict(n_estimators=75, num_leaves=7, min_child_samples=40, learning_rate=0.03, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)
SEEDS = range(7, 17)
FWD = {'H1->H2': (range(1, 7), range(7, 13)), 'H2->H1': (range(7, 13), range(1, 7)),
       'Jan-Jun->Sep-Dec': (range(1, 7), range(9, 13)), 'Jul-Dec->Jan-Apr': (range(7, 13), range(1, 5))}


def norm(fl):
    """Doubled airline prefixes (TARAR753, WMTMT4SG, SVII3006) -> TAR753 (runs/lirf_row_audit/dossier.norm)."""
    if not isinstance(fl, str) or len(fl) < 4:
        return fl
    p, r = fl[:3], fl[3:]
    for k in (p, p[1:], p[2:]):
        if r.startswith(k) and len(r) > len(k):
            return p + r[len(k):]
    return fl


def _context(dep: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Label-free same-period context, windows cut to the covered local dates (2025, Jan + Jul 2026)."""
    covered = set(dep.sd.dropna().unique())
    by_fn = {k: v for k, v in dep.groupby('fn')}
    by_al = {k: v for k, v in dep.groupby('al')}
    out = []
    for r in s.itertuples():
        sd = r.sched_loc.normalize()
        o = by_fn.get(r.fn, dep.iloc[:0])
        o = o[o.MVT_ID_mvt != r.MVT_ID_mvt]
        avail = [sd + pd.Timedelta(days=i) for i in range(-7, 8) if i != 0 and sd + pd.Timedelta(days=i) in covered]
        wk = o[o.sd.isin(avail)]
        nd, pd_ = sd + pd.Timedelta('1D'), sd - pd.Timedelta('1D')
        nxt = o[o.sd == nd]
        a = by_al.get(r.al, dep.iloc[:0])
        a = a[a.MVT_ID_mvt != r.MVT_ID_mvt]
        a3 = a[(a.sched_loc >= r.sched_loc - pd.Timedelta('3D')) & (a.sched_loc <= r.sched_loc + pd.Timedelta('3D'))]
        n3 = sum(1 for i in range(-3, 4) if sd + pd.Timedelta(days=i) in covered)
        a3d = a3[(a3.D > LO) & (a3.D <= HI)]
        bank = a[(a.sched_loc - r.sched_loc).abs() <= pd.Timedelta('2h')]
        out.append(dict(MVT_ID_mvt=r.MVT_ID_mvt, n_avail=len(avail),
                        fn_wk_days=wk.sd.nunique() / max(len(avail), 1) * 14,
                        fn_wk_unm=(~wk.nm).mean() if len(wk) else np.nan,
                        fn_wk_late=(wk.D > LO).mean() if len(wk) else np.nan,
                        next_n=len(nxt) if nd in covered else np.nan,
                        prev_n=int((o.sd == pd_).sum()) if pd_ in covered else np.nan,
                        next_late=float((nxt.D > LO).any()) if len(nxt) else np.nan,
                        al3_late_n=len(a3d) / max(n3, 1) * 7, al3_late_unm=(~a3d.nm).mean() if len(a3d) else np.nan,
                        bank_unm=int((~bank.nm).sum())))
    return pd.DataFrame(out)


def frame() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(AUD / 'lirf_rows.parquet')      # LIRF departures both years, v57 proxy / v57 file as p (scripts/271 frame)
    g = pd.read_parquet(GATE_TABLE, columns=['MVT_ID_mvt', 'U'])   # final-file change: was a literal RUNS/v54_gate path
    f = f.merge(g, on='MVT_ID_mvt', how='left')
    f['st'] = b53.status(f)
    f['fn'] = f.FLIGHT_mvt.map(norm)
    f['al'] = f.fn.str[:3]
    f = f[f.is_heli == 0]
    dep = f[['MVT_ID_mvt', 'fn', 'al', 'nm', 'D', 'sched_loc']].copy()
    dep['sd'] = dep.sched_loc.dt.normalize()
    s = f[~f.nm & f.valid.fillna(False) & (f.D > LO) & (f.D <= HI)].copy()
    s['U2'] = s.U.fillna(1010.0)
    s['w_imp'] = (s.p - s.U2) / (s.D - s.U2)
    s['bel'] = s.fn.str.startswith('BEL', na=False)
    s['in_scope'] = (s.st == 'no_recovered_lobt') & ~s.bel & s.w_imp.between(-0.02, 1.02)
    s['k'] = np.select([s.split.eq('rank'), s.cls.eq('copy'), s.cls.eq('normal')], ['', 'copy', 'normal'], 'other')
    s = s.merge(_context(dep, s), on='MVT_ID_mvt', how='left', validate='one_to_one')
    keep = ['MVT_ID_mvt', 'split', 'fold', 'month', 'st', 'in_scope', 'FLIGHT_mvt', 'fn', 'al', 'STAND_mvt', 'sched_loc',
            'tko_loc', 'D', 'y', 'cls', 'k', 'p', 'U', 'U2', 'w_imp', 't', 'lobt', 'bf_lobt', 'n_avail'] + YEAR
    s = s[keep].sort_values(['split', 'sched_loc']).reset_index(drop=True)
    s.to_parquet(OUT / 'frame.parquet', index=False)
    print(s.groupby(['split', 'in_scope', 'k']).size().unstack(fill_value=0))
    return s


def load(scope_only: bool = True) -> pd.DataFrame:
    s = pd.read_parquet(OUT / 'frame.parquet')
    if scope_only:
        s = s[s.in_scope].reset_index(drop=True)
    s['copy'] = (s.k == 'copy').astype(float)
    s['wc'] = s.w_imp.clip(EPS, 1 - EPS)
    s['lw'] = np.log(s.wc / (1 - s.wc))
    return s


def fit(tr: pd.DataFrame, te: pd.DataFrame, cols=None) -> np.ndarray:
    cols = cols or X
    raw = np.zeros(len(te))
    for sd in SEEDS:
        m = lgb.LGBMClassifier(random_state=sd, **PRM).fit(tr[cols], tr['copy'], init_score=tr.lw.to_numpy())
        raw += m.predict(te[cols], raw_score=True) / len(SEEDS)
    return 1 / (1 + np.exp(-(raw + te.lw.to_numpy())))


def moved(t: pd.DataFrame, P: np.ndarray) -> np.ndarray:
    return np.maximum(t.p.to_numpy() + (P - t.wc.to_numpy()) * (t.D.to_numpy() - t.U2.to_numpy()), b53.FLOOR)


def dsse(t: pd.DataFrame, P: np.ndarray) -> np.ndarray:
    return (moved(t, P) - t.y.to_numpy()) ** 2 - (t.p - t.y).to_numpy() ** 2


def parts(t, P) -> dict:
    d = dsse(t, P)
    mv = moved(t, P) - t.p.to_numpy()
    m2 = float((mv ** 2).sum())
    return {'rows': int(len(t)), 'dsse_M': round(d.sum() / 1e6, 1), 'sum_move2_M': round(m2 / 1e6, 1),
            'gain_per_move2': round(d.sum() / m2, 2) if m2 else None,
            'best_scale': round(-(d.sum() - m2) / (2 * m2), 2) if m2 else None}


def oof(tr: pd.DataFrame, cols=None) -> np.ndarray:
    P = np.full(len(tr), np.nan)
    for fd in sorted(tr.fold.unique()):
        mk = (tr.fold == fd).to_numpy()
        P[mk] = fit(tr[~mk], tr[mk], cols)
    return P


def buckets(t, P):
    b = pd.cut(P, [0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0], include_lowest=True)
    return {str(k): {'rows': int(len(x)), 'copy': int((x.k == 'copy').sum()), 'normal': int((x.k == 'normal').sum()),
                     'other': int((x.k == 'other').sum()), 'P_mean': round(float(x.P.mean()), 3)}
            for k, x in t.assign(P=P, b=b).groupby('b', observed=True)}


def caught(t, P):
    out = {}
    for k in ('copy', 'normal'):
        q = P[(t.k == k).to_numpy()]
        q = q if k == 'copy' else 1 - q
        out[k] = {'right_ge_0.5': int((q >= 0.5).sum()), 'hedged_0.1_0.5': int(((q >= 0.1) & (q < 0.5)).sum()),
                  'wrong_lt_0.1': int((q < 0.1).sum())}
    return out


def screen():
    s = load()
    tr, rk = s[s.split == 'train'].copy(), s[s.split == 'rank'].copy()
    P = oof(tr)
    d = dsse(tr, P)
    tr['P'], tr['dsse'] = P, d
    srt = np.sort(d)
    rep = {'rows_2025': int(len(tr)), 'rows_2026': int(len(rk)), 'classes_2025': tr.k.value_counts().to_dict(),
           'classifier': {'logloss_v57': round(log_loss(tr['copy'], tr.wc), 4), 'logloss_new': round(log_loss(tr['copy'], P), 4),
                          'auc_v57': round(roc_auc_score(tr['copy'], tr.wc), 4), 'auc_new': round(roc_auc_score(tr['copy'], P), 4),
                          'brier_v57': round(float(np.mean((tr.wc - tr['copy']) ** 2)), 4),
                          'brier_new': round(float(np.mean((P - tr['copy']) ** 2)), 4)},
           'buckets_v57': buckets(tr, tr.wc.to_numpy()), 'buckets_new': buckets(tr, P),
           'caught_v57': caught(tr, tr.wc.to_numpy()), 'caught_new': caught(tr, P),
           'oof': parts(tr, P), 'd_rmse_vs_proxy': round(float(np.sqrt(R_PROXY ** 2 + d.sum() / N_2025) - R_PROXY), 4),
           'by_fold_M': tr.groupby('fold').dsse.sum().div(1e6).round(1).to_dict(),
           'by_month_M': {int(k): v for k, v in tr.groupby('month').dsse.sum().div(1e6).round(1).items()},
           'by_class_M': tr.groupby('k').dsse.sum().div(1e6).round(1).to_dict(),
           'by_band_M': {str(k): round(v / 1e6, 1) for k, v in tr.groupby(pd.cut(tr.D, [LO, 7200, 14400, HI]), observed=True).dsse.sum().items()},
           'rows_better_worse': [int((d < 0).sum()), int((d > 0).sum())],
           'top10_share': round(float(srt[:10].sum() / d.sum()), 3), 'without_top10_M': round(float(srt[10:].sum() / 1e6), 1),
           'forward': {}, 'commit': {}, 'past_layer_signals_through_this_harness': {}}
    for name, (a_m, b_m) in FWD.items():
        A, B = tr[tr.month.isin(a_m)], tr[tr.month.isin(b_m)]
        Pb = fit(A, B)
        rep['forward'][name] = {**parts(B, Pb), 'auc_v57': round(roc_auc_score(B['copy'], B.wc), 4), 'auc_new': round(roc_auc_score(B['copy'], Pb), 4)}
    for th in (0.9, 0.95, 0.98):
        Q = np.where(P >= th, 1.0, np.where(P <= 1 - th, 0.0, P))
        rep['commit'][str(th)] = {'rows_committed': int(((P >= th) | (P <= 1 - th)).sum()),
                                  'vs_mixture_M': round((dsse(tr, Q).sum() - d.sum()) / 1e6, 1)}
    # signals of earlier candidate layers, re-run through the same harness
    ctx = pd.read_parquet(AUD / 'lirf_rows.parquet', columns=['MVT_ID_mvt', 'arr_copy_6h', 'inb_copy', 'has_inbound', 'day_gap'])
    t2 = tr.merge(ctx, on='MVT_ID_mvt', how='left')
    t2['mid'] = (t2.day_gap >= 1).astype(float)
    t2['inb_copy'] = t2.inb_copy.astype(float)
    t2['has_inbound'] = t2.has_inbound.astype(float)
    for name, cols in {'arrival-copy rate (v58 candidate)': ['arr_copy_6h'], 'next-day signals (v61 candidate)': ['next_n', 'mid'],
                       'lead #17 inbound copy (never shipped)': ['inb_copy', 'has_inbound'], 'this layer': YEAR}.items():
        c = ['lw'] + cols
        Pc = oof(t2, c)
        rep['past_layer_signals_through_this_harness'][name] = {'oof': parts(t2, Pc), **{k: parts(t2[t2.month.isin(b)], fit(t2[t2.month.isin(a)], t2[t2.month.isin(b)], c))['dsse_M']
                                                                                          for k, (a, b) in FWD.items()}}
    P26 = fit(tr, rk)
    mv26 = moved(rk, P26) - rk.p.to_numpy()
    m2 = float((mv26 ** 2).sum())
    fw_g = [v['gain_per_move2'] for v in rep['forward'].values()]
    rep['rank_2026'] = {'rows': int(len(rk)), 'sum_move2_M': round(m2 / 1e6, 1), 'rms_move_s': round(float(np.sqrt(np.mean(mv26 ** 2))), 1),
                        'mean_move_s': round(float(mv26.mean()), 1), 'rows_abs_move_gt_1000': int((np.abs(mv26) > 1000).sum()),
                        'dsse_2026_M_at_forward_rates': {f'gain/move2 {g}': round(g * m2 / 1e6, 1) for g in fw_g},
                        'dsse_2026_M_at_oof_rate': round(rep['oof']['gain_per_move2'] * m2 / 1e6, 1),
                        'dsse_2026_M_if_no_information': round(m2 / 1e6, 1)}
    tr.drop(columns=['dsse']).assign(dsse_M=d / 1e6).to_parquet(OUT / 'oof.parquet', index=False)
    (OUT / 'screen.json').write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


def build():
    if V63.exists():
        raise SystemExit(f'{V63} exists; refusing to overwrite')
    s = load()
    tr, rk = s[s.split == 'train'].copy(), s[s.split == 'rank'].reset_index(drop=True)
    sub = pd.read_parquet(V57)
    assert len(sub) == final_rank.n_rank() and sub.MVT_ID_mvt.is_unique   # final-file change: was a fixed 344,841
    base = sub.set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)
    assert np.allclose(rk.MVT_ID_mvt.map(base).to_numpy(), rk.p.to_numpy()), 'frame p is not the v57 file'
    P = fit(tr, rk)
    st = np.array(['no_recovered_lobt'] * len(rk), dtype=object)
    new = v55.rules_v55(rk, moved(rk, P), st)[0]
    pos = pd.Series(np.arange(len(sub)), index=sub.MVT_ID_mvt.to_numpy())[rk.MVT_ID_mvt.to_numpy()].to_numpy()
    out = sub.copy()
    out.loc[pos, 'TAXITIME_SEC_mvt'] = new
    v = out.TAXITIME_SEC_mvt.astype(float).to_numpy()
    assert np.isfinite(v).all() and (v >= b53.FLOOR).all()
    changed = v != sub.TAXITIME_SEC_mvt.to_numpy(float)
    assert changed.sum() == int((np.abs(new - rk.p.to_numpy()) > 0).sum()) and changed.sum() <= len(rk)
    out.to_parquet(V63, index=False)
    pkg = common.RUNS / 'models' / f'A70_{datetime.now().strftime("%Y%m%d_%H%M%S")}_v63'
    pkg.mkdir(parents=True)
    mv = new - rk.p.to_numpy()
    rows = rk[['MVT_ID_mvt', 'FLIGHT_mvt', 'fn', 'al', 'STAND_mvt', 'sched_loc', 'tko_loc', 'D', 'U2', 'w_imp'] + YEAR].assign(
        P=P, v57=rk.p.to_numpy(), v63=new, move_s=mv)
    rows.to_parquet(pkg / 'layer_rows_2026.parquet', index=False)
    scr = json.loads((OUT / 'screen.json').read_text())
    info = {'approach': 'A70', 'version': 63, 'base': V57.name, 'base_sha256': hashlib.sha256(V57.read_bytes()).hexdigest(),
            'file_sha256': hashlib.sha256(V63.read_bytes()).hexdigest(), 'layer': 'unmatched_context_layer',
            'rows_2026_scope': int(len(rk)), 'rows_changed': int(changed.sum()),
            'sum_sq_move_2026_M': round(float((mv ** 2).sum() / 1e6), 2), 'rms_move_2026_s': round(float(np.sqrt(np.mean(mv ** 2))), 1),
            'model': {'features': X, 'params': PRM, 'seeds': list(SEEDS), 'init_score': 'logit(clip(w_imp, 0.005, 0.995))',
                      'scope': 'LIRF NM-unmatched, gate-valid, no heli, 3600 < D <= 28800, status no_recovered_lobt, no BEL, w_imp in [-0.02, 1.02]'},
            'evidence_2025': {k: scr[k] for k in ('classifier', 'oof', 'd_rmse_vs_proxy', 'by_fold_M', 'forward', 'top10_share',
                                                  'caught_v57', 'caught_new', 'commit', 'past_layer_signals_through_this_harness')},
            'expected_2026': scr['rank_2026']}
    (pkg / 'build.json').write_text(json.dumps(info, indent=2, default=str))
    (pkg / '274_unmatched_context_layer.py').write_bytes(Path(__file__).read_bytes())
    for n in ('screen.json', 'frame.parquet'):
        (pkg / n).write_bytes((OUT / n).read_bytes())
    print(json.dumps({k: v for k, v in info.items() if k != 'evidence_2025'}, indent=1, default=str))


def ledger():
    """2025 per-row v63 proxy for the Explorer: the v57 proxy (runs/v56_build/rows.parquet) with the out-of-fold layer move
    on the scope rows (screen's oof.parquet) -> runs/v63_context_layer/rows.parquet."""
    rows = pd.read_parquet(common.RUNS / 'v56_build' / 'rows.parquet', columns=['MVT_ID_mvt', 'split', 'fold', 'y', 'v57_pred_s'])
    o = pd.read_parquet(OUT / 'oof.parquet', columns=['MVT_ID_mvt', 'p', 'P', 'wc', 'D', 'U2'])
    o['v63'] = moved(o, o.P.to_numpy())
    rows = rows.merge(o[['MVT_ID_mvt', 'p', 'P', 'v63']], on='MVT_ID_mvt', how='left', validate='one_to_one')
    hit = rows.v63.notna()
    assert np.allclose(rows.loc[hit, 'p'], rows.loc[hit, 'v57_pred_s']), 'oof p is not the v57 proxy'
    rows['v63_pred_s'] = np.where(hit, rows.v63, rows.v57_pred_s)
    rows['layer63'] = hit
    rows['P63'] = rows.P
    rows = rows.drop(columns=['p', 'P', 'v63'])
    tr = rows[rows.split == 'train']
    r57, r63 = (float(np.sqrt(((tr[c] - tr.y) ** 2).mean())) for c in ('v57_pred_s', 'v63_pred_s'))
    rows.to_parquet(OUT / 'rows.parquet', index=False)
    print(json.dumps({'rows': int(len(rows)), 'layer_rows_2025': int((hit & (rows.split == 'train')).sum()),
                      'rmse_v57_proxy': round(r57, 4), 'rmse_v63_proxy': round(r63, 4)}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['frame', 'screen', 'build', 'ledger'])
    a = ap.parse_args()
    {'frame': frame, 'screen': screen, 'build': build, 'ledger': ledger}[a.mode]()
