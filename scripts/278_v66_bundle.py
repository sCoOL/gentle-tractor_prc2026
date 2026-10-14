"""v66 = v63 + L1 push-phase residual layer + L2 LIRF copy layer (v66 bundle, built 2026-10-07).

  py -3.14 scripts/278_v66_bundle.py l1frame    # label-free L1 inputs, both years -> runs/v66_bundle/l1_frame.parquet
  py -3.14 scripts/278_v66_bundle.py l1screen --mode sixfold|forward   # 2025 out of fold / forward, 3 seeds
  py -3.14 scripts/278_v66_bundle.py l2screen   # cross-fitted pure-number rule + F_range rule on 2025
  py -3.14 scripts/278_v66_bundle.py build      # fit L1 on all 2025, apply L1 + L2 to v63 -> v66 + package A73


L1 (bundle item 1+, replaces item 1): per airport, LightGBM on v63's residual r = y - v63 with the 15 v2 ADS-B columns,
the local hour, v63 itself and the ADS-B push-back phases (scripts/adsb_push_phases.py -> <PRC_ADSB_DIR>/
push_phases.parquet: ground-speed push start, push end, first forward move under own power, hold, push distance / turn).
Only EDDF, EDDM, EGLL, EHAM, LEBL, LEMD (LIRF and LSZH lost in the 2026-10-07 screen); only rows with a v2 ADS-B match.
p66 = rules_v55(v63 + alpha * correction).
L2 (bundle item 4), LIRF only: recovered-LOBT rows (status F_range) 1-8 h late -> the copy label D = take-off -
schedule, through rules_v55. Bundle item 2 ((a) NM-matched rows whose flight number was a schedule copy on >= 95 % of its
>= 15 2025 matched rows with D > 0.5 h -> D) is screened by l2screen but NOT applied (WITH_PURE = False): right on 99 % of
rows yet cross-fitted -2.6 M s^2 and forward +16.7 M. L1 alpha 0.75 chosen from {0.5, 0.75, 1} on the forward split.
Built 2026-10-07 22:16: v66 = A73, package runs/models/A73_20261007_221655_v66. Bundle item 3 (Jan 1 G10 lead-in) is NOT in v66: it needs the A51 chain re-run
on rebuilt stand-timeline features. Items 8-9 are excluded.
All inputs are label-free; labels are used only to fit / judge on 2025. No official score is used.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import final_rank  # noqa: E402  final-file change: rank-population size (was a hard-coded 344,841)

b53 = importlib.import_module('263_v53_build')
v55 = importlib.import_module('267_v55_build')
OUT = common.RUNS / 'v66_bundle'
V63 = common.SUBS / 'gentle-tractor_v63.parquet'
V66 = common.SUBS / 'gentle-tractor_v66.parquet'
ROWS63 = common.RUNS / 'v63_context_layer' / 'rows.parquet'
ADSB = common.prc_config.ADSB_DIR   # v2 features + push phases
N25 = 2084660
L1_APTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD']
FOLDS = list('ABCDEF')
SEEDS = (7, 8, 9)
ALPHAS = (0.0, 0.5, 0.75, 1.0)
PARAMS = dict(objective='regression', learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8, deterministic=True,
              force_row_wise=True)
ROUNDS = 300
OLD = ['adsb_how', 'adsb_at_stand', 'adsb_censored', 'adsb_after_landing', 'adsb_taxi_leave', 'adsb_taxi_leave_hi',
       'adsb_taxi_first', 'adsb_taxi_clear', 'adsb_wait', 'adsb_gap', 'adsb_dstand', 'adsb_gs0', 'adsb_nground',
       'adsb_leave_minus_a3', 'adsb_first_minus_a3', 'hour', 'pred']
NEW = ['x_taxi_move', 'x_taxi_pend', 'x_taxi_fwd', 'x_taxi_first', 'push_s', 'push_m', 'push_turn', 'hold_s',
       'x_ptype', 'x_fstate', 'stand_pts', 'u_move', 'u_fwd', 'u_push_end', 'gap_push', 'max_gap_s', 'n_runs',
       'x_rel_first', 'x_a3_move', 'x_a3_fwd']
X = OLD + NEW
PTYPES = ['nomove', 'nocusp', 'forward_seen', 'forward_stand', 'cusp_push', 'push_stand', 'push_heading']
FSTATES = ['standing', 'pushing', 'forward', 'moving']
COPY_TOL, D_PURE, N_PURE, SHARE_PURE = 120.0, 1800.0, 15, 0.95
F_LO, F_HI = 3600.0, 28800.0
WITH_PURE = False   # L2a (pure numbers) failed its screen: cross-fitted -2.6 M, forward +16.7 M (l2_screen.json)
ALPHA = 0.75        # best forward total (-383.7 M) with every fold and January better in both protocols


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rmse(e2) -> float:
    return float(np.sqrt(np.mean(e2)))


# ---------------------------------------------------------------------------------------------------------------- L1
def l1frame():
    """Label-free L1 inputs for every v2-matched departure of both years (2025 y / fold / v63 proxy attached)."""
    OUT.mkdir(parents=True, exist_ok=True)
    f2 = pd.read_parquet(ADSB / 'adsb_features_v2.parquet',
                         columns=['MVT_ID_mvt', 'APT', 't_mvt'] + [c for c in OLD if c.startswith('adsb_')])
    ph = pd.read_parquet(ADSB / 'push_phases.parquet').drop(columns=['APT', 'date', 'version'], errors='ignore')
    d = f2.merge(ph, on='MVT_ID_mvt', how='left', validate='one_to_one')
    base = pd.concat([pd.read_parquet(common.DATA / f, columns=['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt',
                                                                 'month']).assign(split=s)
                      for f, s in [('base_train.parquet', 'train'), ('base_rank.parquet', 'rank')]], ignore_index=True)
    base['hour'] = np.nan
    for a, g in base.groupby('APT'):
        if a in common.AIRPORT_META:
            base.loc[g.index, 'hour'] = pd.to_datetime(g.MVT_TIME_UTC_mvt, utc=True).dt.tz_convert(
                common.AIRPORT_META[a]['tz']).dt.hour.to_numpy()
    base['t_a3'] = b53._epoch(base.AOBT_3_flt)
    d = d.merge(base[['MVT_ID_mvt', 'split', 'month', 'hour', 't_a3']], on='MVT_ID_mvt', how='inner', validate='one_to_one')
    rows = pd.read_parquet(ROWS63, columns=['MVT_ID_mvt', 'split', 'fold', 'y', 'v63_pred_s'])
    sub = pd.read_parquet(V63).rename(columns={'TAXITIME_SEC_mvt': 'v63_file'})
    d = d.merge(rows.drop(columns='split'), on='MVT_ID_mvt', how='left').merge(sub, on='MVT_ID_mvt', how='left')
    d = d[(d.split == 'train') | d.MVT_ID_mvt.isin(sub.MVT_ID_mvt)].reset_index(drop=True)   # final-file change (no-op on the final file)
    d['pred'] = np.where(d.split == 'train', d.v63_pred_s, d.v63_file).astype(float)
    assert d.pred.notna().all(), 'every v2-matched row must have a v63 value'
    t = d.t_mvt.astype(float)
    d['x_taxi_move'], d['x_taxi_pend'], d['x_taxi_fwd'], d['x_taxi_first'] = t - d.t_move, t - d.t_push_end, t - d.t_fwd, t - d.t_first
    d['x_ptype'] = d.push_type.map({p: i for i, p in enumerate(PTYPES)})
    d['x_fstate'] = d.first_state.map({p: i for i, p in enumerate(FSTATES)})
    d['x_rel_first'] = ((d.hd_first - d.hd_stand + 180) % 360 - 180).abs()
    d['x_a3_move'], d['x_a3_fwd'] = d.t_a3 - d.t_move, d.t_a3 - d.t_fwd
    for c in X:
        d[c] = pd.to_numeric(d[c], errors='coerce').astype(float)
    d['r'] = d.y - d.pred
    keep = ['MVT_ID_mvt', 'APT', 'split', 'fold', 'month', 'y', 'r', 'v63_file', 't_mvt'] + X
    d[keep].to_parquet(OUT / 'l1_frame.parquet', index=False)
    print(json.dumps({'rows': int(len(d)), 'by_split': d.split.value_counts().to_dict(),
                      'with_push_phases': int(d.push_type.notna().sum()),
                      'rank_rows_at_l1_airports': int(((d.split == 'rank') & d.APT.isin(L1_APTS)).sum())}, indent=1))


def _fit_predict(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(te))
    for s in SEEDS:
        m = lgb.train({**PARAMS, 'seed': s}, lgb.Dataset(tr[X], tr.r), ROUNDS)
        out += m.predict(te[X]) / len(SEEDS)
    return out


def _l1_train() -> pd.DataFrame:
    d = pd.read_parquet(OUT / 'l1_frame.parquet')
    return d[(d.split == 'train') & d.APT.isin(L1_APTS)].reset_index(drop=True)


def _with_rules(ids: np.ndarray, p: np.ndarray) -> np.ndarray:
    f = b53.frame().set_index('MVT_ID_mvt').loc[ids].reset_index()
    return v55.rules_v55(f, p, b53.status(f))[0]


def l1screen(mode: str):
    d = _l1_train()
    if mode == 'sixfold':
        splits = [(f, d.fold.ne(f).to_numpy(), d.fold.eq(f).to_numpy()) for f in FOLDS]
    else:
        h1 = (d.month <= 6).to_numpy()
        splits = [('H1->H2', h1, ~h1), ('H2->H1', ~h1, h1)]
    c = np.full(len(d), np.nan)
    for name, trm, tem in splits:
        for a in L1_APTS:
            m = (d.APT == a).to_numpy()
            c[tem & m] = _fit_predict(d[trm & m], d[tem & m])
        print(f'  {mode} {name} done', flush=True)
    d['corr'] = c
    base = d.r.to_numpy() ** 2
    res = {'mode': mode, 'rows': int(len(d)), 'rows_corrected': int(np.isfinite(c).sum())}
    for al in ALPHAS:
        p = _with_rules(d.MVT_ID_mvt.to_numpy(), d.y.to_numpy() - d.r.to_numpy() + al * np.nan_to_num(c))
        e = (d.y.to_numpy() - p) ** 2 - base
        res[f'alpha_{al}'] = {
            'dsse_M': round(e.sum() / 1e6, 1), 'body_M': round(e[d.y <= 3600].sum() / 1e6, 1),
            'by_fold_M': (pd.Series(e).groupby(d.fold).sum() / 1e6).round(1).to_dict(),
            'by_month_M': (pd.Series(e).groupby(d.month).sum() / 1e6).round(1).to_dict(),
            'by_apt_M': (pd.Series(e).groupby(d.APT).sum() / 1e6).round(1).to_dict(),
            'by_apt_rmse_before_after': {a: [round(rmse(base[(d.APT == a).to_numpy()]), 2),
                                             round(rmse((d.y.to_numpy() - p)[(d.APT == a).to_numpy()] ** 2), 2)] for a in L1_APTS},
            'per_row_s2_by_apt': (pd.Series(e).groupby(d.APT).mean()).round(0).to_dict()}
        print(f'alpha {al}: {res[f"alpha_{al}"]["dsse_M"]} M (body {res[f"alpha_{al}"]["body_M"]}), folds '
              f'{res[f"alpha_{al}"]["by_fold_M"]}, Jan {res[f"alpha_{al}"]["by_month_M"].get(1)}', flush=True)
    d[['MVT_ID_mvt', 'APT', 'fold', 'month', 'y', 'r', 'corr']].to_parquet(OUT / f'l1_oof_{mode}.parquet', index=False)
    (OUT / f'l1_screen_{mode}.json').write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------------------------------------------------------- L2
def _l2_frame() -> pd.DataFrame:
    f = b53.frame()
    f = f[f.apt.eq('LIRF') & ~f.is_heli.astype(bool)].copy()
    f['st'] = b53.status(f)
    f['D'] = f.t - f.t_sched
    fl = pd.concat([pd.read_parquet(common.DATA / n, columns=['MVT_ID_mvt', 'FLIGHT_mvt'])
                    for n in ('base_train.parquet', 'base_rank.parquet')], ignore_index=True)
    f = f.merge(fl, on='MVT_ID_mvt', how='left', validate='one_to_one')
    f['fn'] = f.FLIGHT_mvt.astype('string').str.strip()
    rows = pd.read_parquet(ROWS63, columns=['MVT_ID_mvt', 'fold', 'v63_pred_s'])
    sub = pd.read_parquet(V63).rename(columns={'TAXITIME_SEC_mvt': 'v63_file'})
    f = f.merge(rows, on='MVT_ID_mvt', how='left').merge(sub, on='MVT_ID_mvt', how='left')
    f = f[(f.split == 'train') | f.MVT_ID_mvt.isin(sub.MVT_ID_mvt)].reset_index(drop=True)   # final-file change (no-op on the final file)
    f['p'] = np.where(f.split == 'train', f.v63_pred_s, f.v63_file).astype(float)
    f['is_copy'] = (f.y - f.D).abs() < COPY_TOL
    return f


def pure_numbers(tr: pd.DataFrame) -> pd.DataFrame:
    """Flight numbers whose 2025 NM-matched rows with D > 0.5 h were schedule copies on >= 95 % of >= 15 rows."""
    m = tr[(tr.st == 'matched') & (tr.D > D_PURE) & tr.fn.notna()]
    g = m.groupby('fn').agg(n=('is_copy', 'size'), copies=('is_copy', 'sum'))
    g['share'] = g.copies / g.n
    return g[(g.n >= N_PURE) & (g.share >= SHARE_PURE)].sort_values('n', ascending=False)


def l2_rows(f: pd.DataFrame, numbers) -> tuple[np.ndarray, np.ndarray]:
    pure = (f.st == 'matched') & (f.D > D_PURE) & f.fn.isin(set(numbers))
    frng = (f.st == 'F_range') & (f.D > F_LO) & (f.D <= F_HI)
    return pure.to_numpy(), frng.to_numpy()


def _l2_apply(f: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Base everywhere; on the masked rows the copy label D, then rules_v55 on those rows only."""
    p = f.p.to_numpy().copy()
    if mask.any():
        g = f[mask]
        p[mask] = v55.rules_v55(g, g.D.to_numpy(), g.st.to_numpy())[0]
    return p


def l2screen():
    OUT.mkdir(parents=True, exist_ok=True)
    f = _l2_frame()
    tr = f[f.split == 'train'].reset_index(drop=True)
    pn = pure_numbers(tr)
    res = {'pure_numbers_full_2025': {k: [int(v.n), round(float(v.share), 3)] for k, v in pn.iterrows()}}
    res['proxy_rows_rules_would_move'] = int((np.abs(v55.rules_v55(tr, tr.p.to_numpy(), tr.st.to_numpy())[0]
                                                     - tr.p.to_numpy()) > 1e-6).sum())
    e0 = (tr.y - tr.p).to_numpy() ** 2

    def judge(mask, name):
        p = _l2_apply(tr, mask)
        e = (tr.y.to_numpy() - p) ** 2 - e0
        return {'rows': int(mask.sum()), 'dsse_M': round(e.sum() / 1e6, 2),
                'by_fold_M': (pd.Series(e).groupby(tr.fold).sum() / 1e6).round(2).to_dict(),
                'right_copy': int((mask & tr.is_copy.to_numpy()).sum()), 'wrong_not_copy': int((mask & ~tr.is_copy.to_numpy()).sum()),
                'worst_rows': tr.loc[mask].assign(dsse=e[mask]).nlargest(3, 'dsse')[['fn', 'D', 'y', 'p', 'dsse']].round(0).to_dict('records')}
    # (a) pure numbers, cross-fitted: numbers chosen on the other five folds, applied to the held-out fold
    m_cf = np.zeros(len(tr), bool)
    for fo in FOLDS:
        nums = pure_numbers(tr[tr.fold != fo]).index
        m_cf |= l2_rows(tr, nums)[0] & (tr.fold == fo).to_numpy()
    res['pure_crossfit'] = judge(m_cf, 'pure_crossfit')
    h1 = (tr.month <= 6).to_numpy()
    m_fw = (l2_rows(tr, pure_numbers(tr[h1]).index)[0] & ~h1) | (l2_rows(tr, pure_numbers(tr[~h1]).index)[0] & h1)
    res['pure_forward'] = judge(m_fw, 'pure_forward')
    # (b) F_range rows (fixed rule, no fitted parameter)
    res['f_range_2025'] = judge(l2_rows(tr, [])[1], 'f_range')
    rk = f[f.split == 'rank']
    a, b = l2_rows(rk, pn.index)
    res['rank_2026'] = {'pure_rows': int(a.sum()), 'f_range_rows': int(b.sum()),
                        'pure_sum_sq_move_M': round(float(((_l2_apply(rk.reset_index(drop=True), a) - rk.p.to_numpy()) ** 2).sum() / 1e6), 2),
                        'f_range_sum_sq_move_M': round(float(((_l2_apply(rk.reset_index(drop=True), b) - rk.p.to_numpy()) ** 2).sum() / 1e6), 2)}
    (OUT / 'l2_screen.json').write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != 'pure_numbers_full_2025'}, indent=1, default=str))
    print(f'{len(pn)} pure numbers, {int(pn.n.sum())} 2025 rows: {", ".join(pn.index[:15])} ...')


# ---------------------------------------------------------------------------------------------------------------- L3
# Step-2 turnaround link layer ("new link for summer, old link for winter"; omitted from the first v66
# build, added the same night). Rows: NM-matched departures that G10 left without an inbound (st_turn_in_gap NaN) and
# that scripts/231 step 2 links by 'rotation' or 'stand_relaxed' (rotation_tow excluded: 70.7 % precision and it hurt
# in the V52 fill screen), not moved by L1. Small pooled LightGBM on v63's residual with the link's turnaround gap,
# inbound delay, link source, towed flag, scheduled turnaround, v63, local hour, airport. Applied to July 2026 only.
L3_SRC = {'rotation': 3, 'stand_relaxed': 4}
L3_X = ['l3_turn_gap', 'l3_turn_delay', 'l3_src', 'l3_towed', 'l3_sched_turn', 'pred', 'hour', 'apt_code']
L3_PARAMS = dict(objective='regression', learning_rate=0.03, num_leaves=7, min_data_in_leaf=100, feature_fraction=0.9,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8, deterministic=True,
                 force_row_wise=True)
L3_ROUNDS, L3_CLIP = 200, 1800.0
L3_MONTHS_2026 = (7,)          # step-2 link for summer only (the final chain passes 6, 7)
L3_ALPHA = 0.5                 # best in both protocols (six-fold -1.11 M, forward -0.07 M; July +0.11 / +0.05 M = noise)


def l3frame():
    raw = ['MVT_ID_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'STAND_mvt', 'PHASE_mvt']
    m = pd.concat([pd.read_parquet(f, columns=raw) for f in sorted(common.DATA.glob('training_2025-*.parquet'))]
                  + [pd.read_parquet(common.DATA / 'ranking.parquet', columns=raw)], ignore_index=True)
    arr = m[m.PHASE_mvt == 'ARR'].set_index('MVT_ID_mvt')
    dep = m[m.PHASE_mvt == 'DEP'].set_index('MVT_ID_mvt')
    lk = pd.read_parquet(common.RUNS / 'v51_turnaround' / 'links.parquet')
    lk = lk[lk.link_source.isin(L3_SRC)]
    a = arr.reindex(lk.arr_mvt)
    st_a = a.STAND_mvt.astype('string').fillna('').to_numpy()
    st_d = dep.STAND_mvt.reindex(lk.dep_mvt).astype('string').fillna('').to_numpy()
    sched_d = b53._epoch(dep.SCHED_TIME_UTC_mvt.reindex(lk.dep_mvt))
    f = pd.DataFrame({'MVT_ID_mvt': lk.dep_mvt.to_numpy(), 'link_source': lk.link_source.to_numpy(),
                      'l3_turn_gap': np.minimum(lk.turn_s.to_numpy(), 43200.0),
                      'l3_turn_delay': b53._epoch(a.BLOCK_TIME_UTC_mvt) - b53._epoch(a.SCHED_TIME_UTC_mvt),
                      'l3_src': lk.link_source.map(L3_SRC).to_numpy(float),
                      'l3_towed': np.where((st_a != '') & (st_d != ''), (st_a != st_d).astype(float), np.nan),
                      'l3_sched_turn': sched_d - b53._epoch(a.SCHED_TIME_UTC_mvt)})
    g10 = pd.concat([pd.read_parquet(common.DATA / n, columns=['MVT_ID_mvt', 'st_turn_in_gap'])
                     for n in ('features_v5_train.parquet', 'features_v5_rank.parquet')], ignore_index=True)
    fr = b53.frame()
    fr['st'] = b53.status(fr)
    d = fr.merge(f, on='MVT_ID_mvt', how='inner').merge(g10, on='MVT_ID_mvt', how='left')
    d = d[(d.st == 'matched') & d.st_turn_in_gap.isna() & ~d.is_heli.astype(bool)].copy()
    l1 = pd.read_parquet(OUT / 'l1_frame.parquet', columns=['MVT_ID_mvt', 'APT'])
    d = d[~d.MVT_ID_mvt.isin(l1.MVT_ID_mvt[l1.APT.isin(L1_APTS)])]
    rows = pd.read_parquet(ROWS63, columns=['MVT_ID_mvt', 'fold', 'v63_pred_s'])
    sub = pd.read_parquet(V63).rename(columns={'TAXITIME_SEC_mvt': 'v63_file'})
    d = d.merge(rows, on='MVT_ID_mvt', how='left').merge(sub, on='MVT_ID_mvt', how='left')
    d = d[(d.split == 'train') | d.MVT_ID_mvt.isin(sub.MVT_ID_mvt)].reset_index(drop=True)   # final-file change (no-op on the final file)
    d['pred'] = np.where(d.split == 'train', d.v63_pred_s, d.v63_file).astype(float)
    tz = {a: common.AIRPORT_META[a]['tz'] for a in common.AIRPORT_META}
    d['hour'] = [pd.Timestamp(t, unit='s', tz='UTC').tz_convert(tz[a]).hour for t, a in zip(d.t, d.apt)]
    d['apt_code'] = d.apt.map({a: i for i, a in enumerate(sorted(common.AIRPORT_META))}).astype(float)
    d['r'] = d.y - d.pred
    d = d.reset_index(drop=True)
    d.to_parquet(OUT / 'l3_frame.parquet', index=False)
    print(json.dumps({'rows': int(len(d)), 'by_split': d.split.value_counts().to_dict(),
                      'by_source': d.link_source.value_counts().to_dict(),
                      'rank_by_month': {int(k): int(v) for k, v in d[d.split == 'rank'].month.value_counts().items()},
                      'by_apt': d.apt.value_counts().to_dict()}, indent=1))


def _l3_fit_predict(tr, te):
    out = np.zeros(len(te))
    y = tr.r.clip(-L3_CLIP, L3_CLIP)
    for sd in SEEDS:
        m = lgb.train({**L3_PARAMS, 'seed': sd}, lgb.Dataset(tr[L3_X], y), L3_ROUNDS)
        out += m.predict(te[L3_X]) / len(SEEDS)
    return out


def l3screen():
    d = pd.read_parquet(OUT / 'l3_frame.parquet')
    d = d[d.split == 'train'].reset_index(drop=True)
    res = {'rows': int(len(d))}
    h1 = (d.month <= 6).to_numpy()
    for mode, splits in (('sixfold', [(f, d.fold.ne(f).to_numpy(), d.fold.eq(f).to_numpy()) for f in FOLDS]),
                         ('forward', [('H1->H2', h1, ~h1), ('H2->H1', ~h1, h1)])):
        c = np.full(len(d), np.nan)
        for name, trm, tem in splits:
            c[tem] = _l3_fit_predict(d[trm], d[tem])
        base = d.r.to_numpy() ** 2
        summer = d.month.between(4, 9).to_numpy()
        res[mode] = {}
        for al in ALPHAS:
            p = v55.rules_v55(d, d.pred.to_numpy() + al * np.nan_to_num(c), d.st.to_numpy())[0]
            e = (d.y.to_numpy() - p) ** 2 - base
            res[mode][f'alpha_{al}'] = {
                'dsse_M': round(e.sum() / 1e6, 2), 'summer_M': round(e[summer].sum() / 1e6, 2),
                'winter_M': round(e[~summer].sum() / 1e6, 2), 'july_M': round(e[d.month.eq(7).to_numpy()].sum() / 1e6, 2),
                'body_M': round(e[d.y <= 3600].sum() / 1e6, 2),
                'by_fold_M': (pd.Series(e).groupby(d.fold).sum() / 1e6).round(2).to_dict(),
                'by_month_M': {int(k): round(v / 1e6, 2) for k, v in pd.Series(e).groupby(d.month).sum().items()},
                'by_apt_M': (pd.Series(e).groupby(d.apt).sum() / 1e6).round(2).to_dict(),
                'rmse_before_after_summer': [round(rmse(base[summer]), 1), round(rmse((d.y.to_numpy() - p)[summer] ** 2), 1)]}
            r = res[mode][f'alpha_{al}']
            print(f'{mode} alpha {al}: {r["dsse_M"]} M (summer {r["summer_M"]}, winter {r["winter_M"]}, July {r["july_M"]}, '
                  f'body {r["body_M"]}) folds {r["by_fold_M"]}', flush=True)
        d[f'corr_{mode}'] = c
    d[['MVT_ID_mvt', 'apt', 'fold', 'month', 'y', 'r', 'corr_sixfold', 'corr_forward']].to_parquet(OUT / 'l3_oof.parquet', index=False)
    (OUT / 'l3_screen.json').write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------------------- build
def _projection(alpha: float, rk: pd.DataFrame) -> dict:
    """2026 size of L1 (dSSE) from the 2025 per-row gain at each airport x that airport's 2026 rows."""
    out = {}
    for mode in ('sixfold', 'forward'):
        f = OUT / f'l1_screen_{mode}.json'
        if not f.exists():
            continue
        per = json.loads(f.read_text())[f'alpha_{alpha}']['per_row_s2_by_apt']
        tot = sum(per[a] * int((rk.APT == a).sum()) for a in L1_APTS)
        out[mode] = {'dsse_2026_M': round(tot / 1e6, 1)}
    return out


def build(alpha: float):
    if V66.exists():
        raise SystemExit(f'{V66} exists; refusing to overwrite')
    sub = pd.read_parquet(V63)
    assert len(sub) == final_rank.n_rank() and sub.MVT_ID_mvt.is_unique   # final-file change: was a fixed 344,841
    pos = pd.Series(np.arange(len(sub)), index=sub.MVT_ID_mvt.to_numpy())
    p63 = sub.TAXITIME_SEC_mvt.astype(float).to_numpy()
    new = p63.copy()
    # L1
    d = pd.read_parquet(OUT / 'l1_frame.parquet')
    tr = d[(d.split == 'train') & d.APT.isin(L1_APTS)]
    rk = d[(d.split == 'rank') & d.APT.isin(L1_APTS)].reset_index(drop=True)
    assert np.allclose(rk.v63_file.to_numpy(), p63[pos[rk.MVT_ID_mvt].to_numpy()]), 'L1 frame is not the v63 file'
    corr = np.zeros(len(rk))
    for a in L1_APTS:
        m = (rk.APT == a).to_numpy()
        corr[m] = _fit_predict(tr[tr.APT == a], rk[m])
    l1_new = _with_rules(rk.MVT_ID_mvt.to_numpy(), rk.v63_file.to_numpy() + alpha * corr)
    i1 = pos[rk.MVT_ID_mvt].to_numpy()
    new[i1] = l1_new
    # L2
    f = _l2_frame()
    pn = pure_numbers(f[f.split == 'train'])
    r2 = f[f.split == 'rank'].reset_index(drop=True)
    a2, b2 = l2_rows(r2, pn.index)
    m2 = b2 if not WITH_PURE else (a2 | b2)
    assert np.allclose(r2.p.to_numpy(), p63[pos[r2.MVT_ID_mvt].to_numpy()]), 'L2 frame is not the v63 file'
    l2_new = _l2_apply(r2, m2)
    i2 = pos[r2.MVT_ID_mvt[m2]].to_numpy()
    assert not set(i1) & set(i2), 'L1 and L2 must move disjoint rows'
    new[i2] = l2_new[m2]
    # L3: step-2 link layer, summer months only (new link for summer, old link for winter)
    l3 = pd.read_parquet(OUT / 'l3_frame.parquet')
    tr3 = l3[l3.split == 'train']
    rk3 = l3[(l3.split == 'rank') & l3.month.isin(L3_MONTHS_2026)].reset_index(drop=True)
    assert np.allclose(rk3.v63_file.to_numpy(), p63[pos[rk3.MVT_ID_mvt].to_numpy()]), 'L3 frame is not the v63 file'
    c3 = _l3_fit_predict(tr3, rk3)
    l3_new = v55.rules_v55(rk3, rk3.v63_file.to_numpy() + L3_ALPHA * c3, rk3.st.to_numpy())[0]
    i3 = pos[rk3.MVT_ID_mvt].to_numpy()
    assert not set(i3) & (set(i1) | set(i2)), 'L3 must move rows L1 and L2 do not'
    new[i3] = l3_new
    assert np.isfinite(new).all() and (new >= b53.FLOOR).all()
    out = sub.copy()
    out['TAXITIME_SEC_mvt'] = new
    out.to_parquet(V66, index=False)
    mv = new - p63
    changed = mv != 0
    pkg = common.RUNS / 'models' / f'A73_{datetime.now().strftime("%Y%m%d_%H%M%S")}_v66'
    pkg.mkdir(parents=True)
    l1rows = rk[['MVT_ID_mvt', 'APT']].assign(layer='L1_push_phases', corr=corr, v63=rk.v63_file.to_numpy(), v66=l1_new)
    l2rows = r2.loc[m2, ['MVT_ID_mvt', 'fn', 'st', 'D', 'p']].rename(columns={'p': 'v63'}).assign(
        APT='LIRF', layer=np.where(a2[m2] & WITH_PURE, 'L2_pure_number', 'L2_f_range'), v66=l2_new[m2])
    l3rows = rk3[['MVT_ID_mvt', 'apt', 'link_source']].rename(columns={'apt': 'APT'}).assign(
        layer='L3_step2_link', corr=c3, v63=rk3.v63_file.to_numpy(), v66=l3_new)
    lr = pd.concat([l1rows, l2rows, l3rows], ignore_index=True)
    lr['move_s'] = lr.v66 - lr.v63
    lr.to_parquet(pkg / 'layer_rows_2026.parquet', index=False)
    ev = {k: json.loads((OUT / k).read_text()) for k in ('l1_screen_sixfold.json', 'l1_screen_forward.json', 'l2_screen.json',
                                                          'l3_screen.json')
          if (OUT / k).exists()}
    by_layer = lr.groupby('layer').agg(rows=('move_s', 'size'), moved=('move_s', lambda s: int((s != 0).sum())),
                                       sum_sq_move_M=('move_s', lambda s: round(float((s ** 2).sum() / 1e6), 2)),
                                       rms_move_s=('move_s', lambda s: round(float(np.sqrt((s ** 2).mean())), 1)))
    info = {'approach': 'A73', 'version': 66, 'base': V63.name, 'base_sha256': sha(V63), 'file_sha256': sha(V66),
            'rows_changed': int(changed.sum()), 'sum_sq_move_2026_M': round(float((mv ** 2).sum() / 1e6), 2),
            'layers': by_layer.reset_index().to_dict('records'),
            'L1': {'airports': L1_APTS, 'alpha': alpha, 'features': X, 'params': PARAMS, 'rounds': ROUNDS, 'seeds': list(SEEDS),
                   'rows_by_airport': rk.APT.value_counts().to_dict(),
                   'push_phases_sha256': sha(ADSB / 'push_phases.parquet')},
            'L2': {'rule': f'LIRF recovered-LOBT rows (status F_range), {F_LO:.0f} < D <= {F_HI:.0f} s -> schedule copy D, then '
                           'rules_v55 (rule F keeps them >= recovered LOBT + 3600 s)',
                   'pure_numbers_dropped': 'L2a (LIRF matched pure flight numbers -> D) failed its screen: right on 99 % of rows '
                                           'but cross-fitted -2.6 M, forward +16.7 M; not in v66',
                   'pure_numbers_full_2025': {k: [int(v.n), round(float(v.share), 3)] for k, v in pn.iterrows()}},
            'projection_2026_L1': _projection(alpha, rk),
            'L3': {'rule': 'NM-matched departures with no G10 inbound that scripts/231 step 2 links by rotation or '
                           'stand_relaxed, not moved by L1; summer months only (new link for summer, old for winter)',
                   'alpha': L3_ALPHA, 'features': L3_X, 'params': L3_PARAMS, 'rounds': L3_ROUNDS, 'clip': L3_CLIP,
                   'seeds': list(SEEDS), 'rows_2026': int(len(rk3)),
                   'links_sha256': sha(common.RUNS / 'v51_turnaround' / 'links.parquet')},
            'not_in_v66': {'item 3 Jan 1 lead-in': 'needs the A51 chain re-run on rebuilt G10 features; '
                                                   'OPEN since 2026-09-20',
                           'items 8-9': 'excluded (2026-10-07)'},
            'item_5_normaliser': 'DONE 2026-10-07: scripts/flightnorm.py in 255 / 231 / 260; L1 reads the rebuilt v2 match, '
                                 'L3 the rebuilt step-2 links',
            'evidence_2025': ev}
    (pkg / 'build.json').write_text(json.dumps(info, indent=2, default=str))
    for src in (Path(__file__), Path(__file__).parent / 'adsb_push_phases.py', Path(__file__).parent / 'flightnorm.py'):
        (pkg / src.name).write_bytes(src.read_bytes())
    for n in ('l1_screen_sixfold.json', 'l1_screen_forward.json', 'l2_screen.json', 'l3_screen.json'):
        if (OUT / n).exists():
            (pkg / n).write_bytes((OUT / n).read_bytes())
    print(json.dumps({k: v for k, v in info.items() if k not in ('evidence_2025', 'L2')}, indent=1, default=str))
    print(f'package {pkg}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['l1frame', 'l1screen', 'l2screen', 'l3frame', 'l3screen', 'build'])
    ap.add_argument('--mode', default='sixfold', choices=['sixfold', 'forward'])
    ap.add_argument('--alpha', type=float, default=ALPHA)
    a = ap.parse_args()
    if a.cmd == 'l1frame':
        l1frame()
    elif a.cmd == 'l1screen':
        l1screen(a.mode)
    elif a.cmd == 'l2screen':
        l2screen()
    elif a.cmd == 'l3frame':
        l3frame()
    elif a.cmd == 'l3screen':
        l3screen()
    else:
        build(a.alpha)
