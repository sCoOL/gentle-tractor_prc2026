"""ADS-B for the departures WITHOUT an NM match: a twin-difference correction of the unmatched normal expert (2026-09-22).

  py -3.14 scripts/124_unmatched_adsb.py stops [--workers 12]              # per-phase stop sequence -> adsb_stops.parquet
  py -3.14 scripts/124_unmatched_adsb.py fit --block AB [--seeds 41 42 43]  # strict twins, cached in runs/tuning
  py -3.14 scripts/124_unmatched_adsb.py rule --tag A46f --bundle runs/oof/bundle_A45f_strict.parquet --base-rule runs/unmatched_mode_rule_A45f.json
  py -3.14 scripts/124_unmatched_adsb.py deploy --base-package runs/models/A45f_<stamp> --rule runs/unmatched_adsb_rule_A46f.json --evidence <json> --submit

Why: 110's normal experts never read an ADS-B column, so the rows with no AOBT_3 got nothing from the
ADS-B programme. Two LightGBM twins per seed, fitted exactly on 110's population (other months, fixed-wing, no schedule
copies, no date bugs, every NM input masked): BLIND (110's columns) and AB (+ the ADS-B off-block columns without the
two NM-derived ones, + the stop sequence of the ground phase). The delivered unmatched expert becomes

    normal' = normal + lam * mean_seeds(AB - BLIND)        on rows ADS-B saw, unchanged elsewhere

so the set expert keeps what it knows and only what ADS-B ADDS is moved across. Gate weight, mode rule and Pb stay.
The stop sequence exists because 121's off-block is the FIRST move away from where the aircraft was first seen: a tow
or a return to stand makes that hours too early, and the trace shows the re-park (ADS-B taxi >= 1 h is right in 8 % of
2025 rows as measured and in 79 % counted from the last stand departure).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_context  # noqa: E402
import common  # noqa: E402
import fallback_context as fc  # noqa: E402
import tail_experts as te  # noqa: E402
import train_eval  # noqa: E402
import tune_residual as tr  # noqa: E402
from common import RUNS  # noqa: E402

un = importlib.import_module('110_unmatched_normal')
a35 = importlib.import_module('111_bundle_a35')
m114 = importlib.import_module('114_unmatched_mode')

ADSB = [c for c in adsb_context.COLUMNS if not c.endswith('_minus_a3')]     # *_minus_a3 carry AOBT_3: NM-derived
PARAMS = {'learning_rate': 0.05, 'num_leaves': 255, 'min_data_in_leaf': 100}
ROUNDS = 700
SEEDS = [41, 42, 43]
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
MU_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
MU2_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
NEAR_M = 200.0          # the second trust tier: ADS-B saw the aircraft, but this far from the nearest known stand
STOP_R, STOP_S, STAND_STOP_S = 20.0, 180.0, 300.0
CACHE = RUNS / 'models' / 'member_cache_a46'


# ---------------------------------------------------------------- stop sequence (one row per matched ground phase)
def _events():
    spec = importlib.util.spec_from_file_location('p121', Path(__file__).resolve().parent / '121_adsb_events.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def stops_of(t, x, y, S, at_stand_m):
    """[(i0, i1, at_stand)]: runs of points within STOP_R of the run's first point that last >= STOP_S."""
    out, k, n = [], 0, len(t)
    while k < n:
        k2 = k
        while k2 + 1 < n and (x[k2 + 1] - x[k]) ** 2 + (y[k2 + 1] - y[k]) ** 2 <= STOP_R ** 2:
            k2 += 1
        if t[k2] - t[k] >= STOP_S:
            d = float(np.min(np.hypot(S[:, 0] - np.median(x[k:k2 + 1]), S[:, 1] - np.median(y[k:k2 + 1])))) if len(S) else np.nan
            out.append((k, k2, d <= at_stand_m))
        k = k2 + 1
    return out


def stop_row(t, x, y, t_mvt, S, at_stand_m):
    st = stops_of(t, x, y, S, at_stand_m)
    stand = [s for s in st if s[2] and t[s[1]] - t[s[0]] >= STAND_STOP_S]
    off = [s for s in st if not s[2]]
    last = stand[-1] if stand else None
    cum = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    dur = lambda s: float(t[s[1]] - t[s[0]])
    return dict(st_n=len(st), st_n_stand=len(stand), st_n_off=len(off),
                st_taxi_last=float(t_mvt - t[last[1]]) if last else np.nan,
                st_taxi_first_stop=float(t_mvt - t[st[0][1]]) if st and st[0][0] == 0 else np.nan,
                st_last_dwell=dur(last) if last else np.nan,
                st_repark_s=float(sum(dur(s) for s in stand[1:])),
                st_moved_before_last=float(cum[last[0]]) if last else np.nan,
                st_same_stand=float(np.hypot(x[stand[0][0]] - x[last[0]], y[stand[0][0]] - y[last[0]]) <= 60) if len(stand) > 1 else np.nan,
                st_off_longest=float(max((dur(s) for s in off), default=0.0)), st_off_total=float(sum(dur(s) for s in off)),
                st_off_after_last=float(sum(dur(s) for s in off if last is None or s[0] > last[1])))


def _stops_day(date):
    E = _events()
    f = E.OUT / 'stops' / f'stops_{date}.parquet'
    if f.exists():
        return date
    ph = E.phases(date)[0]
    ev, d = E.match_day(date)
    rows = []
    for mvt, k, t_mvt in zip(d.MVT_ID_mvt.values, d.k.values, d.t_mvt.values) if len(d) else []:
        icao, apt, cs, t_tko, j, i, after, t, x, y, gs0 = ph[int(k)]
        if abs(t_tko - ev.t_tko.values[int(k)]) > 1e-6:
            raise RuntimeError(f'{date}: phase order differs from the events file')
        rows.append(dict(MVT_ID_mvt=mvt, **stop_row(t, x, y, t_mvt, E.stands_of(apt), E.AT_STAND_M)))
    f.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(f, index=False)
    return date


def stops_cmd(a):
    E = _events()
    days = [x for x in E.days() if (E.OUT / 'events' / f'events_{x}.parquet').exists()]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        for n, _ in enumerate(pool.map(_stops_day, days), 1):
            if n % 50 == 0:
                print(f'  {n}/{len(days)} days [{time.time() - t0:.0f}s]', flush=True)
    d = pd.concat([pd.read_parquet(f) for f in sorted((E.OUT / 'stops').glob('stops_*.parquet'))], ignore_index=True)
    d = d[d.MVT_ID_mvt.notna()].drop_duplicates('MVT_ID_mvt')
    stops_path = E.OUT / 'adsb_stops.parquet'   # final-file change: was adsb_context.STOPS_PATH (= the SHARED folder)
    d.to_parquet(stops_path, index=False)
    print(f'{stops_path}: {len(d):,} rows [{time.time() - t0:.0f}s]')


# ---------------------------------------------------------------- the twins
def with_trace(df):
    """ADS-B off-block columns + stop sequence + the repaired taxi (from the LAST stand departure when it re-parked)."""
    out = df.merge(adsb_context.table()[['MVT_ID_mvt'] + ADSB], on='MVT_ID_mvt', how='left', validate='one_to_one') \
            .merge(adsb_context.stops_table(), on='MVT_ID_mvt', how='left', validate='one_to_one')
    repark = (out.st_n_stand.values >= 1) & (out.st_taxi_last.values < out.adsb_taxi_leave.values - 300)
    out['st_taxi_fix'] = np.where(repark, out.st_taxi_last.values, out.adsb_taxi_leave.values)
    return out


AB_COLS = ADSB + adsb_context.STOPS_COLUMNS + ['st_taxi_fix']


def fit_twins(train, held, cols, cats, seeds, label):
    import lightgbm as lgb
    res = pd.DataFrame({'MVT_ID_mvt': held.MVT_ID_mvt.values})
    y = train.TAXITIME_SEC_mvt.values
    for name, cc in [('blind', cols), ('ab', cols + AB_COLS)]:
        X = train_eval.prep(train, cc, cats)
        Xh = train_eval.prep(held, cc, cats, train_eval._cats_of(X, cats))
        for s in seeds:
            t0 = time.time()
            m = lgb.train(train_eval._lgb_params({**PARAMS, 'seed': s}),
                          lgb.Dataset(X, y, categorical_feature=[c for c in cats if c in X.columns], free_raw_data=False), ROUNDS)
            res[f'{name}_s{s}'] = m.predict(Xh)
            print(f'{label} {name} seed {s}: {len(train):,} rows x {len(cc)} cols in {time.time() - t0:.0f}s', flush=True)
    return res


def _population(data, extra_mask):
    bug = te.datebug_label(data.TAXITIME_SEC_mvt.values, data.atot_minus_sched.values)
    return extra_mask & ~bug & (data.is_heli.values == 0) & (data.fallback.values == 0)


def fit_cmd(a):
    data, _, manifest = common.load_features(5)
    for block in a.block:
        cache = RUNS / 'tuning' / f'unmatched_adsb_{block}.parquet'
        if cache.exists():
            continue
        blk = tr.load_block(block, 5, '_v5tfb5rd', data=data)
        blocked = common.FOLDS[block[0]] + common.FOLDS[block[1]]
        keep = _population(data, ~data.month.isin(blocked).values)
        drop = set(un.NM_NAN) | set(un.NM_CONST)
        cols = [c for c in blk['cols'] if c not in drop]; cats = [c for c in blk['cats'] if c not in drop]
        assert cols == [c for c in a35._strict_cols(manifest)[0] if c not in drop], 'strict columns differ from 111'
        held = blk['held'][blk['held'].fpl_matched.values == 0]
        res = fit_twins(with_trace(un.mask_nm(data[keep])), with_trace(un.mask_nm(held)), cols, cats, a.seeds, block)
        res.to_parquet(cache, index=False); print(cache, flush=True)


def delta_of(d, seeds=None):
    ab = [c for c in d.columns if c.startswith('ab_s') and (seeds is None or int(c[4:]) in seeds)]
    return d[ab].mean(axis=1).values - d[[c.replace('ab_', 'blind_') for c in ab]].mean(axis=1).values


def ab_of(d, seeds=None):
    ab = [c for c in d.columns if c.startswith('ab_s') and (seeds is None or int(c[4:]) in seeds)]
    return d[ab].mean(axis=1).values


def saw_mask(ids):
    return pd.Series(ids).isin(adsb_context.table().MVT_ID_mvt.values).values


def pushback_mask(ids):
    """Rows whose own push-back ADS-B measured (first seen standing at a stand, a leave time found)."""
    f = adsb_context.table()
    ok = f.MVT_ID_mvt.values[(f.adsb_at_stand.values == 1) & np.isfinite(f.adsb_taxi_leave.values)]
    return pd.Series(ids).isin(ok).values


def near_mask(ids, dmax=NEAR_M):
    """Rows ADS-B saw rolling but NOT standing at their stand, first seen within ``dmax`` metres of one.

    The second trust tier (A48). ``adsb_at_stand`` is a hard 40 m cut, and the off-block's error degrades gracefully
    past it rather than falling off: on 2025 labels the median error goes -8 s at <= 40 m to -161 / -199 / -261 s out
    to 80 m and -335 / -432 s out to 200 m, while the median-centred trimmed spread only widens from 173 s to
    177 / 208 / 190 / 211 / 278 s. The MATCHED trees already price that themselves (best shrink alpha* = 0.00-0.02 in
    every band), but the unmatched lane's mu blend is masked on ``pushback_mask`` and so never reaches these rows -
    24 % of the 2026 unmatched rows, which get the twin difference and nothing else. Past 200 m it is worth nothing
    (the cross-fit drives mu2 to 0) and ``after_landing`` rows are excluded: their phase starts on the runway, so the
    distance says nothing about the off-block.
    """
    f = adsb_context.table()
    ok = f.MVT_ID_mvt.values[(f.adsb_at_stand.values != 1) & (f.adsb_after_landing.values == 0)
                             & np.isfinite(f.adsb_taxi_leave.values)
                             & (f.adsb_dstand.values > 40.0) & (f.adsb_dstand.values <= dmax)]
    return pd.Series(ids).isin(ok).values


def corrected(nrm, delta, ab, push, lam, mu, near=None, mu2=0.0):
    """normal' = normal + lam * (AB - BLIND); where the push-back itself was measured, a share mu of the AB expert
    read directly. The twin difference carries the set expert's level error with it (set 2,700, blind twin 1,900, AB
    900 -> 1,700); the AB expert alone loses the set expert's skill. 2025: set 275.9, twin difference 220.5, AB direct
    215.6, their mean 199.7 on the non-copy unmatched rows with a push-back.

    ``near`` / ``mu2`` are the second tier (``near_mask``): the same read of the AB expert at its own weight on the
    rows ADS-B saw but not at their stand. mu2 = 0 reproduces A46/A47 exactly.
    """
    td = nrm + lam * delta
    out = np.where(push & np.isfinite(ab), (1 - mu) * td + mu * ab, td)
    if near is None or not mu2:
        return out
    return np.where(near & ~push & np.isfinite(ab), (1 - mu2) * td + mu2 * ab, out)


# ---------------------------------------------------------------- strict rule
def rule_cmd(a):
    t = un.unmatched_table()
    um = t.fpl_matched.values == 0; y = t.y.values; D = t.atot_minus_sched.values
    nrm = a35._cached('unmatched_normal_{block}_{name}.parquet', list(a35.NORMAL), 'normal').reindex(t.MVT_ID_mvt).values
    nrm = np.where(um & np.isfinite(nrm), nrm, t.normal_pred.values)
    base = json.loads(Path(a.base_rule).read_text(encoding='utf-8'))
    pu = pd.concat([pd.concat([pd.read_parquet(RUNS / 'tuning' / f'unmatched_mode_{l}{r}_{n}.parquet') for l, r in te.stg.BLOCKS])
                    .set_index('MVT_ID_mvt').p.rename(n) for n in base['candidates']], axis=1).mean(axis=1).reindex(t.MVT_ID_mvt).values
    pu = np.where(np.isfinite(pu), pu, t.p_fallback.values)
    al = np.zeros(len(t))
    for b, c in base['cross_fitted'].items():
        rows = t.block.values == b
        al[rows] = c['alpha'] * ((D[rows] <= c['dmax_h'] * 3600) & ~np.isin(t.APT.values[rows], base['exclude']))
    w = (1 - al) * t.w.values + al * pu; p = (1 - al) * t.p_fallback.values + al * pu
    ref = pd.read_parquet(a.bundle)
    r = t[['MVT_ID_mvt']].merge(ref[['MVT_ID_mvt', 'pred']], on='MVT_ID_mvt', how='left', validate='one_to_one').pred.values
    gap = float(np.nanmax(np.abs(m114.compose(t, nrm, w, p)[um] - r[um])))
    print(f'base unmatched rows reproduced: max gap {gap:.2e}', flush=True)
    if gap > 1e-6:
        raise SystemExit('the base bundle is not reproduced')
    tw = pd.concat([pd.read_parquet(RUNS / 'tuning' / f'unmatched_adsb_{l}{r_}.parquet') for l, r_ in te.stg.BLOCKS])
    delta = pd.Series(delta_of(tw, a.seeds), index=tw.MVT_ID_mvt.values).reindex(t.MVT_ID_mvt.values).values
    delta = np.where(um & np.isfinite(delta) & saw_mask(t.MVT_ID_mvt.values), delta, 0.0)
    ab = pd.Series(ab_of(tw, a.seeds), index=tw.MVT_ID_mvt.values).reindex(t.MVT_ID_mvt.values).values
    push = um & pushback_mask(t.MVT_ID_mvt.values) & (t.is_heli.values == 0)
    near = um & near_mask(t.MVT_ID_mvt.values, a.near_m) & (t.is_heli.values == 0)
    print(f'rows with a measured push-back {int(push.sum())}; second tier (<= {a.near_m:.0f} m) {int(near.sum())}', flush=True)

    def pred(g, rows):
        return m114.compose(t.iloc[np.where(rows)[0]],
                            corrected(nrm[rows], delta[rows], ab[rows], push[rows], g[0], g[1], near[rows], g[2]),
                            w[rows], p[rows])

    pairs = [(l, m_, m2) for l in GRID for m_ in MU_GRID for m2 in (MU2_GRID if a.tier2 else [0.0])]
    best = lambda src: pairs[int(np.argmin([np.sum((pred(g, src) - y[src]) ** 2) for g in pairs]))]
    new = r.copy(); chosen = {}
    for b in sorted(t.block.unique()):
        chosen[b] = best(um & (t.block.values != b))
        rows = um & (t.block.values == b); new[rows] = pred(chosen[b], rows)
    lam, mu, mu2 = best(um)
    out = ref.copy()
    out['pred'] = out[['MVT_ID_mvt']].merge(pd.DataFrame({'MVT_ID_mvt': t.MVT_ID_mvt.values, 'p': np.where(um, new, r)}),
                                            on='MVT_ID_mvt', how='left', validate='one_to_one').p.values
    rm = lambda q, yy: float(np.sqrt(np.mean((q - yy) ** 2)))
    print(f'cross-fitted (lam, mu, mu2) {chosen}; deploy lam {lam}, mu {mu}, mu2 {mu2}')
    print(f'pooled: {rm(ref.pred.values, ref.y.values):.4f} -> {rm(out.pred.values, out.y.values):.4f}')
    for f in sorted(ref.fold.unique()):
        s = ref.fold.values == f
        print(f'  fold {f}: {rm(out.pred.values[s], ref.y.values[s]) - rm(ref.pred.values[s], ref.y.values[s]):+.4f}')
    for x in sorted(ref.APT.unique()):
        s = ref.APT.values == x
        print(f'  {x}: {rm(out.pred.values[s], ref.y.values[s]) - rm(ref.pred.values[s], ref.y.values[s]):+.4f}')
    for g in [(l, m_, m2) for l in (0.75, 1.0) for m_ in MU_GRID for m2 in ({mu2, 0.0} if a.tier2 else {0.0})]:
        print(f'  flat (lam, mu, mu2) {g}: pooled {np.sqrt((np.sum((ref.pred.values - ref.y.values) ** 2) - np.sum((r[um] - y[um]) ** 2) + np.sum((pred(g, um) - y[um]) ** 2)) / len(ref)):.4f}')
    path = RUNS / 'oof' / f'bundle_{a.tag}_strict.parquet'
    out.to_parquet(path, index=False); print(path)
    (RUNS / f'unmatched_adsb_rule_{a.tag}.json').write_text(json.dumps(
        {'base_rule': str(a.base_rule), 'seeds': a.seeds, 'cross_fitted': chosen, 'lam': lam, 'mu': mu,
         'mu2': mu2, 'near_m': a.near_m, 'params': PARAMS, 'rounds': ROUNDS}, indent=1), encoding='utf-8')


# ---------------------------------------------------------------- deployment
def rank_twins(seeds):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f'unmatched_adsb_rank_{"_".join(map(str, seeds))}.parquet'
    if f.exists():
        return pd.read_parquet(f)
    train, rank, manifest = common.load_features(5)
    cols, cats = a35._strict_cols(manifest)
    drop = set(un.NM_NAN) | set(un.NM_CONST)
    cols = [c for c in cols if c not in drop]; cats = [c for c in cats if c not in drop]
    keep = _population(train, np.ones(len(train), bool))
    held = rank[rank.fpl_matched.values == 0]
    res = fit_twins(with_trace(un.mask_nm(train[keep])), with_trace(un.mask_nm(held)), cols, cats, seeds, 'deploy')
    res.to_parquet(f, index=False)
    return res


def deploy_cmd(a):
    import submit
    base = Path(a.base_package); rule = json.loads(Path(a.rule).read_text(encoding='utf-8'))
    d = pd.read_parquet(base / 'rank_detail.parquet')
    ids = d.MVT_ID_mvt.values; um = d.fpl_matched.values == 0
    rk = pd.read_parquet(common.DATA / 'features_v5_rank.parquet', columns=['MVT_ID_mvt', 'is_heli'])
    heli = d[['MVT_ID_mvt']].merge(rk, on='MVT_ID_mvt', how='left', validate='one_to_one').is_heli.values
    exp = RUNS / 'models' / 'A27_20260917_141019'
    p_old = d[['MVT_ID_mvt']].merge(pd.read_parquet(exp / 'p_fallback_rank.parquet'), on='MVT_ID_mvt', how='left',
                                     validate='one_to_one').p_fallback.values
    heli_const = float(json.loads((exp / 'M3rpa' / 'state.json').read_text(encoding='utf-8'))['heli_const'])
    D = d.atot_minus_sched.values; al = d.alpha.values; pu = d.p_unmatched_mode.values
    w = (1 - al) * d.w.values + al * pu; p = (1 - al) * p_old + al * pu

    def unmatched_pred(nrm):
        mix = fc.mixture(p, D, nrm, heli, np.full(len(d), heli_const))
        two = np.where(d.gate_valid.values, np.maximum(nrm + np.clip(w, 0, 1) * (D - nrm), 30.0), mix)
        return te.combine(two, np.nan_to_num(d.pb.values), nrm, d.day_shift.values)

    gap = float(np.max(np.abs(unmatched_pred(d.normal_unmatched_set.values)[um] - d.pred.values[um])))
    print(f'base package unmatched rows reproduced: max gap {gap:.2e}', flush=True)
    if gap > 1e-6:
        raise SystemExit('base unmatched predictions are not reproduced')
    tw = rank_twins(rule['seeds'])
    delta = pd.Series(delta_of(tw), index=tw.MVT_ID_mvt.values).reindex(ids).values
    delta = np.where(um & np.isfinite(delta) & saw_mask(ids) & (heli == 0), delta, 0.0)
    ab = pd.Series(ab_of(tw), index=tw.MVT_ID_mvt.values).reindex(ids).values
    nrm_new = corrected(d.normal_unmatched_set.values, delta, ab, um & pushback_mask(ids) & (heli == 0),
                        rule['lam'], rule.get('mu', 0.0),
                        um & near_mask(ids, rule.get('near_m', NEAR_M)) & (heli == 0), rule.get('mu2', 0.0))
    new = np.where(um, unmatched_pred(nrm_new), d.pred.values)
    stamp = time.strftime('%Y%m%d_%H%M%S'); pkg = RUNS / 'models' / f'{a.approach}_{stamp}'; pkg.mkdir(parents=True, exist_ok=True)
    d.assign(pred_before_adsb_unmatched=d.pred.values, adsb_unmatched_delta=delta, normal_unmatched_adsb=nrm_new, pred=new) \
     .to_parquet(pkg / 'rank_detail.parquet', index=False)
    diff = new - d.pred.values
    summary = {'approach': a.approach, 'base_package': str(base), 'rule': rule, 'base_unmatched_reproduction_gap': gap,
               'rows_changed': int((np.abs(diff) > 1).sum()), 'rms_change_unmatched': float(np.sqrt(np.mean(diff[um] ** 2))),
               'rows_with_delta': int((delta != 0).sum()), 'mean_delta': float(delta[delta != 0].mean()),
               'rows': int(len(new)), 'finite': bool(np.isfinite(new).all()), 'min_pred': float(new.min())}
    (pkg / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1), flush=True)
    if a.submit:
        ev = json.loads(Path(a.evidence).read_text(encoding='utf-8'))
        if not ev.get('passes'):
            raise SystemExit('evidence does not pass the noise guard; no submission written')
        team = submit.team_name(); version = submit.next_version(team)
        path = submit.write_submission(team, version, ids, new)
        desc = (f"{ev['summary']}; strict pooled {ev['pooled']:.2f} ({ev['delta_vs_ref']:+.2f} vs {ev['reference']}), "
                f"worst fold {ev['worst_fold_vs_ref']:+.2f}")
        log = pd.read_csv(common.SUBS / 'submission_log.csv')
        base_models = json.loads(log.loc[log.approach_id == a.log_base, 'models_json'].iloc[-1])
        row = dict(file=path.name, ts=time.strftime('%Y-%m-%d %H:%M:%S'), team=team, version_n=version, approach_id=a.approach_id or a.approach,
                   feature_version=5, models_json=json.dumps(base_models + a.model_id), foldA_rmse=ev['fold_A'], foldB_rmse=ev['fold_B'],
                   result_rmse='', result_rank='', decision='candidate', notes=desc)
        pd.DataFrame([row]).to_csv(common.SUBS / 'submission_log.csv', mode='a', index=False, header=False, lineterminator='\n')
        submit.append_manifest([f'# Candidate {a.approach} {time.strftime("%Y-%m-%d %H:%M")} - team {team}', '',
                                '| file | approach | description | strict Fold A | strict Fold B | pooled six-pair |', '|---|---|---|---:|---:|---:|',
                                f"| {path.name} | {a.approach} | {desc} | {ev['fold_A']:.2f} | {ev['fold_B']:.2f} | {ev['pooled']:.2f} |"])
        (pkg / 'submission.txt').write_text(path.name, encoding='utf-8')
        print(f'{path.name} <- {a.approach}: {desc}\npackage: {pkg}')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('stops'); s.add_argument('--workers', type=int, default=12)
    f = sub.add_parser('fit'); f.add_argument('--block', nargs='+', default=['AB', 'CD', 'EF']); f.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    r = sub.add_parser('rule'); r.add_argument('--tag', required=True); r.add_argument('--bundle', required=True)
    r.add_argument('--base-rule', required=True); r.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    r.add_argument('--tier2', action='store_true', help='also fit mu2 on the rows ADS-B saw but not at their stand (A48)')
    r.add_argument('--near-m', type=float, default=NEAR_M)
    d = sub.add_parser('deploy'); d.add_argument('--base-package', required=True); d.add_argument('--rule', required=True)
    d.add_argument('--approach', default='A46f'); d.add_argument('--approach-id', default='A46'); d.add_argument('--log-base', default='A42')
    d.add_argument('--evidence'); d.add_argument('--submit', action='store_true')
    d.add_argument('--model-id', nargs='+', default=['adsb_unmatched_expert'])
    a = p.parse_args()
    {'stops': stops_cmd, 'fit': fit_cmd, 'rule': rule_cmd, 'deploy': deploy_cmd}[a.cmd](a)


if __name__ == '__main__':
    main()
