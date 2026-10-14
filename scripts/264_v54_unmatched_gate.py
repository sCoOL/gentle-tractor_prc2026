"""Lead #17: label-free airport-recording signals for the unmatched copy gate (v54 candidate).

  py -3.14 scripts/264_v54_unmatched_gate.py table     # rebuild the deployed unmatched lane on 2025 + 2026, add signals
  py -3.14 scripts/264_v54_unmatched_gate.py nested    # 2025 nested evidence, with and without the new signals
  py -3.14 scripts/264_v54_unmatched_gate.py fit       # full-2025 refit -> runs/v54_gate/rank_gate.parquet

The deployed unmatched prediction (A48u, unchanged in A49-A51 = v45) is
    pred = combine(max(U' + w (D - U'), 30), Pb)      on gate-valid rows (finite D, |D - old normal| >= 60 s, no heli)
U' = NM-masked set normal + ADS-B twin correction (124), w = (1 - a) W + a p_u (114's mode rule), D = take-off - schedule,
Pb = LIRF date-bug expert. `table` rebuilds U', w and every other input on the 2025 strict folds exactly as `124 rule`
does (the A48u bundle is reproduced to 1e-6) and on 2026 from A48u's rank_detail (also reproduced).

New label-free signals (arrivals are not scored, so their recorded in-block is known in 2026 too; the turnaround links
of scripts/231 read no departure block time):
    has_inbound       a linked inbound arrival exists (NaN in the first 12 h of each data block, Jan 1 / Jul 1)
    inb_gap_h         inbound in-block - departure schedule (h)
    inb_copy          the inbound arrival's recorded in-block equals its schedule (+-10 s)
    arr_copy_6h       share of the airport's arrivals scheduled within +-3 h whose in-block equals the schedule
The refitted gate is a LightGBM regression of z = (y - U') / (D - U') (clipped to [-0.25, 1.25]) with weight (D - U')^2,
i.e. exactly the squared prediction error of U' + w (D - U'); new w = clip(prediction, 0, 1). Nested: outer
leave-one-fold-out, the shrink toward the deployed w picked by an inner leave-one-fold-out. Arm 'base' uses only the
deployed gate's own inputs (w, D, U', airport, hour) so that 'signals' minus 'base' is the value of the new columns.
"""
import argparse
import glob
import importlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

OUT = common.RUNS / 'v54_gate'
RANK_PKG = common.RUNS / 'models' / 'A48u_20260922_132003'
A27_PKG = common.RUNS / 'models' / 'A27_20260917_141019'   # final-file change: was a literal in rank_lane(); final_chain repoints it
MODE_RULE = common.RUNS / 'unmatched_mode_rule_A48f.json'
ADSB_RULE = common.RUNS / 'unmatched_adsb_rule_A48u.json'
BUNDLE = common.RUNS / 'oof' / 'bundle_A48u_strict.parquet'
LINKS = common.RUNS / 'v51_turnaround' / 'links.parquet'
ADSB_SNAPSHOT = common.prc_config.ADSB_SNAPSHOT / 'adsb_features.parquet'   # the v1 table the 2025 lane was built on
SIGNALS = ['has_inbound', 'inb_gap_h', 'inb_copy', 'arr_copy_6h']
BASE_FEATS = ['w', 'D_h', 'U_h', 'apt_code', 'hour']
ALPHAS = np.array([0, .25, .5, .75, 1.0])
# final-file change: was ['2025-01-01', '2026-01-01', '2026-07-01'] = the first 12 h of each contiguous data block of
# the OLD export (Jan + Jul). The final file's blocks start 2025-01-01, 2026-01-01 (Jan + Feb) and 2026-06-01 (Jun + Jul):
# 07-01 departures now link to 06-30 arrivals (790 links) and 06-01 has no May movements. has_inbound feeds no
# deployed layer (268 layer / 274 build do not read it), so this changes no prediction.
EDGES = ['2025-01-01', '2026-01-01', '2026-06-01']
RANK_DETAIL_DF = None   # final-file change: a DataFrame with A48u rank_detail's columns (+ p_fallback) replaces RANK_PKG's file
TRAIN_TABLE = None      # final-file change: a table.parquet whose 2025 rows replace train_lane() (the reference rows = the deployed fit)


def _epoch(s):
    return ((pd.to_datetime(s, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


# ------------------------------------------------------------------ signals (both years, label-free)
def signals() -> pd.DataFrame:
    cols = ['MVT_ID_mvt', 'ADEP_mvt', 'ADES_mvt', 'PHASE_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt']
    m = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(common.DATA / 'training_2025-*.parquet')))]
                  + [pd.read_parquet(common.DATA / 'ranking.parquet', columns=cols)], ignore_index=True)
    for c in ('SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt'):
        m[c] = _epoch(m[c])
    arr = m[m.PHASE_mvt == 'ARR'].rename(columns={'ADES_mvt': 'apt'})
    arr['acopy'] = (np.abs(arr.BLOCK_TIME_UTC_mvt - arr.SCHED_TIME_UTC_mvt) <= 10).astype(float)
    dep = m[m.PHASE_mvt == 'DEP'].rename(columns={'ADEP_mvt': 'apt'})[['MVT_ID_mvt', 'apt', 'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt']]
    out = []
    for apt, g in dep.groupby('apt'):
        a = arr[arr.apt == apt].sort_values('SCHED_TIME_UTC_mvt')
        ts, cs = a.SCHED_TIME_UTC_mvt.to_numpy(), np.concatenate([[0.0], np.cumsum(a.acopy.to_numpy())])
        s = g.SCHED_TIME_UTC_mvt.to_numpy()
        lo, hi = np.searchsorted(ts, s - 10800), np.searchsorted(ts, s + 10800)
        out.append(pd.DataFrame({'MVT_ID_mvt': g.MVT_ID_mvt.to_numpy(),
                                 'arr_copy_6h': np.where(hi > lo, (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1), np.nan)}))
    sig = dep.merge(pd.concat(out), on='MVT_ID_mvt', how='left')
    links = pd.read_parquet(LINKS, columns=['dep_mvt', 'arr_mvt'])
    sig = sig.merge(links.rename(columns={'dep_mvt': 'MVT_ID_mvt'}), on='MVT_ID_mvt', how='left') \
        .merge(arr[['MVT_ID_mvt', 'BLOCK_TIME_UTC_mvt', 'acopy']].rename(columns={'MVT_ID_mvt': 'arr_mvt', 'BLOCK_TIME_UTC_mvt': 'arr_blk'}),
               on='arr_mvt', how='left')
    edge = np.zeros(len(sig), bool)
    for e in EDGES:
        t0 = pd.Timestamp(e, tz='UTC').timestamp()
        edge |= (sig.MVT_TIME_UTC_mvt.to_numpy() >= t0) & (sig.MVT_TIME_UTC_mvt.to_numpy() < t0 + 43200)
    has = sig.arr_mvt.notna().astype(float).to_numpy()
    sig['has_inbound'] = np.where(edge, np.nan, has)
    sig['inb_gap_h'] = (sig.arr_blk - sig.SCHED_TIME_UTC_mvt) / 3600.0
    sig['inb_copy'] = sig.acopy
    return sig[['MVT_ID_mvt'] + SIGNALS]


# ------------------------------------------------------------------ the deployed unmatched lane
def _modules():
    return (importlib.import_module('110_unmatched_normal'), importlib.import_module('111_bundle_a35'),
            importlib.import_module('114_unmatched_mode'), importlib.import_module('124_unmatched_adsb'))


def train_lane() -> pd.DataFrame:
    """2025 strict: U', w, p, D and the composition inputs, reproducing bundle_A48u_strict on every unmatched row."""
    un, a35, m114, m124 = _modules()
    te = m124.te
    # the A48u bundle (2026-09-22) read the ADS-B table from before the 2026-09-26 rebuild: its masks must too
    orig = m124.adsb_context.table
    m124.adsb_context.table = lambda path=None: orig(ADSB_SNAPSHOT)
    t = un.unmatched_table()
    um = t.fpl_matched.values == 0
    D = t.atot_minus_sched.values
    nrm = a35._cached('unmatched_normal_{block}_{name}.parquet', list(a35.NORMAL), 'normal').reindex(t.MVT_ID_mvt).values
    nrm = np.where(um & np.isfinite(nrm), nrm, t.normal_pred.values)
    base = json.loads(MODE_RULE.read_text(encoding='utf-8'))
    pu = pd.concat([pd.concat([pd.read_parquet(common.RUNS / 'tuning' / f'unmatched_mode_{l}{r}_{n}.parquet') for l, r in te.stg.BLOCKS])
                    .set_index('MVT_ID_mvt').p.rename(n) for n in base['candidates']], axis=1).mean(axis=1).reindex(t.MVT_ID_mvt).values
    pu = np.where(np.isfinite(pu), pu, t.p_fallback.values)
    al = np.zeros(len(t))
    for b, c in base['cross_fitted'].items():
        rows = t.block.values == b
        al[rows] = c['alpha'] * ((D[rows] <= c['dmax_h'] * 3600) & ~np.isin(t.APT.values[rows], base['exclude']))
    w = (1 - al) * t.w.values + al * pu
    p = (1 - al) * t.p_fallback.values + al * pu
    rule = json.loads(ADSB_RULE.read_text(encoding='utf-8'))
    tw = pd.concat([pd.read_parquet(common.RUNS / 'tuning' / f'unmatched_adsb_{l}{r}.parquet') for l, r in te.stg.BLOCKS])
    delta = pd.Series(m124.delta_of(tw, rule['seeds']), index=tw.MVT_ID_mvt.values).reindex(t.MVT_ID_mvt.values).values
    delta = np.where(um & np.isfinite(delta) & m124.saw_mask(t.MVT_ID_mvt.values), delta, 0.0)
    ab = pd.Series(m124.ab_of(tw, rule['seeds']), index=tw.MVT_ID_mvt.values).reindex(t.MVT_ID_mvt.values).values
    push = um & m124.pushback_mask(t.MVT_ID_mvt.values) & (t.is_heli.values == 0)
    near = um & m124.near_mask(t.MVT_ID_mvt.values, rule['near_m']) & (t.is_heli.values == 0)
    U = nrm.copy()
    for b, g in rule['cross_fitted'].items():                       # the bundle used per-block (lam, mu, mu2)
        rows = t.block.values == b
        U[rows] = m124.corrected(nrm[rows], delta[rows], ab[rows], push[rows], g[0], g[1], near[rows], g[2])
    pred = m114.compose(t, U, w, p)
    ref = t[['MVT_ID_mvt']].merge(pd.read_parquet(BUNDLE, columns=['MVT_ID_mvt', 'pred']), on='MVT_ID_mvt', how='left',
                                  validate='one_to_one').pred.values
    gap = float(np.nanmax(np.abs(pred[um] - ref[um])))
    print(f'2025 unmatched rows {int(um.sum())}: max gap vs bundle_A48u_strict {gap:.2e}', flush=True)
    assert gap < 1e-6, 'the deployed unmatched lane is not reproduced'
    valid = np.isfinite(D) & (t.is_heli.values == 0) & (np.abs(D - t.normal_pred.values) >= 60)
    return pd.DataFrame({'MVT_ID_mvt': t.MVT_ID_mvt.values, 'split': 'train', 'APT': t.APT.values, 'fold': t.fold.values,
                         'month': t.month.values, 'y': t.y.values, 'D': D, 'U': U, 'w': w, 'p': p, 'pb': np.nan_to_num(t.pb.values),
                         'day_shift': t.day_shift.values, 'is_heli': t.is_heli.values, 'pred_rd': t.pred_rd.values,
                         'valid': valid, 'pred': pred})[um]


def rank_lane() -> pd.DataFrame:
    """2026: the same inputs from A48u's rank_detail (U' = normal_unmatched_adsb), reproducing its unmatched preds."""
    _, _, _, m124 = _modules()
    te, fc = m124.te, m124.fc
    d = pd.read_parquet(RANK_PKG / 'rank_detail.parquet') if RANK_DETAIL_DF is None else RANK_DETAIL_DF.copy()   # final-file change
    um = d.fpl_matched.values == 0
    rk = pd.read_parquet(common.DATA / 'features_v5_rank.parquet', columns=['MVT_ID_mvt', 'is_heli'])
    heli = d[['MVT_ID_mvt']].merge(rk, on='MVT_ID_mvt', how='left', validate='one_to_one').is_heli.values
    exp = A27_PKG   # final-file change: was a literal RUNS/models/A27_20260917_141019
    if RANK_DETAIL_DF is not None and 'p_fallback' in d:   # final-file change: the rebuilt A27 context probability
        p_old = d.p_fallback.values
    else:
        p_old = d[['MVT_ID_mvt']].merge(pd.read_parquet(exp / 'p_fallback_rank.parquet'), on='MVT_ID_mvt', how='left',
                                         validate='one_to_one').p_fallback.values
    heli_const = float(json.loads((exp / 'M3rpa' / 'state.json').read_text(encoding='utf-8'))['heli_const'])
    D, al, pu = d.atot_minus_sched.values, d.alpha.values, d.p_unmatched_mode.values
    w = (1 - al) * d.w.values + al * pu
    p = (1 - al) * p_old + al * pu
    U = d.normal_unmatched_adsb.values
    mix = fc.mixture(p, D, U, heli, np.full(len(d), heli_const))
    two = np.where(d.gate_valid.values, np.maximum(U + np.clip(w, 0, 1) * (D - U), 30.0), mix)
    pred = te.combine(two, np.nan_to_num(d.pb.values), U, d.day_shift.values)
    gap = float(np.max(np.abs(pred[um] - d.pred.values[um])))
    print(f'2026 unmatched rows {int(um.sum())}: max gap vs A48u rank_detail {gap:.2e}', flush=True)
    assert gap < 1e-6
    return pd.DataFrame({'MVT_ID_mvt': d.MVT_ID_mvt.values, 'split': 'rank', 'APT': d.APT.values, 'fold': None, 'month': np.nan,
                         'y': np.nan, 'D': D, 'U': U, 'w': w, 'p': p, 'pb': np.nan_to_num(d.pb.values), 'day_shift': d.day_shift.values,
                         'is_heli': heli, 'pred_rd': np.full(len(d), heli_const), 'valid': d.gate_valid.values.astype(bool),
                         'pred': pred})[um]


def compose(f: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    """The deployed composition with gate weight ``w`` swapped in (mixture rows and Pb unchanged)."""
    _, _, _, m124 = _modules()
    te, fc = m124.te, m124.fc
    D, U = f.D.to_numpy(), f.U.to_numpy()
    mix = fc.mixture(f.p.to_numpy(), D, U, f.is_heli.to_numpy(), f.pred_rd.to_numpy())
    two = np.where(f.valid.to_numpy(), np.maximum(U + np.clip(w, 0, 1) * (D - U), 30.0), mix)
    return te.combine(two, f.pb.to_numpy(), U, f.day_shift.to_numpy())


def table():
    OUT.mkdir(parents=True, exist_ok=True)
    if TRAIN_TABLE is None:
        f = pd.concat([train_lane(), rank_lane()], ignore_index=True)
    else:   # final-file change: 2025 rows (incl. hour + signals) verbatim from TRAIN_TABLE, rank rows rebuilt
        t25 = pd.read_parquet(TRAIN_TABLE)
        t25 = t25[t25.split == 'train']
        f = pd.concat([t25[['MVT_ID_mvt', 'split', 'APT', 'fold', 'month', 'y', 'D', 'U', 'w', 'p', 'pb', 'day_shift', 'is_heli',
                            'pred_rd', 'valid', 'pred']], rank_lane()], ignore_index=True)
    for s, g in f.groupby('split'):
        chk = np.max(np.abs(compose(g, g.w.to_numpy()) - g.pred.to_numpy()))
        print(f'{s}: compose() reproduces the lane, max gap {chk:.2e}', flush=True)
        assert chk < 1e-6
    tz = pd.read_parquet(common.DATA / 'base_train.parquet', columns=['MVT_ID_mvt', 'hour_utc'])
    tz = pd.concat([tz, pd.read_parquet(common.DATA / 'base_rank.parquet', columns=['MVT_ID_mvt', 'hour_utc'])])
    f = f.merge(tz.rename(columns={'hour_utc': 'hour'}), on='MVT_ID_mvt', how='left').merge(signals(), on='MVT_ID_mvt', how='left')
    if TRAIN_TABLE is not None:   # final-file change: 2025 hour + signals exactly as in TRAIN_TABLE
        t25 = pd.read_parquet(TRAIN_TABLE)
        t25 = t25[t25.split == 'train'].reset_index(drop=True)
        n25 = int((f.split == 'train').sum())
        assert n25 == len(t25) and (f.MVT_ID_mvt.to_numpy()[:n25] == t25.MVT_ID_mvt.to_numpy()).all()
        f = pd.concat([t25[f.columns], f.iloc[n25:]], ignore_index=True)
    f.to_parquet(OUT / 'table.parquet', index=False)
    print(f.groupby('split').agg(rows=('MVT_ID_mvt', 'size'), valid=('valid', 'mean'), has_inbound=('has_inbound', 'mean'),
                                 arr_copy_6h=('arr_copy_6h', 'mean')).round(3).to_string(), flush=True)


# ------------------------------------------------------------------ the refitted gate
APTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']


def feats(f: pd.DataFrame, cols) -> pd.DataFrame:
    x = pd.DataFrame({'w': np.clip(f.w.to_numpy(), 0, 1), 'D_h': f.D.to_numpy() / 3600, 'U_h': f.U.to_numpy() / 3600,
                      'apt_code': f.APT.map({a: i for i, a in enumerate(APTS)}).to_numpy(), 'hour': f.hour.to_numpy()}, index=f.index)
    for c in SIGNALS:
        x[c] = f[c].to_numpy()
    return x[cols]


def target(f: pd.DataFrame):
    g = f.D.to_numpy() - f.U.to_numpy()
    z = np.clip((f.y.to_numpy() - f.U.to_numpy()) / g, -0.25, 1.25)
    return z, g ** 2


def fit(x, z, wt, seed=7):
    prm = dict(objective='regression', learning_rate=0.03, num_leaves=7, min_data_in_leaf=100, feature_fraction=0.9,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=seed, num_threads=16)
    return lgb.train(prm, lgb.Dataset(x, z, weight=wt / wt.mean(), categorical_feature=['apt_code']), num_boost_round=200)


def nested_arm(tr: pd.DataFrame, cols, name: str) -> dict:
    """tr = gate-valid unmatched 2025 rows. Returns the per-row new w (out of fold) and the report."""
    z, wt = target(tr)
    x = feats(tr, cols)
    folds = sorted(tr.fold.unique())
    w_new, chosen = tr.w.to_numpy().copy(), {}
    for k in folds:
        o = (tr.fold != k).to_numpy()
        sse = np.zeros(len(ALPHAS))
        for j in folds:
            if j == k:
                continue
            ii, io = o & (tr.fold != j).to_numpy(), (tr.fold == j).to_numpy()
            h = np.clip(fit(x[ii], z[ii], wt[ii]).predict(x[io]), 0, 1)
            sub = tr[io]
            for i, a in enumerate(ALPHAS):
                q = compose(sub, (1 - a) * sub.w.to_numpy() + a * h)
                sse[i] += ((q - sub.y.to_numpy()) ** 2).sum()
        a = float(ALPHAS[sse.argmin()])
        chosen[str(k)] = a
        h = (tr.fold == k).to_numpy()
        w_new[h] = (1 - a) * tr.w.to_numpy()[h] + a * np.clip(fit(x[o], z[o], wt[o]).predict(x[h]), 0, 1)
    return {'name': name, 'cols': list(cols), 'alpha_by_fold': chosen, 'w_new': w_new}


def on_v53(tr: pd.DataFrame, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """v53 2025 proxy before / after adding ``delta`` to its pre-rule value and re-applying W/F/R/B (scripts/263)."""
    b = importlib.import_module('263_v53_build')
    rows = pd.read_parquet(b.OUT / 'rows_2025.parquet')
    x = tr[['MVT_ID_mvt']].merge(rows, on='MVT_ID_mvt', how='left', validate='one_to_one')
    fr = tr[['MVT_ID_mvt']].merge(b.frame(), on='MVT_ID_mvt', how='left', validate='one_to_one')
    pre = x.v52_proxy.to_numpy() + x.opdi_corr.to_numpy()
    st = b.status(fr)
    assert np.allclose(b.apply_rules(fr, pre, st)[0], x.v53_proxy.to_numpy()), 'v53 proxy not reproduced'
    return x.v53_proxy.to_numpy(), b.apply_rules(fr, pre + delta, st)[0]


def nested():
    f = pd.read_parquet(OUT / 'table.parquet')
    tr = f[(f.split == 'train') & f.valid].reset_index(drop=True)
    N, R = 2085047, 232.5159                                     # rows and RMSE of the v53 2025 proxy
    sse0 = N * R ** 2
    base_pred = compose(tr, tr.w.to_numpy())
    y = tr.y.to_numpy()
    rep = {'rows_gate_valid_2025': int(len(tr)), 'arms': {}}
    keep = pd.DataFrame({'MVT_ID_mvt': tr.MVT_ID_mvt})
    for name, cols in (('base', BASE_FEATS), ('signals', BASE_FEATS + SIGNALS)):
        r = nested_arm(tr, cols, name)
        lane = compose(tr, r['w_new'])
        v53, q = on_v53(tr, lane - base_pred)
        de = (q - y) ** 2 - (v53 - y) ** 2
        keep[f'w_{name}'] = r['w_new']
        keep[f'delta_{name}'] = lane - base_pred
        arm = {'alpha_by_fold': r['alpha_by_fold'], 'dsse_M': round(float(de.sum() / 1e6), 2),
               'd_rmse_2025_vs_v53_proxy': float(np.sqrt((sse0 + de.sum()) / N) - R),
               'dsse_M_by_fold': {str(k): round(float(de[(tr.fold == k).to_numpy()].sum() / 1e6), 2) for k in sorted(tr.fold.unique())},
               'dsse_M_by_airport': {a: round(float(de[(tr.APT == a).to_numpy()].sum() / 1e6), 2) for a in APTS},
               'dsse_M_by_month': {str(m): round(float(de[(tr.month == m).to_numpy()].sum() / 1e6), 2) for m in sorted(tr.month.unique())},
               'rows_moved_100s': int((np.abs(q - base_pred) > 100).sum())}
        arm['folds_worse'] = sum(v > 0 for v in arm['dsse_M_by_fold'].values())
        arm['months_worse'] = sum(v > 0 for v in arm['dsse_M_by_month'].values())
        rep['arms'][name] = arm
        print(name, json.dumps({k: v for k, v in arm.items() if k != 'dsse_M_by_month'}), flush=True)
    rep['signals_minus_base_dsse_M'] = round(rep['arms']['signals']['dsse_M'] - rep['arms']['base']['dsse_M'], 2)
    keep.to_parquet(OUT / 'nested_oof.parquet', index=False)
    (OUT / 'nested.json').write_text(json.dumps(rep, indent=2))
    print('signals - base (M s^2):', rep['signals_minus_base_dsse_M'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['table', 'nested'])
    a = ap.parse_args()
    table() if a.mode == 'table' else nested()
