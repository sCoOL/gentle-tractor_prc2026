"""OPDI off-block correction: a per-airport residual model on the OPDI ground events (lead #3 of LEADS_20261006).

  py -3.14 scripts/262_opdi_correction.py frame                 # runs/v53_opdi/frame.parquet (both years, label-free inputs)
  py -3.14 scripts/262_opdi_correction.py nested --apt LSZH     # 2025 nested evidence -> runs/v53_opdi/nested_<APT>.json + oof
  py -3.14 scripts/262_opdi_correction.py ablate --apt LSZH     # same, without the OPDI columns / OPDI columns only
  py -3.14 scripts/262_opdi_correction.py fit --apt LSZH        # full-2025 fit (3 seeds) -> runs/v53_opdi/rank_<APT>.parquet

Base prediction p: on 2025 rows the v52 proxy (A51 strict OOF + v51's W/F + v52's R; runs/v52_recovered_lobt/rows.parquet,
233.4620 on the 2,085,047 rows); on 2026 rows the v52 file. Target y - p. Inputs (all label-free, scripts/260):
  d_<event> = (recorded take-off - OPDI event time) - p   for the first taxiway entry, first apron exit, the parking exit
              at the recorded stand, the first parking exit, the first ground event and the last runway entry
  n_gnd, n_tw (OPDI ground/taxiway event counts), adsb.lol state (none/censored/seen/pushback), NM lane, p.
A row is in scope when it has at least one OPDI ground event before the runway (OPD_EVENTS) and is not a helicopter.
LightGBM, Huber loss (alpha 300 s), fixed hyper-parameters. Nested: for each outer fold the model is trained on the other
five folds; its shrink factor (0 / .25 / .5 / .75 / 1) is chosen by an inner leave-one-fold-out over those five.
No label, block time or official score enters a feature; nothing is fitted on 2026.
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import final_rank  # noqa: E402  final-file change: rank-population size (was a hard-coded 344,841)

OUT = common.RUNS / 'v53_opdi'
OPDI = common.prc_config.OPDI_DIR / 'opdi_movements.parquet'     # 260 output
ADSB_V1 = common.prc_config.ADSB_DIR / 'adsb_features.parquet'   # 121 features (v1 match)
V52 = common.SUBS / 'gentle-tractor_v52.parquet'
LEDGER = common.RUNS / 'v52_recovered_lobt' / 'rows.parquet'
OOF = common.RUNS / 'a51_final' / 'chosen_ensemble_oof.parquet'
N_2025, RMSE_V52_PROXY = 2085047, 233.4620   # rows and RMSE of the v52 2025 proxy
EVENTS = ['tw_first_t', 'xa_first_t', 'xs_t', 'xp_first_t', 'g_first_t', 'er_t']
OPD_EVENTS = ['tw_first_t', 'xa_first_t', 'xs_t', 'xp_first_t', 'g_first_t']
STATES = {'none': 0, 'censored': 1, 'seen': 2, 'pushback': 3}
OPDI_COLS = ['d_' + e for e in EVENTS] + ['n_gnd', 'n_tw']
FEATS = OPDI_COLS + ['st', 'fpl_matched', 'p']
ALPHAS = np.array([0, .25, .5, .75, 1.0])
SEEDS = (7, 11, 23)


def _epoch(s):
    return (pd.to_datetime(s, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')) / pd.Timedelta(seconds=1)


def frame() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'TAXITIME_SEC_mvt', 'fpl_matched', 'is_heli', 'month']
    b = pd.concat([pd.read_parquet(common.DATA / f, columns=cols).assign(split=s)
                   for f, s in [('base_train.parquet', 'train'), ('base_rank.parquet', 'rank')]], ignore_index=True)
    fa = pd.read_parquet(ADSB_V1, columns=['MVT_ID_mvt', 'adsb_at_stand', 'adsb_censored', 'adsb_match_gap'])
    fa = fa.sort_values('adsb_match_gap', na_position='last').drop_duplicates('MVT_ID_mvt')
    mv = pd.read_parquet(OPDI, columns=['MVT_ID_mvt', 'role', 'n_gnd', 'n_tw'] + EVENTS)
    mv = mv[mv.role == 'dep'].drop(columns='role')
    assert mv.MVT_ID_mvt.is_unique
    d = b.merge(fa, on='MVT_ID_mvt', how='left').merge(mv, on='MVT_ID_mvt', how='left')
    d['tko'] = _epoch(d.MVT_TIME_UTC_mvt)
    for e in EVENTS:
        d[e] = _epoch(d[e])
    d['adsb'] = np.select([d.adsb_at_stand.isna(), d.adsb_at_stand == 1, d.adsb_censored == 1], ['none', 'pushback', 'censored'], 'seen')
    d['y'] = d.TAXITIME_SEC_mvt.astype(float)
    oof = pd.read_parquet(OOF, columns=['MVT_ID_mvt', 'fold'])
    led = pd.read_parquet(LEDGER, columns=['MVT_ID_mvt', 'split', 'v52_pred_s'])
    led = led[led.split == 'train'].drop(columns='split')
    s52 = pd.read_parquet(V52).rename(columns={'TAXITIME_SEC_mvt': 'sub52'})
    d = d.merge(oof, on='MVT_ID_mvt', how='left').merge(led, on='MVT_ID_mvt', how='left').merge(s52, on='MVT_ID_mvt', how='left')
    d['p'] = np.where(d.split == 'train', d.v52_pred_s, d.sub52.astype(float))
    assert d.loc[d.split == 'train', ['p', 'fold']].notna().all().all() and len(d[d.split == 'train']) == N_2025
    # final-file change: rank rows = the ids of the V52 file (the whole final file; rows do not interact here), and the
    # count is the template's (was a fixed 344,841)
    d = d[(d.split == 'train') | d.MVT_ID_mvt.isin(s52.MVT_ID_mvt)].reset_index(drop=True)
    assert d.loc[d.split == 'rank', 'p'].notna().all() and (d.split == 'rank').sum() == final_rank.n_rank()
    d['scope'] = d[OPD_EVENTS].notna().any(axis=1) & (d.is_heli != 1)
    keep = ['MVT_ID_mvt', 'APT', 'split', 'month', 'fold', 'fpl_matched', 'adsb', 'scope', 'y', 'p', 'tko', 'n_gnd', 'n_tw'] + EVENTS
    d = d[keep]
    d.to_parquet(OUT / 'frame.parquet', index=False)
    print(pd.crosstab([d.split, d.APT], d.scope).to_string(), flush=True)
    return d


def features(d: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame({'d_' + e: (d.tko - d[e]) - d.p for e in EVENTS}, index=d.index)
    x['n_gnd'], x['n_tw'] = d.n_gnd, d.n_tw
    x['st'] = d.adsb.map(STATES).astype(int)
    x['fpl_matched'], x['p'] = d.fpl_matched.astype(float), d.p
    return x


def fit(x: pd.DataFrame, r: np.ndarray, cols, seed: int):
    prm = dict(objective='huber', alpha=300.0, learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=seed, num_threads=16)
    cats = ['st'] if 'st' in cols else []
    return lgb.train(prm, lgb.Dataset(x[cols], r, categorical_feature=cats), num_boost_round=300)


def load(apt: str) -> pd.DataFrame:
    d = pd.read_parquet(OUT / 'frame.parquet')
    return d[(d.APT == apt) & d.scope].reset_index(drop=True)


def nested(apt: str, cols=FEATS, tag: str = '') -> dict:
    d = load(apt)
    tr = d[d.split == 'train'].reset_index(drop=True)
    x, r = features(tr), (tr.y - tr.p).to_numpy()
    folds = sorted(tr.fold.unique())
    corr, chosen = np.zeros(len(tr)), {}
    for k in folds:
        o = (tr.fold != k).to_numpy()
        sse = np.zeros(len(ALPHAS))
        for j in folds:
            if j == k:
                continue
            ii, io = o & (tr.fold != j).to_numpy(), (tr.fold == j).to_numpy()
            h = fit(x[ii], r[ii], cols, SEEDS[0]).predict(x.loc[io, cols])
            sse += np.array([((r[io] - a * h) ** 2).sum() for a in ALPHAS])
        a = float(ALPHAS[sse.argmin()])
        chosen[str(k)] = a
        h = (tr.fold == k).to_numpy()
        corr[h] = a * fit(x[o], r[o], cols, SEEDS[0]).predict(x.loc[h, cols])
    de = (r - corr) ** 2 - r ** 2
    sse0 = N_2025 * RMSE_V52_PROXY ** 2

    def drmse(mask):
        return float(np.sqrt((sse0 + de[mask].sum()) / N_2025) - RMSE_V52_PROXY)
    rep = {'apt': apt, 'tag': tag, 'features': list(cols), 'rows_in_scope_2025': int(len(tr)),
           'rows_in_scope_2026': int((d.split == 'rank').sum()), 'alpha_by_fold': chosen,
           'd_rmse_2025': drmse(np.ones(len(tr), bool)),
           'dsse_M_by_fold': {str(k): round(float(de[(tr.fold == k).to_numpy()].sum() / 1e6), 2) for k in folds},
           'dsse_M_by_month': {str(m): round(float(de[(tr.month == m).to_numpy()].sum() / 1e6), 2) for m in sorted(tr.month.unique())},
           'dsse_M_by_state': {s: round(float(de[(tr.adsb == s).to_numpy()].sum() / 1e6), 2) for s in STATES},
           'dsse_M_by_lane': {str(int(m)): round(float(de[(tr.fpl_matched == m).to_numpy()].sum() / 1e6), 2) for m in (0, 1)},
           'rows_2025_by_state': tr.adsb.value_counts().to_dict(),
           'rows_2026_by_state': d[d.split == 'rank'].adsb.value_counts().to_dict(),
           'rows_better_100s': int((np.abs(r) - np.abs(r - corr) > 100).sum()),
           'rows_worse_100s': int((np.abs(r - corr) - np.abs(r) > 100).sum()),
           'rmse_in_scope_before_after': [float(np.sqrt(np.mean(r ** 2))), float(np.sqrt(np.mean((r - corr) ** 2)))]}
    rep['folds_worse'] = sum(v > 0 for v in rep['dsse_M_by_fold'].values())
    rep['months_worse'] = sum(v > 0 for v in rep['dsse_M_by_month'].values())
    name = f'nested_{apt}{tag}'
    (OUT / f'{name}.json').write_text(json.dumps(rep, indent=2))
    if not tag:
        pd.DataFrame({'MVT_ID_mvt': tr.MVT_ID_mvt, 'corr': corr}).to_parquet(OUT / f'oof_{apt}.parquet', index=False)
    print(json.dumps(rep), flush=True)
    return rep


def fit_rank(apt: str, alpha: float = 1.0) -> pd.DataFrame:
    d = load(apt)
    tr, rk = d[d.split == 'train'], d[d.split == 'rank']
    x, r = features(tr), (tr.y - tr.p).to_numpy()
    xr = features(rk)
    h = np.mean([fit(x, r, FEATS, s).predict(xr[FEATS]) for s in SEEDS], axis=0)
    out = pd.DataFrame({'MVT_ID_mvt': rk.MVT_ID_mvt.to_numpy(), 'APT': apt, 'adsb': rk.adsb.to_numpy(), 'p': rk.p.to_numpy(),
                        'corr': alpha * h})
    out.to_parquet(OUT / f'rank_{apt}.parquet', index=False)
    print(apt, 'rank rows', len(out), 'corr mean/rms', round(float(out['corr'].mean()), 1),
          round(float(np.sqrt((out['corr'] ** 2).mean())), 1), out.groupby('adsb')['corr'].agg(['size', 'mean']).round(1).to_dict(), flush=True)
    return out


# ------------------------------------------------------------------ v54: stand-level events only (EDDF, added 2026-10-06)
# EDDF's first taxiway entry changed meaning between 2025 and 2026 (404 s vs 69 s after the adsb.lol push-back), but its
# stand exit / apron exit did not (2025: 49 / 24 s, 2026: 13 / 27 s) and their coverage starts in Sep 2025 (3.9-7.8 % of
# rows, 2026: 7.5-10.5 %). Scope = rows with a stand or apron exit; inputs = the three stand-level events, adsb.lol state,
# lane, prediction and an EDDF flag; trained on LSZH + EDDF stand-event rows together, evaluated / applied on EDDF only.
STAND_COLS = ['d_xs_t', 'd_xa_first_t', 'd_xp_first_t', 'st', 'fpl_matched', 'p', 'is_target']
STAND_POOL = ('LSZH',)


def stand_frame(apt: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_parquet(OUT / 'frame.parquet')
    d = d[d.APT.isin((apt,) + STAND_POOL) & (d.xs_t.notna() | d.xa_first_t.notna()) & d.scope].reset_index(drop=True)
    x = features(d)
    x['is_target'] = (d.APT == apt).astype(float).to_numpy()
    return d, x


def stand_nested(apt: str) -> dict:
    d, x = stand_frame(apt)
    tr = d[d.split == 'train']
    xt, r = x.loc[tr.index], (tr.y - tr.p).to_numpy()
    tg = (tr.APT == apt).to_numpy()
    folds = sorted(tr.fold.unique())
    corr, chosen = np.zeros(len(tr)), {}
    for k in folds:
        o = (tr.fold != k).to_numpy()
        sse = np.zeros(len(ALPHAS))
        for j in folds:
            if j == k:
                continue
            ii, io = o & (tr.fold != j).to_numpy(), (tr.fold == j).to_numpy() & tg
            if io.sum() == 0:
                continue
            h = fit(xt[ii], r[ii], STAND_COLS, SEEDS[0]).predict(xt.loc[io, STAND_COLS])
            sse += np.array([((r[io] - a * h) ** 2).sum() for a in ALPHAS])
        a = float(ALPHAS[sse.argmin()])
        chosen[str(k)] = a
        h = (tr.fold == k).to_numpy() & tg
        if h.any():
            corr[h] = a * fit(xt[o], r[o], STAND_COLS, SEEDS[0]).predict(xt.loc[h, STAND_COLS])
    de = ((r - corr) ** 2 - r ** 2)[tg]
    t = tr[tg]
    sse0 = N_2025 * RMSE_V52_PROXY ** 2
    rep = {'apt': apt, 'pool': list(STAND_POOL), 'features': STAND_COLS, 'rows_2025': int(tg.sum()),
           'rows_2026': int(((d.split == 'rank') & (d.APT == apt)).sum()), 'alpha_by_fold': chosen,
           'd_rmse_2025': float(np.sqrt((sse0 + de.sum()) / N_2025) - RMSE_V52_PROXY),
           'rows_by_fold': t.fold.value_counts().to_dict(),
           'dsse_M_by_fold': {str(k): round(float(de[(t.fold == k).to_numpy()].sum() / 1e6), 2) for k in folds},
           'dsse_M_by_state': {s: round(float(de[(t.adsb == s).to_numpy()].sum() / 1e6), 2) for s in STATES},
           'rows_2025_by_state': t.adsb.value_counts().to_dict(),
           'rows_2026_by_state': d[(d.split == 'rank') & (d.APT == apt)].adsb.value_counts().to_dict()}
    (OUT / f'nested_{apt}_stand.json').write_text(json.dumps(rep, indent=2))
    pd.DataFrame({'MVT_ID_mvt': t.MVT_ID_mvt, 'corr': corr[tg]}).to_parquet(OUT / f'oof_{apt}_stand.parquet', index=False)
    print(json.dumps(rep), flush=True)
    return rep


def stand_fit(apt: str, alpha: float = 1.0) -> pd.DataFrame:
    d, x = stand_frame(apt)
    tr = d[d.split == 'train']
    rk = d[(d.split == 'rank') & (d.APT == apt)]
    r = (tr.y - tr.p).to_numpy()
    h = np.mean([fit(x.loc[tr.index], r, STAND_COLS, s).predict(x.loc[rk.index, STAND_COLS]) for s in SEEDS], axis=0)
    out = pd.DataFrame({'MVT_ID_mvt': rk.MVT_ID_mvt.to_numpy(), 'APT': apt, 'adsb': rk.adsb.to_numpy(), 'p': rk.p.to_numpy(),
                        'corr': alpha * h})
    out.to_parquet(OUT / f'rank_{apt}_stand.parquet', index=False)
    print(apt, 'stand rank rows', len(out), 'corr mean/rms', round(float(out['corr'].mean()), 1),
          round(float(np.sqrt((out['corr'] ** 2).mean())), 1), out.groupby('adsb')['corr'].agg(['size', 'mean']).round(1).to_dict(), flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['frame', 'nested', 'ablate', 'fit', 'stand_nested', 'stand_fit'])
    ap.add_argument('--apt', default='LSZH')
    ap.add_argument('--alpha', type=float, default=1.0)
    a = ap.parse_args()
    if a.mode == 'frame':
        frame()
    elif a.mode == 'nested':
        for apt in a.apt.split(','):
            nested(apt)
    elif a.mode == 'ablate':
        for apt in a.apt.split(','):
            nested(apt, [c for c in FEATS if c not in OPDI_COLS], '_no_opdi')
            nested(apt, FEATS, '')
    elif a.mode == 'stand_nested':
        for apt in a.apt.split(','):
            stand_nested(apt)
    elif a.mode == 'stand_fit':
        for apt in a.apt.split(','):
            stand_fit(apt, a.alpha)
    else:
        for apt in a.apt.split(','):
            fit_rank(apt, a.alpha)
