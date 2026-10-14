"""Re-predict every v45 input for the 670,790 final rows from SAVED models only (no refit), for variant A or B.

  py -3.14 tools/members/rescore.py --variant B [--check]   # -> runs/models/final_B/...
  py -3.14 tools/members/rescore.py --variant A [--check]   # -> runs/models/final_A/...

Variant A and B share every non-ADS-B input; they differ only in the 26 ADS-B members (16 LightGBM, 7 set networks,
3 AB twins): B = trained on the rebuilt (= reference current) 2025 ADS-B table, A = trained on the 09-26 snapshot's 2025 rows.
Both score the final rows with the ADS-B tables built from the final ranking file (PRC_ADSB_DIR).

Saved models used:
  M3rpa (reference A27 boosters), A27 gate (reference gate_lgb.txt), context detector / rd members (runs/models/R_A27/models),
  LightGBM members + twins (runs/models/member_models, variant A ADS-B ones in runs/models/variant_A/member_models),
  networks (runs/models/member_models/nn/*.pt, variant A ADS-B ones in runs/models/variant_A/member_models/nn).
The surge context table of the ranking file is rebuilt from the current data first (it is a cache of data/).
--check compares every output with the training-run prediction of the same member (exact when data/ is unchanged).
"""
from __future__ import annotations

import mcommon as mc

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import probe106_ext
sys.modules['106_member_probe'] = probe106_ext

from mcommon import REFM, OUT, DATA, say, compare, fmt, save_report

ADSB = mc.prc_config.ADSB_DIR
A27 = REFM / 'A27_20260917_141019'
STATE = json.loads((A27 / 'M3rpa' / 'state.json').read_text(encoding='utf-8'))
ADSB_LGB = ['lgb_ab', 'lgb_ab_s1', 'lgb_sg_ab', 'lgb_sg_ab_s1', 'lgb_ab_apt', 'lgb_sg_ab_apt', 'lgb_q12', 'lgb_sg_q12',
            'lgb_ab_trust', 'lgb_sg_ab_trust', 'lgb_ab_st', 'lgb_ab_st_s1', 'lgb_ab_st_s2', 'lgb_sg_ab_st', 'lgb_sg_ab_st_s1',
            'lgb_sg_ab_st_s2']
ADSB_NN = ['mlp_setn_ab'] + [f'mlp_setn_ab_s{i}' for i in range(1, 7)]
A48_NN = ['mlp', 'mlp_s1', 'mlp_s2', 'mlp_e20', 'mlp_wide', 'mlp_set', 'mlp_set2', 'mlp_set_s1', 'mlp_setn', 'mlp_setn2',
          'mlp_setn_s1', 'mlp_setn_r', 'mlp_setn2_r', 'mlp_setn_s1_r']
A35_NORMAL = ['nm_masked_set', 'nm_masked_self', 'nm_masked_set2']
A35_CLF = ['set_clf', 'self_clf', 'set_clf2']
A37 = ['u_self', 'u_self_s1', 'u_self_s2']
CHECKS = {}


def check(label, ids, vals, ref_path, col):
    if not ARGS.check or not Path(ref_path).exists():
        return
    r = pd.read_parquet(ref_path)
    c = compare(ids, vals, r.MVT_ID_mvt, r[col], label)
    CHECKS[label] = c; say(fmt(c))


def write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def a27(rank, out):
    import a27_members as am
    import detectors
    import fallback_context as fc
    import residual_variants as rv
    import train_eval
    d = out / 'R_A27'
    t0 = time.time()
    normal = am.m3rpa_predict(rank)
    write(pd.DataFrame({'MVT_ID_mvt': rank.MVT_ID_mvt.values, 'normal_pred': normal}), d / 'normal_rank.parquet')
    check('A27 normal_rank', rank.MVT_ID_mvt, normal, OUT / 'R_A27' / 'normal_rank.parquet', 'normal_pred')
    det = lgb.Booster(model_file=str(OUT / 'R_A27' / 'models' / 'context_detector.txt'))
    p = det.predict(detectors._prep(rank, fc.FEATS, fc.CATS))
    write(pd.DataFrame({'MVT_ID_mvt': rank.MVT_ID_mvt.values, 'p_fallback': p}), d / 'p_fallback_rank.parquet')
    check('A27 p_fallback', rank.MVT_ID_mvt, p, OUT / 'R_A27' / 'p_fallback_rank.parquet', 'p_fallback')
    cols, cats = STATE['cols'], STATE['cats']
    a3 = rank.atot_minus_aobt3.values.astype(float); has = np.isfinite(a3)
    rg1 = np.full(len(rank), np.nan); ra = np.full(len(rank), np.nan)
    m = lgb.Booster(model_file=str(OUT / 'R_A27' / 'models' / 'r_global_seed1.txt'))
    rg1[has] = m.predict(train_eval.prep(rank[has], cols, cats))
    for apt in rank.APT.unique():
        b = (rank.APT.values == apt) & has
        if b.any():
            ra[b] = lgb.Booster(model_file=str(OUT / 'R_A27' / 'models' / f'r_airport_{apt}.txt')).predict(train_eval.prep(rank[b], cols, cats))
    res = pd.DataFrame({'MVT_ID_mvt': rank.MVT_ID_mvt.values, 'aobt3_anchor': a3, 'r_global': normal - a3,
                        'r_global_seed1': rg1, 'r_airport': ra})
    write(res, d / 'residual_members_rank.parquet')
    for c in ('r_global_seed1', 'r_airport'):
        check(f'A27 {c}', res.MVT_ID_mvt, res[c], OUT / 'R_A27' / 'residual_members_rank.parquet', c)
    rd = rv.blend_members(normal, a3, rg1, ra, rank.is_heli.values)
    write(pd.DataFrame({'MVT_ID_mvt': rank.MVT_ID_mvt.values, 'normal_pred': rd}), d / 'normal_rank_rd.parquet')
    detail = am.gate_detail(False, src=d, write=True)
    check('A27 gate w', detail.MVT_ID_mvt, detail.w, OUT / 'R_A27' / 'rank_detail.parquet', 'w')
    say(f'A27 block re-predicted [{time.time() - t0:.0f}s]')


def lgb_members(rank, out, variant):
    import lgb_members as lm
    import train_eval
    p = probe106_ext
    cols, cats = STATE['cols'], STATE['cats']
    for name in ['lgb_sg'] + ADSB_LGB:
        t0 = time.time()
        mdir = (OUT / 'variant_A' / 'member_models' if (variant == 'A' and name in ADSB_LGB) else OUT / 'member_models') / name
        kind0 = p.CANDIDATES[name][0]
        blk, kind = lm.transform(kind0, {'train': rank.iloc[:0], 'held': rank, 'cols': cols, 'cats': cats, 'block': 'deploy'},
                                 held_only=True)
        h, cc = blk['held'], blk['cols']
        r = np.full(len(h), np.nan)
        if kind == 'lgb':
            has = np.isfinite(h.atot_minus_aobt3.values.astype(float))
            r[has] = lgb.Booster(model_file=str(mdir / 'model.txt')).predict(train_eval.prep(h[has], cc, cats))
        else:
            for f in sorted(mdir.glob('model_*.txt')):
                apt = f.stem.split('_', 1)[1]
                b = (h.APT.values == apt) & np.isfinite(h.atot_minus_aobt3.values.astype(float))
                if b.any():
                    r[b] = lgb.Booster(model_file=str(f)).predict(train_eval.prep(h[b], cc, cats))
        write(pd.DataFrame({'MVT_ID_mvt': h.MVT_ID_mvt.values, f'r_{name}': r}), out / 'member_cache' / f'member_rank_{name}.parquet')
        ref = (OUT / 'variant_A' if (variant == 'A' and name in ADSB_LGB) else OUT) / 'member_cache' / f'member_rank_{name}.parquet'
        check(f'{name}', h.MVT_ID_mvt, r, ref, f'r_{name}')
        say(f'{name} re-predicted from {mdir} [{time.time() - t0:.0f}s]')


def twins(rank, out, variant, seeds=(41, 42, 43)):
    """Unmatched twins from their saved boosters. `seeds` beyond 41-43 (v73: --twin-seeds 41 ... 55) are written
    to unmatched_adsb_rank_<seeds>.parquet next to the 3-seed file; only 41-43 have reference caches to check."""
    import common
    import train_eval
    m124 = importlib.import_module('124_unmatched_adsb')
    un = importlib.import_module('110_unmatched_normal')
    a35 = importlib.import_module('111_bundle_a35')
    cols, cats = a35._strict_cols(common.read_manifest(5))
    drop = set(un.NM_NAN) | set(un.NM_CONST)
    cols = [c for c in cols if c not in drop]; cats = [c for c in cats if c not in drop]
    held = un.mask_nm(rank[rank.fpl_matched.values == 0])
    res = pd.DataFrame({'MVT_ID_mvt': held.MVT_ID_mvt.values})
    Xb = train_eval.prep(held, cols, cats)
    for s in seeds:
        res[f'blind_s{s}'] = lgb.Booster(model_file=str(OUT / 'member_models' / 'unmatched_twins' / f'blind_s{s}.txt')).predict(Xb)
    ht = m124.with_trace(held)
    Xa = train_eval.prep(ht, cols + m124.AB_COLS, cats)
    adir = (OUT / 'variant_A' if variant == 'A' else OUT) / 'member_models' / 'unmatched_twins'
    for s in seeds:
        res[f'ab_s{s}'] = lgb.Booster(model_file=str(adir / f'ab_s{s}.txt')).predict(Xa)
    write(res, out / 'member_cache_a46' / f'unmatched_adsb_rank_{"_".join(map(str, seeds))}.parquet')
    ref = (OUT / 'variant_A' if variant == 'A' else OUT) / 'member_cache_a46' / 'unmatched_twins_ab_41_42_43.parquet'
    for s in [s for s in seeds if s in (41, 42, 43)]:
        check(f'twin blind_s{s}', res.MVT_ID_mvt, res[f'blind_s{s}'], OUT / 'member_cache_a46' / 'unmatched_twins_blind_41_42_43.parquet', f'blind_s{s}')
        check(f'twin ab_s{s}', res.MVT_ID_mvt, res[f'ab_s{s}'], ref, f'ab_s{s}')


_TOK = {}


LEAD_IN_MIN = 3600.0     # continuity rule: 1 Jan 2026 tokens see December 2025 (token window -60/+20 min)


def tokens(ids, ctx):
    """Ranking-file neighbour tokens with a December lead-in of at least the token window (surge contexts already
    carry 3 h). Only departures in the first hour of 1 Jan 2026 can see a lead-in neighbour."""
    import nbr_set
    ctx = dict(ctx or {})
    ctx['lead_in'] = max(float(ctx.get('lead_in', 0.0)), LEAD_IN_MIN)
    key = tuple(sorted(ctx.items()))
    if key not in _TOK:
        _TOK.clear()
        _TOK[key] = nbr_set.build('rank', **ctx)
    return nbr_set.aligned(*_TOK[key], ids)


def networks(rank, out, variant):
    import adsb_context
    un = importlib.import_module('110_unmatched_normal')
    flag = lambda df: df.assign(is_unmatched=(df.fpl_matched.values == 0).astype('float32'))
    frames = {'plain': rank, 'adsb': rank.merge(adsb_context.table(), on='MVT_ID_mvt', how='left', validate='one_to_one'),
              'masked': un.mask_nm(rank), 'masked_flag': un.mask_nm(flag(rank))}
    plan = ([(n, 'plain', 'member_cache', f'member_rank_{n}.parquet', f'r_{n}') for n in A48_NN] +
            [(n, 'adsb', 'member_cache', f'member_rank_{n}.parquet', f'r_{n}') for n in ADSB_NN] +
            [(n, 'masked', 'member_cache_a35', f'{n}.parquet', 'v') for n in A35_NORMAL] +
            [(n, 'plain', 'member_cache_a35', f'{n}.parquet', 'v') for n in A35_CLF] +
            [(n, 'masked_flag', 'member_cache_a37', f'{n}.parquet', 'p') for n in A37])
    # group by token context to build each token set once
    def ctx_of(n):
        return json.dumps(probe106_ext.CANDIDATES.get(n, (None, {}))[1].get('ctx') if n in probe106_ext.CANDIDATES else None, sort_keys=True)
    plan.sort(key=lambda x: ctx_of(x[0]))
    for n, fr, folder, fname, col in plan:
        t0 = time.time()
        base = OUT / 'variant_A' if (variant == 'A' and n in ADSB_NN) else OUT
        path = base / 'member_models' / 'nn' / f'{n}.pt'
        frame = frames[fr]
        r = probe106_ext.predict_saved(path, frame, tokens)
        write(pd.DataFrame({'MVT_ID_mvt': frame.MVT_ID_mvt.values, col: r}), out / folder / fname)
        check(f'{n}', frame.MVT_ID_mvt, r, base / folder / fname, col)
        say(f'{n} re-predicted from {path} [{time.time() - t0:.0f}s]')


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['A', 'B'], required=True)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--parts', nargs='+', default=['surge', 'a27', 'lgb', 'twins', 'nn'])
    ap.add_argument('--out')
    ap.add_argument('--twin-seeds', nargs='+', type=int, default=[41, 42, 43], help='twins part: booster seeds (v73: 41 ... 55)')
    ARGS = ap.parse_args()
    out = Path(ARGS.out) if ARGS.out else OUT / f'final_{ARGS.variant}'
    mc.point_adsb(ADSB)
    t0 = time.time()
    if 'surge' in ARGS.parts:                      # the ranking-file surge cache must follow data/
        f = mc.prc_config.RUNS / 'tuning' / 'surge_context_rank.parquet'
        if f.exists():
            f.rename(f.with_name(f'surge_context_rank.before_{time.strftime("%H%M%S")}.parquet'))
        probe106_ext.surge_table('rank')
        say('surge_context_rank rebuilt from data/')
    rank = mc.load_rank(5)
    if 'a27' in ARGS.parts:
        a27(rank, out)
    if 'lgb' in ARGS.parts:
        lgb_members(rank, out, ARGS.variant)
    if 'twins' in ARGS.parts:
        twins(rank, out, ARGS.variant, ARGS.twin_seeds)
    if 'nn' in ARGS.parts:
        networks(rank, out, ARGS.variant)
    save_report(f'rescore_{ARGS.variant}_{time.strftime("%H%M")}', {'variant': ARGS.variant, 'out': str(out), 'checks': CHECKS,
                                                                   'seconds': round(time.time() - t0, 1)})
    say(f'rescore {ARGS.variant} done -> {out} [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
