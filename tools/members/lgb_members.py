"""LightGBM members of the A48 blend and the unmatched twins, refitted once and scored on NEW and OLD rank rows.

  py -3.14 tools/members/lgb_members.py surge                 # surge_context tables: train / rank (final) / rank_old
  py -3.14 tools/members/lgb_members.py member lgb_sg [...]   # 107_extra_members.fit_rank_member, any LightGBM kind
  py -3.14 tools/members/lgb_members.py twins --part blind     # 124_unmatched_adsb.rank_twins, BLIND half (no ADS-B)
  py -3.14 tools/members/lgb_members.py twins --part ab --adsb <folder>   # AB half (needs the v1 ADS-B tables)

The fit paths are 107.fit_rank_member / 106.fit_lightgbm_per_airport / 124.fit_twins verbatim (same columns, params,
rounds, seeds, training rows); the only addition is that the booster is kept and also scores the reference old rank rows
(features data_old/, surge context from the old export, G12 from data_old/features_v6_rank) for the reproduction check.
NEW -> runs/models/member_cache/member_rank_<name>.parquet (reference format); OLD -> runs/members_repro/old_scored/...;
boosters -> runs/models/member_models/<name>/.
"""
from __future__ import annotations

import mcommon as mc

import argparse
import contextlib
import importlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from mcommon import REFM, OUT, REPRO, OLD, DATA, say, compare, fmt, save_report

STATE = json.loads((REFM / 'A27_20260917_141019' / 'M3rpa' / 'state.json').read_text(encoding='utf-8'))
CACHE_NEW = OUT / 'member_cache'
CACHE_OLD = REPRO / 'old_scored' / 'member_cache'
MODELS = OUT / 'member_models'
SURGE_OLD = mc.prc_config.RUNS / 'tuning' / 'surge_context_rank_old.parquet'
VARIANT = [None]
TRAIN_ADSB = [None, None]      # (training-side folder, scoring-side folder) for variant A


def probe():
    return importlib.import_module('106_member_probe')


def cmd_surge(a):
    import surge_context
    p = probe()
    t0 = time.time()
    tr = p.surge_table('train'); rk = p.surge_table('rank')
    say(f'surge train {tr.shape} rank {rk.shape} [{time.time() - t0:.0f}s]')
    if not SURGE_OLD.exists():
        with mc.old_rank_source():
            surge_context.build('rank', lead_in=p.SURGE_LEAD_IN).to_parquet(SURGE_OLD, index=False)
    old = pd.read_parquet(SURGE_OLD)
    rep = {'rows': {'train': len(tr), 'rank': len(rk), 'rank_old': len(old)}}
    main_tr = mc.REF / 'runs' / 'tuning' / 'surge_context_train.parquet'   # optional reference caches (skipped when absent)
    main_rk = mc.REF / 'runs' / 'tuning' / 'surge_context_rank.parquet'
    for label, mine, ref in (('train vs reference cache', tr, main_tr), ('rank_old vs reference cache', old, main_rk), ('rank (final) vs reference cache', rk, main_rk)):
        if not ref.exists():
            continue
        r = pd.read_parquet(ref)
        cols = [c for c in r.columns if c != 'MVT_ID_mvt']
        out = []
        for c in cols:
            x = compare(mine.MVT_ID_mvt, mine[c], r.MVT_ID_mvt, r[c], c)
            out.append(x)
        worst = max(out, key=lambda x: x.get('max_abs', 0))
        rep[label] = {'n_ref': len(r), 'n_mine': len(mine), 'columns_exact': sum(x.get('max_abs', 1) == 0 and x['nan_mismatch'] == 0 for x in out),
                      'columns': len(out), 'worst': worst}
        say(label, rep[label]['columns_exact'], '/', len(out), 'exact; worst', fmt(worst))
    save_report('step5_surge_tables', rep)


@contextlib.contextmanager
def old_tables():
    """106's 2026-side tables swapped for the reference old export while the OLD held frame is transformed."""
    p = probe()
    st, qt = p.surge_table, p.queue_table

    def surge_table(which):
        return pd.read_parquet(SURGE_OLD) if which == 'rank' else st(which)

    def queue_table(which):
        if which == 'rank':
            return pd.read_parquet(OLD / 'features_v6_rank.parquet', columns=['MVT_ID_mvt'] + p.queue_columns())
        return qt(which)
    p.surge_table, p.queue_table = surge_table, queue_table
    try:
        yield
    finally:
        p.surge_table, p.queue_table = st, qt


def deploy_rows():
    """107._deploy_rows(5): clean (fallback 0, non-heli) 2025 rows without date bugs, with an AOBT_3."""
    import tail_experts as te
    import train_eval
    train = pd.read_parquet(DATA / 'features_v5_train.parquet')
    bug = te.training_exclusion(train.TAXITIME_SEC_mvt.values, train.atot_minus_sched.values, 'datebug')
    clean = train_eval._clean(train[~bug])
    return clean[np.isfinite(clean.atot_minus_aobt3.values)]


def transform(kind, blk, held_only=False):
    """107.fit_rank_member's block transformations; returns (blk, final kind). held_only skips the training filters."""
    p = probe()
    if kind.endswith('_plus') or kind == 'lgb_plus_apt':
        blk = p.with_surge(blk)
        kind = {'lgb_plus': 'lgb', 'mlp_plus': 'mlp', 'mlp_set_plus': 'mlp_set'}.get(kind, kind)
    if kind.startswith(('lgb_adsb', 'lgb_sg_adsb', 'mlp_set_adsb')):
        blk = p.with_adsb(p.with_surge(blk) if kind.startswith('lgb_sg_adsb') else blk)
        if kind.endswith('_nbr'):
            blk = p.with_nbr(blk)
        if kind.endswith('_geom'):
            blk = p.with_geom(blk)
        if kind.endswith('_st') or kind.endswith('_st_ty'):
            blk = p.with_stops(blk)
        if kind.endswith('_ty'):
            blk = p.with_stand_type(blk)
        if '_queue' in kind:
            blk = p.with_queue(blk)
        if kind.endswith('_debias'):
            blk = p.with_adsb_debias(blk) if held_only else p.only_trust(p.with_adsb_debias(blk))
        if kind.endswith('_trust') and not held_only:
            blk = p.only_trust(blk)
        if kind.endswith('_cens') and not held_only:
            blk = p.only_cens(blk)
        kind = 'mlp_set' if kind.startswith('mlp_set') else ('lgb_plus_apt' if kind.endswith('_apt') else 'lgb')
    return blk, kind


def fit_member(name, cm=None):
    import train_eval
    p = probe()
    kind0, params, rounds, _ = p.CANDIDATES[name]
    if not kind0.startswith(('lgb',)):
        raise SystemExit(f'{name}: kind {kind0} is not a LightGBM member')
    cols, cats = STATE['cols'], STATE['cats']
    cm = deploy_rows() if cm is None else cm
    new = mc.load_rank(5); old = mc.load_rank(5, old=True)
    if TRAIN_ADSB[0] is None:
        blk, kind = transform(kind0, {'train': cm, 'held': new, 'cols': cols, 'cats': cats, 'block': 'deploy'})
        with old_tables():
            blk_old, kind_o = transform(kind0, {'train': cm.iloc[:0], 'held': old, 'cols': cols, 'cats': cats, 'block': 'deploy'},
                                        held_only=True)
    else:                 # variant A: training rows read the TRAIN_ADSB tables, the final rows the --adsb tables
        mc.point_adsb(TRAIN_ADSB[0])
        blk, kind = transform(kind0, {'train': cm, 'held': new.iloc[:0], 'cols': cols, 'cats': cats, 'block': 'deploy'})
        with old_tables():   # old rows scored on the same tables as training = comparable with reference
            blk_old, kind_o = transform(kind0, {'train': cm.iloc[:0], 'held': old, 'cols': cols, 'cats': cats, 'block': 'deploy'},
                                        held_only=True)
        mc.point_adsb(TRAIN_ADSB[1])
        blk_new, kind_n = transform(kind0, {'train': cm.iloc[:0], 'held': new, 'cols': cols, 'cats': cats, 'block': 'deploy'},
                                    held_only=True)
        assert kind_n == kind and blk_new['cols'] == blk['cols']
        blk = {**blk, 'held': blk_new['held']}
    assert kind_o == kind and blk_old['cols'] == blk['cols'], (kind, kind_o)
    tr_rows, cols = blk['train'], blk['cols']
    mdir = MODELS / name; mdir.mkdir(parents=True, exist_ok=True)
    helds = {'new': blk['held'], 'old': blk_old['held']}
    out = {k: np.full(len(h), np.nan) for k, h in helds.items()}
    t0 = time.time()
    if kind == 'lgb':                                   # 107: train_eval.fit_lgb on all clean matched rows
        X = train_eval.prep(tr_rows, cols, cats)
        y = tr_rows.TAXITIME_SEC_mvt.values - tr_rows.atot_minus_aobt3.values
        m = train_eval.fit_lgb(X, y, cats, dict(params), rounds, f'member_{name}')
        m.save_model(str(mdir / 'model.txt'))
        for k, h in helds.items():
            has = np.isfinite(h.atot_minus_aobt3.values.astype(float))
            out[k][has] = m.predict(train_eval.prep(h[has], cols, cats, train_eval._cats_of(X, cats)))
    elif kind == 'lgb_plus_apt':                        # 106.fit_lightgbm_per_airport -> tune_residual.fit_lightgbm
        pl = train_eval._lgb_params(params)
        for apt in sorted(helds['new'].APT.unique()):
            ta = tr_rows[tr_rows.APT.values == apt]
            if len(ta) < 1000:
                continue
            X = train_eval.prep(ta, cols, cats)
            y = ta.TAXITIME_SEC_mvt.values - ta.atot_minus_aobt3.values
            dtr = lgb.Dataset(X, y, categorical_feature=[c for c in cats if c in X.columns], free_raw_data=False,
                              params={'max_bin': pl.get('max_bin', 255)})
            m = lgb.train(pl, dtr, rounds)
            m.save_model(str(mdir / f'model_{apt}.txt'))
            for k, h in helds.items():
                b = (h.APT.values == apt) & np.isfinite(h.atot_minus_aobt3.values.astype(float))
                if b.any():
                    out[k][b] = m.predict(train_eval.prep(h[b], cols, cats, train_eval._cats_of(X, cats)))
            say(f'  {name} {apt}: {len(ta):,} rows [{time.time() - t0:.0f}s]')
    else:
        raise SystemExit(f'{name}: unexpected kind {kind}')
    secs = time.time() - t0
    rep = {'name': name, 'kind': kind0, 'params': params, 'rounds': rounds, 'train_rows': int(len(tr_rows)),
           'n_cols': len(cols), 'seconds': round(secs, 1)}
    for k, h in helds.items():
        d = CACHE_NEW if k == 'new' else CACHE_OLD
        d.mkdir(parents=True, exist_ok=True)
        res = pd.DataFrame({'MVT_ID_mvt': h.MVT_ID_mvt.values, f'r_{name}': out[k]})
        res.to_parquet(d / f'member_rank_{name}.parquet', index=False)
        ref = pd.read_parquet(REFM / 'member_cache' / f'member_rank_{name}.parquet')
        c = compare(res.MVT_ID_mvt, res[f'r_{name}'], ref.MVT_ID_mvt, ref[f'r_{name}'], f'{name} ({k}-scored) vs reference')
        rep[k] = c; say(fmt(c))
    say(f'{name}: fitted on {len(tr_rows):,} rows x {len(cols)} cols in {secs:.0f}s')
    save_report(f'member_{name}' + (f'__{VARIANT[0]}' if VARIANT[0] else ''), rep)
    return rep


def set_variant(v):
    """--variant X: write NEW/OLD outputs and boosters under runs/members_repro/variant_X/ (reproduction-only runs,
    e.g. the ADS-B members fitted on the 09-26 snapshot tables); the deliverable caches are not touched."""
    global CACHE_NEW, CACHE_OLD, MODELS
    VARIANT[0] = v
    if v == 'A':          # deliverable variant A: ADS-B members trained on the 09-26 snapshot's 2025 rows
        CACHE_NEW, CACHE_OLD, MODELS = OUT / 'variant_A' / 'member_cache', REPRO / 'variant_A_old' / 'member_cache', OUT / 'variant_A' / 'member_models'
    elif v:
        base = REPRO / f'variant_{v}'
        CACHE_NEW, CACHE_OLD, MODELS = base / 'new' / 'member_cache', base / 'old' / 'member_cache', base / 'member_models'


def cmd_member(a):
    set_variant(a.variant)
    if a.train_adsb:
        if not a.adsb:
            raise SystemExit('--train-adsb needs --adsb (the scoring-side tables)')
        TRAIN_ADSB[0], TRAIN_ADSB[1] = a.train_adsb, a.adsb
    if a.adsb:
        mc.point_adsb(a.adsb)
    elif any(probe().CANDIDATES[n][0].startswith(('lgb_adsb', 'lgb_sg_adsb')) for n in a.names):
        raise SystemExit('--adsb <folder> is required for the ADS-B members')
    cm = deploy_rows()
    for name in a.names:
        fit_member(name, cm)


def twin_dir(variant, k):
    if variant == 'A':
        return (OUT / 'variant_A' if k == 'new' else REPRO / 'variant_A_old') / 'member_cache_a46'
    if variant:
        return REPRO / f'variant_{variant}' / k / 'member_cache_a46'
    return (OUT if k == 'new' else REPRO / 'old_scored') / 'member_cache_a46'


def cmd_twins(a):
    """124.rank_twins(seeds) split in halves: BLIND reads no ADS-B column (with_trace only merges), AB needs F + S."""
    import common
    import train_eval
    m124 = importlib.import_module('124_unmatched_adsb')
    un = importlib.import_module('110_unmatched_normal')
    a35 = importlib.import_module('111_bundle_a35')
    if a.part == 'ab':
        if not a.adsb:
            raise SystemExit('--adsb <folder> is required for the AB half')
        mc.point_adsb(a.adsb)
    manifest = common.read_manifest(5)
    train = pd.read_parquet(DATA / 'features_v5_train.parquet')
    cols, cats = a35._strict_cols(manifest)
    drop = set(un.NM_NAN) | set(un.NM_CONST)
    cols = [c for c in cols if c not in drop]; cats = [c for c in cats if c not in drop]
    keep = m124._population(train, np.ones(len(train), bool))
    tr_rows = un.mask_nm(train[keep]); del train
    helds = {}
    for k, old in (('new', False), ('old', True)):
        r = mc.load_rank(5, old=old)
        helds[k] = un.mask_nm(r[r.fpl_matched.values == 0])
    if a.part == 'ab' and a.train_adsb:      # variant A: training and old rows on the snapshot, final rows on --adsb
        mc.point_adsb(a.train_adsb)
        tr_rows = m124.with_trace(tr_rows); helds['old'] = m124.with_trace(helds['old'])
        mc.point_adsb(a.adsb)
        helds['new'] = m124.with_trace(helds['new'])
        cc = cols + m124.AB_COLS
    elif a.part == 'ab':
        tr_rows = m124.with_trace(tr_rows); helds = {k: m124.with_trace(h) for k, h in helds.items()}
        cc = cols + m124.AB_COLS
    else:
        cc = cols
    say(f'twins {a.part}: {len(tr_rows):,} training rows x {len(cc)} cols; held new {len(helds["new"]):,}, old {len(helds["old"]):,}')
    y = tr_rows.TAXITIME_SEC_mvt.values
    X = train_eval.prep(tr_rows, cc, cats)
    Xh = {k: train_eval.prep(h, cc, cats, train_eval._cats_of(X, cats)) for k, h in helds.items()}
    set_variant(a.variant)
    mdir = MODELS / 'unmatched_twins'; mdir.mkdir(parents=True, exist_ok=True)
    res = {k: pd.DataFrame({'MVT_ID_mvt': h.MVT_ID_mvt.values}) for k, h in helds.items()}
    for s in a.seeds:
        t0 = time.time()
        m = lgb.train(train_eval._lgb_params({**m124.PARAMS, 'seed': s}),
                      lgb.Dataset(X, y, categorical_feature=[c for c in cats if c in X.columns], free_raw_data=False), m124.ROUNDS)
        m.save_model(str(mdir / f'{a.part}_s{s}.txt'))
        for k in helds:
            res[k][f'{a.part}_s{s}'] = m.predict(Xh[k])
        say(f'{a.part} seed {s}: {len(tr_rows):,} rows x {len(cc)} cols in {time.time() - t0:.0f}s')
    ref = pd.read_parquet(REFM / 'member_cache_a46' / 'unmatched_adsb_rank_41_42_43.parquet')
    rep = {'part': a.part, 'train_rows': int(len(tr_rows)), 'n_cols': len(cc)}
    for k, r in res.items():
        d = twin_dir(a.variant, k)
        d.mkdir(parents=True, exist_ok=True)
        r.to_parquet(d / f'unmatched_twins_{a.part}_{"_".join(map(str, a.seeds))}.parquet', index=False)
        rep[k] = []
        for s in a.seeds:
            if f'{a.part}_s{s}' not in ref.columns:   # extra seeds (v73) have no reference counterpart
                continue
            c = compare(r.MVT_ID_mvt, r[f'{a.part}_s{s}'], ref.MVT_ID_mvt, ref[f'{a.part}_s{s}'], f'{a.part}_s{s} ({k}-scored) vs reference')
            rep[k].append(c); say(fmt(c))
    save_report(f'twins_{a.part}' + (f'__{a.variant}' if a.variant else ''), rep)
    tag = "_".join(map(str, a.seeds))
    for k in res:                                   # both halves present -> 124's rank file layout (blind_s*, ab_s*)
        d = twin_dir(a.variant, k)
        fb, fa = d / f'unmatched_twins_blind_{tag}.parquet', d / f'unmatched_twins_ab_{tag}.parquet'
        if fb.exists() and fa.exists():
            both = pd.read_parquet(fb).merge(pd.read_parquet(fa), on='MVT_ID_mvt', how='inner', validate='one_to_one')
            both.to_parquet(d / f'unmatched_adsb_rank_{tag}.parquet', index=False)
            say(f'{k}: wrote {d / f"unmatched_adsb_rank_{tag}.parquet"} ({len(both):,} rows)')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('surge')
    m = sub.add_parser('member'); m.add_argument('names', nargs='+'); m.add_argument('--adsb'); m.add_argument('--variant'); m.add_argument('--train-adsb')
    t = sub.add_parser('twins'); t.add_argument('--part', choices=['blind', 'ab'], required=True)
    t.add_argument('--seeds', nargs='+', type=int, default=[41, 42, 43]); t.add_argument('--adsb'); t.add_argument('--variant'); t.add_argument('--train-adsb')
    a = ap.parse_args()
    {'surge': cmd_surge, 'member': cmd_member, 'twins': cmd_twins}[a.cmd](a)


if __name__ == '__main__':
    main()
