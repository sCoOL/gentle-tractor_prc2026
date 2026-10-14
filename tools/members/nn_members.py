"""Neural members of v45 refitted once on the GPU and scored on NEW and OLD rank rows (docs/METHOD.md).

  py -3.14 tools/members/nn_members.py mlp_set mlp_set2 ...          # A48 members (107.fit_rank_member path)
  py -3.14 tools/members/nn_members.py set_clf nm_masked_set ...     # A35 nets (111.rank_members path)
  py -3.14 tools/members/nn_members.py u_self u_self_s1 u_self_s2    # A37 mode nets (114.rank_p path)

Training is the reference deploy path verbatim (columns, rows, params, seeds). The network is the one in
tools/members/probe106_ext.py (= scripts/106_member_probe.py + the post-training extra-scoring hook), registered as
the '106_member_probe' module so that 107/108/110/111/114 use it too. OLD rows get their neighbour tokens / masks from
the reference old export (data_old). Seed-noise yardstick: the RMS difference between two of the reference member's own sibling seeds.
"""
from __future__ import annotations

import mcommon as mc

import argparse
import importlib
import json
import sys
import time

import numpy as np
import pandas as pd

import probe106_ext
sys.modules['106_member_probe'] = probe106_ext

from mcommon import REFM, OUT, REPRO, DATA, say, compare, fmt, save_report

STATE = json.loads((REFM / 'A27_20260917_141019' / 'M3rpa' / 'state.json').read_text(encoding='utf-8'))
A48_NN = {'mlp', 'mlp_s1', 'mlp_s2', 'mlp_e20', 'mlp_wide', 'mlp_set', 'mlp_set2', 'mlp_set_s1', 'mlp_setn', 'mlp_setn2',
          'mlp_setn_s1', 'mlp_setn_r', 'mlp_setn2_r', 'mlp_setn_s1_r'} | {f'mlp_setn_ab{s}' for s in ('', '_s1', '_s2', '_s3', '_s4', '_s5', '_s6')}
A35_NORMAL = ['nm_masked_set', 'nm_masked_self', 'nm_masked_set2']
A35_CLF = ['set_clf', 'self_clf', 'set_clf2']
A37 = ['u_self', 'u_self_s1', 'u_self_s2']
TRAIN_ADSB = [None, None]
NO_REPLAY = [False]
NO_OLD = [False]
VARIANT = [None]     # --variant X: reproduction-only outputs under runs/members_repro/variant_X/{new,old}/
SIBLING = {'mlp': 'mlp_s1', 'mlp_s1': 'mlp', 'mlp_s2': 'mlp', 'mlp_e20': 'mlp', 'mlp_wide': 'mlp',
           'mlp_set': 'mlp_set_s1', 'mlp_set_s1': 'mlp_set', 'mlp_set2': 'mlp_set',
           'mlp_setn': 'mlp_setn_r', 'mlp_setn_r': 'mlp_setn', 'mlp_setn2': 'mlp_setn2_r', 'mlp_setn2_r': 'mlp_setn2',
           'mlp_setn_s1': 'mlp_setn_s1_r', 'mlp_setn_s1_r': 'mlp_setn_s1',
           'set_clf': 'set_clf2', 'set_clf2': 'set_clf', 'self_clf': 'set_clf',
           'nm_masked_set': 'nm_masked_set2', 'nm_masked_set2': 'nm_masked_set', 'nm_masked_self': 'nm_masked_set',
           'u_self': 'u_self_s1', 'u_self_s1': 'u_self', 'u_self_s2': 'u_self',
           'mlp_setn_ab': 'mlp_setn_ab_s1', 'mlp_setn_ab_s1': 'mlp_setn_ab', 'mlp_setn_ab_s2': 'mlp_setn_ab',
           'mlp_setn_ab_s3': 'mlp_setn_ab', 'mlp_setn_ab_s4': 'mlp_setn_ab', 'mlp_setn_ab_s5': 'mlp_setn_ab',
           'mlp_setn_ab_s6': 'mlp_setn_ab'}


def out_dir(k, folder):
    v = VARIANT[0]
    if v == 'A':
        return (OUT / 'variant_A' if k == 'new' else REPRO / 'variant_A_old') / folder
    if v:
        return REPRO / f'variant_{v}' / k / folder
    return (OUT if k == 'new' else REPRO / 'old_scored') / folder


def report_name(name):
    return f'nn_{name}__' + (VARIANT[0] if VARIANT[0] else 'saved')


def model_path(name):
    base = OUT / 'variant_A' if VARIANT[0] == 'A' else (OUT if not VARIANT[0] else REPRO / f'variant_{VARIANT[0]}')
    return base / 'member_models' / 'nn' / f'{name}.pt'


def new_tokens(ids, ctx):
    import nbr_set
    all_ids, T = nbr_set.build('rank', **(ctx or {}))
    return nbr_set.aligned(all_ids, T, ids)


def old_tokens(ids, ctx):
    import nbr_set
    with mc.old_rank_source():
        all_ids, T = nbr_set.build('rank', **(ctx or {}))
    return nbr_set.aligned(all_ids, T, ids)


def a48_block(name, train_v5, new, old):
    import tail_experts as te
    import train_eval
    p = probe106_ext
    kind, params, _, _ = p.CANDIDATES[name]
    assert kind in ('mlp', 'mlp_set', 'mlp_set_adsb'), (name, kind)
    bug = te.training_exclusion(train_v5.TAXITIME_SEC_mvt.values, train_v5.atot_minus_sched.values, 'datebug')
    clean = train_eval._clean(train_v5[~bug])
    cm = clean[np.isfinite(clean.atot_minus_aobt3.values)]
    blk = {'train': cm, 'held': new, 'cols': STATE['cols'], 'cats': STATE['cats'], 'block': 'deploy',
           'extra_held': [dict(tag='old', frame=old, tokens=old_tokens)], 'extra_out': {}}
    if kind == 'mlp_set_adsb':                      # 107: with_adsb(blk), then the set network (needs point_adsb first)
        import adsb_context
        if TRAIN_ADSB[0] is None:
            blk = p.with_adsb(blk)
            blk['extra_held'][0]['frame'] = old.merge(adsb_context.table(), on='MVT_ID_mvt', how='left', validate='one_to_one')
        else:                                       # variant A: training + old rows on the snapshot, final rows on --adsb
            tA = adsb_context.table(mc.Path(TRAIN_ADSB[0]) / 'adsb_features.parquet')
            tH = adsb_context.table(mc.Path(TRAIN_ADSB[1]) / 'adsb_features.parquet')
            add = lambda df, f: df.merge(f, on='MVT_ID_mvt', how='left', validate='one_to_one')
            blk = {**blk, 'train': add(blk['train'], tA), 'held': add(blk['held'], tH), 'cols': blk['cols'] + adsb_context.COLUMNS}
            blk['extra_held'][0]['frame'] = add(old, tA)
        kind = 'mlp_set'
    fit = p.fit_mlp if kind == 'mlp' else p.fit_mlp_set
    return blk, fit, params, ('member_cache', f'member_rank_{name}.parquet', f'r_{name}')


def a35_block(name, train_v5, new, old, manifest):
    import tail_experts as te
    un = importlib.import_module('110_unmatched_normal')
    det = importlib.import_module('108_matched_detector')
    a35 = importlib.import_module('111_bundle_a35')
    cols, cats = a35._strict_cols(manifest)
    bug = te.datebug_label(train_v5.TAXITIME_SEC_mvt.values, train_v5.atot_minus_sched.values)
    fixed = (~bug) & (train_v5.is_heli.values == 0)
    drop = set(un.NM_NAN) | set(un.NM_CONST)
    if name in A35_NORMAL:
        blk = {'train': un.mask_nm(train_v5[fixed & (train_v5.fallback.values == 0)]), 'held': un.mask_nm(new), 'block': 'deploy',
               'cols': [c for c in cols if c not in drop], 'cats': [c for c in cats if c not in drop],
               'extra_held': [dict(tag='old', frame=un.mask_nm(old), tokens=old_tokens)], 'extra_out': {}}
        params = un.CANDIDATES[name]
    else:
        blk = {'train': train_v5[fixed & np.isfinite(train_v5.atot_minus_aobt3.values)], 'held': new, 'block': 'deploy',
               'cols': cols, 'cats': cats, 'extra_held': [dict(tag='old', frame=old, tokens=old_tokens)], 'extra_out': {}}
        params = det.CANDIDATES[name]
    return blk, probe106_ext.fit_mlp_set, params, ('member_cache_a35', f'{name}.parquet', 'v')


def a37_block(name, train_v5, new, old, manifest):
    import tail_experts as te
    un = importlib.import_module('110_unmatched_normal')
    a35 = importlib.import_module('111_bundle_a35')
    m114 = importlib.import_module('114_unmatched_mode')
    cols, cats = a35._strict_cols(manifest)
    bug = te.datebug_label(train_v5.TAXITIME_SEC_mvt.values, train_v5.atot_minus_sched.values)
    flag = lambda df: df.assign(is_unmatched=(df.fpl_matched.values == 0).astype('float32'))
    drop = set(un.NM_NAN) | set(un.NM_CONST)
    blk = {'train': un.mask_nm(flag(train_v5[~bug & (train_v5.is_heli.values == 0)])), 'held': un.mask_nm(flag(new)),
           'block': 'deploy', 'cols': [c for c in cols if c not in drop] + ['is_unmatched'],
           'cats': [c for c in cats if c not in drop],
           'extra_held': [dict(tag='old', frame=un.mask_nm(flag(old)), tokens=old_tokens)], 'extra_out': {}}
    return blk, probe106_ext.fit_mlp_set, {**m114.CANDIDATES[name], 'taxi_expert': True}, ('member_cache_a37', f'{name}.parquet', 'p')


def run(name, train_v5, new, old, manifest):
    import gc
    train_v5 = pd.read_parquet(DATA / 'features_v5_train.parquet') if train_v5 is None else train_v5
    if name in A48_NN:
        blk, fit, params, (folder, fname, col) = a48_block(name, train_v5, new, old)
    elif name in A35_NORMAL + A35_CLF:
        blk, fit, params, (folder, fname, col) = a35_block(name, train_v5, new, old, manifest)
    elif name in A37:
        blk, fit, params, (folder, fname, col) = a37_block(name, train_v5, new, old, manifest)
    else:
        raise SystemExit(f'unknown neural member {name}')
    del train_v5; gc.collect()            # blk['train'] is its own copy; the full table is not needed during the fit
    if NO_OLD[0]:                          # refits that only add a saved network: the old-id check was done on the first fit
        blk['extra_held'] = []
    t0 = time.time()
    blk['save_to'] = model_path(name)
    say(f'{name}: training rows {len(blk["train"]):,}, {len(blk["cols"])} cols, params {params}')
    r, _ = fit(blk, params)
    secs = time.time() - t0
    # the saved network must re-predict the NEW held frame exactly as the training run did (one member of each kind)
    if NO_REPLAY[0]:
        reload_gap = {'label': 'replay skipped (--no-replay); checked on one member of the same kind'}
    else:
        r2 = probe106_ext.predict_saved(blk['save_to'], blk['held'], new_tokens)
        reload_gap = compare(blk['held'].MVT_ID_mvt.values, r2, blk['held'].MVT_ID_mvt.values, r, f'{name} saved-network replay vs training-run prediction')
        say(fmt(reload_gap))
    outs = {'new': (blk['held'], r)}
    if blk['extra_held']:
        outs['old'] = (blk['extra_held'][0]['frame'], blk['extra_out']['old'])
    ref = pd.read_parquet(REFM / folder / fname)
    rep = {'name': name, 'model': str(blk['save_to']), 'reload_gap': reload_gap,
           'params': {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
           'train_rows': int(len(blk['train'])), 'n_cols': len(blk['cols']), 'seconds': round(secs, 1)}
    sib = SIBLING.get(name)
    if sib:
        sref = pd.read_parquet(REFM / folder / fname.replace(name, sib))
        scol = col if col in ('v', 'p') else f'r_{sib}'
        rep['seed_noise_main'] = compare(ref.MVT_ID_mvt, ref[col], sref.MVT_ID_mvt, sref[scol], f'reference {name} vs reference {sib} (seed yardstick)')
        say(fmt(rep['seed_noise_main']))
    for k, (frame, vals) in outs.items():
        d = out_dir(k, folder)
        prev = d / fname
        if prev.exists():          # the refit (now with a saved network) against the earlier rebuild prediction
            pv = pd.read_parquet(prev)
            rep[f'{k}_vs_previous_rebuild'] = compare(frame.MVT_ID_mvt.values, vals, pv.MVT_ID_mvt, pv[col], f'{name} ({k}) vs previous rebuild run')
            say(fmt(rep[f'{k}_vs_previous_rebuild']))
        d.mkdir(parents=True, exist_ok=True)
        res = pd.DataFrame({'MVT_ID_mvt': frame.MVT_ID_mvt.values, col: vals})
        res.to_parquet(d / fname, index=False)
        c = compare(res.MVT_ID_mvt, res[col], ref.MVT_ID_mvt, ref[col], f'{name} ({k}-scored) vs reference')
        rep[k] = c; say(fmt(c))
    say(f'{name}: done in {secs:.0f}s')
    save_report(report_name(name), rep)
    probe106_ext._TOKENS.clear()
    import gc; gc.collect()
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='+')
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--adsb', help='folder with the v1 ADS-B tables (required for mlp_setn_ab*)')
    ap.add_argument('--variant')
    ap.add_argument('--no-replay', action='store_true', help='skip the saved-network replay check (time)')
    ap.add_argument('--no-old', action='store_true', help='skip scoring the old export (time)')
    ap.add_argument('--train-adsb', help='variant A: ADS-B tables for the TRAINING rows (and the old rows); --adsb scores the final rows')
    a = ap.parse_args()
    VARIANT[0] = a.variant
    NO_REPLAY[0] = a.no_replay
    NO_OLD[0] = a.no_old
    if a.train_adsb:
        TRAIN_ADSB[0], TRAIN_ADSB[1] = a.train_adsb, a.adsb
    if a.adsb:
        mc.point_adsb(a.adsb)
    elif any(n.startswith('mlp_setn_ab') for n in a.names):
        raise SystemExit('--adsb <folder> is required for the ADS-B set members')
    import common
    manifest = common.read_manifest(5)
    new = mc.load_rank(5); old = mc.load_rank(5, old=True)
    for name in a.names:
        if a.skip_existing and (mc.REPRO / (report_name(name) + '.json')).exists():
            say(f'{name}: report exists, skipped'); continue
        run(name, None, new, old, manifest)


if __name__ == '__main__':
    main()
