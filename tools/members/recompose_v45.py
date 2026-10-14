"""Recompose v45 (A51) from its member predictions, exactly as the reference package chain did (docs/METHOD.md).

The reference chain is 96 (A27) -> 105 (A30/A31) -> 107 (A48m) -> 111 (A35) -> 113 (A48) -> 114 (A48f) -> 124 (A48u)
-> 132 (A51/v45). Each step only recomposes rows from saved member predictions, so this tool writes the whole chain as
one function of the members and replays A51's five saved boosters.

  py -3.14 tools/members/recompose_v45.py main      # the reference saved inputs + old features -> must equal v45 bit for bit
  py -3.14 tools/members/recompose_v45.py rebuild --rank new|old [--allow-missing]   # the rebuilt member caches

Inputs (a dict of paths, see SOURCES): A27-like package (normal_rank, normal_rank_rd, p_fallback_rank, rank_detail with
gate_valid/w/pb/day_shift/normal_pred), the rd member table (aobt3_anchor, r_global_seed1, r_airport), member_cache
(member_rank_<name>.parquet, column r_<name>), member_cache_a35 (v), member_cache_a37 (p), the unmatched twins
(blind_s*/ab_s*), the v1 ADS-B table folder (cell labels and masks), base_rank (A51 raw fields) and the A51 package.
"""
from __future__ import annotations

import mcommon as mc  # noqa: F401  (threads, sys.path)

import argparse
import importlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from mcommon import REB, REFM, REF, OLD, DATA, SNAP_ADSB, OUT, say, compare, fmt, save_report

FLOOR = 30.0
A48_MEMBERS = ['mlp+mlp_s1+mlp_s2+mlp_e20+mlp_wide', 'mlp_set+mlp_set2+mlp_set_s1',
               'mlp_setn+mlp_setn2+mlp_setn_s1+mlp_setn_r+mlp_setn2_r+mlp_setn_s1_r', 'lgb_sg', 'lgb_ab+lgb_ab_s1',
               'lgb_sg_ab+lgb_sg_ab_s1',
               'mlp_setn_ab+mlp_setn_ab_s1+mlp_setn_ab_s2+mlp_setn_ab_s3+mlp_setn_ab_s4+mlp_setn_ab_s5+mlp_setn_ab_s6',
               'lgb_ab_apt', 'lgb_sg_ab_apt', 'lgb_q12', 'lgb_sg_q12', 'lgb_ab_trust+lgb_sg_ab_trust',
               'lgb_ab_st+lgb_ab_st_s1+lgb_ab_st_s2+lgb_sg_ab_st+lgb_sg_ab_st_s1+lgb_sg_ab_st_s2']
CLF = ['set_clf', 'self_clf', 'set_clf2']                      # 111.CLF order
NORMAL = ['nm_masked_set', 'nm_masked_self', 'nm_masked_set2']  # 111.NORMAL order
A27 = 'A27_20260917_141019'
A51 = 'A51_20260923_010203'

SOURCES = {
    'main': dict(feat=OLD / 'features_v5_rank.parquet', base_rank=OLD / 'base_rank.parquet',
                 a27=REFM / A27, resm=REFM / 'A31_20260918_075753' / 'residual_members_rank.parquet',
                 cache=REFM / 'member_cache', a35=REFM / 'member_cache_a35', a37=REFM / 'member_cache_a37',
                 twins=REFM / 'member_cache_a46' / 'unmatched_adsb_rank_41_42_43.parquet', adsb=SNAP_ADSB,
                 a51=REFM / A51, n=mc.N_OLD),
}


def rebuild_sources(rank='new', adsb=None, base_dir=None):
    """The rebuilt caches; --rank old reads the OLD-scored copies written next to them by the members tools;
    ``base_dir`` (e.g. runs/models/final_A) overrides where the NEW caches are read from."""
    sub = '' if rank == 'new' else 'old_scored'
    base = (Path(base_dir) if base_dir else OUT) if rank == 'new' else mc.REPRO / 'old_scored'
    return dict(feat=(DATA if rank == 'new' else OLD) / 'features_v5_rank.parquet',
                base_rank=(DATA if rank == 'new' else OLD) / 'base_rank.parquet',
                a27=base / 'R_A27', resm=base / 'R_A27' / 'residual_members_rank.parquet',
                cache=base / 'member_cache', a35=base / 'member_cache_a35', a37=base / 'member_cache_a37',
                twins=base / 'member_cache_a46' / 'unmatched_adsb_rank_41_42_43.parquet',
                adsb=Path(adsb) if adsb else SNAP_ADSB, a51=REFM / A51, n=mc.N_FINAL if rank == 'new' else mc.N_OLD,
                tag=sub or 'new')


def _first(dirs, fname):
    """The first folder (in order) holding ``fname``; the last folder's path when none does."""
    dirs = [dirs] if isinstance(dirs, (str, Path)) else list(dirs)
    for d in dirs:
        if (Path(d) / fname).exists():
            return Path(d) / fname
    return Path(dirs[-1]) / fname


def hybrid_sources():
    """OLD rows: every rebuilt OLD-scored input that exists, the reference saved prediction for the rest (ADS-B members,
    AB twins). A27 inputs (expert, rd members, detector, gate) are taken as one block from the rebuild once its
    rank_detail exists, so the rd refit gap is that of one fit (0)."""
    ro = mc.REPRO / 'old_scored'
    a27 = ro / 'R_A27' if (ro / 'R_A27' / 'rank_detail.parquet').exists() else REFM / A27
    tw = pd.read_parquet(REFM / 'member_cache_a46' / 'unmatched_adsb_rank_41_42_43.parquet')
    fb = ro / 'member_cache_a46' / 'unmatched_twins_blind_41_42_43.parquet'
    if fb.exists():
        b = pd.read_parquet(fb)
        tw = tw.drop(columns=[c for c in tw.columns if c.startswith('blind_')]).merge(b, on='MVT_ID_mvt', how='left', validate='one_to_one')
    return dict(feat=OLD / 'features_v5_rank.parquet', base_rank=OLD / 'base_rank.parquet', a27=a27,
                resm=(ro / 'R_A27' / 'residual_members_rank.parquet') if a27 != REFM / A27 else REFM / 'A31_20260918_075753' / 'residual_members_rank.parquet',
                cache=[ro / 'member_cache', REFM / 'member_cache'], a35=[ro / 'member_cache_a35', REFM / 'member_cache_a35'],
                a37=[ro / 'member_cache_a37', REFM / 'member_cache_a37'], twins=tw, adsb=SNAP_ADSB, a51=REFM / A51,
                n=mc.N_OLD, blind_from=str(fb if fb.exists() else 'reference'))


def _align(order, df, col):
    return order.merge(df, on='MVT_ID_mvt', how='left', validate='one_to_one')[col].values


def fc_mixture(p, D, normal, is_heli, heli_pred, floor=30.0):
    """fallback_context.mixture, verbatim."""
    p = np.clip(np.asarray(p, float), 0, 1); D = np.asarray(D, float); normal = np.asarray(normal, float)
    ok = np.isfinite(D) & (D > 0) & np.isfinite(p)
    mix = np.where(ok, p * D + (1 - p) * normal, normal)
    mix = np.where(np.asarray(is_heli) == 1, heli_pred, mix)
    return np.maximum(mix, floor)


def te_combine(two_expert_pred, pb, normal, day_shift, floor=FLOOR):
    """tail_experts.combine, verbatim."""
    k = np.clip(np.asarray(day_shift, float), 1, None)
    pb = np.asarray(pb, float)
    out = (1.0 - pb) * np.asarray(two_expert_pred, float) + pb * (k * 86400.0 + np.asarray(normal, float))
    return np.maximum(out, floor)


def compose_a48u(S, allow_missing=False):
    """Every row of the A48u package (v42 = max(pred, 30)) from member predictions. Returns a frame with intermediates."""
    m107 = importlib.import_module('107_extra_members')
    m124 = importlib.import_module('124_unmatched_adsb')
    mc.point_adsb(S['adsb'])
    feat = pd.read_parquet(S['feat'], columns=['MVT_ID_mvt', 'APT', 'fpl_matched', 'is_heli', 'atot_minus_sched', 'atot_minus_aobt3'])
    assert len(feat) == S['n'], len(feat)
    ids = feat.MVT_ID_mvt.values
    order = pd.DataFrame({'MVT_ID_mvt': ids})
    a27 = Path(S['a27'])
    state = json.loads((REFM / A27 / 'M3rpa' / 'state.json').read_text(encoding='utf-8'))
    heli_const = float(state['heli_const'])
    det = order.merge(pd.read_parquet(a27 / 'rank_detail.parquet'), on='MVT_ID_mvt', how='left', validate='one_to_one')
    plain = _align(order, pd.read_parquet(a27 / 'normal_rank.parquet'), 'normal_pred')
    rd_saved = _align(order, pd.read_parquet(a27 / 'normal_rank_rd.parquet'), 'normal_pred')
    p_old = _align(order, pd.read_parquet(a27 / 'p_fallback_rank.parquet'), 'p_fallback')
    mem = order.merge(pd.read_parquet(S['resm']), on='MVT_ID_mvt', how='left', validate='one_to_one')
    mem = mem.merge(feat[['MVT_ID_mvt', 'is_heli', 'APT', 'atot_minus_sched']], on='MVT_ID_mvt', how='left', validate='one_to_one')
    a3 = mem.aobt3_anchor.values; is_heli = mem.is_heli.values; apt = mem.APT.values
    D = feat.atot_minus_sched.values
    missing = []
    cols = []
    for spec in A48_MEMBERS:                                  # 107.rank_member: np.mean over the spec's cached members
        rs = []
        for name in spec.split('+'):
            f = _first(S['cache'], f'member_rank_{name}.parquet')
            S.setdefault('used', {})[name] = str(f.parent)
            if not f.exists():
                missing.append(name); rs.append(np.full(len(order), np.nan)); continue
            rs.append(order.merge(pd.read_parquet(f), on='MVT_ID_mvt', how='left', validate='one_to_one')[f'r_{name}'].values)
        cols.append(np.mean(rs, axis=0))
    if missing and not allow_missing:
        raise SystemExit(f'missing members: {missing}')
    R = np.column_stack([plain - a3, mem.r_global_seed1.values, mem.r_airport.values] + cols)
    has = np.isfinite(a3)
    rd_refit = np.where(has, a3 + R[:, :3] @ np.array([0.25, 0.25, 0.5]), plain)
    rd_refit = np.where(is_heli == 1, plain, rd_refit)
    gap = rd_saved - rd_refit
    wrep = json.loads((REF / 'runs' / 'residual_weights_A48_20260922.json').read_text(encoding='utf-8'))
    weights = wrep['deploy']
    rank_cell = m107.cell_labels(mem.assign(atot_minus_aobt3=mem.aobt3_anchor.values), wrep.get('cells', 'apt'), 0)
    rank_cell = np.where(np.isin(rank_cell, list(weights)), rank_cell, mem.APT.values.astype(str))
    miss_cells = sorted({x for x in set(rank_cell) if x not in weights})
    if miss_cells:
        raise SystemExit(f'no deploy weights for cells {miss_cells}')
    w = np.array([weights.get(x, [0.25, 0.25, 0.5] + [0.0] * len(A48_MEMBERS)) for x in rank_cell], float)
    out = np.where(has, a3 + (np.nan_to_num(R) * w).sum(axis=1), plain)
    normal_new = np.where(is_heli == 1, plain, out) + gap
    matched = det.fpl_matched.values == 1
    normal_members = np.where(matched, normal_new, det.normal_pred.values)            # A48m rank_detail.normal_members

    # A35: copy classifiers into the matched probability, NM-masked set experts as the unmatched normal
    for n in CLF + NORMAL:
        S.setdefault('used', {})[n] = str(_first(S['a35'], f'{n}.parquet').parent)
    clf = pd.concat([pd.read_parquet(_first(S['a35'], f'{n}.parquet')).set_index('MVT_ID_mvt').v.rename(n) for n in CLF], axis=1)
    nrm = pd.concat([pd.read_parquet(_first(S['a35'], f'{n}.parquet')).set_index('MVT_ID_mvt').v.rename(n) for n in NORMAL], axis=1)
    n_new = nrm.mean(axis=1).reindex(ids).values; c_new = clf.mean(axis=1).reindex(ids).values
    mt = matched
    n_new = np.where(~mt & np.isfinite(n_new), n_new, det.normal_pred.values)
    p_new = np.where(np.isfinite(c_new), 0.5 * (c_new + p_old), p_old)

    def matched_pred(normal, p):                                                     # 113.deploy_cmd.matched_pred
        pp = np.clip(p, 0, 1); ok = np.isfinite(D) & (D > 0) & np.isfinite(pp)
        mix = np.where(ok, pp * D + (1 - pp) * normal, normal)
        return np.maximum(np.where(is_heli == 1, heli_const, mix), FLOOR)

    def unmatched_pred(normal, w_, p_):                                              # 111/114/124 unmatched_pred
        mix = fc_mixture(p_, D, normal, is_heli, np.full(len(det), heli_const))
        two = np.where(det.gate_valid.values, np.maximum(normal + np.clip(w_, 0, 1) * (D - normal), 30.0), mix)
        return te_combine(two, np.nan_to_num(det.pb.values), normal, det.day_shift.values)

    pred_a48_matched = matched_pred(normal_members, p_new)
    # A48f: the unmatched mode rule
    rule_f = json.loads((REF / 'runs' / 'unmatched_mode_rule_A48f.json').read_text(encoding='utf-8'))
    for n in rule_f['candidates']:
        S.setdefault('used', {})[n] = str(_first(S['a37'], f'{n}.parquet').parent)
    pu = pd.concat([pd.read_parquet(_first(S['a37'], f'{n}.parquet')).set_index('MVT_ID_mvt').p.rename(n)
                    for n in rule_f['candidates']], axis=1).mean(axis=1).reindex(ids).values
    pu = np.where(np.isfinite(pu), pu, p_old)
    al = np.array([rule_f['deploy'].get(x, 0.0) for x in det.APT.values])
    if 'dmax' in rule_f:
        al = al * (np.nan_to_num(D, nan=np.inf) <= rule_f['dmax'])
    w_f = (1 - al) * det.w.values + al * pu
    p_f = (1 - al) * p_old + al * pu
    pred_a48f_unm = unmatched_pred(n_new, w_f, p_f)
    # A48u: the unmatched ADS-B twins
    rule_u = json.loads((REF / 'runs' / 'unmatched_adsb_rule_A48u.json').read_text(encoding='utf-8'))
    um = det.fpl_matched.values == 0
    if isinstance(S['twins'], pd.DataFrame) or Path(S['twins']).exists():
        tw = S['twins'] if isinstance(S['twins'], pd.DataFrame) else pd.read_parquet(S['twins'])
        if [c for c in tw.columns if c.startswith('ab_s')] and not allow_missing:
            pass
        delta = pd.Series(m124.delta_of(tw), index=tw.MVT_ID_mvt.values).reindex(ids).values if any(c.startswith('ab_s') for c in tw.columns) else np.full(len(ids), np.nan)
        ab = pd.Series(m124.ab_of(tw), index=tw.MVT_ID_mvt.values).reindex(ids).values if any(c.startswith('ab_s') for c in tw.columns) else np.full(len(ids), np.nan)
    else:
        if not allow_missing:
            raise SystemExit(f"missing twins {S['twins']}")
        delta = np.full(len(ids), np.nan); ab = np.full(len(ids), np.nan)
    delta = np.where(um & np.isfinite(delta) & m124.saw_mask(ids) & (is_heli == 0), delta, 0.0)
    nrm_new = m124.corrected(n_new, delta, ab, um & m124.pushback_mask(ids) & (is_heli == 0), rule_u['lam'], rule_u.get('mu', 0.0),
                             um & m124.near_mask(ids, rule_u.get('near_m', m124.NEAR_M)) & (is_heli == 0), rule_u.get('mu2', 0.0))
    pred_unm = unmatched_pred(nrm_new, w_f, p_f)
    pred = np.where(um, pred_unm, pred_a48_matched)
    return pd.DataFrame({'MVT_ID_mvt': ids, 'APT': feat.APT.values, 'fpl_matched': feat.fpl_matched.values,
                         'is_heli': is_heli, 'atot_minus_sched': D, 'cell': rank_cell, 'gap': gap,
                         'normal_members_a36': normal_members, 'p_matched_mix': p_new, 'normal_unmatched_set': n_new,
                         'p_unmatched_mode': pu, 'alpha': al, 'pred_a48f_unm': pred_a48f_unm,
                         'adsb_unmatched_delta': delta, 'normal_unmatched_adsb': nrm_new, 'pred_a48u': pred,
                         'p_fallback': p_old}), missing


def _predict(model, state, score, threads=6):
    """129_identity_search.predict with 6 threads (prediction is per row, so the thread count cannot change it)."""
    X = score[state['features']].copy()
    for c, cats in state['categories'].items():
        v = X[c].astype('string').fillna('__NA__')
        X[c] = pd.Categorical(v.where(v.isin(cats), '__NA__'), categories=cats)
    raw = model.predict(X, num_threads=threads)
    if state['config']['mode'] == 'residual':
        raw = raw + score.w.to_numpy()
    return np.clip(raw, 0, 1)


def a51_replay(full, S):
    """132.build's rank path with the saved A51 boosters and history; returns (eligible mask, proposed, detail frame)."""
    s = importlib.import_module('129_identity_search')
    x = importlib.import_module('130_source_identity_search')
    b = importlib.import_module('131_outer_identity_detector')
    g = s.g; old = g.old
    pkg = Path(S['a51'])
    history = pd.read_parquet(pkg / 'history.parquet')
    raw = old._raw(Path(S['base_rank']))
    f = full[['MVT_ID_mvt', 'APT', 'fpl_matched', 'atot_minus_sched', 'p_matched_mix', 'normal_members_a36']].merge(
        raw, on='MVT_ID_mvt', how='left', validate='one_to_one')
    eligible = (f.APT.eq('LIRF') & f.fpl_matched.eq(1) & f.is_heli.eq(0)).to_numpy()
    rank = f.loc[eligible].copy(); rank['D'] = rank.atot_minus_sched; rank['a3taxi'] = rank.atot_minus_aobt3
    rank['actual_p'] = rank.p_matched_mix; rank['y'] = np.nan
    rank = b.enrich(g.frame(rank, rank.normal_members_a36), history)
    # 130.extra_table(rank=True) with explicit sources
    br = pd.read_parquet(S['base_rank'], columns=['MVT_ID_mvt', 'APT'] + x.RAW_COLS)
    ex = br.loc[br.APT.eq('LIRF')].drop(columns='APT').set_index('MVT_ID_mvt')
    parts = [pd.read_parquet(_first(S['a35'], f'{n}.parquet')).set_index('MVT_ID_mvt').v.rename('p_' + n) for n in ['set_clf', 'self_clf', 'set_clf2']]
    parts.append(full.set_index('MVT_ID_mvt').p_fallback.rename('p_context'))
    for v in parts:
        ex[v.name] = v.reindex(ex.index)
    rank = rank.merge(ex.reset_index(), on='MVT_ID_mvt', how='left', validate='one_to_one')
    for mode in ['weighted', 'plain']:
        folder = pkg / f'outer_detector_{mode}'
        rank['p_identity_' + mode] = _predict(lgb.Booster(model_file=str(folder / 'model.txt')),
                                              json.loads((folder / 'state.json').read_text()), rank)
    gates = {}
    for name in b.CONFIGS:
        folder = pkg / name
        gates[name] = _predict(lgb.Booster(model_file=str(folder / 'model.txt')), json.loads((folder / 'state.json').read_text()), rank)
    summ = json.loads((pkg / 'summary.json').read_text())
    gate = (gates['detector_absolute'] + gates['detector_residual'] + gates['detector_binary']) / 3.
    proposed = g.compose(rank, gate, summ['alpha'])
    det = rank[['MVT_ID_mvt', 'normal', 'D', 'w', 'p_identity_weighted', 'p_identity_plain']].assign(gate=gate, pred=proposed)
    return eligible, proposed, det


def compose_v45(S, allow_missing=False):
    full, missing = compose_a48u(S, allow_missing)
    eligible, proposed, det = a51_replay(full, S)
    v45 = np.maximum(full.pred_a48u.values, FLOOR)
    v45[eligible] = proposed
    full['eligible_a51'] = eligible
    full['pred_v45'] = v45
    return full, det, missing


def main_check():
    S = SOURCES['main']
    full, det, _ = compose_v45(S)
    rep = {'sources': {k: str(v) for k, v in S.items()}}
    ref = {n: pd.read_parquet(REFM / p / 'rank_detail.parquet') for n, p in
           [('A48m', 'A48m_20260922_131955'), ('A48', 'A48_20260922_131958'), ('A48f', 'A48f_20260922_132000'),
            ('A48u', 'A48u_20260922_132003'), ('A35', 'A35_20260918_181126'), ('A51', A51)]}
    v45 = pd.read_parquet(REF / 'submissions' / 'gentle-tractor_v45.parquet')
    mt = full.fpl_matched.values == 1; um = ~mt
    checks = [('A48m normal_members (matched)', full.loc[mt], 'normal_members_a36', ref['A48m'], 'normal_members'),
              ('A35 p_matched_mix', full, 'p_matched_mix', ref['A35'], 'p_matched_mix'),
              ('A35 normal_unmatched_set', full, 'normal_unmatched_set', ref['A35'], 'normal_unmatched_set'),
              ('A48f p_unmatched_mode', full, 'p_unmatched_mode', ref['A48f'], 'p_unmatched_mode'),
              ('A48f alpha', full, 'alpha', ref['A48f'], 'alpha'),
              ('A48f pred (unmatched)', full.loc[um], 'pred_a48f_unm', ref['A48f'], 'pred'),
              ('A48u adsb_unmatched_delta', full, 'adsb_unmatched_delta', ref['A48u'], 'adsb_unmatched_delta'),
              ('A48u normal_unmatched_adsb', full, 'normal_unmatched_adsb', ref['A48u'], 'normal_unmatched_adsb'),
              ('A48u pred', full, 'pred_a48u', ref['A48u'], 'pred'),
              ('A51 pred (eligible)', det, 'pred', ref['A51'], 'pred'),
              ('A51 gate (eligible)', det, 'gate', ref['A51'], 'gate')]
    rep['checks'] = []
    for label, a, col, r, rcol in checks:
        c = compare(a.MVT_ID_mvt.values, a[col].values, r.MVT_ID_mvt.values, r[rcol].values, label)
        rep['checks'].append(c); say(fmt(c))
    c = compare(full.MVT_ID_mvt.values, full.pred_v45.values, v45.MVT_ID_mvt.values, v45.TAXITIME_SEC_mvt.values, 'v45 (all rows)')
    rep['checks'].append(c); say(fmt(c))
    rep['bit_exact_v45'] = bool(c['max_abs'] == 0.0 and c['nan_mismatch'] == 0 and c['n_both_finite'] == mc.N_OLD)
    rep['max_abs_gap_rd_refit'] = float(np.nanmax(np.abs(full.gap.values)))
    p = save_report('step0_recompose_main', rep)
    say(f'bit-exact v45: {rep["bit_exact_v45"]}; report {p}')
    return rep


def hybrid_check():
    """Whole-chain training reproduction on the old ids so far: rebuilt OLD-scored members where they exist."""
    S = hybrid_sources()
    full, det, missing = compose_v45(S)
    v45 = pd.read_parquet(REF / 'submissions' / 'gentle-tractor_v45.parquet')
    rep = {'a27': str(S['a27']), 'blind_from': S['blind_from'], 'used': S.get('used', {})}
    rebuilt = sorted(k for k, v in rep['used'].items() if 'members_repro' in v)
    rep['rebuilt_members'] = rebuilt
    mt = full.fpl_matched.values == 1
    rep['v45_all'] = compare(full.MVT_ID_mvt, full.pred_v45, v45.MVT_ID_mvt, v45.TAXITIME_SEC_mvt, 'hybrid v45 vs v45 (all)')
    rep['v45_matched'] = compare(full.MVT_ID_mvt[mt], full.pred_v45[mt], v45.MVT_ID_mvt, v45.TAXITIME_SEC_mvt, 'matched rows')
    rep['v45_unmatched'] = compare(full.MVT_ID_mvt[~mt], full.pred_v45[~mt], v45.MVT_ID_mvt, v45.TAXITIME_SEC_mvt, 'unmatched rows')
    lirf = full.eligible_a51.values
    rep['v45_lirf_a51'] = compare(full.MVT_ID_mvt[lirf], full.pred_v45[lirf], v45.MVT_ID_mvt, v45.TAXITIME_SEC_mvt, 'A51 rows')
    for k in ('v45_all', 'v45_matched', 'v45_unmatched', 'v45_lirf_a51'):
        say(fmt(rep[k]))
    say(f'rebuilt inputs used: A27 block = {S["a27"]}; blind twins = {S["blind_from"]}; members = {rebuilt}')
    import time as _t
    save_report('hybrid_recompose_' + _t.strftime('%H%M'), rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['main', 'rebuild', 'hybrid'])
    ap.add_argument('--rank', choices=['new', 'old'], default='new')
    ap.add_argument('--adsb', help='folder with the v1 adsb_features/adsb_stops tables (default: the 09-26 snapshot)')
    ap.add_argument('--allow-missing', action='store_true')
    ap.add_argument('--out', help='write the composed frame here (parquet)')
    ap.add_argument('--base-dir', help='rebuild/new: folder holding R_A27, member_cache* (default runs/models)')
    ap.add_argument('--twins', help='rebuild/new: unmatched twins file (default <base-dir>/member_cache_a46/unmatched_adsb_rank_41_42_43.parquet; v73 uses the 15-seed file)')
    ap.add_argument('--tag', default='rebuild', help='file name: v45_<tag>_<stamp>.parquet')
    ap.add_argument('--write', action='store_true', help='rebuild/new only: write submissions_rebuild/v45_rebuild_<stamp>.parquet '
                    '(template order of data/submitting.parquet; refuses when any member is missing)')
    a = ap.parse_args()
    if a.mode == 'main':
        main_check(); return
    if a.mode == 'hybrid':
        hybrid_check(); return
    S = rebuild_sources(a.rank, a.adsb, a.base_dir)
    if a.twins:
        S['twins'] = Path(a.twins)
    full, det, missing = compose_v45(S, a.allow_missing)
    say(f'missing members: {missing}')
    if a.out:
        full.to_parquet(a.out, index=False); say(f'-> {a.out}')
    if a.write:
        import time as _t
        if a.rank != 'new' or missing or not a.adsb:
            raise SystemExit('--write needs --rank new, every member present and --adsb <rebuild tables>')
        tpl = pd.read_parquet(DATA / 'submitting.parquet', columns=['MVT_ID_mvt'])
        sub = tpl.merge(full[['MVT_ID_mvt', 'pred_v45']], on='MVT_ID_mvt', how='left', validate='one_to_one')
        v = sub.pred_v45.values
        if len(sub) != mc.N_FINAL or sub.MVT_ID_mvt.nunique() != mc.N_FINAL or not np.isfinite(v).all() or v.min() < FLOOR:
            raise SystemExit('final checks failed (rows / ids / finite / floor)')
        outdir = REB / 'submissions_rebuild'; outdir.mkdir(exist_ok=True)
        path = outdir / f'v45_{a.tag}_{_t.strftime("%Y%m%d_%H%M%S")}.parquet'
        pd.DataFrame({'MVT_ID_mvt': sub.MVT_ID_mvt.values, 'TAXITIME_SEC_mvt': v}).to_parquet(path, index=False)
        full.to_parquet(path.with_name(path.stem + '_detail.parquet'), index=False)
        say(f'-> {path} (not uploaded; outside submissions/)')
    ref_v45 = REF / 'submissions' / 'gentle-tractor_v45.parquet'   # optional reference (the earlier Jan/Jul v45 file)
    if ref_v45.exists():
        v45 = pd.read_parquet(ref_v45)
        c = compare(full.MVT_ID_mvt.values, full.pred_v45.values, v45.MVT_ID_mvt.values, v45.TAXITIME_SEC_mvt.values,
                    f'rebuild({a.rank}) vs v45 on old ids')
        say(fmt(c))


if __name__ == '__main__':
    main()
