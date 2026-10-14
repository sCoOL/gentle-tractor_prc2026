"""Unmatched mode model: a set-token schedule-copy classifier for NM-unmatched departures (2026-09-19 evening).

  py -3.14 scripts/114_unmatched_mode.py fit --block AB --candidates u_clf u_clf2 u_self
  py -3.14 scripts/114_unmatched_mode.py eval          # per-airport alpha on every delay (first look; superseded by rule)
  py -3.14 scripts/114_unmatched_mode.py rule          # the A37 rule -> runs/oof/bundle_A37_strict.parquet + rule JSON
  py -3.14 scripts/114_unmatched_mode.py deploy --base-package runs/models/A36_<stamp> [--submit]

Why: in A36's strict OOF the unmatched rows hold 38.7 % of the squared error. Most of it outside the 15 date bugs is
mode uncertainty: real taxis <= 1 h are predicted at 758 s RMSE although the normal expert alone is at ~311 s,
because the gate hedges toward the schedule-copy target D (= take-off - schedule). A34's gate weight W comes from the
context detector's p (LightGBM), trained on all rows. Here a neighbour-set classifier (106 ``fit_mlp_set``, task clf)
learns ``fallback`` from all fixed-wing rows of the other eight months with every NM-derived input masked as on an
unmatched row (110 ``mask_nm``), plus the true ``is_unmatched`` flag so that it can tell the two populations apart.
Arrival tokens carry the neighbours' schedule-copy flags, which is the kind of evidence the mode needs.

eval: on unmatched rows, W' = (1 - a) * W + a * p_u and p' = (1 - a) * p + a * p_u (rows outside the gate), with
``a`` per airport chosen on the other two blocks (grid 0.0 ... 1.0), everything else as A35 (normal expert, Pb).
Matched rows are A36's.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import detectors  # noqa: E402
import fallback_context as fc  # noqa: E402
import tail_experts as te  # noqa: E402
import tune_residual as tr  # noqa: E402
from common import RUNS  # noqa: E402

probe = importlib.import_module('106_member_probe')
un = importlib.import_module('110_unmatched_normal')
a35 = importlib.import_module('111_bundle_a35')

_SET = {**probe.MLP, 'd_tok': 64, 'heads': 4, 'task': 'clf', 'mask_self_nm': True}
CANDIDATES = {
    'u_clf': {**_SET, 'layers': 1, 'seed': 80},
    'u_clf2': {**_SET, 'layers': 2, 'seed': 81},
    'u_self': {**_SET, 'layers': 0, 'seed': 82, 'kinds': ()},       # no neighbour tokens (ablation / diversity)
    # u_self was the strongest (all six folds better): two more seeds for a stable mean
    'u_self_s1': {**_SET, 'layers': 0, 'seed': 83, 'kinds': ()},
    'u_self_s2': {**_SET, 'layers': 0, 'seed': 84, 'kinds': ()},
    # three more seeds for the mean (A38 probe)
    'u_self_s3': {**_SET, 'layers': 0, 'seed': 85, 'kinds': ()},
    'u_self_s4': {**_SET, 'layers': 0, 'seed': 86, 'kinds': ()},
    'u_self_s5': {**_SET, 'layers': 0, 'seed': 87, 'kinds': ()},
    # genuinely unmatched rows weighted x10 in the BCE (106 weight='unm')
    'u_self_w10': {**_SET, 'layers': 0, 'seed': 88, 'kinds': (), 'weight': 'unm', 'unm_w': 10.0},
    'u_self_w10_s1': {**_SET, 'layers': 0, 'seed': 89, 'kinds': (), 'weight': 'unm', 'unm_w': 10.0},
}
A36_OOF = RUNS / 'oof' / 'bundle_A36_strict.parquet'
GRID = np.round(np.arange(0.0, 1.0001, 0.1), 2)


def masked_block(block, data):
    """Training: every fixed-wing row of the other eight months (date bugs out), NM inputs masked; held: the block."""
    blk = tr.load_block(block, 5, '_v5tfb5rd', data=data)
    blocked = common.FOLDS[block[0]] + common.FOLDS[block[1]]
    bug = te.datebug_label(data.TAXITIME_SEC_mvt.values, data.atot_minus_sched.values)
    keep = ~data.month.isin(blocked).values & ~bug & (data.is_heli.values == 0)
    flag = lambda df: df.assign(is_unmatched=(df.fpl_matched.values == 0).astype('float32'))
    train = un.mask_nm(flag(data[keep])); held = un.mask_nm(flag(blk['held']))
    drop = set(un.NM_NAN) | set(un.NM_CONST)
    cols = [c for c in blk['cols'] if c not in drop] + ['is_unmatched']; cats = [c for c in blk['cats'] if c not in drop]
    return {**blk, 'train': train, 'held': held, 'cols': cols, 'cats': cats}


def candidate_p(name, block, data):
    cache = RUNS / 'tuning' / f'unmatched_mode_{block}_{name}.parquet'
    if cache.exists():
        return pd.read_parquet(cache)
    blk = masked_block(block, data)
    t0 = time.time()
    # fit_mlp_set scores only anchored held rows unless it is a taxi expert; the mode model scores every held row
    r, _ = probe.fit_mlp_set(blk, {**CANDIDATES[name], 'taxi_expert': True})
    res = pd.DataFrame({'MVT_ID_mvt': blk['held'].MVT_ID_mvt.values, 'p': r})
    res.to_parquet(cache, index=False)
    print(f'{name} {block}: fitted in {time.time() - t0:.0f}s', flush=True)
    return res


def fit_cmd(a):
    data = None
    for name in a.candidates:
        if data is None and not (RUNS / 'tuning' / f'unmatched_mode_{a.block}_{name}.parquet').exists():
            data, _, _ = common.load_features(5)
        candidate_p(name, a.block, data)


def compose(t, nrm, w, p):
    """A35's unmatched composition (un.compose) with gate weight ``w`` and mixture probability ``p`` swapped in."""
    D = t.atot_minus_sched.values
    valid = np.isfinite(D) & (t.is_heli.values == 0) & (np.abs(D - t.normal_pred.values) >= 60)
    mix = fc.mixture(p, D, nrm, t.is_heli.values, t.pred_rd.values)
    two = np.where(valid, np.maximum(nrm + np.clip(w, 0, 1) * (D - nrm), 30.0), mix)
    return te.combine(two, np.nan_to_num(t.pb.values), nrm, t.day_shift.values)


def eval_cmd(a):
    t = un.unmatched_table()
    um = t.fpl_matched.values == 0
    nrm = a35._cached('unmatched_normal_{block}_{name}.parquet', a35.NORMAL, 'normal').reindex(t.MVT_ID_mvt).values
    nrm = np.where(um & np.isfinite(nrm), nrm, t.normal_pred.values)
    base = compose(t, nrm, t.w.values, t.p_fallback.values)
    names = a.candidates
    pu = pd.concat([pd.concat([pd.read_parquet(RUNS / 'tuning' / f'unmatched_mode_{l}{r}_{n}.parquet') for l, r in te.stg.BLOCKS])
                    .set_index('MVT_ID_mvt').p.rename(n) for n in names], axis=1).mean(axis=1).reindex(t.MVT_ID_mvt).values
    ref = pd.read_parquet(A36_OOF)
    m36 = t[['MVT_ID_mvt']].merge(ref[['MVT_ID_mvt', 'pred']], on='MVT_ID_mvt', how='left', validate='one_to_one').pred.values
    y = t.y.values
    chk = float(np.nanmax(np.abs(base[um] - m36[um])))
    print(f'A36 unmatched rows reproduced: max gap {chk:.2e}', flush=True)
    fb = t.fallback.values.astype(int); real = um & (t.is_heli.values == 0) & ~te.datebug_label(y, t.atot_minus_sched.values)
    for lab, s in [('p context', t.p_fallback.values), ('gate W', np.clip(t.w.values, 0, 1)), ('p_u', pu)]:
        ok = real & np.isfinite(s)
        per = {x: detectors._auc(fb[ok & (t.APT.values == x)], s[ok & (t.APT.values == x)]) for x in ['LIRF', 'LFPG', 'LSZH', 'LTFM']
               if (ok & (t.APT.values == x)).sum() > 50}
        print(f'  AUC on unmatched fixed-wing rows ({lab}): {detectors._auc(fb[ok], s[ok]):.4f}  ' + ' '.join(f'{k} {v:.3f}' for k, v in per.items()))
    apts = sorted(t.APT.unique()); blocks = sorted(t.block.unique())
    pu_f = np.where(np.isfinite(pu), pu, t.p_fallback.values)

    def pred_with(alpha_by_apt, rows):
        al = np.array([alpha_by_apt.get(x, 0.0) for x in t.APT.values[rows]])
        w = (1 - al) * t.w.values[rows] + al * pu_f[rows]
        p = (1 - al) * t.p_fallback.values[rows] + al * pu_f[rows]
        return compose(t.iloc[np.where(rows)[0]], nrm[rows], w, p)

    new = base.copy(); chosen = {}
    for b in blocks:
        src = um & (t.block.values != b)
        alpha = {}
        for x in apts:
            rows = src & (t.APT.values == x)
            if rows.sum() == 0:
                alpha[x] = 0.0; continue
            sse = [np.sum((pred_with({x: g}, rows) - y[rows]) ** 2) for g in GRID]
            alpha[x] = float(GRID[int(np.argmin(sse))])
        chosen[b] = alpha
        rows = um & (t.block.values == b)
        new[rows] = pred_with(alpha, rows)
    full = np.where(um, new, m36)
    out = ref.copy()
    out['pred'] = out[['MVT_ID_mvt']].merge(pd.DataFrame({'MVT_ID_mvt': t.MVT_ID_mvt.values, 'p': full}), on='MVT_ID_mvt',
                                            how='left', validate='one_to_one').p.values
    rm = lambda p, yy: float(np.sqrt(np.mean((p - yy) ** 2)))
    print(f'alpha per block (fitted on the other two): {json.dumps(chosen)}')
    print(f'pooled: A36 {rm(ref.pred.values, ref.y.values):.4f} -> {rm(out.pred.values, out.y.values):.4f}')
    for f in sorted(ref.fold.unique()):
        s = ref.fold.values == f
        print(f'  fold {f}: {rm(out.pred.values[s], ref.y.values[s]) - rm(ref.pred.values[s], ref.y.values[s]):+.3f}')
    for x in apts:
        s = ref.APT.values == x
        print(f'  {x}: {rm(out.pred.values[s], ref.y.values[s]) - rm(ref.pred.values[s], ref.y.values[s]):+.3f}')
    path = RUNS / 'oof' / f'unmatched_mode_{"_".join(names)}_strict.parquet'
    out.to_parquet(path, index=False); print(path)
    # deployment strengths: fitted on all three blocks' unmatched rows
    deploy = {}
    for x in apts:
        rows = um & (t.APT.values == x)
        sse = [np.sum((pred_with({x: g}, rows) - y[rows]) ** 2) for g in GRID]
        deploy[x] = float(GRID[int(np.argmin(sse))])
    (RUNS / f'unmatched_mode_alpha_{"_".join(names)}.json').write_text(
        json.dumps({'candidates': names, 'cross_fitted': chosen, 'deploy': deploy}, indent=1), encoding='utf-8')
    print(f'deploy alpha: {deploy}')


H = 3600.0
DMAX_GRID = [4 * H, 5 * H, 6 * H, 7 * H, 8 * H, 9 * H, 10 * H, 12 * H]
EXCLUDE = ('LTFM',)      # its Jan/Feb/Jul/Aug rows (block AB, the 2026 test months) prefer the old gate in every variant


def rule_cmd(a):
    """The A37 rule: on unmatched rows with schedule delay D <= Dmax, W' = (1 - a) W + a p_u (p' likewise); longer
    delays keep the gate (the 8-16 h copies are where the gate is right and p_u is not). One global ``a`` and ``Dmax``,
    chosen on the other two blocks; the airports in EXCLUDE keep the gate everywhere."""
    t = un.unmatched_table()
    um = t.fpl_matched.values == 0; y = t.y.values; D = t.atot_minus_sched.values
    nrm = a35._cached('unmatched_normal_{block}_{name}.parquet', a.normal, 'normal').reindex(t.MVT_ID_mvt).values
    nrm = np.where(um & np.isfinite(nrm), nrm, t.normal_pred.values)
    names = a.candidates
    pu = pd.concat([pd.concat([pd.read_parquet(RUNS / 'tuning' / f'unmatched_mode_{l}{r}_{n}.parquet') for l, r in te.stg.BLOCKS])
                    .set_index('MVT_ID_mvt').p.rename(n) for n in names], axis=1).mean(axis=1).reindex(t.MVT_ID_mvt).values
    pu = np.where(np.isfinite(pu), pu, t.p_fallback.values)
    ref = pd.read_parquet(a.bundle)
    m36 = t[['MVT_ID_mvt']].merge(ref[['MVT_ID_mvt', 'pred']], on='MVT_ID_mvt', how='left', validate='one_to_one').pred.values
    keep_gate = np.isin(t.APT.values, list(a.exclude))

    def pred(al, rows):
        w = (1 - al) * t.w.values[rows] + al * pu[rows]; p = (1 - al) * t.p_fallback.values[rows] + al * pu[rows]
        return compose(t.iloc[np.where(rows)[0]], nrm[rows], w, p)

    def best(src):
        out = None
        for dm in a.dmax:
            for g in GRID:
                al = g * ((D[src] <= dm) & ~keep_gate[src])
                v = float(np.sum((pred(al, src) - y[src]) ** 2))
                if out is None or v < out[0]:
                    out = (v, dm, float(g))
        return out[1], out[2]

    new = m36.copy(); chosen = {}
    for b in sorted(t.block.unique()):
        dm, g = best(um & (t.block.values != b)); chosen[b] = {'dmax_h': dm / H, 'alpha': g}
        rows = um & (t.block.values == b)
        new[rows] = pred(g * ((D[rows] <= dm) & ~keep_gate[rows]), rows)
    dm, g = best(um)
    rm = lambda p_, yy: float(np.sqrt(np.mean((p_ - yy) ** 2)))
    out = ref.copy()
    out['pred'] = out[['MVT_ID_mvt']].merge(pd.DataFrame({'MVT_ID_mvt': t.MVT_ID_mvt.values, 'p': np.where(um, new, m36)}),
                                            on='MVT_ID_mvt', how='left', validate='one_to_one').p.values
    print(f'cross-fitted (Dmax h, alpha): {chosen}; deploy: Dmax {dm / H:.0f} h, alpha {g}')
    print(f'pooled: A36 {rm(ref.pred.values, ref.y.values):.4f} -> {rm(out.pred.values, out.y.values):.4f}')
    for f in sorted(ref.fold.unique()):
        s_ = ref.fold.values == f
        print(f'  fold {f}: {rm(out.pred.values[s_], ref.y.values[s_]) - rm(ref.pred.values[s_], ref.y.values[s_]):+.3f}')
    for x in sorted(ref.APT.unique()):
        s_ = ref.APT.values == x
        print(f'  {x}: {rm(out.pred.values[s_], ref.y.values[s_]) - rm(ref.pred.values[s_], ref.y.values[s_]):+.3f}')
    path = RUNS / 'oof' / f'bundle_{a.tag}_strict.parquet'
    out.to_parquet(path, index=False); print(path)
    (RUNS / f'unmatched_mode_rule_{a.tag}.json').write_text(json.dumps(
        {'candidates': names, 'exclude': list(a.exclude), 'cross_fitted': chosen, 'dmax': dm, 'alpha': g,
         'deploy': {x: (0.0 if x in a.exclude else g) for x in sorted(t.APT.unique())}}, indent=1), encoding='utf-8')


def rank_p(names):
    """The mode classifiers refitted on all 2025 fixed-wing rows (NM masked, date bugs out) and scored on 2026."""
    cache = RUNS / 'models' / 'member_cache_a37'
    cache.mkdir(parents=True, exist_ok=True)
    need = [n for n in names if not (cache / f'{n}.parquet').exists()]
    if need:
        train, rank, manifest = common.load_features(5)
        cols, cats = a35._strict_cols(manifest)
        bug = te.datebug_label(train.TAXITIME_SEC_mvt.values, train.atot_minus_sched.values)
        flag = lambda df: df.assign(is_unmatched=(df.fpl_matched.values == 0).astype('float32'))
        drop = set(un.NM_NAN) | set(un.NM_CONST)
        blk = {'train': un.mask_nm(flag(train[~bug & (train.is_heli.values == 0)])), 'held': un.mask_nm(flag(rank)),
               'block': 'deploy', 'cols': [c for c in cols if c not in drop] + ['is_unmatched'],
               'cats': [c for c in cats if c not in drop]}
        for n in need:
            t0 = time.time()
            r, _ = probe.fit_mlp_set(blk, {**CANDIDATES[n], 'taxi_expert': True})
            pd.DataFrame({'MVT_ID_mvt': rank.MVT_ID_mvt.values, 'p': r}).to_parquet(cache / f'{n}.parquet', index=False)
            print(f'{n}: rank mode model fitted in {time.time() - t0:.0f}s', flush=True)
    return pd.concat([pd.read_parquet(cache / f'{n}.parquet').set_index('MVT_ID_mvt').p.rename(n) for n in names], axis=1).mean(axis=1)


def deploy_cmd(a):
    """A36's package with the unmatched rows recomposed: W' = (1 - a) W + a p_u, p' likewise (a per airport)."""
    import submit, train_eval
    base = Path(a.base_package)
    d = pd.read_parquet(base / 'rank_detail.parquet')
    ids = d.MVT_ID_mvt.values; um = d.fpl_matched.values == 0
    rk = pd.read_parquet(common.DATA / 'features_v5_rank.parquet', columns=['MVT_ID_mvt', 'is_heli'])
    heli = d[['MVT_ID_mvt']].merge(rk, on='MVT_ID_mvt', how='left', validate='one_to_one').is_heli.values
    exp = RUNS / 'models' / 'A27_20260917_141019'
    p_old = d[['MVT_ID_mvt']].merge(pd.read_parquet(exp / 'p_fallback_rank.parquet'), on='MVT_ID_mvt', how='left',
                                     validate='one_to_one').p_fallback.values
    heli_const = float(json.loads((exp / 'M3rpa' / 'state.json').read_text(encoding='utf-8'))['heli_const'])
    D = d.atot_minus_sched.values; nrm = d.normal_unmatched_set.values

    def unmatched_pred(w, p):
        mix = fc.mixture(p, D, nrm, heli, np.full(len(d), heli_const))
        two = np.where(d.gate_valid.values, np.maximum(nrm + np.clip(w, 0, 1) * (D - nrm), 30.0), mix)
        return te.combine(two, np.nan_to_num(d.pb.values), nrm, d.day_shift.values)

    gap = float(np.max(np.abs(unmatched_pred(d.w.values, p_old)[um] - d.pred.values[um])))
    print(f'base package unmatched rows reproduced: max gap {gap:.2e}', flush=True)
    if gap > 1e-6:
        raise SystemExit('base unmatched predictions are not reproduced')
    alpha = json.loads(Path(a.alpha).read_text(encoding='utf-8'))
    pu = rank_p(alpha['candidates']).reindex(ids).values
    pu = np.where(np.isfinite(pu), pu, p_old)
    al = np.array([alpha['deploy'].get(x, 0.0) for x in d.APT.values])
    if 'dmax' in alpha:                # the A37 rule: only rows with schedule delay <= Dmax
        al = al * (np.nan_to_num(D, nan=np.inf) <= alpha['dmax'])
    new_u = unmatched_pred((1 - al) * d.w.values + al * pu, (1 - al) * p_old + al * pu)
    new = np.where(um, new_u, d.pred.values)
    stamp = time.strftime('%Y%m%d_%H%M%S'); pkg = RUNS / 'models' / f'{a.approach}_{stamp}'; pkg.mkdir(parents=True, exist_ok=True)
    d.assign(pred_base=d.pred.values, p_unmatched_mode=pu, alpha=al, pred=new).to_parquet(pkg / 'rank_detail.parquet', index=False)
    diff = new - d.pred.values
    summary = {'approach': a.approach, 'base_package': str(base), 'alpha': alpha, 'base_unmatched_reproduction_gap': gap,
               'rows_changed': int((np.abs(diff) > 1).sum()), 'rms_change_unmatched': float(np.sqrt(np.mean(diff[um] ** 2))),
               'mean_p_u_unmatched': float(np.nanmean(pu[um])), 'mean_p_old_unmatched': float(np.nanmean(p_old[um])),
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
        base_models = json.loads(log.loc[log.approach_id == 'A35', 'models_json'].iloc[-1])
        row = dict(file=path.name, ts=time.strftime('%Y-%m-%d %H:%M:%S'), team=team, version_n=version, approach_id=a.approach,
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
    f = sub.add_parser('fit'); f.add_argument('--block', required=True); f.add_argument('--candidates', nargs='+', default=list(CANDIDATES))
    e = sub.add_parser('eval'); e.add_argument('--candidates', nargs='+', default=list(CANDIDATES))
    d = sub.add_parser('deploy'); d.add_argument('--base-package', required=True); d.add_argument('--approach', default='A37')
    d.add_argument('--alpha', default=str(RUNS / 'unmatched_mode_rule_A37.json'))
    d.add_argument('--evidence', default=str(RUNS / 'evidence_A37_20260919.json')); d.add_argument('--submit', action='store_true')
    d.add_argument('--model-id', nargs='+', default=['set_unmatched_mode'])
    r = sub.add_parser('rule'); r.add_argument('--candidates', nargs='+', default=['u_self', 'u_self_s1', 'u_self_s2'])
    r.add_argument('--dmax', nargs='+', type=float, default=DMAX_GRID); r.add_argument('--exclude', nargs='*', default=list(EXCLUDE))
    r.add_argument('--tag', default='A37')
    r.add_argument('--bundle', default=str(A36_OOF),
                   help='matched bundle OOF the rule builds on (A38 = A36 members + the ADS-B member)')
    r.add_argument('--normal', nargs='+', default=list(a35.NORMAL), help='110 normal-expert candidates (mean)')
    a = p.parse_args()
    {'fit': fit_cmd, 'eval': eval_cmd, 'deploy': deploy_cmd, 'rule': rule_cmd}[a.cmd](a)


if __name__ == '__main__':
    main()
