"""A36 = A35 with the surge context (2026-09-19) in the matched-row member blend.

  # members on every block first (106), then six-fold weights for the seven members
  py -3.14 scripts/107_extra_members.py weights --members mlp+mlp_s1+mlp_s2+mlp_e20+mlp_wide mlp_set+mlp_set2+mlp_set_s1 \
      mlp_setn+mlp_setn2+mlp_setn_s1+mlp_setn_r+mlp_setn2_r+mlp_setn_s1_r lgb_sg --tag A36m --json runs/residual_weights_A36_20260919.json
  py -3.14 scripts/113_bundle_a36.py check     # A34 members + A34 weights must reproduce A35's strict OOF exactly
  py -3.14 scripts/113_bundle_a36.py strict    # -> runs/oof/bundle_A36_strict.parquet, then 97_evidence vs A35
  py -3.14 scripts/107_extra_members.py deploy ... --approach A36m   # package with the re-blended matched normal
  py -3.14 scripts/113_bundle_a36.py deploy --base-package runs/models/A36m_<stamp> [--submit]

Two new matched members: the mean of six neighbour-set networks whose tokens carry every movement's
stand x runway normal taxi and its NM taxi / taxi-in minus that normal (nbr_set.SURGE), and a LightGBM residual member
with surge_context.COLUMNS (neighbours' taxi vs normal, gaps to the k-th previous / next take-off, movement counts).
Everything else is A35: the NM-masked set normal expert on unmatched rows and ``p = 0.5 * (set classifiers + p_context)``
in the matched mixture.
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
import fallback_context as fc  # noqa: E402
import submit  # noqa: E402
import train_eval  # noqa: E402
from common import RUNS, SUBS  # noqa: E402

em = importlib.import_module('107_extra_members')
det = importlib.import_module('108_matched_detector')
a35 = importlib.import_module('111_bundle_a35')

A34_MEMBERS = ['mlp+mlp_s1+mlp_s2+mlp_e20+mlp_wide', 'mlp_set+mlp_set2+mlp_set_s1']
A34_WEIGHTS = RUNS / 'residual_weights_A34_20260918.json'
A36_MEMBERS = A34_MEMBERS + ['mlp_setn+mlp_setn2+mlp_setn_s1+mlp_setn_r+mlp_setn2_r+mlp_setn_s1_r', 'lgb_sg']
A36_WEIGHTS = RUNS / 'residual_weights_A36_20260919.json'
A35_OOF = RUNS / 'oof' / 'bundle_A35_strict.parquet'
A34_PKG = a35.A34_PKG
A35_PKG = RUNS / 'models' / 'A35_20260918_181126'


def strict_pred(members, weights_file):
    """A35's strict OOF with the matched rows recomposed from ``members`` under cross-fitted per-airport weights."""
    t, R = em.strict_table(members)
    rep = json.loads(Path(weights_file).read_text(encoding='utf-8'))
    w = rep['cross_fitted']
    # the weights may be fitted per airport (through A42) or per airport x ADS-B coverage cell; label rows the
    # same way 107 did, or the blend applies weights fitted for a different population
    cell = em.cell_labels(t, rep.get('cells', 'apt'), rep.get('cell_min', 3000))
    normal = np.full(len(t), np.nan)
    for blk, wb in w.items():
        m = t.block.values == blk
        wm = np.array([wb[x] for x in cell[m]])
        normal[m] = np.where(t.has.values[m], t.atot_minus_aobt3.values[m] + (R[m] * wm).sum(axis=1), t.n_rd.values[m])
    pc = a35._cached('detector_probe_{block}_{name}.parquet', a35.CLF, 'p').reindex(t.MVT_ID_mvt).values
    p = np.where(np.isfinite(pc), 0.5 * (pc + t.p_fallback.values), t.p_fallback.values)
    pm = fc.mixture(p, t.atot_minus_sched.values, normal, t.is_heli.values, t.pred_rd.values)
    ref = pd.read_parquet(A35_OOF)
    new = ref.merge(pd.DataFrame({'MVT_ID_mvt': t.MVT_ID_mvt.values, 'pm': pm}), on='MVT_ID_mvt', how='left', validate='one_to_one')
    mt = new.fpl_matched.values == 1
    if new.pm.values[mt].size and not np.isfinite(new.pm.values[mt]).all():
        raise SystemExit('matched rows missing from the member table')
    new['pred'] = np.where(mt, new.pm.values, new.pred.values)
    return ref, new.drop(columns=['pm']), mt


def check_cmd(a):
    ref, new, mt = strict_pred(A34_MEMBERS, A34_WEIGHTS)
    gap = float(np.max(np.abs(new.pred.values - ref.pred.values)))
    print(f'A35 strict OOF from A34 members + weights: max gap {gap:.3e} (matched rows {mt.sum()})')
    if gap > 1e-6:
        raise SystemExit('A35 strict OOF is not reproduced')


def strict_cmd(a):
    ref, new, mt = strict_pred(A36_MEMBERS + list(a.extra_members), a.weights)
    y = ref.y.values
    out = RUNS / 'oof' / f'bundle_{a.tag}_strict.parquet'
    new.to_parquet(out, index=False)
    print(f'{a.tag} strict pooled {common.rmse(new.pred.values, y):.4f} (A35 {common.rmse(ref.pred.values, y):.4f}); '
          f'matched {common.rmse(new.pred.values[mt], y[mt]):.2f} (A35 {common.rmse(ref.pred.values[mt], y[mt]):.2f}); {out}')
    for f in sorted(ref.fold.unique()):
        m = ref.fold.values == f
        print(f'  fold {f}: {common.rmse(new.pred.values[m], y[m]) - common.rmse(ref.pred.values[m], y[m]):+.3f}')
    for apt in sorted(ref.APT.unique()):
        m = ref.APT.values == apt
        print(f'  {apt}: {common.rmse(new.pred.values[m], y[m]) - common.rmse(ref.pred.values[m], y[m]):+.3f}')


def deploy_cmd(a):
    """A35's rank predictions with the matched normal taken from a 107 package built with the A36 members."""
    t0 = time.time()
    base = Path(a.base_package)
    d = pd.read_parquet(base / 'rank_detail.parquet')
    d34 = pd.read_parquet(A34_PKG / 'rank_detail.parquet')
    d35 = pd.read_parquet(A35_PKG / 'rank_detail.parquet')
    if not (np.array_equal(d.MVT_ID_mvt.values, d35.MVT_ID_mvt.values) and np.array_equal(d34.MVT_ID_mvt.values, d35.MVT_ID_mvt.values)):
        raise SystemExit('package row orders differ')
    mt = d35.fpl_matched.values == 1
    # the 107 package changes matched rows only: its unmatched rows must be A34's (the base A35 builds on)
    gap_u = float(np.max(np.abs(d.pred.values[~mt] - d34.pred.values[~mt])))
    rk = pd.read_parquet(common.DATA / 'features_v5_rank.parquet', columns=['MVT_ID_mvt', 'is_heli'])
    heli = d35[['MVT_ID_mvt']].merge(rk, on='MVT_ID_mvt', how='left', validate='one_to_one').is_heli.values
    exp = RUNS / 'models' / 'A27_20260917_141019'
    heli_const = float(json.loads((exp / 'M3rpa' / 'state.json').read_text(encoding='utf-8'))['heli_const'])
    D = d35.atot_minus_sched.values

    def matched_pred(normal, p):
        pp = np.clip(p, 0, 1); ok = np.isfinite(D) & (D > 0) & np.isfinite(pp)
        mix = np.where(ok, pp * D + (1 - pp) * normal, normal)
        return np.maximum(np.where(heli == 1, heli_const, mix), train_eval.FLOOR)

    # A35 check: its matched rows are matched_pred(A34 normal, p_matched_mix)
    gap35 = float(np.max(np.abs(matched_pred(d34.normal_members.values, d35.p_matched_mix.values)[mt] - d35.pred.values[mt])))
    print(f'unmatched rows equal A34: {gap_u:.2e}; A35 matched reproduction: {gap35:.2e}', flush=True)
    if gap_u > 1e-6 or gap35 > 1e-6:
        raise SystemExit('reproduction failed')
    new = np.where(mt, matched_pred(d.normal_members.values, d35.p_matched_mix.values), d35.pred.values)
    stamp = time.strftime('%Y%m%d_%H%M%S'); pkg = RUNS / 'models' / f'{a.approach}_{stamp}'; pkg.mkdir(parents=True, exist_ok=True)
    d35.assign(pred_a35=d35.pred.values, normal_members_a36=d.normal_members.values, pred=new).to_parquet(pkg / 'rank_detail.parquet', index=False)
    diff = new - d35.pred.values
    summary = {'approach': a.approach, 'base_package': str(base), 'a35_package': str(A35_PKG), 'members': A36_MEMBERS,
               'reproduction': {'unmatched_vs_a34': gap_u, 'a35_matched': gap35}, 'rows_changed': int((np.abs(diff) > 1).sum()),
               'rms_change': float(np.sqrt(np.mean(diff ** 2))), 'rms_change_matched': float(np.sqrt(np.mean(diff[mt] ** 2))),
               'rows': int(len(new)), 'finite': bool(np.isfinite(new).all()), 'min_pred': float(new.min()),
               'seconds': round(time.time() - t0, 1)}
    (pkg / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1), flush=True)
    if a.submit:
        ev = json.loads(Path(a.evidence).read_text(encoding='utf-8'))
        if not ev.get('passes'):
            raise SystemExit('evidence does not pass the noise guard; no submission written')
        team = submit.team_name(); version = submit.next_version(team)
        path = submit.write_submission(team, version, d35.MVT_ID_mvt.values, new)
        desc = (f"{ev['summary']}; strict pooled {ev['pooled']:.2f} ({ev['delta_vs_ref']:+.2f} vs {ev['reference']}), "
                f"worst fold {ev['worst_fold_vs_ref']:+.2f}")
        log = pd.read_csv(SUBS / 'submission_log.csv')
        base_file = (A35_PKG / 'submission.txt').read_text(encoding='utf-8').strip()
        base_models = json.loads(log.loc[log.file == base_file, 'models_json'].iloc[-1])
        row = dict(file=path.name, ts=time.strftime('%Y-%m-%d %H:%M:%S'), team=team, version_n=version, approach_id=a.approach,
                   feature_version=5, models_json=json.dumps(base_models + ['surge_set_member', 'surge_lgb_member']),
                   foldA_rmse=ev['fold_A'], foldB_rmse=ev['fold_B'], result_rmse='', result_rank='', decision='candidate',
                   notes=desc)
        pd.DataFrame([row]).to_csv(SUBS / 'submission_log.csv', mode='a', index=False, header=False, lineterminator='\n')
        submit.append_manifest([f'# Candidate {a.approach} {time.strftime("%Y-%m-%d %H:%M")} - team {team}', '',
                                '| file | approach | description | strict Fold A | strict Fold B | pooled six-pair |', '|---|---|---|---:|---:|---:|',
                                f"| {path.name} | {a.approach} | {desc} | {ev['fold_A']:.2f} | {ev['fold_B']:.2f} | {ev['pooled']:.2f} |"])
        (pkg / 'submission.txt').write_text(path.name, encoding='utf-8')
        print(f'{path.name} <- {a.approach}: {desc}\npackage: {pkg}')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('check')
    s = sub.add_parser('strict'); s.add_argument('--weights', default=str(A36_WEIGHTS))
    s.add_argument('--extra-members', nargs='*', default=[], help="members beyond A36's, e.g. lgb_ab (ADS-B, §0x)")
    s.add_argument('--tag', default='A36', help='names runs/oof/bundle_<tag>_strict.parquet')
    d = sub.add_parser('deploy'); d.add_argument('--base-package', required=True); d.add_argument('--approach', default='A36')
    d.add_argument('--evidence', default=str(RUNS / 'evidence_A36_20260919.json')); d.add_argument('--submit', action='store_true')
    a = p.parse_args()
    {'check': check_cmd, 'strict': strict_cmd, 'deploy': deploy_cmd}[a.cmd](a)


if __name__ == '__main__':
    main()
