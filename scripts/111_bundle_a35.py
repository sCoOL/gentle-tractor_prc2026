"""A35 = A34 + a set-encoder normal expert for NM-unmatched rows + a set-classifier mix in the matched copy probability.

  py -3.14 scripts/111_bundle_a35.py strict        # six-fold strict OOF from the cached block probes, then 97_evidence
  py -3.14 scripts/111_bundle_a35.py deploy [--submit]

Unmatched rows (110_unmatched_normal.py): ``normal`` = mean of the three NM-masked set experts (nm_masked_set,
nm_masked_self, nm_masked_set2), swapped into A34's composition with A34's gate weight W and date-bug Pb unchanged.
Matched rows (108_matched_detector.py): ``p = 0.5 * (mean(set_clf, self_clf, set_clf2) + p_context)`` in A34's matched
mixture with A34's member blend unchanged.
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
import strict_tail_gate as stg  # noqa: E402
import submit  # noqa: E402
import tail_experts as te  # noqa: E402
import train_eval  # noqa: E402
from common import RUNS, SUBS  # noqa: E402

un = importlib.import_module('110_unmatched_normal')
det = importlib.import_module('108_matched_detector')
probe = importlib.import_module('106_member_probe')
NORMAL = ['nm_masked_set', 'nm_masked_self', 'nm_masked_set2']
CLF = ['set_clf', 'self_clf', 'set_clf2']
A34_PKG = RUNS / 'models' / 'A34_20260918_160803'


def _cached(pattern, names, col):
    parts = []
    for name in names:
        res = pd.concat([pd.read_parquet(RUNS / 'tuning' / pattern.format(block=f'{l}{r}', name=name)) for l, r in stg.BLOCKS],
                        ignore_index=True)
        parts.append(res.set_index('MVT_ID_mvt')[col].rename(name))
    return pd.concat(parts, axis=1).mean(axis=1)


def strict_cmd(a):
    t = un.unmatched_table()
    m34 = det.a34_table()[['MVT_ID_mvt', 'normal34']]
    t = t.merge(m34, on='MVT_ID_mvt', how='left', validate='one_to_one')
    um = t.fpl_matched.values == 0
    nrm = _cached('unmatched_normal_{block}_{name}.parquet', NORMAL, 'normal').reindex(t.MVT_ID_mvt).values
    nrm = np.where(um & np.isfinite(nrm), nrm, t.normal_pred.values)
    pu = un.compose(t, nrm)
    pc = _cached('detector_probe_{block}_{name}.parquet', CLF, 'p').reindex(t.MVT_ID_mvt).values
    p = np.where(np.isfinite(pc), 0.5 * (pc + t.p_fallback.values), t.p_fallback.values)
    pm = fc.mixture(p, t.atot_minus_sched.values, t.normal34.values, t.is_heli.values, t.pred_rd.values)
    ref = pd.read_parquet(RUNS / 'oof' / 'extra_members_A34_strict.parquet')
    new = ref.merge(pd.DataFrame({'MVT_ID_mvt': t.MVT_ID_mvt.values, 'pu': pu, 'pm': pm}), on='MVT_ID_mvt', how='left', validate='one_to_one')
    mt = new.fpl_matched.values == 1
    new['pred'] = np.where(mt, new.pm.values, new.pu.values)
    for part, rows in [('unmatched only', ~mt), ('matched only', mt)]:
        alt = np.where(rows, new.pred.values, ref.pred.values)
        print(f'{part}: pooled {common.rmse(alt, ref.y.values):.4f}', flush=True)
    out = RUNS / 'oof' / 'bundle_A35_strict.parquet'
    new.drop(columns=['pu', 'pm']).to_parquet(out, index=False)
    print(f'A35 strict pooled {common.rmse(new.pred.values, new.y.values):.4f} (A34 {common.rmse(ref.pred.values, ref.y.values):.4f}); {out}')


def _strict_cols(manifest):
    target_cols = set(manifest['groups'].get('G8', [])) | stg.TARGET_ROUTE | stg.TARGET_G9
    cols = [c for c in train_eval.feature_columns(manifest, list(manifest['groups'])) if c not in target_cols]
    return cols, train_eval._cats(manifest, cols)


def rank_members():
    """The six networks refitted on all 2025 rows (validated target-free columns) and scored on the ranking file."""
    cache = RUNS / 'models' / 'member_cache_a35'
    cache.mkdir(parents=True, exist_ok=True)
    need = [n for n in NORMAL + CLF if not (cache / f'{n}.parquet').exists()]
    if need:
        train, rank, manifest = common.load_features(5)
        cols, cats = _strict_cols(manifest)
        bug = te.datebug_label(train.TAXITIME_SEC_mvt.values, train.atot_minus_sched.values)
        fixed = (~bug) & (train.is_heli.values == 0)
        drop = set(un.NM_NAN) | set(un.NM_CONST)
        for name in need:
            t0 = time.time()
            if name in NORMAL:
                blk = {'train': un.mask_nm(train[fixed & (train.fallback.values == 0)]), 'held': un.mask_nm(rank), 'block': 'deploy',
                       'cols': [c for c in cols if c not in drop], 'cats': [c for c in cats if c not in drop]}
                r, _ = probe.fit_mlp_set(blk, un.CANDIDATES[name])
            else:
                blk = {'train': train[fixed & np.isfinite(train.atot_minus_aobt3.values)], 'held': rank, 'block': 'deploy',
                       'cols': cols, 'cats': cats}
                r, _ = probe.fit_mlp_set(blk, det.CANDIDATES[name])
            pd.DataFrame({'MVT_ID_mvt': rank.MVT_ID_mvt.values, 'v': r}).to_parquet(cache / f'{name}.parquet', index=False)
            print(f'{name}: rank member fitted in {time.time() - t0:.0f}s', flush=True)

    def get(names):
        return pd.concat([pd.read_parquet(cache / f'{n}.parquet').set_index('MVT_ID_mvt').v.rename(n) for n in names], axis=1)
    return get(NORMAL), get(CLF)


def deploy_cmd(a):
    t0 = time.time()
    d = pd.read_parquet(A34_PKG / 'rank_detail.parquet')
    ids = d.MVT_ID_mvt.values
    rk = pd.read_parquet(common.DATA / 'features_v5_rank.parquet', columns=['MVT_ID_mvt', 'is_heli'])
    d = d.merge(rk, on='MVT_ID_mvt', how='left', validate='one_to_one')
    exp = RUNS / 'models' / 'A27_20260917_141019'
    p_old = d[['MVT_ID_mvt']].merge(pd.read_parquet(exp / 'p_fallback_rank.parquet'), on='MVT_ID_mvt', how='left',
                                     validate='one_to_one').p_fallback.values
    heli_const = float(json.loads((exp / 'M3rpa' / 'state.json').read_text(encoding='utf-8'))['heli_const'])
    D = d.atot_minus_sched.values; heli = d.is_heli.values; mt = d.fpl_matched.values == 1

    def matched_pred(normal, p):
        pp = np.clip(p, 0, 1); ok = np.isfinite(D) & (D > 0) & np.isfinite(pp)
        mix = np.where(ok, pp * D + (1 - pp) * normal, normal)
        return np.maximum(np.where(heli == 1, heli_const, mix), train_eval.FLOOR)

    def unmatched_pred(normal):
        mix = fc.mixture(p_old, D, normal, heli, np.full(len(d), heli_const))
        two = np.where(d.gate_valid.values, np.maximum(normal + np.clip(d.w.values, 0, 1) * (D - normal), 30.0), mix)
        return te.combine(two, np.nan_to_num(d.pb.values), normal, d.day_shift.values)

    gap_m = float(np.max(np.abs(matched_pred(d.normal_members.values, p_old)[mt] - d.pred.values[mt])))
    gap_u = float(np.max(np.abs(unmatched_pred(d.normal_pred.values)[~mt] - d.pred.values[~mt])))
    print(f'A34 reproduction: matched {gap_m:.2e}, unmatched {gap_u:.2e}', flush=True)
    if gap_m > 1e-6 or gap_u > 1e-6:
        raise SystemExit('A34 rank predictions are not reproduced')
    if a.check_only:
        return
    nrm, clf = rank_members()
    n_new = nrm.mean(axis=1).reindex(ids).values; c_new = clf.mean(axis=1).reindex(ids).values
    n_new = np.where(~mt & np.isfinite(n_new), n_new, d.normal_pred.values)
    p_new = np.where(np.isfinite(c_new), 0.5 * (c_new + p_old), p_old)
    new = np.where(mt, matched_pred(d.normal_members.values, p_new), unmatched_pred(n_new))
    stamp = time.strftime('%Y%m%d_%H%M%S'); pkg = RUNS / 'models' / f'A35_{stamp}'; pkg.mkdir(parents=True, exist_ok=True)
    d.drop(columns=['is_heli']).assign(pred_a34=d.pred.values, normal_unmatched_set=n_new, p_matched_mix=p_new, pred=new) \
     .to_parquet(pkg / 'rank_detail.parquet', index=False)
    diff = new - d.pred.values
    summary = {'approach': 'A35', 'base_package': str(A34_PKG), 'normal_members': NORMAL, 'clf_members': CLF,
               'a34_reproduction_gap': {'matched': gap_m, 'unmatched': gap_u}, 'rows_changed': int((np.abs(diff) > 1).sum()),
               'rms_change': float(np.sqrt(np.mean(diff ** 2))), 'rms_change_unmatched': float(np.sqrt(np.mean(diff[~mt] ** 2))),
               'rms_change_matched': float(np.sqrt(np.mean(diff[mt] ** 2))), 'finite': bool(np.isfinite(new).all()),
               'min_pred': float(new.min()), 'seconds': round(time.time() - t0, 1)}
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
        log = pd.read_csv(SUBS / 'submission_log.csv')
        base_file = (A34_PKG / 'submission.txt').read_text(encoding='utf-8').strip()
        base_models = json.loads(log.loc[log.file == base_file, 'models_json'].iloc[-1])
        row = dict(file=path.name, ts=time.strftime('%Y-%m-%d %H:%M:%S'), team=team, version_n=version, approach_id='A35', feature_version=5,
                   models_json=json.dumps(base_models + ['set_unmatched_normal', 'set_copy_classifier']), foldA_rmse=ev['fold_A'],
                   foldB_rmse=ev['fold_B'], result_rmse='', result_rank='', decision='candidate', notes=desc)
        pd.DataFrame([row]).to_csv(SUBS / 'submission_log.csv', mode='a', index=False, header=False, lineterminator='\n')
        submit.append_manifest([f'# Candidate A35 {time.strftime("%Y-%m-%d %H:%M")} - team {team}', '',
                                '| file | approach | description | strict Fold A | strict Fold B | pooled six-pair |', '|---|---|---|---:|---:|---:|',
                                f"| {path.name} | A35 | {desc} | {ev['fold_A']:.2f} | {ev['fold_B']:.2f} | {ev['pooled']:.2f} |"])
        (pkg / 'submission.txt').write_text(path.name, encoding='utf-8')
        print(f'{path.name} <- A35: {desc}\npackage: {pkg}')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('strict')
    d = sub.add_parser('deploy'); d.add_argument('--evidence', default=str(RUNS / 'evidence_A35_20260918.json'))
    d.add_argument('--submit', action='store_true'); d.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    strict_cmd(a) if a.cmd == 'strict' else deploy_cmd(a)


if __name__ == '__main__':
    main()
