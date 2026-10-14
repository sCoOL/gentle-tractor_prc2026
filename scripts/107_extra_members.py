"""Add residual members of another kind (106_member_probe.py candidates) to A31's per-airport matched-row blend.

  # strict members: run 106 on every block first (it caches runs/tuning/member_probe_<block>_<name>.parquet)
  py -3.14 scripts/107_extra_members.py weights --members lgb_extra --tag A32 --json runs/residual_weights_A32_20260918.json
  py -3.14 scripts/97_evidence.py --candidate runs/oof/extra_members_A32_strict.parquet \
      --reference runs/oof/residual_weights_v5tfb5rw_strict_20260917.parquet --reference-name A31 --summary "..." \
      --out runs/evidence_A32_20260918.json
  py -3.14 scripts/107_extra_members.py deploy --members lgb_extra --approach A32 --weights runs/residual_weights_A32_20260918.json \
      --base-package runs/models/A31_20260918_075753 --expert-package runs/models/A27_20260917_141019 \
      --members-file runs/models/A30_20260917_194203/residual_members_rank.parquet --evidence runs/evidence_A32_20260918.json --submit

weights: the three existing members (104's recovery from the ``_v5tfb5`` frames) plus the cached candidates. Simplex
weights (0.05 grid) per airport are fitted on two strict blocks and applied to the third; the deployment weights are
fitted on all six folds. Only NM-matched fixed-wing rows change, so the candidate strict OOF is A31's strict OOF with
those rows replaced by ``p * D + (1 - p) * normal``.

deploy: each candidate member is refitted on all clean 2025 matched rows with the expert package's deployment
columns, exactly like the existing members. The base package's matched predictions must be reproduced from the saved
members first, and the three-member refit gap of the saved package is carried over as in 105.
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

rw = importlib.import_module('104_residual_weights')
probe = importlib.import_module('106_member_probe')
A31_OOF = RUNS / 'oof' / 'residual_weights_v5tfb5rw_strict_20260917.parquet'


def strict_table(members, blocks=None):
    """Existing members plus one column per member spec; a spec 'a+b+c' is the mean of those candidates' residuals.

    ``blocks`` (e.g. ['AB']) keeps only those strict blocks, for an early read on candidates that 106 has not run
    everywhere yet; the table is then restricted to them and nothing else may be deployed from it.
    """
    t = rw.member_table(5, '_v5tfb5')
    if blocks is not None:
        t = t[t.block.isin(blocks).values].reset_index(drop=True)
    want = [(l, r) for l, r in stg.BLOCKS if blocks is None or f'{l}{r}' in blocks]
    cols = [t.rg0.values, t.rg1.values, t.ra.values]
    for spec in members:
        rs = []
        for name in spec.split('+'):
            parts = [pd.read_parquet(RUNS / 'tuning' / f'member_probe_{l}{r}_{name}.parquet') for l, r in want]
            res = pd.concat(parts, ignore_index=True)
            r = t[['MVT_ID_mvt']].merge(res, on='MVT_ID_mvt', how='left', validate='one_to_one').r.values
            if not np.isfinite(r[t.has.values]).all():
                raise SystemExit(f'{name}: strict residuals missing on matched rows (run 106 on every block)')
            rs.append(np.where(t.has.values, r, 0.0))
        cols.append(np.mean(rs, axis=0))
    return t, np.column_stack(cols)



def cell_labels(frame, cells='apt', cell_min=3000):
    """The weight-fitting cell of every row: the airport (every approach through A42), or the airport split on
    whether ADS-B measured this aircraft's push-back.

    ``frame`` needs APT and, for 'adsb', MVT_ID_mvt so adsb_context can be joined. Used by ``weights``, by
    113's bundle and by ``deploy``, so all three label a row identically -- a mismatch there would silently
    deploy weights fitted for a different population.
    """
    if cells == 'apt':
        return frame.APT.values.astype(str)
    if cells not in ('adsb', 'adsb3'):
        raise SystemExit(f'unknown cell scheme {cells!r}')
    import adsb_context
    f = frame[['MVT_ID_mvt', 'APT']].merge(adsb_context.table()[['MVT_ID_mvt', 'adsb_at_stand', 'adsb_censored']],
                                           on='MVT_ID_mvt', how='left', validate='one_to_one')
    if 'has' in frame.columns:
        f['has'] = frame.has.values
    else:                                   # the 2026 ranking rows: anchored means there is an AOBT_3 to reside on
        f['has'] = np.isfinite(frame.atot_minus_aobt3.values.astype(float))
    return probe.adsb_cells(f, cell_min, three=(cells == 'adsb3'))


def weights_cmd(a):
    t, R = strict_table(a.members)
    k = R.shape[1]
    step = getattr(a, 'grid_step', 0.05)
    exact = getattr(a, 'exact', False); lam = getattr(a, 'ridge', 0.0)
    cells = getattr(a, 'cells', 'apt'); cell_min = getattr(a, 'cell_min', 3000)
    # past nine members the 0.05 grid is 10 M+ points; the exact solve does not need it, so it is skipped there
    grid = None if (exact and k > 9) else probe.simplex_grid(k, step)
    fit = t.has.values; target = t.y.values - t.mix_c.values
    cell = cell_labels(t, cells, cell_min)
    keys = sorted(set(cell))
    solve = (lambda *args: probe.fit_weights_exact(*args, lam=lam)) if exact else probe.fit_weights

    def fit_on(rows):
        return {x: solve(R[rows & (cell == x)], t.mix_m.values[rows & (cell == x)],
                         target[rows & (cell == x)], grid) for x in keys}

    nd = 6 if exact else 2          # the deploy entry is what 'deploy' reads back, so exact weights keep their digits
    normal = np.full(len(t), np.nan)
    report = {'members': ['r_global', 'r_global_seed1', 'r_airport'] + list(a.members),
              'grid_step': step, 'exact': exact, 'ridge': lam, 'cells': cells, 'cell_min': cell_min,
              'cross_fitted': {}}
    for blk in t.block.unique():
        w = fit_on(fit & (t.block.values != blk))
        report['cross_fitted'][blk] = {x: [round(float(v), nd) for v in w[x]] for x in keys}
        m = t.block.values == blk
        wm = np.array([w[x] for x in cell[m]])
        normal[m] = np.where(t.has.values[m], t.atot_minus_aobt3.values[m] + (R[m] * wm).sum(axis=1), t.n_rd.values[m])
    report['deploy'] = {x: [round(float(v), nd) for v in w] for x, w in fit_on(fit).items()}
    pred_m = fc.mixture(t.p_fallback.values, t.atot_minus_sched.values, normal, t.is_heli.values, t.pred_rd.values)
    ref = pd.read_parquet(A31_OOF)
    new = ref.merge(pd.DataFrame({'MVT_ID_mvt': t.MVT_ID_mvt.values, 'pm': pred_m, 'mt': t.fpl_matched.values}),
                    on='MVT_ID_mvt', how='left', validate='one_to_one')
    if new.pm.isna().any():
        raise SystemExit('strict rows missing')
    mt = new.fpl_matched.values == 1
    new['pred'] = np.where(mt, new.pm.values, new.pred.values)
    out = RUNS / 'oof' / f'extra_members_{a.tag}_strict.parquet'
    new.drop(columns=['pm', 'mt']).to_parquet(out, index=False)
    y = new.y.values
    report['matched_rmse_a31'] = common.rmse(ref.pred.values[mt], y[mt]); report['matched_rmse_new'] = common.rmse(new.pred.values[mt], y[mt])
    report['pooled_a31'] = common.rmse(ref.pred.values, y); report['pooled_new'] = common.rmse(new.pred.values, y)
    report['strict_oof'] = str(out)
    Path(a.json).write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'cross_fitted'}, indent=1))


_DATA = {}


def _deploy_rows(fv):
    if fv not in _DATA:
        train, rank, _ = common.load_features(fv)
        bug = te.training_exclusion(train.TAXITIME_SEC_mvt.values, train.atot_minus_sched.values, 'datebug')
        clean = train_eval._clean(train[~bug])
        _DATA[fv] = (clean[np.isfinite(clean.atot_minus_aobt3.values)], rank)
    return _DATA[fv]


def fit_rank_member(name, fv, cols, cats):
    """Refit one candidate on all clean 2025 matched rows with the deployment columns (like rv.rank_blend_members)."""
    kind, params, rounds, _ = probe.CANDIDATES[name]
    cm, rank = _deploy_rows(fv)
    a3 = rank.atot_minus_aobt3.values.astype(float); has = np.isfinite(a3)
    blk = {'train': cm, 'held': rank, 'cols': cols, 'cats': cats, 'block': 'deploy'}
    if kind.endswith('_plus') or kind == 'lgb_plus_apt':   # surge_context columns (2026 rows get the December lead-in)
        blk = probe.with_surge(blk)
        cm, rank, cols = blk['train'], blk['held'], blk['cols']
        kind = {'lgb_plus': 'lgb', 'mlp_plus': 'mlp', 'mlp_set_plus': 'mlp_set'}.get(kind, kind)
    if kind.startswith(('lgb_adsb', 'lgb_sg_adsb', 'mlp_set_adsb')):          # adsb_context off-block columns
        blk = probe.with_adsb(probe.with_surge(blk) if kind.startswith('lgb_sg_adsb') else blk)
        if kind.endswith('_nbr'):                                     # + the measured surface context (121 neighbours)
            blk = probe.with_nbr(blk)
        if kind.endswith('_geom'):                                    # + the observed track geometry (122 geometry)
            blk = probe.with_geom(blk)
        if kind.endswith('_st') or kind.endswith('_st_ty'):           # + the ground-phase stop sequence (124 stops)
            blk = probe.with_stops(blk)
        if kind.endswith('_ty'):                                      # + the X-Plane stand type (scripts/stand_type.py)
            blk = probe.with_stand_type(blk)
        if '_queue' in kind:              # + the G12 queue / pushback cohort (features v6); '_queue_apt' ends in _apt
            blk = probe.with_queue(blk)
        if kind.endswith('_debias'):      # + the ADS-B anchor corrected by the arrival-measured airport-month bias
            blk = probe.only_trust(probe.with_adsb_debias(blk))
        if kind.endswith('_trust'):       # train only where ADS-B measured the push-back
            blk = probe.only_trust(blk)
        if kind.endswith('_cens'):        # train only where ADS-B saw the aircraft already rolling
            blk = probe.only_cens(blk)
        cm, rank, cols = blk['train'], blk['held'], blk['cols']
        kind = 'mlp_set' if kind.startswith('mlp_set') else ('lgb_plus_apt' if kind.endswith('_apt') else 'lgb')
    if kind in ('mlp', 'mlp_set'):
        fit = probe.fit_mlp if kind == 'mlp' else probe.fit_mlp_set     # mlp_set scores the 2026 ranking tokens
        r, _ = fit(blk, params)
        return r
    if kind == 'lgb_plus_apt':
        r, _ = probe.fit_lightgbm_per_airport(blk, params, rounds)
        return r
    if kind not in ('lgb', 'lgb_direct'):
        raise SystemExit(f'{name}: deployment is implemented for LightGBM and MLP members only')
    X = train_eval.prep(cm, cols, cats)
    y = cm.TAXITIME_SEC_mvt.values - (0.0 if kind == 'lgb_direct' else cm.atot_minus_aobt3.values)
    m = train_eval.fit_lgb(X, y, cats, dict(params), rounds, f'member_{name}')
    r = np.full(len(rank), np.nan)
    r[has] = m.predict(train_eval.prep(rank[has], cols, cats, train_eval._cats_of(X, cats))) - (a3[has] if kind == 'lgb_direct' else 0.0)
    return r


def rank_member(spec, fv, cols, cats, cache_dir, reuse):
    """One deployment member column for a spec 'a+b+c' (mean of the candidates), with a per-candidate cache."""
    cm, rank = _deploy_rows(fv)
    rs = []
    for name in spec.split('+'):
        cache = Path(cache_dir) / f'member_rank_{name}.parquet'
        if reuse and cache.exists():
            r = rank[['MVT_ID_mvt']].merge(pd.read_parquet(cache), on='MVT_ID_mvt', how='left', validate='one_to_one')[f'r_{name}'].values
        else:
            r = fit_rank_member(name, fv, cols, cats)
            cache.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'MVT_ID_mvt': rank.MVT_ID_mvt.values, f'r_{name}': r}).to_parquet(cache, index=False)
        rs.append(r)
    return pd.DataFrame({'MVT_ID_mvt': rank.MVT_ID_mvt.values, f'r_{spec}': np.mean(rs, axis=0)})


def deploy_cmd(a):
    t0 = time.time()
    base_pkg, exp_pkg = Path(a.base_package), Path(a.expert_package)
    state = json.loads((exp_pkg / 'M3rpa' / 'state.json').read_text(encoding='utf-8'))
    fv = int(state['feature_version'])
    mem = pd.read_parquet(a.members_file)
    rk = pd.read_parquet(common.DATA / f'features_v{fv}_rank.parquet', columns=['MVT_ID_mvt', 'is_heli', 'APT', 'atot_minus_sched'])
    mem = mem.merge(rk, on='MVT_ID_mvt', how='left', validate='one_to_one')
    cols, cats = state['cols'], state['cats']
    cache_dir = RUNS / 'models' / 'member_cache'
    if a.member_cols == 'strict':      # exactly the target-free columns the strict members were validated with
        manifest = common.read_manifest(fv)
        target_cols = set(manifest['groups'].get('G8', [])) | stg.TARGET_ROUTE | stg.TARGET_G9
        cols = [c for c in train_eval.feature_columns(manifest, list(manifest['groups'])) if c not in target_cols]
        cats = train_eval._cats(manifest, cols)
        cache_dir = RUNS / 'models' / 'member_cache_strict'
    for spec in a.members:
        extra = rank_member(spec, fv, cols, cats, cache_dir, a.reuse_rank_members)
        mem = mem.merge(extra, on='MVT_ID_mvt', how='left', validate='one_to_one')
        print(f'{spec}: rank member ready [{time.time() - t0:.0f}s]', flush=True)
    ids = mem.MVT_ID_mvt.values; a3 = mem.aobt3_anchor.values; is_heli = mem.is_heli.values; apt = mem.APT.values
    order = pd.DataFrame({'MVT_ID_mvt': ids})
    plain = order.merge(pd.read_parquet(exp_pkg / 'normal_rank.parquet'), on='MVT_ID_mvt', how='left', validate='one_to_one').normal_pred.values
    rd_saved = order.merge(pd.read_parquet(exp_pkg / 'normal_rank_rd.parquet'), on='MVT_ID_mvt', how='left', validate='one_to_one').normal_pred.values
    p_fb = order.merge(pd.read_parquet(exp_pkg / 'p_fallback_rank.parquet'), on='MVT_ID_mvt', how='left', validate='one_to_one').p_fallback.values
    det = order.merge(pd.read_parquet(base_pkg / 'rank_detail.parquet'), on='MVT_ID_mvt', how='left', validate='one_to_one')
    D = mem.atot_minus_sched.values; heli_const = float(state['heli_const'])
    R = np.column_stack([plain - a3, mem.r_global_seed1.values, mem.r_airport.values] + [mem[f'r_{n}'].values for n in a.members])
    has = np.isfinite(a3)
    if not np.isfinite(R[has]).all():
        raise SystemExit('non-finite rank member on an anchored row')
    rd_refit = np.where(has, a3 + R[:, :3] @ np.array([0.25, 0.25, 0.5]), plain)
    rd_refit = np.where(is_heli == 1, plain, rd_refit)
    gap = rd_saved - rd_refit

    def blend(weights, cell=None):
        keys = apt if cell is None else cell
        w = np.array([weights.get(x, [0.25, 0.25, 0.5] + [0.0] * len(a.members)) for x in keys], float)
        out = np.where(has, a3 + (np.nan_to_num(R) * w).sum(axis=1), plain)
        return np.where(is_heli == 1, plain, out) + gap

    def matched_pred(normal):
        pp = np.clip(p_fb, 0, 1)
        ok = np.isfinite(D) & (D > 0) & np.isfinite(pp)
        mix = np.where(ok, pp * D + (1 - pp) * normal, normal)
        return np.maximum(np.where(is_heli == 1, heli_const, mix), train_eval.FLOOR)

    matched = det.fpl_matched.values == 1
    base_w = json.loads(Path(a.base_weights).read_text(encoding='utf-8'))['deploy']
    base_w = {x: list(v) + [0.0] * len(a.members) for x, v in base_w.items()}
    repro = float(np.max(np.abs(matched_pred(blend(base_w))[matched] - det.pred.values[matched])))
    if repro > 1e-6:
        raise SystemExit(f'base package matched predictions are not reproduced (max gap {repro})')
    wrep = json.loads(Path(a.weights).read_text(encoding='utf-8'))
    weights = wrep['deploy']
    # the 2026 rows get the same cell labels the weights were fitted with (mem carries MVT_ID_mvt, APT, aobt3_anchor)
    # Which airports were collapsed is a property of the FITTED weights, not of the 2026 row counts: re-deciding it here
    # put LIRF (1,473 measured rows in 2026, under cell_min) into a 'LIRF' cell the strict fit never produced. So label
    # without collapsing, then fall back to the airport key only where the fit itself had collapsed (LFPG, LTFM).
    rank_cell = cell_labels(mem.assign(atot_minus_aobt3=mem.aobt3_anchor.values), wrep.get('cells', 'apt'), 0)
    rank_cell = np.where(np.isin(rank_cell, list(weights)), rank_cell, mem.APT.values.astype(str))
    if wrep.get('cells', 'apt') != 'apt':
        miss = sorted({x for x in set(rank_cell) if x not in weights})
        if miss:
            raise SystemExit(f'no deploy weights for cells {miss} (the strict fit never saw them)')
        print(f"cells {wrep['cells']}: {len(set(rank_cell))} on the 2026 rows", flush=True)
    normal_new = blend(weights, rank_cell)
    new = det.pred.values.copy(); new[matched] = matched_pred(normal_new)[matched]
    stamp = time.strftime('%Y%m%d_%H%M%S'); pkg = RUNS / 'models' / f'{a.approach}_{stamp}'; pkg.mkdir(parents=True, exist_ok=True)
    det.assign(pred_before_members=det.pred.values, normal_members=np.where(matched, normal_new, det.normal_pred.values), pred=new) \
       .to_parquet(pkg / 'rank_detail.parquet', index=False)
    mem.drop(columns=['is_heli', 'APT', 'atot_minus_sched']).to_parquet(pkg / 'residual_members_rank.parquet', index=False)
    diff = new - det.pred.values
    summary = {'approach': a.approach, 'members': a.members, 'member_cols': a.member_cols, 'n_member_cols': len(cols),
               'base_package': str(base_pkg), 'weights_file': a.weights, 'weights': weights,
               'base_matched_reproduction_gap': repro, 'rows_changed': int((np.abs(diff) > 1).sum()),
               'rms_change': float(np.sqrt(np.mean(diff ** 2))), 'rows': int(len(new)), 'finite': bool(np.isfinite(new).all()),
               'min_pred': float(new.min()), 'seconds': round(time.time() - t0, 1)}
    (pkg / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'weights'}, indent=1), flush=True)
    if a.submit:
        ev = json.loads(Path(a.evidence).read_text(encoding='utf-8'))
        if not ev.get('passes'):
            raise SystemExit('evidence does not pass the noise guard; no submission written')
        team = submit.team_name(); version = submit.next_version(team)
        path = submit.write_submission(team, version, ids, new)
        desc = (f"{ev['summary']}; strict pooled {ev['pooled']:.2f} ({ev['delta_vs_ref']:+.2f} vs {ev['reference']}), "
                f"worst fold {ev['worst_fold_vs_ref']:+.2f}")
        log = pd.read_csv(SUBS / 'submission_log.csv')
        base_file = (base_pkg / 'submission.txt').read_text(encoding='utf-8').strip()
        base_models = json.loads(log.loc[log.file == base_file, 'models_json'].iloc[-1])
        row = dict(file=path.name, ts=time.strftime('%Y-%m-%d %H:%M:%S'), team=team, version_n=version, approach_id=a.approach, feature_version=fv,
                   models_json=json.dumps(base_models + list(a.model_id)), foldA_rmse=ev['fold_A'], foldB_rmse=ev['fold_B'],
                   result_rmse='', result_rank='', decision='candidate', notes=desc)
        pd.DataFrame([row]).to_csv(SUBS / 'submission_log.csv', mode='a', index=False, header=False, lineterminator='\n')
        submit.append_manifest([f'# Candidate {a.approach} {time.strftime("%Y-%m-%d %H:%M")} - team {team}', '',
                                '| file | approach | description | strict Fold A | strict Fold B | pooled six-pair |', '|---|---|---|---:|---:|---:|',
                                f"| {path.name} | {a.approach} | {desc} | {ev['fold_A']:.2f} | {ev['fold_B']:.2f} | {ev['pooled']:.2f} |"])
        (pkg / 'submission.txt').write_text(path.name, encoding='utf-8')
        print(f'{path.name} <- {a.approach}: {desc}\npackage: {pkg}')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    w = sub.add_parser('weights'); w.add_argument('--members', nargs='+', required=True); w.add_argument('--tag', required=True)
    w.add_argument('--json', required=True)
    w.add_argument('--grid-step', type=float, default=0.05, help='simplex grid resolution (0.05 through A40)')
    w.add_argument('--exact', action='store_true',
                   help='solve the simplex QP exactly (106.fit_weights_exact) instead of rounding to the grid')
    w.add_argument('--ridge', type=float, default=0.0, help='ridge on the exact solve; keeps near-collinear seed twins stable')
    w.add_argument('--cells', choices=['apt', 'adsb', 'adsb3'], default='apt',
                   help="weight-fitting cell: 'apt' (through A42) or 'adsb', which splits each airport on whether "
                        "ADS-B measured the push-back -- what makes a coverage-matched member usable")
    w.add_argument('--cell-min', type=int, default=3000)
    d = sub.add_parser('deploy'); d.add_argument('--members', nargs='+', required=True); d.add_argument('--approach', required=True)
    d.add_argument('--weights', required=True); d.add_argument('--base-package', required=True); d.add_argument('--expert-package', required=True)
    d.add_argument('--members-file', required=True, help="residual_members_rank.parquet with the three existing members")
    d.add_argument('--base-weights', default=str(RUNS / 'residual_weights_A27_20260917.json'),
                   help='weights that reproduce the base package (A31: 104 deployment weights)')
    d.add_argument('--reuse-rank-members', action='store_true')
    d.add_argument('--model-id', nargs='+', default=['mlp_residual_member'],
                   help="model ids appended to the base package's models_json in submission_log (one per new member family)")
    d.add_argument('--member-cols', choices=['deploy', 'strict'], default='deploy',
                   help="deploy: the expert package's columns (target encodings included, like the existing members); "
                        "strict: the target-free columns the strict members were fitted with")
    d.add_argument('--evidence'); d.add_argument('--submit', action='store_true')
    a = p.parse_args()
    weights_cmd(a) if a.cmd == 'weights' else deploy_cmd(a)


if __name__ == '__main__':
    main()
