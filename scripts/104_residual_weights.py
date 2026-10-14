"""Per-airport, block-cross-fitted weights for the three matched-row residual members (strict frames).

  py -3.14 scripts/104_residual_weights.py --frames-suffix _v5tfb5 --out-suffix _v5tfb5rw --json runs/residual_weights_A27_20260917.json

The rd expert is ``aobt3 + 0.25 r_global + 0.25 r_global_seed1 + 0.5 r_airport`` with fixed weights. Here the weights
(non-negative, summing to 1, 0.05 grid) are chosen per airport to minimise the matched-lane mixture error
``p * D + (1 - p) * normal`` of the *other two* strict blocks, then applied to the held block. The members are recovered
from the saved frames: r_global = <suffix> normal - aobt3, r_airport = <suffix>ra normal - aobt3, and
r_global_seed1 = 4 (<suffix>rd normal - aobt3) - r_global - 2 r_airport. Unmatched rows, helicopters and the fallback
probability are unchanged, so the strict three-expert diagnostic isolates the weights. The JSON also holds weights fitted
on all six frames for deployment.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import fallback_context as fc  # noqa: E402
import strict_tail_gate as stg  # noqa: E402

GRID = np.array([w for w in itertools.product(np.arange(0, 1.0001, 0.05), repeat=2) if w[0] + w[1] <= 1.0 + 1e-9])
DEFAULT = (0.25, 0.25, 0.5)


def member_table(feature_version, suffix):
    feats = pd.read_parquet(common.DATA / f'features_v{feature_version}_train.parquet', columns=['MVT_ID_mvt', 'atot_minus_aobt3', 'atot_minus_sched', 'is_heli'])
    parts = []
    for left, right in stg.BLOCKS:
        for fold in (left, right):
            src = pd.read_parquet(fc._frame_path(left, right, fold, suffix))
            ra = pd.read_parquet(fc._frame_path(left, right, fold, suffix + 'ra'), columns=['MVT_ID_mvt', 'normal_pred']).rename(columns={'normal_pred': 'n_ra'})
            rd = pd.read_parquet(fc._frame_path(left, right, fold, suffix + 'rd'), columns=['MVT_ID_mvt', 'normal_pred', 'pred']).rename(
                columns={'normal_pred': 'n_rd', 'pred': 'pred_rd'})
            t = src.merge(ra, on='MVT_ID_mvt', validate='one_to_one').merge(rd, on='MVT_ID_mvt', validate='one_to_one') \
                   .merge(feats, on='MVT_ID_mvt', how='left', validate='one_to_one')
            t['block'] = f'{left}{right}'; t['fold'] = fold
            parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    a3 = t.atot_minus_aobt3.values.astype(float)
    t['has'] = np.isfinite(a3) & (t.is_heli.values == 0)
    t['rg0'] = t.normal_pred.values - a3
    t['ra'] = t.n_ra.values - a3
    t['rg1'] = 4.0 * (t.n_rd.values - a3) - t.rg0.values - 2.0 * t.ra.values
    p = np.clip(t.p_fallback.values.astype(float), 0, 1); D = t.atot_minus_sched.values.astype(float)
    ok = np.isfinite(D) & (D > 0) & np.isfinite(p)
    t['mix_m'] = np.where(ok, 1 - p, 1.0)
    t['mix_c'] = np.where(ok, p * D + (1 - p) * a3, a3)
    return t


def fit_simplex(rows):
    """Weights (w0, w1, wa) on the 0.05 simplex grid minimising the matched mixture SSE of ``rows``."""
    R = np.column_stack([rows.rg0.values, rows.rg1.values, rows.ra.values]) * rows.mix_m.values[:, None]
    target = rows.y.values - rows.mix_c.values
    A = R.T @ R; b = R.T @ target
    W = np.column_stack([GRID[:, 0], GRID[:, 1], 1 - GRID[:, 0] - GRID[:, 1]])
    sse = np.einsum('ij,jk,ik->i', W, A, W) - 2 * W @ b
    return tuple(float(x) for x in W[int(np.argmin(sse))])


def normal_with(rows, weights_by_apt):
    w = np.array([weights_by_apt.get(a, DEFAULT) for a in rows.APT.values])
    a3 = rows.atot_minus_aobt3.values.astype(float)
    new = a3 + w[:, 0] * rows.rg0.values + w[:, 1] * rows.rg1.values + w[:, 2] * rows.ra.values
    return np.where(rows.has.values, new, rows.n_rd.values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--feature-version', type=int, default=5)
    p.add_argument('--frames-suffix', required=True)
    p.add_argument('--out-suffix', required=True)
    p.add_argument('--json', required=True)
    a = p.parse_args()
    t = member_table(a.feature_version, a.frames_suffix)
    # the recovered members must reproduce the rd frames exactly
    chk = normal_with(t, {})
    if float(np.nanmax(np.abs(chk - t.n_rd.values))) > 1e-6:
        raise SystemExit('recovered members do not reproduce the rd expert')
    fit_rows = t[t.has.values]
    report = {'frames_suffix': a.frames_suffix, 'grid_step': 0.05, 'default': DEFAULT, 'cross_fitted': {}, 'deploy': {}}
    new_normal = np.full(len(t), np.nan)
    for blk in t.block.unique():
        other = fit_rows[fit_rows.block != blk]
        weights = {apt: fit_simplex(other[other.APT == apt]) for apt in sorted(t.APT.unique())}
        report['cross_fitted'][blk] = weights
        m = t.block.values == blk
        new_normal[m] = normal_with(t[m], weights)
    report['deploy'] = {apt: fit_simplex(fit_rows[fit_rows.APT == apt]) for apt in sorted(t.APT.unique())}
    t['n_rw'] = new_normal
    t['pred_rw'] = fc.mixture(t.p_fallback.values, t.atot_minus_sched.values, t.n_rw.values, t.is_heli.values, t.pred_rd.values)
    # matched-lane effect before the gate/tail diagnostic
    y = t.y.values; mt = t.fpl_matched.values == 1
    report['matched_rmse_rd'] = common.rmse(t.pred_rd.values[mt], y[mt]); report['matched_rmse_rw'] = common.rmse(t.pred_rw.values[mt], y[mt])
    report['matched_rmse_by_block'] = {blk: {'rd': common.rmse(t.pred_rd.values[mt & (t.block.values == blk)], y[mt & (t.block.values == blk)]),
                                             'rw': common.rmse(t.pred_rw.values[mt & (t.block.values == blk)], y[mt & (t.block.values == blk)])}
                                       for blk in t.block.unique()}
    for left, right in stg.BLOCKS:
        for fold in (left, right):
            src = pd.read_parquet(fc._frame_path(left, right, fold, a.frames_suffix + 'rd'))
            upd = t[t.fold == fold][['MVT_ID_mvt', 'n_rw', 'pred_rw']]
            out = src.merge(upd, on='MVT_ID_mvt', how='left', validate='one_to_one')
            if out.n_rw.isna().any():
                raise SystemExit(f'fold {fold}: missing weighted expert rows')
            out['normal_pred'] = out.n_rw.values; out['pred'] = out.pred_rw.values
            out.drop(columns=['n_rw', 'pred_rw']).to_parquet(fc._frame_path(left, right, fold, a.out_suffix), index=False)
    Path(a.json).write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('cross_fitted',)}, indent=1))


if __name__ == '__main__':
    main()
