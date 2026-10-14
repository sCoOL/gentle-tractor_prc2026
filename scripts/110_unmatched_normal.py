"""A set-encoder normal-taxi expert for departures without an NM match (no AOBT_3): sizing on the strict blocks.

  py -3.14 scripts/110_unmatched_normal.py check                         # reproduce A34's unmatched predictions
  py -3.14 scripts/110_unmatched_normal.py probe --block AB --candidates nm_masked_set

A34's unmatched rows use the plain clean expert (per-airport LightGBM) as ``normal``. It learns the no-AOBT_3 case
from about 1 % of the training rows (real taxis 19,669 rows < 1 h, normal RMSE 389 vs 190 on matched rows, 9.8 % of
A34 strict SSE). The candidate is the neighbour-set network (106 ``fit_mlp_set``) trained on the taxi itself over
all clean rows of the block's other eight months, with every NM-derived input of the matched rows masked exactly as
it is missing on unmatched rows (NM_NAN, NM_CONST and the self token's AOBT_3 / EOBT_1 / wake). Neighbour tokens are
unchanged: other flights' AOBT_3 stays visible, as it is for an unmatched departure.

The gate weight W and the date-bug probability Pb of A34's strict OOF are kept, and only ``normal`` is swapped in
``(1 - Pb) * (normal + W * (D - normal)) + Pb * (k * 86400 + normal)``, so this sizes the expert, not a refitted gate.
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
import nbr_set  # noqa: E402
import tail_experts as te  # noqa: E402
import tune_residual as tr  # noqa: E402
from common import RUNS  # noqa: E402

rw = importlib.import_module('104_residual_weights')
probe = importlib.import_module('106_member_probe')

NM_NAN = ['atot_minus_eobt1', 'atot_minus_lobt', 'atot_minus_aobt3', 'aobt3_minus_lobt', 'aobt3_minus_sched',
          'lobt_minus_sched', 'lobt_minus_iobt', 'plan_block_time', 'arr_delay', 'n_out_at_aobt3', 'n_in_at_aobt3',
          'push_prev15_at_aobt3', 'dep_same_pier_at_aobt3']
NM_CONST = {'wake': 'NA', 'segment': 'NA', 'operator': 'NA', 'flight_type': 'NA', 'src_conflict_ades': 1,
            'src_conflict_type': 1, 'iobt_eq_lobt': 0, 'sched_eq_eobt1': 0, 'aobt3_eq_lobt': 0,
            'dep_same_rwy_between_aobt3_atot': 0, 'arr_same_rwy_between_aobt3_atot': 0, 'dep_any_rwy_between_aobt3_atot': 0,
            'fpl_matched': 0}
_SET = {**probe.MLP, 'd_tok': 64, 'heads': 4, 'taxi_expert': True, 'mask_self_nm': True}
CANDIDATES = {
    'nm_masked_set': {**_SET, 'layers': 1, 'seed': 50},
    'nm_masked_set2': {**_SET, 'layers': 2, 'seed': 51},
    'nm_masked_self': {**_SET, 'layers': 0, 'seed': 52, 'kinds': ()},     # ablation: no neighbour tokens
    # trained on masked MATCHED rows only: "what a comparable matched flight takes", without learning the genuinely
    # unmatched subpopulation (2026 check: the all-rows expert extrapolates very long taxis to EHAM's January snow week)
    'nm_mo_set': {**_SET, 'layers': 1, 'seed': 53, 'matched_only': True},
    'nm_mo_set2': {**_SET, 'layers': 2, 'seed': 54, 'matched_only': True},
    'nm_mo_self': {**_SET, 'layers': 0, 'seed': 55, 'kinds': (), 'matched_only': True},
    # 2026-09-19: surge tokens (nbr_set.SURGE: stand x runway normal and NM taxi / taxi-in minus it);
    # the own NM residual is masked with the rest of the self token's NM fields. Same seeds as nm_masked_set / _set2.
    'nm_masked_setn': {**_SET, 'layers': 1, 'seed': 50, 'ctx': {'surge': True, 'lead_in': 10800.0}},
    'nm_masked_setn2': {**_SET, 'layers': 2, 'seed': 51, 'ctx': {'surge': True, 'lead_in': 10800.0}},
    # 2026-09-19 night (after v36): the true is_unmatched flag as an input, as the A37 copy classifiers have it, so the
    # expert can tell genuinely unmatched rows from masked matched ones instead of averaging the two populations
    'nm_masked_set_u': {**_SET, 'layers': 1, 'seed': 60, 'flag': True},
    'nm_masked_set2_u': {**_SET, 'layers': 2, 'seed': 61, 'flag': True},
    'nm_masked_self_u': {**_SET, 'layers': 0, 'seed': 62, 'kinds': (), 'flag': True},
    # 2026-09-22: the three deployed experts' twins at the same seeds, reading the ADS-B off-block columns and
    # the ground-phase stop sequence (124.with_trace) - the set-network version of 124's LightGBM AB twin
    'nm_masked_set_ab': {**_SET, 'layers': 1, 'seed': 50, 'adsb': True},
    'nm_masked_set2_ab': {**_SET, 'layers': 2, 'seed': 51, 'adsb': True},
    'nm_masked_self_ab': {**_SET, 'layers': 0, 'seed': 52, 'kinds': (), 'adsb': True},
}


def mask_nm(df):
    """The NM-derived inputs of every row set to what an unmatched row carries."""
    out = df.copy()
    for c in NM_NAN:
        if c in out:
            out[c] = np.nan
    for c, v in NM_CONST.items():
        if c in out:
            out[c] = v if not isinstance(out[c].dtype, pd.CategoricalDtype) else pd.Categorical([v] * len(out))
    return out


def unmatched_table():
    """A34's strict OOF with what its unmatched composition needs (old normal, W, Pb, D, day shift, p, validity)."""
    t = rw.member_table(5, '_v5tfb5')
    o = pd.read_parquet(RUNS / 'oof' / 'extra_members_A34_strict.parquet', columns=['MVT_ID_mvt', 'pred', 'w', 'pb'])
    t = t.merge(o.rename(columns={'pred': 'p34'}), on='MVT_ID_mvt', how='left', validate='one_to_one')
    raw = pd.read_parquet(common.DATA / 'base_train.parquet', columns=['MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'])
    raw['day_shift'] = te.local_day_shift(raw.MVT_TIME_UTC_mvt.values, raw.SCHED_TIME_UTC_mvt.values)
    t = t.merge(raw[['MVT_ID_mvt', 'day_shift']], on='MVT_ID_mvt', how='left', validate='one_to_one')
    f = pd.read_parquet(common.DATA / 'features_v5_train.parquet', columns=['MVT_ID_mvt', 'fallback'])
    return t.merge(f, on='MVT_ID_mvt', how='left', validate='one_to_one')


def compose(t, normal):
    """A34's unmatched prediction with ``normal`` swapped in (W, Pb and gate validity from the old expert)."""
    D = t.atot_minus_sched.values
    old = t.normal_pred.values
    valid = np.isfinite(D) & (t.is_heli.values == 0) & (np.abs(D - old) >= 60)
    mix = fc.mixture(t.p_fallback.values, D, normal, t.is_heli.values, t.pred_rd.values)
    two = np.where(valid, np.maximum(normal + np.clip(t.w.values, 0, 1) * (D - normal), 30.0), mix)
    return te.combine(two, np.nan_to_num(t.pb.values), normal, t.day_shift.values)


def check_cmd(a):
    t = unmatched_table()
    um = t.fpl_matched.values == 0
    rep = compose(t, t.normal_pred.values)
    gap = np.abs(rep[um] - t.p34.values[um])
    print(f'unmatched rows {um.sum()}: max gap {gap.max():.3e}, rows > 1 s {(gap > 1).sum()}')


def candidate_normal(name, block, data):
    cache = RUNS / 'tuning' / f'unmatched_normal_{block}_{name}.parquet'
    if cache.exists():
        return pd.read_parquet(cache)
    blk = tr.load_block(block, 5, '_v5tfb5rd', data=data)
    blocked = common.FOLDS[block[0]] + common.FOLDS[block[1]]
    bug = te.datebug_label(data.TAXITIME_SEC_mvt.values, data.atot_minus_sched.values)
    keep = (~data.month.isin(blocked).values & ~bug & (data.is_heli.values == 0) & (data.fallback.values == 0))
    if CANDIDATES[name].get('matched_only'):
        keep &= np.isfinite(data.atot_minus_aobt3.values.astype(float))
    flag = (lambda df: df.assign(is_unmatched=(df.fpl_matched.values == 0).astype('float32'))) if CANDIDATES[name].get('flag')         else (lambda df: df)
    train = mask_nm(flag(data[keep])); held = mask_nm(flag(blk['held']))
    drop = set(NM_NAN) | set(NM_CONST)          # missing or constant without an NM match: out of the row inputs
    cols = [c for c in blk['cols'] if c not in drop]; cats = [c for c in blk['cats'] if c not in drop]
    if CANDIDATES[name].get('flag'):
        cols = cols + ['is_unmatched']
    if CANDIDATES[name].get('adsb'):
        m124 = importlib.import_module('124_unmatched_adsb')       # lazy: 124 imports this module
        train, held = m124.with_trace(train), m124.with_trace(held); cols = cols + m124.AB_COLS
    t0 = time.time()
    r, _ = probe.fit_mlp_set({**blk, 'train': train, 'held': held, 'cols': cols, 'cats': cats}, CANDIDATES[name])
    res = pd.DataFrame({'MVT_ID_mvt': held.MVT_ID_mvt.values, 'normal': r})
    res.to_parquet(cache, index=False)
    print(f'{name}: fitted in {time.time() - t0:.0f}s', flush=True)
    return res


def probe_cmd(a):
    t = unmatched_table()
    tb = t[t.block.values == a.block].reset_index(drop=True)
    um = tb.fpl_matched.values == 0; y = tb.y.values
    real = um & (tb.is_heli.values == 0) & (tb.fallback.values == 0) & (y < 3600)
    base = compose(tb, tb.normal_pred.values)

    def report_line(label, pred, normal):
        folds = {f: common.rmse(pred[tb.fold.values == f], y[tb.fold.values == f]) for f in sorted(tb.fold.unique())}
        out = {'block': common.rmse(pred, y), 'folds': folds, 'unmatched': common.rmse(pred[um], y[um]),
               'normal_rmse_real_lt1h': common.rmse(normal[real], y[real]), 'final_rmse_real_lt1h': common.rmse(pred[real], y[real])}
        print(f'  {label}: block {out["block"]:.3f} | ' + ', '.join(f'{f} {v:.3f}' for f, v in folds.items())
              + f' | unmatched {out["unmatched"]:.1f}; real <1 h: normal {out["normal_rmse_real_lt1h"]:.1f}, '
              f'final {out["final_rmse_real_lt1h"]:.1f}', flush=True)
        return out

    pred34 = np.where(um, base, tb.p34.values)
    report = {'block': a.block, 'a34': report_line('A34', pred34, tb.normal_pred.values)}
    data = None; cols = []
    for name in a.candidates:
        if data is None and not (RUNS / 'tuning' / f'unmatched_normal_{a.block}_{name}.parquet').exists():
            data, _, _ = common.load_features(5)
        res = candidate_normal(name, a.block, data)
        nn_ = tb[['MVT_ID_mvt']].merge(res, on='MVT_ID_mvt', how='left', validate='one_to_one').normal.values
        cols.append(nn_)
        for lab, nrm in [(name, nn_), (f'mean({name}, old)', 0.5 * (nn_ + tb.normal_pred.values))]:
            nrm = np.where(um & np.isfinite(nrm), nrm, tb.normal_pred.values)
            report[lab] = report_line(lab, np.where(um, compose(tb, nrm), tb.p34.values), nrm)
    if len(cols) > 1:
        avg = np.where(um, np.mean(cols, axis=0), tb.normal_pred.values)
        report['avg'] = report_line('mean(candidates)', np.where(um, compose(tb, avg), tb.p34.values), avg)
        avg2 = np.where(um, 0.5 * (avg + tb.normal_pred.values), tb.normal_pred.values)
        report['avg+old'] = report_line('mean(avg, old)', np.where(um, compose(tb, avg2), tb.p34.values), avg2)
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1), encoding='utf-8')


NEW_NORMAL = ['nm_masked_set', 'nm_masked_self', 'nm_masked_set2']


def frames_cmd(a):
    """Strict frames ``<source><tag>``: the unmatched rows' normal expert replaced by the mean of the cached block
    probes, and their frame mixture recomputed with it (matched rows untouched)."""
    feats = pd.read_parquet(common.DATA / 'features_v5_train.parquet', columns=['MVT_ID_mvt', 'atot_minus_sched', 'is_heli'])
    for left, right in te.stg.BLOCKS:
        parts = [pd.read_parquet(RUNS / 'tuning' / f'unmatched_normal_{left}{right}_{n}.parquet').set_index('MVT_ID_mvt').normal
                 for n in NEW_NORMAL]
        nrm = pd.concat(parts, axis=1).mean(axis=1)
        for fold in (left, right):
            src = pd.read_parquet(fc._frame_path(left, right, fold, a.source))
            info = src[['MVT_ID_mvt']].merge(feats, on='MVT_ID_mvt', how='left', validate='one_to_one')
            um = src.fpl_matched.values == 0
            new = nrm.reindex(src.MVT_ID_mvt).values
            if not np.isfinite(new[um]).all():
                raise SystemExit(f'fold {fold}: new normal missing on unmatched rows')
            out = src.copy()
            out['normal_pred'] = np.where(um, new, src.normal_pred.values)
            mix = fc.mixture(src.p_fallback.values, info.atot_minus_sched.values, out.normal_pred.values, info.is_heli.values, src.pred.values)
            out['pred'] = np.where(um, mix, src.pred.values)
            out.to_parquet(fc._frame_path(left, right, fold, a.source + a.tag), index=False)
            print(f'fold {fold}: {um.sum()} unmatched rows replaced', flush=True)


def gate_cmd(a):
    """te.strict_diagnostic on ``suffix`` (gate and Pb refitted per blocked pair), written as a strict OOF."""
    report, frame = te.strict_diagnostic(output=RUNS / f'unmatched_gate_{a.suffix.strip("_")}.json', candidate_version=5,
                                         candidate_suffix=a.suffix)
    out = RUNS / 'oof' / f'unmatched_gate{a.suffix}_strict.parquet'
    frame.to_parquet(out, index=False)
    print(json.dumps(report.get('pooled', {}), indent=1, default=float)[:800]); print(out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('check')
    pr = sub.add_parser('probe'); pr.add_argument('--block', default='AB'); pr.add_argument('--candidates', nargs='+', default=['nm_masked_set'])
    pr.add_argument('--json')
    fr = sub.add_parser('frames'); fr.add_argument('--source', default='_v5tfb5rd'); fr.add_argument('--tag', default='U')
    ga = sub.add_parser('gate'); ga.add_argument('--suffix', required=True)
    a = p.parse_args()
    {'check': check_cmd, 'probe': probe_cmd, 'frames': frames_cmd, 'gate': gate_cmd}[a.cmd](a)


if __name__ == '__main__':
    main()
