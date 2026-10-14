"""Schedule-copy detector for NM-matched rows (A34 matched lane): how much is left, and does a set-token classifier help?

  py -3.14 scripts/108_matched_detector.py oracle
  py -3.14 scripts/108_matched_detector.py probe --block AB --candidates set_clf

A34 predicts a matched fixed-wing row as ``p * D + (1 - p) * normal`` (D = take-off - schedule, the copy mode; normal =
AOBT_3 + the per-airport member blend). ``p`` is the context detector fitted on all rows, which also drives the unmatched
gate. Here only the matched mixture changes: the unmatched lane keeps the default detector, because detector changes
swing LIRF unmatched by several seconds (§0o).

oracle   A34's matched SSE split by copy / non-copy rows, and the SSE with a perfect p (upper bound).
probe    fit a candidate copy classifier on one strict block (other eight months, all matched fixed-wing rows, label
         ``fallback``), and score the matched mixture with p_new and with mean(p_old, p_new), per fold.
Candidate probabilities are cached in runs/tuning/detector_probe_<block>_<name>.parquet.
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
from common import RUNS  # noqa: E402

em = importlib.import_module('107_extra_members')
probe = importlib.import_module('106_member_probe')
import detectors  # noqa: E402
import tail_experts as te  # noqa: E402
import tune_residual as tr  # noqa: E402

_SET = {**probe.MLP, 'd_tok': 64, 'heads': 4, 'task': 'clf'}
CANDIDATES = {
    # set-token copy classifiers (BCE on ``fallback``), same encoder as the mlp_set residual member
    'set_clf': {**_SET, 'layers': 1, 'seed': 40},
    'set_clf2': {**_SET, 'layers': 2, 'seed': 41},
    # ablation: the same network with only the self token (no neighbours)
    'self_clf': {**_SET, 'layers': 0, 'seed': 42, 'kinds': ()},
    # cost-weighted: BCE weighted by ((D - NM taxi) / 15 min)^2, capped at 16 (mean 1)
    'set_clf_w': {**_SET, 'layers': 1, 'seed': 43, 'weight': 'gap'},
    'self_clf_w': {**_SET, 'layers': 0, 'seed': 44, 'kinds': (), 'weight': 'gap'},
}
A34_MEMBERS = ['mlp+mlp_s1+mlp_s2+mlp_e20+mlp_wide', 'mlp_set+mlp_set2+mlp_set_s1']
A34_WEIGHTS = RUNS / 'residual_weights_A34_20260918.json'


def a34_table():
    """Strict member table with A34's cross-fitted matched ``normal`` (weights fitted on the other two blocks)."""
    t, R = em.strict_table(A34_MEMBERS)
    w = json.loads(A34_WEIGHTS.read_text(encoding='utf-8'))['cross_fitted']
    normal = np.full(len(t), np.nan)
    for blk, wb in w.items():
        m = t.block.values == blk
        wm = np.array([wb[x] for x in t.APT.values[m]])
        normal[m] = np.where(t.has.values[m], t.atot_minus_aobt3.values[m] + (R[m] * wm).sum(axis=1), t.n_rd.values[m])
    t['normal34'] = normal
    feats = pd.read_parquet(common.DATA / 'features_v5_train.parquet', columns=['MVT_ID_mvt', 'fallback', 'month'])
    t = t.merge(feats, on='MVT_ID_mvt', how='left', validate='one_to_one')
    t['pred34'] = fc.mixture(t.p_fallback.values, t.atot_minus_sched.values, t.normal34.values, t.is_heli.values, t.pred_rd.values)
    return t


def matched_rows(t):
    return (t.fpl_matched.values == 1) & (t.is_heli.values == 0)


def oracle_cmd(a):
    t = a34_table()
    mt = matched_rows(t)
    y = t.y.values; e = (t.pred34.values - y) ** 2
    ref = pd.read_parquet(RUNS / 'oof' / 'extra_members_A34_strict.parquet', columns=['MVT_ID_mvt', 'pred'])
    chk = t[['MVT_ID_mvt']].merge(ref, on='MVT_ID_mvt', how='left').pred.values
    fbm = t.fpl_matched.values == 1
    print(f'reproduces A34 strict OOF on matched rows: max gap {np.nanmax(np.abs(chk[fbm] - t.pred34.values[fbm])):.2e}')
    tot_all = float(((ref.pred.values - pd.read_parquet(RUNS / 'oof' / 'extra_members_A34_strict.parquet', columns=['y']).y.values) ** 2).sum())
    cp = mt & (t.fallback.values == 1)
    D = t.atot_minus_sched.values; nrm = t.normal34.values
    oracle = np.where(cp & np.isfinite(D) & (D > 0), D, nrm)
    oracle = np.maximum(oracle, 30.0)
    e_or = (oracle - y) ** 2
    p = np.clip(t.p_fallback.values, 0, 1)
    out = {'matched_sse_share': float(e[mt].sum() / tot_all), 'copy_rows': int(cp.sum()),
           'copy_sse_share': float(e[cp].sum() / tot_all), 'noncopy_sse_share': float(e[mt & ~cp].sum() / tot_all),
           'oracle_matched_sse_share': float(e_or[mt].sum() / tot_all),
           'mean_p_copy': float(p[cp].mean()), 'mean_p_noncopy': float(p[mt & ~cp].mean())}
    # the error that a better p can remove, split by |D - normal|
    gap = np.abs(D - nrm)
    for lo, hi in [(0, 300), (300, 900), (900, 1800), (1800, 3600), (3600, 1e9)]:
        s = mt & (gap >= lo) & (gap < hi)
        out[f'gap_{lo}_{int(min(hi, 99999))}'] = {'rows': int(s.sum()), 'copies': int((s & cp).sum()),
                                                   'sse_share': float(e[s].sum() / tot_all),
                                                   'oracle_share': float(e_or[s].sum() / tot_all)}
    pooled = np.sqrt(tot_all / len(ref)); pooled_or = np.sqrt((tot_all - e[mt].sum() + e_or[mt].sum()) / len(ref))
    out['pooled_a34'] = float(pooled); out['pooled_with_oracle_matched_p'] = float(pooled_or)
    print(json.dumps(out, indent=1))


def clf_block(block, data):
    """Strict block for a copy classifier: training = all matched fixed-wing rows of the other eight months (copies
    included, date bugs out), label ``fallback``; held = the block's rows (as in the residual probe)."""
    blk = tr.load_block(block, 5, '_v5tfb5rd', data=data)
    blocked = common.FOLDS[block[0]] + common.FOLDS[block[1]]
    bug = te.datebug_label(data.TAXITIME_SEC_mvt.values, data.atot_minus_sched.values)
    keep = ~data.month.isin(blocked).values & ~bug & (data.is_heli.values == 0) & np.isfinite(data.atot_minus_aobt3.values)
    return {**blk, 'train': data[keep]}


def candidate_p(name, block, data):
    cache = RUNS / 'tuning' / f'detector_probe_{block}_{name}.parquet'
    if cache.exists():
        return pd.read_parquet(cache)
    blk = clf_block(block, data)
    t0 = time.time()
    r, _ = probe.fit_mlp_set(blk, CANDIDATES[name])
    res = pd.DataFrame({'MVT_ID_mvt': blk['held'].MVT_ID_mvt.values, 'p': r})
    res.to_parquet(cache, index=False)
    print(f'{name}: fitted in {time.time() - t0:.0f}s', flush=True)
    return res


def score(tb, p, ref_block, label):
    """Matched mixture with probability ``p`` on the block's matched rows; unmatched rows keep A34."""
    mt = matched_rows(tb)
    pm = fc.mixture(p, tb.atot_minus_sched.values, tb.normal34.values, tb.is_heli.values, tb.pred_rd.values)
    pred = np.where(tb.fpl_matched.values == 1, pm, ref_block.pred.values)
    y = tb.y.values; out = {}
    for f in sorted(tb.fold.unique()):
        m = tb.fold.values == f
        out[f'fold_{f}'] = common.rmse(pred[m], y[m]); out[f'matched_{f}'] = common.rmse(pred[m & mt], y[m & mt])
    out['block'] = common.rmse(pred, y)
    fb = tb.fallback.values[mt].astype(int); pp = np.clip(np.asarray(p, float)[mt], 1e-6, 1 - 1e-6)
    out['auc'] = detectors._auc(fb, pp); out['logloss'] = float(-np.mean(fb * np.log(pp) + (1 - fb) * np.log(1 - pp)))
    print(f'  {label}: block {out["block"]:.3f} | ' + ', '.join(f'{f} {out[f"fold_{f}"]:.3f}' for f in sorted(tb.fold.unique()))
          + f' | matched AUC {out["auc"]:.4f}, log loss {out["logloss"]:.4f}', flush=True)
    return out


def probe_cmd(a):
    t = a34_table()
    ref = pd.read_parquet(RUNS / 'oof' / 'extra_members_A34_strict.parquet', columns=['MVT_ID_mvt', 'pred'])
    report = {'block': a.block}
    tb = t[t.block.values == a.block].reset_index(drop=True)
    rb = tb[['MVT_ID_mvt']].merge(ref, on='MVT_ID_mvt', how='left', validate='one_to_one')
    report['a34'] = score(tb, tb.p_fallback.values, rb, 'A34 (context detector)')
    data = None
    ps = {}
    for name in a.candidates:
        cache = RUNS / 'tuning' / f'detector_probe_{a.block}_{name}.parquet'
        if data is None and not cache.exists():
            data, _, _ = common.load_features(5)
        res = candidate_p(name, a.block, data)
        pn = tb[['MVT_ID_mvt']].merge(res, on='MVT_ID_mvt', how='left', validate='one_to_one').p.values
        mt = matched_rows(tb)
        if not np.isfinite(pn[mt & np.isfinite(tb.atot_minus_aobt3.values)]).all():
            raise SystemExit(f'{name}: probabilities missing on matched rows')
        pn = np.where(np.isfinite(pn), pn, tb.p_fallback.values)      # matched rows without AOBT_3 keep the old p
        ps[name] = pn
        report[name] = score(tb, pn, rb, name)
        report[name + '+old'] = score(tb, 0.5 * (pn + tb.p_fallback.values), rb, f'mean({name}, old)')
    if len(ps) > 1:
        avg = np.mean(list(ps.values()), axis=0)
        report['avg'] = score(tb, avg, rb, 'mean(' + ', '.join(ps) + ')')
        report['avg+old'] = score(tb, 0.5 * (avg + tb.p_fallback.values), rb, 'mean(avg, old)')
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('oracle')
    pr = sub.add_parser('probe'); pr.add_argument('--block', default='AB'); pr.add_argument('--candidates', nargs='+', default=['set_clf'])
    pr.add_argument('--json')
    a = p.parse_args()
    if a.cmd == 'oracle':
        oracle_cmd(a)
    else:
        probe_cmd(a)


if __name__ == '__main__':
    main()
