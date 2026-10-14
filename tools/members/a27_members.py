"""A27 package inputs for the final rows (docs/REPRODUCE.md, stage 3).

  py -3.14 tools/members/a27_members.py m3rpa      # re-predict the 11 saved M3rpa boosters (NEW + OLD)
  py -3.14 tools/members/a27_members.py detector   # refit the context fallback detector once, score NEW + OLD
  py -3.14 tools/members/a27_members.py rd         # refit r_global_seed1 + r_airport x10 once, score NEW + OLD
  py -3.14 tools/members/a27_members.py gate       # re-predict the saved strict tail gate; pb, day_shift -> rank_detail

Outputs (NEW, 670,790 rows) in runs/models/R_A27/ under the reference A27 file names; OLD-scored copies (344,841 rows, the reference
old features) in runs/members_repro/old_scored/R_A27/; refitted boosters in runs/models/R_A27/models/.
Every refit is the reference code path's parameters verbatim (96_deploy_candidate.py --approach A27 ...; residual_variants;
fallback_context.score_seeds; tail_experts.rank_prediction).
"""
from __future__ import annotations

import mcommon as mc

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from mcommon import REFM, OUT, REPRO, OLD, DATA, say, compare, fmt, save_report

A27 = REFM / 'A27_20260917_141019'
NEWD = OUT / 'R_A27'
OLDD = REPRO / 'old_scored' / 'R_A27'
STATE = json.loads((A27 / 'M3rpa' / 'state.json').read_text(encoding='utf-8'))


def _dirs():
    for d in (NEWD, OLDD, NEWD / 'models'):
        d.mkdir(parents=True, exist_ok=True)


def m3rpa_predict(rank):
    """tail_experts.fit_full_m3rpa's normal on ``rank`` from the saved boosters (train_eval._m3r per_airport)."""
    import train_eval
    cols, cats = STATE['cols'], STATE['cats']
    cols2 = [c for c in cols if c != 'APT']
    metas = [json.loads((A27 / 'M3rpa' / f'{i:03d}_meta.json').read_text()) for i in range(11)]
    by_tag = {m['tag']: i for i, m in enumerate(metas)}
    reg = np.full(len(rank), np.nan)
    for apt in rank.APT.unique():
        i = by_tag[f'M3rpa_{apt}']
        assert metas[i]['columns'] == cols2
        b = rank.APT.values == apt
        model = lgb.Booster(model_file=str(A27 / 'M3rpa' / f'{i:03d}_lgb.txt'))
        reg[b] = model.predict(train_eval.prep(rank[b], cols2, cats))
    i = by_tag['M3r_residual']
    assert metas[i]['columns'] == cols
    model = lgb.Booster(model_file=str(A27 / 'M3rpa' / f'{i:03d}_lgb.txt'))
    r = model.predict(train_eval.prep(rank, cols, cats))
    a3 = rank.atot_minus_aobt3.values.astype(float)
    reg = np.where(np.isfinite(a3), a3 + r, reg)
    return np.where(rank.is_heli.values == 1, float(STATE['heli_const']), reg)


def cmd_m3rpa(a):
    _dirs()
    rep = {}
    for tag, old in (('new', False), ('old', True)):
        rank = mc.load_rank(5, old=old)
        t0 = time.time()
        normal = m3rpa_predict(rank)
        out = pd.DataFrame({'MVT_ID_mvt': rank.MVT_ID_mvt.values, 'normal_pred': normal})
        out.to_parquet((OLDD if old else NEWD) / 'normal_rank.parquet', index=False)
        ref = pd.read_parquet(A27 / 'normal_rank.parquet')
        c = compare(out.MVT_ID_mvt, out.normal_pred, ref.MVT_ID_mvt, ref.normal_pred, f'M3rpa normal ({tag}) vs reference A27')
        rep[tag] = c; say(fmt(c), f'[{time.time() - t0:.0f}s]')
        assert np.isfinite(normal).all()
    save_report('step1_a27_m3rpa', rep)


def cmd_detector(a):
    """96 --detector context: fc.score_seeds(train_v5, rank_v5, fc.FEATS, fc.CATS, 400, None) — fitted once, both
    rank tables scored by the same booster (detectors.score_fallback, verbatim apart from returning the model)."""
    import detectors
    import fallback_context as fc
    _dirs()
    need = ['MVT_ID_mvt', 'month', 'fallback'] + [c for c in fc.FEATS if c != 'month']
    train_d = pd.read_parquet(DATA / 'features_v5_train.parquet', columns=need)
    new = mc.load_rank(5, columns=[c for c in need if c != 'fallback'])
    old = mc.load_rank(5, old=True, columns=[c for c in need if c != 'fallback'])
    P = {**detectors.CLF_PARAMS}
    y = detectors.fallback_label(train_d).values
    X = detectors._prep(train_d, fc.FEATS, fc.CATS)
    categories = {c: X[c].cat.categories for c in fc.CATS}
    t0 = time.time()
    model = lgb.train(P, lgb.Dataset(X, y, categorical_feature=fc.CATS), 400)
    say(f'context detector fitted [{time.time() - t0:.0f}s]')
    model.save_model(str(NEWD / 'models' / 'context_detector.txt'))
    rep = {}
    for tag, rk, d in (('new', new, NEWD), ('old', old, OLDD)):
        p = model.predict(detectors._prep(rk, fc.FEATS, fc.CATS, categories))
        out = pd.DataFrame({'MVT_ID_mvt': rk.MVT_ID_mvt.values, 'p_fallback': p})
        out.to_parquet(d / 'p_fallback_rank.parquet', index=False)
        ref = pd.read_parquet(A27 / 'p_fallback_rank.parquet')
        c = compare(out.MVT_ID_mvt, out.p_fallback, ref.MVT_ID_mvt, ref.p_fallback, f'p_fallback ({tag}) vs reference A27')
        rep[tag] = c; say(fmt(c))
    save_report('step2_a27_detector', rep)


def cmd_rd(a):
    """residual_variants.rank_blend_members(5, cols, cats, seed=1, rounds=800, params=residual_params,
    airport_params={'learning_rate': 0.04}, airport_rounds=1320), fitted once and scored on both rank tables."""
    import common
    import tail_experts as te
    import train_eval
    _dirs()
    cols, cats = STATE['cols'], STATE['cats']
    params = STATE['params']['residual_params']; rounds = int(STATE['params']['residual_rounds'])
    airport_params = {'learning_rate': 0.04}; airport_rounds = 1320
    train = pd.read_parquet(DATA / 'features_v5_train.parquet')
    bug = te.training_exclusion(train.TAXITIME_SEC_mvt.values, train.atot_minus_sched.values, 'datebug')
    clean = train_eval._clean(train[~bug])
    cm = clean[np.isfinite(clean.atot_minus_aobt3.values)]
    del train, clean
    say(f'clean matched training rows {len(cm):,} (reference meta: 1,963,611)')
    ranks = {'new': mc.load_rank(5), 'old': mc.load_rank(5, old=True)}
    res = {k: {'rg1': np.full(len(r), np.nan), 'ra': np.full(len(r), np.nan)} for k, r in ranks.items()}
    t0 = time.time()
    X = train_eval.prep(cm, cols, cats)
    m = train_eval.fit_lgb(X, cm.TAXITIME_SEC_mvt.values - cm.atot_minus_aobt3.values, cats, {**dict(params), 'seed': 1}, rounds,
                           'M3r_residual_s1')
    m.save_model(str(NEWD / 'models' / 'r_global_seed1.txt'))
    for k, r in ranks.items():
        has = np.isfinite(r.atot_minus_aobt3.values.astype(float))
        res[k]['rg1'][has] = m.predict(train_eval.prep(r[has], cols, cats, train_eval._cats_of(X, cats)))
    say(f'r_global_seed1 fitted [{time.time() - t0:.0f}s]')
    del X, m
    for apt in ranks['new'].APT.unique():
        t1 = time.time()
        aa = cm[cm.APT.values == apt]
        Xa = train_eval.prep(aa, cols, cats)
        ma = train_eval.fit_lgb(Xa, aa.TAXITIME_SEC_mvt.values - aa.atot_minus_aobt3.values, cats, dict(airport_params), airport_rounds,
                                f'M3r_residual_{apt}')
        ma.save_model(str(NEWD / 'models' / f'r_airport_{apt}.txt'))
        for k, r in ranks.items():
            b = (r.APT.values == apt) & np.isfinite(r.atot_minus_aobt3.values.astype(float))
            if b.any():
                res[k]['ra'][b] = ma.predict(train_eval.prep(r[b], cols, cats, train_eval._cats_of(Xa, cats)))
        say(f'r_airport {apt}: {len(aa):,} rows [{time.time() - t1:.0f}s]')
    ref = pd.read_parquet(REFM / 'A31_20260918_075753' / 'residual_members_rank.parquet')
    ref_rd = pd.read_parquet(A27 / 'normal_rank_rd.parquet')
    rep = {}
    import residual_variants as rv
    for k, r in ranks.items():
        d = NEWD if k == 'new' else OLDD
        plain = r[['MVT_ID_mvt']].merge(pd.read_parquet(d / 'normal_rank.parquet'), on='MVT_ID_mvt', how='left',
                                        validate='one_to_one').normal_pred.values
        a3 = r.atot_minus_aobt3.values.astype(float)
        out = pd.DataFrame({'MVT_ID_mvt': r.MVT_ID_mvt.values, 'aobt3_anchor': a3, 'r_global': plain - a3,
                            'r_global_seed1': res[k]['rg1'], 'r_airport': res[k]['ra']})
        out.to_parquet(d / 'residual_members_rank.parquet', index=False)
        rd = rv.blend_members(plain, a3, res[k]['rg1'], res[k]['ra'], r.is_heli.values)
        pd.DataFrame({'MVT_ID_mvt': r.MVT_ID_mvt.values, 'normal_pred': rd}).to_parquet(d / 'normal_rank_rd.parquet', index=False)
        rep[k] = [compare(out.MVT_ID_mvt, out[c], ref.MVT_ID_mvt, ref[c], f'{c} ({k}) vs reference A30/A31') for c in ('r_global_seed1', 'r_airport')]
        rep[k].append(compare(r.MVT_ID_mvt, rd, ref_rd.MVT_ID_mvt, ref_rd.normal_pred, f'normal_rank_rd ({k}) vs reference A27'))
        for c in rep[k]:
            say(fmt(c))
    rep['seconds'] = round(time.time() - t0, 1)
    save_report('step3_a27_rd', rep)


def gate_detail(old, src=None, write=True):
    """tail_experts.rank_prediction's rank side with the SAVED gate booster: base mixture, gate_rank, w, pb, day_shift.
    ``src`` overrides the folder holding normal_rank_rd / p_fallback_rank (the reference A27 for the pure replay check)."""
    import tail_experts as te
    import tail_gate
    import train_eval
    import strict_tail_gate as stg
    d = src if src is not None else (OLDD if old else NEWD)
    rank_features = mc.load_rank(5, old=old)
    ids = rank_features.MVT_ID_mvt.values
    order = pd.DataFrame({'MVT_ID_mvt': ids})
    normal = order.merge(pd.read_parquet(d / 'normal_rank_rd.parquet'), on='MVT_ID_mvt', how='left', validate='one_to_one').normal_pred.values
    p = order.merge(pd.read_parquet(d / 'p_fallback_rank.parquet'), on='MVT_ID_mvt', how='left', validate='one_to_one').p_fallback.values
    rank = rank_features.copy(); rank['p_fallback'] = p
    base = train_eval._mixture(rank, normal, {'p_scale': 1.0}, float(STATE['heli_const']))
    base = np.maximum(base, train_eval.FLOOR)
    rank_features['p_fallback'] = p
    skeleton = rank_features[['MVT_ID_mvt', 'APT', 'month', 'fpl_matched']].copy()
    skeleton['pred'] = base; skeleton['normal_pred'] = normal; skeleton['y'] = np.nan
    gate_rank = tail_gate.prepare_frame(skeleton, rank_features, base_p_scale=1.0)
    gate_rank = te.with_expert(gate_rank)
    summ = json.loads((A27 / 'summary.json').read_text(encoding='utf-8'))
    feats, cats = summ['gate']['features'], summ['gate']['categoricals']
    assert feats == list(stg.STRICT_GATE_FEATURES) and cats == list(stg.STRICT_GATE_CATS)
    X = gate_rank[feats].copy()
    for c in cats:
        X[c] = pd.Categorical(X[c].astype(str))
    model = lgb.Booster(model_file=str(A27 / 'gate_lgb.txt'))
    w = np.clip(model.predict(X), 0, 1)
    two = tail_gate.apply_gate(gate_rank, w, 1.0)
    base_rank = pd.read_parquet((OLD if old else DATA) / 'base_rank.parquet', columns=['MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'])
    base_rank['day_shift'] = te.local_day_shift(base_rank.MVT_TIME_UTC_mvt, base_rank.SCHED_TIME_UTC_mvt)
    shift = gate_rank[['MVT_ID_mvt']].merge(base_rank[['MVT_ID_mvt', 'day_shift']], on='MVT_ID_mvt', how='left',
                                           validate='one_to_one').day_shift.values
    dom = te.b_domain(gate_rank.APT.values, gate_rank.fpl_matched.values, shift, gate_rank.is_heli.values)
    rates = np.asarray(summ['pb_rates'], float)
    pb = te.apply_pb_bins(rates, gate_rank.atot_minus_sched.values / 3600.0, dom, summ['pb_edges'])
    pred = te.combine(two, pb, gate_rank.normal_pred.values, shift)
    detail = gate_rank[['MVT_ID_mvt', 'APT', 'fpl_matched', 'atot_minus_sched', 'normal_pred', 'gate_valid']].assign(
        base_pred=base, w=w, pb=pb, day_shift=shift, pred=pred)
    if write:
        detail.to_parquet(d / 'rank_detail.parquet', index=False)
    return detail


def cmd_gate(a):
    _dirs()
    ref = pd.read_parquet(A27 / 'rank_detail.parquet')
    rep = {}
    for tag, old, src in (('replay_main_inputs_old_features', True, A27), ('old', True, None), ('new', False, None)):
        det = gate_detail(old, src=src, write=src is None)
        um = det.fpl_matched.values == 0
        rep[tag] = []
        for col in ('w', 'pb', 'day_shift', 'normal_pred', 'base_pred', 'pred'):
            c = compare(det.MVT_ID_mvt, det[col], ref.MVT_ID_mvt, ref[col], f'{col} ({tag}) vs reference A27')
            rep[tag].append(c); say(fmt(c))
        c = compare(det.MVT_ID_mvt[um], det.w[um], ref.MVT_ID_mvt, ref.w, f'w on unmatched rows ({tag})'); rep[tag].append(c); say(fmt(c))
        gv = det[['MVT_ID_mvt', 'gate_valid']].merge(ref[['MVT_ID_mvt', 'gate_valid']], on='MVT_ID_mvt', suffixes=('', '_ref'))
        rep[tag].append({'label': f'gate_valid agreement ({tag})', 'n': int(len(gv)),
                         'differ': int((gv.gate_valid != gv.gate_valid_ref).sum())})
        say(rep[tag][-1])
    save_report('step4_a27_gate', rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['m3rpa', 'detector', 'rd', 'gate'])
    a = ap.parse_args()
    {'m3rpa': cmd_m3rpa, 'detector': cmd_detector, 'rd': cmd_rd, 'gate': cmd_gate}[a.cmd](a)


if __name__ == '__main__':
    main()
