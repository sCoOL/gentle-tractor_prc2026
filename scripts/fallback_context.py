"""Fallback (schedule-copy) detector with stand-timeline and operations-context inputs.

The v1 detector (``detectors.FALLBACK_FEATS``) predates G10/G11. Two new rank-available signals
separate copies at LIRF: the airline's arrival schedule-copy rate that day (matched AUC 0.64,
top decile 40 % copies vs 13 %) and the inbound turnaround arrival's own copied in-block
(38.1 % vs 15.4 %). This module refits the detector per strict block with those inputs, writes
candidate strict frames whose mixture uses the new probability, and leaves every other model
input untouched, so the strict three-expert diagnostic isolates the detector change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
import detectors
import strict_tail_gate as stg
from common import DATA, RUNS

G10 = ['st_last_in_gap', 'st_next_in_gap', 'st_n_in_60', 'st_last_same_type', 'st_last_same_al', 'st_turn_in_gap',
       'st_turn_in_delay', 'st_prev_dep_gap']
G11 = ['cx_arr_copy_rate_day', 'cx_arr_copy_rate_al_day', 'cx_arr_copy_rate_c180', 'cx_arr_delay_med_c180', 'st_turn_in_is_copy',
       'cx_dep_a3taxi_med_c60', 'cx_arr_taxiin_med_c60']
EXTRA = ['airline', 'flight_is_callsign', 'sched_eq_eobt1', 'lobt_minus_iobt', 'dep_same_rwy_prev30', 'arr_prev60']
FEATS = detectors.FALLBACK_FEATS + EXTRA + G10 + G11
CATS = detectors.FALLBACK_CATS + ['airline']


def _frame_path(left, right, fold, suffix):
    return RUNS / 'oof' / f'strict_block_{left}{right}_fold{fold}{suffix}.parquet'


def score_seeds(train, score, feats=FEATS, cats=CATS, rounds=400, seeds=None, params=None):
    """Detector probability, averaged over LightGBM seeds when ``seeds`` is given (``params`` override CLF_PARAMS)."""
    if not seeds:
        return detectors.score_fallback(train, score, feature_cols=feats, cat_cols=cats, rounds=rounds, params=dict(params or {}))
    parts = [detectors.score_fallback(train, score, feature_cols=feats, cat_cols=cats, rounds=rounds,
                                      params={**dict(params or {}), 'seed': int(s), 'bagging_seed': int(s), 'feature_fraction_seed': int(s)})
             for s in seeds]
    return np.mean(parts, axis=0)


def mixture(p, D, normal, is_heli, heli_pred, floor=30.0):
    """Strict-frame mixture exactly as ``train_eval._mixture`` with p_scale 1 (helicopters keep their constant)."""
    p = np.clip(np.asarray(p, float), 0, 1); D = np.asarray(D, float); normal = np.asarray(normal, float)
    ok = np.isfinite(D) & (D > 0) & np.isfinite(p)
    mix = np.where(ok, p * D + (1 - p) * normal, normal)
    mix = np.where(np.asarray(is_heli) == 1, heli_pred, mix)
    return np.maximum(mix, floor)


def build_candidate_frames(feature_version, source_suffix, out_suffix, rounds=400, feats=FEATS, cats=CATS, seeds=None, params=None):
    """Refit the detector per block (other eight months), rewrite frames with the new probability.

    ``seeds`` (e.g. [11, 22, 33]) averages detectors fitted with those LightGBM seeds; None keeps the
    library default seed, as every earlier detector did.
    """
    data, _, _ = common.load_features(feature_version)
    missing = [c for c in feats if c not in data.columns]
    if missing:
        raise ValueError(f'detector features missing from v{feature_version}: {missing}')
    meta = {'feature_version': feature_version, 'source_suffix': source_suffix, 'features': list(feats), 'blocks': {}}
    for left, right in stg.BLOCKS:
        blocked = common.FOLDS[left] + common.FOLDS[right]
        train_mask = ~data.month.isin(blocked).values
        score = data[~train_mask]
        p_new = score_seeds(data[train_mask], score, feats, cats, rounds, seeds, params)
        pmap = pd.Series(p_new, index=score.MVT_ID_mvt.values)
        block = {}
        for fold in (left, right):
            src = pd.read_parquet(_frame_path(left, right, fold, source_suffix))
            info = src[['MVT_ID_mvt']].merge(score[['MVT_ID_mvt', 'atot_minus_sched', 'is_heli', 'fallback']], on='MVT_ID_mvt',
                                             how='left', validate='one_to_one')
            D = info.atot_minus_sched.values; heli = info.is_heli.values
            old = mixture(src.p_fallback.values, D, src.normal_pred.values, heli, src.pred.values)
            check = float(np.abs(old - src.pred.values).max())
            if check > 1e-6:
                raise ValueError(f'fold {fold}: mixture does not reproduce the source frame ({check})')
            p = src.MVT_ID_mvt.map(pmap).values
            if not np.isfinite(p).all():
                raise ValueError(f'fold {fold}: missing new probabilities')
            out = src.copy()
            out['p_fallback'] = p
            out['pred'] = mixture(p, D, src.normal_pred.values, heli, src.pred.values)
            out.to_parquet(_frame_path(left, right, fold, out_suffix), index=False)
            y = info.fallback.values.astype(int)
            block[fold] = {'auc_old': detectors._auc(y, src.p_fallback.values), 'auc_new': detectors._auc(y, p),
                           'logloss_old': _logloss(y, src.p_fallback.values), 'logloss_new': _logloss(y, p),
                           'lirf_matched_auc_old': _sub_auc(y, src.p_fallback.values, info, src, 'LIRF', 1),
                           'lirf_matched_auc_new': _sub_auc(y, p, info, src, 'LIRF', 1),
                           'lirf_unmatched_auc_old': _sub_auc(y, src.p_fallback.values, info, src, 'LIRF', 0),
                           'lirf_unmatched_auc_new': _sub_auc(y, p, info, src, 'LIRF', 0)}
        meta['blocks'][f'{left}{right}'] = block
    return meta


def transplant_probability(feature_version, source_suffix, donor_suffix, out_suffix):
    """Frames of ``source_suffix`` with the detector probability of ``donor_suffix`` (same strict blocks).

    The detector does not read G12+ columns, so a detector fitted for one frame family is valid for
    another; only the mixture is recomputed.
    """
    data = pd.read_parquet(DATA / f'features_v{feature_version}_train.parquet', columns=['MVT_ID_mvt', 'atot_minus_sched', 'is_heli'])
    for left, right in stg.BLOCKS:
        for fold in (left, right):
            src = pd.read_parquet(_frame_path(left, right, fold, source_suffix))
            donor = pd.read_parquet(_frame_path(left, right, fold, donor_suffix), columns=['MVT_ID_mvt', 'p_fallback'])
            info = src[['MVT_ID_mvt']].merge(data, on='MVT_ID_mvt', how='left', validate='one_to_one')
            old = mixture(src.p_fallback.values, info.atot_minus_sched.values, src.normal_pred.values, info.is_heli.values, src.pred.values)
            if float(np.abs(old - src.pred.values).max()) > 1e-6:
                raise ValueError(f'fold {fold}: mixture does not reproduce the source frame')
            p = src[['MVT_ID_mvt']].merge(donor, on='MVT_ID_mvt', how='left', validate='one_to_one').p_fallback.values
            if not np.isfinite(p).all():
                raise ValueError(f'fold {fold}: donor probabilities missing')
            out = src.copy()
            out['p_fallback'] = p
            out['pred'] = mixture(p, info.atot_minus_sched.values, src.normal_pred.values, info.is_heli.values, src.pred.values)
            out.to_parquet(_frame_path(left, right, fold, out_suffix), index=False)


def _logloss(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _sub_auc(y, p, info, src, apt, lane):
    m = (src.APT.values == apt) & (src.fpl_matched.values == lane) & (info.is_heli.values == 0)
    return detectors._auc(y[m], np.asarray(p)[m])
