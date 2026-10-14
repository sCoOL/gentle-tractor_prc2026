"""AOBT_3-residual expert variants for matched rows under the strict blocked-pair protocol.

M3rpa predicts matched rows as ``atot_minus_aobt3 + r`` with one global LightGBM residual model
(``M3r_residual``), while rows without AOBT_3 use per-airport clean experts. Matched body rows are
56 % of strict squared error. This module fits per-airport residual models on the same clean
training rows (other eight months, date bugs excluded, same features and parameters) and writes two
frame families next to an existing global-residual family:

  <source>ra   normal = aobt3 anchor + per-airport residual
  <source>rb   normal = aobt3 anchor + mean(global, per-airport) residual

Rows without AOBT_3, helicopters and the fallback probability are unchanged, so the strict
three-expert diagnostic isolates the residual expert.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common
import fallback_context as fc
import strict_tail_gate as stg
import tail_experts as te
import train_eval
from common import RUNS


def build(feature_version, source_suffix, rounds=660, params=None, exclusion='datebug'):
    data, _, manifest = common.load_features(feature_version)
    target_cols = set(manifest['groups'].get('G8', [])) | stg.TARGET_ROUTE | stg.TARGET_G9
    cols = [c for c in train_eval.feature_columns(manifest, list(manifest['groups'])) if c not in target_cols]
    cats = train_eval._cats(manifest, cols)
    bug = te.training_exclusion(data.TAXITIME_SEC_mvt.values, data.atot_minus_sched.values, exclusion)
    meta = {'feature_version': feature_version, 'source_suffix': source_suffix, 'blocks': {}}
    for left, right in stg.BLOCKS:
        blocked = common.FOLDS[left] + common.FOLDS[right]
        train_mask = ~data.month.isin(blocked).values
        clean = train_eval._clean(data[train_mask & ~bug])
        cm = clean[np.isfinite(clean.atot_minus_aobt3.values)]
        score = data[~train_mask]
        a3 = score.atot_minus_aobt3.values.astype(float)
        r_apt = np.full(len(score), np.nan)
        for apt in score.APT.unique():
            a = cm[cm.APT.values == apt]
            b = (score.APT.values == apt) & np.isfinite(a3)
            if not b.any():
                continue
            X = train_eval.prep(a, cols, cats)
            m = train_eval.fit_lgb(X, a.TAXITIME_SEC_mvt.values - a.atot_minus_aobt3.values, cats, dict(params or {}), rounds, f'M3r_residual_{apt}')
            r_apt[b] = m.predict(train_eval.prep(score[b], cols, cats, train_eval._cats_of(X, cats)))
        rmap = pd.Series(r_apt, index=score.MVT_ID_mvt.values)
        amap = pd.Series(a3, index=score.MVT_ID_mvt.values)
        info_cols = score[['MVT_ID_mvt', 'atot_minus_sched', 'is_heli']]
        block = {}
        for fold in (left, right):
            src = pd.read_parquet(fc._frame_path(left, right, fold, source_suffix))
            info = src[['MVT_ID_mvt']].merge(info_cols, on='MVT_ID_mvt', how='left', validate='one_to_one')
            anchor = src.MVT_ID_mvt.map(amap).values
            ra = src.MVT_ID_mvt.map(rmap).values
            has = np.isfinite(anchor)
            if not np.isfinite(ra[has]).all():
                raise ValueError(f'fold {fold}: missing per-airport residual predictions')
            rg = src.normal_pred.values - anchor           # global residual implied by the saved expert
            for tag, r in (('ra', ra), ('rb', 0.5 * (rg + ra))):
                out = src.copy()
                normal = np.where(has, anchor + r, src.normal_pred.values)
                out['normal_pred'] = normal
                out['pred'] = fc.mixture(src.p_fallback.values, info.atot_minus_sched.values, normal, info.is_heli.values, src.pred.values)
                out.to_parquet(fc._frame_path(left, right, fold, f'{source_suffix}{tag}'), index=False)
            block[fold] = {'matched_rows': int(has.sum()),
                           'rms_global_vs_airport_residual': float(np.sqrt(np.nanmean((rg[has] - ra[has]) ** 2)))}
        meta['blocks'][f'{left}{right}'] = block
    return meta


def rank_blend_members(feature_version, cols, cats, seed=1, rounds=660, params=None, airport_params=None, airport_rounds=660,
                       exclusion='datebug'):
    """Deployment members of the rd blend on ranking rows: ``(ids, aobt3 anchor, r_global_seed, r_airport, is_heli)``.

    The members are fitted on the clean all-2025 rows (``exclusion`` rows left out) with ``cols``/``cats``.
    ``params``/``rounds`` set the global seed-``seed`` member, ``airport_params``/``airport_rounds`` the
    per-airport members, matching how the strict frames were built.
    """
    train, rank, _ = common.load_features(feature_version)
    bug = te.training_exclusion(train.TAXITIME_SEC_mvt.values, train.atot_minus_sched.values, exclusion)
    clean = train_eval._clean(train[~bug])
    cm = clean[np.isfinite(clean.atot_minus_aobt3.values)]
    a3 = rank.atot_minus_aobt3.values.astype(float)
    has = np.isfinite(a3)
    X = train_eval.prep(cm, cols, cats)
    m = train_eval.fit_lgb(X, cm.TAXITIME_SEC_mvt.values - cm.atot_minus_aobt3.values, cats, {**dict(params or {}), 'seed': seed}, rounds,
                           f'M3r_residual_s{seed}')
    rg1 = np.full(len(rank), np.nan)
    rg1[has] = m.predict(train_eval.prep(rank[has], cols, cats, train_eval._cats_of(X, cats)))
    ra = np.full(len(rank), np.nan)
    for apt in rank.APT.unique():
        a = cm[cm.APT.values == apt]
        b = (rank.APT.values == apt) & has
        if not b.any():
            continue
        Xa = train_eval.prep(a, cols, cats)
        ma = train_eval.fit_lgb(Xa, a.TAXITIME_SEC_mvt.values - a.atot_minus_aobt3.values, cats, dict(airport_params or {}), airport_rounds,
                                f'M3r_residual_{apt}')
        ra[b] = ma.predict(train_eval.prep(rank[b], cols, cats, train_eval._cats_of(Xa, cats)))
    return rank.MVT_ID_mvt.values, a3, rg1, ra, rank.is_heli.values, rank.APT.values


def blend_members(normal, a3, rg1, ra, is_heli, apt=None, weights=None):
    """``aobt3 + w0 r_global + w1 r_global_seed + wa r_airport`` on anchored fixed-wing rows (default 0.25/0.25/0.5)."""
    normal = np.asarray(normal, float)
    rg0 = normal - a3
    w = np.tile(np.array([0.25, 0.25, 0.5]), (len(normal), 1))
    if weights:
        w = np.array([weights.get(x, (0.25, 0.25, 0.5)) for x in np.asarray(apt)], float)
    has = np.isfinite(a3)
    out = np.where(has, a3 + w[:, 0] * rg0 + w[:, 1] * rg1 + w[:, 2] * ra, normal)
    out = np.where(np.asarray(is_heli) == 1, normal, out)
    if not np.isfinite(out).all():
        raise ValueError('non-finite blended rank expert')
    return out


def rank_blend_normal(feature_version, cols, cats, normal, seed=1, rounds=660, params=None, airport_params=None, airport_rounds=660,
                      exclusion='datebug'):
    """Deployment: replace the matched-row expert ``aobt3 + r_global`` by the rd blend on ranking rows.

    ``normal`` is the full-2025 M3rpa expert fitted with ``cols``/``cats``; the extra members come from
    ``rank_blend_members`` and are blended 0.25 / 0.25 / 0.5.
    """
    ids, a3, rg1, ra, is_heli, _ = rank_blend_members(feature_version, cols, cats, seed, rounds, params, airport_params, airport_rounds, exclusion)
    return ids, blend_members(normal, a3, rg1, ra, is_heli)


def build_seed_blend(feature_version, source_suffix, seed=1, rounds=660, params=None, exclusion='datebug'):
    """<source>rd: 0.25 global (default seed) + 0.25 global (``seed``) + 0.5 per-airport residual.

    The same 50/50 global/airport structure as <source>rb, with the global half averaged over two
    LightGBM seeds; decided before scoring, so no blend weight is tuned on the strict folds.
    Needs the <source>ra frames from ``build``.
    """
    data, _, manifest = common.load_features(feature_version)
    target_cols = set(manifest['groups'].get('G8', [])) | stg.TARGET_ROUTE | stg.TARGET_G9
    cols = [c for c in train_eval.feature_columns(manifest, list(manifest['groups'])) if c not in target_cols]
    cats = train_eval._cats(manifest, cols)
    bug = te.training_exclusion(data.TAXITIME_SEC_mvt.values, data.atot_minus_sched.values, exclusion)
    meta = {'feature_version': feature_version, 'source_suffix': source_suffix, 'seed': seed, 'blocks': {}}
    for left, right in stg.BLOCKS:
        blocked = common.FOLDS[left] + common.FOLDS[right]
        train_mask = ~data.month.isin(blocked).values
        clean = train_eval._clean(data[train_mask & ~bug])
        cm = clean[np.isfinite(clean.atot_minus_aobt3.values)]
        score = data[~train_mask]
        a3 = score.atot_minus_aobt3.values.astype(float)
        has_score = np.isfinite(a3)
        X = train_eval.prep(cm, cols, cats)
        m = train_eval.fit_lgb(X, cm.TAXITIME_SEC_mvt.values - cm.atot_minus_aobt3.values, cats, {**dict(params or {}), 'seed': seed}, rounds,
                               f'M3r_residual_s{seed}')
        r1 = np.full(len(score), np.nan)
        r1[has_score] = m.predict(train_eval.prep(score[has_score], cols, cats, train_eval._cats_of(X, cats)))
        rmap = pd.Series(r1, index=score.MVT_ID_mvt.values)
        amap = pd.Series(a3, index=score.MVT_ID_mvt.values)
        info_cols = score[['MVT_ID_mvt', 'atot_minus_sched', 'is_heli']]
        block = {}
        for fold in (left, right):
            src = pd.read_parquet(fc._frame_path(left, right, fold, source_suffix))
            ra_frame = pd.read_parquet(fc._frame_path(left, right, fold, f'{source_suffix}ra'), columns=['MVT_ID_mvt', 'normal_pred'])
            info = src[['MVT_ID_mvt']].merge(info_cols, on='MVT_ID_mvt', how='left', validate='one_to_one')
            anchor = src.MVT_ID_mvt.map(amap).values
            has = np.isfinite(anchor)
            rg0 = src.normal_pred.values - anchor
            rg1 = src.MVT_ID_mvt.map(rmap).values
            ra = src[['MVT_ID_mvt']].merge(ra_frame, on='MVT_ID_mvt', how='left', validate='one_to_one').normal_pred.values - anchor
            if not (np.isfinite(rg1[has]).all() and np.isfinite(ra[has]).all()):
                raise ValueError(f'fold {fold}: missing residual members')
            normal = np.where(has, anchor + 0.25 * rg0 + 0.25 * rg1 + 0.5 * ra, src.normal_pred.values)
            out = src.copy()
            out['normal_pred'] = normal
            out['pred'] = fc.mixture(src.p_fallback.values, info.atot_minus_sched.values, normal, info.is_heli.values, src.pred.values)
            out.to_parquet(fc._frame_path(left, right, fold, f'{source_suffix}rd'), index=False)
            block[fold] = {'rms_seed0_vs_seed1': float(np.sqrt(np.mean((rg0[has] - rg1[has]) ** 2)))}
        meta['blocks'][f'{left}{right}'] = block
    return meta
