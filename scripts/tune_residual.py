"""Hyperparameter search for the matched-row AOBT_3 residual expert (strict blocked pairs).

The residual model predicts ``taxi - (take-off - AOBT_3)`` on clean matched rows and sets the
matched-lane expert ``aobt3 + r``; matched body rows are about 56 % of strict squared error. The
model had never been tuned (LightGBM lr 0.08, 255 leaves, 660 rounds).

Protocol: a block's model trains on the other eight months (date bugs excluded, clean rows,
the strict target-free feature set) exactly like ``tail_experts.build_strict_frames``. Candidates are
searched on one block (early stopping may use that block), and the chosen configuration is then
confirmed on the other two blocks with its round count fixed. The score that matters is the
matched-lane pipeline RMSE: the strict mixture ``p*D + (1-p)*(aobt3 + r)`` with the saved
context-detector probability, on every matched row of the held pair.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common
import fallback_context as fc
import strict_tail_gate as stg
import tail_experts as te
import train_eval
from common import RUNS

TUNE_DIR = RUNS / 'tuning'


def load_block(block, feature_version=5, frames_suffix='_v5fb5', data=None):
    """Training rows, scoring rows and the saved frame of one strict block."""
    left, right = block[0], block[1]
    if data is None:
        data, _, manifest = common.load_features(feature_version)
    else:
        manifest = common.read_manifest(feature_version)
    target_cols = set(manifest['groups'].get('G8', [])) | stg.TARGET_ROUTE | stg.TARGET_G9
    cols = [c for c in train_eval.feature_columns(manifest, list(manifest['groups'])) if c not in target_cols]
    cats = train_eval._cats(manifest, cols)
    bug = te.datebug_label(data.TAXITIME_SEC_mvt.values, data.atot_minus_sched.values)
    blocked = common.FOLDS[left] + common.FOLDS[right]
    train_mask = ~data.month.isin(blocked).values
    clean = train_eval._clean(data[train_mask & ~bug])
    cm = clean[np.isfinite(clean.atot_minus_aobt3.values)]
    held = data[~train_mask]
    frame = pd.concat([pd.read_parquet(fc._frame_path(left, right, f, frames_suffix)) for f in (left, right)], ignore_index=True)
    held = held.drop(columns=['p_fallback']).merge(
        frame[['MVT_ID_mvt', 'p_fallback', 'pred', 'normal_pred']].rename(columns={'pred': 'frame_pred'}),
        on='MVT_ID_mvt', how='left', validate='one_to_one')
    return {'cols': cols, 'cats': cats, 'train': cm, 'held': held, 'block': block}


def matched_metrics(held, residual):
    """Matched-lane pipeline RMSE and clean-row expert RMSE for a residual prediction on held rows."""
    a3 = held.atot_minus_aobt3.values.astype(float)
    m = np.isfinite(a3) & (held.is_heli.values == 0)
    y = held.TAXITIME_SEC_mvt.values.astype(float)
    normal = a3 + np.asarray(residual, float)
    pred = fc.mixture(held.p_fallback.values, held.atot_minus_sched.values, normal, held.is_heli.values, held.frame_pred.values)
    clean = m & (held.fallback.values == 0)
    return {'matched_rmse': common.rmse(pred[m], y[m]), 'clean_expert_rmse': common.rmse(normal[clean], y[clean]),
            'matched_body_rmse': common.rmse(pred[m & (y <= 3600)], y[m & (y <= 3600)]), 'n_matched': int(m.sum())}


def baseline_metrics(held):
    a3 = held.atot_minus_aobt3.values.astype(float)
    return matched_metrics(held, held.normal_pred.values - a3)


def fit_lightgbm(blk, params, rounds, early_stopping=None):
    import lightgbm as lgb
    tr = blk['train']; held = blk['held']
    X = train_eval.prep(tr, blk['cols'], blk['cats'])
    y = tr.TAXITIME_SEC_mvt.values - tr.atot_minus_aobt3.values
    m = np.isfinite(held.atot_minus_aobt3.values.astype(float))
    Xh = train_eval.prep(held[m], blk['cols'], blk['cats'], train_eval._cats_of(X, blk['cats']))
    p = train_eval._lgb_params(params)
    dtr = lgb.Dataset(X, y, categorical_feature=[c for c in blk['cats'] if c in X.columns], free_raw_data=False,
                      params={'max_bin': p.get('max_bin', 255)})
    callbacks, valid = [], []
    if early_stopping:
        # the held labels exist only on 2025 blocks; when deploying, ``held`` is the 2026 ranking table and has
        # none, which is why this is read here and not above (it is what early stopping needs, nothing else)
        yh = held.TAXITIME_SEC_mvt.values[m] - held.atot_minus_aobt3.values[m]
        # stop on the rows the residual expert is meant for: clean (not schedule-copied) fixed-wing matched rows
        es = (held.fallback.values[m] == 0) & (held.is_heli.values[m] == 0)
        valid = [lgb.Dataset(Xh[es], yh[es], reference=dtr)]
        callbacks = [lgb.early_stopping(early_stopping, verbose=False)]
    model = lgb.train(p, dtr, rounds, valid_sets=valid, callbacks=callbacks)
    r = np.full(len(held), np.nan)
    r[m] = model.predict(Xh, num_iteration=model.best_iteration or None)
    return r, int(model.best_iteration or rounds)


def fit_xgboost(blk, params, rounds, early_stopping=None):
    import xgboost as xgb
    tr = blk['train']; held = blk['held']
    X = train_eval.prep(tr, blk['cols'], blk['cats'])
    y = tr.TAXITIME_SEC_mvt.values - tr.atot_minus_aobt3.values
    m = np.isfinite(held.atot_minus_aobt3.values.astype(float))
    Xh = train_eval.prep(held[m], blk['cols'], blk['cats'], train_eval._cats_of(X, blk['cats']))
    yh = held.TAXITIME_SEC_mvt.values[m] - held.atot_minus_aobt3.values[m]
    base = dict(tree_method='hist', device='cuda', enable_categorical=True, max_cat_to_onehot=4, n_estimators=rounds,
                random_state=0, eval_metric='rmse')
    if early_stopping:
        base['early_stopping_rounds'] = early_stopping
    model = xgb.XGBRegressor(**{**base, **params})
    model.fit(X, y, eval_set=[(Xh, yh)] if early_stopping else None, verbose=False)
    best = getattr(model, 'best_iteration', None)
    r = np.full(len(held), np.nan)
    r[m] = model.predict(Xh, iteration_range=(0, best + 1) if (early_stopping and best is not None) else None)
    return r, int(best + 1 if (early_stopping and best is not None) else rounds)


def log_trial(path, row):
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(row) + '\n')


def done_trials(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
