"""Model zoo on the fixed fold protocol, including residual LightGBM/XGBoost experts."""
from __future__ import annotations
import json, time, pathlib
import numpy as np, pandas as pd, lightgbm as lgb
import xgboost as xgb
import common
from common import DATA, RUNS
from assemble import CAT_COLS

DEFAULT_PARAMS = dict(objective='regression', learning_rate=0.08, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                      bagging_freq=1, lambda_l2=10, cat_smooth=50, cat_l2=10, max_cat_to_onehot=4, verbose=-1, num_threads=0)
FLOOR = 30.0
NON_LGB = {'p_col', 'p_scale', 'seed', 'residual_params', 'residual_rounds'}
SAVE_DIR = None          # when set to a directory, every fitted booster is saved there (submission-time model persistence)
_SAVE_N = [0]


def rmse_np(p, y): return common.rmse(p, y)


def feature_columns(manifest, groups_on):
    return [c for g in groups_on for c in manifest['groups'][g]]


def prep(df, cols, cats, categories=None):
    X = df[cols].copy()
    for c in cats:
        if c in X: X[c] = pd.Categorical(X[c].astype(str), categories=None if categories is None else categories[c])
    return X


def _lgb_params(params):
    p = {**DEFAULT_PARAMS, **{k: v for k, v in params.items() if k not in NON_LGB}}
    if 'seed' in params: p.update(seed=params['seed'], bagging_seed=params['seed'], feature_fraction_seed=params['seed'])
    return p


def _save(model, kind, extra=None):
    if SAVE_DIR is None: return
    d = pathlib.Path(SAVE_DIR); d.mkdir(parents=True, exist_ok=True); n = _SAVE_N[0]; _SAVE_N[0] += 1
    if kind == 'lgb': model.save_model(str(d / f'{n:03d}_lgb.txt'))
    else: model.save_model(str(d / f'{n:03d}_xgb.json'))
    if extra: (d / f'{n:03d}_meta.json').write_text(json.dumps(extra), encoding='utf-8')


def fit_lgb(X, y, cats, params, rounds, tag=''):
    m = lgb.train(_lgb_params(params), lgb.Dataset(X, y, categorical_feature=[c for c in cats if c in X.columns]), rounds)
    _save(m, 'lgb', {'tag': tag, 'n_rows': int(len(X)), 'columns': list(X.columns), 'rounds': rounds}); return m


def _cats_of(X, cats): return {c: X[c].cat.categories for c in cats if c in X.columns}


def _m1(train, test, cols, cats, params, rounds, tag='M1'):
    X = prep(train, cols, cats); m = fit_lgb(X, train.TAXITIME_SEC_mvt.values, cats, params, rounds, tag)
    return m.predict(prep(test, cols, cats, _cats_of(X, cats)))


def _m2(train, test, cols, cats, params, rounds, tag='M2'):
    cols2 = [c for c in cols if c != 'APT']; out = np.full(len(test), np.nan)
    for apt in test.APT.unique():
        a = train[train.APT == apt]; b = test.APT.values == apt
        if len(a) < 500: out[b] = train.TAXITIME_SEC_mvt.mean(); continue
        X = prep(a, cols2, cats); m = fit_lgb(X, a.TAXITIME_SEC_mvt.values, cats, params, rounds, f'{tag}_{apt}'); out[b] = m.predict(prep(test[b], cols2, cats, _cats_of(X, cats)))
    return out


def _mixture(test, reg, params, heli_const):
    p = test[params.get('p_col', 'p_fallback')].values.astype(float)
    p = np.clip(p * float(params.get('p_scale', 1.0)), 0.0, 1.0)
    ams = test.atot_minus_sched.values.astype(float)
    ok = np.isfinite(ams) & (ams > 0) & np.isfinite(p); mix = np.where(ok, p * ams + (1 - p) * reg, reg)
    return np.where(test.is_heli.values == 1, heli_const, mix)


def _clean(train): return train[(train.fallback.values == 0) & (train.is_heli.values == 0)]


def _m3(train, test, cols, cats, params, rounds, heli_const, per_airport=False):
    clean = _clean(train); reg = _m2(clean, test, cols, cats, params, rounds, 'M3pa') if per_airport else _m1(clean, test, cols, cats, params, rounds, 'M3')
    return _mixture(test, reg, params, heli_const)


def _m3r(train, test, cols, cats, params, rounds, heli_const, per_airport=False,
         return_normal=False):
    clean = _clean(train); reg = _m2(clean, test, cols, cats, params, rounds, 'M3rpa') if per_airport else _m1(clean, test, cols, cats, params, rounds, 'M3r_global')
    cm = clean[np.isfinite(clean.atot_minus_aobt3.values)]; X = prep(cm, cols, cats)
    # the AOBT_3 residual expert may carry its own tuned parameters/rounds (params['residual_params'], params['residual_rounds'])
    r_params = {**params, **params.get('residual_params', {})}; r_rounds = int(params.get('residual_rounds', rounds))
    m = fit_lgb(X, cm.TAXITIME_SEC_mvt.values - cm.atot_minus_aobt3.values, cats, r_params, r_rounds, 'M3r_residual'); r = m.predict(prep(test, cols, cats, _cats_of(X, cats)))
    a3 = test.atot_minus_aobt3.values.astype(float); reg = np.where(np.isfinite(a3), a3 + r, reg)
    pred = _mixture(test, reg, params, heli_const)
    return (pred, reg) if return_normal else pred


def _m4(train, test, cols, cats, params, rounds, heli_const, min_rows=20000):
    clean = _clean(train); reg = _m1(clean, test, cols, cats, params, rounds, 'M4_global')
    def seg(df): return (df.fpl_matched.values.astype(int) * 2 + np.isfinite(df.taxi_dist_m.values.astype(float)).astype(int)) if 'taxi_dist_m' in df else df.fpl_matched.values.astype(int) * 2
    st, ss = seg(clean), seg(test)
    for s in np.unique(ss):
        a = clean[st == s]; b = ss == s
        if len(a) >= min_rows and b.any(): X = prep(a, cols, cats); reg[b] = fit_lgb(X, a.TAXITIME_SEC_mvt.values, cats, params, rounds, f'M4_seg{s}').predict(prep(test[b], cols, cats, _cats_of(X, cats)))
    return _mixture(test, reg, params, heli_const)


def _m5(train, test, cols, cats, params, rounds, heli_const):
    clean = _clean(train); X = prep(clean, cols, cats); Xt = prep(test, cols, cats, _cats_of(X, cats))
    m = xgb.XGBRegressor(tree_method='hist', enable_categorical=True, n_estimators=rounds, learning_rate=0.08, max_leaves=255, grow_policy='lossguide', max_depth=0,
                         min_child_weight=100, subsample=0.8, colsample_bytree=0.8, reg_lambda=10, random_state=params.get('seed', 0), n_jobs=0)
    m.fit(X, clean.TAXITIME_SEC_mvt.values); _save(m, 'xgb', {'tag': 'M5', 'n_rows': int(len(X)), 'columns': list(X.columns), 'rounds': rounds})
    return _mixture(test, m.predict(Xt), params, heli_const)


def _fit_xgb(X, y, params, rounds, tag):
    """Fit the native-categorical XGBoost learner used by M5 variants."""
    m = xgb.XGBRegressor(
        tree_method='hist', enable_categorical=True, n_estimators=rounds,
        learning_rate=0.08, max_leaves=255, grow_policy='lossguide', max_depth=0,
        min_child_weight=100, subsample=0.8, colsample_bytree=0.8, reg_lambda=10,
        random_state=params.get('seed', 0), n_jobs=0,
    )
    m.fit(X, y)
    _save(m, 'xgb', {'tag': tag, 'n_rows': int(len(X)), 'columns': list(X.columns),
                     'rounds': rounds})
    return m


def _m5r(train, test, cols, cats, params, rounds, heli_const, return_normal=False):
    """M5 with the AOBT_3 residual anchor used successfully by M3r.

    A direct XGBoost clean expert supplies rows without AOBT_3.  A second XGBoost
    learner predicts ``taxi_time - atot_minus_aobt3`` wherever that anchor exists.
    The usual fallback mixture is applied only after the clean expert is assembled.
    """
    clean = _clean(train)
    X = prep(clean, cols, cats)
    categories = _cats_of(X, cats)
    direct = _fit_xgb(X, clean.TAXITIME_SEC_mvt.values, params, rounds, 'M5r_direct')
    reg = direct.predict(prep(test, cols, cats, categories))

    anchored = clean[np.isfinite(clean.atot_minus_aobt3.values)]
    Xr = prep(anchored, cols, cats)
    residual = _fit_xgb(
        Xr,
        anchored.TAXITIME_SEC_mvt.values - anchored.atot_minus_aobt3.values,
        params, rounds, 'M5r_residual',
    )
    residual_pred = residual.predict(prep(test, cols, cats, _cats_of(Xr, cats)))
    a3 = test.atot_minus_aobt3.values.astype(float)
    reg = np.where(np.isfinite(a3), a3 + residual_pred, reg)
    pred = _mixture(test, reg, params, heli_const)
    return (pred, reg) if return_normal else pred


def predict_model(model_id, train, test, cols, cats, params, rounds, heli_const,
                  return_normal=False):
    if model_id == 'M1': p = _m1(train, test, cols, cats, params, rounds)
    elif model_id == 'M2': p = _m2(train, test, cols, cats, params, rounds)
    elif model_id == 'M3': p = _m3(train, test, cols, cats, params, rounds, heli_const)
    elif model_id == 'M3pa': p = _m3(train, test, cols, cats, params, rounds, heli_const, per_airport=True)
    elif model_id == 'M3r': p = _m3r(train, test, cols, cats, params, rounds, heli_const,
                                     return_normal=return_normal)
    elif model_id == 'M3rpa': p = _m3r(train, test, cols, cats, params, rounds, heli_const,
                                       per_airport=True, return_normal=return_normal)
    elif model_id == 'M4': p = _m4(train, test, cols, cats, params, rounds, heli_const)
    elif model_id == 'M5': p = _m5(train, test, cols, cats, params, rounds, heli_const)
    elif model_id == 'M5r': p = _m5r(train, test, cols, cats, params, rounds, heli_const,
                                     return_normal=return_normal)
    else: raise ValueError(model_id)
    if return_normal:
        if not isinstance(p, tuple):
            raise ValueError(f'normal expert output is not supported for {model_id}')
        pred, normal = p
        return np.maximum(np.asarray(pred, dtype=float), FLOOR), np.asarray(normal, dtype=float)
    return np.maximum(np.asarray(p, dtype=float), FLOOR)


def _heli_const():
    return float(json.load(open(DATA / 'det_heli.json'))['heli_const']) if (DATA / 'det_heli.json').exists() else 120.0


def _cats(man, cols): return [c for c in man.get('cat_cols', CAT_COLS) if c in cols]


def run(model_id, feature_version, groups_on, params=None, rounds=600, folds=('A', 'B'), tag='', save_oof=True):
    params = params or {}; tr, rk, man = common.load_features(feature_version); cols = feature_columns(man, groups_on)
    cats = _cats(man, cols); run_id = common.new_run_id(f'{model_id}_v{feature_version}{("_" + tag) if tag else ""}')
    for fold in folds:
        trm, tem = common.fold_masks(tr.month, fold); a, b = tr[trm], tr[tem]; t0 = time.time()
        predicted = predict_model(model_id, a, b, cols, cats, params, rounds, _heli_const(),
                                  return_normal=save_oof and model_id in {'M3r', 'M3rpa', 'M5r'})
        p, normal = predicted if isinstance(predicted, tuple) else (predicted, None)
        y = b.TAXITIME_SEC_mvt.values
        m = common.metrics(p, y, b.APT.values, b.month.values, b.fpl_matched.values)
        row = dict(run_id=run_id, ts=time.strftime('%Y-%m-%d %H:%M:%S'), feature_version=feature_version, groups_on='+'.join(groups_on), model=model_id,
                   params_json=json.dumps(params), fold=fold, rmse_jan=m.pop('rmse_m1'), rmse_jul=m.pop('rmse_m2'), n_train=int(trm.sum()), n_test=int(tem.sum()),
                   fit_seconds=round(time.time() - t0, 1), notes=tag, seed=params.get('seed', 0), **m)
        common.append_result(row)
        if save_oof:
            oof = pd.DataFrame({'MVT_ID_mvt': b.MVT_ID_mvt.values, 'APT': b.APT.values,
                                'month': b.month.values, 'y': y, 'pred': p,
                                'fpl_matched': b.fpl_matched.values})
            if normal is not None: oof['normal_pred'] = normal
            oof.to_parquet(RUNS / 'oof' / f'{run_id}_fold{fold}.parquet')
        print(f'{run_id} fold {fold}: RMSE {row["rmse_raw"]:.1f} m1 {row["rmse_jan"]:.1f} m2 {row["rmse_jul"]:.1f} matched {row["rmse_matched"]:.1f} unmatched {row["rmse_unmatched"]:.1f} body {row["rmse_body"]:.1f} top100 {row["ss_top100_share"]:.2f} [{row["fit_seconds"]}s]', flush=True)
    return run_id


def fit_full(model_id, feature_version, groups_on, params=None, rounds=660):
    """Train on all 2025 and return a callable state for predict_full."""
    params = params or {}; tr, rk, man = common.load_features(feature_version); cols = feature_columns(man, groups_on); cats = _cats(man, cols)
    return dict(model_id=model_id, train=tr, cols=cols, cats=cats, params=params, rounds=rounds, heli_const=_heli_const(), feature_version=feature_version, groups_on=groups_on)


def predict_full(state, rank_df):
    return predict_model(state['model_id'], state['train'], rank_df, state['cols'], state['cats'], state['params'], state['rounds'], state['heli_const'])
