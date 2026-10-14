"""Past-only all-airport normal-expert screen for v49; never writes a submission.

The baseline is a freshly fitted ADS-B residual tree, NOT v48. A passing result
only warrants a full ensemble comparison; it cannot qualify a v49 submission.
No official results or scored-submission geometry are read.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import adsb_context as adsb
import common
import strict_tail_gate as stg
import tail_experts as te
import train_eval as tr

OUT = common.RUNS / 'v49_history_normal'
WINDOWS = {'summer': (4, [5, 6], [7, 8]), 'winter': (8, [9, 10], [11, 12])}
KEYS = {'airport': ['APT'], 'airline': ['APT', 'airline'],
        'route': ['APT', 'airline', 'ADES_mvt'],
        'flight': ['APT', 'FLIGHT_mvt', 'ADES_mvt'],
        'stand': ['APT', 'STAND_mvt'], 'runway': ['APT', 'RUNWAY_mvt']}
HISTORY_COLS = [f'past_{name}_{stat}' for name in KEYS for stat in ['mean', 'n']]
PARAMS = dict(objective='regression', learning_rate=.05, num_leaves=255,
              min_data_in_leaf=100, feature_fraction=.85, bagging_fraction=1.,
              lambda_l2=30., cat_smooth=200., cat_l2=1., max_cat_threshold=128,
              max_bin=255, seed=20260923, num_threads=4, verbosity=-1)
ROUNDS = 600
ALPHAS = [0., .1, .25, .5, 1.]
LABEL = 'TAXITIME_SEC_mvt'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def id_digest(values):
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def clean_mask(d):
    return (d.fallback.eq(0) & d.is_heli.eq(0) & np.isfinite(d.atot_minus_aobt3)
            & ~te.datebug_label(d[LABEL], d.atot_minus_sched))


def history_features(frame, history):
    """All target statistics use strictly earlier months, never scored labels."""
    if len(history):
        if history.month.max() >= frame.month.min():
            raise ValueError('history must precede every scored month')
        if np.intersect1d(frame.MVT_ID_mvt, history.MVT_ID_mvt).size:
            raise ValueError('history and scoring IDs overlap')
    out = frame.copy()
    h = history.loc[clean_mask(history)].copy() if len(history) else history.copy()
    h['_residual'] = (h[LABEL] - h.atot_minus_aobt3).clip(-3600., 3600.)
    # Clipping protects the historical feature, not the regression labels/loss.
    for name, keys in KEYS.items():
        mu, count = f'past_{name}_mean', f'past_{name}_n'
        prior = np.zeros(len(out)) if name == 'airport' else out.past_airport_mean.to_numpy()
        if h.empty:
            out[mu] = prior
            out[count] = 0.
            continue
        stats = h.groupby(keys, observed=True, dropna=False)._residual.agg(['sum', 'size']).reset_index()
        stats = stats.rename(columns={'sum': '_sum', 'size': '_n'})
        mapped = out[keys].merge(stats, on=keys, how='left', validate='many_to_one')
        n = mapped._n.fillna(0).to_numpy()
        out[mu] = ((mapped._sum.fillna(0).to_numpy() + 50. * prior) / (n + 50.)).astype('float32')
        out[count] = n.astype('float32')
    return out


def rolling_history(train):
    parts = []
    for month in sorted(train.month.unique()):
        parts.append(history_features(train.loc[train.month.eq(month)], train.loc[train.month.lt(month)]))
    return pd.concat(parts).reindex(train.index)


def same_schedule_month(d):
    """Exclude movements spanning month/year boundaries from this time screen."""
    return d.SCHED_TIME_UTC_mvt.dt.year.eq(2025) & d.SCHED_TIME_UTC_mvt.dt.month.eq(d.month)


def load_window(window):
    cutoff, cal, test = WINDOWS[window]
    manifest = common.read_manifest(5)
    forbidden = set(manifest['groups']['G8']) | stg.TARGET_ROUTE | stg.TARGET_G9
    # These arrival-derived rates have a different batch definition in the
    # historic table; exclude them from both sides of this controlled screen.
    cols = [c for c in tr.feature_columns(manifest, list(manifest['groups']))
            if c not in forbidden and not c.startswith('arr_fb_rate_')]
    need = list(dict.fromkeys(cols + ['MVT_ID_mvt', 'month', 'APT', LABEL,
                                    'fallback', 'is_heli', 'fpl_matched']))
    d = pd.read_parquet(common.DATA / 'features_v5_train.parquet', columns=need,
                        filters=[('month', '<=', max(test))])
    raw_cols = list(dict.fromkeys(['MVT_ID_mvt', 'SCHED_TIME_UTC_mvt'] + [k for keys in KEYS.values() for k in keys if k not in d]))
    raw = pd.read_parquet(common.DATA / 'base_train.parquet', columns=raw_cols)
    d = d.merge(raw, on='MVT_ID_mvt', validate='one_to_one')
    d = d.loc[same_schedule_month(d)].copy()
    del raw
    # Compute history on a narrow frame rather than copying the wide feature
    # table for every month. Preserve its original index when joining below.
    keys = list(dict.fromkeys(['MVT_ID_mvt', 'month', LABEL, 'atot_minus_aobt3',
        'atot_minus_sched', 'fallback', 'is_heli'] + [k for ks in KEYS.values() for k in ks]))
    ht = d.loc[d.month.le(cutoff), keys]
    hs = d.loc[d.month.gt(cutoff), keys]
    features = pd.concat([rolling_history(ht)[HISTORY_COLS], history_features(hs, ht)[HISTORY_COLS]])
    d = d.join(features)
    del ht, hs, features
    obs = adsb.COLUMNS + adsb.STOPS_COLUMNS + adsb.GEOM_COLUMNS + adsb.NBR_COLUMNS
    for read in [adsb.table, adsb.stops_table, adsb.geom_table, adsb.nbr_table]:
        d = d.merge(read(), on='MVT_ID_mvt', how='left', validate='one_to_one')
    cols += obs
    for c in cols:
        if pd.api.types.is_float_dtype(d[c]) and c not in ['atot_minus_aobt3', 'atot_minus_sched']:
            d[c] = d[c].astype('float32')
    train = d.loc[d.month.le(cutoff)].copy()
    held = d.loc[d.month.gt(cutoff)].copy()
    del d
    gc.collect()
    assert train.SCHED_TIME_UTC_mvt.max() < held.SCHED_TIME_UTC_mvt.min()
    assert not np.intersect1d(train.MVT_ID_mvt, held.MVT_ID_mvt).size
    train = train.loc[clean_mask(train)].copy()
    held = held.loc[np.isfinite(held.atot_minus_aobt3) & held.is_heli.eq(0)].copy()
    return train, held, cols, tr._cats(manifest, cols), sorted(forbidden)


def predict(model, state, frame):
    X = tr.prep(frame, state['columns'], state['cats'], state['categories'])
    return model.predict(X, num_threads=4) + frame.atot_minus_aobt3.to_numpy()


def fit_predict(folder, train, held, cols, cats, input_fingerprint=None):
    folder.mkdir(parents=True, exist_ok=True)
    path, statepath = folder/'model.txt', folder/'state.json'
    if path.exists():
        state = json.loads(statepath.read_text())
        assert state['columns'] == cols
        assert state['train_ids_sha256'] == id_digest(train.MVT_ID_mvt)
        assert state['parameters'] == PARAMS and state['rounds'] == ROUNDS
        assert state.get('input_fingerprint') == input_fingerprint, 'cached model inputs/code changed'
        model = lgb.Booster(model_file=str(path))
    else:
        X = tr.prep(train, cols, cats)
        state = {'columns': cols, 'cats': cats,
                 'categories': {c: list(v) for c, v in tr._cats_of(X, cats).items()},
                 'train_ids_sha256': id_digest(train.MVT_ID_mvt),
                 'parameters': PARAMS, 'rounds': ROUNDS,
                 'input_fingerprint': input_fingerprint}
        model = lgb.train(PARAMS, lgb.Dataset(X, train[LABEL]-train.atot_minus_aobt3,
                          categorical_feature=cats), num_boost_round=ROUNDS)
        model.save_model(str(path))
        statepath.write_text(json.dumps(state, indent=2))
        del X
        gc.collect()
    p = predict(model, state, held)
    poisoned = held.copy()
    for c in [LABEL, 'fallback']:
        poisoned[c] = 1e12
    np.testing.assert_array_equal(p, predict(model, state, poisoned))
    replay = predict(lgb.Booster(model_file=str(path)), state, held)
    np.testing.assert_array_equal(p, replay)
    assert np.isfinite(p).all()
    return p


def metric(frame, control, candidate, name):
    y = frame[LABEL].to_numpy()
    a, b = (control-y)**2, (candidate-y)**2
    keep = np.ones(len(y), dtype=bool)
    keep[np.argsort(-np.abs(b-a))[:10]] = False
    return {'slice': name, 'n': len(y), 'control_rmse': float(np.sqrt(a.mean())),
            'candidate_rmse': float(np.sqrt(b.mean())),
            'delta': float(np.sqrt(b.mean())-np.sqrt(a.mean())),
            'delta_without_top10': float(np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean()))}


def run(window):
    folder = OUT/window
    folder.mkdir(parents=True, exist_ok=True)
    protocol = {'window': window, 'months': WINDOWS[window], 'parameters': PARAMS,
                'rounds': ROUNDS, 'alphas': ALPHAS, 'history_keys': KEYS,
                'history_prior_strength': 50., 'history_feature_clip': [-3600., 3600.],
                'month_boundary_rule': 'Exclude schedule year/month mismatches before splitting; all training timestamps precede scoring timestamps.',
                'scope': 'Normal-expert screen only; newly fitted ADS-B tree control, NOT v48.',
                'selection': 'Global alpha selected on calibration clean-row SSE only.',
                'promotion': 'Both seasons and each test month, including top10 removal, must improve before ensemble testing.',
                'official_results_read': False}
    # Content fingerprints prevent a resume from silently using a model trained
    # before an input table, dependency, or experiment definition changed.
    paths = [common.DATA/'features_v5_train.parquet', common.DATA/'base_train.parquet',
             adsb.PATH, adsb.STOPS_PATH, adsb.GEOM_PATH, adsb.NBR_PATH,
             Path(__file__), Path(tr.__file__), Path(te.__file__), Path(adsb.__file__)]
    provenance = {str(p): digest(p) for p in paths}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    protocol.update(input_sha256=provenance, input_fingerprint=fingerprint)
    if (folder/'protocol.json').exists() and (folder/'control/model.txt').exists():
        previous = json.loads((folder/'protocol.json').read_text())
        assert previous.get('input_fingerprint') == fingerprint, 'resume fingerprint mismatch; preserve this run and use a fresh directory'
    (folder/'protocol.json').write_text(json.dumps(protocol, indent=2))
    print('PREPARE', window, flush=True)
    train, held, cols, cats, excluded = load_window(window)
    protocol.update(train_ids_sha256=id_digest(train.MVT_ID_mvt),
                    score_ids_sha256=id_digest(held.MVT_ID_mvt), excluded=excluded,
                    source_sha256=digest(__file__), train_rows=len(train), score_rows=len(held))
    (folder/'protocol.json').write_text(json.dumps(protocol, indent=2))
    (folder/'157_history_normal.executed.py').write_bytes(Path(__file__).read_bytes())
    ledger = held[['MVT_ID_mvt', 'APT', 'month', LABEL, 'fallback', 'is_heli',
                   'atot_minus_aobt3', 'atot_minus_sched']].copy()
    masked = held.copy()
    masked[LABEL] = np.nan
    masked['fallback'] = np.nan
    for name, features in [('control', cols), ('history', cols+HISTORY_COLS)]:
        print('FIT', window, name, len(train), len(held), flush=True)
        ledger[name] = fit_predict(folder/name, train, masked, features, cats, fingerprint)
        print('DONE', window, name, flush=True)
    clean = clean_mask(ledger).to_numpy()
    cal = clean & ledger.month.isin(WINDOWS[window][1]).to_numpy()
    y = ledger[LABEL].to_numpy()
    control, history = ledger.control.to_numpy(), ledger.history.to_numpy()
    losses = {str(a): float(np.mean((control[cal]+a*(history[cal]-control[cal])-y[cal])**2)) for a in ALPHAS}
    alpha = min(ALPHAS, key=lambda a: (losses[str(a)], a))
    ledger['candidate'] = control + alpha*(history-control)
    ledger.to_parquet(folder/'predictions.parquet', index=False)
    test = clean & ledger.month.isin(WINDOWS[window][2]).to_numpy()
    rows = [metric(ledger.loc[test], control[test], ledger.candidate.to_numpy()[test], 'all')]
    for month in WINDOWS[window][2]:
        z = test & ledger.month.eq(month).to_numpy()
        rows.append(metric(ledger.loc[z], control[z], ledger.candidate.to_numpy()[z], f'month_{month}'))
    for apt in sorted(ledger.APT.unique()):
        z = test & ledger.APT.eq(apt).to_numpy()
        rows.append(metric(ledger.loc[z], control[z], ledger.candidate.to_numpy()[z], apt))
    report = {'window': window, 'scope': protocol['scope'], 'alpha': alpha,
              'calibration_losses': losses, 'rows': rows,
              'held_label_poisoning_and_model_replay': 'exact',
              'submission_qualified': False,
              'expert_screen_pass': alpha > 0 and all(r['delta'] < 0 and r['delta_without_top10'] < 0 for r in rows[:3]),
              'ledger_sha256': digest(folder/'predictions.parquet')}
    (folder/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', choices=WINDOWS, required=True)
    run(parser.parse_args().window)
