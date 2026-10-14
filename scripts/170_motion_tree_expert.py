"""Fixed paired tree control for new motion features; nested training-only test.

The direct physical blend cannot learn when timing observations are trustworthy.
This second, separately specified comparison adds the same new timings to a
fixed tree with existing target-safe context. Both trees exclude the held block.
No official results, parameter search, rank predictions or submission writes.
"""
import gc
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

m = importlib.import_module('166_motion_nested_validation')
h = importlib.import_module('157_history_normal')
OUT = m.common.RUNS/'v49_motion_trees'
PARAMS = dict(h.PARAMS, num_leaves=127, seed=20260924)
ROUNDS = 400


def load():
    manifest = m.common.read_manifest(5)
    forbidden = set(manifest['groups']['G8']) | h.stg.TARGET_ROUTE | h.stg.TARGET_G9
    cols = [c for c in h.tr.feature_columns(manifest, list(manifest['groups']))
            if c not in forbidden and not c.startswith('arr_fb_rate_')]
    need = list(dict.fromkeys(cols+['MVT_ID_mvt', 'month', 'APT', h.LABEL, 'fallback', 'is_heli', 'fpl_matched']))
    features = pd.read_parquet(m.FEATURE_PATH)
    d = pd.read_parquet(m.common.DATA/'features_v5_train.parquet', columns=need)
    d = d.loc[d.MVT_ID_mvt.isin(features.MVT_ID_mvt) & d.APT.ne('LIRF') & d.fpl_matched.eq(1) & d.is_heli.eq(0)].copy()
    d = d.merge(features.drop(columns='date'), on='MVT_ID_mvt', validate='one_to_one')
    for read in [h.adsb.table, h.adsb.stops_table, h.adsb.geom_table, h.adsb.nbr_table]:
        d = d.merge(read(), on='MVT_ID_mvt', how='left', validate='one_to_one')
    cols += h.adsb.COLUMNS+h.adsb.STOPS_COLUMNS+h.adsb.GEOM_COLUMNS+h.adsb.NBR_COLUMNS+['original_taxi']
    new = list(m.motion.COLUMNS)
    for col in m.motion.COLUMNS:
        delta = col+'_minus_original'
        d[delta] = d[col]-d.original_taxi
        new.append(delta)
    for c in cols+new:
        if pd.api.types.is_float_dtype(d[c]) and c not in ['atot_minus_aobt3', 'atot_minus_sched']:
            d[c] = d[c].astype('float32')
    assert not set(cols+new).intersection([h.LABEL, 'fallback', 'b', 'BLOCK_TIME_UTC_mvt'])
    return d, cols, new, h.tr._cats(manifest, cols)


def evaluate(predictions):
    raw = pd.read_parquet(m.OUT/'upstream.parquet')
    raw = raw.merge(predictions, on='MVT_ID_mvt', how='left', validate='one_to_one')
    rcols = [c for c in raw if c.startswith('R') and c[1:].isdigit()]
    ledgers, selections = [], {}
    for block in sorted(raw.block.unique()):
        for fold in block:
            cal = raw.loc[raw.block.eq(block) & raw.fold.ne(fold)].copy()
            score = raw.loc[raw.fold.eq(fold)].copy()
            normal_inner = m.g.crossfit_normal(cal, cal[rcols].to_numpy(), cal.month)
            normal = m.g.normal_from(score, score[rcols].to_numpy(), m.g.fit_blend(cal, cal[rcols].to_numpy()))
            ledger = score[['MVT_ID_mvt', 'APT', 'fold', 'month']].copy()
            selections[fold] = {}
            for expert, key in [('tree_control', 'original_control'), ('tree_motion', 'candidate')]:
                alpha, losses = m.choose_alpha(cal, normal_inner, cal[expert].to_numpy())
                selections[fold][expert] = {'alpha': alpha, 'losses': losses}
                result = m.compose(score.assign(y=np.nan), normal, score[expert].to_numpy(), alpha)
                np.testing.assert_array_equal(result, m.compose(score.assign(y=1e12), normal, score[expert].to_numpy(), alpha))
                ledger[key] = result
            ledger['baseline'] = m.compose(score, normal, score.tree_motion.to_numpy(), 0.)
            ledger['y'] = score.y.to_numpy()
            ledger['normal'] = normal
            ledger['tree_control'], ledger['tree_motion'] = score.tree_control.to_numpy(), score.tree_motion.to_numpy()
            ledgers.append(ledger)
            print('fold', fold, selections[fold], m.metric(ledger), flush=True)
    ledger = pd.concat(ledgers, ignore_index=True)
    ledger.to_parquet(OUT/'predictions.parquet', index=False)
    slices = [('pooled', ledger)]+[(f'fold_{k}', z) for k, z in ledger.groupby('fold')]+[(f'month_{k}', z) for k, z in ledger.groupby('month')]
    metrics = {name: m.metric(z) for name, z in slices}
    novelty = {name: m.metric(z, 'original_control') for name, z in slices}
    report = {'scope': __doc__, 'selections': selections, 'metrics': metrics, 'motion_vs_tree_control': novelty,
              'branch_gate_pass': metrics['pooled']['delta'] < 0 and all(r['delta'] <= 0 and r['delta_without_top10'] <= 0 for r in metrics.values()),
              'new_information_gate_pass': novelty['pooled']['delta'] < 0 and all(r['delta'] <= 0 and r['delta_without_top10'] <= 0 for r in novelty.values()),
              'airport_metrics': {str(k): m.metric(z) for k, z in ledger.groupby('APT', observed=True)},
              'poisoning_gap': 0., 'submission_qualified': False, 'ledger_sha256': h.digest(OUT/'predictions.parquet')}
    (OUT/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(m.__file__), Path(h.__file__), Path(h.adsb.__file__), Path(h.tr.__file__)]
    inputs = [m.FEATURE_PATH, m.OUT/'upstream.parquet', m.common.DATA/'features_v5_train.parquet',
              h.adsb.PATH, h.adsb.STOPS_PATH, h.adsb.GEOM_PATH, h.adsb.NBR_PATH]
    protocol = {'sources': {str(p): h.digest(p) for p in sources}, 'inputs': {str(p): h.digest(p) for p in inputs},
                'params': PARAMS, 'rounds': ROUNDS, 'alphas': m.ALPHAS, 'scope': __doc__}
    path = OUT/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Cached source/input drift'
    else:
        path.write_text(json.dumps(protocol, indent=2))
    for p in sources:
        (OUT/p.name).write_bytes(p.read_bytes())
    d, cols, new, cats = load()
    h.PARAMS, h.ROUNDS = PARAMS, ROUNDS
    parts = []
    upstream = pd.read_parquet(m.OUT/'upstream.parquet', columns=['block', 'month'])
    for block in sorted(upstream.block.unique()):
        months = sorted(upstream.loc[upstream.block.eq(block), 'month'].unique().tolist())
        train = d.loc[~d.month.isin(months) & h.clean_mask(d)].copy()
        score = d.loc[d.month.isin(months)].copy()
        assert not np.intersect1d(train.MVT_ID_mvt, score.MVT_ID_mvt).size
        score[h.LABEL] = np.nan
        score['fallback'] = np.nan
        part = score[['MVT_ID_mvt']].copy()
        for tag, columns in [('control', cols), ('motion', cols+new)]:
            print('Training', block, tag, len(train), 'rows', flush=True)
            part['tree_'+tag] = h.fit_predict(OUT/block/tag, train, score, columns, cats, protocol)
        parts.append(part)
        del train, score
        gc.collect()
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_parquet(OUT/'expert_predictions.parquet', index=False)
    evaluate(predictions)


if __name__ == '__main__':
    run()
