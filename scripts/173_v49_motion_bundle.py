"""The sustained-motion normal expert (A57) on the v45 matched component.

  py -3.14 scripts/173_v49_motion_bundle.py fit     # train the A57 booster on 2025 -> runs/models/A57_20260924_v49/

fit: the motion tree of scripts/170 (same columns, parameters, rounds and seed), fitted once on every clean 2025 row with
motion features (non-LIRF, NM-matched, fixed-wing, finite AOBT_3) and saved with its state; alpha = the shrink chosen
unanimously by the inner calibration of all six training folds (scripts/170 report, which must pass its branch and
new-information gates). clean_component() applies the saved booster to any rank population; tools/final_chain.py
step 49 uses it on the final rows. No official score is used.
"""
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

t = importlib.import_module('170_motion_tree_expert')
m = t.m
h = t.h
ROOT = m.common.ROOT
PACKAGE = m.common.RUNS/'models/A57_20260924_v49'


def rank_features(columns):
    path = m.common.DATA/'features_v5_rank.parquet'
    present = set(pq.read_schema(path).names)
    metadata = ['MVT_ID_mvt', 'APT', 'month', 'fpl_matched', 'is_heli', 'atot_minus_aobt3']
    names = list(dict.fromkeys([c for c in columns if c in present]+metadata))
    d = pd.read_parquet(path, columns=names)
    observations = pd.read_parquet(m.FEATURE_PATH)
    d = d.loc[d.APT.ne('LIRF') & d.fpl_matched.eq(1) & d.is_heli.eq(0)
              & np.isfinite(d.atot_minus_aobt3) & d.MVT_ID_mvt.isin(observations.MVT_ID_mvt)].copy()
    d = d.merge(observations.drop(columns='date'), on='MVT_ID_mvt', validate='one_to_one')
    for read in [h.adsb.table, h.adsb.stops_table, h.adsb.geom_table, h.adsb.nbr_table]:
        d = d.merge(read(), on='MVT_ID_mvt', how='left', validate='one_to_one')
    for col in m.motion.COLUMNS:
        d[col+'_minus_original'] = d[col]-d.original_taxi
    for col in columns:
        if pd.api.types.is_float_dtype(d[col]) and col not in ['atot_minus_aobt3', 'atot_minus_sched']:
            d[col] = d[col].astype('float32')
    assert d.MVT_ID_mvt.is_unique and set(columns).issubset(d.columns)
    return d


def add_correction(baseline, old_normal, new_normal, probability, delay, alpha, source_weight):
    """The exact floor-aware difference of the v45 matched component, added to `baseline` with `source_weight`."""
    p = m.g.matched_probability(probability, delay)
    delay = np.where(np.isfinite(delay), delay, 0.)
    old = np.maximum(old_normal+p*(delay-old_normal), 30.)
    normal = old_normal+float(alpha)*(new_normal-old_normal)
    new = np.maximum(normal+p*(delay-normal), 30.)
    final = np.maximum(baseline+float(source_weight)*(new-old), 30.)
    return final, old, new



def clean_component(detail: pd.DataFrame, package=None, feature_path=None) -> pd.DataFrame:
    """The motion change on the v45 component for any rank population (tools/final_chain.py step 49), with the SAVED A57
    booster (motion/model.txt + state.json; no refit) and alpha from the package summary, at source weight 1, for the
    rows of `detail` (MVT_ID_mvt, normal_members_a36, p_matched_mix, atot_minus_sched: the v45 detail of
    tools/members/recompose_v45.py). Eligibility as build(): the row is in rank_features() (non-LIRF,
    NM-matched, fixed-wing, finite AOBT_3, an original at-stand ground phase in the motion features).
    feature_path: the motion feature table (165 output) for these rows; default m.FEATURE_PATH."""
    import lightgbm as lgb
    package = Path(package or PACKAGE)
    state = json.loads((package/'motion'/'state.json').read_text())
    alpha = float(json.loads((package/'summary.json').read_text())['alpha'])
    if feature_path is not None:
        m.FEATURE_PATH = Path(feature_path)
    rank = rank_features(state['columns'])
    rank = rank[rank.MVT_ID_mvt.isin(detail.MVT_ID_mvt)].reset_index(drop=True)
    model = lgb.Booster(model_file=str(package/'motion'/'model.txt'))
    expert = h.predict(model, state, rank)
    assert np.isfinite(expert).all()
    q = detail[detail.MVT_ID_mvt.isin(rank.MVT_ID_mvt)].reset_index(drop=True)
    expert = pd.Series(expert, index=rank.MVT_ID_mvt.to_numpy()).reindex(q.MVT_ID_mvt).to_numpy()
    zero = np.zeros(len(q))
    _, old, new = add_correction(zero, q.normal_members_a36.to_numpy(), expert, q.p_matched_mix.to_numpy(),
                                 q.atot_minus_sched.to_numpy(), alpha, 1.0)
    rd = q[['MVT_ID_mvt', 'atot_minus_sched', 'p_matched_mix', 'normal_members_a36']].copy()
    rd['motion_expert'], rd['old_v45_component'], rd['new_v45_component'] = expert, old, new
    rd = rd.merge(rank[['MVT_ID_mvt', 'APT', 'month']], on='MVT_ID_mvt', validate='one_to_one')
    rd.attrs['alpha'] = alpha
    return rd


def fit(package=PACKAGE):
    """Train and save the A57 motion booster (+ summary.json with alpha). Rank rows are not predicted here."""
    report = json.loads((t.OUT/'report.json').read_text())
    assert report['branch_gate_pass'] and report['new_information_gate_pass']
    assert json.loads((t.OUT/'audit.json').read_text())['passed']
    alphas = [v['tree_motion']['alpha'] for v in report['selections'].values()]
    assert len(alphas) == 6 and len(set(alphas)) == 1 and alphas[0] > 0
    alpha = alphas[0]
    package = Path(package)
    if (package/'motion'/'model.txt').exists():
        raise FileExistsError(f'{package} already holds a motion model')
    package.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(t.__file__), Path(m.__file__), Path(h.__file__), Path(h.tr.__file__), Path(h.adsb.__file__)]
    protocol = {'scope': __doc__, 'alpha': alpha, 'sources': {str(p.name): h.digest(p) for p in sources},
                'params': t.PARAMS, 'rounds': t.ROUNDS,
                'eligibility': 'non-LIRF, matched, non-helicopter, finite AOBT3, original at-stand non-after-landing ground phase with >=3 points',
                'official_results_used': False}
    (package/'protocol.json').write_text(json.dumps(protocol, indent=2))
    train, cols, new, cats = t.load()
    train = train.loc[h.clean_mask(train)].copy()
    columns = cols+new
    rank = rank_features(columns)
    assert not np.intersect1d(train.MVT_ID_mvt, rank.MVT_ID_mvt).size
    h.PARAMS, h.ROUNDS = t.PARAMS, t.ROUNDS
    print('Full motion fit', len(train), 'training rows', flush=True)
    h.fit_predict(package/'motion', train, rank, columns, cats, protocol)
    summary = {'approach': 'A57', 'alpha': alpha, 'train_rows': len(train), 'validation': report['metrics'],
               'paired_control': report['motion_vs_tree_control'],
               'model_sha256': h.digest(package/'motion/model.txt'), 'state_sha256': h.digest(package/'motion/state.json')}
    (package/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['fit'])
    parser.parse_args()
    fit()
