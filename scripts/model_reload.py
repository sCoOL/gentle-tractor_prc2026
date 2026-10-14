"""Reload a persisted M3rpa package and reproduce its clean expert predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import common
import train_eval


def load_tagged_boosters(model_dir):
    """Return ``tag -> (booster, metadata)`` for a saved model directory."""
    model_dir = Path(model_dir)
    tagged = {}
    for meta_path in sorted(model_dir.glob('*_meta.json')):
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        tag = meta.get('tag')
        if not tag:
            raise ValueError(f'missing model tag in {meta_path}')
        model_path = meta_path.with_name(meta_path.name.replace('_meta.json', '_lgb.txt'))
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        if tag in tagged:
            raise ValueError(f'duplicate model tag: {tag}')
        tagged[tag] = (lgb.Booster(model_file=str(model_path)), meta)
    return tagged


def _predict_with_training_categories(model, meta, train, score, cats):
    cols = meta['columns']
    missing = sorted(set(cols) - set(train.columns))
    if missing:
        raise ValueError(f'missing persisted features: {missing}')
    X = train_eval.prep(train, cols, cats)
    Xt = train_eval.prep(score, cols, cats, train_eval._cats_of(X, cats))
    return model.predict(Xt)


def reload_m3rpa_clean_expert(package_dir, feature_version=None):
    """Reproduce the pre-mixture M3rpa expert for ranking rows.

    Categorical domains are rebuilt from the exact full-training subsets used by
    the original fit. This is required because LightGBM stores category codes,
    not the source pandas category labels, in its text model.
    """
    package_dir = Path(package_dir)
    model_dir = package_dir / 'M3rpa' if (package_dir / 'M3rpa').is_dir() else package_dir
    state = json.loads((model_dir / 'state.json').read_text(encoding='utf-8'))
    if state.get('model_id') != 'M3rpa':
        raise ValueError(f"expected M3rpa package, got {state.get('model_id')!r}")
    version = int(feature_version or state['feature_version'])
    train, rank, _ = common.load_features(version)
    cats = state['cats']
    clean = train[(train.fallback.values == 0) & (train.is_heli.values == 0)]
    tagged = load_tagged_boosters(model_dir)

    normal = np.full(len(rank), np.nan)
    for apt in rank.APT.unique():
        tag = f'M3rpa_{apt}'
        if tag not in tagged:
            raise ValueError(f'missing persisted airport model: {tag}')
        fit = clean[clean.APT == apt]
        mask = rank.APT.values == apt
        model, meta = tagged[tag]
        if len(fit) != int(meta['n_rows']):
            raise ValueError(f'{tag} training row mismatch: {len(fit)} != {meta["n_rows"]}')
        normal[mask] = _predict_with_training_categories(model, meta, fit, rank[mask], cats)

    residual_model, residual_meta = tagged['M3r_residual']
    residual_train = clean[np.isfinite(clean.atot_minus_aobt3.values)]
    if len(residual_train) != int(residual_meta['n_rows']):
        raise ValueError(
            f'residual training row mismatch: {len(residual_train)} != {residual_meta["n_rows"]}'
        )
    residual = _predict_with_training_categories(
        residual_model, residual_meta, residual_train, rank, cats,
    )
    a3 = rank.atot_minus_aobt3.values.astype(float)
    normal = np.where(np.isfinite(a3), a3 + residual, normal)
    normal = np.where(rank.is_heli.values == 1, float(state['heli_const']), normal)
    if not np.isfinite(normal).all():
        raise ValueError(f'non-finite clean expert predictions: {int((~np.isfinite(normal)).sum())}')
    return rank, normal, state


def verify_and_write(package_dir, base_submission, output=None, atol=1e-6):
    rank, normal, state = reload_m3rpa_clean_expert(package_dir)
    reproduced = train_eval._mixture(rank, normal, state.get('params', {}), state['heli_const'])
    reproduced = np.maximum(reproduced, train_eval.FLOOR)
    base = pd.read_parquet(base_submission)[['MVT_ID_mvt', 'TAXITIME_SEC_mvt']]
    check = rank[['MVT_ID_mvt']].merge(base, on='MVT_ID_mvt', how='left', validate='one_to_one')
    delta = np.abs(reproduced - check.TAXITIME_SEC_mvt.values)
    if not np.isfinite(delta).all() or float(delta.max()) > atol:
        raise ValueError(
            f'saved-model reproduction failed: max_abs_delta={float(np.nanmax(delta)):.9g}, atol={atol}'
        )
    output = Path(output) if output else Path(package_dir) / 'normal_rank.parquet'
    pd.DataFrame({'MVT_ID_mvt': rank.MVT_ID_mvt.values, 'normal_pred': normal}).to_parquet(output, index=False)
    return output, {'n_rows': len(rank), 'max_abs_delta': float(delta.max()), 'mean_abs_delta': float(delta.mean())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--base-submission', required=True)
    parser.add_argument('--output')
    parser.add_argument('--atol', type=float, default=1e-6)
    args = parser.parse_args()
    path, metrics = verify_and_write(args.package, args.base_submission, args.output, args.atol)
    print(json.dumps({'output': str(path), **metrics}, indent=1))
