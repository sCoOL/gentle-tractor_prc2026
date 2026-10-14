"""Build feature v3 with row-wise seasonal-pair OOF target features and corrected semantics.

The existing v2 artifact is preserved. Training target-derived columns are rebuilt with
six seasonal pair folds (Jan+Jul, ..., Jun+Dec); ranking columns use all 2025 labels.
This excludes a row's complete pair from its own values, but a static table cannot make the
outer model fit fully nested: other training rows may still use labels from the evaluated pair.
Use ``strict_tail_gate.py`` or a per-outer-fold feature build for audit-clean validation.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import time

import numpy as np
import pandas as pd

import common
import detectors
import ext_calendar
from common import DATA

_spec = importlib.util.spec_from_file_location('encodings_nested', pathlib.Path(__file__).with_name('encodings.py'))
enc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(enc)

TARGET_ENCODING_COLS = [s['name'] for s in enc.SPECS]
ROUTE_COLS = ['pair_p10_period', 'route_period_id', 'pair_n_periods']
LEGACY_FALLBACK_COLS = ['p_fallback_v2', 'p_fallback_lane', 'p_fallback_best']
INTERVAL_COUNT_COLS = [
    'dep_same_rwy_between_aobt3_atot', 'arr_same_rwy_between_aobt3_atot',
    'dep_any_rwy_between_aobt3_atot',
]


def _pier_of(s):
    m = re.match(r'^([A-Za-z]+)', s)
    return m.group(1) if m else s[:1]


def _encoding_rank_keys(rank):
    """Apply the reviewed unseen-stand map only to keys used for rank encodings."""
    out = rank.copy(); p = DATA / 'det_unseen_stands.csv'
    if not p.exists(): return out
    um = pd.read_csv(p); um = um[um.method != 'none']
    mp = {(a, str(s)): str(m) for a, s, m in zip(um.APT, um.stand, um.mapped_stand)}
    raw = out.apt_stand.astype(str).str.split('|', regex=False).str[-1]
    keys = list(zip(out.APT.astype(str), raw))
    hit = np.array([k in mp for k in keys])
    if hit.any():
        mapped = [f'{a}|{mp[k]}' for a, k in zip(out.loc[hit, 'APT'], [keys[i] for i in np.flatnonzero(hit)])]
        out.loc[hit, 'apt_stand'] = mapped
        out.loc[hit, 'apt_pier'] = [s.split('|')[0] + '|' + _pier_of(s.split('|')[1]) for s in mapped]
    return out


def _update_by_id(dst, src, columns):
    values = src.set_index('MVT_ID_mvt')[columns].reindex(dst.MVT_ID_mvt).reset_index(drop=True)
    if len(values) != len(dst) or values.isna().all(axis=None):
        raise ValueError(f'failed ID alignment for {columns}')
    for c in columns: dst[c] = values[c].values


def build(version=3, parent_version=2, force=False, fallback_rounds=400):
    common.ensure_dirs(); out_manifest = common.manifest_path(version)
    if out_manifest.exists() and not force:
        print(f'features v{version} exist; use --force to rebuild')
        return common.load_features(version)

    t0 = time.time(); train, rank, parent = common.load_features(parent_version)
    train = train.copy(); rank = rank.copy()
    fold_ids = enc.seasonal_pair_folds(train.month.values)

    # Pull corrected G3 counts and UTC timestamps from the rebuilt base tables.
    g3 = parent['groups']['G3']
    base_cols = ['MVT_ID_mvt', 'MVT_TIME_UTC_mvt'] + g3
    base_train = pd.read_parquet(DATA / 'base_train.parquet', columns=base_cols)
    base_rank = pd.read_parquet(DATA / 'base_rank.parquet', columns=base_cols)
    _update_by_id(train, base_train, g3); _update_by_id(rank, base_rank, g3)
    for df in (train, rank):
        for c in INTERVAL_COUNT_COLS:
            if c in df: df[c] = df[c].clip(lower=0)
    print(f'  corrected congestion columns [{time.time()-t0:.0f}s]', flush=True)

    # Calendar features use airport-local civil date, not UTC date.
    cal = pd.read_parquet(DATA / 'ext_calendar.parquet')
    cal_cols = ext_calendar.LOCAL_COLS + [c for c in ext_calendar.CAL_COLS if c != 'event_type']
    for dst, base in ((train, base_train), (rank, base_rank)):
        aligned = base.set_index('MVT_ID_mvt').reindex(dst.MVT_ID_mvt)
        lf = ext_calendar.local_features(dst.APT.values, aligned.MVT_TIME_UTC_mvt.reset_index(drop=True), cal)
        for c in cal_cols: dst[c] = lf[c].values
    print(f'  corrected local-date calendar columns [{time.time()-t0:.0f}s]', flush=True)

    # G8 target/fallback encodings: each row's seasonal pair is excluded as a unit.
    enc_train, enc_rank = enc.encode_frames(train, _encoding_rank_keys(rank), fold_ids=fold_ids)
    _update_by_id(train, enc_train, TARGET_ENCODING_COLS)
    _update_by_id(rank, enc_rank, TARGET_ENCODING_COLS)
    print(f'  nested G8 encodings [{time.time()-t0:.0f}s]', flush=True)

    # G4 route statistics: held pair gets last-period features fitted without either month.
    route_train, route_rank = detectors.route_features_oof(train, rank, fold_ids)
    route_train.insert(0, 'MVT_ID_mvt', train.MVT_ID_mvt.values)
    route_rank.insert(0, 'MVT_ID_mvt', rank.MVT_ID_mvt.values)
    _update_by_id(train, route_train, ROUTE_COLS); _update_by_id(rank, route_rank, ROUTE_COLS)
    print(f'  nested route features [{time.time()-t0:.0f}s]', flush=True)

    # The production mixture probability gets the same seasonal-pair OOF boundary.
    fb_train, fb_rank = detectors.detect_fallback(
        train, rank, detectors.FALLBACK_FEATS, detectors.FALLBACK_CATS,
        rounds=fallback_rounds, fold_ids=fold_ids,
    )
    _update_by_id(train, fb_train, ['p_fallback']); _update_by_id(rank, fb_rank, ['p_fallback'])
    print(f'  nested p_fallback [{time.time()-t0:.0f}s]', flush=True)

    groups = {k: list(v) for k, v in parent['groups'].items()}
    groups['G9'] = [c for c in groups['G9'] if c not in LEGACY_FALLBACK_COLS]
    feature_cols = [c for g in groups.values() for c in g]
    train = train[['MVT_ID_mvt', 'TAXITIME_SEC_mvt', 'fallback'] + feature_cols]
    rank = rank[['MVT_ID_mvt'] + feature_cols]
    train.to_parquet(DATA / f'features_v{version}_train.parquet')
    rank.to_parquet(DATA / f'features_v{version}_rank.parquet')

    columns = {
        c: {
            'group': g, 'source': ('rowwise_seasonal_pair_oof_v3' if c in TARGET_ENCODING_COLS + ROUTE_COLS + ['p_fallback'] else 'corrected_v2'),
            'rank_available': True, 'dtype': str(rank[c].dtype),
        }
        for g, cols in groups.items() for c in cols
    }
    manifest = {
        'version': version, 'parent_version': parent_version,
        'built_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
        'validation': 'rowwise_seasonal_pair_oof_not_strict_outer_nested',
        'columns': columns, 'groups': groups,
        'cat_cols': [c for c in parent.get('cat_cols', []) if c in feature_cols],
        'missing_tables': parent.get('missing_tables', []),
        'notes': [
            "Each row's G8, route-period and p_fallback values exclude its complete seasonal pair.",
            'For strict outer validation, target-derived predictors must also be absent/rebuilt on every outer-training row.',
            'p_fallback_v2/lane/best removed from G9 because they were redundant and lost taxi RMSE.',
            'Calendar joins use airport-local date; interval congestion counts are non-negative.',
        ],
    }
    out_manifest.write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(f'features v{version}: train {train.shape}, rank {rank.shape}, {len(feature_cols)} features [{time.time()-t0:.0f}s]', flush=True)
    return train, rank, manifest
