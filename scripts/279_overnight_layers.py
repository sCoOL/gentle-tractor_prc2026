"""Overnight 2026-10-08 layers that passed the pass rule (runs/overnight_20261008/ov.py judge), each a separate step on v66.

  py -3.14 scripts/279_overnight_layers.py build --item 5     # v67 = v66 + recorded-take-off shift layer   (A74)
  py -3.14 scripts/279_overnight_layers.py build --item 4     # v68 = v66 + stand-error layer               (A75)
  py -3.14 scripts/279_overnight_layers.py build --item 2     # v69 = v66 + LTFM month-end cut layer        (A76)
  py -3.14 scripts/279_overnight_layers.py build --item 11    # v70 = v66 + joint coverage-shadow layer     (A77, low confidence)

Frames (label-free inputs, both years; 2025 evidence in the same folder):
  item 5  runs/overnight_20261008/item05_dateshift.py -> item05_rows.parquet. Departures with no v2 ADS-B match whose
          recorded take-off is 180 s < |s| <= 3600 s from an UNCLAIMED adsb.lol take-off with the same normalised
          callsign (scripts/flightnorm.py keys of NM callsign / flight number / destination-arrival callsign + 255
          aliases) at the same airport. Residual LightGBM on v66 with s, v66, v66 - s, ground points, at-stand, hour,
          airport (4 leaves, 150 rounds, 3 seeds), alpha 1.0 (best forward). 2025: 883 rows, six folds -15.1 M s^2,
          6/6 folds, forward -6.3 / -5.9 M, January -1.5 M, 99.5 % body.
  item 4  runs/overnight_20261008/item04_park_extract.py + item04y_stand_error.py -> item04y_rows.parquet. ADS-B parked
          position (first ground point of the v2-matched departure) vs the per-year median parked position of its
          RECORDED stand (stands with >= 5 standing departures); rows seen standing with distance > 60 m. Residual
          LightGBM on v66 with log distance, v66, hour, airport, distance to nearest learned stand, the stand's spread
          (7 leaves, 200 rounds, 3 seeds), alpha 0.75 (best forward). 2025: 57,959 rows, six folds -17.7 M, 6/6 folds,
          forward -9.4 / -4.8 M, January -3.5 M, 70 % body.
  item 2  runs/overnight_20261008/item02_gaps.py -> item02_frame.parquet. LTFM rows within 3 h before the 21:00 UTC
          month-end cut. Residual LightGBM with minutes to the cut, v66, hour (4 leaves, 100 rounds), alpha 0.5.
          2025: 884 rows, six folds -0.40 M, 4/6 folds, forward -0.20 / -0.33 M. 162 rows in 2026.
  item 11 runs/overnight_20261008/item11_arr_tail_extract.py + item11_shadow_layer.py -> item11_rows.parquet. v2-matched
          departures first seen already moving (first_state forward / moving, or not at a stand) with v66 <= 1800 s
          (label-free gate). Same-period arrival tail at the stand (recorded in-block - last ADS-B ground point of
          arrivals, airport x stand x year-month, shrunk to the airport-month): per-airport LightGBM fitted on the y <=
          3600 rows only (residual clipped +-900 s), alpha 0.25. 2025: six folds -27.5 M, 4/6 folds (C +3.3, D +0.6),
          forward -2.6 / -12.7 M, January -2.0 M, body share 1.22. LOW CONFIDENCE: third variant tried for this item,
          LEMD worse in every variant, alpha 0.5 already fails forward; used at alpha 0.25 in the A78 combination.
Every layer (items 2, 4, 5): fit on all 2025 rows of its frame (residual y - v66 proxy, clipped +-1800 s), apply to the 2026 rows, then
267.rules_v55 on the moved rows. No official score is used.
"""
import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

OV = common.RUNS / 'overnight_20261008'
sys.path.insert(0, str(OV))
import ov  # noqa: E402

V66 = common.SUBS / 'gentle-tractor_v66.parquet'
APT_CODES = {a: i for i, a in enumerate(sorted(common.AIRPORT_META))}

ITEMS = {
    5: dict(version=67, approach='A74', tag='v67', name='recorded_takeoff_shift_layer', alpha=1.0, rounds=150,
            params={**ov.PARAMS, 'num_leaves': 4, 'min_data_in_leaf': 40},
            X=['s', 'base', 'base_minus_s', 'n_ground', 'at_stand', 'hour', 'apt_code']),
    4: dict(version=68, approach='A75', tag='v68', name='stand_error_layer', alpha=0.75, rounds=200, params=ov.PARAMS,
            X=['logd', 'base', 'hour', 'apt_code', 'd_stand_first', 'spread']),
    2: dict(version=69, approach='A76', tag='v69', name='ltfm_month_end_cut_layer', alpha=0.5, rounds=100,
            params={**ov.PARAMS, 'min_data_in_leaf': 30, 'num_leaves': 4}, X=['min_to_cut', 'base', 'hour']),
    11: dict(version=70, approach='A77', tag='v70', name='joint_coverage_shadow_layer', alpha=0.25, rounds=200,
             params={**ov.PARAMS, 'num_leaves': 15, 'min_data_in_leaf': 200}, by='APT', fit_body=True, clip=900.0,
             X=['x_first', 'tail_stand', 'gone_stand', 'm_am', 'g_am', 'd_stand_first', 'fs', 'base', 'hour', 'apt_code']),
}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frame(item: int, v66: pd.Series) -> pd.DataFrame:
    """Rows of the layer, both years, with base = v66 proxy (2025) / v66 file (2026) and the layer's inputs."""
    if item == 5:
        d = pd.read_parquet(OV / 'item05_rows.parquet')
        d = d[d.s.abs() <= 3600].copy()
        d['hour'] = pd.to_datetime(d.t, unit='s', utc=True).dt.hour
        d['apt_code'] = d.APT.map(APT_CODES).astype(float)
        d['at_stand'] = d.at_stand.astype(float)
    elif item == 4:
        d = pd.read_parquet(OV / 'item04y_rows.parquet')
        d = d[d.flag.astype(bool)].copy()
        fr = ov.frame()[['MVT_ID_mvt', 't']]
        d = d.merge(fr, on='MVT_ID_mvt', how='left')
        d['hour'] = pd.to_datetime(d.t, unit='s', utc=True).dt.hour
        d['apt_code'] = d.APT.map(APT_CODES).astype(float)
        d['logd'] = np.log1p(d.dist)
    elif item == 2:
        d = pd.read_parquet(OV / 'item02_frame.parquet')
        d = d[d.min_to_cut.notna() & (d.min_to_cut <= 180)].copy()
    else:
        d = pd.read_parquet(OV / 'item11_rows.parquet')
    p = ov.load_proxy()[['MVT_ID_mvt', 'v66_pred_s']].rename(columns={'v66_pred_s': 'proxy'})
    d = d.drop(columns=[c for c in ('base', 'v66_pred_s', 'r') if c in d.columns]).merge(p, on='MVT_ID_mvt', how='left')
    d['base'] = np.where(d.split == 'rank', v66.reindex(d.MVT_ID_mvt).to_numpy(), d.proxy)
    if item == 5:
        d['base_minus_s'] = d.base - d.s
    d['r'] = d.y - d.base
    if item == 11:
        d = d[d.base <= 1800.0]
    return d.reset_index(drop=True)


def build(item: int):
    cfg = ITEMS[item]
    out_f = common.SUBS / f"gentle-tractor_{cfg['tag']}.parquet"
    if out_f.exists():
        raise SystemExit(f'{out_f} exists; refusing to overwrite')
    sub = pd.read_parquet(V66)
    v66 = sub.set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)
    d = frame(item, v66)
    tr, rk = d[d.split == 'train'], d[d.split == 'rank'].reset_index(drop=True)
    assert rk.base.notna().all() and tr.base.notna().all()
    clip = cfg.get('clip', 1800.0)
    fit = tr[tr.y <= 3600] if cfg.get('fit_body') else tr
    if cfg.get('by'):
        corr = np.zeros(len(rk))
        for g in rk[cfg['by']].unique():
            mr, mf = (rk[cfg['by']] == g).to_numpy(), (fit[cfg['by']] == g).to_numpy()
            if mf.sum() > 200:
                corr[mr] = ov.fit_predict(fit[mf], rk[mr], cfg['X'], rounds=cfg['rounds'], clip=clip, params=cfg['params'])
    else:
        corr = ov.fit_predict(fit, rk, cfg['X'], rounds=cfg['rounds'], clip=clip, params=cfg['params'])
    new_rows = ov.rules(rk.MVT_ID_mvt, rk.base.to_numpy() + cfg['alpha'] * corr)
    new = v66.copy()
    new.loc[rk.MVT_ID_mvt.to_numpy()] = new_rows
    out = sub.copy()
    out['TAXITIME_SEC_mvt'] = new.reindex(out.MVT_ID_mvt).to_numpy()
    assert np.isfinite(out.TAXITIME_SEC_mvt).all() and (out.TAXITIME_SEC_mvt >= ov.b53.FLOOR).all()
    mv = out.TAXITIME_SEC_mvt.to_numpy() - sub.TAXITIME_SEC_mvt.to_numpy()
    out.to_parquet(out_f, index=False)
    pkg = common.RUNS / 'models' / f"{cfg['approach']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg['tag']}"
    pkg.mkdir(parents=True)
    lr = rk[['MVT_ID_mvt', 'APT' if 'APT' in rk.columns else 'apt'] + cfg['X']].assign(corr=corr, v66=rk.base.to_numpy(), new=new_rows)
    lr['move_s'] = lr.new - lr.v66
    lr.to_parquet(pkg / 'layer_rows_2026.parquet', index=False)
    ev = {}
    for n in {5: ['item05b_layer.json', 'item05_summary.json'], 4: ['item04y_stand_error.json'], 2: ['item02_gaps.json'],
              11: ['item11_shadow_layer.json', 'item11b_body_fit.json', 'item11c_gated.json']}[item]:
        if (OV / n).exists():
            (pkg / n).write_bytes((OV / n).read_bytes())
            ev[n] = json.loads((OV / n).read_text())
    for src in (Path(__file__), OV / 'ov.py'):
        (pkg / src.name).write_bytes(src.read_bytes())
    aptcol = 'APT' if 'APT' in lr.columns else 'apt'
    info = {'approach': cfg['approach'], 'version': cfg['version'], 'layer': cfg['name'], 'item': item, 'base': V66.name,
            'base_sha256': sha(V66), 'file_sha256': sha(out_f), 'rows_in_layer_2026': int(len(rk)),
            'rows_changed': int((mv != 0).sum()), 'sum_sq_move_2026_M': round(float((mv ** 2).sum() / 1e6), 3),
            'rms_move_s': round(float(np.sqrt((lr.move_s ** 2).mean())), 1), 'mean_move_s': round(float(lr.move_s.mean()), 1),
            'by_airport_2026': lr.groupby(aptcol).move_s.agg(['size', 'mean', lambda s: float(np.sqrt((s ** 2).mean()))]).round(1)
                                 .rename(columns={'<lambda_0>': 'rms'}).reset_index().to_dict('records'),
            'alpha': cfg['alpha'], 'features': cfg['X'], 'params': cfg['params'], 'rounds': cfg['rounds'], 'seeds': [7, 8, 9],
            'rows_2025_fit': int(len(fit)), 'evidence_2025': ev}
    (pkg / 'build.json').write_text(json.dumps(ov._clean(info), indent=2, default=str))
    print(json.dumps({k: v for k, v in info.items() if k != 'evidence_2025'}, indent=1, default=str))
    print(f'package {pkg}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build'])
    ap.add_argument('--item', type=int, required=True, choices=sorted(ITEMS))
    a = ap.parse_args()
    build(a.item)
