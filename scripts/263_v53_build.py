"""v53 = v52 + the LSZH OPDI off-block correction (scripts/262) + the +-1 h band rule + the VFR guard.

  py -3.14 scripts/263_v53_build.py evidence   # 2025 check -> runs/v53_build/evidence.json + rows.parquet
  py -3.14 scripts/263_v53_build.py build      # submissions/gentle-tractor_v53.parquet + runs/models/A61_*_v53/

Order on every row: base (v52 file on 2026; the v52 proxy = A51 strict OOF + W/F/R on 2025) -> + OPDI correction
(LSZH rows with an OPDI ground event; 2025: nested out-of-fold, 2026: full-2025 fit, 3 seeds) -> the NM-window rules
again (they only bound labels, so re-applying them after a learned move can only lower each row's error):
  W  NM-matched: clip to take-off - LOBT +- 3,660 s (scripts/251).
  F  unmatched, LOBT recovered at the destination (scripts/250), take-off - LOBT < 3,540 s: floor take-off - LOBT + 3,540 s.
  R  unmatched, recovered LOBT, schedule within 3,540 s of it, take-off - LOBT > 3,570 s: cap take-off - LOBT - 3,540 s.
  B  (new, 2026-09-26) the other unmatched rows with a recovered LOBT ('sched_far': schedule 59+ min from
     it): a prediction INSIDE take-off - LOBT +- 3,540 s moves to the lower edge, take-off - LOBT - 3,540 s. 2025: all
     100 sched_far labels lie below the band (98 with the schedule more than 1 h BEFORE LOBT, where a schedule copy
     would lie above it). A prediction above the band is left alone: the rule allows it (one 2026 row).
F, R and B apply only where the airline normally matches NM at the airport (rate >= 0.5, v51's guard) AND the airport
recorded the flight as not VFR: the builder never matches FLIGHT_RULE_mvt = 'V' (0 of ~7,000 rows, both years), so a
VFR row is unmatched for identity, not for time (2025: 12 VFR rows with a recovered LOBT, all inside the band).
Floor 30 s; helicopters keep the base value (never in the OPDI scope). No official score is used.
"""
import argparse
import hashlib
import importlib
import json
from datetime import datetime

import numpy as np
import pandas as pd

import common
import final_rank  # final-file change: rank-population size (was a hard-coded 344,841)

opdi = importlib.import_module('262_opdi_correction')
OUT = common.RUNS / 'v53_build'
V52, V53 = common.SUBS / 'gentle-tractor_v52.parquet', common.SUBS / 'gentle-tractor_v53.parquet'
LEDGER = common.RUNS / 'v52_recovered_lobt' / 'rows.parquet'
OOF = common.RUNS / 'a51_final' / 'chosen_ensemble_oof.parquet'
BACKFILL = common.RUNS / 'nm_backfill' / 'backfill.parquet'
W_CLIP, F_GAP, OP_MIN = 3660.0, 3540.0, 0.5
OPDI_APTS = ('LSZH',)
FLOOR = 30.0


def _epoch(s):
    return ((pd.to_datetime(s, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def frame() -> pd.DataFrame:
    """One row per departure, both years: the inputs of W/F/R/B (scripts/251 + 256 from the base tables)."""
    cols = ['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_RULE_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
            'LOBT_flt', 'TAXITIME_SEC_mvt', 'month', 'is_heli']
    b = pd.concat([pd.read_parquet(common.DATA / f, columns=cols).assign(split=s)
                   for f, s in [('base_train.parquet', 'train'), ('base_rank.parquet', 'rank')]], ignore_index=True)
    b['al'] = b.FLIGHT_mvt.str.extract(r'^([A-Z]{3}|[A-Z0-9]{2})', expand=False)
    b['nm'] = b.FLIGHT_ID_mvt.notna()
    op = b.groupby(['APT', 'al'], dropna=False).nm.mean().rename('op_rate').reset_index()   # NULL airline = its own group (DuckDB GROUP BY)
    b = b.merge(op, on=['APT', 'al'], how='left')
    bf = pd.read_parquet(BACKFILL, columns=['MVT_ID_mvt', 'bf_lobt'])
    b = b.merge(bf, on='MVT_ID_mvt', how='left', validate='one_to_one')
    return pd.DataFrame({'MVT_ID_mvt': b.MVT_ID_mvt, 'split': b.split, 'apt': b.APT, 'month': b.month, 'is_heli': b.is_heli,
                         't': _epoch(b.MVT_TIME_UTC_mvt), 'lobt': _epoch(b.LOBT_flt), 'nm': b.nm, 'bf_lobt': _epoch(b.bf_lobt),
                         'op_rate': b.op_rate, 't_sched': _epoch(b.SCHED_TIME_UTC_mvt), 'vfr': b.FLIGHT_RULE_mvt.eq('V').fillna(False),
                         'y': b.TAXITIME_SEC_mvt.astype(float)})


def status(f: pd.DataFrame) -> np.ndarray:
    """scripts/256 status() with the VFR guard in front of the airline guard."""
    L = f.t.to_numpy() - f.bf_lobt.to_numpy()
    rec = (~f.nm.to_numpy()) & np.isfinite(L)
    op = f.op_rate.fillna(0).to_numpy() >= OP_MIN
    near = np.abs(f.t_sched.to_numpy() - f.bf_lobt.to_numpy()) < F_GAP
    late = L > F_GAP + 30.0
    return np.select([f.nm.to_numpy(), ~rec, f.vfr.to_numpy(), ~op, ~late, ~near],
                     ['matched', 'no_recovered_lobt', 'vfr', 'airline_rarely_nm', 'F_range', 'sched_far'], 'R')


def apply_rules(f: pd.DataFrame, p: np.ndarray, st: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out, why = p.astype(float).copy(), np.full(len(p), '', dtype=object)
    L = f.t.to_numpy() - f.lobt.to_numpy()
    w = np.isfinite(L)
    low, high = w & (out < L - W_CLIP), w & (out > L + W_CLIP)
    out[low], out[high] = (L - W_CLIP)[low], (L + W_CLIP)[high]
    why[low], why[high] = 'W_low', 'W_high'
    Lb = f.t.to_numpy() - f.bf_lobt.to_numpy()
    fl = (st == 'F_range') & (out < Lb + F_GAP)
    out[fl], why[fl] = (Lb + F_GAP)[fl], 'F'
    rc = (st == 'R') & (out > Lb - F_GAP)
    out[rc], why[rc] = (Lb - F_GAP)[rc], 'R'
    bd = (st == 'sched_far') & (np.abs(out - Lb) < F_GAP)
    out[bd], why[bd] = (Lb - F_GAP)[bd], 'B'
    lo = out < FLOOR
    out[lo] = FLOOR
    return out, why


def rmse(e2) -> float:
    return float(np.sqrt(np.mean(e2)))


def evidence():
    OUT.mkdir(parents=True, exist_ok=True)
    f = frame()
    led = pd.read_parquet(LEDGER, columns=['MVT_ID_mvt', 'split', 'v52_status', 'v52_pred_s'])
    chk = f[['MVT_ID_mvt']].assign(st=status(f)).merge(led[['MVT_ID_mvt', 'v52_status']], on='MVT_ID_mvt', validate='one_to_one')
    # the base-table frame must reproduce scripts/256's statuses; only the new 'vfr' class may differ
    diff = chk[chk.st != chk.v52_status]
    assert len(chk) == len(f) and set(diff.st) <= {'vfr'}, diff.groupby(['v52_status', 'st']).size()
    tr = f[f.split == 'train'].merge(led[led.split == 'train'][['MVT_ID_mvt', 'v52_pred_s']], on='MVT_ID_mvt', validate='one_to_one') \
        .merge(pd.read_parquet(OOF, columns=['MVT_ID_mvt', 'fold']), on='MVT_ID_mvt', validate='one_to_one')
    assert len(tr) == 2085047
    corr = pd.concat([pd.read_parquet(opdi.OUT / f'oof_{a}.parquet') for a in OPDI_APTS])
    tr = tr.merge(corr, on='MVT_ID_mvt', how='left')
    tr['corr'] = tr['corr'].fillna(0.0)
    y, p52 = tr.y.to_numpy(), tr.v52_pred_s.to_numpy()
    st = status(tr)
    p_c = p52 + tr['corr'].to_numpy()
    p53, why = apply_rules(tr, p_c, st)
    p_rules_only, why_r = apply_rules(tr, p52, st)
    e52, e53, ec, er = (p52 - y) ** 2, (p53 - y) ** 2, (p_c - y) ** 2, (p_rules_only - y) ** 2
    rep = {'rows': int(len(tr)), 'rmse_v52_proxy': rmse(e52), 'rmse_v53_proxy': rmse(e53), 'delta': rmse(e53) - rmse(e52),
           'components': {'opdi_correction_only': rmse(ec) - rmse(e52), 'rules_only (B + VFR guard)': rmse(er) - rmse(e52)},
           'status_2025': pd.Series(st).value_counts().to_dict(),
           'rules_moved_2025': {k: int((why == k).sum()) for k in ('W_low', 'W_high', 'F', 'R', 'B')},
           'band_rule_2025': {'moved': int((why_r == 'B').sum()), 'worse': int(((er > e52 + 1e-9) & (why_r == 'B')).sum()),
                              'sched_far_labels_inside_band': int(((st == 'sched_far') & (np.abs(y - (tr.t - tr.bf_lobt).to_numpy()) < 3606)).sum()),
                              'sched_far_labels_above_band': int(((st == 'sched_far') & (y >= (tr.t - tr.bf_lobt).to_numpy() + 3606)).sum())},
           'opdi_rows_2025': int((tr['corr'] != 0).sum()), 'by_fold': {}, 'by_month': {}, 'by_airport': {}}
    for col, key in (('fold', 'by_fold'), ('month', 'by_month'), ('apt', 'by_airport')):
        for g, idx in tr.groupby(col).indices.items():
            rep[key][str(g)] = {'rows': int(len(idx)), 'delta': rmse(e53[idx]) - rmse(e52[idx])}
    rep['folds_worse'] = sum(v['delta'] > 0 for v in rep['by_fold'].values())
    rep['months_worse'] = sum(v['delta'] > 0 for v in rep['by_month'].values())
    rep['airports_worse'] = [k for k, v in rep['by_airport'].items() if v['delta'] > 1e-12]
    pd.DataFrame({'MVT_ID_mvt': tr.MVT_ID_mvt, 'v52_proxy': p52, 'opdi_corr': tr['corr'], 'v53_proxy': p53, 'rule': why,
                  'status': st}).to_parquet(OUT / 'rows_2025.parquet', index=False)
    (OUT / 'evidence.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k not in ('by_month',)}, indent=2))
    return rep


def sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build():
    if V53.exists():
        raise SystemExit(f'{V53} exists; refusing to overwrite')
    f = frame()
    rk = f[f.split == 'rank']
    s52 = pd.read_parquet(V52)
    assert len(s52) == final_rank.n_rank() and s52.MVT_ID_mvt.is_unique   # final-file change: was == 344841
    x = s52[['MVT_ID_mvt']].merge(rk, on='MVT_ID_mvt', how='left', validate='one_to_one')
    assert x.split.notna().all(), 'a v52 row is missing from the ranking departures'
    corr = pd.concat([pd.read_parquet(opdi.OUT / f'rank_{a}.parquet', columns=['MVT_ID_mvt', 'corr']) for a in OPDI_APTS])
    x = x.merge(corr, on='MVT_ID_mvt', how='left', validate='one_to_one')
    x['corr'] = x['corr'].fillna(0.0)
    assert (x.loc[x.is_heli == 1, 'corr'] == 0).all()
    p = s52.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    st = status(x)
    q, why = apply_rules(x, p + x['corr'].to_numpy(), st)
    heli = (x.is_heli == 1).to_numpy()
    q[heli] = p[heli]
    sub = s52.copy()
    sub['TAXITIME_SEC_mvt'] = np.round(q).astype(s52.TAXITIME_SEC_mvt.dtype) if np.issubdtype(s52.TAXITIME_SEC_mvt.dtype, np.integer) else q
    v = sub.TAXITIME_SEC_mvt.astype(float)
    assert np.isfinite(v).all() and (v >= FLOOR).all() and (sub.MVT_ID_mvt.to_numpy() == s52.MVT_ID_mvt.to_numpy()).all()
    sub.to_parquet(V53, index=False)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pkg = common.RUNS / 'models' / f'A61_{stamp}_v53'
    pkg.mkdir(parents=True)
    moved = pd.DataFrame({'MVT_ID_mvt': x.MVT_ID_mvt, 'APT': x.apt, 'status': st, 'opdi_corr': x['corr'], 'rule': why,
                          'v52': p, 'v53': v})
    moved = moved[(moved.v53 != moved.v52)]
    moved.to_parquet(pkg / 'moved_rows.parquet', index=False)
    d = (v - p)
    ev = OUT / 'evidence.json'
    info = {'approach': 'A61', 'version': 53, 'base': V52.name, 'base_sha256': sha(V52), 'file_sha256': sha(V53),
            'opdi_airports': list(OPDI_APTS), 'opdi_rows': int((x['corr'] != 0).sum()),
            'opdi_corr_rms': float(np.sqrt((x['corr'] ** 2).mean() * len(x) / max((x['corr'] != 0).sum(), 1))),
            'rules': {'W_CLIP': W_CLIP, 'F_GAP': F_GAP, 'OP_MIN': OP_MIN, 'vfr_guard': True, 'band': 'sched_far inside +-F_GAP -> lower edge'},
            'rules_moved_2026': {k: int((why == k).sum()) for k in ('W_low', 'W_high', 'F', 'R', 'B')},
            'status_2026': pd.Series(st).value_counts().to_dict(), 'rows': int(len(sub)), 'rows_changed': int((d != 0).sum()),
            'rows_changed_by_airport': moved.APT.value_counts().to_dict(), 'rms_change_all_rows': float(np.sqrt(np.mean(d ** 2))),
            'band_rows': moved[moved.rule == 'B'][['MVT_ID_mvt', 'APT', 'v52', 'v53']].to_dict('records'),
            'opdi_rank_sha256': {a: sha(opdi.OUT / f'rank_{a}.parquet') for a in OPDI_APTS}, 'backfill_sha256': sha(BACKFILL)}
    (pkg / 'build.json').write_text(json.dumps(info, indent=2))
    for s in ('263_v53_build.py', '262_opdi_correction.py'):
        (pkg / s).write_bytes((common.ROOT / 'scripts' / s).read_bytes())
    for a in OPDI_APTS:
        (pkg / f'rank_{a}.parquet').write_bytes((opdi.OUT / f'rank_{a}.parquet').read_bytes())
        (pkg / f'nested_{a}.json').write_bytes((opdi.OUT / f'nested_{a}.json').read_bytes())
    if ev.exists():
        (pkg / 'evidence.json').write_bytes(ev.read_bytes())
    print(json.dumps({k: v for k, v in info.items() if k != 'band_rows'}, indent=2))
    print('band rows:', info['band_rows'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['evidence', 'build'])
    a = ap.parse_args()
    evidence() if a.mode == 'evidence' else build()
