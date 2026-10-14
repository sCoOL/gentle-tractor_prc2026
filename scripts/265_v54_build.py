"""v54 = v53 + the EDDF stand-level OPDI correction (scripts/262 stand_nested / stand_fit), NM-window rules re-applied.

  py -3.14 scripts/265_v54_build.py evidence   # 2025 check on the v53 proxy -> runs/v54_build/evidence.json
  py -3.14 scripts/265_v54_build.py build      # submissions/gentle-tractor_v54.parquet + runs/models/A62_*_v54/
  py -3.14 scripts/265_v54_build.py ledger     # runs/v54_build/rows.parquet: per-row v53/v54 values for the explorer

2025: v53 proxy before its rules (v52 proxy + LSZH OPDI correction) + the nested out-of-fold EDDF correction, then
scripts/263's W/F/R/B rules. 2026: v53 file + the full-2025 EDDF correction (3 seeds), then the same rules (idempotent on
v53's own rows). Only EDDF rows with an OPDI stand or apron exit move. No official score is used.

Plus the sched_far cap: a 'sched_far' prediction ABOVE the +-1 h band also
moves to the band's lower edge, take-off - recovered LOBT - 3,540 s (rule B moved only predictions inside the band). 2025:
all 100 sched_far labels lie below the band (98 with the schedule > 1 h before LOBT, where a copy would lie above it) and no
v53-proxy prediction sits above it, so the 2025 check cannot see the cap. 2026: one row, LIRF RYR8ZF (MVT 213141547),
10,112 s -> 2,103 s in the original run.
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

b53 = importlib.import_module('263_v53_build')
opdi = importlib.import_module('262_opdi_correction')
OUT = common.RUNS / 'v54_build'
V53, V54 = common.SUBS / 'gentle-tractor_v53.parquet', common.SUBS / 'gentle-tractor_v54.parquet'
APT = 'EDDF'
STRICT_FIXED_POINT = True   # final-file change: see build()


def rmse(e2) -> float:
    return float(np.sqrt(np.mean(e2)))


def cap_sched_far(f: pd.DataFrame, q: np.ndarray, st: np.ndarray, why: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sched_far rows predicted above the band's lower edge (after rule B, i.e. above the band) -> the lower edge."""
    out, why = q.copy(), why.copy()
    edge = f.t.to_numpy() - f.bf_lobt.to_numpy() - b53.F_GAP
    k = (st == 'sched_far') & (out > edge)
    out[k], why[k] = edge[k], 'B_cap'
    return out, why


def rules_v54(f: pd.DataFrame, p: np.ndarray, st: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q, why = b53.apply_rules(f, p, st)
    return cap_sched_far(f, q, st, why)


def evidence():
    OUT.mkdir(parents=True, exist_ok=True)
    f = b53.frame()
    rows = pd.read_parquet(b53.OUT / 'rows_2025.parquet')
    tr = f[f.split == 'train'].merge(rows, on='MVT_ID_mvt', validate='one_to_one') \
        .merge(pd.read_parquet(b53.OOF, columns=['MVT_ID_mvt', 'fold']), on='MVT_ID_mvt', validate='one_to_one')
    corr = pd.read_parquet(opdi.OUT / f'oof_{APT}_stand.parquet')
    tr = tr.merge(corr, on='MVT_ID_mvt', how='left')
    tr['corr'] = tr['corr'].fillna(0.0)
    st = b53.status(tr)
    pre = tr.v52_proxy.to_numpy() + tr.opdi_corr.to_numpy()
    assert np.allclose(b53.apply_rules(tr, pre, st)[0], tr.v53_proxy.to_numpy())
    p54, why = rules_v54(tr, pre + tr['corr'].to_numpy(), st)
    y, p53 = tr.y.to_numpy(), tr.v53_proxy.to_numpy()
    e53, e54 = (p53 - y) ** 2, (p54 - y) ** 2
    rep = {'rows': int(len(tr)), 'rmse_v53_proxy': rmse(e53), 'rmse_v54_proxy': rmse(e54), 'delta': rmse(e54) - rmse(e53),
           'rows_moved': int((p54 != p53).sum()), 'rows_worse_100s': int((np.sqrt(e54) - np.sqrt(e53) > 100).sum()),
           'rows_better_100s': int((np.sqrt(e53) - np.sqrt(e54) > 100).sum()),
           'sched_far_cap_2025': {'rows_moved': int((why == 'B_cap').sum()),
                                  'sched_far_labels_above_band': int(((st == 'sched_far') & (y >= (tr.t - tr.bf_lobt).to_numpy() + 3606)).sum()),
                                  'sched_far_rows': int((st == 'sched_far').sum())},
           'by_fold': {}, 'by_month': {}, 'by_airport': {}}
    for col, key in (('fold', 'by_fold'), ('month', 'by_month'), ('apt', 'by_airport')):
        for g, idx in tr.groupby(col).indices.items():
            rep[key][str(g)] = {'rows': int(len(idx)), 'moved': int((p54[idx] != p53[idx]).sum()), 'delta': rmse(e54[idx]) - rmse(e53[idx])}
    rep['folds_worse'] = sum(v['delta'] > 1e-12 for v in rep['by_fold'].values())
    rep['months_worse'] = sum(v['delta'] > 1e-12 for v in rep['by_month'].values())
    nst = json.loads((opdi.OUT / f'nested_{APT}_stand.json').read_text())
    per = {s: nst['dsse_M_by_state'][s] * 1e6 / max(nst['rows_2025_by_state'].get(s, 0), 1) for s in opdi.STATES}
    dsse26 = sum(per[s] * nst['rows_2026_by_state'].get(s, 0) for s in opdi.STATES)
    rep['projection_2026'] = {'per_row_2025_by_state': {k: round(v, 0) for k, v in per.items()}, 'dsse_M': round(dsse26 / 1e6, 1)}
    (OUT / 'evidence.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k not in ('by_month', 'by_airport')}, indent=2))


def sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build():
    if V54.exists():
        raise SystemExit(f'{V54} exists; refusing to overwrite')
    f = b53.frame()
    rk = f[f.split == 'rank']
    s53 = pd.read_parquet(V53)
    assert len(s53) == final_rank.n_rank() and s53.MVT_ID_mvt.is_unique   # final-file change: was == 344841
    x = s53[['MVT_ID_mvt']].merge(rk, on='MVT_ID_mvt', how='left', validate='one_to_one')
    corr = pd.read_parquet(opdi.OUT / f'rank_{APT}_stand.parquet', columns=['MVT_ID_mvt', 'corr'])
    x = x.merge(corr, on='MVT_ID_mvt', how='left', validate='one_to_one')
    x['corr'] = x['corr'].fillna(0.0)
    assert (x.loc[x.is_heli == 1, 'corr'] == 0).all() and (x.loc[x.apt != APT, 'corr'] == 0).all()
    p = s53.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    st = b53.status(x)
    if STRICT_FIXED_POINT:   # final-file change: on in every chained run
        assert np.allclose(b53.apply_rules(x, p, st)[0], np.maximum(p, b53.FLOOR)), 'v53 is not a fixed point of the rules'
    q, why = rules_v54(x, p + x['corr'].to_numpy(), st)
    heli = (x.is_heli == 1).to_numpy()
    q[heli] = p[heli]
    sub = s53.copy()
    sub['TAXITIME_SEC_mvt'] = q
    v = sub.TAXITIME_SEC_mvt.astype(float)
    assert np.isfinite(v).all() and (v >= b53.FLOOR).all() and (sub.MVT_ID_mvt.to_numpy() == s53.MVT_ID_mvt.to_numpy()).all()
    changed = v.to_numpy() != p
    assert set(x.apt[changed & (why != 'B_cap')]) <= {APT} or not STRICT_FIXED_POINT   # final-file change: same flag
    sub.to_parquet(V54, index=False)
    pkg = common.RUNS / 'models' / f'A62_{datetime.now().strftime("%Y%m%d_%H%M%S")}_v54'
    pkg.mkdir(parents=True)
    pd.DataFrame({'MVT_ID_mvt': x.MVT_ID_mvt, 'APT': x.apt, 'opdi_corr': x['corr'], 'rule': why, 'v53': p, 'v54': v})[changed] \
        .to_parquet(pkg / 'moved_rows.parquet', index=False)
    d = v.to_numpy() - p
    info = {'approach': 'A62', 'version': 54, 'base': V53.name, 'base_sha256': sha(V53), 'file_sha256': sha(V54),
            'opdi_stand_airport': APT, 'pool': list(opdi.STAND_POOL), 'rows_changed': int(changed.sum()),
            'rules_moved_2026': {k: int((why == k).sum()) for k in ('W_low', 'W_high', 'F', 'R', 'B', 'B_cap')},
            'sched_far_cap_rows': pd.DataFrame({'MVT_ID_mvt': x.MVT_ID_mvt, 'APT': x.apt, 'v53': p, 'v54': q})[why == 'B_cap'].to_dict('records'),
            'rms_change_changed_rows': float(np.sqrt(np.mean(d[changed] ** 2))), 'mean_change_changed_rows': float(d[changed].mean()),
            'rms_change_all_rows': float(np.sqrt(np.mean(d ** 2))), 'opdi_rank_sha256': sha(opdi.OUT / f'rank_{APT}_stand.parquet')}
    (pkg / 'build.json').write_text(json.dumps(info, indent=2))
    for s in ('265_v54_build.py', '262_opdi_correction.py', '263_v53_build.py'):
        (pkg / s).write_bytes((common.ROOT / 'scripts' / s).read_bytes())
    for name in (f'rank_{APT}_stand.parquet', f'nested_{APT}_stand.json'):
        (pkg / name).write_bytes((opdi.OUT / name).read_bytes())
    ev = OUT / 'evidence.json'
    if ev.exists():
        (pkg / 'evidence.json').write_bytes(ev.read_bytes())
    print(json.dumps(info, indent=2))


def ledger():
    """One row per departure, both years: v53 / v54 prediction (2025: the proxies, 2026: the files), the OPDI correction
    each row received (LSZH all-event model of v53, EDDF stand-event model of v54; 2025 out of fold) and the rule that
    set the final value. Read by explorer/db.py (columns v54_*)."""
    OUT.mkdir(parents=True, exist_ok=True)
    f = b53.frame()
    st = b53.status(f)
    f['status'] = st
    r25 = pd.read_parquet(b53.OUT / 'rows_2025.parquet').drop(columns=['status', 'rule'])
    tr = f[f.split == 'train'].merge(r25, on='MVT_ID_mvt', validate='one_to_one')
    tr = tr.merge(pd.read_parquet(b53.OOF, columns=['MVT_ID_mvt', 'fold']), on='MVT_ID_mvt', validate='one_to_one')
    lsz = pd.read_parquet(opdi.OUT / 'oof_LSZH.parquet').rename(columns={'corr': 'c_lszh'})
    edf = pd.read_parquet(opdi.OUT / f'oof_{APT}_stand.parquet').rename(columns={'corr': 'c_eddf'})
    tr = tr.merge(lsz, on='MVT_ID_mvt', how='left').merge(edf, on='MVT_ID_mvt', how='left')
    pre = tr.v52_proxy.to_numpy() + tr.opdi_corr.to_numpy() + tr.c_eddf.fillna(0).to_numpy()
    p54, why = rules_v54(tr, pre, tr.status.to_numpy())
    a = pd.DataFrame({'MVT_ID_mvt': tr.MVT_ID_mvt, 'split': 'train', 'fold': tr.fold, 'y': tr.y, 'v53_pred_s': tr.v53_proxy, 'v54_pred_s': p54,
                      'opdi_corr_s': tr.c_lszh.fillna(0) + tr.c_eddf.fillna(0),
                      'opdi_model': np.select([tr.c_lszh.notna(), tr.c_eddf.notna()], ['LSZH all events', 'EDDF stand events'], ''),
                      'v54_rule': why, 'v54_status': tr.status})
    rk = f[f.split == 'rank']
    s53 = pd.read_parquet(V53).rename(columns={'TAXITIME_SEC_mvt': 'v53_pred_s'})
    s54 = pd.read_parquet(V54).rename(columns={'TAXITIME_SEC_mvt': 'v54_pred_s'})
    c53 = pd.read_parquet(opdi.OUT / 'rank_LSZH.parquet', columns=['MVT_ID_mvt', 'corr']).rename(columns={'corr': 'c_lszh'})
    c54 = pd.read_parquet(opdi.OUT / f'rank_{APT}_stand.parquet', columns=['MVT_ID_mvt', 'corr']).rename(columns={'corr': 'c_eddf'})
    m53 = pd.read_parquet(sorted((common.RUNS / 'models').glob('A61_*_v53'))[-1] / 'moved_rows.parquet', columns=['MVT_ID_mvt', 'rule'])
    m54 = pd.read_parquet(sorted((common.RUNS / 'models').glob('A62_*_v54'))[-1] / 'moved_rows.parquet', columns=['MVT_ID_mvt', 'rule'])
    x = rk[['MVT_ID_mvt', 'status']].merge(s53, on='MVT_ID_mvt').merge(s54, on='MVT_ID_mvt')
    x = x.merge(c53, on='MVT_ID_mvt', how='left').merge(c54, on='MVT_ID_mvt', how='left')
    x = x.merge(m53.rename(columns={'rule': 'r53'}), on='MVT_ID_mvt', how='left')
    x = x.merge(m54.rename(columns={'rule': 'r54'}), on='MVT_ID_mvt', how='left')
    assert len(x) == final_rank.n_rank()   # final-file change: was == 344841
    rule = np.where(x.r54.fillna('').ne(''), x.r54.fillna(''), x.r53.fillna(''))
    b = pd.DataFrame({'MVT_ID_mvt': x.MVT_ID_mvt, 'split': 'rank', 'fold': None, 'y': np.nan, 'v53_pred_s': x.v53_pred_s.astype(float), 'v54_pred_s': x.v54_pred_s.astype(float),
                      'opdi_corr_s': x.c_lszh.fillna(0) + x.c_eddf.fillna(0),
                      'opdi_model': np.select([x.c_lszh.notna(), x.c_eddf.notna()], ['LSZH all events', 'EDDF stand events'], ''),
                      'v54_rule': rule, 'v54_status': x.status})
    out = pd.concat([a, b], ignore_index=True)
    t = out[out.split == 'train']
    print('2025 v53 / v54 proxy RMSE', rmse((t.v53_pred_s - t.y) ** 2), rmse((t.v54_pred_s - t.y) ** 2), flush=True)
    out.to_parquet(OUT / 'rows.parquet', index=False)
    print(out.groupby(['split', 'opdi_model']).size().to_string())
    print(out.groupby(['split', 'v54_rule']).size().to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['evidence', 'build', 'ledger'])
    a = ap.parse_args()
    {'evidence': evidence, 'build': build, 'ledger': ledger}[a.mode]()
