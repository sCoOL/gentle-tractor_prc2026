"""v55 = v54 + the gate-retention / overtake residual correction (scripts/266 arm 'ret6ov') + tightened NM-window edges.

  py -3.14 scripts/267_v55_build.py rank       # full-2025 fit (3 seeds) -> runs/v55_build/rank_corr.parquet (2026 rows)
  py -3.14 scripts/267_v55_build.py evidence   # 2025 check on the v54 proxy -> runs/v55_build/evidence.json
  py -3.14 scripts/267_v55_build.py build      # submissions/gentle-tractor_v55.parquet + runs/models/A63_*_v55/
  py -3.14 scripts/267_v55_build.py ledger     # runs/v55_build/rows.parquet: per-row v54/v55 values for the explorer

Correction: LightGBM Huber residual model on y - v54 proxy (scripts/266 fit(), fixed parameters), inputs p, airport, adsb.lol
state, five existing queue columns, six gate-retention / runway-cadence columns (read in team outgoing-boat's public repo,
src/pista.py; the 15-min gate-delay column dropped by ablation) and two own overtake columns (same-runway departures that
pushed after my AOBT_3 but took off before me, and the reverse). Scope: NM-matched rows with AOBT_3, fixed-wing, NOT LIRF
(LIRF was worse in the nested test). Shrink 0.5 (picked by the inner folds in all six outer folds). 2025: the nested
out-of-fold correction of scripts/266 (oof_ret6ov); 2026: the mean of three full-2025 fits.

Rules, re-applied after the correction (they only bound labels):
  W  NM-matched: take-off - LOBT +- 3,606 s (was 3,660): max |BLOCK - LOBT| over the 2,062,577 2025 rows with a LOBT
     is exactly 3,606 s.
  F / R / B / B_cap (unmatched, LOBT recovered at the destination): edges at take-off - LOBT +- 3,600 s (was 3,540):
     the closest of the 1,887 2025 labels is 3,602 s from the recovered LOBT (R), so 3,600 leaves 0 labels inside.
  The status classes (scripts/263 status(), 3,540 s thresholds) are unchanged. Floor 30 s; helicopters keep v54.
No official score is used.
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
s266 = importlib.import_module('266_gate_retention_screen')
OUT = common.RUNS / 'v55_build'
V54, V55 = common.SUBS / 'gentle-tractor_v54.parquet', common.SUBS / 'gentle-tractor_v55.parquet'
ROWS54 = common.RUNS / 'v54_build' / 'rows.parquet'
COLS = [c for c in s266.BASE + s266.NEW if c != 'ret_gate_delay_15m'] + s266.OV
ALPHA, SEEDS, SKIP_APT = 0.5, (7, 11, 23), 'LIRF'
W_CLIP, EDGE = 3606.0, 3600.0
N_2025, N_2026 = 2085047, None   # final-file change: N_2026 was 344841; now final_rank.n_rank()


def rmse(e2) -> float:
    return float(np.sqrt(np.mean(e2)))


def sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rules_v55(f: pd.DataFrame, p: np.ndarray, st: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """scripts/263 apply_rules + scripts/265 cap_sched_far with the tightened edges."""
    out, why = p.astype(float).copy(), np.full(len(p), '', dtype=object)
    L = f.t.to_numpy() - f.lobt.to_numpy()
    w = np.isfinite(L)
    low, high = w & (out < L - W_CLIP), w & (out > L + W_CLIP)
    out[low], out[high] = (L - W_CLIP)[low], (L + W_CLIP)[high]
    why[low], why[high] = 'W_low', 'W_high'
    Lb = f.t.to_numpy() - f.bf_lobt.to_numpy()
    fl = (st == 'F_range') & (out < Lb + EDGE)
    out[fl], why[fl] = (Lb + EDGE)[fl], 'F'
    rc = (st == 'R') & (out > Lb - EDGE)
    out[rc], why[rc] = (Lb - EDGE)[rc], 'R'
    bd = (st == 'sched_far') & (np.abs(out - Lb) < EDGE)
    out[bd], why[bd] = (Lb - EDGE)[bd], 'B'
    bc = (st == 'sched_far') & (out > Lb - EDGE)
    out[bc], why[bc] = (Lb - EDGE)[bc], 'B_cap'
    lo = out < b53.FLOOR
    out[lo] = b53.FLOOR
    return out, why


def _rank_frame() -> pd.DataFrame:
    """2026 inputs of the correction, same columns as scripts/266 load() builds for 2025."""
    # final-file change: the rank rows' v54_status came from runs/v54_build/rows.parquet (= b53.status() of the old
    # export's rank rows, scripts/265 ledger); it is now computed the same way from the current base tables, so every row
    # of the final file has one. Rows are limited to the V54 file's ids below (the merge with s54).
    rows = b53.frame()
    rows = rows[rows.split == 'rank'].copy()
    rows['v54_status'] = b53.status(rows)
    rows = rows[['MVT_ID_mvt', 'split', 'v54_status']]
    f = pd.read_parquet(s266.OUT / 'frame.parquet').drop(columns='split')
    ov = pd.read_parquet(s266.OUT / 'overtakes.parquet')
    st = pd.read_parquet(s266.OPDI_FRAME, columns=['MVT_ID_mvt', 'adsb'])   # final-file change: was a literal RUNS/v53_opdi path
    old = pd.read_parquet(common.DATA / 'features_v5_rank.parquet', columns=['MVT_ID_mvt', 'is_heli'] + s266.OLD)
    s54 = pd.read_parquet(V54).rename(columns={'TAXITIME_SEC_mvt': 'p'})
    d = rows.merge(f, on='MVT_ID_mvt').merge(ov, on='MVT_ID_mvt').merge(st, on='MVT_ID_mvt').merge(old, on='MVT_ID_mvt') \
        .merge(s54, on='MVT_ID_mvt')
    assert len(d) == final_rank.n_rank()   # final-file change: was == N_2026 (344841)
    d['p'] = d.p.astype(float)
    d['st'] = d.adsb.map(s266.STATES).astype(int)
    return d


def _scope(d: pd.DataFrame, heli: np.ndarray) -> np.ndarray:
    return ((d.v54_status == 'matched').to_numpy() & d.ret_overdue_apt.notna().to_numpy()
            & (d.apt.astype(str) != SKIP_APT).to_numpy() & ~heli)


def rank():
    OUT.mkdir(parents=True, exist_ok=True)
    tr = s266.load()
    sc = (tr.v54_status == 'matched').to_numpy() & tr.ret_overdue_apt.notna().to_numpy()   # the scope 266 trained on
    tr = tr[sc].reset_index(drop=True)
    r = (tr.y - tr.p).to_numpy()
    rk = _rank_frame()
    rk['apt'] = pd.Categorical(rk.apt, categories=tr.apt.cat.categories)
    keep = _scope(rk, rk.is_heli.fillna(0).astype(bool).to_numpy())
    h = np.mean([s266.fit(tr, r, COLS, seed=s).predict(rk.loc[keep, COLS]) for s in SEEDS], axis=0)
    out = pd.DataFrame({'MVT_ID_mvt': rk.MVT_ID_mvt[keep], 'APT': rk.apt[keep].astype(str), 'corr': ALPHA * h})
    out.to_parquet(OUT / 'rank_corr.parquet', index=False)
    print(json.dumps({'rows_2026': int(keep.sum()), 'rms_corr': float(np.sqrt(np.mean(out['corr'] ** 2))),
                      'mean_corr': float(out['corr'].mean()),
                      'by_apt_mean': out.groupby('APT')['corr'].mean().round(1).to_dict()}, indent=1), flush=True)


def _train() -> pd.DataFrame:
    f = b53.frame()
    f['status'] = b53.status(f)
    rows = pd.read_parquet(ROWS54, columns=['MVT_ID_mvt', 'split', 'fold', 'v54_pred_s'])
    tr = f[f.split == 'train'].merge(rows[rows.split == 'train'].drop(columns='split'), on='MVT_ID_mvt', validate='one_to_one')
    oof = pd.read_parquet(s266.OUT / 'oof_ret6ov.parquet')
    tr = tr.merge(oof, on='MVT_ID_mvt', how='left')
    keep = (tr.apt != SKIP_APT).to_numpy() & (tr.is_heli.fillna(0) != 1).to_numpy()
    tr['corr'] = np.where(keep, tr['corr'].fillna(0.0), 0.0)
    assert len(tr) == N_2025
    return tr


def evidence():
    OUT.mkdir(parents=True, exist_ok=True)
    tr = _train()
    st = tr.status.to_numpy()
    y, p54 = tr.y.to_numpy(), tr.v54_pred_s.to_numpy()
    arms = {'rules_only': rules_v55(tr, p54, st)[0],
            'correction_only': b53.apply_rules(tr, p54 + tr['corr'].to_numpy(), st)[0],
            'v55': rules_v55(tr, p54 + tr['corr'].to_numpy(), st)[0]}
    e54 = (p54 - y) ** 2
    body = y <= 3600
    rep = {'rows': int(len(tr)), 'rmse_v54_proxy': rmse(e54)}
    for k, q in arms.items():
        e = (q - y) ** 2
        R0, R1 = rmse(e54), rmse(e)
        part = (e - e54) / (len(y) * (R0 + R1))      # exact per-row decomposition of the RMSE change
        rep[k] = {'rmse': R1, 'delta': R1 - R0, 'delta_body_y_le_3600': float(part[body].sum()),
                  'delta_tail': float(part[~body].sum()), 'rows_moved': int((q != p54).sum()),
                  'rows_worse_100s': int((np.sqrt(e) - np.sqrt(e54) > 100).sum()),
                  'rows_better_100s': int((np.sqrt(e54) - np.sqrt(e) > 100).sum()),
                  'top20_share_of_body_gain': float(np.sort(part[body])[:20].sum() / part[body].sum()) if part[body].sum() < 0 else None}
    q = arms['v55']
    e55 = (q - y) ** 2
    for col, key in (('fold', 'by_fold'), ('month', 'by_month'), ('apt', 'by_airport')):
        rep[key] = {str(g): rmse(e55[idx]) - rmse(e54[idx]) for g, idx in tr.groupby(col).indices.items()}
    rep['folds_worse'] = sum(v > 1e-12 for v in rep['by_fold'].values())
    rep['months_worse'] = sum(v > 1e-12 for v in rep['by_month'].values())
    rep['airports_worse'] = sum(v > 1e-12 for v in rep['by_airport'].values())
    (OUT / 'evidence.json').write_text(json.dumps(rep, indent=2))
    pd.DataFrame({'MVT_ID_mvt': tr.MVT_ID_mvt, 'v55_proxy': q, 'corr': tr['corr'], 'rule': rules_v55(tr, p54 + tr['corr'].to_numpy(), st)[1]}) \
        .to_parquet(OUT / 'rows_2025.parquet', index=False)
    print(json.dumps({k: v for k, v in rep.items() if k not in ('by_month', 'by_airport')}, indent=2))


def build():
    if V55.exists():
        raise SystemExit(f'{V55} exists; refusing to overwrite')
    f = b53.frame()
    rk = f[f.split == 'rank']
    s54 = pd.read_parquet(V54)
    assert len(s54) == final_rank.n_rank() and s54.MVT_ID_mvt.is_unique   # final-file change: was == N_2026 (344841)
    x = s54[['MVT_ID_mvt']].merge(rk, on='MVT_ID_mvt', how='left', validate='one_to_one')
    corr = pd.read_parquet(OUT / 'rank_corr.parquet', columns=['MVT_ID_mvt', 'corr'])
    x = x.merge(corr, on='MVT_ID_mvt', how='left', validate='one_to_one')
    x['corr'] = x['corr'].fillna(0.0)
    assert (x.loc[x.is_heli == 1, 'corr'] == 0).all() and (x.loc[x.apt == SKIP_APT, 'corr'] == 0).all()
    p = s54.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    st = b53.status(x)
    q, why = rules_v55(x, p + x['corr'].to_numpy(), st)
    heli = (x.is_heli == 1).to_numpy()
    q[heli], why[heli] = p[heli], ''
    sub = s54.copy()
    sub['TAXITIME_SEC_mvt'] = q
    v = sub.TAXITIME_SEC_mvt.astype(float)
    assert np.isfinite(v).all() and (v >= b53.FLOOR).all() and (sub.MVT_ID_mvt.to_numpy() == s54.MVT_ID_mvt.to_numpy()).all()
    changed = v.to_numpy() != p
    sub.to_parquet(V55, index=False)
    pkg = common.RUNS / 'models' / f'A63_{datetime.now().strftime("%Y%m%d_%H%M%S")}_v55'
    pkg.mkdir(parents=True)
    pd.DataFrame({'MVT_ID_mvt': x.MVT_ID_mvt, 'APT': x.apt, 'ret_corr': x['corr'], 'rule': why, 'v54': p, 'v55': v})[changed] \
        .to_parquet(pkg / 'moved_rows.parquet', index=False)
    d = v.to_numpy() - p
    ev = OUT / 'evidence.json'
    info = {'approach': 'A63', 'version': 55, 'base': V54.name, 'base_sha256': sha(V54), 'file_sha256': sha(V55),
            'correction_columns': COLS, 'alpha': ALPHA, 'seeds': list(SEEDS), 'skip_airport': SKIP_APT,
            'w_clip_s': W_CLIP, 'recovered_lobt_edge_s': EDGE, 'rows_changed': int(changed.sum()),
            'rows_corrected': int((x['corr'] != 0).sum()),
            'rules_moved_2026': {k: int((why == k).sum()) for k in ('W_low', 'W_high', 'F', 'R', 'B', 'B_cap')},
            'rms_change_changed_rows': float(np.sqrt(np.mean(d[changed] ** 2))), 'mean_change_changed_rows': float(d[changed].mean()),
            'rms_change_all_rows': float(np.sqrt(np.mean(d ** 2))), 'rank_corr_sha256': sha(OUT / 'rank_corr.parquet')}
    if ev.exists():
        e = json.loads(ev.read_text())
        info['evidence_2025'] = {'by_2025_total': e['v55']['delta'], 'by_2025_body_share': e['v55']['delta_body_y_le_3600']}
    (pkg / 'build.json').write_text(json.dumps(info, indent=2))
    for s in ('267_v55_build.py', '266_gate_retention_screen.py', '265_v54_build.py', '263_v53_build.py'):
        (pkg / s).write_bytes((common.ROOT / 'scripts' / s).read_bytes())
    for src in (OUT / 'rank_corr.parquet', ev, s266.OUT / 'nested_ret6ov.json'):
        if src.exists():
            (pkg / src.name).write_bytes(src.read_bytes())
    print(json.dumps(info, indent=2))


def ledger():
    """One row per departure, both years: v54 / v55 prediction (2025: proxies, 2026: files), the correction each row got
    (2025 out of fold) and the rule that set the final value. Read by explorer/db.py (columns v55_*)."""
    tr = _train()
    r25 = pd.read_parquet(OUT / 'rows_2025.parquet')
    a = tr[['MVT_ID_mvt', 'fold', 'y', 'v54_pred_s', 'status']].merge(r25, on='MVT_ID_mvt', validate='one_to_one')
    a = pd.DataFrame({'MVT_ID_mvt': a.MVT_ID_mvt, 'split': 'train', 'fold': a.fold, 'y': a.y, 'v54_pred_s': a.v54_pred_s,
                      'v55_pred_s': a.v55_proxy, 'ret_corr_s': a['corr'], 'v55_rule': a.rule, 'v55_status': a.status})
    f = b53.frame()
    f['status'] = b53.status(f)
    rk = f[f.split == 'rank'][['MVT_ID_mvt', 'status']]
    s54 = pd.read_parquet(V54).rename(columns={'TAXITIME_SEC_mvt': 'v54_pred_s'})
    s55 = pd.read_parquet(V55).rename(columns={'TAXITIME_SEC_mvt': 'v55_pred_s'})
    c = pd.read_parquet(OUT / 'rank_corr.parquet', columns=['MVT_ID_mvt', 'corr'])
    m = pd.read_parquet(sorted((common.RUNS / 'models').glob('A63_*_v55'))[-1] / 'moved_rows.parquet', columns=['MVT_ID_mvt', 'rule'])
    x = rk.merge(s54, on='MVT_ID_mvt').merge(s55, on='MVT_ID_mvt').merge(c, on='MVT_ID_mvt', how='left').merge(m, on='MVT_ID_mvt', how='left')
    assert len(x) == final_rank.n_rank()   # final-file change: was == N_2026 (344841)
    b = pd.DataFrame({'MVT_ID_mvt': x.MVT_ID_mvt, 'split': 'rank', 'fold': None, 'y': np.nan, 'v54_pred_s': x.v54_pred_s.astype(float),
                      'v55_pred_s': x.v55_pred_s.astype(float), 'ret_corr_s': x['corr'].fillna(0.0), 'v55_rule': x.rule.fillna(''),
                      'v55_status': x.status})
    out = pd.concat([a, b], ignore_index=True)
    t = out[out.split == 'train']
    print('2025 v54 / v55 proxy RMSE', rmse((t.v54_pred_s - t.y) ** 2), rmse((t.v55_pred_s - t.y) ** 2), flush=True)
    out.to_parquet(OUT / 'rows.parquet', index=False)
    print(out.groupby(['split', 'v55_rule']).size().to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['rank', 'evidence', 'build', 'ledger'])
    a = ap.parse_args()
    {'rank': rank, 'evidence': evidence, 'build': build, 'ledger': ledger}[a.mode]()
