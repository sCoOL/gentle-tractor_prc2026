"""v56 = v55 + the departure-neighbour residual correction (scripts/269, lead #21) + LEMD 3,606 s cap (+ BEL hedge).

  py -3.14 scripts/270_v56_build.py rank --arm nb       # full-2025 fit (3 seeds) -> runs/v56_build/rank_corr.parquet
  py -3.14 scripts/270_v56_build.py evidence --arm nb   # 2025 check on the v55 proxy -> runs/v56_build/evidence.json
  py -3.14 scripts/270_v56_build.py build [--bel 0.65]  # submissions/gentle-tractor_v56.parquet + runs/models/A64_*_v56/

Correction: LightGBM Huber residual model on y - v55 proxy (scripts/266 fit(), fixed parameters) with the columns of the
269 arm; scope NM-matched rows with AOBT_3, fixed-wing, all ten airports (the nested arm's own scope).
Shrink = the value the inner folds picked most often. 2025 evidence = 269's nested out-of-fold correction.
LEMD cap: 2025 LEMD labels max 3,598 s (0 of 212,242 above 3,600) -> predictions above 3,606 s go to 3,606.
BEL hedge (lead #20, only with --bel q): LIRF unmatched Brussels Airlines rows ->
q * (Rome UTC offset + 776 s) + (1 - q) * v55.
Rules: rules_v55 re-applied after the correction (they only bound labels). No official score is used.
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
v55m = importlib.import_module('267_v55_build')
n269 = importlib.import_module('269_neighbour_lateness')
OUT = common.RUNS / 'v56_build'
V55, V56 = common.SUBS / 'gentle-tractor_v55.parquet', common.SUBS / 'gentle-tractor_v56.parquet'
ROWS55 = common.RUNS / 'v55_build' / 'rows.parquet'
SEEDS, SKIP = (7, 11, 23), ()     # no airport skipped: the nested arm trained and was judged on all ten (LIRF -9.5 M)
LEMD_CAP, BEL_TAXI = 3606.0, 776.0
N_2025, N_2026 = 2085047, None   # final-file change: N_2026 was 344841; now final_rank.n_rank()


def sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rmse(e2) -> float:
    return float(np.sqrt(np.mean(e2)))


def _alpha(arm: str) -> float:
    a = list(json.loads((n269.OUT / f'nested_{arm}.json').read_text())['alpha_by_fold'].values())
    v, c = np.unique(a, return_counts=True)
    return float(v[c.argmax()])


def _rank_frame() -> pd.DataFrame:
    # final-file change: the rank rows' v55_status came from runs/v55_build/rows.parquet (= b53.status() of the old
    # export's rank rows, scripts/267 ledger); it is now computed the same way from the current base tables.
    rows = b53.frame()
    rows = rows[rows.split == 'rank'].copy()
    rows['v55_status'] = b53.status(rows)
    rows = rows[['MVT_ID_mvt', 'v55_status']]
    f = pd.read_parquet(s266.OUT / 'frame.parquet').drop(columns='split')
    ov = pd.read_parquet(s266.OUT / 'overtakes.parquet')
    st = pd.read_parquet(s266.OPDI_FRAME, columns=['MVT_ID_mvt', 'adsb'])   # final-file change: was a literal RUNS/v53_opdi path
    old = pd.read_parquet(common.DATA / 'features_v5_rank.parquet', columns=['MVT_ID_mvt', 'is_heli'] + s266.OLD)
    nb = pd.read_parquet(n269.OUT / 'frame.parquet').drop(columns='split')
    s55 = pd.read_parquet(V55).rename(columns={'TAXITIME_SEC_mvt': 'p'})
    d = rows.merge(f, on='MVT_ID_mvt').merge(ov, on='MVT_ID_mvt').merge(st, on='MVT_ID_mvt').merge(old, on='MVT_ID_mvt') \
        .merge(nb, on='MVT_ID_mvt').merge(s55, on='MVT_ID_mvt')
    assert len(d) == final_rank.n_rank()   # final-file change: was == N_2026 (344841)
    d['p'] = d.p.astype(float)
    d['st'] = d.adsb.map(s266.STATES).astype(int)
    return d


def _scope(d: pd.DataFrame) -> np.ndarray:
    return ((d.v55_status == 'matched').to_numpy() & d.ret_overdue_apt.notna().to_numpy()
            & ~d.apt.astype(str).isin(SKIP).to_numpy() & (d.is_heli.fillna(0) != 1).to_numpy())


def rank(arm: str):
    OUT.mkdir(parents=True, exist_ok=True)
    cols, a = n269.ARMS[arm], _alpha(arm)
    tr = n269.load()
    sc = (tr.v55_status == 'matched').to_numpy() & tr.ret_overdue_apt.notna().to_numpy() & (tr.is_heli.fillna(0) != 1).to_numpy()
    tr = tr[sc].reset_index(drop=True)                      # the scope 269 trained on (all airports)
    r = (tr.y - tr.p).to_numpy()
    rk = _rank_frame()
    rk['apt'] = pd.Categorical(rk.apt, categories=tr.apt.cat.categories)
    keep = _scope(rk)
    h = np.mean([s266.fit(tr, r, cols, seed=s).predict(rk.loc[keep, cols]) for s in SEEDS], axis=0)
    out = pd.DataFrame({'MVT_ID_mvt': rk.MVT_ID_mvt[keep], 'APT': rk.apt[keep].astype(str), 'corr': a * h})
    out.to_parquet(OUT / 'rank_corr.parquet', index=False)
    (OUT / 'rank_corr.json').write_text(json.dumps({'arm': arm, 'alpha': a, 'cols': cols}))
    print(json.dumps({'arm': arm, 'alpha': a, 'rows_2026': int(keep.sum()), 'rms_corr': float(np.sqrt(np.mean(out['corr'] ** 2))),
                      'mean_corr': float(out['corr'].mean()), 'by_apt_mean': out.groupby('APT')['corr'].mean().round(1).to_dict()},
                     indent=1), flush=True)


def _train(arm: str) -> pd.DataFrame:
    f = b53.frame()
    f['status'] = b53.status(f)
    rows = pd.read_parquet(ROWS55, columns=['MVT_ID_mvt', 'split', 'fold', 'v55_pred_s'])
    tr = f[f.split == 'train'].merge(rows[rows.split == 'train'].drop(columns='split'), on='MVT_ID_mvt', validate='one_to_one')
    tr = tr.merge(pd.read_parquet(n269.OUT / f'oof_{arm}.parquet'), on='MVT_ID_mvt', how='left')
    keep = ~tr.apt.isin(SKIP).to_numpy() & (tr.is_heli.fillna(0) != 1).to_numpy()
    tr['corr'] = np.where(keep, tr['corr'].fillna(0.0), 0.0)
    assert len(tr) == N_2025
    return tr


def finish(f: pd.DataFrame, p: np.ndarray, corr: np.ndarray, st: np.ndarray):
    q, why = v55m.rules_v55(f, p + corr, st)
    cap = (f.apt.astype(str) == 'LEMD').to_numpy() & (q > LEMD_CAP)
    q[cap], why[cap] = LEMD_CAP, 'LEMD_cap'
    return q, why


def evidence(arm: str):
    OUT.mkdir(parents=True, exist_ok=True)
    tr = _train(arm)
    st = tr.status.to_numpy()
    y, p = tr.y.to_numpy(), tr.v55_pred_s.to_numpy()
    zero = np.zeros(len(tr))
    arms = {'cap_only': finish(tr, p, zero, st)[0], 'v56': finish(tr, p, tr['corr'].to_numpy(), st)[0]}
    e0 = (p - y) ** 2
    body = y <= 3600
    rep = {'arm': arm, 'rows': int(len(tr)), 'rmse_v55_proxy': rmse(e0)}
    for k, q in arms.items():
        e = (q - y) ** 2
        R0, R1 = rmse(e0), rmse(e)
        part = (e - e0) / (len(y) * (R0 + R1))
        rep[k] = {'rmse': R1, 'delta': R1 - R0, 'delta_body_y_le_3600': float(part[body].sum()), 'delta_tail': float(part[~body].sum()),
                  'rows_moved': int((q != p).sum()), 'rows_worse_100s': int((np.sqrt(e) - np.sqrt(e0) > 100).sum()),
                  'rows_better_100s': int((np.sqrt(e0) - np.sqrt(e) > 100).sum())}
    e = (arms['v56'] - y) ** 2
    for col, key in (('fold', 'by_fold'), ('month', 'by_month'), ('apt', 'by_airport')):
        rep[key] = {str(g): rmse(e[i]) - rmse(e0[i]) for g, i in tr.groupby(col).indices.items()}
    for key in ('by_fold', 'by_month', 'by_airport'):
        rep[key.replace('by_', '') + 's_worse'] = sum(v > 1e-12 for v in rep[key].values())
    (OUT / 'evidence.json').write_text(json.dumps(rep, indent=2))
    pd.DataFrame({'MVT_ID_mvt': tr.MVT_ID_mvt, 'v56_proxy': arms['v56'], 'corr': tr['corr']}).to_parquet(OUT / 'rows_2025.parquet', index=False)
    print(json.dumps({k: v for k, v in rep.items() if k not in ('by_month', 'by_airport')}, indent=2))


def _bel(x: pd.DataFrame, q: np.ndarray, share: float) -> np.ndarray:
    """LIRF unmatched Brussels Airlines rows (lead #20): rows and their hedge target."""
    b = pd.read_parquet(common.DATA / 'base_rank.parquet', columns=['MVT_ID_mvt', 'FLIGHT_mvt', 'FLIGHT_ID_mvt', 'MVT_TIME_UTC_mvt'])
    b = x[['MVT_ID_mvt']].merge(b, on='MVT_ID_mvt', how='left', validate='one_to_one')
    rows = ((x.apt.astype(str) == 'LIRF').to_numpy() & b.FLIGHT_ID_mvt.isna().to_numpy()
            & b.FLIGHT_mvt.str.startswith('BEL', na=False).to_numpy())
    t = pd.to_datetime(b.MVT_TIME_UTC_mvt, utc=True)
    off = (t.dt.tz_convert('Europe/Rome').apply(lambda z: z.utcoffset().total_seconds() if pd.notna(z) else np.nan)).to_numpy()
    tgt = off + BEL_TAXI
    out = q.copy()
    out[rows] = share * tgt[rows] + (1 - share) * q[rows]
    return out


def build(bel: float | None):
    if V56.exists():
        raise SystemExit(f'{V56} exists; refusing to overwrite')
    f = b53.frame()
    rk = f[f.split == 'rank']
    s55 = pd.read_parquet(V55)
    assert len(s55) == final_rank.n_rank() and s55.MVT_ID_mvt.is_unique   # final-file change: was == N_2026 (344841)
    x = s55[['MVT_ID_mvt']].merge(rk, on='MVT_ID_mvt', how='left', validate='one_to_one')
    corr = pd.read_parquet(OUT / 'rank_corr.parquet', columns=['MVT_ID_mvt', 'corr'])
    x = x.merge(corr, on='MVT_ID_mvt', how='left', validate='one_to_one')
    x['corr'] = x['corr'].fillna(0.0)
    assert (x.loc[x.is_heli == 1, 'corr'] == 0).all() and (x.loc[x.apt.isin(SKIP), 'corr'] == 0).all()
    p = s55.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    st = b53.status(x)
    q, why = finish(x, p, x['corr'].to_numpy(), st)
    if bel is not None:
        q2 = _bel(x, q, bel)
        why[q2 != q] = 'BEL'
        q = q2
    heli = (x.is_heli == 1).to_numpy()
    q[heli], why[heli] = p[heli], ''
    sub = s55.copy()
    sub['TAXITIME_SEC_mvt'] = q
    v = sub.TAXITIME_SEC_mvt.astype(float)
    assert np.isfinite(v).all() and (v >= b53.FLOOR).all() and (sub.MVT_ID_mvt.to_numpy() == s55.MVT_ID_mvt.to_numpy()).all()
    changed = v.to_numpy() != p
    sub.to_parquet(V56, index=False)
    pkg = common.RUNS / 'models' / f'A64_{datetime.now().strftime("%Y%m%d_%H%M%S")}_v56'
    pkg.mkdir(parents=True)
    pd.DataFrame({'MVT_ID_mvt': x.MVT_ID_mvt, 'APT': x.apt, 'nb_corr': x['corr'], 'rule': why, 'v55': p, 'v56': v})[changed] \
        .to_parquet(pkg / 'moved_rows.parquet', index=False)
    d = v.to_numpy() - p
    ev = OUT / 'evidence.json'
    info = {'approach': 'A64', 'version': 56, 'base': V55.name, 'base_sha256': sha(V55), 'file_sha256': sha(V56),
            **json.loads((OUT / 'rank_corr.json').read_text()), 'seeds': list(SEEDS), 'skip_airports': list(SKIP),
            'lemd_cap_s': LEMD_CAP, 'bel_share': bel, 'rows_changed': int(changed.sum()), 'rows_corrected': int((x['corr'] != 0).sum()),
            'rules_moved_2026': {k: int((why == k).sum()) for k in ('W_low', 'W_high', 'F', 'R', 'B', 'B_cap', 'LEMD_cap', 'BEL')},
            'rms_change_changed_rows': float(np.sqrt(np.mean(d[changed] ** 2))), 'mean_change_changed_rows': float(d[changed].mean()),
            'rms_change_all_rows': float(np.sqrt(np.mean(d ** 2))), 'rank_corr_sha256': sha(OUT / 'rank_corr.parquet')}
    if ev.exists():
        e = json.loads(ev.read_text())
        info['evidence_2025'] = {'by_2025_total': e['v56']['delta'], 'by_2025_body_share': e['v56']['delta_body_y_le_3600']}
    (pkg / 'build.json').write_text(json.dumps(info, indent=2))
    for s in ('270_v56_build.py', '269_neighbour_lateness.py', '267_v55_build.py', '266_gate_retention_screen.py', '263_v53_build.py'):
        (pkg / s).write_bytes((common.ROOT / 'scripts' / s).read_bytes())
    for src in (OUT / 'rank_corr.parquet', OUT / 'rank_corr.json', ev):
        if src.exists():
            (pkg / src.name).write_bytes(src.read_bytes())
    print(json.dumps(info, indent=2))


def ledger():
    """runs/v56_build/rows.parquet, one row per departure, both years: v55 / v56 / v57 (2025: proxies with the nested
    out-of-fold correction and the fold-crossed reuse layer; the BEL hedge is not in the 2025 proxy; 2026: the files)."""
    r268 = importlib.import_module('268_v56_stand_reuse')
    rows = pd.read_parquet(ROWS55, columns=['MVT_ID_mvt', 'split', 'fold', 'y', 'v55_pred_s', 'v55_status'])
    tr = rows[rows.split == 'train'].merge(pd.read_parquet(OUT / 'rows_2025.parquet'), on='MVT_ID_mvt', validate='one_to_one')
    tr = tr.rename(columns={'v56_proxy': 'v56_pred_s', 'corr': 'nb_corr_s'})
    f = r268._cells(pd.read_parquet(r268.OUT / 'table.parquet'))
    t = f[(f.split == 'train') & r268._layer_scope(f)].reset_index(drop=True)
    p, q = t.v55.to_numpy(), t.v55.to_numpy().copy()
    for k in sorted(t.fold.unique()):
        h = (t.fold == k).to_numpy()
        q[h] = r268._apply(t[h], p[h], r268._contrast(t[~h], p[~h]))
    lay = pd.Series(q, index=t.MVT_ID_mvt.to_numpy())
    tr['v57_pred_s'] = tr.v56_pred_s.to_numpy()
    m = tr.MVT_ID_mvt.isin(lay.index).to_numpy()
    assert np.allclose(tr.v56_pred_s[m], tr.v55_pred_s[m]), 'layer rows must be untouched by v56'
    tr.loc[m, 'v57_pred_s'] = lay.reindex(tr.MVT_ID_mvt[m]).to_numpy()
    tr['layer'] = m
    s = {v: pd.read_parquet(common.SUBS / f'gentle-tractor_v{v}.parquet').rename(columns={'TAXITIME_SEC_mvt': f'v{v}_pred_s'}) for v in (55, 56, 57)}
    rk = rows[rows.split == 'rank'][['MVT_ID_mvt', 'split', 'fold', 'y', 'v55_status']]
    for v in (55, 56, 57):
        rk = rk.merge(s[v], on='MVT_ID_mvt', validate='one_to_one')
    rk = rk.merge(pd.read_parquet(OUT / 'rank_corr.parquet', columns=['MVT_ID_mvt', 'corr']).rename(columns={'corr': 'nb_corr_s'}),
                  on='MVT_ID_mvt', how='left')
    rk['nb_corr_s'] = rk.nb_corr_s.fillna(0.0)
    rk['layer'] = rk.v57_pred_s != rk.v56_pred_s
    assert len(rk) == final_rank.n_rank() and len(tr) == N_2025   # final-file change: was len(rk) == N_2026 (344841)
    cols = ['MVT_ID_mvt', 'split', 'fold', 'y', 'v55_pred_s', 'v56_pred_s', 'v57_pred_s', 'nb_corr_s', 'layer']
    out = pd.concat([tr[cols], rk[cols]], ignore_index=True)
    e = out[out.split == 'train']
    print('2025 proxy RMSE v55 / v56 / v57:', *(round(rmse((e[f'v{v}_pred_s'] - e.y) ** 2), 4) for v in (55, 56, 57)), flush=True)
    out.to_parquet(OUT / 'rows.parquet', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['rank', 'evidence', 'build', 'ledger'])
    ap.add_argument('--arm', default='nb', choices=list(n269.ARMS))
    ap.add_argument('--bel', type=float, default=None)
    a = ap.parse_args()
    {'rank': lambda: rank(a.arm), 'evidence': lambda: evidence(a.arm), 'build': lambda: build(a.bel), 'ledger': ledger}[a.mode]()
