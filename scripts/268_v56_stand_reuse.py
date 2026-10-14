"""Stand reuse (lead #19, OpenAir's E45 "stand release" idea) for LIRF NM-unmatched departures: the v57 layer (A65).

  py -3.14 scripts/268_v56_stand_reuse.py table     # runs/v56_reuse/table.parquet: 264's lane table + reuse signals + v55
  py -3.14 scripts/268_v56_stand_reuse.py layer     # the separate reuse layer on v56 -> v57 (runs/v56_reuse/layer.json)

Signal (label-free): another movement used the departure's recorded stand between schedule + 5 min and take-off - 5 min,
i.e. another departure's take-off or an arrival's recorded in-block (arrivals are not scored, so their block time is
known in 2026; no departure block time is read). Columns:
    reused        any such event
    n_rel         how many
    rel_gap_h     take-off - first such event (h), NaN when not reused
    rel_after_h   first such event - schedule (h), NaN when not reused
The layer (below) moves LIRF gate-valid unmatched rows with 1 h < D <= 4 h toward D (reused) or away from it (not reused),
judged out of fold by fold on the v55 2025 proxy. No official score is used.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

g264 = importlib.import_module('264_v54_unmatched_gate')
b53 = importlib.import_module('263_v53_build')
v55 = importlib.import_module('267_v55_build')
OUT = common.RUNS / 'v56_reuse'
ROWS55 = common.RUNS / 'v55_build' / 'rows.parquet'
REUSE = ['reused', 'n_rel', 'rel_gap_h', 'rel_after_h']
N_2025, R_V55 = 2085047, 232.2726   # rows and RMSE of the v55 2025 proxy
ALPHAS = np.array([0, .25, .5, .75, 1.0])


def reuse_signals() -> pd.DataFrame:
    con = duckdb.connect()
    d = str(common.DATA).replace('\\', '/')
    mv = con.sql(f"""
     SELECT MVT_ID_mvt, PHASE_mvt, ADEP_mvt, ADES_mvt, STAND_mvt, epoch(MVT_TIME_UTC_mvt) t, epoch(BLOCK_TIME_UTC_mvt) blk,
            epoch(SCHED_TIME_UTC_mvt) sch FROM read_parquet('{d}/training_*.parquet')
     UNION ALL
     SELECT MVT_ID_mvt, PHASE_mvt, ADEP_mvt, ADES_mvt, STAND_mvt, epoch(MVT_TIME_UTC_mvt), epoch(BLOCK_TIME_UTC_mvt),
            epoch(SCHED_TIME_UTC_mvt) FROM read_parquet('{d}/ranking.parquet')""").df()
    mv['apt'] = np.where(mv.PHASE_mvt == 'DEP', mv.ADEP_mvt, mv.ADES_mvt)
    ok = mv.STAND_mvt.notna() & ~mv.STAND_mvt.isin(['', 'NA'])
    dep = mv[(mv.PHASE_mvt == 'DEP') & ok][['MVT_ID_mvt', 'apt', 'STAND_mvt', 't', 'sch']]
    ev = pd.concat([mv[(mv.PHASE_mvt == 'DEP') & ok][['MVT_ID_mvt', 'apt', 'STAND_mvt', 't']].rename(columns={'t': 'e'}),
                    mv[(mv.PHASE_mvt == 'ARR') & ok][['MVT_ID_mvt', 'apt', 'STAND_mvt', 'blk']].rename(columns={'blk': 'e'})]
                   ).dropna(subset=['e'])
    con.register('dep', dep)
    con.register('ev', ev)
    j = con.sql("""SELECT d.MVT_ID_mvt, d.t, d.sch, count(e.e) AS n_rel, min(e.e) AS first_rel FROM dep d LEFT JOIN ev e
                   ON e.apt = d.apt AND e.STAND_mvt = d.STAND_mvt AND e.MVT_ID_mvt <> d.MVT_ID_mvt
                      AND e.e > d.sch + 300 AND e.e < d.t - 300
                   GROUP BY d.MVT_ID_mvt, d.t, d.sch""").df()
    return pd.DataFrame({'MVT_ID_mvt': j.MVT_ID_mvt.astype(float), 'reused': (j.n_rel > 0).astype(float),
                         'n_rel': j.n_rel.astype(float), 'rel_gap_h': (j.t - j.first_rel) / 3600.0,
                         'rel_after_h': (j.first_rel - j.sch) / 3600.0})


def table():
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(g264.OUT / 'table.parquet')
    f = f.merge(reuse_signals(), on='MVT_ID_mvt', how='left', validate='one_to_one')   # no stand -> NaN (unknown)
    r = pd.read_parquet(ROWS55, columns=['MVT_ID_mvt', 'split', 'v55_pred_s', 'v55_status'])
    s55 = pd.read_parquet(v55.V55).rename(columns={'TAXITIME_SEC_mvt': 'v55_file'})
    # final-file change: runs/v55_build/rows.parquet holds the 2025 proxy and the OLD export's rank rows. Keep its 2025
    # rows; the rank rows' v55_status is b53.status() of the current base tables (as scripts/267 ledger computed it) for
    # every id of the V55 file, and the 264 table's rank rows are limited to those ids.
    r = r[r.split == 'train']
    fb = b53.frame()
    fb = fb[(fb.split == 'rank') & fb.MVT_ID_mvt.isin(s55.MVT_ID_mvt)]
    r = pd.concat([r, pd.DataFrame({'MVT_ID_mvt': fb.MVT_ID_mvt.to_numpy(), 'split': 'rank', 'v55_pred_s': np.nan,
                                    'v55_status': b53.status(fb)})], ignore_index=True)
    f = f[(f.split == 'train') | f.MVT_ID_mvt.isin(s55.MVT_ID_mvt)].reset_index(drop=True)
    r = r.merge(s55, on='MVT_ID_mvt', how='left')
    r['v55'] = np.where(r.split == 'rank', r.v55_file, r.v55_pred_s)
    f = f.merge(r[['MVT_ID_mvt', 'split', 'v55', 'v55_status']], on=['MVT_ID_mvt', 'split'], how='left', validate='one_to_one')
    fr = b53.frame()[['MVT_ID_mvt', 't', 'lobt', 'bf_lobt']]
    f = f.merge(fr, on='MVT_ID_mvt', how='left', validate='one_to_one')
    assert f.v55.notna().all()
    f.to_parquet(OUT / 'table.parquet', index=False)
    print(f.groupby(['split', 'APT']).agg(rows=('MVT_ID_mvt', 'size'), has_stand=('n_rel', lambda s: s.notna().mean()),
                                           reused=('reused', 'mean')).round(3).unstack(0).to_string(), flush=True)


def feats(f: pd.DataFrame, cols) -> pd.DataFrame:
    x = g264.feats(f.assign(**{c: np.nan for c in g264.SIGNALS if c not in f}), [c for c in cols if c not in REUSE])
    for c in REUSE:
        if c in cols:
            x[c] = f[c].to_numpy()
    return x[list(cols)]


def report(tr: pd.DataFrame, q: np.ndarray) -> dict:
    y, p = tr.y.to_numpy(), tr.v55.to_numpy()
    de = (q - y) ** 2 - (p - y) ** 2
    R1 = np.sqrt(R_V55 ** 2 + de.sum() / N_2025)
    part = de / (N_2025 * (R_V55 + R1))
    body = y <= 3600
    Db = pd.cut(tr.D / 3600, [-99, 1, 2, 4, 7], labels=['<1h', '1-2h', '2-4h', '4-7h']).astype(str).to_numpy()
    rep = {'rows': int(len(tr)), 'dsse_M': round(float(de.sum() / 1e6), 2), 'd_rmse_vs_v55_proxy': float(R1 - R_V55),
           'd_body_y_le_3600': float(part[body].sum()), 'd_tail': float(part[~body].sum()),
           'rows_moved_100s': int((np.abs(q - p) > 100).sum()),
           'rows_better_100s': int((np.abs(p - y) - np.abs(q - y) > 100).sum()),
           'rows_worse_100s': int((np.abs(q - y) - np.abs(p - y) > 100).sum()),
           'dsse_M_by_fold': {str(k): round(float(de[i].sum() / 1e6), 2) for k, i in tr.groupby('fold').indices.items()},
           'dsse_M_by_airport': {str(k): round(float(de[i].sum() / 1e6), 2) for k, i in tr.groupby('APT').indices.items()},
           'dsse_M_by_delay': {k: round(float(de[Db == k].sum() / 1e6), 2) for k in ('<1h', '1-2h', '2-4h', '4-7h')},
           'dsse_M_by_month': {str(int(k)): round(float(de[i].sum() / 1e6), 2) for k, i in tr.groupby('month').indices.items()}}
    gains = np.sort(de)
    rep['top10_rows_share_of_gain'] = float(gains[:10].sum() / de.sum()) if de.sum() < 0 else None
    rep['folds_worse'] = sum(v > 0 for v in rep['dsse_M_by_fold'].values())
    rep['months_worse'] = sum(v > 0 for v in rep['dsse_M_by_month'].values())
    return rep


# ------------------------------------------------------------------ the separate reuse layer (v57 = A65)
# Used for > 1 h rows only, as a separate step that does not affect any other row. Scope: LIRF gate-valid unmatched rows with 1 h < D <= 4 h (4-7 h was worse on 2025 in
# every shrink, > 7 h is the long tail), BEL rows (lead #20) excluded. Within each delay bucket (1-2 h, 2-4 h) the rows move
# toward D (reused) or away from it (not reused) by the reused-vs-not contrast of z = (y - v55)/(D - v55), weight
# (D - v55)^2, the bucket mean left at 0 (moving the whole bucket toward D lost out of fold). Prior strength = SHRINK
# median weights: 80, the heaviest of the three tried (5 / 20 / 80), also the best out of fold.
LAYER_BUCKETS, SHRINK = [3600.0, 7200.0, 14400.0], 80
V56 = common.SUBS / 'gentle-tractor_v56.parquet'
V56R = common.SUBS / 'gentle-tractor_v57.parquet'   # v57 = v56 + this layer only


def _layer_scope(f: pd.DataFrame) -> np.ndarray:
    return (f.valid.to_numpy() & (f.APT == 'LIRF').to_numpy() & (f.D.to_numpy() > LAYER_BUCKETS[0])
            & (f.D.to_numpy() <= LAYER_BUCKETS[-1]))


def _contrast(t: pd.DataFrame, p: np.ndarray) -> pd.Series:
    """dz per (bucket, reused) from rows t with base prediction p."""
    g = t.D.to_numpy() - p
    z = np.clip((t.y.to_numpy() - p) / g, -1, 1.25)
    d = pd.DataFrame({'zw': z * g ** 2, 'w': g ** 2, 'b': t.b.to_numpy(), 'r': t.r.to_numpy()})
    pri = float(np.median(g ** 2)) * SHRINK
    c = d.groupby(['b', 'r']).sum()
    c['z'] = c.zw / (c.w + pri)
    bb = d.groupby('b').sum()
    bb['zb'] = bb.zw / (bb.w + pri)
    c = c.join(bb.zb, on='b')
    return (c.z - c.zb).rename('dz')


def _cells(f: pd.DataFrame) -> pd.DataFrame:
    return f.assign(b=np.searchsorted(LAYER_BUCKETS, f.D.to_numpy(), 'left'), r=f.reused.fillna(0).to_numpy())


def _apply(t: pd.DataFrame, p: np.ndarray, dz: pd.Series) -> np.ndarray:
    k = t[['b', 'r']].merge(dz.reset_index(), on=['b', 'r'], how='left').dz.fillna(0).to_numpy()
    return v55.rules_v55(t, p + k * (t.D.to_numpy() - p), t.v55_status.to_numpy())[0]


def layer():
    f = _cells(pd.read_parquet(OUT / 'table.parquet'))
    tr = f[(f.split == 'train') & _layer_scope(f)].reset_index(drop=True)
    y, p = tr.y.to_numpy(), tr.v55.to_numpy()
    q = p.copy()
    for k in sorted(tr.fold.unique()):                                # evidence: cross-fitted by fold
        h = (tr.fold == k).to_numpy()
        q[h] = _apply(tr[h], p[h], _contrast(tr[~h], p[~h]))
    rep = report(tr.assign(month=tr.month), q)
    dz = _contrast(tr, p)                                             # 2026: fit on all of 2025
    rep['dz_full_2025'] = {f'{b}|{int(r)}': round(float(v), 4) for (b, r), v in dz.items()}
    bel = pd.read_parquet(common.DATA / 'base_rank.parquet', columns=['MVT_ID_mvt', 'FLIGHT_mvt'])
    rk = f[(f.split == 'rank') & _layer_scope(f)].merge(bel, on='MVT_ID_mvt', how='left')
    rk = rk[~rk.FLIGHT_mvt.str.startswith('BEL', na=False)].reset_index(drop=True)
    s56 = pd.read_parquet(V56)
    base = rk[['MVT_ID_mvt']].merge(s56, on='MVT_ID_mvt', how='left', validate='one_to_one').TAXITIME_SEC_mvt.to_numpy(float)
    assert np.allclose(base, rk.v55.to_numpy()), 'v56 moved a layer row: the layer would not be separable'
    new = _apply(rk, base, dz)
    out = s56.copy()
    idx = pd.Series(np.arange(len(out)), index=out.MVT_ID_mvt.to_numpy())[rk.MVT_ID_mvt.to_numpy()].to_numpy()
    out.loc[idx, 'TAXITIME_SEC_mvt'] = new
    moved = out.TAXITIME_SEC_mvt.to_numpy(float) != s56.TAXITIME_SEC_mvt.to_numpy(float)
    rep['rows_2026_scope'] = int(len(rk))
    rep['rows_2026_moved'] = int(moved.sum())
    rep['rows_2026_reused'] = int(rk.r.sum())
    rep['sum_sq_move_2026_M'] = round(float(((new - base) ** 2).sum() / 1e6), 2)
    rep['mean_move_2026'] = {f'reused={int(r)}': round(float((new - base)[rk.r.to_numpy() == r].mean()), 1) for r in (0, 1)}
    if V56R.exists():
        raise SystemExit(f'{V56R} exists; refusing to overwrite')
    assert np.isfinite(out.TAXITIME_SEC_mvt).all() and len(out) == len(s56)
    out.to_parquet(V56R, index=False)
    import hashlib
    rep['file'] = V56R.name
    rep['file_sha256'] = hashlib.sha256(V56R.read_bytes()).hexdigest()
    rep['base_file_sha256'] = hashlib.sha256(V56.read_bytes()).hexdigest()
    pd.DataFrame({'MVT_ID_mvt': rk.MVT_ID_mvt, 'reused': rk.r, 'bucket': rk.b, 'D': rk.D, 'v56': base, 'v56r': new}) \
        .to_parquet(OUT / 'layer_rows_2026.parquet', index=False)
    (OUT / 'layer.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != 'dsse_M_by_month'}, indent=1), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['table', 'layer'])
    a = ap.parse_args()
    {'table': table, 'layer': layer}[a.mode]()
