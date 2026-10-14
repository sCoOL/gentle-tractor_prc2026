"""v51 = v49 + two label-free constraints from the dataset builder's NM matching rule (2026-09-25).

  py -3.14 scripts/251_v51_nm_window.py evidence   # 2025 check on the A51 strict OOF -> runs/v51_nm_window/evidence.json
  py -3.14 scripts/251_v51_nm_window.py build      # submissions/gentle-tractor_v51.parquet + runs/models/A59_*/

The builder gave a departure its NM (`_flt`) columns only when |BLOCK_TIME - LOBT_flt| <= ~3,606 s (all 2,062,577
matched 2025 departures; the 5-min counts stop at 60 min). Both rules are fixed in advance, have no fitted parameter
and read no departure block time or label:

  W  NM-matched departure (LOBT present): the label lies in take-off - LOBT +- 3,606 s, so the prediction is clipped
     to take-off - LOBT +- W_CLIP (3,660 s: one minute wider than any 2025 row). A projection onto an interval that
     holds the label can only lower each row's error.
  F  NM-unmatched departure whose NM flight was found on the destination's arrival record (scripts/250_nm_backfill.py,
     tiers >= 99.89 % right), with take-off - LOBT < F_GAP (3,540 s): the recorded off-block cannot be more than 1 h
     after LOBT (the flight had already left), so it is more than 1 h before it and the label exceeds take-off - LOBT +
     3,600 s. The prediction is raised to at least take-off - LOBT + F_GAP. Only where the airline normally matches NM
     at this airport (>= OP_MIN of its departures over both years, label-free): operators that never match (LEBL ICC,
     NCG, PCT, FCK in 2025) are unmatched for identity reasons, where the floor does not hold.

Evidence runs both rules on the A51 strict OOF (v49 has no 2025 prediction; the rules do not depend on the model).
"""
import argparse
import hashlib
import json
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd

import common
import final_rank  # final-file change: rank-population size (was a hard-coded 344,841)

OUT = common.RUNS / 'v51_nm_window'
OOF = common.RUNS / 'a51_final' / 'chosen_ensemble_oof.parquet'
V49 = common.SUBS / 'gentle-tractor_v49.parquet'
V51 = common.SUBS / 'gentle-tractor_v51.parquet'
BACKFILL = common.RUNS / 'nm_backfill' / 'backfill.parquet'
W_CLIP, F_GAP, OP_MIN = 3660.0, 3540.0, 0.5


def frame(con) -> pd.DataFrame:
    """One row per departure (both years) with the inputs of W and F; label and fold on 2025 rows."""
    con.execute("SET TimeZone = 'UTC'")
    con.execute("""CREATE TABLE m AS SELECT *, 'train' AS split FROM read_parquet(?, union_by_name=true)
                   UNION ALL BY NAME SELECT *, 'rank' AS split FROM read_parquet(?)""",
                [[str(p) for p in sorted(common.DATA.glob('training_2025-*.parquet'))], str(common.DATA / 'ranking.parquet')])
    return con.execute(f"""
        WITH dep AS (SELECT *, regexp_extract(FLIGHT_mvt, '^([A-Z]{{3}}|[A-Z0-9]{{2}})', 1) AS al FROM m WHERE PHASE_mvt = 'DEP'),
        op AS (SELECT ADEP_mvt, al, avg((FLIGHT_ID_mvt IS NOT NULL)::DOUBLE) AS op_rate FROM dep GROUP BY ALL)
        SELECT dep.MVT_ID_mvt, dep.split, dep.ADEP_mvt AS apt, month(dep.MVT_TIME_UTC_mvt) AS month,
               epoch(dep.MVT_TIME_UTC_mvt) AS t, epoch(dep.LOBT_flt) AS lobt, dep.FLIGHT_ID_mvt IS NOT NULL AS nm,
               epoch(bf.bf_lobt) AS bf_lobt, op.op_rate, CAST(dep.TAXITIME_SEC_mvt AS DOUBLE) AS y
        FROM dep LEFT JOIN op USING (ADEP_mvt, al) LEFT JOIN read_parquet('{BACKFILL.as_posix()}') bf USING (MVT_ID_mvt)""").df()


def apply_rules(f: pd.DataFrame, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Constrained predictions and the rule that moved each row ('' / 'W_low' / 'W_high' / 'F')."""
    out, why = p.copy(), np.full(len(p), '', dtype=object)
    L = f.t.to_numpy() - f.lobt.to_numpy()
    w = np.isfinite(L)
    lo, hi = L - W_CLIP, L + W_CLIP
    low, high = w & (out < lo), w & (out > hi)
    out[low], out[high] = lo[low], hi[high]
    why[low], why[high] = 'W_low', 'W_high'
    Lb = f.t.to_numpy() - f.bf_lobt.to_numpy()
    fl = (~f.nm.to_numpy()) & np.isfinite(Lb) & (Lb < F_GAP) & (f.op_rate.fillna(0).to_numpy() >= OP_MIN)
    floor = Lb + F_GAP
    up = fl & (out < floor)
    out[up] = floor[up]
    why[up] = 'F'
    return out, why


def rmse(e2) -> float:
    return float(np.sqrt(np.mean(e2)))


def evidence():
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    f = frame(con)
    tr = f[f.split == 'train'].merge(pd.read_parquet(OOF, columns=['MVT_ID_mvt', 'fold', 'candidate']), on='MVT_ID_mvt', how='inner')
    p = tr.candidate.to_numpy()
    q, why = apply_rules(tr, p)
    y = tr.y.to_numpy()
    e0, e1 = (p - y) ** 2, (q - y) ** 2
    L = tr.t.to_numpy() - tr.lobt.to_numpy()
    Lb = tr.t.to_numpy() - tr.bf_lobt.to_numpy()
    applies_f = (~tr.nm.to_numpy()) & np.isfinite(Lb) & (Lb < F_GAP) & (tr.op_rate.fillna(0).to_numpy() >= OP_MIN)
    rep = {'rows': int(len(tr)), 'rmse_a51': rmse(e0), 'rmse_constrained': rmse(e1), 'delta': rmse(e1) - rmse(e0),
           'rule_premises_on_2025_labels': {
               'W_labels_outside_window': int((np.isfinite(L) & ((y < L - W_CLIP) | (y > L + W_CLIP))).sum()),
               'W_rows_with_lobt': int(np.isfinite(L).sum()),
               'F_rows_where_floor_applies': int(applies_f.sum()),
               'F_labels_below_floor': int((applies_f & (y < Lb + F_GAP)).sum())},
           'moved': {}, 'by_fold': {}, 'by_month': {}, 'by_airport': {}}
    for r in ('W_low', 'W_high', 'F'):
        k = why == r
        rep['moved'][r] = {'rows': int(k.sum()), 'sse_before': float(e0[k].sum()), 'sse_after': float(e1[k].sum()),
                           'rows_worse': int((e1[k] > e0[k] + 1e-9).sum())}
    for col, key in (('fold', 'by_fold'), ('month', 'by_month'), ('apt', 'by_airport')):
        for g, idx in tr.groupby(col).indices.items():
            rep[key][str(g)] = {'rows': int(len(idx)), 'moved': int((why[idx] != '').sum()),
                                'delta': rmse(e1[idx]) - rmse(e0[idx])}
    (OUT / 'evidence.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k not in ('by_month', 'by_airport')}, indent=2))
    return rep


def sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build():
    con = duckdb.connect()
    f = frame(con)
    rk = f[f.split == 'rank']
    v49 = pd.read_parquet(V49)
    assert len(v49) == final_rank.n_rank() and v49.MVT_ID_mvt.is_unique   # final-file change: was == 344841
    x = v49[['MVT_ID_mvt']].merge(rk, on='MVT_ID_mvt', how='left', validate='one_to_one')
    assert x.split.notna().all(), 'a v49 row is missing from the ranking departures'
    p = v49.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    q, why = apply_rules(x, p)
    sub = v49.copy()
    sub['TAXITIME_SEC_mvt'] = np.round(q).astype(v49.TAXITIME_SEC_mvt.dtype) if np.issubdtype(v49.TAXITIME_SEC_mvt.dtype, np.integer) else q
    assert np.isfinite(sub.TAXITIME_SEC_mvt.astype(float)).all() and (sub.MVT_ID_mvt.to_numpy() == v49.MVT_ID_mvt.to_numpy()).all()
    sub.to_parquet(V51, index=False)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pkg = common.RUNS / 'models' / f'A59_{stamp}_v51'
    pkg.mkdir(parents=True)
    moved = pd.DataFrame({'MVT_ID_mvt': x.MVT_ID_mvt, 'APT': x.apt, 'rule': why, 'v49': p, 'v51': sub.TAXITIME_SEC_mvt.astype(float)})
    moved[moved.rule != ''].to_parquet(pkg / 'moved_rows.parquet', index=False)
    d = moved.v51 - moved.v49
    info = {'approach': 'A59', 'version': 51, 'base': V49.name, 'base_sha256': sha(V49), 'file_sha256': sha(V51),
            'rules': {'W_CLIP': W_CLIP, 'F_GAP': F_GAP, 'OP_MIN': OP_MIN}, 'backfill_sha256': sha(BACKFILL),
            'rows': int(len(sub)), 'moved': moved[moved.rule != ''].rule.value_counts().to_dict(),
            'rms_change_all_rows': float(np.sqrt(np.mean(d ** 2))),
            'min_sse_reduction_vs_v49_if_rules_hold': float((d ** 2).sum())}   # no official-score forecast
    (pkg / 'build.json').write_text(json.dumps(info, indent=2))
    (pkg / '251_v51_nm_window.py').write_bytes(open(__file__, 'rb').read())
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['evidence', 'build'])
    a = ap.parse_args()
    evidence() if a.mode == 'evidence' else build()
