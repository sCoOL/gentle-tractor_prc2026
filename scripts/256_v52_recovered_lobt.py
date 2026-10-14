"""v52 = v51 + rule R, a third label-free constraint from the dataset builder's NM match rule (2026-09-26).

  py -3.14 scripts/256_v52_recovered_lobt.py ledger   # 2025 check + per-row ledger -> runs/v52_recovered_lobt/
  py -3.14 scripts/256_v52_recovered_lobt.py build    # submissions/gentle-tractor_v52.parquet + runs/models/A60_*_v52/

R  NM-unmatched departure whose NM flight was found on the destination's arrival record (scripts/250: the recovered
   LOBT that v51's rule F reads), airline NM match rate at the airport >= OP_MIN, and schedule within F_GAP (3,540 s)
   of that LOBT. A schedule copy (off-block = schedule) would then lie within ~1 h of LOBT and the builder would have
   matched the flight, so the record is NOT a copy and lies more than 1 h from LOBT. When the take-off is more than
   F_GAP + 30 s after LOBT the record is the late one, label <= take-off - LOBT - 3,606 s, and the prediction is capped
   at take-off - LOBT - F_GAP. The other branch (record more than 1 h BEFORE LOBT, i.e. before the schedule) is not
   excluded by the builder rule; `ledger` counts it on the 2025 labels (evidence.json, premise block).
   R is disjoint from F (F needs take-off - LOBT < F_GAP). No fitted parameter; the constants are v51's.

The ledger also carries the copy/date-bug/long detector verdict of every 2025 departure under the v51 2025 proxy
(A51 strict OOF + v51's W/F rules, 2025 RMSE 233.6637) and under v52 (explorer columns det_* / v52_*, row sets `v52-*` and
`detector-*`). Verdicts are evaluation only; `ref` (airport median of normal 2025 labels) never enters a prediction:
  artefact label (copy / date bug / over 20 h / EOBT-LOBT copy / long): share = (pred - ref) / (label - ref);
      label - ref < 900 s -> looks_normal (a copy of an on-time flight), share >= 0.5 caught, >= 0.1 hedged, else missed
  normal label: delay = off-block - schedule; share = (pred - label) / delay when delay >= 900 s
      pred >= 12 h -> false_alarm_datebug; share >= 0.5 -> false_alarm_copy; pred >= 1 h -> false_alarm_long;
      share >= 0.1 -> copy_hedge; else ok
"""
import argparse
import hashlib
import importlib
import json
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd

import common
import final_rank  # final-file change: rank-population size (was a hard-coded 344,841)

v51 = importlib.import_module('251_v51_nm_window')
OUT = common.RUNS / 'v52_recovered_lobt'
V51, V52 = v51.V51, common.SUBS / 'gentle-tractor_v52.parquet'
F_GAP, OP_MIN, BUILDER_WINDOW = v51.F_GAP, v51.OP_MIN, 3606.0
ART = ['date_bug', 'over_20h', 'long_other', 'eobt_lobt_copy', 'sched_copy']


def frame(con) -> pd.DataFrame:
    """251's frame (both years) plus schedule, recorded off-block and EOBT."""
    f = v51.frame(con)
    x = con.execute("""SELECT MVT_ID_mvt, epoch(SCHED_TIME_UTC_mvt) AS t_sched, epoch(BLOCK_TIME_UTC_mvt) AS t_blk,
                              epoch(EOBT_1_flt) AS eobt FROM m WHERE PHASE_mvt = 'DEP'""").df()
    return f.merge(x, on='MVT_ID_mvt', how='left', validate='one_to_one')


def status(f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Where each departure sits relative to rule R, and R's cap (NaN where R does not apply)."""
    L = f.t.to_numpy() - f.bf_lobt.to_numpy()
    rec = (~f.nm.to_numpy()) & np.isfinite(L)
    op = f.op_rate.fillna(0).to_numpy() >= OP_MIN
    near = np.abs(f.t_sched.to_numpy() - f.bf_lobt.to_numpy()) < F_GAP
    late = L > F_GAP + 30.0
    st = np.select([f.nm.to_numpy(), ~rec, ~op, ~late, ~near],
                   ['matched', 'no_recovered_lobt', 'airline_rarely_nm', 'F_range', 'sched_far'], 'R')
    cap = np.where(st == 'R', L - F_GAP, np.nan)
    return st, cap


def apply_r(p: np.ndarray, cap: np.ndarray) -> np.ndarray:
    out = p.copy()
    k = np.isfinite(cap) & (out > cap)
    out[k] = cap[k]
    return out


def label_class(f: pd.DataFrame) -> np.ndarray:
    """explorer/db.py LABEL_CLASS_SQL (scripts/91 rules, 10 s tolerance)."""
    y = f.y.to_numpy()
    blk = f.t_blk.to_numpy()
    return np.select([np.abs(blk - f.t_sched.to_numpy()) <= 10, (y >= 72000) & (y % 86400 < 5400), y >= 72000,
                      (y > 3600) & ((np.abs(blk - f.eobt.to_numpy()) <= 10) | (np.abs(blk - f.lobt.to_numpy()) <= 10)),
                      y > 3600], ['sched_copy', 'date_bug', 'over_20h', 'eobt_lobt_copy', 'long_other'], 'normal')


def verdict(p, y, lc, ref, delay) -> tuple[np.ndarray, np.ndarray]:
    art = lc != 'normal'
    gap = y - ref
    with np.errstate(divide='ignore', invalid='ignore'):
        sa = (p - ref) / gap
        sn = np.where(delay >= 900, (p - y) / np.where(delay >= 900, delay, 1.0), np.nan)
    va = np.select([gap < 900, sa >= .5, sa >= .1], ['looks_normal', 'caught', 'hedged'], 'missed')
    vn = np.select([p >= 43200, sn >= .5, p >= 3600, sn >= .1],
                   ['false_alarm_datebug', 'false_alarm_copy', 'false_alarm_long', 'copy_hedge'], 'ok')
    return np.where(art, va, vn), np.where(art, sa, sn)


def rmse(e2) -> float:
    return float(np.sqrt(np.mean(e2)))


def counts(v, lc) -> dict:
    out = {c: {k: int(((lc == c) & (v == k)).sum()) for k in ('caught', 'hedged', 'missed', 'looks_normal')} for c in ART}
    out['normal'] = {k: int(((lc == 'normal') & (v == k)).sum())
                     for k in ('false_alarm_datebug', 'false_alarm_copy', 'false_alarm_long', 'copy_hedge')}
    return out


def ledger():
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    f = frame(con)
    st, cap = status(f)
    f['v52_status'], f['v52_cap_s'] = st, cap
    # 2025: the v51 proxy (A51 strict OOF + W/F), then R
    tr = f[f.split == 'train'].merge(pd.read_parquet(v51.OOF, columns=['MVT_ID_mvt', 'fold', 'candidate']),
                                     on='MVT_ID_mvt', how='inner', validate='one_to_one')
    assert len(tr) == 2085047
    p51, _ = v51.apply_rules(tr, tr.candidate.to_numpy())
    p52 = apply_r(p51, tr.v52_cap_s.to_numpy())
    y = tr.y.to_numpy()
    lc = label_class(tr)
    ref = pd.Series(y[lc == 'normal']).groupby(tr.apt.to_numpy()[lc == 'normal']).median()
    r = tr.apt.map(ref).to_numpy()
    delay = tr.t_blk.to_numpy() - tr.t_sched.to_numpy()
    v1, s1 = verdict(p51, y, lc, r, delay)
    v2, s2 = verdict(p52, y, lc, r, delay)
    e1, e2 = (p51 - y) ** 2, (p52 - y) ** 2
    moved = p52 != p51
    L = tr.t.to_numpy() - tr.bf_lobt.to_numpy()
    isr = tr.v52_status.to_numpy() == 'R'
    early = isr & (tr.t_blk.to_numpy() < tr.bf_lobt.to_numpy() - BUILDER_WINDOW)
    # diagnostic only (not a rule): recovered-LOBT rows outside F and R whose prediction sits inside the forbidden band
    band = np.isin(tr.v52_status.to_numpy(), ['sched_far']) & (np.abs(p51 - L) < BUILDER_WINDOW)
    rep = {'rows': int(len(tr)), 'rmse_v51_proxy': rmse(e1), 'rmse_v52_proxy': rmse(e2), 'delta': rmse(e2) - rmse(e1),
           'status_2025': pd.Series(st[f.split == 'train']).value_counts().to_dict(),
           'premise': {'R_rows': int(isr.sum()), 'R_rows_schedule_copies': int((isr & (lc == 'sched_copy')).sum()),
                       'R_rows_record_more_than_1h_before_lobt': int(early.sum()),
                       'R_labels_above_cap': int((isr & (y > tr.v52_cap_s.to_numpy())).sum()),
                       'R_label_classes': pd.Series(lc[isr]).value_counts().to_dict()},
           'moved': {'rows': int(moved.sum()), 'rows_worse': int((e2 > e1 + 1e-9).sum()), 'sse_before': float(e1[moved].sum()),
                     'sse_after': float(e2[moved].sum()), 'by_airport': pd.Series(tr.apt[moved]).value_counts().to_dict(),
                     'by_label_class': pd.Series(lc[moved]).value_counts().to_dict(),
                     'by_v51_verdict': pd.Series(v1[moved]).value_counts().to_dict()},
           'verdicts_v51_proxy': counts(v1, lc), 'verdicts_v52_proxy': counts(v2, lc),
           'band_diagnostic_sched_far': {'rows_inside_band': int(band.sum()),
                                         'labels_below_band': int((band & (y <= L - BUILDER_WINDOW)).sum()),
                                         'labels_above_band': int((band & (y >= L + BUILDER_WINDOW)).sum()),
                                         'sse_share': float(e1[band].sum() / e1.sum())},
           'by_fold': {}, 'by_month': {}, 'by_airport': {}}
    for col, key in (('fold', 'by_fold'), ('month', 'by_month'), ('apt', 'by_airport')):
        for g, idx in tr.groupby(col).indices.items():
            rep[key][str(g)] = {'rows': int(len(idx)), 'moved': int(moved[idx].sum()), 'delta': rmse(e2[idx]) - rmse(e1[idx])}
    # 2026: v51 file, then R
    rk = f[f.split == 'rank']
    s51 = pd.read_parquet(V51)
    x = s51[['MVT_ID_mvt']].merge(rk, on='MVT_ID_mvt', how='left', validate='one_to_one')
    q51 = s51.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    q52 = apply_r(q51, x.v52_cap_s.to_numpy())
    mv = q52 != q51
    rep['rank_2026'] = {'rows': int(len(x)), 'status': x.v52_status.value_counts().to_dict(), 'moved': int(mv.sum()),
                        'moved_by_airport': x.apt[mv].value_counts().to_dict(),
                        'mean_cut_s': float((q51 - q52)[mv].mean()) if mv.any() else 0.0,
                        'min_sse_reduction_vs_v51_if_rule_holds': float(((q51 - q52) ** 2).sum())}
    led = pd.concat([
        pd.DataFrame({'MVT_ID_mvt': tr.MVT_ID_mvt, 'split': 'train', 'v52_status': tr.v52_status, 'v52_cap_s': tr.v52_cap_s,
                      'det_pred_s': p51, 'det_err_s': p51 - y, 'v52_pred_s': p52, 'v52_moved': moved, 'det_label': lc, 'det_ref_s': r,
                      'det_verdict': v1, 'det_share': s1, 'det_verdict_v52': v2}),
        pd.DataFrame({'MVT_ID_mvt': x.MVT_ID_mvt, 'split': 'rank', 'v52_status': x.v52_status, 'v52_cap_s': x.v52_cap_s,
                      'det_pred_s': q51, 'v52_pred_s': q52, 'v52_moved': mv})], ignore_index=True)
    led.to_parquet(OUT / 'rows.parquet', index=False)
    (OUT / 'evidence.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k not in ('by_month', 'by_airport')}, indent=2))
    return rep


def sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build():
    if V52.exists():
        raise SystemExit(f'{V52} exists; refusing to overwrite')
    con = duckdb.connect()
    f = frame(con)
    f['v52_status'], f['v52_cap_s'] = status(f)
    rk = f[f.split == 'rank']
    s51 = pd.read_parquet(V51)
    assert len(s51) == final_rank.n_rank() and s51.MVT_ID_mvt.is_unique   # final-file change: was == 344841
    x = s51[['MVT_ID_mvt']].merge(rk, on='MVT_ID_mvt', how='left', validate='one_to_one')
    assert x.split.notna().all(), 'a v51 row is missing from the ranking departures'
    p = s51.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    q = apply_r(p, x.v52_cap_s.to_numpy())
    sub = s51.copy()
    sub['TAXITIME_SEC_mvt'] = np.round(q).astype(s51.TAXITIME_SEC_mvt.dtype) if np.issubdtype(s51.TAXITIME_SEC_mvt.dtype, np.integer) else q
    assert np.isfinite(sub.TAXITIME_SEC_mvt.astype(float)).all() and (sub.MVT_ID_mvt.to_numpy() == s51.MVT_ID_mvt.to_numpy()).all()
    assert (sub.TAXITIME_SEC_mvt.astype(float) >= 30.0).all()
    sub.to_parquet(V52, index=False)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pkg = common.RUNS / 'models' / f'A60_{stamp}_v52'
    pkg.mkdir(parents=True)
    moved = pd.DataFrame({'MVT_ID_mvt': x.MVT_ID_mvt, 'APT': x.apt, 'rule': np.where(q != p, 'R', ''), 'v51': p,
                          'v52': sub.TAXITIME_SEC_mvt.astype(float), 'cap': x.v52_cap_s})
    moved[moved.rule != ''].to_parquet(pkg / 'moved_rows.parquet', index=False)
    d = moved.v52 - moved.v51
    info = {'approach': 'A60', 'version': 52, 'base': V51.name, 'base_sha256': sha(V51), 'file_sha256': sha(V52),
            'rules': {'F_GAP': F_GAP, 'OP_MIN': OP_MIN, 'cap': 'take-off - recovered LOBT - F_GAP',
                      'applies': 'no NM match, recovered LOBT, airline NM rate >= OP_MIN, |schedule - LOBT| < F_GAP, '
                                 'take-off - LOBT > F_GAP + 30 s'},
            'backfill_sha256': sha(v51.BACKFILL), 'rows': int(len(sub)), 'moved': int((moved.rule != '').sum()),
            'moved_by_airport': moved[moved.rule != ''].APT.value_counts().to_dict(),
            'rms_change_all_rows': float(np.sqrt(np.mean(d ** 2))),
            'min_sse_reduction_vs_v51_if_rule_holds': float((d ** 2).sum())}   # no official-score forecast
    (pkg / 'build.json').write_text(json.dumps(info, indent=2))
    (pkg / '256_v52_recovered_lobt.py').write_bytes(open(__file__, 'rb').read())
    ev = OUT / 'evidence.json'
    if ev.exists():
        (pkg / 'evidence.json').write_bytes(ev.read_bytes())
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['ledger', 'build'])
    a = ap.parse_args()
    ledger() if a.mode == 'ledger' else build()
