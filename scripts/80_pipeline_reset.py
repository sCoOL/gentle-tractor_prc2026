"""Reproducible strict v18 ledger, source parity audit, and bounded signal probes.

All scored labels remain the untouched movement labels. Historical corrections use
only source rows strictly earlier than the scored departure. Arrival windows close
at arrival in-block time and likewise precede the scored departure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import common
import strict_tail_gate
import tail_gate

OUT = common.RUNS / "pipeline_reset_20260916"
BASE_COLS = ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "BLOCK_TIME_UTC_mvt",
             "SCHED_TIME_UTC_mvt", "FLIGHT_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
             "TAXITIME_SEC_mvt", "atot_minus_aobt3", "atot_minus_sched", "fpl_matched"]


def strict_predictions() -> pd.DataFrame:
    frames, _ = strict_tail_gate.load_saved_frames()
    rows = []
    for left, right in strict_tail_gate.BLOCKS:
        for calibration, test in ((left, right), (right, left)):
            d = frames[test]
            weight, _, _ = tail_gate.fit_predict_gate(
                frames[calibration], d, features=strict_tail_gate.STRICT_GATE_FEATURES,
                cats=strict_tail_gate.STRICT_GATE_CATS)
            rows.append(d[["MVT_ID_mvt", "APT", "month", "y", "pred", "normal_pred",
                           "p_fallback", "fpl_matched", "gate_valid"]].assign(
                               fold=test, v18_pred=tail_gate.apply_gate(d, weight)))
    result = pd.concat(rows, ignore_index=True)
    if result.MVT_ID_mvt.duplicated().any() or len(result) != 2_085_047:
        raise ValueError("strict OOF IDs are not unique and complete")
    rmse = common.rmse(result.v18_pred, result.y)
    if abs(rmse - 274.1594) > 0.02:
        raise ValueError(f"v18 control changed: {rmse}")
    return result


def make_ledger() -> pd.DataFrame:
    p = strict_predictions()
    b = pd.read_parquet(common.DATA / "base_train.parquet", columns=BASE_COLS)
    if b.MVT_ID_mvt.duplicated().any():
        raise ValueError("duplicate base movement IDs")
    d = p.merge(b, on="MVT_ID_mvt", validate="one_to_one", how="left", suffixes=("", "_base"))
    if len(d) != len(p) or d.MVT_TIME_UTC_mvt.isna().any() or not np.allclose(d.y, d.TAXITIME_SEC_mvt):
        raise ValueError("strict OOF / base join or raw labels disagree")
    d["error"] = d.v18_pred - d.y
    d["sse"] = d.error.pow(2)
    d["sse_share"] = d.sse / d.sse.sum()
    d["nm_gap"] = d.y - d.atot_minus_aobt3
    d["nm_mismatch_600"] = d.nm_gap.abs().gt(600) & d.atot_minus_aobt3.notna()
    d["schedule_defaulted"] = (d.BLOCK_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).abs().le(
        pd.Timedelta(seconds=10))
    d["label_tail"] = np.select([d.y.lt(0), d.y.gt(14400), d.y.gt(3600)],
                                ["negative", "over_4h", "1h_to_4h"], default="body")
    # Off-block schedule defaults are distinct from missing NM, and overlap it.
    d["fallback_subtype"] = np.select(
        [d.fpl_matched.eq(0) & d.schedule_defaulted,
         d.fpl_matched.eq(0),
         d.schedule_defaulted,
         d.atot_minus_aobt3.notna()],
        ["nm_missing_schedule_default", "nm_missing_other", "matched_schedule_default",
         "matched_other"], default="matched_nm_missing")
    d["month"] = d.month.astype(int)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "strict_v18_ledger.parquet", index=False)
    return d


def segment(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = d.groupby(cols, dropna=False, observed=True).agg(
        n=("sse", "size"), sse=("sse", "sum"), mae=("error", lambda x: x.abs().mean()),
        mismatch_n=("nm_mismatch_600", "sum"), mean_label=("y", "mean"))
    g["rmse"] = np.sqrt(g.sse / g.n)
    g["sse_share"] = g.sse / d.sse.sum()
    return g.reset_index().sort_values("sse", ascending=False)


def parity_audit() -> pd.DataFrame:
    tr = pd.read_parquet(common.DATA / "features_v3_train.parquet")
    rk = pd.read_parquet(common.DATA / "features_v3_rank.parquet")
    man = common.read_manifest(3)
    used = set(c for group in man["groups"].values() for c in group)
    strict_drop = set(man["groups"].get("G8", [])) | strict_tail_gate.TARGET_ROUTE | strict_tail_gate.TARGET_G9
    rows = []
    for c in sorted(used):
        a, b = tr[c], rk[c]
        na = float(a.isna().mean()); nb = float(b.isna().mean())
        cat = c in man.get("cat_cols", [])
        va = a.dropna().astype(str) if cat else a.dropna()
        vb = b.dropna().astype(str) if cat else b.dropna()
        unique_a = int(va.nunique()); unique_b = int(vb.nunique())
        unseen = float((~vb.isin(set(va.unique()))).mean()) if cat and len(vb) else np.nan
        rows.append(dict(column=c, strict_v18_base_used=c not in strict_drop,
                         is_categorical=cat, train_null=na, rank_null=nb,
                         null_delta=nb-na, train_unique=unique_a, rank_unique=unique_b,
                         train_constant_rank_varies=unique_a <= 1 and unique_b > 1,
                         rank_unseen_category_share=unseen))
    return pd.DataFrame(rows)


def time_safe_history(source: pd.DataFrame, score: pd.DataFrame,
                      key: list[str], prior: float = 5.0) -> pd.DataFrame:
    """Shrunk proxy residual mean from strictly earlier source movements only."""
    src = source[["MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "atot_minus_aobt3", *key]].copy()
    src = src[np.isfinite(src.atot_minus_aobt3) & src.TAXITIME_SEC_mvt.between(0, 3600)]
    src["residual"] = (src.TAXITIME_SEC_mvt - src.atot_minus_aobt3).clip(-1800, 1800)
    src = src.dropna(subset=key).sort_values("MVT_TIME_UTC_mvt")
    q = score[["MVT_TIME_UTC_mvt", *key]].copy().reset_index(drop=True).reset_index(names="_row")
    q = q.dropna(subset=key).sort_values("MVT_TIME_UTC_mvt")
    # Build one cumulative history stream. Source at the exact score time is
    # ordered after the query, making the boundary strictly earlier.
    src["_query"] = False; q["_query"] = True
    stream = pd.concat([src, q], ignore_index=True, sort=False).sort_values(
        ["MVT_TIME_UTC_mvt", "_query"], ascending=[True, False], kind="stable")
    stream["_source"] = (~stream._query).astype("int32")
    stream["_residual_zero"] = stream.residual.fillna(0)
    groups = stream.groupby(key, dropna=False, sort=False)
    stream["_n"] = groups["_source"].cumsum()
    stream["_sum"] = groups["_residual_zero"].cumsum()
    got = stream[stream._query][["_row", "_n", "_sum"]].set_index("_row")
    out = pd.DataFrame(index=np.arange(len(score)))
    out["n"] = got._n.reindex(out.index).fillna(0).to_numpy()
    out["correction"] = (got._sum / (got._n + prior)).reindex(out.index).fillna(0).to_numpy()
    return out


def arrival_trailing(arr: pd.DataFrame, score: pd.DataFrame, group: list[str],
                     window="60min") -> pd.DataFrame:
    """Arrival taxi-in summary known by in-block time, strictly before score time."""
    a = arr[["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", *group]].copy()
    a = a[a.BLOCK_TIME_UTC_mvt.notna() & a.TAXITIME_SEC_mvt.between(0, 3600)]
    a = a.sort_values("BLOCK_TIME_UTC_mvt")
    q = score[["MVT_TIME_UTC_mvt", *group]].reset_index(drop=True).reset_index(names="_row")
    pieces = []
    for name, s in a.groupby(group, sort=False, dropna=False):
        s = s.set_index("BLOCK_TIME_UTC_mvt")
        roll = s.TAXITIME_SEC_mvt.rolling(window, closed="both")
        t = pd.DataFrame({"count": roll.count(), "median": roll.median(),
                          "p90": roll.quantile(.9)}).reset_index()
        for k, v in zip(group, name if isinstance(name, tuple) else (name,)):
            t[k] = v
        pieces.append(t)
    if not pieces:
        return pd.DataFrame({"count": np.zeros(len(score)), "median": np.nan, "p90": np.nan})
    t = pd.concat(pieces, ignore_index=True)
    # At a departure time, only arrivals with in-block strictly before that time.
    result = pd.merge_asof(q.sort_values("MVT_TIME_UTC_mvt"),
                           t.sort_values("BLOCK_TIME_UTC_mvt"),
                           left_on="MVT_TIME_UTC_mvt", right_on="BLOCK_TIME_UTC_mvt",
                           by=group, direction="backward", allow_exact_matches=False,
                           tolerance=pd.Timedelta(window))
    return result.set_index("_row")[["count", "median", "p90"]].reindex(range(len(score)))


def report() -> dict:
    d = make_ledger()
    summary = {"strict_oof_rows": len(d), "strict_v18_rmse": common.rmse(d.v18_pred, d.y),
               "top100_sse_share": float(d.nlargest(100, "sse").sse.sum()/d.sse.sum()),
               "join_unique": True}
    for name, cols in {"airport": ["APT"], "month": ["month"], "fold": ["fold"],
                       "lane": ["fpl_matched"], "fallback": ["fallback_subtype"],
                       "tail": ["label_tail"], "nm_mismatch": ["nm_mismatch_600"],
                       "schedule_default": ["schedule_defaulted"],
                       "airport_lane": ["APT", "fpl_matched"],
                       "airport_month": ["APT", "month"],
                       "airport_source": ["APT", "fallback_subtype"]}.items():
        segment(d, cols).to_csv(OUT / f"segment_{name}.csv", index=False)
    d.nlargest(100, "sse").to_csv(OUT / "top100.csv", index=False)
    parity = parity_audit()
    parity.to_csv(OUT / "train_rank_parity.csv", index=False)
    summary["parity"] = {
        "train_constant_rank_varies": parity.loc[parity.train_constant_rank_varies, "column"].tolist(),
        "largest_null_deltas": parity.reindex(parity.null_delta.abs().sort_values(ascending=False).index)
        .head(12)[["column", "train_null", "rank_null", "strict_v18_base_used"]].to_dict("records")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
