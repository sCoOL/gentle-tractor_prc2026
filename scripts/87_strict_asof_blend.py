"""Strict six-pair as-of source ablation and one fixed blend with v18.

Each AB/CD/EF block trains its base and fallback detector on the other eight
months. The gate is calibrated on one held pair and tested on the other, then
reversed. All target-derived base columns are excluded. The as-of variant also
removes retrospective or future-window sources from both base and detector.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

import common
import detectors
import strict_tail_gate
import tail_gate
import train_eval
import importlib.util

_spec = importlib.util.spec_from_file_location("forward_validation", common.ROOT / "scripts/82_forward_validation.py")
forward = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(forward)

OUT = common.RUNS / "pipeline_reset_20260916"


def _metric(d: pd.DataFrame, col: str) -> dict:
    e2 = (d[col].to_numpy()-d.y.to_numpy())**2
    body = d.y.between(0, 3600).to_numpy()
    return {"rmse": float(np.sqrt(e2.mean())),
            "top100_sse_share": float(np.sort(e2)[-100:].sum()/e2.sum()),
            "body_rmse": float(np.sqrt(e2[body].mean())) if body.any() else None,
            "tail_rmse": float(np.sqrt(e2[~body].mean())) if (~body).any() else None}


def block_result(data: pd.DataFrame, man: dict, left: str, right: str,
                 v18: pd.DataFrame, rounds=660, fallback_rounds=400) -> pd.DataFrame:
    blocked = common.FOLDS[left] + common.FOLDS[right]
    source = data[~data.month.isin(blocked)].copy()
    held = data[data.month.isin(blocked)].copy()
    fallback_cols, fallback_cats = forward.fallback_columns(True)
    held["p_fallback"] = detectors.score_fallback(
        source, held, feature_cols=fallback_cols, cat_cols=fallback_cats,
        rounds=fallback_rounds)
    cols = forward.model_columns(man, True)
    cats = train_eval._cats(man, cols)
    pred, normal = train_eval.predict_model(
        "M3rpa", source, held, cols, cats, {}, rounds,
        strict_tail_gate._heli_const(source), return_normal=True)
    frames = {}
    for fold in (left, right):
        m = held.month.isin(common.FOLDS[fold]).to_numpy()
        raw = pd.DataFrame({"MVT_ID_mvt": held.loc[m, "MVT_ID_mvt"].to_numpy(),
                            "APT": held.loc[m, "APT"].to_numpy(),
                            "month": held.loc[m, "month"].to_numpy(),
                            "y": held.loc[m, "TAXITIME_SEC_mvt"].to_numpy(),
                            "pred": pred[m], "normal_pred": normal[m],
                            "fpl_matched": held.loc[m, "fpl_matched"].to_numpy(),
                            "p_fallback": held.loc[m, "p_fallback"].to_numpy()})
        frames[fold] = tail_gate.prepare_frame(raw, held.loc[m])
    pieces = []
    for calibration, test in ((left, right), (right, left)):
        frame = frames[test]
        weight, _, _ = tail_gate.fit_predict_gate(
            frames[calibration], frame,
            features=strict_tail_gate.STRICT_GATE_FEATURES,
            cats=strict_tail_gate.STRICT_GATE_CATS)
        asof = tail_gate.apply_gate(frame, weight)
        baseline = frame[["MVT_ID_mvt", "y"]].merge(
            v18[["MVT_ID_mvt", "y", "v18_pred"]], on="MVT_ID_mvt",
            how="left", validate="one_to_one", suffixes=("", "_v18"))
        if baseline.v18_pred.isna().any() or not np.allclose(baseline.y, baseline.y_v18):
            raise ValueError(f"strict v18 alignment failed for {test}")
        result = frame[["MVT_ID_mvt", "APT", "month", "y", "fpl_matched"]].copy()
        result["fold"] = test
        result["v18_pred"] = baseline.v18_pred.to_numpy()
        result["asof_pred"] = asof
        result["blend_pred"] = np.maximum(30, (result.v18_pred+result.asof_pred)/2)
        pieces.append(result)
    return pd.concat(pieces, ignore_index=True)


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    man = common.read_manifest(3)
    paths = [OUT / f"strict_asof_{left}{right}.parquet" for left, right in strict_tail_gate.BLOCKS]
    data = None if all(p.exists() for p in paths) else pd.read_parquet(
        common.DATA / "features_v3_train.parquet")
    v18 = pd.read_parquet(OUT / "strict_v18_ledger.parquet",
                          columns=["MVT_ID_mvt", "y", "v18_pred"])
    pieces = []
    for left, right in strict_tail_gate.BLOCKS:
        name = f"strict_asof_{left}{right}.parquet"
        path = OUT / name
        if path.exists():
            part = pd.read_parquet(path)
        else:
            if data is None:
                raise ValueError("training data unexpectedly unavailable")
            part = block_result(data, man, left, right, v18)
            part.to_parquet(path, index=False)
        print(f"{left}{right}: n={len(part)}, v18={_metric(part,'v18_pred')['rmse']:.4f}, "
              f"asof={_metric(part,'asof_pred')['rmse']:.4f}, "
              f"blend={_metric(part,'blend_pred')['rmse']:.4f}", flush=True)
        pieces.append(part)
    d = pd.concat(pieces, ignore_index=True)
    if len(d) != len(v18) or d.MVT_ID_mvt.duplicated().any():
        raise ValueError("strict as-of six-pair coverage failed")
    report = {"protocol": "strict paired-block target-free as-of base and detector; pairwise gate",
              "n": len(d), "n_features": len(forward.model_columns(man, True)),
              "n_removed_vs_v18": len(forward.model_columns(man, False))-len(forward.model_columns(man, True)),
              "candidates": {}}
    d["body"] = np.where(d.y.between(0, 3600), "body", "tail_or_negative")
    d["label_tail"] = np.select([d.y.lt(0), d.y.gt(14400), d.y.gt(3600)],
                                ["negative", "over_4h", "1h_to_4h"], default="body")
    segment_rows = []
    for name, col in (("v18", "v18_pred"), ("asof", "asof_pred"), ("blend_equal", "blend_pred")):
        metric = _metric(d, col)
        metric["folds"] = {fold: _metric(g, col)["rmse"] for fold, g in d.groupby("fold")}
        metric["months"] = {str(month): _metric(g, col)["rmse"] for month, g in d.groupby("month")}
        metric["airports"] = {apt: _metric(g, col)["rmse"] for apt, g in d.groupby("APT")}
        metric["lanes"] = {str(lane): _metric(g, col)["rmse"] for lane, g in d.groupby("fpl_matched")}
        report["candidates"][name] = metric
        total_sse = float(((d[col]-d.y)**2).sum())
        for dimension in ("fold", "APT", "month", "fpl_matched", "body", "label_tail"):
            for segment, g in d.groupby(dimension, observed=True):
                sse = float(((g[col]-g.y)**2).sum())
                segment_rows.append({"candidate": name, "dimension": dimension,
                                     "segment": str(segment), "n": len(g),
                                     "rmse": float(np.sqrt(sse/len(g))),
                                     "sse_share": sse/total_sse})
    base = report["candidates"]["v18"]
    for name in ("asof", "blend_equal"):
        c = report["candidates"][name]
        c["pooled_delta_vs_v18"] = c["rmse"]-base["rmse"]
        c["fold_deltas_vs_v18"] = {f: c["folds"][f]-base["folds"][f] for f in base["folds"]}
        c["passes_existing_rule"] = (c["pooled_delta_vs_v18"] <= -1 and
                                      max(c["fold_deltas_vs_v18"].values()) <= 1)
    (OUT / "strict_asof_blend.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(segment_rows).to_csv(OUT / "strict_asof_segments.csv", index=False)
    d.to_parquet(OUT / "strict_asof_blend.parquet", index=False)
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
