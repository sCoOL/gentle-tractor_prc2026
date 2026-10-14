"""Chronological 2025 backtest with a source-availability stress variant.

The gate is calibrated on May-June predictions from a Jan-Apr base and tested
on July-December predictions from a Jan-June base. No validation target enters
its own base or gate fit. Static target-derived G4/G8/G9 predictors are removed.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

import common
import detectors
import strict_tail_gate
import tail_gate
import train_eval

OUT = common.RUNS / "pipeline_reset_20260916"
ASOF_DROP = {
    # Windows reaching after the movement timestamp.
    "dep_next15", "arr_pm15", "dep_same_rwy_pm10", "rwy_share_pm30",
    "n_rwys_active_pm30", "mixed_mode_rwy",
    # Retrospective same-day and same-month aggregates. In July 2026 the
    # monthly source is explicitly a 2025 fallback rather than a live month.
    "atfm_arr_delay_min", "atfm_arr_flights", "atfm_arr_delayed_flights",
    "atfm_delay_per_flight", "traffic_dep_day", "traffic_arr_day",
    "addl_taxi_out_min_month", "asma_addl_min_month", "ec_monthly_is_prior",
    # Configuration windows may incorporate later movements in the day.
    "rwy_config_id", "min_since_config_change", "config_dep_share",
    "n_dep_rwys_config", "n_arr_rwys_config",
}


def fallback_columns(asof: bool) -> tuple[list[str], list[str]]:
    cols = [c for c in detectors.FALLBACK_FEATS if not asof or c not in ASOF_DROP]
    cats = [c for c in detectors.FALLBACK_CATS if c in cols]
    return cols, cats


def model_columns(manifest: dict, asof: bool) -> list[str]:
    target_drop = (set(manifest["groups"].get("G8", [])) |
                   strict_tail_gate.TARGET_ROUTE | strict_tail_gate.TARGET_G9)
    return [c for c in train_eval.feature_columns(manifest, list(manifest["groups"]))
            if c not in target_drop and (not asof or c not in ASOF_DROP)]


def _frame(data: pd.DataFrame, pred: np.ndarray, normal: np.ndarray) -> pd.DataFrame:
    raw = data[["MVT_ID_mvt", "APT", "month", "TAXITIME_SEC_mvt",
                "fpl_matched", "p_fallback"]].copy().rename(columns={"TAXITIME_SEC_mvt": "y"})
    raw["pred"] = pred
    raw["normal_pred"] = normal
    return tail_gate.prepare_frame(raw, data)


def _fit_score(train: pd.DataFrame, score: pd.DataFrame, cols: list[str], cats: list[str],
               rounds=660, fallback_rounds=400, asof=False):
    a = train.copy(); b = score.copy()
    fallback_cols, fallback_cats = fallback_columns(asof)
    b["p_fallback"] = detectors.score_fallback(
        a, b, feature_cols=fallback_cols, cat_cols=fallback_cats, rounds=fallback_rounds)
    pred, normal = train_eval.predict_model(
        "M3rpa", a, b, cols, cats, {}, rounds, strict_tail_gate._heli_const(a),
        return_normal=True)
    return _frame(b, pred, normal)


def run_variant(data: pd.DataFrame, manifest: dict, asof: bool,
                rounds: int = 660, fallback_rounds: int = 400) -> dict:
    cols = model_columns(manifest, asof)
    cats = train_eval._cats(manifest, cols)
    early = data[data.month.isin([1, 2, 3, 4])].copy()
    calibrate = data[data.month.isin([5, 6])].copy()
    source = data[data.month.isin([1, 2, 3, 4, 5, 6])].copy()
    held = data[data.month.isin([7, 8, 9, 10, 11, 12])].copy()
    cal_frame = _fit_score(early, calibrate, cols, cats, rounds, fallback_rounds, asof=asof)
    test_frame = _fit_score(source, held, cols, cats, rounds, fallback_rounds, asof=asof)
    weight, _, _ = tail_gate.fit_predict_gate(
        cal_frame, test_frame, features=strict_tail_gate.STRICT_GATE_FEATURES,
        cats=strict_tail_gate.STRICT_GATE_CATS)
    gated = tail_gate.apply_gate(test_frame, weight)
    name = "forward_asof" if asof else "forward_retrospective"
    result = {"variant": name, "n_source": len(source), "n_calibration": len(calibrate),
              "n_held": len(held), "n_features": len(cols), "dropped_asof_columns":
              sorted(ASOF_DROP.intersection(manifest["columns"])) if asof else [],
              "rounds": rounds, "fallback_rounds": fallback_rounds,
              "base_rmse": common.rmse(test_frame.pred, test_frame.y),
              "gated_rmse": common.rmse(gated, test_frame.y),
              "months": {str(m): common.rmse(gated[test_frame.month.eq(m)],
                                               test_frame.loc[test_frame.month.eq(m), "y"])
                         for m in range(7, 13)}}
    pd.DataFrame({"MVT_ID_mvt": test_frame.MVT_ID_mvt, "y": test_frame.y,
                  "pred": gated, "base_pred": test_frame.pred,
                  "month": test_frame.month, "APT": test_frame.APT,
                  "fpl_matched": test_frame.fpl_matched}).to_parquet(
                      OUT / f"{name}.parquet", index=False)
    return result


def main(variants=(False, True), rounds=660, fallback_rounds=400) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    data, _, manifest = common.load_features(3)
    path = OUT / "forward_validation.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "protocol": "Jan-Apr base -> May-Jun gate calibration; Jan-Jun base -> Jul-Dec test",
        "strict_v18_reference": 274.1594,
        "caveat": "2025 chronological test cannot reproduce Jan/Jul 2026 source availability; asof variant drops retrospective aggregates and lookahead windows."}
    for asof in variants:
        result = run_variant(data, manifest, asof, rounds, fallback_rounds)
        report[result["variant"]] = result
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["both", "retrospective", "asof"], default="both")
    parser.add_argument("--rounds", type=int, default=660)
    parser.add_argument("--fallback-rounds", type=int, default=400)
    args = parser.parse_args()
    variants = {"both": (False, True), "retrospective": (False,), "asof": (True,)}[args.variant]
    print(json.dumps(main(variants, args.rounds, args.fallback_rounds), indent=2))
