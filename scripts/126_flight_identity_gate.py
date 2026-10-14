"""Fully nested LIRF flight-identity correction for the final A48 copy gate.

Why this exists
---------------
The current copy detector never sees exact flight identity.  A first diagnostic
found stable flight/route signal, but a conventional cross-block stack can let a
held block influence base predictions used to train the meta-model.  This probe
uses the stricter blocked-pair protocol:

* target fold A is scored by a gate trained on paired fold B (and vice versa);
* the other eight months supply smoothed historical flight-route copy rates;
* the blend with A48 is selected by an inner two-month cross-fit inside the
  calibration fold; and
* target-fold labels are read only after its predictions are fixed.

The script is intentionally a probe first.  ``deploy`` is enabled only when the
saved probe report satisfies the promotion checks in this file.

Examples
--------
    py -3.14 scripts/126_flight_identity_gate.py probe
    py -3.14 scripts/126_flight_identity_gate.py deploy
    py -3.14 scripts/126_flight_identity_gate.py build-v43
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import common


ROOT = common.WORK   # data / runs / submissions root (PRC_WORK)
OOF_PATH = ROOT / "runs" / "oof" / "bundle_A48u_strict.parquet"
ANATOMY_PATH = ROOT / "runs" / "a48" / "matched_anatomy_2025.parquet"
BASE_TRAIN = ROOT / "data" / "base_train.parquet"
BASE_RANK = ROOT / "data" / "base_rank.parquet"
RANK_DETAIL = ROOT / "runs" / "models" / "A48u_20260922_132003" / "rank_detail.parquet"
REPORT_PATH = ROOT / "runs" / "flight_identity_gate_A49_probe.json"
OOF_OUT = ROOT / "runs" / "oof" / "flight_identity_gate_A49_strict.parquet"
RANK_OUT = ROOT / "runs" / "flight_identity_gate_A49_rank.parquet"
BASE_SUBMISSION = ROOT / "submissions" / "gentle-tractor_v42.parquet"
V43_SUBMISSION = ROOT / "submissions" / "gentle-tractor_v43.parquet"

PAIR = {"A": "B", "B": "A", "C": "D", "D": "C", "E": "F", "F": "E"}
BLOCK = {"A": "AB", "B": "AB", "C": "CD", "D": "CD", "E": "EF", "F": "EF"}
ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)

CONTROL_FEATURES = [
    "w", "D", "normal", "gap", "abs_gap", "delay_h", "a3taxi", "month", "dow", "hour_utc",
    "minute_of_day_utc", "sched_minute", "aobt3_minus_sched", "lobt_minus_sched", "aobt3_minus_lobt",
    "airline", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "STAND_mvt", "RUNWAY_mvt", "flight_is_callsign",
]
IDENTITY_FEATURES = CONTROL_FEATURES + [
    "FLIGHT_mvt", "flight_route", "flight_dow", "flight_route_rate_outer", "flight_route_count_outer",
]
CATEGORICAL = [
    "airline", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "STAND_mvt", "RUNWAY_mvt", "FLIGHT_mvt",
    "flight_route", "flight_dow",
]


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(y, float) - np.asarray(p, float)))))


def _raw(path: Path) -> pd.DataFrame:
    cols = [
        "MVT_ID_mvt", "FLIGHT_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "STAND_mvt", "RUNWAY_mvt",
        "SCHED_TIME_UTC_mvt", "airline", "flight_is_callsign", "dow", "month", "hour_utc",
        "minute_of_day_utc", "aobt3_minus_sched", "lobt_minus_sched", "aobt3_minus_lobt", "is_heli",
        "atot_minus_aobt3",
    ]
    return pd.read_parquet(path, columns=cols)


def load_train() -> pd.DataFrame:
    m = pd.read_parquet(ANATOMY_PATH)
    d = m.merge(_raw(BASE_TRAIN), on="MVT_ID_mvt", how="left", validate="one_to_one", suffixes=("", "_raw"))
    d = d[(d.APT == "LIRF") & (d.fpl_matched == 1) & (d.is_heli == 0)].copy()
    d["block"] = d.fold.map(BLOCK)
    d["is_copy"] = (d.cls == "copy").astype(int)
    recover = d.w < 0.999999
    d["recover"] = recover
    d["normal"] = np.where(recover, (d.pred - d.w * d.D) / (1.0 - d.w), d.a3taxi)
    d["gap"] = d.D - d.normal
    d["w_star"] = np.clip((d.y - d.normal) / d.gap.replace(0, np.nan), 0.0, 1.0).fillna(d.w)
    d["loss_weight"] = np.minimum(np.square(d.gap), 86400.0 ** 2)
    return engineer(d.reset_index(drop=True))


def engineer(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["abs_gap"] = d.gap.abs()
    d["delay_h"] = d.D / 3600.0
    d["sched_minute"] = d.SCHED_TIME_UTC_mvt.dt.hour * 60 + d.SCHED_TIME_UTC_mvt.dt.minute
    d["flight_route"] = d.FLIGHT_mvt.astype("string") + ">" + d.ADES_mvt.astype("string")
    d["flight_dow"] = d.FLIGHT_mvt.astype("string") + ":" + d.dow.astype("string")
    return d


def add_outer_rates(frame: pd.DataFrame, history: pd.DataFrame, strength: float = 5.0) -> pd.DataFrame:
    """Smoothed exact flight+route rate from disjoint history."""
    key = ["airline", "FLIGHT_mvt", "ADES_mvt"]
    prior_key = ["airline", "ADES_mvt"]
    global_rate = float(history.is_copy.mean())
    prior = history.groupby(prior_key, dropna=False).is_copy.agg(["sum", "size"]).reset_index()
    prior["prior"] = (prior["sum"] + 20.0 * global_rate) / (prior["size"] + 20.0)
    stats = history.groupby(key, dropna=False).is_copy.agg(["sum", "size"]).reset_index()
    stats = stats.merge(prior[prior_key + ["prior"]], on=prior_key, how="left", validate="many_to_one")
    stats["rate"] = (stats["sum"] + strength * stats["prior"]) / (stats["size"] + strength)
    out = frame.merge(stats[key + ["rate", "size"]], on=key, how="left", validate="many_to_one")
    out = out.merge(prior[prior_key + ["prior"]], on=prior_key, how="left", validate="many_to_one")
    out["flight_route_rate_outer"] = out.rate.fillna(out.prior).fillna(global_rate)
    out["flight_route_count_outer"] = out["size"].fillna(0.0)
    return out.drop(columns=["rate", "size", "prior"])


def add_crossfit_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Rate feature for full-fit training without letting a row label encode itself."""
    parts = []
    for held in sorted(frame.fold.unique()):
        parts.append(add_outer_rates(frame[frame.fold == held].copy(), frame[frame.fold != held]))
    return pd.concat(parts, ignore_index=True).sort_values("MVT_ID_mvt").reset_index(drop=True)


def _prep(train: pd.DataFrame, score: pd.DataFrame, features: list[str]):
    cats = [c for c in CATEGORICAL if c in features]
    x_train, x_score = train[features].copy(), score[features].copy()
    for c in cats:
        train_values = x_train[c].astype("string").fillna("__NA__")
        categories = pd.Index(train_values.unique()).append(pd.Index(["__NA__"])).unique()
        x_train[c] = pd.Categorical(train_values, categories=categories)
        score_values = x_score[c].astype("string").fillna("__NA__")
        score_values = score_values.where(score_values.isin(categories), "__NA__")
        x_score[c] = pd.Categorical(score_values, categories=categories)
    return x_train, x_score, cats


def fit_gate(train: pd.DataFrame, score: pd.DataFrame, features: list[str]) -> np.ndarray:
    usable = train.recover & np.isfinite(train.loss_weight) & (train.loss_weight > 0)
    x_train, x_score, cats = _prep(train.loc[usable], score, features)
    params = dict(
        objective="regression",
        learning_rate=0.03,
        num_leaves=15,
        min_data_in_leaf=150,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l2=150.0,
        cat_smooth=150.0,
        cat_l2=30.0,
        max_cat_to_onehot=4,
        verbose=-1,
        seed=20260922,
    )
    model = lgb.train(
        params,
        lgb.Dataset(
            x_train,
            train.loc[usable, "w_star"],
            weight=train.loc[usable, "loss_weight"],
            categorical_feature=cats,
        ),
        num_boost_round=120,
    )
    return np.clip(model.predict(x_score), 0.0, 1.0)


def mix(frame: pd.DataFrame, gate_w: np.ndarray, alpha: float) -> np.ndarray:
    w = (1.0 - alpha) * frame.w.to_numpy() + alpha * gate_w
    pred = frame.normal.to_numpy() + w * frame.gap.to_numpy()
    pred[~frame.recover.to_numpy()] = frame.loc[~frame.recover, "pred"].to_numpy()
    return pred


def select_alpha_inner(cal: pd.DataFrame, features: list[str]) -> tuple[float, dict[str, float]]:
    """Two-way month cross-fit within the calibration fold; no target-fold labels."""
    months = sorted(cal.month.unique())
    if len(months) != 2:
        raise ValueError(f"expected two months in calibration fold, got {months}")
    gate = np.zeros(len(cal), dtype=float)
    for held_month in months:
        tr = cal[cal.month != held_month]
        te = cal[cal.month == held_month]
        gate[cal.month.to_numpy() == held_month] = fit_gate(tr, te, features)
    scores = {str(a): rmse(cal.y, mix(cal, gate, a)) for a in ALPHAS}
    best = min(ALPHAS, key=lambda a: (scores[str(a)], a))
    return float(best), scores


def probe() -> dict:
    d = load_train()
    baseline = rmse(d.y, d.pred)
    candidate = d.pred.to_numpy().copy()
    control = d.pred.to_numpy().copy()
    selected = {}
    inner = {}

    for target_fold in "ABCDEF":
        pair_fold = PAIR[target_fold]
        block = BLOCK[target_fold]
        outer_history = d[d.block != block]
        block_rows = add_outer_rates(d[d.block == block].copy(), outer_history)
        cal = block_rows[block_rows.fold == pair_fold].copy()
        target = block_rows[block_rows.fold == target_fold].copy()

        alpha, inner_scores = select_alpha_inner(cal, IDENTITY_FEATURES)
        identity_w = fit_gate(cal, target, IDENTITY_FEATURES)
        control_w = fit_gate(cal, target, CONTROL_FEATURES)
        pred_identity = mix(target, identity_w, alpha)
        pred_control = mix(target, control_w, alpha)
        positions = pd.Series(np.arange(len(d)), index=d.MVT_ID_mvt.values)
        idx = positions.reindex(target.MVT_ID_mvt.values).to_numpy()
        if pd.isna(idx).any():
            raise RuntimeError("target ID alignment failed")
        idx = idx.astype(int)
        candidate[idx] = pred_identity
        control[idx] = pred_control
        selected[target_fold] = alpha
        inner[target_fold] = inner_scores
        print(
            f"fold {target_fold} <- {pair_fold}: alpha {alpha:.2f}, "
            f"original {rmse(target.y, target.pred):.4f}, control {rmse(target.y, pred_control):.4f}, "
            f"identity {rmse(target.y, pred_identity):.4f}",
            flush=True,
        )

    # Map the LIRF-only result back to the whole A48 strict population.
    whole = pd.read_parquet(OOF_PATH)
    cmap = pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt, "candidate": candidate, "control": control})
    whole = whole.merge(cmap, on="MVT_ID_mvt", how="left", validate="one_to_one")
    whole["candidate"] = whole.candidate.fillna(whole.pred)
    whole["control"] = whole.control.fillna(whole.pred)
    fold_rows = []
    for f in "ABCDEF":
        z = whole.fold == f
        fold_rows.append({
            "fold": f,
            "n": int(z.sum()),
            "original": rmse(whole.loc[z, "y"], whole.loc[z, "pred"]),
            "control": rmse(whole.loc[z, "y"], whole.loc[z, "control"]),
            "identity": rmse(whole.loc[z, "y"], whole.loc[z, "candidate"]),
        })
    for row in fold_rows:
        row["identity_delta"] = row["identity"] - row["original"]
        row["control_delta"] = row["control"] - row["original"]

    report = {
        "approach": "A49-flight-identity-gate-probe",
        "protocol": "blocked-pair gate with inner month cross-fit for alpha",
        "features_control": CONTROL_FEATURES,
        "features_identity": IDENTITY_FEATURES,
        "selected_alpha_by_target_fold": selected,
        "inner_alpha_scores": inner,
        "lirf_rows": len(d),
        "lirf_original": baseline,
        "lirf_control": rmse(d.y, control),
        "lirf_identity": rmse(d.y, candidate),
        "whole_original": rmse(whole.y, whole.pred),
        "whole_control": rmse(whole.y, whole.control),
        "whole_identity": rmse(whole.y, whole.candidate),
        "whole_identity_delta": rmse(whole.y, whole.candidate) - rmse(whole.y, whole.pred),
        "folds": fold_rows,
    }
    report["passes"] = bool(
        report["whole_identity_delta"] <= -0.30
        and max(x["identity_delta"] for x in fold_rows) <= 0.0
        and report["whole_identity"] < report["whole_control"]
    )
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    whole[["MVT_ID_mvt", "APT", "month", "fold", "fpl_matched", "y", "pred", "candidate", "control", "w", "pb"]].to_parquet(
        OOF_OUT, index=False
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f"-> {REPORT_PATH}\n-> {OOF_OUT}", flush=True)
    return report


def deploy() -> None:
    if not REPORT_PATH.exists():
        raise SystemExit("run probe first")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if not report.get("passes"):
        raise SystemExit("probe did not pass the promotion gate; deployment refused")

    train_raw = load_train()
    train = add_crossfit_rates(train_raw)
    rank_detail = pd.read_parquet(RANK_DETAIL)
    rank_raw = _raw(BASE_RANK)
    rank = rank_detail.merge(rank_raw, on="MVT_ID_mvt", how="left", validate="one_to_one")
    rank = rank[(rank.APT == "LIRF") & (rank.fpl_matched == 1) & (rank.is_heli == 0)].copy()
    rank["D"] = rank.atot_minus_sched
    rank["a3taxi"] = rank.atot_minus_aobt3
    rank["recover"] = rank.w < 0.999999
    rank["normal"] = np.where(rank.recover, (rank.pred - rank.w * rank.D) / (1.0 - rank.w), rank.normal_pred)
    rank["gap"] = rank.D - rank.normal
    rank = engineer(rank)
    rank = add_outer_rates(rank, train_raw)

    gate_w = fit_gate(train, rank, IDENTITY_FEATURES)
    chosen = [float(v) for v in report["selected_alpha_by_target_fold"].values()]
    alpha = float(np.median(chosen))
    rank["flight_identity_gate_w"] = gate_w
    rank["flight_identity_alpha"] = alpha
    rank["pred_a49"] = mix(rank, gate_w, alpha)
    rank[["MVT_ID_mvt", "pred_a49", "flight_identity_gate_w", "flight_identity_alpha"]].to_parquet(RANK_OUT, index=False)
    print(f"deployment alpha {alpha:.2f}; {len(rank):,} LIRF matched rows -> {RANK_OUT}")


def targeted_fold_a_guard(report: dict) -> tuple[float, dict]:
    """Validate the narrower Jan/Jul evidence used for the v43 candidate.

    This is deliberately separate from the six-fold deployment gate.  It does
    not reinterpret the unstable E/F results as passing; it permits only the
    competition target's Jan/Jul seasonal pair, whose labels were untouched
    until its prediction and alpha had been fixed.
    """
    fold_a = next((row for row in report.get("folds", []) if row.get("fold") == "A"), None)
    if fold_a is None:
        raise SystemExit("probe report has no Fold A result")
    alpha = float(report.get("selected_alpha_by_target_fold", {}).get("A", np.nan))
    checks = {
        "protocol_is_nested": report.get("protocol") == "blocked-pair gate with inner month cross-fit for alpha",
        "alpha_is_allowed": bool(np.isfinite(alpha) and alpha in ALPHAS and alpha > 0),
        "fold_a_gain_at_least_0_15s": float(fold_a.get("identity_delta", np.inf)) <= -0.15,
        "identity_beats_nonidentity_control": float(fold_a.get("identity", np.inf)) < float(fold_a.get("control", -np.inf)),
    }
    if not all(checks.values()):
        raise SystemExit(f"targeted Fold A evidence gate failed: {checks}")
    return alpha, {**fold_a, "checks": checks}


def build_v43() -> None:
    """Build, package and log the requested Jan/Jul-only v43 candidate.

    Nothing is uploaded.  All non-target rows are copied bit for
    bit from v42; only matched, non-helicopter LIRF rows can change.
    """
    import submit

    if not REPORT_PATH.exists():
        raise SystemExit("run probe first")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    alpha, fold_a = targeted_fold_a_guard(report)

    train_raw = load_train()
    train = add_crossfit_rates(train_raw)
    rank_detail = pd.read_parquet(RANK_DETAIL)
    rank_raw = _raw(BASE_RANK)
    full = rank_detail.merge(rank_raw, on="MVT_ID_mvt", how="left", validate="one_to_one")
    rank_months = sorted(int(x) for x in full.month.dropna().unique())
    if rank_months != [1, 7]:
        raise SystemExit(f"v43 is Fold-A-targeted but rank months are {rank_months}, expected [1, 7]")

    target_mask = (full.APT == "LIRF") & (full.fpl_matched == 1) & (full.is_heli == 0)
    rank = full.loc[target_mask].copy()
    if rank.empty:
        raise SystemExit("no matched non-helicopter LIRF rank rows")
    rank["D"] = rank.atot_minus_sched
    rank["a3taxi"] = rank.atot_minus_aobt3
    rank["recover"] = rank.w < 0.999999
    rank["normal"] = np.where(
        rank.recover,
        (rank.pred - rank.w * rank.D) / (1.0 - rank.w),
        rank.normal_pred,
    )
    rank["gap"] = rank.D - rank.normal
    rank = add_outer_rates(engineer(rank), train_raw)

    gate_w = fit_gate(train, rank, IDENTITY_FEATURES)
    rank["flight_identity_gate_w"] = gate_w
    rank["flight_identity_alpha"] = alpha
    rank["pred_a49"] = mix(rank, gate_w, alpha)
    rank[["MVT_ID_mvt", "pred_a49", "flight_identity_gate_w", "flight_identity_alpha"]].to_parquet(
        RANK_OUT, index=False
    )

    base = pd.read_parquet(BASE_SUBMISSION)
    if base.MVT_ID_mvt.duplicated().any() or rank_detail.MVT_ID_mvt.duplicated().any():
        raise SystemExit("duplicate movement IDs in base submission or A48 package")
    base_by_id = pd.Series(base.TAXITIME_SEC_mvt.values, index=base.MVT_ID_mvt.values)
    base_pred = base_by_id.reindex(rank_detail.MVT_ID_mvt.values).to_numpy()
    if not np.isfinite(base_pred).all():
        raise SystemExit("v42 submission ID set does not match the A48 package")
    package_base = np.maximum(rank_detail.pred.to_numpy(dtype=float), 30.0)
    reproduction_gap = float(np.max(np.abs(base_pred - package_base)))
    if reproduction_gap > 1e-9:
        raise SystemExit(f"A48 package does not reproduce v42 (max gap {reproduction_gap:.3g})")

    candidate_by_id = pd.Series(rank.pred_a49.values, index=rank.MVT_ID_mvt.values)
    target_candidate = candidate_by_id.reindex(rank_detail.loc[target_mask, "MVT_ID_mvt"].values).to_numpy()
    if not np.isfinite(target_candidate).all():
        raise SystemExit("target prediction alignment failed")
    new_pred = base_pred.copy()
    new_pred[target_mask.to_numpy()] = np.maximum(target_candidate, 30.0)
    diff = new_pred - base_pred
    changed = np.abs(diff) > 1e-12
    if np.any(changed & ~target_mask.to_numpy()):
        raise SystemExit("a non-target row changed")

    if V43_SUBMISSION.exists():
        raise SystemExit(f"refusing to overwrite existing {V43_SUBMISSION}")
    expected_version = submit.next_version(submit.team_name())
    if expected_version != 43:
        raise SystemExit(f"next submission version is {expected_version}, not requested v43")
    submission = submit.write_submission(submit.team_name(), 43, rank_detail.MVT_ID_mvt.values, new_pred)
    if submission != V43_SUBMISSION:
        raise SystemExit(f"unexpected submission path {submission}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    package = ROOT / "runs" / "models" / f"A49_{stamp}"
    package.mkdir(parents=True, exist_ok=False)
    gate_by_id = pd.Series(rank.flight_identity_gate_w.values, index=rank.MVT_ID_mvt.values)
    detail = rank_detail.copy()
    detail["pred_a48"] = detail.pred.to_numpy()
    detail["flight_identity_gate_w"] = gate_by_id.reindex(detail.MVT_ID_mvt.values).to_numpy()
    detail["flight_identity_alpha"] = np.where(target_mask.to_numpy(), alpha, 0.0)
    detail["flight_identity_applied"] = changed
    detail["pred"] = new_pred
    detail.to_parquet(package / "rank_detail.parquet", index=False)

    estimated_gain = [0.20, 0.35]
    summary = {
        "approach": "A49",
        "candidate_version": 43,
        "base_approach": "A48",
        "base_package": str(RANK_DETAIL.relative_to(ROOT)).replace("\\", "/"),
        "base_submission": str(BASE_SUBMISSION.relative_to(ROOT)).replace("\\", "/"),
        "submission": str(submission.relative_to(ROOT)).replace("\\", "/"),
        "protocol": "Jan/Jul Fold-A-targeted exact-flight identity gate; no change outside matched non-helicopter LIRF",
        "rank_months": rank_months,
        "alpha": alpha,
        "fold_a_evidence": fold_a,
        "general_six_fold_gate_passes": bool(report.get("passes", False)),
        "base_reproduction_max_abs_gap": reproduction_gap,
        "rows": int(len(detail)),
        "eligible_rows": int(target_mask.sum()),
        "changed_rows": int(changed.sum()),
        "unchanged_rows_exact": int((diff == 0).sum()),
        "rms_change_all": float(np.sqrt(np.mean(np.square(diff)))),
        "rms_change_eligible": float(np.sqrt(np.mean(np.square(diff[target_mask.to_numpy()])))),
        "mean_change_eligible": float(np.mean(diff[target_mask.to_numpy()])),
        "max_abs_change": float(np.max(np.abs(diff))),
        "finite": bool(np.isfinite(new_pred).all()),
        "min_prediction": float(new_pred.min()),
        "upload_performed": False,
        "decision": "candidate",
    }
    (package / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (package / "submission.txt").write_text(submission.name, encoding="utf-8")
    evidence_path = ROOT / "runs" / f"evidence_A49_{time.strftime('%Y%m%d')}.json"
    evidence_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log_path = ROOT / "submissions" / "submission_log.csv"
    log = pd.read_csv(log_path)
    if (log.file == submission.name).any():
        raise SystemExit(f"submission log already contains {submission.name}")
    base_models = json.loads(log.loc[log.file == BASE_SUBMISSION.name, "models_json"].iloc[-1])
    notes = (
        f"A49 = A48 + Jan/Jul-targeted exact-flight identity correction on matched non-helicopter LIRF only; "
        f"nested Fold A {fold_a['identity']:.4f} ({fold_a['identity_delta']:+.4f} vs A48, "
        f"{fold_a['identity'] - fold_a['control']:+.4f} vs non-identity control), alpha {alpha:.2f}; "
        "six-fold general gate remains failed; candidate built locally, not uploaded"
    )
    row = dict(
        file=submission.name,
        ts=time.strftime("%Y-%m-%d %H:%M:%S"),
        team=submit.team_name(),
        version_n=43,
        approach_id="A49",
        feature_version=5,
        models_json=json.dumps(base_models + ["flight_identity_gate"]),
        foldA_rmse=fold_a["identity"],
        # The v43 policy is Jan/Jul-only.  Outside Fold A it is A48 unchanged,
        # so record A48's Fold B score rather than implying an untested gain.
        foldB_rmse=next(x["original"] for x in report["folds"] if x["fold"] == "B"),
        result_rmse="",
        result_rank="",
        decision="candidate",
        notes=notes,
    )
    pd.DataFrame([row]).to_csv(log_path, mode="a", index=False, header=False, lineterminator="\n")
    submit.append_manifest([
        f"# Candidate A49 {time.strftime('%Y-%m-%d %H:%M')} - team {submit.team_name()}",
        "",
        "| file | approach | description | strict Fold A | upload |",
        "|---|---|---|---:|---|",
        f"| {submission.name} | A49 | {notes} | {fold_a['identity']:.4f} | no |",
    ])
    print(json.dumps({**summary, "package": str(package), "evidence": str(evidence_path)}, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["probe", "deploy", "build-v43"])
    a = p.parse_args()
    if a.command == "probe":
        probe()
    elif a.command == "deploy":
        deploy()
    else:
        build_v43()


if __name__ == "__main__":
    main()
