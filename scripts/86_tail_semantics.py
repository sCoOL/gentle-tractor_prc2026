"""Read-only timestamp and source breakdown for extreme strict OOF errors."""
from __future__ import annotations

import json
import pandas as pd

import common

OUT = common.RUNS / "pipeline_reset_20260916"


def main() -> dict:
    cols = ["MVT_ID_mvt", "APT", "month", "y", "sse", "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "fpl_matched",
            "schedule_defaulted", "nm_mismatch_600", "fallback_subtype"]
    d = pd.read_parquet(OUT / "strict_v18_ledger.parquet", columns=cols)
    extreme = d[d.y > 14400].copy()
    extreme["near_24h"] = extreme.y.between(23*3600, 25*3600)
    extreme["block_day_before"] = (
        extreme.MVT_TIME_UTC_mvt.dt.floor("D") -
        extreme.BLOCK_TIME_UTC_mvt.dt.floor("D") == pd.Timedelta(days=1))
    extreme["label_equals_timestamp_delta"] = (
        (extreme.MVT_TIME_UTC_mvt-extreme.BLOCK_TIME_UTC_mvt).dt.total_seconds()-extreme.y).abs().le(1)
    extreme[["MVT_ID_mvt", "APT", "month", "y", "sse", "near_24h",
             "block_day_before", "label_equals_timestamp_delta", "fpl_matched",
             "schedule_defaulted", "fallback_subtype"]].to_csv(OUT / "extreme_tail_semantics.csv", index=False)
    report = {"n_over_4h": len(extreme), "near_24h": int(extreme.near_24h.sum()),
              "block_day_before": int(extreme.block_day_before.sum()),
              "label_equals_timestamp_delta": int(extreme.label_equals_timestamp_delta.sum()),
              "nm_missing": int(extreme.fpl_matched.eq(0).sum()),
              "schedule_defaulted": int(extreme.schedule_defaulted.sum()),
              "sse_share": float(extreme.sse.sum()/d.sse.sum()),
              "by_airport": extreme.APT.value_counts().to_dict()}
    (OUT / "extreme_tail_semantics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
