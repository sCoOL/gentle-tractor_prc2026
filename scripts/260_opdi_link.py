"""Link our movements (2025 training + 2026 ranking, departures and arrivals) to OPDI flights (scripts/259).

  py -3.14 scripts/260_opdi_link.py

Label-free: no block time is read for linking. A movement is linked to an OPDI flight of the same airport and role:
  pass 1 'icao'  the aircraft's icao24 from the adsb.lol match (<PRC_ADSB_DIR>/adsb_times.parquet) = OPDI icao24
  pass 1 'cs'    a normalized callsign candidate (NM callsign, airport flight number, destination-arrival NM callsign;
                 scripts/255 norm(): IATA prefixes mapped to ICAO, leading zeros dropped) = OPDI flt_id normalized
     time check, departures: OPDI runway entry - recorded take-off in [-40 min, +10 min], or without a runway entry
                 the take-off inside [first_seen - 10 min, last_seen + 10 min] and first_seen at most 4 h before it.
     arrivals:   OPDI first runway exit - recorded landing in [-10 min, +40 min], or the landing inside the flight span.
  pass 2 'time'  unclaimed OPDI flights of the airport whose runway entry (dep) / exit (arr) is within -180..+60 s
                 (dep) / -60..+180 s (arr) of the recorded time, dropped when both sides name an airline and they differ.
One OPDI flight per movement and one movement per OPDI flight (closest time first).
Recorded-stand match: the departure's last parking EXIT (arrival: first parking ENTRY) whose osm_ref equals STAND_mvt
after normalising (upper case, alphanumerics only, leading zeros after a letter dropped: 'A08' = 'A8').

Output <PRC_OPDI_DIR>/opdi_movements.parquet: MVT_ID_mvt, PHASE_mvt, APT, split, opdi_how, opdi_gap_s, flight_id and
the opdi_flights columns, plus xs_t / xs_ref / xs_in_use_s (dep, at the recorded stand) or es_t / es_ref / es_in_use_s
(arr).
"""
import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

m255 = importlib.import_module("255_adsb_match_v2")
ROOT = common.prc_config.OPDI_DIR   # reads 259 output and writes here (PRC_OPDI_DIR)
MIN = pd.Timedelta(minutes=1)


def stand_norm(x):
    if not isinstance(x, str):
        return ""
    x = re.sub(r"[^A-Z0-9]", "", x.upper())
    return re.sub(r"(?<=[A-Z])0+(?=\d)", "", x)


def movements():
    raw = pd.concat([pd.read_parquet(p, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt",
                                                 "CALLSIGN_flt", "MVT_TIME_UTC_mvt", "STAND_mvt"]).assign(split=s)
                     for p, s in [(x, "train") for x in sorted(common.DATA.glob("training_2025-*.parquet"))]
                     + [(common.DATA / "ranking.parquet", "rank")]], ignore_index=True)
    raw["APT"] = np.where(raw.PHASE_mvt == "DEP", raw.ADEP_mvt, raw.ADES_mvt)
    raw["role"] = np.where(raw.PHASE_mvt == "DEP", "dep", "arr")
    raw["t"] = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True)
    tm = pd.read_parquet(common.prc_config.ADSB_DIR / "adsb_times.parquet", columns=["MVT_ID_mvt", "adsb_icao"]).dropna()
    raw = raw.merge(tm.drop_duplicates("MVT_ID_mvt"), on="MVT_ID_mvt", how="left")
    raw["bf_cs"] = m255.backfill_callsigns().reindex(raw.MVT_ID_mvt).to_numpy()
    return raw


def base_norm(x) -> str:
    """Suffix-tolerant key (scripts/flightnorm.py base_key), after 255 has set the IATA->ICAO map."""
    m255._ensure_map()
    return m255.fn.base_key(x)


def normcol(s, fn=None):
    """fn (default scripts/255 norm) over the unique values only."""
    fn = fn or m255.norm
    u = pd.Series(s).dropna().unique()
    return pd.Series(s).map(dict(zip(u, [fn(x) for x in u]))).fillna("").to_numpy()


def anchor(f):
    """The OPDI time compared with the recorded movement time: runway entry (dep) / first runway exit (arr)."""
    return np.where(f.role == "dep", f.er_t, f.xr_first_t)


def keyed(mv, fl, how):
    # same airport, role and key, and the OPDI reference day (runway time, else first_seen) within +-1 day: without the
    # day an aircraft's every flight would meet its every movement over 19 months
    fl = fl.assign(day=pd.to_datetime(fl.anchor_t, utc=True).fillna(fl.first_seen).dt.floor("D"))
    base = ["MVT_ID_mvt", "APT", "role", "t"]
    if how == "icao":
        m = mv.dropna(subset=["adsb_icao"])[base + ["adsb_icao"]].rename(columns={"adsb_icao": "k"})
        fk = fl.drop(columns=["ncs", "ncs_base"]).rename(columns={"icao24": "k"})
    else:
        # 2026-10-07: exact and suffix-tolerant keys (scripts/flightnorm.py) of the NM callsign, the airport flight number
        # and the destination-arrival callsign, plus the flight number's learned NM callsign aliases (255.aliases)
        cand = pd.concat([pd.DataFrame({"MVT_ID_mvt": mv.MVT_ID_mvt.to_numpy(), "k": normcol(mv[col], f)})
                          for col in ("CALLSIGN_flt", "FLIGHT_mvt", "bf_cs") for f in (m255.norm, base_norm)], ignore_index=True)
        al = m255.aliases()
        fk_air = normcol(mv.FLIGHT_mvt)
        extra = [(m, c) for m, a, k in zip(mv.MVT_ID_mvt.to_numpy(), mv.APT.to_numpy(), fk_air) for c in al.get((a, k), [])]
        if extra:
            cand = pd.concat([cand, pd.DataFrame(extra, columns=["MVT_ID_mvt", "k"])], ignore_index=True)
        cand = cand[cand.k != ""].drop_duplicates()
        m = mv[base].merge(cand, on="MVT_ID_mvt")
        fk = pd.concat([fl.drop(columns=["icao24", "ncs_base"]).rename(columns={"ncs": "k"}),
                        fl.drop(columns=["icao24", "ncs"]).rename(columns={"ncs_base": "k"})], ignore_index=True)
        fk = fk[fk.k != ""].drop_duplicates(["flight_id", "apt", "role", "k"])
    c = pd.concat([m.assign(day=m.t.dt.floor("D") + pd.Timedelta(days=off))
                   .merge(fk, left_on=["APT", "role", "k", "day"], right_on=["apt", "role", "k", "day"])
                   for off in (-1, 0, 1)], ignore_index=True)
    a = pd.to_datetime(c.anchor_t, utc=True)
    gap = (a - c.t)
    dep = c.role == "dep"
    ok_a = np.where(dep, (gap >= -40 * MIN) & (gap <= 10 * MIN), (gap >= -10 * MIN) & (gap <= 40 * MIN)) & a.notna()
    span = (c.t >= c.first_seen - 10 * MIN) & (c.t <= c.last_seen + 10 * MIN)
    span &= np.where(dep, c.t - c.first_seen <= 240 * MIN, True)
    c = c[ok_a | (a.isna() & span)].copy()
    c["opdi_gap_s"] = np.where(a.loc[c.index].notna(), gap.loc[c.index].dt.total_seconds(),
                               (c.t - c.first_seen).dt.total_seconds() + 1e6)  # span-only links sort last
    return c.assign(opdi_how=how)


def main():
    fl = pd.read_parquet(ROOT / "opdi_flights.parquet")
    fl["ncs"] = normcol(fl.flt_id)
    fl["ncs_base"] = normcol(fl.flt_id, base_norm)
    fl["nal"] = normcol(fl.flt_id, m255.airline)
    fl["anchor_t"] = anchor(fl)
    mv = movements()
    print(f"movements {len(mv):,}; OPDI flight rows {len(fl):,}", flush=True)
    fcols = ["flight_id", "apt", "role", "icao24", "ncs", "ncs_base", "nal", "anchor_t", "first_seen", "last_seen"]
    c = pd.concat([keyed(mv, fl[fcols], "icao"), keyed(mv, fl[fcols], "cs")], ignore_index=True)
    c["k"] = c.opdi_gap_s.abs() + np.where(c.opdi_how == "icao", 0, 1)  # icao wins a tie
    c = c.sort_values("k").drop_duplicates("MVT_ID_mvt").drop_duplicates(["flight_id", "apt", "role"])
    link = c[["MVT_ID_mvt", "flight_id", "apt", "role", "opdi_how", "opdi_gap_s"]]
    # pass 2: time only
    rest = mv[~mv.MVT_ID_mvt.isin(link.MVT_ID_mvt)]
    free = fl[fcols].merge(link[["flight_id", "apt", "role"]], how="left", indicator=True)
    free = free[(free._merge == "left_only") & free.anchor_t.notna()].drop(columns="_merge")
    rest = rest.assign(day=rest.t.dt.floor("D"))
    parts = []
    for off in (pd.Timedelta(0), pd.Timedelta(days=1), pd.Timedelta(days=-1)):
        f2 = free.assign(day=pd.to_datetime(free.anchor_t, utc=True).dt.floor("D") + off)
        parts.append(rest.merge(f2, left_on=["APT", "role", "day"], right_on=["apt", "role", "day"]))
    p = pd.concat(parts, ignore_index=True)
    gap = (pd.to_datetime(p.anchor_t, utc=True) - p.t).dt.total_seconds()
    p["opdi_gap_s"] = gap
    p = p[np.where(p.role == "dep", (gap >= -180) & (gap <= 60), (gap >= -60) & (gap <= 180))]
    a1, a2 = normcol(p.CALLSIGN_flt, m255.airline), normcol(p.FLIGHT_mvt, m255.airline)
    mal = pd.Series(np.where(a1 != "", a1, a2), index=p.index)
    p = p[~((mal != "") & (p.nal != "") & (mal != p.nal))]
    p = p.assign(k=p.opdi_gap_s.abs()).sort_values("k").drop_duplicates("MVT_ID_mvt").drop_duplicates(["flight_id", "apt", "role"])
    link = pd.concat([link, p[["MVT_ID_mvt", "flight_id", "apt", "role", "opdi_gap_s"]].assign(opdi_how="time")],
                     ignore_index=True)
    link = link.astype({"flight_id": "UInt64"})  # full-range uint64: a NaN-bearing merge must not turn it into float
    fl = fl.astype({"flight_id": "UInt64"})
    out = mv[["MVT_ID_mvt", "PHASE_mvt", "APT", "role", "split", "t", "STAND_mvt"]].merge(link.drop(columns=["apt", "role"]),
                                                                                          on="MVT_ID_mvt", how="left")
    out = out.merge(fl.drop(columns=["ncs", "ncs_base", "nal", "anchor_t"]), on=["flight_id"], how="left", suffixes=("", "_opdi"))
    out = out[(out.flight_id.isna()) | ((out.apt == out.APT) & (out.role_opdi == out.role))] \
        if "role_opdi" in out else out[(out.flight_id.isna()) | (out.apt == out.APT)]
    # recorded-stand match
    pk = pd.read_parquet(ROOT / "opdi_parking.parquet")
    pk["sn"] = normcol(pk.osm_ref, stand_norm)
    o = out[["MVT_ID_mvt", "flight_id", "APT", "role", "STAND_mvt"]].dropna(subset=["flight_id"])
    o = o.assign(sn=normcol(o.STAND_mvt, stand_norm))
    j = o.merge(pk, left_on=["flight_id", "APT", "role", "sn"], right_on=["flight_id", "apt", "role", "sn"])
    j = j[j.sn != ""].sort_values("t")
    jd = j[j.role == "dep"].drop_duplicates("MVT_ID_mvt", keep="last")
    ja = j[j.role == "arr"].drop_duplicates("MVT_ID_mvt", keep="first")
    out = out.merge(jd[["MVT_ID_mvt", "t", "osm_ref", "in_use_s"]].rename(columns={"t": "xs_t", "osm_ref": "xs_ref", "in_use_s": "xs_in_use_s"}),
                    on="MVT_ID_mvt", how="left")
    out = out.merge(ja[["MVT_ID_mvt", "t", "osm_ref", "in_use_s"]].rename(columns={"t": "es_t", "osm_ref": "es_ref", "in_use_s": "es_in_use_s"}),
                    on="MVT_ID_mvt", how="left")
    out.to_parquet(ROOT / "opdi_movements.parquet", index=False)
    s = out.assign(linked=out.flight_id.notna())
    print(s.pivot_table(index="APT", columns=["split", "role"], values="linked", aggfunc="mean").round(3).to_string())
    print(s.groupby(["split", "role"]).opdi_how.value_counts(dropna=False).unstack().to_string())
    print("EXIT", flush=True)


if __name__ == "__main__":
    main()
