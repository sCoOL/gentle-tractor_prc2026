"""One row per OPDI flight x our airport x role (dep / arr), from the extracted events of scripts/258.

  py -3.14 scripts/259_opdi_flights.py

Role: 'dep' when the flight list's ADEP is the airport of the ground events, 'arr' when ADES is (a round trip with
ADEP = ADES gets both rows, split at the flight's first runway exit). Ground events are OPDI's own: entering /
leaving an OSM aeroway element (parking_position, apron, taxiway, runway, threshold, deicing_pad, hangar) at
5-second resolution; `in_use_s` is the time the aircraft spent in that element (scripts/258 docstring).

Outputs (<PRC_OPDI_DIR>/):
  opdi_flights.parquet   per flight x airport x role; times are UTC timestamps. Departure columns:
      xp_first_*   first exit from a parking_position (t, ref, in_use_s)        <- the stand it started from, if seen
      xp_last_*    last parking exit before the final runway entry               <- can be a stand passed while taxiing
      xp_long_*    the parking exit with the longest in_use_s before the runway  <- the stand it waited longest at
      xa_first_t   first apron exit;  tw_first_t first taxiway entry;  g_first_t / g_first_type first ground event
      dc_t, dc_in_use_s, dc_ref   de-icing pad entry and time spent there
      n_tw, tw_in_use_sum_s, tw_in_use_max_s, tw_route (taxiway refs in order, '>'-joined)   between stand exit and runway
      er_t, er_ref, er_in_use_s, n_er   last runway entry before take-off; n_er - 1 = runways entered before (crossings)
      eth_t      last threshold entry;  to_t   OPDI take-off event (sparse in v0.0.2)
      dist_xp_er_nm, time_xp_er_s   measurement difference between the first parking exit and the runway entry
    Arrival columns: xr_first_t/ref (first runway exit after landing), ld_t (landing event), ep_first_* / ep_last_* /
      ep_long_* (parking entries), n_tw, tw_in_use_*, g_last_t.
  opdi_parking.parquet   every parking exit of a dep flight / parking entry of an arr flight (flight_id, apt, role,
                         t, ref, in_use_s, seq) for the recorded-stand match in scripts/260.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc_config  # noqa: E402

ROOT = prc_config.OPDI_DIR   # reads 258 output and writes here (PRC_OPDI_DIR)
NPART = 8
KEY = ["flight_id", "apt", "role"]


def load_part(k, files, fl):
    parts = []
    for f in files:
        d = pd.read_parquet(f)
        d = d[(d.flight_id % NPART) == k]
        parts.append(d)
    # legacy event ids are not deterministic and re-processed months were appended (opdi repo config.py:1363-1369)
    d = pd.concat(parts, ignore_index=True).drop_duplicates(["flight_id", "type", "event_time", "osm_id"])
    d = d.rename(columns={"event_time": "t"})
    d = remap_ids(d, fl)
    air = d[d.apt.isna()]
    g = d[d.apt.notna()].merge(fl[["id", "adep", "ades"]].rename(columns={"id": "flight_id"}), on="flight_id", how="left")
    return g.sort_values(["flight_id", "t"]), air


def remap_ids(d, fl):
    """Ground events whose flight_id is not in the flight list (flight_events_20251201_20251211 uses another id
    scheme than flight_list_202512) get the id of the flight-list row with the same icao24 + callsign, ADEP or ADES
    = the event's airport, and a span overlapping the events (+-2 h); the closest first_seen wins."""
    miss = d.apt.notna() & ~d.flight_id.isin(fl.id)
    if not miss.any():
        return d
    c = d[miss].groupby(["flight_id", "apt"]).agg(icao24=("icao24", "first"), cs=("cs", "first"),
                                                  t0=("t", "min"), t1=("t", "max")).reset_index()
    f = fl[["id", "icao24", "flt_id", "adep", "ades", "first_seen", "last_seen"]].assign(cs=fl.flt_id.str.strip())
    for x in ("first_seen", "last_seen"):
        f[x] = pd.to_datetime(f[x]).dt.tz_localize("UTC")
    j = c.merge(f.drop(columns="flt_id"), on=["icao24", "cs"])
    h2 = pd.Timedelta(hours=2)
    j = j[(j.adep.eq(j.apt) | j.ades.eq(j.apt)) & (j.t1 >= j.first_seen - h2) & (j.t0 <= j.last_seen + h2)]
    j = j.assign(k=(j.t0 - j.first_seen).abs()).sort_values("k").drop_duplicates(["flight_id", "apt"])
    # ids are full-range uint64: keep them out of float (a NaN-bearing merge would round them)
    m = d.loc[miss, ["flight_id", "apt"]].reset_index().merge(
        j[["flight_id", "apt", "id"]].astype({"id": "UInt64"}), on=["flight_id", "apt"], how="left")
    hit = m["id"].notna().to_numpy()
    d.loc[m["index"].to_numpy()[hit], "flight_id"] = m["id"].to_numpy()[hit].astype("uint64")
    print(f"  remapped {int(hit.sum()):,} of {int(miss.sum()):,} unjoined ground events", flush=True)
    return d


def pick(df, key, keep, cols, prefix):
    x = df.drop_duplicates(key, keep=keep)[key + cols]
    return x.rename(columns={c: f"{prefix}_{c}" for c in cols})


def summarise(g, air):
    g = g.copy()
    # round trips: events before the first runway exit belong to the departure, the rest to the arrival
    rt = g.adep.eq(g.apt) & g.ades.eq(g.apt)
    g["role"] = np.select([g.adep.eq(g.apt) & ~rt, g.ades.eq(g.apt) & ~rt], ["dep", "arr"], "other")
    if rt.any():
        r = g[rt]
        cut = r[r.type == "exit-runway"].groupby("flight_id").t.min()
        g.loc[rt, "role"] = np.where(r.t.values > r.flight_id.map(cut).fillna(pd.Timestamp.max.tz_localize("UTC")).values,
                                     "arr", "dep")
    g = g[g.role != "other"]
    rows, parking = [], []
    # ---------------- departures
    d = g[g.role == "dep"]
    base = d.groupby(KEY).agg(n_gnd=("t", "size"), g_first_t=("t", "first"), g_first_type=("type", "first"),
                              g_last_t=("t", "last")).reset_index()
    er = d[d.type == "entry-runway"]
    er_last = pick(er, KEY, "last", ["t", "osm_ref", "in_use_s", "meas_dist_nm", "meas_time_s"], "er")
    er_last = er_last.merge(er.groupby(KEY).size().rename("n_er").reset_index(), on=KEY)
    base = base.merge(er_last, on=KEY, how="left")
    d = d.merge(base[KEY + ["er_t"]], on=KEY, how="left")
    pre = d[d.er_t.isna() | (d.t <= d.er_t)]
    xp = pre[pre.type == "exit-parking_position"]
    for keep, name in [("first", "xp_first"), ("last", "xp_last")]:
        base = base.merge(pick(xp, KEY, keep, ["t", "osm_ref", "in_use_s", "meas_dist_nm", "meas_time_s"], name),
                          on=KEY, how="left")
    base = base.merge(pick(xp.sort_values(["in_use_s", "t"], na_position="first"), KEY, "last",
                           ["t", "osm_ref", "in_use_s"], "xp_long"), on=KEY, how="left")
    parking.append(xp.assign(seq=xp.groupby(KEY).cumcount())[KEY + ["t", "osm_ref", "in_use_s", "seq"]])
    base = base.merge(pick(pre[pre.type == "exit-apron"], KEY, "first", ["t"], "xa_first"), on=KEY, how="left")
    base = base.merge(pick(pre[pre.type == "entry-taxiway"], KEY, "first", ["t"], "tw_first"), on=KEY, how="left")
    base = base.merge(pick(pre[pre.type == "entry-deicing_pad"], KEY, "first", ["t", "osm_ref", "in_use_s"], "dc"),
                      on=KEY, how="left")
    base = base.merge(pick(pre[pre.type == "entry-threshold"], KEY, "last", ["t"], "eth"), on=KEY, how="left")
    # taxi between the first parking exit (or the first ground event) and the runway entry
    d = d.merge(base[KEY + ["xp_first_t"]], on=KEY, how="left")
    tx = d[(d.type == "entry-taxiway") & (d.xp_first_t.isna() | (d.t >= d.xp_first_t)) & (d.er_t.isna() | (d.t <= d.er_t))]
    tws = tx.groupby(KEY).agg(n_tw=("t", "size"), tw_in_use_sum_s=("in_use_s", "sum"), tw_in_use_max_s=("in_use_s", "max"),
                              tw_route=("osm_ref", lambda s: ">".join(s.fillna("?").astype(str)))).reset_index()
    base = base.merge(tws, on=KEY, how="left")
    base["dist_xp_er_nm"] = base.er_meas_dist_nm - base.xp_first_meas_dist_nm
    base["time_xp_er_s"] = base.er_meas_time_s - base.xp_first_meas_time_s
    to = air[air.type == "take-off"].sort_values("t").drop_duplicates("flight_id")[["flight_id", "t"]].rename(columns={"t": "to_t"})
    base = base.merge(to, on="flight_id", how="left")
    base = base.rename(columns={"er_osm_ref": "er_ref", "xp_first_osm_ref": "xp_first_ref", "xp_last_osm_ref": "xp_last_ref",
                                "xp_long_osm_ref": "xp_long_ref", "dc_osm_ref": "dc_ref"})
    rows.append(base)
    # ---------------- arrivals
    a = g[g.role == "arr"]
    base = a.groupby(KEY).agg(n_gnd=("t", "size"), g_first_t=("t", "first"), g_first_type=("type", "first"),
                              g_last_t=("t", "last")).reset_index()
    xr = a[a.type == "exit-runway"]
    base = base.merge(pick(xr, KEY, "first", ["t", "osm_ref"], "xr_first"), on=KEY, how="left")
    a = a.merge(base[KEY + ["xr_first_t"]], on=KEY, how="left")
    post = a[a.xr_first_t.isna() | (a.t >= a.xr_first_t)]
    ep = post[post.type == "entry-parking_position"]
    for keep, name in [("first", "ep_first"), ("last", "ep_last")]:
        base = base.merge(pick(ep, KEY, keep, ["t", "osm_ref", "in_use_s"], name), on=KEY, how="left")
    base = base.merge(pick(ep.sort_values(["in_use_s", "t"], na_position="first"), KEY, "last", ["t", "osm_ref", "in_use_s"],
                           "ep_long"), on=KEY, how="left")
    parking.append(ep.assign(seq=ep.groupby(KEY).cumcount())[KEY + ["t", "osm_ref", "in_use_s", "seq"]])
    tx = post[post.type == "entry-taxiway"]
    base = base.merge(tx.groupby(KEY).agg(n_tw=("t", "size"), tw_in_use_sum_s=("in_use_s", "sum"),
                                          tw_in_use_max_s=("in_use_s", "max")).reset_index(), on=KEY, how="left")
    ld = air[air.type == "landing"].sort_values("t").drop_duplicates("flight_id", keep="last")[["flight_id", "t"]]
    base = base.merge(ld.rename(columns={"t": "ld_t"}), on="flight_id", how="left")
    base = base.rename(columns={"xr_first_osm_ref": "xr_first_ref", "ep_first_osm_ref": "ep_first_ref",
                                "ep_last_osm_ref": "ep_last_ref", "ep_long_osm_ref": "ep_long_ref"})
    rows.append(base)
    return pd.concat(rows, ignore_index=True), pd.concat(parking, ignore_index=True)


def main():
    files = sorted((ROOT / "ours").glob("ev_*.parquet"))
    fl = pd.read_parquet(ROOT / "ours" / "flight_list_ours.parquet")
    fl = fl.drop_duplicates("id")
    print(f"{len(files)} extracted periods, {len(fl):,} flights", flush=True)
    out, park = [], []
    for k in range(NPART):
        g, air = load_part(k, files, fl)
        s, p = summarise(g, air)
        out.append(s); park.append(p)
        print(f"part {k}: {len(g):,} ground events -> {len(s):,} flight rows", flush=True)
    s = pd.concat(out, ignore_index=True)
    s = s.merge(fl[["id", "icao24", "flt_id", "adep", "ades", "typecode", "icao_operator", "registration",
                    "first_seen", "last_seen"]].rename(columns={"id": "flight_id"}), on="flight_id", how="left")
    for c in ("first_seen", "last_seen"):
        s[c] = pd.to_datetime(s[c]).dt.tz_localize("UTC")
    s.to_parquet(ROOT / "opdi_flights.parquet", index=False)
    pd.concat(park, ignore_index=True).to_parquet(ROOT / "opdi_parking.parquet", index=False)
    print(s.groupby(["apt", "role"]).size().unstack().to_string())
    print("EXIT", flush=True)


if __name__ == "__main__":
    sys.exit(main())
