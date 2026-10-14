"""Cut the downloaded OPDI v002 files (scripts/257) down to what touches our ten airports.

  py -3.14 scripts/258_opdi_extract.py               # every downloaded 10-day period not yet extracted
  py -3.14 scripts/258_opdi_extract.py --force       # redo all

Outputs in <PRC_OPDI_DIR>/ours/:
  flight_list_ours.parquet   flights with ADEP or ADES in our ten airports (all monthly files)
  ev_<start>_<end>.parquet   per 10-day period: (a) every GROUND event whose info.osm_airport is one of ours
                             (runway / taxiway / apron / parking_position / threshold / hangar / deicing_pad
                             entries and exits, whatever flight they belong to) and (b) take-off, landing,
                             first_seen, last_seen of the flights in flight_list_ours. `info` is parsed into
                             osm_id, osm_aeroway, osm_ref, apt (osm_airport), in_use_s (opdi_time_in_use_s),
                             icao24 (osn_icao24), cs (osn_flight_id, stripped). Measurements of each kept event
                             are joined as meas_dist_nm ('Distance flown (NM)') and meas_time_s ('Time Passed (s)').
File versions differ (flight_id / id int64 vs uint64, event_time string vs timestamp): ids are normalised to uint64
(two's complement), times to UTC timestamps.
"""
import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc_config  # noqa: E402

ROOT = prc_config.OPDI_DIR   # outputs (PRC_OPDI_DIR)
RAW = prc_config.OPDI_RAW    # OPDI downloads, read only (PRC_OPDI_RAW)
OUT = ROOT / "ours"
OURS = ["EGLL", "LFPG", "EHAM", "EDDF", "EDDM", "LEMD", "LEBL", "LIRF", "LSZH", "LTFM"]
AIR = ["take-off", "landing", "first_seen", "last_seen"]
APT_RX = '"osm_airport":"(' + "|".join(OURS) + ')"'
FL_COLS = ["id", "icao24", "flt_id", "dof", "adep", "ades", "adep_p", "ades_p", "registration", "typecode",
           "icao_aircraft_class", "icao_operator", "first_seen", "last_seen"]


def u64(a):
    return a if a.type == pa.uint64() else pc.cast(a, pa.uint64(), safe=False)


def ts(a):
    if pa.types.is_timestamp(a.type):
        return pc.cast(a, pa.timestamp("us"))
    return pc.cast(pc.strptime(a, format="%Y-%m-%d %H:%M:%S", unit="us"), pa.timestamp("us"))


def build_flight_list():
    parts = []
    for f in sorted((RAW / "flight_list").glob("flight_list_*.parquet")):
        names = pq.read_schema(f).names
        t = pq.read_table(f, columns=[c for c in FL_COLS if c in names])
        t = t.set_column(0, "id", u64(t["id"]))
        for c in ("dof", "first_seen", "last_seen"):
            t = t.set_column(t.schema.get_field_index(c), c, ts(t[c]))
        m = pc.or_(pc.is_in(t["adep"], value_set=pa.array(OURS)), pc.is_in(t["ades"], value_set=pa.array(OURS)))
        t = t.filter(pc.fill_null(m, False))
        parts.append(t.append_column("src_month", pa.array([f.stem[-6:]] * t.num_rows)))
    t = pa.concat_tables(parts, promote_options="permissive")
    pq.write_table(t, OUT / "flight_list_ours.parquet")
    return t.num_rows


def parse_info(df):
    s = df.pop("info").fillna("")
    for col, rx in [("osm_id", r'"osm_id":(\d+)'), ("osm_aeroway", r'"osm_aeroway":"([^"]*)"'),
                    ("osm_ref", r'"osm_ref":"([^"]*)"'), ("apt", r'"osm_airport":"([^"]*)"'),
                    ("in_use_s", r'"opdi_time_in_use_s":(\d+)'), ("icao24", r'"osn_icao24":"([^"]*)"'),
                    ("cs", r'"osn_flight_id":"([^"]*)"')]:
        df[col] = s.str.extract(rx, expand=False)
    df["osm_id"] = pd.to_numeric(df.osm_id, errors="coerce").astype("Int64")
    df["in_use_s"] = pd.to_numeric(df.in_use_s, errors="coerce").astype("Int64")
    df["cs"] = df.cs.str.strip()
    df.loc[df.osm_ref == "nan", "osm_ref"] = pd.NA
    return df


def extract(ev_path, ms_path, out_path):
    ids = pq.read_table(OUT / "flight_list_ours.parquet", columns=["id"])["id"]
    pf = pq.ParquetFile(ev_path)
    keep = []
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=["id", "flight_id", "type", "event_time", "longitude", "latitude",
                                          "altitude", "info"])
        gnd = pc.fill_null(pc.match_substring_regex(t["info"], APT_RX), False)
        air = pc.and_(pc.is_in(t["type"], value_set=pa.array(AIR)), pc.is_in(u64(t["flight_id"]), value_set=ids))
        t = t.filter(pc.or_(gnd, air))
        t = t.set_column(0, "id", u64(t["id"])).set_column(1, "flight_id", u64(t["flight_id"]))
        t = t.set_column(3, "event_time", ts(t["event_time"]))
        keep.append(t)
    t = pa.concat_tables(keep)
    eids = t["id"]
    df = parse_info(t.to_pandas())
    df["event_time"] = df.event_time.dt.tz_localize("UTC")
    mf = pq.ParquetFile(ms_path)
    if pa.types.is_integer(mf.schema_arrow.field("event_id").type):
        meas = []
        for i in range(mf.metadata.num_row_groups):
            m = mf.read_row_group(i, columns=["event_id", "type", "value"])
            m = m.set_column(0, "event_id", u64(m["event_id"]))
            meas.append(m.filter(pc.is_in(m["event_id"], value_set=eids)))
        m = pa.concat_tables(meas).to_pandas()
        m = m.pivot_table(index="event_id", columns="type", values="value", aggfunc="first")
        m = m.rename(columns={"Distance flown (NM)": "meas_dist_nm", "Time Passed (s)": "meas_time_s"})
        df = df.merge(m, left_on="id", right_index=True, how="left")
    else:
        # older vintages key measurements as 'YYYYMMDD_<sequence>' while their events carry xxhash ids: no join exists
        df["meas_dist_nm"] = float("nan"); df["meas_time_s"] = float("nan")
    df.to_parquet(out_path, index=False)
    return out_path.name, len(df), int(df.apt.notna().sum()), float(df.meas_time_s.notna().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    print("flight_list_ours rows:", build_flight_list(), flush=True)
    jobs = []
    for ev in sorted((RAW / "flight_events").glob("flight_events_*.parquet")):
        span = ev.stem[len("flight_events_"):]
        ms = RAW / "measurements" / f"measurements_{span}.parquet"
        out = OUT / f"ev_{span}.parquet"
        if ms.exists() and (a.force or not out.exists()):
            jobs.append((ev, ms, out))
    print(f"{len(jobs)} periods to extract", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(extract, *j) for j in jobs]
        for f in as_completed(futs):
            name, n, ng, mj = f.result()
            print(f"{name}: {n:,} events, {ng:,} ground, measurements joined {mj:.1%}", flush=True)
    print("EXIT", flush=True)


if __name__ == "__main__":
    main()
