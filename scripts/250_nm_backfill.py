"""NM flight-plan times for NM-unmatched departures, taken from the destination airport's arrival record.

  py -3.14 scripts/250_nm_backfill.py

How the dataset builder matched (2026-09-25): a departure row got its `_flt` columns only
when |BLOCK_TIME - LOBT_flt| <= ~3,606 s. A departure left unmatched may still have its NM flight in the data: when it
flew to another study airport, that airport's arrival row of the same flight may have matched, and an arrival's
AOBT_3/LOBT/EOBT_1/IOBT describe the departure airport's off-block (DEP and ARR rows sharing an NM flight id carry the
identical AOBT_3 on 315,234 of 315,273 pairs in 2025).

nm_backfill.py (09-17) linked on an identical flight number only. The two airports write it the same way on half of the
NM-identified pairs (LH992 vs DLH992, KLM vs KL, SWR vs LX, alphanumeric callsigns), so this links in four tiers, each
on the same route with the landing 20 min - 8 h after the departure's take-off:
  1 same_number   identical FLIGHT_mvt
  2 nm_callsign   the arrival's NM callsign equals the departure's flight number
  3 mapped_prefix same numeric suffix, prefix pair seen >= 5 times on NM-identified pairs
  4 route_time    same airline (or mapped prefix) and aircraft type, landing within 15 min of the route's median
                  airborne time, exactly one candidate (99.92 % precise on NM-identified 2025 pairs)
The lowest tier wins, an arrival serves one departure, and an arrival whose NM flight is already held by a matched
departure row is skipped (that NM flight belongs to the other row). Rank-available inputs only: take-off, route,
flight numbers, types and the arrival rows; no departure block time is read.

Output runs/nm_backfill/backfill.parquet: MVT_ID_mvt (the departure), bf_arr_mvt, bf_tier, bf_nm_flight_id,
bf_callsign, bf_lobt, bf_eobt1, bf_iobt, bf_aobt3, bf_arvt3. audit.json: per-tier counts and each tier's precision
when run on NM-matched departures (truth = the shared NM flight id).
"""
import json

import duckdb

import common

OUT = common.RUNS / 'nm_backfill'
STUDY = ", ".join(f"'{a}'" for a in common.APTS)

MOVES = """CREATE TABLE m AS SELECT MVT_ID_mvt, PHASE_mvt, FLIGHT_ID_mvt, FLIGHT_mvt, CALLSIGN_flt, ADEP_mvt, ADES_mvt, AIRCRAFT_TYPE_mvt,
    epoch(MVT_TIME_UTC_mvt) AS tm, LOBT_flt, EOBT_1_flt, IOBT_flt, AOBT_3_flt, ARVT_3_flt, split,
    regexp_extract(FLIGHT_mvt, '^([A-Z]{3}|[A-Z0-9]{2})', 1) AS pre,
    ltrim(substr(FLIGHT_mvt, length(regexp_extract(FLIGHT_mvt, '^([A-Z]{3}|[A-Z0-9]{2})', 1)) + 1), '0') AS suf
    FROM (SELECT *, 'train' AS split FROM read_parquet(?, union_by_name=true)
          UNION ALL BY NAME SELECT *, 'rank' AS split FROM read_parquet(?))"""

# every candidate arrival for a departure set `dp`, with its tier; `excl` removes arrivals whose NM flight a matched departure holds
CANDIDATES = """
WITH dp AS ({dep}),
ar AS (SELECT * FROM m WHERE PHASE_mvt = 'ARR' {excl}),
c AS (
  SELECT dp.MVT_ID_mvt, dp.FLIGHT_ID_mvt AS dep_fid, ar.MVT_ID_mvt AS arr_mvt, ar.FLIGHT_ID_mvt AS arr_fid, ar.tm AS land,
    CASE WHEN ar.FLIGHT_mvt = dp.FLIGHT_mvt THEN 1
         WHEN ar.CALLSIGN_flt = dp.FLIGHT_mvt THEN 2
         WHEN pm.n IS NOT NULL AND ar.suf = dp.suf AND ar.suf <> '' THEN 3
         WHEN (ar.pre = dp.pre OR pm.n IS NOT NULL) AND ar.AIRCRAFT_TYPE_mvt = dp.AIRCRAFT_TYPE_mvt
              AND abs(ar.tm - dp.tm - air.med) <= 900 THEN 4 END AS tier
  FROM dp JOIN ar ON ar.ADEP_mvt = dp.ADEP_mvt AND ar.ADES_mvt = dp.ADES_mvt AND ar.tm - dp.tm BETWEEN 1200 AND 28800
  LEFT JOIN pmap pm ON pm.dp = dp.pre AND pm.ap = ar.pre
  LEFT JOIN air ON air.a = dp.ADEP_mvt AND air.b = dp.ADES_mvt
)
SELECT * FROM c WHERE tier IS NOT NULL"""


def choose(con, cand: str) -> None:
    """Lowest tier, then earliest landing; tier 4 only with a single candidate; each arrival used once."""
    con.execute(f"""CREATE OR REPLACE TABLE pick AS
        WITH c AS (SELECT *, count(*) FILTER (WHERE tier = 4) OVER (PARTITION BY MVT_ID_mvt) AS n4 FROM ({cand})),
        c2 AS (SELECT * FROM c WHERE tier < 4 OR n4 = 1),
        b AS (SELECT * FROM c2 QUALIFY row_number() OVER (PARTITION BY MVT_ID_mvt ORDER BY tier, land) = 1)
        SELECT * FROM b QUALIFY row_number() OVER (PARTITION BY arr_mvt ORDER BY tier, land) = 1""")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute("SET threads TO 6")   # final-file change: shared 20-core machine (was DuckDB default = all cores)
    con.execute(MOVES, [[str(p) for p in sorted(common.DATA.glob('training_2025-*.parquet'))], str(common.DATA / 'ranking.parquet')])
    # label-free learning from NM-identified pairs (a DEP and an ARR row sharing an NM flight id)
    con.execute("""CREATE TABLE pairs AS SELECT d.pre dp, a.pre ap, d.suf ds, a.suf asf, d.ADEP_mvt ad, d.ADES_mvt aa, a.tm - d.tm AS airborne
                   FROM m d JOIN m a ON a.FLIGHT_ID_mvt = d.FLIGHT_ID_mvt AND a.PHASE_mvt = 'ARR' WHERE d.PHASE_mvt = 'DEP'""")
    con.execute("CREATE TABLE pmap AS SELECT dp, ap, count(*) n FROM pairs WHERE ds = asf AND dp <> ap GROUP BY ALL HAVING count(*) >= 5")
    con.execute("CREATE TABLE air AS SELECT ad a, aa b, median(airborne) med FROM pairs GROUP BY ALL")
    held = "AND (FLIGHT_ID_mvt IS NULL OR FLIGHT_ID_mvt NOT IN (SELECT FLIGHT_ID_mvt FROM m WHERE PHASE_mvt = 'DEP' AND FLIGHT_ID_mvt IS NOT NULL))"

    # audit: run the same rules on NM-matched departures, where the true arrival is the one sharing the NM flight id
    choose(con, CANDIDATES.format(dep=f"SELECT * FROM m WHERE PHASE_mvt = 'DEP' AND AOBT_3_flt IS NOT NULL AND ADES_mvt IN ({STUDY})", excl=''))
    aud = con.execute("""SELECT tier, count(*) n, avg((arr_fid = dep_fid)::int) AS prec FROM pick GROUP BY 1 ORDER BY 1""").fetchall()
    audit = {'precision_on_nm_matched_departures': {f'tier{t}': {'n': n, 'precision': round(p, 5)} for t, n, p in aud},
             'prefix_pairs': con.execute("SELECT count(*) FROM pmap").fetchone()[0]}

    choose(con, CANDIDATES.format(dep=f"SELECT * FROM m WHERE PHASE_mvt = 'DEP' AND AOBT_3_flt IS NULL AND ADES_mvt IN ({STUDY})", excl=held))
    names = "CASE p.tier WHEN 1 THEN 'same_number' WHEN 2 THEN 'nm_callsign' WHEN 3 THEN 'mapped_prefix' ELSE 'route_time' END"
    con.execute(f"""CREATE TABLE out AS SELECT p.MVT_ID_mvt, p.arr_mvt AS bf_arr_mvt, {names} AS bf_tier, a.FLIGHT_ID_mvt AS bf_nm_flight_id,
        a.CALLSIGN_flt AS bf_callsign, a.LOBT_flt AS bf_lobt, a.EOBT_1_flt AS bf_eobt1, a.IOBT_flt AS bf_iobt, a.AOBT_3_flt AS bf_aobt3,
        a.ARVT_3_flt AS bf_arvt3, d.split
        FROM pick p JOIN m a ON a.MVT_ID_mvt = p.arr_mvt JOIN m d ON d.MVT_ID_mvt = p.MVT_ID_mvt""")
    con.execute(f"COPY (SELECT * EXCLUDE (split) FROM out ORDER BY MVT_ID_mvt) TO '{(OUT / 'backfill.parquet').as_posix()}' (FORMAT parquet)")
    cnt = con.execute("""SELECT split, bf_tier, count(*) n, count(bf_aobt3) with_aobt3 FROM out GROUP BY ALL ORDER BY ALL""").fetchall()
    audit['unmatched_departures_linked'] = [dict(split=s, tier=t, n=n, with_aobt3=k) for s, t, n, k in cnt]
    audit['unmatched_departures_to_study_airport'] = dict(con.execute(f"""SELECT split, count(*) FROM m WHERE PHASE_mvt = 'DEP'
        AND AOBT_3_flt IS NULL AND ADES_mvt IN ({STUDY}) GROUP BY 1""").fetchall())
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
