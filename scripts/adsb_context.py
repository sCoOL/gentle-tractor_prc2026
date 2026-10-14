"""ADS-B off-block context for the models (built by 121_adsb_events.py).

One row per departure that adsb.lol saw taking off, keyed by MVT_ID_mvt, for 2025 and the 2026 ranking days alike.
Every column is observation, never a label: the feature table on disk also carries `y`, `t_blk`, `t_mvt`, `date` and
`APT` for `121 check`, and `table()` drops them — `t_blk` IS the off-block the target is built from.

  adsb_taxi_leave        take-off minus the last position at the stand = the ADS-B off-block estimate
  adsb_taxi_leave_hi     the same from the first position that had moved away (the other side of the bracket)
  adsb_taxi_first        take-off minus the first sighting; a LOWER bound on the taxi when the row is censored
  adsb_taxi_clear        take-off minus the first position > 80 m from every stand
  adsb_at_stand          the aircraft was first seen standing at a stand (the estimate is trustworthy)
  adsb_censored          first seen > 80 m from every stand: it had already left when ADS-B picked it up
  adsb_after_landing     the ground phase began with a landing (the transponder stayed on over the turnaround)
  adsb_wait / adsb_gap   seconds between first sighting and leaving / width of the leave bracket (uncertainty)
  adsb_dstand            metres from the first position to the nearest known stand
  adsb_gs0, adsb_nground ground speed at the first point, number of ground points
  adsb_*_minus_a3        leaving / first sighting minus NM's AOBT_3 (agreement with the independent anchor)
  adsb_how               1 = matched by callsign, 2 = matched by take-off time only
  adsb_match_gap         seconds between the ADS-B take-off and the airport's take-off time

How good ADS-B is at this airport in this month, measured on ARRIVALS, whose in-block times are given in both years
(only departure block times are masked). It lets a model trust ADS-B where it is good now rather than where it was good
last year: EGLL arrivals land within 60 s of their recorded in-block 13 % of the time on 2025-07-15 but 68 % on
2026-07-20.

  adsb_q_within60        share of that airport-month's arrivals whose ADS-B stop is within 60 s of the record
  adsb_q_med             median signed offset, adsb_q_mad the median absolute one
  adsb_q_cov             share of arrivals for which ADS-B saw a stop at all
  adsb_q_n               how many arrivals the share rests on

`nbr_table()` (`121 neighbours`) is the second file: what the OTHER aircraft on the surface were doing around each
departure. The columns above need this row's own aircraft to have been seen at its stand (25 % of the 2026 rows);
these reach every row at an airport with ground coverage, and each window has the row's own aircraft taken out of it.

  adsb_nb_taxi_c60/_prev60     mean measured taxi of the neighbours seen leaving within +-30 min / the hour before
  adsb_nb_taxi_n               how many aircraft that mean rests on
  adsb_nb_queue_c60            mean measured time from clearing the stands to take-off (the runway queue)
  adsb_nb_surface              aircraft measured between stand-leave and take-off at this take-off instant
  adsb_nb_tko_c60              ADS-B take-offs within +-30 min; adsb_nb_day_n the day's ground phases at the airport
  adsb_nb_bias_c60/_n/_prev60  mean (ADS-B off-block - AOBT_3) of the neighbours: how far NM's anchor is running now
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc_config  # noqa: E402

PATH = prc_config.ADSB_DIR / 'adsb_features.parquet'
COLUMNS = ['adsb_how', 'adsb_at_stand', 'adsb_censored', 'adsb_after_landing', 'adsb_taxi_leave',
           'adsb_taxi_leave_hi', 'adsb_taxi_first', 'adsb_taxi_clear', 'adsb_wait', 'adsb_gap', 'adsb_dstand',
           'adsb_gs0', 'adsb_nground', 'adsb_leave_minus_a3', 'adsb_first_minus_a3', 'adsb_match_gap',
           'adsb_q_within60', 'adsb_q_med', 'adsb_q_mad', 'adsb_q_cov', 'adsb_q_n']
_CACHE = {}


NBR_PATH = prc_config.ADSB_DIR / 'adsb_nbr.parquet'
NBR_COLUMNS = ['adsb_nb_taxi_c60', 'adsb_nb_taxi_n', 'adsb_nb_taxi_prev60', 'adsb_nb_queue_c60',
               'adsb_nb_surface', 'adsb_nb_tko_c60', 'adsb_nb_day_n',
               'adsb_nb_bias_c60', 'adsb_nb_bias_n', 'adsb_nb_bias_prev60']


GEOM_PATH = prc_config.ADSB_DIR / 'adsb_geom.parquet'
GEOM_COLUMNS = ['adsb_gm_x0', 'adsb_gm_y0', 'adsb_gm_r0', 'adsb_gm_bear0', 'adsb_gm_x1', 'adsb_gm_y1',
                'adsb_gm_dist', 'adsb_gm_disp', 'adsb_gm_wander', 'adsb_gm_span', 'adsb_gm_stops',
                'adsb_gm_stopped', 'adsb_gm_hold', 'adsb_gm_v_mean', 'adsb_gm_v_max', 'adsb_gm_v_last',
                'adsb_gm_npts']


STOPS_PATH = prc_config.ADSB_DIR / 'adsb_stops.parquet'
STOPS_COLUMNS = ['st_n', 'st_n_stand', 'st_n_off', 'st_taxi_last', 'st_taxi_first_stop', 'st_last_dwell', 'st_repark_s',
                 'st_moved_before_last', 'st_same_stand', 'st_off_longest', 'st_off_total', 'st_off_after_last']


def stops_table(path=STOPS_PATH):
    """MVT_ID_mvt + STOPS_COLUMNS (124_unmatched_adsb.py stops): the stop sequence of the ground phase. 121's off-block
    is the FIRST move away from the first sighting; a tow or a return to stand makes it hours early, and these columns
    say whether the aircraft parked at a stand again (st_taxi_last = take-off minus the LAST stand departure), for how
    long, how far it had rolled before that, and how long it stood still away from any stand (de-icing, remote hold)."""
    return _read(path, STOPS_COLUMNS)


def geom_table(path=GEOM_PATH):
    """MVT_ID_mvt + GEOM_COLUMNS (122_adsb_shadow.py): what the trace shows of the taxi we DID see - where the
    aircraft was first observed, how far it actually rolled, how long it stood still, how fast it reached the
    runway. A measurement of this flight, not a prediction about it."""
    return _read(path, GEOM_COLUMNS)


def nbr_table(path=NBR_PATH):
    """MVT_ID_mvt + NBR_COLUMNS: the surface context measured around each departure, self excluded."""
    return _read(path, NBR_COLUMNS)


def table(path=PATH):
    """MVT_ID_mvt + COLUMNS only; rows without ADS-B simply do not appear (merge left, trees read the NaNs).

    Columns the file does not have yet (the quality block needs `121 quality`) are reported and skipped rather than
    raising, so an older feature table still fits — but then the member is not the one the reported numbers used.
    """
    return _read(path, COLUMNS)


def _read(path, columns):
    key = str(path)
    if key not in _CACHE:
        import pyarrow.parquet as pq
        have = set(pq.ParquetFile(path).schema.names)
        cols = [c for c in columns if c in have]
        if len(cols) < len(columns):
            print(f'adsb_context: {path} lacks {sorted(set(columns) - have)}; fitting without them', flush=True)
        d = pd.read_parquet(path, columns=['MVT_ID_mvt'] + cols)
        _CACHE[key] = d.drop_duplicates('MVT_ID_mvt')
    return _CACHE[key]
