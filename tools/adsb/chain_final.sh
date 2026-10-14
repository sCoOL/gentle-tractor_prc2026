#!/bin/bash
# The ADS-B / OPDI derived tables for every day that has an extracted points file (docs/REPRODUCE.md, stage 2).
# usage: tools/adsb/chain_final.sh <prefix> [<day list file>]     (no list = every complete day, 2025-01-01 on)
# lane 1: 121 quality -> reference splice -> 121 features -> 255 features -> 121 neighbours
# lane 2: 122 geometry -> 124 stops -> 232 times -> 255 compare -> push run/collect -> 213 -> 231 -> 260
p=$1
[ -n "$2" ] && export PRC_ADSB_DAYS=$2
PY=${PY:-py -3.14}
cd "$(dirname "$0")/../.."
R=tools/adsb/run_step.sh
LOG=${PRC_WORK:-.}/logs/adsb; mkdir -p "$LOG"
bash $R ${p}_121_events 121_adsb_events.py events --workers 6 || exit 1
(
  bash $R ${p}_121_quality 121_adsb_events.py quality --workers 6 &&
  $PY tools/adsb/quality_splice.py > "$LOG/${p}_quality_splice.log" 2>&1 &&
  bash $R ${p}_121_features 121_adsb_events.py features --workers 6 &&
  bash $R ${p}_255_features 255_adsb_match_v2.py features --workers 6 &&
  bash $R ${p}_121_neighbours 121_adsb_events.py neighbours --workers 6 &&
  echo "lane1 $p done"
) > "$LOG/${p}_lane1.out" 2>&1 &
l1=$!
(
  bash $R ${p}_122_geometry 122_adsb_shadow.py geometry --workers 6 &&
  bash $R ${p}_124_stops 124_unmatched_adsb.py stops --workers 6 &&
  bash $R ${p}_232_times 232_adsb_movement_times.py --workers 6 &&
  bash $R ${p}_255_compare 255_adsb_match_v2.py compare --workers 6 &&
  bash $R ${p}_push_run adsb_push_phases.py run --workers 6 &&
  bash $R ${p}_push_collect adsb_push_phases.py collect &&
  bash $R ${p}_213_turnaround 213_turnaround_features.py --workers 6 &&
  bash $R ${p}_231_links 231_turnaround_links.py &&
  bash $R ${p}_260_opdi_link 260_opdi_link.py &&
  echo "lane2 $p done"
) > "$LOG/${p}_lane2.out" 2>&1 &
l2=$!
wait $l1; r1=$?; wait $l2; r2=$?
echo "final $p: lane1 rc=$r1 lane2 rc=$r2"
