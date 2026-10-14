#!/bin/bash
# usage: tools/adsb/run_step.sh <name> <script> [args...]
# Runs $PY scripts/<script> from the repository root with <= 6 threads, logs to $PRC_WORK/logs/adsb/<name>.log and
# appends a timing row to $PRC_WORK/logs/adsb/timings.tsv. PRC_ADSB_DAYS (a frozen day list) is passed through.
name=$1; shift
PY=${PY:-py -3.14}
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_MAX_THREADS=6 POLARS_MAX_THREADS=6 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/../.."
LOG=${PRC_WORK:-.}/logs/adsb; mkdir -p "$LOG"
s=$(date '+%F %T'); s0=$(date +%s)
echo "[$s] START $PY scripts/$* (PRC_ADSB_DAYS=${PRC_ADSB_DAYS:-unset})" > "$LOG/$name.log"
$PY scripts/"$@" >> "$LOG/$name.log" 2>&1; rc=$?
e=$(date '+%F %T'); e0=$(date +%s)
echo "[$e] END rc=$rc runtime=$((e0-s0))s" >> "$LOG/$name.log"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$PY scripts/$*" "${PRC_ADSB_DAYS:-}" "$s" "$e" "$((e0-s0))" "$rc" >> "$LOG/timings.tsv"
exit $rc
