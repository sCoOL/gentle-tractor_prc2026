# Reproducing the final file

The final file is `runs/clean_chain_final/A15LXRHDS/gentle-tractor_final_v68_A15LXRHDS.parquet`: 670,790 rows (every
departure of the final ranking file), in the row order of the challenge's `submitting.parquet`, all finite, minimum
30 s (maximum 102,327.7 s, mean 1,009.65 s; sha256 `957695e0cced359ae70c8afbaadff67326eb577552b8ad777be55ecd7ff87541`).
It is the final chain's file `A15` (§6; sha256 `560fed0ac0233d0b0d83ebaf66b54f503d8598953cec8a1a813f522fcc70590f`)
followed by the five post-steps of §7. The method is in `docs/METHOD.md`; the inputs that are read but not rebuilt are
in `docs/FROZEN_ARTEFACTS.md`.

Bit-for-bit identity is not expected on another machine (LightGBM thread-count rounding, hash-seed tie-breaks in the
ADS-B callsign vote, revised downloads); the per-step row counts below are the comparison points.

## 0. Environment and locations

- Python 3.14 with `requirements.txt` (`pip install -r requirements.txt`); a CUDA GPU for the neural members (stage 3).
  Commands are run from the repository root; `py -3.14` on Windows, `python3` elsewhere (`PY=python3` for the shell
  drivers).
- Hardware of the original run: Windows 11, Intel Core i5-13600K, 31.7 GB RAM, NVIDIA GeForce RTX 4080 SUPER (16 GB).
  Heavy steps ran with 4–6 threads each and waited for ≥ 8 GB free memory (GPU members: ≥ 9 GB free GPU memory,
  ≥ 17 GB free RAM).
- Every location is set in `scripts/prc_config.py` (print them with `py -3.14 scripts/prc_config.py`):

| variable | default | holds |
|---|---|---|
| `PRC_WORK` | the repository root | `data/`, `runs/`, `submissions_rebuild/`, `logs/` |
| `PRC_DATA` | `$PRC_WORK/data` | challenge files + derived tables |
| `PRC_EXTERNAL` | `$PRC_WORK/external` | METAR, X-Plane, OurAirports, EUROCONTROL, aircraft files |
| `PRC_ADSB_DIR` | `$PRC_WORK/adsb` | derived ADS-B tables and per-day caches |
| `PRC_ADSB_POINTS` | `$PRC_ADSB_DIR` | adsb.lol downloads (`raw/`) and `points_<date>.parquet` (≈ 1.5 TB in all for 487 days) |
| `PRC_OPDI_DIR` | `$PRC_WORK/opdi` | derived OPDI tables |
| `PRC_OPDI_RAW` | `$PRC_OPDI_DIR` | OPDI downloads (≈ 38 GB) |
| `PRC_FROZEN` | `$PRC_WORK/frozen` | the frozen artefacts (`ref_layers/`, `ref_main/`, `ref_adsb/`; ≈ 2.5 GB) |
| `PRC_DATA_OLD` | `$PRC_WORK/data_old` | rank tables of the earlier Jan/Jul export (member reproduction statistics only) |

## 1. Data

1. Challenge files in `PRC_DATA`: the twelve `training_2025-MM-01_*.parquet` (the December file is
   `training_2025-12-01_2026-01-01.parquet`; it is also the December 2025 look-back of the rank side), the final
   ranking file as `ranking.parquet` and the final submission template as `submitting.parquet`.
2. External sources: `py -3.14 scripts/fetch_external.py` (X-Plane Gateway packs, METAR to 2026-07, OurAirports,
   EUROCONTROL workbooks and the daily `all_pre_departure_delays_{2025,2026}.csv` of the post-steps, whose sha256 it
   checks against the copies used), then `scripts/20_ext_layout.py`, `20_ext_weather.py`, `20_ext_calendar.py`,
   `20_ext_aircraft.py`, `20_ext_eurocontrol.py`, `stand_type.py`.
3. The frozen artefacts in `PRC_FROZEN` (`docs/FROZEN_ARTEFACTS.md`); copy `PRC_FROZEN/ref_adsb/stands.parquet` to
   `PRC_ADSB_DIR/stands.parquet`.

## 2. Stage 1: base and feature tables (≈ 55 min)

| step | command | runtime |
|---|---|---:|
| base | `py -3.14 scripts/10_build_base.py --force` | 148 s |
| detectors | `py -3.14 scripts/30_detect.py --force --stage all` | 1,187 s |
| features v1 | `py -3.14 scripts/40_assemble.py --version 1 --force` | 57 s |
| encodings | `py -3.14 scripts/35_encode.py --force` | 321 s |
| features v2 | `py -3.14 scripts/40_assemble.py --version 2 --force` | 224 s |
| features v3 | `py -3.14 scripts/36_build_nested.py --version 3 --parent-version 2 --force` | 976 s |
| v4 stand timeline | `py -3.14 scripts/37_build_v4_stand.py` | 85 s |
| v5 ops context | `py -3.14 scripts/38_build_v5_context.py` | 104 s |
| v6 queue | `py -3.14 scripts/39_build_v6_queue.py` | 120 s |
| v7 NM backfill | `py -3.14 scripts/41_build_v7_backfill.py` | 75 s |
| NM backfill links | `py -3.14 scripts/250_nm_backfill.py` | 42 s |

The original run built these tables first without the December look-back and then re-ran the rank side only with
`py -3.14 scripts/42_rank_continuity.py` (28 min, of which 16 min waiting for memory); the scripts above already
contain the continuity rule, so one pass gives the same rank tables (up to LightGBM refit noise of ≤ 6.5e-4 in the
copy probabilities; the original kept the first pass's probabilities on rows whose inputs did not change).

## 3. Stage 2: ADS-B and OPDI tables

1. Download and extract adsb.lol (resumable; 7–15 min of extraction per day, two at a time):
   `py -3.14 scripts/119_adsb_download.py 2025-01-01:2025-12-31 2026-01-01:2026-03-01 2026-05-31:2026-07-31 --preferred --workers 4 --extract`
2. OPDI: `py -3.14 scripts/257_opdi_download.py`, `258_opdi_extract.py`, `259_opdi_flights.py`.
3. Everything derived, for every complete day (487 days):
   `bash tools/adsb/chain_final.sh fin` — lane 1: `121 events`, `121 quality`, `tools/adsb/quality_splice.py` (keeps
   the reference 2025 quality rows), `121 features`, `255 features`, `121 neighbours`; lane 2: `122 geometry`,
   `124 stops`, `232`, `255 compare`, `adsb_push_phases run / collect`, `213`, `231`, `260_opdi_link.py`.
   The original run took about 2.5 h of derived-table computation on 6 workers (logs per step in `logs/adsb/`).
4. Checks: `py -3.14 tools/adsb/coverage.py` (share of departures with a v1 / v2 match, push-back, OPDI link by month
   and airport; all airports: v2 match 58 / 54 / 54 / 55 % for Jan / Feb / Jun / Jul), `py -3.14 tools/adsb/edge_check.py`.

## 4. Stage 3: v45 for the final rows (members)

`bash tools/members/run_a15.sh` runs, one at a time: the A27 block (`a27_members.py m3rpa / detector / rd / gate`),
the LightGBM members without ADS-B and the BLIND twins (seeds 41–55), the 23 networks, the 26 ADS-B members of
variant A (trained on `PRC_FROZEN/ref_main/adsb_snapshot_0926`, scoring the final rows on `PRC_ADSB_DIR`), the AB twins
(seeds 41–55), the re-prediction of every input from its saved model on the final rank tables (`rescore.py --variant A`)
and the composition (`recompose_v45.py rebuild ... --tag A15 --write`) →
`$PRC_WORK/submissions_rebuild/v45_A15_<stamp>.parquet` + `_detail.parquet`.

Runtimes of the original run: the ADS-B LightGBM members 20–60 min per group (≈ 4 h in all), the refitted rd members
1.8 h, every network 6–10 min on the GPU (≈ 4 h for 30), twins 26–27 min per 12 seeds, re-prediction ≈ 45–60 min,
composition 30 s. The member tools also score the earlier Jan/Jul export and compare every refit with the reference
predictions (`PRC_DATA_OLD`, `PRC_FROZEN/ref_main/models/member_cache*`); those numbers are statistics only.

## 5. Stage 4: layer frames (≈ 1 h)

```bash
py -3.14 tools/layers/build_frames.py static rows66 ovcache f266 f269 lirfbase item05 item02 longrows
py -3.14 tools/layers/build_frames.py motion item04 item11
```

`static` copies the frozen 2025-side files into `runs/`; every other frame is rebuilt for the 670,790 rank rows and its
2025 rows are replaced by the frozen reference rows (diagnostics in `logs/layers/record.json`). `motion` extracts the
sustained-motion features of the 120 departure days of 2026 (≈ 6 min on 4 workers).

## 6. Stage 5: the final chain (≈ 6–8 min)

```bash
py -3.14 tools/final_chain.py --variant A15 --v45 submissions_rebuild/v45_A15_<stamp>.parquet \
    --a27-detail runs/models/final_A/R_A27/rank_detail.parquet --l3-months 6,7 all
py -3.14 tools/layers/final_report.py A15
```

Outputs in `runs/clean_chain_final/A15/`: `cv49 … cv68.parquet`, `chain_log.json` (per step: rows moved, sum of squared
moves), `splice_record.json`, `packages/` (one `build.json` per layer) and `gentle-tractor_final_v68_A15.parquet`.
The run checks: the v45 file covers the 670,790 template ids, finite, ≥ 30 s; step 49 asserts that v45 equals the
detail's v45 component on every motion-eligible row (172,484); step 57 asserts that the rebuilt unmatched lane
reproduces the detail's prediction; step 68 asserts that the gate's 2025 inputs equal those it was evaluated on; the
export asserts 670,790 unique ids in template order, finite, ≥ 30 s.

Rows moved per step in the original run (compare with `chain_log.json`):

| step | 49 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 63 | 66 | 67 | 68 gate | manual rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rows | 172,481 | 57 | 44 | 39,245 | 5,651 | 610,232 | 661,623 | 482 | 426 | 316,046 | 221,488 | 95,153 | 15 |

### The manual rows (step 68, last)

`config/manual_rows.csv` holds the 15 LIRF rows of METHOD.md §9 (columns `MVT_ID_mvt`, `flight`, `rule`, `value_s`;
values as in the final file, including each row's re-exported take-off time). Switches:

- `--manual-rules 8-13h,ita_overnight` (default: both) — drop a rule from the list to leave its rows at the chain's
  value;
- `--manual-rows off` — no manual rows at all.
- `8-13h`: 14 rows set to D = take-off − schedule. `ita_overnight`: ITY672 2026-07-09 set to the full date-bug label
  87,420.5 s (the value rule R4 of step 5 also gives this row's side; METHOD.md §9).

The table is not in the git repository (it lists flight ids of the challenge data); it is in the frozen download
(`docs/FROZEN_ARTEFACTS.md`, folder `config/`). Without it the chain stops at step 68 unless `--manual-rows off` is given.

## 7. Stage 6: the post-steps (≈ 6 min)

```bash
py -3.14 tools/last_day/run_last_day.py               # A15 -> A15L -> A15LX -> A15LXR -> A15LXRH -> A15LXRHDS, then the output check
py -3.14 tools/last_day/run_last_day.py --from c18    # resume at a step after a failure
```

Inputs: the chain run `runs/clean_chain_final/A15/` (its final file, `manual_rows_applied.parquet`,
`v54_gate/table.parquet`, `lirf_row_audit/lirf_rows.parquet`), `runs/overnight_20261008/frame_cache.parquet`
(stage 4), `PRC_ADSB_DIR/push_phases.parquet` (stage 2), `PRC_EXTERNAL/eurocontrol/all_pre_departure_delays_{2025,2026}.csv`
(stage 1), the turnaround links `runs/v51_turnaround/links.parquet` (stage 2, `scripts/231`) and the frozen 2025 proxy `PRC_FROZEN/ref_layers/runs/leads_20261008/rows.parquet`. Fits, out-of-fold frames
and logs go to `runs/last_day/`; each step's file to `runs/clean_chain_final/<tag>/`.

| order | step | script | runtime | output / check |
|---:|---|---|---:|---|
| 1 | identity check | `c9_step1_identity.py` | 3 s | `c9_step1_identity.json` (per-airport correlation of 60 · DLY_ALL_PRE_2 with the 2025 labels' delay sums) |
| 2 | 8–13 h rule rows | `c0_long_delay_rule.py` | 1 s | `c0_rows.parquet` (16 final rows in scope, 2 not set by the manual step) |
| 3 | cap-at-D counts | `c3_cap_at_D.py` | 8 s | `c3_cap_at_D.json` (2025: 8 rows, all labels at D) |
| 4 | **A15L** | `build_A15L.py` | 2 s | 9 rows move vs A15 |
| 5 | daily layer version 1 (reference) | `c9_daily_layer.py` | 107 s | `c9_oof_2025.parquet` |
| 6 | stack, 2025 out of fold | `c15_combined.py --tag x` | 34 s | `c15_oof_x.parquet` |
| 7 | **A15LX** | `build_A15LX.py` | 15 s | 10 airport-months scaled by the guard; 570,912 rows differ from A15 |
| 8 | stack + enumeration, 2025 out of fold | `c16_stack_recentred.py --tag x` | 41 s | `c16_oof_x.parquet`; asserts the port equals `c13` and the stack equals `c15_oof_x` |
| 9 | check | `build_A15LXR.py --idea4 0` | 12 s | must reproduce A15LX exactly |
| 10 | **A15LXR** | `build_A15LXR.py --idea4 1` | 13 s | no cell scaled by the increment guard |
| 11 | crossers, 2025 out of fold | `c18_crossers_hardened.py --tag h` | 82 s | `c18_oof_h.parquet`; asserts its base equals `c16_oof_x` |
| 12 | **A15LXRH** | `build_A15LXRH.py` | 22 s | asserts the recomputed stack + enumeration equals A15LXR outside the guarded cells; 827 rows move by > 1 s vs A15LXR; 571,220 rows differ from A15 |
| 13 | LIRF rules, 2025 counts | `c19_lirf_datebug_rules.py` | 4 s | `c19_lirf_rules.json` (in-sample counts per rule: R0 1, R1 15, R2 5, R3 2, R4 9, R5 2, R6 8, no error) |
| 14 | **A15LXRHDS** | `build_A15LXRHDS.py` | 2 s | 8 LIRF rows move vs A15LXRH (6 copies, 2 date bugs); manual rows unchanged |

Every build writes through `build_A15L.write`, which asserts 670,790 rows in the template order of
`data/submitting.parquet`, unique ids, all finite, minimum ≥ 30 s, and the rows of the manual step unchanged; the
runner then re-reads the final file and prints its row count, order, minimum, maximum, mean and sha256.

sha256 of each file of the original run (from the A15 file above):

| file | sha256 |
|---|---|
| `A15L` | `de2e58c2cb37f0d86a306cf717b9ecacd84a182ac8f0d891210e7db6eee7fdd4` |
| `A15LX` | `e337fb0fcd6c2a71d098d62a85caebfbaffecb12c7c3faf8140d2932bf058400` |
| `A15LXR` | `c269e432d6dfb28e215f0a875cf9ad5793023c8ab970a5fe58f0c56c3169adde` |
| `A15LXRH` | `609f159afbfecdcdd318286385201dce11dad607935b63206a70ffc61a84a3a2` |
| `A15LXRHDS` (final) | `957695e0cced359ae70c8afbaadff67326eb577552b8ad777be55ecd7ff87541` |

The hashes are those of a full run of `run_last_day.py` from the A15 file above with the stage-2 push-phase table
`PRC_ADSB_DIR/push_phases.parquet` of the same run. Run from the A15 file without RYRR388G in the 8–13 h rule
(`de09657e…`), the same code gives A15L `8e2ebc62…`, A15LX `46b4b1d7…` and A15LXR `f3ca8d7a…`, the files of the
original run; its A15LXRH differs from the original `6ddfc1c8…` on 99 rows by at most 83 s (Σ 0.04 M s²), because the
original run read an earlier push-phase table.

The post-steps are deterministic (no random seed, no LightGBM): given the same A15 file and the same input files, the
files are reproduced bit for bit (two runs from the same A15 file and inputs give the same five sha256). A rebuilt A15 that differs by float rounding gives a final file that differs by
about as much. The EUROCONTROL daily files can be revised by their publisher; `scripts/fetch_external.py` reports when
a download differs from the copies used.

## 8. Tests

`py -3.14 -m pytest tests -q` (unit tests that need no data).
