# gentle-tractor — PRC Data Challenge 2026: departure taxi-out time

**Our main competition pipeline intentionally excludes labels inferred from submission results and ensemble weights
fitted to those results.** Experiments that use submission feedback are documented separately under
[Experimental alternative models](#experimental-alternative-models).

Code of team **gentle-tractor** for the EUROCONTROL Performance Review Commission (PRC) Data Challenge 2026: predict
the **taxi-out time** (take-off time − actual off-block time, seconds) of every departure at ten European airports
(EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM) in **January, February, June and July 2026** (the final
ranking file, 670,790 departures), from the challenge's airport movement records and Network Manager (NM) flight
data, plus open surveillance, flight-event and weather data.

Licensed under the **GNU General Public License v3.0** (see `LICENSE`).

## Authors

Team gentle-tractor, all three of ACV – Airports Corporation of Vietnam:

- Hoang Anh Thong LE (Lê Hoàng Anh Thông) - thongle040288@gmail.com
- Dieu Huyen DO (Đỗ Diệu Huyền)
- Canh Trieu TRAN (Trần Cảnh Triệu)

## The approach in one page

1. **Labels are not all taxi times.** About 5 % of 2025 departures carry an off-block equal to the schedule (a
   "schedule copy"), a few carry an off-block on the wrong day ("date bug"), and some carry the first-filed or last
   off-block plan time. These rows dominate the squared error, so the model predicts a **mixture**: a normal-taxi
   expert N and the copy value D = take-off − schedule (and, for date bugs, D plus a day), weighted by learned copy /
   date-bug gates: `p = N + w · (D − N)`.
2. **Two lanes.** Departures that NM matched (an NM actual off-block `AOBT_3` exists) and NM-unmatched departures
   (no NM anchor, where most artefacts live — above all at LIRF) get separate experts, gates and rules.
3. **Features** (≈ 240 columns): flight / airline / stand / runway keys, local and UTC time, take-off-time deltas to
   schedule and NM times, traffic and runway-configuration context, layout taxi distances from the X-Plane Gateway
   stand graph, METAR weather, EUROCONTROL daily traffic, ATFM delay and monthly additional taxi-out time, aircraft
   dimensions, out-of-fold target encodings, the same-stand timeline, and set encoders over the airport's movement
   timeline around each departure. January + February and June + July are treated as continuous periods.
4. **ADS-B (adsb.lol open archive)**: a stand map, per-departure ground events (first sighting, first real move off the
   stand, the stop sequence, the push-back phases) and matched-flight identity. **OPDI** ground events give
   per-airport off-block corrections where their coverage is good (LSZH, EDDF).
5. **v45**, the base: a fixed composition of 63 trained members (LightGBM, MLPs, set networks, ADS-B members, copy
   classifiers, unmatched-lane twins, a LIRF identity gate) with per-cell weights fitted on 2025 out-of-fold frames.
6. **The chain after v45** (`tools/final_chain.py`): a motion expert on ADS-B ground motion, rules derived from how
   the dataset was built (NM matching window ≈ ±1 h of LOBT), OPDI corrections, gate retention, neighbour lateness,
   LIRF stand reuse and same-period context, push-back phases, step-2 turnaround links, four small overnight layers
   and a long-label gate — each a residual correction or classifier on the rows its signal is about, each validated on
   its own — then a documented step that sets 15 LIRF rows (`docs/METHOD.md` §9). This gives the chain's file `A15`.
7. **Post-steps on that file** (`tools/last_day/`, `docs/METHOD.md` §11): two fixed rules (the 8–13 h copy rule on every
   row it selects, p ≤ D for D > 2 h), then a **daily reconciliation with EUROCONTROL's daily all pre-departure delay**
   (60 · `DLY_ALL_PRE_2` per airport and day equals the sum of max(0, off-block − schedule) over that day's recorded
   off-blocks, so it is an aggregate of the masked quantity; a linear-Gaussian posterior spreads each day's residual over
   the rows that can absorb it, in three passes, on IFR rows, with a count-gap correction), a re-centred copy / held
   enumeration of the long LIRF unmatched rows against the same daily residual, and a guarded midnight-crossers step
   that uses EUROCONTROL's daily departure counts. Every 2026 move is bounded per airport-month by the size of the same
   step's 2025 out-of-fold moves. The last step decides copy or date bug for LIRF unmatched rows that take off on a
   later local day, by rules read off the 2025 rows (delay band, ITA overnight long-haul take-off clock, inbound
   turnaround), and gives each decided row its full value (`docs/METHOD.md` §11.5).
8. **Validation.** 2025 is split into six seasonal folds (A = Jan + Jul, …, F = Jun + Dec). Every parameter is chosen
   inside the training folds, and every layer must also improve **forward in time** (fit Jan–Jun → test Jul–Dec and
   back). Only label-free inputs are used on 2026 rows; nothing is fitted on 2026 rows.

The final file is `runs/clean_chain_final/A15LXRHDS/gentle-tractor_final_v68_A15LXRHDS.parquet` (670,790 rows,
template order; `docs/REPRODUCE.md` §7).

Frozen artefacts and the manual-rows table that is not in git (`config/manual_rows.csv`):
[download folder](https://vietnamairports-my.sharepoint.com/:f:/g/personal/thonglha_acv_vn/IgCzNW9H2-GZTJqxE1HPhHoVAayFZCgnU5qYnxc6zNYUjYU?e=sQpp4l) (`docs/FROZEN_ARTEFACTS.md`).

Details: `docs/METHOD.md`. Design decisions, data, features and the ideas tested and left out: `docs/DESIGN_DECISIONS.md`.
How much single rows move the RMSE, and what the leaderboard showed about them: `docs/A_FEW_ROWS.md`. How to run, stage by stage: `docs/REPRODUCE.md`. Inputs that are read but not rebuilt:
`docs/FROZEN_ARTEFACTS.md`. Data sources and licences: `docs/DATA_SOURCES.md`. Every script with its usage line:
`docs/SCRIPTS.md`.

## Experimental alternative models

The files in [Alternative Models](Alternative%20Models/) are provided **for testing**. This experiment fits ensemble
weights to previous January / July 2026 submission results, checks candidate blends against saved 2025 predictions,
and applies the frozen v85 weights to all four 2026 ranking months. These alternatives also treat **RYRR388G
(2026-07-26) as a normal taxi**, following the evidence from submission-result comparisons described in
[`docs/A_FEW_ROWS.md`](docs/A_FEW_ROWS.md#2-ryrr388g-2026-07-26-a-9-h-late-unmatched-flight-that-was-a-normal-taxi).
Unlike the main pipeline described above, these alternatives explicitly use leaderboard feedback to fit their weights
and inform this flight's treatment.

| File | Coverage | Result / status |
|---|---|---|
| [gentle-tractor_v85.parquet](Alternative%20Models/gentle-tractor_v85.parquet) | January + July 2026; 344,841 departures | Estimated RMSE **212.5166 seconds (about 212.5 seconds)**, calculated from previous submission results; not a measured official score for this file. |
| [gentle-tractor_final_Fitted.parquet](Alternative%20Models/gentle-tractor_final_Fitted.parquet) | January, February, June + July 2026; 670,790 departures | Frozen v85 weights applied to rebuilt source models and the final ranking inputs; official RMSE is unverified. |

Some original source checkpoints were unavailable, so the four-month file rebuilds those sources from their recorded
recipes. Its January / July predictions therefore differ from the original test by 7.93 seconds RMS, or 6.41 seconds
after aligning the changed take-off timestamps. These are prediction differences, not measured error increases; the
212.5-second estimate does not establish the rebuilt final file's score.

On the saved 2025 comparison, the original v85 blend improved February + June RMSE by approximately 0.41 seconds
relative to v84. These checks reused predictions and labels involved in model selection, and the rebuilt sources have
not received fresh independent full-pipeline validation. February / June 2026 predictions are included for final
evaluation, but their hidden labels are unavailable: better performance on those months or on the complete final
ranking has not been established.

## Repository layout

```
scripts/        the pipeline (numbered by stage; shared modules unnumbered); scripts/prc_config.py = every location
tools/          final_chain.py (the final chain), members/ (v45 for the final rows), layers/ (layer frames, runtime),
                adsb/ (stage-2 drivers and checks), last_day/ (the post-steps A15 -> A15LXRHDS, run_last_day.py)
runs/           the layer scripts that write their frames next to their outputs (overnight_20261008/,
                long_rows_20261008/, lirf_row_audit/); everything else under runs/ is output and not tracked
config/         manual_rows.csv (not tracked; in the frozen download, see docs/REPRODUCE.md)
docs/           METHOD.md, DESIGN_DECISIONS.md, A_FEW_ROWS.md, REPRODUCE.md, FROZEN_ARTEFACTS.md, DATA_SOURCES.md,
                SCRIPTS.md
tests/          unit tests that need no data
Alternative Models/  experimental v85 January / July and four-month prediction files (see above)
requirements.txt
LICENSE         GNU GPL v3
```

No data is redistributed: every open source is downloaded by the scripts from its owner (see `docs/DATA_SOURCES.md`).

## Acknowledgements

EUROCONTROL / PRC for the challenge data; adsb.lol and its feeders (ODbL); the OPDI project; the X-Plane Scenery
Gateway contributors (GPL-2.0-or-later); Iowa Environmental Mesonet (METAR archive); OurAirports; OpenAP. Several layer
ideas were re-implemented from descriptions in other teams' public repositories — credited in the scripts that use them
(teams outgoing-boat, versatile-violin and OpenAir); the daily reconciliation layer and the cap at D were implemented
after reading the public descriptions of teams gracious-tornado and reliable-hamburger (no code copied).
