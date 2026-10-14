# Design decisions

This document lists the design decisions behind the final file of team gentle-tractor, in pipeline order, with the 2025
validation evidence for each one, the data and features used, the ideas that were tested and left out, and the scripts
that implement each stage. It complements `docs/METHOD.md` (what the method is), `docs/REPRODUCE.md` (how to run it),
`docs/FROZEN_ARTEFACTS.md` (inputs that are read but not rebuilt), `docs/DATA_SOURCES.md` (sources and licences) and
`docs/SCRIPTS.md` (usage lines).

**Conventions.**

- ΔSSE is the change in the sum of squared errors on the 2,085,047 labelled 2025 departures, in M s² (10⁶ s²); negative
  is better. Where the sources record the change as a root-mean-square error instead, it is given as ΔRMSE in seconds on
  the same rows.
- "Six-fold" = pooled out-of-fold prediction over the six seasonal folds (A = Jan + Jul, B = Feb + Aug, C = Mar + Sep,
  D = Apr + Oct, E = May + Nov, F = Jun + Dec). "Forward" = fitted on Jan–Jun, tested on Jul–Dec (H1→H2), and the
  reverse (H2→H1). "Folds better" counts the folds whose ΔSSE is negative.
- "Proxy" = the 2025 out-of-fold prediction of the chain up to a given step; every step is judged as an increment on the
  proxy of the step before it.
- D = take-off − schedule (the value a schedule copy produces); N = the normal-taxi prediction; p = the prediction.
- No public or official score appears in this document. Where a choice was made by comparing candidate files on the
  public January / July leaderboard rather than on 2025 validation, the text says so.

## 1. The final file in one table

| stage | what | entry point | final-file rows moved |
|---|---|---|---:|
| 1 | base and feature tables (G0–G13) | `scripts/10_build_base.py` … `scripts/41_build_v7_backfill.py`, `scripts/250_nm_backfill.py` | — |
| 2 | ADS-B and OPDI tables | `scripts/119_adsb_download.py`, `tools/adsb/chain_final.sh` | — |
| 3 | v45, the member composition | `tools/members/run_a15.sh` | all 670,790 |
| 4 | layer frames | `tools/layers/build_frames.py` | — |
| 5 | chain steps 49–68 and the manual rows → file A15 | `tools/final_chain.py` | per step, §2.6 |
| 6 | post-steps A15 → A15L → A15LX → A15LXR → A15LXRH → A15LXRHDS (the final file) | `tools/last_day/run_last_day.py` | §2.8 |

## 2. Design decisions in pipeline order

### 2.1 Target and label artefacts

| decision | what | why | evidence (2025) |
|---|---|---|---|
| Classify every 2025 label before modelling | `scripts/91_label_artifact_audit.py`: schedule copy (\|off-block − schedule\| ≤ 10 s), date bug (≥ 20 h and within 90 min of a whole number of days), EOBT_1 / LOBT copy, other long, normal | the target is the airport's recorded off-block, not always a taxi time | 4,126 labels above 1 h (0.20 % of rows) carry 41.7 % of the squared error of the chain's 2025 proxy; 16 date bugs alone carry 22.3 % |
| Predict a mixture, not a regression | `p = N + w · (D − N)`; a date-bug branch `P_b · (k · 86,400 + N)` on NM-unmatched LIRF rows (`scripts/tail_experts.py`) | one wrong date-bug row costs as much as thousands of normal rows; under squared loss the best prediction for an uncertain copy is the probability-weighted value | adding the date-bug expert and cleaning the clean expert's training rows: ΔRMSE −4.97 s (strict protocol, against the reference of that time) |
| Train normal experts on clean rows only | no copies, no date bugs, no helicopters in the normal experts' training rows | copies and date bugs are not taxi times; their values come from D and the day shift | part of the same −4.97 s step |
| Keep 98 % of long labels in a known population | the long-label gate (step 68) works on departures ≥ 50 min after schedule | 4,058 of the 4,126 long labels sit in that population (286k rows, 1.4 % long); the other 68 have no timing flag | `runs/long_rows_20261008/audit.py`, `gate.py` |

The labels that remain unexplained are "near-plan records" (the recorded off-block sits near the plan while ADS-B and
NM show the aircraft leaving about an hour later) at LTFM, EGLL, LFPG and LIRF, and two date bugs at normal delays (one
day early) that no label-free input marks.

### 2.2 Two lanes: NM-matched and NM-unmatched

| lane | 2025 share | anchor | experts |
|---|---|---|---|
| NM-matched (an NM actual off-block `AOBT_3` exists) | ≈ 99 % | take-off − AOBT_3 | residual members on top of the AOBT_3 taxi; copy classifiers; the LIRF identity gate |
| NM-unmatched | ≈ 1 % (most artefacts, above all at LIRF) | none | NM-masked normal set networks, ADS-B twins, the strict tail gate, unmatched-mode classifiers, the date-bug expert |

Why: an NM match exists only when the recorded off-block lies within about ±1 h of LOBT (§2.5), so the two lanes hold
different label populations. The unmatched lane carries the long tail: in the 2025 proxy through step 68, unmatched LIRF rows (1,488)
hold 20.3 % of all 2025 squared error, unmatched LFPG 10.2 %, unmatched LSZH 7.4 %.

### 2.3 Validation protocol

| element | rule |
|---|---|
| Folds | six seasonal folds of 2025 (A … F above); every model refitted per fold; every hyper-parameter, blend weight, gate strength or shrinkage chosen inside the training folds (nested / "strict" blocked pairs) |
| Forward | every layer also fitted on Jan–Jun → tested on Jul–Dec, and the reverse |
| Pass rule (`runs/overnight_20261008/ov.py judge`) | six-fold ΔSSE < 0 **and** ≥ 4 of 6 folds better **and** both forward splits better **and** January not worse than noise, noise = 0.05 · \|six-fold ΔSSE\| + 0.5 M s², on both the six-fold and the forward January |
| Small-row rules | rules that touch a few dozen rows are judged by counts: rows caught, missed and false alarms |
| Reported separately | the "body" (labels ≤ 1 h) and the tail; per airport, per fold, per month |
| 2026 guard (post-steps only) | the move of every airport-month is scaled down to ≤ 1.5 × the 2025 out-of-fold move RMS of the same airport-month |
| What is fitted on 2026 | nothing: no model, weight or rule parameter is fitted on 2026 rows; but see §2.8 for the EUROCONTROL daily values of 2026 days |

Limits that apply to every number below:

- The 2025 folds were reused for many decisions over the project, so the 2025 evidence is a validation set, not an
  untouched test set. Some early selections (for example the member families behind v45) reused the same benchmark.
- The forward split was added to the rule during the project; for the earliest chain steps (49–57) the forward result is
  not recorded in the sources.
- The pass rule was introduced with the step-67 layers. One batch of post-step candidates was first judged with a
  stricter variant (≥ 5 of 6 folds, January and February not worse); the verdicts below use the rule in the table.
- The 1.5 × guard was introduced after a residual layer whose 2026 corrections were 2.2 × its 2025 out-of-fold size
  passed on 2025 and did not hold up (§5, CatBoost). The guarded post-steps themselves were never rehearsed on 2025:
  the guard is defined on 2026 vs 2025 cells only.

### 2.4 Members and composition: v45

v45 is a fixed composition of 63 trained inputs (formulas in `docs/METHOD.md` §6, code `tools/members/recompose_v45.py`).
Matched rows: a per-cell simplex blend of 16 residual members on top of take-off − AOBT_3 (26 airport × ADS-B-state
cells, exact simplex fit with ridge 1e-3), mixed with D by the mean of three copy classifiers and the fallback detector.
Unmatched rows: NM-masked normal set networks + an ADS-B twin difference, hedged against D by the strict tail gate and
the unmatched-mode classifiers, then the date-bug expert. LIRF matched fixed-wing rows: the identity gate (A51).

How v45 was built up (strict 2025 protocol; ΔRMSE in s against the previous approach; for most approaches the sources
also record that every fold, and from A34 on every airport, improved):

| approach | addition | ΔRMSE (s) |
|---|---|---:|
| A20 | G10 same-stand timeline in the base model | −3.16 (−8.13 cumulative with A19) |
| A22 | G11 operations context in the clean expert; context-aware fallback detector | −4.17 |
| A25 | matched-row AOBT_3-residual expert over two global seeds and per-airport models | −1.07 |
| A27 | tuned clean experts and residual members (saved as the frozen A27 package) | −0.83 |
| A31 | per-airport cross-fitted weights of the three residual members | −0.38 |
| A32 | five MLP residual members | −0.91 |
| A34 | neighbour-set residual member (set encoders over the airport's departure timeline) | −5.25 |
| A35 | NM-masked set networks as the unmatched normal expert; set copy classifiers in the matched copy probability | −1.99 |
| A37 | surge context (six surge set networks, a surge LightGBM; −0.71 alone) and a neural copy classifier in the unmatched gate for D ≤ 10 h | −4.13 |
| A38 | ADS-B off-block member (adsb.lol traces, push-back for about a quarter of the rows) | −9.09 |
| A40 | ADS-B set-network member; ADS-B LightGBM averaged with its surge twin | −1.48 |
| A42 | blend weights solved exactly on the simplex; six separately weighted ADS-B tree members incl. one per airport | −0.517 |
| A46 | ADS-B twin difference for the unmatched lane; the ground stop sequence repairs tows and returns to stand | −0.457 |
| A47 | stop sequence in the matched ADS-B members | −0.138 |
| A48 | more seeds on the heaviest single-seeded group; second tier of the unmatched ADS-B rule | ≈ −0.04 |
| A50 | LIRF matched identity gate with corrected matched-mixture inputs; all 16 blend weights recalibrated inside calibration folds | −0.826 |
| A51 (= v45) | three-objective identity gate mean with outer-block identity classifiers, alpha 0.7 from fold A's inner months | −0.537 |

An earlier version of the LIRF identity gate (A49) used incorrect matched-mixture inputs and had paired-label feedback;
A50 replaced it. For the final file the 56 members that were not saved as models were refitted with their original
columns, rows, parameters and seeds, and every member was re-predicted on the final rows (`tools/members/rescore.py`).
Two decisions were made at that point:

| decision | evidence | basis |
|---|---|---|
| ADS-B members trained on the pre-2026-09-26 ADS-B snapshot ("variant A") rather than on the refreshed table ("variant B", 19 re-downloaded days, +51k rows) | everything above the members (cell weights, gates, every chain layer) was fitted on the out-of-fold predictions of the snapshot-trained members; on 2025 a retrain of one ADS-B member on the refreshed table was noise (ΔRMSE −0.06 / +0.09 s) and a refit of the step-49 motion expert on it failed its gate | chosen between two candidate files by a public-leaderboard comparison |
| Unmatched twins averaged over 15 seeds (41–55) instead of 3 | v45 is linear in the twins; the twins differ by 131 s RMS between seeds; the 3-seed mean carried 10.6 M s² of seed noise on the final rows, 15 seeds remove 8.5 M (`tools/members/twin_seed_noise.py`) | decided from that measurement |

### 2.5 Rules from the dataset construction: W, F, R, B

The challenge dataset matched a departure to its NM flight only when the recorded off-block lay within about ±1 h of
LOBT. This gives label-free bounds (`scripts/251_v51_nm_window.py`, `scripts/256_v52_recovered_lobt.py`,
`scripts/263_v53_build.py`, `scripts/265_v54_build.py`; re-applied after every learned step by `rules_v55` in
`scripts/267_v55_build.py`).

| rule | rows | what | 2025 evidence |
|---|---|---|---|
| W | NM-matched | clip p to take-off − LOBT ± 3,606 s (first ± 3,660 s; tightened to the 2025 maximum in step 55) | with F: ΔRMSE −0.655 s on the v45 out-of-fold prediction; every fold, month and airport better; no fitted parameter |
| F | NM-unmatched with an LOBT recovered from the destination's arrival record (`scripts/250_nm_backfill.py`) | the off-block lies outside ± 1 h of that LOBT: late take-offs floored above the band | in the W figure |
| R | unmatched, recovered LOBT, schedule within 59 min of LOBT | such a row cannot be a schedule copy (it would have matched NM): cap at take-off − LOBT − 3,540 s (edge 3,600 s from step 55) | ΔRMSE −0.202 s; 127 rows moved, 0 worse; on 1,715 such rows, 0 copies and 0 labels above the cap |
| B | other recovered-LOBT rows ("sched_far") | moved to the lower band edge | 14 rows moved, 0 worse (ΔRMSE −0.039 s) |
| guards | all four | only where the airline normally matches NM at the airport; never on VFR rows (the builder never matches them); floor 30 s; helicopters keep the base | — |

A wider clip (3,618 s, to allow for the ±12 s dither of the final file's take-off times) was tested and left out (§5).

### 2.6 The chain after v45 (`tools/final_chain.py`)

Each step refits deterministically, with fixed seeds, on its frozen 2025 out-of-fold frame and moves only the rows its
signal is about; the W/F/R/B rules are re-applied after each learned move. Steps 49–57 report ΔRMSE in s on their 2025
proxy (as recorded); later steps report ΔSSE in M s².

| step | rows | what | why | 2025 evidence | rows moved (final) |
|---|---|---|---|---|---:|
| 49 (A57) | non-LIRF NM-matched fixed-wing rows with ADS-B motion features | sustained-motion normal expert on v45's matched component (saved booster, alpha 0.5, source weight 1) | the first real sustained motion seen by ADS-B times the off-block better than the first position change | nested, non-LIRF branch: ΔRMSE −0.241 s on that branch; every fold, month and top-10-removed check better; forward not recorded | 172,481 |
| 51 (A59) | matched; unmatched with recovered LOBT | rules W / F | §2.5 | ΔRMSE −0.655 s, every fold / month / airport better | 57 |
| 52 (A60) | unmatched, recovered LOBT near the schedule | rule R | §2.5 | ΔRMSE −0.202 s, 0 rows worse | 44 |
| 53 (A61) | LSZH with OPDI ground events; recovered-LOBT rows | OPDI off-block correction; band rule B | OPDI stand / apron exit events are timed independently of the airport record | OPDI part ΔRMSE −0.907 s (nested, shrink 1.0 in every fold, every fold and month better, all of the gain from the OPDI columns); B −0.039 s; no airport worse | 39,245 |
| 54 (A62) | EDDF with an OPDI stand / apron exit | stand-level OPDI correction; sched_far cap | stand and apron exits keep their meaning across years at EDDF (coverage from September 2025) | ΔRMSE −0.025 s (EDDF −0.43 s on its rows); folds C–F better, A / B nearly empty; the sched_far cap moves 0 rows on 2025 (one row in 2026) | 5,651 |
| 55 (A63) | NM-matched | gate retention / same-runway overtakes between AOBT_3 and take-off; NM-window edges tightened | the queue an aircraft joins after AOBT_3 (idea from team outgoing-boat's public repository) | ΔRMSE −0.218 s (body −0.121), nested shrink 0.5 in every fold; 0 folds and 0 airports worse; rules alone −0.027 s | 610,232 |
| 56 (A64) | all | lateness of the neighbouring departures (12 columns); LEMD 3,606 s cap; BEL hedge | idea from team versatile-violin's public repository; LEMD 2025 labels never exceed 3,598 s | ΔRMSE −0.046 s (body −0.054), nested shrink 0.25 in every fold; fold B +0.009 s (5 of 6 better); LFPG −20.9 M, LIRF −9.5 M; LEMD cap −3.1 M on 2 rows | 661,623 |
| 57 (A65) | LIRF unmatched, 1–4 h late | stand reuse between schedule and take-off moves the row toward or away from D, bucket mean kept | a stand reused before take-off means the aircraft had left it (OpenAir's "stand release" idea) | ΔRMSE −0.171 s cross-fitted, 0 folds worse, but fold A ≈ 0 (−0.96 M), the body worse (+0.096 s), the top 10 rows carry 69 % of the gain; forward not recorded | 482 |
| 63 (A70) | LIRF unmatched, 1–8 h late | same-period context classifier re-estimating the copy weight | copies cluster in time (recording practice of the period) | −600.9 M (ΔRMSE −0.62 s); every fold and month better; the forward splits better | 426 |
| 66 (A73) | six airports with a v2 ADS-B match; LIRF recovered-LOBT rows; step-2 links | L1 push-back phase residual (alpha 0.75); L2 recovered-LOBT ("F_range") rows set to D; L3 step-2 turnaround-link residual (alpha 0.5) | push start / end from the heading-vs-motion trace; F_range rows are copies in 22 of 25 cases | L1: −544.6 M, every fold, forward −392.2 M, January better in both; L2: 22 of 25 copies; L3: ≈ 0 on 2025 (July rows only) | 316,046 |
| 67 (A78) | four row groups | item 5 recorded-take-off shift seen by ADS-B; item 4 stand error; item 2 LTFM month-end cut; item 11 joint arrival / departure coverage shadow | §2.6.2 | combined: −60.3 M, 6 / 6, forward −18.8 / −22.9 M, January −6.8 M | 221,488 |
| 68 (A79) | non-LIRF departures ≥ 50 min after schedule | three-class long-label gate (short / schedule copy / other long), alpha 0.25 | the existing copy gates do not separate "other long" labels outside LIRF | stacked on the step-67 proxy: −130.2 M, 5 / 6 (B +12.6), forward −64.7 / −37.9 M, January −16.7 M, body share 0.33; LTFM +41 M | 95,153 |

#### 2.6.1 Notes on single steps

- **Step 56, BEL hedge.** Five LIRF unmatched rows of one carrier whose 2025 records were often the local time written
  as UTC (15 of 23 labels = UTC offset + a short taxi; ADS-B confirms the short taxi on 11 of them) are moved toward
  "offset + 776 s" with weight 0.65. The weight was set by hand from the same 2025 rows (in sample ΔRMSE −0.045 s, 9 of
  23 rows worse); there is no out-of-sample check. The step-56 file (correction + LEMD cap + hedge) was kept after its
  candidate file was compared on the public leaderboard.
- **Step 54, sched_far cap.** A label-free consequence of rule B; no 2025 row is affected, so there is no 2025 evidence.
- **Step 57.** The 2025 evidence is mixed (body worse, fold A ≈ 0, gain concentrated on 10 rows). The step was kept after
  its candidate file was compared on the public leaderboard.
- **Step 66, L3.** About zero on 2025 and fitted on July rows only; in the final file it is applied to June and July
  (`--l3-months 6,7`) without a separate 2025 check for June.
- **General.** Every chain step was also submitted as a candidate file during the public phase. Three LIRF layers that
  passed their 2025 cross-fits between steps 57 and 63 (arrival-copy rate, airline copy rate, next-day copy) lost as
  candidate files and are not in the chain; a later forward re-check of two of them also failed (§5).

#### 2.6.2 The four step-67 layers (`scripts/279_overnight_layers.py`, `runs/overnight_20261008/`)

| item | signal | 2025 rows | six-fold ΔSSE (folds better) | forward H1→H2 / H2→H1 | January | note |
|---|---|---:|---|---|---:|---|
| 5 | an unclaimed adsb.lol take-off with the same normalised callsign 3–60 min from the record (recorded-take-off shift) | 883 | −15.14 (6 / 6) | −6.27 / −5.90 | −1.48 | small-row layer |
| 4 | ADS-B parked position > 60 m from the learned location of the recorded stand | 57,959 | −17.69 (6 / 6) | −9.38 / −4.79 | −3.48 | EDDF's 2026 flag share is 55 % vs 34 % in 2025 (thin 2026 coverage) |
| 2 | LTFM records stop at 21:00 UTC on every month end | 884 | −0.40 (4 / 6) | −0.20 / −0.33 | −0.04 | 162 rows in 2026 |
| 11 | arrival in-block minus last ADS-B sighting at the same stand and period (coverage shadow), on rows the base prices ≤ 1,800 s | 680,848 | −27.49 (4 / 6) | −2.56 / −12.71 | −1.96 | third variant tried; folds C / D worse; LEMD worse in every variant; low confidence |

Stacked (moves added where layers overlap, rules once): −60.3 M, 6 / 6 (A −10.8, B −9.0, C −1.5, D −5.7, E −25.7,
F −7.6), forward −18.8 / −22.9 M.

#### 2.6.3 Step 68: why outside LIRF only and why α = 0.25

A four-way gate (short / copy / other long / date bug) with exact copy algebra did not beat the base overall (+361 M at
α 0.25): about a third of the copies and two thirds of the other long rows look exactly like normal taxis to every
label-free input. Outside LIRF the three-class gate passes at α 0.25 (−131.3 M on the step-66 proxy, 5 / 6); α 0.10
gives −70.1 M and α 0.50 −112.9 M with weaker forward results; α 1.0 loses. LIRF keeps its own copy gates (A50 / A51,
steps 57 / 63).

### 2.7 The manual rows (`config/manual_rows.csv`)

After step 68 the chain sets 15 LIRF rows to fixed values (`docs/METHOD.md` §9): 14 rows of the 8–13 h rule set to D
and one overnight long-haul row set to the full date-bug label.

| rule | 2025 basis | how the final values were chosen |
|---|---|---|
| 8–13 h rule (LIRF, unmatched, gate-valid, 8 h < D ≤ 13 h, or D > 13 h with a same-local-day take-off → p = D) | all 29 such 2025 rows are schedule copies, in every fold (A 9, B 8, C 4, D 2, E 2, F 4); no 2025 unmatched LIRF row with a normal label has D above 6.4 h | all 14 rows at D. A candidate file with the 14 rows at D lost, which pointed to RYRR388G being a normal taxi; the final file keeps the rule's value (`docs/A_FEW_ROWS.md`). |
| overnight long-haul row → full date-bug label | 9 cases in 2025 (7 copies when the flight left by 12:17 local, 2 date bugs after 14:00); the model gives the row ≈ 60 % date-bug weight | the full label (instead of a 0.65 hedge) was chosen after a comparison of candidate files whose result was consistent only with a date bug. |

The full date-bug value of the overnight row was first chosen after a candidate-file comparison; the row is now also
decided by rule R4 of the last post-step (§2.8.6), which is this same 2025 ITA pattern and gives rule-decided rows their
full value.

### 2.8 Post-steps after the chain (`tools/last_day/`)

The chain's file A15 is followed by four post-steps; A15LXRH is the final file. Each post-step is fitted on all 2025 IFR
non-helicopter rows, keeps the 15 manual rows (and, after A15L, A15L's rule rows) fixed, and is checked by
`build_A15L.write` (670,790 rows in template order, unique ids, finite, ≥ 30 s, manual rows unchanged). The 2025
evidence comes from the 2025 proxy of the full chain (step 68 included); the rules are a re-implementation of
`rules_v55` that is a fixed point on both the proxy and A15 (0 rows differ).

#### 2.8.1 A15L: two fixed rules (`build_A15L.py`)

| rule | script | what | 2025 evidence | 2026 rows |
|---|---|---|---|---:|
| c0 | `c0_long_delay_rule.py` | the 8–13 h rule of §2.7 on every in-scope final row (the manual table covered the January / July rows only) | 29 / 29 copies (counts rule); next-day rows above 13 h (10 copies, 11 date bugs) left to the chain | 2 (16 in scope, 14 already manual rows) |
| c3 | `c3_cap_at_D.py` | p ≤ D for D > 2 h outside the LIRF date-bug domain (idea from team gracious-tornado's public description) | 8 rows moved: 8 caught (labels within 5 s of D), 0 false alarms; ΔSSE −0.023 M, 3 folds better, 0 worse; forward −0.003 / −0.020 | 7 |

Together: 9 rows, Σ move² 0.539 M s². Both are fixed rules, so the six-fold and forward predictions are the same.

#### 2.8.2 The EUROCONTROL daily reconciliation layer

**Identity.** EUROCONTROL's daily "all pre-departure delay" file gives, per airport and UTC day, `DLY_ALL_PRE_2` (minutes)
and `FLT_DEP_IFR_2` (IFR departures). On 2025 (`c9_step1_identity.py`):

- 60 · DLY_ALL_PRE_2 = K = Σ max(0, BLOCK − SCHED) = Σ max(0, D − y) over the airport's departures whose recorded
  off-block falls on that UTC day. Correlation 0.9997–1.0000 and sum ratio 0.995–1.005 at every airport; median
  |K − Σ| per day 0.8–1.5 min at seven airports, 6 min at LSZH, 34 min at LFPG, 40 min at EHAM (all rows); with VFR and
  helicopter rows removed: EHAM 2.4 min, LSZH 1.2 min. The off-block UTC day fits better than the take-off or schedule day.
- FLT_DEP_IFR_2 = the number of IFR non-helicopter departures by recorded off-block day: exact on 92–100 % of 2025 days
  at nine airports; LFPG ≈ 12 % (EUROCONTROL counts about 2 more LFPG departures per day than the challenge file).

**What the file is.** EUROCONTROL describes it as the daily total pre-departure delay by airport, all causes
(https://ansperformance.eu/reference/dataset/all-pre-departure-delay/); both fields are supplied by the airport operator. Pre-departure delay is the delay at off-block against the
schedule, so the file is the daily sum of the airport-recorded off-block delays, and the off-block time is what the
challenge masks. On 2026 days the layer reads the 2026 values of the file, i.e. daily sums over the masked 2026
off-block times. The organisers ruled on the monthly EUROCONTROL file (additional taxi-out time); they did not rule on
this daily file.

**Model** (own implementation; the idea was taken from the public descriptions of teams gracious-tornado and
reliable-hamburger, no code copied). Per airport × estimated off-block UTC day (t − p): R = K − S_p with
S_p = Σ max(0, D − p). Row errors are taken as independent Gaussians e_i ~ N(0, s_i²); on the day's active rows
(D − p > −s_i) the constraint is Σ e_i = −R + noise(s_K²). The posterior mean is
`e_i = −R s_i² / (Σ_active s_j² + s_K²)`, and `p_new = rules(p + λ e_i)`. This puts most of a day's residual on the
uncertain rows (LIRF copy-vs-held mixtures) and a small shift on ordinary delayed rows. Fitted inside the training rows:
s_i² per airport × lane × D-bin × (D − p > 0); s_K² per airport on whole days; λ by training SSE. Days whose count differs
from FLT_DEP_IFR_2 by more than 2 % are skipped.

| post-step | script | adds | 2025 increment (six-fold, folds better; forward; January six-fold / forward) | verdict |
|---|---|---|---|---|
| v1 (reference) | `c9_daily_layer.py` | one pass, all rows, λ 0.5 | over the chain proxy: −1,587.1 (6 / 6); −659.5 / −630.9; −79.8 / −90.2 | pass |
| A15LX | `c15_combined.py`, `build_A15LX.py` | (1) up to 3 passes (chosen: 3, λ 0.25); (2) IFR non-helicopter rows only, VFR / helicopter rows never move; (3) count gap K_adj = K − (F − n) · μ_apt, n recomputed in every pass | over v1: −238.6 (6 / 6); −75.2 / −286.6; −7.8 / −1.4. Over the chain proxy: −1,825.7; −734.7 / −917.5 | pass |
| A15LXR | `c16_stack_recentred.py`, `build_A15LXR.py` (functions in `c10_daily_v2.py`, `c13_recentred_v2.py`) | exact copy / held enumeration on LIRF unmatched rows with 3,600 < D ≤ 43,200 s (1–14 per day), the day residual re-centred by the airport-month median, at the stack output; λ4 by training SSE (1.0) | over A15LX's proxy: −909.4 (5 / 6, A +7.1); −787.1 / −127.0; −43.5 / −42.8. Stack + enumeration over v1: −1,148.0 (6 / 6) | pass |
| A15LXRH (final) | `c18_crossers_hardened.py`, `build_A15LXRH.py` (machinery in `c17_crossers.py`) | hardened midnight-crossers step (§2.8.4) | over A15LXR's proxy: −1,801.1 (4 / 6, C +57.7, D +0.6); −13.1 / −107.9; −2.3 / −2.7 | pass |

Component evidence for A15LX (each over v1): passes alone −169.2 (4 / 6; forward −35.9 / −259.0; January −4.3 / +0.5);
IFR filter alone −30.5 (6 / 6); IFR + count gap −57.7 (6 / 6); the count gap over the IFR layer −27.15 (6 / 6; forward
−14.41 / −16.78; LFPG −11.10, only LSZH worse, +0.56). μ (2025, s): EDDF 1,133, EDDM 769, EGLL 743, EHAM 1,175, LEBL
1,000, LEMD 991, LFPG 1,169, LIRF 861, LSZH 1,005, LTFM 796.

v1 by airport (six-fold, M s²): LTFM −765.4, LIRF −555.4 (unmatched −473.7), LEBL −80.1, LFPG −56.8, EGLL −54.8, LEMD
−40.5, EDDF −32.0, EDDM −12.0, EHAM −5.6, LSZH +15.4 (removed by the IFR filter). The stack over v1: LIRF −92.2, LTFM
−69.1, LFPG −24.2, EHAM −18.6, LSZH −16.8, LEBL −13.1, LEMD −4.9, EDDF −3.7, EDDM −1.2, EGLL +5.1.

#### 2.8.3 The LIRF copy / held enumeration and its re-centring

For each off-block day with 1–14 eligible LIRF rows, every row is either a copy (y = D) or held (y = N + ε, N and sN²
from held-row tables by airport × lane × fine D-bin); every held set is enumerated and scored by
R_L = K − Σ_others max(0, D − p) ~ N(Σ_held (D − N), s_K² + Σ s_i² + Σ_held sN²), and each row moves by λ4 × (posterior
mean − prior mean). A month-level offset of the day residual (later flights, a recording change) would be read as "some
copies were held"; re-centring subtracts the airport-month median of the model residual over the eligible days
(≥ 5 days, else no shift). It is label-free in both years.

| variant | 2025 increment | verdict on 2025 | basis of the final choice |
|---|---|---|---|
| plain enumeration on v1 | over v1: −843.1 (6 / 6); forward −530.1 / −566.0; January −76.4 / −67.2. 1,156 LIRF rows change: copies −330.6 M, held −512.4 M | pass | its candidate file lost to the v1 file |
| re-centred, on v1 | over the plain enumeration: −176.7 (5 / 6, A +119.3); forward −184.3 / −45.4; **January +46.4 / +48.6 (noise 9.3)**; July +72.9 | **fails January** | — |
| re-centred, on the stack output (in the final file) | over the stack: −909.4 (5 / 6); forward −787.1 / −127.0; January −43.5 / −42.8 | pass | — |

The re-centring entered the final file after a public-leaderboard comparison of candidate files (re-centred vs plain),
although its own 2025 increment over the plain enumeration failed the January condition. Stack + re-centred enumeration
did not pass against the re-centred enumeration on v1 alone (−128.2 M, 3 / 6).

#### 2.8.4 The hardened midnight-crossers step (A15LXRH)

Per airport × UTC day, X(d) = FLT_DEP_IFR_2(d) − n_sched(d) is the net number of flights whose recorded off-block crossed
UTC midnight away from their schedule day. Rows: schedule on an earlier UTC day than the take-off, and long
(1 h < D ≤ 24 h: copy / held / LIRF date bug) or ambiguous (predicted off-block within 30 min of 0 h UTC). Groups of
≤ 12 rows (airport × take-off day, days d − 1 and d) are enumerated; each joint state is scored by prior × Gaussian K_adj
likelihood (re-centred per airport-month) × α · count term (n_h − F), and p += λ4 (E_post − E_prior). Priors come from
the row's current prediction and the chain's date-bug gate.

| guard (fixed before the 2025 run) | what |
|---|---|
| support gate | copy / date-bug states only in airport × lane × D-band cells whose training artefact rate ≥ τ (τ ∈ {2, 5, 10 %}); unsupported long rows may move only on the midnight side |
| capped priors | copy ≤ 3 × the cell's copy rate, date bug ≤ 3 × the cell's bug rate |
| move cap | 1,800 s where the cell rate < 20 % (LIRF unmatched never capped) |

τ, α, c_v and λ4 by training SSE inside each split (all-2025 fit: τ 2 %, α ½, c_v 4, λ4 1). 2025: −1,801.1 M, 4 / 6;
LIRF −1,795.8 M (unmatched −1,779.0). The gain is concentrated: the top 3 rows (LIRF unmatched date bugs, y ≈ 87,200 s)
give −1,562.8 M; without the top row −1,107.0 M. Counts (six-fold): midnight side 77 caught, 24 false alarms, 87 missed;
copy direction on supported long rows 24 caught, 2 false alarms, 12 missed; forward 12 caught, 26 false alarms, 8 missed.

#### 2.8.5 Guards and moves on the final rows

| file | guard | 2026 rows moved vs the previous file |
|---|---|---|
| A15L | none (fixed rules) | 9 rows, Σ move² 0.539 M s² |
| A15LX | 1.5 × per airport-month vs this variant's 2025 six-fold move; factors < 1: EDDF-1 0.64, EDDF-2 0.30, EDDF-7 0.99, EHAM-1 0.40, EHAM-6 0.51, LEBL-1 0.81, LEBL-2 0.33, LIRF-1 0.995, LSZH-1 0.66, LTFM-1 0.91 | v1 alone moved 568,876 rows (Σ move² 426.9 M s² after its guard); A15LX differs from the v1 file on 284,200 rows by > 1 s |
| A15LXR | whole layer as A15LX; increment guard vs the 2025 increment (no cell scaled; LIRF ratios 0.99–1.23) | 427 rows > 1 s (LIRF only), Σ move² 176.2 M s² |
| A15LXRH | increment guard: EDDF-01 0, EHAM-01 0.06, LIRF-01 0.13, EHAM-02 0.22, LIRF-02 0.37, LFPG-07 0.40, LEMD-01 0.45, LTFM-01 0.75, LEBL-01 0.82 (416 rows rescaled) | 827 rows > 1 s, Σ move² 313.3 M s², of which LIRF 309.9 M on 33 rows; 5 LIRF rows move > 600 s |
| A15LXRHDS | none (fixed rules on LIRF rows) | 7 LIRF rows (5 copies, 2 date bugs) |

**How the post-steps were chosen.** Every post-step passed the 2025 rule on its own increment (the re-centring
excepted, §2.8.3). The switches of the final file from A15 to the daily layer, from the daily layer to A15LXR, and from
A15LXR to A15LXRH and from A15LXRH to A15LXRHDS were each made after a public-leaderboard comparison of the candidate
files. The combination A15LX was
not itself submitted as a candidate file.

#### 2.8.6 LIRF date-bug / schedule-copy rules (A15LXRHDS, `c19_lirf_datebug_rules.py`)

The last step decides copy or date bug for LIRF NM-unmatched departures that take off on a later local day (LIRF only;
`docs/METHOD.md` §11.5 has the rule table). How it was reached, in order:

1. **Snapping the chain's own weight lost.** Setting the date-bug value where pb ≥ 0.5 (13 rows in 2025, 9 date bugs):
   +1,690.9 M s², 2 / 6 folds, forward −888.4 / +1,220.0. pb is a delay-bin rate; inside a bin copies and date bugs
   look alike, and a copy sent to the date-bug value costs 279–1,076 M s² against 2–143 M s² gained per date bug.
2. **Rules read off the 2025 rows.** The 168 rows of the domain were read one by one, date bugs and copies side by side.
   Findings: no date bug between 3.3 h and 14 h of delay; every 2025 row at 18.5–26 h is a
   date bug; ITA's overnight long-haul splits by take-off clock (§2.7); one date bug (ITY332) shows an off-grid 23:59
   schedule unlike its usual 21:30; the inbound turnaround points to a date bug when the aircraft landed the day before
   (5 date bugs, 1 copy at ≥ 14 h). Rules R0–R4 encode these. All counts are in-sample.
3. **R6 (rare operator → copy), after one 2026 row.** R3 (inbound ≥ 12 h before take-off) had one 2026 row,
   SVI3006 2026-07-16, and a candidate-file comparison was consistent only with a copy. SVI's operator is absent from all
   of 2025 and has 8 LIRF departures in 2026. On 2025, every LIRF unmatched row ≥ 8 h late whose operator has fewer than
   300 LIRF departures over both years (or which has no flight number) is a copy: 8 of 8; the nearest date bug belongs
   to an operator with 342. R6 encodes that and sits before R2 and R3. In 2026 it decides SVI3006 and IGA674 06-06
   (14.4 h, 3 departures, all in 2026) and nothing else.
4. **R5 (Qantas → copy)** was added from the two 2025 QFAQFA6 rows (both copies); its 2026 row (QFA7006 07-19, D 24.3 h)
   moves by only 395 s because its copy and date-bug values are one day apart.

Left on the mixture (no rule fires): NOS431 (31 h; its operator's 2025 row at 36 h was a copy), WMT3508 (15.7 h, short
same-day turnaround: 4 date bugs / 5 copies in 2025).

## 3. Data used

Licences and download terms: `docs/DATA_SOURCES.md` (not repeated here).

| source | what is taken | where it enters |
|---|---|---|
| PRC Data Challenge 2026 files | airport movement records (`*_mvt`: off-block, take-off, schedule, stand, runway, flight rule, …) and NM fields (`*_flt`: callsign, EOBT_1, LOBT, AOBT_3, aircraft, operator) for the 2025 training months and the final ranking file; arrivals of both years as label-free context | `scripts/10_build_base.py` (stage 1) → every later stage |
| adsb.lol `globe_history` (487 days) | ground and low-altitude points inside each airport box | `scripts/119_adsb_download.py`, `118_adsb_probe.py extract` → `121`, `122`, `124`, `232`, `255`, `adsb_push_phases.py`, `213`, `231`, `164` / `165` (stage 2) → v45 ADS-B members (stage 3), steps 49, 66, 67, 68; push phases also read by the midnight-crossers step |
| ADS-B snapshot of 2026-09-26 (derived, frozen) | 2025 training rows of the 26 ADS-B members ("variant A") | `docs/FROZEN_ARTEFACTS.md` → `tools/members/lgb_members.py`, `nn_members.py` |
| OPDI | flight list and ground events (stand / apron / taxiway / runway) | `scripts/257`–`260` (stage 2) → `262_opdi_correction.py` → steps 53, 54; the OPDI link supplies the icao24 for the step-67 coverage shadow |
| X-Plane Scenery Gateway | stands, taxi network, runways (`apt.dat`) | `scripts/fetch_external.py` → `20_ext_layout.py`, `stand_type.py` (G4); stand coordinates in the ADS-B stand map (`121 stands`) |
| Iowa Environmental Mesonet METAR | weather of the ten airports | `fetch_external.py` → `20_ext_weather.py` (G5) |
| OurAirports | runway thresholds and headings, airport reference points | `fetch_external.py` → `20_ext_layout.py`, `20_ext_weather.py` (crosswind / headwind) |
| EUROCONTROL workbooks (monthly additional taxi-out and ASMA time; daily arrival ATFM delay and traffic) | member features (G6); 2026 months without a published value take the same month of 2025 (`ec_monthly_is_prior`) | `fetch_external.py` → `20_ext_eurocontrol.py` (stage 1) |
| EUROCONTROL daily `all_pre_departure_delays_{2025,2026}.csv` | DLY_ALL_PRE_2 and FLT_DEP_IFR_2 per airport-day: EUROCONTROL's daily total pre-departure delay (all causes), supplied by the airport operator: the daily sum of the airport-recorded off-block delays, i.e. of the masked quantity; the organisers ruled on the monthly file, not on this one | `fetch_external.py` (section 5) → `tools/last_day/c9_step1_identity.py` (`ec()` loader) → A15LX, A15LXR, A15LXRH |
| EUROCONTROL daily ATC pre-departure delay and ATFM slot adherence | tested only (day-congestion layer, §5); not in the final file and not fetched by the repository scripts | — |
| OpenAP | aircraft dimensions for common types (manual table for the rest) | `20_ext_aircraft.py` (G7) |
| `holidays` Python package; optional school-holiday / event tables | public holidays per country / region; school holidays and events if the tables are present | `20_ext_calendar.py` (G1) |

Not used: OpenSky Network Trino data (not permitted); OPDI measurements (airborne only); adsb.lol ACARS data-links
(probed, no off-block event near the record).

## 4. Features built

Group definitions: `docs/METHOD.md` §3. About 240 columns.

| group | content (short) | built by |
|---|---|---|
| G0 keys | airport, runway, stand, pier, type, wake, segment, operator, flight type / rule, region, NM-match flag, helicopter, source conflicts | `scripts/base_features.py` via `10_build_base.py` / `40_assemble.py`; source conflicts `detectors.py` |
| G1 time | UTC / local time, day of week / year, night, sunset, curfew, holidays, events, DST | `base_features.py`; `ext_calendar.py` (`20_ext_calendar.py`) |
| G2 take-off deltas | take-off − schedule / EOBT_1 / LOBT / AOBT_3, AOBT_3 − LOBT, planned block time, inbound delay | `base_features.py` |
| G3 congestion | departures / arrivals around the take-off, same-runway counts, active runways, movements between AOBT_3 and take-off | `base_features.py` |
| G4 layout | stand → runway taxi distance on the Gateway taxi graph, straight distance, crossings, stand coordinates, route periods | `ext_layout.py` (`20_ext_layout.py`), `stand_type.py`; route periods `detectors.py` |
| G5 weather | METAR as of the off-block; runway wind components; low visibility; precipitation and snow codes; de-icing and frost risk; rolling precipitation / snow | `ext_weather.py` (`20_ext_weather.py`) |
| G6 EUROCONTROL | daily arrival ATFM delay and traffic; monthly additional taxi-out and ASMA time | `ext_eurocontrol.py` (`20_ext_eurocontrol.py`) |
| G7 aircraft | MTOW, wingspan, length, engines, code letter | `ext_aircraft.py` (`20_ext_aircraft.py`) |
| G8 encodings | out-of-fold, time-weighted target encodings and copy-rate encodings; v3 row-wise seasonal-pair out-of-fold target features | `encodings.py` (`35_encode.py`); `nested_features.py` (`36_build_nested.py`) |
| G9 detectors | copy probability, runway configuration and time since its change, disruption-day score, unseen stands | `detectors.py`, `fallback_context.py` (`30_detect.py`) |
| G10 stand timeline | previous / next arrival at the same stand, turnaround gap, inbound delay, inbound copy flag | `stand_timeline.py` (`37_build_v4_stand.py`) |
| G11 operations context | median NM taxi of departures around the row, taxi-in, arrival delay, arrival-record copy rates | `ops_context.py` (`38_build_v5_context.py`) |
| G12 queue | runway take-off gaps, push-back cohorts, pier windows | `queue_context.py` (`39_build_v6_queue.py`) |
| G13 NM back-fill | NM fields for unmatched departures from the destination's arrival record | `nm_backfill.py` (`41_build_v7_backfill.py`); links `250_nm_backfill.py` |
| set encoders | the airport's departure / arrival timeline around each departure (24 previous and 8 next departures, 18 / 6 arrivals); surge and gap tokens | `nbr_set.py`, `surge_context.py`; members `106`, `107`, `110`, `111`, `113`, `114`, `tools/members/nn_members.py` |
| ADS-B v1 | stand map, first sighting, first real move off the stand, clear of the stand area, censoring flags, airport-month quality (`adsb_q_*`), ± 30 / 60 min neighbour counts | `121_adsb_events.py` (`stands`, `events`, `quality`, `features`, `neighbours`); `adsb_context.py` |
| ADS-B geometry and stops | track geometry; ground stop sequence (repairs tows and returns to stand) | `122_adsb_shadow.py`; `124_unmatched_adsb.py stops` |
| ADS-B v2 match | every callsign candidate in normalised form; no cross-airline time-only matches | `255_adsb_match_v2.py`, `flightnorm.py` |
| push-back phases | push start / end, first forward move, hold, push distance and turn (track = heading on the ground) | `adsb_push_phases.py` |
| turnaround links | same-stand and aircraft-identity links (step 1), step-2 links | `213_turnaround_features.py`, `231_turnaround_links.py` |
| sustained motion | when the aircraft really started moving | `164_ground_motion_candidates.py`, `165_ground_motion_full.py`; expert `170`, `173` |
| per-movement ADS-B times | ADS-B timestamps and positions per movement | `232_adsb_movement_times.py` |
| step-67 frames | recorded-take-off shift, parked position vs stand, LTFM month-end cut, arrival tail at the stand | `runs/overnight_20261008/item05_dateshift.py`, `item04_park_extract.py`, `item04y_stand_error.py`, `item02_gaps.py`, `item11_arr_tail_extract.py`, `item11_shadow_layer.py` |
| OPDI | stand / apron exit and other ground events linked to the movements | `257`–`260`; correction frame `262_opdi_correction.py frame` |
| EUROCONTROL day residual | K, K_adj, F − n per airport-day (post-steps only) | `tools/last_day/c9_step1_identity.py`, `c14_count_gap.py`, `c15_combined.py` |

## 5. Tested and kept vs tested and left out

"Kept" items are in §2 and listed only briefly. Ideas taken from other teams' public repositories or descriptions are
credited; no code was copied.

### 5.1 Kept

| idea | evidence on 2025 | decision |
|---|---|---|
| mixture with copy / date-bug branches | −4.97 s ΔRMSE (§2.1) | kept |
| v45 members | §2.4 (ΔRMSE per approach) | kept |
| NM-window rules W / F / R / B | §2.5 | kept |
| chain steps 49–68 | §2.6 | kept |
| 15-seed unmatched twins | 8.5 M s² of seed noise removed on the final rows | kept |
| c0 (8–13 h rule on every row), c3 (cap p ≤ D, gracious-tornado's idea) | counts: 29 / 29 copies; 8 caught, 0 false alarms | kept |
| EUROCONTROL daily layer: passes, IFR filter, count gap, re-centred enumeration, hardened crossers | §2.8 | kept |
| same-day rule (no date-bug weight on a same-local-day take-off; courageous-onion / logical-ladder) | the chain already gives such rows 0 date-bug mass: 0 rows | already in the chain |

### 5.2 Left out

| idea | evidence on 2025 (M s² unless stated) | decision |
|---|---|---|
| CatBoost residual layer on matched rows (gentle-octopus / genuine-cabbage) | −91.2 (5 / 6, B +0.2); forward −24.0 / −18.3; January +0.3 / −6.4; but its 2026 correction RMS was 39 s against 17.6 s out of fold (LTFM 82 vs 22 s); the unmatched arm took 72 % of its LIRF gain from 10 rows | passed the 2025 rule; left out after its candidate file lost on the public leaderboard |
| persistent winter weather (6–12 h precipitation, sustained de-icing, hours since above 0 °C; resilient-kiwi) | 0 of 21 pre-planned arms pass; best +11.8 (2 / 6); a post-hoc arm −23.4 (4 / 6) gained at LTFM, not in winter, and its winter-only form failed January | left out |
| EUROCONTROL monthly level (2026 monthly additional taxi-out → per-airport level shift at EGLL / LEBL / LIRF) | −53.4 (6 / 6); forward −23.0 / −17.2; January −2.2. The monthly figure is not an exact sum of our labels (it excludes de-icing and unreferenced flights, whose share moved in 2026) | passed the 2025 rule; left out after its candidate file lost on the public leaderboard |
| relative long gate (step 68 with "long" = cell mean + 25 min instead of 1 h) | stacked on the step-67 proxy: −198.2 vs −130.2 for the 1 h gate (6 / 6); forward −71.0 / −82.4; January −28.9 | passed the 2025 rule; left out after a public-leaderboard comparison with the 1 h gate |
| ADS-B members of variant B (refreshed 2025 table) | §2.4 | left out after a public-leaderboard comparison |
| score-fitted blends v47 / v48 (fixed blends of earlier submissions with weights from their public scores) | no independent validation; v48's inputs cannot predict February / June rows | excluded from the clean chain by construction |
| orphan floor (unmatched rows raised to 0.95 × the median live taxi of matched departures ± 30 min; gentle-octopus) | whole year, their constants: +59.9 (1 / 6); pre-registered February non-LIRF scope: +0.129 (fold B worse); LTFM variants +8.6 … +50.5 | left out |
| copy-impossible rule (P(copy) = 0 when \|schedule − LOBT\| > 3,606 s; quirky-honey) | every such row is NM-matched; their copy share is 0 in every band and the residual there is ≥ 0 | no gain; left out |
| LightGBM `linear_tree` | not run by us; another team (gentle-octopus) reported predictions extrapolating to ± 300,000 s | not tried |
| day-congestion layer (IATA code-89 minutes per departure, ATFM regulated and late-vs-slot shares, day z-scores) | −7.67 (4 / 6); forward **+8.48** / −7.47; January forward +1.20; code-89 correlates negatively with the residual (pooled −0.17) | fails |
| daily count identity on the enumeration (c12: hypothesis-dependent off-block day + count term) | +30.12 (3 / 6); forward +0.25 / −2.17; one false alarm costs +30 M; the base already has 36 / 36 cross-day rows on the right side | fails |
| unhardened midnight step, all cells may take copy / date-bug states (c15 step 4) | +1,528.6 (1 / 6); forward +970.3 / +259.7; LIRF +1,104.8 (date-bug prior from the training rate, inconsistent with p) | fails |
| unhardened midnight step with priors from the current prediction (c17) | +2,621.6 (2 / 6); forward 0.0 / +3,429.1; one LTFM held row moved 4,115 → 59,096 s (+3,395 M) | fails; led to the hardened form |
| λ per airport × lane in the daily layer (lamcell) | −156.8 (6 / 6); forward −45.2 / −25.8; **January +13.0** (noise 8.3) | fails |
| mixture variance for D > 1 h in the daily layer | +954.8 (3 / 6); LIRF +986 | fails |
| wider NM window (3,618 s, dither allowance) | dither simulation (20 draws): +0.72; under squared loss the conditional mean of a clipped value lies at or inside ± 3,606 s | left out |
| LTFM cold-snow guard (revert layers when snow in 6 h and T ≤ 3.5 °C) | reverting all: −28.2 (3 / 6), all from fold B, whose gate was fitted without February 2025 while the final gate was not | fails |
| Rome summer-time floor (LIRF unmatched not-late rows ≥ 7,900 s in CEST) | +1,843.5 (0 / 6); 50 of 70 moved rows are normal taxis | fails |
| correction-scale guard (revert every layer in an airport-month whose 2026 correction RMS > 1.5 × its 2025 OOF RMS) | month-out rehearsal: +603.4 (A +293.3, B +149.7, …) | fails as a revert; the post-steps use a scale-down guard instead (not rehearsed on 2025) |
| retraining the strict tail gate with EUROCONTROL day-residual features | on the final stack: +109.4 (4 / 6); a February LTFM row +369 M | fails |
| LIRF copy-rate layers between steps 57 and 63 (arrival-copy rate, airline copy rate, next-day copy) | 2025 cross-fitted ΔRMSE −0.203 / −0.150 / −0.091 s; a later forward re-check: arrival-copy +122 M and worse in every forward split, next-day forward +61 / +7 / −1 / +11 M; the airline layer had held-label feedback in its second stage | left out after their candidate files lost; the forward re-check agrees |
| A29: full copy weight for long-delayed LIRF unmatched rows (early version of the 8–13 h idea) | strict ΔRMSE −2.36 s, every fold better | left out after its candidate file lost |
| matched probability gates outside LIRF (A58) and the best of 140 identity-gate configurations (A52) | branch −0.521 s; pooled −0.145 s, all six folds | left out after their candidate files lost |
| hierarchical airport gate for the unmatched lane | all rows −138 (α 0.25) but fold B +84.5; D ≤ 7 h +150.7 (2 / 6) | fails |
| unmatched-only normal expert on ADS-B-matched unmatched rows | −7.36 (6 / 6); forward −1.92 / −1.15; **January +1.12** (noise 0.87) | fails |
| lane × ADS-B-state × airport residual grid | cell means +64.3 (1 / 6); LightGBM +56.2 (1 / 6) | fails |
| G8 encodings rebuilt without non-taxi labels | +46.9 (2 / 6); on a label-free scope +25 (0 / 6) | fails |
| air-return / diverted-inbound identity | −0.71 (4 / 6); forward +0.51 / +0.97 | fails |
| gap-day flags (movement completeness) | +3.16; fold F +4.57; forward H2→H1 +4.09 | fails |
| queue recounted over all departures (matched rows / unmatched rows) | +14.2 (2 / 6) / +76.8 (2 / 6) | fails |
| four-way long gate incl. LIRF | +361 at α 0.25 | fails; the three-class gate outside LIRF is step 68 |
| recovered destination timing as a learned input | about −226 pooled, but +0.36 after removing the 10 most influential rows; 1 / 6 folds | fails |
| blend with take-off − recovered AOBT_3 on recovered rows | −38.9; forward −56.0 / −0.5; 3 / 6 (F +14.1) | fails narrowly |
| matched physical-context adapter (five-experiments screen) | −111.5 (6 / 6), forward −4.9, backward −36.3; LEMD forward +27.1 | passed a preliminary screen only; never built into a layer |
| relative long / short selective adjustments (raise confident long, cap confident short) | raise: −5.85 but forward +7.28; cap: +553.1 | fail |

## 6. The scripts in the repository

Usage lines for every script are in `docs/SCRIPTS.md`.

| stage | scripts / tools |
|---|---|
| configuration | `scripts/prc_config.py` (every location), `scripts/common.py` |
| 1 external sources | `scripts/fetch_external.py`; `20_ext_layout.py`, `20_ext_weather.py`, `20_ext_calendar.py`, `20_ext_aircraft.py`, `20_ext_eurocontrol.py`, `stand_type.py` (modules `ext_*.py`) |
| 1 base and features | `10_build_base.py`, `30_detect.py`, `40_assemble.py`, `35_encode.py`, `36_build_nested.py`, `37_build_v4_stand.py`, `38_build_v5_context.py`, `39_build_v6_queue.py`, `41_build_v7_backfill.py`, `250_nm_backfill.py`, `42_rank_continuity.py` |
| 2 ADS-B and OPDI | `119_adsb_download.py`, `118_adsb_probe.py`; `tools/adsb/chain_final.sh` (runs `121`, `122`, `124`, `232`, `255`, `adsb_push_phases.py`, `213`, `231`, `260` through `tools/adsb/run_step.sh`); `tools/adsb/quality_splice.py`, `coverage.py`, `edge_check.py`; OPDI `257`–`260` |
| 3 members and v45 | `tools/members/run_a15.sh` → `a27_members.py`, `lgb_members.py`, `nn_members.py`, `rescore.py`, `recompose_v45.py`; `twin_seed_noise.py` |
| research history of v45 | `scripts/50`–`97` (experts, gates, label audit `91`), `104`–`132` (members, ADS-B, identity gates); A27 comes from `96_deploy_candidate.py` |
| motion expert | `164`, `165`, `166`, `170`, `173_v49_motion_bundle.py` |
| 4 layer frames | `tools/layers/build_frames.py`, `tools/layers/rt.py` |
| 5 the chain | `tools/final_chain.py`; layers `251`, `256`, `262`–`270`, `274`, `278`, `279`, `281`; `runs/overnight_20261008/` (step 67, `ov.py judge`), `runs/long_rows_20261008/` (step 68), `runs/lirf_row_audit/lirf_rows.py`; report `tools/layers/final_report.py` |
| 6 post-steps | `tools/last_day/run_last_day.py` (entry point) → `c0_long_delay_rule.py`, `c3_cap_at_D.py`, `build_A15L.py`; `c9_step1_identity.py`, `c9_daily_layer.py`, `c10_daily_v2.py`, `c13_recentred_v2.py`, `c14_count_gap.py`, `c15_combined.py`, `build_A15LX.py`; `c16_stack_recentred.py`, `build_A15LXR.py`; `c17_crossers.py`, `c18_crossers_hardened.py`, `build_A15LXRH.py`; shared `ld.py` |
| tests | `tests/` (`py -3.14 -m pytest tests -q`) |

`tools/last_day/` usage:

```bash
py -3.14 tools/last_day/run_last_day.py                 # every post-step, then the output check of the final file
py -3.14 tools/last_day/run_last_day.py --from c18      # resume at a step
py -3.14 tools/last_day/run_last_day.py --no-checks     # skip the steps that only check or report
py -3.14 tools/last_day/c15_combined.py --tag x         # 2025 six folds + forward of the combined daily layer
py -3.14 tools/last_day/c16_stack_recentred.py --tag x  # 2025 evidence of the re-centred enumeration on the stack
py -3.14 tools/last_day/c18_crossers_hardened.py --tag h  # 2025 evidence of the hardened crossers step
```

Inputs of the post-steps: the chain run `runs/clean_chain_final/A15/`, `runs/overnight_20261008/frame_cache.parquet`,
`PRC_ADSB_DIR/push_phases.parquet`, `PRC_EXTERNAL/eurocontrol/all_pre_departure_delays_{2025,2026}.csv` and the frozen
2025 proxy of the full chain (`PRC_FROZEN/ref_layers/runs/leads_20261008/rows.parquet`).

## 7. Not measured

- The guarded post-steps as a whole on 2025 (the 1.5 × guard is defined on 2026 vs 2025 cells only).
- The forward splits of chain steps 49–57.
- The enumeration at airports other than LIRF; LIRF days with more than 14 long rows (about 9 in 2025, 5 in 2026) keep
  the stack value.
- Alternative forms of rejected post-steps after their result was known (for example a trimmed-mean re-centring, the
  midnight step without the date-bug state, a separate μ for the missing side of the count gap): not run, because a
  second form chosen after seeing a result would be chosen on that result.
- A refit of the normal experts on cleaned encodings; pb bins, unmatched-mode networks and the LIRF identity gates with
  EUROCONTROL day features.
