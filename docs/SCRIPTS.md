# Scripts in this repository

Generated from each file's own docstring (first paragraph) and usage lines. Run from the repository root with Python
3.14 (`py -3.14` on Windows, `python3` elsewhere). "final path" = executed (directly or as an imported module) when
the final file is rebuilt (`docs/REPRODUCE.md`); the other files are the research and validation history that produced
the evidence and the frozen artefacts (`docs/FROZEN_ARTEFACTS.md`).

## Shared modules (imported by the numbered scripts)

| file | final path | what it does | usage |
|---|:-:|---|---|
| `scripts/adsb_context.py` | yes | ADS-B off-block context for the models (built by 121_adsb_events.py). |  |
| `scripts/adsb_push_phases.py` | yes | Push-back phases of every ADS-B-matched departure: push start, push end, first forward move under own power. | `py -3.14 scripts/adsb_push_phases.py day 2026-07-20 --apt EDDM --show 5`<br>`py -3.14 scripts/adsb_push_phases.py run --workers 8`<br>`py -3.14 scripts/adsb_push_phases.py collect` |
| `scripts/airport_tail_gate.py` |  | Hierarchical airport-specific gate overrides evaluated on strict paired blocks. |  |
| `scripts/assemble.py` | yes | Assemble feature tables from base + detector + external tables; write manifest and shift report. |  |
| `scripts/base_features.py` | yes | Base feature builder: keys (G0), UTC time (G1), take-off deltas (G2), congestion (G3). |  |
| `scripts/blend.py` | yes | Non-negative least-squares blending of OOF predictions, per airport with a global fallback; optional equal weights. |  |
| `scripts/common.py` | yes | Shared constants and helpers for the PRC 2026 taxi-out pipeline. |  |
| `scripts/detectors.py` | yes | Detectors: 1 fallback (v1, v2, unmatched-lane specialist), 2 helicopters, 3 route periods + 2026 drift, 4 runway configuration, 5 disruption days, 6 unseen stands, 7 source conflicts. |  |
| `scripts/encodings.py` | yes | Out-of-fold, time-weighted target encodings (G8) and fallback-rate encodings for the unmatched lane. |  |
| `scripts/ext_aircraft.py` | yes | Aircraft dimensions (G7): openap for 37 common types, manual CSV for the rest. |  |
| `scripts/ext_calendar.py` | yes | Calendar per airport-date (holidays, school holidays, events, sun times, DST) and local-time row features. |  |
| `scripts/ext_eurocontrol.py` | yes | EUROCONTROL PRU tables (G6): daily arrival ATFM delay and traffic, monthly additional taxi-out and ASMA time. |  |
| `scripts/ext_layout.py` | yes | X-Plane Gateway apt.dat -> taxi graph -> stand x runway taxi distances (G4) and stand coordinates. |  |
| `scripts/ext_weather.py` | yes | METAR-derived weather table (G5): units, sky/visibility, precipitation codes, de-icing/frost flags, rolling snow, as-of join with crosswind. |  |
| `scripts/fallback_context.py` | yes | Fallback (schedule-copy) detector with stand-timeline and operations-context inputs. |  |
| `scripts/fetch_external.py` | yes | Download the open external sources into PRC_EXTERNAL (default external/). Safe to re-run: finished files are skipped, partial downloads are never kept (temp file + rename), each source is isolated, METAR falls back from … |  |
| `scripts/final_rank.py` | yes | The rank-population size the layer builds assert against. |  |
| `scripts/flightnorm.py` | yes | Universal flight-number / callsign normaliser (2026-10-07: "normalising the callsign / flight number should be a universal preprocess ... discover all similarity ... even fuzzy ... airlines add 'D' for delayed flights, … |  |
| `scripts/model_reload.py` |  | Reload a persisted M3rpa package and reproduce its clean expert predictions. |  |
| `scripts/nbr_set.py` | yes | Neighbour-set tokens: the airport's movement timeline around each departure, for a set-encoder residual member. |  |
| `scripts/nested_features.py` | yes | Build feature v3 with row-wise seasonal-pair OOF target features and corrected semantics. |  |
| `scripts/nested_rich_base.py` |  | Fully nested rich-base diagnostic against the strict v18 protocol. |  |
| `scripts/nm_backfill.py` | yes | NM flight-plan backfill for NM-unmatched departures from their destination arrival records. |  |
| `scripts/ops_context.py` | yes | G11 operations context: contemporaneous taxi levels and schedule-copy activity at the airport. |  |
| `scripts/prc_config.py` | yes | Every location the pipeline reads or writes, in one place. Override any of them with an environment variable. |  |
| `scripts/queue_context.py` | yes | G12 queue and pushback-cohort context (rank-available), complementing G11's take-off windows. |  |
| `scripts/rescore_mixture.py` | yes | Re-score an existing two-stage OOF run under deterministic p_fallback scaling. |  |
| `scripts/residual_variants.py` | yes | AOBT_3-residual expert variants for matched rows under the strict blocked-pair protocol. |  |
| `scripts/rich_strict_gate.py` |  | Outer-safe rich features layered on the strict blocked-pair gate. |  |
| `scripts/stand_timeline.py` | yes | G10 same-stand timeline features from arrival in-block times (fully known in the ranking file). |  |
| `scripts/stand_type.py` | yes | Stand type per airport stand from the X-Plane Gateway packs. | `py -3.14 scripts/stand_type.py` |
| `scripts/strict_tail_gate.py` | yes | Blocked-pair gate diagnostic with no target-derived predictor crossing the block. |  |
| `scripts/submit.py` | yes | Daily submission batches: one file per approach, manifest, log; fitted models saved under runs/models/<approach>_<ts>/. |  |
| `scripts/surge_context.py` | yes | Surge and gap context for tree models (2026-09-19). |  |
| `scripts/tail_experts.py` | yes | Three-expert tail mixture: normal taxi / schedule default / +24 h date bug. |  |
| `scripts/tail_gate.py` | yes | Cost-sensitive fallback gate diagnostics on nested-clean OOF predictions. |  |
| `scripts/train_eval.py` | yes | Model zoo on the fixed fold protocol, including residual LightGBM/XGBoost experts. |  |
| `scripts/tune_residual.py` | yes | Hyperparameter search for the matched-row AOBT_3 residual expert (strict blocked pairs). |  |

## Stage 1: external tables, base and feature tables (10–42)

| file | final path | what it does | usage |
|---|:-:|---|---|
| `scripts/10_build_base.py` | yes |  |  |
| `scripts/20_ext_aircraft.py` | yes |  |  |
| `scripts/20_ext_calendar.py` | yes |  |  |
| `scripts/20_ext_eurocontrol.py` | yes |  |  |
| `scripts/20_ext_layout.py` | yes |  |  |
| `scripts/20_ext_weather.py` | yes |  |  |
| `scripts/30_detect.py` | yes |  |  |
| `scripts/35_encode.py` | yes |  |  |
| `scripts/36_build_nested.py` | yes |  |  |
| `scripts/37_build_v4_stand.py` | yes | features v4 = features v3 + G10 same-stand timeline (scripts/stand_timeline.py). | `py -3.14 scripts/37_build_v4_stand.py` |
| `scripts/38_build_v5_context.py` | yes | features v5 = features v4 + G11 operations context (scripts/ops_context.py). | `py -3.14 scripts/38_build_v5_context.py` |
| `scripts/39_build_v6_queue.py` | yes | features v6 = features v5 + G12 queue and pushback-cohort context (scripts/queue_context.py). | `py -3.14 scripts/39_build_v6_queue.py` |
| `scripts/40_assemble.py` | yes |  |  |
| `scripts/41_build_v7_backfill.py` | yes | features v7 = features v5 with NM columns backfilled for unmatched departures (scripts/nm_backfill.py). | `py -3.14 scripts/41_build_v7_backfill.py` |
| `scripts/42_rank_continuity.py` | yes | Rank-side continuity pass (continuity rule): rebuild ONLY the rank-side tables so that | `py -3.14 scripts/42_rank_continuity.py` |

## Research history of the experts and gates (50–97); A27 comes from 96

| file | final path | what it does | usage |
|---|:-:|---|---|
| `scripts/50_train_eval.py` |  |  |  |
| `scripts/70_blend.py` |  |  |  |
| `scripts/72_rescore_mixture.py` |  |  |  |
| `scripts/73_tail_gate.py` |  |  |  |
| `scripts/75_multifold_tail_gate.py` |  | Six-pair leave-one-pair-out validation for the cost-weighted tail gate. |  |
| `scripts/77_strict_tail_gate.py` |  | Run the strict blocked-pair base/gate diagnostic. |  |
| `scripts/79_airport_gate.py` |  | Run hierarchical airport-specific strict-gate diagnostics. |  |
| `scripts/79_nested_rich_base.py` |  | Run the fully nested rich-base comparison against strict v18. |  |
| `scripts/79_outer_safe_rich_gate.py` |  | Run the fully outer-safe rich-gate experiment; diagnostic only. |  |
| `scripts/79_outer_safe_rich_search.py` |  | Ablate outer-safe rich gate feature families against strict v18. |  |
| `scripts/79_strict_guardrails.py` |  | Evaluate cross-block calibration and maximum-change caps for the strict gate. |  |
| `scripts/80_pipeline_reset.py` |  | Reproducible strict v18 ledger, source parity audit, and bounded signal probes. |  |
| `scripts/82_forward_validation.py` |  | Chronological 2025 backtest with a source-availability stress variant. |  |
| `scripts/86_tail_semantics.py` |  | Read-only timestamp and source breakdown for extreme strict OOF errors. |  |
| `scripts/87_strict_asof_blend.py` |  | Strict six-pair as-of source ablation and one fixed blend with v18. |  |
| `scripts/88_three_expert_tail.py` |  | Three-expert tail mixture: strict diagnostic, v18 reproduction check, deployment and submission. | `py -3.14 scripts/88_three_expert_tail.py --retrain-blocks`<br>`py -3.14 scripts/88_three_expert_tail.py --diagnostic`<br>`py -3.14 scripts/88_three_expert_tail.py --reproduce-v18`<br>`py -3.14 scripts/88_three_expert_tail.py --fit-rank --submit` |
| `scripts/89_stand_v4.py` |  | features v4 (G10 stand timeline) under the three-expert tail: strict frames, diagnostic, deployment. | `py -3.14 scripts/89_stand_v4.py --build-frames`<br>`py -3.14 scripts/89_stand_v4.py --diagnostic`<br>`py -3.14 scripts/89_stand_v4.py --fit-rank --submit` |
| `scripts/91_label_artifact_audit.py` |  | Label-artifact audit of every 2025 departure off-block, priced with v21 strict OOF errors. | `py -3.14 scripts/91_label_artifact_audit.py` |
| `scripts/96_deploy_candidate.py` |  | Deploy a strict-validated three-expert candidate to a ranking submission. | `py -3.14 scripts/96_deploy_candidate.py --approach A24 --reuse-package runs/models/A20_20260917_005317 \` |
| `scripts/97_evidence.py` |  | Promotion evidence for a strict six-pair OOF candidate against a reference strict OOF. | `py -3.14 scripts/97_evidence.py --candidate runs/oof/<cand>.parquet --reference runs/oof/<ref>.parquet \` |

## Members, ADS-B and identity gates (104–132)

| file | final path | what it does | usage |
|---|:-:|---|---|
| `scripts/104_residual_weights.py` | yes | Per-airport, block-cross-fitted weights for the three matched-row residual members (strict frames). | `py -3.14 scripts/104_residual_weights.py --frames-suffix _v5tfb5 --out-suffix _v5tfb5rw --json runs/residual_weights_A27_20260917.json` |
| `scripts/106_member_probe.py` | yes | Does a fourth matched-row residual member of a different kind add anything to the rd blend? (block probe) | `py -3.14 scripts/106_member_probe.py --block AB --candidates xgb_d10 lgb_extra` |
| `scripts/107_extra_members.py` | yes | Add residual members of another kind (106_member_probe.py candidates) to A31's per-airport matched-row blend. | `py -3.14 scripts/107_extra_members.py weights --members lgb_extra --tag A32 --json runs/residual_weights_A32_20260918.json`<br>`py -3.14 scripts/97_evidence.py --candidate runs/oof/extra_members_A32_strict.parquet \`<br>`py -3.14 scripts/107_extra_members.py deploy --members lgb_extra --approach A32 --weights runs/residual_weights_A32_20260918.json \` |
| `scripts/108_matched_detector.py` | yes | Schedule-copy detector for NM-matched rows (A34 matched lane): how much is left, and does a set-token classifier help? | `py -3.14 scripts/108_matched_detector.py oracle`<br>`py -3.14 scripts/108_matched_detector.py probe --block AB --candidates set_clf` |
| `scripts/110_unmatched_normal.py` | yes | A set-encoder normal-taxi expert for departures without an NM match (no AOBT_3): sizing on the strict blocks. | `py -3.14 scripts/110_unmatched_normal.py check`<br>`py -3.14 scripts/110_unmatched_normal.py probe --block AB --candidates nm_masked_set` |
| `scripts/111_bundle_a35.py` | yes | A35 = A34 + a set-encoder normal expert for NM-unmatched rows + a set-classifier mix in the matched copy probability. | `py -3.14 scripts/111_bundle_a35.py strict`<br>`py -3.14 scripts/111_bundle_a35.py deploy [--submit]` |
| `scripts/113_bundle_a36.py` | yes | A36 = A35 with the surge context (2026-09-19) in the matched-row member blend. | `py -3.14 scripts/107_extra_members.py weights --members mlp+mlp_s1+mlp_s2+mlp_e20+mlp_wide mlp_set+mlp_set2+mlp_set_s1 \`<br>`py -3.14 scripts/113_bundle_a36.py check`<br>`py -3.14 scripts/113_bundle_a36.py strict`<br>`py -3.14 scripts/107_extra_members.py deploy ... --approach A36m`<br>`py -3.14 scripts/113_bundle_a36.py deploy --base-package runs/models/A36m_<stamp> [--submit]` |
| `scripts/114_unmatched_mode.py` | yes | Unmatched mode model: a set-token schedule-copy classifier for NM-unmatched departures (2026-09-19 evening). | `py -3.14 scripts/114_unmatched_mode.py fit --block AB --candidates u_clf u_clf2 u_self`<br>`py -3.14 scripts/114_unmatched_mode.py eval`<br>`py -3.14 scripts/114_unmatched_mode.py rule`<br>`py -3.14 scripts/114_unmatched_mode.py deploy --base-package runs/models/A36_<stamp> [--submit]` |
| `scripts/118_adsb_probe.py` | yes | ADS-B ground coverage probe: can open adsb.lol traces see the push-back? (one day, 2025 labels only) | `py -3.14 scripts/118_adsb_probe.py extract --date 2025-07-15 --keep-raw`<br>`py -3.14 scripts/118_adsb_probe.py match   --date 2025-07-15`<br>`py -3.14 scripts/118_adsb_probe.py coverage --date 2026-07-20 --flights ITY674 QFA7006` |
| `scripts/119_adsb_download.py` | yes | Download adsb.lol globe_history daily releases (ODbL-1.0) for date ranges into <PRC_ADSB_POINTS>/raw. | `py -3.14 scripts/119_adsb_download.py 2026-07-01:2026-07-31 2026-01-01:2026-01-31 --workers 4 --extract`<br>`py -3.14 scripts/119_adsb_download.py recommended --workers 4 --extract`<br>`py -3.14 scripts/119_adsb_download.py 2025-07-15 --dry-run`<br>`py -3.14 scripts/119_adsb_download.py 2025-05-28:2025-06-12 2025-10-15 2025-12-31 --preferred --replace --extract` |
| `scripts/120_adsb_stand_events.py` |  | ADS-B stand-departure events (2026-09-19): off-block = the first real move away from the stand. | `py -3.14 scripts/120_adsb_stand_events.py 2025-07-15`<br>`py -3.14 scripts/120_adsb_stand_events.py 2025-07-15 --dep-stands` |
| `scripts/121_adsb_events.py` | yes | ADS-B off-block events for every extracted day -> one feature table for the challenge departures. | `py -3.14 scripts/121_adsb_events.py stands   --workers 6`<br>`py -3.14 scripts/121_adsb_events.py events   --workers 6`<br>`py -3.14 scripts/121_adsb_events.py features`<br>`py -3.14 scripts/121_adsb_events.py neighbours`<br>`py -3.14 scripts/121_adsb_events.py check` |
| `scripts/122_adsb_shadow.py` | yes | Where ADS-B loses sight of the ground, and what the track shows of the taxi we DID see. | `py -3.14 scripts/122_adsb_shadow.py geometry --workers 6`<br>`py -3.14 scripts/122_adsb_shadow.py shadow`<br>`py -3.14 scripts/122_adsb_shadow.py check` |
| `scripts/124_unmatched_adsb.py` | yes | ADS-B for the departures WITHOUT an NM match: a twin-difference correction of the unmatched normal expert (2026-09-22). | `py -3.14 scripts/124_unmatched_adsb.py stops [--workers 12]`<br>`py -3.14 scripts/124_unmatched_adsb.py fit --block AB [--seeds 41 42 43]`<br>`py -3.14 scripts/124_unmatched_adsb.py rule --tag A46f --bundle runs/oof/bundle_A45f_strict.parquet --base-rule runs/unmatched_mode_rule_A45f.json`<br>`py -3.14 scripts/124_unmatched_adsb.py deploy --base-package runs/models/A45f_<stamp> --rule runs/unmatched_adsb_rule_A46f.json --evidence <json> --submit` |
| `scripts/126_flight_identity_gate.py` | yes | Fully nested LIRF flight-identity correction for the final A48 copy gate. | `py -3.14 scripts/126_flight_identity_gate.py probe`<br>`py -3.14 scripts/126_flight_identity_gate.py deploy`<br>`py -3.14 scripts/126_flight_identity_gate.py build-v43` |
| `scripts/128_corrected_identity_gate.py` | yes | A50/v44: correct matched-mixture identity gate with nested blend calibration. |  |
| `scripts/129_identity_search.py` | yes | Current-data A51 research: nested alternatives to the shipped A50 gate. |  |
| `scripts/130_source_identity_search.py` | yes | Source-component and schedule-family gate probe, existing data only. |  |
| `scripts/131_outer_identity_detector.py` | yes | Eight-month, identity-aware copy detector added to a nested A50 gate. |  |
| `scripts/132_v45_identity_bundle.py` |  | Validate and package A51/v45 with inner-only strategy selection. |  |

## Motion expert (157–173)

| file | final path | what it does | usage |
|---|:-:|---|---|
| `scripts/157_history_normal.py` | yes | Past-only all-airport normal-expert screen for v49; never writes a submission. |  |
| `scripts/164_ground_motion_candidates.py` | yes | Extract new off-block candidates from existing ground trajectories. |  |
| `scripts/165_ground_motion_full.py` | yes | Freeze and extract sustained-motion features from all existing cached dates. |  |
| `scripts/166_motion_nested_validation.py` | yes | Training-only nested evaluation of sustained-motion measurements. |  |
| `scripts/170_motion_tree_expert.py` | yes | Fixed paired tree control for new motion features; nested training-only test. |  |
| `scripts/173_v49_motion_bundle.py` | yes | The sustained-motion normal expert (A57) on the v45 matched component. | `py -3.14 scripts/173_v49_motion_bundle.py fit` |

## ADS-B / OPDI tables and links (213–260)

| file | final path | what it does | usage |
|---|:-:|---|---|
| `scripts/213_turnaround_features.py` | yes | Turnaround-link features: fixed same-stand rule + ADS-B aircraft-identity inbound. | `py -3.14 scripts/213_turnaround_features.py --workers 6` |
| `scripts/231_turnaround_links.py` | yes | Best arrival<->departure link per departure, for the explorer (and anyone who needs the pairs themselves). | `py -3.14 scripts/231_turnaround_links.py` |
| `scripts/232_adsb_movement_times.py` | yes | ADS-B timestamps and positions per movement, for the explorer. | `py -3.14 scripts/232_adsb_movement_times.py --workers 6` |
| `scripts/250_nm_backfill.py` | yes | NM flight-plan times for NM-unmatched departures, taken from the destination airport's arrival record. | `py -3.14 scripts/250_nm_backfill.py` |
| `scripts/251_v51_nm_window.py` | yes | v51 = v49 + two label-free constraints from the dataset builder's NM matching rule (2026-09-25). | `py -3.14 scripts/251_v51_nm_window.py evidence`<br>`py -3.14 scripts/251_v51_nm_window.py build` |
| `scripts/255_adsb_match_v2.py` | yes | ADS-B take-off matching v2: normalized callsigns in pass 1, no cross-airline pass-2 matches (2026-09-26). | `py -3.14 scripts/255_adsb_match_v2.py compare`<br>`py -3.14 scripts/255_adsb_match_v2.py features` |
| `scripts/256_v52_recovered_lobt.py` | yes | v52 = v51 + rule R, a third label-free constraint from the dataset builder's NM match rule (2026-09-26). | `py -3.14 scripts/256_v52_recovered_lobt.py ledger`<br>`py -3.14 scripts/256_v52_recovered_lobt.py build` |
| `scripts/257_opdi_download.py` | yes | Download OPDI v002 (EUROCONTROL PRU Open Performance Data Initiative) parquet files into PRC_OPDI_RAW (scripts/prc_config.py). | `py -3.14 scripts/257_opdi_download.py`<br>`py -3.14 scripts/257_opdi_download.py --dry-run`<br>`py -3.14 scripts/257_opdi_download.py --datasets flight_events --workers 3` |
| `scripts/258_opdi_extract.py` | yes | Cut the downloaded OPDI v002 files (scripts/257) down to what touches our ten airports. | `py -3.14 scripts/258_opdi_extract.py`<br>`py -3.14 scripts/258_opdi_extract.py --force` |
| `scripts/259_opdi_flights.py` | yes | One row per OPDI flight x our airport x role (dep / arr), from the extracted events of scripts/258. | `py -3.14 scripts/259_opdi_flights.py` |
| `scripts/260_opdi_link.py` | yes | Link our movements (2025 training + 2026 ranking, departures and arrivals) to OPDI flights (scripts/259). | `py -3.14 scripts/260_opdi_link.py` |

## Layers of the final chain (262–281)

| file | final path | what it does | usage |
|---|:-:|---|---|
| `scripts/262_opdi_correction.py` | yes | OPDI off-block correction: a per-airport residual model on the OPDI ground events (lead #3 of LEADS_20261006). | `py -3.14 scripts/262_opdi_correction.py frame`<br>`py -3.14 scripts/262_opdi_correction.py nested --apt LSZH`<br>`py -3.14 scripts/262_opdi_correction.py ablate --apt LSZH`<br>`py -3.14 scripts/262_opdi_correction.py fit --apt LSZH` |
| `scripts/263_v53_build.py` | yes | v53 = v52 + the LSZH OPDI off-block correction (scripts/262) + the +-1 h band rule + the VFR guard. | `py -3.14 scripts/263_v53_build.py evidence`<br>`py -3.14 scripts/263_v53_build.py build` |
| `scripts/264_v54_unmatched_gate.py` | yes | Lead #17: label-free airport-recording signals for the unmatched copy gate (v54 candidate). | `py -3.14 scripts/264_v54_unmatched_gate.py table`<br>`py -3.14 scripts/264_v54_unmatched_gate.py nested`<br>`py -3.14 scripts/264_v54_unmatched_gate.py fit` |
| `scripts/265_v54_build.py` | yes | v54 = v53 + the EDDF stand-level OPDI correction (scripts/262 stand_nested / stand_fit), NM-window rules re-applied. | `py -3.14 scripts/265_v54_build.py evidence`<br>`py -3.14 scripts/265_v54_build.py build`<br>`py -3.14 scripts/265_v54_build.py ledger` |
| `scripts/266_gate_retention_screen.py` | yes | Gate-retention / runway-cadence screen on the v54 2025 proxy (idea read in team outgoing-boat's public repo, src/pista.py). | `py -3.14 scripts/266_gate_retention_screen.py frame`<br>`py -3.14 scripts/266_gate_retention_screen.py nested` |
| `scripts/267_v55_build.py` | yes | v55 = v54 + the gate-retention / overtake residual correction (scripts/266 arm 'ret6ov') + tightened NM-window edges. | `py -3.14 scripts/267_v55_build.py rank`<br>`py -3.14 scripts/267_v55_build.py evidence`<br>`py -3.14 scripts/267_v55_build.py build`<br>`py -3.14 scripts/267_v55_build.py ledger` |
| `scripts/268_v56_stand_reuse.py` | yes | Stand reuse (lead #19, OpenAir's E45 "stand release" idea) for LIRF NM-unmatched departures: the v57 layer (A65). | `py -3.14 scripts/268_v56_stand_reuse.py table`<br>`py -3.14 scripts/268_v56_stand_reuse.py layer` |
| `scripts/269_neighbour_lateness.py` | yes | Lead #21: departure-neighbour lateness (team versatile-violin's groups L / G, arnavhm/prc_26) on the v55 2025 proxy. | `py -3.14 scripts/269_neighbour_lateness.py frame`<br>`py -3.14 scripts/269_neighbour_lateness.py nested --arm nb` |
| `scripts/270_v56_build.py` | yes | v56 = v55 + the departure-neighbour residual correction (scripts/269, lead #21) + LEMD 3,606 s cap (+ BEL hedge). | `py -3.14 scripts/270_v56_build.py rank --arm nb`<br>`py -3.14 scripts/270_v56_build.py evidence --arm nb`<br>`py -3.14 scripts/270_v56_build.py build [--bel 0.65]` |
| `scripts/274_unmatched_context_layer.py` | yes | v63 = v57 + a same-period context layer on LIRF NM-unmatched departures 1-8 h late. | `py -3.14 scripts/274_unmatched_context_layer.py frame`<br>`py -3.14 scripts/274_unmatched_context_layer.py screen`<br>`py -3.14 scripts/274_unmatched_context_layer.py build`<br>`py -3.14 scripts/274_unmatched_context_layer.py ledger` |
| `scripts/278_v66_bundle.py` | yes | v66 = v63 + L1 push-phase residual layer + L2 LIRF copy layer (v66 bundle, built 2026-10-07). | `py -3.14 scripts/278_v66_bundle.py l1frame`<br>`py -3.14 scripts/278_v66_bundle.py l1screen --mode sixfold|forward`<br>`py -3.14 scripts/278_v66_bundle.py l2screen`<br>`py -3.14 scripts/278_v66_bundle.py build` |
| `scripts/279_overnight_layers.py` | yes | Overnight 2026-10-08 layers that passed the pass rule (runs/overnight_20261008/ov.py judge), each a separate step on v66. | `py -3.14 scripts/279_overnight_layers.py build --item 5`<br>`py -3.14 scripts/279_overnight_layers.py build --item 4`<br>`py -3.14 scripts/279_overnight_layers.py build --item 2`<br>`py -3.14 scripts/279_overnight_layers.py build --item 11` |
| `scripts/281_long_gate.py` | yes | v68 = A79 = v67 + the long-label gate outside LIRF (a separate gate for the > 1 h rows). | `py -3.14 scripts/281_long_gate.py build` |

## tools/

| file | final path | what it does | usage |
|---|:-:|---|---|
| `tools/adsb/chain_final.sh` | yes | The ADS-B / OPDI derived tables for every day that has an extracted points file (docs/REPRODUCE.md, stage 2). usage: tools/adsb/chain_final.sh <prefix> [<day list file>] (no list = every complete day, 2025-01-01 on) lane 1: … |  |
| `tools/adsb/coverage.py` |  | Coverage of the rebuilt ADS-B / OPDI tables on the FINAL ranking departures, by month x airport. | `py -3.14 tools/adsb/coverage.py` |
| `tools/adsb/edge_check.py` |  | Month-edge check (continuity rule: windows / links use the neighbouring days present in the data). | `py -3.14 tools/adsb/edge_check.py` |
| `tools/adsb/quality_splice.py` | yes | Keep the reference 2025 rows of adsb_quality.parquet (run after every `121 quality`, before `121 features` / `255 features`). | `py -3.14 tools/adsb/quality_splice.py` |
| `tools/adsb/run_step.sh` | yes | usage: tools/adsb/run_step.sh <name> <script> [args...] Runs $PY scripts/<script> from the repository root with <= 6 threads, logs to $PRC_WORK/logs/adsb/<name>.log and appends a timing row to $PRC_WORK/logs/adsb/timings.tsv. … |  |
| `tools/final_chain.py` | yes | The final chain: v45 (the member composition, tools/members) + the motion correction + every validated layer, then the long-label gate, then (switchable) the 15 manual rows, then the export in the template order. | `py -3.14 tools/final_chain.py --variant A15 --v45 submissions_rebuild/v45_A15_<stamp>.parquet \\` |
| `tools/layers/build_frames.py` | yes | Label-free, base-prediction-independent layer frames for the final file (670,790 rank rows), built once before the final chain (tools/final_chain.py) fills in the base prediction. | `py -3.14 tools/layers/build_frames.py static rows66 f266 f269 ovcache item05 item04 item02 item11 longrows lirfbase`<br>`py -3.14 tools/layers/build_frames.py motion` |
| `tools/layers/final_report.py` | yes | Report of a final-chain run: rows moved per step, the distance to its v45, and the mean / median prediction by airport and month. Light (reads the cvNN files only). | `py -3.14 tools/layers/final_report.py A15` |
| `tools/layers/rt.py` | yes | Runtime helpers of the final chain (tools/final_chain.py, tools/layers/build_frames.py): thread caps, memory gate, train-row splicing, small IO helpers. |  |
| `tools/members/a27_members.py` | yes | A27 package inputs for the final rows (docs/REPRODUCE.md, stage 3). | `py -3.14 tools/members/a27_members.py m3rpa`<br>`py -3.14 tools/members/a27_members.py detector`<br>`py -3.14 tools/members/a27_members.py rd`<br>`py -3.14 tools/members/a27_members.py gate` |
| `tools/members/lgb_members.py` | yes | LightGBM members of the A48 blend and the unmatched twins, refitted once and scored on NEW and OLD rank rows. | `py -3.14 tools/members/lgb_members.py surge`<br>`py -3.14 tools/members/lgb_members.py member lgb_sg [...]`<br>`py -3.14 tools/members/lgb_members.py twins --part blind`<br>`py -3.14 tools/members/lgb_members.py twins --part ab --adsb <folder>` |
| `tools/members/make_probe106_ext.py` | yes | Write tools/members/probe106_ext.py = scripts/106_member_probe.py + an extra-scoring hook (no training change). |  |
| `tools/members/mcommon.py` | yes | Shared helpers of the v45 member tools (docs/METHOD.md, "v45 member composition"). |  |
| `tools/members/nn_members.py` | yes | Neural members of v45 refitted once on the GPU and scored on NEW and OLD rank rows (docs/METHOD.md). | `py -3.14 tools/members/nn_members.py mlp_set mlp_set2 ...`<br>`py -3.14 tools/members/nn_members.py set_clf nm_masked_set ...`<br>`py -3.14 tools/members/nn_members.py u_self u_self_s1 u_self_s2` |
| `tools/members/probe106_ext.py` | yes | Does a fourth matched-row residual member of a different kind add anything to the rd blend? (block probe) | `py -3.14 scripts/106_member_probe.py --block AB --candidates xgb_d10 lgb_extra` |
| `tools/members/recompose_v45.py` | yes | Recompose v45 (A51) from its member predictions, exactly as the reference package chain did (docs/METHOD.md). | `py -3.14 tools/members/recompose_v45.py main`<br>`py -3.14 tools/members/recompose_v45.py rebuild --rank new|old [--allow-missing]` |
| `tools/members/rescore.py` | yes | Re-predict every v45 input for the 670,790 final rows from SAVED models only (no refit), for variant A or B. | `py -3.14 tools/members/rescore.py --variant B [--check]`<br>`py -3.14 tools/members/rescore.py --variant A [--check]` |
| `tools/members/run_a15.sh` | yes | Stage 3 of docs/REPRODUCE.md: every trained input of v45 for the final rows, variant A, unmatched twins over 15 seeds, then the composition -> $PRC_WORK/submissions_rebuild/v45_A15_<stamp>.parquet (+ _detail.parquet). One … |  |
| `tools/members/twin_seed_noise.py` | yes | How much of the unmatched twins' seed noise reaches v45 (sizing for v73 = more twin seeds). | `py -3.14 tools/members/twin_seed_noise.py [--base-dir runs/models/final_A] [--seeds 41 42 43]` |

## tools/last_day/ (post-steps on the chain's file, docs/METHOD.md §11)

| file | final path | what it does | usage |
|---|:-:|---|---|
| `tools/last_day/build_A15L.py` | yes | A15L = A15 + two fixed rules, each through rules_v55: c0  the 8-13 h rule (c0_long_delay_rule.py) on every final row it selects, rows of the manual step untouched; c3  p <= D for D > 2 h outside the L … | `py -3.14 tools/last_day/build_A15L.py` |
| `tools/last_day/build_A15LX.py` | yes | A15LX = A15L + the combined EUROCONTROL daily layer (c15_combined.py): up to three passes + the IFR / non-heli identity + the count-gap target K_adj recomputed in every pass. | `py -3.14 tools/last_day/build_A15LX.py` |
| `tools/last_day/build_A15LXR.py` | yes | A15LXR = A15LX (A15L + the combined daily layer) followed by the re-centred LIRF copy / held enumeration computed on the stack output (c16_stack_recentred.py). | `py -3.14 tools/last_day/build_A15LXR.py --idea4 0`<br>`py -3.14 tools/last_day/build_A15LXR.py --idea4 1` |
| `tools/last_day/build_A15LXRH.py` | yes | A15LXRH = A15LXR + the hardened midnight-crossers step (c18_crossers_hardened.py). | `py -3.14 tools/last_day/build_A15LXRH.py` |
| `tools/last_day/build_A15LXRHDS.py` | yes | A15LXRHDS = A15LXRH + the LIRF date-bug / schedule-copy rules (c19_lirf_datebug_rules.py). This is the final file. | `py -3.14 tools/last_day/build_A15LXRHDS.py` |
| `tools/last_day/c0_long_delay_rule.py` | yes | The 8-13 h rule on every final row it selects (the rule of the chain's manual step, `8-13h`, METHOD.md §9): LIRF, NM-unmatched, gate-valid, no helicopters; 8 h < D <= 13 h, or D > 13 h with a same-loc … | `py -3.14 tools/last_day/c0_long_delay_rule.py` |
| `tools/last_day/c10_daily_v2.py` | yes | EUROCONTROL daily reconciliation layer on integer arrays: the shared machinery of the combined layer (c15_combined.py) and of the LIRF copy / held enumeration (c13_recentred_v2.py, c16_stack_recentred.py). |  |
| `tools/last_day/c13_recentred_v2.py` | yes | LIRF copy / held enumeration with the day residual re-centred per month (functions used by the post-steps). |  |
| `tools/last_day/c14_count_gap.py` | yes | Count-gap adjustment of the EUROCONTROL daily target and the IFR / non-helicopter row filter (functions used by the combined layer, c15_combined.py). |  |
| `tools/last_day/c15_combined.py` | yes | The combined EUROCONTROL daily reconciliation layer ("stack"): version 1 (c9_daily_layer.py) + (1) up to three passes + (2) the IFR / non-helicopter identity + (3) the count-gap target K_adj recompute … | `py -3.14 tools/last_day/c15_combined.py --tag x` |
| `tools/last_day/c16_stack_recentred.py` | yes | The combined stack (c15_combined.py) followed by the re-centred LIRF copy / held enumeration (c13_recentred_v2.py), computed on the stack output. | `py -3.14 tools/last_day/c16_stack_recentred.py --tag x` |
| `tools/last_day/c17_crossers.py` | yes | Midnight-crossers machinery (functions used by the hardened step, c18_crossers_hardened.py, and its build). |  |
| `tools/last_day/c18_crossers_hardened.py` | yes | The hardened midnight-crossers step, applied after the stack + re-centred enumeration (base = A15LXR). | `py -3.14 tools/last_day/c18_crossers_hardened.py --tag h` |
| `tools/last_day/c19_lirf_datebug_rules.py` | yes | LIRF date-bug / schedule-copy rules R0–R6 for LIRF NM-unmatched departures taking off on a later local day; 2025 in-sample counts. | `py -3.14 tools/last_day/c19_lirf_datebug_rules.py` |
| `tools/last_day/c3_cap_at_D.py` |  | The cap p <= D (take-off - schedule) for D > 2 h, unless the row carries date-bug mass (LIRF NM-unmatched next-local-day rows = tail_experts.b_domain). A fixed rule, so the six-fold and forward predic … | `py -3.14 tools/last_day/c3_cap_at_D.py` |
| `tools/last_day/c9_daily_layer.py` | yes | EUROCONTROL daily reconciliation layer, version 1 (one pass, all rows). Own implementation; the public descriptions of teams gracious-tornado and reliable-hamburger were read for the idea only, no code copied. | `py -3.14 tools/last_day/c9_daily_layer.py` |
| `tools/last_day/c9_step1_identity.py` | yes | EUROCONTROL daily identity check: does 60 x DLY_ALL_PRE_2 (EUROCONTROL daily all pre-departure delay, min) equal sum max(0, BLOCK - SCHED) = sum max(0, D - y) of our 2025 departures per airport-day? T … | `py -3.14 tools/last_day/c9_step1_identity.py` |
| `tools/last_day/ld.py` | yes | Post-steps on the chain's final file (docs/METHOD.md §11): shared paths, loaders, the dataset rules and the 2025 judge. |  |
| `tools/last_day/run_last_day.py` | yes | The post-steps after the final chain, in order: A15 -> A15L -> A15LX -> A15LXR -> A15LXRH -> A15LXRHDS (docs/METHOD.md §11, docs/REPRODUCE.md §7). Each step is its own script; this runs them one after the other wi … | `py -3.14 tools/last_day/run_last_day.py`<br>`py -3.14 tools/last_day/run_last_day.py --from c18`<br>`py -3.14 tools/last_day/run_last_day.py --no-checks` |

## runs/ (layer scripts that live next to their outputs)

| file | final path | what it does | usage |
|---|:-:|---|---|
| `runs/lirf_row_audit/lirf_rows.py` | yes | LIRF departures, both years, one row each with raw record fields + the v57 proxy + ADS-B link. -> lirf_rows.parquet |  |
| `runs/long_rows_20261008/audit.py` | yes | The errors come mostly from rows with y > 1 h: what are they, can they be told apart beforehand? |  |
| `runs/long_rows_20261008/gate.py` | yes | Step 3: a separate "long-label gate" on the population that can contain long labels: departures whose take-off is >= 50 min after schedule (98 % of all 2025 long labels; 286k rows, 1.4 % long). |  |
| `runs/long_rows_20261008/gate3_no_lirf.py` | yes | Step 4: four-way gate on the late-vs-schedule population (take-off >= 50 min after schedule), exact copy algebra. |  |
| `runs/long_rows_20261008/offsets.py` | yes | Step 2: for long rows not explained by a plain copy, what does the recorded block time equal? Offsets of the block from: schedule (whole hours = local-time / timezone shifts), LOBT / EOBT_1 / AOBT_3 (+-1 h, +-2 h), the inbound … |  |
| `runs/long_rows_20261008/stack_v67.py` | yes | Step 5: the non-LIRF long gate (alpha 0.25) stacked on the v67 proxy (v66 proxy + the four v67 layers, combo_oof). Gate move = 0.25 (p_gate - v66), added to v67's move on shared rows, rules_v55 once. Judged against the v67 proxy. |  |
| `runs/overnight_20261008/combo_eval.py` |  | The four overnight layers (279 items 5, 4, 2, 11; A74-A77) combined into one v67 (A78). Check on 2025 that the four layers stacked on the v66 proxy (moves added where two layers overlap, rules_v55 once on the sum) still pass: … |  |
| `runs/overnight_20261008/item02_gaps.py` | yes | Item 2 (lead #9): LTFM month-end cut + gap-day flags as residual layers on the v66 proxy. |  |
| `runs/overnight_20261008/item04_park_extract.py` | yes | Item 4 step 1: the parked position of every v2-matched departure = its first ADS-B ground point (push_phases t_first), plus the median position of its first 5 ground points within 30 s. Label-free. -> item04_park.parquet |  |
| `runs/overnight_20261008/item04y_stand_error.py` | yes | Item 4 (lead #10): stand-error flag = ADS-B parked position of the departure vs the learned location of its RECORDED stand; flag > 60 m. Learned location = per (airport, stand) median parked position over its departures that … |  |
| `runs/overnight_20261008/item05_dateshift.py` | yes | Item 5 (lead #12): date-shift / recorded-time-shift detection by ADS-B within +-24 h. |  |
| `runs/overnight_20261008/item11_arr_tail_extract.py` | yes | Item 11 step 1 (A9, joint coverage shadow): for every arrival record (both years; arrival in-block is given in both), the aircraft's last ADS-B ground point after landing -> tail = recorded in-block - last seen, and that … |  |
| `runs/overnight_20261008/item11_shadow_layer.py` | yes | Item 11 (A9): joint coverage shadow — arrival disappearance informs departure appearance, same period. |  |
| `runs/overnight_20261008/ov.py` | yes | Test harness of the overnight layers: the 2025 v66 proxy + the pass rule for residual layers on it. | `py -3.14 runs/overnight_20261008/ov.py proxy` |
