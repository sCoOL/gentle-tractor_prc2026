# Frozen artefacts

The final chain re-runs every step that touches the final ranking file from the challenge data and open sources. A few
inputs it reads are **not rebuilt by the final chain**: models whose training belongs to the 2025 research history,
fixed composition weights, and the 2025 side of the layer frames (the out-of-fold "proxy" predictions every layer was
validated and fitted on). They are listed here with the script that produced each, so that every one of them can be
re-derived; the final file was built with exactly these files.

They are not in the git repository (size, and most of them hold per-flight rows of the challenge data). They are read
from `PRC_FROZEN` (default `<PRC_WORK>/frozen`, see `scripts/prc_config.py`), in the layout below.

**Download:** [https://vietnamairports-my.sharepoint.com/:f:/g/personal/thonglha_acv_vn/IgCzNW9H2-GZTJqxE1HPhHoVAayFZCgnU5qYnxc6zNYUjYU?e=sQpp4l](https://vietnamairports-my.sharepoint.com/:f:/g/personal/thonglha_acv_vn/IgCzNW9H2-GZTJqxE1HPhHoVAayFZCgnU5qYnxc6zNYUjYU?e=sQpp4l), one folder holding `frozen/` (point `PRC_FROZEN` at it), `data_old/`
(`PRC_DATA_OLD`), `config/manual_rows.csv` (the manual rows of `docs/METHOD.md` §9; copy it into the repository's `config/`), the two EUROCONTROL daily files
the final file was built with (`external/eurocontrol/`; the live 2026 file has grown since) and `MANIFEST.csv` (size and
sha256 of every file). Its `README.txt` gives the steps. The files hold per-flight rows derived from the challenge data,
including 2025 labels, and are shared under the challenge terms.

Sizes are those of the files the final file was built with (the chain's file A15 and the post-steps of
`tools/last_day/`, docs/METHOD.md §11).

| folder under `PRC_FROZEN` | needed by | files | size |
|---|---|---:|---:|
| `ref_layers/` | `tools/layers/build_frames.py`, `tools/final_chain.py` | 45 | 1,475 MB |
| `ref_layers/runs/leads_20261008/rows.parquet` | `tools/last_day/` (the 2025 out-of-fold proxy of the whole chain) | 1 | 163 MB |
| `ref_main/models/A27_20260917_141019/`, `ref_main/models/A51_20260923_010203/` | `tools/members/*`, `tools/final_chain.py` (step 57) | 37 | 409 MB |
| `ref_main/runs/` (3 JSON files of fixed weights / rules) | `tools/members/recompose_v45.py` | 3 | < 0.1 MB |
| `ref_main/adsb_snapshot_0926/` | `tools/members/lgb_members.py`, `nn_members.py` (variant A training rows) | 2 | 95 MB |
| `ref_adsb/` (`stands.parquet`, `adsb_quality.parquet`) | stage 2 (stand map; `tools/adsb/quality_splice.py`) | 2 | 0.2 MB |
| `ref_main/models/member_cache*`, A27 / A31 reference predictions | the member tools' reproduction statistics (read unconditionally by `a27_members.py`, `lgb_members.py`, `nn_members.py`) | 53 | 199 MB |
| `ref_main/models/{A35_20260918_181126, A48_20260922_131958, A48m_20260922_131955, A48f_20260922_132000, A48u_20260922_132003, A51_20260923_010203}/rank_detail.parquet`, `ref_main/submissions/gentle-tractor_v45.parquet` | `tools/members/recompose_v45.py` (`main` and `hybrid` modes; the `rebuild --write` path that builds the final chain's input reads the v45 file only if it is present) | 7 | 212 MB |
| `PRC_DATA_OLD` (rank tables of the earlier Jan/Jul ranking export) | the same tools also score the earlier export (statistics only; the final rows never read it) | 8 | 181 MB |
| **total** | | **158** | **≈ 2.73 GB** |

Largest single file: `ref_layers/runs/long_rows_20261008/frame.parquet`, 206 MB.

## 1. Models and fixed parameters

| file(s) | what | produced by |
|---|---|---|
| `ref_main/models/A27_20260917_141019/M3rpa/000–010_lgb.txt`, `M3rpa/*_meta.json`, `M3rpa/state.json` (371 MB) | the clean per-airport expert (10 airport boosters + 1 global AOBT_3-residual booster) and the helicopter constant | `scripts/96_deploy_candidate.py --approach A27` (→ `tail_experts.fit_full_m3rpa`) |
| `ref_main/models/A27_20260917_141019/gate_lgb.txt`, `summary.json` | the strict tail gate (copy weight `w`) and the date-bug bins (`pb_rates`, `pb_edges`) | same run (→ `tail_experts.rank_prediction`, `tail_gate.fit_predict_gate` on the six strict 2025 frames) |
| `ref_main/models/A51_20260923_010203/{outer_detector_weighted, outer_detector_plain, detector_absolute, detector_residual, detector_binary}/model.txt + state.json`, `history.parquet`, `summary.json` | the LIRF NM-matched identity gate (two outer identity detectors, three gates, alpha 0.7) and its 2025 label history | `scripts/129_identity_search.py`, `130`, `131`, `132_v45_identity_bundle.py validate / build` |
| `ref_main/runs/residual_weights_A48_20260922.json` | per-cell member weights of the matched-row blend (26 airport × ADS-B-state cells; exact simplex, ridge 1e-3) | `scripts/106_member_probe.py` (strict member predictions per block) → `scripts/107_extra_members.py weights --exact --ridge 1e-3` on the 2025 strict out-of-fold frames |
| `ref_main/runs/unmatched_mode_rule_A48f.json` | NM-unmatched mode rule (alpha 0.9, D ≤ 10 h, LTFM 0) | `scripts/114_unmatched_mode.py rule` (2025 strict folds) |
| `ref_main/runs/unmatched_adsb_rule_A48u.json` | NM-unmatched ADS-B twin rule (lam 1, mu 0.75, mu2 0.5, near 200 m) | `scripts/124_unmatched_adsb.py rule --tier2` (2025 strict folds) |
| `ref_layers/runs/models/A57_20260924_v49/motion/model.txt`, `motion/state.json`, `summary.json` (5 MB) | the sustained-motion normal expert (A57) and its alpha 0.5 | `scripts/170_motion_tree_expert.py` (nested evidence) + `scripts/173_v49_motion_bundle.py fit` |
| `ref_main/adsb_snapshot_0926/adsb_features.parquet`, `adsb_stops.parquet` | the 2025 ADS-B v1 tables the 26 ADS-B members of v45 were designed and validated on ("variant A"); the members are retrained on their 2025 rows | `scripts/121_adsb_events.py features` (2026-09-20) and `scripts/124_unmatched_adsb.py stops` (2026-09-22) on the adsb.lol days downloaded before 2026-09-26; 19 days were re-downloaded later, so a fresh download gives slightly different 2025 rows (that is variant B) |
| `ref_adsb/stands.parquet` | the stand map (ADS-B parking clusters + X-Plane stand coordinates); copy it to `<PRC_ADSB_DIR>/stands.parquet` before stage 2 | `scripts/121_adsb_events.py stands` (2026-09-20 day set) |
| `ref_adsb/adsb_quality.parquet` | the 2025 airport-month ADS-B quality rows (`adsb_q_*`) every 2025 frame was built with | `scripts/121_adsb_events.py quality` (its callsign tie-break depends on the hash seed, see `tools/adsb/quality_splice.py`) |

## 2. The 2025 side of the layers (`ref_layers/runs/...`)

Each layer is fitted at build time on its 2025 frame; those frames carry the 2025 out-of-fold predictions of the chain
up to that layer. The final chain keeps them verbatim and rebuilds only the rank rows (`rt.splice_train`).

| file | what | produced by |
|---|---|---|
| `a51_final/chosen_ensemble_oof.parquet` | 2025 out-of-fold prediction of v45 (A51 strict OOF): the base of every 2025 proxy | `scripts/132_v45_identity_bundle.py validate` |
| `v52_recovered_lobt/rows.parquet` | 2025 proxy after the W/F/R rules | `scripts/256_v52_recovered_lobt.py ledger` |
| `v53_opdi/oof_LSZH.parquet`, `nested_LSZH.json`, `oof_EDDF_stand.parquet`, `nested_EDDF_stand.json` | nested OPDI corrections (LSZH, EDDF stand) | `scripts/262_opdi_correction.py nested --apt LSZH`, `stand_nested --apt EDDF` |
| `v53_build/evidence.json`, `v54_build/{evidence.json,rows.parquet}`, `v55_build/{evidence.json,rows.parquet}`, `v56_build/evidence.json` | 2025 proxies and evidence of steps 53–56 | `263 evidence`, `265 evidence / ledger`, `267 evidence / ledger`, `270 evidence` |
| `v55_retention/{frame,overtakes,oof_ret6ov}.parquet`, `nested_ret6ov.json` | gate-retention frame and its nested arm | `scripts/266_gate_retention_screen.py frame / overtakes / nested --arm ret6ov` |
| `v56_neighbour/{frame,oof_nb}.parquet`, `nested_nb.json` | neighbour-lateness frame and its nested arm | `scripts/269_neighbour_lateness.py frame / nested --arm nb` |
| `v54_gate/table.parquet` | the NM-unmatched lane table (2025 rows) | `scripts/264_v54_unmatched_gate.py table` |
| `v63_context_layer/{rows.parquet,screen.json}` | v63 2025 proxy and screen | `scripts/274_unmatched_context_layer.py screen / ledger` |
| `v66_bundle/rows.parquet`, `l1_screen_sixfold.json`, `l1_screen_forward.json`, `l2_screen.json`, `l3_screen.json` | v66 2025 proxy and screens | `scripts/278_v66_bundle.py l1screen / l2screen / l3screen`, `runs/overnight_20261008/ov.py proxy` |
| `overnight_20261008/{frame_cache,item05_rows,item04_park,item04y_rows,item02_frame,item11_arr_tail,item11_rows}.parquet` | 2025 frames of the four overnight layers (step 67) | `runs/overnight_20261008/*.py` |
| `long_rows_20261008/{frame,gate3x_oof}.parquet`, `gate3_no_lirf.json`, `stack_v67.json` | the long-label gate's 2025 frame and out-of-fold evidence (step 68) | `runs/long_rows_20261008/audit.py`, `gate.py`, `gate3_no_lirf.py`, `stack_v67.py` |
| `clean_chain/v53_opdi/frame.parquet`, `clean_chain/lirf_row_audit/lirf_rows.parquet`, `clean_chain/v66_bundle/{l1_frame,l3_frame}.parquet` | 2025 rows of the frames the chain builds inside steps 53, 63 and 66 | `scripts/262 frame`, `runs/lirf_row_audit/lirf_rows.py` (on the LIRF row table of the layer screens), `scripts/278 l1frame / l3frame`, each run on the 2025 proxy chain |

## 2b. The 2025 side of the post-steps (`tools/last_day/`)

The post-steps after the chain (A15 → A15LXRHDS) are fitted and judged on the 2025 out-of-fold proxy of the whole chain.
They read these inputs, none of which they rebuild:

| file | what | produced by | size |
|---|---|---|---:|
| `PRC_FROZEN/ref_layers/runs/leads_20261008/rows.parquet` | one row per departure; the 2025 rows (`split = train`, 2,085,047) carry the label `y`, the six-fold `fold`, and `v67_pred_s` = the 2025 out-of-fold proxy of the chain through step 68 (the v66 proxy + the step-67 layers six-fold + the step-68 gate six-fold), plus label-free descriptors (`D`, `lane`, `adsb`, ...). Only the 2025 rows are read | `runs/leads_20261008/load.py` of the research history, from the v67 proxy ledger (v66 proxy + the A78 six-fold move + 0.25 × the A79 gate on its rows, as judged in `runs/long_rows_20261008/stack_v67.py`) | 162.6 MB |
| `runs/overnight_20261008/frame_cache.parquet` | per-departure times (`t`, `lobt`, `bf_lobt`, `t_sched`, rule status `st`, `nm`) of both years; rebuilt by `tools/layers/build_frames.py ovcache`, its 2025 rows are the frozen `ref_layers/runs/overnight_20261008/frame_cache.parquet` rows (section 2) | stage 4 | 45.0 MB |
| `runs/clean_chain_final/A15/v54_gate/table.parquet`, `.../lirf_row_audit/lirf_rows.parquet` | written by the final chain (steps 57 / 63); their 2025 rows are the frozen `ref_layers` rows of section 2 | stage 5 | 2.0 MB, 36.7 MB |

The EUROCONTROL daily files are downloads (`docs/DATA_SOURCES.md`), not frozen artefacts; the sha256 of the copies used
is checked by `scripts/fetch_external.py`.

When the 2025 side is rebuilt instead of copied, the rebuilt 2025 rows differ from the frozen ones only where a
quantity pools both years (the counts are logged in `splice_record.json` / `logs/layers/record.json`): op_rate
(airline NM-match rate over both years, kept frozen for step 68 on purpose), 15 rows of the neighbour frame on
2025-12-31, a handful of re-matched ADS-B rows.

## 3. What is reproducible from the repository alone

- Stage 1 (base and feature tables), stage 2 (ADS-B / OPDI tables), the 56 member refits of v45, the motion features,
  every layer refit and the final chain: yes, from the challenge data, the open sources and the frozen files above.
- The frozen files: each from the scripts named above, but only by re-running the 2025 research chain that produced
  them (A27 → A51 → the six strict folds and the layer screens), which is many days of computation and depends on
  LightGBM thread-count rounding; the files themselves are the reference.
- Bit-for-bit identity is not expected: LightGBM refits on another thread count move splits by float rounding; the
  ADS-B callsign tie-breaks depend on the hash seed; the adsb.lol, OPDI, METAR and EUROCONTROL downloads may have been
  revised since. `tools/final_chain.py` checks its inputs (row count 670,790, ids = the template's, finite, ≥ 30 s)
  and logs every step's moved rows, which can be compared with the counts in `docs/REPRODUCE.md`.
