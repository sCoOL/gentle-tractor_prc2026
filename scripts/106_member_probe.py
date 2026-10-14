"""Does a fourth matched-row residual member of a different kind add anything to the rd blend? (block probe)

  py -3.14 scripts/106_member_probe.py --block AB --candidates xgb_d10 lgb_extra

A31 predicts matched rows as ``aobt3 + w0 r_global + w1 r_global_seed1 + wa r_airport`` with per-airport
weights. The residual model is information-limited (tuning moved block AB by < 1 s), so a new member only helps
if its errors differ. This probe fits candidate residual models on one strict block exactly like the existing
members (other eight months, clean matched rows, strict target-free columns, date bugs out), recovers the three
existing members from the ``_v5tfb5`` frames, and compares per-airport simplex weights (0.05 grid) with and
without the candidate:

  in-sample     weights fitted and scored on the whole held block (optimistic, same for both sides)
  cross-fitted  weights fitted on one held fold and scored on the other (A <-> B), the honest number

Scores are matched-lane pipeline RMSE (fallback mixture with the saved context-detector probability).
Candidate residuals are cached in runs/tuning/member_probe_<block>_<name>.parquet.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import fallback_context as fc  # noqa: E402
import train_eval  # noqa: E402
import tune_residual as tr  # noqa: E402
from common import RUNS  # noqa: E402

rw = importlib.import_module('104_residual_weights')

T01 = {'num_leaves': 1023, 'min_data_in_leaf': 100, 'feature_fraction': 0.85, 'bagging_fraction': 1.0, 'lambda_l2': 30, 'lambda_l1': 0,
       'min_gain_to_split': 0, 'cat_smooth': 200, 'cat_l2': 1, 'max_cat_threshold': 128, 'max_bin': 255, 'learning_rate': 0.04}
CANDIDATES = {
    # depth-wise trees: a different split structure from LightGBM's leaf-wise growth
    'xgb_d10': ('xgb', dict(max_depth=10, learning_rate=0.05, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                            reg_lambda=10, max_bin=256, max_cat_to_onehot=4, max_cat_threshold=128), 4000, 100),
    # randomised thresholds at t01's structure
    'lgb_extra': ('lgb', {**T01, 'extra_trees': True, 'seed': 7}, 800, None),
    # piecewise-linear leaves at a smaller structure
    'lgb_linear': ('lgb', {**T01, 'num_leaves': 255, 'linear_tree': True, 'linear_lambda': 10.0, 'seed': 3}, 800, None),
    # direct target (taxi, with take-off - AOBT_3 as an ordinary feature) instead of the AOBT_3 residual
    'lgb_direct': ('lgb_direct', {**T01, 'seed': 5}, 800, None),
    # two more extra-trees seeds (averaged with lgb_extra into one member when both are cached)
    'lgb_extra2': ('lgb', {**T01, 'extra_trees': True, 'seed': 17}, 800, None),
    # MLP with categorical embeddings (torch, CPU): a genuinely different function class
    'mlp': ('mlp', dict(hidden=(512, 256, 128), dropout=0.1, lr=2e-3, epochs=10, batch=2048, emb_max=16, seed=0), None, None),
}
MLP = CANDIDATES['mlp'][1]
CANDIDATES.update({
    'mlp_s1': ('mlp', {**MLP, 'seed': 1}, None, None),
    'mlp_s2': ('mlp', {**MLP, 'seed': 2}, None, None),
    'mlp_e20': ('mlp', {**MLP, 'epochs': 20, 'seed': 3}, None, None),
    'mlp_wide': ('mlp', {**MLP, 'hidden': (1024, 512, 256), 'seed': 4}, None, None),
    # the base MLP refitted on the GPU (sanity check of the CUDA path against the CPU 'mlp' cache)
    'mlp_gpu': ('mlp', {**MLP}, None, None),
    # structural variants (2026-09-18): one net per airport; direct target
    'mlp_apt': ('mlp_apt', {**MLP, 'epochs': 15, 'seed': 10}, None, None),
    'mlp_direct': ('mlp', {**MLP, 'direct': True, 'seed': 11}, None, None),
    # tuning round 2 (2026-09-18)
    'mlp_q': ('mlp', {**MLP, 'quantile': True, 'seed': 5}, None, None),
    'mlp_emb32': ('mlp', {**MLP, 'emb_max': 32, 'seed': 6}, None, None),
    'mlp_reg': ('mlp', {**MLP, 'dropout': 0.2, 'wd': 1e-4, 'epochs': 15, 'seed': 7}, None, None),
    'mlp_deep': ('mlp', {**MLP, 'hidden': (512, 512, 256, 128), 'seed': 8}, None, None),
    'mlp_b4k': ('mlp', {**MLP, 'batch': 4096, 'lr': 3e-3, 'epochs': 12, 'seed': 9}, None, None),
    # neighbour-set members (2026-09-18 evening): the MLP plus a set encoder over the airport timeline (nbr_set.py)
    'mlp_set': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 20}, None, None),
    'mlp_set0': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 0, 'seed': 21}, None, None),
    'mlp_set2': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 2, 'seed': 22}, None, None),
    # ablations: only departure tokens / only arrival tokens (plus the self token)
    'mlp_set_dep': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 23, 'kinds': ('dep',)}, None, None),
    'mlp_set_arr': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 24, 'kinds': ('arr',)}, None, None),
    # round 2: another seed; a larger encoder; a wider departure context (40/12 departures, 12/4 arrivals, -90/+30 min)
    'mlp_set_s1': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 25}, None, None),
    'mlp_set_big': ('mlp_set', {**MLP, 'd_tok': 128, 'heads': 8, 'layers': 2, 'seed': 26}, None, None),
    'mlp_set_ctx': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 2, 'seed': 27,
                                'ctx': {'dep_before': 40, 'dep_after': 12, 'arr_before': 12, 'arr_after': 4,
                                        'window': (-5400.0, 1800.0)}}, None, None),
    # round 3: v2 tokens (stand offset, runway identity, taxi distance/crossings; nbr_set.EXTRA)
    'mlp_setx': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 30, 'ctx': {'extra': True}}, None, None),
    'mlp_setx2': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 2, 'seed': 31, 'ctx': {'extra': True}}, None, None),
    # direct target (taxi itself) with tokens; month-unstable for LightGBM and the plain MLP
    'mlp_set_direct': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 32, 'direct': True}, None, None),
    # round 4: schedule / regularisation / more seeds for the set group
    'mlp_set_e15': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'epochs': 15, 'seed': 33}, None, None),
    'mlp_set_reg': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'epochs': 15, 'dropout': 0.2, 'wd': 1e-4, 'seed': 34}, None, None),
    'mlp_set_s2': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 35}, None, None),
    'mlp_set2_s1': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 2, 'seed': 36}, None, None),
    # round 5 (2026-09-19): surge tokens - every movement's stand x runway normal taxi and its NM taxi /
    # taxi-in minus that normal (nbr_set.SURGE); same configurations and seeds as A34's set group. lead_in only acts
    # on the 2026 ranking tokens (1 January sees 31 December), so the strict fits are the same with or without it
    'mlp_setn': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 20, 'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    'mlp_setn2': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 2, 'seed': 22, 'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    'mlp_setn_s1': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 25, 'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    # seed control for the surge group: plain tokens, the same three configurations, fresh seeds
    'mlp_setc': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 40}, None, None),
    'mlp_setc2': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 2, 'seed': 42}, None, None),
    'mlp_setc_s1': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 45}, None, None),
    # a second surge replicate (fresh seeds) to separate the token effect from seed luck
    'mlp_setn_r': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 50, 'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    'mlp_setn2_r': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 2, 'seed': 52, 'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    'mlp_setn_s1_r': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 55, 'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    # the same idea for trees: LightGBM with surge_context.COLUMNS added, and its control without them (same seed)
    'lgb_t01c': ('lgb', {**T01, 'seed': 41}, 800, None),
    'lgb_sg': ('lgb_plus', {**T01, 'seed': 41}, 800, None),
    # round 6 (2026-09-19 evening), toward a 2 s bundle on top of A36: the surge columns in other member kinds
    'lgb_sg_apt': ('lgb_plus_apt', {**T01, 'seed': 43}, 800, None),          # one LightGBM per airport, like r_airport
    # round 7 (2026-09-19 night, after v36): per-airport fine-tuning of the surge set network (same seed as mlp_setn,
    # so the difference to mlp_setn is the fine-tuning): 2 epochs at 0.2 x the peak rate on each airport's rows
    'mlp_setn_ft': ('mlp_set', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 20, 'ctx': {'surge': True, 'lead_in': 10800.0},
                                'ft_epochs': 2, 'ft_lr': 0.2}, None, None),
    'lgb_sg_x': ('lgb_plus', {**T01, 'extra_trees': True, 'seed': 44}, 800, None),
    # round 8 (2026-09-20): the ADS-B off-block columns (adsb_context, built by 121_adsb_events.py).
    # lgb_ab is the control's twin (same seed as lgb_t01c) so the difference is the ADS-B columns alone.
    'lgb_ab': ('lgb_adsb', {**T01, 'seed': 41}, 800, None),
    'lgb_sg_ab': ('lgb_sg_adsb', {**T01, 'seed': 41}, 800, None),
    # 2026-09-22: lgb_ab / lgb_sg_ab twins that also read the ground-phase stop sequence (124 stops) and the
    # repaired ADS-B off-block against AOBT_3 - 121's off-block is the FIRST move, a tow or a return makes it hours early
    'lgb_ab_st': ('lgb_adsb_st', {**T01, 'seed': 41}, 800, None),
    'lgb_sg_ab_st': ('lgb_sg_adsb_st', {**T01, 'seed': 41}, 800, None),
    # A48 (2026-09-22, after v41 scored): seed twins for the stop-sequence pair. It is the heaviest LightGBM group in
    # A47's blend (mean weight 0.220, max 0.739 over the 26 cells) and it is the only heavily-weighted group still on
    # ONE seed - which is exactly where §0aa's rule says seed averaging pays ("put seeds where the weight is"; the
    # ADS-B set member 2->5 seeds was -0.265 and 5->7 a further -0.066, and lgb_ab/lgb_sg_ab's own seed twins at 71
    # were part of A42). Same seed numbers as those twins so the pattern is one thing, not two.
    'lgb_ab_st_s1': ('lgb_adsb_st', {**T01, 'seed': 71}, 800, None),
    'lgb_sg_ab_st_s1': ('lgb_sg_adsb_st', {**T01, 'seed': 71}, 800, None),
    'lgb_ab_st_s2': ('lgb_adsb_st', {**T01, 'seed': 72}, 800, None),
    'lgb_sg_ab_st_s2': ('lgb_sg_adsb_st', {**T01, 'seed': 72}, 800, None),
    # 2026-09-22 late (remote vs VDGS stands): the stop-sequence twins + the X-Plane
    # stand type (gate / tie_down / misc) of the departure stand, same seed as lgb_ab_st so the difference is the type alone
    'lgb_ab_st_ty': ('lgb_adsb_st_ty', {**T01, 'seed': 41}, 800, None),
    'lgb_sg_ab_st_ty': ('lgb_sg_adsb_st_ty', {**T01, 'seed': 41}, 800, None),
    'mlp_setn_ab': ('mlp_set_adsb', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 20,
                                     'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    # round 9 (2026-09-20): the surface context measured over the OTHER aircraft (121 neighbours). Same seeds as the
    # round-8 members, so the difference to lgb_ab / mlp_setn_ab is exactly the neighbourhood columns.
    'lgb_abn': ('lgb_adsb_nbr', {**T01, 'seed': 41}, 800, None),
    'lgb_ab_apt': ('lgb_adsb_apt', {**T01, 'seed': 43}, 800, None),          # one ADS-B LightGBM per airport
    # round 10 (2026-09-20 night): the observed track geometry (122_adsb_shadow.py). Same seed as lgb_ab, so
    # on a base that already holds lgb_ab the difference is exactly the geometry columns.
    'lgb_gm': ('lgb_adsb_geom', {**T01, 'seed': 41}, 800, None),
    'mlp_setn_gm': ('mlp_set_adsb_geom', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 20,
                                          'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    # more seeds of the ADS-B set member (A39's biggest one), to average as the surge group is averaged
    'mlp_setn_ab_s1': ('mlp_set_adsb', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 25,
                                        'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    'mlp_setn_ab_s2': ('mlp_set_adsb', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 50,
                                        'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    # round 11 (2026-09-21): seed averaging aimed at the two members the A40 blend actually leans on (weight 0.30-0.60
    # at eight airports); the plain-MLP group carries weight 0 everywhere, so no seeds go there.
    'mlp_setn_ab_s3': ('mlp_set_adsb', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 75,
                                        'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    'mlp_setn_ab_s4': ('mlp_set_adsb', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 101,
                                        'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    'mlp_setn_ab_s5': ('mlp_set_adsb', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 137,
                                        'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    'mlp_setn_ab_s6': ('mlp_set_adsb', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 211,
                                        'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    # tree-side seeds for member 7 (lgb_ab + lgb_sg_ab): feature_fraction 0.85 makes the seed a real variant
    'lgb_ab_s1': ('lgb_adsb', {**T01, 'seed': 71}, 800, None),
    'lgb_sg_ab_s1': ('lgb_sg_adsb', {**T01, 'seed': 71}, 800, None),
    'lgb_ab_x': ('lgb_adsb', {**T01, 'extra_trees': True, 'seed': 44}, 800, None),
    # the G12 queue / pushback cohort as a member instead of a feature-set change: lgb_ab's twin (same seed), so on
    # a base that already holds lgb_ab the difference is exactly the G12 columns
    # per-airport twin of lgb_sg_ab: lgb_ab_apt is worth -0.159 on block AB in the real blend (2026-09-21), far more
    # than its thin-base probe suggested, so the surge+ADS-B version of the same idea is worth a fit
    'lgb_sg_ab_apt': ('lgb_sg_adsb_apt', {**T01, 'seed': 43}, 800, None),
    'lgb_q12': ('lgb_adsb_queue', {**T01, 'seed': 41}, 800, None),
    'lgb_sg_q12': ('lgb_sg_adsb_queue', {**T01, 'seed': 41}, 800, None),
    # round 12 (2026-09-21, after A42): the two things that paid were per-airport structure and extra columns, and
    # nothing that only changed a seed. These cross them: per-airport G12, and G12 in the set network's row input
    # (the shape that made mlp_setn_ab the blend's largest member). G12 reaches LTFM, which ADS-B never does.
    'lgb_q12_apt': ('lgb_adsb_queue_apt', {**T01, 'seed': 43}, 800, None),
    # round 13 (2026-09-21, after v39 transferred at x1.67 on no new ADS-B data): members fitted ONLY on rows whose
    # push-back ADS-B measured -- the population that is 21 % of 2025 training and 27 % of the scored 2026 rows
    # (EGLL 4.8 % -> 52.5 %). Same seeds as their pooled twins, so the difference is the training population alone.
    'lgb_ab_trust': ('lgb_adsb_trust', {**T01, 'seed': 41}, 800, None),
    'lgb_ab_cens': ('lgb_adsb_cens', {**T01, 'seed': 41}, 800, None),
    # round 14: the ADS-B anchor de-biased by the airport-month offset arrivals measure, on the trust population
    'lgb_ab_db': ('lgb_adsb_debias', {**T01, 'seed': 41}, 800, None),
    'lgb_sg_ab_db': ('lgb_sg_adsb_debias', {**T01, 'seed': 41}, 800, None),
    'lgb_sg_ab_cens': ('lgb_sg_adsb_cens', {**T01, 'seed': 41}, 800, None),
    'lgb_sg_ab_trust': ('lgb_sg_adsb_trust', {**T01, 'seed': 41}, 800, None),
    'mlp_setn_ab_trust': ('mlp_set_adsb_trust', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 20,
                                                 'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    'mlp_setn_q12': ('mlp_set_adsb_queue', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 20,
                                            'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    'mlp_setn_abn': ('mlp_set_adsb_nbr', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 20,
                                          'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    'mlp_sg': ('mlp_plus', {**MLP, 'seed': 60}, None, None),                 # plain MLP + surge columns
    'mlp_sg_s1': ('mlp_plus', {**MLP, 'seed': 61}, None, None),
    'mlp_sg_s2': ('mlp_plus', {**MLP, 'seed': 62}, None, None),
    # surge tokens AND the surge columns in the row input of the set network
    'mlp_setn_sg': ('mlp_set_plus', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 70, 'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    'mlp_setn_sg2': ('mlp_set_plus', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 2, 'seed': 72, 'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
    'mlp_setn_sg_s1': ('mlp_set_plus', {**MLP, 'd_tok': 64, 'heads': 4, 'layers': 1, 'seed': 75, 'ctx': {'surge': True, 'lead_in': 10800.0}}, None, None),
})


def fit_lightgbm_per_airport(blk, params, rounds):
    """One LightGBM residual model per airport (trained and scored airport by airport)."""
    held = blk['held']; r = np.full(len(held), np.nan)
    for apt in sorted(held.APT.unique()):
        ta = blk['train'][blk['train'].APT.values == apt]; ha = held.APT.values == apt
        if len(ta) < 1000:
            continue
        ra, _ = tr.fit_lightgbm({**blk, 'train': ta, 'held': held[ha]}, params, rounds)
        r[ha] = ra
    return r, rounds


SURGE_LEAD_IN = 3 * 3600.0      # 2026 context reaches 3 h back into December 2025 (the longest surge window is 2 h)


def surge_table(which):
    """surge_context columns of the 2025 file or the 2026 ranking file (with the December lead-in), cached."""
    import surge_context
    cache = RUNS / 'tuning' / f'surge_context_{which}.parquet'
    if not cache.exists():
        surge_context.build(which, lead_in=SURGE_LEAD_IN if which == 'rank' else 0.0).to_parquet(cache, index=False)
    return pd.read_parquet(cache)


def with_surge(blk):
    """The block with surge_context columns merged into its training and held rows (held = 2026 rows when deploying)."""
    import surge_context
    ft = surge_table('train'); fh = ft if blk['block'] != 'deploy' else surge_table('rank')
    add = lambda df, f: df.merge(f, on='MVT_ID_mvt', how='left', validate='one_to_one')
    return {**blk, 'train': add(blk['train'], ft), 'held': add(blk['held'], fh), 'cols': blk['cols'] + surge_context.COLUMNS}


def with_adsb(blk):
    """The block with the ADS-B off-block columns merged in (one table covers 2025 and the 2026 ranking rows)."""
    import adsb_context
    f = adsb_context.table()
    add = lambda df: df.merge(f, on='MVT_ID_mvt', how='left', validate='one_to_one')
    return {**blk, 'train': add(blk['train']), 'held': add(blk['held']), 'cols': blk['cols'] + adsb_context.COLUMNS}


def with_stops(blk):
    """``with_adsb`` + the stop sequence of the ground phase, the taxi from the LAST stand departure where the aircraft
    re-parked, and that repaired off-block against NM's AOBT_3 (the repaired twin of ``adsb_leave_minus_a3``)."""
    import adsb_context
    f = adsb_context.stops_table()

    def add(df):
        out = df.merge(f, on='MVT_ID_mvt', how='left', validate='one_to_one')
        repark = (out.st_n_stand.values >= 1) & (out.st_taxi_last.values < out.adsb_taxi_leave.values - 300)
        out['st_taxi_fix'] = np.where(repark, out.st_taxi_last.values, out.adsb_taxi_leave.values)
        out['st_fix_minus_a3'] = out.atot_minus_aobt3.values.astype(float) - out.st_taxi_fix.values
        return out
    blk = blk if 'adsb_taxi_leave' in blk['train'] else with_adsb(blk)      # 107's deployment route has merged them already
    return {**blk, 'train': add(blk['train']), 'held': add(blk['held']),
            'cols': blk['cols'] + adsb_context.STOPS_COLUMNS + ['st_taxi_fix', 'st_fix_minus_a3']}


def with_geom(blk):
    """The block with the observed-track geometry merged in (reaches every row ADS-B saw rolling, censored or not)."""
    import adsb_context
    f = adsb_context.geom_table()
    add = lambda df: df.merge(f, on='MVT_ID_mvt', how='left', validate='one_to_one')
    return {**blk, 'train': add(blk['train']), 'held': add(blk['held']), 'cols': blk['cols'] + adsb_context.GEOM_COLUMNS}


def queue_columns():
    """G12's columns: queue_context.COLUMNS plus qx_own_minus_rwy_c60, which 39_build_v6_queue.py adds on top."""
    import queue_context
    return list(queue_context.COLUMNS) + ['qx_own_minus_rwy_c60']


def queue_table(which):
    """The G12 queue / pushback-cohort columns, read out of the prebuilt features v6 tables.

    39_build_v6_queue.py already wrote them for 2025 and for the 2026 ranking rows, so nothing is recomputed here.
    G12 was rejected in §0l as a FEATURE-SET change against A22 (-0.204, worst fold +0.463); this exposes it as a
    member instead, the shape that worked for the surge and ADS-B columns.
    """
    name = 'rank' if which == 'rank' else 'train'
    return pd.read_parquet(common.DATA / f'features_v6_{name}.parquet', columns=['MVT_ID_mvt'] + queue_columns())


def with_stand_type(blk):
    """+ the X-Plane stand type of the departure stand (scripts/stand_type.py): ``stand_xp_type`` (0 unmatched, 1 gate,
    2 tie_down = remote apron, 3 misc) and ``stand_remote``. The ADS-B off-block estimate behaves differently at remote
    positions (no push-back), and a stand categorical cannot express "type x ADS-B value" cheaply."""
    import stand_type
    return {**blk, 'train': stand_type.add(blk['train']), 'held': stand_type.add(blk['held']), 'cols': blk['cols'] + stand_type.COLUMNS}


def with_queue(blk):
    """The block with the G12 columns merged into its training and held rows (held = 2026 rows when deploying)."""
    ft = queue_table('train'); fh = ft if blk['block'] != 'deploy' else queue_table('rank')
    add = lambda df, f: df.merge(f, on='MVT_ID_mvt', how='left', validate='one_to_one')
    return {**blk, 'train': add(blk['train'], ft), 'held': add(blk['held'], fh), 'cols': blk['cols'] + queue_columns()}


def adsb_cells(t, cell_min=3000, three=False):
    """Cell label per row: ``<APT>`` as always, or ``<APT>:T`` / ``<APT>:N`` split on whether ADS-B actually saw this
    aircraft leave its stand.

    The blend has carried ONE weight vector per airport since A30, so the ADS-B members are weighted the same on a
    row whose push-back was measured and on a row where their columns are all NaN. In 2025 only 21 % of matched rows
    have a trustworthy push-back and in the scored 2026 rows 27 % do -- at EGLL, 4.8 % against 52.5 %. Splitting the
    cell lets the blend lean on those members exactly where their columns exist.

    A cell with fewer than ``cell_min`` fitting rows keeps the airport-level label: LFPG has 1,636 such rows and
    LTFM none, and a 14-member simplex fitted on a few hundred rows is noise.
    """
    apt = t.APT.values.astype(object)
    trust = (t.adsb_at_stand.values == 1)
    if three:
        # :C = ADS-B saw the aircraft already rolling, so it has a BOUND on the off-block but not the off-block.
        # 382,899 matched 2025 rows, RMSE 176.0 against 206.4 for rows ADS-B never saw at all.
        cens = (t.adsb_censored.values == 1) & ~trust
        state = np.where(trust, ':T', np.where(cens, ':C', ':N'))
    else:
        state = np.where(trust, ':T', ':N')
    lab = np.char.add(apt.astype(str), state)
    fit = t.has.values
    apt_s = apt.astype(str)
    # Both sides are checked per airport, not just the labels that happen to occur: an airport with NO measured
    # push-back at all (LTFM) never produces a ':T' label, so scanning observed labels would leave it split into a
    # single ':N' cell. Leaving a thin side its own weights is exactly what this guard exists to prevent, and LFPG
    # has 1,636 such rows against a 15-member simplex.
    small = set()
    states = (':T', ':C', ':N') if three else (':T', ':N')
    for x in np.unique(apt_s):
        here = fit & (apt_s == x)
        if min((here & (state == st)).sum() for st in states) < cell_min:
            small.add(x)
    return np.where(np.isin(apt_s, list(small)), apt_s, lab)


def only_trust(blk):
    """Restrict the TRAINING rows to those whose push-back ADS-B actually measured (``adsb_at_stand`` = 1).

    Why: the ADS-B members are fitted on all matched rows, of which only
    21 % carry a trustworthy push-back in 2025 while 27 % of the scored 2026 rows do -- and at EGLL it is 4.8 %
    against 52.5 %. A pooled fit therefore spends almost all of its capacity on rows where its own columns are
    missing, and where it does see a push-back it has learned little about using one: the push-back halves the error
    at EDDM (0.49) and LSZH (0.47) but buys only 24 % at EGLL, which has 11,508 such training rows against EDDM's
    82,001. This member trains only on the population that grows in 2026. It still SCORES every held row, because
    the blend needs a finite residual everywhere; the ``adsb`` weight cells are what keep it off the rows it is not
    for (123_weight_solver.py --cells adsb).

    Must be applied after ``with_adsb``, which is what puts ``adsb_at_stand`` on the frame.
    """
    tr_rows = blk['train']
    keep = (tr_rows.adsb_at_stand.values == 1)
    if keep.sum() < 50000:
        raise SystemExit(f'only_trust: {keep.sum()} training rows with a measured push-back is too few')
    return {**blk, 'train': tr_rows[keep]}


DEBIAS_COLUMNS = ['adsb_leave_corr', 'adsb_first_corr', 'adsb_leave_z']


def with_adsb_debias(blk):
    """Add the ADS-B anchor CORRECTED by the airport-month bias that arrivals measure.

    ``adsb_q_med`` is the median signed offset between ADS-B and the record for that airport-month, measured on
    ARRIVALS, whose in-block times are real in both years -- so it is rank-available and it tracks calibration as it
    moves. It moves a lot: EGLL's departures sit 325 s after NM's AOBT_3 in 2025 and 222 s in 2026 (MAD 39.7 -> 28.3),
    which is why a member that learned EGLL's relationship on 11,536 poorly-calibrated 2025 rows serves 20,918
    better ones badly. A tree cannot subtract two columns, exactly as with G12's qx_own_minus_rwy_c60, so the
    correction has to be handed to it.

    Must be applied after ``with_adsb``.
    """
    def add(df):
        d = df.copy()
        q = d.adsb_q_med.values.astype(float)
        d['adsb_leave_corr'] = d.adsb_leave_minus_a3.values.astype(float) - q
        d['adsb_first_corr'] = d.adsb_first_minus_a3.values.astype(float) - q
        d['adsb_leave_z'] = d['adsb_leave_corr'].values / np.maximum(d.adsb_q_mad.values.astype(float), 1.0)
        return d
    return {**blk, 'train': add(blk['train']), 'held': add(blk['held']), 'cols': blk['cols'] + DEBIAS_COLUMNS}


def only_cens(blk):
    """Restrict the TRAINING rows to those ADS-B saw already rolling (``adsb_censored`` = 1): a hard bound on the
    off-block but not the off-block itself. The censored share also moves between the years (EGLL 8.2 % -> 16.5 %,
    LSZH 20.1 % -> 27.0 %, LEMD 36.1 % -> 39.1 %), so the same argument as ``only_trust`` applies to it.

    Must be applied after ``with_adsb``.
    """
    tr_rows = blk['train']
    keep = (tr_rows.adsb_censored.values == 1)
    if keep.sum() < 50000:
        raise SystemExit(f'only_cens: {keep.sum()} censored training rows is too few')
    return {**blk, 'train': tr_rows[keep]}


def with_nbr(blk):
    """The block with the measured surface-context columns merged in (they reach rows ADS-B never saw itself)."""
    import adsb_context
    f = adsb_context.nbr_table()
    add = lambda df: df.merge(f, on='MVT_ID_mvt', how='left', validate='one_to_one')
    return {**blk, 'train': add(blk['train']), 'held': add(blk['held']), 'cols': blk['cols'] + adsb_context.NBR_COLUMNS}


def mlp_inputs(blk, params):
    """Numeric block (clipped to the training 0.5-99.5 % range, standardised, NaN indicators) and embedded-category
    codes for the training rows and the anchored held rows; shared by the plain and the neighbour-set MLP."""
    tr_rows = blk['train']; held = blk['held']
    cats = [c for c in blk['cats'] if c in blk['cols']]
    nums = [c for c in blk['cols'] if c not in cats]
    m = np.isfinite(held.atot_minus_aobt3.values.astype(float))
    if params.get('taxi_expert'):               # 110: a normal-taxi expert scores every held row
        m = np.ones(len(held), bool)
    hs = held[m]

    def num_block(df, stats=None):
        X = df[nums].apply(pd.to_numeric, errors='coerce').astype('float64').values
        if stats is None:
            lo = np.nanpercentile(X, 0.5, axis=0); hi = np.nanpercentile(X, 99.5, axis=0)
            Xc = np.clip(X, lo, hi)
            mu = np.nanmean(Xc, axis=0); sd = np.nanstd(Xc, axis=0)
            mu = np.where(np.isfinite(mu), mu, 0.0); sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
            nan_cols = np.where(np.isnan(X).any(axis=0))[0]
            stats = (lo, hi, mu, sd, nan_cols)
        lo, hi, mu, sd, nan_cols = stats[:5]
        if params.get('quantile'):
            # rank-gauss: each clipped column mapped through its 2025 quantiles to a standard normal
            from scipy.special import ndtri
            if len(stats) == 5:
                qs = np.linspace(0, 1, 257)
                Xq = np.clip(X, lo, hi)
                qv = np.nanquantile(Xq, qs, axis=0).T + np.arange(257)[None, :] * 1e-9
                stats = stats + (qs, qv)
            qs, qv = stats[5], stats[6]
            Xc = np.clip(X, lo, hi)
            for j in range(Xc.shape[1]):
                col = Xc[:, j]; ok = np.isfinite(col)
                if hi[j] > lo[j]:
                    col[ok] = ndtri(np.clip(np.interp(col[ok], qv[j], qs), 1e-3, 1 - 1e-3))
                else:
                    col[ok] = 0.0
        else:
            Xc = (np.clip(X, lo, hi) - mu) / sd
        ind = np.isnan(X[:, nan_cols]).astype('float32')
        return np.hstack([np.nan_to_num(Xc, nan=0.0).astype('float32'), ind]), stats

    def cat_block(df, maps=None):
        if maps is None:
            maps = {}
            for c in cats:
                vc = df[c].astype(str).value_counts()
                keep = vc.index[vc.values >= 20]
                maps[c] = {v: i + 1 for i, v in enumerate(keep)}
        return np.column_stack([df[c].astype(str).map(maps[c]).fillna(0).astype('int64').values for c in cats]), maps

    Xn, stats = num_block(tr_rows); Xk, maps = cat_block(tr_rows)
    Hn, _ = num_block(hs, stats); Hk, _ = cat_block(hs, maps)
    direct = bool(params.get('direct'))
    taxi = direct or bool(params.get('taxi_expert'))
    y = (tr_rows.TAXITIME_SEC_mvt.values - (0.0 if taxi else tr_rows.atot_minus_aobt3.values)).astype('float64')
    y_mu, y_sd = float(np.mean(y)), float(np.std(y))
    yt = ((y - y_mu) / y_sd).astype('float32')
    return dict(Xn=Xn, Xk=Xk, Hn=Hn, Hk=Hk, maps=maps, cats=cats, m=m, hs=hs, y_mu=y_mu, y_sd=y_sd, yt=yt, direct=direct)


def fit_mlp(blk, params):
    """MLP on the AOBT_3 residual: numeric columns clipped/standardised with NaN indicators, categoricals embedded."""
    import torch
    from torch import nn
    torch.manual_seed(params['seed']); np.random.seed(params['seed'])
    torch.set_num_threads(16)
    held = blk['held']
    I = mlp_inputs(blk, params)
    Xn, Xk, Hn, Hk, maps, cats, m, hs = I['Xn'], I['Xk'], I['Hn'], I['Hk'], I['maps'], I['cats'], I['m'], I['hs']
    y_mu, y_sd, yt, direct = I['y_mu'], I['y_sd'], I['yt'], I['direct']

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(len(maps[c]) + 1, min(params['emb_max'], (len(maps[c]) + 2) // 2)) for c in cats])
            d = Xn.shape[1] + sum(e.embedding_dim for e in self.embs)
            layers = []
            for h in params['hidden']:
                layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(params['dropout'])]; d = h
            layers.append(nn.Linear(d, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, xn, xk):
            z = torch.cat([xn] + [e(xk[:, i]) for i, e in enumerate(self.embs)], dim=1)
            return self.mlp(z).squeeze(1)

    # GPU when available (params['device'] = 'cpu' forces CPU); the v32/v33 members were fitted on CPU
    dev = torch.device('cuda' if params.get('device', 'auto') != 'cpu' and torch.cuda.is_available() else 'cpu')
    if dev.type == 'cuda':
        torch.cuda.manual_seed_all(params['seed'])
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=params['lr'], weight_decay=params.get('wd', 1e-5))
    n = len(yt); steps = params['epochs'] * ((n + params['batch'] - 1) // params['batch'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=params['lr'], total_steps=steps)
    Xn_t, Xk_t, y_t = torch.from_numpy(Xn).to(dev), torch.from_numpy(Xk).to(dev), torch.from_numpy(yt).to(dev)
    for ep in range(params['epochs']):
        net.train(); perm = torch.randperm(n, device=dev); tot = torch.zeros((), device=dev)
        for i in range(0, n, params['batch']):
            idx = perm[i:i + params['batch']]
            opt.zero_grad()
            loss = torch.mean((net(Xn_t[idx], Xk_t[idx]) - y_t[idx]) ** 2)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
            tot += loss.detach() * len(idx)
        print(f'    mlp epoch {ep + 1} [{dev.type}]: train mse (scaled) {float(tot) / n:.4f}', flush=True)
    net.eval(); out = []
    with torch.no_grad():
        Hn_t, Hk_t = torch.from_numpy(Hn), torch.from_numpy(Hk)
        for i in range(0, len(Hn), 65536):
            out.append(net(Hn_t[i:i + 65536].to(dev), Hk_t[i:i + 65536].to(dev)).cpu().numpy())
    r = np.full(len(held), np.nan)
    r[m] = np.concatenate(out).astype('float64') * y_sd + y_mu - (hs.atot_minus_aobt3.values if direct else 0.0)
    return r, params['epochs']


def fit_mlp_per_airport(blk, params):
    """One MLP per airport (like the per-airport LightGBM member), fitted and scored airport by airport."""
    held = blk['held']; r = np.full(len(held), np.nan)
    for apt in sorted(held.APT.unique()):
        tr_a = blk['train'][blk['train'].APT.values == apt]
        ha = held.APT.values == apt
        if len(tr_a) < 1000:
            continue
        sub = {**blk, 'train': tr_a, 'held': held[ha]}
        ra, _ = fit_mlp(sub, params)
        r[ha] = ra
    return r, params['epochs']


_TOKENS = {}


def tokens_for(which, ids, ctx=None):
    """Neighbour-set tokens (nbr_set.py) of the 2025 training year or the 2026 ranking file, aligned to ``ids``;
    ``ctx`` overrides nbr_set.build's context (dep_before, dep_after, arr_before, arr_after, window)."""
    import nbr_set
    key = (which, tuple(sorted((ctx or {}).items())))
    if key not in _TOKENS:
        for k in [k for k in _TOKENS if k[0] == which]:     # one context per file in memory (about 5 GB each)
            del _TOKENS[k]
        _TOKENS[key] = nbr_set.build(which, **(ctx or {}))
    return nbr_set.aligned(*_TOKENS[key], ids)


def fit_mlp_set(blk, params):
    """The plain MLP plus a set encoder over the airport's movement timeline around each departure (nbr_set.py).

    Tokens pass a per-token MLP and ``layers`` self-attention layers; the row's own embedding queries them
    (multi-head attention pooling) and their masked mean is added, both concatenated to the row input of the
    usual 512-256-128 head. Residual target, scaling and training schedule are the plain MLP's."""
    import torch
    from torch import nn
    import nbr_set
    torch.manual_seed(params['seed']); np.random.seed(params['seed'])
    torch.set_num_threads(16)
    held = blk['held']
    I = mlp_inputs(blk, params)
    Xn, Xk, Hn, Hk, maps, cats, m, hs = I['Xn'], I['Xk'], I['Hn'], I['Hk'], I['maps'], I['cats'], I['m'], I['hs']
    y_mu, y_sd, yt = I['y_mu'], I['y_sd'], I['yt']
    clf = params.get('task') == 'clf'           # schedule-copy classifier (label ``fallback``), 108_matched_detector.py
    wt = None
    if clf:
        yt = blk['train'].fallback.values.astype('float32')
        if params.get('weight') == 'gap':
            # the mixture's cost of a wrong p grows with (D - normal)^2; AOBT_3's NM taxi stands in for normal
            D = blk['train'].atot_minus_sched.values.astype(float); a3 = blk['train'].atot_minus_aobt3.values.astype(float)
            g = np.where(np.isfinite(D) & (D > 0), ((D - a3) / 900.0) ** 2, 0.0)
            wt = np.clip(np.nan_to_num(g, nan=0.0), 0.01, params.get('weight_cap', 16.0))
            wt = (wt / wt.mean()).astype('float32')
        elif params.get('weight') == 'unm':
            # 114: genuinely unmatched rows (1 % of the masked training rows, all of the use) weighted up
            wt = np.where(blk['train'].is_unmatched.values > 0.5, float(params.get('unm_w', 10.0)), 1.0)
            wt = (wt / wt.mean()).astype('float32')
    which = 'rank' if blk['block'] == 'deploy' else 'train'
    ctx = params.get('ctx')
    Tt = tokens_for('train', blk['train'].MVT_ID_mvt.values, ctx)
    Th = tokens_for(which, hs.MVT_ID_mvt.values, ctx)
    kinds = params.get('kinds', ('dep', 'arr'))
    for kind in ('dep', 'arr'):
        if kind not in kinds:           # ablation: blank that kind of token entirely (valid = 0)
            for T in (Tt, Th):
                T[T[..., nbr_set.F[f'is_{kind}']] > 0.5] = 0
    if params.get('mask_self_nm'):      # 110: the departure's own NM fields as an unmatched row carries them
        F = nbr_set.F
        for T in (Tt, Th):
            for c in ('anchor', 'dur', 'plan', 'wake_L', 'wake_M', 'wake_H', 'wake_J'):
                T[:, 0, F[c]] = 0
            T[:, 0, F['anchor_missing']] = 1; T[:, 0, F['plan_missing']] = 1
            if (ctx or {}).get('surge'):    # the own NM taxi residual is NM-derived; the stand x runway normal is not
                T[:, 0, nbr_set.surge_index('dur_res', T.shape[-1])] = 0
    nf = Tt.shape[-1]; dm = params['d_tok']

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(len(maps[c]) + 1, min(params['emb_max'], (len(maps[c]) + 2) // 2)) for c in cats])
            d_row = Xn.shape[1] + sum(e.embedding_dim for e in self.embs)
            self.tok_in = nn.Sequential(nn.Linear(nf, dm), nn.SiLU(), nn.Linear(dm, dm))
            self.enc = None
            if params['layers']:
                layer = nn.TransformerEncoderLayer(dm, params['heads'], dim_feedforward=2 * dm, dropout=0.0, activation='gelu',
                                                   batch_first=True, norm_first=True)
                self.enc = nn.TransformerEncoder(layer, params['layers'], enable_nested_tensor=False)
            self.q = nn.Linear(d_row, dm)
            self.pool = nn.MultiheadAttention(dm, params['heads'], batch_first=True)
            d = d_row + 2 * dm
            layers = []
            for h in params['hidden']:
                layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(params['dropout'])]; d = h
            layers.append(nn.Linear(d, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, xn, xk, tok):
            z = torch.cat([xn] + [e(xk[:, i]) for i, e in enumerate(self.embs)], dim=1)
            tok = tok.float()
            pad = tok[..., 0] < 0.5                      # the self token is always valid
            h = self.tok_in(tok)
            if self.enc is not None:
                h = self.enc(h, src_key_padding_mask=pad)
            a, _ = self.pool(self.q(z).unsqueeze(1), h, h, key_padding_mask=pad, need_weights=False)
            keep = (~pad).unsqueeze(-1).to(h.dtype)
            mean = (h * keep).sum(1) / keep.sum(1)
            return self.mlp(torch.cat([z, a.squeeze(1), mean], dim=1)).squeeze(1)

    dev = torch.device('cuda' if params.get('device', 'auto') != 'cpu' and torch.cuda.is_available() else 'cpu')
    if dev.type == 'cuda':
        torch.cuda.manual_seed_all(params['seed'])
    amp = dev.type == 'cuda' and params.get('amp', True)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=params['lr'], weight_decay=params.get('wd', 1e-5))
    n = len(yt); steps = params['epochs'] * ((n + params['batch'] - 1) // params['batch'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=params['lr'], total_steps=steps)
    Xn_t, Xk_t, y_t = torch.from_numpy(Xn).to(dev), torch.from_numpy(Xk).to(dev), torch.from_numpy(yt).to(dev)
    w_t = torch.from_numpy(wt).to(dev) if wt is not None else None
    T_t = torch.from_numpy(Tt).to(dev)
    del Tt
    t0 = time.time()
    for ep in range(params['epochs']):
        net.train(); perm = torch.randperm(n, device=dev); tot = torch.zeros((), device=dev)
        for i in range(0, n, params['batch']):
            idx = perm[i:i + params['batch']]
            opt.zero_grad()
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                out = net(Xn_t[idx], Xk_t[idx], T_t[idx])
            if clf:
                loss = nn.functional.binary_cross_entropy_with_logits(out.float(), y_t[idx],
                                                                      weight=None if w_t is None else w_t[idx])
            else:
                loss = torch.mean((out.float() - y_t[idx]) ** 2)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
            tot += loss.detach() * len(idx)
        print(f'    mlp_set epoch {ep + 1} [{dev.type}]: train {"log loss" if clf else "mse (scaled)"} {float(tot) / n:.4f} '
              f'[{time.time() - t0:.0f}s]', flush=True)

    def predict(rows):
        net.eval(); out = []
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            for i in range(0, len(rows), 32768):
                j = rows[i:i + 32768]
                out.append(net(torch.from_numpy(Hn[j]).to(dev), torch.from_numpy(Hk[j]).to(dev),
                               torch.from_numpy(Th[j]).to(dev)).float().cpu().numpy())
        return np.concatenate(out).astype('float64') if out else np.zeros(0)

    ft = int(params.get('ft_epochs', 0))
    if ft:
        # 2026-09-19 night: per-airport fine-tuning - the global network continued for ``ft`` epochs on one airport's
        # training rows at ft_lr x the peak rate (cosine to 0), then that airport's held rows are scored
        import copy
        base_state = copy.deepcopy(net.state_dict())
        apt_tr = blk['train'].APT.values; apt_h = hs.APT.values; z = np.full(len(Hn), np.nan)
        for apt in sorted(np.unique(apt_h)):
            net.load_state_dict(base_state)
            tr_idx = torch.from_numpy(np.where(apt_tr == apt)[0]).to(dev); n_a = len(tr_idx)
            if n_a >= 1000:
                opt_a = torch.optim.AdamW(net.parameters(), lr=params['lr'] * params.get('ft_lr', 0.2), weight_decay=params.get('wd', 1e-5))
                sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=ft * ((n_a + params['batch'] - 1) // params['batch']))
                for ep in range(ft):
                    net.train(); perm = tr_idx[torch.randperm(n_a, device=dev)]
                    for i in range(0, n_a, params['batch']):
                        idx = perm[i:i + params['batch']]
                        opt_a.zero_grad()
                        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                            o_ = net(Xn_t[idx], Xk_t[idx], T_t[idx])
                        loss = (nn.functional.binary_cross_entropy_with_logits(o_.float(), y_t[idx]) if clf
                                else torch.mean((o_.float() - y_t[idx]) ** 2))
                        loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt_a.step(); sched_a.step()
            z[apt_h == apt] = predict(np.where(apt_h == apt)[0])
        print(f'    mlp_set per-airport fine-tuning: {ft} epochs at x{params.get("ft_lr", 0.2)} [{time.time() - t0:.0f}s]', flush=True)
    del T_t, Xn_t, Xk_t
    r = np.full(len(held), np.nan)
    if not ft:
        z = predict(np.arange(len(Hn)))
    r[m] = 1.0 / (1.0 + np.exp(-z)) if clf else z * y_sd + y_mu - (hs.atot_minus_aobt3.values if I['direct'] else 0.0)
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return r, params['epochs']


def fit_direct(blk, params, rounds):
    """LightGBM on the taxi itself (clean matched rows); returns the implied residual ``pred - aobt3``."""
    tr_rows = blk['train']; held = blk['held']
    X = train_eval.prep(tr_rows, blk['cols'], blk['cats'])
    m = np.isfinite(held.atot_minus_aobt3.values.astype(float))
    model = train_eval.fit_lgb(X, tr_rows.TAXITIME_SEC_mvt.values, blk['cats'], params, rounds, 'direct_member')
    r = np.full(len(held), np.nan)
    r[m] = model.predict(train_eval.prep(held[m], blk['cols'], blk['cats'], train_eval._cats_of(X, blk['cats']))) \
        - held.atot_minus_aobt3.values[m]
    return r, rounds


def simplex_grid(k, step=0.05):
    """Every weight vector whose first k-1 entries are multiples of ``step`` and sum to at most 1.

    Built dimension by dimension, in the same lexicographic order a filtered ``itertools.product`` gives, because the
    cube it walks is 21^(k-1) while the simplex inside it is only C(20+k-1, k-1): at k=9 that is 38 G against 3.1 M.
    """
    n = int(round(1 / step))
    memo = {}

    def build(d, m):                      # all d-tuples of 0..m summing to at most m, lexicographic
        if d == 0:
            return np.zeros((1, 0), np.int16)
        if (d, m) not in memo:
            parts = [np.column_stack([np.full(len(sub), c, np.int16), sub])
                     for c, sub in ((c, build(d - 1, m - c)) for c in range(m + 1))]
            memo[(d, m)] = np.concatenate(parts)
        return memo[(d, m)]

    return build(k - 1, n).astype(float) * step


def fit_weights(R, mix_m, target, grid):
    """Simplex weights over the member columns of R minimising sum((mix_m * R w - target)^2)."""
    Rm = R * mix_m[:, None]
    A = Rm.T @ Rm; b = Rm.T @ target
    W = np.column_stack([grid, 1 - grid.sum(axis=1)])
    sse = np.einsum('ij,jk,ik->i', W, A, W) - 2 * W @ b
    return W[int(np.argmin(sse))]


def _solve_support(A, b, S, lam):
    """Equality-constrained minimiser on one support: min w'Aw - 2b'w over {w_S >= 0, sum w_S = 1, w_-S = 0}.

    KKT of the support problem is [[2 A_SS + 2 lam I, 1], [1', 0]] [w_S; mu] = [2 b_S; 1]; ``lam`` is an optional
    ridge that keeps A_SS invertible when two members are collinear (the seed twins are, nearly).
    """
    k = len(S)
    M = np.zeros((k + 1, k + 1)); M[:k, :k] = 2 * (A[np.ix_(S, S)] + lam * np.eye(k))
    M[:k, k] = 1.0; M[k, :k] = 1.0
    rhs = np.zeros(k + 1); rhs[:k] = 2 * b[S]; rhs[k] = 1.0
    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
    return sol[:k], sol[k]


def fit_weights_exact(R, mix_m, target, grid, lam=0.0, tol=1e-9, kmax=16):
    """The same objective solved exactly on the simplex instead of on a 0.05 grid.

    ``fit_weights`` rounds every weight to a multiple of 0.05, and the grid cost grows as C(20+k-1, k-1), which is
    what capped the blend at nine members. The objective is a convex quadratic over a simplex, so its minimiser is
    the best KKT point over the 2^k - 1 supports; at k <= 14 that enumeration is milliseconds and is exact.

    ``grid`` is only a safety floor: when one is given and ``lam`` is 0 the grid optimum is evaluated too and
    returned if it scores lower, so this can never be worse than ``fit_weights`` on the rows it is fitted on. Pass
    ``grid=None`` (what 107 does past nine members, where the grid alone is 30 M points and the floor costs more
    than the solve) and the enumeration stands on its own. Above 14 members the enumeration is skipped.

    ``lam`` > 0 adds a ridge of that fraction of the members' mean own-SSE, which shrinks the fitted weights toward
    equal shares inside the support -- the near-collinear seed twins are what it is for. The ridged objective then
    also decides between supports, so the grid floor does not apply (the grid minimises the unridged one).

    Cost is 2^k supports per call and quadruples per member: a whole ``107 weights`` run (10 airports x 4 fits) is
    about 30 s at k=11, 70 s at k=14 (A42) and a few minutes at ``kmax``. Raise ``kmax`` only with that in mind.
    """
    Rm = R * mix_m[:, None]
    A = Rm.T @ Rm; b = Rm.T @ target
    k = A.shape[0]
    lam = lam * float(np.trace(A)) / k       # scale-free: ``lam`` is a fraction of the members' mean own-SSE
    obj = ((lambda w: float(w @ A @ w - 2 * b @ w)) if lam <= 0 else
           (lambda w: float(w @ A @ w - 2 * b @ w + lam * w @ w)))
    if grid is None or lam > 0:
        best = np.full(k, 1.0 / k); best_v = obj(best)
    else:
        W = np.column_stack([grid, 1 - grid.sum(axis=1)])
        sse = np.einsum('ij,jk,ik->i', W, A, W) - 2 * W @ b
        best = W[int(np.argmin(sse))]; best_v = float(sse.min())
    if k > kmax:
        if grid is None:
            raise ValueError(f'fit_weights_exact needs a grid above {kmax} members')
        return best
    for mask in range(1, 1 << k):
        S = [j for j in range(k) if mask >> j & 1]
        wS, mu = _solve_support(A, b, S, lam)
        if (wS < -tol).any():
            continue
        w = np.zeros(k); w[S] = np.maximum(wS, 0.0)
        if w.sum() <= 0:
            continue
        w /= w.sum()
        rest = [j for j in range(k) if not mask >> j & 1]
        if rest:                       # stationarity off the support: the objective must not fall by leaving it
            g = 2 * (A @ w - b) + mu
            if (g[rest] < -tol * max(1.0, abs(mu))).any():
                continue
        v = obj(w)
        if v < best_v - 1e-12:
            best, best_v = w, v
    return best


def pipeline_rmse(t, R, weights_by_apt, rows):
    w = np.array([weights_by_apt[a] for a in t.APT.values[rows]])
    normal = t.atot_minus_aobt3.values[rows] + (R[rows] * w).sum(axis=1)
    normal = np.where(t.has.values[rows], normal, t.n_rd.values[rows])
    pred = fc.mixture(t.p_fallback.values[rows], t.atot_minus_sched.values[rows], normal, t.is_heli.values[rows], t.pred_rd.values[rows])
    return common.rmse(pred, t.y.values[rows])


def evaluate(t, R, label):
    k = R.shape[1]; grid = simplex_grid(k)
    fit = t.has.values
    mt = t.fpl_matched.values == 1
    apts = sorted(t.APT.unique())
    out = {}
    w_in = {a: fit_weights(R[fit & (t.APT.values == a)], t.mix_m.values[fit & (t.APT.values == a)],
                           (t.y.values - t.mix_c.values)[fit & (t.APT.values == a)], grid) for a in apts}
    out['in_sample'] = pipeline_rmse(t, R, w_in, mt)
    out['weights_in_sample'] = {a: [round(float(x), 2) for x in w] for a, w in w_in.items()}
    folds = sorted(t.fold.unique())
    sse = 0.0; n = 0
    for f in folds:
        src = fit & (t.fold.values != f)
        w = {a: fit_weights(R[src & (t.APT.values == a)], t.mix_m.values[src & (t.APT.values == a)],
                            (t.y.values - t.mix_c.values)[src & (t.APT.values == a)], grid) for a in apts}
        rows = mt & (t.fold.values == f)
        r = pipeline_rmse(t, R, w, rows)
        out[f'cross_{f}'] = r
        sse += r ** 2 * rows.sum(); n += rows.sum()
    out['cross_fitted'] = float(np.sqrt(sse / n))
    print(f'  {label}: in-sample {out["in_sample"]:.3f}, cross-fitted {out["cross_fitted"]:.3f} '
          f'({", ".join(f"{f} {out[f"cross_{f}"]:.3f}" for f in folds)})', flush=True)
    return out


def candidate_residual(name, blk):
    cache = RUNS / 'tuning' / f'member_probe_{blk["block"]}_{name}.parquet'
    if cache.exists():
        return pd.read_parquet(cache), None
    kind, params, rounds, es = CANDIDATES[name]
    t0 = time.time()
    if kind == 'xgb':
        r, best = tr.fit_xgboost(blk, params, rounds, early_stopping=es)
    elif kind == 'lgb_direct':
        r, best = fit_direct(blk, params, rounds)
    elif kind == 'mlp':
        r, best = fit_mlp(blk, params)
    elif kind == 'mlp_apt':
        r, best = fit_mlp_per_airport(blk, params)
    elif kind == 'mlp_set':
        r, best = fit_mlp_set(blk, params)
    elif kind == 'lgb_adsb':
        r, best = tr.fit_lightgbm(with_adsb(blk), params, rounds, early_stopping=es)
    elif kind == 'lgb_sg_adsb':
        r, best = tr.fit_lightgbm(with_adsb(with_surge(blk)), params, rounds, early_stopping=es)
    elif kind == 'lgb_adsb_st':
        r, best = tr.fit_lightgbm(with_stops(blk), params, rounds, early_stopping=es)
    elif kind == 'lgb_sg_adsb_st':
        r, best = tr.fit_lightgbm(with_stops(with_surge(blk)), params, rounds, early_stopping=es)
    elif kind == 'lgb_adsb_st_ty':
        r, best = tr.fit_lightgbm(with_stand_type(with_stops(blk)), params, rounds, early_stopping=es)
    elif kind == 'lgb_sg_adsb_st_ty':
        r, best = tr.fit_lightgbm(with_stand_type(with_stops(with_surge(blk))), params, rounds, early_stopping=es)
    elif kind == 'mlp_set_adsb':
        r, best = fit_mlp_set(with_adsb(blk), params)
    elif kind == 'lgb_adsb_apt':
        r, best = fit_lightgbm_per_airport(with_adsb(blk), params, rounds)
    elif kind == 'lgb_sg_adsb_apt':
        r, best = fit_lightgbm_per_airport(with_adsb(with_surge(blk)), params, rounds)
    elif kind == 'lgb_adsb_debias':
        r, best = tr.fit_lightgbm(only_trust(with_adsb_debias(with_adsb(blk))), params, rounds, early_stopping=es)
    elif kind == 'lgb_sg_adsb_debias':
        r, best = tr.fit_lightgbm(only_trust(with_adsb_debias(with_adsb(with_surge(blk)))), params, rounds, early_stopping=es)
    elif kind == 'lgb_adsb_cens':
        r, best = tr.fit_lightgbm(only_cens(with_adsb(blk)), params, rounds, early_stopping=es)
    elif kind == 'lgb_sg_adsb_cens':
        r, best = tr.fit_lightgbm(only_cens(with_adsb(with_surge(blk))), params, rounds, early_stopping=es)
    elif kind == 'lgb_adsb_trust':
        r, best = tr.fit_lightgbm(only_trust(with_adsb(blk)), params, rounds, early_stopping=es)
    elif kind == 'lgb_sg_adsb_trust':
        r, best = tr.fit_lightgbm(only_trust(with_adsb(with_surge(blk))), params, rounds, early_stopping=es)
    elif kind == 'mlp_set_adsb_trust':
        r, best = fit_mlp_set(only_trust(with_adsb(blk)), params)
    elif kind == 'lgb_adsb_queue':
        r, best = tr.fit_lightgbm(with_queue(with_adsb(blk)), params, rounds, early_stopping=es)
    elif kind == 'lgb_adsb_queue_apt':
        r, best = fit_lightgbm_per_airport(with_queue(with_adsb(blk)), params, rounds)
    elif kind == 'mlp_set_adsb_queue':
        r, best = fit_mlp_set(with_queue(with_adsb(blk)), params)
    elif kind == 'lgb_sg_adsb_queue':
        r, best = tr.fit_lightgbm(with_queue(with_adsb(with_surge(blk))), params, rounds, early_stopping=es)
    elif kind == 'lgb_adsb_geom':
        r, best = tr.fit_lightgbm(with_geom(with_adsb(blk)), params, rounds, early_stopping=es)
    elif kind == 'mlp_set_adsb_geom':
        r, best = fit_mlp_set(with_geom(with_adsb(blk)), params)
    elif kind == 'lgb_adsb_nbr':
        r, best = tr.fit_lightgbm(with_nbr(with_adsb(blk)), params, rounds, early_stopping=es)
    elif kind == 'mlp_set_adsb_nbr':
        r, best = fit_mlp_set(with_nbr(with_adsb(blk)), params)
    elif kind == 'lgb_plus':
        r, best = tr.fit_lightgbm(with_surge(blk), params, rounds, early_stopping=es)
    elif kind == 'lgb_plus_apt':
        r, best = fit_lightgbm_per_airport(with_surge(blk), params, rounds)
    elif kind == 'mlp_plus':
        r, best = fit_mlp(with_surge(blk), params)
    elif kind == 'mlp_set_plus':
        r, best = fit_mlp_set(with_surge(blk), params)
    else:
        r, best = tr.fit_lightgbm(blk, params, rounds, early_stopping=es)
    res = pd.DataFrame({'MVT_ID_mvt': blk['held'].MVT_ID_mvt.values, 'r': r})
    cache.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(cache, index=False)
    print(f'{name}: fitted in {time.time() - t0:.0f}s, rounds {best}', flush=True)
    return res, {'seconds': round(time.time() - t0, 1), 'rounds': best}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--block', default='AB')
    p.add_argument('--candidates', nargs='+', default=['xgb_d10', 'lgb_extra'])
    p.add_argument('--average', nargs='*', help="groups of candidates averaged into one member, e.g. mlp+mlp_s1+mlp_s2")
    p.add_argument('--base-avg', nargs='*', default=[],
                   help="cached candidate groups added to the BASE members, e.g. mlp+mlp_s1+mlp_s2+mlp_e20+mlp_wide (A32's blend)")
    p.add_argument('--json')
    a = p.parse_args()
    t = rw.member_table(5, '_v5tfb5')
    t = t[t.block.values == a.block].reset_index(drop=True)
    base = np.column_stack([t.rg0.values, t.rg1.values, t.ra.values])
    report = {'block': a.block, 'rows_matched': int((t.fpl_matched.values == 1).sum()), 'base_avg': a.base_avg}
    print(f'block {a.block}: {len(t)} rows, {report["rows_matched"]} matched', flush=True)
    report['rd_fixed'] = pipeline_rmse(t, base, {x: (0.25, 0.25, 0.5) for x in t.APT.unique()}, t.fpl_matched.values == 1)
    print(f'  rd fixed 0.25/0.25/0.5: {report["rd_fixed"]:.3f}', flush=True)
    for group in a.base_avg:          # e.g. A32's five-MLP mean: new candidates are then scored on top of it
        rs = []
        for name in group.split('+'):
            res = pd.read_parquet(RUNS / 'tuning' / f'member_probe_{a.block}_{name}.parquet')
            r = t[['MVT_ID_mvt']].merge(res, on='MVT_ID_mvt', how='left', validate='one_to_one').r.values
            if not np.isfinite(r[t.has.values]).all():
                raise SystemExit(f'{name}: base member missing on matched rows (run it on block {a.block} first)')
            rs.append(np.where(t.has.values, r, 0.0))
        base = np.column_stack([base, np.mean(rs, axis=0)])
    # 'three' is the baseline key (three members, or more with --base-avg)
    report['three'] = evaluate(t, base, f'base ({base.shape[1]} members)')
    data = None
    extra_cols = []
    for name in a.candidates:
        if data is None:
            data, _, _ = common.load_features(5)
        blk = tr.load_block(a.block, 5, '_v5tfb5rd', data=data)
        res, fit_info = candidate_residual(name, blk)
        r = t[['MVT_ID_mvt']].merge(res, on='MVT_ID_mvt', how='left', validate='one_to_one').r.values
        if not np.isfinite(r[t.has.values]).all():
            raise SystemExit(f'{name}: missing residuals on matched rows')
        r = np.where(t.has.values, r, 0.0)
        alone = pipeline_rmse(t, r[:, None], {x: (1.0,) for x in t.APT.unique()}, t.fpl_matched.values == 1)
        corr = float(np.corrcoef(r[t.has.values], (0.25 * base[:, 0] + 0.25 * base[:, 1] + 0.5 * base[:, 2])[t.has.values])[0, 1])
        corr_last = float(np.corrcoef(r[t.has.values], base[t.has.values, -1])[0, 1])
        print(f'  {name} alone: {alone:.3f}; correlation with the rd residual {corr:.4f}, with the last base member {corr_last:.4f}', flush=True)
        four = evaluate(t, np.column_stack([base, r]), f'base + {name}')
        report[name] = {'alone': alone, 'corr_with_rd': corr, 'corr_with_last_base': corr_last, 'four': four, 'fit': fit_info,
                        'gain_cross_fitted': four['cross_fitted'] - report['three']['cross_fitted'],
                        'gain_in_sample': four['in_sample'] - report['three']['in_sample']}
        print(f'  => {name}: cross-fitted {report[name]["gain_cross_fitted"]:+.3f} s matched, in-sample {report[name]["gain_in_sample"]:+.3f}', flush=True)
        extra_cols.append(r)
    if len(extra_cols) > 1:
        if base.shape[1] + len(extra_cols) <= 6:      # the 0.05 simplex grid grows as 21^(k-1)
            allm = evaluate(t, np.column_stack([base] + extra_cols), 'base + ' + ' + '.join(a.candidates))
            report['all'] = {'members': a.candidates, 'eval': allm, 'gain_cross_fitted': allm['cross_fitted'] - report['three']['cross_fitted']}
            print(f'  => all candidates: cross-fitted {report["all"]["gain_cross_fitted"]:+.3f} s matched', flush=True)
        for group in a.average or []:
            names = group.split('+')
            avg = np.mean([extra_cols[a.candidates.index(n)] for n in names], axis=0)
            alone = pipeline_rmse(t, avg[:, None], {x: (1.0,) for x in t.APT.unique()}, t.fpl_matched.values == 1)
            ev = evaluate(t, np.column_stack([base, avg]), f'base + mean({group})')
            report[f'avg:{group}'] = {'alone': alone, 'eval': ev, 'gain_cross_fitted': ev['cross_fitted'] - report['three']['cross_fitted']}
            print(f'  => mean({group}) alone {alone:.3f}: cross-fitted {report[f"avg:{group}"]["gain_cross_fitted"]:+.3f} s matched', flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
