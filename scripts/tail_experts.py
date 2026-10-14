"""Three-expert tail mixture: normal taxi / schedule default / +24 h date bug.

Diagnosis (2026-09-17): the M3rpa per-airport clean expert
was trained on every non-fallback row, including 13 "date-bug" labels (12 LIRF + 1 LSZH, all
NM-unmatched) whose value is k*86400 s plus an ordinary taxi. For NM-unmatched LIRF rows with
long schedule delays that expert predicted 8-9 hour taxis, and the cost-weighted gate learned to
lean toward the schedule expert to escape it. Both halves of the mixture were distorted.

The fix keeps v18's structure and protocol and changes three things:
1. the clean expert is fitted without date-bug labels (they are not clean taxis);
2. the strict gate is refitted against that expert;
3. a third expert, ``k*86400 + normal``, receives probability ``Pb`` on LIRF NM-unmatched rows
   whose take-off local date is after the scheduled local date. ``Pb`` is a binned,
   Beta-smoothed rate by schedule delay, fitted on labels only and never extrapolated.

Final prediction on gate-valid rows:
    (1 - Pb) * (normal + W * (D - normal)) + Pb * (k * 86400 + normal)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import common
import strict_tail_gate as stg
import tail_gate
import train_eval
from common import RUNS

DATEBUG_MIN_LABEL = 72000.0
DATEBUG_MAX_REMAINDER = 5400.0
DAY = 86400.0
PB_EDGES = (0.0, 13.0, 16.0, 20.0, 26.0)
PB_LONG_FROM = 13.0
PB_STRENGTH = 2.0
LOCAL_TZ = 'Europe/Rome'
NODB_TAG = 'M3rpa_nodb'


RICH_GATE = ['fb_rate_airline_apt_unm', 'fb_rate_stand_unm', 'fb_rate_apt_hour_unm', 'fb_rate_airline_apt', 'fb_rate_stand',
             'te_airline_apt_mean']
STAND_GATE = ['st_last_minus_D', 'st_last2_minus_D', 'st_turn_minus_D', 'st_last_is_turn', 'st_prevdep_minus_D']


def add_stand_gate_features(d):
    """Stand-timeline gaps expressed relative to the schedule delay D (negative = after scheduled off-block)."""
    d = d.copy(); D = d.atot_minus_sched.values.astype(float)
    d['st_last_minus_D'] = d.st_last_in_gap.values - D
    d['st_last2_minus_D'] = d.st_last2_in_gap.values - D
    d['st_turn_minus_D'] = d.st_turn_in_gap.values - D
    d['st_last_is_turn'] = (np.abs(d.st_last_in_gap.values - d.st_turn_in_gap.values) < 1).astype(float)
    d['st_prevdep_minus_D'] = d.st_prev_dep_gap.values - D
    return d


def load_safe_rich():
    """Outer-safe G8 rates for every 2025 departure: fitted on the eight months outside its strict block."""
    parts = [pd.read_parquet(RUNS / 'oof' / f'outer_safe_rich_block_{l}{r}.parquet', columns=['MVT_ID_mvt'] + RICH_GATE)
             for l, r in stg.BLOCKS]
    s = pd.concat(parts, ignore_index=True)
    if s.MVT_ID_mvt.duplicated().any():
        raise ValueError('duplicate ids in outer-safe rich blocks')
    return s


def _replace_rich(frame, safe):
    d = frame.drop(columns=[c for c in RICH_GATE if c in frame.columns]).merge(safe, on='MVT_ID_mvt', how='left', validate='one_to_one')
    if d[RICH_GATE].isna().all(axis=None):
        raise ValueError('outer-safe rich alignment failed')
    return d


def datebug_label(y, schedule_delay):
    """Labels that equal k days plus an ordinary taxi and are not schedule defaults."""
    y = np.asarray(y, float); d = np.asarray(schedule_delay, float)
    with np.errstate(invalid='ignore'):
        return (y >= DATEBUG_MIN_LABEL) & (np.abs(y - d) > 10) & (np.mod(y, DAY) < DATEBUG_MAX_REMAINDER)


def training_exclusion(y, schedule_delay, kind='datebug'):
    """Rows kept out of every normal-taxi model's training. Evaluation always keeps them.

    datebug    k x 86400 + taxi labels (A19 onwards)
    artifacts  datebug, plus labels <= 60 s (no pushback and taxi fits in a minute), plus off-blocks recorded
               more than 3 h before the scheduled time on labels over 3 h (the off-block of another event,
               e.g. a tow or the previous evening). Schedule copies are already outside the clean experts.
    """
    y = np.asarray(y, float); d = np.asarray(schedule_delay, float)
    m = datebug_label(y, d)
    if kind == 'datebug':
        return m
    if kind == 'artifacts':
        with np.errstate(invalid='ignore'):
            return m | (y <= 60) | ((y > 3 * 3600) & (y > d + 3 * 3600))
    raise ValueError(f'unknown training exclusion {kind!r}')


def local_day_shift(takeoff_utc, sched_utc, tz=LOCAL_TZ):
    """Whole local calendar days between the scheduled time and the take-off (rank-available)."""
    def day(s):
        s = pd.to_datetime(pd.Series(s))
        if s.dt.tz is None:
            s = s.dt.tz_localize('UTC')
        return s.dt.tz_convert(tz).dt.tz_localize(None).dt.floor('D')
    return (day(takeoff_utc) - day(sched_utc)).dt.days.to_numpy()


def b_domain(apt, fpl_matched, day_shift, is_heli):
    """Rows eligible for the date-bug expert: LIRF, NM-unmatched, later local day, not helicopter."""
    return ((np.asarray(apt) == 'LIRF') & (np.asarray(fpl_matched) == 0)
            & (np.asarray(day_shift) >= 1) & (np.asarray(is_heli) == 0))


def fit_pb_bins(delay_hours, is_datebug, edges=PB_EDGES, strength=PB_STRENGTH, long_from=PB_LONG_FROM):
    """Beta-smoothed date-bug rate per schedule-delay bin.

    Bins at or beyond ``long_from`` shrink toward the pooled long-delay rate; shorter bins shrink
    toward zero. The last bin is open-ended, so no rate is extrapolated from a fitted slope.
    """
    x = np.asarray(delay_hours, float); b = np.asarray(is_datebug, bool)
    long = x >= long_from
    pooled = (b[long].sum() + 0.5) / (long.sum() + 1.0)
    bounds = list(edges) + [np.inf]
    rates = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        s = (x >= lo) & (x < hi)
        prior = pooled if lo >= long_from else 0.5 / (s.sum() + 1.0)
        rates.append((b[s].sum() + strength * prior) / (s.sum() + strength))
    return np.asarray(rates, float)


def apply_pb_bins(rates, delay_hours, domain, edges=PB_EDGES):
    idx = np.clip(np.searchsorted(np.asarray(edges, float), np.asarray(delay_hours, float), side='right') - 1,
                  0, len(rates) - 1)
    return np.where(np.asarray(domain, bool), np.asarray(rates)[idx], 0.0)


def combine(two_expert_pred, pb, normal, day_shift, floor=train_eval.FLOOR):
    k = np.clip(np.asarray(day_shift, float), 1, None)
    pb = np.asarray(pb, float)
    out = (1.0 - pb) * np.asarray(two_expert_pred, float) + pb * (k * DAY + np.asarray(normal, float))
    return np.maximum(out, floor)


LONG_COPY_HOURS = 7.0


def long_delay_copy(pred, apt, fpl_matched, is_heli, schedule_delay, pb, normal, day_shift, hours=LONG_COPY_HOURS,
                    floor=train_eval.FLOOR):
    """LIRF NM-unmatched fixed-wing departures delayed more than ``hours``: schedule copy or date bug, never a real taxi.

    In 2025 the longest schedule delay with a real recorded off-block among these rows was 6.4 h; above 7 h there
    were 43 schedule copies, 11 date bugs and no real taxis. The gate's weight W still leaves 5-20 % on the
    normal expert there (its leaves average with shorter delays), which costs thousands of seconds per row.
    Those rows get W = 1: ``(1 - Pb) * D + Pb * (k * 86400 + normal)``. Returns the new predictions and the mask.
    """
    d = np.asarray(schedule_delay, float)
    with np.errstate(invalid='ignore'):
        m = ((np.asarray(apt) == 'LIRF') & (np.asarray(fpl_matched) == 0) & (np.asarray(is_heli) == 0)
             & np.isfinite(d) & (d > hours * 3600.0))
    k = np.clip(np.asarray(day_shift, float), 1, None); pb = np.asarray(pb, float)
    out = np.asarray(pred, float).copy()
    out[m] = np.maximum((1.0 - pb[m]) * d[m] + pb[m] * (k[m] * DAY + np.asarray(normal, float)[m]), floor)
    return out, m


def with_expert(frame, normal_override=None):
    """Recompute every gate input that depends on the clean expert.

    ``normal_override`` maps MVT_ID_mvt -> expert for NM-unmatched rows; matched rows keep the
    saved AOBT_3-residual expert, which the date-bug exclusion does not touch (all 13 date-bug
    labels are NM-unmatched).
    """
    d = frame.copy()
    if normal_override is not None:
        unm = d.fpl_matched.values == 0
        new = d.loc[unm, 'MVT_ID_mvt'].map(normal_override)
        if new.isna().any():
            raise ValueError(f'missing override expert for {int(new.isna().sum())} unmatched rows')
        d.loc[unm, 'normal_pred'] = new.values
    e = d.normal_pred.values.astype(float)
    d['expert_gap'] = d.atot_minus_sched.values - e
    d['abs_expert_gap'] = np.abs(d.expert_gap.values)
    d['log_abs_expert_gap'] = np.log1p(d.abs_expert_gap.values)
    valid = np.isfinite(d.atot_minus_sched.values) & (d.is_heli.values == 0)
    d['gate_valid'] = valid & (d.fpl_matched.values == 0) & (d.abs_expert_gap.values >= 60)
    if 'y' in d and np.isfinite(d.y.values.astype(float)).any():
        raw = np.divide(d.y.values - e, d.expert_gap.values, out=np.zeros(len(d)),
                        where=np.abs(d.expert_gap.values) >= 60)
        d['optimal_weight'] = np.clip(raw, 0, 1)
    return d


def _nodb_path(left, right):
    return RUNS / 'oof' / f'strict_block_{left}{right}_nodb_expert.parquet'


def retrain_block_experts(feature_version=3, rounds=660):
    """Strict-block clean expert without date-bug labels, scored on the blocked NM-unmatched rows."""
    data, _, manifest = common.load_features(feature_version)
    target_cols = set(manifest['groups'].get('G8', [])) | stg.TARGET_ROUTE | stg.TARGET_G9
    cols = [c for c in train_eval.feature_columns(manifest, list(manifest['groups'])) if c not in target_cols]
    cats = train_eval._cats(manifest, cols)
    bug = datebug_label(data.TAXITIME_SEC_mvt.values, data.atot_minus_sched.values)
    meta = {'datebug_labels': int(bug.sum()), 'datebug_matched': int(data.fpl_matched.values[bug].sum()), 'blocks': {}}
    for left, right in stg.BLOCKS:
        blocked = common.FOLDS[left] + common.FOLDS[right]
        train_mask = ~data.month.isin(blocked).values
        base = data[train_mask]
        clean = base[(base.fallback.values == 0) & (base.is_heli.values == 0) & ~bug[train_mask]]
        score = data[data.month.isin(blocked).values & (data.fpl_matched.values == 0)]
        reg = train_eval._m2(clean, score, cols, cats, {}, rounds, NODB_TAG)
        pd.DataFrame({'MVT_ID_mvt': score.MVT_ID_mvt.values, 'normal_nodb': reg}).to_parquet(_nodb_path(left, right), index=False)
        meta['blocks'][f'{left}{right}'] = {'clean_rows': int(len(clean)), 'scored_unmatched': int(len(score))}
    return meta


def _frame_path(left, right, fold, suffix):
    return RUNS / 'oof' / f'strict_block_{left}{right}_fold{fold}{suffix}.parquet'


def build_strict_frames(feature_version, suffix, rounds=660, extra_drop=(), params=None, exclusion='datebug'):
    """Blocked-pair M3rpa frames on another feature version, with date-bug labels excluded.

    Same protocol as ``strict_tail_gate.build_block_predictions`` (other eight months, target
    predictors dropped, 660 rounds), but the fallback probability is reused from the audited v3
    frames because the detector's inputs are unchanged. ``exclusion`` selects the training rows
    left out of the experts (``training_exclusion``); scored rows are never filtered.
    """
    data, _, manifest = common.load_features(feature_version)
    target_cols = set(manifest['groups'].get('G8', [])) | stg.TARGET_ROUTE | stg.TARGET_G9 | set(extra_drop)
    cols = [c for c in train_eval.feature_columns(manifest, list(manifest['groups'])) if c not in target_cols]
    cats = train_eval._cats(manifest, cols)
    bug = training_exclusion(data.TAXITIME_SEC_mvt.values, data.atot_minus_sched.values, exclusion)
    heli = stg._heli_const
    meta = {'feature_version': feature_version, 'n_model_features': len(cols), 'rounds': rounds, 'params': dict(params or {}),
            'exclusion': exclusion, 'excluded_rows': int(bug.sum()), 'blocks': {}}
    for left, right in stg.BLOCKS:
        blocked = common.FOLDS[left] + common.FOLDS[right]
        train_mask = ~data.month.isin(blocked).values
        base_train = data[train_mask & ~bug].copy()
        score = data[data.month.isin(blocked).values].copy()
        saved = pd.concat([pd.read_parquet(_frame_path(left, right, f, '')) for f in (left, right)], ignore_index=True)
        score = score.merge(saved[['MVT_ID_mvt', 'p_fallback']], on='MVT_ID_mvt', how='left', validate='one_to_one',
                            suffixes=('_v3feature', ''))
        if not np.isfinite(score.p_fallback.values.astype(float)).all():
            raise ValueError('missing saved strict p_fallback')
        pred, normal = train_eval.predict_model('M3rpa', base_train, score, cols, cats, dict(params or {}), rounds,
                                                heli(data[train_mask]), return_normal=True)
        for fold in (left, right):
            m = score.month.isin(common.FOLDS[fold]).values
            pd.DataFrame({'MVT_ID_mvt': score.MVT_ID_mvt.values[m], 'APT': score.APT.values[m], 'month': score.month.values[m],
                          'y': score.TAXITIME_SEC_mvt.values[m], 'pred': pred[m], 'normal_pred': normal[m],
                          'fpl_matched': score.fpl_matched.values[m], 'p_fallback': score.p_fallback.values[m]}
                         ).to_parquet(_frame_path(left, right, fold, suffix), index=False)
        meta['blocks'][f'{left}{right}'] = {'train_rows': int(len(base_train)), 'score_rows': int(len(score))}
    return meta


def load_frames(feature_version, suffix):
    """Strict frames prepared exactly like ``strict_tail_gate.load_saved_frames``."""
    if not suffix:
        return stg.load_saved_frames(feature_version)[0]
    features = pd.read_parquet(common.DATA / f'features_v{feature_version}_train.parquet')
    frames = {}
    for left, right in stg.BLOCKS:
        for fold in (left, right):
            frames[fold] = tail_gate.prepare_frame(pd.read_parquet(_frame_path(left, right, fold, suffix)), features)
    return frames


def load_nodb_override():
    parts = [pd.read_parquet(_nodb_path(l, r)) for l, r in stg.BLOCKS]
    s = pd.concat(parts, ignore_index=True)
    if s.MVT_ID_mvt.duplicated().any():
        raise ValueError('duplicate ids across strict nodb expert blocks')
    return s.set_index('MVT_ID_mvt').normal_nodb


def _with_day_shift(frames):
    base = pd.read_parquet(common.DATA / 'base_train.parquet', columns=['MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'])
    base['day_shift'] = local_day_shift(base.MVT_TIME_UTC_mvt, base.SCHED_TIME_UTC_mvt)
    base = base[['MVT_ID_mvt', 'day_shift']]
    return {k: v.merge(base, on='MVT_ID_mvt', how='left', validate='one_to_one') for k, v in frames.items()}


def _segments(frame, pred, base_pred):
    y = frame.y.values; rows = []
    groups = {
        'APT': frame.APT.values,
        'lane': np.where(frame.fpl_matched.values == 1, 'matched', 'unmatched'),
        'label_tail': pd.cut(y, [-np.inf, 3600, 14400, 72000, np.inf], labels=['body', '1h_4h', '4h_20h', 'over_20h']).astype(str),
        'fold': frame.fold.values,
    }
    tot = float(((base_pred - y) ** 2).sum())
    for name, g in groups.items():
        for key in pd.unique(g):
            m = g == key
            rows.append(dict(segment=name, key=str(key), n=int(m.sum()),
                             rmse_v18=float(np.sqrt(np.mean((base_pred[m] - y[m]) ** 2))),
                             rmse_new=float(np.sqrt(np.mean((pred[m] - y[m]) ** 2))),
                             sse_delta_share=float((((pred[m] - y[m]) ** 2).sum() - ((base_pred[m] - y[m]) ** 2).sum()) / tot)))
    return pd.DataFrame(rows)


def strict_diagnostic(feature_version=3, output=None, segments_output=None, use_datebug_expert=True,
                      edges=PB_EDGES, strength=PB_STRENGTH, long_from=PB_LONG_FROM,
                      candidate_version=None, candidate_suffix='', rich_gate=False, gate_extra=()):
    """v18's blocked-pair protocol with the no-date-bug expert, refitted gate and date-bug expert.

    Gate: fitted on one blocked pair, evaluated on the other (v18 params). With ``rich_gate`` the
    gate also sees outer-safe fallback/airline rates (fitted on the block's other eight months) and
    the stand timeline relative to the schedule delay (needs a v4 candidate).
    Pb: label-only bins fitted on every fold except the evaluated one.
    The v18 reference always uses the audited v3 frames. With ``candidate_suffix`` the candidate
    uses frames from ``build_strict_frames`` (already date-bug free, so no expert override).
    """
    frames, _ = stg.load_saved_frames(feature_version)
    frames = _with_day_shift(frames)
    if candidate_suffix:
        cand = _with_day_shift(load_frames(candidate_version or feature_version, candidate_suffix))
        prepared = {k: with_expert(v) for k, v in cand.items()}
        for k in prepared:
            if not np.array_equal(prepared[k].MVT_ID_mvt.values, frames[k].MVT_ID_mvt.values):
                raise ValueError(f'candidate frame {k} ids differ from v18 frame')
    else:
        override = load_nodb_override()
        prepared = {k: with_expert(v, override) for k, v in frames.items()}
    gate_features = list(stg.STRICT_GATE_FEATURES)
    if rich_gate:
        safe = load_safe_rich()
        prepared = {k: add_stand_gate_features(_replace_rich(v, safe)) for k, v in prepared.items()}
        gate_features += RICH_GATE + STAND_GATE
    gate_features += [c for c in gate_extra if c not in gate_features]
    for v in prepared.values():
        v['b_domain'] = b_domain(v.APT.values, v.fpl_matched.values, v.day_shift.values, v.is_heli.values)
        v['datebug'] = datebug_label(v.y.values, v.atot_minus_sched.values)
    base_frames = {}
    preds, bases, keep = [], [], []
    report = {'protocol': 'blocked_pair_strict_three_expert', 'feature_version': feature_version,
              'candidate_version': candidate_version or feature_version, 'candidate_suffix': candidate_suffix,
              'gate_features': gate_features,
              'pb': {'edges': list(edges), 'strength': strength, 'long_from': long_from, 'enabled': use_datebug_expert},
              'folds': {}}
    for left, right in stg.BLOCKS:
        for cal, test in ((left, right), (right, left)):
            a, b = prepared[cal], prepared[test]
            w, _, _ = tail_gate.fit_predict_gate(a, b, features=gate_features, cats=stg.STRICT_GATE_CATS)
            two = tail_gate.apply_gate(b, w, 1.0)
            if use_datebug_expert:
                others = pd.concat([prepared[f] for f in prepared if f != test], ignore_index=True)
                o = others[others.b_domain]
                rates = fit_pb_bins(o.atot_minus_sched.values / 3600.0, o.datebug.values, edges, strength, long_from)
                pb = apply_pb_bins(rates, b.atot_minus_sched.values / 3600.0, b.b_domain.values, edges)
                pred = combine(two, pb, b.normal_pred.values, b.day_shift.values)
            else:
                rates, pb, pred = np.array([]), np.zeros(len(b)), two
            # v18 reference on the same rows: saved expert, original gate
            v18_frame = frames[test]
            w0, _, _ = tail_gate.fit_predict_gate(frames[cal], v18_frame, features=stg.STRICT_GATE_FEATURES, cats=stg.STRICT_GATE_CATS)
            v18 = tail_gate.apply_gate(v18_frame, w0, 1.0)
            y = b.y.values
            report['folds'][test] = {'calibration': cal, 'rmse_v18': float(np.sqrt(np.mean((v18 - y) ** 2))),
                                     'rmse_new': float(np.sqrt(np.mean((pred - y) ** 2))),
                                     'pb_rates': [round(float(r), 4) for r in rates],
                                     'n_pb_rows': int((pb > 0).sum())}
            report['folds'][test]['delta'] = report['folds'][test]['rmse_new'] - report['folds'][test]['rmse_v18']
            preds.append(pred); bases.append(v18); keep.append(b.assign(fold=test, pb=pb, w=w, pred=pred, v18_pred=v18))
    frame = pd.concat(keep, ignore_index=True)
    P, B = np.concatenate(preds), np.concatenate(bases)
    mn = common.metrics(P, frame.y.values, frame.APT.values, frame.month.values, frame.fpl_matched.values)
    mb = common.metrics(B, frame.y.values, frame.APT.values, frame.month.values, frame.fpl_matched.values)
    report['pooled'] = {'v18': {k: v for k, v in mb.items()}, 'new': {k: v for k, v in mn.items()},
                        'delta': mn['rmse_raw'] - mb['rmse_raw'],
                        'worst_fold_delta': max(v['delta'] for v in report['folds'].values()),
                        'n_rows': int(len(frame)), 'ids_unique': bool(frame.MVT_ID_mvt.is_unique)}
    report['decision_rule'] = 'promote if pooled delta <= -1.0 s and no fold delta > +1.0 s'
    report['passes'] = bool(report['pooled']['delta'] <= -1.0 and report['pooled']['worst_fold_delta'] <= 1.0)
    output = Path(output) if output else RUNS / 'three_expert_tail_strict.json'
    output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    seg = _segments(frame, P, B)
    seg.to_csv(segments_output or output.with_name(output.stem + '_segments.csv'), index=False)
    return report, frame


def fit_rank_expert(package_dir, rounds=None, save_dir=None, feature_version=None):
    """Full-2025 per-airport clean expert without date-bug labels, with the package's own features.

    Matched and helicopter rank rows keep the package's saved expert (residual/heli constant).
    """
    package_dir = Path(package_dir)
    model_dir = package_dir / 'M3rpa'
    state = json.loads((model_dir / 'state.json').read_text(encoding='utf-8'))
    version = int(feature_version or state['feature_version'])
    rounds = int(rounds or state['rounds'])
    train, rank, _ = common.load_features(version)
    saved = pd.read_parquet(package_dir / 'normal_rank.parquet')
    normal = rank[['MVT_ID_mvt']].merge(saved, on='MVT_ID_mvt', how='left', validate='one_to_one').normal_pred.values
    bug = datebug_label(train.TAXITIME_SEC_mvt.values, train.atot_minus_sched.values)
    clean = train[(train.fallback.values == 0) & (train.is_heli.values == 0) & ~bug]
    score_mask = (rank.fpl_matched.values == 0) & (rank.is_heli.values == 0) & ~np.isfinite(rank.atot_minus_aobt3.values.astype(float))
    old_save = train_eval.SAVE_DIR
    try:
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            train_eval.SAVE_DIR = str(save_dir); train_eval._SAVE_N[0] = 0
        reg = train_eval._m2(clean, rank[score_mask], state['cols'], state['cats'], state.get('params', {}), rounds, NODB_TAG)
    finally:
        train_eval.SAVE_DIR = old_save
    new = normal.copy(); new[score_mask] = reg
    if not np.isfinite(new).all():
        raise ValueError('non-finite rank expert')
    meta = {'package': str(package_dir), 'rounds': rounds, 'datebug_labels_removed': int(bug.sum()),
            'clean_rows': int(len(clean)), 'rank_rows_refit': int(score_mask.sum()),
            'changed_rows': int((np.abs(new - normal) > 1e-9).sum())}
    return rank, new, state, meta


def fit_full_m3rpa(feature_version, save_dir, rounds=660, p_scale=1.4, exclude=(), params=None, exclusion='datebug'):
    """A13-style M3rpa on all 2025 (every feature group) with date-bug labels excluded.

    ``exclude`` drops columns, e.g. the target-derived set the strict frames never see. ``exclusion``
    selects the training rows left out (``training_exclusion``).
    """
    train, rank, manifest = common.load_features(feature_version)
    cols = [c for c in train_eval.feature_columns(manifest, list(manifest['groups'])) if c not in set(exclude)]
    cats = train_eval._cats(manifest, cols)
    bug = training_exclusion(train.TAXITIME_SEC_mvt.values, train.atot_minus_sched.values, exclusion)
    heli = train_eval._heli_const()
    old = train_eval.SAVE_DIR
    try:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        train_eval.SAVE_DIR = str(save_dir); train_eval._SAVE_N[0] = 0
        _, normal = train_eval.predict_model('M3rpa', train[~bug], rank, cols, cats, {**dict(params or {}), 'p_scale': p_scale}, rounds, heli,
                                             return_normal=True)
    finally:
        train_eval.SAVE_DIR = old
    normal = np.where(rank.is_heli.values == 1, heli, normal)
    state = {'model_id': 'M3rpa', 'feature_version': feature_version, 'groups': list(manifest['groups']), 'cols': cols, 'cats': cats,
             'params': {**dict(params or {}), 'p_scale': p_scale}, 'rounds': rounds, 'heli_const': heli,
             'datebug_labels_removed': int(bug.sum()), 'exclusion': exclusion}
    (Path(save_dir) / 'state.json').write_text(json.dumps(state, indent=1), encoding='utf-8')
    return rank, normal, state


def rank_prediction(rank, normal, state, feature_version=3, base_p_scale=1.4,
                    edges=PB_EDGES, strength=PB_STRENGTH, long_from=PB_LONG_FROM,
                    use_nodb_frames=True, use_datebug_expert=True, frame_version=3, frame_suffix='', rich_gate=False,
                    rank_p_fallback=None, matched_p_scale=None, gate_extra=()):
    """Deployment: A13-style mixture base, gate fitted on all six strict frames, date-bug expert.

    With ``use_nodb_frames=False``, ``use_datebug_expert=False`` and the package's saved expert this
    is exactly the v18 deployment path (used as a regression check). ``frame_suffix`` selects frames
    from ``build_strict_frames`` (already date-bug free). With ``rich_gate`` the gate trains on
    outer-safe rates (as evaluated) and scores rank rows with the full-2025 G8 rates.
    ``rank_p_fallback`` (aligned with ``rank``) replaces the feature table's detector probability in
    both the mixture and the gate input. ``matched_p_scale`` overrides ``base_p_scale`` on NM-matched
    rows only (the strict protocol scores matched rows with scale 1.0).
    """
    if rank_p_fallback is not None:
        rank = rank.copy()
        rank['p_fallback'] = np.asarray(rank_p_fallback, float)
    base = train_eval._mixture(rank, normal, {'p_scale': base_p_scale}, float(state['heli_const']))
    if matched_p_scale is not None:
        matched_base = train_eval._mixture(rank, normal, {'p_scale': matched_p_scale}, float(state['heli_const']))
        base = np.where(rank.fpl_matched.values == 1, matched_base, base)
    base = np.maximum(base, train_eval.FLOOR)
    frames = _with_day_shift(load_frames(frame_version, frame_suffix))
    override = load_nodb_override() if (use_nodb_frames and not frame_suffix) else None
    gate_features = list(stg.STRICT_GATE_FEATURES)
    if rich_gate:
        safe = load_safe_rich()
        frames = {k: add_stand_gate_features(_replace_rich(v, safe)) for k, v in frames.items()}
        gate_features += RICH_GATE + STAND_GATE
    gate_features += [c for c in gate_extra if c not in gate_features]
    gate_train = pd.concat([with_expert(v, override) for v in frames.values()], ignore_index=True)
    gate_train['b_domain'] = b_domain(gate_train.APT.values, gate_train.fpl_matched.values, gate_train.day_shift.values, gate_train.is_heli.values)
    gate_train['datebug'] = datebug_label(gate_train.y.values, gate_train.atot_minus_sched.values)
    rank_features = pd.read_parquet(common.DATA / f'features_v{feature_version}_rank.parquet')
    if rank_p_fallback is not None:
        if not np.array_equal(rank_features.MVT_ID_mvt.values, rank.MVT_ID_mvt.values):
            raise ValueError('rank_p_fallback must be aligned with the feature table rows')
        rank_features['p_fallback'] = rank.p_fallback.values
    skeleton = rank_features[['MVT_ID_mvt', 'APT', 'month', 'fpl_matched']].copy()
    skeleton['pred'] = base; skeleton['normal_pred'] = normal; skeleton['y'] = np.nan
    gate_rank = tail_gate.prepare_frame(skeleton, rank_features, base_p_scale=base_p_scale)
    gate_rank = with_expert(gate_rank)
    if rich_gate:
        gate_rank = add_stand_gate_features(gate_rank)
    w, model, gate_meta = tail_gate.fit_predict_gate(gate_train, gate_rank, features=gate_features, cats=stg.STRICT_GATE_CATS)
    two = tail_gate.apply_gate(gate_rank, w, 1.0)
    times = pd.read_parquet(common.DATA / 'base_rank.parquet', columns=['MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'])
    times['day_shift'] = local_day_shift(times.MVT_TIME_UTC_mvt, times.SCHED_TIME_UTC_mvt)
    shift = gate_rank[['MVT_ID_mvt']].merge(times[['MVT_ID_mvt', 'day_shift']], on='MVT_ID_mvt', how='left', validate='one_to_one').day_shift.values
    dom = b_domain(gate_rank.APT.values, gate_rank.fpl_matched.values, shift, gate_rank.is_heli.values)
    if use_datebug_expert:
        t = gate_train[gate_train.b_domain]
        rates = fit_pb_bins(t.atot_minus_sched.values / 3600.0, t.datebug.values, edges, strength, long_from)
        pb = apply_pb_bins(rates, gate_rank.atot_minus_sched.values / 3600.0, dom, edges)
        pred = combine(two, pb, gate_rank.normal_pred.values, shift)
    else:
        rates, pb, pred = np.array([]), np.zeros(len(gate_rank)), two
    detail = gate_rank[['MVT_ID_mvt', 'APT', 'fpl_matched', 'atot_minus_sched', 'normal_pred', 'gate_valid']].assign(
        base_pred=base, w=w, pb=pb, day_shift=shift, pred=pred)
    return pred, detail, model, {'gate': gate_meta, 'pb_rates': [float(r) for r in rates],
                                 'pb_edges': list(edges), 'pb_strength': strength, 'n_pb_rows': int((pb > 0).sum())}
