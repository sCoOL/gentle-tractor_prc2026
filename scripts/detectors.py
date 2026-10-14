"""Detectors: 1 fallback (v1, v2, unmatched-lane specialist), 2 helicopters, 3 route periods + 2026 drift, 4 runway configuration,
5 disruption days, 6 unseen stands, 7 source conflicts."""
from __future__ import annotations
import json, re, time
import numpy as np, pandas as pd, lightgbm as lgb
import common
from common import DATA

FALLBACK_FEATS = ['APT', 'apt_rwy', 'apt_stand', 'actype', 'wake', 'segment', 'operator', 'flight_type', 'ades_region', 'fpl_matched', 'is_heli',
                  'hour_utc', 'dow', 'month', 'minute_of_day_utc', 'atot_minus_sched', 'atot_minus_eobt1', 'atot_minus_lobt', 'atot_minus_aobt3',
                  'aobt3_minus_lobt', 'lobt_minus_sched', 'plan_block_time', 'dep_prev15', 'dep_prev60', 'arr_pm15', 'dep_same_rwy_pm10', 'n_out_at_aobt3']
FALLBACK_CATS = ['APT', 'apt_rwy', 'apt_stand', 'actype', 'wake', 'segment', 'operator', 'flight_type', 'ades_region']
CLF_PARAMS = dict(objective='binary', learning_rate=0.08, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=10, cat_smooth=50, cat_l2=10, max_cat_to_onehot=4, verbose=-1)
FALLBACK_FEATS_V2 = FALLBACK_FEATS + ['airline', 'flight_is_callsign', 'arr_fb_rate_airline_apt', 'arr_fb_rate_airline_apt_month', 'sched_eq_eobt1', 'lobt_minus_iobt', 'dep_same_rwy_prev30', 'arr_prev60', 'doy']
FALLBACK_CATS_V2 = FALLBACK_CATS + ['airline']


def fallback_label(df: pd.DataFrame) -> pd.Series:
    if 'fallback' in df:
        return df.fallback.astype(int)
    return ((df.b - df.sched_s).abs() <= 10).astype(int)


def _prep(df, cols, cats, categories=None):
    X = df[cols].copy()
    for c in cats:
        X[c] = pd.Categorical(X[c].astype(str), categories=None if categories is None else categories[c])
    return X


def _auc(y, s):
    r = pd.Series(s).rank().values; n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')


def detect_fallback(train, rank, feature_cols=FALLBACK_FEATS, cat_cols=FALLBACK_CATS, n_folds=5, params=None, rounds=400, fold_ids=None):
    """OOF p_fallback for training rows and full-fit p_fallback for scoring rows."""
    P = {**CLF_PARAMS, **(params or {})}; y = fallback_label(train).values
    Xall = _prep(train, feature_cols, cat_cols); cats = {c: Xall[c].cat.categories for c in cat_cols}
    oof = np.zeros(len(train)); fold = train.month.values % n_folds if fold_ids is None else np.asarray(fold_ids)
    if len(fold) != len(train): raise ValueError('fold_ids length must match train')
    for k in np.unique(fold):
        tr_m, te_m = fold != k, fold == k
        m = lgb.train(P, lgb.Dataset(Xall[tr_m], y[tr_m], categorical_feature=cat_cols), rounds)
        oof[te_m] = m.predict(Xall[te_m])
    full = lgb.train(P, lgb.Dataset(Xall, y, categorical_feature=cat_cols), rounds)
    p_rank = full.predict(_prep(rank, feature_cols, cat_cols, cats))
    return (pd.DataFrame({'MVT_ID_mvt': train.MVT_ID_mvt.values, 'fallback': y, 'p_fallback': oof}),
            pd.DataFrame({'MVT_ID_mvt': rank.MVT_ID_mvt.values, 'p_fallback': p_rank}))


def score_fallback(train, score, feature_cols=FALLBACK_FEATS, cat_cols=FALLBACK_CATS,
                   params=None, rounds=400):
    """Fit once on ``train`` and score a disjoint frame; no OOF work or target reuse."""
    P = {**CLF_PARAMS, **(params or {})}; y = fallback_label(train).values
    X = _prep(train, feature_cols, cat_cols)
    categories = {c: X[c].cat.categories for c in cat_cols}
    model = lgb.train(P, lgb.Dataset(X, y, categorical_feature=cat_cols), rounds)
    return model.predict(_prep(score, feature_cols, cat_cols, categories))


def detect_fallback_lane(train, rank, feature_cols=FALLBACK_FEATS_V2, cat_cols=FALLBACK_CATS_V2, n_folds=5, params=None, rounds=300, fold_ids=None):
    """Specialist classifier fitted on NM-unmatched rows only (operator/wake/segment are null there; airline carries the signal)."""
    tr = train[train.fpl_matched.values == 0]; rk = rank[rank.fpl_matched.values == 0]
    P = {**CLF_PARAMS, 'num_leaves': 63, 'min_data_in_leaf': 40, **(params or {})}; y = fallback_label(tr).values
    Xall = _prep(tr, feature_cols, cat_cols); cats = {c: Xall[c].cat.categories for c in cat_cols}
    oof = np.zeros(len(tr))
    fold = tr.month.values % n_folds if fold_ids is None else np.asarray(fold_ids)[train.fpl_matched.values == 0]
    if len(fold) != len(tr): raise ValueError('fold_ids length must match train')
    for k in np.unique(fold):
        a, b = fold != k, fold == k
        if b.sum() == 0: continue
        m = lgb.train(P, lgb.Dataset(Xall[a], y[a], categorical_feature=cat_cols), rounds); oof[b] = m.predict(Xall[b])
    full = lgb.train(P, lgb.Dataset(Xall, y, categorical_feature=cat_cols), rounds)
    p_rank = full.predict(_prep(rk, feature_cols, cat_cols, cats)) if len(rk) else np.zeros(0)
    return (pd.DataFrame({'MVT_ID_mvt': tr.MVT_ID_mvt.values, 'p_fallback_lane': oof}), pd.DataFrame({'MVT_ID_mvt': rk.MVT_ID_mvt.values, 'p_fallback_lane': p_rank}))


def permutation_auc(train, feature_cols=FALLBACK_FEATS_V2, cat_cols=FALLBACK_CATS_V2, rounds=150, params=None, n_folds=5, seed=0):
    """OOF AUC after shuffling the label within airport: the noise floor a classifier reaches on nothing (expect ~0.5)."""
    rng = np.random.default_rng(seed); y = fallback_label(train).values.copy()
    for apt in np.unique(train.APT.values):
        m = train.APT.values == apt; y[m] = rng.permutation(y[m])
    P = {**CLF_PARAMS, 'num_leaves': 63, 'min_data_in_leaf': 40, **(params or {})}; X = _prep(train, feature_cols, cat_cols); oof = np.zeros(len(train)); fold = train.month.values % n_folds
    for k in range(n_folds):
        a, b = fold != k, fold == k
        if b.sum() == 0 or y[a].sum() == 0: continue
        oof[b] = lgb.train(P, lgb.Dataset(X[a], y[a], categorical_feature=cat_cols), rounds).predict(X[b])
    return _auc(y, oof)


def detect_helicopters(train):
    flag = ((train.RUNWAY_mvt == 'H') | train.AIRCRAFT_TYPE_mvt.isin(common.HELI_TYPES)).astype(int)
    const = float(train.loc[flag == 1, 'TAXITIME_SEC_mvt'].mean()) if flag.sum() else 120.0
    return flag, const


def detect_source_conflicts(df):
    return pd.DataFrame({'MVT_ID_mvt': df.MVT_ID_mvt.values,
                         'src_conflict_adep': (df.ADEP_mvt != df.ADEP_flt).fillna(False).astype(int).values,
                         'src_conflict_ades': (df.ADES_mvt != df.ADES_flt).fillna(False).astype(int).values,
                         'src_conflict_type': (df.AIRCRAFT_TYPE_mvt != df.AIRCRAFT_TYPE_flt).fillna(False).astype(int).values})


def run_all(force=False):
    common.ensure_dirs(); t0 = time.time()
    if not force and (DATA / 'det_fallback_rank.parquet').exists(): print('detector tables exist; --force to rebuild'); return
    tr = pd.read_parquet(DATA / 'base_train.parquet'); rk = pd.read_parquet(DATA / 'base_rank.parquet')
    a, b = detect_fallback(tr, rk); a.to_parquet(DATA / 'det_fallback_train.parquet'); b.to_parquet(DATA / 'det_fallback_rank.parquet')
    y, p = a.fallback.values, a.p_fallback.values; auc = _auc(y, p)
    print(f'  fallback: rate {y.mean():.4f}, OOF AUC {auc:.3f}, by airport {pd.Series(y).groupby(tr.APT.values).mean().round(3).to_dict()} [{time.time()-t0:.0f}s]', flush=True)
    flag, const = detect_helicopters(tr); json.dump({'heli_const': const, 'n_train': int(flag.sum())}, open(DATA / 'det_heli.json', 'w'))
    print(f'  helicopters: {int(flag.sum())} training rows, constant {const:.0f} s', flush=True)
    detect_source_conflicts(tr).to_parquet(DATA / 'det_conflicts_train.parquet'); detect_source_conflicts(rk).to_parquet(DATA / 'det_conflicts_rank.parquet')
    print('  conflicts written', flush=True)


def run_fallback_v2(force=False):
    common.ensure_dirs(); t0 = time.time()
    if not force and (DATA / 'det_fallback_v2_rank.parquet').exists(): print('det_fallback_v2 exists; --force to rebuild'); return
    tr = pd.read_parquet(DATA / 'base_train.parquet'); rk = pd.read_parquet(DATA / 'base_rank.parquet')
    a, b = detect_fallback(tr, rk, FALLBACK_FEATS_V2, FALLBACK_CATS_V2); a = a.rename(columns={'p_fallback': 'p_fallback_v2'}); b = b.rename(columns={'p_fallback': 'p_fallback_v2'})
    la, lb = detect_fallback_lane(tr, rk); a = a.merge(la, on='MVT_ID_mvt', how='left'); b = b.merge(lb, on='MVT_ID_mvt', how='left')
    y = a.fallback.values; unm = tr.fpl_matched.values == 0; v1 = pd.read_parquet(DATA / 'det_fallback_train.parquet').set_index('MVT_ID_mvt').p_fallback.reindex(a.MVT_ID_mvt).values
    auc_v1_u, auc_v2_u, auc_lane = _auc(y[unm], v1[unm]), _auc(y[unm], a.p_fallback_v2.values[unm]), _auc(y[unm], a.p_fallback_lane.values[unm])
    use_lane = auc_lane > auc_v2_u
    for df in (a, b):
        df['p_fallback_best'] = np.where(df.p_fallback_lane.notna() & use_lane, df.p_fallback_lane, df.p_fallback_v2)
    a.to_parquet(DATA / 'det_fallback_v2_train.parquet'); b.to_parquet(DATA / 'det_fallback_v2_rank.parquet')
    print(f'  fallback v2: OOF AUC all {_auc(y, a.p_fallback_v2.values):.3f} (v1 {_auc(y, v1):.3f}); unmatched lane AUC v1 {auc_v1_u:.3f} v2 {auc_v2_u:.3f} lane-specialist {auc_lane:.3f} -> best uses {"lane" if use_lane else "v2"} [{time.time()-t0:.0f}s]', flush=True)
    l = tr.APT.values == 'LIRF'; cal = pd.DataFrame({'y': y[l & unm], 'p': a.p_fallback_best.values[l & unm]}); cal['bin'] = pd.cut(cal.p, [0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)
    print('  LIRF unmatched calibration (best):'); print(cal.groupby('bin', observed=True).y.agg(['size', 'mean']).round(2).to_string())
    print(f'  permutation control (unmatched rows, label shuffled within airport): AUC {permutation_auc(tr[unm]):.3f}', flush=True)


# ---------------- detector 3: route periods (stand x runway p10 step changes) and 2026 drift ----------------
def detect_route_changes(dep, key='apt_stand', min_n=30, thr=90, agree=45):
    g = dep.groupby(['APT', key, 'apt_rwy', 'month']).TAXITIME_SEC_mvt.agg(n='size', p10=lambda s: s.quantile(0.10)).reset_index(); g = g[g.n >= min_n]; rows = []
    for (apt, k, rwy), grp in g.groupby(['APT', key, 'apt_rwy']):
        grp = grp.sort_values('month'); months = grp.month.tolist(); p = grp.p10.tolist(); n = grp.n.tolist()
        start = 0; pid = 0; i = 1
        while i < len(months):
            mean_cur = float(np.mean(p[start:i])); follow = p[i + 1:i + 3]; follow_months = months[i + 1:i + 3]
            persistent = len(follow) == 2 and follow_months == [months[i] + 1, months[i] + 2]
            if abs(p[i] - mean_cur) >= thr and persistent and all(abs(x - p[i]) <= agree for x in follow):
                rows.append(dict(APT=apt, key=k, apt_rwy=rwy, period_id=pid, start_month=months[start], end_month=months[i - 1], p10_period=mean_cur, n=int(sum(n[start:i])))); start = i; pid += 1
            i += 1
        rows.append(dict(APT=apt, key=k, apt_rwy=rwy, period_id=pid, start_month=months[start], end_month=months[-1], p10_period=float(np.mean(p[start:])), n=int(sum(n[start:]))))
    return pd.DataFrame(rows, columns=['APT', 'key', 'apt_rwy', 'period_id', 'start_month', 'end_month', 'p10_period', 'n'])


def _period_lookup(rows, per, key, is_rank):
    if per.empty: return pd.DataFrame({'pair_p10_period': np.nan, 'route_period_id': np.nan, 'pair_n_periods': np.nan}, index=rows.index)
    per = per.rename(columns={'key': key}); npr = per.groupby(['APT', key, 'apt_rwy']).period_id.max().rename('pair_n_periods') + 1
    if is_rank:
        last = per.sort_values('period_id').groupby(['APT', key, 'apt_rwy']).tail(1)
        j = rows[['APT', key, 'apt_rwy']].merge(last[['APT', key, 'apt_rwy', 'period_id', 'p10_period']], on=['APT', key, 'apt_rwy'], how='left')
        j = j.merge(npr.reset_index(), on=['APT', key, 'apt_rwy'], how='left')
    else:
        j = rows[['APT', key, 'apt_rwy', 'month']].reset_index().merge(per, on=['APT', key, 'apt_rwy'], how='left')
        j = j[(j.month >= j.start_month) & (j.month <= j.end_month) | j.period_id.isna()].drop_duplicates('index').set_index('index').reindex(rows.index)
        j = j.join(npr, on=['APT', key, 'apt_rwy'])
    out = pd.DataFrame({'pair_p10_period': j.p10_period.values, 'route_period_id': j.period_id.values, 'pair_n_periods': j.pair_n_periods.values}, index=rows.index); return out


def route_features(rows, periods_stand, periods_pier, is_rank):
    a = _period_lookup(rows, periods_stand, 'apt_stand', is_rank); b = _period_lookup(rows, periods_pier, 'apt_pier', is_rank)
    m = np.repeat(a.pair_p10_period.notna().values[:, None], a.shape[1], axis=1); return a.where(m, b)


def route_features_oof(train, rank, fold_ids, min_n=30):
    """Seasonal-pair-safe route features for train and full-history features for rank."""
    fold = np.asarray(fold_ids)
    if len(fold) != len(train): raise ValueError('fold_ids length must match train')
    cols = ['pair_p10_period', 'route_period_id', 'pair_n_periods']
    out = pd.DataFrame(np.nan, index=train.index, columns=cols)
    for k in np.unique(fold):
        src = train[fold != k]; dst = train[fold == k]
        ps = detect_route_changes(src, 'apt_stand', min_n=min_n)
        pp = detect_route_changes(src, 'apt_pier', min_n=min_n)
        # Validation represents 2026 deployment: use the last period learned without the
        # held seasonal pair, exactly as ranking rows use the last full-2025 period.
        out.loc[dst.index, cols] = route_features(dst, ps, pp, is_rank=True).values
    ps = detect_route_changes(train, 'apt_stand', min_n=min_n)
    pp = detect_route_changes(train, 'apt_pier', min_n=min_n)
    return out, route_features(rank, ps, pp, is_rank=True)


def detect_drift_2026(arr_train, arr_rank, min_n=40):
    def p10(df): return df.groupby(['APT', 'RUNWAY_mvt', 'apt_pier', 'month']).TAXITIME_SEC_mvt.agg(n='size', p10=lambda s: s.quantile(0.10)).reset_index()
    a = p10(arr_train[arr_train.month.isin(sorted(arr_rank.month.unique()))]); b = p10(arr_rank); j = a.merge(b, on=['APT', 'RUNWAY_mvt', 'apt_pier', 'month'], suffixes=('_25', '_26'))  # final-file change: 2025 months = ranking months (was [1, 7])
    j = j[(j.n_25 >= min_n) & (j.n_26 >= min_n)]; j['drift_ratio_2026'] = (j.p10_26 / j.p10_25).clip(0.5, 2.0)
    return j[['APT', 'RUNWAY_mvt', 'apt_pier', 'month', 'drift_ratio_2026']]


# ---------------- detector 4: runway configuration windows ----------------
def detect_runway_config(mov, window=1800):
    m = mov[['APT', 'PHASE_mvt', 'RUNWAY_mvt', 't']].copy(); m['w_start'] = (m.t // window * window).astype(float); m['RUNWAY_mvt'] = m.RUNWAY_mvt.astype(str)
    c = m.groupby(['APT', 'w_start', 'PHASE_mvt', 'RUNWAY_mvt']).size().rename('n').reset_index(); c2 = c[c.n >= 2]
    def sets(ph): return c2[c2.PHASE_mvt == ph].groupby(['APT', 'w_start']).RUNWAY_mvt.apply(lambda s: ','.join(sorted(s)))
    d, a = sets('DEP').rename('dep_rwys'), sets('ARR').rename('arr_rwys')
    tot = m.groupby(['APT', 'w_start', 'PHASE_mvt']).size().unstack(fill_value=0).rename(columns={'DEP': 'n_dep', 'ARR': 'n_arr'})
    w = tot.join(d).join(a).reset_index(); w['dep_rwys'] = w.dep_rwys.fillna(''); w['arr_rwys'] = w.arr_rwys.fillna('')
    for col in ('n_dep', 'n_arr'):
        if col not in w: w[col] = 0
    w['config_id'] = 'D:' + w.dep_rwys + '|A:' + w.arr_rwys; w['n_dep_rwys'] = w.dep_rwys.map(lambda s: len(s.split(',')) if s else 0); w['n_arr_rwys'] = w.arr_rwys.map(lambda s: len(s.split(',')) if s else 0)
    w = w.sort_values(['APT', 'w_start']).reset_index(drop=True); since = np.zeros(len(w))
    for apt, idx in w.groupby('APT').indices.items():
        ids = w.config_id.values[idx]; ws = w.w_start.values[idx]; run_start = ws[0]
        for j in range(len(idx)):
            if j > 0 and ids[j] != ids[j - 1]: run_start = ws[j]
            since[idx[j]] = (ws[j] - run_start) / 60.0
    w['min_since_change'] = since
    wr = c[c.PHASE_mvt == 'DEP'][['APT', 'w_start', 'RUNWAY_mvt', 'n']].rename(columns={'n': 'n_dep_rwy'})
    return w[['APT', 'w_start', 'config_id', 'dep_rwys', 'arr_rwys', 'n_dep', 'n_arr', 'n_dep_rwys', 'n_arr_rwys', 'min_since_change']], wr


def config_features(rows, windows, window_rwy, window=1800):
    k = pd.DataFrame({'APT': rows.APT.values, 'w_start': (rows.t.values // window * window).astype(float), 'RUNWAY_mvt': rows.RUNWAY_mvt.astype(str).values})
    j = k.merge(windows, on=['APT', 'w_start'], how='left').merge(window_rwy, on=['APT', 'w_start', 'RUNWAY_mvt'], how='left')
    return pd.DataFrame({'rwy_config_id': j.config_id.fillna('D:|A:').values, 'min_since_config_change': j.min_since_change.values, 'config_dep_share': (j.n_dep_rwy.fillna(0) / j.n_dep.replace(0, np.nan)).values,
                         'n_dep_rwys_config': j.n_dep_rwys.values, 'n_arr_rwys_config': j.n_arr_rwys.values}, index=rows.index)


# ---------------- detector 5: disruption days from arrivals' taxi-in ----------------
def detect_disruption_days(arr_daily, wx_daily=None, forward_from=None):
    out = []
    for apt, g in arr_daily.sort_values('date').groupby('APT'):
        g = g.reset_index(drop=True); dates = pd.to_datetime(g.date); wk = (dates.dt.dayofweek >= 5).values; med = g.taxi_in_median.values; z = np.full(len(g), np.nan); base = np.full(len(g), np.nan)
        for i in range(len(g)):
            m = (dates.values < dates.values[i]) & (dates.values >= dates.values[i] - np.timedelta64(28, 'D')) & (wk == wk[i])
            if m.sum() >= 4:
                b = np.median(med[m]); mad = np.median(np.abs(med[m] - b)); base[i] = b; z[i] = (med[i] - b) / (1.4826 * mad + 1e-9)
        out.append(pd.DataFrame(dict(APT=apt, date=dates.values, taxi_in_median=med, baseline=base, disruption_z=z)))
    d = pd.concat(out, ignore_index=True)
    d['baseline_forward'] = 0
    if forward_from is not None:   # continuity rule: a day on/after forward_from whose
        # backward baseline is missing (fewer than 4 same-type days in the previous 28, e.g. June 2026 without May) takes the
        # same rule looking FORWARD: same day type (weekday/weekend) in the next 28 days, >= 4 days. Earlier days untouched.
        lim = np.datetime64(pd.Timestamp(forward_from))
        z = d.disruption_z.to_numpy(copy=True); base = d.baseline.to_numpy(copy=True); fw = np.zeros(len(d), dtype=int)
        for apt, idx in d.groupby('APT').indices.items():
            dates = d.date.values[idx]; med = d.taxi_in_median.values[idx]; wk = pd.DatetimeIndex(dates).dayofweek.values >= 5
            for k, i in enumerate(idx):
                if dates[k] < lim or np.isfinite(z[i]): continue
                m = (dates > dates[k]) & (dates <= dates[k] + np.timedelta64(28, 'D')) & (wk == wk[k])
                if m.sum() >= 4:
                    b = np.median(med[m]); mad = np.median(np.abs(med[m] - b))
                    base[i] = b; z[i] = (med[k] - b) / (1.4826 * mad + 1e-9); fw[i] = 1
        d['baseline'] = base; d['disruption_z'] = z; d['baseline_forward'] = fw
    d['is_disruption'] = (d.disruption_z >= 3).astype(int)
    if wx_daily is not None:
        d = d.merge(wx_daily, on=['APT', 'date'], how='left'); expl = (d.snow_day.fillna(0) == 1) | (d.lvp_share.fillna(0) > 0.10) | (d.gust_max.fillna(0) >= 35)
        d['weather_explained'] = (d.is_disruption.astype(bool) & expl).astype(int); d = d.drop(columns=['snow_day', 'lvp_share', 'gust_max'])
    else: d['weather_explained'] = 0
    return d


# ---------------- detector 6: unseen ranking stands ----------------
def detect_unseen_stands(train_counts, rank_counts, coords):
    seen = set(zip(train_counts.APT, train_counts.STAND_mvt.astype(str))); loc = coords[coords.matched == 1].set_index(['APT', 'stand']); rows = []
    def pier(s): m = re.match(r'^([A-Za-z]+)', s); return m.group(1) if m else s[:1]
    for _, r in rank_counts.iterrows():
        s = str(r.STAND_mvt)
        if (r.APT, s) in seen: continue
        cand = train_counts[train_counts.APT == r.APT]; mapped, method, dist = 'NONE', 'none', np.nan
        if (r.APT, s) in loc.index:
            c = cand[[(r.APT, x) in loc.index for x in cand.STAND_mvt.astype(str)]]
            if len(c):
                la, lo = loc.loc[(r.APT, s), ['lat', 'lon']]; cl = loc.loc[[(r.APT, x) for x in c.STAND_mvt.astype(str)]]
                d = common.haversine(cl.lat.values, cl.lon.values, la, lo); k = int(np.argmin(d)); mapped, method, dist = str(c.STAND_mvt.astype(str).values[k]), 'coords', float(d[k])
        if method == 'none':
            same = cand[cand.STAND_mvt.astype(str).map(pier) == pier(s)]
            if len(same): mapped, method = str(same.sort_values('n', ascending=False).STAND_mvt.iloc[0]), 'pier'
        rows.append(dict(APT=r.APT, stand=s, n_rank_rows=int(r.n), mapped_stand=mapped, method=method, dist_m=dist))
    return pd.DataFrame(rows, columns=['APT', 'stand', 'n_rank_rows', 'mapped_stand', 'method', 'dist_m'])


def leadin_movements():
    """continuity rule: Dec 2025 movements as APT / RUNWAY_mvt / t / PHASE_mvt rows for the rank runway windows."""
    c = common.rank_leadin(['PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'RUNWAY_mvt', 'MVT_TIME_UTC_mvt'])
    return pd.DataFrame({'APT': np.where(c.PHASE_mvt.eq('DEP'), c.ADEP_mvt, c.ADES_mvt), 'RUNWAY_mvt': c.RUNWAY_mvt.values,
                         't': common.sec(c.MVT_TIME_UTC_mvt).values, 'PHASE_mvt': c.PHASE_mvt.values})


def run_detectors_3_to_6(force=False):
    common.ensure_dirs(); t0 = time.time()
    if not force and (DATA / 'det_unseen_stands.csv').exists(): print('detectors 3-6 exist; --force to rebuild'); return
    tr = pd.read_parquet(DATA / 'base_train.parquet', columns=['MVT_ID_mvt', 'APT', 'apt_stand', 'apt_pier', 'apt_rwy', 'RUNWAY_mvt', 'STAND_mvt', 'month', 't', 'TAXITIME_SEC_mvt'])
    rk = pd.read_parquet(DATA / 'base_rank.parquet', columns=['MVT_ID_mvt', 'APT', 'apt_stand', 'apt_pier', 'apt_rwy', 'RUNWAY_mvt', 'STAND_mvt', 'month', 't'])
    at = pd.read_parquet(DATA / 'arr_train.parquet'); ar = pd.read_parquet(DATA / 'arr_rank.parquet')
    ps = detect_route_changes(tr, 'apt_stand'); pp = detect_route_changes(tr, 'apt_pier'); ps.to_parquet(DATA / 'det_route_periods_stand.parquet'); pp.to_parquet(DATA / 'det_route_periods_pier.parquet')
    print(f'  route periods: {len(ps)} stand-pair periods ({int((ps.period_id > 0).sum())} changes), {len(pp)} pier-pair periods [{time.time()-t0:.0f}s]', flush=True)
    dr = detect_drift_2026(at, ar); dr.to_parquet(DATA / 'det_route_drift_2026.parquet'); print(f'  drift 2026: {len(dr)} runway x pier x month cells; ratio quantiles {dr.drift_ratio_2026.quantile([.1, .5, .9]).round(2).tolist()}', flush=True)
    for name, dep, arr in (('train', tr, at), ('rank', rk, ar)):
        mov = pd.concat([dep[['APT', 'RUNWAY_mvt', 't']].assign(PHASE_mvt='DEP'), arr.assign(t=common.sec(arr.MVT_TIME_UTC_mvt))[['APT', 'RUNWAY_mvt', 't']].assign(PHASE_mvt='ARR')])
        if name == 'rank': mov = pd.concat([mov, leadin_movements()])   # continuity rule: Dec 2025 lead-in
        w, wr = detect_runway_config(mov); w.to_parquet(DATA / f'det_runway_config_windows_{name}.parquet'); wr.to_parquet(DATA / f'det_runway_config_window_rwy_{name}.parquet')
        print(f'  runway config {name}: {len(w):,} windows, {w.config_id.nunique()} distinct configs, median minutes since change {w.min_since_change.median():.0f}', flush=True)
    arr_all = pd.concat([at, ar]); arr_all['date'] = arr_all.MVT_TIME_UTC_mvt.dt.floor('D').dt.tz_localize(None)
    daily = arr_all.groupby(['APT', 'date']).TAXITIME_SEC_mvt.agg(taxi_in_median='median', n='size').reset_index()
    wxd = None
    if (DATA / 'ext_weather.parquet').exists():
        wx = pd.read_parquet(DATA / 'ext_weather.parquet'); wx['date'] = wx.valid.dt.floor('D').dt.tz_localize(None)
        wxd = wx.groupby(['station', 'date']).agg(snow_day=('wx_snow', 'max'), lvp_share=('lvp', 'mean'), gust_max=('gust_kt', 'max')).reset_index().rename(columns={'station': 'APT'})
    dd = detect_disruption_days(daily, wxd, forward_from='2026-01-01'); dd.to_parquet(DATA / 'det_disruption_days.parquet')   # continuity rule: forward_from
    print(f'  disruption days: {int(dd.is_disruption.sum())} flagged of {len(dd)} airport-days, {int(dd.weather_explained.sum())} weather-explained; by airport {dd.groupby("APT").is_disruption.sum().to_dict()}', flush=True)
    coords = pd.read_parquet(DATA / 'ext_stand_coords.parquet') if (DATA / 'ext_stand_coords.parquet').exists() else pd.DataFrame(columns=['APT', 'stand', 'lat', 'lon', 'matched'])
    tc = tr.groupby(['APT', 'STAND_mvt']).size().rename('n').reset_index(); rc = rk.groupby(['APT', 'STAND_mvt']).size().rename('n').reset_index()
    um = detect_unseen_stands(tc, rc, coords); um.sort_values('n_rank_rows', ascending=False).to_csv(DATA / 'det_unseen_stands.csv', index=False)
    print(f'  unseen stands: {len(um)} stands / {int(um.n_rank_rows.sum())} ranking rows; methods {um.method.value_counts().to_dict()} (review: data/det_unseen_stands.csv)', flush=True)
