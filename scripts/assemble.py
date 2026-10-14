"""Assemble feature tables from base + detector + external tables; write manifest and shift report."""
from __future__ import annotations
import importlib.util, pathlib, time, json
import numpy as np, pandas as pd
import common
from common import DATA

G2 = ['atot_minus_sched', 'atot_minus_eobt1', 'atot_minus_lobt', 'atot_minus_aobt3', 'aobt3_minus_lobt', 'aobt3_minus_sched',
      'lobt_minus_sched', 'lobt_minus_iobt', 'plan_block_time', 'arr_delay', 'iobt_eq_lobt', 'sched_eq_eobt1', 'aobt3_eq_lobt']
G3 = ['dep_prev15', 'dep_next15', 'dep_prev60', 'arr_pm15', 'arr_prev60', 'dep_same_rwy_pm10', 'dep_same_rwy_prev30', 'rwy_share_pm30',
      'n_rwys_active_pm30', 'n_out_at_aobt3', 'n_in_at_aobt3', 'push_prev15_at_aobt3', 'dep_same_rwy_between_aobt3_atot',
      'arr_same_rwy_between_aobt3_atot', 'dep_any_rwy_between_aobt3_atot', 'dep_same_pier_at_aobt3', 'mixed_mode_rwy']
GROUPS_V1 = {
    'G0': ['APT', 'apt_rwy', 'apt_stand', 'apt_pier', 'actype', 'wake', 'segment', 'operator', 'flight_type', 'flight_rule', 'ades_region',
           'fpl_matched', 'is_heli', 'src_conflict_adep', 'src_conflict_ades', 'src_conflict_type'],
    'G1': ['hour_utc', 'minute_of_day_utc', 'dow', 'month', 'doy', 'is_weekend'],
    'G2': G2, 'G3': G3, 'G9': ['p_fallback']}
CAT_COLS = ['APT', 'apt_rwy', 'apt_stand', 'apt_pier', 'actype', 'wake', 'segment', 'operator', 'flight_type', 'flight_rule', 'ades_region']
ID_TRAIN = ['MVT_ID_mvt', 'TAXITIME_SEC_mvt', 'fallback']; ID_RANK = ['MVT_ID_mvt']

import ext_weather, ext_calendar, detectors, base_features
_spec = importlib.util.spec_from_file_location('encodings_', pathlib.Path(__file__).with_name('encodings.py')); enc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(enc)
G4 = ['taxi_dist_m', 'taxi_dist_straight_m', 'n_edges_path', 'n_rwy_crossings', 'stand_lat', 'stand_lon', 'layout_source', 'pair_p10_period', 'route_period_id', 'pair_n_periods', 'drift_ratio_2026']
G6 = ['atfm_arr_delay_min', 'atfm_arr_flights', 'atfm_arr_delayed_flights', 'atfm_delay_per_flight', 'traffic_dep_day', 'traffic_arr_day', 'addl_taxi_out_min_month', 'asma_addl_min_month', 'ec_monthly_is_prior']
G7 = ['mtow_kg', 'wingspan_m', 'length_m', 'n_engines', 'engine_type', 'icao_code_letter', 'is_code_f', 'aircraft_source']
G9_V2 = ['p_fallback', 'p_fallback_v2', 'p_fallback_lane', 'p_fallback_best', 'rwy_config_id', 'min_since_config_change', 'config_dep_share', 'n_dep_rwys_config', 'n_arr_rwys_config', 'disruption_z', 'is_disruption', 'disruption_weather_explained']
GROUPS_V2 = {'G0': GROUPS_V1['G0'] + ['airline', 'flight_is_callsign'],
             'G1': GROUPS_V1['G1'] + ext_calendar.LOCAL_COLS + [c for c in ext_calendar.CAL_COLS if c != 'event_type'],
             'G2': G2, 'G3': G3, 'G4': G4, 'G5': ext_weather.G5, 'G6': G6, 'G7': G7, 'G8': [s['name'] for s in enc.SPECS] + [n for n, _ in base_features.ARR_RATE_SPECS], 'G9': G9_V2}
CAT_COLS_V2 = CAT_COLS + ['airline', 'layout_source', 'engine_type', 'icao_code_letter', 'aircraft_source', 'rwy_config_id']


def _opt(name):
    p = DATA / name
    if not p.exists(): print(f'  missing table {name}: its group is skipped', flush=True); return None
    return pd.read_csv(p) if name.endswith('.csv') else pd.read_parquet(p)


def layout_features(rows, pairs):
    """Stand x runway distances with pier and airport fallbacks; rows need APT, stand_eff, RUNWAY_mvt."""
    key = rows[['APT', 'stand_eff', 'RUNWAY_mvt']].rename(columns={'stand_eff': 'stand', 'RUNWAY_mvt': 'rwy'}).astype(str); vals = ['taxi_dist_m', 'taxi_dist_straight_m', 'n_edges_path', 'n_rwy_crossings']
    p = pairs.assign(stand=pairs.stand.astype(str), rwy=pairs.rwy.astype(str)); p['pier'] = p.stand.str.extract(r'^([A-Za-z]+)')[0].fillna(p.stand.str[:1])
    ex = key.merge(p[['APT', 'stand', 'rwy'] + vals], on=['APT', 'stand', 'rwy'], how='left'); src = np.where(ex.taxi_dist_m.notna(), 'xplane', 'none')
    key['pier'] = key.stand.str.extract(r'^([A-Za-z]+)')[0].fillna(key.stand.str[:1])
    pm = p.groupby(['APT', 'pier', 'rwy'])[vals].median().reset_index(); am = p.groupby(['APT', 'rwy'])[vals].median().reset_index()
    jp = key.merge(pm, on=['APT', 'pier', 'rwy'], how='left'); ja = key.merge(am, on=['APT', 'rwy'], how='left')
    for v in vals:
        ex[v] = ex[v].where(ex[v].notna(), jp[v].values); src = np.where((src == 'none') & jp[v].notna().values, 'pier', src)
        ex[v] = ex[v].where(ex[v].notna(), ja[v].values); src = np.where((src == 'none') & ja[v].notna().values & (jp[v].isna().values), 'airport', src)
    out = ex[vals].copy(); out['layout_source'] = src; out.index = rows.index; return out


def shift_report(train, rank, cols):
    rows = []; tj = train[train.month.isin([1, 7])] if 'month' in train else train
    for c in cols:
        if not pd.api.types.is_numeric_dtype(train[c]): continue
        sd = tj[c].std(); shift_sd = float((rank[c].mean() - tj[c].mean()) / sd) if sd and sd > 0 else np.nan
        for apt in sorted(train.APT.unique()):
            a = train.loc[train.APT == apt, c]; b = rank.loc[rank.APT == apt, c]
            ma, mb = a.median(), b.median(); ratio = (mb / ma) if (pd.notna(ma) and ma != 0) else np.nan
            rows.append(dict(column=c, apt=apt, med_train=ma, med_rank=mb, ratio=ratio, p90_train=a.quantile(.9), p90_rank=b.quantile(.9), shift_sd=shift_sd,
                             flag=bool(pd.notna(ratio) and (ratio > 1.3 or ratio < 0.7))))
    return pd.DataFrame(rows)


NULL_TOKENS = ('NA', 'nan', 'None', '', '<NA>')


def _null_mask(s: pd.Series, is_cat: bool) -> np.ndarray:
    m = s.isna().to_numpy()
    if is_cat:
        m = m | s.astype(str).isin(NULL_TOKENS).to_numpy()
    return m


def _psi(a: pd.Series, b: pd.Series, eps=1e-4) -> float:
    """Population stability index of b against a over a's decile bins plus a missing bin."""
    av = a.to_numpy(dtype=float); bv = b.to_numpy(dtype=float)
    fa = av[np.isfinite(av)]
    if len(fa) == 0 or len(bv) == 0:
        return float('nan')
    edges = np.unique(np.quantile(fa, np.linspace(0, 1, 11)))
    def dist(v):
        fin = np.isfinite(v); idx = np.clip(np.searchsorted(edges, v[fin], 'right') - 1, 0, max(len(edges) - 1, 0))
        counts = np.bincount(idx, minlength=len(edges)).astype(float)
        return np.append(counts, (~fin).sum()) / len(v)
    pa, pb = dist(av) + eps, dist(bv) + eps
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def drift_report(train, rank, cols, cats=(), months=(1, 7)):
    """Missingness, categorical and distribution drift of the ranking months against the same 2025 months.

    One row per column x airport (plus 'ALL'). ``train_constant`` marks a column that never varies in
    all of 2025 but varies in the ranking file (a tree cannot learn its effect). Categorical columns get
    the ranking share of values never seen in 2025 and the total-variation distance between the two
    month-matched distributions; numeric columns get the PSI over 2025 decile bins.
    """
    tj = train[train.month.isin(months)] if 'month' in train else train
    groups = [('ALL', None)] + [(a, a) for a in sorted(rank.APT.unique())]
    rows = []
    for c in cols:
        is_cat = c in cats or not pd.api.types.is_numeric_dtype(train[c])
        for name, apt in groups:
            full = train[c] if apt is None else train.loc[train.APT.values == apt, c]
            a = tj[c] if apt is None else tj.loc[tj.APT.values == apt, c]
            b = rank[c] if apt is None else rank.loc[rank.APT.values == apt, c]
            if len(b) == 0:
                continue
            na, nb = _null_mask(a, is_cat), _null_mask(b, is_cat)
            nf = _null_mask(full, is_cat)
            vf, vb = full[~nf], b[~nb]
            row = dict(column=c, apt=name, kind='cat' if is_cat else 'num', n_train=len(a), n_rank=len(b),
                       null_train=float(na.mean()) if len(a) else np.nan, null_rank=float(nb.mean()),
                       nunique_train=int(vf.nunique()), nunique_rank=int(vb.nunique()),
                       train_constant=bool(vf.nunique() <= 1 and vb.nunique() > 1))
            row['null_delta'] = row['null_rank'] - row['null_train']
            if is_cat:
                sa, sb = a[~na].astype(str), vb.astype(str)
                pa, pb = sa.value_counts(normalize=True), sb.value_counts(normalize=True)
                idx = pa.index.union(pb.index)
                row['tvd'] = float(0.5 * (pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).abs().sum()) if len(sb) and len(sa) else np.nan
                row['unseen_share_rank'] = float((~sb.isin(set(vf.astype(str).unique()))).mean()) if len(sb) else 0.0
                if len(pb):
                    row['top_rank'] = str(pb.index[0]); row['top_share_rank'] = float(pb.iloc[0]); row['top_share_train'] = float(pa.get(pb.index[0], 0.0))
            else:
                row['psi'] = _psi(a, b)
                row['med_train'] = float(a.median()) if len(a) else np.nan; row['med_rank'] = float(b.median())
            rows.append(row)
    rep = pd.DataFrame(rows)
    for col in ('tvd', 'unseen_share_rank', 'psi'):
        if col not in rep:
            rep[col] = np.nan
    rep['flag'] = ((rep.null_delta.abs() > 0.05) | rep.train_constant | (rep.tvd > 0.2) | (rep.unseen_share_rank > 0.02) | (rep.psi > 0.25))
    return rep


def assemble(version=1, force=False):
    if version >= 2: return assemble_v2(version, force)
    common.ensure_dirs(); groups = GROUPS_V1
    if not force and common.manifest_path(version).exists(): print(f'features v{version} exist; --force to rebuild'); return common.load_features(version)
    tr = pd.read_parquet(DATA / 'base_train.parquet'); rk = pd.read_parquet(DATA / 'base_rank.parquet')
    tr = tr.merge(pd.read_parquet(DATA / 'det_fallback_train.parquet'), on='MVT_ID_mvt', how='left').merge(pd.read_parquet(DATA / 'det_conflicts_train.parquet'), on='MVT_ID_mvt', how='left')
    rk = rk.merge(pd.read_parquet(DATA / 'det_fallback_rank.parquet'), on='MVT_ID_mvt', how='left').merge(pd.read_parquet(DATA / 'det_conflicts_rank.parquet'), on='MVT_ID_mvt', how='left')
    cols = [c for g in groups.values() for c in g]
    missing = [c for c in cols if c not in rk.columns]; assert not missing, f'ranking table lacks {missing}'
    ftr = tr[ID_TRAIN + cols].copy(); frk = rk[ID_RANK + cols].copy()
    for c in CAT_COLS: ftr[c] = ftr[c].astype(str); frk[c] = frk[c].astype(str)
    ftr.to_parquet(DATA / f'features_v{version}_train.parquet'); frk.to_parquet(DATA / f'features_v{version}_rank.parquet')
    manifest_cols = {c: {'group': g, 'source': 'base' if g != 'G9' else 'detector', 'rank_available': True, 'dtype': str(ftr[c].dtype)} for g, cs in groups.items() for c in cs}
    manifest_cols.update({c: {'group': 'ID', 'source': 'raw', 'rank_available': c == 'MVT_ID_mvt', 'dtype': str(ftr[c].dtype)} for c in ID_TRAIN})
    common.write_manifest(version, manifest_cols, groups)
    rep = shift_report(ftr, frk, cols); rep.to_csv(DATA / f'features_v{version}_shift.csv', index=False)
    print(f'features v{version}: train {ftr.shape}, rank {frk.shape}; shift flags: {int(rep.flag.sum())} of {len(rep)} (see features_v{version}_shift.csv)', flush=True)
    if rep.flag.any(): print(rep[rep.flag].sort_values('ratio').to_string(index=False)[:3000])
    return ftr, frk, common.read_manifest(version)


def assemble_v2(version=2, force=False):
    common.ensure_dirs(); t0 = time.time()
    if not force and common.manifest_path(version).exists(): print(f'features v{version} exist; --force to rebuild'); return common.load_features(version)
    tr = pd.read_parquet(DATA / 'base_train.parquet'); rk = pd.read_parquet(DATA / 'base_rank.parquet'); missing = []
    def merge_ids(df, name):
        t = _opt(name)
        if t is None: missing.append(name); return df
        t = t[[c for c in t.columns if c == 'MVT_ID_mvt' or c not in df.columns]]   # e.g. `fallback` exists in both fallback tables
        return df.merge(t, on='MVT_ID_mvt', how='left')
    for name in ('det_fallback_train.parquet', 'det_fallback_v2_train.parquet', 'det_conflicts_train.parquet', 'enc_train.parquet'): tr = merge_ids(tr, name)
    for name in ('det_fallback_rank.parquet', 'det_fallback_v2_rank.parquet', 'det_conflicts_rank.parquet', 'enc_rank.parquet'): rk = merge_ids(rk, name)
    print(f'  merged id tables [{time.time()-t0:.0f}s]', flush=True)
    # unseen-stand map -> stand_eff (rank only)
    um = _opt('det_unseen_stands.csv'); tr['stand_eff'] = tr.STAND_mvt.astype(str); rk['stand_eff'] = rk.STAND_mvt.astype(str)
    if um is not None:
        mp = {(a, str(s)): str(m) for a, s, m, meth in zip(um.APT, um.stand, um.mapped_stand, um.method) if meth != 'none'}
        rk['stand_eff'] = [mp.get((a, s), s) for a, s in zip(rk.APT, rk.stand_eff)]
    # G4 layout + coords + route periods + drift
    pairs = _opt('ext_layout_pairs.parquet'); coords = _opt('ext_stand_coords.parquet'); ps = _opt('det_route_periods_stand.parquet'); pp = _opt('det_route_periods_pier.parquet'); dr = _opt('det_route_drift_2026.parquet')
    for df, is_rank in ((tr, False), (rk, True)):
        if pairs is not None: df[['taxi_dist_m', 'taxi_dist_straight_m', 'n_edges_path', 'n_rwy_crossings', 'layout_source']] = layout_features(df, pairs)
        if coords is not None:
            c = df[['APT', 'stand_eff']].rename(columns={'stand_eff': 'stand'}).merge(coords[['APT', 'stand', 'lat', 'lon']].assign(stand=coords.stand.astype(str)).drop_duplicates(['APT', 'stand']), on=['APT', 'stand'], how='left'); df['stand_lat'] = c.lat.values; df['stand_lon'] = c.lon.values
        if ps is not None and pp is not None: df[['pair_p10_period', 'route_period_id', 'pair_n_periods']] = detectors.route_features(df, ps, pp, is_rank)
        if dr is not None:
            if is_rank: d = df[['APT', 'RUNWAY_mvt', 'apt_pier', 'month']].merge(dr, on=['APT', 'RUNWAY_mvt', 'apt_pier', 'month'], how='left'); df['drift_ratio_2026'] = d.drift_ratio_2026.fillna(1.0).values
            else: df['drift_ratio_2026'] = 1.0
    print(f'  G4 done [{time.time()-t0:.0f}s]', flush=True)
    # G5 weather
    wx = _opt('ext_weather.parquet')
    if wx is not None:
        for df in (tr, rk):
            when = df.AOBT_3_flt.where(df.AOBT_3_flt.notna(), df.MVT_TIME_UTC_mvt - pd.Timedelta(minutes=15)); df[ext_weather.G5] = ext_weather.join_weather(df.APT.values, when, df.RUNWAY_mvt.values, wx)
    print(f'  G5 done [{time.time()-t0:.0f}s]', flush=True)
    # G1 local + calendar
    cal = _opt('ext_calendar.parquet')
    if cal is not None:
        lc = ext_calendar.LOCAL_COLS + [c for c in ext_calendar.CAL_COLS if c != 'event_type']
        for df in (tr, rk):
            f = ext_calendar.local_features(df.APT.values, df.MVT_TIME_UTC_mvt, cal); df[lc] = f[lc]
    print(f'  G1-local done [{time.time()-t0:.0f}s]', flush=True)
    # G6 EUROCONTROL
    ecd = _opt('ext_eurocontrol_daily.parquet'); ecm = _opt('ext_eurocontrol_monthly.parquet')
    for df in (tr, rk):
        df['date_dt'] = pd.to_datetime(df.date)
        if ecd is not None: j = df[['APT', 'date_dt']].merge(ecd.rename(columns={'date': 'date_dt'}), on=['APT', 'date_dt'], how='left'); df[G6[:6]] = j[G6[:6]].values
        if ecm is not None: j = df[['APT', 'year', 'month']].merge(ecm, on=['APT', 'year', 'month'], how='left'); df[G6[6:]] = j[G6[6:]].values
    # G7 aircraft
    ac = _opt('ext_aircraft.csv')
    if ac is not None:
        for df in (tr, rk):
            j = df[['AIRCRAFT_TYPE_mvt']].merge(ac.rename(columns={'icao': 'AIRCRAFT_TYPE_mvt'}), on='AIRCRAFT_TYPE_mvt', how='left'); df[G7] = j[G7].values
            df['engine_type'] = df.engine_type.fillna('unknown'); df['aircraft_source'] = df.aircraft_source.fillna('missing'); df['icao_code_letter'] = df.icao_code_letter.fillna('U'); df['is_code_f'] = df.is_code_f.fillna(0)
    # G9 runway config + disruption
    dd = _opt('det_disruption_days.parquet')
    for suffix, df in (('train', tr), ('rank', rk)):
        w = _opt(f'det_runway_config_windows_{suffix}.parquet'); wr = _opt(f'det_runway_config_window_rwy_{suffix}.parquet')
        if w is not None and wr is not None: df[['rwy_config_id', 'min_since_config_change', 'config_dep_share', 'n_dep_rwys_config', 'n_arr_rwys_config']] = detectors.config_features(df, w, wr)
        if dd is not None: j = df[['APT', 'date_dt']].merge(dd.rename(columns={'date': 'date_dt'})[['APT', 'date_dt', 'disruption_z', 'is_disruption', 'weather_explained']], on=['APT', 'date_dt'], how='left'); df['disruption_z'] = j.disruption_z.values; df['is_disruption'] = j.is_disruption.fillna(0).values; df['disruption_weather_explained'] = j.weather_explained.fillna(0).values
    print(f'  G6/G7/G9 done [{time.time()-t0:.0f}s]', flush=True)
    # select, cast, write
    groups = {g: [c for c in cs if c in rk.columns and c in tr.columns] for g, cs in GROUPS_V2.items()}; dropped = {g: [c for c in cs if c not in groups[g]] for g, cs in GROUPS_V2.items()}
    cols = [c for g in groups.values() for c in g]; cats = [c for c in CAT_COLS_V2 if c in cols]
    ftr = tr[ID_TRAIN + cols].copy(); frk = rk[ID_RANK + cols].copy()
    for c in cats: ftr[c] = ftr[c].astype(str).fillna('NA'); frk[c] = frk[c].astype(str).fillna('NA')
    for c in cols:
        if c not in cats and ftr[c].dtype == object: ftr[c] = pd.to_numeric(ftr[c], errors='coerce'); frk[c] = pd.to_numeric(frk[c], errors='coerce')
    ftr.to_parquet(DATA / f'features_v{version}_train.parquet'); frk.to_parquet(DATA / f'features_v{version}_rank.parquet')
    manifest_cols = {c: {'group': g, 'source': 'v2', 'rank_available': True, 'dtype': str(ftr[c].dtype)} for g, cs in groups.items() for c in cs}
    manifest_cols.update({c: {'group': 'ID', 'source': 'raw', 'rank_available': c == 'MVT_ID_mvt', 'dtype': str(ftr[c].dtype)} for c in ID_TRAIN})
    p = common.write_manifest(version, manifest_cols, groups); m = json.loads(p.read_text(encoding='utf-8')); m['cat_cols'] = cats; m['missing_tables'] = missing; m['dropped_columns'] = dropped; p.write_text(json.dumps(m, indent=1), encoding='utf-8')
    rep = shift_report(ftr, frk, cols); rep.to_csv(DATA / f'features_v{version}_shift.csv', index=False)
    drift_report(ftr, frk, cols, cats).to_csv(DATA / f'features_v{version}_drift.csv', index=False)
    big = rep[rep.shift_sd.abs() > 0.25].column.unique().tolist()
    print(f'features v{version}: train {ftr.shape}, rank {frk.shape}; groups {{ {", ".join(f"{g}:{len(c)}" for g, c in groups.items())} }}; missing tables {missing}; |shift| > 0.25 sd: {big}; median-ratio flags {int(rep.flag.sum())} [{time.time()-t0:.0f}s]', flush=True)
    return ftr, frk, common.read_manifest(version)
