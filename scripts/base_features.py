"""Base feature builder: keys (G0), UTC time (G1), take-off deltas (G2), congestion (G3)."""
from __future__ import annotations
import glob, time
import numpy as np, pandas as pd
import common
from common import sec, DATA, HELI_TYPES


def load_raw():
    files = sorted(glob.glob(str(DATA / 'training_*.parquet')))
    tr = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rk = pd.read_parquet(DATA / 'ranking.parquet')
    return tr, rk


def _s(x: pd.Series) -> pd.Series:
    """String view with missing values as 'NA' (pandas 3 str dtype keeps NaN through astype(str))."""
    return x.astype(str).fillna('NA')


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['APT'] = np.where(df.PHASE_mvt.eq('DEP'), df.ADEP_mvt, df.ADES_mvt)
    stand = _s(df.STAND_mvt)
    df['apt_rwy'] = df.APT + '|' + _s(df.RUNWAY_mvt)
    df['apt_stand'] = df.APT + '|' + stand
    pier = stand.str.extract(r'^([A-Za-z]+)')[0]
    df['apt_pier'] = df.APT + '|' + pier.fillna(stand.str[:1])
    df['actype'] = df.AIRCRAFT_TYPE_mvt; df['wake'] = df.WK_TBL_CAT_flt; df['segment'] = df.MARKET_SEGMENT_flt
    df['operator'] = df.AIRCRAFT_OPERATOR_flt; df['flight_type'] = df.FLIGHT_TYPE_flt; df['flight_rule'] = df.FLIGHT_RULE_mvt
    df['ades_region'] = df.ADES_mvt.astype(str).str[0]
    df['fpl_matched'] = df.AOBT_3_flt.notna().astype(int)
    df['is_heli'] = ((df.RUNWAY_mvt == 'H') | df.AIRCRAFT_TYPE_mvt.isin(HELI_TYPES)).astype(int)
    fl = _s(df.FLIGHT_mvt)
    df['airline'] = fl.str.extract(r'^([A-Z]{2,3})')[0].fillna('NA')
    df['flight_is_callsign'] = fl.str.contains(r'\d[A-Z]+$', regex=True).astype(int)
    return df


def add_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); t = df.MVT_TIME_UTC_mvt
    df['hour_utc'] = t.dt.hour; df['minute_of_day_utc'] = t.dt.hour * 60 + t.dt.minute
    df['dow'] = t.dt.dayofweek; df['month'] = t.dt.month; df['doy'] = t.dt.dayofyear; df['year'] = t.dt.year
    df['is_weekend'] = (df.dow >= 5).astype(int); df['date'] = t.dt.date
    return df


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['t'] = sec(df.MVT_TIME_UTC_mvt); df['a3'] = sec(df.AOBT_3_flt); df['b'] = sec(df.BLOCK_TIME_UTC_mvt)
    df['sched_s'] = sec(df.SCHED_TIME_UTC_mvt); lobt = sec(df.LOBT_flt); iobt = sec(df.IOBT_flt); eobt = sec(df.EOBT_1_flt)
    df['atot_minus_sched'] = df.t - df.sched_s; df['atot_minus_eobt1'] = df.t - eobt; df['atot_minus_lobt'] = df.t - lobt
    df['atot_minus_aobt3'] = df.t - df.a3; df['aobt3_minus_lobt'] = df.a3 - lobt; df['aobt3_minus_sched'] = df.a3 - df.sched_s
    df['lobt_minus_sched'] = lobt - df.sched_s; df['lobt_minus_iobt'] = lobt - iobt
    df['plan_block_time'] = sec(df.ARVT_1_flt) - eobt; df['arr_delay'] = sec(df.ARVT_3_flt) - sec(df.ARVT_1_flt)
    df['iobt_eq_lobt'] = (df.IOBT_flt == df.LOBT_flt).astype(int); df['sched_eq_eobt1'] = (df.SCHED_TIME_UTC_mvt == df.EOBT_1_flt).astype(int)
    df['aobt3_eq_lobt'] = (df.AOBT_3_flt == df.LOBT_flt).astype(int)
    return df


def _count(sorted_arr, lo, hi):
    """Number of values v with lo < v <= hi (arrays)."""
    return np.searchsorted(sorted_arr, hi, 'right') - np.searchsorted(sorted_arr, lo, 'right')


def congestion(df: pd.DataFrame) -> pd.DataFrame:
    """Congestion counts for DEP rows from all movements of the same dataset. Requires add_keys/add_deltas."""
    out = []
    for apt, g in df.groupby('APT'):
        dep = g[g.PHASE_mvt == 'DEP']; arr = g[g.PHASE_mvt == 'ARR']
        t = dep.t.values; a = dep.a3.values; ok = ~np.isnan(a)
        ts = np.sort(t); a_s = np.sort(a[ok]); t_of_a = np.sort(t[ok])
        l_s = np.sort(arr.t.values); b_arr = arr.b.values; b_s = np.sort(b_arr[~np.isnan(b_arr)])
        r = pd.DataFrame(index=dep.index)
        r['dep_prev15'] = _count(ts, t - 900, t) - 1; r['dep_next15'] = _count(ts, t, t + 900)
        r['dep_prev60'] = _count(ts, t - 3600, t) - 1
        r['arr_pm15'] = _count(l_s, t - 900, t + 900); r['arr_prev60'] = _count(l_s, t - 3600, t)
        n = np.full(len(dep), np.nan); n[ok] = np.searchsorted(a_s, a[ok], 'left') - np.searchsorted(t_of_a, a[ok], 'right'); r['n_out_at_aobt3'] = n
        n = np.full(len(dep), np.nan); n[ok] = np.searchsorted(l_s, a[ok], 'left') - np.searchsorted(b_s, a[ok], 'right'); r['n_in_at_aobt3'] = n
        n = np.full(len(dep), np.nan); n[ok] = _count(a_s, a[ok] - 900, a[ok]) - 1; r['push_prev15_at_aobt3'] = n
        rw = _s(dep.RUNWAY_mvt).values; arw = _s(arr.RUNWAY_mvt).values
        same10 = np.zeros(len(dep)); same30 = np.zeros(len(dep)); same_w = np.zeros(len(dep)); arr_w = np.zeros(len(dep)); mixed = np.zeros(len(dep)); share_num = np.zeros(len(dep))
        for rwy in np.unique(rw):
            mk = rw == rwy; tt = np.sort(t[mk]); la = np.sort(arr.t.values[arw == rwy])
            same10[mk] = _count(tt, t[mk] - 600, t[mk] + 600) - 1; same30[mk] = _count(tt, t[mk] - 1800, t[mk]) - 1
            share_num[mk] = _count(tt, t[mk] - 1800, t[mk] + 1800)
            lo = np.where(ok[mk], a[mk], t[mk]); valid = ok[mk] & (a[mk] < t[mk])
            same_w[mk] = np.where(valid, np.maximum(_count(tt, lo, t[mk]) - 1, 0), 0)
            arr_w[mk] = np.where(valid, np.maximum(_count(la, lo, t[mk]), 0), 0)
            mixed[mk] = (_count(la, t[mk] - 1800, t[mk] + 1800) > 0).astype(int)
        tot = _count(ts, t - 1800, t + 1800)
        r['dep_same_rwy_pm10'] = same10; r['dep_same_rwy_prev30'] = same30; r['rwy_share_pm30'] = share_num / np.maximum(tot, 1)
        pres = np.zeros(len(dep))
        for rwy in np.unique(rw):
            tt = np.sort(t[rw == rwy]); pres += (_count(tt, t - 1800, t + 1800) > 0)
        r['n_rwys_active_pm30'] = pres
        r['dep_same_rwy_between_aobt3_atot'] = same_w; r['arr_same_rwy_between_aobt3_atot'] = arr_w
        lo = np.where(ok, a, t); valid = ok & (a < t)
        r['dep_any_rwy_between_aobt3_atot'] = np.where(valid, np.maximum(_count(ts, lo, t) - 1, 0), 0)
        pier = dep.apt_pier.values; sp = np.full(len(dep), np.nan)
        for p in np.unique(pier):
            mk = (pier == p) & ok; aa = np.sort(a[mk]); sp[mk] = _count(aa, a[mk] - 300, a[mk] + 300) - 1
        r['dep_same_pier_at_aobt3'] = sp; r['mixed_mode_rwy'] = mixed
        out.append(r)
    return pd.concat(out).reindex(df.index[df.PHASE_mvt == 'DEP'])


PRIOR_N = 20
ARR_RATE_SPECS = [('arr_fb_rate_airline_apt', ['APT', 'airline']), ('arr_fb_rate_airline_apt_month', ['APT', 'airline', 'month'])]


def arrival_defaulting_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Share of ARRIVALS whose in-block time equals the schedule (the defaulting the departure label suffers), per airport x airline (x month),
    from the same file as the rows (2025 arrivals for training rows, 2026 arrivals for ranking rows), smoothed toward the airport rate.
    Uses arrival labels only, never a departure's own label. Returned for the DEP rows of df; NaN where the airport has no arrivals."""
    arr = df[df.PHASE_mvt == 'ARR']; dep = df[df.PHASE_mvt == 'DEP']; fb = ((arr.b - arr.sched_s).abs() <= 10).astype(float)
    prior = fb.groupby(arr.APT.values).mean(); out = pd.DataFrame(index=dep.index)
    for name, keys in ARR_RATE_SPECS:
        kk = pd.Series(list(zip(*[arr[c].astype(str).values for c in keys])), index=arr.index)
        g = fb.groupby(kk).agg(['sum', 'size'])
        pr = pd.Series([k[0] for k in g.index]).map(prior).values
        tab = pd.Series((g['sum'].values + PRIOR_N * pr) / (g['size'].values + PRIOR_N), index=g.index)
        k = pd.Series(list(zip(*[dep[c].astype(str).values for c in keys])), index=dep.index)
        out[name] = k.map(tab).astype(float).fillna(dep.APT.map(prior)).values
    return out


ARR_COLS = ['MVT_ID_mvt', 'APT', 'RUNWAY_mvt', 'STAND_mvt', 'apt_pier', 'airline', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'TAXITIME_SEC_mvt', 'month', 'year']


def build_base(force: bool = False, rank_only: bool = False) -> None:
    common.ensure_dirs()
    if not force and (DATA / 'base_train.parquet').exists() and (DATA / 'base_rank.parquet').exists():
        print('base tables exist; use --force to rebuild'); return
    t0 = time.time(); tr, rk = load_raw()
    for name, df in (('train', tr), ('rank', rk)):
        if rank_only and name == 'train': continue   # continuity rule: leave the 2025 side untouched
        df = add_deltas(add_time(add_keys(df)))
        if name == 'rank':   # continuity rule: congestion windows see the Dec 2025 lead-in; pooled rates stay 2026-only
            ctx = add_deltas(add_time(add_keys(common.rank_leadin())))
            both = pd.concat([df, ctx.set_axis(ctx.index + len(df))])
            cg = congestion(both).reindex(df.index[df.PHASE_mvt == 'DEP'])
        else:
            cg = congestion(df)
        dep = df[df.PHASE_mvt == 'DEP'].join(cg).join(arrival_defaulting_rates(df))
        dep.to_parquet(DATA / f'base_{name}.parquet'); df.loc[df.PHASE_mvt == 'ARR', ARR_COLS].to_parquet(DATA / f'arr_{name}.parquet')
        print(f'  base_{name}: {len(dep):,} departures, {cg.shape[1]} congestion cols [{time.time()-t0:.0f}s]', flush=True)
