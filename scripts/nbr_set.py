"""Neighbour-set tokens: the airport's movement timeline around each departure, for a set-encoder residual member.

For departure i (take-off t_i) the tokens are

  self        the departure itself (always valid, so attention never sees an empty set)
  departures  the DEP_BEFORE previous and DEP_AFTER next departures at the airport in take-off order,
              kept when their take-off lies in [t_i - 60 min, t_i + 20 min]
  arrivals    the ARR_BEFORE previous and ARR_AFTER next arrivals in landing order, same window

Token features (times relative to t_i in units of 15 min, clipped to +-6 h):

  valid, is_self, is_dep, is_arr
  move        take-off / landing - t_i
  anchor      dep: NM AOBT_3 - t_i; arr: in-block - t_i          (anchor_missing flag)
  dur         dep: take-off - AOBT_3 (NM's taxi); arr: in-block - landing (taxi-in)
  sched       schedule - t_i
  plan        dep: EOBT_1 - t_i (plan_missing flag); arr: 0
  same_rwy, same_stand, same_pier, same_airline   (against departure i)
  stand_km    stand-to-stand distance to i's stand (km, clipped at 4; stand_missing flag)
  wake        one-hot L / M / H / J
  arr_copy    arrival in-block within 10 s of its schedule

Every field is rank-available. A departure's block time (the label) is never read: departures come from
``base_*`` without the ``BLOCK_TIME``/``b``/``TAXITIME`` columns, arrivals from ``arr_*`` (complete in both years).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import common
from common import DATA

DEP_BEFORE, DEP_AFTER = 24, 8
ARR_BEFORE, ARR_AFTER = 18, 6
WINDOW = (-3600.0, 1200.0)
SCALE = 900.0
CLIP = 21600.0 / SCALE
FEATURES = ['valid', 'is_self', 'is_dep', 'is_arr', 'move', 'anchor', 'anchor_missing', 'dur', 'sched', 'plan',
            'plan_missing', 'same_rwy', 'same_stand', 'same_pier', 'same_airline', 'stand_km', 'stand_missing',
            'wake_L', 'wake_M', 'wake_H', 'wake_J', 'arr_copy']
F = {n: i for i, n in enumerate(FEATURES)}
# optional token features (build(..., extra=True)), appended after FEATURES so the v1 indices do not move:
#   dx_km, dy_km   stand offset east / north of departure i's stand (km, clipped at +-4)
#   taxi_km        the movement's own stand-to-runway taxi distance (ext_layout_pairs; taxi_missing flag)
#   rwy_cross      runway crossings on that path
#   rwy0..rwy5     the movement's runway among the airport's six busiest runways (one-hot; others all zero)
EXTRA = ['dx_km', 'dy_km', 'taxi_km', 'taxi_missing', 'rwy_cross', 'rwy0', 'rwy1', 'rwy2', 'rwy3', 'rwy4', 'rwy5']
FX = {n: len(FEATURES) + i for i, n in enumerate(EXTRA)}
# optional surge features (build(..., surge=True)), appended last (after EXTRA when both are on):
#   norm          the movement's normal taxi: median over 2025 of its (airport, stand, runway) NM taxi for departures or
#                 taxi-in for arrivals, falling back to (pier, runway), (runway), airport (>= 20 values per cell)
#   dur_res       its NM taxi / taxi-in minus that normal (clipped +-30 min), 0 when the taxi is missing or implausible
#   norm_missing  no normal at any level
SURGE = ['norm', 'dur_res', 'norm_missing']
SURGE_MIN_N = 20
N_TOKENS = 1 + DEP_BEFORE + DEP_AFTER + ARR_BEFORE + ARR_AFTER


def surge_index(name, n_features):
    """Index of a SURGE feature in tokens with ``n_features`` features (the surge block is always last)."""
    return n_features - len(SURGE) + SURGE.index(name)

_DEP_COLS = ['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt', 'SCHED_TIME_UTC_mvt', 'EOBT_1_flt', 'RUNWAY_mvt',
             'STAND_mvt', 'apt_pier', 'airline', 'wake']
_ARR_COLS = ['MVT_ID_mvt', 'APT', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'apt_pier', 'airline']


def _sec(s):
    """Timestamps to float epoch seconds (NaN for NaT)."""
    t = pd.to_datetime(s, utc=True)
    out = (t - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().to_numpy(dtype='float64')
    return out


def _load(which):
    """Departures (never with their block time) and arrivals of the 2025 training year or the 2026 ranking file."""
    dep = pd.read_parquet(DATA / f'base_{which}.parquet', columns=_DEP_COLS)
    arr = pd.read_parquet(DATA / f'arr_{which}.parquet', columns=_ARR_COLS)
    # arrival schedules and wake categories live in the raw movement files
    raw = ([DATA / 'ranking.parquet'] if which == 'rank' else sorted(DATA.glob('training_2025-*.parquet')))
    rs = pd.concat([pd.read_parquet(f, columns=['MVT_ID_mvt', 'PHASE_mvt', 'SCHED_TIME_UTC_mvt', 'WK_TBL_CAT_flt']) for f in raw],
                   ignore_index=True)
    rs = rs[rs.PHASE_mvt == 'ARR'].drop(columns='PHASE_mvt')
    arr = arr.merge(rs, on='MVT_ID_mvt', how='left', validate='one_to_one')
    return dep, arr


def _coords():
    c = pd.read_parquet(DATA / 'ext_stand_coords.parquet')
    c = c[c.matched == 1]
    return {(a, s): (la, lo) for a, s, la, lo in zip(c.APT, c.stand.astype(str), c.lat, c.lon)}


def _latlon(apt, stands, coords):
    ll = np.array([coords.get((apt, s), (np.nan, np.nan)) for s in stands], dtype='float64').reshape(-1, 2)
    return ll[:, 0], ll[:, 1]


def _pairs():
    p = pd.read_parquet(DATA / 'ext_layout_pairs.parquet')
    return {(a, str(s), str(r)): (d / 1000.0, c) for a, s, r, d, c in zip(p.APT, p.stand, p.rwy, p.taxi_dist_m, p.n_rwy_crossings)}


def _taxi(apt, stands, runways, pairs):
    v = np.array([pairs.get((apt, s, r), (np.nan, np.nan)) for s, r in zip(stands, runways)], dtype='float64').reshape(-1, 2)
    return v[:, 0], v[:, 1]


_NORMALS = {}


def _taxis(dep, arr):
    """NM taxi of departures (take-off - AOBT_3, 0-5400 s) and taxi-in of arrivals (in-block - landing, 0-3600 s); NaN else."""
    nm = _sec(dep.MVT_TIME_UTC_mvt) - _sec(dep.AOBT_3_flt)
    ti = _sec(arr.BLOCK_TIME_UTC_mvt) - _sec(arr.MVT_TIME_UTC_mvt)
    return np.where((nm > 0) & (nm < 5400), nm, np.nan), np.where((ti > 0) & (ti < 3600), ti, np.nan)


def _normal_tables():
    """Median NM taxi / taxi-in per (APT, stand, runway), (APT, pier, runway), (APT, runway), APT over the 2025 year.
    Only target-free inputs are read (no departure block time), so held months may enter their own normals."""
    if not _NORMALS:
        dep, arr = _load('train')
        nm, ti = _taxis(dep, arr)
        for kind, df, v in (('dep', dep, nm), ('arr', arr, ti)):
            df = pd.DataFrame({'APT': df.APT.astype(str).values, 'stand': df.STAND_mvt.astype(str).values,
                               'pier': df.apt_pier.astype(str).values, 'rwy': df.RUNWAY_mvt.astype(str).values, 'v': v})
            levels = []
            for keys in (['APT', 'stand', 'rwy'], ['APT', 'pier', 'rwy'], ['APT', 'rwy'], ['APT']):
                g = df.groupby(keys)['v'].agg(['median', 'count'])
                levels.append((keys, g.loc[g['count'] >= SURGE_MIN_N, 'median'].rename('m').reset_index()))
            _NORMALS[kind] = levels
    return _NORMALS


def _normal(kind, df):
    """Each movement's normal taxi from the 2025 tables (NaN when no level has enough values)."""
    key = pd.DataFrame({'APT': df.APT.astype(str).values, 'stand': df.STAND_mvt.astype(str).values,
                        'pier': df.apt_pier.astype(str).values, 'rwy': df.RUNWAY_mvt.astype(str).values})
    out = np.full(len(df), np.nan)
    for keys, tab in _normal_tables()[kind]:
        j = key[keys].merge(tab, on=keys, how='left')['m'].to_numpy(dtype='float64')
        out = np.where(np.isfinite(out), out, j)
    return out


def _wake(w):
    w = pd.Series(w).astype(str).str.upper().to_numpy()
    return np.column_stack([(w == k).astype('float32') for k in ('L', 'M', 'H', 'J')])


def _codes(*arrays):
    """Shared integer codes for the same categorical field across departures and arrivals (NaN -> -1)."""
    allv = pd.concat([pd.Series(a, dtype='object') for a in arrays], ignore_index=True)
    allv = allv.where(allv.notna() & ~allv.astype(str).isin(['nan', 'None', 'NaN', '', '<NA>']), None)
    codes, _ = pd.factorize(allv, use_na_sentinel=True)
    out, i = [], 0
    for a in arrays:
        out.append(codes[i:i + len(a)]); i += len(a)
    return out


def build(which, dep_before=DEP_BEFORE, dep_after=DEP_AFTER, arr_before=ARR_BEFORE, arr_after=ARR_AFTER, window=WINDOW,
          extra=False, surge=False, lead_in=0.0):
    """Return (MVT_ID_mvt of every departure, tokens float16 [n, 1 + dep_before + dep_after + arr_before + arr_after,
    len(FEATURES)]). The defaults are the first probe's context (24/8 departures, 18/6 arrivals, -60/+20 min).

    ``lead_in`` (seconds, ranking file only): movements of the last ``lead_in`` seconds of 2025 join the timeline as
    neighbours, so the first departures of 1 January 2026 see 31 December; they are not returned as rows."""
    t0 = time.time()
    n_tokens = 1 + dep_before + dep_after + arr_before + arr_after
    dep, arr = _load(which)
    dep = dep.assign(_ctx=False); arr = arr.assign(_ctx=False)
    if which == 'rank' and lead_in > 0:
        pdep, parr = _load('train')
        start = pd.Timestamp('2026-01-01', tz='UTC') - pd.Timedelta(seconds=float(lead_in))
        pdep = pdep[pd.to_datetime(pdep.MVT_TIME_UTC_mvt, utc=True) >= start].assign(_ctx=True)
        parr = parr[pd.to_datetime(parr.MVT_TIME_UTC_mvt, utc=True) >= start].assign(_ctx=True)
        dep = pd.concat([pdep, dep], ignore_index=True); arr = pd.concat([parr, arr], ignore_index=True)
        print(f'  lead-in: {len(pdep)} departures and {len(parr)} arrivals from 2025 after {start}', flush=True)
    coords = _coords()
    pairs = _pairs() if extra else None
    nf = len(FEATURES) + (len(EXTRA) if extra else 0) + (len(SURGE) if surge else 0)
    if surge:
        dep['_norm'] = _normal('dep', dep); arr['_norm'] = _normal('arr', arr)
    ids_out, tok_out = [], []
    for apt in sorted(dep.APT.unique()):
        d = dep[dep.APT.values == apt].copy()
        a = arr[arr.APT.values == apt].copy()
        d['t'] = _sec(d.MVT_TIME_UTC_mvt); a['t'] = _sec(a.MVT_TIME_UTC_mvt)
        d = d[np.isfinite(d.t.values)].sort_values('t', kind='stable').reset_index(drop=True)
        a = a[np.isfinite(a.t.values)].sort_values('t', kind='stable').reset_index(drop=True)
        nd, na = len(d), len(a)
        td = d.t.values; ta = a.t.values
        # per-movement fields, departures then arrivals
        d_anchor = _sec(d.AOBT_3_flt); d_sched = _sec(d.SCHED_TIME_UTC_mvt); d_plan = _sec(d.EOBT_1_flt)
        a_anchor = _sec(a.BLOCK_TIME_UTC_mvt); a_sched = _sec(a.SCHED_TIME_UTC_mvt)
        rw_d, rw_a = _codes(d.RUNWAY_mvt.to_numpy(), a.RUNWAY_mvt.to_numpy())
        st_d, st_a = _codes(d.STAND_mvt.astype(str).to_numpy(), a.STAND_mvt.astype(str).to_numpy())
        pi_d, pi_a = _codes(d.apt_pier.to_numpy(), a.apt_pier.to_numpy())
        al_d, al_a = _codes(d.airline.to_numpy(), a.airline.to_numpy())
        lat_d, lon_d = _latlon(apt, d.STAND_mvt.astype(str).to_numpy(), coords)
        lat_a, lon_a = _latlon(apt, a.STAND_mvt.astype(str).to_numpy(), coords)
        wk_d = _wake(d.wake.to_numpy()); wk_a = _wake(a.WK_TBL_CAT_flt.to_numpy())
        copy_a = (np.abs(a_anchor - a_sched) <= 10.0).astype('float32')
        ex_d = ex_a = None
        if extra:
            tk_d, cr_d = _taxi(apt, d.STAND_mvt.astype(str).to_numpy(), d.RUNWAY_mvt.astype(str).to_numpy(), pairs)
            tk_a, cr_a = _taxi(apt, a.STAND_mvt.astype(str).to_numpy(), a.RUNWAY_mvt.astype(str).to_numpy(), pairs)
            codes = np.r_[rw_d, rw_a]
            top = pd.Series(codes[codes >= 0]).value_counts().index[:6].to_numpy()
            oh = lambda rw: np.column_stack([(rw == c).astype('float32') for c in top] +
                                            [np.zeros(len(rw), 'float32')] * (6 - len(top)))
            ex_d = (tk_d, cr_d, oh(rw_d)); ex_a = (tk_a, cr_a, oh(rw_a))
        sg_d = sg_a = None
        if surge:
            nm_d, ti_a = _taxis(d, a)
            sg_d = (d._norm.to_numpy(dtype='float64'), nm_d - d._norm.to_numpy(dtype='float64'))
            sg_a = (a._norm.to_numpy(dtype='float64'), ti_a - a._norm.to_numpy(dtype='float64'))

        # neighbour indices: self, departures in take-off order, arrivals in landing order
        doffs = np.r_[np.arange(-dep_before, 0), np.arange(1, dep_after + 1)]
        didx = np.arange(nd)[:, None] + doffs[None, :]
        dval = (didx >= 0) & (didx < nd)
        didx = np.clip(didx, 0, nd - 1)
        dval &= (td[didx] - td[:, None] >= window[0]) & (td[didx] - td[:, None] <= window[1])
        pos = np.searchsorted(ta, td, side='left')
        aoffs = np.r_[np.arange(-arr_before, 0), np.arange(0, arr_after)]
        aidx = pos[:, None] + aoffs[None, :]
        aval = (aidx >= 0) & (aidx < na) if na else np.zeros_like(aidx, bool)
        aidx = np.clip(aidx, 0, max(na - 1, 0))
        if na:
            aval &= (ta[aidx] - td[:, None] >= window[0]) & (ta[aidx] - td[:, None] <= window[1])

        T = np.zeros((nd, n_tokens, nf), dtype='float32')
        rel = lambda x: np.clip(np.nan_to_num((x - td[:, None]) / SCALE, nan=0.0), -CLIP, CLIP)
        dur = lambda x: np.clip(np.nan_to_num(x / SCALE, nan=0.0), -CLIP, CLIP)

        def fill(sl, idx, val, kind, t_move, anchor, sched, plan, rw, st, pi, al, lat, lon, wk, copy, ex=None, sg=None):
            T[:, sl, F['valid']] = val
            T[:, sl, F[kind]] = val
            T[:, sl, F['move']] = rel(t_move[idx])
            T[:, sl, F['anchor']] = rel(anchor[idx])
            T[:, sl, F['anchor_missing']] = ~np.isfinite(anchor[idx])
            T[:, sl, F['dur']] = dur((t_move[idx] - anchor[idx]) * (1 if kind != 'is_arr' else -1))
            T[:, sl, F['sched']] = rel(sched[idx])
            if plan is not None:
                T[:, sl, F['plan']] = rel(plan[idx]); T[:, sl, F['plan_missing']] = ~np.isfinite(plan[idx])
            me = np.arange(nd)[:, None]
            T[:, sl, F['same_rwy']] = (rw[idx] == rw_d[me]) & (rw_d[me] >= 0)
            T[:, sl, F['same_stand']] = (st[idx] == st_d[me]) & (st_d[me] >= 0)
            T[:, sl, F['same_pier']] = (pi[idx] == pi_d[me]) & (pi_d[me] >= 0)
            T[:, sl, F['same_airline']] = (al[idx] == al_d[me]) & (al_d[me] >= 0)
            dkm = common.haversine(lat[idx], lon[idx], lat_d[me], lon_d[me]) / 1000.0
            T[:, sl, F['stand_km']] = np.clip(np.nan_to_num(dkm, nan=0.0), 0, 4)
            T[:, sl, F['stand_missing']] = ~np.isfinite(dkm)
            for j, k in enumerate(('wake_L', 'wake_M', 'wake_H', 'wake_J')):
                T[:, sl, F[k]] = wk[idx, j]
            if copy is not None:
                T[:, sl, F['arr_copy']] = copy[idx]
            if ex is not None:
                dx = (lon[idx] - lon_d[me]) * np.cos(np.radians(lat_d[me])) * 111.32
                dy = (lat[idx] - lat_d[me]) * 110.57
                T[:, sl, FX['dx_km']] = np.clip(np.nan_to_num(dx, nan=0.0), -4, 4)
                T[:, sl, FX['dy_km']] = np.clip(np.nan_to_num(dy, nan=0.0), -4, 4)
                tk, cr, oh = ex
                T[:, sl, FX['taxi_km']] = np.clip(np.nan_to_num(tk[idx], nan=0.0), 0, 8)
                T[:, sl, FX['taxi_missing']] = ~np.isfinite(tk[idx])
                T[:, sl, FX['rwy_cross']] = np.clip(np.nan_to_num(cr[idx], nan=0.0), 0, 4)
                for j in range(6):
                    T[:, sl, FX[f'rwy{j}']] = oh[idx, j]
            if sg is not None:
                norm, res = sg
                T[:, sl, surge_index('norm', nf)] = np.clip(np.nan_to_num(norm[idx] / SCALE, nan=0.0), 0, 6)
                T[:, sl, surge_index('dur_res', nf)] = np.clip(np.nan_to_num(res[idx] / SCALE, nan=0.0), -2, 2)
                T[:, sl, surge_index('norm_missing', nf)] = ~np.isfinite(norm[idx])
            T[:, sl, :] *= val[..., None]

        s_dep = slice(1, 1 + len(doffs)); s_arr = slice(1 + len(doffs), n_tokens)
        fill(slice(0, 1), np.arange(nd)[:, None], np.ones((nd, 1), bool), 'is_self', td, d_anchor, d_sched, d_plan,
             rw_d, st_d, pi_d, al_d, lat_d, lon_d, wk_d, None, ex_d, sg_d)
        fill(s_dep, didx, dval, 'is_dep', td, d_anchor, d_sched, d_plan, rw_d, st_d, pi_d, al_d, lat_d, lon_d, wk_d, None, ex_d,
             sg_d)
        if na and len(aoffs):
            fill(s_arr, aidx, aval, 'is_arr', ta, a_anchor, a_sched, None, rw_a, st_a, pi_a, al_a, lat_a, lon_a, wk_a, copy_a, ex_a,
                 sg_a)
        keep = ~d._ctx.to_numpy(dtype=bool)          # lead-in movements are neighbours only
        ids_out.append(d.MVT_ID_mvt.to_numpy()[keep]); tok_out.append(T[keep].astype('float16'))
        print(f'  tokens {which} {apt}: {nd} departures, {na} arrivals, mean valid dep {dval.sum(1).mean():.1f} '
              f'arr {aval.sum(1).mean():.1f} [{time.time() - t0:.0f}s]', flush=True)
    return np.concatenate(ids_out), np.concatenate(tok_out)


def aligned(ids, tokens, want_ids):
    """Rows of ``tokens`` in the order of ``want_ids`` (every wanted id must exist)."""
    pos = pd.Series(np.arange(len(ids)), index=ids)
    idx = pos.reindex(want_ids).to_numpy()
    if np.isnan(idx.astype('float64')).any():
        raise SystemExit('neighbour tokens missing for some departures')
    return tokens[idx.astype('int64')]
