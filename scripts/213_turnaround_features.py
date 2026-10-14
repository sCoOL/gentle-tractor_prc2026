"""Turnaround-link features: fixed same-stand rule + ADS-B aircraft-identity inbound.

  py -3.14 scripts/213_turnaround_features.py --workers 6

Every ARR and DEP movement gets the icao24 of the aircraft ADS-B saw landing / taking off with the same callsign
within 180 s (unique on both sides). A departure's identity inbound is the latest identified arrival of the same
aircraft at the same airport before its take-off, within 48 h, with no identified departure of it in between. Only
rank-available inputs: arrival rows (block times are given in both years), departure take-off, schedule, stand and
callsign, and the ADS-B traces. No departure block time or label is read.
Outputs runs/v51_turnaround/{ident.parquet, features.parquet, link_audit.json}.
"""
import argparse
import glob
import importlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import common

OUT = common.RUNS / 'v51_turnaround'
TOL = 180.0
SKIP_S, LOOKBACK_S, STEPS, ID_MAX_S = 1800.0, 72000.0, 6, 48 * 3600.0
RAW = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'STAND_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt',
       'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'AIRCRAFT_TYPE_mvt', 'CALLSIGN_flt']
COLS = ['tn_turn_gap', 'tn_turn_delay', 'tn_id_gap', 'tn_id_delay', 'tn_id_towed', 'tn_id_agree',
        'tn_best_gap', 'tn_best_delay', 'tn_best_src', 'tn_sched_slack', 'tn_inb_taxi_in']


def secs(s):
    s = pd.to_datetime(s)
    return (s.astype('int64') // 1_000_000).astype('float64').where(s.notna()).to_numpy()


def airline(s):
    return s.fillna('').astype(str).str.extract(r'^([A-Z]{3}|[A-Z0-9]{2})', expand=False).fillna('')


def movements():
    """2025 training + 2026 ranking movements with the few raw columns this needs."""
    tr = pd.concat([pd.read_parquet(f, columns=RAW) for f in sorted(glob.glob(str(common.DATA / 'training_2025-*.parquet')))],
                   ignore_index=True)
    rk = pd.read_parquet(common.DATA / 'ranking.parquet', columns=RAW)
    rk = rk.assign(BLOCK_TIME_UTC_mvt=rk.BLOCK_TIME_UTC_mvt.where(rk.PHASE_mvt == 'ARR'))  # departures: never read
    d = pd.concat([tr, rk], ignore_index=True)
    d.loc[d.PHASE_mvt == 'DEP', 'BLOCK_TIME_UTC_mvt'] = pd.NaT
    assert d.MVT_ID_mvt.is_unique
    d['APT'] = np.where(d.PHASE_mvt == 'DEP', d.ADEP_mvt, d.ADES_mvt)
    d['cs'] = d.CALLSIGN_flt.astype('string').str.strip().replace('', pd.NA)
    d['t'] = secs(d.MVT_TIME_UTC_mvt)
    d['ts'] = secs(d.SCHED_TIME_UTC_mvt)
    d['tb'] = secs(d.BLOCK_TIME_UTC_mvt)
    d['date'] = d.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d')
    return d


def _unique(rows, ev):
    c = rows.merge(ev, on=['APT', 'cs'])
    c = c[(c.t_ev - c.t).abs() <= TOL]
    ok = c.groupby('MVT_ID_mvt').MVT_ID_mvt.transform('size').eq(1) & c.groupby('k').k.transform('size').eq(1)
    return c[ok]


def identify_day(job):
    """icao24 for one day's ARR (air->ground transition) and DEP (ground->air) movements."""
    date, arr_rows, dep_rows = job
    f = OUT / 'days' / f'{date}.parquet'
    if f.exists():
        return pd.read_parquet(f)
    e = importlib.import_module('121_adsb_events')
    p = pd.read_parquet(e.P118.points_path(date), columns=['icao', 'APT', 't', 'ground', 'cs'])
    p = p.sort_values(['icao', 'APT', 't'], kind='stable').reset_index(drop=True)
    p['cs'] = p.cs.astype('string').str.strip().replace('', pd.NA)
    gr, t = p.ground.to_numpy(bool), p.t.to_numpy()
    ic, ap = p.icao.to_numpy(), p.APT.to_numpy()
    same = np.r_[False, (ic[1:] == ic[:-1]) & (ap[1:] == ap[:-1])]
    dt = np.r_[np.inf, np.diff(t)]
    land = same & gr & ~np.r_[True, gr[:-1]] & (dt <= 120)                  # first ground point after airborne
    tko = np.r_[same[1:] & ~gr[1:] & gr[:-1] & (dt[1:] <= 120), False]      # last ground point before airborne
    events = {'ARR': p.loc[land, ['icao', 'APT', 'cs']].assign(t_ev=t[np.flatnonzero(land) - 1]),
              'DEP': p.loc[tko, ['icao', 'APT', 'cs']].assign(t_ev=t[np.flatnonzero(tko) + 1])}
    out = []
    for phase, rows in (('ARR', arr_rows), ('DEP', dep_rows)):
        ev = events[phase].dropna(subset=['cs']).reset_index(drop=True).assign(k=lambda z: np.arange(len(z)))
        m = _unique(rows, ev)
        out.append(pd.DataFrame({'MVT_ID_mvt': m.MVT_ID_mvt.to_numpy(), 'icao': m.icao.to_numpy(), 'phase': phase}))
    r = pd.concat(out, ignore_index=True).assign(date=date)
    r.to_parquet(f, index=False)
    return r


def identify(d, workers):
    (OUT / 'days').mkdir(parents=True, exist_ok=True)
    e = importlib.import_module('121_adsb_events')
    days = e.days()
    k = d.dropna(subset=['cs'])[['MVT_ID_mvt', 'APT', 'cs', 't', 'date', 'PHASE_mvt']]
    by = dict(tuple(k.groupby('date')))
    jobs = [(x, by[x].loc[by[x].PHASE_mvt == 'ARR', ['MVT_ID_mvt', 'APT', 'cs', 't']],
             by[x].loc[by[x].PHASE_mvt == 'DEP', ['MVT_ID_mvt', 'APT', 'cs', 't']]) for x in days if x in by]
    with ProcessPoolExecutor(workers) as pool:
        res = list(pool.map(identify_day, jobs, chunksize=2))
    ident = pd.concat(res, ignore_index=True).drop_duplicates('MVT_ID_mvt', keep=False)
    ident.to_parquet(OUT / 'ident.parquet', index=False)
    return ident


def stand_rule(d):
    """G10's turnaround search with the last SKIP_S before take-off skipped; returns the chosen arrival's row index."""
    key = d.APT.astype(str) + '|' + d.STAND_mvt.fillna('').astype(str)
    al = airline(d.FLIGHT_mvt).to_numpy()
    typ = d.AIRCRAFT_TYPE_mvt.fillna('').astype(str).to_numpy()
    t, ts, tb = d.t.to_numpy(), d.ts.to_numpy(), d.tb.to_numpy()
    is_arr = (d.PHASE_mvt == 'ARR').to_numpy() & np.isfinite(tb)
    arr_pos = np.flatnonzero(is_arr)
    arr_pos = arr_pos[np.lexsort((tb[arr_pos], key.to_numpy()[arr_pos]))]
    groups = pd.Series(arr_pos).groupby(key.to_numpy()[arr_pos], sort=False).indices
    link = np.full(len(d), -1, dtype=np.int64)
    dep_pos = np.flatnonzero((d.PHASE_mvt == 'DEP').to_numpy())
    for k, rows in pd.Series(dep_pos).groupby(key.to_numpy()[dep_pos], sort=False).indices.items():
        if k.endswith('|') or k.endswith('|NA') or k not in groups:
            continue
        ai = arr_pos[groups[k]]
        ab = tb[ai]
        for r in dep_pos[rows]:
            limit = t[r] - SKIP_S
            if np.isfinite(ts[r]):
                limit = min(limit, ts[r] + 7200.0)
            p, steps = np.searchsorted(ab, limit, side='right') - 1, 0
            while p >= 0 and steps < STEPS:
                c = ai[p]
                if t[r] - tb[c] > LOOKBACK_S:
                    break
                if al[c] == al[r] and typ[c] == typ[r]:
                    link[r] = c
                    break
                p -= 1
                steps += 1
    return link


def identity_inbound(d, ident):
    """Row index of each departure's identity inbound (or -1)."""
    pos = pd.Series(np.arange(len(d)), index=d.MVT_ID_mvt.to_numpy())
    rows = pos.reindex(ident.MVT_ID_mvt)
    assert rows.notna().all(), 'identified movement missing from the movement table'
    x = ident.assign(row=rows.to_numpy().astype(np.int64))
    x = x.assign(APT=d.APT.to_numpy()[x.row], t=d.t.to_numpy()[x.row], phase=d.PHASE_mvt.to_numpy()[x.row])
    x = x.sort_values(['icao', 'APT', 't'], kind='stable').reset_index(drop=True)
    ic, ap, ph, t, row = x.icao.to_numpy(), x.APT.to_numpy(), x.phase.to_numpy(), x.t.to_numpy(), x.row.to_numpy()
    ok = np.r_[False, (ic[1:] == ic[:-1]) & (ap[1:] == ap[:-1]) & (ph[:-1] == 'ARR') & (ph[1:] == 'DEP')
               & (t[1:] - t[:-1] <= ID_MAX_S)]
    inbound = np.full(len(d), -1, dtype=np.int64)
    dep_rows, arr_rows = row[ok], row[np.flatnonzero(ok) - 1]
    good = d.tb.to_numpy()[arr_rows] < d.t.to_numpy()[dep_rows]            # in-block before take-off
    inbound[dep_rows[good]] = arr_rows[good]
    return inbound


def build(d, ident):
    link = stand_rule(d)
    inb = identity_inbound(d, ident)
    dep = np.flatnonzero((d.PHASE_mvt == 'DEP').to_numpy())
    t, ts, tb, tl = d.t.to_numpy(), d.ts.to_numpy(), d.tb.to_numpy(), d.t.to_numpy()
    stand = d.STAND_mvt.astype('string').fillna('').to_numpy()

    def take(v, idx):
        return np.where(idx >= 0, v[np.maximum(idx, 0)], np.nan)

    L, I = link[dep], inb[dep]
    f = pd.DataFrame({'MVT_ID_mvt': d.MVT_ID_mvt.to_numpy()[dep]})
    f['tn_turn_gap'] = t[dep] - take(tb, L)
    f['tn_turn_delay'] = take(tb, L) - take(ts, L)
    f['tn_id_gap'] = t[dep] - take(tb, I)
    f['tn_id_delay'] = take(tb, I) - take(ts, I)
    si = np.where(I >= 0, stand[np.maximum(I, 0)], '')
    known = (I >= 0) & (si != '') & (si != 'NA') & (stand[dep] != '') & (stand[dep] != 'NA')
    f['tn_id_towed'] = np.where(known, (si != stand[dep]).astype(float), np.nan)
    f['tn_id_agree'] = np.where(I >= 0, (L == I).astype(float), np.nan)
    best = np.where(I >= 0, I, L)
    f['tn_best_gap'] = t[dep] - take(tb, best)
    f['tn_best_delay'] = take(tb, best) - take(ts, best)
    f['tn_best_src'] = np.select([I >= 0, L >= 0], [2., 1.], 0.)
    f['tn_sched_slack'] = ts[dep] - take(tb, best)
    f['tn_inb_taxi_in'] = take(tb, best) - take(tl, best)
    for c in COLS:
        f[c] = f[c].astype('float32')
    return f, link, inb


def audit(d, ident, link, inb):
    """Link precision/recall against identity (same numbers as the protocol's audit, for the fixed rule)."""
    icao = pd.Series(ident.icao.to_numpy(), index=ident.MVT_ID_mvt.to_numpy())
    ids = d.MVT_ID_mvt.to_numpy()
    dep = np.flatnonzero((d.PHASE_mvt == 'DEP').to_numpy())
    di = icao.reindex(ids[dep]).to_numpy()
    li = icao.reindex(np.where(link[dep] >= 0, ids[np.maximum(link[dep], 0)], -1)).to_numpy()
    chk = (link[dep] >= 0) & pd.notna(di) & pd.notna(li)
    known = inb[dep] >= 0
    return {'departures': int(len(dep)), 'stand_rule_links': int((link[dep] >= 0).sum()),
            'identity_inbounds': int(known.sum()),
            'stand_rule_precision_on_identified': float((di[chk] == li[chk]).mean()),
            'stand_rule_recall_correct': float((link[dep][known] == inb[dep][known]).mean()),
            'stand_rule_wrong': float(((link[dep][known] >= 0) & (link[dep][known] != inb[dep][known])).mean()),
            'stand_rule_none': float((link[dep][known] < 0).mean())}


def main(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    d = movements()
    print(f'{len(d):,} movements', flush=True)
    ident = identify(d, workers)
    print(f'identified {len(ident):,} movements', flush=True)
    f, link, inb = build(d, ident)
    f.to_parquet(OUT / 'features.parquet', index=False)
    rep = audit(d, ident, link, inb)
    rep['coverage'] = {c: float(f[c].notna().mean()) for c in COLS}
    (OUT / 'link_audit.json').write_text(json.dumps(rep, indent=2))
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(rep, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    main(ap.parse_args().workers)
