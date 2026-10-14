"""Extract new off-block candidates from existing ground trajectories.

Different from prior aggregate geometry/shadow features: several sustained
motion onsets, including speed and displacement over time. Observation-only
feature extraction; labels are joined afterward for a bounded source screen.
"""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import common

e = importlib.import_module('121_adsb_events')
OUT = common.RUNS/'v49_motion_candidates'
DISTANCES = [8., 20., 40., 80.]
SPEEDS = [1., 3., 5.]
COLUMNS = ['motion_d8', 'motion_d20', 'motion_d40', 'motion_d80',
           'motion_v1', 'motion_v3', 'motion_v5', 'motion_window15']
DATES = ['2025-01-15', '2025-07-15', '2025-09-15', '2025-11-15', '2025-12-15']


def onsets(t, x, y, speed, takeoff):
    """Return candidate taxi durations, with missing evidence left missing."""
    if len(t) < 3:
        return np.full(len(COLUMNS), np.nan)
    t, x, y, speed = map(np.asarray, [t, x, y, speed])
    distance = np.hypot(x-x[0], y-y[0])
    values = []
    def first(mask):
        # Two observations, separated by no more than30s, must support onset.
        good = np.where(mask[:-1] & mask[1:] & (np.diff(t) <= 30.))[0]
        if not len(good):
            return np.nan
        return float(takeoff-t[max(int(good[0])-1, 0)])
    for radius in DISTANCES:
        values.append(first(distance >= radius))
    for threshold in SPEEDS:
        values.append(first(speed >= threshold))
    future = np.searchsorted(t, t+15.)
    valid = future < len(t)
    safe = np.minimum(future, len(t)-1)
    movement = np.hypot(x[safe]-x, y[safe]-y)
    mask = valid & ((t[safe]-t) <= 30.) & (movement >= 8.)
    good = np.where(mask)[0]
    values.append(float(takeoff-t[int(good[0])]) if len(good) else np.nan)
    return np.asarray(values)


def extract(date):
    path = OUT/f'features_{date}.parquet'
    if path.exists():
        return pd.read_parquet(path)
    ev, matches = e.match_day(date)
    wanted = ev.loc[ev.at_stand & ~ev.after_landing & ev.n_ground.ge(3)]
    # final-file change: was e.OUT/points_<date> (e.OUT is the derived-table folder; raw points may live elsewhere)
    points = pd.read_parquet(e.P118.points_path(date), columns=['icao', 'APT', 't', 'lat', 'lon', 'gs'])
    points = points.sort_values('t')
    grouped = {key: g for key, g in points.groupby(['icao', 'APT'], sort=False)}
    rows = []
    for row in wanted.itertuples():
        g = grouped.get((row.icao, row.APT))
        if g is None:
            continue
        t = g.t.to_numpy()
        a, b = np.searchsorted(t, [row.t_first, row.t_tko])
        g = g.iloc[a:b]
        x, y = e.xy(row.APT, g.lat.to_numpy(), g.lon.to_numpy())
        values = onsets(g.t.to_numpy(), x, y, g.gs.to_numpy(), row.t_tko)
        # Convert from event-relative to each movement-relative take-off below.
        rows.append({'k': row.k, 'event_takeoff': row.t_tko, **dict(zip(COLUMNS, values))})
    f = matches[['MVT_ID_mvt', 'k', 't_mvt', 'leave_lo']].merge(
        pd.DataFrame(rows, columns=['k', 'event_takeoff']+COLUMNS), on='k', validate='many_to_one')
    for col in COLUMNS:
        f[col] += f.t_mvt-f.event_takeoff
    f['original_taxi'] = f.t_mvt-f.leave_lo
    f['date'] = date
    f = f[['MVT_ID_mvt', 'date', 'original_taxi']+COLUMNS]
    f.to_parquet(path, index=False)
    return f


def screen(features):
    raw = pd.read_parquet(common.DATA/'base_train.parquet', columns=['MVT_ID_mvt', 'APT',
        'TAXITIME_SEC_mvt', 'atot_minus_sched', 'month', 'is_heli'])
    quality = pd.read_parquet(e.OUT / 'adsb_features.parquet', columns=['MVT_ID_mvt', 'adsb_q_med'])
    d = features.merge(raw, on='MVT_ID_mvt', validate='one_to_one').merge(quality, on='MVT_ID_mvt', validate='one_to_one')
    d = d.loc[(d.TAXITIME_SEC_mvt-d.atot_minus_sched).abs().gt(10) & d.is_heli.eq(0)
              & d.original_taxi.notna() & d.adsb_q_med.notna()].copy()
    # Existing measurement-member calibration from arrivals, available in both years.
    base = d.original_taxi.to_numpy()+d.adsb_q_med.to_numpy()
    observed = d[COLUMNS].to_numpy()
    P = np.where(np.isfinite(observed), observed, d.original_taxi.to_numpy()[:, None])+d.adsb_q_med.to_numpy()[:, None]
    y = d.TAXITIME_SEC_mvt.to_numpy()
    train = d.month.le(9).to_numpy()
    test = d.month.ge(11).to_numpy()
    # Bounded positive corrections from the existing measurement, with total
    # exposure <=1. Candidate weights/bias use earlier dates only.
    from scipy.optimize import minimize
    delta = P-base[:, None]
    centre = np.mean(y[train]-base[train])
    def loss(w):
        err = base[train]+centre+delta[train]@w-y[train]
        return (np.mean(err**2)+100.*np.dot(w, w))/1e6
    def gradient(w):
        err = base[train]+centre+delta[train]@w-y[train]
        return (2.*delta[train].T@err/train.sum()+200.*w)/1e6
    opt = minimize(loss, np.zeros(len(COLUMNS)), bounds=[(0., 1.)]*len(COLUMNS),
                   jac=gradient, constraints=[{'type': 'ineq', 'fun': lambda w: 1.-w.sum()}], method='SLSQP',
                   options={'maxiter': 500, 'ftol': 1e-8})
    assert opt.success, opt.message
    # The control gets the identical training-fitted offset; the comparison
    # measures candidate timing information, not a free extra intercept.
    baseline = base+centre
    candidate = baseline+delta@opt.x
    rows = []
    for name, mask in [('pooled', test)]+[(str(m), test & d.month.eq(m).to_numpy()) for m in [11, 12]]:
        a, b = (baseline[mask]-y[mask])**2, (candidate[mask]-y[mask])**2
        keep = np.ones(len(a), bool)
        keep[np.argsort(-np.abs(a-b))[:10]] = False
        rows.append({'slice': name, 'n': int(mask.sum()), 'baseline': float(np.sqrt(a.mean())),
            'candidate': float(np.sqrt(b.mean())), 'delta': float(np.sqrt(b.mean())-np.sqrt(a.mean())),
            'delta_without_top10': float(np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean()))})
    report = {'scope': 'Five-date source timing screen; Jan/Jul/Sep fitting, Nov/Dec held. Not submission RMSE.',
              'weights': dict(zip(COLUMNS, opt.x.tolist())), 'training_offset': float(centre),
              'train_rows': int(train.sum()), 'rows': rows, 'submission_qualified': False}
    d.assign(baseline=baseline, candidate=candidate).to_parquet(OUT/'screen_ledger.parquet', index=False)
    (OUT/'screen.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dates', nargs='+', default=DATES)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT/'164_ground_motion_candidates.executed.py').write_bytes(Path(__file__).read_bytes())
    parts = []
    for date in args.dates:
        f = extract(date)
        parts.append(f)
        print(date, len(f), 'motion rows', flush=True)
    screen(pd.concat(parts, ignore_index=True))
