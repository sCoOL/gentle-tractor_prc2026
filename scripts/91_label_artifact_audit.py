"""Label-artifact audit of every 2025 departure off-block, priced with v21 strict OOF errors.

  py -3.14 scripts/91_label_artifact_audit.py

Writes runs/label_artifact_audit_<tag>/{summary.json, copy_modes.csv, long_label_classes.csv,
minute_resolution.csv, lirf_inbound_copy.csv}. Read-only: no model, feature or submission changes.
"""
import argparse
import glob
import json
import time

import numpy as np
import pandas as pd

from common import DATA, RUNS

parser = argparse.ArgumentParser()
parser.add_argument('--tag', default=time.strftime('%Y%m%d'))
parser.add_argument('--oof', default=str(RUNS / 'oof' / 'three_expert_tail_v4_richgate_strict_20260917.parquet'))
args = parser.parse_args()
out = RUNS / f'label_artifact_audit_{args.tag}'
out.mkdir(parents=True, exist_ok=True)
TOL = 10.0


def secs(s):
    return (s.astype('int64') // 1_000_000).astype('float64').where(s.notna())


raw = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(DATA / 'training_2025-*.parquet')))], ignore_index=True)
for col, name in [('MVT_TIME_UTC_mvt', 't'), ('BLOCK_TIME_UTC_mvt', 'b'), ('SCHED_TIME_UTC_mvt', 's'), ('AOBT_3_flt', 'a3'),
                  ('EOBT_1_flt', 'e1'), ('LOBT_flt', 'lo'), ('ARVT_3_flt', 'ar3'), ('ARVT_1_flt', 'ar1')]:
    raw[name] = secs(raw[col])
raw['APT'] = np.where(raw.PHASE_mvt == 'DEP', raw.ADEP_mvt, raw.ADES_mvt)
raw['month'] = raw.MVT_TIME_UTC_mvt.dt.month
dep = raw[raw.PHASE_mvt == 'DEP'].copy()
arr = raw[raw.PHASE_mvt == 'ARR'].copy()
dep['y'] = dep.TAXITIME_SEC_mvt
dep['matched'] = dep.a3.notna()
dep['A'] = dep.t - dep.a3
assert ((dep.t - dep.b) == dep.y).all(), 'label is not take-off minus off-block'

dep['copy_sched'] = (dep.b - dep.s).abs() <= TOL
dep['copy_aobt3'] = (dep.b - dep.a3).abs() <= TOL
dep['copy_eobt_lobt'] = ((dep.b - dep.e1).abs() <= TOL) | ((dep.b - dep.lo).abs() <= TOL)
dep['off_1h_early_vs_nm'] = dep.matched & ((dep.b - dep.a3 + 3600).abs() <= 120)
dep['off_1h_late_vs_nm'] = dep.matched & ((dep.b - dep.a3 - 3600).abs() <= 120)
dep['datebug'] = (dep.y >= 72000) & ~dep.copy_sched & (np.mod(dep.y, 86400) < 5400)

oof = pd.read_parquet(args.oof, columns=['MVT_ID_mvt', 'pred'])
dep = dep.merge(oof, on='MVT_ID_mvt', how='left', validate='one_to_one')
if dep.pred.isna().any():
    raise ValueError('OOF predictions do not cover every departure')
dep['e2'] = (dep.pred - dep.y) ** 2
TOT = float(dep.e2.sum())


def price(mask):
    m = np.asarray(mask, bool)
    return dict(rows=int(m.sum()), rmse=float(np.sqrt(dep.e2[m].mean())) if m.any() else None,
                sse_share=float(dep.e2[m].sum() / TOT))


# 1) copy modes by airport and lane
copy = dep.groupby(['APT', 'matched'])[['copy_sched', 'copy_aobt3', 'copy_eobt_lobt', 'off_1h_early_vs_nm', 'off_1h_late_vs_nm']].mean()
copy['rows'] = dep.groupby(['APT', 'matched']).size()
copy.round(5).to_csv(out / 'copy_modes.csv')

# 2) long-label classes
cls = np.select(
    [dep.y >= 72000, dep.copy_sched, dep.copy_eobt_lobt, dep.off_1h_early_vs_nm, dep.off_1h_late_vs_nm,
     dep.matched & ((dep.y - dep.A).abs() <= 600), dep.matched],
    ['over_20h', 'schedule_default', 'eobt_lobt_default', 'offblock_1h_early_vs_nm', 'offblock_1h_late_vs_nm',
     'real_long_nm_agrees', 'pushback_then_hold_nm_estimate_only'], 'no_nm_match')
long = dep.y > 3600
classes = []
for c in pd.unique(cls[long]):
    m = long & (cls == c)
    classes.append(dict(label_class=c, **price(m), airports=dep.loc[m, 'APT'].value_counts().to_dict()))
pd.DataFrame(classes).sort_values('sse_share', ascending=False).to_csv(out / 'long_label_classes.csv', index=False)

# 3) minute-resolution timestamps (share of :00 seconds; uniform would be 1/60)
mres = pd.DataFrame({
    'dep_offblock_sec0': dep.groupby('APT').b.apply(lambda s: float((s % 60 == 0).mean())),
    'dep_takeoff_sec0': dep.groupby('APT').t.apply(lambda s: float((s % 60 == 0).mean())),
    'arr_inblock_sec0': arr.groupby('APT').b.apply(lambda s: float((s % 60 == 0).mean())),
    'offblock_sec0_month_min': dep.assign(z=dep.b % 60 == 0).groupby(['APT', 'month']).z.mean().groupby('APT').min(),
    'offblock_sec0_month_max': dep.assign(z=dep.b % 60 == 0).groupby(['APT', 'month']).z.mean().groupby('APT').max(),
})
mres.round(4).to_csv(out / 'minute_resolution.csv')

# 4) LIRF: does the inbound arrival's schedule-copied in-block flag the departure's schedule default?
arr['key'] = arr.APT + '|' + arr.STAND_mvt.fillna('').astype(str)
arr['al'] = arr.FLIGHT_mvt.fillna('').astype(str).str.extract(r'^([A-Z]{3}|[A-Z0-9]{2})', expand=False).fillna('')
dep['key'] = dep.APT + '|' + dep.STAND_mvt.fillna('').astype(str)
dep['al'] = dep.FLIGHT_mvt.fillna('').astype(str).str.extract(r'^([A-Z]{3}|[A-Z0-9]{2})', expand=False).fillna('')
la = arr[(arr.APT == 'LIRF') & arr.b.notna()].sort_values(['key', 'b']).reset_index(drop=True)
la['in_copy_sched'] = (la.b - la.s).abs() <= TOL
ld = dep[dep.APT == 'LIRF'].reset_index(drop=True)
flag = np.full(len(ld), np.nan)
idx = la.groupby('key').indices
ab, aal = la.b.to_numpy(), la.al.to_numpy()
aty = la.AIRCRAFT_TYPE_mvt.fillna('').astype(str).to_numpy(); acs = la.in_copy_sched.to_numpy()
dt, ds, dal = ld.t.to_numpy(), ld.s.to_numpy(), ld.al.to_numpy()
dty = ld.AIRCRAFT_TYPE_mvt.fillna('').astype(str).to_numpy()
for k, di in ld.groupby('key').indices.items():
    if k.endswith('|') or k not in idx:
        continue
    ai = idx[k]; b = ab[ai]
    for row in di:
        limit = min(dt[row], ds[row] + 7200.0) if np.isfinite(ds[row]) else dt[row]
        p = np.searchsorted(b, limit, side='right') - 1; steps = 0
        while p >= 0 and steps < 6:
            c = ai[p]
            if dt[row] - ab[c] > 72000:
                break
            if aal[c] == dal[row] and aty[c] == dty[row]:
                flag[row] = acs[c]; break
            p -= 1; steps += 1
ld['inbound_inblock_is_schedule'] = flag
inbound = ld.dropna(subset=['inbound_inblock_is_schedule']).groupby(['matched', 'inbound_inblock_is_schedule']).agg(
    rows=('y', 'size'), departure_schedule_default_rate=('copy_sched', 'mean'))
inbound.round(4).to_csv(out / 'lirf_inbound_copy.csv')

# 5) near-on-time extreme labels outside Rome: look-alike base rates (the probability any model could assign)
dep['airline_code'] = dep.al
dep['near_on_time'] = (dep.t - dep.s).abs() <= 3600
dep['extreme'] = (dep.y > 3 * 3600) & (dep.y > (dep.t - dep.s) + 3 * 3600)
stand = dep.STAND_mvt.fillna('').astype(str)
look = {
    'non_LIRF_near_on_time': (dep.APT != 'LIRF') & dep.near_on_time,
    'non_LIRF_near_on_time_unmatched': (dep.APT != 'LIRF') & dep.near_on_time & ~dep.matched,
    'non_LIRF_near_on_time_matched': (dep.APT != 'LIRF') & dep.near_on_time & dep.matched,
    'LSZH_unmatched_near_on_time': (dep.APT == 'LSZH') & ~dep.matched & dep.near_on_time,
    'LFPG_unmatched_near_on_time': (dep.APT == 'LFPG') & ~dep.matched & dep.near_on_time,
    'LFPG_EJU_unmatched_near_on_time': (dep.APT == 'LFPG') & ~dep.matched & dep.near_on_time & (dep.airline_code == 'EJU'),
    'LSZH_GA_stand_MFG': (dep.APT == 'LSZH') & (stand == 'MFG'),
    'EHAM_hangar_stands_HG': (dep.APT == 'EHAM') & stand.str.startswith('HG'),
    'LIRF_near_on_time': (dep.APT == 'LIRF') & dep.near_on_time,
}
lookalike = pd.DataFrame([dict(group=k, rows=int(m.sum()), extreme=int((m & dep.extreme).sum()),
                               base_rate=float((m & dep.extreme).sum() / m.sum()) if m.sum() else None,
                               extreme_sse_share=float(dep.e2[m & dep.extreme].sum() / TOT)) for k, m in look.items()])
lookalike.to_csv(out / 'near_on_time_extreme_lookalikes.csv', index=False)
ext = dep[(dep.APT != 'LIRF') & dep.near_on_time & dep.extreme]
ext[['MVT_ID_mvt', 'APT', 'FLIGHT_mvt', 'AIRCRAFT_TYPE_mvt', 'STAND_mvt', 'SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'y', 'pred']].to_csv(
    out / 'near_on_time_extreme_rows_outside_LIRF.csv', index=False)

summary = {
    'oof': args.oof, 'departures': int(len(dep)), 'total_sse': TOT,
    'labels_over_1h': price(long), 'body_le_1h': price(~long),
    'matched_copy_sched': price(dep.matched & dep.copy_sched),
    'matched_copy_aobt3': price(dep.matched & dep.copy_aobt3),
    'matched_copy_aobt3_rmse_if_label_known': float(np.sqrt(((dep.A - dep.y)[dep.matched & dep.copy_aobt3] ** 2).mean())),
    'matched_copy_eobt_lobt_not_sched': price(dep.matched & dep.copy_eobt_lobt & ~dep.copy_sched),
    'offblock_1h_vs_nm': price(dep.off_1h_early_vs_nm | dep.off_1h_late_vs_nm),
    'datebug_rows': int(dep.datebug.sum()), 'datebug_matched_rows': int((dep.datebug & dep.matched).sum()),
    'labels_le_60s': price(dep.y <= 60),
    'near_on_time_extreme_outside_LIRF': price((dep.APT != 'LIRF') & dep.near_on_time & dep.extreme),
}
(out / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
print(json.dumps(summary, indent=1))
print(pd.read_csv(out / 'long_label_classes.csv').drop(columns='airports').round(4).to_string(index=False))
print(inbound.round(4))
print(lookalike.round(6).to_string(index=False))
print(f'wrote {out}')
