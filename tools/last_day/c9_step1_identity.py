"""EUROCONTROL daily identity check: does 60 x DLY_ALL_PRE_2 (EUROCONTROL daily all pre-departure delay, min) equal
sum max(0, BLOCK - SCHED) = sum max(0, D - y) of our 2025 departures per airport-day? Three day definitions compared
(take-off, off-block and schedule UTC day). `ec(year)` is the loader every later post-step uses.

  py -3.14 tools/last_day/c9_step1_identity.py          # -> runs/last_day/c9_step1_identity.json
"""
import numpy as np
import pandas as pd
import ld

EC = ld.EC_DIR
APTS = list(ld.TZ)


def ec(year):
    e = pd.read_csv(EC / f'all_pre_departure_delays_{year}.csv', usecols=['FLT_DATE', 'APT_ICAO', 'FLT_DEP_1', 'FLT_DEP_IFR_2', 'DLY_ALL_PRE_2'])
    e = e[e.APT_ICAO.isin(APTS)].rename(columns={'APT_ICAO': 'APT'})
    e['day'] = pd.to_datetime(e.FLT_DATE).dt.date
    e['K'] = 60.0 * e.DLY_ALL_PRE_2
    return e


if __name__ == '__main__':
    a = ld.load_2025()
    a['blk'] = a.t - a.y
    a['dly'] = np.maximum(0.0, a.D - a.y)
    e = ec(2025)
    res = {}
    for name, col in (('takeoff_utc', 't'), ('offblock_utc', 'blk'), ('sched_utc', 't_sched')):
        a['day'] = pd.to_datetime(a[col], unit='s', utc=True).dt.date
        g = a.groupby(['APT', 'day']).agg(S=('dly', 'sum'), n=('dly', 'size')).reset_index()
        m = g.merge(e, on=['APT', 'day'], how='inner')
        m = m[m.K.notna()]
        per = {}
        for apt, h in m.groupby('APT'):
            per[apt] = {'days': len(h), 'corr': round(float(np.corrcoef(h.S, h.K)[0, 1]), 5),
                        'ratio_sum': round(float(h.K.sum() / h.S.sum()), 4),
                        'median_abs_diff_min': round(float(((h.K - h.S).abs() / 60).median()), 2),
                        'days_count_mismatch_gt2pct': int(((h.n - h.FLT_DEP_IFR_2).abs() > 0.02 * h.FLT_DEP_IFR_2).sum()),
                        'count_ratio_mean': round(float((h.n / h.FLT_DEP_IFR_2).mean()), 4)}
        res[name] = per
        print(name, 'pooled corr', round(float(np.corrcoef(m.S, m.K)[0, 1]), 5))
        for apt, v in per.items():
            print('   ', apt, v)
    ld.save('c9_step1_identity', res)
