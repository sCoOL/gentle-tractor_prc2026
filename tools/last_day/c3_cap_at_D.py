"""The cap p <= D (take-off - schedule) for D > 2 h, unless the row carries date-bug mass (LIRF NM-unmatched
next-local-day rows = tail_experts.b_domain). A fixed rule, so the six-fold and forward predictions are the same; this
script gives its 2025 counts (labels at / above / below D on the rows it moves) and its 2026 rows. build_A15L.py
applies it. The idea is from team gracious-tornado's public method description (no code copied).

  py -3.14 tools/last_day/c3_cap_at_D.py          # -> runs/last_day/c3_cap_at_D.json
"""
import numpy as np
import pandas as pd
import ld

a, b = ld.load_2025(), ld.load_2026()
res = {}
for d in (a, b):
    nd = ld.local_day(d.t, d.APT) > ld.local_day(d.t_sched, d.APT)
    d['bugdom'] = (d.APT == 'LIRF') & (d.lane == 'unmatched') & nd & (d.is_heli == 0)
for thr in (7200.0, 10800.0):
    out = {}
    for name, d in (('2025', a), ('2026', b)):
        m = (d.D > thr) & (d.p > d.D) & ~d.bugdom
        newp = np.where(m, ld.rules(d, np.where(m, d.D, d.p)), d.p)
        mv = newp - d.p.to_numpy()
        d[f'mv{int(thr)}'] = mv
        out[name] = {'rows': int((np.abs(mv) > 1e-6).sum()), 'sum_move2_M': round(float((mv ** 2).sum() / 1e6), 3),
                     'by_apt_lane': d[np.abs(mv) > 1e-6].groupby(['APT', 'lane']).size().to_dict(),
                     'by_month': d[np.abs(mv) > 1e-6].month.value_counts().sort_index().to_dict()}
        if name == '2025':
            y = d.y.to_numpy()
            mm = np.abs(mv) > 1e-6
            out[name]['labels_above_D'] = int((mm & (y > d.D.to_numpy() + 60)).sum())
            out[name]['labels_at_D_120'] = int((mm & (np.abs(y - d.D.to_numpy()) < 120)).sum())
            out[name]['labels_below_D'] = int((mm & (y < d.D.to_numpy() - 120)).sum())
            j = ld.judge(d, newp, newp, f'cap_at_D_{int(thr)}')
            out['judge'] = j
            out['rms'] = None
    out['rms_ratio'] = ld.rms_ratio(a, a[f'mv{int(thr)}'], b, b[f'mv{int(thr)}'])
    res[f'D>{int(thr)}'] = out
    print(int(thr), out)
# context: all 2025 rows D > 2 h outside the date-bug domain: how often is y > D?
m = (a.D > 7200) & ~a.bugdom
print('2025 D>2h non-bug-domain rows', int(m.sum()), 'y > D+60:', int((m & (a.y > a.D + 60)).sum()),
      'p > D:', int((m & (a.p > a.D)).sum()))
ld.save('c3_cap_at_D', res)
