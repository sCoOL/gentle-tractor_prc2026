"""Month-edge check (continuity rule: windows / links use the neighbouring days present in the data).

  py -3.14 tools/adsb/edge_check.py

1. adsb_nbr `adsb_nb_tko_c60` (ADS-B take-offs within +-30 min, own aircraft excluded) on final-file departures within
   30 min of a month edge: recount it from the events of ALL days and from the row's own day only; the table must agree
   with the all-days count. Edges: 01-31/02-01, 02-28/03-01, 05-31/06-01, 06-30/07-01.
2. runs/v51_turnaround/links.parquet: departures of the first day of a month linked to an arrival of the previous day.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import prc_config  # noqa: E402
REB = prc_config.WORK
e = importlib.import_module('121_adsb_events')
EDGES = [('2026-01-31', '2026-02-01'), ('2026-02-28', '2026-03-01'), ('2026-05-31', '2026-06-01'), ('2026-06-30', '2026-07-01')]


def main():
    nb = pd.read_parquet(prc_config.ADSB_DIR / 'adsb_nbr.parquet', columns=['MVT_ID_mvt', 'adsb_nb_tko_c60']).set_index('MVT_ID_mvt')
    rows = []
    for a, b in EDGES:
        E = []
        for d in (a, b):
            f = prc_config.ADSB_DIR / 'events' / f'events_{d}.parquet'
            if f.exists():
                ev, mt = e.match_day(d)
                ev = ev.assign(date=d)
                E.append((d, ev, mt))
        if not E:
            continue
        edge = pd.Timestamp(b, tz='UTC').timestamp()
        allev = pd.concat([x[1] for x in E], ignore_index=True)
        for d, ev, mt in E:
            if mt is None or len(mt) == 0 or 'MVT_ID_mvt' not in mt:
                continue
            dep = e.departures(d)
            dep = dep[(dep.t_mvt - edge).abs() <= 1800].merge(mt[['MVT_ID_mvt', 'k']], on='MVT_ID_mvt', how='left')
            for r in dep.itertuples():
                w = allev[(allev.APT == r.APT) & ((allev.t_tko - r.t_mvt).abs() <= 1800)]
                own = 1 if (r.k == r.k) else 0                     # matched rows exclude their own take-off
                n_all = len(w) - own
                n_own_day = int(((w.date == d)).sum()) - own
                tab = nb.adsb_nb_tko_c60.get(r.MVT_ID_mvt, np.nan)
                rows.append(dict(edge=f'{a}|{b}', date=d, APT=r.APT, n_all_days=n_all, n_own_day=n_own_day, table=tab))
    x = pd.DataFrame(rows)
    x['agrees_all_days'] = x.table == x.n_all_days
    x['agrees_own_day_only'] = x.table == x.n_own_day
    x['crosses'] = x.n_all_days != x.n_own_day
    s = x.groupby('edge').agg(rows=('table', 'size'), rows_with_other_day_neighbours=('crosses', 'sum'),
                              table_eq_all_days=('agrees_all_days', 'mean'), table_eq_own_day_only=('agrees_own_day_only', 'mean'))
    print('1. adsb_nb_tko_c60 near month edges (departures within 30 min of the edge)')
    print(s.round(4).to_string())
    x.to_csv(REB / 'logs/adsb/edge_check_nbr.csv', index=False)
    # 2. turnaround links across month edges
    lk = pd.read_parquet(prc_config.RUNS / 'v51_turnaround/links.parquet')
    rk = pd.read_parquet(prc_config.DATA / 'ranking.parquet', columns=['MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt'])
    day = rk.set_index('MVT_ID_mvt').MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d')
    lk['dep_day'] = day.reindex(lk.dep_mvt).to_numpy()
    lk['arr_day'] = day.reindex(lk.arr_mvt).to_numpy()
    print('\n2. turnaround links whose arrival is on the previous day of a month edge')
    for a, b in EDGES:
        m = (lk.dep_day == b) & (lk.arr_day == a)
        print(f'  {b} departures linked to {a} arrivals: {int(m.sum()):,}  '
              f'(of {int((lk.dep_day == b).sum()):,} links of {b}); by source {lk[m].link_source.value_counts().to_dict()}')


if __name__ == '__main__':
    main()
