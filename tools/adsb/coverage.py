"""Coverage of the rebuilt ADS-B / OPDI tables on the FINAL ranking departures, by month x airport.

  py -3.14 tools/adsb/coverage.py            # -> logs/adsb/coverage_by_airport.csv (+ printed tables)

Columns (share of the final-ranking departures of that month and airport):
  v1        a row in adsb/adsb_features.parquet (121 match: callsign or take-off time)
  v2        a row in adsb/adsb_features_v2.parquet (255 normalised match)
  v2_stand  v2 match AND adsb_at_stand == 1 AND adsb_taxi_leave known (usable ADS-B off-block)
  push      push_phases row with a measured push-back (t_push_end known)
  move      push_phases row with a first real move (t_move known)
  times_ob  adsb_times.parquet off-block known (v1 match)
  opdi      opdi/opdi_movements.parquet linked (flight_id known)
  tn_link   runs/v51_turnaround/links.parquet has an inbound arrival for it
Days of the month with no complete points file are listed separately (logs/adsb/coverage_missing_days.csv).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import prc_config  # noqa: E402

REB = prc_config.WORK
ID = 'MVT_ID_mvt'


def main():
    rk = pd.read_parquet(prc_config.DATA / 'ranking.parquet', columns=[ID, 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt'])
    d = rk[rk.PHASE_mvt == 'DEP'].copy()
    d['month'] = d.MVT_TIME_UTC_mvt.dt.strftime('%m')
    d['date'] = d.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d')
    d['APT'] = d.ADEP_mvt.astype(str)

    def has(path, cols=None, cond=None, key=ID):
        if not path.exists():
            return np.full(len(d), np.nan)
        t = pd.read_parquet(path, columns=[key] + (cols or []))
        if cond is not None:
            t = t[cond(t)]
        return d[ID].isin(set(t[key].tolist())).to_numpy().astype(float)

    A = prc_config.ADSB_DIR
    d['v1'] = has(A / 'adsb_features.parquet')
    d['v2'] = has(A / 'adsb_features_v2.parquet')
    d['v2_stand'] = has(A / 'adsb_features_v2.parquet', ['adsb_at_stand', 'adsb_taxi_leave'],
                        lambda t: (t.adsb_at_stand == 1) & t.adsb_taxi_leave.notna())
    d['push'] = has(A / 'push_phases.parquet', ['t_push_end'], lambda t: t.t_push_end.notna())
    d['move'] = has(A / 'push_phases.parquet', ['t_move'], lambda t: t.t_move.notna())
    d['times_ob'] = has(A / 'adsb_times.parquet', ['adsb_offblock_t'], lambda t: t.adsb_offblock_t.notna())
    d['opdi'] = has(prc_config.OPDI_DIR / 'opdi_movements.parquet', ['flight_id'], lambda t: t.flight_id.notna())
    d['tn_link'] = has(prc_config.RUNS / 'v51_turnaround/links.parquet', key='dep_mvt')
    cols = ['v1', 'v2', 'v2_stand', 'push', 'move', 'times_ob', 'opdi', 'tn_link']
    g = d.groupby(['month', 'APT'])[cols].mean().round(3)
    g.insert(0, 'deps', d.groupby(['month', 'APT']).size())
    tot = d.groupby('month')[cols].mean().round(3)
    tot.insert(0, 'deps', d.groupby('month').size())
    tot['APT'] = 'ALL'
    out = pd.concat([g.reset_index(), tot.reset_index()], ignore_index=True).sort_values(['month', 'APT'])
    out.to_csv(REB / 'logs/adsb/coverage_by_airport.csv', index=False)
    # days of the final file without an events file (no complete points file when the run started)
    ev = {p.name[7:17] for p in (A / 'events').glob('events_*.parquet')}
    miss = sorted(set(d.date) - ev)
    pd.DataFrame({'date': miss, 'departures': [int((d.date == x).sum()) for x in miss]}).to_csv(
        REB / 'logs/adsb/coverage_missing_days.csv', index=False)
    pd.set_option('display.width', 200)
    for c in ['v2', 'v2_stand', 'push', 'opdi', 'tn_link']:
        print(f'\n{c}: share of final departures, month x airport')
        print(pd.concat([g.reset_index(), tot.reset_index()]).pivot_table(index='APT', columns='month', values=c).to_string())
    print(f'\nfinal-file days without events: {len(miss)} {miss[:5]}{"..." if len(miss) > 5 else ""}')


if __name__ == '__main__':
    main()
