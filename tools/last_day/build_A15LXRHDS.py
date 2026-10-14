"""A15LXRHDS = A15LXRH + the LIRF date-bug / schedule-copy rules (c19_lirf_datebug_rules.py). This is the final file.

  py -3.14 tools/last_day/build_A15LXRHDS.py    # -> runs/clean_chain_final/A15LXRHDS/, log runs/last_day/build_A15LXRHDS.json

On the 2026 rows of the date-bug domain (LIRF, NM-unmatched, fixed-wing, take-off on a later local day): a row a rule
decides takes D (copy) or B = k * 86,400 + N (date bug). A row moves only when its new value differs from A15LXRH by more than 0.5 s. Rows of the chain's manual step never move.
build_A15L.write checks.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ld  # noqa: E402
import c19_lirf_datebug_rules as c19  # noqa: E402
from build_A15L import OWN, write  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='A15LXRH')
    ap.add_argument('--tag', default='A15LXRHDS')
    x = ap.parse_args()
    a15 = pd.read_parquet(ld.A15_FILE).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)
    base = pd.read_parquet(ld.FIN / x.base / f'gentle-tractor_final_v68_{x.base}.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)
    d = c19.decide(c19.features(2026))
    manual = set(OWN.MVT_ID_mvt)
    cfg = Path(__file__).resolve().parents[2] / 'config' / 'manual_rows.csv'
    if cfg.exists():                                   # belt and braces: the table itself, if the run's copy is missing
        manual |= set(pd.read_csv(cfg).MVT_ID_mvt)
    d = d[d.side.notna() & ~d.MVT_ID_mvt.isin(manual)].copy()
    d['before'] = base.reindex(d.MVT_ID_mvt).to_numpy()
    d = d[np.abs(d.value - d.before) > 0.5]
    s = base.copy()
    s.loc[d.MVT_ID_mvt.to_numpy()] = d.value.to_numpy()
    out = {'base': x.base,
           'rows': d[['MVT_ID_mvt', 'FLIGHT_mvt', 'Dh', 'sched_clock', 'takeoff_clock', 'turn_h', 'rule', 'side', 'before', 'value']]
           .round(2).astype({'MVT_ID_mvt': int}).to_dict('records')}
    print(d[['MVT_ID_mvt', 'FLIGHT_mvt', 'Dh', 'rule', 'side', 'before', 'value']].round(2).to_string(index=False), flush=True)
    write(x.tag, s, a15, out)
    ld.save(f'build_{x.tag}', out)


if __name__ == '__main__':
    main()
