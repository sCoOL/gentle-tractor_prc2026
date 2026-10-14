"""A15L = A15 + two fixed rules, each through rules_v55:
  c0  the 8-13 h rule (c0_long_delay_rule.py) on every final row it selects, rows of the manual step untouched;
  c3  p <= D for D > 2 h outside the LIRF date-bug domain (c3_cap_at_D.py).

  py -3.14 tools/last_day/build_A15L.py          # -> runs/clean_chain_final/A15L/, log runs/last_day/build_A15L.json

`write` is the output check of every post-step: 670,790 rows in the template order of data/submitting.parquet,
unique ids, finite, >= 30 s, and the rows of the chain's manual step equal to A15.
"""
import hashlib

import numpy as np
import pandas as pd
import ld

OWN = ld.manual_rows()
TPL = pd.read_parquet(ld.DATA / 'submitting.parquet', columns=['MVT_ID_mvt'])


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(tag: str, s: pd.Series, a15: pd.Series, info: dict):
    out_dir = ld.FIN / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    f = pd.DataFrame({'MVT_ID_mvt': TPL.MVT_ID_mvt.to_numpy(), 'TAXITIME_SEC_mvt': s.reindex(TPL.MVT_ID_mvt).to_numpy()})
    x = f.TAXITIME_SEC_mvt.to_numpy()
    assert len(f) == 670790 and f.MVT_ID_mvt.is_unique and (f.MVT_ID_mvt.to_numpy() == TPL.MVT_ID_mvt.to_numpy()).all()
    assert np.isfinite(x).all() and x.min() >= 30
    own = s.reindex(OWN.MVT_ID_mvt).to_numpy()
    assert np.array_equal(own, a15.reindex(OWN.MVT_ID_mvt).to_numpy()), 'a manual row moved'
    p = out_dir / f'gentle-tractor_final_v68_{tag}.parquet'
    f.to_parquet(p, index=False)
    d = x - a15.reindex(TPL.MVT_ID_mvt).to_numpy()
    info[tag] = {'file': str(p), 'sha256': sha(p), 'rows': int(len(f)), 'min': float(x.min()), 'max': float(x.max()),
                 'mean': float(x.mean()), 'finite': True, 'order_final_submitting': True, 'manual_rows_exact': True,
                 'rows_moved_vs_A15': int((np.abs(d) > 1e-9).sum()), 'sum_move2_vs_A15_M': round(float((d ** 2).sum() / 1e6), 4)}
    print(tag, info[tag], flush=True)


def main():
    info = {}
    a15 = pd.read_parquet(ld.A15_FILE).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)
    d = ld.load_2026().set_index('MVT_ID_mvt', drop=False)
    own = d.MVT_ID_mvt.isin(OWN.MVT_ID_mvt).to_numpy()
    s = a15.reindex(d.index).copy()
    # c0: the 8-13 h rule (rows from c0_long_delay_rule.py, rows of the manual step excluded)
    c0 = pd.read_parquet(ld.OUT / 'c0_rows.parquet')
    c0 = c0[~c0.manual_row]
    ids0 = c0.MVT_ID_mvt.to_numpy()
    new0 = ld.rules(d.loc[ids0], c0.D.to_numpy())
    info['c0'] = {'ids': ids0.tolist(), 'D': c0.D.tolist(), 'a15': s.loc[ids0].tolist(), 'new': new0.tolist()}
    s.loc[ids0] = new0
    # c3: p <= D for D > 2 h outside the LIRF date-bug domain (not manual rows, not c0 rows)
    nd = ld.local_day(d.t, d.APT) > ld.local_day(d.t_sched, d.APT)
    bug = ((d.APT == 'LIRF') & (d.lane == 'unmatched') & (d.is_heli == 0)).to_numpy() & nd
    m3 = (d.D.to_numpy() > 7200) & (s.to_numpy() > d.D.to_numpy()) & ~bug & ~own & ~d.MVT_ID_mvt.isin(ids0).to_numpy()
    ids3 = d.MVT_ID_mvt.to_numpy()[m3]
    new3 = ld.rules(d.loc[ids3], d.D.to_numpy()[m3])
    info['c3'] = {'ids': ids3.tolist(), 'D': d.D.to_numpy()[m3].tolist(), 'a15': s.loc[ids3].tolist(), 'new': new3.tolist()}
    s.loc[ids3] = new3
    write('A15L', s, a15, info)
    ld.save('build_A15L', info)


if __name__ == '__main__':
    main()
