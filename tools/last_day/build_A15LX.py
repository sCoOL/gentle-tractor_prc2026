"""A15LX = A15L + the combined EUROCONTROL daily layer (c15_combined.py): up to three passes + the IFR / non-heli
identity + the count-gap target K_adj recomputed in every pass.

  py -3.14 tools/last_day/build_A15LX.py          # -> runs/clean_chain_final/A15LX/, log runs/last_day/build_A15LX.json

Everything fitted on all 2025 IFR non-heli rows; the rows of the manual step and A15L's rule rows are fixed. Guard: the
move of each airport-month is scaled down to <= 1.5x the 2025 six-fold out-of-fold move RMS of the same airport-month
(c15_oof_x.parquet, p6_stack), then rules_v55 on the rescaled rows. build_A15L.write checks.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ld  # noqa: E402
import c14_count_gap as c14  # noqa: E402
import c15_combined as c15  # noqa: E402
from c10_daily_v2 import ec_arrays  # noqa: E402
from c9_step1_identity import ec  # noqa: E402
from build_A15L import OWN, write  # noqa: E402

FIN = ld.FIN


def final(tag, idx):
    return pd.read_parquet(FIN / tag / f'gentle-tractor_final_v68_{tag}.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt \
        .astype(float).reindex(idx).to_numpy()


def cmp(pg, ref, apt):
    dd = pg - ref
    return {'rows_differ': int((np.abs(dd) > 1e-9).sum()), 'rows_differ_1s': int((np.abs(dd) > 1).sum()),
            'sum_s': round(float(dd.sum()), 1), 'sum_sq_M': round(float((dd ** 2).sum() / 1e6), 3),
            'by_apt': {x: {'rows_1s': int((np.abs(dd[apt == x]) > 1).sum()), 'sum_s': round(float(dd[apt == x].sum()), 1),
                           'sum_sq_M': round(float((dd[apt == x] ** 2).sum() / 1e6), 3)} for x in sorted(set(apt))}}


def make(cfg, guard_p6, a, C, d, s, fixed_all, k26, EA25, EA26):
    dl = d.copy()
    dl['p'] = s.to_numpy()
    dk = dl[k26].reset_index(drop=True)
    Cb = c15.mk(dk)
    fx = fixed_all[k26]
    pt3, info, P = c15.split_run(C, Cb, EA25, cfg, fixed=fx, EA_te=EA26)
    pk = pt3.copy()
    pk[fx] = dk.p.to_numpy()[fx]
    mv = np.zeros(len(dl))
    mv[k26] = pk - dk.p.to_numpy()
    mv[fixed_all] = 0.0
    mv25 = guard_p6 - a.p.to_numpy()
    r25 = np.sqrt(pd.Series(mv25 ** 2).groupby([a.APT.to_numpy(), a.month.to_numpy()]).mean())
    r26 = pd.Series(mv ** 2).groupby([dl.APT.to_numpy(), dl.month.to_numpy()]).mean() ** 0.5
    fac = np.minimum(1.0, 1.5 * r25.reindex(r26.index) / r26.replace(0, np.nan)).fillna(1.0)
    kf = fac.reindex(pd.MultiIndex.from_arrays([dl.APT.to_numpy(), dl.month.to_numpy()])).to_numpy()
    pg = dl.p.to_numpy() + mv * kf
    ch = np.abs(mv * kf) > 1e-9
    pg[ch] = ld.rules(dl[ch], pg[ch])
    pg[fixed_all] = dl.p.to_numpy()[fixed_all]
    info = {k: v for k, v in info.items() if k != 'stack_log'}
    info['guard_factors_below_1'] = {f'{x}-{mm}': round(float(v), 3) for (x, mm), v in fac[fac < 1].items()}
    info['unguarded_sum_move2_M'] = round(float((mv ** 2).sum() / 1e6), 2)
    info['sum_move2_vs_A15L_M'] = round(float(((pg - dl.p.to_numpy()) ** 2).sum() / 1e6), 2)
    return pg, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof', default='c15_oof_x.parquet')
    ap.add_argument('--tag', default='A15LX')
    x = ap.parse_args()
    out = {}
    a15 = pd.read_parquet(ld.A15_FILE).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)
    d = ld.load_2026().set_index('MVT_ID_mvt', drop=False)
    own = d.MVT_ID_mvt.isin(OWN.MVT_ID_mvt).to_numpy()
    s = pd.Series(final('A15L', d.index), index=d.index)
    fixed_all = own | (s.to_numpy() != a15.reindex(d.index).to_numpy())
    EA25, EA26 = ec_arrays(ec(2025)), ec_arrays(ec(2026))
    a, k25 = c15.load(2025, True)
    C = c15.mk(a[k25].reset_index(drop=True))
    k26 = c14.ifr_keep(d.reset_index(drop=True), 'base_rank.parquet')
    apt = d.APT.astype(str).to_numpy()
    o = pd.read_parquet(ld.OUT / x.oof)
    assert (o.MVT_ID_mvt.to_numpy() == a.MVT_ID_mvt.to_numpy()).all()
    pg, out['fit'] = make({'passes': [1, 2, 3], 'mu': True}, o.p6_stack.to_numpy(), a, C, d, s, fixed_all, k26, EA25, EA26)
    out['components'] = 'passes + IFR + count-gap'
    out['vs_A15L'] = cmp(pg, s.to_numpy(), apt)
    print(json.dumps(out, default=str)[:4000], flush=True)
    write(x.tag, pd.Series(pg, index=d.index), a15, out)
    ld.save(f'build_{x.tag}', out)


if __name__ == '__main__':
    main()
