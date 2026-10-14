"""A15LXR = A15LX (A15L + the combined daily layer) followed by the re-centred LIRF copy / held enumeration computed on
the stack output (c16_stack_recentred.py).

  py -3.14 tools/last_day/build_A15LXR.py --idea4 0     # check: must reproduce the A15LX final file (nothing written)
  py -3.14 tools/last_day/build_A15LXR.py --idea4 1     # -> runs/clean_chain_final/A15LXR/, log runs/last_day/build_A15LXR.json

Everything fitted on all 2025 IFR non-heli rows; the rows of the manual step and A15L's rule rows are fixed. Guards:
(1) whole layer, 1.5x per airport-month vs this variant's 2025 six-fold out-of-fold move (as build_A15LX); (2) increment
guard: (result - A15LX) vs 2025 (c16 OOF p6 - stack OOF p6_stack), 1.5x per airport-month, rules_v55 on rescaled rows,
fixed rows reset. build_A15L.write checks.
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
import c16_stack_recentred as c16  # noqa: E402
import c10_daily_v2 as v2  # noqa: E402
from c9_step1_identity import ec  # noqa: E402
from build_A15L import OWN, write  # noqa: E402
from build_A15LX import final, cmp  # noqa: E402

FIN = ld.FIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idea4', type=int, required=True)
    ap.add_argument('--oof', default='c16_oof_x.parquet')
    ap.add_argument('--tag', default='A15LXR')
    x = ap.parse_args()
    out = {'idea4': bool(x.idea4)}
    a15 = pd.read_parquet(ld.A15_FILE).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)
    d = ld.load_2026().set_index('MVT_ID_mvt', drop=False)
    own = d.MVT_ID_mvt.isin(OWN.MVT_ID_mvt).to_numpy()
    s = pd.Series(final('A15L', d.index), index=d.index)
    fixed_all = own | (s.to_numpy() != a15.reindex(d.index).to_numpy())
    EA25, EA26 = v2.ec_arrays(ec(2025)), v2.ec_arrays(ec(2026))
    a, k25 = c15.load(2025, True)
    C = v2.prep(a[k25].reset_index(drop=True))
    k26 = c14.ifr_keep(d.reset_index(drop=True), 'base_rank.parquet')
    apt = d.APT.astype(str).to_numpy()
    o = pd.read_parquet(ld.OUT / x.oof)
    assert (o.MVT_ID_mvt.to_numpy() == a.MVT_ID_mvt.to_numpy()).all()
    gp6 = (o.p6 if x.idea4 else o.p6_stack).to_numpy()
    # ---- stack (+ 4R) on 2026, as build_A15LX.make
    dl = d.copy()
    dl['p'] = s.to_numpy()
    dk = dl[k26].reset_index(drop=True)
    Cb = v2.prep(dk)
    fx = fixed_all[k26]
    diag = []
    pt3, pt4, el, info, P = c16.split16(C, Cb, EA25, fixed=fx, EA_te=EA26, do4=bool(x.idea4), diag=diag)
    out['fit'] = info
    out['offsets_2026'] = diag[-1] if diag else None
    pk = (pt4 if x.idea4 else pt3).copy()
    pk[fx] = dk.p.to_numpy()[fx]
    mv = np.zeros(len(dl))
    mv[k26] = pk - dk.p.to_numpy()
    mv[fixed_all] = 0.0
    mv25 = gp6 - a.p.to_numpy()
    r25 = np.sqrt(pd.Series(mv25 ** 2).groupby([a.APT.to_numpy(), a.month.to_numpy()]).mean())
    r26 = pd.Series(mv ** 2).groupby([dl.APT.to_numpy(), dl.month.to_numpy()]).mean() ** 0.5
    fac = np.minimum(1.0, 1.5 * r25.reindex(r26.index) / r26.replace(0, np.nan)).fillna(1.0)
    kf = fac.reindex(pd.MultiIndex.from_arrays([dl.APT.to_numpy(), dl.month.to_numpy()])).to_numpy()
    pg = dl.p.to_numpy() + mv * kf
    ch = np.abs(mv * kf) > 1e-9
    pg[ch] = ld.rules(dl[ch], pg[ch])
    pg[fixed_all] = dl.p.to_numpy()[fixed_all]
    out['whole_layer_guard_below_1'] = {f'{k_[0]}-{k_[1]}': round(float(v), 3) for k_, v in fac[fac < 1].items()}
    lx = final('A15LX', d.index)
    if not x.idea4:
        out['step0_repro_A15LX'] = {'final_max_abs': float(np.abs(pg - lx).max())}
        print('STEP0', out['step0_repro_A15LX'], flush=True)
        assert max(out['step0_repro_A15LX'].values()) < 1e-6
        ld.save(f'build_{x.tag}_step0', out)
        return
    # ---- increment guard over A15LX (c13 / c10_guard_increment form)
    inc25 = o.p6.to_numpy() - o.p6_stack.to_numpy()
    inc26 = pg - lx
    r25 = np.sqrt(pd.Series(inc25 ** 2).groupby([a.APT.to_numpy(), a.month.to_numpy()]).mean())
    r26 = np.sqrt(pd.Series(inc26 ** 2).groupby([d.APT.to_numpy(), d.month.to_numpy()]).mean())
    t = pd.DataFrame({'rms25': r25.reindex(r26.index), 'rms26': r26})
    t['ratio'] = np.where(t.rms26 > 0, t.rms26 / t.rms25.replace(0, np.nan), 0.0)
    fi = np.minimum(1.0, 1.5 / t.ratio.replace(0, np.nan)).fillna(1.0)
    k = fi.reindex(pd.MultiIndex.from_arrays([d.APT.to_numpy(), d.month.to_numpy()])).to_numpy()
    res = lx + inc26 * k
    chg = (np.abs(inc26) > 1e-9) & (k < 1)
    if chg.any():
        res[chg] = ld.rules(d[chg], res[chg])
    res[fixed_all] = s.to_numpy()[fixed_all]
    out['increment_guard'] = {'ratio_cells': {f'{a_}-{m:02d}': {'rms25': round(float(r.rms25), 2), 'rms26': round(float(r.rms26), 2),
                                                                'ratio': round(float(r.ratio), 3)} for (a_, m), r in t[t.rms26 > 0].iterrows()},
                              'scale_below_1': {f'{a_}-{m:02d}': round(float(v), 3) for (a_, m), v in fi[fi < 1].items()},
                              'rows_rescaled': int(chg.sum())}
    inc = np.zeros(len(d))
    inc[k26] = np.where(fx, 0.0, pt4 - pt3)
    out['idea4_2026_unguarded'] = {'eligible': int(el.sum()), 'rows_changed_1s': int((np.abs(inc) > 1).sum()),
                                   'sum_s': round(float(inc.sum()), 1), 'sum_sq_M': round(float((inc ** 2).sum() / 1e6), 2),
                                   'by_month_rows_1s': {int(m): int(v) for m, v in pd.Series(np.abs(inc) > 1).groupby(d.month.to_numpy()).sum().items() if v}}
    print(json.dumps({k_: out[k_] for k_ in ('fit', 'offsets_2026', 'whole_layer_guard_below_1', 'idea4_2026_unguarded')}, default=str), flush=True)
    print('INC_GUARD', json.dumps({k_: v for k_, v in out['increment_guard'].items() if k_ != 'ratio_cells'}),
          json.dumps({k_: v for k_, v in out['increment_guard']['ratio_cells'].items() if k_.startswith('LIRF')}), flush=True)
    out['vs_A15LX'] = cmp(res, lx, apt)
    print('VS A15LX', json.dumps(out['vs_A15LX']), flush=True)
    write(x.tag, pd.Series(res, index=d.index), a15, out)
    ld.save(f'build_{x.tag}', out)


if __name__ == '__main__':
    main()
