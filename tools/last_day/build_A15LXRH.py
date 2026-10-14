"""A15LXRH = A15LXR + the hardened midnight-crossers step (c18_crossers_hardened.py). This is the final file.

  py -3.14 tools/last_day/build_A15LXRH.py      # -> runs/clean_chain_final/A15LXRH/, log runs/last_day/build_A15LXRH.json

Everything fitted on all 2025 IFR non-heli rows (stack + enumeration exactly as c16 / build_A15LXR; step configuration by
training SSE at the training p4). Applied to the A15LXR final values. Fixed: the rows of the manual step, A15L's rule
rows, the 2026 rows the enumeration set. Increment guard (as build_A15LXR): (result - A15LXR) vs 2025 (c18 OOF p6 - its
base p6), 1.5x per airport-month, rules_v55 on rescaled rows, fixed rows reset. build_A15L.write checks.
Check before the step: the recomputed stack + enumeration equals A15LXR on every airport-month its guards left alone
(read from runs/last_day/build_A15LXR.json).
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
import c17_crossers as c17  # noqa: E402
import c18_crossers_hardened as c18  # noqa: E402
import c10_daily_v2 as v2  # noqa: E402
from c9_step1_identity import ec  # noqa: E402
from build_A15L import OWN, write  # noqa: E402
from build_A15LX import final, cmp  # noqa: E402

FIN = ld.FIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof', default='c18_oof_h.parquet')
    ap.add_argument('--tag', default='A15LXRH')
    x = ap.parse_args()
    out = {}
    a15 = pd.read_parquet(ld.A15_FILE).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)
    d = ld.load_2026().set_index('MVT_ID_mvt', drop=False)
    own = d.MVT_ID_mvt.isin(OWN.MVT_ID_mvt).to_numpy()
    s = pd.Series(final('A15L', d.index), index=d.index)
    fixed_all = own | (s.to_numpy() != a15.reindex(d.index).to_numpy())
    lxr = final('A15LXR', d.index)
    EA25, EA26 = v2.ec_arrays(ec(2025)), v2.ec_arrays(ec(2026))
    a, k25 = c15.load(2025, True)
    C = c17.mk(a[k25].reset_index(drop=True), 'train')
    k26 = c14.ifr_keep(d.reset_index(drop=True), 'base_rank.parquet')
    apt = d.APT.astype(str).to_numpy()
    fx = fixed_all[k26]
    # ---- stack + idea 4 fit on all 2025; 2026 idea-4 eligible rows from A15L (as build_A15LXR)
    dl = d.copy()
    dl['p'] = s.to_numpy()
    b = c17.stack4(C, EA25, v2.prep(dl[k26].reset_index(drop=True)), EA26, fx)
    out['stack'] = {'passes': b['passes'], 'lam': b['lam'], 'lam4_idea4': b['lam4_idea4'], 'idea4_eligible_2026': int(b['elt'].sum())}
    # step 0: the recomputed stack + enumeration reproduces A15LXR on the airport-months its guards left alone
    bx = json.loads((ld.OUT / 'build_A15LXR.json').read_text())
    guarded = set(bx['whole_layer_guard_below_1']) | {f'{k_.split("-")[0]}-{int(k_.split("-")[1])}' for k_ in bx['increment_guard']['scale_below_1']}
    pre = np.where(fx, dl.p.to_numpy()[k26], b['pt4'])
    ok_cells = ~pd.Series(dl.APT.to_numpy()[k26] + '-' + dl.month.astype(str).to_numpy()[k26]).isin(sorted(guarded)).to_numpy()
    out['step0_guarded_cells_A15LXR'] = sorted(guarded)
    out['step0_unguarded_cells_max_abs_vs_A15LXR'] = float(np.abs(pre - lxr[k26])[ok_cells].max())
    print('STEP0', out['stack'], out['step0_unguarded_cells_max_abs_vs_A15LXR'], flush=True)
    assert out['stack']['idea4_eligible_2026'] == bx['idea4_2026_unguarded']['eligible'] and out['step0_unguarded_cells_max_abs_vs_A15LXR'] < 1e-6
    # ---- step fit on all 2025 at the training p4
    A, R, best, info = c18.fit_step(C, b['ptr4'], b['P'], EA25, b['eltr'])
    out['step_fit'] = {k: v for k, v in info.items() if k != 'train_sse_by_cfg_M'}
    print('FIT', json.dumps({k: info[k] for k in ('tau', 'k', 'alpha', 'c_v', 'lam4', 'train_dsse_best_M', 'train_dsse_best_by_tau_M')}), flush=True)
    # ---- apply to the A15LXR final values
    d2 = d.copy()
    d2['p'] = lxr
    Cb = c17.mk(d2[k26].reset_index(drop=True), 'rank')
    frozen = fx | b['elt']
    pt, el, kd, st = c18.apply_step(Cb, Cb['p0'], b['P'], EA26, A, R, best, frozen, fx)
    pt[frozen] = Cb['p0'][frozen]
    out['step_2026'] = {'eligible': int(el.sum()), 'eligible_long': int((el & (kd == 1)).sum()), 'eligible_amb': int((el & (kd == 2)).sum()), 'eligible_demoted': int((el & (kd == 3)).sum()),
                        'stats': st}
    inc26 = np.zeros(len(d))
    inc26[k26] = pt - Cb['p0']
    inc26[fixed_all] = 0.0
    out['step_2026']['rows_changed_1s'] = int((np.abs(inc26) > 1).sum())
    out['step_2026']['sum_s'] = round(float(inc26.sum()), 1)
    out['step_2026']['sum_sq_M'] = round(float((inc26 ** 2).sum() / 1e6), 3)
    # ---- increment guard over A15LXR
    o = pd.read_parquet(ld.OUT / x.oof)
    assert (o.MVT_ID_mvt.to_numpy() == a.MVT_ID_mvt.to_numpy()).all()
    inc25 = o.p6.to_numpy() - o.p6_base.to_numpy()
    r25 = np.sqrt(pd.Series(inc25 ** 2).groupby([a.APT.to_numpy(), a.month.to_numpy()]).mean())
    r26 = np.sqrt(pd.Series(inc26 ** 2).groupby([d.APT.to_numpy(), d.month.to_numpy()]).mean())
    t = pd.DataFrame({'rms25': r25.reindex(r26.index).fillna(0.0), 'rms26': r26})
    t['ratio'] = np.where(t.rms26 > 0, np.where(t.rms25 > 0, t.rms26 / t.rms25.where(t.rms25 > 0, 1), np.inf), 0.0)
    fi = pd.Series(np.where(t.ratio > 1.5, 1.5 / t.ratio, 1.0), index=t.index)
    k = fi.reindex(pd.MultiIndex.from_arrays([d.APT.to_numpy(), d.month.to_numpy()])).to_numpy()
    res = lxr + inc26 * k
    chg = (np.abs(inc26) > 1e-9) & (k < 1)
    if chg.any():
        res[chg] = ld.rules(d[chg], res[chg])
    res[fixed_all] = s.to_numpy()[fixed_all]
    out['increment_guard'] = {'ratio_cells': {f'{a_}-{m:02d}': {'rms25': round(float(r.rms25), 2), 'rms26': round(float(r.rms26), 2),
                                                                'ratio': round(float(r.ratio), 3)} for (a_, m), r in t[t.rms26 > 0].iterrows()},
                              'scale_below_1': {f'{a_}-{m:02d}': round(float(v), 4) for (a_, m), v in fi[fi < 1].items()},
                              'rows_rescaled': int(chg.sum())}
    print('STEP2026', json.dumps({k_: v for k_, v in out['step_2026'].items() if k_ != 'stats'}), flush=True)
    print('INC_GUARD', json.dumps(out['increment_guard']), flush=True)
    out['vs_A15LXR'] = cmp(res, lxr, apt)
    out['vs_A15LXR']['by_month_rows_1s'] = {int(m): int(v) for m, v in pd.Series(np.abs(res - lxr) > 1).groupby(d.month.to_numpy()).sum().items() if v}
    print('VS A15LXR', json.dumps(out['vs_A15LXR']), flush=True)
    kind = np.zeros(len(d), np.int8)
    kind[np.where(k26)[0]] = kd
    big = np.abs(res - lxr) > 600
    rows = d.loc[big, ['MVT_ID_mvt', 'FLIGHT_mvt', 'APT', 'month', 'lane', 'st', 'D', 't', 't_sched']].copy()
    rows['takeoff_utc'] = pd.to_datetime(rows.t, unit='s').astype(str)
    rows['sched_utc'] = pd.to_datetime(rows.t_sched, unit='s').astype(str)
    rows = rows.drop(columns=['t', 't_sched'])
    rows['A15LXR'], rows['A15LXRH'] = lxr[big], res[big]
    rows['move'] = res[big] - lxr[big]
    rows['kind'] = np.where(kind[big] == 1, 'long', np.where(kind[big] == 2, 'amb', np.where(kind[big] == 3, 'demoted', '-')))
    out['rows_over_600s'] = rows.round(1).astype({'MVT_ID_mvt': int}).to_dict('records')
    print('rows > 600 s:', int(big.sum()), flush=True)
    print(rows.round(1).to_string(index=False), flush=True)
    write(x.tag, pd.Series(res, index=d.index), a15, out)
    ld.save(f'build_{x.tag}', out)


if __name__ == '__main__':
    main()
