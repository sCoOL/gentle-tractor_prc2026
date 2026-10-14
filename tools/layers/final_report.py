"""Report of a final-chain run: rows moved per step, the distance to its v45, and the mean / median prediction by airport
and month. Light (reads the cvNN files only).

  py -3.14 tools/layers/final_report.py A15        # -> runs/clean_chain_final/A15/report.json + report.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import prc_config  # noqa: E402

REB = prc_config.WORK
STEPS = ['49', '51', '52', '53', '54', '55', '56', '57', '63', '66', '67a', '68a', '68']


def ser(p):
    return pd.read_parquet(p).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)


def rms(x):
    x = np.asarray(x, float)
    return round(float(np.sqrt(np.mean(x ** 2))), 3) if len(x) else None


def main(v):
    H = REB / 'runs/clean_chain_final' / v
    args = json.loads((H / 'run_args.json').read_text())
    base = pd.read_parquet(prc_config.DATA / 'base_rank.parquet', columns=['MVT_ID_mvt', 'APT', 'month', 'fpl_matched']).set_index('MVT_ID_mvt')
    rep = {'variant': v, 'v45': args['v45'], 'l3_months': args['l3_months'], 'manual_rows': args.get('manual_rows')}
    prev = ser(args['v45'])
    mv = []
    for s in STEPS:
        f = ser(H / f'cv{s}.parquet')
        df = f - prev.reindex(f.index)
        mf = df.abs() > 1e-9
        mv.append({'step': s, 'rows_moved': int(mf.sum()), 'sum_sq_move_M': round(float((df ** 2).sum() / 1e6), 2),
                   **{f'rows_moved_month_{k}': int(g.sum()) for k, g in mf.groupby(base.month.reindex(f.index))}})
        prev = f
    rep['rows_moved'] = mv
    out = ser(H / f'gentle-tractor_final_v68_{v}.parquet')
    v45 = ser(args['v45']).reindex(out.index)
    m = base.month.reindex(out.index)
    dd = out - v45
    rep['final_vs_v45_rms_s'] = {'all': rms(dd), **{f'month_{k}': rms(g) for k, g in dd.groupby(m)},
                                 'rows_changed': int((dd.abs() > 1e-9).sum())}
    t = pd.DataFrame({'p': out, 'APT': base.APT.reindex(out.index), 'month': m})
    g = t.groupby(['APT', 'month']).p.agg(['mean', 'median']).round(0)
    rep['by_airport_month'] = {f'{a}|{mo}': [float(r['mean']), float(r['median'])] for (a, mo), r in g.iterrows()}
    rep['file'] = {'rows': int(len(out)), 'min': float(out.min()), 'max': float(out.max()), 'mean': float(out.mean())}
    (H / 'report.json').write_text(json.dumps(rep, indent=1))
    lines = [f'variant {v}: v45 {Path(args["v45"]).name}, L3 months {args["l3_months"]}, manual rows {args.get("manual_rows")}', '',
             '| step | rows moved | sum sq move (M s^2) |', '|---|---|---|']
    for r in mv:
        lines.append(f"| {r['step']} | {r['rows_moved']:,} | {r['sum_sq_move_M']} |")
    lines += ['', f"final vs its v45 (RMS s): {rep['final_vs_v45_rms_s']}", '',
              '| airport | Jan | Feb | Jun | Jul |', '|---|---|---|---|---|']
    for a in sorted(t.APT.dropna().unique()):
        cells = [f"{int(g.loc[(a, mo), 'mean'])}/{int(g.loc[(a, mo), 'median'])}" if (a, mo) in g.index else '-' for mo in (1, 2, 6, 7)]
        lines.append(f"| {a} | " + ' | '.join(cells) + ' |')
    (H / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1])
