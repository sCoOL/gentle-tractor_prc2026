"""features v4 (G10 stand timeline) under the three-expert tail: strict frames, diagnostic, deployment.

  py -3.14 scripts/89_stand_v4.py --build-frames        # blocked-pair M3rpa v4 frames (~15 min)
  py -3.14 scripts/89_stand_v4.py --diagnostic          # vs v18 on the same six held pairs
  py -3.14 scripts/89_stand_v4.py --fit-rank --submit   # all-2025 M3rpa v4 + next submission
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import submit
import tail_experts as te
from common import RUNS, SUBS

parser = argparse.ArgumentParser()
parser.add_argument('--build-frames', action='store_true')
parser.add_argument('--diagnostic', action='store_true')
parser.add_argument('--fit-rank', action='store_true')
parser.add_argument('--submit', action='store_true')
parser.add_argument('--tag', default=time.strftime('%Y%m%d'))
parser.add_argument('--suffix', default='_v4')
parser.add_argument('--approach', default='A20')
parser.add_argument('--rich-gate', action='store_true', help='gate also sees outer-safe fallback/airline rates + D-relative stand timeline')
parser.add_argument('--reuse-package', help='existing package dir with M3rpa/state.json and normal_rank.parquet (skips the all-2025 refit)')
args = parser.parse_args()
stamp = time.strftime('%Y%m%d_%H%M%S')
variant = '_richgate' if args.rich_gate else ''

if args.build_frames:
    t0 = time.time(); meta = te.build_strict_frames(4, args.suffix)
    meta['seconds'] = round(time.time() - t0, 1)
    (RUNS / f'stand_v4_strict_frames_{args.tag}.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(json.dumps(meta, indent=1), flush=True)

if args.diagnostic:
    out = RUNS / f'three_expert_tail_v4{variant}_strict_{args.tag}.json'
    report, frame = te.strict_diagnostic(output=out, candidate_version=4, candidate_suffix=args.suffix, rich_gate=args.rich_gate)
    frame[['MVT_ID_mvt', 'APT', 'month', 'fold', 'fpl_matched', 'y', 'v18_pred', 'pred', 'w', 'pb']].to_parquet(
        RUNS / 'oof' / f'three_expert_tail_v4{variant}_strict_{args.tag}.parquet', index=False)
    p = report['pooled']
    print(f"pooled v18 {p['v18']['rmse_raw']:.4f} -> v4 {p['new']['rmse_raw']:.4f} (delta {p['delta']:+.4f}); "
          f"worst fold {p['worst_fold_delta']:+.4f}; body {p['v18']['rmse_body']:.2f} -> {p['new']['rmse_body']:.2f}; "
          f"matched {p['v18']['rmse_matched']:.2f} -> {p['new']['rmse_matched']:.2f}; passes={report['passes']}", flush=True)
    for f, v in sorted(report['folds'].items()):
        print(f"  fold {f}: {v['rmse_v18']:.4f} -> {v['rmse_new']:.4f} ({v['delta']:+.4f})")

if args.fit_rank:
    pkg = RUNS / 'models' / f'{args.approach}_{stamp}'
    if args.reuse_package:
        import common
        from pathlib import Path
        src = Path(args.reuse_package)
        state = json.loads((src / 'M3rpa' / 'state.json').read_text(encoding='utf-8'))
        _, rank, _ = common.load_features(4)
        normal = rank[['MVT_ID_mvt']].merge(pd.read_parquet(src / 'normal_rank.parquet'), on='MVT_ID_mvt', how='left', validate='one_to_one').normal_pred.values
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / 'base_package.txt').write_text(str(src), encoding='utf-8')
    else:
        rank, normal, state = te.fit_full_m3rpa(4, pkg / 'M3rpa')
        pd.DataFrame({'MVT_ID_mvt': rank.MVT_ID_mvt.values, 'normal_pred': normal}).to_parquet(pkg / 'normal_rank.parquet', index=False)
    pred, detail, gate_model, meta = te.rank_prediction(rank, normal, state, feature_version=4, frame_version=4, frame_suffix=args.suffix,
                                                        rich_gate=args.rich_gate)
    gate_model.save_model(str(pkg / 'gate_lgb.txt'))
    detail.to_parquet(pkg / 'rank_detail.parquet', index=False)
    refs = [v for v in ('v18', 'v19', 'v20') if (SUBS / f'gentle-tractor_{v}.parquet').exists()]
    cmp = detail.copy()
    for v in refs:
        cmp = cmp.merge(pd.read_parquet(SUBS / f'gentle-tractor_{v}.parquet').rename(columns={'TAXITIME_SEC_mvt': v}), on='MVT_ID_mvt', validate='one_to_one')
    summary = {**meta, 'rows': int(len(cmp)), 'rich_gate': bool(args.rich_gate)}
    for v in refs:
        summary[f'rms_change_vs_{v}'] = float(np.sqrt(np.mean((cmp.pred - cmp[v]) ** 2)))
        summary[f'matched_rows_changed_vs_{v}'] = int(((np.abs(cmp.pred - cmp[v]) > 1) & (cmp.fpl_matched == 1)).sum())
    (pkg / 'summary.json').write_text(json.dumps(summary, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'gate'}, indent=1, default=str), flush=True)
    if args.submit:
        diag = json.loads((RUNS / f'three_expert_tail_v4{variant}_strict_{args.tag}.json').read_text(encoding='utf-8'))
        if not diag['passes']:
            raise SystemExit('strict diagnostic did not pass the promotion rule; no submission written')
        team = submit.team_name(); version = submit.next_version(team)
        path = submit.write_submission(team, version, detail.MVT_ID_mvt.values, pred)
        pooled = diag['pooled']['new']['rmse_raw']; fa, fb = diag['folds']['A']['rmse_new'], diag['folds']['B']['rmse_new']
        gate_desc = 'rich gate (outer-safe fallback/airline rates + D-relative stand timeline)' if args.rich_gate else 'refit strict gate'
        desc = (f"Features v4 (G10 same-stand timeline) M3rpa without date-bug labels + {gate_desc} + date-bug expert; "
                f"strict pooled {pooled:.2f} ({diag['pooled']['delta']:+.2f} vs v18), worst fold {diag['pooled']['worst_fold_delta']:+.2f}")
        row = dict(file=path.name, ts=time.strftime('%Y-%m-%d %H:%M:%S'), team=team, version_n=version, approach_id=args.approach,
                   feature_version=4, models_json=json.dumps(['M3rpa_v4_nodb', 'tail_gate_strict_rich' if args.rich_gate else 'tail_gate_strict', 'datebug_expert']),
                   foldA_rmse=fa, foldB_rmse=fb, result_rmse='', result_rank='', decision='candidate', notes=desc)
        pd.DataFrame([row]).to_csv(SUBS / 'submission_log.csv', mode='a', index=False, header=False)
        submit.append_manifest([f'# Stand v4 batch {time.strftime("%Y-%m-%d %H:%M")} - team {team}', '',
                                '| file | approach | description | strict Fold A | strict Fold B | pooled six-pair |', '|---|---|---|---:|---:|---:|',
                                f'| {path.name} | {args.approach} | {desc} | {fa:.2f} | {fb:.2f} | {pooled:.2f} |'])
        print(f'{path.name} <- {args.approach}: {desc}\npackage: {pkg}')
