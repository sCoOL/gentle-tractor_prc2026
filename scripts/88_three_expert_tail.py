"""Three-expert tail mixture: strict diagnostic, v18 reproduction check, deployment and submission.

  py -3.14 scripts/88_three_expert_tail.py --retrain-blocks        # strict nodb clean experts (~5 min)
  py -3.14 scripts/88_three_expert_tail.py --diagnostic            # blocked-pair evaluation vs v18
  py -3.14 scripts/88_three_expert_tail.py --reproduce-v18         # deployment path == v18 file
  py -3.14 scripts/88_three_expert_tail.py --fit-rank --submit     # full-2025 expert + next version
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import submit
import tail_experts as te
from common import RUNS, SUBS

PACKAGE = RUNS / 'models' / 'A13_20260915_105919'
parser = argparse.ArgumentParser()
parser.add_argument('--retrain-blocks', action='store_true')
parser.add_argument('--diagnostic', action='store_true')
parser.add_argument('--reproduce-v18', action='store_true')
parser.add_argument('--fit-rank', action='store_true')
parser.add_argument('--submit', action='store_true')
parser.add_argument('--package', default=str(PACKAGE))
parser.add_argument('--tag', default=time.strftime('%Y%m%d'))
parser.add_argument('--approach', default='A19')
args = parser.parse_args()
stamp = time.strftime('%Y%m%d_%H%M%S')

if args.retrain_blocks:
    t0 = time.time(); meta = te.retrain_block_experts()
    meta['seconds'] = round(time.time() - t0, 1)
    (RUNS / f'three_expert_tail_block_experts_{args.tag}.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(json.dumps(meta, indent=1), flush=True)

if args.diagnostic:
    out = RUNS / f'three_expert_tail_strict_{args.tag}.json'
    report, frame = te.strict_diagnostic(output=out)
    frame[['MVT_ID_mvt', 'APT', 'month', 'fold', 'fpl_matched', 'y', 'v18_pred', 'pred', 'w', 'pb']].to_parquet(
        RUNS / 'oof' / f'three_expert_tail_strict_{args.tag}.parquet', index=False)
    p = report['pooled']
    print(f"pooled v18 {p['v18']['rmse_raw']:.4f} -> new {p['new']['rmse_raw']:.4f} (delta {p['delta']:+.4f}); "
          f"worst fold {p['worst_fold_delta']:+.4f}; passes={report['passes']}", flush=True)
    for f, v in sorted(report['folds'].items()):
        print(f"  fold {f}: {v['rmse_v18']:.4f} -> {v['rmse_new']:.4f} ({v['delta']:+.4f}) pb rates {v['pb_rates']}")

if args.reproduce_v18:
    import model_reload
    rank, normal, state = model_reload.reload_m3rpa_clean_expert(args.package)
    pred, detail, _, meta = te.rank_prediction(rank, normal, state, use_nodb_frames=False, use_datebug_expert=False)
    v18 = pd.read_parquet(SUBS / 'gentle-tractor_v18.parquet')
    chk = detail[['MVT_ID_mvt']].assign(pred=np.maximum(pred, 30.0)).merge(v18, on='MVT_ID_mvt', validate='one_to_one')
    delta = np.abs(chk.pred.values - chk.TAXITIME_SEC_mvt.values)
    print(f'v18 reproduction: rows {len(chk)}, max abs delta {delta.max():.6g}, mean {delta.mean():.3g}', flush=True)
    if delta.max() > 1e-6:
        raise SystemExit('deployment path does not reproduce v18')

if args.fit_rank:
    pkg = RUNS / 'models' / f'{args.approach}_{stamp}'
    rank, normal, state, fit_meta = te.fit_rank_expert(args.package, save_dir=pkg / 'M3rpa_nodb')
    pd.DataFrame({'MVT_ID_mvt': rank.MVT_ID_mvt.values, 'normal_pred': normal}).to_parquet(pkg / 'normal_rank_nodb.parquet', index=False)
    pred, detail, gate_model, meta = te.rank_prediction(rank, normal, state)
    gate_model.save_model(str(pkg / 'gate_lgb.txt'))
    detail.to_parquet(pkg / 'rank_detail.parquet', index=False)
    v18 = pd.read_parquet(SUBS / 'gentle-tractor_v18.parquet').rename(columns={'TAXITIME_SEC_mvt': 'v18'})
    cmp = detail.merge(v18, on='MVT_ID_mvt', validate='one_to_one')
    diff = cmp.pred.values - cmp.v18.values
    summary = {**fit_meta, **meta, 'base_package': args.package, 'rows': int(len(cmp)),
               'rows_changed_over_1s': int((np.abs(diff) > 1).sum()), 'rows_changed_over_600s': int((np.abs(diff) > 600).sum()),
               'mean_abs_change': float(np.abs(diff).mean()), 'rms_change': float(np.sqrt(np.mean(diff ** 2))),
               'matched_rows_changed_over_1s': int(((np.abs(diff) > 1) & (cmp.fpl_matched.values == 1)).sum())}
    (pkg / 'state.json').write_text(json.dumps(summary, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'gate'}, indent=1, default=str), flush=True)
    if args.submit:
        diag = json.loads((RUNS / f'three_expert_tail_strict_{args.tag}.json').read_text(encoding='utf-8'))
        team = submit.team_name(); version = submit.next_version(team)
        path = submit.write_submission(team, version, detail.MVT_ID_mvt.values, pred)
        pooled = diag['pooled']['new']['rmse_raw']
        fa, fb = diag['folds']['A']['rmse_new'], diag['folds']['B']['rmse_new']
        desc = (f"Three-expert tail: clean expert without date-bug labels + refit strict gate + binned date-bug expert; "
                f"strict pooled {pooled:.2f} ({diag['pooled']['delta']:+.2f} vs v18), worst fold {diag['pooled']['worst_fold_delta']:+.2f}")
        row = dict(file=path.name, ts=time.strftime('%Y-%m-%d %H:%M:%S'), team=team, version_n=version, approach_id=args.approach,
                   feature_version=3, models_json=json.dumps(['M3rpa_nodb', 'tail_gate_strict', 'datebug_expert']),
                   foldA_rmse=fa, foldB_rmse=fb, result_rmse='', result_rank='', decision='candidate', notes=desc)
        pd.DataFrame([row]).to_csv(SUBS / 'submission_log.csv', mode='a', index=False, header=False)
        submit.append_manifest([f'# Three-expert tail batch {time.strftime("%Y-%m-%d %H:%M")} - team {team}', '',
                                '| file | approach | description | strict Fold A | strict Fold B | pooled six-pair |', '|---|---|---|---:|---:|---:|',
                                f'| {path.name} | {args.approach} | {desc} | {fa:.2f} | {fb:.2f} | {pooled:.2f} |'])
        print(f'{path.name} <- {args.approach}: {desc}')
        print(f'package: {pkg}')
