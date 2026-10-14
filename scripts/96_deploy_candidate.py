"""Deploy a strict-validated three-expert candidate to a ranking submission.

  # A24: A20 package + context detector on v4fb5 frames, fallback scale 1.0; with only --reuse-package,
  # --feature-version 4 --frame-version 4 --frames-suffix _v4 this path reproduces v20 exactly (checked 2026-09-17)
  py -3.14 scripts/96_deploy_candidate.py --approach A24 --reuse-package runs/models/A20_20260917_005317 \
      --feature-version 4 --frame-version 4 --frames-suffix _v4fb5 --detector context --base-p-scale 1.0 \
      --evidence runs/evidence_A24_20260917.json --submit

  --strict-base      full-2025 clean expert without the target-derived columns the strict frames drop
  --detector context refit the fallback detector with G10/G11 inputs (scripts/fallback_context.py)
  --detector-seeds   average that detector over LightGBM seeds

The evidence JSON must hold pooled, fold_A, fold_B, delta_vs_ref, worst_fold_vs_ref, reference,
passes and summary; --submit refuses a candidate whose evidence does not pass.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common
import fallback_context as fc
import strict_tail_gate as stg
import submit
import tail_experts as te
from common import RUNS, SUBS

parser = argparse.ArgumentParser()
parser.add_argument('--approach', required=True)
parser.add_argument('--feature-version', type=int, required=True, help='features of the full-2025 clean expert')
parser.add_argument('--frame-version', type=int, required=True, help='features the strict frames / gate were built on')
parser.add_argument('--frames-suffix', required=True)
parser.add_argument('--reuse-package', help='package with M3rpa/state.json and normal_rank.parquet')
parser.add_argument('--strict-base', action='store_true')
parser.add_argument('--exclusion', choices=['datebug', 'artifacts'], default='datebug',
                    help='training rows left out of the experts (tail_experts.training_exclusion)')
parser.add_argument('--detector', choices=['table', 'context'], default='table')
parser.add_argument('--detector-version', type=int, default=5)
parser.add_argument('--detector-seeds', type=int, nargs='*', default=None, help='average context detectors over these LightGBM seeds')
parser.add_argument('--residual-blend', choices=['none', 'rd'], default='none',
                    help='rd: matched-row expert = 0.25 global + 0.25 global seed 1 + 0.5 per-airport AOBT_3 residual')
parser.add_argument('--params', default='{}', help='JSON LightGBM params for the clean experts, e.g. residual_params/residual_rounds')
parser.add_argument('--rounds', type=int, default=660)
parser.add_argument('--blend-params', default='{}', help='JSON params for the rd blend global seed-1 member (default: residual_params)')
parser.add_argument('--airport-params', default='{}', help='JSON params for the rd blend per-airport residual members')
parser.add_argument('--airport-rounds', type=int, default=660)
parser.add_argument('--matched-p-scale', type=float, default=None)
parser.add_argument('--base-p-scale', type=float, default=1.4)
parser.add_argument('--models', default='', help='comma-separated model ids for the submission log')
parser.add_argument('--evidence')
parser.add_argument('--submit', action='store_true')
args = parser.parse_args()
stamp = time.strftime('%Y%m%d_%H%M%S')
pkg = RUNS / 'models' / f'{args.approach}_{stamp}'
pkg.mkdir(parents=True, exist_ok=True)
t0 = time.time()

target_cols = set()
if args.strict_base:
    manifest = common.read_manifest(args.feature_version)
    target_cols = set(manifest['groups'].get('G8', [])) | stg.TARGET_ROUTE | stg.TARGET_G9

if args.reuse_package:
    src = Path(args.reuse_package)
    state = json.loads((src / 'M3rpa' / 'state.json').read_text(encoding='utf-8'))
    _, rank, _ = common.load_features(args.feature_version)
    normal = rank[['MVT_ID_mvt']].merge(pd.read_parquet(src / 'normal_rank.parquet'), on='MVT_ID_mvt', how='left',
                                        validate='one_to_one').normal_pred.values
    (pkg / 'base_package.txt').write_text(str(src), encoding='utf-8')
else:
    rank, normal, state = te.fit_full_m3rpa(args.feature_version, pkg / 'M3rpa', rounds=args.rounds, exclude=target_cols,
                                            params=json.loads(args.params), exclusion=args.exclusion)
    pd.DataFrame({'MVT_ID_mvt': rank.MVT_ID_mvt.values, 'normal_pred': normal}).to_parquet(pkg / 'normal_rank.parquet', index=False)
    print(f'clean expert fitted: {len(state["cols"])} columns [{time.time() - t0:.0f}s]', flush=True)

if args.residual_blend == 'rd':
    import residual_variants as rv
    model_params = json.loads(args.params)
    blend_params = json.loads(args.blend_params) or dict(model_params.get('residual_params', {}))
    blend_rounds = int(model_params.get('residual_rounds', state.get('rounds', 660)))
    ids, normal = rv.rank_blend_normal(args.feature_version, state['cols'], state['cats'], normal, rounds=blend_rounds, params=blend_params,
                                       airport_params=json.loads(args.airport_params), airport_rounds=args.airport_rounds,
                                       exclusion=args.exclusion)
    if not np.array_equal(ids, rank.MVT_ID_mvt.values):
        raise ValueError('blended expert rows are not aligned with the expert rows')
    pd.DataFrame({'MVT_ID_mvt': ids, 'normal_pred': normal}).to_parquet(pkg / 'normal_rank_rd.parquet', index=False)
    print(f'residual blend rd fitted [{time.time() - t0:.0f}s]', flush=True)

rank_p = None
if args.detector == 'context':
    need = ['MVT_ID_mvt', 'month', 'fallback'] + [c for c in fc.FEATS if c != 'month']
    train_d = pd.read_parquet(common.DATA / f'features_v{args.detector_version}_train.parquet', columns=need)
    rank_d = pd.read_parquet(common.DATA / f'features_v{args.detector_version}_rank.parquet', columns=[c for c in need if c != 'fallback'])
    if not np.array_equal(rank_d.MVT_ID_mvt.values, rank.MVT_ID_mvt.values):
        raise ValueError('detector feature rows are not aligned with the expert rows')
    rank_p = fc.score_seeds(train_d, rank_d, fc.FEATS, fc.CATS, 400, args.detector_seeds)
    pd.DataFrame({'MVT_ID_mvt': rank.MVT_ID_mvt.values, 'p_fallback': rank_p}).to_parquet(pkg / 'p_fallback_rank.parquet', index=False)
    print(f'context detector scored: mean p {rank_p.mean():.4f} vs table {rank.p_fallback.mean():.4f} [{time.time() - t0:.0f}s]', flush=True)
    del train_d, rank_d

pred, detail, gate_model, meta = te.rank_prediction(
    rank, normal, state, feature_version=args.feature_version, base_p_scale=args.base_p_scale,
    frame_version=args.frame_version, frame_suffix=args.frames_suffix, rank_p_fallback=rank_p,
    matched_p_scale=args.matched_p_scale)
gate_model.save_model(str(pkg / 'gate_lgb.txt'))
detail.to_parquet(pkg / 'rank_detail.parquet', index=False)
cmp = detail[['MVT_ID_mvt', 'fpl_matched', 'pred']].copy()
summary = {**meta, 'approach': args.approach, 'feature_version': args.feature_version, 'frame_version': args.frame_version,
           'frames_suffix': args.frames_suffix, 'reuse_package': args.reuse_package, 'strict_base': args.strict_base,
           'n_expert_columns': len(state['cols']), 'residual_blend': args.residual_blend, 'exclusion': args.exclusion,
           'detector': args.detector, 'detector_seeds': args.detector_seeds,
           'matched_p_scale': args.matched_p_scale,
           'base_p_scale': args.base_p_scale, 'rows': int(len(cmp)), 'finite': bool(np.isfinite(pred).all()),
           'min_pred': float(pred.min()), 'seconds': round(time.time() - t0, 1)}
for f in sorted(SUBS.glob(f'{submit.team_name()}_v*.parquet'), key=lambda p: int(p.stem.split('_v')[1])):
    v = int(f.stem.split('_v')[1])
    if v < 18:
        continue
    other = cmp[['MVT_ID_mvt']].merge(pd.read_parquet(f), on='MVT_ID_mvt', how='left', validate='one_to_one').TAXITIME_SEC_mvt.values
    diff = cmp.pred.values - other
    summary[f'vs_v{v}'] = {'rms_change': float(np.sqrt(np.mean(diff ** 2))),
                           'matched_rows_changed': int(((np.abs(diff) > 1) & (cmp.fpl_matched.values == 1)).sum()),
                           'unmatched_rows_changed': int(((np.abs(diff) > 1) & (cmp.fpl_matched.values == 0)).sum())}
(pkg / 'summary.json').write_text(json.dumps(summary, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k != 'gate'}, indent=1, default=str), flush=True)

if args.submit:
    if not args.evidence:
        raise SystemExit('--submit needs --evidence')
    ev = json.loads(Path(args.evidence).read_text(encoding='utf-8'))
    if not ev.get('passes'):
        raise SystemExit('evidence does not pass the promotion rule; no submission written')
    team = submit.team_name(); version = submit.next_version(team)
    path = submit.write_submission(team, version, detail.MVT_ID_mvt.values, pred)
    desc = (f"{ev['summary']}; strict pooled {ev['pooled']:.2f} ({ev['delta_vs_ref']:+.2f} vs {ev['reference']}), "
            f"worst fold {ev['worst_fold_vs_ref']:+.2f}")
    models = [m for m in args.models.split(',') if m]
    row = dict(file=path.name, ts=time.strftime('%Y-%m-%d %H:%M:%S'), team=team, version_n=version, approach_id=args.approach,
               feature_version=args.feature_version, models_json=json.dumps(models), foldA_rmse=ev['fold_A'], foldB_rmse=ev['fold_B'],
               result_rmse='', result_rank='', decision='candidate', notes=desc)
    pd.DataFrame([row]).to_csv(SUBS / 'submission_log.csv', mode='a', index=False, header=False, lineterminator='\n')
    submit.append_manifest([f'# Candidate {args.approach} {time.strftime("%Y-%m-%d %H:%M")} - team {team}', '',
                            '| file | approach | description | strict Fold A | strict Fold B | pooled six-pair |', '|---|---|---|---:|---:|---:|',
                            f"| {path.name} | {args.approach} | {desc} | {ev['fold_A']:.2f} | {ev['fold_B']:.2f} | {ev['pooled']:.2f} |"])
    (pkg / 'submission.txt').write_text(path.name, encoding='utf-8')
    print(f'{path.name} <- {args.approach}: {desc}\npackage: {pkg}')
