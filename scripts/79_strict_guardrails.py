"""Evaluate cross-block calibration and maximum-change caps for the strict gate."""
import argparse
from pathlib import Path

import strict_tail_gate
from common import RUNS


parser = argparse.ArgumentParser()
parser.add_argument('--features', type=int, default=3)
parser.add_argument('--output', default=str(RUNS / 'tail_gate_strict_guardrails_20260916.json'))
parser.add_argument('--shrinks', default='0.5,0.75,1.0')
parser.add_argument('--caps', default='300,600,900,1200,1800,none')
args = parser.parse_args()

shrinks = tuple(float(x) for x in args.shrinks.split(','))
caps = tuple(None if x.strip().lower() == 'none' else float(x) for x in args.caps.split(','))
report = strict_tail_gate.guardrail_diagnostic(
    feature_version=args.features,
    output=Path(args.output),
    shrinks=shrinks,
    max_deltas=caps,
)

base = report['pooled']['baseline']['rmse_raw']
print(f'pooled baseline: {base:.4f}')
print(f'champion: {report["champion"]}')
if report['champion']:
    row = report['pooled']['metrics'][report['champion']]
    print(f'pooled champion: {row["rmse_raw"]:.4f} ({row["delta_vs_base"]:+.4f})')
    for fold, delta in row['fold_deltas'].items():
        print(f'  fold {fold}: {delta:+.4f}')
print(f'wrote {args.output}')
