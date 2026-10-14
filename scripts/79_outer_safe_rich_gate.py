"""Run the fully outer-safe rich-gate experiment; diagnostic only."""
import argparse
from pathlib import Path

import rich_strict_gate
from common import RUNS


parser = argparse.ArgumentParser()
parser.add_argument('--features', type=int, default=3)
parser.add_argument('--output', default=str(RUNS / 'tail_gate_outer_safe_rich_20260916.json'))
args = parser.parse_args()

report, _ = rich_strict_gate.diagnostic(args.features, Path(args.output))
print(f"champion: {report['champion']}")
for shrink, row in report['pooled']['metrics'].items():
    print(
        f"shrink={shrink}: pooled {row['rmse_raw']:.4f}, "
        f"vs base {row['delta_vs_base']:+.4f}, vs v18 {row['delta_vs_v18']:+.4f}, "
        f"max fold regression {row['v18_max_fold_regression']:+.4f}, "
        f"accepted={row['accepted_vs_v18']}"
    )
print(f'wrote {args.output}')
