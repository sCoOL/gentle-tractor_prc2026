"""Run the fully nested rich-base comparison against strict v18."""
import argparse
from pathlib import Path

import nested_rich_base
from common import RUNS


parser = argparse.ArgumentParser()
parser.add_argument('--features', type=int, default=3)
parser.add_argument('--rounds', type=int, default=660)
parser.add_argument('--fallback-rounds', type=int, default=400)
parser.add_argument('--rich-groups', default='G4,G8')
parser.add_argument('--output', default=str(RUNS / 'tail_gate_nested_rich_base_20260916.json'))
args = parser.parse_args()

report = nested_rich_base.diagnostic(
    feature_version=args.features,
    rounds=args.rounds,
    fallback_rounds=args.fallback_rounds,
    rich_groups=tuple(x for x in args.rich_groups.split(',') if x),
    output=Path(args.output),
)
print(f"champion: {report['champion']}")
for key, row in report['pooled']['metrics'].items():
    print(
        f"shrink={key}: pooled={row['rmse_raw']:.4f} "
        f"vs_base={row['delta_vs_base']:+.4f} vs_v18={row['delta_vs_v18']:+.4f} "
        f"max_fold={row['max_fold_regression']:+.4f} accepted={row['accepted']}"
    )
print(f'wrote {args.output}')

