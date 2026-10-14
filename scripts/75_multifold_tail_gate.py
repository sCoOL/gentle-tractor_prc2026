"""Six-pair leave-one-pair-out validation for the cost-weighted tail gate."""
import argparse

import tail_gate


parser = argparse.ArgumentParser()
parser.add_argument('--fold-runs', required=True,
                    help='comma list such as A:run1,B:run1,C:run2,...,F:run2')
parser.add_argument('--features', type=int, default=3)
parser.add_argument('--output')
args = parser.parse_args()
fold_runs = dict(item.split(':', 1) for item in args.fold_runs.split(','))
report = tail_gate.multi_pair_diagnostic(fold_runs, args.features, args.output)
for fold, item in report['folds'].items():
    base = item['baseline']['rmse_raw']
    scores = ', '.join(f'{s}={m["rmse_raw"]:.2f}' for s, m in item['metrics'].items())
    print(f'{fold}: base={base:.2f}; {scores}')
pooled = report['pooled']
scores = ', '.join(f'{s}={m["rmse_raw"]:.2f}' for s, m in pooled['metrics'].items())
print(f'pooled: base={pooled["baseline"]["rmse_raw"]:.2f}; {scores}')
