"""Run the strict blocked-pair base/gate diagnostic."""
import argparse

import strict_tail_gate


parser = argparse.ArgumentParser()
parser.add_argument('--features', type=int, default=3)
parser.add_argument('--rounds', type=int, default=660)
parser.add_argument('--fallback-rounds', type=int, default=400)
parser.add_argument('--output')
args = parser.parse_args()
report = strict_tail_gate.diagnostic(
    args.features, args.rounds, args.fallback_rounds, args.output,
)
for block in report['blocks'].values():
    for direction, item in block['directions'].items():
        scores = ', '.join(f'{s}={m["rmse_raw"]:.2f}' for s, m in item['metrics'].items())
        print(f'{direction}: base={item["baseline"]["rmse_raw"]:.2f}; {scores}')
p = report['pooled']
scores = ', '.join(f'{s}={m["rmse_raw"]:.2f}' for s, m in p['metrics'].items())
print(f'pooled: base={p["baseline"]["rmse_raw"]:.2f}; {scores}')
