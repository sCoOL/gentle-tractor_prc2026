import argparse
import json

import tail_gate

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--features', type=int, default=3)
p.add_argument('--output')
a = p.parse_args()
r = tail_gate.cross_pair_diagnostic(a.run, a.features, a.output)
for direction, result in r['directions'].items():
    print(direction)
    for shrink, m in result['metrics'].items():
        print(f'  shrink={shrink}: rmse {m["rmse_raw"]:.2f}, months {m["rmse_m1"]:.2f}/{m["rmse_m2"]:.2f}, unmatched {m["rmse_unmatched"]:.1f}, top100 {m["ss_top100_share"]:.3f}')
