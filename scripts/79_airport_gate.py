"""Run hierarchical airport-specific strict-gate diagnostics."""
import airport_tail_gate
from common import RUNS


path = RUNS / 'tail_gate_airport_hierarchical_20260916.json'
report = airport_tail_gate.diagnostic(path)
for key, row in sorted(report['pooled']['metrics'].items(), key=lambda kv: kv[1]['rmse_raw']):
    print(
        f"{key}: pooled={row['rmse_raw']:.4f} vs_v18={row['delta_vs_v18']:+.4f} "
        f"max_fold={row['max_fold_regression']:+.4f} accepted={row['accepted']}"
    )
print(f"champion: {report['champion']}")
print(f'wrote {path}')
