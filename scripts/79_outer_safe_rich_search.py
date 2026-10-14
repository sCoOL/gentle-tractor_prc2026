"""Ablate outer-safe rich gate feature families against strict v18."""
import json

import nested_features
import rich_strict_gate
from common import RUNS


target = [c for c in nested_features.TARGET_ENCODING_COLS if c.startswith('te_')]
fallback = [c for c in nested_features.TARGET_ENCODING_COLS if c.startswith('fb_rate_')]
route = list(nested_features.ROUTE_COLS)
variants = {
    'target': target,
    'fallback': fallback,
    'route': route,
    'target_fallback': target + fallback,
    'target_route': target + route,
    'fallback_route': fallback + route,
    'all': target + fallback + route,
}

frames, metadata = rich_strict_gate.build_frames(rebuild=False)
summary = {'selection_rule': 'same v18 gate; pooled >=1s better and no fold >1s worse', 'variants': {}}
accepted = []
for name, extras in variants.items():
    output = RUNS / f'tail_gate_outer_safe_{name}_20260916.json'
    report, _ = rich_strict_gate.diagnostic(
        output=output, extra_features=extras, frames=frames, metadata=metadata,
    )
    best = min(report['pooled']['metrics'].items(), key=lambda kv: kv[1]['rmse_raw'])
    summary['variants'][name] = {
        'extra_features': extras,
        'best_shrink': best[0],
        'best': best[1],
        'accepted_champion': report['champion'],
        'diagnostic': output.relative_to(RUNS.parent).as_posix(),
    }
    if report['champion'] is not None:
        row = report['pooled']['metrics'][report['champion']]
        accepted.append((row['rmse_raw'], name, report['champion']))
    print(
        f"{name}: best shrink={best[0]} pooled={best[1]['rmse_raw']:.4f} "
        f"vs_v18={best[1]['delta_vs_v18']:+.4f} "
        f"max_fold={best[1]['v18_max_fold_regression']:+.4f} "
        f"accepted={report['champion']}"
    )
summary['champion'] = (
    {'variant': min(accepted)[1], 'shrink': min(accepted)[2], 'rmse_raw': min(accepted)[0]}
    if accepted else None
)
path = RUNS / 'tail_gate_outer_safe_rich_search_20260916.json'
path.write_text(json.dumps(summary, indent=1), encoding='utf-8')
print(f"champion: {summary['champion']}")
print(f'wrote {path}')
