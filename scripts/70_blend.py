import argparse, blend
p = argparse.ArgumentParser(); p.add_argument('--runs', required=True, help='comma-separated run ids'); p.add_argument('--tag', default=''); p.add_argument('--equal', action='store_true'); a = p.parse_args()
blend.blend_runs(a.runs.split(','), a.tag, equal=a.equal)
