import argparse
import nested_features

p = argparse.ArgumentParser()
p.add_argument('--version', type=int, default=3)
p.add_argument('--parent-version', type=int, default=2)
p.add_argument('--fallback-rounds', type=int, default=400)
p.add_argument('--force', action='store_true')
a = p.parse_args()
nested_features.build(a.version, a.parent_version, a.force, a.fallback_rounds)
