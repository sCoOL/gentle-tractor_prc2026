import argparse, base_features
p = argparse.ArgumentParser(); p.add_argument('--force', action='store_true'); p.add_argument('--rank-only', action='store_true'); a = p.parse_args()   # continuity rule: --rank-only
base_features.build_base(force=a.force, rank_only=a.rank_only)
