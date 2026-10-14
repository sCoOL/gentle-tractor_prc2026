import argparse, assemble
p = argparse.ArgumentParser(); p.add_argument('--version', type=int, default=1); p.add_argument('--force', action='store_true'); a = p.parse_args()
assemble.assemble(version=a.version, force=a.force)
