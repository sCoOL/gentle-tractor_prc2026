import argparse, ext_layout
p = argparse.ArgumentParser(); p.add_argument("--force", action="store_true"); a = p.parse_args(); ext_layout.build(force=a.force)
