import argparse, ext_eurocontrol
p = argparse.ArgumentParser(); p.add_argument("--force", action="store_true"); a = p.parse_args(); ext_eurocontrol.build(force=a.force)
