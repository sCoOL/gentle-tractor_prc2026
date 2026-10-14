import argparse, ext_aircraft
p = argparse.ArgumentParser(); p.add_argument("--force", action="store_true"); a = p.parse_args(); ext_aircraft.build(force=a.force)
