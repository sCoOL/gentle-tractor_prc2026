import argparse, ext_calendar
p = argparse.ArgumentParser(); p.add_argument("--force", action="store_true"); a = p.parse_args(); ext_calendar.build(force=a.force)
