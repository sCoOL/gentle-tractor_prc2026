import argparse, detectors
p = argparse.ArgumentParser(); p.add_argument('--force', action='store_true'); p.add_argument('--stage', default='all', help='all | v1 | fallback_v2 | det36'); a = p.parse_args()
if a.stage in ('all', 'v1'): detectors.run_all(force=a.force)
if a.stage in ('all', 'fallback_v2'): detectors.run_fallback_v2(force=a.force)
if a.stage in ('all', 'det36'): detectors.run_detectors_3_to_6(force=a.force)
