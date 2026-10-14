"""The post-steps after the final chain, in order: A15 -> A15L -> A15LX -> A15LXR -> A15LXRH -> A15LXRHDS (docs/METHOD.md §11,
docs/REPRODUCE.md §7). Each step is its own script; this runs them one after the other with the same interpreter and
stops at the first failure.

  py -3.14 tools/last_day/run_last_day.py                 # every step, then the output check of the final file
  py -3.14 tools/last_day/run_last_day.py --from c18      # resume at a step (names below)
  py -3.14 tools/last_day/run_last_day.py --no-checks     # skip the steps that only check or report

Needs: the final-chain run runs/clean_chain_final/A15/ (final file, manual_rows_applied.parquet, v54_gate/table.parquet,
lirf_row_audit/lirf_rows.parquet), runs/overnight_20261008/frame_cache.parquet (tools/layers/build_frames.py ovcache),
PRC_ADSB_DIR/push_phases.parquet, PRC_EXTERNAL/eurocontrol/all_pre_departure_delays_{2025,2026}.csv
(scripts/fetch_external.py), runs/v51_turnaround/links.parquet (scripts/231) and
PRC_FROZEN/ref_layers/runs/leads_20261008/rows.parquet (docs/FROZEN_ARTEFACTS.md).
"""
import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# (name, script, args, check-only)
STEPS = [
    ('identity', 'c9_step1_identity.py', [], True),        # 60 x DLY_ALL_PRE_2 = sum max(0, BLOCK - SCHED), 2025
    ('c0', 'c0_long_delay_rule.py', [], False),            # 8-13 h rule rows
    ('c3', 'c3_cap_at_D.py', [], True),                    # 2025 counts of the cap at D
    ('A15L', 'build_A15L.py', [], False),
    ('c9', 'c9_daily_layer.py', [], True),                 # version-1 daily layer: the 2025 reference of the stack
    ('c15', 'c15_combined.py', ['--tag', 'x'], False),     # stack out of fold (guard of A15LX)
    ('A15LX', 'build_A15LX.py', [], False),
    ('c16', 'c16_stack_recentred.py', ['--tag', 'x'], False),   # stack + re-centred enumeration out of fold
    ('A15LXR_check', 'build_A15LXR.py', ['--idea4', '0'], True),  # must reproduce A15LX
    ('A15LXR', 'build_A15LXR.py', ['--idea4', '1'], False),
    ('c18', 'c18_crossers_hardened.py', ['--tag', 'h'], False),  # hardened crossers out of fold (guard of A15LXRH)
    ('A15LXRH', 'build_A15LXRH.py', [], False),
    ('c19', 'c19_lirf_datebug_rules.py', [], True),         # LIRF date-bug / copy rules: 2025 in-sample counts
    ('A15LXRHDS', 'build_A15LXRHDS.py', [], False),
]


def verify(tag='A15LXRHDS'):
    sys.path.insert(0, str(HERE))
    import numpy as np
    import pandas as pd
    import ld
    p = ld.FIN / tag / f'gentle-tractor_final_v68_{tag}.parquet'
    f = pd.read_parquet(p)
    tpl = pd.read_parquet(ld.DATA / 'submitting.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy()
    x = f.TAXITIME_SEC_mvt.to_numpy()
    ok = (list(f.columns) == ['MVT_ID_mvt', 'TAXITIME_SEC_mvt'] and len(f) == 670790 and f.MVT_ID_mvt.is_unique
          and (f.MVT_ID_mvt.to_numpy() == tpl).all() and np.isfinite(x).all() and x.min() >= 30)
    print(f'{p}: rows {len(f):,}, template order {bool((f.MVT_ID_mvt.to_numpy() == tpl).all())}, min {x.min():.1f}, '
          f'max {x.max():.1f}, mean {x.mean():.2f}, sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}')
    if not ok:
        raise SystemExit('output check failed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='start', default=STEPS[0][0], choices=[s[0] for s in STEPS])
    ap.add_argument('--no-checks', action='store_true')
    a = ap.parse_args()
    names = [s[0] for s in STEPS]
    for name, script, args, check in STEPS[names.index(a.start):]:
        if check and a.no_checks:
            continue
        t0 = time.time()
        print(f'--- {name}: {script} {" ".join(args)}', flush=True)
        r = subprocess.run([sys.executable, str(HERE / script), *args], cwd=REPO)
        if r.returncode:
            raise SystemExit(f'step {name} failed (exit {r.returncode}); resume with --from {name}')
        print(f'--- {name} done in {time.time() - t0:.0f} s', flush=True)
    verify()


if __name__ == '__main__':
    main()
