"""Every location the pipeline reads or writes, in one place. Override any of them with an environment variable.

    PRC_WORK         working root that holds data/, runs/, submissions/          default: the repository root
    PRC_DATA         challenge files + derived tables                           default: $PRC_WORK/data
    PRC_EXTERNAL     downloaded open data (METAR, X-Plane, EUROCONTROL, ...)    default: $PRC_WORK/external
    PRC_ADSB_DIR     derived ADS-B tables and per-day caches                    default: $PRC_WORK/adsb
    PRC_ADSB_POINTS  adsb.lol downloads (raw/) and points_<date>.parquet        default: $PRC_ADSB_DIR
    PRC_OPDI_DIR     derived OPDI tables                                        default: $PRC_WORK/opdi
    PRC_OPDI_RAW     OPDI downloads (flight_list/, flight_events/, ...)          default: $PRC_OPDI_DIR
    PRC_FROZEN       frozen reference artefacts (docs/FROZEN_ARTEFACTS.md)      default: $PRC_WORK/frozen
    PRC_DATA_OLD     derived tables of the earlier Jan/Jul ranking export,      default: $PRC_WORK/data_old
                     read only by the member tools' reproduction checks

Relative values are taken relative to the repository root. Nothing else in the code names a drive or a user folder.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _path(name: str, default: Path) -> Path:
    v = os.environ.get(name, '').strip()
    if not v:
        return default
    p = Path(v).expanduser()
    return p if p.is_absolute() else (REPO / p)


WORK = _path('PRC_WORK', REPO)
DATA = _path('PRC_DATA', WORK / 'data')
RUNS = WORK / 'runs'
SUBS = WORK / 'submissions'
EXTERNAL = _path('PRC_EXTERNAL', WORK / 'external')
ADSB_DIR = _path('PRC_ADSB_DIR', WORK / 'adsb')
ADSB_POINTS = _path('PRC_ADSB_POINTS', ADSB_DIR)
OPDI_DIR = _path('PRC_OPDI_DIR', WORK / 'opdi')
OPDI_RAW = _path('PRC_OPDI_RAW', OPDI_DIR)
FROZEN = _path('PRC_FROZEN', WORK / 'frozen')
DATA_OLD = _path('PRC_DATA_OLD', WORK / 'data_old')

# Frozen reference artefacts (docs/FROZEN_ARTEFACTS.md): models, fixed weights and 2025-side validation frames that
# the final chain reads but does not rebuild.
REF_LAYERS = FROZEN / 'ref_layers'      # 2025-side layer frames, the A57 motion booster, ...
REF_MAIN = FROZEN / 'ref_main'          # A27 / A51 packages, fixed composition weights, reference member predictions
ADSB_SNAPSHOT = REF_MAIN / 'adsb_snapshot_0926'   # ADS-B v1 tables the variant-A ADS-B members are trained on


def describe() -> dict:
    return {k: str(v) for k, v in globals().items() if k.isupper() and isinstance(v, Path)}


if __name__ == '__main__':
    for k, v in describe().items():
        print(f'{k:12s} {v}')
