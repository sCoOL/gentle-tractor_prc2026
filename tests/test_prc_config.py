"""Locations (scripts/prc_config.py): defaults relative to the repository, environment overrides."""
import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))


def load(monkeypatch, **env):
    for k in ('PRC_WORK', 'PRC_DATA', 'PRC_EXTERNAL', 'PRC_ADSB_DIR', 'PRC_ADSB_POINTS', 'PRC_OPDI_DIR', 'PRC_OPDI_RAW',
              'PRC_FROZEN', 'PRC_DATA_OLD'):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    import prc_config
    return importlib.reload(prc_config)


def test_defaults(monkeypatch):
    c = load(monkeypatch)
    assert c.WORK == REPO and c.DATA == REPO / 'data' and c.RUNS == REPO / 'runs'
    assert c.ADSB_POINTS == c.ADSB_DIR == REPO / 'adsb' and c.OPDI_RAW == c.OPDI_DIR == REPO / 'opdi'
    assert c.REF_LAYERS == REPO / 'frozen' / 'ref_layers'
    assert c.ADSB_SNAPSHOT == REPO / 'frozen' / 'ref_main' / 'adsb_snapshot_0926'


def test_overrides(monkeypatch, tmp_path):
    c = load(monkeypatch, PRC_WORK=tmp_path, PRC_ADSB_POINTS=tmp_path / 'pts', PRC_FROZEN='relative/frozen')
    assert c.DATA == tmp_path / 'data' and c.ADSB_DIR == tmp_path / 'adsb' and c.ADSB_POINTS == tmp_path / 'pts'
    assert c.FROZEN == REPO / 'relative' / 'frozen'          # relative values are taken from the repository root
    load(monkeypatch)
