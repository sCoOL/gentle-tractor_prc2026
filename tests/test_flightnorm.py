"""Universal flight-number / callsign normaliser (scripts/flightnorm.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import flightnorm as fn  # noqa: E402


def setup_module():
    fn.set_prefix_map({'LH': 'DLH', 'FR': 'RYR', 'TK': 'THY', 'TP': 'TAP', 'W4': 'WMT', 'U2': 'EZY'})


def test_prefix_zeros_and_doubled_prefix():
    assert fn.canon('LH0114') == ('DLH', '114', '')
    assert fn.key('RYRR388G') == 'RYR388G' and fn.key('WMTMT437') == 'WMT437'
    assert fn.key('FR1123') == 'RYR1123'          # IATA number: no doubled-prefix rule
    assert fn.key('DLH3TP') == 'DLH3TP'           # ICAO alphanumeric callsign untouched
    assert fn.key(' ryr 0388g ') == 'RYR388G' and fn.key(None) == '' and fn.key('') == ''


def test_suffix_tolerant_and_fuzzy():
    a, b = 'THY7PXD', 'THY7PX'
    assert {fn.key(a), fn.base_key(a)} & {fn.key(b), fn.base_key(b)}
    assert fn.base_key('TAP931L') == fn.key('TP931') == 'TAP931'
    assert fn.base_key('NJE389D') == 'NJE389'
    assert set(fn.fuzzy_keys('EZY91LJ')) & set(fn.fuzzy_keys('EZY91KJ'))
    assert not set(fn.fuzzy_keys('EZY91LJ')) & set(fn.fuzzy_keys('RYR91LJ'))   # airline must agree
