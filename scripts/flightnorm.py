"""Universal flight-number / callsign normaliser (2026-10-07: "normalising the callsign / flight number should be a
universal preprocess ... discover all similarity ... even fuzzy ... airlines add 'D' for delayed flights, airports
prefix charters or special flights").

One canonical form for every identifier we compare across sources: the airport flight number (FLIGHT_mvt), the NM
callsign (CALLSIGN_flt), the destination arrival's callsign (scripts/250 backfill), the ADS-B callsign (adsb.lol) and
OPDI's flt_id. Used by scripts/255 (ADS-B match), 260 (OPDI link) and 231 (turnaround rotation rule).

  canon('LH0114')   -> ('DLH', '114', '')      IATA prefix -> ICAO (map learned from NM-matched rows, scripts/255)
  canon('RYRR388G') -> ('RYR', '388', 'G')     LIRF doubled prefix dropped (only after a 3-letter prefix, before digits)
  canon('WMTMT437') -> ('WMT', '437', '')
  canon('THY7PXD')  -> ('THY', '7', 'PXD')     ICAO alphanumeric callsign: number = leading digits, suffix = the rest
  key()      = airline + number + suffix       exact canonical key
  base_key() = key without one trailing letter when the key still ends in a digit+letters part ('THY7PXD' ~ 'THY7PX',
               'TAP931L' ~ 'TAP931', 'NJE389D' ~ 'NJE389')
  fuzzy_keys() = one-deletion neighbourhood of the key (same airline): two keys within edit distance 1 share one, so
               'EZY91LJ' ~ 'EZY91KJ'. Only for a last, time-tight pass; never on its own.
"""
import re
from functools import lru_cache

_CLEAN = re.compile(r'[^A-Z0-9]')
_PREFIX = re.compile(r'^([A-Z]{3}|[A-Z0-9]{2})')
_NUM = re.compile(r'^(\d*)(.*)$')
_MAP: dict = {}


def set_prefix_map(m: dict):
    """IATA (2-char) -> ICAO (3-letter) airline prefix map; scripts/255 learns it from NM-matched rows."""
    _MAP.clear()
    _MAP.update(m)
    canon.cache_clear()


def _undouble(p: str, r: str) -> str:
    """'RYR' + 'R388G' -> 'R388G' minus the repeated 'R'. Only letters repeated from the prefix's tail, followed by a digit,
    so IATA-style numbers ('FR1123') and ICAO alphanumeric callsigns ('DLH3TP') are never touched."""
    if not p.isalpha():
        return r
    for k in (p, p[1:], p[2:]):
        if r.startswith(k) and len(r) > len(k) and r[len(k)].isdigit():
            return r[len(k):]
    return r


@lru_cache(maxsize=1_000_000)
def canon(s) -> tuple:
    """(airline, number, suffix); ('', '', '') when s is empty / not a string."""
    if not isinstance(s, str):
        return ('', '', '')
    s = _CLEAN.sub('', s.upper())
    if not s:
        return ('', '', '')
    m = _PREFIX.match(s)
    if not m:
        return ('', s, '')
    p, r = m.group(1), s[len(m.group(1)):]
    al = _MAP.get(p, p)
    r = _undouble(al if len(al) == 3 else p, r)
    num, suf = _NUM.match(r).groups()
    num = num.lstrip('0') or ('0' if num else '')
    return (al, num, suf)


def airline(s) -> str:
    return canon(s)[0]


def key(s) -> str:
    a, n, x = canon(s)
    return a + n + x


def base_key(s) -> str:
    """Suffix-tolerant key: one trailing letter dropped when the key keeps at least one character after the airline."""
    a, n, x = canon(s)
    k = a + n + x
    if x and x[-1].isalpha() and len(n + x) >= 2:
        return k[:-1]
    return k


def fuzzy_keys(s) -> list:
    """Keys of the one-deletion neighbourhood inside the part after the airline (the airline itself must agree)."""
    a, n, x = canon(s)
    t = n + x
    if not a or len(t) < 2:
        return []
    return sorted({a + t[:i] + t[i + 1:] for i in range(len(t))} | {a + t})


def learn_aliases(apt, flight, callsign, min_n: int = 3, min_share: float = 0.2, top: int = 2) -> dict:
    """(airport, key(flight number)) -> up to `top` callsign keys that NM used for that number, each on >= min_n rows and
    >= min_share of the number's rows. Learned from NM-matched rows (label-free: NM callsign and the airport's flight
    number); covers the IATA-number-vs-ATC-callsign airports (EDDF LH964 -> DLH8WY, EHAM KL1337 -> KLM55Z) where no
    string rule can map one onto the other."""
    from collections import Counter, defaultdict
    cnt = defaultdict(Counter)
    for a, f, c in zip(apt, flight, callsign):
        kf, kc = key(f), key(c)
        if kf and kc and kf != kc:
            cnt[(a, kf)][kc] += 1
    out = {}
    for k, c in cnt.items():
        tot = sum(c.values())
        keep = [x for x, n in c.most_common(top) if n >= min_n and n >= min_share * tot]
        if keep:
            out[k] = keep
    return out
