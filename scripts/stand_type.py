"""Stand type per airport stand from the X-Plane Gateway packs.

  py -3.14 scripts/stand_type.py            # build data/ext_stand_type.parquet and print coverage

The challenge data carries only the stand NAME. X-Plane's startup locations (apt.dat row 1300) carry a location type:
``gate`` (contact stand), ``tie_down`` (remote apron position) or ``misc``. `data/ext_stand_coords.parquet` already holds the
coordinates of every dataset stand that was matched to a pack (2,200 stands, LEBL/LSZH/LTFM included via the normalised
name match), and every one of them coincides with exactly one 1300 row, so the type comes from a coordinate join.

Why: the ADS-B off-block estimate behaves differently at remote positions (no push-back: the record and the physical move
coincide at EHAM's remotes, 81 % within 60 s against 61 % at its gates; the opposite at LIRF, 9 % vs 25 %), and the stand id
as a categorical cannot express "type x ADS-B value" cheaply. Columns: ``stand_xp_type`` (0 unmatched, 1 gate, 2 tie_down,
3 misc) and ``stand_remote`` (tie_down or misc). Read through ``table()`` keyed by ``apt_stand`` ('EGLL|501').
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import prc_config  # noqa: E402
DATA = prc_config.DATA
PATH = DATA / 'ext_stand_type.parquet'
CODES = {'unmatched': 0, 'gate': 1, 'tie_down': 2, 'misc': 3}
COLUMNS = ['stand_xp_type', 'stand_remote']
_CACHE = {}


def build():
    rows = []
    for f in sorted((prc_config.EXTERNAL / 'xplane').glob('*.dat')):
        for line in f.read_text(encoding='utf-8', errors='replace').splitlines():
            p = line.split()
            if p and p[0] == '1300' and len(p) >= 7:
                rows.append((f.stem.upper(), float(p[1]), float(p[2]), p[4]))
    xp = pd.DataFrame(rows, columns=['APT', 'lat', 'lon', 'xp_type']).drop_duplicates(['APT', 'lat', 'lon'])
    s = pd.read_parquet(DATA / 'ext_stand_coords.parquet')
    m = s.merge(xp, on=['APT', 'lat', 'lon'], how='left')
    m['xp_type'] = m.xp_type.fillna('unmatched')
    m['stand'] = m.stand.astype('string').str.strip()
    m['apt_stand'] = m.APT.astype(str) + '|' + m.stand.astype(str)
    out = m[['APT', 'stand', 'apt_stand', 'xp_type']].drop_duplicates('apt_stand')
    out['stand_xp_type'] = out.xp_type.map(CODES).astype('int8')
    out['stand_remote'] = out.xp_type.isin(['tie_down', 'misc']).astype('int8')
    out.to_parquet(PATH, index=False)
    print(out.groupby(['APT', 'xp_type']).size().unstack(fill_value=0).to_string())
    print(f'{PATH}: {len(out):,} stands')
    return out


def table(path=PATH):
    """apt_stand + COLUMNS; stands not in the file are unmatched (code 0, not remote) after a left merge + fillna."""
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = pd.read_parquet(path, columns=['apt_stand'] + COLUMNS)
    return _CACHE[key]


def add(df):
    """Merge the two columns onto a frame that has ``apt_stand``; unknown stands get 0 / 0."""
    out = df.merge(table(), on='apt_stand', how='left', validate='many_to_one')
    for c in COLUMNS:
        out[c] = out[c].fillna(0).astype('int8')
    return out


if __name__ == '__main__':
    build()
