"""Aircraft dimensions (G7): openap for 37 common types, manual CSV for the rest."""
from __future__ import annotations
import numpy as np, pandas as pd
from openap import prop
import common
from common import DATA, EXT
ENGINE = {'turbofan': 'jet', 'turbojet': 'jet', 'turboprop': 'turboprop', 'piston': 'piston'}


def code_letter(span) -> str:
    if span is None or np.isnan(span): return 'U'
    for letter, lim in (('A', 15), ('B', 24), ('C', 36), ('D', 52), ('E', 65)):
        if span < lim: return letter
    return 'F'


def from_openap() -> pd.DataFrame:
    rows = []
    for code in prop.available_aircraft():
        a = prop.aircraft(code)
        rows.append(dict(icao=code.upper(), mtow_kg=float(a['mtow']), wingspan_m=float(a['wing']['span']), length_m=float(a['fuselage']['length']),
                         n_engines=int(a['engine']['number']), engine_type=ENGINE.get(str(a['engine']['type']).lower(), 'jet')))
    return pd.DataFrame(rows)


def load_manual() -> pd.DataFrame:
    return pd.read_csv(EXT / 'aircraft' / 'manual_types.csv').drop_duplicates('icao')


def combine(o: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    o = o.assign(aircraft_source='openap'); m = m[~m.icao.isin(set(o.icao))].assign(aircraft_source='manual')
    t = pd.concat([o, m], ignore_index=True); t['icao_code_letter'] = t.wingspan_m.map(code_letter); t['is_code_f'] = (t.icao_code_letter == 'F').astype(int)
    return t[['icao', 'mtow_kg', 'wingspan_m', 'length_m', 'n_engines', 'engine_type', 'icao_code_letter', 'is_code_f', 'aircraft_source']]


def build(force=False):
    common.ensure_dirs()
    if not force and (DATA / 'ext_aircraft.csv').exists(): print('ext_aircraft exists; --force to rebuild'); return
    t = combine(from_openap(), load_manual()); t.to_csv(DATA / 'ext_aircraft.csv', index=False)
    tr = pd.read_parquet(DATA / 'base_train.parquet', columns=['AIRCRAFT_TYPE_mvt']); cov = tr.AIRCRAFT_TYPE_mvt.isin(set(t.icao)).mean()
    print(f'ext_aircraft: {len(t)} types ({int((t.aircraft_source == "openap").sum())} openap, {int((t.aircraft_source == "manual").sum())} manual); departures covered {cov:.3f}')
    miss = tr.AIRCRAFT_TYPE_mvt[~tr.AIRCRAFT_TYPE_mvt.isin(set(t.icao))].value_counts().head(15); print('  most frequent uncovered types:', miss.to_dict())
