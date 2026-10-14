"""METAR-derived weather table (G5): units, sky/visibility, precipitation codes, de-icing/frost flags, rolling snow, as-of join with crosswind."""
from __future__ import annotations
import time
import numpy as np, pandas as pd
import common
from common import DATA, EXT
TOLERANCE = pd.Timedelta(hours=6)
G5 = ['temp_c', 'dewpoint_c', 'dew_spread_c', 'rh', 'wind_kt', 'gust_kt', 'wind_dir', 'crosswind_kt', 'headwind_kt', 'vis_m', 'ceiling_ft', 'lvp', 'wx_snow', 'wx_freezing',
      'wx_ts', 'wx_fog', 'wx_rain', 'wx_any_precip', 'deice_risk', 'frost_risk', 'precip_3h', 'snow_3h', 'snow_24h', 'snow_day', 'metar_age_min']
OBS_COLS = [c for c in G5 if c not in ('crosswind_kt', 'headwind_kt', 'metar_age_min')]


def derive(obs: pd.DataFrame) -> pd.DataFrame:
    d = obs.sort_values(['station', 'valid']).reset_index(drop=True).copy()
    d['temp_c'] = (d.tmpf - 32) * 5 / 9; d['dewpoint_c'] = (d.dwpf - 32) * 5 / 9; d['dew_spread_c'] = d.temp_c - d.dewpoint_c; d['rh'] = d.relh
    d['wind_kt'] = d.sknt; d['gust_kt'] = d.gust.fillna(0.0); d['wind_dir'] = d.drct
    d['vis_m'] = (d.vsby * 1609.34).clip(upper=10000)
    ceil = pd.DataFrame({i: d[f'skyl{i}'].where(d[f'skyc{i}'].isin(['BKN', 'OVC', 'VV', 'VV '])) for i in (1, 2, 3, 4)}).min(axis=1); d['ceiling_ft'] = ceil
    d['lvp'] = ((d.vis_m < 600) | (d.ceiling_ft < 200)).astype(int)
    wx = d.wxcodes.fillna('').astype(str)
    d['wx_snow'] = wx.str.contains('SN|SG').astype(int); d['wx_freezing'] = wx.str.contains('FZ').astype(int); d['wx_ts'] = wx.str.contains('TS').astype(int)
    d['wx_fog'] = wx.str.contains('FG').astype(int); d['wx_rain'] = wx.str.contains('RA|DZ').astype(int)
    d['wx_any_precip'] = wx.str.contains('RA|DZ|SN|SG|PL|GR|GS|UP|IC').astype(int)
    d['deice_risk'] = ((d.temp_c <= 3) & ((d.wx_any_precip == 1) | (d.wx_fog == 1) | (d.rh > 90))).astype(int)
    d['frost_risk'] = ((d.temp_c <= 3) & (d.dew_spread_c <= 3) & (d.wind_kt <= 6) & (d.wx_any_precip == 0)).astype(int)
    d['date'] = d.valid.dt.floor('D')
    d['snow_day'] = d.groupby(['station', 'date']).wx_snow.transform('max')
    for col, hours, name in (('wx_any_precip', 3, 'precip_3h'), ('wx_snow', 3, 'snow_3h'), ('wx_snow', 24, 'snow_24h')):
        out = np.zeros(len(d))
        for _, idx in d.groupby('station').indices.items():
            s = pd.Series(d[col].values[idx], index=pd.DatetimeIndex(d.valid.values[idx])); out[idx] = s.rolling(f'{hours}h').sum().values
        d[name] = out
    return d.drop(columns=['date'])


def runway_headings() -> dict:
    r = pd.read_csv(EXT / 'ourairports' / 'runways_10apts.csv'); out = {}
    for _, x in r.iterrows():
        out[(x.airport_ident, str(x.le_ident))] = float(x.le_heading_degT); out[(x.airport_ident, str(x.he_ident))] = float(x.he_heading_degT)
    return out


def join_weather(apt, when: pd.Series, rwy, wx: pd.DataFrame) -> pd.DataFrame:
    assert when.dt.tz is not None and wx.valid.dt.tz is not None, 'both sides of the weather join must be tz-aware UTC'
    left = pd.DataFrame({'station': np.asarray(apt).astype(str), 'when': when.dt.tz_convert('UTC').reset_index(drop=True), 'rwy': np.asarray(rwy).astype(str), '_i': np.arange(len(when))}).sort_values('when')
    right = wx.assign(valid=wx.valid.dt.tz_convert('UTC'))
    right = right[['station', 'valid'] + OBS_COLS].sort_values('valid')
    j = pd.merge_asof(left, right, left_on='when', right_on='valid', by='station', direction='backward', tolerance=TOLERANCE).sort_values('_i')
    j['metar_age_min'] = (j.when - j.valid).dt.total_seconds() / 60
    hd = runway_headings(); h = np.array([hd.get((a, r), np.nan) for a, r in zip(j.station, j.rwy)], float)
    delta = np.radians(j.wind_dir.values - h); j['crosswind_kt'] = np.abs(j.wind_kt.values * np.sin(delta)); j['headwind_kt'] = j.wind_kt.values * np.cos(delta)
    out = j[G5].reset_index(drop=True); out.index = when.index; return out


def build(force=False):
    common.ensure_dirs(); t0 = time.time()
    if not force and (DATA / 'ext_weather.parquet').exists(): print('ext_weather exists; --force to rebuild'); return
    obs = pd.read_parquet(EXT / 'metar' / 'metar_all.parquet'); d = derive(obs)
    d[['station', 'valid'] + OBS_COLS].to_parquet(DATA / 'ext_weather.parquet')
    pd.Series(runway_headings()).rename('heading_deg').rename_axis(['APT', 'rwy']).reset_index().to_csv(DATA / 'ext_runway_headings.csv', index=False)
    print(f'ext_weather: {len(d):,} observations, snow obs {int(d.wx_snow.sum())}, lvp obs {int(d.lvp.sum())}, deice-risk obs {int(d.deice_risk.sum())} [{time.time()-t0:.0f}s]', flush=True)
    print(d.groupby('station')[['wx_snow', 'lvp', 'deice_risk', 'frost_risk']].mean().round(3).to_string())
