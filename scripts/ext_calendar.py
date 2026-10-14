"""Calendar per airport-date (holidays, school holidays, events, sun times, DST) and local-time row features."""
from __future__ import annotations
import datetime as dt, time
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd, holidays
from astral import LocationInfo
from astral.sun import sun
import common
from common import DATA, EXT, AIRPORT_META, APTS
CURFEW = {'EGLL': (23, 6), 'LSZH': (23, 6), 'EDDF': (23, 5)}
CAL_COLS = ['is_public_holiday', 'days_to_next_holiday', 'days_since_holiday', 'is_school_holiday', 'event_flag', 'event_type', 'is_dst']
LOCAL_COLS = ['hour_local', 'minute_of_day_local', 'is_night', 'min_since_sunset', 'curfew_flag']


def _optional_csv(name):
    p = EXT / 'calendar' / name
    if not p.exists(): print(f'  note: {p} absent -> zeros (may be added by hand)'); return None
    return pd.read_csv(p)


def build_calendar(start='2025-01-01', end='2026-07-31') -> pd.DataFrame:
    days = pd.date_range(start, end, freq='D'); years = sorted({d.year for d in days} | {days[-1].year + 1}); rows = []
    school = _optional_csv('school_holidays.csv'); events = _optional_csv('events.csv')
    for apt in APTS:
        m = AIRPORT_META[apt]; tz = ZoneInfo(m['tz']); hol = holidays.country_holidays(m['country'], subdiv=m['subdiv'], years=years); hdays = sorted(hol)
        obs = LocationInfo(apt, '', 'UTC', m['lat'], m['lon']).observer
        sch = school[school.apt == apt] if school is not None else None; ev = events[events.apt == apt] if events is not None else None
        for d in days:
            day = d.date(); nxt = min((h for h in hdays if h >= day), default=None); prv = max((h for h in hdays if h <= day), default=None)
            s = sun(obs, date=day, tzinfo=dt.timezone.utc)
            is_school = int(bool(sch is not None and ((pd.to_datetime(sch.start).dt.date <= day) & (pd.to_datetime(sch.end).dt.date >= day)).any()))
            e = ev[pd.to_datetime(ev.date).dt.date == day] if ev is not None else None
            rows.append(dict(APT=apt, date=d, is_public_holiday=int(day in hol), days_to_next_holiday=(nxt - day).days if nxt else 99, days_since_holiday=(day - prv).days if prv else 99,
                             is_school_holiday=is_school, event_flag=int(e is not None and len(e) > 0), event_type=(str(e.event_type.iloc[0]) if e is not None and len(e) else 'none'),
                             dawn_utc=s['dawn'], sunrise_utc=s['sunrise'], sunset_utc=s['sunset'], dusk_utc=s['dusk'],
                             is_dst=int(dt.datetime(day.year, day.month, day.day, 12, tzinfo=tz).dst() != dt.timedelta(0))))
    return pd.DataFrame(rows)


def local_features(apt, when: pd.Series, cal: pd.DataFrame) -> pd.DataFrame:
    assert when.dt.tz is not None
    apt = np.asarray(apt).astype(str); out = pd.DataFrame(index=when.index)
    hl = np.zeros(len(when), int); ml = np.zeros(len(when), int); local_date = np.empty(len(when), dtype='datetime64[ns]')
    for a in np.unique(apt):
        m = apt == a; loc = when[m].dt.tz_convert(AIRPORT_META[a]['tz']); hl[m] = loc.dt.hour.values; ml[m] = (loc.dt.hour * 60 + loc.dt.minute).values
        local_date[m] = loc.dt.tz_localize(None).dt.floor('D').values
    out['hour_local'] = hl; out['minute_of_day_local'] = ml
    key = pd.DataFrame({'APT': apt, 'date': local_date}); j = key.merge(cal, on=['APT', 'date'], how='left')
    w = when.dt.tz_convert('UTC').values
    dusk = pd.to_datetime(j.dusk_utc, utc=True).dt.tz_localize(None).values; dawn = pd.to_datetime(j.dawn_utc, utc=True).dt.tz_localize(None).values; sunset = pd.to_datetime(j.sunset_utc, utc=True).dt.tz_localize(None).values
    w = pd.to_datetime(pd.Series(w)).dt.tz_localize(None).values if getattr(pd.Series(w).dt, 'tz', None) is not None else w
    out['is_night'] = ((w >= dusk) | (w <= dawn)).astype(int)
    out['min_since_sunset'] = np.where(w >= sunset, (w - sunset) / np.timedelta64(1, 'm'), (w - (sunset - np.timedelta64(1, 'D'))) / np.timedelta64(1, 'm'))
    h = out.hour_local.values; cf = np.zeros(len(out), int)
    for a, (s, e) in CURFEW.items(): mk = apt == a; cf[mk] = ((h[mk] >= s) | (h[mk] < e)).astype(int)
    out['curfew_flag'] = cf
    for c in CAL_COLS: out[c] = j[c].values
    return out


def build(force=False):
    common.ensure_dirs(); t0 = time.time()
    if not force and (DATA / 'ext_calendar.parquet').exists(): print('ext_calendar exists; --force to rebuild'); return
    cal = build_calendar(); cal.to_parquet(DATA / 'ext_calendar.parquet')
    print(f'ext_calendar: {len(cal):,} airport-days; public holidays per airport {cal.groupby("APT").is_public_holiday.sum().to_dict()} [{time.time()-t0:.0f}s]')
