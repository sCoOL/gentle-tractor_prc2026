"""EUROCONTROL PRU tables (G6): daily arrival ATFM delay and traffic, monthly additional taxi-out and ASMA time."""
from __future__ import annotations
import time
import numpy as np, pandas as pd, openpyxl
import common
from common import DATA, EXT, APTS
EC = EXT / 'eurocontrol'


def stream_rows(file, sheet, apt_col) -> pd.DataFrame:
    wb = openpyxl.load_workbook(EC / file, read_only=True); ws = wb[sheet]; hdr = None; rows = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0: hdr = [str(c) for c in row]; continue
        if row[apt_col] in APTS: rows.append(row)
    wb.close(); return pd.DataFrame(rows, columns=hdr)


def daily_table(atfm: pd.DataFrame, traffic: pd.DataFrame) -> pd.DataFrame:
    a = atfm.rename(columns={'APT_ICAO': 'APT'}).assign(date=lambda x: pd.to_datetime(x.FLT_DATE))
    a['atfm_arr_delay_min'] = a.DLY_APT_ARR_1.astype(float).fillna(0.0); a['atfm_arr_flights'] = a.FLT_ARR_1.astype(float); a['atfm_arr_delayed_flights'] = a.FLT_ARR_1_DLY.astype(float).fillna(0.0)
    a['atfm_delay_per_flight'] = a.atfm_arr_delay_min / a.atfm_arr_flights.replace(0, np.nan)
    t = traffic.rename(columns={'APT_ICAO': 'APT'}).assign(date=lambda x: pd.to_datetime(x.FLT_DATE)); t['traffic_dep_day'] = t.FLT_DEP_1.astype(float); t['traffic_arr_day'] = t.FLT_ARR_1.astype(float)
    return a[['APT', 'date', 'atfm_arr_delay_min', 'atfm_arr_flights', 'atfm_arr_delayed_flights', 'atfm_delay_per_flight']].merge(t[['APT', 'date', 'traffic_dep_day', 'traffic_arr_day']], on=['APT', 'date'], how='outer').sort_values(['APT', 'date']).reset_index(drop=True)


def _per_flight(df, name):
    d = df.rename(columns={'APT_ICAO': 'APT'}); d = d[d.VALID_FL.astype(float) > 0].copy(); d['year'] = d.YEAR.astype(int); d['month'] = d.MONTH_NUM.astype(int)
    d[name] = d.TOTAL_ADD_TIME_MIN.astype(float) / d.TOTAL_REF_NB_FL.astype(float).replace(0, np.nan); return d[['APT', 'year', 'month', name]]


def monthly_table(txo: pd.DataFrame, asma: pd.DataFrame) -> pd.DataFrame:
    m = _per_flight(txo, 'addl_taxi_out_min_month').merge(_per_flight(asma, 'asma_addl_min_month'), on=['APT', 'year', 'month'], how='outer'); m['ec_monthly_is_prior'] = 0
    have = set(zip(m.APT, m.year, m.month)); prior = []
    for apt in sorted(m.APT.unique()):
        for mo in range(1, 13):
            if (apt, 2026, mo) not in have and (apt, 2025, mo) in have:
                r = m[(m.APT == apt) & (m.year == 2025) & (m.month == mo)].iloc[0]; prior.append(dict(APT=apt, year=2026, month=mo, addl_taxi_out_min_month=r.addl_taxi_out_min_month, asma_addl_min_month=r.asma_addl_min_month, ec_monthly_is_prior=1))
    return pd.concat([m, pd.DataFrame(prior)], ignore_index=True).sort_values(['APT', 'year', 'month']).reset_index(drop=True)


def build(force=False):
    common.ensure_dirs(); t0 = time.time()
    if not force and (DATA / 'ext_eurocontrol_monthly.parquet').exists(): print('ext_eurocontrol exists; --force to rebuild'); return
    atfm = stream_rows('Airport_Arrival_ATFM_Delay.xlsx', 'DATA', 4); traffic = stream_rows('Airport_Traffic.xlsx', 'DATA', 4); print(f'  streamed daily rows: atfm {len(atfm):,}, traffic {len(traffic):,} [{time.time()-t0:.0f}s]', flush=True)
    txo = stream_rows('Taxi-Out_Additional_Time.xlsx', 'DATA', 3); asma = stream_rows('ASMA_Additional_Time.xlsx', 'DATA', 3)
    d = daily_table(atfm, traffic); d = d[d.date >= '2025-01-01']; d.to_parquet(DATA / 'ext_eurocontrol_daily.parquet')
    m = monthly_table(txo, asma); m = m[m.year >= 2025]; m.to_parquet(DATA / 'ext_eurocontrol_monthly.parquet')
    print(f'ext_eurocontrol: daily {len(d):,} rows to {d.date.max().date()}, monthly {len(m)} rows ({int(m.ec_monthly_is_prior.sum())} priors) [{time.time()-t0:.0f}s]')
