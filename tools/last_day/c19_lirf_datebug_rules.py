"""LIRF date-bug / schedule-copy rules (docs/METHOD.md §11.5): label-free rules that decide copy or date bug for LIRF
NM-unmatched fixed-wing departures taking off on a later local day than scheduled (the date-bug domain of the chain,
tail_experts.b_domain). LIRF only: no other airport has date bugs in 2025 (1 of 13 elsewhere).

A rule's verdict replaces the chain's mixture (1 - pb) * two + pb * (k * 86,400 + N) by one side:
  copy     -> D = take-off - schedule
  date bug -> B = k * 86,400 + N   (k = local day shift, N = the chain's normal-taxi value U)

Rules, in order (first match wins):
  R0  schedule 23:59 local while the same flight number's usual schedule (mode over >= 5 departures within +-7 days) is
      at least 60 min away -> date bug                         (2025: ITY332 2025-06-25, 1 of 1)
  R4  ITA overnight long-haul (ITY6xx scheduled 21:30 or later): took off after 14:00 local -> date bug,
      took off by 12:17 local -> copy                          (2025: 2 of 2 date bugs, 7 of 7 copies)
  R1  8 h <= D < 14 h -> copy                                  (2025: no date bug between 3.3 h and 14 h; 15 of 15 copies)
  R5  Qantas (QFA...) with D >= 8 h -> copy                    (2025: QFAQFA6 2 of 2 copies)
  R6  rare operator: fewer than 300 LIRF departures over both years under its 3-letter prefix, or no flight number,
      with D >= 8 h -> copy                                    (2025: 8 of 8 copies; nearest date bug: CHH, 342)
  R2  18.5 h <= D <= 26 h -> date bug                          (2025: 5 of 5 date bugs)
  R3  D <= 26 h and the inbound arrival (turnaround link, scripts/231) landed >= 12 h before take-off -> date bug
                                                               (2025: 2 of 2 date bugs after R5)
Every count is in-sample: the rules were read off the 2025 rows one by one (2026-10-14). A row set by the chain's manual
step never moves.

  py -3.14 tools/last_day/c19_lirf_datebug_rules.py      # 2025 in-sample check -> runs/last_day/c19_lirf_rules.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ld  # noqa: E402

TZ = 'Europe/Rome'
LINKS = ld.prc_config.RUNS / 'v51_turnaround' / 'links.parquet'      # scripts/231_turnaround_links.py
RULE_NAMES = {'R0': '23:59 schedule unlike the same flight +-7 d', 'R4b': 'ITA overnight long-haul, took off after 14:00',
              'R4c': 'ITA overnight long-haul, took off by 12:17', 'R1': 'delay 8-14 h', 'R5': 'Qantas delayed to a later day',
              'R6': 'rare operator (< 300 LIRF departures) or no flight number', 'R2': 'delay 18.5-26 h',
              'R3': 'inbound landed >= 12 h before take-off'}
RARE_OPERATOR = 300


def _local(s) -> pd.Series:
    return pd.to_datetime(pd.Series(s), utc=True).dt.tz_convert(TZ)


def _arrivals(year: int) -> pd.DataFrame:
    if year == 2025:
        files = sorted(ld.DATA.glob('training_2025-*.parquet'))
        a = pd.concat([pd.read_parquet(f, columns=['MVT_ID_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt']) for f in files])
    else:
        a = pd.read_parquet(ld.DATA / 'ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt'])
    return a[a.PHASE_mvt == 'ARR'][['MVT_ID_mvt', 'MVT_TIME_UTC_mvt']]


def _inbound_landing(dep_ids, year: int) -> pd.Series:
    """Landing time of each departure's inbound arrival: the turnaround link dep -> arr, else the arrival whose first
    linked departure (by take-off time) is this one (the same lookup as the Explorer's LinkFlight_Actual)."""
    lk = pd.read_parquet(LINKS, columns=['dep_mvt', 'arr_mvt'])
    arr = _arrivals(year).set_index('MVT_ID_mvt').MVT_TIME_UTC_mvt
    base = pd.read_parquet(ld.DATA / ('base_train.parquet' if year == 2025 else 'base_rank.parquet'),
                           columns=['MVT_ID_mvt', 'MVT_TIME_UTC_mvt'])
    t = base.set_index('MVT_ID_mvt').MVT_TIME_UTC_mvt
    lk = lk[lk.arr_mvt.isin(arr.index)]
    first = lk.assign(t=lk.dep_mvt.map(t)).dropna(subset=['t']).sort_values('t').drop_duplicates('arr_mvt')
    direct = lk.drop_duplicates('dep_mvt').set_index('dep_mvt').arr_mvt
    rev = first.set_index('dep_mvt').arr_mvt
    ids = pd.Index(dep_ids)
    a = direct.reindex(ids)
    a = a.where(a.notna(), rev.reindex(ids))
    return pd.Series(arr.reindex(a.to_numpy()).to_numpy(), index=ids)


def features(year: int) -> pd.DataFrame:
    """Domain rows of one year with D, B and the label-free inputs of the rules."""
    g = pd.read_parquet(ld.GATE_TABLE)
    g = g[(g.split == ('train' if year == 2025 else 'rank')) & (g.APT == 'LIRF') & (g.day_shift >= 1) & (g.is_heli == 0)].copy()
    cols = ['MVT_ID_mvt', 'FLIGHT_mvt', 'SCHED_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'APT', 'PHASE_mvt']
    both = pd.concat([pd.read_parquet(ld.DATA / f, columns=cols) for f in ('base_train.parquet', 'base_rank.parquet')])
    lirf = both[(both.APT == 'LIRF') & (both.PHASE_mvt == 'DEP')].copy()
    lirf['S'] = _local(lirf.SCHED_TIME_UTC_mvt)
    lirf['cm'] = lirf.S.dt.hour * 60 + lirf.S.dt.minute
    d = g.merge(lirf[['MVT_ID_mvt', 'FLIGHT_mvt', 'S', 'cm', 'MVT_TIME_UTC_mvt']], on='MVT_ID_mvt', how='left', validate='one_to_one')
    assert d.S.notna().all()
    d['Dh'] = d.D / 3600.0
    d['B'] = np.clip(d.day_shift.to_numpy(float), 1, None) * 86400.0 + d.U
    tk = _local(d.MVT_TIME_UTC_mvt)
    d['sched_clock'] = d.S.dt.strftime('%H:%M')
    d['takeoff_clock'] = tk.dt.strftime('%H:%M').to_numpy()
    land = _inbound_landing(d.MVT_ID_mvt, year)
    d['turn_h'] = ((pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True) - pd.to_datetime(land.to_numpy(), utc=True))
                   .dt.total_seconds() / 3600.0).to_numpy()
    ops = lirf.FLIGHT_mvt.str[:3].value_counts()
    d['op_n'] = d.FLIGHT_mvt.str[:3].map(ops).fillna(0).astype(int).where(d.FLIGHT_mvt.notna(), 0)
    same = lirf[lirf.FLIGHT_mvt.isin(set(d.FLIGHT_mvt.dropna()))]
    n7, dev = [], []
    for i, r in d.iterrows():
        m = same[(same.FLIGHT_mvt == r.FLIGHT_mvt) & (same.MVT_ID_mvt != r.MVT_ID_mvt) & ((same.S - r.S).abs() <= pd.Timedelta(days=7))]
        if m.empty:
            n7.append(0); dev.append(np.nan); continue
        med = m.cm.mode().iloc[0]
        n7.append(len(m)); dev.append(min(abs(r.cm - med), 1440 - abs(r.cm - med)))
    d['n_same7'], d['sched_dev_min'] = n7, dev
    return d


def verdict(r) -> str | None:
    """Rule code of the first rule that fires, or None (the chain's mixture stays)."""
    f = r.FLIGHT_mvt if isinstance(r.FLIGHT_mvt, str) else ''
    if r.sched_clock == '23:59' and r.n_same7 >= 5 and r.sched_dev_min >= 60:
        return 'R0'
    if not r.Dh >= 3.5:
        return None
    if f.startswith(('ITY6', 'ITYTY6')) and r.sched_clock >= '21:30':
        if r.takeoff_clock > '14:00':
            return 'R4b'
        if r.takeoff_clock <= '12:17':
            return 'R4c'
    if 8 <= r.Dh < 14:
        return 'R1'
    if f.startswith('QFA') and r.Dh >= 8:
        return 'R5'
    if r.Dh >= 8 and (not f or r.op_n < RARE_OPERATOR):
        return 'R6'
    if 18.5 <= r.Dh <= 26:
        return 'R2'
    if r.Dh <= 26 and r.turn_h >= 12:
        return 'R3'
    return None


SIDE = {'R0': 'bug', 'R4b': 'bug', 'R4c': 'copy', 'R1': 'copy', 'R5': 'copy', 'R6': 'copy', 'R2': 'bug', 'R3': 'bug'}


def decide(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d['rule'] = [verdict(r) for r in d.itertuples()]
    d['side'] = d.rule.map(SIDE)
    d['value'] = np.where(d.side == 'bug', d.B, np.where(d.side == 'copy', d.D, np.nan))
    return d


def main():
    d = decide(features(2025))
    o = pd.read_parquet(ld.OUT / 'c18_oof_h.parquet', columns=['MVT_ID_mvt', 'p6'])
    d = d.merge(o, on='MVT_ID_mvt', how='left', validate='one_to_one')
    y, D = d.y.to_numpy(), d.D.to_numpy()
    bug = (y >= 72000) & (np.abs(y - D) > 10) & (np.mod(y, 86400) < 5400)
    d['true'] = np.where(bug, 'bug', np.where(np.abs(y - D) <= 10, 'copy', 'other'))
    s = d[d.rule.notna()]
    new = np.where(d.value.notna(), d.value, d.p6)
    d['dsse'] = ((new - y) ** 2 - (d.p6 - y) ** 2) / 1e6
    out = {'domain_rows_2025': int(len(d)), 'date_bugs_2025': int(bug.sum()),
           'by_rule': {k: {'name': RULE_NAMES[k], 'side': SIDE[k], **{c: int(v) for c, v in g.true.value_counts().items()},
                           'dSSE_vs_chain_oof_M': round(float(d.dsse[d.rule == k].sum()), 1)} for k, g in s.groupby('rule')},
           'in_sample_dSSE_vs_chain_oof_M': round(float(d.dsse.sum()), 1),
           'note': 'in-sample: the rules were read off these 2025 rows'}
    print(json.dumps(out, indent=1))
    ld.save('c19_lirf_rules', out)


if __name__ == '__main__':
    main()
