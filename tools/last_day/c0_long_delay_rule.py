"""The 8-13 h rule on every final row it selects (the rule of the chain's manual step, `8-13h`, METHOD.md §9): LIRF,
NM-unmatched, gate-valid, no helicopters; 8 h < D <= 13 h, or D > 13 h with a same-local-day take-off -> p = D.
Counts on 2025 (the rule as fixed before the count), rows on the final file -> runs/last_day/c0_rows.parquet
(read by build_A15L.py; rows already set by the manual step are flagged and never moved).

  py -3.14 tools/last_day/c0_long_delay_rule.py
"""
import numpy as np
import pandas as pd
import ld

LO, MID = 8 * 3600.0, 13 * 3600.0
f = pd.read_parquet(ld.A15 / 'lirf_row_audit' / 'lirf_rows.parquet')
print(f.columns.tolist())
fr = pd.read_parquet(ld.FRAME, columns=['MVT_ID_mvt', 'nm', 'bf_lobt', 'op_rate', 'vfr'])
f = f.merge(fr[['MVT_ID_mvt', 'nm']], on='MVT_ID_mvt', how='left', suffixes=('', '_fr')) if 'nm' not in f else f
sel = (~f.nm.astype(bool)) & (f.is_heli == 0) & f.valid.fillna(False).astype(bool) & (f.D > 3600)
g = f[sel].copy()
g['scope'] = ((g.D > LO) & (g.D <= MID)) | ((g.D > MID) & (g.day_gap == 0))
tr = g[g.split == 'train']
y = tr.y
k = np.select([(y - tr.D).abs() < 120, y > 70000, y <= 1800], ['copy', 'bug', 'normal'], 'long')
tr = tr.assign(k=k)
print('2025 in scope by class:', tr[tr.scope].k.value_counts().to_dict())
print(pd.crosstab(tr[tr.scope].fold, tr[tr.scope].k))
print('2025 out of scope D>8h by class (next-day >13 h):', tr[(~tr.scope) & (tr.D > LO)].k.value_counts().to_dict())
rk = g[(g.split == 'rank') & g.scope].copy()
a15 = pd.read_parquet(ld.A15_FILE).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
own = ld.manual_rows()
base = pd.read_parquet(ld.DATA / 'base_rank.parquet', columns=['MVT_ID_mvt', 'FLIGHT_mvt', 'month'])
rk = rk[['MVT_ID_mvt', 'D', 'day_gap']].merge(base, on='MVT_ID_mvt')
rk['a15'] = rk.MVT_ID_mvt.map(a15)
rk['manual_row'] = rk.MVT_ID_mvt.isin(own.MVT_ID_mvt)
rk['manual_rule'] = rk.MVT_ID_mvt.map(own.set_index('MVT_ID_mvt').rule)
rk['move'] = np.where(rk.manual_row, 0.0, rk.D - rk.a15)
pd.set_option('display.width', 250)
print(rk.sort_values(['month', 'D']).to_string())
print('rows by month', rk.month.value_counts().sort_index().to_dict())
rk.to_parquet(ld.OUT / 'c0_rows.parquet', index=False)
ld.save('c0_long_delay_rule', {'counts_2025_in_scope': tr[tr.scope].k.value_counts().to_dict(),
                               'by_fold': pd.crosstab(tr[tr.scope].fold, tr[tr.scope].k).to_dict(),
                               'out_of_scope_next_day_gt13h': tr[(~tr.scope) & (tr.D > LO)].k.value_counts().to_dict(),
                               'final_rows': rk.to_dict('records')})
