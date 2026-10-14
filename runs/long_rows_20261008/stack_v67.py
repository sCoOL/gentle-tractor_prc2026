"""Step 5: the non-LIRF long gate (alpha 0.25) stacked on the v67 proxy (v66 proxy + the four v67 layers, combo_oof).
Gate move = 0.25 (p_gate - v66), added to v67's move on shared rows, rules_v55 once. Judged against the v67 proxy."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OV = Path(__file__).resolve().parents[1] / 'overnight_20261008'
sys.path.insert(0, str(OV))
import ov  # noqa: E402

A = 0.25
p = ov.load_proxy()
p = p[p.split == 'train'][['MVT_ID_mvt', 'fold', 'month', 'apt', 'y', 'v66_pred_s']]
c = pd.read_parquet(ov.OUT / 'combo_oof.parquet', columns=['MVT_ID_mvt', 'p6', 'pf'])
g = pd.read_parquet(ov.common.RUNS / 'long_rows_20261008' / 'gate3x_oof.parquet', columns=['MVT_ID_mvt', 'p_gate6', 'p_gatef', 'cls'])
d = p.merge(c, on='MVT_ID_mvt', how='left').merge(g, on='MVT_ID_mvt', how='left')
d = d[d.p_gate6.notna()].reset_index(drop=True)
b66 = d.v66_pred_s.to_numpy()
b67_6 = d.p6.fillna(d.v66_pred_s).to_numpy()
b67_f = d.pf.fillna(d.v66_pred_s).to_numpy()
d['base'] = b67_6
s6 = ov.rules(d.MVT_ID_mvt, b67_6 + A * (d.p_gate6.to_numpy() - b66))
sf = ov.rules(d.MVT_ID_mvt, b67_f + A * (d.p_gatef.to_numpy() - b66))
# judge against v67 (six-fold base for both; forward compares to the forward v67)
y = d.y.to_numpy()
r = ov.judge(d, s6, sf, 'gate on v67')
ef = (y - sf) ** 2 - (y - b67_f) ** 2
h1 = (d.month <= 6).to_numpy()
r['forward_H1toH2_M'], r['forward_H2toH1_M'] = round(ef[~h1].sum() / 1e6, 2), round(ef[h1].sum() / 1e6, 2)
r['shared_rows_with_v67'] = int(d.p6.notna().sum())
e = (y - s6) ** 2 - (y - b67_6) ** 2
r['by_apt_M'] = (pd.Series(e).groupby(d.apt).sum() / 1e6).round(1).to_dict()
r['by_class_M'] = (pd.Series(e).groupby(d.cls).sum() / 1e6).round(1).to_dict()
print({k: r[k] for k in ('rows_2025', 'shared_rows_with_v67', 'sixfold_dsse_M', 'by_fold_M', 'folds_better', 'forward_H1toH2_M',
                         'forward_H2toH1_M', 'jan_sixfold_M', 'jul_sixfold_M', 'body_share', 'by_apt_M', 'by_class_M')})
ov.save('../long_rows_20261008/stack_v67', r)
