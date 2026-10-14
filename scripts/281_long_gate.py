"""v68 = A79 = v67 + the long-label gate outside LIRF (a separate gate for the > 1 h rows).

  py -3.14 scripts/281_long_gate.py build

Population: departures (not LIRF — LIRF keeps its dedicated copy gates A50/A51 and v57/v63) whose take-off is >= 50 min
after schedule; 98 % of all 2025 labels > 1 h are in it. Three-class gate, all inputs label-free:
  q = P(short y <= 1 h | schedule copy |y - D| < 120 s | other long)   multiclass LightGBM
  S = E[y | short], Lo = E[y | other long]                              LightGBM regressors; copy value = D exactly
  p = q_short S + q_copy D + q_long Lo;  v68 = rules_v55(v67 + 0.25 (p - v66)) on the population rows.
Inputs: airport, NM status, D, take-off - LOBT / EOBT_1 / IOBT / AOBT_3, AOBT_3 - LOBT, ADS-B take-off - stand leave and
- first sighting, ADS-B first state / at stand, hour, weekday, airline code, airline NM rate, v66.
2025 evidence (runs/long_rows_20261008/: audit.py, offsets.py, gate.py, gate3.py, gate3_no_lirf.py, stack_v67.py):
on the v66 proxy alpha 0.25: -131.3 M s^2, 5/6 folds (B +12.6), forward -65.8 / -37.7 M, January -16.6 M, body share
0.33; stacked on the v67 proxy: -130.2 M, forward -64.7 / -37.9 M. LTFM loses (+41 M), every other airport gains.
No official score is used.
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

OV = common.RUNS / 'overnight_20261008'
LR = common.RUNS / 'long_rows_20261008'
sys.path.insert(0, str(common.ROOT / 'runs' / 'overnight_20261008'))   # ov.py (repository)
import ov  # noqa: E402

V66 = common.SUBS / 'gentle-tractor_v66.parquet'
V67 = common.SUBS / 'gentle-tractor_v67.parquet'
V68 = common.SUBS / 'gentle-tractor_v68.parquet'
ALPHA = 0.25
X = ['apt_c', 'st_c', 'D', 'L', 'E', 'I', 'A3', 'a3_lobt', 'adsb_taxi', 'adsb_seen', 'fs', 'at_stand', 'hour', 'dow', 'al_c', 'op_rate', 'base']
CAT = ['apt_c', 'st_c', 'al_c']
PM = dict(objective='multiclass', num_class=3, learning_rate=0.03, num_leaves=31, min_data_in_leaf=30, feature_fraction=0.8,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=12, deterministic=True, force_row_wise=True)
PR = {k: v for k, v in PM.items() if k != 'num_class'} | {'objective': 'regression'}
# final-file change: the categorical codes apt_c / st_c / al_c were computed over the frame's rows of BOTH years
# (al_c: the 150 most frequent airline codes among them). On the final file the rank rows change, so the codes could
# shift and the 2025 inputs would no longer be the evaluated gate's (the assert in build()). CODE_FRAME = the frame the
# gate was evaluated on (the frozen reference runs/long_rows_20261008/frame.parquet); its category lists are frozen and applied to
# every row. None = the original behaviour.
CODE_FRAME = None
# final-file change: op_rate (airline NM-match rate at the airport, label-free) is a mean over BOTH years' rows, so the
# final file moves it on 2025 rows too. TRAIN_OP_RATE = a table (MVT_ID_mvt, op_rate) with the values the gate was
# evaluated on (the frozen reference runs/overnight_20261008/frame_cache.parquet); 2025 rows take theirs, rank rows the current ones.
TRAIN_OP_RATE = None


def _code_maps(src: pd.DataFrame) -> dict:
    src = src[(src.t - src.sch) > 3000]
    al = src.FLIGHT_mvt.astype('string').str.extract(r'^([A-Z]{3}|[A-Z0-9]{2})', expand=False).fillna('')
    top = al.value_counts().index[:150]
    return {'apt': src.APT.astype('category').cat.categories, 'st': src.st.astype('category').cat.categories,
            'top': top, 'al': pd.Categorical(al.where(al.isin(top), 'other')).categories}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frame(v66: pd.Series) -> pd.DataFrame:
    """Same construction as runs/long_rows_20261008/gate.py (categorical codes over both years), base = v66 proxy / file."""
    b = pd.read_parquet(LR / 'frame.parquet')
    b = b.merge(ov.frame()[['MVT_ID_mvt', 'op_rate']], on='MVT_ID_mvt', how='left')
    if TRAIN_OP_RATE is not None:   # final-file change (see TRAIN_OP_RATE)
        o25 = pd.read_parquet(TRAIN_OP_RATE, columns=['MVT_ID_mvt', 'op_rate']).set_index('MVT_ID_mvt').op_rate
        tr_rows = (b.split == 'train').to_numpy()
        b.loc[tr_rows, 'op_rate'] = o25.reindex(b.MVT_ID_mvt[tr_rows]).to_numpy()
    b = b[(b.t - b.sch) > 3000].copy()
    b['D'], b['L'], b['E'], b['I'], b['A3'] = b.t - b.sch, b.t - b.lobt, b.t - b.eobt, b.t - b.iobt, b.t - b.a3
    b['a3_lobt'] = b.a3 - b.lobt
    b['adsb_taxi'], b['adsb_seen'] = b.t - b.t_start, b.t - b.t_first
    b['fs'] = b.first_state.map({'standing': 0, 'pushing': 1, 'forward': 2, 'moving': 3}).astype(float)
    b['at_stand'] = b.at_stand.astype(float)
    ts = pd.to_datetime(b.t, unit='s', utc=True)
    b['hour'], b['dow'] = ts.dt.hour, ts.dt.dayofweek
    if CODE_FRAME is None:
        b['apt_c'] = b.APT.astype('category').cat.codes.astype(float)
        b['st_c'] = b.st.astype('category').cat.codes.astype(float)
        al = b.FLIGHT_mvt.astype('string').str.extract(r'^([A-Z]{3}|[A-Z0-9]{2})', expand=False).fillna('')
        top = al.value_counts().index[:150]
        b['al_c'] = pd.Categorical(al.where(al.isin(top), 'other')).codes.astype(float)
    else:   # final-file change: frozen category lists (see CODE_FRAME)
        cm = _code_maps(pd.read_parquet(CODE_FRAME, columns=['APT', 'st', 'FLIGHT_mvt', 't', 'sch']))
        b['apt_c'] = pd.Categorical(b.APT, categories=cm['apt']).codes.astype(float)
        b['st_c'] = pd.Categorical(b.st, categories=cm['st']).codes.astype(float)
        al = b.FLIGHT_mvt.astype('string').str.extract(r'^([A-Z]{3}|[A-Z0-9]{2})', expand=False).fillna('')
        b['al_c'] = pd.Categorical(al.where(al.isin(cm['top']), 'other'), categories=cm['al']).codes.astype(float)
        assert (b.apt_c >= 0).all() and (b.al_c >= 0).all(), 'a category outside the evaluated gate lists'
    b['base'] = np.where(b.split == 'rank', v66.reindex(b.MVT_ID_mvt).to_numpy(), b.v66_pred_s)
    b = b[(b.split == 'train') | b.MVT_ID_mvt.isin(v66.index)]   # final-file change: rank rows = the V66 file's ids (no-op on the final file)
    b = b[b.APT != 'LIRF'].reset_index(drop=True)
    return b


def build():
    if V68.exists():
        raise SystemExit(f'{V68} exists; refusing to overwrite')
    s66, s67 = pd.read_parquet(V66), pd.read_parquet(V67)
    assert (s66.MVT_ID_mvt.to_numpy() == s67.MVT_ID_mvt.to_numpy()).all()
    v66 = s66.set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)
    b = frame(v66)
    tr, rk = b[b.split == 'train'].reset_index(drop=True), b[b.split == 'rank'].reset_index(drop=True)
    # same 2025 inputs as the evaluated gate (check the codes / base against gate3x_oof)
    chk = pd.read_parquet(LR / 'gate3x_oof.parquet', columns=['MVT_ID_mvt'] + X).set_index('MVT_ID_mvt').loc[tr.MVT_ID_mvt]
    assert np.allclose(chk[X].to_numpy(float), tr[X].to_numpy(float), equal_nan=True), '2025 inputs differ from the evaluated gate'
    y, D = tr.y.to_numpy(), tr.D.to_numpy()
    cls = np.select([y <= 3600.0, np.abs(y - D) < 120], [0, 1], 2)
    q = np.zeros((len(rk), 3))
    S = Lo = 0.0
    for sd in (7, 8, 9):
        q += lgb.train({**PM, 'seed': sd}, lgb.Dataset(tr[X], cls, categorical_feature=CAT), 400).predict(rk[X]) / 3
        S = S + lgb.train({**PR, 'seed': sd}, lgb.Dataset(tr[X][cls == 0], tr.y[cls == 0], categorical_feature=CAT), 400).predict(rk[X]) / 3
        Lo = Lo + lgb.train({**PR, 'seed': sd, 'min_data_in_leaf': 20}, lgb.Dataset(tr[X][cls == 2], tr.y[cls == 2], categorical_feature=CAT),
                            300).predict(rk[X]) / 3
    p = q[:, 0] * S + q[:, 1] * rk.D.to_numpy() + q[:, 2] * Lo
    move = ALPHA * (p - rk.base.to_numpy())
    v67 = s67.set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)
    ids = rk.MVT_ID_mvt.to_numpy()
    new_rows = ov.rules(ids, v67.reindex(ids).to_numpy() + move)
    new = v67.copy()
    new.loc[ids] = new_rows
    out = s67.copy()
    out['TAXITIME_SEC_mvt'] = new.reindex(out.MVT_ID_mvt).to_numpy()
    assert np.isfinite(out.TAXITIME_SEC_mvt).all() and (out.TAXITIME_SEC_mvt >= ov.b53.FLOOR).all()
    out.to_parquet(V68, index=False)
    mv = out.TAXITIME_SEC_mvt.to_numpy() - s67.TAXITIME_SEC_mvt.to_numpy()
    pkg = common.RUNS / 'models' / f"A79_{datetime.now().strftime('%Y%m%d_%H%M%S')}_v68"
    pkg.mkdir(parents=True)
    lr = rk[['MVT_ID_mvt', 'APT', 'D'] + [c for c in X if c not in ('D',)]].assign(q_short=q[:, 0], q_copy=q[:, 1], q_long=q[:, 2], S=S, Lo=Lo,
                                                                                 p_gate=p, v67=v67.reindex(ids).to_numpy(), v68=new_rows)
    lr['move_s'] = lr.v68 - lr.v67
    lr.to_parquet(pkg / 'gate_rows_2026.parquet', index=False)
    ev = {n: json.loads((LR / f'{n}.json').read_text()) for n in ('gate3_no_lirf', 'stack_v67') if (LR / f'{n}.json').exists()}
    by = lr.groupby('APT').move_s.agg(['size', 'mean', lambda s: float(np.sqrt((s ** 2).mean()))]).round(1)
    by.columns = ['rows', 'mean_move_s', 'rms_move_s']
    info = {'approach': 'A79', 'version': 68, 'base': V67.name, 'base_sha256': sha(V67), 'v66_sha256': sha(V66), 'file_sha256': sha(V68),
            'alpha': ALPHA, 'rows_in_gate_2026': int(len(rk)), 'rows_changed': int((mv != 0).sum()),
            'sum_sq_move_2026_M': round(float((mv ** 2).sum() / 1e6), 2), 'rms_move_s': round(float(np.sqrt((lr.move_s ** 2).mean())), 1),
            'mean_move_s': round(float(lr.move_s.mean()), 1), 'by_airport_2026': by.reset_index().to_dict('records'),
            'q_copy_ge_0.5_2026': int((q[:, 1] >= 0.5).sum()), 'q_long_ge_0.5_2026': int((q[:, 2] >= 0.5).sum()),
            'features': X, 'params_class': PM, 'params_reg': PR, 'seeds': [7, 8, 9], 'rows_2025_fit': int(len(tr)),
            'class_counts_2025': {int(k): int(v) for k, v in pd.Series(cls).value_counts().items()}, 'evidence_2025': ev}
    (pkg / 'build.json').write_text(json.dumps(ov._clean(info), indent=2, default=str))
    code_lr = common.ROOT / 'runs' / 'long_rows_20261008'      # the 2025 evidence scripts (repository)
    for src in (Path(__file__), code_lr / 'gate3_no_lirf.py', code_lr / 'stack_v67.py', code_lr / 'audit.py', code_lr / 'offsets.py'):
        if src.exists():
            (pkg / src.name).write_bytes(src.read_bytes())
    print(json.dumps({k: v for k, v in info.items() if k not in ('evidence_2025', 'params_class', 'params_reg')}, indent=1, default=str))
    print(f'package {pkg}')


if __name__ == '__main__':
    assert sys.argv[1:] == ['build']
    build()
