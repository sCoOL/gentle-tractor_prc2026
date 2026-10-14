"""Post-steps on the chain's final file (docs/METHOD.md §11): shared paths, loaders, the dataset rules and the 2025 judge.

2025 (out of fold): the 2025 proxy of the full chain (PRC_FROZEN/ref_layers/runs/leads_20261008/rows.parquet,
`v67_pred_s` = the v66 proxy + the step-67 layers six-fold + the step-68 long gate six-fold). 2026: the final file of
the chain run `BASE` (runs/clean_chain_final/<BASE>/gentle-tractor_final_v68_<BASE>.parquet, 670,790 rows).
Rules: a re-implementation of scripts/267 rules_v55 with the W clip as a parameter (3,606 s = the chain's value); it is
a fixed point on both the 2025 proxy and the chain's final file.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import prc_config  # noqa: E402

BASE = 'A15'                                            # the final-chain run the post-steps start from
FIN = prc_config.RUNS / 'clean_chain_final'
OUT = prc_config.RUNS / 'last_day'                      # fits, out-of-fold frames, logs of the post-steps
A15 = FIN / BASE
A15_FILE = A15 / f'gentle-tractor_final_v68_{BASE}.parquet'
MANUAL_FILE = A15 / 'manual_rows_applied.parquet'       # the manual rows of the chain's last step (written by step 68)
GATE_TABLE = A15 / 'v54_gate' / 'table.parquet'         # the unmatched-lane gate table of the chain run (pb)
LEADS = prc_config.REF_LAYERS / 'runs' / 'leads_20261008' / 'rows.parquet'
FRAME = prc_config.RUNS / 'overnight_20261008' / 'frame_cache.parquet'
DATA = prc_config.DATA
EC_DIR = prc_config.EXTERNAL / 'eurocontrol'            # all_pre_departure_delays_<year>.csv (scripts/fetch_external.py)
PUSH = prc_config.ADSB_DIR / 'push_phases.parquet'
FOLDS = list('ABCDEF')
FLOOR, W_CLIP, EDGE = 30.0, 3606.0, 3600.0
TZ = {'EDDF': 'Europe/Berlin', 'EDDM': 'Europe/Berlin', 'EGLL': 'Europe/London', 'EHAM': 'Europe/Amsterdam',
      'LEBL': 'Europe/Madrid', 'LEMD': 'Europe/Madrid', 'LFPG': 'Europe/Paris', 'LIRF': 'Europe/Rome',
      'LSZH': 'Europe/Zurich', 'LTFM': 'Europe/Istanbul'}
BASE_COLS = ['MVT_ID_mvt', 'APT', 'fpl_matched', 'is_heli', 'a3', 'FLIGHT_mvt', 'apt_stand', 'apt_rwy']
OUT.mkdir(parents=True, exist_ok=True)


def manual_rows() -> pd.DataFrame:
    """The rows set by the chain's manual step (empty when the chain ran with --manual-rows off). They never move here."""
    if MANUAL_FILE.exists():
        return pd.read_parquet(MANUAL_FILE)
    return pd.DataFrame({'MVT_ID_mvt': pd.Series([], dtype=float), 'rule': pd.Series([], dtype=str)})


def rules(f: pd.DataFrame, p, w_clip=W_CLIP) -> np.ndarray:
    """scripts/267 rules_v55 (W / F / R / B / B_cap / floor); f needs t, lobt, bf_lobt, st."""
    out = np.asarray(p, float).copy()
    st = f.st.to_numpy()
    L = f.t.to_numpy() - f.lobt.to_numpy()
    w = np.isfinite(L)
    lo, hi = w & (out < L - w_clip), w & (out > L + w_clip)
    out[lo], out[hi] = (L - w_clip)[lo], (L + w_clip)[hi]
    Lb = f.t.to_numpy() - f.bf_lobt.to_numpy()
    m = (st == 'F_range') & (out < Lb + EDGE)
    out[m] = (Lb + EDGE)[m]
    m = (st == 'R') & (out > Lb - EDGE)
    out[m] = (Lb - EDGE)[m]
    m = (st == 'sched_far') & (np.abs(out - Lb) < EDGE)
    out[m] = (Lb - EDGE)[m]
    m = (st == 'sched_far') & (out > Lb - EDGE)
    out[m] = (Lb - EDGE)[m]
    return np.maximum(out, FLOOR)


def local_day(t_s, apt) -> np.ndarray:
    """Local calendar day (days since epoch) of epoch seconds t_s at each row's airport."""
    ts = pd.to_datetime(pd.Series(np.asarray(t_s, float)), unit='s', utc=True)
    out = np.full(len(ts), -1, dtype=np.int64)
    apt = np.asarray(apt)
    for a, tz in TZ.items():
        m = apt == a
        if m.any():
            loc = ts[m].dt.tz_convert(tz).dt.tz_localize(None).dt.floor('D')
            out[m] = ((loc - pd.Timestamp('1970-01-01')) // pd.Timedelta(days=1)).to_numpy()
    return out


def load_2025() -> pd.DataFrame:
    r = pd.read_parquet(LEADS, columns=['MVT_ID_mvt', 'split', 'fold', 'month', 'y', 'v67_pred_s', 'APT', 'fpl_matched',
                                        'is_heli', 'atot_minus_aobt3', 'adsb', 't_start', 'hour_loc', 'D', 'lane'])
    r = r[r.split == 'train'].drop(columns='split')
    f = pd.read_parquet(FRAME, columns=['MVT_ID_mvt', 'split', 't', 'lobt', 'bf_lobt', 't_sched', 'st', 'nm'])
    f = f[f.split == 'train'].drop(columns='split')
    d = r.merge(f, on='MVT_ID_mvt', how='left', validate='one_to_one').rename(columns={'v67_pred_s': 'p'})
    assert d.t.notna().all() and len(d) == 2085047
    d['year'] = 2025
    return d.reset_index(drop=True)


def load_2026() -> pd.DataFrame:
    s = pd.read_parquet(A15_FILE)
    f = pd.read_parquet(FRAME, columns=['MVT_ID_mvt', 'split', 't', 'lobt', 'bf_lobt', 't_sched', 'st', 'nm', 'month'])
    f = f[f.split == 'rank'].drop(columns='split')
    b = pd.read_parquet(DATA / 'base_rank.parquet', columns=BASE_COLS + ['atot_minus_aobt3'])
    d = s.rename(columns={'TAXITIME_SEC_mvt': 'p'}).merge(f, on='MVT_ID_mvt', how='left', validate='one_to_one') \
        .merge(b, on='MVT_ID_mvt', how='left', validate='one_to_one')
    assert len(d) == 670790 and d.t.notna().all() and d.APT.notna().all()
    d['D'] = d.t - d.t_sched
    d['lane'] = np.where(d.st == 'matched', 'matched', 'unmatched')
    d['year'] = 2026
    return d


def judge(d: pd.DataFrame, p6, pf, name: str) -> dict:
    """Strict form of the pass rule: six-fold dSSE < 0 with >= 5/6 folds better, both forward splits better, January and
    February not worse (six-fold and forward). d: 2025 rows with y, p (base), fold, month."""
    y, b = d.y.to_numpy(), d.p.to_numpy()
    p6, pf = np.asarray(p6, float), np.asarray(pf, float)
    e6 = (y - p6) ** 2 - (y - b) ** 2
    ef = (y - pf) ** 2 - (y - b) ** 2
    h1 = (d.month <= 6).to_numpy()
    fold = (pd.Series(e6).groupby(d.fold.to_numpy()).sum() / 1e6).reindex(FOLDS).fillna(0).round(3)
    mon = d.month.to_numpy()
    tot = e6.sum()
    res = {'name': name, 'rows_moved_2025_sixfold': int((np.abs(p6 - b) > 1e-6).sum()),
           'rows_moved_2025_forward': int((np.abs(pf - b) > 1e-6).sum()),
           'sixfold_dsse_M': round(tot / 1e6, 3), 'by_fold_M': fold.to_dict(), 'folds_better': int((fold < 0).sum()),
           'folds_worse': int((fold > 0).sum()),
           'forward_H1toH2_M': round(ef[~h1].sum() / 1e6, 3), 'forward_H2toH1_M': round(ef[h1].sum() / 1e6, 3),
           'jan_sixfold_M': round(e6[mon == 1].sum() / 1e6, 3), 'feb_sixfold_M': round(e6[mon == 2].sum() / 1e6, 3),
           'jan_forward_M': round(ef[mon == 1].sum() / 1e6, 3), 'feb_forward_M': round(ef[mon == 2].sum() / 1e6, 3),
           'body_share': round(float(e6[y <= 3600].sum() / tot), 3) if tot != 0 else None,
           'sum_move2_2025_M': round(float(((p6 - b) ** 2).sum() / 1e6), 3)}
    res['pass_2025'] = bool(tot < 0 and res['folds_better'] >= 5 and res['forward_H1toH2_M'] < 0 and res['forward_H2toH1_M'] < 0
                            and res['jan_sixfold_M'] <= 0 and res['feb_sixfold_M'] <= 0
                            and res['jan_forward_M'] <= 0 and res['feb_forward_M'] <= 0)
    return res


def rms_ratio(d25: pd.DataFrame, mv25, d26: pd.DataFrame, mv26, months=(1, 2, 6, 7)) -> dict:
    """2026 correction RMS per airport-month vs the 2025 out-of-fold correction RMS of the same airport-month (all rows
    of the cell; a rule that moves nothing in 2025 gets ratio inf where it moves 2026 rows)."""
    a = pd.DataFrame({'APT': d25.APT.to_numpy(), 'month': d25.month.to_numpy(), 'm2': np.asarray(mv25, float) ** 2})
    b = pd.DataFrame({'APT': d26.APT.to_numpy(), 'month': d26.month.to_numpy(), 'm2': np.asarray(mv26, float) ** 2})
    r25 = np.sqrt(a[a.month.isin(months)].groupby(['APT', 'month']).m2.mean())
    r26 = np.sqrt(b.groupby(['APT', 'month']).m2.mean())
    t = pd.DataFrame({'rms25': r25, 'rms26': r26}).fillna(0.0)
    t['ratio'] = np.where(t.rms25 > 0, t.rms26 / t.rms25.where(t.rms25 > 0, 1), np.where(t.rms26 > 0, np.inf, 0.0))
    over = t[t.ratio > 1.5]
    return {'max_ratio': float(t.ratio.max()), 'cells_over_1.5': {f'{k[0]}-{k[1]:02d}': round(float(v), 2) for k, v in over.ratio.items()},
            'overall_rms25_janfebjunjul': round(float(np.sqrt(a[a.month.isin(months)].m2.mean())), 3),
            'overall_rms26': round(float(np.sqrt(b.m2.mean())), 3)}


def save(name, obj):
    import json

    def c(o):
        if isinstance(o, dict):
            return {str(k): c(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [c(v) for v in o]
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        return o
    (OUT / f'{name}.json').write_text(json.dumps(c(obj), indent=1, default=str))
