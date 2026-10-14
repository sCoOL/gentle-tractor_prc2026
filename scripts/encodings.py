"""Out-of-fold, time-weighted target encodings (G8) and fallback-rate encodings for the unmatched lane."""
from __future__ import annotations
import re, time
import numpy as np, pandas as pd
import common
from common import DATA
PRIOR_N = 20
SPECS = [
    dict(name='te_stand_rwy_mean', keys=['apt_stand', 'apt_rwy'], target='y', stat='mean', subset=None),
    dict(name='te_stand_rwy_p10', keys=['apt_stand', 'apt_rwy'], target='y', stat='p10', subset=None),
    dict(name='te_stand_rwy_n', keys=['apt_stand', 'apt_rwy'], target='y', stat='count', subset=None),
    dict(name='te_pier_rwy_mean', keys=['apt_pier', 'apt_rwy'], target='y', stat='mean', subset=None),
    dict(name='te_operator_apt_mean', keys=['APT', 'operator'], target='y', stat='mean', subset=None),
    dict(name='te_actype_apt_mean', keys=['APT', 'actype'], target='y', stat='mean', subset=None),
    dict(name='te_airline_apt_mean', keys=['APT', 'airline'], target='y', stat='mean', subset=None),
    dict(name='fb_rate_stand', keys=['apt_stand'], target='fallback', stat='mean', subset=None),
    dict(name='fb_rate_airline_apt', keys=['APT', 'airline'], target='fallback', stat='mean', subset=None),
    dict(name='fb_rate_airline_apt_unm', keys=['APT', 'airline'], target='fallback', stat='mean', subset='unmatched'),
    dict(name='fb_rate_stand_unm', keys=['apt_stand'], target='fallback', stat='mean', subset='unmatched'),
    dict(name='fb_rate_apt_hour_unm', keys=['APT', 'hour_utc'], target='fallback', stat='mean', subset='unmatched')]


def _target(df, target):
    if target == 'y':
        return df.TAXITIME_SEC_mvt.astype(float)
    if 'fallback' in df:
        return df.fallback.astype(float)
    return ((df.b - df.sched_s).abs() <= 10).astype(float)


def _fit(src, keys, target, stat):
    """Table keyed by `keys` with the smoothed statistic and the prior; src rows already restricted to the subset."""
    t = _target(src, target); w = np.where(src.month.values >= 10, 1.0, 0.5)
    g = pd.DataFrame({'k': list(zip(*[src[k].astype(str).values for k in keys])), 't': t.values, 'w': w, 'tw': t.values * w})
    if stat == 'count': tab = g.groupby('k').size().astype(float) / max(src.month.nunique(), 1); prior = 0.0   # per fitted month: OOF fits see 9-10 months, the ranking fit sees 12
    elif stat == 'p10': tab = g.groupby('k').t.quantile(0.10); prior = float(np.nanpercentile(t, 10)) if len(t) else np.nan
    else:
        prior = float(np.average(t, weights=w)) if len(t) else np.nan
        s = g.groupby('k').agg(sw=('w', 'sum'), stw=('tw', 'sum')); tab = (s.stw + PRIOR_N * prior) / (s.sw + PRIOR_N)
    return tab, prior


def _lookup(df, keys, tab, prior):
    k = pd.Series(list(zip(*[df[c].astype(str).values for c in keys])), index=df.index)
    return k.map(tab).astype(float).fillna(prior)


def _subset(df, subset):
    return df if subset is None else df[df.fpl_matched.values == (0 if subset == 'unmatched' else 1)]


def seasonal_pair_folds(month):
    """0..5 fold ids: Jan+Jul, Feb+Aug, ..., Jun+Dec."""
    return (np.asarray(month, dtype=int) - 1) % 6


def oof_encode(train, rank, spec, n_folds=5, fold_ids=None):
    keys, target, stat, subset = spec['keys'], spec['target'], spec['stat'], spec['subset']
    out = pd.Series(np.nan, index=train.index)
    fold = train.month.values % n_folds if fold_ids is None else np.asarray(fold_ids)
    if len(fold) != len(train): raise ValueError('fold_ids length must match train')
    for k in np.unique(fold):
        te = fold == k
        if not te.any(): continue
        tab, prior = _fit(_subset(train[~te], subset), keys, target, stat); out[te] = _lookup(train[te], keys, tab, prior).values
    tab, prior = _fit(_subset(train, subset), keys, target, stat)
    return out, _lookup(rank, keys, tab, prior)


def encode_frames(train, rank, fold_ids=None):
    """Pure frame API used by the nested feature builder and leakage tests."""
    a = pd.DataFrame({'MVT_ID_mvt': train.MVT_ID_mvt.values})
    b = pd.DataFrame({'MVT_ID_mvt': rank.MVT_ID_mvt.values})
    for spec in SPECS:
        x, y = oof_encode(train, rank, spec, fold_ids=fold_ids)
        a[spec['name']] = x.values; b[spec['name']] = y.values
    return a, b


def encode_score(train, score):
    """Fit every encoding on a disjoint training frame and score another frame.

    Unlike :func:`encode_frames`, this does no inner OOF work. It is intended for
    an outer validation block whose labels are wholly absent from ``train``.
    """
    out = pd.DataFrame({'MVT_ID_mvt': score.MVT_ID_mvt.values})
    for spec in SPECS:
        src = _subset(train, spec['subset'])
        tab, prior = _fit(src, spec['keys'], spec['target'], spec['stat'])
        out[spec['name']] = _lookup(score, spec['keys'], tab, prior).values
    return out


def _pier_of(s):
    m = re.match(r'^([A-Za-z]+)', s); return m.group(1) if m else s[:1]


def build(force=False, use_unseen_map=True):
    common.ensure_dirs(); t0 = time.time()
    if not force and (DATA / 'enc_rank.parquet').exists(): print('encodings exist; --force to rebuild'); return
    cols = ['MVT_ID_mvt', 'APT', 'apt_stand', 'apt_rwy', 'apt_pier', 'operator', 'actype', 'airline', 'month', 'hour_utc', 'fpl_matched', 'b', 'sched_s', 'TAXITIME_SEC_mvt']
    tr = pd.read_parquet(DATA / 'base_train.parquet', columns=cols); rk = pd.read_parquet(DATA / 'base_rank.parquet', columns=[c for c in cols if c not in ('b', 'TAXITIME_SEC_mvt')])
    p = DATA / 'det_unseen_stands.csv'
    if use_unseen_map and p.exists():
        um = pd.read_csv(p); um = um[um.method != 'none']; mp = {(a, s): m for a, s, m in zip(um.APT, um.stand.astype(str), um.mapped_stand.astype(str))}
        key = list(zip(rk.APT, rk.apt_stand.str.split('|').str[1].astype(str))); hit = np.array([k in mp for k in key])
        if hit.any():
            new_stand = [f'{a}|{mp[k]}' for a, k in zip(rk.APT.values[hit], [key[i] for i in np.flatnonzero(hit)])]
            rk.loc[hit, 'apt_stand'] = new_stand
            rk.loc[hit, 'apt_pier'] = [s.split('|')[0] + '|' + _pier_of(s.split('|')[1]) for s in new_stand]
        print(f'  unseen-stand map applied to {int(hit.sum()):,} ranking rows', flush=True)
    a = pd.DataFrame({'MVT_ID_mvt': tr.MVT_ID_mvt.values}); b = pd.DataFrame({'MVT_ID_mvt': rk.MVT_ID_mvt.values})
    for spec in SPECS:
        x, y = oof_encode(tr, rk, spec); a[spec['name']] = x.values; b[spec['name']] = y.values
        print(f"  {spec['name']}: train mean {np.nanmean(x):.2f} rank mean {np.nanmean(y):.2f} [{time.time()-t0:.0f}s]", flush=True)
    a.to_parquet(DATA / 'enc_train.parquet'); b.to_parquet(DATA / 'enc_rank.parquet'); print(f'encodings written: {a.shape}, {b.shape}')
