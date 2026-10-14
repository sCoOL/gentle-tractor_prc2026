"""Training-only nested evaluation of sustained-motion measurements.

Upstream member predictions exclude the whole four-month block. Physical timing
experts also exclude that block. Normal blend weights use only the paired fold;
alpha uses predictions cross-fitted between its two months. Evaluation retains
every target, including copy and date-bug labels. LIRF is outside this experiment.
Historical model families/folds were explored previously: this is not a fresh
unseen estimate, nor an official-set score forecast. Never writes a submission.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import common
import tail_experts as te

em = importlib.import_module('107_extra_members')
g = importlib.import_module('128_corrected_identity_gate')
bundle = importlib.import_module('113_bundle_a36')
motion = importlib.import_module('164_ground_motion_candidates')
OUT = common.RUNS/'v49_motion_nested'
ALPHAS = [0., .1, .25, .5, 1.]
FEATURE_PATH = common.RUNS/'v49_motion_full/features.parquet'


def prepare():
    """Cache only upstream predictions, with fingerprints of all source files."""
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [g.WEIGHTS, common.DATA/'base_train.parquet',
             common.DATA/'features_v5_train.parquet']
    specs = json.loads(g.WEIGHTS.read_text())['members']
    for left, right in em.stg.BLOCKS:
        for name in sorted({n for spec in specs[3:] for n in spec.split('+')}):
            paths.append(common.RUNS/'tuning'/f'member_probe_{left}{right}_{name}.parquet')
        for name in bundle.a35.CLF:
            paths.append(common.RUNS/'tuning'/f'detector_probe_{left}{right}_{name}.parquet')
    # Frame paths are owned by fallback_context, not assumed directory names.
    for left, right in em.stg.BLOCKS:
        for fold in [left, right]:
            for suffix in ['_v5tfb5', '_v5tfb5ra', '_v5tfb5rd']:
                paths.append(em.fc._frame_path(left, right, fold, suffix))
    sources = [Path(__file__), Path(em.__file__), Path(em.rw.__file__), Path(g.__file__)]
    fingerprint = {str(p): g.digest(p) for p in paths+sources}
    manifest = OUT/'upstream_inputs.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == fingerprint, 'Upstream inputs changed'
        if (OUT/'upstream.parquet').exists():
            return pd.read_parquet(OUT/'upstream.parquet')
    print('Loading fixed upstream member caches', flush=True)
    t, R = em.strict_table(specs[3:])
    t['cell'] = em.cell_labels(t, 'adsb3', 3000)
    pc = bundle.a35._cached('detector_probe_{block}_{name}.parquet', bundle.a35.CLF, 'p').reindex(t.MVT_ID_mvt).to_numpy()
    t['actual_p'] = np.where(np.isfinite(pc), .5*(pc+t.p_fallback.to_numpy()), t.p_fallback.to_numpy())
    mask = t.has & t.fpl_matched.eq(1) & t.APT.ne('LIRF')
    d = t.loc[mask, ['MVT_ID_mvt', 'APT', 'fold', 'block', 'y', 'cell', 'actual_p',
                     'atot_minus_aobt3', 'atot_minus_sched']].rename(columns={'atot_minus_aobt3': 'a3taxi', 'atot_minus_sched': 'D'})
    raw = pd.read_parquet(common.DATA/'base_train.parquet', columns=['MVT_ID_mvt', 'month'])
    d = d.merge(raw, on='MVT_ID_mvt', validate='one_to_one')
    assert d.MVT_ID_mvt.is_unique and np.isfinite(d[['y', 'D', 'a3taxi', 'actual_p']]).all().all()
    for j in range(R.shape[1]):
        d[f'R{j}'] = R[mask, j]
    d.to_parquet(OUT/'upstream.parquet', index=False)
    manifest.write_text(json.dumps(fingerprint, indent=2))
    print('Prepared', len(d), 'non-LIRF matched rows', flush=True)
    return d


def physical_matrix(d):
    original = d.original_taxi.to_numpy(float)
    observed = d[motion.COLUMNS].to_numpy(float)
    P = np.column_stack([original, np.where(np.isfinite(observed), observed, original[:, None])])
    return P+d.adsb_q_med.to_numpy(float)[:, None]


def fit_physical(d):
    """Per-airport positive timing blend and intercept, fitted outside held block."""
    y = d.TAXITIME_SEC_mvt.to_numpy(float)
    clean = (d.is_heli.eq(0).to_numpy() & (np.abs(y-d.atot_minus_sched.to_numpy()) > 10.)
             & ~te.datebug_label(y, d.atot_minus_sched) & np.isfinite(d.original_taxi)
             & np.isfinite(d.adsb_q_med))
    d = d.loc[clean]
    states = {}
    for apt, z in d.groupby('APT', observed=True):
        if len(z) < 500:
            continue
        P, y = physical_matrix(z), z.TAXITIME_SEC_mvt.to_numpy(float)
        mean_P, mean_y = P.mean(axis=0), float(y.mean())
        w = g.simplex_fit(P-mean_P, np.ones(len(P)), y-mean_y, ridge=.001)
        states[str(apt)] = {'weights': w.tolist(), 'bias': float(mean_y-mean_P@w),
                            'original_bias': float(mean_y-mean_P[0]), 'n': len(z)}
    return states


def predict_physical(d, states):
    """Uses observations only, even if callers include a poisoned target column."""
    old = np.full(len(d), np.nan)
    new = old.copy()
    available = np.isfinite(d.original_taxi) & np.isfinite(d.adsb_q_med)
    for apt, state in states.items():
        mask = available & d.APT.eq(apt)
        P = physical_matrix(d.loc[mask])
        old[mask] = P[:, 0]+state['original_bias']
        new[mask] = P@np.asarray(state['weights'])+state['bias']
    return old, new


def compose(raw, normal, expert, alpha):
    corrected = normal+float(alpha)*np.where(np.isfinite(expert), expert-normal, 0.)
    p = g.matched_probability(raw.actual_p, raw.D)
    return np.maximum(corrected+p*(raw.D.to_numpy()-corrected), 30.)


def choose_alpha(raw, normal, expert):
    losses = {str(a): float(np.mean((compose(raw, normal, expert, a)-raw.y.to_numpy())**2)) for a in ALPHAS}
    return min(ALPHAS, key=lambda a: (losses[str(a)], a)), losses


def nested_direction(cal, score, rcols, states):
    assert not np.intersect1d(cal.MVT_ID_mvt, score.MVT_ID_mvt).size
    normal_inner = g.crossfit_normal(cal, cal[rcols].to_numpy(), cal.month)
    oi, ni = predict_physical(cal, states)
    ao, lo = choose_alpha(cal, normal_inner, oi)
    an, ln = choose_alpha(cal, normal_inner, ni)
    normal = g.normal_from(score, score[rcols].to_numpy(), g.fit_blend(cal, cal[rcols].to_numpy()))
    os, ns = predict_physical(score, states)
    baseline = compose(score, normal, ns, 0.)
    old = compose(score, normal, os, ao)
    candidate = compose(score, normal, ns, an)
    poisoned = score.assign(y=1e12, TAXITIME_SEC_mvt=-1e12)
    np.testing.assert_array_equal(predict_physical(poisoned, states)[1], ns)
    np.testing.assert_array_equal(compose(poisoned, normal, ns, an), candidate)
    ledger = score[['MVT_ID_mvt', 'APT', 'fold', 'month']].copy()
    ledger['baseline'], ledger['original_control'], ledger['candidate'] = baseline, old, candidate
    ledger['normal'], ledger['physical_original'], ledger['physical_motion'] = normal, os, ns
    state = {'alpha_original': ao, 'alpha_motion': an, 'loss_original': lo, 'loss_motion': ln,
             'poisoning_gap': 0., 'calibration_ids_sha256': hashlib.sha256(cal.MVT_ID_mvt.to_numpy().tobytes()).hexdigest()}
    return ledger, state


def metric(d, reference='baseline'):
    a = (d[reference].to_numpy()-d.y.to_numpy())**2
    b = (d.candidate.to_numpy()-d.y.to_numpy())**2
    keep = np.ones(len(d), bool)
    keep[np.argsort(-np.abs(a-b))[:min(10, len(d)-1)]] = False
    return {'n': len(d), 'baseline': float(np.sqrt(a.mean())), 'candidate': float(np.sqrt(b.mean())),
            'delta': float(np.sqrt(b.mean())-np.sqrt(a.mean())),
            'delta_without_top10': float(np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean()))}


def run(prepare_only=False):
    d = prepare()
    if prepare_only:
        return
    assert FEATURE_PATH.exists(), 'Complete script165 extraction first'
    features = pd.read_parquet(FEATURE_PATH)
    quality = pd.read_parquet(motion.e.OUT / 'adsb_features.parquet', columns=['MVT_ID_mvt', 'adsb_q_med'])
    observations = features.merge(quality, on='MVT_ID_mvt', validate='one_to_one')
    raw = pd.read_parquet(common.DATA/'base_train.parquet', columns=['MVT_ID_mvt', 'APT', 'month', 'is_heli', 'atot_minus_sched', 'TAXITIME_SEC_mvt'])
    training = raw.merge(observations, on='MVT_ID_mvt', validate='one_to_one')
    d = d.merge(observations.drop(columns='date'), on='MVT_ID_mvt', how='left', validate='one_to_one')
    rcols = [c for c in d if c.startswith('R') and c[1:].isdigit()]
    protocol = {'source_sha256': {str(p): g.digest(p) for p in [Path(__file__), Path(motion.__file__), Path(g.__file__)]},
                'feature_sha256': g.digest(FEATURE_PATH), 'upstream_sha256': g.digest(OUT/'upstream.parquet'),
                'quality_sha256': g.digest(motion.e.OUT / 'adsb_features.parquet'),
                'alpha_grid': ALPHAS, 'scope': __doc__, 'submission_qualified': False}
    path = OUT/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Protocol/input drift'
    else:
        path.write_text(json.dumps(protocol, indent=2))
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    results, selections = [], {}
    for block in sorted(d.block.unique()):
        months = sorted(d.loc[d.block.eq(block), 'month'].unique())
        base = training.loc[~training.month.isin(months)]
        state = fit_physical(base)
        (OUT/f'physical_{block}.json').write_text(json.dumps(state, indent=2))
        assert not np.intersect1d(base.MVT_ID_mvt, d.loc[d.block.eq(block), 'MVT_ID_mvt']).size
        for fold in block:
            cal = d.loc[d.block.eq(block) & d.fold.ne(fold)].copy()
            score = d.loc[d.fold.eq(fold)].copy()
            truth = score.y.copy()
            score['y'] = np.nan
            ledger, selection = nested_direction(cal, score, rcols, state)
            ledger['y'] = truth.to_numpy()
            ledger.to_parquet(OUT/f'ledger_{fold}.parquet', index=False)
            results.append(ledger)
            selections[fold] = selection
            print(fold, selection['alpha_motion'], metric(ledger), flush=True)
    ledger = pd.concat(results, ignore_index=True)
    ledger.to_parquet(OUT/'predictions.parquet', index=False)
    rows = {'pooled': metric(ledger)}
    rows.update({f'fold_{k}': metric(z) for k, z in ledger.groupby('fold')})
    rows.update({f'month_{k}': metric(z) for k, z in ledger.groupby('month')})
    novelty = {'pooled': metric(ledger, 'original_control')}
    novelty.update({f'fold_{k}': metric(z, 'original_control') for k, z in ledger.groupby('fold')})
    report = {'scope': __doc__, 'selections': selections, 'metrics': rows, 'motion_vs_original_expert': novelty,
              'airport_metrics': {str(k): metric(z) for k, z in ledger.groupby('APT', observed=True)},
              'branch_gate_pass': rows['pooled']['delta'] < 0 and all(r['delta'] <= 0 and r['delta_without_top10'] <= 0 for r in rows.values()),
              'new_information_gate_pass': all(r['delta'] < 0 and r['delta_without_top10'] < 0 for r in novelty.values()),
              'submission_qualified': False, 'ledger_sha256': g.digest(OUT/'predictions.parquet')}
    (OUT/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    run(parser.parse_args().prepare_only)
