"""Daily submission batches: one file per approach, manifest, log; fitted models saved under runs/models/<approach>_<ts>/."""
from __future__ import annotations
import json, re, time
import numpy as np, pandas as pd
import common, train_eval, blend
from common import DATA, RUNS, SUBS


def load_template(): return pd.read_parquet(DATA / 'submitting.parquet')
def load_approaches(): return json.load(open(SUBS / 'approaches.json', encoding='utf-8'))


def team_name():
    p = SUBS / 'team.txt'; return p.read_text(encoding='utf-8').strip() if p.exists() and p.read_text(encoding='utf-8').strip() else 'team'


def next_version(team):
    vs = [int(m.group(1)) for f in SUBS.glob(f'{team}_v*.parquet') if (m := re.search(r'_v(\d+)\.parquet$', f.name))]
    return (max(vs) + 1) if vs else 1


def write_submission(team, version_n, rank_ids, preds):
    t = load_template(); preds = np.asarray(preds, dtype=float)
    assert len(rank_ids) == len(t) and set(np.asarray(rank_ids, dtype=float)) == set(t.MVT_ID_mvt.astype(float)), 'id set differs from template'
    if not np.isfinite(preds).all():
        bad = np.flatnonzero(~np.isfinite(preds))[:10].tolist()
        raise ValueError(f'non-finite submission predictions at row positions {bad}')
    out = pd.DataFrame({'MVT_ID_mvt': np.asarray(rank_ids, dtype=float), 'TAXITIME_SEC_mvt': np.maximum(preds, train_eval.FLOOR)})
    SUBS.mkdir(parents=True, exist_ok=True); p = SUBS / f'{team}_v{version_n}.parquet'; out.to_parquet(p, index=False); return p


def _last_scores(ap):
    """Latest Fold A / Fold B RMSE for an approach from results.csv (blend: by blend id; equal blend: by approach id in notes; else by model, feature version, groups, params), or NaN."""
    p = RUNS / 'results.csv'
    if not p.exists(): return np.nan, np.nan
    r = pd.read_csv(p)
    if ap.get('blend_equal'):
        r = r[(r.model == 'blend_equal') & (r.notes == ap.get('id', ''))]
        if r.empty: return np.nan, np.nan
        last = r[r.run_id == r.run_id.iloc[-1]]; return float(last[last.fold == 'A'].rmse_raw.mean()), float(last[last.fold == 'B'].rmse_raw.mean())
    if ap.get('blend'): r = r[r.run_id == ap['blend']]
    else:
        m0 = ap['models'][0]; r = r[(r.model == m0['id']) & (r.feature_version == ap['feature_version']) & (r.groups_on == '+'.join(ap['groups']))]
        rp = r[r.params_json == json.dumps(m0.get('params', {}))]; r = rp if len(rp) else r
    if r.empty: return np.nan, np.nan
    last = r[r.run_id == r.run_id.iloc[-1]]
    return float(last[last.fold == 'A'].rmse_raw.mean()), float(last[last.fold == 'B'].rmse_raw.mean())


def predict_approach(ap, save_models=True):
    tr, rk, man = common.load_features(ap['feature_version']); preds = {}
    mdir = RUNS / 'models' / f"{ap.get('id', 'approach')}_{time.strftime('%Y%m%d_%H%M%S')}" if save_models else None
    for m in ap['models']:
        train_eval.SAVE_DIR = (mdir / m['id']) if mdir else None; train_eval._SAVE_N[0] = 0
        st = train_eval.fit_full(m['id'], ap['feature_version'], ap['groups'], m.get('params', {}), m.get('rounds', 660)); preds[m['id']] = train_eval.predict_full(st, rk)
        if mdir: (mdir / m['id'] / 'state.json').write_text(json.dumps(dict(model_id=m['id'], feature_version=ap['feature_version'], groups=ap['groups'], cols=st['cols'], cats=st['cats'], params=m.get('params', {}), rounds=m.get('rounds', 660), heli_const=st['heli_const'])), encoding='utf-8')
    train_eval.SAVE_DIR = None
    if ap.get('blend_equal'):
        w = {'global': {m['id']: 1.0 / len(ap['models']) for m in ap['models']}, 'by_apt': {}}
        if mdir: (mdir / 'blend_weights.json').write_text(json.dumps(w), encoding='utf-8')
        return blend.apply_weights(w, preds, rk.APT.values), rk.MVT_ID_mvt.values
    if ap.get('blend'):
        w = json.load(open(RUNS / f"blend_{ap['blend']}.json"))['weights']; ids = list(w['global']); keyed = dict(zip(ids, [preds[m['id']] for m in ap['models']]))
        if mdir: (mdir / 'blend_weights.json').write_text(json.dumps(w), encoding='utf-8')
        return blend.apply_weights(w, keyed, rk.APT.values), rk.MVT_ID_mvt.values
    return preds[ap['models'][0]['id']], rk.MVT_ID_mvt.values


def append_manifest(lines):
    mp = SUBS / f'manifest_{time.strftime("%Y-%m-%d")}.md'
    block = '\n'.join(lines) + '\n'
    if mp.exists() and mp.read_text(encoding='utf-8').strip():
        old = mp.read_text(encoding='utf-8')
        block = old.rstrip() + '\n\n' + block.replace('# Submission batch', '## Submission batch', 1)
    mp.write_text(block, encoding='utf-8')


def make_batch(approach_ids, team=None):
    team = team or team_name(); aps = load_approaches(); paths = []
    lines = [f'# Submission batch {time.strftime("%Y-%m-%d %H:%M")} - team {team}', '', '| file | approach | description | fold A | fold B |', '|---|---|---|---|---|']
    for aid in approach_ids[:5]:
        ap = {**aps[aid], 'id': aid}; t0 = time.time(); preds, ids = predict_approach(ap); v = next_version(team); p = write_submission(team, v, ids, preds)
        fa, fb = _last_scores(ap)
        row = dict(file=p.name, ts=time.strftime('%Y-%m-%d %H:%M:%S'), team=team, version_n=v, approach_id=aid, feature_version=ap['feature_version'],
                   models_json=json.dumps([m['id'] for m in ap['models']]), foldA_rmse=fa, foldB_rmse=fb, result_rmse='', result_rank='', decision='', notes=ap.get('desc', ''))
        lp = SUBS / 'submission_log.csv'; pd.DataFrame([row]).to_csv(lp, mode='a', index=False, header=not lp.exists())
        lines.append(f'| {p.name} | {aid} | {ap.get("desc","")} | {fa:.1f} | {fb:.1f} |'); paths.append(p)
        print(f'  {p.name} <- {aid} ({ap.get("desc","")}) foldA {fa:.1f} foldB {fb:.1f} [{time.time()-t0:.0f}s]', flush=True)
    append_manifest(lines); return paths
