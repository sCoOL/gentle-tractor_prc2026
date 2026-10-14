"""The final chain: v45 (the member composition, tools/members) + the motion correction + every validated layer, then
the long-label gate, then (switchable) the 15 manual rows, then the export in the template order.

  py -3.14 tools/final_chain.py --variant A15 --v45 submissions_rebuild/v45_A15_<stamp>.parquet \\
      --a27-detail runs/models/final_A/R_A27/rank_detail.parquet --l3-months 6,7 all
  steps: 49 51 52 53 54 55 56 57 63 66 67 68 export   ('all' = every step in that order; a failed run resumes with the
         remaining steps, each step reads the previous cvNN of this run)

Steps and the scripts behind them (docs/METHOD.md):
  49  motion expert on the v45 component, source weight 1 (A57)  scripts/173 clean_component (saved A57 booster)
  51  NM window rules W / F (A59)                                 scripts/251 build
  52  recovered-LOBT cap R (A60)                                  scripts/256 build
  53  LSZH OPDI correction + B band (A61)                         scripts/262 frame, fit_rank LSZH; scripts/263 build
  54  EDDF stand OPDI + sched_far cap (A62)                       scripts/262 stand_fit EDDF; scripts/265 build
  55  gate retention + NM tighten (A63)                           scripts/267 rank, build
  56  neighbour lateness + LEMD cap + BEL 0.65 (A64)              scripts/270 rank nb, build(0.65)
  57  LIRF stand reuse (A65)                                      scripts/264 table, scripts/268 table + layer
  63  LIRF unmatched context (A70)                                scripts/274 frame, build
  66  push-phase L1 / F_range L2 / step-2 link L3 (A73)           scripts/278 l1frame, l3frame, build(0.75)
  67  overnight layers items 5, 4, 2, 11 (A74-A77 -> A78)         scripts/279 frame + runs/overnight_20261008/ov.py
  68  long-label gate (A79), then the manual rows                 scripts/281 build; --manual-rows / --manual-rules

Every layer's 2025 side is the validated one: each layer refits deterministically, with its fixed seeds, on its 2025
out-of-fold frame (the frozen reference frames in PRC_FROZEN/ref_layers, docs/FROZEN_ARTEFACTS.md); every frame that
mixes both years is rebuilt from the final tables and its 2025 rows are replaced by the reference rows (rt.splice_train).
Rank rows always come from the final tables. <= 4 threads per process (tools/layers/rt.py).

Outputs: runs/clean_chain_final/<variant>/cvNN.parquet, chain_log.json, splice_record.json, packages/, and
gentle-tractor_final_v68_<variant>.parquet (670,790 rows, template order of data/submitting.parquet).
"""
import argparse
import hashlib
import importlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'layers'))
import rt  # noqa: E402  (thread caps before numpy / lightgbm)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402  (rt put scripts/ on sys.path)
import final_rank  # noqa: E402

REB, REF, RUNS = rt.REB, rt.REF, rt.REB / 'runs'
REF_CC = REF / 'runs' / 'clean_chain'                  # frozen 2025-side rows of the frames built inside the chain
MOTION = RUNS / 'v49_motion_full_final' / 'features.parquet'
A57 = RUNS / 'models' / 'A57_20260924_v49'             # the saved motion booster + summary (tools/layers/build_frames.py static)
ORDER = ['49', '51', '52', '53', '54', '55', '56', '57', '63', '66', '67', '68', 'export']
PKG_RE = re.compile(r'^A\d+_\d{8}_\d{6}_v\d+$')
N_FINAL = 670790
MANUAL_ROWS_DEFAULT = common.ROOT / 'config' / 'manual_rows.csv'

HERE = V45 = DETAIL = None
VARIANT = ''
A27_DETAIL = RUNS / 'models/R_A27/rank_detail.parquet'
MANUAL_ROWS = None
MANUAL_RULES = ('8-13h', 'ita_overnight')   # the rule column of the manual-rows table
MANUAL_ON = set(MANUAL_RULES)


def CV(v):
    return HERE / ('cv67a.parquet' if str(v) == '67a' else f'cv{v}.parquet')


def _log(step, info):
    p = HERE / 'chain_log.json'
    d = json.loads(p.read_text()) if p.exists() else {}
    d[str(step)] = info
    p.write_text(json.dumps(d, indent=1, default=str))
    rt.say(f'step {step}: ' + json.dumps(info, default=str)[:1500])


def _packages():
    return {p.name for p in (common.RUNS / 'models').iterdir()}


def _move_new_packages(before: set):
    """Move the package folders a build just created (names like A61_20261009_101500_v53) to <run dir>/packages."""
    (HERE / 'packages').mkdir(exist_ok=True)
    moved = []
    for p in (common.RUNS / 'models').iterdir():
        if p.name not in before and PKG_RE.match(p.name):
            shutil.move(str(p), str(HERE / 'packages' / p.name))
            moved.append(p.name)
    return moved


def _series(path):
    s = pd.read_parquet(path)
    return s.set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)


def _summary(path):
    a = _series(path)
    return {'rows': int(len(a)), 'min': float(a.min()), 'max': float(a.max()), 'mean': float(a.mean())}


def _step_delta(prev, new):
    a, b = _series(prev), _series(new).reindex(_series(prev).index)
    d = b - a
    return {'rows_moved': int((d.abs() > 1e-9).sum()), 'sum_sq_move_M': round(float((d ** 2).sum() / 1e6), 3),
            'rms_move_s': round(float(np.sqrt((d ** 2).mean())), 3)}


def _simple(mod_name, prev_attr, next_attr, prev_v, next_v, call=lambda m: m.build()):
    m = importlib.import_module(mod_name)
    setattr(m, prev_attr, CV(prev_v))
    setattr(m, next_attr, CV(next_v))
    CV(next_v).unlink(missing_ok=True)          # the builds refuse to overwrite
    rt.wait_mem(8, f'{mod_name} build')
    before = _packages()
    call(m)
    moved = _move_new_packages(before)
    _log(next_v, {'packages': moved, **_step_delta(CV(prev_v), CV(next_v)), **_summary(CV(next_v))})


def _splice_frame(path, ref_path, name):
    new, ref = pd.read_parquet(path), pd.read_parquet(ref_path)
    out, diag = rt.splice_train(new, ref)
    out.to_parquet(path, index=False)
    rt.record(f'final.splice.{name}', diag, HERE / 'splice_record.json')
    rt.say(f'{name}: 2025 rows = reference ({len(out) - diag["rank_rows"]}), rank rows {diag["rank_rows"]}; rebuilt 2025 vs '
           f'reference: only_new {diag["only_new"]} only_ref {diag["only_ref"]} differing {diag["rows_any_column_differs"]}')


def _clean_out(m, name, keep=()):
    out = HERE / name
    out.mkdir(exist_ok=True)
    for k in keep:
        if (m.OUT / k).exists() and not (out / k).exists():
            shutil.copy2(m.OUT / k, out / k)
    m.OUT = out
    return m


# --------------------------------------------------------------------------------------------------------- inputs
def detail_frame() -> pd.DataFrame:
    """The v45 run's rank detail (`<v45>_detail.parquet`, tools/members/recompose_v45.py) + its A27 gate table
    (w, gate_valid, pb, day_shift)."""
    d = pd.read_parquet(DETAIL)
    g = pd.read_parquet(A27_DETAIL, columns=['MVT_ID_mvt', 'w', 'gate_valid', 'pb', 'day_shift'])
    g = g[[c for c in g.columns if c == 'MVT_ID_mvt' or c not in d.columns]]   # the detail's own columns win
    d = d.merge(g, on='MVT_ID_mvt', how='left', validate='one_to_one')
    if 'pred' not in d:
        d['pred'] = d.pred_a48u
    need = ['MVT_ID_mvt', 'APT', 'fpl_matched', 'atot_minus_sched', 'normal_members_a36', 'p_matched_mix', 'alpha',
            'p_unmatched_mode', 'normal_unmatched_adsb', 'w', 'gate_valid', 'pb', 'day_shift', 'pred', 'p_fallback']
    miss = [c for c in need if c not in d]
    assert not miss, f'v45 detail lacks {miss}'
    return d


# ---------------------------------------------------------------------------------------------------------- steps
def step49():
    """The A57 motion change on v45's NM-matched component, recomputed with the saved booster on the motion features of
    the final rows (scripts/173 clean_component), at source weight 1."""
    m173 = importlib.import_module('173_v49_motion_bundle')
    v45 = pd.read_parquet(V45)
    s = v45.set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)
    det = detail_frame()
    det = det[det.MVT_ID_mvt.isin(s.index)]
    rt.wait_mem(8, 'step 49 motion re-application')
    rd = m173.clean_component(det, package=A57, feature_path=MOTION)
    assert np.allclose(rd.old_v45_component.to_numpy(), s.reindex(rd.MVT_ID_mvt).to_numpy(), atol=1e-6), \
        'the v45 file is not the detail\'s v45 component on the eligible rows'
    s.loc[rd.MVT_ID_mvt.to_numpy()] = rd.new_v45_component.to_numpy()
    out = v45.copy()
    out['TAXITIME_SEC_mvt'] = s.reindex(v45.MVT_ID_mvt).to_numpy()
    assert np.isfinite(out.TAXITIME_SEC_mvt).all() and out.TAXITIME_SEC_mvt.min() >= 30
    out.to_parquet(CV(49), index=False)
    rd.to_parquet(HERE / 'v49_rank_detail.parquet', index=False)
    info = {'alpha': rd.attrs.get('alpha'), 'eligible_rows': int(len(rd)),
            'eligible_by_month': rd.month.value_counts().sort_index().to_dict(),
            'rows_changed_vs_v45': int((np.abs(rd.new_v45_component - rd.old_v45_component) > 1e-9).sum())}
    _log(49, {**info, **_step_delta(V45, CV(49)), **_summary(CV(49))})


def step51():
    _simple('251_v51_nm_window', 'V49', 'V51', 49, 51)


def step52():
    _simple('256_v52_recovered_lobt', 'V51', 'V52', 51, 52)


def _opdi():
    """262 with OUT switched to <run dir>/v53_opdi (2025 oof/nested files copied; frame + rank files rebuilt); the 266
    family reads the adsb state of that frame."""
    m = importlib.import_module('262_opdi_correction')
    out = HERE / 'v53_opdi'
    if m.OUT != out:
        out.mkdir(exist_ok=True)
        for f in m.OUT.iterdir():
            if f.name.startswith(('oof_', 'nested_')) and not (out / f.name).exists():
                shutil.copy2(f, out / f.name)
        m.OUT = out
    m.V52 = CV(52)
    importlib.import_module('266_gate_retention_screen').OPDI_FRAME = out / 'frame.parquet'
    return m


def step53_frame_only(m):
    m.frame()
    _splice_frame(m.OUT / 'frame.parquet', REF_CC / 'v53_opdi/frame.parquet', 'v53_opdi.frame')


def step53():
    m = _opdi()
    rt.wait_mem(8, '262 frame')
    step53_frame_only(m)           # rank p = cv52 (2025 p = the unchanged v52 ledger)
    m.fit_rank('LSZH', 1.0)        # alpha 1.0 in all six nested folds (A61 nested_LSZH.json)
    _simple('263_v53_build', 'V52', 'V53', 52, 53)


def step54():
    m = _opdi()
    if not (m.OUT / 'frame.parquet').exists():
        step53_frame_only(m)
    m.stand_fit('EDDF', 1.0)       # alpha 1.0 in all six folds (A62 nested_EDDF_stand.json)
    _simple('265_v54_build', 'V53', 'V54', 53, 54)


def step55():
    _opdi()
    m = _clean_out(importlib.import_module('267_v55_build'), 'v55_build', keep=('evidence.json',))
    m.V54, m.V55 = CV(54), CV(55)
    rt.wait_mem(8, '267 rank')
    m.rank()                       # rank p = v54; the correction model is refitted on the unchanged 2025 frame
    _simple('267_v55_build', 'V54', 'V55', 54, 55)


def step56():
    _opdi()
    m = _clean_out(importlib.import_module('270_v56_build'), 'v56_build', keep=('evidence.json',))
    m.V55, m.V56 = CV(55), CV(56)
    rt.wait_mem(8, '270 rank')
    m.rank('nb')                   # arm nb (A64)
    _simple('270_v56_build', 'V55', 'V56', 55, 56, call=lambda mm: mm.build(0.65))  # bel_share 0.65


def gate_table():
    """264's unmatched-lane table for this run: 2025 rows = the reference table, rank rows = this run's v45 detail."""
    g = importlib.import_module('264_v54_unmatched_gate')
    g.OUT = HERE / 'v54_gate'
    g.TRAIN_TABLE = REF / 'runs/v54_gate/table.parquet'
    g.A27_PKG = common.prc_config.REF_MAIN / 'models' / 'A27_20260917_141019'
    v45_ids = pd.read_parquet(V45, columns=['MVT_ID_mvt']).MVT_ID_mvt
    d = detail_frame()
    g.RANK_DETAIL_DF = d[d.MVT_ID_mvt.isin(v45_ids)].reset_index(drop=True)
    rt.wait_mem(8, '264 table')
    g.table()
    t = pd.read_parquet(g.OUT / 'table.parquet')
    rt.say(f'264 table: {int((t.split == "train").sum())} 2025 + {int((t.split == "rank").sum())} rank rows')
    return g


def step57():
    g = gate_table()
    m = importlib.import_module('268_v56_stand_reuse')
    m.g264 = g
    m.v55.V55 = CV(55)             # table(): 2026 v55 values
    _clean_out(m, 'v56_reuse')
    m.V56, m.V56R = CV(56), CV(57)
    CV(57).unlink(missing_ok=True)
    rt.wait_mem(8, '268 table + layer')
    m.table()
    before = _packages()
    m.layer()
    _log(57, {'packages': _move_new_packages(before), **_step_delta(CV(56), CV(57)), **_summary(CV(57))})


def lirf_rows():
    """runs/lirf_row_audit/lirf_rows.parquet for this run: the reference 2025 rows + the final file's LIRF rank rows
    (build_frames lirfbase) with `valid` from this run's 264 table and p = v57 (NaN for ids outside the input file)."""
    ref = pd.read_parquet(REF_CC / 'lirf_row_audit/lirf_rows.parquet')
    rk = pd.read_parquet(RUNS / 'lirf_row_audit' / 'lirf_rows_rank_base.parquet')
    gv = pd.read_parquet(HERE / 'v54_gate' / 'table.parquet', columns=['MVT_ID_mvt', 'split', 'valid'])
    gv = gv[gv.split == 'rank'].set_index('MVT_ID_mvt').valid
    rk['valid'] = gv.reindex(rk.MVT_ID_mvt).to_numpy()
    s = _series(CV(57))
    rk['p'] = s.reindex(rk.MVT_ID_mvt).to_numpy()
    rk['v57_file'] = rk.p
    out, diag = rt.splice_train(rk, ref)
    (HERE / 'lirf_row_audit').mkdir(exist_ok=True)
    out.to_parquet(HERE / 'lirf_row_audit' / 'lirf_rows.parquet', index=False)
    rt.say(f'lirf_rows: {len(out) - len(rk)} 2025 rows (reference) + {len(rk)} rank rows, p on {int(rk.p.notna().sum())}')
    return out


def step63():
    if not (HERE / 'v54_gate' / 'table.parquet').exists():
        gate_table()
    lirf_rows()
    m = importlib.import_module('274_unmatched_context_layer')
    m.AUD = HERE / 'lirf_row_audit'
    m.GATE_TABLE = HERE / 'v54_gate' / 'table.parquet'
    _clean_out(m, 'v63_context_layer', keep=('screen.json',))
    rt.wait_mem(8, '274 frame')
    m.frame()
    _simple('274_unmatched_context_layer', 'V57', 'V63', 57, 63)


def step66(l3_months):
    m = importlib.import_module('278_v66_bundle')
    src = m.OUT
    _clean_out(m, 'v66_bundle', keep=tuple(p.name for p in src.iterdir() if p.suffix == '.json'))
    m.V63, m.V66 = CV(63), CV(66)
    m.L3_MONTHS_2026 = tuple(l3_months)
    rt.wait_mem(8, '278 frames')
    m.l1frame()                    # 'pred' (an L1 input) = v63 on 2026 rows; 2025 = the unchanged v63 proxy
    _splice_frame(m.OUT / 'l1_frame.parquet', REF_CC / 'v66_bundle/l1_frame.parquet', 'v66_bundle.l1_frame')
    m.l3frame()                    # same for L3
    _splice_frame(m.OUT / 'l3_frame.parquet', REF_CC / 'v66_bundle/l3_frame.parquet', 'v66_bundle.l3_frame')
    _simple('278_v66_bundle', 'V63', 'V66', 63, 66, call=lambda mm: mm.build(0.75))
    _log('66_L3_months', {'L3_MONTHS_2026': list(m.L3_MONTHS_2026)})


def step67():
    """A78 on v66: the four overnight layers (279 items 5, 4, 2, 11; same configs), each fitted on its unchanged 2025
    frame and applied with 'base' = v66, moves summed, rules_v55 once on the union (280's combination)."""
    m = importlib.import_module('279_overnight_layers')
    ov = m.ov
    sub = pd.read_parquet(CV(66))
    v66 = sub.set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)
    base = v66.reindex(sub.MVT_ID_mvt).to_numpy()
    total, nlay, per = np.zeros(len(sub)), np.zeros(len(sub), int), {}
    pos = pd.Series(np.arange(len(sub)), index=sub.MVT_ID_mvt.to_numpy())
    rt.wait_mem(8, '279 overnight layers')
    for item in (5, 4, 2, 11):
        cfg = m.ITEMS[item]
        d = m.frame(item, v66)
        tr = d[d.split == 'train']
        rk = d[(d.split == 'rank') & d.MVT_ID_mvt.isin(pos.index)].reset_index(drop=True)   # rows of the input file
        assert rk.base.notna().all()
        clip = cfg.get('clip', 1800.0)
        fit = tr[tr.y <= 3600] if cfg.get('fit_body') else tr
        if cfg.get('by'):
            corr = np.zeros(len(rk))
            for g in rk[cfg['by']].unique():
                mr, mf = (rk[cfg['by']] == g).to_numpy(), (fit[cfg['by']] == g).to_numpy()
                if mf.sum() > 200:
                    corr[mr] = ov.fit_predict(fit[mf], rk[mr], cfg['X'], rounds=cfg['rounds'], clip=clip, params=cfg['params'])
        else:
            corr = ov.fit_predict(fit, rk, cfg['X'], rounds=cfg['rounds'], clip=clip, params=cfg['params'])
        new_rows = ov.rules(rk.MVT_ID_mvt, rk.base.to_numpy() + cfg['alpha'] * corr)
        i = pos[rk.MVT_ID_mvt.to_numpy()].to_numpy()
        mv = np.zeros(len(sub))
        mv[i] = new_rows - base[i]
        total += mv
        nlay += mv != 0
        per[cfg['approach']] = {'rows': int((mv != 0).sum()), 'sum_sq_move_M': round(float((mv ** 2).sum() / 1e6), 3),
                                'rows_in_layer': int(len(rk)), 'rows_2025_fit': int(len(fit))}
    u = nlay > 0
    new = base.copy()
    new[u] = ov.rules(sub.MVT_ID_mvt[u], base[u] + total[u])
    out = sub.copy()
    out['TAXITIME_SEC_mvt'] = new
    assert np.isfinite(new).all() and (new >= ov.b53.FLOOR).all()
    out.to_parquet(CV('67a'), index=False)
    _log(67, {'components': per, 'rows_changed': int(u.sum()),
              'sum_sq_move_M': round(float(((new - base) ** 2).sum() / 1e6), 3), **_summary(CV('67a'))})


def manual_rows(path) -> pd.DataFrame:
    """The manual rows of the final step (config/manual_rows.csv; docs/METHOD.md, "LIRF long-delay unmatched rows"):
    absolute values in seconds for the final file, one `rule` per row (8-13h, ita_overnight); only the
    rules switched on (--manual-rules) are returned."""
    t = pd.read_csv(path)
    assert {'MVT_ID_mvt', 'value_s', 'rule'} <= set(t.columns), f'{path}: needs MVT_ID_mvt, rule and value_s'
    unknown = set(t.rule) - set(MANUAL_RULES)
    assert not unknown, f'{path}: unknown rules {unknown}'
    t = t[t.rule.isin(MANUAL_ON)].reset_index(drop=True)
    t['MVT_ID_mvt'] = t.MVT_ID_mvt.astype(float)      # the ids are float in the challenge files
    assert t.MVT_ID_mvt.is_unique and np.isfinite(t.value_s).all() and (t.value_s >= 30).all()
    return t


def step68():
    """A79 long-label gate on the chain (base v66, v67 = cv67a), then the manual rows (unless --manual-rows off)."""
    m = importlib.import_module('281_long_gate')
    m.V66, m.V67, m.V68 = CV(66), CV('67a'), HERE / 'cv68a.parquet'
    m.CODE_FRAME = REF / 'runs/long_rows_20261008/frame.parquet'
    m.TRAIN_OP_RATE = REF / 'runs/overnight_20261008/frame_cache.parquet'
    (HERE / 'cv68a.parquet').unlink(missing_ok=True)
    rt.wait_mem(8, '281 long gate')
    before = _packages()
    m.build()
    moved = _move_new_packages(before)
    s = pd.read_parquet(HERE / 'cv68a.parquet')
    info = {'packages': moved, **_step_delta(CV('67a'), HERE / 'cv68a.parquet'), 'cv68a': _summary(HERE / 'cv68a.parquet')}
    v = s.set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.astype(float)
    if MANUAL_ROWS is None:
        info['manual_rows'] = 'off'
    else:
        keep = manual_rows(MANUAL_ROWS)
        before15 = v.reindex(keep.MVT_ID_mvt).to_numpy()
        assert np.isfinite(before15).all(), 'a manual row is missing from the file'
        v.loc[keep.MVT_ID_mvt.to_numpy()] = keep.value_s.to_numpy()
        keep.assign(chain_value_s=before15).to_parquet(HERE / 'manual_rows_applied.parquet', index=False)
        info['manual_rows'] = {'file': str(MANUAL_ROWS), 'rules_on': sorted(MANUAL_ON), 'rows': int(len(keep)),
                               'rows_by_rule': keep.rule.value_counts().to_dict(),
                               'sum_sq_move_M': round(float(((keep.value_s.to_numpy() - before15) ** 2).sum() / 1e6), 2)}
    out = s.copy()
    out['TAXITIME_SEC_mvt'] = v.reindex(s.MVT_ID_mvt).to_numpy()
    out.to_parquet(CV(68), index=False)
    _log(68, {**info, 'cv68': _summary(CV(68))})


def export():
    """The final file in the template order of data/submitting.parquet."""
    tpl = pd.read_parquet(common.DATA / 'submitting.parquet', columns=['MVT_ID_mvt'])
    sfx = f'_{VARIANT}' if VARIANT else ''
    name = f'gentle-tractor_final_v68{sfx}.parquet'
    s = _series(CV(68))
    out = pd.DataFrame({'MVT_ID_mvt': tpl.MVT_ID_mvt.to_numpy(), 'TAXITIME_SEC_mvt': s.reindex(tpl.MVT_ID_mvt).to_numpy()})
    x = out.TAXITIME_SEC_mvt.to_numpy()
    assert len(out) == N_FINAL and out.MVT_ID_mvt.is_unique and np.isfinite(x).all() and x.min() >= 30
    out.to_parquet(HERE / name, index=False)
    _log('export', {name: {'rows': int(len(out)), 'sha256': hashlib.sha256((HERE / name).read_bytes()).hexdigest(),
                           'min': float(x.min()), 'max': float(x.max()), 'mean': float(x.mean())}})


STEPS = {'49': step49, '51': step51, '52': step52, '53': step53, '54': step54, '55': step55, '56': step56, '57': step57,
         '63': step63, '67': step67, '68': step68, 'export': export}


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else REB / p


def main():
    global HERE, V45, DETAIL, VARIANT, A27_DETAIL, MANUAL_ROWS, MANUAL_ON
    ap = argparse.ArgumentParser()
    ap.add_argument('steps', nargs='+')
    ap.add_argument('--v45', required=True, help='the v45 file of tools/members/recompose_v45.py (submissions_rebuild/v45_<tag>_<stamp>.parquet)')
    ap.add_argument('--detail', help='its detail file (default <v45 stem>_detail.parquet)')
    ap.add_argument('--l3-months', default='6,7', help='2026 months of the L3 step-2 link layer of step 66 (comma list)')
    ap.add_argument('--variant', default='', help='run tag: output runs/clean_chain_final/<tag>/, file gentle-tractor_final_v68_<tag>')
    ap.add_argument('--a27-detail', help='A27 gate table (w, gate_valid, pb, day_shift) of the v45 run '
                    '(default runs/models/final_<tag>/R_A27/rank_detail.parquet when it exists)')
    ap.add_argument('--manual-rows', default=str(MANUAL_ROWS_DEFAULT),
                    help="step 68: CSV of manual rows (MVT_ID_mvt, rule, value_s) applied last, or 'off' (docs/METHOD.md)")
    ap.add_argument('--manual-rules', default=','.join(MANUAL_RULES),
                    help='step 68: comma list of the manual-row rules to apply (default all: ' + ', '.join(MANUAL_RULES) + ')')
    a = ap.parse_args()
    VARIANT = a.variant
    HERE = RUNS / 'clean_chain_final' / VARIANT if VARIANT else RUNS / 'clean_chain_final'
    if a.a27_detail:
        A27_DETAIL = _abs(a.a27_detail)
    elif VARIANT and (RUNS / 'models' / f'final_{VARIANT}' / 'R_A27' / 'rank_detail.parquet').exists():
        A27_DETAIL = RUNS / 'models' / f'final_{VARIANT}' / 'R_A27' / 'rank_detail.parquet'
    MANUAL_ON = {x.strip() for x in a.manual_rules.split(',') if x.strip()}
    assert MANUAL_ON <= set(MANUAL_RULES), f'unknown --manual-rules {MANUAL_ON - set(MANUAL_RULES)}'
    if a.manual_rows.lower() != 'off' and MANUAL_ON:
        p = Path(a.manual_rows)
        MANUAL_ROWS = p if p.is_absolute() else common.ROOT / p      # relative to the repository root
        if not MANUAL_ROWS.exists():
            raise SystemExit(f'manual rows table {MANUAL_ROWS} not found: pass --manual-rows off to build without them')
    HERE.mkdir(parents=True, exist_ok=True)
    V45 = _abs(a.v45)
    DETAIL = Path(a.detail) if a.detail else V45.with_name(V45.stem + '_detail.parquet')
    tpl = pd.read_parquet(common.DATA / 'submitting.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt
    v = pd.read_parquet(V45)
    assert len(v) == len(tpl) == N_FINAL and set(v.MVT_ID_mvt) == set(tpl), 'the v45 file must cover the final template'
    assert np.isfinite(v.TAXITIME_SEC_mvt).all() and v.TAXITIME_SEC_mvt.min() >= 30
    final_rank.N = len(v)
    steps = ORDER if a.steps == ['all'] else a.steps
    rt.say(f'=== final_chain v45={V45} rows={final_rank.N} steps={steps} (L3 months {a.l3_months})')
    rt.say(f'    variant {VARIANT or "-"}; A27 gate detail {A27_DETAIL}; manual rows {MANUAL_ROWS or "off"} {sorted(MANUAL_ON) if MANUAL_ROWS else ""}')
    (HERE / 'run_args.json').write_text(json.dumps({'variant': VARIANT, 'a27_detail': str(A27_DETAIL), 'v45': str(V45),
                                                    'detail': str(DETAIL), 'steps': steps, 'l3_months': a.l3_months,
                                                    'manual_rows': str(MANUAL_ROWS or 'off'),
                                                    'manual_rules': sorted(MANUAL_ON) if MANUAL_ROWS else [],
                                                    'started': time.strftime('%Y-%m-%d %H:%M:%S')}, indent=1))
    for s in steps:
        t0 = time.time()
        rt.say(f'--- step {s}')
        if s == '66':
            step66([int(x) for x in a.l3_months.split(',')])
        else:
            STEPS[s]()
        rt.say(f'--- step {s} done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
