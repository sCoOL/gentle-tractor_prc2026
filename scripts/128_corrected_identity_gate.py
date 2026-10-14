"""A50/v44: correct matched-mixture identity gate with nested blend calibration.

validate: reuse eight-month-trained member/classifier caches, but fit all blend
weights, gate models and alpha choices inside the paired calibration fold.
build: full fit on corrected A48 OOF inputs; replace A49 on the A48 rank package.
This is local packaging only; no challenge upload is implemented.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import common

old = importlib.import_module('126_flight_identity_gate')
ROOT = common.WORK   # data / runs / submissions root (PRC_WORK)
REPORT = ROOT / 'runs/corrected_identity_A50_validation.json'
OOF = ROOT / 'runs/oof/corrected_identity_A50_nested.parquet'
WEIGHTS = ROOT / 'runs/residual_weights_A48_20260922.json'
BASE_PACKAGE = old.RANK_DETAIL.parent
FEATURES = old.IDENTITY_FEATURES
ALPHAS = old.ALPHAS
PARAMS = dict(objective='regression', learning_rate=.03, num_leaves=15,
              min_data_in_leaf=150, feature_fraction=.85, bagging_fraction=.85,
              bagging_freq=1, lambda_l2=150., cat_smooth=150., cat_l2=30.,
              max_cat_to_onehot=4, verbose=-1, seed=20260922)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def matched_probability(probability, delay):
    p, delay = np.asarray(probability,float), np.asarray(delay,float)
    if not np.isfinite(p).all():
        raise ValueError('non-finite matched copy probability')
    return np.where(np.isfinite(delay) & (delay > 0), np.clip(p,0,1), 0.0)


def frame(raw, normal):
    """Build from an explicit expert and actual probability, never legacy w."""
    d = raw.copy()
    d['normal'] = np.asarray(normal,float)
    d['w'] = matched_probability(d.actual_p,d.D)
    d['gap'] = d.D-d.normal
    d['pred'] = np.maximum(d.normal + d.w*d.gap,30.0)
    d['recover'] = (d.D > 0) & np.isfinite(d.normal) & np.isfinite(d.gap) & (np.abs(d.gap)>1e-6)
    d['w_star'] = ((d.y-d.normal)/d.gap.replace(0,np.nan)).fillna(0.)
    d['loss_weight'] = d.gap**2
    return old.engineer(d)


def compose(d, gate, alpha):
    """Use the identical positive-delay mask and floor in tuning and deployment."""
    p = d.w.to_numpy() + float(alpha)*(np.clip(gate,0,1)-d.w.to_numpy())
    out = np.maximum(d.normal.to_numpy()+p*d.gap.to_numpy(),30.0)
    return np.where(d.recover.to_numpy(),out,d.pred.to_numpy())


def train_gate(train, features=FEATURES):
    ok = train.recover & np.isfinite(train.w_star) & np.isfinite(train.loss_weight) & (train.loss_weight>0)
    x, _, cats = old._prep(train.loc[ok], train.iloc[:1],features)
    model = lgb.train(PARAMS,lgb.Dataset(x,train.loc[ok,'w_star'],
                      weight=train.loc[ok,'loss_weight'],categorical_feature=cats),num_boost_round=120)
    state = {'features':features,'categories':{c:x[c].cat.categories.tolist() for c in cats},
             'params':PARAMS,'rounds':120,'train_rows':int(ok.sum()),
             'target':'unbounded (y-normal)/(D-normal); weight=(D-normal)^2'}
    return model,state


def predict_gate(model,state,score):
    x = score[state['features']].copy()
    for c,categories in state['categories'].items():
        v = x[c].astype('string').fillna('__NA__')
        x[c] = pd.Categorical(v.where(v.isin(categories),'__NA__'),categories=categories)
    return np.clip(model.predict(x),0,1)


def fit_predict(train,score,features=FEATURES):
    return predict_gate(*train_gate(train,features),score)


def simplex_fit(residuals,mixture_multiplier,target,ridge=.001):
    """Convex simplex QP, scaled analytically; check KKT rather than just status."""
    X = np.asarray(residuals,float)*np.asarray(mixture_multiplier,float)[:,None]
    y = np.asarray(target,float)
    if not len(X) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('empty/non-finite blend fit')
    k = X.shape[1]
    A,b = X.T@X,X.T@y
    H = A + ridge*np.trace(A)/k*np.eye(k)
    scale = max(float(np.max(np.abs(H))),float(np.max(np.abs(b))),1.)
    H,b = H/scale,b/scale
    result = minimize(lambda w:float(w@H@w-2*b@w),np.full(k,1/k),
        jac=lambda w:2*(H@w-b),bounds=[(0.,1.)]*k,
        constraints=[{'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones(k)}],
        method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
    w = np.clip(result.x,0,1);w/=w.sum()
    active=w>1e-7
    grad=2*(H@w-b);level=float(grad[active].mean())
    kkt=max(float(np.max(np.abs(grad[active]-level))),
            max(0.,float(level-np.min(grad[~active]))) if (~active).any() else 0.)
    if kkt>1e-5:
        raise RuntimeError(f'blend QP KKT failure: {kkt}; {result.message}')
    return w


def fit_blend(raw,R):
    p=matched_probability(raw.actual_p,raw.D)
    target=raw.y.to_numpy()-p*raw.D.to_numpy()-(1-p)*raw.a3taxi.to_numpy()
    weights={'all':simplex_fit(R,1-p,target)}
    for cell in sorted(raw.cell.unique()):
        z=raw.cell.to_numpy()==cell
        if z.sum()>=80:
            weights[cell]=simplex_fit(R[z],(1-p)[z],target[z])
    return weights


def normal_from(raw,R,weights):
    w=np.array([weights.get(c,weights['all']) for c in raw.cell])
    return raw.a3taxi.to_numpy()+np.sum(R*w,axis=1)


def crossfit_normal(raw,R,groups):
    groups=np.asarray(groups)
    if len(np.unique(groups))<2:
        raise ValueError('blend cross-fit needs two disjoint groups')
    out=np.full(len(raw),np.nan)
    for held in np.unique(groups):
        z=groups==held
        weights=fit_blend(raw.loc[~z],R[~z])
        out[z]=normal_from(raw.loc[z],R[z],weights)
    if not np.isfinite(out).all():
        raise ValueError('incomplete blend cross-fit')
    return out


def add_history(d,history):
    if set(d.MVT_ID_mvt)&set(history.MVT_ID_mvt):
        raise ValueError('history overlaps the scored/training frame')
    return old.add_outer_rates(d,history)


def nested_direction(cal,Rcal,test,Rtest,history,features=FEATURES):
    """Test labels are not used by any member blend, gate or alpha fit here."""
    inner_gate=np.full(len(cal),np.nan);inner_frame=[]
    # For inner alpha selection, refit the blend WITHOUT the inner held month.
    for held in sorted(cal.month.unique()):
        z=cal.month.to_numpy()==held
        tr,te=cal.loc[~z].copy(),cal.loc[z].copy()
        # Within the remaining calibration month, generate out-of-day weights.
        day_groups=tr.SCHED_TIME_UTC_mvt.dt.day.to_numpy()%2
        trframe=add_history(frame(tr,crossfit_normal(tr,Rcal[~z],day_groups)),history)
        teframe=add_history(frame(te,normal_from(te,Rcal[z],fit_blend(tr,Rcal[~z]))),history)
        inner_gate[z]=fit_predict(trframe,teframe,features)
        inner_frame.append(teframe.assign(_position=np.flatnonzero(z)))
    inner=pd.concat(inner_frame).sort_values('_position')
    scores={str(a):old.rmse(inner.y,compose(inner,inner_gate,a)) for a in ALPHAS}
    alpha=min(ALPHAS,key=lambda a:(scores[str(a)],a))
    training=add_history(frame(cal,crossfit_normal(cal,Rcal,cal.month)),history)
    scoring=add_history(frame(test,normal_from(test,Rtest,fit_blend(cal,Rcal))),history)
    gate=fit_predict(training,scoring,features)
    return scoring,compose(scoring,gate,alpha),alpha,scores


def load_data(with_members=False):
    # Only rank-available raw columns enter the feature construction.
    raw=old._raw(old.BASE_TRAIN)
    anatomy=pd.read_parquet(old.ANATOMY_PATH)
    d=anatomy.merge(raw,on='MVT_ID_mvt',validate='one_to_one',suffixes=('','_raw'))
    d=d[d.APT.eq('LIRF')&d.fpl_matched.eq(1)&d.is_heli.eq(0)].copy().reset_index(drop=True)
    d['block']=d.fold.map(old.BLOCK);d['is_copy']=d.cls.eq('copy').astype(int)
    d['cell']=np.select([d.adsb_at_stand.eq(1),d.adsb_censored.eq(1)],['T','C'],default='N')
    pparts=[];member_parts=[]
    specs=json.loads(WEIGHTS.read_text())['members']
    for block in ['AB','CD','EF']:
        ids=d.loc[d.block.eq(block),'MVT_ID_mvt'].to_numpy()
        frames=pd.concat([pd.read_parquet(ROOT/'runs/oof'/f'strict_block_{block}_fold{f}_v5tfb5.parquet',
                          columns=['MVT_ID_mvt','normal_pred','p_fallback']) for f in block]).set_index('MVT_ID_mvt').reindex(ids)
        pcs=[]
        for name in ['set_clf','self_clf','set_clf2']:
            pcs.append(pd.read_parquet(ROOT/'runs/tuning'/f'detector_probe_{block}_{name}.parquet').set_index('MVT_ID_mvt').p.reindex(ids))
        pc=pd.concat(pcs,axis=1).mean(axis=1)
        prob=np.where(np.isfinite(pc),.5*(pc+frames.p_fallback),frames.p_fallback)
        pparts.append(pd.Series(prob,index=ids))
        if not with_members:
            continue
        a3=d.set_index('MVT_ID_mvt').a3taxi.reindex(ids).to_numpy()
        nr0=frames.normal_pred.to_numpy()
        def read_normal(suffix):
            return pd.concat([pd.read_parquet(ROOT/'runs/oof'/f'strict_block_{block}_fold{f}_v5tfb5{suffix}.parquet',
                   columns=['MVT_ID_mvt','normal_pred']) for f in block]).set_index('MVT_ID_mvt').normal_pred.reindex(ids).to_numpy()
        nra,nrd=read_normal('ra'),read_normal('rd')
        cols=[nr0-a3,4*(nrd-a3)-(nr0-a3)-2*(nra-a3),nra-a3]
        for spec in specs[3:]:
            vals=[pd.read_parquet(ROOT/'runs/tuning'/f'member_probe_{block}_{n}.parquet')
                  .set_index('MVT_ID_mvt').r.reindex(ids).to_numpy() for n in spec.split('+')]
            cols.append(np.mean(vals,axis=0))
        member_parts.append(pd.DataFrame(np.column_stack(cols),index=ids,columns=specs))
    d['actual_p']=pd.concat(pparts).reindex(d.MVT_ID_mvt).to_numpy()
    if not np.isfinite(d.actual_p).all():
        raise ValueError('missing matched probabilities')
    if not with_members:
        return d
    R=pd.concat(member_parts).reindex(d.MVT_ID_mvt).to_numpy()
    if not np.isfinite(R).all():
        raise ValueError('missing member predictions')
    return d,R


def metric_row(whole,baseline,candidate,fold,alpha):
    baseline,candidate=np.asarray(baseline),np.asarray(candidate)
    y=whole.y.to_numpy()
    effect=(baseline-y)**2-(candidate-y)**2
    keep=np.ones(len(y),bool);keep[np.argsort(-np.abs(effect))[:10]]=False
    return {'fold':fold,'n':len(y),'alpha':float(alpha),'baseline':old.rmse(y,baseline),
            'candidate':old.rmse(y,candidate),'delta':old.rmse(y,candidate)-old.rmse(y,baseline),
            'delta_without_top10':old.rmse(y[keep],candidate[keep])-old.rmse(y[keep],baseline[keep])}


def validate():
    d,R=load_data(with_members=True)
    whole=pd.read_parquet(old.OOF_PATH)
    allbase=whole.pred.to_numpy().copy();allpred=allbase.copy()
    positions=pd.Series(np.arange(len(whole)),index=whole.MVT_ID_mvt)
    rows=[];alphas={};inner={};ledger=[];invariance=None
    for f in 'ABCDEF':
        block=old.BLOCK[f];calmask=d.fold.eq(old.PAIR[f]).to_numpy();testmask=d.fold.eq(f).to_numpy()
        cal,test=d.loc[calmask].copy(),d.loc[testmask].copy()
        history=d[d.block.ne(block)].copy()
        score,pred,alpha,scores=nested_direction(cal,R[calmask],test,R[testmask],history)
        if f=='A':
            poisoned=test.copy();poisoned['y']=poisoned.y+1e6;poisoned['is_copy']=1-poisoned.is_copy
            check,other,a2,_=nested_direction(cal,R[calmask],poisoned,R[testmask],history)
            invariance=float(np.max(np.abs(other-pred)))
            if invariance>1e-10 or a2!=alpha:
                raise ValueError('held-label invariance failed')
        idx=positions.reindex(score.MVT_ID_mvt).to_numpy()
        allbase[idx]=score.pred;allpred[idx]=pred
        mask=whole.fold.eq(f).to_numpy()
        row=metric_row(whole.loc[mask],allbase[mask],allpred[mask],f,alpha)
        rows.append(row);alphas[f]=alpha;inner[f]=scores
        ledger.append(pd.DataFrame({'MVT_ID_mvt':score.MVT_ID_mvt,'fold':f,'y':score.y,
                                  'baseline':score.pred,'candidate':pred,'normal':score.normal,'p_matched':score.w}))
        print('nested',json.dumps(row),flush=True)
    report={'approach':'A50','protocol':'paired-fold gate with nested calibration of all 16 member blend weights and alpha',
            'scope':'LIRF matched non-helicopters; cached base members/classifiers exclude each outer four-month block',
            'limitation':'Historical model-family/hyperparameter selection is reused; this is not a newly unseen test set.',
            'baseline':old.rmse(whole.y,allbase),'candidate':old.rmse(whole.y,allpred),
            'delta':old.rmse(whole.y,allpred)-old.rmse(whole.y,allbase),'folds':rows,
            'alpha_by_fold':alphas,'inner_alpha_scores':inner,'held_label_invariance_max_gap':invariance,
            'floor':30.,'source_weights_sha256':digest(WEIGHTS),
            'selection_rule':'Use Fold A inner-selected alpha for the January/July ranking population.',
            'passes':bool(all(x['delta']<0 for x in rows) and old.rmse(whole.y,allpred)<old.rmse(whole.y,allbase)-.3)}
    pd.concat(ledger,ignore_index=True).to_parquet(OOF,index=False)
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return report


def full_training_frame():
    d=load_data()
    p=matched_probability(d.actual_p,d.D)
    normal=np.where((p<.999999)&(d.pred>30.),(d.pred-p*d.D)/(1-p),d.a3taxi)
    out=frame(d,normal)
    # Floor-active and saturated rows cannot be inverted; do not fit on invented experts.
    out['recover'] &= (p<.999999)&(d.pred.to_numpy()>30.)
    out['pred']=d.pred.to_numpy()
    parts=[add_history(out[out.block.eq(b)],out[out.block.ne(b)]) for b in ['AB','CD','EF']]
    return pd.concat(parts,ignore_index=True).sort_values('MVT_ID_mvt'),d


def build():
    report=json.loads(REPORT.read_text())
    if not report['passes'] or report['held_label_invariance_max_gap']>1e-10:
        raise ValueError('nested validation did not pass')
    alpha=float(report['alpha_by_fold']['A'])
    if alpha<=0:
        raise ValueError('Fold A selected no correction')
    submission=ROOT/'submissions/gentle-tractor_v44.parquet'
    if submission.exists():
        raise FileExistsError('v44 already exists; refusing to overwrite')
    train,history=full_training_frame()
    base=pd.read_parquet(old.BASE_SUBMISSION)
    prior=pd.read_parquet(old.V43_SUBMISSION)
    detail=base[['MVT_ID_mvt']].merge(pd.read_parquet(old.RANK_DETAIL),on='MVT_ID_mvt',validate='one_to_one')
    raw=old._raw(old.BASE_RANK)
    full=detail.merge(raw,on='MVT_ID_mvt',validate='one_to_one')
    if full.month.isna().any() or sorted(full.month.unique())!=[1,7]:
        raise ValueError('ranking population is not January/July')
    if not np.array_equal(base.MVT_ID_mvt,prior.MVT_ID_mvt):
        raise ValueError('v42/v43 ID order mismatch')
    reproduction=float(np.max(np.abs(np.maximum(detail.pred,30.)-base.TAXITIME_SEC_mvt)))
    if reproduction>1e-9:
        raise ValueError('A48 package does not reproduce v42')
    eligible=full.APT.eq('LIRF')&full.fpl_matched.eq(1)&full.is_heli.eq(0)
    rank=full.loc[eligible].copy();rank['D']=rank.atot_minus_sched
    rank['a3taxi']=rank.atot_minus_aobt3;rank['actual_p']=rank.p_matched_mix;rank['y']=np.nan
    rank=frame(rank,rank.normal_members_a36)
    repro=float(np.max(np.abs(rank.pred-detail.loc[eligible,'pred'])))
    if repro>1e-9:
        raise ValueError(f'actual matched expert does not reproduce A48: {repro}')
    rank=add_history(rank,history)
    model,state=train_gate(train)
    gate=predict_gate(model,state,rank)
    new=base.TAXITIME_SEC_mvt.to_numpy().copy();new[eligible]=compose(rank,gate,alpha)
    if not np.isfinite(new).all() or (new<30).any():
        raise ValueError('invalid submission values')
    if not np.array_equal(new[~eligible],base.TAXITIME_SEC_mvt.to_numpy()[~eligible]):
        raise ValueError('ineligible row changed')
    if len(base)!=344841 or base.MVT_ID_mvt.nunique()!=len(base):
        raise ValueError('wrong submission ID population')
    template=pd.read_parquet(ROOT/'data/submitting.parquet',columns=['MVT_ID_mvt'])
    if set(template.MVT_ID_mvt)!=set(base.MVT_ID_mvt):
        raise ValueError('template ID mismatch')
    stamp=time.strftime('%Y%m%d_%H%M%S');package=ROOT/'runs/models'/f'A50_{stamp}'
    package.mkdir(exist_ok=False)
    model_dir=package/'flight_identity_gate_corrected';model_dir.mkdir()
    model.save_model(str(model_dir/'gate_lgb.txt'))
    (model_dir/'state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    reloaded=lgb.Booster(model_file=str(model_dir/'gate_lgb.txt'))
    reload_gap=float(np.max(np.abs(predict_gate(reloaded,state,rank)-gate)))
    if reload_gap>1e-12:
        raise ValueError('saved model replay mismatch')
    rank[['MVT_ID_mvt']+FEATURES].to_parquet(package/'rank_gate_features.parquet',index=False)
    history[['MVT_ID_mvt','airline','FLIGHT_mvt','ADES_mvt','is_copy']].to_parquet(package/'history.parquet',index=False)
    diff=new-base.TAXITIME_SEC_mvt.to_numpy();diff43=new-prior.TAXITIME_SEC_mvt.to_numpy()
    detail['pred_a48']=detail.pred;detail['pred_v43']=prior.TAXITIME_SEC_mvt.to_numpy()
    detail['pred']=new;detail['identity_gate_alpha']=np.where(eligible,alpha,0.)
    detail['identity_gate_probability']=np.nan;detail.loc[eligible,'identity_gate_probability']=gate
    detail.to_parquet(package/'rank_detail.parquet',index=False)
    out=base.copy();out['TAXITIME_SEC_mvt']=new
    out.to_parquet(submission,index=False)
    check=pd.read_parquet(submission)
    if not check.equals(out):
        raise ValueError('submission round-trip mismatch')
    summary={'approach':'A50','version':44,'base_approach':'A48','replaces':'A49/v43',
        'base_package':str(BASE_PACKAGE.relative_to(ROOT)),'submission':str(submission.relative_to(ROOT)),
        'alpha':alpha,'rows':len(base),'eligible_rows':int(eligible.sum()),'changed_rows_vs_v42':int((diff!=0).sum()),
        'changed_rows_vs_v43':int((diff43!=0).sum()),'unchanged_outside_eligible':True,
        'rms_delta_all_vs_v42':float(np.sqrt(np.mean(diff**2))),
        'rms_delta_eligible_vs_v42':float(np.sqrt(np.mean(diff[eligible]**2))),
        'max_abs_delta_vs_v42':float(np.max(abs(diff))),
        'min_prediction':float(new.min()),'max_prediction':float(new.max()),
        'a48_submission_reproduction_gap':reproduction,'matched_expert_reproduction_gap':repro,
        'model_reload_max_gap':reload_gap,'submission_sha256':digest(submission),
        'validation':report,'validation_sha256':digest(REPORT),
        'upload_performed':False,'decision':'candidate'}
    (package/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (package/'submission.txt').write_text(submission.name,encoding='utf-8')
    (ROOT/'runs/evidence_A50_20260922.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    register(package,summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='validation'},indent=2),flush=True)
    print('package:',package,flush=True)
    return package,summary


def register(package,summary):
    """Register the generated local artifacts."""
    log_path=ROOT/'submissions/submission_log.csv'
    with log_path.open(newline='',encoding='utf-8') as handle:
        reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
    if any(r['version_n']=='44' for r in rows):
        raise ValueError('v44 already in submission registry')
    base=next(r for r in rows if r['version_n']=='42')
    ev=summary['validation'];folds={r['fold']:r for r in ev['folds']}
    notes=(f"A50 replaces A49 on A48 with actual matched probability/normal expert and an unbounded weighted gate target; "
           f"alpha {summary['alpha']:.2f}; nested blend+gate validation {ev['baseline']:.4f}->{ev['candidate']:.4f} "
           f"({ev['delta']:+.4f}), all six folds improve; local candidate")
    new={k:'' for k in fields}
    new.update(file='gentle-tractor_v44.parquet',ts=time.strftime('%Y-%m-%d %H:%M:%S'),team='gentle-tractor',
               version_n='44',approach_id='A50',feature_version='5',
               models_json=json.dumps(json.loads(base['models_json'])+['flight_identity_gate_corrected']),
               foldA_rmse=str(folds['A']['candidate']),foldB_rmse=str(folds['B']['candidate']),
               decision='candidate',notes=notes)
    rows.append(new)
    with log_path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    ap_path=ROOT/'submissions/approaches.json';aps=json.loads(ap_path.read_text())
    if 'A50' in aps:
        raise ValueError('A50 approach already registered')
    ap=copy.deepcopy(aps['A48']);ap['desc']=notes
    ap['models'].append({'id':'flight_identity_gate_corrected','params':{
        'base_approach':'A48','replaces_approach':'A49','airport':'LIRF','alpha':summary['alpha'],
        'validation':str(REPORT.relative_to(ROOT)),'training_target':'unbounded weighted squared error'}})
    ap['custom_script']='128_corrected_identity_gate.py validate; 128_corrected_identity_gate.py build'
    ap['shrink']=summary['alpha'];aps['A50']=ap
    ap_path.write_text(json.dumps(aps,indent=1)+'\n',encoding='utf-8')
    import submit
    submit.append_manifest(['# Candidate A50 '+time.strftime('%Y-%m-%d %H:%M'),'',
        '| file | alpha | nested pooled baseline | nested pooled candidate | status |',
        '|---|---:|---:|---:|---|',
        f"| gentle-tractor_v44.parquet | {summary['alpha']:.2f} | {ev['baseline']:.4f} | {ev['candidate']:.4f} | local candidate |",'',notes])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['validate','build'])
    args=parser.parse_args()
    validate() if args.command=='validate' else build()
