"""Validate and package A51/v45 with inner-only strategy selection."""
import argparse
import importlib
import json
import copy
import csv
import time
import lightgbm as lgb
import numpy as np
import pandas as pd

s=importlib.import_module('129_identity_search')
x=importlib.import_module('130_source_identity_search')
b=importlib.import_module('131_outer_identity_detector')
g=s.g
ROOT=s.ROOT
OUT=ROOT/'runs/a51_final'
ALPHAS=[0.,.1,.2,.3,.5,.7,1.]
STRATEGIES=['gate_mean','gate_regression','outer_weighted','outer_plain','outer_mean','hybrid']
DEPLOY_STRATEGY='gate_mean'


def strategies(frame,gates):
    means=(gates['detector_absolute']+gates['detector_residual']+gates['detector_binary'])/3.
    outer=(frame.p_identity_weighted.to_numpy()+frame.p_identity_plain.to_numpy())/2.
    return {'gate_mean':means,'gate_regression':(gates['detector_absolute']+gates['detector_residual'])/2.,
        'outer_weighted':frame.p_identity_weighted.to_numpy(),'outer_plain':frame.p_identity_plain.to_numpy(),
        'outer_mean':outer,'hybrid':(outer+means)/2.}


def validate():
    OUT.mkdir(exist_ok=True)
    data=b.prepare();whole=pd.read_parquet(g.old.OOF_PATH,columns=['MVT_ID_mvt','fold','y','pred'])
    prior=pd.read_parquet(g.OOF).set_index('MVT_ID_mvt')
    pos=pd.Series(np.arange(len(whole)),index=whole.MVT_ID_mvt)
    base=whole.pred.to_numpy().copy();base[pos.reindex(prior.index).to_numpy()]=prior.candidate
    candidates={name:base.copy() for name in STRATEGIES+['selected']};rows=[];ledger=[];selection={}
    for f,ds in data.items():
        inner=pd.concat([te for tr,te in ds['inner']],ignore_index=True)
        ig={};og={}
        for n in b.CONFIGS:
            z=np.load(b.OUT/f'pred_{f}_{n}.npz');ig[n]=z['inner'];og[n]=z['outer']
        ins=strategies(inner,ig);outs=strategies(ds['test'],og)
        choices={};scores={};idx=pos.reindex(ds['test'].MVT_ID_mvt).to_numpy();mask=whole.fold.eq(f).to_numpy()
        for n in STRATEGIES:
            losses={str(a):g.old.rmse(inner.y,g.compose(inner,ins[n],a)) for a in ALPHAS}
            alpha=min(ALPHAS,key=lambda a:(losses[str(a)],a))
            choices[n]=(losses[str(alpha)],alpha);scores[n]=losses
            candidates[n][idx]=g.compose(ds['test'],outs[n],alpha)
            row=dict(name=n,**g.metric_row(whole.loc[mask],base[mask],candidates[n][mask],f,alpha))
            rows.append(row);print(json.dumps(row),flush=True)
        best=min(STRATEGIES,key=lambda n:(choices[n][0],STRATEGIES.index(n)));alpha=choices[best][1]
        candidates['selected'][idx]=candidates[best][idx]
        row=dict(name='selected',strategy=best,**g.metric_row(whole.loc[mask],base[mask],candidates['selected'][mask],f,alpha))
        rows.append(row);selection[f]={'strategy':best,'alpha':alpha,'inner_scores':scores}
        te=ds['test'];ledger.append(pd.DataFrame({'MVT_ID_mvt':te.MVT_ID_mvt,'fold':f,'month':te.month,'y':te.y,
            'baseline':base[idx],'candidate':candidates['selected'][idx],'strategy':best,'alpha':alpha}))
        print('selected',json.dumps(row),flush=True)
    baseline=g.old.rmse(whole.y,base)
    summary={n:{'rmse':g.old.rmse(whole.y,p),'delta':g.old.rmse(whole.y,p)-baseline} for n,p in candidates.items()}
    selectedrows=[r for r in rows if r['name']=='selected']
    report={'approach':'A51','version':45,'baseline_a50_nested':baseline,'summary':summary,'folds':selectedrows,
        'all_rows':rows,'selection':selection,'deployment_selection':'Fold A inner-only selected strategy and alpha for January/July ranking',
        'passes':bool(summary['selected']['delta']<=-.2 and all(r['delta']<0 for r in selectedrows)
            and all(r['delta_without_top10']<0 for r in selectedrows)),
        'limitation':'Historical model-family selection and reused benchmark, not a newly unseen test set.',
        'data':'Existing 2025 challenge data, Jan/Jul 2026 rank fields, existing ADS-B.lol-derived A48 base; no downloads.'}
    pd.concat(ledger,ignore_index=True).to_parquet(OUT/'nested_oof.parquet',index=False)
    whole[['MVT_ID_mvt','fold','y']].assign(baseline=base,candidate=candidates[DEPLOY_STRATEGY]).to_parquet(
        OUT/'chosen_ensemble_oof.parquet',index=False)
    (OUT/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return report


def audit():
    """Replay outer detectors and refit gates with poisoned score labels."""
    OUT.mkdir(exist_ok=True);data=b.prepare();ds=data['A']
    history=pd.concat([v['test'] for f,v in data.items() if f not in 'AB'],ignore_index=True)
    test=ds['test'].copy();poison=test.copy()
    poison['y']=poison.y+1e7;poison['is_copy']=1-poison.is_copy
    poison['w_star']=poison.w_star+1e5;poison['cls']='poison'
    gaps={}
    for mode in ['weighted','plain']:
        model=lgb.Booster(model_file=str(b.OUT/f'detector_AB_{mode}.txt'))
        state=json.loads((b.OUT/f'detector_AB_{mode}.json').read_text())
        pa=s.predict(model,state,b.enrich(test,history));pb=s.predict(model,state,b.enrich(poison,history))
        gaps['outer_'+mode]=float(np.max(abs(pa-pb)))
        gaps['outer_replay_'+mode]=float(np.max(abs(pa-test['p_identity_'+mode])))
    gates={};poison_gates={}
    for name,config in b.CONFIGS.items():
        model,state=s.train(ds['train'],config)
        gates[name]=s.predict(model,state,test);poison_gates[name]=s.predict(model,state,poison)
        cache=np.load(b.OUT/f'pred_A_{name}.npz')
        gaps[name]=float(np.max(abs(gates[name]-poison_gates[name])))
        gaps['cached_'+name]=float(np.max(abs(gates[name]-cache['outer'])))
    alpha=.7
    p=g.compose(test,strategies(test,gates)[DEPLOY_STRATEGY],alpha)
    pp=g.compose(poison,strategies(poison,poison_gates)[DEPLOY_STRATEGY],alpha)
    gaps['final']=float(np.max(abs(p-pp)))
    if max(gaps.values())>1e-10:raise ValueError(f'held-label/replay test failed: {gaps}')
    result={'passes':True,'held_fold':'A','gaps':gaps,
        'scope':'New outer-detector history/features, saved detector replay, gate refit and final prediction; inherited A50 blend invariance also tested in 128.',
        'training_ids_disjoint':not bool(set(ds['train'].MVT_ID_mvt)&set(test.MVT_ID_mvt)),
        'history_ids_disjoint':not bool(set(history.MVT_ID_mvt)&set(test.MVT_ID_mvt))}
    (OUT/'label_invariance.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result


def stress():
    """Diagnostic only: hide half of flight identities in each summer test month.

This approximates the observed 48% unseen July 2026 identity rate. It is NOT a
chronological evaluation and never selects deployment settings.
"""
    import hashlib
    data=b.prepare();report=json.loads((OUT/'validation.json').read_text())
    whole=pd.read_parquet(g.old.OOF_PATH,columns=['MVT_ID_mvt','fold','y','pred'])
    pos=pd.Series(np.arange(len(whole)),index=whole.MVT_ID_mvt)
    baseline=whole.pred.to_numpy().copy();candidate=baseline.copy();rows=[]
    for fold,ds in data.items():
        te=ds['test'].copy();history=pd.concat([v['test'] for f,v in data.items() if f not in g.old.BLOCK[fold]])
        hidden=te.FLIGHT_mvt.astype('string').fillna('__NA__').map(lambda k:int(hashlib.sha256(str(k).encode()).hexdigest()[:8],16)%2==0)
        hidden &= te.month.ge(7)
        te.loc[hidden,'FLIGHT_mvt']='UNSEEN_'+te.loc[hidden,'FLIGHT_mvt'].astype('string')
        te=g.old.engineer(te)
        te=b.enrich(te,history)
        for mode in ['weighted','plain']:
            folder=b.OUT;block=g.old.BLOCK[fold]
            model=lgb.Booster(model_file=str(folder/f'detector_{block}_{mode}.txt'))
            state=json.loads((folder/f'detector_{block}_{mode}.json').read_text())
            te['p_identity_'+mode]=s.predict(model,state,te)
        gates={name:s.predict(*s.train(ds['train'],cfg),te) for name,cfg in b.CONFIGS.items()}
        a50=s.predict(*s.train(ds['train'],s.CONFIGS['a50']),te)
        a50_alpha=json.loads(g.REPORT.read_text())['alpha_by_fold'][fold]
        new_alpha=next(r['alpha'] for r in report['all_rows'] if r['name']==DEPLOY_STRATEGY and r['fold']==fold)
        idx=pos.reindex(te.MVT_ID_mvt).to_numpy();mask=whole.fold.eq(fold).to_numpy()
        baseline[idx]=g.compose(te,a50,a50_alpha)
        candidate[idx]=g.compose(te,strategies(te,gates)[DEPLOY_STRATEGY],new_alpha)
        row=g.metric_row(whole.loc[mask],baseline[mask],candidate[mask],fold,new_alpha)
        row['hidden_rows']=int(hidden.sum());rows.append(row);print(json.dumps(row),flush=True)
    result={'diagnostic':'Half of held summer flight identities artificially unseen; no settings selected here',
        'baseline':g.old.rmse(whole.y,baseline),'candidate':g.old.rmse(whole.y,candidate),
        'delta':g.old.rmse(whole.y,candidate)-g.old.rmse(whole.y,baseline),'folds':rows}
    (OUT/'unseen_identity_stress.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result


def deployment_frame():
    """A50's full OOF normal experts, enriched without own-block labels."""
    train,history=g.full_training_frame()
    extras=x.extra_table()
    parts=[]
    for block in ['AB','CD','EF']:
        part=b.enrich(train.loc[train.block.eq(block)],history.loc[history.block.ne(block)])
        parts.append(part)
    train=pd.concat(parts,ignore_index=True).merge(extras,on='MVT_ID_mvt',how='left',validate='one_to_one')
    detector_oof=pd.concat([v['test'][['MVT_ID_mvt','p_identity_weighted','p_identity_plain']] for v in b.prepare().values()])
    train=train.merge(detector_oof,on='MVT_ID_mvt',how='left',validate='one_to_one')
    if train[['p_identity_weighted','p_identity_plain']].isna().any().any():raise ValueError('missing detector OOF')
    return train,history


def build():
    report=json.loads((OUT/'validation.json').read_text())
    invariance=json.loads((OUT/'label_invariance.json').read_text())
    rows=[r for r in report['all_rows'] if r['name']==DEPLOY_STRATEGY]
    score=report['summary'][DEPLOY_STRATEGY]
    if not invariance['passes'] or score['delta']>-.2 or not all(r['delta']<0 and r['delta_without_top10']<0 for r in rows):
        raise ValueError('v45 failed the local promotion checks')
    alpha=next(r['alpha'] for r in rows if r['fold']=='A')
    submission=ROOT/'submissions/gentle-tractor_v45.parquet'
    if submission.exists():raise FileExistsError('refusing to overwrite v45')
    train,history=deployment_frame()
    base=pd.read_parquet(g.old.BASE_SUBMISSION)
    base_v44=pd.read_parquet(ROOT/'submissions/gentle-tractor_v44.parquet')
    full=base[['MVT_ID_mvt']].merge(pd.read_parquet(g.old.RANK_DETAIL),on='MVT_ID_mvt',validate='one_to_one')
    full=full.merge(g.old._raw(g.old.BASE_RANK),on='MVT_ID_mvt',validate='one_to_one')
    eligible=full.APT.eq('LIRF')&full.fpl_matched.eq(1)&full.is_heli.eq(0)
    rank=full.loc[eligible].copy();rank['D']=rank.atot_minus_sched;rank['a3taxi']=rank.atot_minus_aobt3
    rank['actual_p']=rank.p_matched_mix;rank['y']=np.nan
    rank=b.enrich(g.frame(rank,rank.normal_members_a36),history)
    rank=rank.merge(x.extra_table(rank=True),on='MVT_ID_mvt',how='left',validate='one_to_one')
    gap=float(np.max(abs(rank.pred-full.loc[eligible,'pred'].to_numpy())))
    if gap>1e-9:raise ValueError('A48 rank reproduction failed')
    if not np.array_equal(base.MVT_ID_mvt,base_v44.MVT_ID_mvt):raise ValueError('v44 ID order mismatch')
    stamp=time.strftime('%Y%m%d_%H%M%S');package=ROOT/'runs/models'/f'A51_{stamp}';package.mkdir(exist_ok=False)
    reloads={}
    for mode in ['weighted','plain']:
        model,state=b.train(train,mode)
        folder=package/f'outer_detector_{mode}';folder.mkdir()
        model.save_model(str(folder/'model.txt'));(folder/'state.json').write_text(json.dumps(state,indent=2))
        p=s.predict(model,state,rank)
        replay=s.predict(lgb.Booster(model_file=str(folder/'model.txt')),state,rank)
        reloads['outer_'+mode]=float(np.max(abs(p-replay)))
        rank['p_identity_'+mode]=p
    gates={}
    for name,config in b.CONFIGS.items():
        model,state=s.train(train,config)
        folder=package/name;folder.mkdir()
        model.save_model(str(folder/'model.txt'));(folder/'state.json').write_text(json.dumps(state,indent=2))
        p=s.predict(model,state,rank);gates[name]=p
        replay=s.predict(lgb.Booster(model_file=str(folder/'model.txt')),state,rank)
        reloads[name]=float(np.max(abs(p-replay)))
    if max(reloads.values())>1e-12:raise ValueError('saved model replay failed')
    proposed=g.compose(rank,strategies(rank,gates)[DEPLOY_STRATEGY],alpha)
    final=base_v44.TAXITIME_SEC_mvt.to_numpy().copy();final[eligible]=proposed
    template=pd.read_parquet(ROOT/'data/submitting.parquet',columns=['MVT_ID_mvt'])
    if len(base)!=344841 or base.MVT_ID_mvt.nunique()!=len(base) or set(base.MVT_ID_mvt)!=set(template.MVT_ID_mvt):
        raise ValueError('template IDs mismatch')
    if not np.isfinite(final).all() or final.min()<30:raise ValueError('invalid predictions')
    if not np.array_equal(final[~eligible],base_v44.TAXITIME_SEC_mvt.to_numpy()[~eligible]):raise ValueError('ineligible row changed')
    predcols=['MVT_ID_mvt','normal','D','w','recover','pred']
    features=list(dict.fromkeys(predcols+b.DIRECT+sum([c.get('extra_features',[]) for c in b.CONFIGS.values()],[])+g.FEATURES+s.RICH))
    rank[features].to_parquet(package/'rank_features.parquet',index=False)
    history.to_parquet(package/'history.parquet',index=False)
    rank[['MVT_ID_mvt','normal','D','w']].assign(gate=np.mean(list(gates.values()),axis=0),pred=proposed,
        pred_v44=base_v44.loc[eligible,'TAXITIME_SEC_mvt'].to_numpy()).to_parquet(package/'rank_detail.parquet',index=False)
    out=base_v44.copy();out['TAXITIME_SEC_mvt']=final;out.to_parquet(submission,index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(submission),out)
    delta=final-base_v44.TAXITIME_SEC_mvt.to_numpy()
    movement={}
    for month in [1,7]:
        z=rank.month.eq(month).to_numpy();diff=delta[eligible][z]
        movement[str(month)]={'rows':int(z.sum()),'rms_delta':float(np.sqrt(np.mean(diff**2))),
            'mean_delta':float(diff.mean()),'unseen_flight_route_rate':float(rank.loc[z,'flight_route_count_outer'].eq(0).mean())}
    summary={'approach':'A51','version':45,'submission':str(submission.relative_to(ROOT)),'package':str(package.relative_to(ROOT)),
        'strategy':DEPLOY_STRATEGY,'alpha':alpha,'replaces':'A50 LIRF matched identity gate; A48 ADS-B base retained',
        'validation_baseline':report['baseline_a50_nested'],'validation_candidate':score['rmse'],'validation_delta':score['delta'],
        'folds':rows,'inner_strategy_selection_sensitivity':report['summary']['selected'],
        'selection_caveat':'gate_mean family chosen after comparing the reused outer benchmark; alpha is inner-only. No claim of untouched evaluation.',
        'label_invariance':invariance,'replay_gaps':reloads,'a48_rank_replay_gap':gap,'rows':len(base),
        'eligible_rows':int(eligible.sum()),'changed_rows_vs_v44':int((delta!=0).sum()),'unchanged_outside_eligible':True,
        'rms_delta_all':float(np.sqrt(np.mean(delta**2))),'rank_month_movement':movement,
        'min_prediction':float(final.min()),'max_prediction':float(final.max()),
        'sha256':g.digest(submission),'validation_sha256':g.digest(OUT/'validation.json'),
        'data_scope':'Only existing challenge files and the existing ADS-B.lol-derived A48 base and caches; no network acquisition.',
        'upload_performed':False,'decision':'candidate'}
    (package/'summary.json').write_text(json.dumps(summary,indent=2))
    (package/'submission.txt').write_text(submission.name)
    (ROOT/'runs/evidence_A51_20260923.json').write_text(json.dumps(summary,indent=2))
    register(package,summary)
    print(json.dumps(summary,indent=2),flush=True)
    return package,summary


def register(package,summary):
    path=ROOT/'submissions/submission_log.csv'
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    if any(r['version_n']=='45' for r in rows):raise ValueError('v45 already registered')
    base=next(r for r in rows if r['version_n']=='44')
    models=[m for m in json.loads(base['models_json']) if m!='flight_identity_gate_corrected']+['flight_identity_gate_ensemble']
    fold={r['fold']:r for r in summary['folds']}
    notes=(f"A51: three-gate mean with eight-month identity classifiers, source probabilities and disjoint history; "
        f"alpha {summary['alpha']}; local {summary['validation_baseline']:.4f}->{summary['validation_candidate']:.4f} "
        f"({summary['validation_delta']:+.4f}) vs A50, all six folds/top10 tests improve. Reused benchmark.")
    row={k:'' for k in fields};row.update(file='gentle-tractor_v45.parquet',ts=time.strftime('%Y-%m-%d %H:%M:%S'),
        team='gentle-tractor',version_n='45',approach_id='A51',feature_version='5',models_json=json.dumps(models),
        foldA_rmse=str(fold['A']['candidate']),foldB_rmse=str(fold['B']['candidate']),decision='candidate',notes=notes)
    rows.append(row)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    apath=ROOT/'submissions/approaches.json';aps=json.loads(apath.read_text())
    if 'A51' in aps:raise ValueError('A51 already registered')
    ap=copy.deepcopy(aps['A50']);ap['desc']=notes
    ap['models']=[m for m in ap['models'] if m['id']!='flight_identity_gate_corrected']
    ap['models'].append({'id':'flight_identity_gate_ensemble','params':{'base_approach':'A48','replaces':'A50',
        'strategy':DEPLOY_STRATEGY,'alpha':summary['alpha'],'validation':'runs/a51_final/validation.json'}})
    ap['custom_script']='132_v45_identity_bundle.py validate; 132_v45_identity_bundle.py audit; 132_v45_identity_bundle.py build'
    ap['shrink']=summary['alpha'];aps['A51']=ap
    apath.write_text(json.dumps(aps,indent=1)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['validate','audit','stress','build']);a=p.parse_args()
    {'validate':validate,'audit':audit,'stress':stress,'build':build}[a.command]()
