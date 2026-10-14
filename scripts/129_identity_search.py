"""Current-data A51 research: nested alternatives to the shipped A50 gate.

No network access. Keeps the A50 member/classifier isolation and refits its
normal blend inside every split. Configuration and shrinkage are selected only
on inner months, never on the outer scored labels. Research results reuse the
historical benchmark and are not a new unseen test set.
"""
from __future__ import annotations
import argparse
import importlib
import json
import pickle
from pathlib import Path
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

g = importlib.import_module('128_corrected_identity_gate')
old = g.old
ROOT = g.ROOT
OUT = ROOT / 'runs/a51'
ALPHAS = [0., .1, .2, .3, .5, .7, 1.]
CONFIGS = {
    'a50': dict(mode='absolute', rich=False, rounds=120, leaves=15),
    'rich_absolute': dict(mode='absolute', rich=True, rounds=120, leaves=15),
    'rich_long': dict(mode='absolute', rich=True, rounds=300, leaves=15),
    'rich_small': dict(mode='absolute', rich=True, rounds=180, leaves=7),
    'residual': dict(mode='residual', rich=False, rounds=120, leaves=15),
    'rich_residual': dict(mode='residual', rich=True, rounds=120, leaves=15),
    'rich_residual_small': dict(mode='residual', rich=True, rounds=180, leaves=7),
    'rich_binary': dict(mode='binary', rich=True, rounds=180, leaves=15),
}
RATE_KEYS = {
    'flight': ['FLIGHT_mvt'],
    'flight_weekday': ['FLIGHT_mvt', 'dow'],
    'flight_season': ['FLIGHT_mvt', 'summer'],
    'flight_stand': ['FLIGHT_mvt', 'STAND_mvt'],
    'flight_delay': ['FLIGHT_mvt', 'delay_bin'],
    'route_delay': ['airline', 'ADES_mvt', 'delay_bin'],
    'airline_stand': ['airline', 'STAND_mvt'],
    'stand_hour': ['STAND_mvt', 'sched_hour'],
}
RICH = [x for k in RATE_KEYS for x in ('rate_'+k, 'count_'+k)] + ['summer', 'delay_bin', 'sched_hour']


def keys(d):
    d=d.copy()
    d['summer']=d.month.between(4,10).astype(int)
    d['delay_bin']=np.digitize((d.D-d.a3taxi).fillna(0),[0,300,900,1800,3600,7200])
    d['sched_hour']=d.SCHED_TIME_UTC_mvt.dt.hour
    return d


def rich_history(frame,history):
    if set(frame.MVT_ID_mvt)&set(history.MVT_ID_mvt):
        raise ValueError('rich history overlaps frame')
    out=keys(frame);h=keys(history)
    prior=float(h.is_copy.mean())
    for name,cols in RATE_KEYS.items():
        stats=h.groupby(cols,dropna=False,observed=True).is_copy.agg(['sum','size']).reset_index()
        strength=10. if name in ['stand_hour','airline_stand'] else 5.
        stats['rate_'+name]=(stats['sum']+strength*prior)/(stats['size']+strength)
        stats['count_'+name]=stats['size']
        out=out.merge(stats[cols+['rate_'+name,'count_'+name]],on=cols,how='left',validate='many_to_one')
        out['rate_'+name]=out['rate_'+name].fillna(prior)
        out['count_'+name]=out['count_'+name].fillna(0.)
    return out


def prepare(force=False):
    OUT.mkdir(exist_ok=True)
    cache=OUT/'nested_frames.pkl'
    if cache.exists() and not force:
        with cache.open('rb') as f:return pickle.load(f)
    d,R=g.load_data(with_members=True)
    data={}
    for f in 'ABCDEF':
        block=old.BLOCK[f];cm=d.fold.eq(old.PAIR[f]).to_numpy();tm=d.fold.eq(f).to_numpy()
        cal,test=d.loc[cm].copy(),d.loc[tm].copy();hist=d[d.block.ne(block)].copy()
        def enrich(raw,normal):return rich_history(g.add_history(g.frame(raw,normal),hist),hist)
        inner=[]
        for m in sorted(cal.month.unique()):
            z=cal.month.to_numpy()==m;tr=cal.loc[~z];te=cal.loc[z]
            a=enrich(tr,g.crossfit_normal(tr,R[cm][~z],tr.SCHED_TIME_UTC_mvt.dt.day.to_numpy()%2))
            b=enrich(te,g.normal_from(te,R[cm][z],g.fit_blend(tr,R[cm][~z])))
            inner.append((a,b))
        tr=enrich(cal,g.crossfit_normal(cal,R[cm],cal.month))
        te=enrich(test,g.normal_from(test,R[tm],g.fit_blend(cal,R[cm])))
        data[f]={'inner':inner,'train':tr,'test':te}
        print('prepared',f,len(tr),len(te),flush=True)
    with cache.open('wb') as f:pickle.dump(data,f)
    return data


def train(frame,config):
    features=old.IDENTITY_FEATURES+(RICH if config['rich'] else [])+config.get('extra_features',[])
    ok=frame.recover & np.isfinite(frame.w_star) & (frame.loss_weight>0)
    d=frame.loc[ok]
    X,_,cats=old._prep(d,d.iloc[:1],features)
    params=dict(g.PARAMS,num_threads=8,num_leaves=config['leaves'])
    if config['mode']=='binary':
        params['objective']='binary';target=d.is_copy
    elif config['mode']=='residual':target=d.w_star-d.w
    else:target=d.w_star
    model=lgb.train(params,lgb.Dataset(X,target,weight=d.loss_weight,categorical_feature=cats),num_boost_round=config['rounds'])
    state={'features':features,'categories':{c:X[c].cat.categories.tolist() for c in cats},'config':config}
    return model,state


def predict(model,state,score):
    X=score[state['features']].copy()
    for c,cats in state['categories'].items():
        v=X[c].astype('string').fillna('__NA__')
        X[c]=pd.Categorical(v.where(v.isin(cats),'__NA__'),categories=cats)
    raw=model.predict(X,num_threads=8)
    if state['config']['mode']=='residual':raw=raw+score.w.to_numpy()
    return np.clip(raw,0,1)


def run(names=None):
    data=prepare();names=names or list(CONFIGS)
    whole=pd.read_parquet(old.OOF_PATH,columns=['MVT_ID_mvt','y','fold','pred'])
    prior=pd.read_parquet(g.OOF).set_index('MVT_ID_mvt')
    base=whole.pred.to_numpy().copy()
    pos=pd.Series(np.arange(len(whole)),index=whole.MVT_ID_mvt)
    base[pos.reindex(prior.index).to_numpy()]=prior.candidate
    preds={name:base.copy() for name in names};preds['nested_select']=base.copy()
    rows=[];ledger=[]
    for f,ds in data.items():
        inner=pd.concat([b for a,b in ds['inner']],ignore_index=True)
        results={};gates={};inner_all={}
        for name in names:
            config=CONFIGS[name]
            cache=OUT/f'pred_{f}_{name}.npz'
            if cache.exists():
                z=np.load(cache);ig=z['inner'];gate=z['outer']
            else:
                ig=np.concatenate([predict(*train(a,config),b) for a,b in ds['inner']])
                gate=predict(*train(ds['train'],config),ds['test'])
                np.savez_compressed(cache,inner=ig,outer=gate)
            scores={str(a):old.rmse(inner.y,g.compose(inner,ig,a)) for a in ALPHAS}
            # Keep the shipped A50 alpha-selection grid as the exact control.
            grid=list(old.ALPHAS) if name=='a50' else ALPHAS
            if name=='a50' and '.05' not in scores:
                scores[str(.05)]=old.rmse(inner.y,g.compose(inner,ig,.05))
            alpha=min(grid,key=lambda a:(scores[str(a)],a))
            results[name]=(scores[str(alpha)],alpha);gates[name]=gate;inner_all[name]=scores
            idx=pos.reindex(ds['test'].MVT_ID_mvt).to_numpy()
            preds[name][idx]=g.compose(ds['test'],gate,alpha)
            z=whole.fold.eq(f).to_numpy()
            row=g.metric_row(whole.loc[z],base[z],preds[name][z],f,alpha)
            row['name']=name;rows.append(row)
            print(json.dumps(row),flush=True)
        selected=min(results,key=lambda k:(results[k][0],names.index(k)))
        alpha=results[selected][1];test=ds['test'];idx=pos.reindex(test.MVT_ID_mvt).to_numpy()
        candidate=g.compose(test,gates[selected],alpha);preds['nested_select'][idx]=candidate
        ledger.append(pd.DataFrame({'MVT_ID_mvt':test.MVT_ID_mvt,'fold':f,'y':test.y,
            'a50':base[idx],'candidate':candidate,'selected':selected,'alpha':alpha}))
        rows.append(dict(name='nested_select',selected=selected,inner=inner_all,
            **g.metric_row(whole.loc[z],base[z],preds['nested_select'][z],f,alpha)))
        print('selected',f,selected,alpha,flush=True)
    summary={name:{'rmse':old.rmse(whole.y,p),'delta_vs_a50':old.rmse(whole.y,p)-old.rmse(whole.y,base)} for name,p in preds.items()}
    report={'baseline':old.rmse(whole.y,base),'summary':summary,'rows':rows,'configs':{n:CONFIGS[n] for n in names},
        'limitations':'Repeated historical benchmark; outer model selection is diagnostic. nested_select chooses config and alpha within inner months.',
        'sources':'Existing local challenge and ADS-B.lol-derived member caches only; no acquisition.'}
    (OUT/'search.json').write_text(json.dumps(report,indent=2))
    pd.concat(ledger,ignore_index=True).to_parquet(OUT/'nested_oof.parquet',index=False)
    print(json.dumps(summary,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','search']);p.add_argument('--names')
    a=p.parse_args()
    prepare() if a.command=='prepare' else run(a.names.split(',') if a.names else None)
