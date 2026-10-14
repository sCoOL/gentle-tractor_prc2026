"""Eight-month, identity-aware copy detector added to a nested A50 gate.

The detector never sees predictions fitted using held-block labels. Its inputs
are raw rank-available fields and rates from block-disjoint label history.
"""
import argparse
import importlib
import pickle
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

s=importlib.import_module('129_identity_search')
x=importlib.import_module('130_source_identity_search')
g=s.g
ROOT=s.ROOT
OUT=ROOT/'runs/a51_detector'
DIRECT=[c for c in g.FEATURES if c not in ['w','normal','gap','abs_gap']] + s.RICH + x.RAW_COLS + x.SCHEDULE_FEATURES + ['nm_gap']
DIRECT=list(dict.fromkeys(DIRECT))
PARAMS=dict(objective='binary',learning_rate=.03,num_leaves=15,min_data_in_leaf=100,
    feature_fraction=.85,bagging_fraction=.85,bagging_freq=1,lambda_l2=10.,cat_smooth=50.,
    cat_l2=10.,max_cat_to_onehot=4,verbose=-1,seed=20260923,num_threads=8)


def enrich(frame,history):
    generated=['flight_route_rate_outer','flight_route_count_outer']+s.RICH+x.SCHEDULE_FEATURES
    frame=frame.drop(columns=[c for c in generated if c in frame])
    out=x.schedule_history(s.rich_history(g.add_history(frame,history),history),history)
    out['nm_gap']=out.D-out.a3taxi
    return out


def train(frame,mode='weighted'):
    valid=np.isfinite(frame.D)&np.isfinite(frame.a3taxi)
    d=frame.loc[valid];X,_,cats=g.old._prep(d,d.iloc[:1],DIRECT)
    weight=np.clip((d.nm_gap/900.)**2,.01,96.**2) if mode=='weighted' else np.ones(len(d))
    model=lgb.train(PARAMS,lgb.Dataset(X,d.is_copy,weight=weight,categorical_feature=cats),num_boost_round=400)
    state={'features':DIRECT,'categories':{c:X[c].cat.categories.tolist() for c in cats},'config':{'mode':'binary'},'params':PARAMS,'rounds':400,'weight_mode':mode}
    return model,state


def prepare():
    OUT.mkdir(exist_ok=True);path=OUT/'nested_frames.pkl'
    if path.exists():
        with path.open('rb') as f:return pickle.load(f)
    data=x.prepare();allrows=pd.concat([v['test'] for v in data.values()],ignore_index=True)
    for block in ['AB','CD','EF']:
        h=allrows.loc[allrows.block.ne(block)].copy()
        ht=pd.concat([enrich(h.loc[h.block.eq(b)],h.loc[h.block.ne(b)]) for b in sorted(h.block.unique())],ignore_index=True)
        for mode in ['weighted','plain']:
            model,state=train(ht,mode)
            model.save_model(str(OUT/f'detector_{block}_{mode}.txt'))
            (OUT/f'detector_{block}_{mode}.json').write_text(json.dumps(state,indent=2))
            for fold in block:
                ds=data[fold]
                def add(d):
                    out=d.copy();e=enrich(d,h)
                    out['p_identity_'+mode]=s.predict(model,state,e)
                    return out
                ds['inner']=[(add(a),add(b)) for a,b in ds['inner']]
                ds['train']=add(ds['train']);ds['test']=add(ds['test'])
            print('detector ready',block,mode,flush=True)
    with path.open('wb') as f:pickle.dump(data,f)
    return data


EXTRA=x.RAW_COLS+x.PROBS+x.SCHEDULE_FEATURES+['p_identity_weighted','p_identity_plain']
CONFIGS={
    'detector_residual':dict(mode='residual',rich=True,rounds=180,leaves=7,extra_features=EXTRA),
    'detector_absolute':dict(mode='absolute',rich=True,rounds=180,leaves=7,extra_features=EXTRA),
    'detector_binary':dict(mode='binary',rich=True,rounds=180,leaves=15,extra_features=EXTRA),
}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names');a=p.parse_args()
    frames=prepare();s.OUT=OUT;s.CONFIGS=CONFIGS;s.prepare=lambda:frames
    s.run(a.names.split(',') if a.names else None)
