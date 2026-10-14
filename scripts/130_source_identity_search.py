"""Source-component and schedule-family gate probe, existing data only."""
import argparse
import importlib
import json
import pickle
import numpy as np
import pandas as pd

s=importlib.import_module('129_identity_search')
g=s.g
ROOT=s.ROOT
OUT=ROOT/'runs/a51_sources'
RAW_COLS=['atot_minus_eobt1','lobt_minus_iobt','plan_block_time','arr_delay','iobt_eq_lobt',
          'sched_eq_eobt1','aobt3_eq_lobt','arr_fb_rate_airline_apt','arr_fb_rate_airline_apt_month',
          'dep_prev15','dep_next15','dep_prev60','arr_prev60','n_out_at_aobt3','n_in_at_aobt3']
PROBS=['p_context','p_set_clf','p_self_clf','p_set_clf2']
SCHEDULE_KEYS={'route_hour':['airline','ADES_mvt','sched_bin'],
    'route_weekday':['airline','ADES_mvt','dow'],
    'route_season':['airline','ADES_mvt','summer'],
    'route_stand':['airline','ADES_mvt','STAND_mvt']}
SCHEDULE_FEATURES=[x for k in SCHEDULE_KEYS for x in ('rate_'+k,'count_'+k)]


def extra_table(rank=False):
    base=g.old.BASE_RANK if rank else g.old.BASE_TRAIN
    d=pd.read_parquet(base,columns=['MVT_ID_mvt','APT']+RAW_COLS)
    d=d.loc[d.APT.eq('LIRF')].drop(columns='APT').set_index('MVT_ID_mvt')
    parts=[]
    if rank:
        for name in ['set_clf','self_clf','set_clf2']:
            parts.append(pd.read_parquet(ROOT/'runs/models/member_cache_a35'/f'{name}.parquet').set_index('MVT_ID_mvt').v.rename('p_'+name))
        parts.append(pd.read_parquet(ROOT/'runs/models/A27_20260917_141019/p_fallback_rank.parquet').set_index('MVT_ID_mvt').p_fallback.rename('p_context'))
    else:
        for name in ['set_clf','self_clf','set_clf2']:
            parts.append(pd.concat([pd.read_parquet(ROOT/'runs/tuning'/f'detector_probe_{b}_{name}.parquet') for b in ['AB','CD','EF']]).set_index('MVT_ID_mvt').p.rename('p_'+name))
        parts.append(pd.concat([pd.read_parquet(ROOT/'runs/oof'/f'strict_block_{b}_fold{f}_v5tfb5.parquet',columns=['MVT_ID_mvt','p_fallback']) for b in ['AB','CD','EF'] for f in b]).set_index('MVT_ID_mvt').p_fallback.rename('p_context'))
    for v in parts:d[v.name]=v.reindex(d.index)
    return d.reset_index()


def schedule_history(frame,history):
    if set(frame.MVT_ID_mvt)&set(history.MVT_ID_mvt):raise ValueError('schedule history overlaps frame')
    out=s.keys(frame);h=s.keys(history)
    out['sched_bin']=out.sched_hour//2;h['sched_bin']=h.sched_hour//2
    prior=float(h.is_copy.mean())
    for name,cols in SCHEDULE_KEYS.items():
        stats=h.groupby(cols,dropna=False,observed=True).is_copy.agg(['sum','size']).reset_index()
        stats['rate_'+name]=(stats['sum']+10.*prior)/(stats['size']+10.)
        stats['count_'+name]=stats['size']
        out=out.merge(stats[cols+['rate_'+name,'count_'+name]],on=cols,how='left',validate='many_to_one')
        out['rate_'+name]=out['rate_'+name].fillna(prior);out['count_'+name]=out['count_'+name].fillna(0.)
    return out


def prepare():
    OUT.mkdir(exist_ok=True);path=OUT/'nested_frames.pkl'
    if path.exists():
        with path.open('rb') as f:return pickle.load(f)
    data=s.prepare();extras=extra_table()
    # Each test frame covers one fold; labels are used ONLY outside its block.
    history=pd.concat([v['test'] for v in data.values()],ignore_index=True)
    for fold,ds in data.items():
        h=history.loc[history.block.ne(g.old.BLOCK[fold])]
        def add(d):
            return schedule_history(d,h).merge(extras,on='MVT_ID_mvt',how='left',validate='one_to_one')
        ds['inner']=[(add(a),add(b)) for a,b in ds['inner']]
        ds['train']=add(ds['train']);ds['test']=add(ds['test'])
        print('source frame',fold,flush=True)
    with path.open('wb') as f:pickle.dump(data,f)
    return data


CONFIGS={
    'source_residual':dict(mode='residual',rich=True,rounds=180,leaves=7,extra_features=RAW_COLS+PROBS+SCHEDULE_FEATURES),
    'source_absolute':dict(mode='absolute',rich=True,rounds=180,leaves=7,extra_features=RAW_COLS+PROBS+SCHEDULE_FEATURES),
    'source_binary':dict(mode='binary',rich=True,rounds=180,leaves=15,extra_features=RAW_COLS+PROBS+SCHEDULE_FEATURES),
    'probs_residual':dict(mode='residual',rich=True,rounds=180,leaves=7,extra_features=PROBS),
    'schedule_residual':dict(mode='residual',rich=True,rounds=180,leaves=7,extra_features=SCHEDULE_FEATURES),
}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names');a=p.parse_args()
    frames=prepare()
    s.OUT=OUT;s.CONFIGS=CONFIGS;s.prepare=lambda:frames
    s.run(a.names.split(',') if a.names else None)
