import argparse, json, train_eval
p = argparse.ArgumentParser(); p.add_argument('--model', default='M3'); p.add_argument('--features', type=int, default=1)
p.add_argument('--groups', default='G0,G1,G2,G3,G9'); p.add_argument('--params', default='{}'); p.add_argument('--rounds', type=int, default=600)
p.add_argument('--folds', default='A,B'); p.add_argument('--tag', default=''); p.add_argument('--seeds', default='', help='comma list; one run per seed'); a = p.parse_args()
params = json.loads(a.params)
if a.seeds:
    for s in a.seeds.split(','):
        train_eval.run(a.model, a.features, a.groups.split(','), {**params, 'seed': int(s)}, a.rounds, tuple(a.folds.split(',')), f'{a.tag}_s{s}')
else:
    train_eval.run(a.model, a.features, a.groups.split(','), params, a.rounds, tuple(a.folds.split(',')), a.tag)
