import argparse

import rescore_mixture

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--features', type=int, required=True)
p.add_argument('--scales', default='1.1,1.25,1.4,1.5')
p.add_argument('--tag', default='p_scale')
a = p.parse_args()
rescore_mixture.rescore(a.run, a.features, [float(x) for x in a.scales.split(',')], a.tag)
