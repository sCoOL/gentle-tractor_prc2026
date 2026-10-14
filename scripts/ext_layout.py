"""X-Plane Gateway apt.dat -> taxi graph -> stand x runway taxi distances (G4) and stand coordinates."""
from __future__ import annotations
import re, time
import numpy as np, pandas as pd, networkx as nx
import common
from common import DATA, EXT, APTS, haversine
STAND_MAX_M = 400.0


def norm_stand(name) -> str:
    n = str(name).upper().strip(); n = re.sub(r'^(GATE|STAND|RAMP|GAC|PARKING|APRON)\s*', '', n); n = re.sub(r'-\(.*\)$', '', n); return n.replace(' ', '')


def parse_apt_dat(path) -> dict:
    nodes, edges, stands, rwys = [], [], [], []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            p = line.split()
            if not p: continue
            c = p[0]
            if c == '1201': nodes.append((int(p[4]), float(p[1]), float(p[2])))
            elif c == '1202': edges.append((int(p[1]), int(p[2]), p[3] == 'oneway', p[4] == 'runway', ' '.join(p[5:])))
            elif c == '1300': nm = ' '.join(p[6:]); stands.append((nm, norm_stand(nm), float(p[1]), float(p[2])))
            elif c == '100' and len(p) >= 20: rwys.append((p[8], float(p[9]), float(p[10]))); rwys.append((p[17], float(p[18]), float(p[19])))
    return dict(nodes=pd.DataFrame(nodes, columns=['id', 'lat', 'lon']), edges=pd.DataFrame(edges, columns=['a', 'b', 'oneway', 'is_runway', 'name']),
                stands=pd.DataFrame(stands, columns=['name', 'norm', 'lat', 'lon']), runways=pd.DataFrame(rwys, columns=['ident', 'lat', 'lon']))


def build_graph(parsed) -> nx.DiGraph:
    n = parsed['nodes'].set_index('id'); G = nx.DiGraph()
    for _, e in parsed['edges'].iterrows():
        if e.a not in n.index or e.b not in n.index: continue
        w = float(haversine(n.lat[e.a], n.lon[e.a], n.lat[e.b], n.lon[e.b]))
        G.add_edge(e.a, e.b, w=w, is_runway=bool(e.is_runway), name=e['name'])
        if not e.oneway: G.add_edge(e.b, e.a, w=w, is_runway=bool(e.is_runway), name=e['name'])
    return G


def _nearest(nodes, lat, lon, ids=None):
    sub = nodes if ids is None else nodes[nodes.id.isin(ids)]
    if sub.empty: return None, np.inf
    d = haversine(sub.lat.values, sub.lon.values, lat, lon); k = int(np.argmin(d)); return int(sub.id.values[k]), float(d[k])


def pair_table(apt, parsed, stands, runways) -> pd.DataFrame:
    edges = parsed['edges']; G = build_graph(parsed); rows = []
    nodes = parsed['nodes'][parsed['nodes'].id.isin(set(G.nodes))]   # only nodes that carry taxi edges (X-Plane packs hold many edge-less nodes)
    rwy_nodes = {}
    for ident in runways:
        e = edges[edges.is_runway & edges['name'].str.split('/').apply(lambda s: ident in s)]
        thr = parsed['runways'][parsed['runways'].ident == ident]
        if e.empty or thr.empty: continue
        nid, _ = _nearest(nodes, float(thr.lat.iloc[0]), float(thr.lon.iloc[0]), set(e.a) | set(e.b)); rwy_nodes[ident] = (nid, float(thr.lat.iloc[0]), float(thr.lon.iloc[0]))
    st = parsed['stands'].drop_duplicates('norm').set_index('norm')
    for s in stands:
        if s not in st.index: continue
        slat, slon = float(st.lat[s]), float(st.lon[s]); nid, dist = _nearest(nodes, slat, slon)
        if nid is None or dist > STAND_MAX_M or nid not in G: continue
        lengths, paths = nx.single_source_dijkstra(G, nid, weight='w')
        for ident, (tn, tlat, tlon) in rwy_nodes.items():
            if tn not in lengths: continue
            path = paths[tn]; crossed = {G[u][v]['name'] for u, v in zip(path[:-1], path[1:]) if G[u][v]['is_runway'] and ident not in G[u][v]['name'].split('/')}
            rows.append(dict(APT=apt, stand=s, rwy=ident, taxi_dist_m=lengths[tn] + dist, taxi_dist_straight_m=float(haversine(slat, slon, tlat, tlon)), n_edges_path=len(path) - 1, n_rwy_crossings=len(crossed)))
    return pd.DataFrame(rows, columns=['APT', 'stand', 'rwy', 'taxi_dist_m', 'taxi_dist_straight_m', 'n_edges_path', 'n_rwy_crossings'])


def build(force=False):
    common.ensure_dirs(); t0 = time.time()
    if not force and (DATA / 'ext_layout_pairs.parquet').exists(): print('ext_layout exists; --force to rebuild'); return
    tr = pd.read_parquet(DATA / 'base_train.parquet', columns=['APT', 'STAND_mvt', 'RUNWAY_mvt']); rk = pd.read_parquet(DATA / 'base_rank.parquet', columns=['APT', 'STAND_mvt', 'RUNWAY_mvt'])
    allm = pd.concat([tr, rk]); pairs, coords = [], []
    for apt in APTS:
        parsed = parse_apt_dat(EXT / 'xplane' / f'{apt}.dat'); m = allm[allm.APT == apt]
        stands = sorted(set(m.STAND_mvt.dropna().astype(str))); rwys = sorted(set(m.RUNWAY_mvt.dropna().astype(str)) - {'NA', 'H', 'HEL'})
        norm_map = {s: norm_stand(s) for s in stands}; st = parsed['stands'].drop_duplicates('norm').set_index('norm')
        t = pair_table(apt, parsed, sorted(set(norm_map.values())), rwys)
        t = t.merge(pd.DataFrame({'stand_raw': list(norm_map), 'stand': list(norm_map.values())}), on='stand').drop(columns='stand').rename(columns={'stand_raw': 'stand'}); pairs.append(t)
        for s, n in norm_map.items():
            ok = n in st.index; coords.append(dict(APT=apt, stand=s, lat=float(st.lat[n]) if ok else np.nan, lon=float(st.lon[n]) if ok else np.nan, matched=int(ok)))
        cov = m.STAND_mvt.astype(str).isin(set(t.stand)).mean() if len(t) else 0.0
        print(f'  {apt}: {len(parsed["nodes"])} nodes, {len(parsed["edges"])} edges, {len(parsed["stands"])} stands; pairs {len(t)}; traffic with a distance {cov:.2f} [{time.time()-t0:.0f}s]', flush=True)
    P = pd.concat(pairs, ignore_index=True); C = pd.DataFrame(coords); P.to_parquet(DATA / 'ext_layout_pairs.parquet'); C.to_parquet(DATA / 'ext_stand_coords.parquet')
    print(f'ext_layout: {len(P):,} stand x runway pairs, {int(C.matched.sum())} of {len(C)} stands located')
