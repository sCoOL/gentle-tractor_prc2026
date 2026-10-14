"""Best arrival<->departure link per departure, for the explorer (and anyone who needs the pairs themselves).

  py -3.14 scripts/231_turnaround_links.py            # links.parquet + link_steps_audit.json

Step 1 uses script 213's two rules unchanged: the ADS-B identity inbound (same icao24, latest identified arrival at the
airport before take-off, within 48 h) where both flights were identified, else the fixed same-stand rule (same airline
and type, arrivals in the 30 min before take-off skipped, in-block no later than schedule + 2 h, 20 h lookback). Reads
213's cached identity days, so it needs 213 to have run.

Step 2 (2026-09-25) links the departures step 1 left unlinked, using only arrivals no step-1 link used, each at
most once, departures in take-off order:
  rotation       the arrival flight number that feeds this departure flight number on step-1 links at this airport
                 (>= 3 links and >= half of them, e.g. AMX070 -> AMX071), in-block 20 min - 48 h before take-off, on
                 any stand (tows) and with no schedule + 2 h cap (a flight delayed overnight takes the next day's
                 inbound); the same stand first, then the latest in-block. Source 'rotation' when that arrival
                 parked on the departure's stand, 'rotation_tow' when it did not (a tow, or an aircraft swap)
  stand_relaxed  the same stand and airline, type equal OR missing on either side (LIRF takes the departure type from
                 the NM match, so unmatched LIRF departures have none), in-block 30 min - 20 h before take-off, no
                 schedule + 2 h cap; the latest in-block
The audit reruns step 2 on top of the stand rule alone and scores it against the ADS-B identity inbound. 2026-09-25:
stand rule 95.7 %, rotation 98.5 %, stand_relaxed 94.2 %, rotation_tow 70.7 % (its misses are aircraft swaps: the
real inbound carried another flight number), so read rotation_tow links as a lead, not a fact.

Output runs/v51_turnaround/links.parquet: dep_mvt, arr_mvt, link_source (identity / stand / rotation / rotation_tow / stand_relaxed),
turn_s (take-off minus the arrival's in-block). Rank-available inputs only; no departure block time is read.
"""
import importlib
import json
from collections import defaultdict

import numpy as np
import pandas as pd

tf = importlib.import_module('213_turnaround_features')
m255 = importlib.import_module('255_adsb_match_v2')
import flightnorm as fn  # noqa: E402


def _fkeys(s) -> np.ndarray:
    """Flight numbers in scripts/flightnorm.py canonical form (2026-10-07: LIRF doubled prefixes, IATA/ICAO prefixes,
    leading zeros), so AMX070/AMX0070 or RYRR388G/RYR388G count as one number in the rotation rule."""
    m255._ensure_map()
    return np.array([fn.key(x) if isinstance(x, str) else '' for x in s.astype('string').fillna('').tolist()], dtype=object)


def _airlines(s) -> np.ndarray:
    m255._ensure_map()
    return np.array([fn.airline(x) if isinstance(x, str) else '' for x in s.astype('string').fillna('').tolist()], dtype=object)

PAIR_MIN, PAIR_SHARE = 3, 0.5
ROT_BACK, ROT_MIN_TURN = 48 * 3600.0, 1200.0
REL_BACK, REL_MIN_TURN, REL_STEPS = 72000.0, 1800.0, 6


def rotation_pairs(d, link):
    """{(APT, departure flight number): {arrival flight numbers}} from the step-1 links."""
    fl = _fkeys(d.FLIGHT_mvt)
    r = np.flatnonzero(link >= 0)
    p = pd.DataFrame({'apt': d.APT.to_numpy()[r], 'dfl': fl[r], 'afl': fl[link[r]]})
    p = p[(p.dfl != '') & (p.afl != '')]
    n = p.groupby(['apt', 'dfl', 'afl']).size().rename('n').reset_index()
    n = n.join(p.groupby(['apt', 'dfl']).size().rename('tot'), on=['apt', 'dfl'])
    n = n[(n.n >= PAIR_MIN) & (n.n >= PAIR_SHARE * n.tot)]
    out = defaultdict(set)
    for a, df, af in n[['apt', 'dfl', 'afl']].itertuples(index=False):
        out[(a, df)].add(af)
    return out


def step2(d, link):
    """Links for the departures `link` (row index of the inbound arrival, -1 = none) leaves unlinked.
    Returns (new link array, source array: '' where step 2 did not link)."""
    link = link.copy()
    src = np.full(len(d), '', dtype=object)
    apt = d.APT.astype(str).to_numpy()
    fl = _fkeys(d.FLIGHT_mvt)
    stand = d.STAND_mvt.astype('string').fillna('').to_numpy()
    typ = d.AIRCRAFT_TYPE_mvt.astype('string').fillna('').to_numpy()
    al = _airlines(d.FLIGHT_mvt)
    t, tb = d.t.to_numpy(), d.tb.to_numpy()
    is_dep = (d.PHASE_mvt == 'DEP').to_numpy()
    used = np.zeros(len(d), bool)
    used[link[link >= 0]] = True
    arr = np.flatnonzero((d.PHASE_mvt == 'ARR').to_numpy() & np.isfinite(tb) & ~used)
    todo = np.flatnonzero(is_dep & (link < 0))
    todo = todo[np.argsort(t[todo], kind='stable')]

    pairs = rotation_pairs(d, link)
    by_flight = defaultdict(list)                     # (APT, arrival flight number) -> arrival rows, in-block order
    for r in arr[np.argsort(tb[arr], kind='stable')]:
        if fl[r]:
            by_flight[(apt[r], fl[r])].append(r)
    for r in todo:
        cands = []
        for af in pairs.get((apt[r], fl[r]), ()):
            for c in by_flight.get((apt[r], af), ()):
                if not used[c] and t[r] - ROT_BACK <= tb[c] <= t[r] - ROT_MIN_TURN:
                    cands.append(c)
        if cands:
            c = max(cands, key=lambda x: (stand[x] != '' and stand[x] == stand[r], tb[x]))
            same = stand[c] != '' and stand[c] == stand[r]
            link[r], src[r], used[c] = c, 'rotation' if same else 'rotation_tow', True

    by_stand = defaultdict(list)                      # (APT, stand) -> unused arrival rows, in-block order
    for r in arr[np.argsort(tb[arr], kind='stable')]:
        if stand[r] not in ('', 'NA') and not used[r]:
            by_stand[(apt[r], stand[r])].append(r)
    keys = {k: np.array([tb[x] for x in v]) for k, v in by_stand.items()}
    for r in todo:
        if link[r] >= 0 or stand[r] in ('', 'NA') or (apt[r], stand[r]) not in by_stand:
            continue
        rows, ab = by_stand[(apt[r], stand[r])], keys[(apt[r], stand[r])]
        p, steps = np.searchsorted(ab, t[r] - REL_MIN_TURN, side='right') - 1, 0
        while p >= 0 and steps < REL_STEPS and t[r] - ab[p] <= REL_BACK:
            c = rows[p]
            if not used[c]:
                steps += 1
                if al[c] == al[r] and (typ[c] == typ[r] or not typ[c] or not typ[r]):
                    link[r], src[r], used[c] = c, 'stand_relaxed', True
                    break
            p -= 1
    return link, src


def audit(d, ident):
    """Step 2 on top of the stand rule alone, scored against the ADS-B identity inbound (the truth where both flights
    were identified)."""
    stand = tf.stand_rule(d)
    inb = tf.identity_inbound(d, ident)
    link, src = step2(d, stand)
    st = d.STAND_mvt.astype('string').fillna('').to_numpy()
    dep = np.flatnonzero((d.PHASE_mvt == 'DEP').to_numpy())
    known = dep[inb[dep] >= 0]
    rep = {'departures': int(len(dep)), 'identity_inbound_known': int(len(known)),
           'stand_rule': {'linked': int((stand[dep] >= 0).sum()),
                          'precision_on_known': float((stand[known][stand[known] >= 0] == inb[known][stand[known] >= 0]).mean())},
           'step2': {}}
    for s in ('rotation', 'rotation_tow', 'stand_relaxed'):
        rows = dep[src[dep] == s]
        k = rows[inb[rows] >= 0]
        rep['step2'][s] = {'linked': int(len(rows)), 'with_identity_truth': int(len(k)),
                           'precision_on_known': float((link[k] == inb[k]).mean()) if len(k) else None,
                           'towed_share': float((st[link[rows]] != st[rows]).mean()) if len(rows) else None}
    miss = known[stand[known] < 0]
    rep['stand_rule_missed_known'] = int(len(miss))
    rep['step2_recovers_correctly'] = int((link[miss] == inb[miss]).sum())
    rep['step2_recovers_wrongly'] = int(((link[miss] >= 0) & (link[miss] != inb[miss])).sum())
    rep['by_airport'] = {a: {'unlinked_after_stand_rule': int(((d.APT.to_numpy()[dep] == a) & (stand[dep] < 0)).sum()),
                             'unlinked_after_step2': int(((d.APT.to_numpy()[dep] == a) & (link[dep] < 0)).sum())}
                         for a in sorted(d.APT.unique())}
    return rep


def main():
    d = tf.movements()
    ident = pd.read_parquet(tf.OUT / 'ident.parquet')
    stand = tf.stand_rule(d)
    inb = tf.identity_inbound(d, ident)
    link = np.full(len(d), -1, dtype=np.int64)
    dep = np.flatnonzero((d.PHASE_mvt == 'DEP').to_numpy())
    link[dep] = np.where(inb[dep] >= 0, inb[dep], stand[dep])
    src1 = np.where(inb >= 0, 'identity', 'stand')
    link2, src2 = step2(d, link)
    keep = dep[link2[dep] >= 0]
    ids = d.MVT_ID_mvt.to_numpy()
    out = pd.DataFrame({'dep_mvt': ids[keep], 'arr_mvt': ids[link2[keep]],
                        'link_source': np.where(src2[keep] != '', src2[keep], src1[keep]),
                        'turn_s': d.t.to_numpy()[keep] - d.tb.to_numpy()[link2[keep]]})
    out.to_parquet(tf.OUT / 'links.parquet', index=False)
    print(f'{len(out):,} links of {len(dep):,} departures;', out.link_source.value_counts().to_dict(), flush=True)
    rep = audit(d, ident)
    rep['deployed'] = out.link_source.value_counts().to_dict()
    (tf.OUT / 'link_steps_audit.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != 'by_airport'}, indent=2))


if __name__ == '__main__':
    main()
