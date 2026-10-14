"""Write tools/members/probe106_ext.py = scripts/106_member_probe.py + an extra-scoring hook (no training change).

Patch: after fit_mlp / fit_mlp_set have trained and scored blk['held'] exactly as
before, each entry of blk.get('extra_held', []) -- dict(tag, frame, tokens) -- is scored by the SAME trained network
and stored in blk['extra_out'][tag]. ``tokens(ids, ctx)`` returns the raw neighbour tokens of the extra frame
(e.g. the earlier Jan/Jul export); the kinds-ablation and mask_self_nm post-processing are applied to them as to blk['held'].
The training loop, seeds, inputs and the returned prediction are untouched; the hook runs after training, so the
random streams that drive training are not consumed differently.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / 'scripts' / '106_member_probe.py'
DST = Path(__file__).resolve().parent / 'probe106_ext.py'

s = SRC.read_text(encoding='utf-8')


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:60], n)
    s = s.replace(old, new)


# --- fit_mlp: score extra frames with the trained net
rep("""    r = np.full(len(held), np.nan)
    r[m] = np.concatenate(out).astype('float64') * y_sd + y_mu - (hs.atot_minus_aobt3.values if direct else 0.0)
    return r, params['epochs']
""", """    r = np.full(len(held), np.nan)
    r[m] = np.concatenate(out).astype('float64') * y_sd + y_mu - (hs.atot_minus_aobt3.values if direct else 0.0)
    for ex in blk.get('extra_held', []):                       # member-tools hook: same net, another frame (no training change)
        I2 = mlp_inputs({**blk, 'held': ex['frame']}, params)
        o2 = []
        with torch.no_grad():
            H2n, H2k = torch.from_numpy(I2['Hn']), torch.from_numpy(I2['Hk'])
            for i in range(0, len(I2['Hn']), 65536):
                o2.append(net(H2n[i:i + 65536].to(dev), H2k[i:i + 65536].to(dev)).cpu().numpy())
        r2 = np.full(len(ex['frame']), np.nan)
        r2[I2['m']] = np.concatenate(o2).astype('float64') * y_sd + y_mu - (I2['hs'].atot_minus_aobt3.values if direct else 0.0)
        blk['extra_out'][ex['tag']] = r2
    return r, params['epochs']
""")

# --- fit_mlp_set: drop the full-file token cache once the two aligned copies exist (memory only; values unchanged)
rep("""    Th = tokens_for(which, hs.MVT_ID_mvt.values, ctx)
""", """    Th = tokens_for(which, hs.MVT_ID_mvt.values, ctx)
    _TOKENS.clear()                     # member tools: free the full-file token arrays (Tt / Th are copies); memory only
""")

# --- fit_mlp_set: factor the token post-processing so extra tokens get the identical treatment
rep("""    kinds = params.get('kinds', ('dep', 'arr'))
    for kind in ('dep', 'arr'):
        if kind not in kinds:           # ablation: blank that kind of token entirely (valid = 0)
            for T in (Tt, Th):
                T[T[..., nbr_set.F[f'is_{kind}']] > 0.5] = 0
    if params.get('mask_self_nm'):      # 110: the departure's own NM fields as an unmatched row carries them
        F = nbr_set.F
        for T in (Tt, Th):
            for c in ('anchor', 'dur', 'plan', 'wake_L', 'wake_M', 'wake_H', 'wake_J'):
                T[:, 0, F[c]] = 0
            T[:, 0, F['anchor_missing']] = 1; T[:, 0, F['plan_missing']] = 1
            if (ctx or {}).get('surge'):    # the own NM taxi residual is NM-derived; the stand x runway normal is not
                T[:, 0, nbr_set.surge_index('dur_res', T.shape[-1])] = 0
""", """    kinds = params.get('kinds', ('dep', 'arr'))

    def _post(Ts):                      # member tools: the original in-place loop body, unchanged, as a function
        for kind in ('dep', 'arr'):
            if kind not in kinds:           # ablation: blank that kind of token entirely (valid = 0)
                for T in Ts:
                    T[T[..., nbr_set.F[f'is_{kind}']] > 0.5] = 0
        if params.get('mask_self_nm'):      # 110: the departure's own NM fields as an unmatched row carries them
            F = nbr_set.F
            for T in Ts:
                for c in ('anchor', 'dur', 'plan', 'wake_L', 'wake_M', 'wake_H', 'wake_J'):
                    T[:, 0, F[c]] = 0
                T[:, 0, F['anchor_missing']] = 1; T[:, 0, F['plan_missing']] = 1
                if (ctx or {}).get('surge'):    # the own NM taxi residual is NM-derived; the stand x runway normal is not
                    T[:, 0, nbr_set.surge_index('dur_res', T.shape[-1])] = 0
    _post((Tt, Th))
""")

rep("""    r[m] = 1.0 / (1.0 + np.exp(-z)) if clf else z * y_sd + y_mu - (hs.atot_minus_aobt3.values if I['direct'] else 0.0)
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return r, params['epochs']
""", """    r[m] = 1.0 / (1.0 + np.exp(-z)) if clf else z * y_sd + y_mu - (hs.atot_minus_aobt3.values if I['direct'] else 0.0)
    if ft and blk.get('extra_held'):
        raise SystemExit('extra_held is not implemented for per-airport fine-tuning')
    for ex in blk.get('extra_held', []):                       # member-tools hook: same net, another frame (no training change)
        I2 = mlp_inputs({**blk, 'held': ex['frame']}, params)
        T2 = np.array(ex['tokens'](I2['hs'].MVT_ID_mvt.values, ctx))
        _post((T2,))
        o2 = []
        net.eval()
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            for i in range(0, len(I2['Hn']), 32768):
                o2.append(net(torch.from_numpy(I2['Hn'][i:i + 32768]).to(dev), torch.from_numpy(I2['Hk'][i:i + 32768]).to(dev),
                              torch.from_numpy(T2[i:i + 32768]).to(dev)).float().cpu().numpy())
        z2 = np.concatenate(o2).astype('float64') if o2 else np.zeros(0)
        r2 = np.full(len(ex['frame']), np.nan)
        r2[I2['m']] = 1.0 / (1.0 + np.exp(-z2)) if clf else z2 * y_sd + y_mu - (I2['hs'].atot_minus_aobt3.values if I['direct'] else 0.0)
        blk['extra_out'][ex['tag']] = r2
        del T2
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return r, params['epochs']
""")


# --- save every trained network so it can be re-predicted without refitting
rep("""    return dict(Xn=Xn, Xk=Xk, Hn=Hn, Hk=Hk, maps=maps, cats=cats, m=m, hs=hs, y_mu=y_mu, y_sd=y_sd, yt=yt, direct=direct)
""", """    return dict(Xn=Xn, Xk=Xk, Hn=Hn, Hk=Hk, maps=maps, cats=cats, m=m, hs=hs, y_mu=y_mu, y_sd=y_sd, yt=yt, direct=direct,
                stats=stats, nums=nums)          # member tools: stats/nums returned too (re-prediction of saved networks)
""")
rep("""    torch.set_num_threads(16)
""", """    torch.set_num_threads(6)            # member tools: CPU threads capped at 6 (was 16); the networks train on the GPU
""", count=2)
rep("""    r[m] = np.concatenate(out).astype('float64') * y_sd + y_mu - (hs.atot_minus_aobt3.values if direct else 0.0)
    for ex in blk.get('extra_held', []):""", """    r[m] = np.concatenate(out).astype('float64') * y_sd + y_mu - (hs.atot_minus_aobt3.values if direct else 0.0)
    if blk.get('save_to'):                                     # member tools: persist the trained network + preprocessing
        save_net(blk['save_to'], 'mlp', net, I, params, blk, None, False)
    for ex in blk.get('extra_held', []):""")
rep("""    if ft and blk.get('extra_held'):
        raise SystemExit('extra_held is not implemented for per-airport fine-tuning')""", """    if ft and (blk.get('extra_held') or blk.get('save_to')):
        raise SystemExit('extra_held / save_to are not implemented for per-airport fine-tuning')
    if blk.get('save_to'):                                     # member tools: persist the trained network + preprocessing
        save_net(blk['save_to'], 'mlp_set', net, I, params, blk, nf, clf)""")

s = ('# GENERATED by tools/members/make_probe106_ext.py from scripts/106_member_probe.py -- do not edit by hand.\n'
     '# Only change: the member-tools extra-scoring hook in fit_mlp / fit_mlp_set (see the generator docstring).\n') + s
s = s + Path(__file__).with_name('probe106_saved_nets.py.txt').read_text(encoding='utf-8')
DST.write_text(s, encoding='utf-8', newline='\n')
print('wrote', DST)
