#!/bin/bash
# Stage 3 of docs/REPRODUCE.md: every trained input of v45 for the final rows, variant A, unmatched twins over 15 seeds,
# then the composition -> $PRC_WORK/submissions_rebuild/v45_A15_<stamp>.parquet (+ _detail.parquet).
# One process at a time (the original run used <= 2 in parallel, gated on free RAM / GPU memory). Each line can be run
# by hand; a failed line can be re-run alone (the networks take --skip-existing).
#   PY        python command (default: py -3.14)
#   ADSB      ADS-B tables of the final file   (default: $PRC_ADSB_DIR, else <repo>/adsb)
#   SNAP      ADS-B tables of variant A training (default: $PRC_FROZEN/ref_main/adsb_snapshot_0926, else <repo>/frozen/...)
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-py -3.14}
WORK=${PRC_WORK:-.}
ADSB=${ADSB:-${PRC_ADSB_DIR:-$WORK/adsb}}
SNAP=${SNAP:-${PRC_FROZEN:-$WORK/frozen}/ref_main/adsb_snapshot_0926}
M=tools/members
S15="41 42 43 44 45 46 47 48 49 50 51 52 53 54 55"
run() { echo "[$(date +%T)] $PY $*"; $PY "$@"; }

# 1. A27 block: saved clean expert re-predicted; context detector and rd members refitted; saved gate re-predicted
run $M/a27_members.py m3rpa
run $M/a27_members.py detector
run $M/a27_members.py rd
run $M/a27_members.py gate
# 2. LightGBM members without ADS-B, the BLIND twins (seeds 41-43, then 44-55)
run $M/lgb_members.py surge
run $M/lgb_members.py member lgb_sg
run $M/lgb_members.py twins --part blind
run $M/lgb_members.py twins --part blind --seeds 44 45 46 47 48 49 50 51 52 53 54 55
# 3. the 23 networks without ADS-B (GPU; each saves its network)
for n in set_clf self_clf set_clf2 nm_masked_set nm_masked_self nm_masked_set2 u_self u_self_s1 u_self_s2 \
         mlp_setn mlp_setn2 mlp_setn_s1 mlp_setn_r mlp_setn2_r mlp_setn_s1_r mlp_set mlp_set2 mlp_set_s1 \
         mlp mlp_s1 mlp_s2 mlp_e20 mlp_wide; do
  run $M/nn_members.py $n --skip-existing
done
# 4. variant A: the 26 ADS-B members trained on the snapshot's 2025 rows, final rows scored on $ADSB
A="--variant A --adsb $ADSB --train-adsb $SNAP"
for n in mlp_setn_ab mlp_setn_ab_s1 mlp_setn_ab_s2 mlp_setn_ab_s3 mlp_setn_ab_s4 mlp_setn_ab_s5 mlp_setn_ab_s6; do
  run $M/nn_members.py $n $A --skip-existing
done
run $M/lgb_members.py member lgb_ab_trust lgb_sg_ab_trust $A
run $M/lgb_members.py member lgb_sg_ab_apt $A
run $M/lgb_members.py member lgb_ab_apt $A
run $M/lgb_members.py member lgb_sg_ab_st lgb_sg_ab_st_s1 lgb_sg_ab_st_s2 $A
run $M/lgb_members.py member lgb_ab_st lgb_ab_st_s1 lgb_ab_st_s2 $A
run $M/lgb_members.py member lgb_sg_q12 lgb_q12 $A
run $M/lgb_members.py member lgb_sg_ab lgb_sg_ab_s1 lgb_ab lgb_ab_s1 $A
run $M/lgb_members.py twins --part ab $A
run $M/lgb_members.py twins --part ab $A --seeds 44 45 46 47 48 49 50 51 52 53 54 55
# 5. every input re-predicted from the saved models on the final rank tables, then the 15-seed twins file
run $M/rescore.py --variant A
run $M/rescore.py --variant A --parts twins --out "$WORK/runs/models/final_A" --twin-seeds $S15
# 6. the composition
TW="$WORK/runs/models/final_A/member_cache_a46/unmatched_adsb_rank_${S15// /_}.parquet"
run $M/recompose_v45.py rebuild --rank new --adsb "$ADSB" --base-dir "$WORK/runs/models/final_A" --twins "$TW" --tag A15 --write
echo "[$(date +%T)] stage 3 done: $(ls -t "$WORK"/submissions_rebuild/v45_A15_*.parquet | grep -v _detail | head -1)"
