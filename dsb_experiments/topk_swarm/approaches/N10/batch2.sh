#!/bin/bash
# N10 batch 2 on umbriel-b200-020: iteration 1 (one-ballot recovery) at warp scope, the tie cost, the level order, and the sub-warp grid.
cd /work/topk_swarm/work/opt-N10/harness
export PATH=/usr/local/cuda/bin:$PATH
BASE="--repo /work/topk_swarm/cccl --root /work/topk_swarm --holder opt-N10 --gpu 0 --node-name umbriel-b200-020 --arch-name b200 --source-commit 32949a4bc6 --keys fp32"
WARP="$BASE --n 32,64,128,256 --k 4,16,32,64"
SUB="$BASE --threads 4,8,16 --n 8,32,128 --k 4,8,16,64"
run() { local lbl=$1; shift; echo "== $lbl $(date -u +%H:%M:%S)"; python3 run.py "$@" --label b2_$lbl > /work/topk_swarm/work/opt-N10/b2_$lbl.log 2>&1; echo "exit=$? $(grep -c compile_error /work/topk_swarm/work/opt-N10/b2_$lbl.log) $(tail -n 1 /work/topk_swarm/work/opt-N10/b2_$lbl.log | cut -c1-160)"; }
for phase in --build-only --measure-only; do
  echo "#### phase $phase $(date -u +%H:%M:%S)"
  run s1 $WARP $phase --adapter N10_ge1_trunc --control N10_u32_trunc --req prefer_smaller_index
  run s2 $BASE --n 64,128,256 --k 16,32 $phase --adapter N10_ge1_trunc --control N10_u32_trunc --req prefer_smaller_index --patterns pivot_tie,all_equal
  run s3 $WARP $phase --adapter N10_ge1_ltrunc --control N10_ge1_trunc --req prefer_smaller_index
  run s4 $WARP $phase --adapter N10_ge1_trunc --control W1_eager_same
  run s5 $SUB $phase --adapter N10_ge1_trunc_sub --control W1_eager_same_sub
  run s6 $SUB $phase --adapter N10_ge1_trunc_sub --control W1_minmax_pairs_sub --req prefer_smaller_index --control-req not_guaranteed
  run s7 $SUB $phase --adapter N10_ge1_ltrunc_sub --control N10_ge1_trunc_sub --req prefer_smaller_index
done
echo "exit=0 batch2 done $(date -u +%H:%M:%S)"
