#!/bin/bash
# N10 batch 3 on umbriel-b200-028: the combined adapter N10_auto against the u64 word and the pairs at warp and sub-warp scope.
cd /work/topk_swarm/work/opt-N10/harness
export PATH=/usr/local/cuda/bin:$PATH
BASE="--repo /work/topk_swarm/cccl --root /work/topk_swarm --holder opt-N10 --gpu 0 --node-name umbriel-b200-028 --arch-name b200 --source-commit e2bb6eccd5 --keys fp32"
WARP="$BASE --n 32,64,128,256 --k 4,16,32,64"
SUB="$BASE --threads 4,8,16 --n 8,32,128 --k 4,8,16,64"
run() { local lbl=$1; shift; echo "== $lbl $(date -u +%H:%M:%S)"; python3 run.py "$@" --label b3_$lbl > /work/topk_swarm/work/opt-N10/b3_$lbl.log 2>&1; echo "exit=$? $(grep -c compile_error /work/topk_swarm/work/opt-N10/b3_$lbl.log) $(tail -n 1 /work/topk_swarm/work/opt-N10/b3_$lbl.log | cut -c1-160)"; }
for phase in --build-only --measure-only; do
  echo "#### phase $phase $(date -u +%H:%M:%S)"
  run s1 $WARP $phase --adapter N10_auto --control W1_eager_same
  run s2 $WARP $phase --adapter N10_auto --control W1_minmax_pairs --req prefer_smaller_index --control-req not_guaranteed
  run s3 $SUB $phase --adapter N10_auto_sub --control W1_eager_same_sub
  run s4 $SUB $phase --adapter N10_auto_sub --control W1_minmax_pairs_sub --req prefer_smaller_index --control-req not_guaranteed
  run s5 $BASE --n 64,128,256 --k 16,32 $phase --adapter N10_auto --control N10_u32_trunc --req prefer_smaller_index --patterns random,pivot_tie,all_equal
done
echo "exit=0 batch3 done $(date -u +%H:%M:%S)"
