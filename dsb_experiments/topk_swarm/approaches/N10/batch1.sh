#!/bin/bash
# N10 batch 1 on umbriel-b200-020: build every section outside the lock, then measure them back to back.
cd /work/topk_swarm/work/opt-N10/harness
export PATH=/usr/local/cuda/bin:$PATH
COMMON="--repo /work/topk_swarm/cccl --root /work/topk_swarm --holder opt-N10 --gpu 0 --node-name umbriel-b200-020 --arch-name b200 --source-commit 3f8ebfe11f --keys fp32 --n 32,64,128,256 --k 4,16,32,64"
run() { local lbl=$1; shift; echo "== $lbl $(date -u +%H:%M:%S)"; python3 run.py $COMMON "$@" --label b1_$lbl > /work/topk_swarm/work/opt-N10/b1_$lbl.log 2>&1; echo "exit=$? $(grep -c compile_error /work/topk_swarm/work/opt-N10/b1_$lbl.log) $(tail -n 1 /work/topk_swarm/work/opt-N10/b1_$lbl.log | cut -c1-160)"; }
for phase in --build-only --measure-only; do
  echo "#### phase $phase $(date -u +%H:%M:%S)"
  run s1 $phase --adapter N10_u32_trunc --control W1_eager_same
  run s2 $phase --adapter N10_u32 --control W1_eager_same
  run s3 $phase --adapter N10_u32_trunc --control W1_minmax_pairs --req prefer_smaller_index --control-req not_guaranteed
  run s4 $phase --adapter N10_u32_trunc --control N10_ctrl_u64rec --req prefer_smaller_index
  run s6 $phase --adapter N10_u32_ltrunc --control N10_u32_trunc --req prefer_smaller_index
  echo "== s5 $(date -u +%H:%M:%S)"; python3 run.py --repo /work/topk_swarm/cccl --root /work/topk_swarm --holder opt-N10 --gpu 0 --node-name umbriel-b200-020 --arch-name b200 --source-commit 3f8ebfe11f --keys bf16 --n 32,64,128,256 --k 4,16,32,64 --req not_guaranteed $phase --adapter W1x_vimnmx_eager --control W1_eager_same --label b1_s5 > /work/topk_swarm/work/opt-N10/b1_s5.log 2>&1; echo "exit=$? $(tail -n 1 /work/topk_swarm/work/opt-N10/b1_s5.log | cut -c1-160)"
done
echo "exit=0 batch1 done $(date -u +%H:%M:%S)"
