#!/usr/bin/env bash
# Build (no GPU) the baseline batches of opt-W2 on a node. Arguments: node-name arch-name
set -u
NODE="$1"; ARCH="$2"
cd /work/topk_swarm/work/opt-W2/harness
export PATH=/usr/local/cuda/bin:$PATH
COMMON="--repo /work/topk_swarm/cccl --root /work/topk_swarm --holder opt-W2 --gpu 0 --node-name $NODE --arch-name $ARCH --timeout-compile 1500"
python3 run.py $COMMON --adapter W2_packed --control W1_reference --control-req not_guaranteed --keys bf16,fp32 --req not_guaranteed,prefer_smaller_index --label grid --build-only > ../build_grid.log 2>&1
echo "grid: $(tail -2 ../build_grid.log | head -1)"
python3 run.py $COMMON --adapter W2_packed_sw --keys bf16,fp32 --req not_guaranteed --k 4,8,16,64 --label subwarp --build-only > ../build_subwarp.log 2>&1
echo "subwarp: $(tail -2 ../build_subwarp.log | head -1)"
python3 run.py $COMMON --adapter B5_hier --control B1_reference --keys bf16,fp32 --req not_guaranteed --label b5 --build-only > ../build_b5.log 2>&1
echo "b5: $(tail -2 ../build_b5.log | head -1)"
echo ALLDONE
