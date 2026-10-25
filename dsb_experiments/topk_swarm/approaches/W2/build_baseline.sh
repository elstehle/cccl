#!/usr/bin/env bash
# Build (no GPU) the baseline batches of opt-W2 on a node. Arguments: node-name arch-name
# The IPT 128 cells with k 256 and 1024 (N 4096 at warp scope) are left out: their translation
# units compile for more than 20 minutes and they are boundary cells the approach does not target.
set -u
NODE="$1"; ARCH="$2"
cd /work/topk_swarm/work/opt-W2/harness
export PATH=/usr/local/cuda/bin:$PATH
COMMON="--repo /work/topk_swarm/cccl --root /work/topk_swarm --holder opt-W2 --gpu 0 --node-name $NODE --arch-name $ARCH --timeout-compile 1500"
W2="--adapter W2_packed --control W1_reference --control-req not_guaranteed --keys bf16,fp32 --req not_guaranteed,prefer_smaller_index"
python3 run.py $COMMON $W2 --n 32,128,512,2048 --label grid_a --build-only > ../build_grid_a.log 2>&1
echo "grid_a: $(tail -2 ../build_grid_a.log | head -1)"
python3 run.py $COMMON $W2 --n 4096 --k 4,16,64 --label grid_b --build-only > ../build_grid_b.log 2>&1
echo "grid_b: $(tail -2 ../build_grid_b.log | head -1)"
python3 run.py $COMMON --adapter W2_packed_sw --keys bf16,fp32 --req not_guaranteed --k 4,8,16,64 --label subwarp --build-only > ../build_subwarp.log 2>&1
echo "subwarp: $(tail -2 ../build_subwarp.log | head -1)"
python3 run.py $COMMON --adapter B5_hier --control B1_reference --keys bf16,fp32 --req not_guaranteed --n 1024,2048,4096,16384 --k 4,8,16,32,64 --label b5 --build-only > ../build_b5.log 2>&1
echo "b5: $(tail -2 ../build_b5.log | head -1)"
echo ALLDONE
