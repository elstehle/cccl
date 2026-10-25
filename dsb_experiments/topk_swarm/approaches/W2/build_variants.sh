#!/usr/bin/env bash
# Build (no GPU) the variant batches of opt-W2 on a node. Arguments: node-name arch-name
set -u
NODE="$1"; ARCH="$2"
cd /work/topk_swarm/work/opt-W2/harness
export PATH=/usr/local/cuda/bin:$PATH
COMMON="--repo /work/topk_swarm/cccl --root /work/topk_swarm --holder opt-W2 --gpu 0 --node-name $NODE --arch-name $ARCH --timeout-compile 1500 --jobs 9"
CELLS="--keys bf16,fp32 --req not_guaranteed --n 256,512,1024,2048 --k 4,8,16,32,64"
python3 run.py $COMMON --adapter W2_packed_v2 --control W2_packed $CELLS --label v2 --build-only > ../build_v2.log 2>&1 &
python3 run.py $COMMON --adapter W2_packed_v3 --control W2_packed_v2 $CELLS --label v3 --build-only > ../build_v3.log 2>&1 &
python3 run.py $COMMON --adapter W2_packed_sw_v2 --control W2_packed_sw --keys bf16,fp32 --req not_guaranteed --k 4,8,16,64 --label sw_v2 --build-only > ../build_sw_v2.log 2>&1 &
wait
echo "v2: $(tail -2 ../build_v2.log | head -1)"
echo "v3: $(tail -2 ../build_v3.log | head -1)"
echo "sw_v2: $(tail -2 ../build_sw_v2.log | head -1)"
grep -h "compile_error\|compile_timeout" ../build_v2.log ../build_v3.log ../build_sw_v2.log | head -5
echo ALLDONE
