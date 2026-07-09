#!/usr/bin/env bash
# Sync a dsb_experiments prototype to the B200 node's CUDA container, build, and run it.
# Usage: ./run_remote.sh <file.cu> [extra nvcc args...]
set -euo pipefail
FILE="$1"
shift || true
BASE="$(basename "$FILE" .cu)"
NODE=umb-b200-250
CTR=quizzical_keldysh
DIR=/cccl_fork/cccl/dsb_experiments

scp -q "$(dirname "$0")/$(basename "$FILE")" "$NODE:/tmp/$(basename "$FILE")"
ssh -o BatchMode=yes "$NODE" "docker cp /tmp/$(basename "$FILE") $CTR:$DIR/ && docker exec $CTR bash -c 'cd $DIR && nvcc -std=c++17 -arch=sm_100 -O3 -I../cub -I../libcudacxx/include -I../thrust --extended-lambda $* $BASE.cu -o $BASE && ./$BASE'"
