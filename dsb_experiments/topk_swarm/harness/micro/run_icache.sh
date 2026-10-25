#!/usr/bin/env bash
# usage: run_icache.sh --out <dir> --node <name> [--gpu <i>] [--lock <path>] [--skip-build]
# Builds icache_micro (sm_100) outside the lock, runs it inside, records nvidia-smi clocks next to it.
set -euo pipefail
OUT=""; NODE="unknown"; GPU=0; LOCK=""; SKIP=0
while [[ $# -gt 0 ]]; do case "$1" in
  --out) OUT="$2"; shift 2;; --node) NODE="$2"; shift 2;; --gpu) GPU="$2"; shift 2;; --lock) LOCK="$2"; shift 2;;
  --skip-build) SKIP=1; shift;; *) echo "unknown argument $1" >&2; exit 1;; esac; done
[[ -n "$OUT" ]] || { echo "--out is required" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
NVCC=nvcc; command -v nvcc >/dev/null 2>&1 || NVCC=/usr/local/cuda/bin/nvcc
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"; DATE="$(date -u +%Y-%m-%d)"; LOG="$OUT/run_icache_$STAMP.log"
if [[ "$SKIP" != "1" || ! -x "$OUT/icache_micro" ]]; then
  "$NVCC" -arch=sm_100 -O3 -std=c++17 -Xptxas -v -o "$OUT/icache_micro" "$HERE/icache_micro.cu" 2> "$OUT/ptxas_icache.txt"
  cuobjdump -sass "$OUT/icache_micro" > "$OUT/sass_icache.txt" 2>>"$LOG" || true
fi
MEASURE="cd '$OUT' && export CUDA_VISIBLE_DEVICES=$GPU && nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw --format=csv > clocks_before_$STAMP.txt && ./icache_micro --node '$NODE' --date '$DATE' > icache_$STAMP.jsonl 2>> '$LOG' && nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw --format=csv > clocks_after_$STAMP.txt"
if [[ -n "$LOCK" ]]; then flock "$LOCK" -c "$MEASURE"; else bash -c "$MEASURE"; fi
echo "lines: $(wc -l < "$OUT/icache_$STAMP.jsonl") results in $OUT (icache_$STAMP.jsonl)"
