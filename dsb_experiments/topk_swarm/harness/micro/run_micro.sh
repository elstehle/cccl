#!/usr/bin/env bash
# Build and run the intrinsics microbenchmark on a node, under the per-GPU lock.
#
# usage: run_micro.sh --out <dir> --node <name> [--gpu <index>] [--lock <path>] [--nvcc <path>] [--only <substr>]
#   --out   directory for the JSON lines, SASS dumps and logs (created)
#   --node  node name recorded in every JSON line
#   --gpu   CUDA device index (default 0), exported as CUDA_VISIBLE_DEVICES for the measurement
#   --lock  flock file for the measurement section (default: no lock, say so in the report)
#   --nvcc  nvcc binary (default: nvcc on PATH, then /usr/local/cuda/bin/nvcc)
#   --only  pass through to the benchmark to run a subset by name substring
#
# Compiles two binaries outside the lock: -arch=sm_100 (the swarm's mandated flags) and
# -arch=sm_100f (family-specific, adds redux.sync.f32). Runs both inside one lock section so the
# sm_100 and sm_100f numbers are a same-session pair. Dumps SASS and resource usage for evidence.

set -euo pipefail

OUT=""
NODE="unknown"
GPU=0
LOCK=""
NVCC=""
ONLY=""
SKIP_BUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --node) NODE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --lock) LOCK="$2"; shift 2 ;;
    --nvcc) NVCC="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift 1 ;;   # binaries already in --out (compiled outside the lease)
    *) echo "unknown argument $1" >&2; exit 1 ;;
  esac
done
[[ -n "$OUT" ]] || { echo "--out is required" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/intrinsics_micro.cu"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"   # absolute, the measurement section cd's into it
if [[ -z "$NVCC" ]]; then
  if command -v nvcc >/dev/null 2>&1; then NVCC=nvcc; else NVCC=/usr/local/cuda/bin/nvcc; fi
fi
DATE="$(date -u +%Y-%m-%d)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$OUT/run_micro_$STAMP.log"

echo "nvcc: $($NVCC --version | tail -n 2 | head -n 1)" | tee -a "$LOG"
echo "arch: $(uname -m)  node: $NODE  gpu: $GPU  date: $DATE" | tee -a "$LOG"

# compiles run outside the lock (and outside the lease: build first with --skip-build absent, then
# acquire the lease and call again with --skip-build)
for V in sm_100 sm_100f; do
  BIN="$OUT/intrinsics_micro_$V"
  FAM=0
  [[ "$V" == "sm_100f" ]] && FAM=1
  if [[ "$SKIP_BUILD" == "1" && -x "$BIN" ]]; then
    echo "using existing $BIN" | tee -a "$LOG"
  else
    echo "compiling $V" | tee -a "$LOG"
    "$NVCC" -arch="$V" -O3 -std=c++17 -lineinfo -Xptxas -v -DMICRO_FAMILY=$FAM -o "$BIN" "$SRC" 2> "$OUT/ptxas_$V.txt" \
      || { echo "compile failed for $V, see $OUT/ptxas_$V.txt" | tee -a "$LOG"; cat "$OUT/ptxas_$V.txt" | tail -n 30; exit 1; }
  fi
  [[ -s "$OUT/sass_$V.txt" ]] || cuobjdump -sass "$BIN" > "$OUT/sass_$V.txt" 2>>"$LOG" || true
  [[ -s "$OUT/res_usage_$V.txt" ]] || cuobjdump -res-usage "$BIN" > "$OUT/res_usage_$V.txt" 2>>"$LOG" || true
done

ONLYARG=""
if [[ -n "$ONLY" ]]; then ONLYARG="--only '$ONLY'"; fi

# three runs in one lock section: sm_100, sm_100f, and sm_100 again as the drift canary
MEASURE="cd '$OUT' && export CUDA_VISIBLE_DEVICES=$GPU && \
  ./intrinsics_micro_sm_100  --node '$NODE' --date '$DATE' --variant sm_100  $ONLYARG > micro_sm_100_$STAMP.jsonl  2>> '$LOG' && \
  ./intrinsics_micro_sm_100f --node '$NODE' --date '$DATE' --variant sm_100f $ONLYARG > micro_sm_100f_$STAMP.jsonl 2>> '$LOG' && \
  ./intrinsics_micro_sm_100  --node '$NODE' --date '$DATE' --variant sm_100_repeat $ONLYARG > micro_sm_100_repeat_$STAMP.jsonl 2>> '$LOG'"

echo "measuring (lock: ${LOCK:-none})" | tee -a "$LOG"
if [[ -n "$LOCK" ]]; then
  flock "$LOCK" -c "$MEASURE"
else
  bash -c "$MEASURE"
fi

echo "lines: $(cat "$OUT"/micro_*_$STAMP.jsonl | wc -l)" | tee -a "$LOG"
echo "done, results in $OUT (micro_*_$STAMP.jsonl, sass_*.txt, res_usage_*.txt, $LOG)"
