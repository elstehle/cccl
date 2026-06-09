#!/usr/bin/env bash
# Reproduce the segmented-topk GPU-time sweep (the baseline latency table).
#
# Env overrides:
#   BUILD_DIR   build tree (default: <repo>/build/topk_repro); run build.sh first
#   OUT_DIR     where to write outputs (default: <repo>/topk_repro/out)
#   SEG_SIZES   MaxSegmentSize axis  (default: 131072,262144,524288,1048576)
#   KS          K axis               (default: 512,1024,2048)
#   NSEGS       NumSegments axis     (default: 1,8,32)
#   PATTERN     pattern(s)           (default: random)
#   DEVICE      GPU index            (default: 0)
#   TIMEOUT     per-state timeout s  (default: 8)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build/topk_repro}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/out}"
BIN="$BUILD_DIR/bin/cub.bench.segmented_topk.variable.keys.base"

SEG_SIZES="${SEG_SIZES:-131072,262144,524288,1048576}"
KS="${KS:-512,1024,2048}"
NSEGS="${NSEGS:-1,8,32}"
PATTERN="${PATTERN:-random}"
DEVICE="${DEVICE:-0}"
TIMEOUT="${TIMEOUT:-8}"

[[ -x "$BIN" ]] || { echo "ERROR: benchmark not built ($BIN). Run topk_repro/build.sh first." >&2; exit 1; }
mkdir -p "$OUT_DIR"
JSON="$OUT_DIR/sweep.json"

echo "== sweep: Seg={$SEG_SIZES} K={$KS} NSeg={$NSEGS} Pattern={$PATTERN}"
"$BIN" --device "$DEVICE" \
  -a "MaxSegmentSize{ct}=[$SEG_SIZES]" -a "K{ct}=[$KS]" \
  -a "NumSegments=[$NSEGS]" -a "Pattern=[$PATTERN]" \
  --timeout "$TIMEOUT" --stopping-criterion entropy --json "$JSON" >/dev/null

python3 "$SCRIPT_DIR/parse_sweep.py" "$JSON" | tee "$OUT_DIR/baseline_sweep.txt"
echo "== wrote $OUT_DIR/baseline_sweep.txt (json: $JSON)"
