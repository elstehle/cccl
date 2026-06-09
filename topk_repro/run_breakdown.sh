#!/usr/bin/env bash
# Reproduce the per-kernel time breakdown for a fixed config across all patterns,
# using nsys traces averaged over steady-state iterations.
#
# Default config matches reference/breakdown_ns32_1M_k2048.md:
#   NumSegments=32, MaxSegmentSize=1048576, K=2048.
#
# Env overrides:
#   BUILD_DIR   build tree (default: <repo>/build/topk_repro); run build.sh first
#   OUT_DIR     outputs (default: <repo>/topk_repro/out)
#   NS, SEG, K  the fixed config (defaults: 32, 1048576, 2048)
#   PATTERNS    comma list (default: random,quantized_random,relu_quantized,tie_heavy,pivot_tie)
#   DEVICE      GPU index (default: 0)
#   MIN_SAMPLES nvbench min samples per pattern (default: 60)
#   TRACE_TIMEOUT  per-pattern trace timeout s (default: 2)
#
# Requirements: nsys on PATH. If nsys cannot auto-import its .qdstrm, this script
# locates a QdstrmImporter; that importer needs libdw (Debian/Ubuntu: apt-get install -y libdw1).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build/topk_repro}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/out}"
BIN="$BUILD_DIR/bin/cub.bench.segmented_topk.variable.keys.base"

NS="${NS:-32}"
SEG="${SEG:-1048576}"
K="${K:-2048}"
PATTERNS="${PATTERNS:-random,quantized_random,relu_quantized,tie_heavy,pivot_tie}"
DEVICE="${DEVICE:-0}"
MIN_SAMPLES="${MIN_SAMPLES:-60}"
TRACE_TIMEOUT="${TRACE_TIMEOUT:-2}"

[[ -x "$BIN" ]] || { echo "ERROR: benchmark not built ($BIN). Run topk_repro/build.sh first." >&2; exit 1; }
command -v nsys >/dev/null 2>&1 || { echo "ERROR: nsys not found on PATH." >&2; exit 1; }
mkdir -p "$OUT_DIR"

find_qdstrm_importer() {
  command -v QdstrmImporter 2>/dev/null && return 0
  local d
  for d in /opt/nvidia/nsight-systems/*/host-linux-* \
           /opt/nvidia/nsight-compute/*/host/linux-* \
           /opt/nvidia/nsight-compute/*/host/target-linux-*; do
    [[ -x "$d/QdstrmImporter" ]] && { echo "$d/QdstrmImporter"; return 0; }
  done
  return 1
}

IFS=',' read -ra PATS <<< "$PATTERNS"

# 1) Official nvbench GPU totals for all patterns (one converged run).
echo "== collecting official nvbench totals (NS=$NS Seg=$SEG K=$K)"
"$BIN" --device "$DEVICE" \
  -a "MaxSegmentSize{ct}=$SEG" -a "K{ct}=$K" -a "NumSegments=$NS" \
  -a "Pattern=[$PATTERNS]" \
  --timeout 10 --stopping-criterion entropy --json "$OUT_DIR/totals.json" >/dev/null

# 2) Per-pattern nsys steady-state traces -> sqlite.
for P in "${PATS[@]}"; do
  echo "== tracing pattern: $P"
  rm -f "$OUT_DIR/pat_$P.nsys-rep" "$OUT_DIR/pat_$P.sqlite" "$OUT_DIR/pat_$P.qdstrm"
  nsys profile -o "$OUT_DIR/pat_$P" --force-overwrite true -t cuda \
    "$BIN" --device "$DEVICE" \
    -a "MaxSegmentSize{ct}=$SEG" -a "K{ct}=$K" -a "NumSegments=$NS" -a "Pattern=$P" \
    --min-samples "$MIN_SAMPLES" --timeout "$TRACE_TIMEOUT" >/dev/null 2>&1 || true
  if [[ ! -f "$OUT_DIR/pat_$P.nsys-rep" && -f "$OUT_DIR/pat_$P.qdstrm" ]]; then
    IMP="$(find_qdstrm_importer || true)"
    [[ -n "$IMP" ]] || { echo "ERROR: only .qdstrm produced and no QdstrmImporter found." >&2; exit 1; }
    "$IMP" --input-file "$OUT_DIR/pat_$P.qdstrm" >/dev/null 2>&1
  fi
  nsys export --type sqlite --force-overwrite true \
    -o "$OUT_DIR/pat_$P.sqlite" "$OUT_DIR/pat_$P.nsys-rep" >/dev/null 2>&1
done

# 3) Aggregate -> Markdown.
META="Device: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown); commit $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo ?)."
python3 "$SCRIPT_DIR/kernel_breakdown.py" \
  --sqlite-dir "$OUT_DIR" --totals "$OUT_DIR/totals.json" \
  --out "$OUT_DIR/breakdown_ns${NS}_${SEG}_k${K}.md" \
  --patterns "$PATTERNS" --ns "$NS" --seg "$SEG" --k "$K" --meta "$META"

echo "== wrote $OUT_DIR/breakdown_ns${NS}_${SEG}_k${K}.md"
echo "   compare against reference/ with:  git diff --no-index topk_repro/reference/breakdown_ns32_1M_k2048.md $OUT_DIR/breakdown_ns${NS}_${SEG}_k${K}.md"
