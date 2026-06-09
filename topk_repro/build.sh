#!/usr/bin/env bash
# Configure + build the segmented-topk benchmark used by the repro scripts.
#
# Env overrides:
#   CUDA_ARCH   GPU arch(s) to compile for (default: 100 = B200)
#   BUILD_DIR   build tree location      (default: <repo>/build/topk_repro)
#   CMAKE       path to a CMake >= 4.0   (auto-detected if unset)
#   PRESET      CMake preset             (default: cub-cpp17)
#   FORCE       set to 1 to re-run configure even if BUILD_DIR exists
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CUDA_ARCH="${CUDA_ARCH:-100}"
PRESET="${PRESET:-cub-cpp17}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build/topk_repro}"
TARGET="cub.bench.segmented_topk.variable.keys.base"

# --- find a CMake >= 4.0 (rapids-cmake, pulled in by the benchmarks, requires it) ---
find_cmake() {
  if [[ -n "${CMAKE:-}" ]] && "$CMAKE" --version >/dev/null 2>&1; then echo "$CMAKE"; return; fi
  local c v major
  for c in $(command -v cmake 2>/dev/null || true) \
           /cccl/cmake/*/bin/cmake /opt/cmake*/bin/cmake /usr/local/cmake*/bin/cmake; do
    [[ -x "$c" ]] || continue
    v="$("$c" --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)"
    major="${v%%.*}"
    if [[ "${major:-0}" -ge 4 ]]; then echo "$c"; return; fi
  done
  echo ""
}

CMAKE_BIN="$(find_cmake)"
if [[ -z "$CMAKE_BIN" ]]; then
  echo "ERROR: no CMake >= 4.0 found (the benchmarks pull in rapids-cmake which needs it)." >&2
  echo "       Install one and/or set CMAKE=/path/to/cmake." >&2
  exit 1
fi
command -v ninja >/dev/null 2>&1 || { echo "ERROR: ninja not found on PATH." >&2; exit 1; }

echo "== repo:   $REPO_ROOT"
echo "== cmake:  $CMAKE_BIN ($("$CMAKE_BIN" --version | head -1))"
echo "== build:  $BUILD_DIR"
echo "== arch:   $CUDA_ARCH   preset: $PRESET"

cd "$REPO_ROOT"
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" || "${FORCE:-0}" == "1" ]]; then
  "$CMAKE_BIN" -G Ninja -B "$BUILD_DIR" --preset "$PRESET" \
    -DCCCL_ENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
fi
"$CMAKE_BIN" --build "$BUILD_DIR" --target "$TARGET"

echo "== built:  $BUILD_DIR/bin/$TARGET"
