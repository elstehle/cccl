#!/usr/bin/env bash
# Brute-force search for the nvdisasm cuda-function-index that corresponds
# to a specific mangled kernel symbol. Iterates indices in a range and prints
# the index whose ".text._<symbol>" section appears in the output.
#
# Usage: find_nvdisasm_index.sh CUBIN MANGLED_SUBSTRING [START END]
set -euo pipefail
CUBIN="$1"
SUBSTR="$2"
START="${3:-100}"
END="${4:-250}"

for i in $(seq "$START" "$END"); do
  out=$(nvdisasm --print-life-ranges -lrm count --cuda-function-index "$i" "$CUBIN" 2>/dev/null \
        | grep -m1 -E "^\.text\._.*${SUBSTR}" || true)
  if [ -n "$out" ]; then
    echo "INDEX $i -> $out" | head -c 200
    echo
    break
  fi
done
