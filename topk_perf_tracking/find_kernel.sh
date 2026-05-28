#!/usr/bin/env bash
# Find filter kernel symbols matching given KeyT, ValueT, and select direction.
# Usage: find_kernel.sh CUBIN KeyT ValueT select [logical_name]
# example: find_kernel.sh pairs.sm_100.cubin "signed char" "signed char" 1 filter
set -euo pipefail
CUBIN="$1"
KT="$2"
VT="$3"
SEL="$4"
LOGICAL="${5:-filter}"

case "$LOGICAL" in
  filter) MATCH='device_segmented_topk_filter_kernel' ;;
  last_filter) MATCH='device_segmented_topk_last_filter_kernel' ;;
  finalize_filter) MATCH='device_segmented_topk_finalize_filter_kernel' ;;
  finalize_histogram) MATCH='device_segmented_topk_finalize_histogram_kernel' ;;
  initial_histogram) MATCH='device_segmented_topk_histogram_kernel' ;;
  single_cta) MATCH='device_segmented_topk_kernel' ;;
  DeviceTopKKernel|filter_main) MATCH='DeviceTopKKernel' ;;
  *) echo "unknown logical: $LOGICAL"; exit 2 ;;
esac

cuobjdump --dump-elf-symbols "$CUBIN" \
  | awk '$1=="STT_FUNC"{print $4}' \
  | grep -v 'finalize' \
  | grep -E "$MATCH" > /tmp/_syms.mangled

# Demangle and align
mangled=$(cat /tmp/_syms.mangled)
demangled=$(echo "$mangled" | c++filt)
paste <(echo "$mangled") <(echo "$demangled") \
  | awk -v kt="$KT" -v vt="$VT" -v sel="$SEL" -F'\t' '
      {
        d=$2; m=$1
        if (index(d, "policy_selector_from_types<" kt ", " vt ",") > 0 \
            && index(d, "topk::select)" sel) > 0) {
          print m
        }
      }'
