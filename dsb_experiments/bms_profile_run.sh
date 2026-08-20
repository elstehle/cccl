#!/bin/bash
# Per-kernel breakdown of DeviceMergeSort (main vs PR #10733) via nsys, plus an ncu attempt for
# SASS-level detail on the block-sort kernel. Runs INSIDE the container after bms_bench_run.sh.
# nvbench --profile: one un-batched invocation per state, made for external profilers.

BENCH_ARGS='-d 0 --profile -a T{ct}=[F32] -a OffsetT{ct}=[I32] -a Elements{io}[pow2]=[28]'

echo ===AXES===
/b_main/bin/cub.bench.merge_sort.keys.base --list 2>&1 | head -30

echo ===NSYS_MAIN===
nsys profile -o /root/prof_main --force-overwrite true \
  /b_main/bin/cub.bench.merge_sort.keys.base $BENCH_ARGS > /root/prof_main.log 2>&1 \
  && echo nsys_main OK || tail -20 /root/prof_main.log

echo ===NSYS_FIX===
nsys profile -o /root/prof_fix --force-overwrite true \
  /b_fix/bin/cub.bench.merge_sort.keys.base $BENCH_ARGS > /root/prof_fix.log 2>&1 \
  && echo nsys_fix OK || tail -20 /root/prof_fix.log

echo ===KERNELS_MAIN===
nsys stats --report cuda_gpu_kern_sum /root/prof_main.nsys-rep 2>/dev/null | head -25
echo ===KERNELS_FIX===
nsys stats --report cuda_gpu_kern_sum /root/prof_fix.nsys-rep 2>/dev/null | head -25

echo ===NCU_BLOCKSORT===
# SASS-level view of just the block-sort kernel; requires counter permissions (may be denied in
# container -> the nsys numbers above remain the authoritative breakdown)
if command -v ncu > /dev/null; then
  ncu -k "regex:BlockSortKernel" --launch-count 1 \
      --section LaunchStats --section Occupancy --section SpeedOfLight \
      /b_main/bin/cub.bench.merge_sort.keys.base $BENCH_ARGS > /root/ncu_main.log 2>&1 \
    && grep -E "BlockSortKernel|Registers|Occupancy|Duration|Elapsed|SM \[%\]|Memory \[%\]" /root/ncu_main.log | head -20 \
    || { echo NCU_MAIN_FAILED; tail -5 /root/ncu_main.log; }
  ncu -k "regex:BlockSortKernel" --launch-count 1 \
      --section LaunchStats --section Occupancy --section SpeedOfLight \
      /b_fix/bin/cub.bench.merge_sort.keys.base $BENCH_ARGS > /root/ncu_fix.log 2>&1 \
    && grep -E "BlockSortKernel|Registers|Occupancy|Duration|Elapsed|SM \[%\]|Memory \[%\]" /root/ncu_fix.log | head -20 \
    || { echo NCU_FIX_FAILED; tail -5 /root/ncu_fix.log; }
else
  echo "ncu not found - nsys breakdown above is the deliverable"
fi
