# Kernel-by-kernel trace: `main` vs current batched (`by_value`) on I32/I32

Workload: `KeyT = ValueT = OffsetT = OutOffsetT = I32`, `Elements = 2^24`,
`SelectedElements = 2^13`, `Entropy = 1.000`. Captured with
`nsys profile --trace=cuda` on `umbriel-b200-072` / `bold_mahavira` /
B200 / CTK 13.1.115, with `nvbench --profile` so the dispatch runs once.

## Main (single-problem `cub::DeviceTopK::MaxPairs`)

- kernels: **4**
- sum of kernel durations: **60.29 us**
- wall-time span (first start -> last end): **70.50 us**

| # | start (us, rel) | duration (us) | kernel |
|---:|---:|---:|---|
| 1 | 0.00 | 24.93 | `DeviceTopKHistogramKernel` |
| 2 | 25.63 | 19.49 | `DeviceTopKKernel` |
| 3 | 45.98 | 11.39 | `DeviceTopKKernel` |
| 4 | 66.02 | 4.48 | `DeviceTopKLastFilterKernel` |

I32 with `bits_per_pass = 11` yields `num_passes = ceil(32/11) = 3`,
so the trace is:

  - **pass 0**: initial histogram (`DeviceTopKHistogramKernel`)
  - **pass 1**: filter + histogram-for-next-pass (`DeviceTopKKernel`)
  - **pass 2**: filter + histogram-for-next-pass (`DeviceTopKKernel`)
  - **final**: last-filter (`DeviceTopKLastFilterKernel`)

Note that `DeviceTopKKernel` is a *combined* per-pass kernel -- the
filter pass and the next-pass histogram setup run inside the same
kernel body. There is no separate "finalize" kernel; the kth-bucket
identification / counter update happens at the end of each pass's
filter body before the next launch.

## Current batched (`by_value` / `tmp/perf-eval-by-value`)

- kernels: **7**
- sum of kernel durations: **74.91 us**
- wall-time span: **78.43 us**

| # | start (us, rel) | duration (us) | kernel |
|---:|---:|---:|---|
| 1 | 0.00 | 22.34 | `device_segmented_topk_histogram_kernel` |
| 2 | 23.04 | 2.91 | `device_segmented_topk_finalize_histogram_kernel` |
| 3 | 26.78 | 21.31 | `device_segmented_topk_filter_kernel` |
| 4 | 48.80 | 3.10 | `device_segmented_topk_finalize_filter_kernel` |
| 5 | 52.61 | 10.05 | `device_segmented_topk_filter_kernel` |
| 6 | 62.94 | 7.71 | `device_segmented_topk_finalize_filter_kernel` |
| 7 | 70.94 | 7.49 | `device_segmented_topk_last_filter_kernel` |

The batched pipeline runs each "phase" as **two kernels**: a per-pass
"work" kernel followed by a per-pass "finalize" kernel. For
`num_passes = 3`:

  - **pass 0**: `histogram_kernel` + `finalize_histogram_kernel` (counter update + kth_bucket compute)
  - **pass 1**: `filter_kernel` + `finalize_filter_kernel`
  - **pass 2**: `filter_kernel` + `finalize_filter_kernel`
  - **final**: `last_filter_kernel` (no per-pass finalize, this is the terminal kernel)

The "finalize" step that `main` folds into the tail of each per-pass
kernel is broken out into its own launch.

## Side-by-side per phase

Pair each phase across the two pipelines:

| phase | main (us) | batched (work + finalize, us) | delta |
|---|---:|---:|---:|
| pass 0 histogram | 24.93 (`Histogram`) | 22.34 + 2.91 = **25.25** | +0.32 |
| pass 1 (filter + finalize) | 19.49 (`DeviceTopK`) | 21.31 + 3.10 = **24.41** | +4.92 |
| pass 2 (filter + finalize) | 11.39 (`DeviceTopK`) | 10.05 + 7.71 = **17.76** | +6.37 |
| pass 3 last_filter | 4.48 (`LastFilter`) | **7.49** | +3.01 |
| **sum** | **60.29** | **74.91** | **+14.62 us (+24%)** |
| **wall-time span** | **70.50** | **78.43** | **+7.93 us (+11%)** |

Reading:

- **Pass 0 (histogram)** is roughly the same total time (within
  ~1.3% sum-of-durations). Main's single `DeviceTopKHistogramKernel`
  takes ~25 us; batched runs an equivalent ~22 us work kernel and a
  ~3 us finalize. Net wash.
- **Pass 1**: main's `DeviceTopKKernel` does filter + histogram-of-next
  in 19.5 us. Batched runs the filter alone in 21.3 us (slower by ~2
  us -- the filter body itself has slightly more state to manage) and
  pays an extra 3.1 us for the separate finalize. Net **+5 us / +25%**.
- **Pass 2**: same shape as pass 1. Main 11.4 us, batched filter is
  *faster* (10.0 us) but the finalize is *slower* (7.7 us, vs the
  pass-1 finalize of 3.1 us). The "later pass = more state to update"
  pattern hits the batched dispatch's separate finalize particularly
  hard here.
- **Last filter**: main 4.5 us, batched 7.5 us. Same kernel shape but
  the batched form pays the same per-segment-state-resolution overhead
  it pays everywhere.

## Where the gap actually lives

Two structural costs explain ~12-14 us of the 15 us total gap:

1. **Extra finalize kernels.** Batched has +3 extra kernel launches
   per dispatch (one per pass that has a finalize), adding ~13.7 us
   to the sum-of-durations. Each individual finalize is small
   (3-8 us) and represents work that `main` is folding into the tail
   of the same kernel as the per-pass filter. The cost is not the
   work itself -- it's the launch + per-segment-state-resolve cycle.
2. **Marginal per-kernel overhead.** Each of batched's kernels does
   the same per-segment state resolution (`resolve_queue_idx` +
   `resolve_segment_state` + smem hist init for histogram-using
   kernels) that `main` does once per pass. With 7 batched kernels vs
   4 in main, even a few microseconds of per-kernel structural
   overhead adds up.

The filter kernels themselves (the ones we spent the last several
days flat-walking) are **competitive**: 21.3 us for pass 1 (batched)
vs 19.5 us for `DeviceTopKKernel` on the same pass in `main`, despite
`DeviceTopKKernel` also doing the next-pass histogram setup inside
the same body. Pass 2 is actually *faster* in batched (10.0 vs
11.4 us). The filter-kernel-internal work is not the bottleneck.

## What would close the remaining gap

1. **Fuse `finalize_*` into the per-pass work kernel.** This is what
   `main` does. The batched dispatch split them because the finalize
   has different launch geometry (one CTA per segment vs many CTAs
   per segment for the filter), which is invariant to the inputs but
   matters for batched: the finalize's CTA-per-segment shape is
   useful when segments are many; merging it back into the work
   kernel would re-introduce the "more CTAs than needed for the
   trailing reduce" cost that motivated the split. So a partial fuse
   (keep the split for `num_large_segments` > some threshold, fuse it
   into the tail of the work kernel when `num_large_segments == 1`)
   is plausible. Estimated savings: 5-13 us per dispatch.
2. **Skip pass-0 finalize when `num_segments == 1`.** Main's first
   kernel does the histogram and computes the kth_bucket inline.
   Batched runs `finalize_histogram_kernel` as a separate launch
   (2.9 us in this trace) -- nearly all kernel-launch overhead, since
   the work is one-CTA reduce / scan.
3. **Skip the per-pass `finalize_filter_kernel` when the pass'
   counter update is a single-CTA operation** (i.e. when
   `num_large_segments == 1`). Same logic as #2. Saves 3-8 us per
   non-terminal pass.

For the I32/I32 / 2^24 / 2^13 / 1.000 workload specifically, doing
all three would close roughly 14 us of the ~15 us gap to main.

## Artifacts

- `topk_perf_tracking/profile/profile_{main,byval}_i32i32.nsys-rep` -- raw nsys reports (also `.qdstrm` companions where applicable)
- `topk_perf_tracking/profile/trace_{main,byval}.csv` -- nsys `cuda_gpu_trace` CSV
- `topk_perf_tracking/profile/trace_{main,byval}.md` -- rendered Markdown tables (above)
- `topk_perf_tracking/profile_trace.py` -- the trace-extraction tool
