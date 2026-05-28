# Kernel-by-kernel trace: `main` vs `lift_part` (I32/I32, Entropy=1.000, 2^24)

Workload: `KeyT = ValueT = OffsetT = OutOffsetT = I32`, `Elements = 2^24`,
`SelectedElements = 2^13`, `Entropy = 1.000`. Captured with
`nsys profile --trace=cuda` on B200 (`umbriel-b200-072` / `bold_mahavira`),
CTK 13.1.115, with `nvbench --profile` so the dispatch runs **once**.

Source artifacts:

- `topk_perf_tracking/profile/profile_main_i32i32_ent1.nsys-rep`
- `topk_perf_tracking/profile/profile_lift_part_i32i32.nsys-rep`
- `topk_perf_tracking/profile/trace_{main_ent1,lift_part}.csv`
- `topk_perf_tracking/profile/trace_{main_ent1,lift_part}.md`

---

## `main` -- single-problem `cub::DeviceTopK::MaxPairs`

- kernels: **4**
- sum of kernel durations: **60.19 us**
- wall-time span (first start -> last end): **71.46 us**

| # | start (us, rel) | duration (us) | kernel |
|---:|---:|---:|---|
| 1 | 0.00 | 24.93 | `DeviceTopKHistogramKernel` |
| 2 | 25.60 | 19.78 | `DeviceTopKKernel` |
| 3 | 46.21 | 11.07 | `DeviceTopKKernel` |
| 4 | 67.04 | 4.42 | `DeviceTopKLastFilterKernel` |

With `bits_per_pass = 11`, I32 needs `ceil(32 / 11) = 3` passes:

- pass 0 -- initial histogram (`DeviceTopKHistogramKernel`)
- pass 1 -- filter + histogram-for-next-pass folded in (`DeviceTopKKernel`)
- pass 2 -- filter + histogram-for-next-pass folded in (`DeviceTopKKernel`)
- final  -- last-filter (`DeviceTopKLastFilterKernel`)

There is **no separate finalize / kth-bucket / counter-update kernel**:
`DeviceTopKKernel` folds it into the tail of each per-pass body.

## `lift_part` -- batched (this CL, `tmp/perf-eval-lift-partition`)

- kernels: **7**
- sum of kernel durations: **74.82 us**
- wall-time span: **78.34 us**

| # | start (us, rel) | duration (us) | kernel |
|---:|---:|---:|---|
| 1 | 0.00 | 22.53 | `device_segmented_topk_histogram_kernel` |
| 2 | 23.23 | 2.75 | `device_segmented_topk_finalize_histogram_kernel` |
| 3 | 26.82 | 21.66 | `device_segmented_topk_filter_kernel` |
| 4 | 49.18 | 3.07 | `device_segmented_topk_finalize_filter_kernel` |
| 5 | 52.96 | 10.24 | `device_segmented_topk_filter_kernel` |
| 6 | 63.49 | 7.71 | `device_segmented_topk_finalize_filter_kernel` |
| 7 | 71.49 | 6.85 | `device_segmented_topk_last_filter_kernel` |

The batched pipeline runs each "phase" as **two kernels**: a per-pass
work kernel + a per-pass finalize kernel:

- pass 0 -- `histogram_kernel` + `finalize_histogram_kernel`
- pass 1 -- `filter_kernel` + `finalize_filter_kernel`
- pass 2 -- `filter_kernel` + `finalize_filter_kernel`
- final  -- `last_filter_kernel` (terminal, no finalize)

## Side-by-side, per phase

| phase | main (us) | lift_part (work + finalize, us) | delta |
|---|---:|---:|---:|
| pass 0 histogram          | 24.93 (`Histogram`)        | 22.53 + 2.75 = **25.28** | +0.35 |
| pass 1 (filter + finalize)| 19.78 (`DeviceTopK`)       | 21.66 + 3.07 = **24.73** | +4.95 |
| pass 2 (filter + finalize)| 11.07 (`DeviceTopK`)       | 10.24 + 7.71 = **17.95** | +6.88 |
| pass 3 last_filter        | 4.42 (`LastFilter`)        | **6.85**                 | +2.43 |
| **sum of kernel durations** | **60.19**                | **74.82**                | **+14.63 us (+24%)** |
| **wall-time span**          | **71.46**                | **78.34**                | **+6.88 us (+10%)** |

Reading:

- **Pass 0 (histogram)**: ~wash. Batched runs its work kernel ~2.4 us
  *faster* than main's combined Histogram kernel, then spends 2.75 us
  in `finalize_histogram_kernel`. Net within 1.4% of main.
- **Pass 1 (filter + finalize)**: main's `DeviceTopKKernel` (which
  also computes the next-pass histogram in the tail of the body) takes
  19.78 us. Batched runs the filter alone in 21.66 us (~2 us slower,
  same `flat_walk`-era cost) and pays an extra 3.07 us for the
  separate finalize. Net **+4.95 us / +25%**.
- **Pass 2 (filter + finalize)**: main 11.07 us. Batched filter is
  *faster* (10.24 us) but the finalize is *substantially heavier*
  (7.71 us vs the pass-1 finalize at 3.07 us). The pass-2 finalize is
  the largest single per-kernel gap to main; later passes have more
  bucket-state to update, and the separate launch geometry pays for
  it. Net **+6.88 us / +62%**.
- **Last filter**: main 4.42 us vs lift_part 6.85 us. Same kernel
  shape; the batched form pays its per-segment-state-resolution
  overhead.

## `lift_part` vs `by_value` at this workload

Recall the previous trace for `by_value` on the *same* workload had
- sum = **74.91 us**, span = **78.43 us**.

`lift_part` is at sum = **74.82 us**, span = **78.34 us** -- effectively
identical (-0.09 us / -0.1%). This is expected: at `Entropy=1.000`
every key is unique, so `cand_reserve_open` never closes (no ties), so
the optimisation that the lift unlocks is dormant. The only effect we
see is a small win in `last_filter` (6.85 us vs 7.49 us, **-0.64 us /
-9%**) from avoiding per-tile partition reconstruction overhead even
when `cand_reserve_open` doesn't fire.

| kernel | by_value (us) | lift_part (us) | delta |
|---|---:|---:|---:|
| histogram_kernel                  | 22.34 | 22.53 |  +0.19 |
| finalize_histogram_kernel         |  2.91 |  2.75 |  -0.16 |
| filter_kernel (pass 1)            | 21.31 | 21.66 |  +0.35 |
| finalize_filter_kernel (pass 1)   |  3.10 |  3.07 |  -0.03 |
| filter_kernel (pass 2)            | 10.05 | 10.24 |  +0.19 |
| finalize_filter_kernel (pass 2)   |  7.71 |  7.71 |   0.00 |
| **last_filter_kernel**            |  **7.49** |  **6.85** |  **-0.64** |
| **sum**                           | 74.91 | 74.82 |  -0.09 |

## Where the remaining gap to main lives (unchanged from prior analysis)

The lift_partition change does not move the `Entropy=1.000` numbers
because the optimisation it enables is tie-driven. The structural cost
breakdown vs main is the same as in `reports/kernel_trace_i32_i32.md`:

1. **+3 extra finalize kernels per dispatch** (~13.5 us). Each
   finalize is small (3-8 us) but represents work that `main` folds
   into the tail of the per-pass work kernel. The cost is the launch
   + per-segment-state-resolve cycle, not the work itself.
2. **Marginal per-kernel structural overhead.** 7 kernels in the
   batched pipeline vs 4 in main; per-segment-state resolution
   (`resolve_queue_idx` + `resolve_segment_state` + per-pass smem
   histogram init for histogram-using kernels) is paid per kernel.

The filter kernels themselves are competitive: pass 1 batched 21.66 us
vs main 19.78 us (despite main also computing the next-pass histogram
inline); pass 2 batched 10.24 us is faster than main 11.07 us.

The two structural costs together account for essentially all of the
+14.63 us kernel-duration gap. Closing them requires fusing the
finalize step into the tail of each work kernel (when feasible
geometrically) -- the same "fold finalize into work kernel" trick
`main`'s `DeviceTopKKernel` already does. That work is independent of
the `cand_reserve_open` fix in this CL.
