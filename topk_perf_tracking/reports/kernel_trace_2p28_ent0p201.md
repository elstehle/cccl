# Kernel-by-kernel trace: `main` vs `lift_part` (I32/I32, 2^28, sel=2^8, entropy=0.201)

Workload: `KeyT = ValueT = OffsetT = OutOffsetT = I32`, `Elements = 2^28 (256 Mi)`,
`SelectedElements = 2^8 (256)`, `Entropy = 0.201`. Captured with
`nsys profile --trace=cuda` on `umbriel-b200-068` / `clever_hellman` /
B200, CTK 13.1.115, with `nvbench --profile` so the dispatch runs **once**.

Artifacts:
- `topk_perf_tracking/profile/profile_{main,lift_part}_i32_2p28_sel2p8_ent0p201.nsys-rep`
- `topk_perf_tracking/profile/trace_{main,lift_part}_2p28_ent0p201.{csv,md}`

---

## `main` -- single-problem `cub::DeviceTopK::MaxPairs`

- kernels: **4**
- sum of kernel durations: **436.23 us**
- wall-time span (first start -> last end): **444.51 us**

| # | start (us, rel) | duration (us) | share | kernel |
|---:|---:|---:|---:|---|
| 1 | 0.00 | 232.67 | **53.3%** | `DeviceTopKHistogramKernel` |
| 2 | 232.93 | 199.90 | **45.8%** | `DeviceTopKKernel` (pass 1) |
| 3 | 433.09 | **1.82** | **0.4%** | `DeviceTopKKernel` (pass 2) |
| 4 | 442.69 | **1.82** | **0.4%** | `DeviceTopKLastFilterKernel` |

## `lift_part` -- batched (this CL, `exp/topk-batched-large-segments-regressions`)

- kernels: **7**
- sum of kernel durations: **1076.87 us**
- wall-time span: **1079.01 us**

| # | start (us, rel) | duration (us) | share | kernel |
|---:|---:|---:|---:|---|
| 1 | 0.00 | 235.65 | **21.9%** | `device_segmented_topk_histogram_kernel` |
| 2 | 235.94 | 4.22 | 0.4% | `device_segmented_topk_finalize_histogram_kernel` |
| 3 | 240.42 | **237.86** | **22.1%** | `device_segmented_topk_filter_kernel` (pass 1) |
| 4 | 478.82 | 4.22 | 0.4% | `device_segmented_topk_finalize_filter_kernel` |
| 5 | 483.58 | **232.67** | **21.6%** | `device_segmented_topk_filter_kernel` (pass 2) |
| 6 | 716.51 | 4.03 | 0.4% | `device_segmented_topk_finalize_filter_kernel` |
| 7 | 720.80 | **358.21** | **33.3%** | `device_segmented_topk_last_filter_kernel` |

---

## Side-by-side, per phase

| phase | main (us) | lift_part (us, work + finalize) | delta | factor |
|---|---:|---:|---:|---:|
| pass 0 histogram          | 232.67 (`Histogram`)    | 235.65 + 4.22 = **239.87** |    +7.20 | 1.03x |
| pass 1 (filter + finalize)| 199.90 (`DeviceTopK`)   | 237.86 + 4.22 = **242.08** |   +42.18 | 1.21x |
| pass 2 (filter + finalize)| **1.82** (`DeviceTopK`) | 232.67 + 4.03 = **236.70** |  **+234.88** | **130x** |
| pass 3 last_filter        | **1.82** (`LastFilter`) | **358.21**                 |  **+356.39** | **197x** |
| **sum of kernel durations** | **436.23**            | **1076.87**                | **+640.64 us (+147%)** | 2.47x |
| **wall-time span**          | **444.51**            | **1079.01**                | **+634.50 us (+143%)** | 2.43x |

Reading the trace, two things jump out:

1. **`pass 2` and `last_filter` cost essentially nothing on `main`** -- both
   are 1.82 us each (combined: 3.64 us, less than 1% of the dispatch). That
   means after pass 1 main has already narrowed the radix-bucket window so
   tightly that pass 2 and the terminal filter are effectively no-ops on a
   handful of candidates.

2. **`lift_part` runs both at near-full input scale** -- pass 2 still costs
   232.67 us (essentially the same as pass 1 at 237.86 us, i.e. the
   batched filter is processing roughly the full input again), and the
   terminal `last_filter_kernel` costs **358.21 us** (1.5x of a full pass).
   Those two kernels together (590.88 us) are bigger than the entire
   `main` dispatch (436.23 us).

The pass-0 and pass-1 work is **competitive** with main (1.03x and 1.21x
respectively). All of the gap comes from passes 2 and 3.

---

## Where the gap actually lives

The batched filter / last-filter kernels keep processing the **original
input** (or close to it) on every pass instead of narrowing to just the
surviving candidates. On the single-problem dispatch, between passes the
kernel-launch grid + the per-CTA work shrink dramatically once the
kth-bucket is narrow:

- **`main` pass 2** sees only the candidate set the pass-1 filter wrote
  (typically O(items_per_pass) = O(k) for `k = 256`, ~tens of thousands of
  items at most). The compute is ~1.82 us, dominated by launch + bucket-
  finalize.
- **`lift_part` pass 2** still does an apparently *full-input-sized* pass.
  237.86 us -> 232.67 us between passes 1 and 2 is roughly the same
  amount of work, so the candidates are not narrowing the grid.

Same shape for last_filter: main 1.82 us vs `lift_part` 358.21 us.
`lift_part`'s last_filter is **even longer than a full filter pass** at
this workload (358 vs 233 us), suggesting last_filter's per-candidate
work is heavier than the per-input-item work in filter (more atomics /
contention) **and** that it doesn't get to skip items.

The `lift_partition` change we just landed *did* materially improve this
case (6.56x of main on `by_value` -> 2.48x of main on `lift_part`, mostly
by avoiding the per-tile `cand_reserve_open` reset), but the structural
miss is unchanged: passes 2 and last_filter still touch the full input.

---

## What would close the remaining gap on entropy=0.201

Both extra costs (pass-2 filter and last_filter at full-input scale) have
the same root cause: the batched dispatch launches each pass and the
terminal filter with a grid sized to the per-segment **input length**
(`segment_length / tile_items`) rather than to the **surviving candidate
count** from the prior pass.

Closing the gap would require:

1. **Read-and-shrink between passes.** After the per-pass `finalize_filter`
   completes, it knows the number of candidates in the kth bucket
   (`num_of_kth_needed`). The next pass should size its grid to that
   number (rounded up to tiles) rather than the original segment length.
   That alone should drop pass-2 from 232 us to a few microseconds at
   `k=256`.
2. **Same shrink for `last_filter`.** It already reads `kth_key_bits` /
   `num_selected_written`. Sizing its grid to `(k_total -
   num_selected_written + tile_items - 1) / tile_items` (i.e. tiles
   covering only the unresolved tail) should match main's 1.82 us.

Neither is an `agent_*` change -- they are at the dispatch level
(`device_topk_segments_get_max_tiles_per_chunk` + the per-pass kernel
launch geometry in `dispatch_batched_topk.cuh`). They are also
independent of (and additive to) the `cand_reserve_open` lifetime fix
that landed in this branch.

For this single workload the upper bound on speedup from the two changes
together is the gap to main: **2.47x** (i.e. close ~590 us of the
640 us total gap). That would bring `lift_part` from 2.48x of main to
near 1.0x on `(I32/I32, 2^28, sel=2^8, ent=0.201)`. The pass-0 and pass-1
margins (3-21%) are presumably the structural costs we identified on the
entropy=1.000 trace (separate `finalize_*` kernels, etc.).
