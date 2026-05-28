# Kernel trace -- I32/I32 entropy = 0.000 (all-equal keys)

Same workload + same profile method as the entropy = 1.000 report, with
`Entropy = 0.000` (i.e. all input keys are identical).

The benchmark sweep had flagged this as the catastrophic case
(`flat_last / main = 27.06x` geomean at `Elements = 2^24`). This trace
attributes the entire gap to **one** kernel.

## Main (single-problem)

- kernels: **4**
- sum of kernel durations: **414.75 us**
- wall-time span: **425.25 us**

| # | start (us) | duration (us) | kernel |
|---:|---:|---:|---|
| 1 | 0.00 | 21.86 | `DeviceTopKHistogramKernel` |
| 2 | 22.56 | 17.15 | `DeviceTopKKernel` |
| 3 | 40.58 | 16.61 | `DeviceTopKKernel` |
| 4 | 66.11 | **359.14** | **`DeviceTopKLastFilterKernel`** |

## Current batched (`by_value`)

- kernels: **7**
- sum of kernel durations: **11 208.01 us**
- wall-time span: **11 211.62 us**

| # | start (us) | duration (us) | kernel |
|---:|---:|---:|---|
| 1 | 0.00 | 20.19 | `device_segmented_topk_histogram_kernel` |
| 2 | 20.90 | 2.62 | `device_segmented_topk_finalize_histogram_kernel` |
| 3 | 24.35 | 18.18 | `device_segmented_topk_filter_kernel` |
| 4 | 43.20 | 2.75 | `device_segmented_topk_finalize_filter_kernel` |
| 5 | 46.66 | 16.86 | `device_segmented_topk_filter_kernel` |
| 6 | 63.81 | 2.62 | `device_segmented_topk_finalize_filter_kernel` |
| 7 | 66.85 | **11 144.77** | **`device_segmented_topk_last_filter_kernel`** |

## Side-by-side per phase

| phase | main (us) | batched (work + finalize, us) | delta |
|---|---:|---:|---:|
| pass 0 histogram | 21.86 | 20.19 + 2.62 = 22.81 | +0.95 |
| pass 1 (filter + finalize) | 17.15 | 18.18 + 2.75 = 20.93 | +3.78 |
| pass 2 (filter + finalize) | 16.61 | 16.86 + 2.62 = 19.48 | +2.87 |
| pass 3 last_filter | **359.14** | **11 144.77** | **+10 785.63 (+3001%)** |
| **sum** | **414.75** | **11 208.01** | **+10 793.26 us (~27x)** |

The histogram + filter kernels look completely normal -- within ~4 us
of main on every phase, mirroring the entropy=1.000 trace.

**`last_filter_kernel` alone accounts for 99.93% of the gap to main.**

## What's going on

At entropy = 0.000 every input key is identical, so:

- The radix histogram passes are trivial: every key lands in the same
  bucket, every pass collapses immediately. That's why kernels 1-6 are
  fast (and `main`'s first three are also fast).
- After `num_passes` of radix narrowing, every surviving element has
  the same key bits as the kth key. The last-filter kernel's job is
  to pick exactly `k` of those (all equal) elements to emit.
- The path through the last-filter agent for this case is the
  candidate-equal-to-kth path: every item runs through the
  candidate stream and the candidate-reserve atomic, with the
  back-grow cap deciding which items make it into the output's
  trailing slots.

The work pattern is structurally the same for both pipelines -- but
the batched `last_filter_kernel` takes 11.1 ms while main's
`DeviceTopKLastFilterKernel` takes 359 us. **Same algorithm, same
input, ~31x speed difference** on this kernel only.

Likely culprits, ranked by suspicion:

1. **The atomic-reserve back-grow scheme differs between the agents.**
   `agent_batched_topk_last_filter` uses
   `back_grow_capped_reserve_op` from the batched dispatch (the one
   that supports the per-segment back-grow cap so different segments
   can be ranked independently), while the single-problem
   `agent_topk_last_filter` uses a simpler whole-output reserve. When
   every item is a candidate, that's 16M atomic-reserve calls in the
   inner loop -- whatever extra branching or counter math the
   per-segment back-grow op does is multiplied by 16M.
2. **Per-segment-state resolution in the inner loop.** The batched
   agent re-resolves segment state via the
   `LargeSegmentTileOffset` table on each `tile_id >= queue_segment_end`
   crossing, but on the all-equal path the inner classify loop also
   touches per-segment counter fields (`k_total`,
   `num_of_kth_needed`, `num_ties_written_to_back`) on every item. For
   `num_segments == 1` those accesses are uniform, but as we saw with
   the static-pass / value-holding experiments, the compiler doesn't
   always hoist them out of the per-item path.
3. **The `block_partition_atomics` candidate-write codepath.** When
   100% of items are candidates and 0% are selected, the partition
   primitive's atomic candidate-reserve is the hot path on every
   item. The flat_walk register-pressure work we did was on the
   filter agent, not the last-filter agent; the candidate-write
   inner loop in last_filter is the same code path and may still be
   carrying the chunked-shape spill we saw before applying flat_last.

Notably, `flat_last` (which flat-walked the last_filter outer loop)
shaved last_filter's entropy=1 time from main's regime but the
entropy=0 time was already this bad before that change. The flat-walk
outer loop is not the bottleneck -- the per-item candidate-write
inner body is.

## Where to look next

To narrow down which of the three culprits dominates, the fastest
path is:

- `ncu --section MemoryWorkloadAnalysis_Tables` on
  `device_segmented_topk_last_filter_kernel` for this workload to
  see global-atomics throughput numbers (does the atomic reserve
  saturate the global atomic unit?), and
- Side-by-side SASS for `block_partition_atomics::partition_impl<true>`
  in the batched last_filter agent vs the single-problem agent. If
  the inner classify+scatter loop has 2-3x more instructions in the
  batched form, that's our gap.

Either or both will isolate whether it's the atomic-reserve scheme
(culprit 1), the per-segment-state pollution of the inner loop
(culprit 2), or the candidate-write codepath in the partition
primitive (culprit 3).

A first-cut fix to test culprit 1 quickly: when `num_segments == 1`,
swap `back_grow_capped_reserve_op` for the simpler
`atomic_reserve_range_op` (the one selected uses in the same agent
already). One template knob; should be a 5-line change.

## Artifacts

- `topk_perf_tracking/reports/kernel_trace_i32_i32.md` -- entropy = 1.000 trace (this report's sibling)
- `topk_perf_tracking/profile/profile_{main,by_value}_i32i32_ent0.nsys-rep` -- raw nsys reports
- `topk_perf_tracking/profile/trace_{main,by_value}_ent0.csv` -- nsys `cuda_gpu_trace` CSVs
- `topk_perf_tracking/profile/trace_{main,by_value}_ent0.md` -- the per-kernel tables above
