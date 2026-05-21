# Round 4 follow-up — split histogram into accumulate + finalize, chunk tiles per CTA

Implements the histogram-pass restructure requested after Round 3:

> "I want to do the looping over tiles that's currently in the kernel into the
> agent. and I want the agent to be smart: we are currently doing these four on
> every tile (while we were smarter in the single-problem implementation):
> init local histogram / add to local histogram / merge histogram / finalize
> pass. This should become (A) init / (B) process tiles for this segment /
> (C) merge, plus a separate finalize kernel that handles all segments at
> once. And introduce a configurable knob for tiles-per-chunk."

Landed in a single commit `66862be78f` on `exp/topk-batched-large-segments-regressions`.

## What changed

Four files, ~350 inserts / ~170 deletes:

1. **`cub/cub/device/dispatch/tuning/tuning_batched_topk.cuh`** —
   `multi_worker_policy` gains `int histogram_tiles_per_chunk`; default `4`.
   The previous `multi_worker_policy{ … }` literals get the field appended;
   `operator==` and `operator<<` updated accordingly.

2. **`cub/cub/agent/agent_batched_topk.cuh`** — the histogram agent's
   `_TempStorage` drops the `block_identify_kth_bucket_t::TempStorage` arm
   (the only thing that was in the union with the smem histogram), so smem
   is now just `OffsetT histogram[num_buckets]` + the keys-source state +
   keys-source scratch. The `run()` method is replaced with a
   `process_chunk(chunk_start, tiles_per_chunk, total_large_tiles, pass)`
   method that:

   - Walks `tiles_per_chunk` consecutive tiles in the queue_idx-space.
   - Resolves each tile's segment via the existing lane-0 +
     `__shfl_sync` `UpperBound` pattern.
   - Caches per-segment derivations (`d_keys_in`, `num_items`,
     `segment_histogram`, `slab_base`, `num_full_tiles`, `partial_items`)
     in registers across tiles of the same segment.
   - On segment change: `__syncthreads()` -> `merge_histogram` of the
     previous segment into its global slab -> `__syncthreads()` ->
     `init_histogram` of the smem for the new segment -> `__syncthreads()`.
   - At chunk end: `merge_histogram` for the last active segment.

   No `finalize_pass`, no per-segment epilogue, no `kth_key_bits` writes.
   The agent is purely write-only with respect to `d_segment_histograms`
   and never touches `segment_counter->finished_block_cnt`.

3. **`cub/cub/device/dispatch/kernels/kernel_batched_topk.cuh`** —
   `device_segmented_topk_histogram_kernel` becomes a nested grid-stride
   loop: outer step of `gridDim.x * tiles_per_chunk`, inner-loop count
   `tiles_per_chunk`, calling `agent.process_chunk(...)`. The
   `tiles_per_chunk` is lifted from the policy via a new
   `histogram_tiles_per_chunk<PolicySelector>` helper. The `reset_histogram`
   parameter is now a no-op for this kernel (consumed by the finalize
   kernel instead). New kernel
   `device_segmented_topk_finalize_histogram_kernel` added: one CTA per
   large segment in a grid-strided loop, runs the prefix-sum +
   bucket-finder + counter update + optional global histogram reset that
   used to live in the per-tile `finalize_pass`.

4. **`cub/cub/device/dispatch/dispatch_batched_topk.cuh`** — instantiates
   the new finalize kernel, queries its `MaxSmOccupancy`, and launches it
   on the same stream right after the histogram kernel for pass 0. Grid
   size is capped at `num_segments_upper_bound` (one CTA per segment
   suffices). The histogram kernel still gets `reset_histogram` but doesn't
   use it; the finalize kernel consumes it.

## Correctness

Verified with the existing Catch2 test suites on the Blackwell container:

```
$ ./bin/cub.test.device.segmented_topk_keys.lid_0
Randomness seeded to: 3481833067
All tests passed (45192 assertions in 42 test cases)

$ ./bin/cub.test.device.segmented_topk_pairs.lid_0.types_0
Randomness seeded to: 1738171676
All tests passed (12960 assertions in 24 test cases)
```

The histogram kernel still produces the same global per-segment histograms;
the new finalize kernel reads them and writes the same per-segment counter
state (`k`, `num_candidates_in/out/written`, `kth_key_bits`) the per-tile
`finalize_pass` epilogue used to. `finished_block_cnt` is no longer touched
on this path -- it stays at `0` after the histogram pass, which is also the
state the post-`finalize_pass` `atomicInc` wrap would have left it in for
the next pass's `finalize_pass`.

## Resource-usage movement (keys.cu, batched vs single-problem on `sm_100`)

`Δ REG` improvements on the **histogram kernel**:

| KeyT  | initial REG | round-4 REG | Δ vs round-3 | single REG | Δ vs single |
|-------|------------:|------------:|-------------:|-----------:|------------:|
| i8    |     56      |    **44**   |   −12        |     28     |  +16        |
| i16   |     54      |    **28**   |  **−26**     |     30     |  **−2** (below single!) |
| f32   |     40      |    **30**   |   −10        |     31     |  **−1** (below single!) |
| i32   |     40      |    **31**   |    −9        |     31     |   0  (at parity) |
| f64   |     40      |    **32**   |    −8        |     30     |  +2 |
| i64   |     40      |    **30**   |   −10        |     30     |   0  (at parity) |
| i128  |     32      |    **27**   |    −5        |     23     |  +4 |

`Δ REG` is 0 on every `filter`, `last_filter`, and `worker` row (the agent
refactor touches only the histogram pass). The histogram movement on the
pairs benchmark is identical (the histogram template body is value-type
independent; the per-`ValueT` instantiations all reduce to the same SASS).

`Δ smem` improvements on the **histogram kernel**:

| bits_per_pass | KeyT          | initial smem | round-4 smem | Δ |
|---------------|---------------|-------------:|-------------:|--:|
| 8             | i8 / i128     |   3072       |   **2052**   | **−1020** |
| 11            | i16/f32/i32/f64/i64 | 9232 |   **9220**   | **−12** |

The 1020 B drop on the narrow-key path is the full prefix-sum scratch
(256 buckets × 4 B = 1024 B) -- it used to be the larger arm of a union
with the smem histogram and dominated smem footprint there.

## New kernel: `device_segmented_topk_finalize_histogram_kernel`

Resource usage (sm_100, `-lineinfo` build):

| KeyT | REG | SHARED | STACK | LOCAL | CONSTANT[0] |
|------|----:|-------:|------:|------:|------------:|
| i8   | 30  |   3072 |   0   |   0   |     957     |
| i16  | 40  |   9216 |   0   |   0   |     957     |
| i32  | 40  |   9216 |   0   |   0   |     957     |
| i64  | 40  |   9216 |   0   |   0   |     957     |
| f32  | 40  |   9216 |   0   |   0   |     957     |
| f64  | 40  |   9216 |   0   |   0   |     957     |
| i128 | 31  |   3072 |   0   |   0   |     957     |

That's the cost of the extra kernel launch per pass: ~30-40 regs, ~3-9 KB
smem, no stack or spills. The per-segment epilogue work is identical to
the old `finalize_pass` epilogue (`block_identify_kth_bucket::find_kth_bucket`
+ counter writes + optional `init_histogram`). The kernel is launched only
once per pass-0 (the histogram pass); subsequent filter passes still run
their own `finalize_pass` internally (those agents are untouched).

## Why the regs go down so much

Three independent factors stack:

1. **Smem footprint shrinks** (especially on bits_per_pass=8 paths)
   because we drop the prefix-sum scratch arm of the smem union. That
   directly shrinks the agent's `_TempStorage` and frees up the smem
   address calculations from that arm too.

2. **`finalize_pass` cost is removed from every tile**. The per-tile
   `__threadfence` + `__syncthreads_or` + lambda capture state for the
   `epilogue_op` closure (capturing `segment_counter`, `segment_histogram`,
   `k`, `pass`, `reset_histogram`, `num_items`) was inflating the per-CTA
   live set. With the closure gone, those captures don't need to be alive
   simultaneously with the tile-load state.

3. **The chunked loop amortises per-tile prologue work** -- per-tile
   `keys_source_t` construction, the segment-id binary search result,
   per-segment scalar caches (`num_items`, `slab_base`, `num_full_tiles`,
   `partial_items`) and per-segment derived pointers (`segment_histogram`)
   stay in registers across same-segment tiles, removing what would have
   been redundant reloads. On a single-large-segment workload the inner
   loop body is essentially just "load tile -> 16 atomic adds to smem"
   for each of the chunk's tiles after the first.

## How far we are from the target

Histogram is now essentially **at parity (or better)** with the
single-problem dispatch on every key type except `i8` (still +16) and
`i128` (+4). The previous worst single-row gap on the histogram pass
(`i8` at +27, narrow-key configurations at +19/+18) is gone.

The remaining headline regressions are all on the **`filter`** kernel
(unchanged by this round) -- `i8` is still at Δ +88, `i8 × i8` pairs
at +72, etc. Those are the next targets if you want to push further.

## What's next

- Optional: sweep `histogram_tiles_per_chunk` (1, 2, 4, 8, 16) and see
  which value works best for the typical workload. Default of 4 is a
  reasonable starting point but is untuned. With chunk=1 we'd lose the
  per-segment amortisation; with very large chunks we waste CTAs at the
  segment-boundary edges.
- Same restructure could be applied to the multi-CTA `filter` kernel
  (the +88 regression on `i8` keys-only filter is the next big target,
  and the same "amortise init/merge across chunked tiles" idea applies
  to that kernel's smem histogram + `finalize_pass`). Worth a try if
  the histogram win translates to runtime gains.

## Files added this round

```
topk_register_pressure_investigation/
├── FOLLOWUP_round_4.md                                  (this file)
├── batched_i8_histogram_max.sass                        (i8 keys-only histogram, post-round-4)
├── batched_finalize_histogram_i8.sass                   (new finalize kernel, i8 instance)
├── keys.after_chunked_histogram.raw                     (resource-usage dump)
├── keys.compare.after_chunked_histogram.txt             (batched-vs-single)
├── pairs.after_chunked_histogram.raw
└── pairs.compare.after_chunked_histogram.txt
```
