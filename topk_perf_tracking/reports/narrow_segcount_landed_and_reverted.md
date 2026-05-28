# `num_segments_val_t` narrowing: landed, end-to-end perf'd, reverted

## TL;DR

The narrowing change was committed to dev (`931ba49866`), end-to-end perf'd against
the pre-narrow build (`581b735ec0`) on the I32 / I32 sweep, and reverted (`580e7b9b6b`)
because it caused up to **21x runtime slowdown** on `Entropy=0.000` workloads despite
the resource snapshot showing only modest changes. Concept is right; the agent/kernel
template plumbing needs more investigation before re-introducing.

## End-to-end I32 / I32 sweep: dev_v3 (pre-narrow) vs narrow

`compare_sweep.py` over 51 (Elements x SelectedElements x Entropy) cells:

| | narrow / dev_v3 |
|---|---|
| Geometric mean | **1.77x slower** |
| Median | 1.00x (essentially neutral on most cases) |
| Min | 0.97x (small wins on Entropy=1.000 / 0.201) |
| Max | **21.21x slower** (Entropy=0.000 / Elements=2^24 / Sel=2^23) |

**The regressions are concentrated on `Entropy=0.000`** -- the workload where
`cand_reserve_open` cross-tile persistence pays off:

| Elements | Sel | Entropy | dev_v3 (us) | narrow (us) | factor |
|---|---|---|---:|---:|---:|
| 2^20 | 2^3 / 2^8 / 2^13 / 2^18 | 0.000 | ~62 | ~636 | **10x** |
| 2^24 | 2^3 / 2^8 / 2^13 / 2^18 | 0.000 | ~108 | ~684 | **6.3x** |
| 2^24 | 2^23 | 0.000 | 280 | 5945 | **21.2x** |
| 2^24 | 2^23 | 0.201 | 471 | 5359 | **11.4x** |
| 2^28 | 2^23 | 0.000 | 1122 | 6705 | **6.0x** |

Entropy=1.000 cases are essentially unchanged or 1-3% better.

## Per-kernel diagnosis on the worst case

`I32/I32 / 2^24 / sel=2^23 / ent=0.000`, captured with `nsys profile --trace=cuda`:

| kernel | dev_v3 (us) | narrow (us) | delta |
|---|---:|---:|---:|
| histogram | 20.45 | 20.06 | ~unchanged |
| finalize_histogram | 2.50 | 2.46 | ~unchanged |
| filter (pass 1) | 18.24 | 18.11 | ~unchanged |
| finalize_filter (pass 1) | 2.82 | 2.66 | ~unchanged |
| filter (pass 2) | 16.45 | 16.26 | ~unchanged |
| finalize_filter (pass 2) | 2.56 | 2.53 | ~unchanged |
| **last_filter** | **201.95** | **5887.18** | **+5685 us / 29x** |

The whole regression lives in `last_filter`. The kernel runs at the same grid
(`grid=444, block=512, regs=40`) and has the same atomic-instruction count
(152 `ATOM` / `RED`), and the SASS body is even ~9% smaller post-narrow. So the
slowdown isn't "the kernel is doing more visible work" -- it's a stall pattern.

## Why it regressed (best hypothesis)

The narrow build's resource snapshot showed:

- I32 last_filter pairs: regs unchanged at 40, **stack 0->8, sp_st 0->4, sp_ld 0->8**.
- I64 last_filter pairs: regs 40 -> 52, stack/spill went the other way (eliminated).

The 4-8 byte spill on I32 last_filter is exactly the size of `cand_reserve_open`.
If ptxas decided to spill that flag to local memory after the type narrowing freed
up a register slot, every tile entry now does:

  - load `cand_reserve_open` from local mem (memory latency, ~50-100 cycles)
  - branch on it
  - dispatch to `HasCandidateStream=true/false` specialisation

Across thousands of tiles in the entropy=0 / all-ties workload, this turns the
"cheap exit hint" optimisation into a memory-latency-bound branch on every tile.
That's the only mechanism I can see that explains a 29x last_filter slowdown
from a kernel that's structurally identical otherwise.

The hypothesis is consistent with:
- Entropy=1.000 is ~unchanged (no all-ties; cand_reserve_open never closes regardless).
- Entropy=0.000 / large-Sel cases regress most (where cand_reserve_open should close
  early and stay closed, but now requires per-tile reload).

I haven't proved it (would need to add a debug printf inside the partition primitive
and rerun, or read the SASS for the `cand_reserve_open` branch directly), but the
shape matches.

## Status

- Reverted on `exp/topk-batched-large-segments-regressions` at commit `580e7b9b6b`.
  Post-revert verification: I32/I32/2^24/sel=2^23/ent=0.000 back to **283 us**
  (matches dev_v3's 280 us, vs narrow's 5945 us).
- The change still exists on the side branch `tmp/perf-eval-narrow-segcount` at
  commit `931ba49866` for future reference.

## Other variables that are 64-bit and could be 32-bit

Investigation only -- no changes made.

### Already narrowed (no action needed)

- **`OffsetT`** -- per-segment offsets / counters. Conditionally `uint32_t` via
  `offset_fits_u32` rule.
- **`OutOffsetT`** -- per-segment `k` counters / num-selected / num-ties counters.
  Conditionally `uint32_t` via `out_offset_fits_u32` rule.
- **`large_segment_tile_offset_t`** -- pinned to `uint32_t` (8.8 trillion items
  ceiling, well beyond realistic).
- **`segment_size_scan_offset_t`** -- `choose_offset_t<num_segments_val_t>`. Inherits
  the dispatch's `num_segments_val_t` width. Was 64-bit, still 64-bit after the revert
  (since `num_segments_val_t` is back to the user's wide type).
- **`current_k`, `current_len`, `input_length`, `num_full_tiles`, `partial_items`,
  `segment_tiles_input`** in `per_segment_state_t` -- all `OffsetT` / `OutOffsetT`,
  inherit the conditional narrowing.

### Candidates that were tried + reverted

- **`num_segments_val_t`** (= `NumSegmentsParameterT::value_type`, often `int64_t`).
  Used for the segment-id buffer's element type, the on-device counters struct's
  `large_segments_count` atomic, the segment-id-provider iterator's value type, the
  scan offset type, and the `num_large_segments` field in the multi-CTA agents'
  cached state. The full narrowing-through-the-agents path regressed performance;
  the dispatch-only narrowing (smaller buffer + counter) wasn't separately tested
  but is plausible.

### New candidates I found

#### 1. `segment_size_val_t` in `agent_batched_topk_worker_per_segment` (line 89)

```cpp
using segment_size_val_t = typename SegmentSizeParameterT::value_type;
```

This is the user's wide `int64_t` segment-size type. Used as the element type of:

- `block_load_epilogue_t  = BlockLoad<segment_size_val_t, ...>`
- `block_scan_epilogue_t  = BlockScan<segment_size_val_t, ...>`
- `block_store_epilogue_t = BlockStore<segment_size_val_t, ...>`

These run inside the worker_per_segment kernel's epilogue, scanning a per-segment
**tile-count** array (indexed by large-segment slot). The values being scanned
are `large_segment_tile_offset_t` (= `uint32_t`); the running total fits in
`uint32_t` because it's bounded by the same `large_segment_tile_offset_t` ceiling.

The comment at line 132 specifically calls out the choice:

```
// the accumulator can hold the running total across all tiles (a large-segment-rich
// workload can accumulate well beyond 2^31 tiles).
```

So the author was thinking about >2^31 tiles. With `large_segment_tile_offset_t =
uint32_t` we already cap at 2^32 tiles, so `segment_size_val_t = uint32_t` would
match the ceiling without losing any addressable workload. **Worth narrowing**, but
only after verifying that the worker_per_segment kernel doesn't see the same
ptxas-spill pathology that hit `last_filter` on this CL.

#### 2. `counting_iterator<num_segments_val_t>` difference type (in dispatch)

`cuda::counting_iterator<T>` chooses its `difference_type` at instantiation time;
for some `T` it ends up as `__int128`. Inspecting the SASS-mangled types:

```
counting_iterator<long, __int128, 0, 0, 0>     // dev_v3 (pre-narrow)
counting_iterator<unsigned int, long, 0, 0, 0> // narrow attempt
```

The `__int128` difference type is concerning -- it forces 128-bit arithmetic for
internal index math (e.g. `iterator + queue_idx`). On B200 that codegens to
multi-instruction sequences. Reducing the iterator's value-type to a narrower
unsigned tightens the difference_type to 64-bit (or 32-bit on some specialisations).

This was inside the `narrow_num_segments_t` change that we reverted. **Worth
revisiting** as a targeted fix on just the iterator types (without touching agent
fields or kernel template signatures), e.g. by templating these iterators on a
`narrow_num_segments_t` derivation locally in the dispatch but plumbing the original
wide type through to the agents.

#### 3. Explicit `int64_t` host-side scratch in dispatch

Lines 556-561 / 817-818: host-side computation of `candidate_buffer_length` and
`total_large_tiles_upper_bound` uses `::cuda::std::int64_t`. These are not on the
device hot path, so leaving them as 64-bit is correct (overflow safety). **No
action needed** -- listed for completeness.

#### 4. `1ull` literal in `atomicAdd` against narrowed counter (line 270)

```cpp
const auto large_segment_queue_idx = atomicAdd(&d_counters->large_segments_count, 1ull);
```

Currently safe because `large_segments_count` is 64-bit (`choose_offset_t<int64_t>
= unsigned long long`). After a future narrowing of the counter struct's element
type, `1ull` becomes a u64-to-u32 implicit conversion -- functionally fine, but
should be changed to `1u` (or to an explicit `static_cast<segment_count_t>(1)`)
to make the type match obvious. **Cosmetic / defensive fix**; track for the
re-attempt of the narrowing.

#### 5. Iterator difference types via default template defaults

`tile_data_source_t<It, ..., OffsetT = int64_t>` and friends in
`detail/topk/tile_data_source.cuh` default the offset to `int64_t`. The agents
override with their own narrowed `OffsetT` (the conditionally-32-bit one), so no
leakage today. **No action**, but keep an eye on this when refactoring.

#### 6. Iterator values returned by `params::*::get_param(segment_id)`

`segment_sizes.get_param(segment_id)` returns `int64_t` (the user's declared
type). Every device-side call I see immediately casts to `OffsetT` / `OutOffsetT`,
e.g.:

```cpp
const OffsetT num_items = static_cast<OffsetT>(segment_sizes.get_param(segment_id));
```

So the wide type is consumed at the host-API boundary and immediately narrows.
**Already handled correctly.**

### Things you'd think are 64-bit but aren't

- `pass`, `total_bits`: `int` (32-bit). Good.
- `bits_per_pass`: compile-time `int`. Good.
- `num_buckets`: compile-time `int`. Good.
- `tile_items`, `block_threads`, `items_per_thread`: compile-time `int`. Good.
- `slab_base`, `queue_segment_end`: `LargeSegmentTileOffsetT` (= `uint32_t`). Good.
- `kth_key_bits`: `key_prefix_storage_t<KeyInT>` -- sized by KeyT. Good.
- `cand_reserve_open`: `bool`. Good.
- Counter struct's `num_candidates_in/out`, `k`, `num_selected_written`,
  `num_ties_written_to_back`, `num_candidates_written`: all `OffsetT` /
  `OutOffsetT`. Good.

## Recommendation

After the current revert:

1. **Leave the narrowing change reverted** until we've root-caused the
   `last_filter` slowdown (the `cand_reserve_open` spill hypothesis above is the
   most likely cause; can be confirmed with a debug-printf rebuild).
2. **Pure dispatch-side narrowing** (smaller `large_segments_ids` buffer + smaller
   `batched_topk_counters::large_segments_count`) is independently safe to land --
   it doesn't touch the agents' register allocation. Could be done as a separate
   small CL.
3. **Iterator difference-type narrowing** (item 2 above) is the cleanest follow-up:
   it directly targets the `counting_iterator<...,__int128,...>` codegen weight
   without affecting agent fields. Lower risk than the full narrowing.
4. **Don't narrow `segment_size_val_t`** (item 1) without first verifying it
   doesn't trigger the same `cand_reserve_open`-style spill pathology -- the
   worker_per_segment kernel is much smaller than `last_filter`, so the risk is
   probably lower, but worth measuring before landing.
