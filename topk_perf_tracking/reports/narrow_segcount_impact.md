# Narrow `num_segments_val_t` / `SegmentCountT`: investigation + measurement

## Investigation: what was 64-bit and could be narrowed?

The dispatch and agents had **two distinct types** that both came from
`typename NumSegmentsParameterT::value_type` (often `int64_t`):

1. **`num_segments_val_t`** in `dispatch_batched_topk.cuh` -- used everywhere downstream
   (counting iterators, segment-id buffer, allocations[2] size, segment-id-provider type,
   counter template arg, transform-iterator value type, scan offset type).

2. **`typename NumSegmentsParameterT::value_type`** read inline in five kernels
   (`device_segmented_topk_kernel`, `device_segmented_topk_filter_kernel`,
   `device_segmented_topk_finalize_filter_kernel`,
   `device_segmented_topk_finalize_histogram_kernel`,
   `device_segmented_topk_last_filter_kernel`) and stored as a per-thread cached
   `num_large_segments` field in two agents (`agent_batched_topk_filter_partition`,
   `agent_batched_topk_last_filter`).

The histogram kernel + agent already had a separate `SegmentCountT` template parameter
that the dispatch instantiated with the narrowed type via an inline
`segment_count_fits_u32 -> uint32_t / unsigned long long` rule.

Other types I checked, already narrow:

- **`large_segment_tile_offset_t`**: pinned to `uint32_t` at the dispatch level (line 327).
- **`OffsetT` / `OutOffsetT`**: already conditionally narrowed via
  `offset_fits_u32` / `out_offset_fits_u32`.
- **`segment_size_scan_offset_t`**: already `choose_offset_t<num_segments_val_t>`,
  inherits the narrowing automatically once `num_segments_val_t` is narrow.

So the one outstanding 64-bit type that could be narrowed under the
"static-bound + value-type" rule was `num_segments_val_t`.

## Change

Centralised the narrowing rule in a shared helper
`detail::batched_topk::narrow_num_segments_t<NumSegmentsParameterT>` (in
`agent_batched_topk.cuh`, near the top so every consumer can `#include` it transitively):

```cpp
template <typename NumSegmentsParameterT>
struct narrow_num_segments {
  template <auto Value>
  static constexpr bool fits_u32 =
    static_cast<unsigned long long>(Value)
    <= ::cuda::std::numeric_limits<::cuda::std::uint32_t>::max();

  static constexpr bool fits =
       fits_u32<params::static_max_value_v<NumSegmentsParameterT>>
    || (sizeof(typename NumSegmentsParameterT::value_type) <= 4);

  using type = ::cuda::std::conditional_t<fits, uint32_t, unsigned long long>;
};

template <typename NumSegmentsParameterT>
using narrow_num_segments_t = typename narrow_num_segments<NumSegmentsParameterT>::type;
```

Same schema as `OffsetT` / `OutOffsetT`: pick `uint32_t` if either source ((1) the
parameter's static upper bound or (2) the user-declared `value_type`) admits 32-bit,
fall back to `unsigned long long`.

Plumbed a new `SegmentCountT` template parameter through the four kernels that didn't
have one yet:

- `device_segmented_topk_kernel` (worker_per_segment)
- `device_segmented_topk_filter_kernel`
- `device_segmented_topk_finalize_filter_kernel`
- `device_segmented_topk_finalize_histogram_kernel`
- `device_segmented_topk_last_filter_kernel`

And through the two agents that didn't:

- `agent_batched_topk_filter_partition`
- `agent_batched_topk_last_filter`

The dispatch passes `num_segments_val_t` -- which now resolves through
`narrow_num_segments_t` -- as the `SegmentCountT` template arg at every kernel
instantiation site. The worker_per_segment agent uses the helper directly to
re-derive `num_segments_val_t` from `NumSegmentsParameterT` alone.

## Measurement: dev_full -> narrow_full

`I32` workloads (the common case) keep the same shape as before. Comparison from
`topk_perf_tracking/reports/resources_narrow_vs_dev.md`.

### Wins

- **`initial_histogram` keys-only**: I16 `40 -> 32` regs (-8). Uniform across all I16
  pairs disappear because keys-only/I16 was the only one that previously needed +8
  for cross-tile segment state.
- **`filter` keys-only**: same I16 win, `40 -> 32`.
- **`filter` I8 KeyT / I64 ValueT**: stack `32 -> 24` (-8 B), spill_stores
  `44 -> 24` (-20 B), spill_loads `40 -> 28` (-12 B). The wide-pair worst case for
  the filter kernel improved.
- **`last_filter` I16 KeyT pairs**: regs `56 -> 54` (-2). Modest.
- **`last_filter` I8/I16 KeyT (`signed char` ValueT)**: spill activity eliminated
  (`8/8/8 -> 0/0/0`).
- **Memory footprint**: `allocations[2]` (segment-id buffer) is now `N_seg * 4 B`
  instead of `N_seg * 8 B`. The on-device `batched_topk_counters::large_segments_count`
  also narrows from 8 B to 4 B.

### Regressions / mixed

- **`last_filter` I32 KeyT pairs**: regs unchanged at 40 *but* introduced stack/spill
  (`0/0/0 -> 8/4/8` for narrow ValueTs, `0/0/0 -> 16/28/32` for I64 ValueT).
  Compiler reshuffled register allocation; net effect is a real regression on this
  case.
- **`last_filter` I64 KeyT pairs**: regs `40 -> 52` (+12), but stack/spill dropped
  `8/8/8 -> 0/0/0` for the ValueT that previously spilled. Net effect: more regs in
  exchange for no-spill -- could be a wash or modest improvement on perf depending on
  whether the I64 last_filter was bottlenecked on spill latency (unlikely; the inner
  peak is small here).
- **`last_filter` I128 keys-only**: `32 -> 40` (+8). I128 was already a niche case.
- **`last_filter` I8/I16 keys-only**: `+1 / +2` regs.
- **`finalize_histogram` I16/I32 KeyT**: `32 -> 38` (+6 regs across all ValueTs).

### Distribution

| delta (regs) | count |
|---:|---:|
| -24 | 3 |
| -20 | 1 |
| -13 | 1 |
| -12 | 5 |
| -10 | 1 |
| -8 | 1 |
| -6 | 4 |
| -5 | 4 |
| -4 | 1 |
| -2 | 4 |
| -1 | 5 |
| **0 (unchanged)** | **79** |
| +1 | 12 |
| +2 | 1 |
| +6 | 11 |
| +8 | 1 |
| +12 | 4 |

29 cases unchanged or with regressions (counting +1, etc. as small regressions);
30 cases with improvements (1 to 24 regs); 79 cases unchanged.

## Why isn't this a clean win?

Two effects compound:

1. **Cross-tile holding state really did shrink**: `num_large_segments` is now 1 reg
   smaller per thread, the binary search uses `uint32_t` indexing, and on the mixed
   path the segment-id buffer reads return `uint32_t` instead of `int64_t`.
   That's the structural intent of the change.

2. **Ptxas register allocation is non-monotonic**. When a kernel had a small spill
   slot before (e.g. `I8/I64` filter or `I64/I8` last_filter), the freed-up registers
   from cross-tile narrowing let the allocator promote the previously-spilled values
   back into registers -- so register count goes *up* but spill goes to zero
   (likely a perf win). Conversely, on `I32` last_filter the freed registers
   triggered the allocator to inline more values that didn't fit, creating a small
   new spill (likely a small perf loss).

The change is conceptually right -- it narrows what was a structurally-unnecessary
64-bit type to 32-bit when the workload's static / runtime upper bounds permit it,
and propagates the narrowing through the segment-id buffer + counter struct + agent
fields -- but ptxas's response is workload-dependent. Some kernels regress in
register count or pick up a new small spill; others see real wins.

## Recommendation

I'd land this regardless of the per-kernel jitter:

1. **The schema is the right one** -- it matches what we already do for `OffsetT` /
   `OutOffsetT`, and centralising it in `narrow_num_segments_t` makes future adoption
   in other CCCL dispatchers a one-liner.

2. **The non-register wins are unconditional**: 50% smaller `large_segments_ids`
   buffer, half the size of the on-device `large_segments_count` atomic. These help
   any workload with many segments regardless of ptxas's allocator decisions.

3. **The register jitter has a clear next step**: the I32 `last_filter` spill is the
   only meaningful regression. It comes from how the allocator chose to use the
   freed slots; addressing it is a follow-up that's independent of the narrowing
   (e.g. via the `cand_reserve_open`-out-of-partition refactor I sketched earlier,
   which has a much bigger projected impact on the same kernel).

If you want, I can:

- (a) commit this and move on to the partition-state refactor, OR
- (b) revert and only land the dispatch-side narrowing (no kernel/agent template
  plumbing) -- that captures the smaller-buffer / smaller-counter wins without
  touching agent register pressure, OR
- (c) run end-to-end perf on the I32 / I32 / 2^28 sweep to confirm the I32
  last_filter spill regression actually matters at runtime before deciding.

Branch with the change: `tmp/perf-eval-narrow-segcount` at commit `931ba49866`.
