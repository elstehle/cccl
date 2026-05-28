# Flat CTA-walk rewrite of `agent_batched_topk_filter_partition::run`

Branch: `exp/flat-cta-walk` (`1fc82b9cbf`).

## Change

`run<TilesPerChunk>(pass)` was a chunked grid-strided loop. Each CTA
iteration owned `TilesPerChunk` tiles (4-8) and split into:

1. **Fast path**: chunk fits within one segment's full-tile range -- a
   fully-unrolled `_CCCL_PRAGMA_UNROLL_FULL` over `TilesPerChunk`
   `dispatch_tile<true>` calls.
2. **Slow path**: chunk crosses a segment boundary, hits a partial-tile
   slot, or runs off the queue tail -- a stretch walk with a power-of-2
   bit decomposition (`if (remaining >= 4) ...; if (remaining >= 2) ...;
   if (remaining >= 1) ...`) plus a trailing-partial branch.

The slow-path variables (`local_tile_start`, `local_stretch_end`,
`local_full_end`, `full_tiles_in_stretch`, `remaining`, `local`,
`stretch_end`, `chunk_cursor`) plus the fast-path-private
`full_tile_boundary` and `local_tile_start` end up alive throughout the
inner tile body because the compiler can't UR-promote them through the
chunk dataflow. The SASS register tracer flagged ~7 of these as
persistent at the int8/int8 peak.

The rewrite drops chunking entirely. One tile per grid-stride iteration:

```cpp
for (LargeSegmentTileOffsetT tile_id = blockIdx.x; tile_id < total; tile_id += gridDim.x)
{
  if (tile_id >= state.queue_segment_end) {
    __syncthreads();
    merge_segment_histogram(state);
    state = resolve_segment_state(resolve_queue_idx(tile_id), pass);
    __syncthreads();
    init_segment_histogram(state);
    __syncthreads();
  }

  if (state.empty) continue;

  const OffsetT local_tile = static_cast<OffsetT>(tile_id - state.slab_base);
  if (local_tile < state.num_full_tiles) {
    dispatch_tile<true>(state, local_tile);
  } else if constexpr (!FullTilesOnly) {
    if (local_tile == state.num_full_tiles && state.partial_items > 0) {
      dispatch_tile<false>(state, state.num_full_tiles);
    }
  }
}
```

`TilesPerChunk` is kept on the template signature for ABI compatibility
but the static_assert is the only place it's referenced.

## Result: filter kernel resources (pairs.cu, OffsetT=int32)

### Per-KeyT register usage

| KeyT | ValueT | dev (chunked) | flat_walk | Δ |
|---|---|---:|---:|---:|
| int8 | int8 | 64 | **40** | **-24** |
| int8 | * (non-int8) | 63 | **40** | **-23** |
| short | * | 40 | 40 | 0 |
| int | * | 32 | 40 | +8 |
| long | * | 40 | 40 | 0 |

### Stack + spill (the regression vs main column)

| metric | (KeyT, ValueT) | dev | flat_walk | Δ |
|---|---|---:|---:|---:|
| stack_frame | int / * | 16-24 | **0** | -16..-24 |
| stack_frame | short / * | 24-32 | **0** | -24..-32 |
| stack_frame | int8 / * | 0 | 24-32 | +24..+32 |
| spill_stores | int / long | 140 | **0** | **-140** |
| spill_stores | int / int, short | 36 | **0** | -36 |
| spill_stores | short / * | 56-80 | **0** | -56..-80 |
| spill_stores | int8 / * | 0 | 28-44 | +28..+44 |
| spill_loads | int / long | 192 | **0** | **-192** |
| spill_loads | int / int, short | 88 | **0** | -88 |
| spill_loads | short / * | 96-120 | **0** | -96..-120 |
| spill_loads | int8 / * | 0 | 20-36 | +20..+36 |

### Summed across all 16 (KeyT, ValueT) pairs, vs main

| metric | dev vs main | flat_walk vs main | Δ (improvement) |
|---|---:|---:|---:|
| registers | +165 | **+104** | -61 |
| stack_frame | +208 | **+104** | -104 |
| spill_stores | +536 | **+128** | **-408** |
| spill_loads | +904 | **+96** | **-808** |
| smem_bytes | -106 688 | -106 688 | 0 (smem unchanged) |

The smem improvement vs main (already in dev) is preserved.

## Result: int8/int8 SASS liveness profile

`KeyT=int8, ValueT=int8` filter kernel, full-kernel liveness histogram:

| | dev (chunked) | flat_walk | change |
|---|---:|---:|---:|
| Total SASS instructions | 36 018 | 2 701 | **-92.5%** |
| Peak live-reg count | 60 | **38** | **-22** |
| Sustained band (>= 5% of instrs) | 50-58 R | **32-37 R** | shifted down by 18-21 |
| Mode of distribution | R=56 (11.0%) | R=36 (11.7%) | -20 |

The full distributions:

```
dev:           50-58 R covers ~70% of kernel
flat_walk:     32-37 R covers ~50% of kernel
```

The "code size" collapse (36 018 -> 2 701 instructions) is the unrolled
chunk body disappearing. Pre-rewrite, the fast path inlined 8 copies of
`dispatch_tile<true>(...)` and the slow path inlined up to 7 more via
the bit decomposition, so the filter pass body was instantiated 8-15
times per chunk-loop iteration. Post-rewrite, the dispatch is once per
grid-stride iteration and the compiler-driven loop does not unroll.
Trade-off: we lose chunk-level ILP. On Blackwell with
`__launch_bounds__(1024)` the warp scheduler should absorb that;
benchmark measurement will tell.

## Side effects

- **`signed char` keys now have a small stack frame (24-32 B) and small
  spill traffic (28-44 B store / 20-36 B load).** Before: zero stack /
  spill, but pinned at the 64-register cap. After: ptxas chose 40
  registers and put the residual in stack. Same occupancy (1 CTA/SM at
  1024 threads/block either way), but introduces a small local-memory
  round-trip. Investigate if benchmark numbers show this as a regression.
- **`int` and `short` keys gain 8 registers** but their spilling (which
  was 28-192 B per thread per kernel call on dev) is gone. Net should
  be positive but their occupancy may step down by one CTA per SM.

## Files

- `cub/cub/agent/agent_batched_topk.cuh` — `run<TilesPerChunk>(pass)`
  rewritten (lines 1833-1944).
- `topk_perf_tracking/snapshots/flat_walk.json` — resource snapshot.
- `topk_perf_tracking/reports/filter_dev_vs_flat.md` — full per-(KeyT,
  ValueT) deltas across all five resource metrics.
- `topk_perf_tracking/reports/int8_int8_flat_peak_liveness.md` —
  source-line attribution of the new peak.
