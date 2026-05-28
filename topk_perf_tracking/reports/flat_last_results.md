# Flat CTA-walk applied to `agent_batched_topk_last_filter::run`

Branch: `exp/flat-cta-walk-plus-last` (`d4ca353952`), on top of the
existing `exp/flat-cta-walk` rewrite of the filter agent.

## Change

Same flat-walk transform we did on the filter agent:

- Drop the `TilesPerChunk`-deep chunk-walk.
- Drop the slow-path stretch walker (`chunk_cursor`, `stretch_end`,
  `local_tile_start`, `local_stretch_end`, `local_full_end`,
  `full_tiles_in_stretch`, `remaining`, `local`) plus the power-of-two
  bit decomposition.
- Replace with a flat tile-per-iteration grid-stride loop. Segment-state
  refresh on `tile_id >= state.queue_segment_end`. Partial-tile
  dispatch is a single conditional.

LOC: -127 / +37 on `cub/cub/agent/agent_batched_topk.cuh`.

`last_filter` has no histogram and no per-segment epilogue, so the body
is even simpler than the filter agent's.

## Last_filter kernel resources (ptxas verbose, sm_100)

Per (KeyT, ValueT), `flat_walk` baseline → `flat_last`:

| KeyT | ValueT | regs | stack | spill_st | spill_ld |
|---|---|---|---|---|---|
| int | int | 40→40 | **16→0** | **56→0** | **72→0** |
| int | long | 40→40 | **16→0** | **84→0** | **92→0** |
| int | short | 40→40 | **16→0** | **56→0** | **72→0** |
| int | signed char | 40→40 | 8→0 | 8→0 | 24→0 |
| long | int | 40→40 | 8→0 | 8→0 | 8→0 |
| long | long | 40→40 | 0→0 | 0→0 | 0→0 |
| long | short | 40→40 | 8→0 | 8→0 | 8→0 |
| long | signed char | 40→40 | 8→0 | 8→0 | 8→0 |
| short | int | **64→40** | 0→24 | 0→68 | 0→60 |
| short | long | **64→40** | 0→24 | 0→68 | 0→60 |
| short | short | **64→40** | 0→24 | 0→68 | 0→60 |
| short | signed char | **64→40** | 0→16 | 0→20 | 0→12 |
| signed char | * | 64→64 | 0→0 | 0→0 | 0→0 |

- `KeyT=int`, `long`: pure win -- the stack frame + spill bytes that
  `last_filter` had in `flat_walk` are gone. No reg change.
- `KeyT=short`: reg cap dropped 64→40 (-24), at the cost of a small
  stack frame (24 B) and ~70 B/iter spill traffic.
- `KeyT=int8`: unchanged. `last_filter` for int8 keys was already
  spill-free, and ptxas picked 64 regs either way.

## End-to-end benchmark (9 workloads, 2^24 elements, 2^13 selected, entropy 1.000)

| KeyT | ValueT |  main | dev | flat_walk | **flat_last** | flat_last/flat_walk |
|---|---|---:|---:|---:|---:|---:|
| **I16** | I64 |  62.10 | 136.55 | 104.92 | **79.17** | **0.755x** |
| **I16** | I8  |  58.53 | 131.83 | 101.41 | **80.47** | **0.793x** |
| I32 | I32  |  63.82 | 115.00 |  90.66 |  87.60 | 0.966x *(new)* |
| I32 | I64  |  63.81 | 115.22 |  90.92 |  87.63 | 0.964x |
| I32 | I8   |  63.51 | 114.64 |  90.61 |  88.60 | 0.978x |
| I64 | I64  | 100.42 | 154.66 | 137.94 | 138.80 | 1.006x (within noise) |
| I64 | I8   | 100.51 | 153.16 | 137.90 | 139.28 | 1.010x (within noise) |
| **I8**  | I64 |  65.68 |  88.70 |  90.09 |  **84.41** | **0.937x** |
| **I8**  | I8  |  65.59 |  94.03 |  93.16 |  **88.19** | **0.947x** |

Reading:
- `flat_last/flat_walk` < 1 means flat_last is *faster* than flat_walk.
- Mean over 9 workloads: flat_last is **7.4% faster than flat_walk** (97.13 us vs 105.18 us mean).
- Biggest wins on **I16** (1.26-1.32x speedup) where last_filter had the
  worst spilling under flat_walk.
- I8 also gains (1.06-1.07x). I8 doesn't run the *filter* kernel
  (`num_passes = 1` so only histogram + last_filter), so last_filter
  dominates and the flat-walk gain shows up here cleanly.
- I64 is within noise (flat_walk already had 0 stack/spill on
  `KeyT=long`, so there was no slack to take back).
- I32 (including the newly-added I32/I32): +2-4%.

## Cumulative gap-closing vs `main`

| KeyT/ValueT | dev vs main | flat_last vs main | gap closed (dev→flat_last) |
|---|---:|---:|---:|
| I16/I64 | 2.20x slower | 1.27x slower | **77%** of regression recovered |
| I16/I8  | 2.25x slower | 1.38x slower | **71%** |
| I32/I32 | 1.80x slower | 1.37x slower | 53% |
| I32/I64 | 1.81x slower | 1.37x slower | 54% |
| I32/I8  | 1.81x slower | 1.40x slower | 51% |
| I64/I64 | 1.54x slower | 1.38x slower | 29% |
| I64/I8  | 1.52x slower | 1.39x slower | 27% |
| I8/I64  | 1.35x slower | 1.29x slower | 22% |
| I8/I8   | 1.43x slower | 1.34x slower | 21% |
| **mean** | **1.72x slower** | **1.37x slower** | **48%** |

Across the 9 workloads, the cumulative effect of `flat_walk` +
`flat_last` is to close roughly **half the regression** the batched
dispatch had vs the single-problem `cub::DeviceTopK::MaxPairs`. I16
specifically goes from ~2.25x slower to ~1.30x slower (gap closed by
~75%).

## What's still on the table

- `KeyT=long` is essentially out of slack on the filter / last_filter
  agents; the remaining ~1.38x gap there is structural to the batched
  pipeline (more kernels per pass, per-segment state resolution,
  iterator-of-iterators wrapping).
- The new `KeyT=short` 64→40 reg drop with introduced stack/spill on
  last_filter is the same trade-off pattern we saw on int8/int8 in the
  filter agent: net positive end-to-end, but worth investigating if the
  spill can be eliminated without forcing the reg cap back up (would
  require ptxas to use a different allocation heuristic, e.g. via
  `__launch_bounds__(threads, blocks_per_sm)` tweaks).
- The cumulative ~50% gap closure suggests the other ~50% comes from
  outside the filter/last_filter agents -- candidates are the
  per-segment state resolution overhead, the extra `finalize_filter`
  and `finalize_histogram` kernels (which `main` doesn't have), and the
  iterator wrapping cost.

## Artifacts

- `topk_perf_tracking/snapshots/flat_last.json`
- `topk_perf_tracking/bench/bench_flat_last.json`
- `topk_perf_tracking/raw_logs/flat_last__pairs.log`
- `topk_perf_tracking/reports/last_filter_flat_walk_vs_flat_last.md`
- Branch: `tmp/perf-eval-flat-last` on origin.
