# Refactor: extract `tile_histogram` block primitive (+ `find_kth_bucket` register overload)

Branch baseline: `508232fdb4` on `exp/topk-batched-large-segments-regressions`.
Verified on `umb-b200-248` (B200, sm_100, CUDA 13.1), `build/sm100-batched`.
Compared against `topk_perf_tracking/baseline/`.

## What changed

- New `detail::topk::tile_histogram<BlockThreads, NumBuckets, CounterT, ExtractBinOpT>` in
  [cub/cub/agent/agent_topk_common.cuh](../../cub/cub/agent/agent_topk_common.cuh): owns the smem
  histogram (nested `TempStorage`), sync-free ops `reset` / `add_full` / `add_partial` (per-call
  `filter_op`) / `load_from` / `flush` / `data` / `make_callback`. Built on existing
  `init_histogram` / `merge_histogram` + new sibling `load_histogram`.
- Adopted in `agent_batched_topk_histogram` (Phase 1) and `agent_batched_topk_filter_partition`
  (Phase 2). All `__syncthreads()` and segment handling stay in the agents.
- `block_identify_kth_bucket`: hoisted `is_full_tile`, added `load_blocked` + a register-input
  `find_kth_bucket(thread_data[bins_per_thread], ...)` overload; the pointer overload now
  loads-then-delegates (kept byte-identical).
- `device_segmented_topk_finalize_histogram_kernel` (Phase 3, `process_partial` mode): stage the
  segment histogram in smem (`load_from` global slab + `add_partial`), read it into blocked
  registers via `load_blocked`, and run the bucket-finder against registers. The staged histogram
  unions with the bucket-finder's `TempStorage` (it is dead before the scan), so smem is unchanged.

## Verification

### Phases 1+2 (extraction): byte-identical
`cuobjdump -sass` of `cub.bench.topk.{keys,pairs}.base` is **identical** to baseline (0-line diff,
both binaries). The refactor relocated code into `tile_histogram` with no codegen change.

### Phase 3 (finalize optimization): localized + measured
- SASS change is confined to **`finalize_histogram`** only: 14 changed kernels (keys) / 32 (pairs)
  = exactly the per-type x direction instantiations. Every other kernel -- including
  `finalize_filter`, which shares `block_identify_kth_bucket` -- is byte-identical.
- `finalize_histogram` resources: **smem unchanged** (union aliasing worked), **no spills/stack**.
  Registers unchanged for f32/f64/i32/i64; +8 (i16), +6 (i128), +18 (i8) for narrow keys
  (max 50 regs).
- Perf (single-segment benchmark, GPU-time mean, v3/baseline):
  - keys (F32): geomean **1.007x** (min 0.998, max 1.033, n=51)
  - pairs (I32/I32): geomean **1.006x** (min 0.999, max 1.032, n=51)
  - Slight regression concentrated at small inputs (2^16); neutral at 2^28.

## Interpretation / recommendation

The single-segment benchmark runs `finalize_histogram` as one CTA ~`num_passes` times, a tiny
fraction of total time; the new path is unconditionally heavier per finalize CTA (always stages
global->smem->registers), so it shows a small net regression here. The optimization is aimed at
finalize-heavy **multi-segment** workloads (many segments each running a finalize CTA), which this
benchmark does not represent.

Options:
1. Keep Phase 3 (bets on multi-segment benefit) -- needs a multi-segment benchmark to confirm.
2. Revert only the Phase 3 finalize change while keeping `tile_histogram` and the
   `find_kth_bucket` register overload (restores byte-identical-everywhere); revisit Phase 3 once a
   multi-segment finalize-heavy benchmark exists.

Artifacts: `topk_perf_tracking/baseline/raw/v3.batched.{keys,pairs}.json` (perf),
`/tmp/topk_resources_v3.csv` + `/tmp/sass.v3.*` on the node (resources / SASS).
