# Async keys-load evaluation: tests + benchmarks

Followed up on the by-reference `multi_source_data_source` refactor by
switching the default `keys_tile_load_kind` to `block_load_to_shared_async`
in `tuning_batched_topk.cuh` and running the topk device tests + the I32/I32
pairs benchmark sweep on `umb-b200-248`.

**Branches:**

| branch | tip | what's there |
|---|---|---|
| `exp/topk-batched-large-segments-regressions` (dev) | `8ba40648d8` | by-reference multi-source + handle fix (async-compatible) |
| `exp/topk-eval-async-keys` | `5f54e1edb1` | dev + default keys-load switched to `block_load_to_shared_async` |

## Bug fix that landed on dev along the way

Switching to async exposed a latent bug in the multi-source's load handles
that I introduced in the prior by-ref refactor. The handles were
aggregate-initialising both arms (`SourceA::full_load_handle a{}; SourceB::full_load_handle b{};`),
which:

1. Doesn't compile when an arm carries `async_to_shared_data_source`'s
   `loader_t::CommitToken` — `CommitToken`'s default ctor is intentionally
   inaccessible and its copy ctor is deleted (move-only).
2. Even with an empty default ctor it would have failed at the `return out;`
   site in `submit_load` — the union of a move-only arm with a user-declared
   ctor/dtor leaves the outer handle with deleted copy *and* deleted move,
   and NRVO is optional.

Fixed by restructuring the handle:

- Union with no-op user-declared ctor / dtor (only the active arm is ever
  materialized).
- Tagged ctors `from_a_t{} / from_b_t{}` that placement-new exactly one arm.
- `submit_load` returns a prvalue via these ctors → C++17 mandatory copy
  elision constructs the handle directly in the caller's slot.
- Explicit move ctor that placement-news the active arm into the
  destination based on the source's `pick_b` (NRVO fallback).
- Explicit dtor that destructs only the active arm.
- Deleted default / copy / assignment ops so misuse fails at the declaration.

Landed on dev as `8ba40648d8` -- "topk(multi-source): refactor load
handles for non-default-constructible arms". **SASS byte-identical** to
the previous dev tip for both `pairs.base` (`a61af970e6c69a5c46e323b1fc929cb9`)
and `keys.base` (`0a1c60ede9b3254e1f716ff2e760c775`): the active-arm
placement-new lowers to the same writes the previous aggregate-init shape
did for direct / sync_block_load children.

## Test results

### Dev tip (default `block_load_vectorize`)

```
20/20 tests passed:
  cub.test.device.topk_api.lid_0                            ✓
  cub.test.device.topk_keys.lid_{0,1,2}                     ✓
  cub.test.device.topk_pairs.lid_{0,1,2}                    ✓
  cub.test.device.topk_tile_data_source.lid_0               ✓
  cub.test.device.segmented_topk_keys.lid_{0,1,2}           ✓
  cub.test.device.segmented_topk_pairs.lid_{0,1,2}.types_{0,1,2} ✓
```

### Async tuning (`block_load_to_shared_async`)

```
Non-segmented + small-segment tests:
  cub.test.device.topk_api.lid_0                            ✓ (1.4s)
  cub.test.device.topk_keys.lid_{0,1,2}                     ✓ (65-88s)
  cub.test.device.topk_pairs.lid_{0,1,2}                    ✓ (21-64s)
  cub.test.device.topk_tile_data_source.lid_0               ✓ (0.8s)

Segmented (multi-CTA all-large):
  cub.test.device.segmented_topk_pairs.lid_2.types_1        ✗ FAIL
    4 of 8 test cases failed with
      `std::bad_alloc: cudaErrorLaunchFailure: unspecified launch failure`
    All failures on "DeviceBatchedTopK::{Min,Max}Pairs work with large
    fixed-size segments (multi-CTA all-large)" workloads.
```

`cudaErrorLaunchFailure` is a real device error (not a timeout), so the
async config isn't just slow on the large-segment multi-CTA cases -- it's
actually mis-executing the kernel.

## Benchmark results (`pairs.base`, I32/I32, full sweep)

```
Per-(KeyT, ValueT) summary: async vs dev baseline:

| KeyT | ValueT | n  | geo mean | median | best  | worst   |
|------|--------|----|----------|--------|-------|---------|
| I32  | I32    | 84 | 1.917x   | 1.437x | 1.040x| 21.233x |
```

Worst regressions (a sample):

```
| Elements    | SelectedElems  | Entropy | dev (us) | async (us) | async/dev |
|-------------|----------------|---------|----------|------------|-----------|
|  16777216   |  8388608       |   0.000 |   278.83 |    5920    | 21.233x   |
|  16777216   |  8388608       |   0.201 |   473.85 |    5540    | 11.689x   |
|   1048576   |    524288      |   0.000 |    61.10 |     659    | 10.790x   |
| 268435456   |  8388608       |   0.000 |  1130    |    7530    |  6.667x   |
```

All worst-case cells cluster around `Entropy=0.000` (the "all-equal keys"
degenerate workload that runs many unbuffered filter passes). For that
class of workloads async makes things 6-21x slower; the rest of the sweep
is between 1.04x and ~3x slower with a 1.44x median. **No cell improved**;
the best entry is still 1.04x slower than dev.

## Why async is broken here

A few candidate causes, all interacting with the multi-source's "both arms
alive" design:

1. **Both arms become async, but only one is ever loaded per submit.** The
   tuning sets `keys_tile_load_kind` globally, so `multi_source<async,
   async>` is instantiated. Each child source then needs its own
   mbarrier in agent smem (`storage.key_src_input_state.barrier` and
   `storage.key_src_buffer_state.barrier`). The agent owns both, but only
   one barrier participates in each tile's TMA. The inactive arm's
   barrier is initialized at the agent's source-construction but never
   used -- and the per-segment destroy-then-construct at the segment
   boundary tears down the loader bound to it. This is at least wasteful
   smem and may be a correctness hazard if the barrier's destructor
   isn't a true no-op.

2. **Multi-CTA on the same segment + async TMA.** The
   `cudaErrorLaunchFailure` is concentrated on multi-CTA-all-large
   segmented workloads. Multiple CTAs operating on the same segment is a
   pattern where the async loader's per-CTA-private barrier should still
   be safe in principle, but the failure suggests something about the
   `mbarrier`/`cp.async` state isn't right -- possibly a missing
   commit/wait pair when the segment boundary's destroy-then-construct
   rebuilds the async source mid-segment-traversal across CTAs.

3. **`Entropy=0.000` runs many filter passes.** With all keys equal, the
   algorithm hits the "stay in the same bucket" degenerate case --
   roughly `ceil(sizeof(KeyT) * 8 / bits_per_pass)` passes, each
   processing every input element. Each pass goes through async load.
   If the async load's per-tile overhead is large (TMA setup, barrier
   wait), it'll get amortized over many fewer items than e.g. the
   buffered path. The 6-21x cluster is exactly where this would show up.

## Recommendation

The current default-tuning experiment ("just flip to async") is not viable
-- both correctness (multi-CTA large-segment failures) and performance
(geo mean 1.9x slower, no improvements) regress.

To benchmark async productively the path forward would be:

1. **Use `<async, direct>` instead of `<async, async>`** — only the
   input arm is interesting to async-load (it's the gmem-resident
   read-only stream); the candidate buffer is already chip-local. This
   needs the policy to support different load-kinds per arm.
2. **Investigate the multi-CTA segmented launch failure**. Likely a
   handshake issue between the destroy-then-construct refresh at segment
   boundaries and the async loader's persistent mbarrier state. The
   loader's `Invalidate()` API exists for exactly this kind of refresh
   -- the agent's segment-boundary code may need to call it on the
   outgoing source before destroying it (today we just destruct and
   reconstruct).
3. **Validate the entropy=0 regression separately**. Likely a structural
   property of how often the async load runs vs how many items each
   pass actually processes -- may not be fixable without a different
   algorithmic shape for the degenerate-bucket case.

None of these are required for the by-ref refactor itself — that part is
landed on dev, SASS-identical, and all 20 tests pass on the default tuning.
The async-keys evaluation is just a forward-looking experiment that
surfaced these issues.

## Files

- Branch with the eval-only tuning change: `exp/topk-eval-async-keys`
  (`5f54e1edb1`). Single-commit delta on top of dev:
  `cub/cub/device/dispatch/tuning/tuning_batched_topk.cuh` line 302
  changed from `block_load_vectorize` to `block_load_to_shared_async`.
- Sweep JSON: `topk_perf_tracking/bench/sweep_async_pairs_i32i32.json`.
- Aggregate report (per K,V): see runtime numbers section above.
