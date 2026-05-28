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

## Why async is broken here — root cause + fix

Bisected the multi-CTA-all-large `cudaErrorLaunchFailure`. The behaviour
was deterministic and the binary hung (SIGTERM at the test's 120 s
timeout) on the `multi-CTA mixed` workloads when run in isolation; with
both `all-large` and `mixed` in the same process the `mixed` case bubbled
an error-719 up through the host-side `verify_unique_indices` thrust
inclusive_scan as `cudaErrorLaunchFailure`. compute-sanitizer memcheck
ran clean — no OOB / sync / race / init violations — so the failure is
*not* a memory error; it's the kernel itself getting stuck or aborting.

**Root cause:** `cub::detail::BlockLoadToShared::Invalidate()` (the
mbarrier reset) is *not* called from the loader's destructor by design.
Per its API docs:

```
// This is not the destructor to avoid overhead when shared memory reuse is not needed.
_CCCL_DEVICE_API _CCCL_FORCEINLINE void Invalidate() { ... mbarrier_inval ... }
```

The dtor is a no-op; the caller is required to call `Invalidate()`
explicitly if the loader's `TempStorage` will be reused. We weren't.

In the batched-topk agents, the keys source is reconstructed in two
shapes that *both* reuse the loader's `TempStorage`:

- **Per-tile, in the filter agent** (`agent_batched_topk_filter_partition`).
  Each `process_tile_{early_stop,buffered,unbuffered}` declares
  `key_src_input` / `key_src_buffer` / `keys_source` as locals bound to
  arm-specific smem slots (`storage.arms.{early_stop,buffered}.key_src_*_state`).
  At construction, `BlockLoadToShared`'s ctor runs `mbarrier_init`. When
  the function returns the locals' dtors are no-ops, leaving the
  mbarriers initialized. The next `process_tile_*` call re-runs
  `mbarrier_init` on already-initialized mbarriers.
- **Per-segment, in `agent_batched_topk_last_filter::run`**. The
  segment-boundary destroy-then-construct (added by my by-ref refactor)
  tears down `keys_source` / `key_src_input` / `key_src_buffer` in
  place, then placement-news new ones in the same smem. Same shape: old
  loader's dtor is no-op, new ctor re-runs `mbarrier_init`.
- **Per-stretch, in the histogram agent's slow path** (and the
  fully-unrolled fast path's `continue`). Same pattern: per-stretch
  ctor without an intervening `Invalidate()`.

Re-initializing an already-active mbarrier produces undefined behaviour
on the TMA path; with multi-CTA traversal of large segments it
manifested as a launch failure / hang. With `block_load_vectorize` (the
dev default), `BlockLoad` carries no persistent mbarrier state across
calls so this is dormant; flipping to `block_load_to_shared_async` (TMA)
exposes it.

**Fix** (commit `75772a6a0e`, on top of dev):

1. Add an `invalidate()` method to every `tile_data_source_t` variant:
   - `direct_data_source` and `sync_block_load_data_source`: no-op
     (these carry no persistent smem state across calls).
   - `async_to_shared_data_source`: was already there — delegates to
     `loader.Invalidate()` (mbarrier_inval + the two syncs the API
     requires).
   - `multi_source_data_source`: delegates to *both* children. Even the
     inactive arm's ctor ran `__init_mbarrier` and needs to be
     invalidated before its smem can be reused.

2. Call `keys_source.invalidate()` at every reconstruction site that
   reuses the loader's smem `TempStorage`:
   - End of each `process_tile_{early_stop,buffered,unbuffered}` in
     `agent_batched_topk_filter_partition` (before locals exit scope).
   - End of each chunk-iteration in `agent_batched_topk_histogram::run`
     (both fast-path before `continue` and slow-path while-loop tail).
   - Before the destroy-then-construct at the segment boundary in
     `agent_batched_topk_last_filter::run`.
   - The single-problem agents (`agent_topk_filter_partition`,
     `agent_topk_last_filter`) construct the keys source *once* per
     `invoke()` / `run()` and never reuse the loader's smem, so they
     need no change.

For non-async sources `invalidate()` is a no-op and compiles to nothing
at every site (verified: tests pass + benchmarks stable with the dev
tuning; see `pairs.dev.json` / `keys.dev.json`).

## Test results — after the fix

Built with `block_load_to_shared_async` everywhere on `umb-b200-236`:

```
19 of 19 test binaries pass, ~245k assertions across ~370 test cases:

Segmented (with async):
  cub.test.device.segmented_topk_keys.lid_{0,1,2}                ✓ 158172 asserts / 126 cases
  cub.test.device.segmented_topk_pairs.lid_{0,1,2}.types_{0,1,2} ✓  88,272 asserts / 144 cases
  (incl. the previously-failing `multi-CTA all-large` and `multi-CTA mixed`)
Single-problem (with async):
  cub.test.device.topk_api.lid_0                                 ✓     12 asserts /   5 cases
  cub.test.device.topk_keys.lid_{0,1,2}                          ✓   6806 asserts /  63 cases
  cub.test.device.topk_pairs.lid_{0,1,2}                         ✓   6624 asserts /  66 cases
```

## Benchmark results — after the fix

With correctness restored I re-ran the full I32-OffsetT/I32-OutOffsetT
sweep on `pairs.base` and `keys.base`. `dev/` here is the dev tip
(`exp/topk-batched-large-segments-regressions`, `block_load_vectorize`);
`async/` is the same dev tip with `block_load_to_shared_async` and the
invalidate fix.

```
Aggregate (async/dev runtime ratio, >1 means async slower):

| binary | rows | geo mean | worst (entropy=0.000) | best   |
|--------|------|----------|-----------------------|--------|
| pairs  | 816  | 2.223x   |  I8/I8/N=2^24/S=2^23   = 26.0x | I16/I64/N=2^28/S=2^23 = 1.02x |
| keys   | 357  | 2.048x   |  I8/N=2^24/S=2^23      = 26.5x | I16/N=2^28/S=2^23     = 1.04x |
```

Worst regressions cluster around `Entropy=0.000` (all-equal-keys
degenerate workload). At entropy ≥ 0.201 ratios drop to 1.02x – ~3x;
at entropy = 1.000 the ratios are 1.04x – ~1.5x. **No cell improved**;
the best entry on each binary is still ~1.04x slower than dev. This
matches the pre-fix benchmark shape (numbers in §"Benchmark results
(pairs.base, I32/I32, full sweep)" above) — the fix changes correctness,
not performance.

## Recommendation (unchanged)

The "just flip to async" default-tuning experiment is correct after this
fix but still ~2x slower in geomean. The structural issues remain:

1. **Use `<async, direct>` instead of `<async, async>`**. Only the input
   arm benefits from TMA (gmem read-only stream); the candidate buffer
   is chip-local so it has nothing to gain. The policy needs to support
   different load-kinds per arm.
2. **The `entropy=0.000` regression is structural**. With all keys
   equal, the radix-pass loop visits every input element on every pass
   (~`8 * sizeof(KeyT) / bits_per_pass` passes for the
   degenerate-bucket case). With async, each tile carries the TMA's
   per-tile setup/barrier-wait overhead; the rebuilt-per-tile shape
   amplifies that. A different algorithmic path for that degenerate
   case (e.g. an early "all-buckets-equal → return any k of them"
   detection) is likely the only way to claw the 6-26x gap back.
3. **Per-tile reconstruction of the keys source is wasteful**. Even on
   the dev (sync) path, every `process_tile_*` rebuilds the trio. With
   async this means an extra `__syncthreads()` pair per tile from the
   invalidate. Hoisting the trio outside `process_tile_*` (or even
   outside the run-loop, like the last_filter agent does) would
   eliminate the per-tile overhead entirely. That refactor wasn't done
   here because the smem footprint differs across arms (the
   `arms_t` union), but it's the natural next step.

The invalidate fix itself is cheap on the dev path (no-op lowering) and
required for any future async work, so it's worth landing independently
of the experiment.

## Files

- Branch with the eval tuning + the fix: `exp/topk-eval-async-keys` tip
  `75772a6a0e` ("topk(fix): invalidate mbarriers before per-tile /
  segment-boundary source reconstruction"). Two-commit delta on top of
  dev: the tuning flip (`5f54e1edb1`) and the invalidate fix
  (`75772a6a0e`).
- Sweep JSONs (post-fix, B200, `umb-b200-236`):
  `topk_perf_tracking/bench/{pairs,keys}.{dev,async}.json`.
- Old sweep JSON (pre-fix, kept for the historical numbers in
  §"Benchmark results (pairs.base, I32/I32, full sweep)" above):
  `topk_perf_tracking/bench/sweep_async_pairs_i32i32.json`.
