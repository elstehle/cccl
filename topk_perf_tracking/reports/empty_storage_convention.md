# Empty-storage convention for the top-k subtree

**Branch:** `exp/topk-block-partition-scratch-refactor` (now extends the prior `block_partition_scratch_refactor` work).
**Commits (top of branch first):**
1. `fcadc5437e` — keep `multi_source_data_source::TempStorage` as a struct (agent accesses `.a` / `.b` directly).
2. `43b7f733a5` — introduce `empty_storage_t` + `is_empty_storage_v<T>` with transitive propagation.

## What was added

### 1. The marker + trait (new file: `cub/detail/topk/empty_storage.cuh`)

```cpp
struct empty_storage_t {
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE empty_storage_t& Alias() { return *this; }
};

template <typename T>
inline constexpr bool is_empty_storage_v =
     ::cuda::std::is_same_v<T, empty_storage_t>
  || ::cuda::std::is_empty_v<T>;
```

The marker is the canonical signal for "this `TempStorage` / `ScratchStorage`
carries no smem state". The trait is **permissive**: it returns `true` for
the marker *and* for legacy `struct {}` declarations (so `BlockLoad`, `BlockScan`,
etc. don't need to migrate).

The marker carries a no-op `Alias()` so consumer code that uniformly does
`buffer.Alias().<member>` keeps compiling whether `buffer` is wrapped in
`Uninitialized<inner>` (non-empty case, where `Alias()` returns the inner
union) or is `empty_storage_t` itself (empty case). In the empty case the
consumer is expected to gate the access via
`if constexpr (!is_empty_storage_v<...>)` and never actually read members
through the empty `Alias()`.

The trait/marker live in `cub/detail/topk/empty_storage.cuh` rather than
`agent_topk_common.cuh` because the latter `#include`s `block_partition.cuh`
and `tile_data_source.cuh` — the primitives need to see the trait at type-
definition time (for transitive propagation), so it has to live below them.
`agent_topk_common.cuh` does include `empty_storage.cuh`, so users see the
symbols via that path too.

### 2. Migrated TempStorage / ScratchStorage publishing

| Type | Before | After | Notes |
|---|---|---|---|
| `direct_data_source::{Temp,Scratch}Storage` | `struct {}` | `empty_storage_t` | Already empty; now uses the canonical marker. |
| `sync_block_load_data_source::TempStorage` | `struct {}` | `empty_storage_t` | Per-tile `BlockLoad::TempStorage` lives in `ScratchStorage`. |
| `multi_source_data_source::TempStorage` | `struct {a; b;}` | `struct {a; b;}` (unchanged) | Agent accesses `.a` / `.b` directly; transitive empties at this level deferred. |
| `multi_source_data_source::ScratchStorage` | `Uninitialized<{a, b}>` | `conditional_t<both empty, empty_storage_t, Uninitialized<{a, b}>>` | **Transitive propagation**: when both children are empty, the aggregate is too. |
| `block_partition_atomics::ScratchStorage` | `Uninitialized<{value_load}>` | `conditional_t<value_load empty, empty_storage_t, Uninitialized<{value_load}>>` | Transitive: empty when the value channel is empty (typical `multi_source<direct, direct>` case). |
| `block_filter_atomics::ScratchStorage` | (same as above) | (same as above) | Same propagation, sister primitive. |
| `block_partition_staged::TempStorage` | `struct {}` | `empty_storage_t` | |
| `block_partition_staged::value_phase_empty` | `struct {}` | `empty_storage_t` (alias) | Replaces the dedicated empty struct. |
| `block_partition_shared_mem::TempStorage` | `struct {}` | `empty_storage_t` | |
| `block_partition_shared_mem::delegate_load_empty` | `struct {}` | `empty_storage_t` (alias) | |
| `block_filter_staged::{TempStorage, value_phase_empty}` | (same as block_partition_staged) | (same) | |
| `block_filter_shared_mem::{TempStorage, delegate_load_empty}` | (same as block_partition_shared_mem) | (same) | |

### 3. Internal use sites updated

`multi_source_data_source::submit_load` and the `block_{partition,filter}_atomics::*_atomics_fused` paths gained `if constexpr (_scratch_storage_is_empty)` branches that pass on-stack stub instances when the aggregate `ScratchStorage` is the empty marker (and there is therefore no `.Alias().value_load` to read). The compiler folds these stubs away — see SASS-identity check below.

## Resource impact

Compared to the immediately-preceding dev baseline (`callback_gated`, `b47e7ca011`):

| metric | callback_gated | empty_storage | delta |
|---|---|---|---|
| registers | (per (KeyT, ValueT)) | identical | **0** |
| stack | identical | identical | **0** |
| spill stores | identical | identical | **0** |
| spill loads | identical | identical | **0** |
| smem | identical | identical | **0** |

Across all 23 `(KeyT, ValueT)` combinations covered by the
`pairs.base + keys.base` benchmarks, every per-kernel resource value
(filter, last-filter / finalize_filter, initial_histogram, finalize_histogram,
single_cta) is bit-identical to `callback_gated`.

## SASS impact

```
$ cuobjdump --dump-sass build_cb_gated/bin/cub.bench.topk.pairs.base | md5sum
08746d77d5da2fa52cc1a6747a5a5494
$ cuobjdump --dump-sass build_empty_storage/bin/cub.bench.topk.pairs.base | md5sum
08746d77d5da2fa52cc1a6747a5a5494
$ cuobjdump --dump-sass build_cb_gated/bin/cub.bench.topk.keys.base | md5sum
f13151ddc6a0eb6114318366f2bb3831
$ cuobjdump --dump-sass build_empty_storage/bin/cub.bench.topk.keys.base | md5sum
f13151ddc6a0eb6114318366f2bb3831
```

**Byte-identical SASS** for both `pairs.base` and `keys.base` against the
immediately-preceding dev baseline.

## Runtime impact

By construction (byte-identical SASS): **zero**. Skipping a perf sweep
because every cycle of work in every kernel is unchanged.

## What this enables (not yet wired)

This change is purely *the convention* — it does not yet add any `__syncthreads()`-bypass
guards on top of the convention. The intended pattern is:

```cpp
// Inside e.g. block_partition_staged::partition_impl, between the keys-source
// load and the partition body's first touch of `partition_scratch`.
if constexpr (!is_empty_storage_v<typename KeysSourceT::ScratchStorage>) {
  __syncthreads();
}
```

Concrete sites where this is *now* legal because of the propagation:

- `multi_source_data_source<direct, direct>::ScratchStorage` is `empty_storage_t` (was `Uninitialized<1B>`),
  so a consumer that holds a `multi_source` value source can detect the empty case across the class boundary.
- `block_{partition,filter}_atomics::ScratchStorage` is `empty_storage_t` whenever its
  `ValueDataSourceScratchT` is empty (the typical config), so the agent's outer arena
  layout can detect that and skip syncs that only existed to protect that storage.
- `block_{partition,filter}_staged::value_phase_empty` and the shared-mem variants'
  `delegate_load_empty` slots now publish as `empty_storage_t`, so the phase unions
  can detect "the value-phase view holds nothing" without re-checking `keys_only`.

Wiring those bypasses is a follow-up — best done at points where SASS shows a
redundant `BAR.SYNC` whose only purpose is to fence empty smem.

## Files touched

- `cub/cub/detail/topk/empty_storage.cuh` (**new**, 86 lines)
- `cub/cub/detail/topk/tile_data_source.cuh` (`direct_data_source`, `sync_block_load_data_source`, `multi_source_data_source::ScratchStorage` propagation)
- `cub/cub/detail/topk/block_partition.cuh` (atomics ScratchStorage propagation, staged + shared_mem TempStorage / empty inner aliases)
- `cub/cub/detail/topk/block_filter.cuh` (sister changes mirroring block_partition)
- `cub/cub/detail/topk/partition_storage_layout.cuh` (just the include — no behaviour change)
- `cub/cub/agent/agent_topk_common.cuh` (just the include — no behaviour change)

No agent-side changes (`agent_topk.cuh`, `agent_batched_topk.cuh`).
