# BlockPartition / BlockFilter ScratchStorage refactor

**Branch:** `exp/topk-block-partition-scratch-refactor`
(stacked on `exp/topk-batched-large-segments-regressions` at `b47e7ca011`).

**Commits (top-of-branch first):**
1. `1c6dfef5ac` — replace explicit ctor/dtor on Scratch types with `Uninitialized<>`.
2. `a3711f59f9` — embed value-source ScratchStorage in `block_filter_atomics`.
3. `97c008fdc8` — refactor atomics scratch + move `partition_storage_layout` out of `block_partition.cuh`.

## What changed

### 1. Per-tile value-source scratch is now smem-resident in the atomics path

`block_partition_atomics::ScratchStorage` and `block_filter_atomics::ScratchStorage` used to be:

```cpp
struct ScratchStorage {};   // empty -- "atomics has no smem state"
```

with the per-tile value-load scratch materialised as a *stack-local* inside the scatter loop:

```cpp
typename ValueSourceT::ScratchStorage chan_scratch{};   // <-- on the per-thread stack
auto h = value_source.submit_load(chan_scratch);
```

That works for `direct_data_source` (whose `ScratchStorage` is an empty struct) but is structurally wrong for any data source that actually needs staging buffers (e.g. `sync_block_load_data_source` carrying `BlockLoad::TempStorage`, or `async_to_shared_data_source` carrying the TMA staging area). Those are smem-shaped types; they don't belong in registers/stack.

After the refactor, the atomics ScratchStorage embeds `ValueDataSourceScratchT` as an `Uninitialized<>`-wrapped member:

```cpp
private:
  struct _ScratchStorage { ValueDataSourceScratchT value_load; };
public:
  struct ScratchStorage : cub::Uninitialized<_ScratchStorage> {};
```

The scatter loop now reads `auto& chan_scratch = buffer.Alias().value_load;`. The agent already routes the BlockPartition's scratch through its own `__shared__ TempStorage` (via `partition_storage_layout::get_partition_scratch()`), so the value-source scratch transitively lives in shared memory. A `static_assert` enforces that the per-call `ValueSourceT::ScratchStorage` matches the class-level `ValueDataSourceScratchT` — agents must commit to the scratch type up front.

The same fix is applied to `block_filter_atomics`. The staged / shared-mem variants of both BlockPartition and BlockFilter already embedded `ValueDataSourceScratchT` correctly; only the atomics variants needed the structural fix.

### 2. `partition_storage_layout` moved out of `block_partition.cuh`

It is an agent-side smem-aliasing helper (it knows about prefix-sum scratch, keys-source scratch, and the partition primitive — concerns the partition itself shouldn't have to know about). Moved to `cub/detail/topk/partition_storage_layout.cuh`. `block_partition.cuh` no longer carries the type or the `is_empty` include for it. Both agents (`agent_topk.cuh`, `agent_batched_topk.cuh`) now `#include <cub/detail/topk/partition_storage_layout.cuh>` directly.

### 3. Explicit ctor / dtor on Scratch types replaced with `Uninitialized<>`

Five sites previously carried a hand-written `ctor() {}` / `~dtor() {}` pair to satisfy union-with-non-trivial-members rules:

| File | Type | Reason |
|---|---|---|
| `tile_data_source.cuh` | `multi_source_data_source::ScratchStorage` | union over `SourceA::ScratchStorage` / `SourceB::ScratchStorage` |
| `block_partition.cuh` | `block_partition_staged::ScratchStorage::phase_t` | union over `keys[]` / `value_phase_t` (and inner `value_phase_full` anonymous union) |
| `block_partition.cuh` | `block_partition_shared_mem::ScratchStorage::phase_t` | union over `delegate_loads` / `kv` |
| `block_filter.cuh` | `block_filter_staged::ScratchStorage::phase_t` | same shape as the partition sister |
| `block_filter.cuh` | `block_filter_shared_mem::ScratchStorage::phase_t` | same shape as the partition sister |

Plus `partition_storage_layout`'s own internal unions (in the moved file).

All replaced with `cub::Uninitialized<>` byte-storage wrappers per the existing CUB convention. Access sites were updated to go through `.Alias()`. The public ScratchStorage / TempStorage types are now free of explicit ctor / dtor declarations across the entire top-k partition / filter family.

## Resource diff (umb-b200-263, B200, sm_100, CTK 13.1)

`dev_baseline_n263_v2.json` (`b47e7ca011`) → `refactor_v2.json` (`1c6dfef5ac`), 253 records each:

| kernel | unchanged | improved | regressed | best Δ regs | worst Δ regs |
|---|---:|---:|---:|---:|---:|
| `initial_histogram` | 46 | 0 | 0 | 0 | 0 |
| `finalize_histogram` | 46 | 0 | 0 | 0 | 0 |
| `filter` | 46 | 0 | 0 | 0 | 0 |
| `finalize_filter` | 46 | 0 | 0 | 0 | 0 |
| `last_filter` | 46 | 0 | 0 | 0 | 0 |
| `single_cta` | 23 | 0 | 0 | 0 | 0 |

Aggregate Σ-deltas across all kernels (stack frame, spill stores, spill loads, smem): **all zero**.

That is the expected outcome on the **current value-source configuration** in both top-k agents:

```cpp
using value_source_t =
    detail::topk::multi_source_data_source<direct_data_source, direct_data_source, OffsetT>;
```

Both alternatives have an empty `ScratchStorage` (1 byte each), so the union scratch is also 1 byte. Whether that 1 byte materialises on the per-thread stack (old) or in the smem `partition_scratch` arena (new) is invisible to ptxas — same address-arithmetic patterns, same SASS.

The structural win lands at the moment a non-trivial value source is wired in (sync `BlockLoad`-based or async TMA staging buffer for the value channel). At that point the old code would have spilled real bytes to per-thread stack; the refactored code keeps them in smem where they belong. `static_assert` in `partition_atomics_fused` catches any future caller that picks a different per-call value source than the class-level `ValueDataSourceScratchT`.

## Runtime diff (B200, 2^28 elements)

`pairs.base` sweep: KeyT × ValueT ∈ {I32, I64}², sel ∈ {2^8, 2^13}, ent ∈ {1.000, 0.201, 0.000} — 24 records.

| | mean Δ% | median Δ% | min Δ% | max Δ% |
|---|---:|---:|---:|---:|
| pairs (24 records) | **-0.02%** | -0.01% | -0.20% | +0.02% |

All differences are within measurement noise. The refactor is runtime-neutral on the current configurations.

## Files touched

- `cub/cub/detail/topk/block_partition.cuh` — atomics scratch fix; staged + shared-mem `phase_t` wrapped in `Uninitialized<>`; `partition_storage_layout` removed.
- `cub/cub/detail/topk/block_filter.cuh` — sister-class fixes for both atomics and staged + shared-mem.
- `cub/cub/detail/topk/tile_data_source.cuh` — `multi_source_data_source::ScratchStorage` wrapped in `Uninitialized<>`; `submit_load` uses `s.Alias().{a,b}`.
- `cub/cub/detail/topk/partition_storage_layout.cuh` — new file; same logic as the old in-`block_partition.cuh` definition with the inner unions wrapped in `Uninitialized<>`.
- `cub/cub/agent/agent_topk.cuh` — added include of `partition_storage_layout.cuh`.
- `cub/cub/agent/agent_batched_topk.cuh` — added include of `partition_storage_layout.cuh`.

No tests changed (`cub/test/catch2_test_device_topk_*.cu`); the public ScratchStorage / TempStorage shapes are unchanged from a caller's perspective (same nested types, same `sizeof`, same alignment), only the inner construction story shifted from "explicit ctor/dtor on inner unions" to "byte-storage wrapper at the outer boundary".

## Status

- ✅ Compiles clean (`cub.bench.topk.pairs.base`, `cub.bench.topk.keys.base`).
- ✅ 253-record resource snapshot byte-identical to dev baseline.
- ✅ Runtime sweep within ±0.20% of dev baseline (mean -0.02%).
- ✅ Changes left on the side branch `exp/topk-block-partition-scratch-refactor` for review (pushed to `origin`).

To review the diffs locally:

```bash
git checkout exp/topk-block-partition-scratch-refactor
git diff exp/topk-batched-large-segments-regressions...HEAD -- cub/cub/detail/topk cub/cub/agent
```

Or per-commit:

```bash
git log --oneline exp/topk-batched-large-segments-regressions..exp/topk-block-partition-scratch-refactor
git show <sha>
```

Note: the structural payoff lands when a real value-source `ScratchStorage` (sync `BlockLoad`-based, async TMA, etc.) is configured for the value channel. Today's `multi_source<direct, direct>` choice means the refactor is runtime-neutral on the benchmark — but the architecture is now ready to absorb a non-trivial value source without spilling its scratch to per-thread stack.
