# `__syncthreads()` elision via empty-storage trait

**Branch:** `exp/topk-block-partition-scratch-refactor`.
**Top-of-branch commits:**
1. `89c79e7658` — document why buffered-tile sync stays unconditional.
2. `1f8a804b3d` — revert buffered-tile sync elision (sel=524288 regression bisect).
3. `c4f8ca9077` — qualify `is_empty_storage_v` with `detail::topk::`.
4. `0fbb826840` — elide superfluous `__syncthreads()` in batched topk via empty_storage trait.

## What changed

### 1. Agent-level post-`complete_load` syncs in `agent_batched_topk.cuh`

The pattern in the `filter` / `last_filter` agents is

```cpp
if constexpr (tile_load_kind_uses_smem) { __syncthreads(); }   // pre-load
key_in_t items[items_per_thread];
auto h = keys_source.submit_load(arena.get_keys_source_scratch());
h.complete_load(items);
__syncthreads();                                                 // post-load
partition.partition(arena.get_partition_scratch(), items, value_source);
```

The pre-load sync was already gated on `tile_load_kind_uses_smem`. The post-load sync was unconditional, kept "in either case" because `keys_source_scratch` and `partition_scratch` alias through the smem union in `partition_storage_layout_for_t` and the previous tile's partition could have written to the partition slot.

The trait `is_empty_storage_v<typename PartitionT::ScratchStorage>` lets us detect the case where the previous partition didn't touch smem (typical `multi_source<direct, direct>` value source -> `block_{partition,filter}_atomics::ScratchStorage` collapses to `empty_storage_t`). Combined with `tile_load_kind_uses_smem`, the post-load sync becomes:

```cpp
if constexpr (tile_load_kind_uses_smem
              || !is_empty_storage_v<typename PartitionT::ScratchStorage>)
{
  __syncthreads();
}
```

Applied to **4 of 6** post-load sites in the batched filter / last-filter agents:
- `process_tile_early_stop` -- full-tile + partial-tile arms (2 sites). Gated on `early_stop_filter_t::ScratchStorage`.
- `process_tile_buffered`   -- full-tile + partial-tile arms (2 sites). **Reverted to unconditional sync** -- see "What got reverted" below.
- `agent_batched_topk_last_filter::process_tile` -- full-tile + partial-tile arms (2 sites). Gated on `partition_t::ScratchStorage`.

### 2. Internal eager-value-load fences in the staged + shared_mem partition / filter strategies

`block_partition_staged` / `block_filter_staged` -- the eager `submit_load(vphase.load)` writes to the value-source scratch via the union with `vphase.values`; the alias swap in the scatter loop needs a fence only if the scratch is non-empty:

```cpp
if constexpr (!is_empty_storage_v<ValueDataSourceScratchT>) { __syncthreads(); }
```

`block_partition_shared_mem` / `block_filter_shared_mem` -- the pre-Phase-1 eager pre-load to `delegate_loads.load` aliases with the kv arena; the transition needs a fence only if the eager pre-load actually wrote smem:

```cpp
if constexpr (!keys_only && !LazyValueLoad && !is_empty_storage_v<ValueDataSourceScratchT>)
{
  __syncthreads();
}
```

These strategies are **not** the policy choice for the current builds (`atomics` is selected for both partition + filter), so the gating is structural -- it shows up only when staged / shared_mem are dialed in.

## What got reverted (and why)

The buffered-tile post-load sync elision **regressed** the `K=I32, V=I32, Elements=2^28, SelectedElements=2^19, Entropy>=0.201` workload by ~+25%:

| label                         | sel=524288, ent=0.201 | sel=524288, ent=0.544 | sel=524288, ent=1.000 |
|---|---|---|---|
| `cb_gated` (baseline)         | 832 us                | 884 us                | 807 us                |
| full elision (buffered + early-stop + last_filter) | 1045 us (1.246x)      | 1080 us (1.224x)      | 1000 us (1.244x)      |
| **reverted buffered, kept early-stop + last_filter** | **832 us (1.000x)**   | **884 us (1.000x)**   | **807 us (1.000x)**   |

The bisect (`1f8a804b3d`) confirmed the regression is fully eliminated by re-adding the buffered-tile sync. Hypothesis: the buffered partition fires `histogram_callback_op::operator()(KeyT)` on every candidate-classified item (one `atomicAdd` to the smem histogram per fire). Without the post-load barrier, the per-tile bursts of those smem atomicAdds overlap with the next tile's bursts, increasing smem-atomic contention. The early-stop and last-filter arms don't fire that callback (early-stop has no histogram; last-filter installs `topk_noop_candidate_callback_op`), so the elision is a clean win there.

## Resource impact

Snapshot diff (`empty_storage` -> `elide_syncs_v2`), 138 `(logical_kernel, KeyT, ValueT)` triples covered:

| metric        | delta histogram                  |
|---|---|
| registers     | 0 for all 138 triples            |
| stack frame   | 0 across the board               |
| spill stores  | 0 across the board               |
| spill loads   | 0 across the board               |
| smem (B)      | 0 across the board               |

## SASS impact

```
                          BAR.SYNC count          binary md5
                          ---------------         -------------
  pairs.base
    cb_gated      :       2324                    08746d77d5da2fa52cc1a6747a5a5494
    empty_storage :       2324                    08746d77d5da2fa52cc1a6747a5a5494
    elide_syncs_v2:       2196  (-128, -5.5%)     a61af970e6c69a5c46e323b1fc929cb9

  keys.base
    cb_gated      :       1299                    f13151ddc6a0eb6114318366f2bb3831
    empty_storage :       1299                    f13151ddc6a0eb6114318366f2bb3831
    elide_syncs_v2:       1243  (-56, -4.3%)      0a1c60ede9b3254e1f716ff2e760c775
```

Per-kernel breakdown (counts are per kernel instance; multiply by # of type instantiations for the totals above):

| kernel              | BAR.SYNC before -> after |
|---|---|
| `device_segmented_topk_filter_kernel`           | 8 -> 6  (-2, early-stop arm only) |
| `device_segmented_topk_finalize_filter_kernel`  | 15 -> 13 (-2, last-filter agent)  |
| `device_segmented_topk_kernel` (single CTA)     | 22 -> 22 (untouched)              |
| `device_segmented_topk_histogram_kernel`        | unchanged                         |

## Runtime impact

Full `K=V=I32, OffsetT=OutOffsetT=I32` sweep (Elements x SelectedElements x Entropy, ~96 cells):

```
mean delta:         ~ 0.997x  (i.e. 0.3% mean improvement)
worst entry:        1.005x    (sel=8388608, ent=0.201, ~0.5% slower -- noise)
best entry:         0.982x    (sel=8, ent=0.201, ~1.8% faster)
no entry:           > 1.005x  (regression)
```

Most cells are 1.000x within measurement noise; the small wins concentrate on workloads where the early-stop or last-filter arms dominate (small SelectedElements).

## Files touched

- `cub/cub/agent/agent_batched_topk.cuh` -- 4 post-load sync sites gated; buffered-arm sites kept unconditional with explanatory comment.
- `cub/cub/detail/topk/block_partition.cuh` -- staged + shared_mem internal eager-load fences gated.
- `cub/cub/detail/topk/block_filter.cuh` -- sister gating.

No other agents (single-problem `agent_topk.cuh`, accumulating / speculative variants) changed by this iteration -- they're a separate follow-up.

## Conclusion

A net-positive structural change: -184 `BAR.SYNC` instructions across the two benchmark binaries (-5.0% combined), zero resource impact, runtime is neutral-to-slightly-positive across the full sweep, no regressions. The buffered-arm exception is documented in-source so future contributors don't reintroduce it.
