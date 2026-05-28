# `last_filter` register-pressure analysis: where the +32 regs on I8 KeyT come from

## Headline

| KeyT | items / thread | dev regs | main regs | delta | dev stack/spill | main stack/spill |
|---|---:|---:|---:|---:|---|---|
| I8 | 16 | 64 | 32 | **+32** | 0 / 0 / 0 | 8 / 12-24 / 8-16 (on pairs) |
| I16 | 8 | 54-56 | 32 | **+22-24** | 0 / 0 / 0 | 0 / 0 / 0 |
| I32 | 4 | 40 | 32 | +8 | 0 / 0 / 0 | 0 / 0 / 0 |
| I64 | 2 | 40 | 32 | +8 | 0 / 0 / 0 | 0 / 0 / 0 |

Dev keeps everything in registers; main spills aggressively on I8 KeyT pairs (8 B stack, 12-24 B spill stores). The I8 / I16 register inflation in dev comes from cross-tile state retained across the grid-stride loop, which prevents `ptxas` from sharing those registers with the inner `partition.partition()` peak.

## The cross-tile state held by dev's `agent_batched_topk_last_filter::run()`

After `581b735ec0`, the run loop is shaped like:

```cpp
per_segment_state_t state = resolve_segment_state(...);     // cross-tile
if (state.empty && cta_doesnt_cross_segment) return;        // fast-empty-exit
partition_t partition = make_partition_for_segment(state);  // cross-tile
keys_source_t keys_source = make_keys_source_for_segment(state);  // cross-tile
for (tile_id ...) {
  if (tile_id >= state.queue_segment_end) { rebuild on segment boundary }
  process_tile<Full>(state, partition, keys_source, local_tile);
}
partition.epilogue();
```

Every variable above lives across all tiles of the segment. For our single-segment benchmark, that's **the full grid-stride loop**.

### `per_segment_state_t` (cross-tile)

```cpp
struct per_segment_state_t {
  bool empty;                                  // 1 reg
  bool load_from_candidates_buffer;            // packed
  inner_key_it_t d_keys_in;                    // ~2 regs
  inner_key_out_it_t d_keys_out;               // ~2 regs
  inner_value_it_t d_values_in;                // ~2 regs (0 keys-only)
  inner_value_out_it_t d_values_out;           // ~2 regs (0 keys-only)
  counter_t* segment_counter;                  // 2 regs
  key_in_t* in_key_buf;                        // 2 regs
  value_in_t* in_val_buf;                      // 2 regs (0 keys-only)
  int pass;                                    // 1 reg
  OutOffsetT k_total;                          // 1 reg (32-bit OutOffsetT)
  OutOffsetT num_of_kth_needed;                // 1 reg
  OffsetT input_length;                        // 1 reg (32-bit OffsetT)
  OffsetT num_full_tiles;                      // 1 reg
  OffsetT partial_items;                       // 1 reg
  OffsetT segment_tiles_input;                 // 1 reg (UNUSED in last_filter loop)
  LargeSegmentTileOffsetT slab_base;           // 2 regs (long)
  LargeSegmentTileOffsetT queue_segment_end;   // 2 regs (long)
};  // ~25 regs total when nothing is folded
```

**But after `partition` and `keys_source` are built**, the loop body only reads:

- `state.queue_segment_end` (segment-crossing check)
- `state.num_full_tiles`, `state.partial_items`, `state.slab_base` (tile dispatch)
- `state.empty` (loop body branch -- redundant with fast-empty-exit but compiler may not prove it)
- `state.d_values_in`, `state.in_val_buf`, `state.load_from_candidates_buffer` (inside `process_tile` for value_source construction -- not lifted)

That's 5-7 regs of *actual* cross-tile use; the other ~18 regs of `per_segment_state` should be dead-code-eliminated **but**:

- The compiler keeps every field alive on the segment-refresh branch (`tile_id >= state.queue_segment_end -> reconstruct state`). Refresh re-reads d_keys_in, segment_counter, k_total, etc.
- For single-segment workloads the refresh branch is provably dead (queue_segment_end == total), but ptxas can't see this -- it's a runtime relation.

### `partition_t partition` (cross-tile, ~13 regs)

```cpp
SelectedReserveOp reserve_sel;       // ptr, ~2 regs
CandidateReserveOp reserve_cand;     // ptr + 2 ints, ~4 regs
SelectedKeyOutIt sel_iter;           // ~2 regs
CandidateKeyOutIt cand_iter;         // ~2 regs (folded with sel_iter for last_filter)
ValueChannelSinksT sinks;            // 2 iters for pairs, 0 for keys-only
IdentifyCandidatesOp identify_op;    // 1-2 regs (kth_key_value)
CandidateCallbackOp callback_op;     // empty
bool cand_reserve_open;              // 1 reg
```

All except `cand_reserve_open` are **loop-invariant per-segment data**. They're live across every tile but never read inside the loop except passed by reference into `partition.partition(...)`.

### `keys_source_t keys_source` (cross-tile, ~5 regs)

`(input_iter, buffer_iter, pick_b_bool)` -- pointers + bool.

### Total cross-tile holding state

About **25-30 regs** that ptxas cannot share with the peak inside `partition.partition()`.

## Why narrow KeyT amplifies the gap

`items_per_thread` is `max(1, 16 / sizeof(KeyT))`:

| KeyT | items/thread | items[] size in regs | inner peak liveness during `partition.partition()` |
|---|---:|---:|---|
| I8 | 16 | 4 | wide -- 16 per-item classifications, atomic-add pos values, ~8-12 regs of inner state |
| I16 | 8 | 4 | ~5-7 regs of inner state |
| I32 | 4 | 4 | ~3 regs |
| I64 | 2 | 4 | ~2 regs |

Because cross-tile state is *alive* during the `partition.partition()` call (it's bound to references / used after), ptxas allocates registers as `cross_tile_state + max(per_iteration_peak)`. For wide-`items_per_thread` types (I8 / I16) the per-iteration peak is large, and the sum hits 54-64 regs.

`main` builds the partition object once at the top of `run()` (same pattern), but its agent has **far fewer cross-tile fields** (no `per_segment_state`, no segmented-input iterators). All ~25 regs of dev's per-segment-state are gone. ptxas can therefore share those slots with the inner peak: 32 regs total.

## Concrete improvement opportunities

Listed by expected impact and invasiveness.

### 1. Lift `value_source` out of `process_tile` (low-risk, ~2-4 regs win)

Currently `value_source` is rebuilt per tile inside `process_tile`, capturing `state.d_values_in`, `state.in_val_buf`, `state.load_from_candidates_buffer` each time. Lifting it (mirror `keys_source`) would let those 3 fields of `state` be DCE'd from the cross-tile holding set.

Affects: `agent_batched_topk_last_filter::run` and `agent_batched_topk_filter_partition::run`.
Risk: low (same lifetime trick we already applied to `keys_source`).

### 2. Trim `per_segment_state_t` to a "tile-loop bookkeeping" struct (low-risk, ~5-10 regs win)

After `make_partition_for_segment` / `make_keys_source_for_segment` / `make_value_source_for_segment` complete, only the loop bookkeeping fields are needed cross-tile. Hoist them into local scalars and let `state` go out of scope:

```cpp
auto state = resolve_segment_state(...);
auto partition    = make_partition_for_segment(state);
auto keys_source  = make_keys_source_for_segment(state);
auto value_source = make_value_source_for_segment(state);

// Extract the only fields the cross-tile loop actually reads.
const LargeSegmentTileOffsetT segment_end = state.queue_segment_end;
const LargeSegmentTileOffsetT slab_base   = state.slab_base;
const OffsetT num_full_tiles              = state.num_full_tiles;
const OffsetT partial_items               = state.partial_items;
// `state` is dead now; ptxas can DCE its fields.
```

The segment-refresh branch needs to rebuild these. Wrap the rebuild in a `__noinline__` helper function so the inliner doesn't pull `per_segment_state_t` back into the hot loop's live range:

```cpp
__attribute__((noinline)) void refresh_segment(LargeSegmentTileOffsetT tile_id, ...);
```

Risk: low. The `__noinline__` helper isolates segment-refresh state from the hot loop.

### 3. Move loop-invariant `partition_t` fields to a per-segment shared-memory descriptor (medium-risk, ~10-15 regs win, biggest potential)

Of `partition_t`'s ~13-15 regs of cross-tile state, **only `cand_reserve_open` actually changes** between tiles. The rest are uniform across the warp and constant across the segment. We're paying `block_threads * sizeof(those_fields)` of shared register file for what is effectively per-block constant data.

Two ways to get them out of registers:

  - **3a. Stash them in a smem slot.** At segment-refresh, write `(reserve_sel, reserve_cand, sel_iter, cand_iter, sinks, identify_op, callback_op)` into a single `_SegmentDescriptor` smem struct. The partition primitive's `partition()` method reads from there each call. Saves ~13 regs / thread, costs ~13 * 8 = 104 B of smem.

  - **3b. Pass them as parameters to `partition()`.** The partition object becomes a thin holder of just `cand_reserve_open`. Each `partition.partition(buffer, keys, value_source, reserve_sel, reserve_cand, sinks, identify_op, callback_op)` call passes them by value -- ptxas can keep them in scalar/uniform registers shared across the warp.

I would prototype 3b first -- it doesn't change smem layout and is closer to the existing API. If 3b doesn't recover enough, 3a moves the fields physically out of regs.

Risk: medium. Touches the partition primitive API; need to update the buffered / accumulating / speculative variants in lock-step.

### 4. Drop `cand_reserve_open` from `partition_t`, pass it as in/out arg (low-risk, enables 3b cleanly)

```cpp
bool cand_reserve_open = true;  // owned by agent's `run()`
for (...) {
  // partition_t now stateless; constructed cheaply per call.
  partition_t partition{...};
  partition.partition(scratch, items, value_source, cand_reserve_open);
  // ^ reads + writes cand_reserve_open
}
```

Once `cand_reserve_open` is the *only* cross-tile state and it's a scalar, the agent has full freedom to put `partition_t` back inside `process_tile`. ptxas can then share registers between `partition`'s short-lived state and the inner peak -- exactly the layout main enjoys.

Risk: low API change (one new method param). Saves ~13 regs / thread on the cross-tile holding set, with no behaviour change.

### 5. Specialise `agent_batched_topk_last_filter` for the common `num_large_segments == 1` case (medium-risk, ~2-5 regs win)

When the dispatch knows there's only one segment, the segment-refresh branch is provably dead. Templating the agent on a `static_eq_one : bool` parameter (false for the general case, true when the dispatch can prove single-segment) lets `if constexpr` collapse:

```cpp
if constexpr (!StaticOneSegment) {
  if (tile_id >= state.queue_segment_end) {
    refresh_segment(...);
  }
}
```

`per_segment_state_t` then no longer needs to be kept alive after construction in the StaticOneSegment instantiation, and the refresh-side reads of agent member iterators (d_key_segments_it etc.) are DCE'd.

Risk: medium. Doubles the number of last_filter / filter kernel instantiations (one for single-segment, one for multi-segment). For the single-problem benchmark path this is a net positive; for genuine multi-segment workloads no change.

### 6. Lower `items_per_thread` on narrow KeyT in the multi-worker policy (low-risk, but tunes both `main` and dev)

The current policy gives I8 KeyT 16 items/thread (chasing 16 B/thread load). Lowering to 8 items/thread halves the inner per-iteration peak liveness, which directly attacks the I8 register cliff. Costs grid-occupancy / waves-per-tile.

This is a **tuning** experiment, not an agent rewrite. Should be measured against the entire I8 sweep, not just the resource numbers.

## Recommendation

Land in this order:

1. **#4 (drop `cand_reserve_open` from `partition_t`, pass as in/out arg)**. Cleanest win, narrow API change. Should drop dev's I8 last_filter from 64 -> ~50 regs and I16 from 56 -> ~45.
2. **#1 (lift `value_source`) + #2 (trim `per_segment_state` cross-tile lifetime)**. Together another 5-10 regs.
3. Consider **#3 (move loop-invariant partition fields out of regs)** if the gap to main is still material after 1+2+4.
4. **#6** as a tuning sweep, separately, to evaluate items_per_thread tradeoffs across the I8 / I16 workloads.

After **(1+2+4)** I'd expect dev's last_filter to land at:

| KeyT | dev now | dev projected | main |
|---|---:|---:|---:|
| I8 | 64 | ~40-44 | 32 |
| I16 | 56 | ~38-42 | 32 |
| I32 | 40 | ~32-34 | 32 |
| I64 | 40 | ~32-34 | 32 |

i.e. close to parity on I32/I64 and within +6-12 of main on I8/I16, while preserving the `cand_reserve_open` cross-tile optimisation.
