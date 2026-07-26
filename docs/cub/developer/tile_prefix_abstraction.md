(cub-tile-prefix-abstraction)=

# Tile-prefix abstraction

:::{note}
This is a design proposal (RFC), not a description of existing code.
Nothing described under {ref}`cub-tile-prefix-proposed-api` exists today.
The analysis sections that precede it do describe the code as it is at the time of writing.
:::

## Motivation

CUB currently contains two unrelated implementations of the same idea: computing, for a tile of a
device-wide scan-like algorithm, the aggregate of all preceding tiles.

- **Decoupled look-back**, in `cub/agent/single_pass_scan_operators.cuh`. It is used by eight CUB
  agents, which run nine independent look-back chains between them because `agent_batch_memcpy`
  needs two, and by two Thrust CUDA-backend algorithms. It runs on every architecture CUB supports.
- **Look-ahead**, in `cub/detail/warpspeed/look_ahead.cuh` and
  `cub/device/dispatch/kernels/kernel_scan_lookahead.cuh`. It is used by exactly one algorithm
  ({cpp:struct}`cub::DeviceScan`) and requires SM90 or newer.

The two are not two implementations of one algorithm. They have different global dependency
structures, different tile-state encodings, different execution models, and different hardware
requirements. What they share is the value contract they present to the algorithm above them.

The goal of this document is to determine whether a single abstraction can back both, so that an
algorithm author writes one implementation and the choice of prefix mechanism becomes a tuning
decision. The concrete payoff would be extending look-ahead's performance characteristics to the
seven agents that have no look-ahead path today.

This document is organized as follows. {ref}`cub-tile-prefix-lookback` and
{ref}`cub-tile-prefix-lookahead` describe how each mechanism works and what is on its critical path.
{ref}`cub-tile-prefix-comparison` extracts the commonalities and the differences that constrain any
shared abstraction. {ref}`cub-tile-prefix-inventory` catalogues what each current consumer actually
needs. {ref}`cub-tile-prefix-proposed-api` proposes a three-layer API, and
{ref}`cub-tile-prefix-groups` addresses the group/squad question it raises. Finally,
{ref}`cub-tile-prefix-risks` covers determinism, cost, risks, open questions, how an implementation
would be validated, and a suggested rollout order.

(cub-tile-prefix-lookback)=

## How decoupled look-back works

### Tile state

`cub::ScanTileState<T>` is an array of per-tile descriptors in global memory, carved out of
`d_temp_storage`. It is sized by `detail::num_tiles_to_num_tile_states`, which adds
`detail::warp_threads` (32) leading entries so that the index `tile_idx - lane - 1` used during
look-back is always in bounds. Those 32 padding entries are permanently marked `SCAN_TILE_OOB`,
which acts as the identity terminator for the search.

A tile is in one of four states:

| status | meaning |
| --- | --- |
| `SCAN_TILE_OOB` | padding before tile 0; terminates a look-back window with no contribution |
| `SCAN_TILE_INVALID` | the owning block has not published anything yet |
| `SCAN_TILE_PARTIAL` | the tile's own aggregate is available, but not the prefix of everything before it |
| `SCAN_TILE_INCLUSIVE` | the aggregate of all tiles up to and including this one is available |

There are two storage specializations. When `{status, value}` fits into a single word that the
hardware can read and write atomically, `ScanTileState<T, true>` packs them into one `TxnWord`
(`unsigned int`, `uint2`, or `ulonglong2`) and every publish is a single relaxed or release
store. Otherwise `ScanTileState<T, false>` keeps three parallel arrays: a status array, a
*partial* payload array, and a separate *inclusive* payload array. The payloads are kept apart
deliberately, so that a block overwriting its own state from partial to inclusive cannot tear a
payload that another block is concurrently reading.

### The prefix callback

`cub::TilePrefixCallbackOp` is a functor, constructed per tile, that fuses three operations into
one blocking call. Its core is `TilePrefixCallbackOp::lookback`:

1. Thread 0 publishes this tile's own aggregate with `SetPartial(tile_idx, block_aggregate)`.
   This happens *before* the search, so that successor tiles can make progress through this tile
   even while it is still resolving its own prefix. This is the "decoupled" part.
2. The 32 lanes of warp 0 each inspect one predecessor, `predecessor_idx = tile_idx - lane - 1`,
   and spin in `ScanTileState::WaitForValid` until *no* lane sees `SCAN_TILE_INVALID`. Note the
   spin condition is `__any_sync`: the whole warp waits for the whole window.
3. The window is collapsed with `WarpReduce::TailSegmentedReduce`, using `SwizzleScanOp` because
   the reduction runs *down* towards lane 0, i.e. in the opposite direction from the scan, and the
   operator is not assumed to be commutative. The segment tail flag is
   `predecessor_status == SCAN_TILE_INCLUSIVE`, so the reduction stops at the first predecessor
   that already knows its full prefix.
4. If no lane in the window saw `SCAN_TILE_INCLUSIVE`, the window slides back another 32 tiles
   and the accumulated result is folded in with
   `exclusive_prefix = scan_op(window_aggregate, exclusive_prefix)`. The loop condition is
   `__all_sync`, so a single lane finding an inclusive predecessor ends the search for the whole
   warp.
5. Thread 0 computes `inclusive_prefix = scan_op(exclusive_prefix, block_aggregate)` and publishes
   it with `SetInclusive(tile_idx, inclusive_prefix)`. This is what bounds the search for
   successor tiles.

The exclusive prefix is returned from lane 0. The aggregate, exclusive prefix, and inclusive prefix
are also stashed in the callback's shared-memory `TempStorage` so that other threads can read them
afterwards through `GetBlockAggregate` / `GetExclusivePrefix` / `GetInclusivePrefix`.

### Where it runs

The callback is invoked from the middle of a block scan. `BlockScanWarpScans::ExclusiveScan`
computes the block aggregate, hands it to warp 0's callback, broadcasts the returned prefix through
shared memory, and then applies it:

```c++
// cub/block/specializations/block_scan_warp_scans.cuh
T block_aggregate;
ExclusiveScan(input, exclusive_output, scan_op, block_aggregate);

if (warp_id == 0)
{
  T block_prefix = block_prefix_callback_op(block_aggregate);
  if (lane_id == 0) { /* stash block_prefix in temp_storage */ }
}

__syncthreads();

T block_prefix = temp_storage.block_prefix;
if (linear_tid > 0) { exclusive_output = scan_op(block_prefix, exclusive_output); }
```

Two consequences follow from this placement, and both matter for the abstraction. The aggregate
is produced by the block scan's upsweep and consumed by its downsweep, so the per-thread and
per-warp partials never leave registers. And the publish, the search, and the inclusive publish all
happen at a single program point, on a single warp.

### Critical path

Within a block, the shape is:

```text
warp 0    | load | sync | upsweep | publish . spin . reduce . publish | sync | downsweep | store |
warp 1    | load | sync | upsweep |     blocked in __syncthreads      | sync | downsweep | store |
...       |      |      |         |                                  |      |           |       |
warp N-1  | load | sync | upsweep |     blocked in __syncthreads      | sync | downsweep | store |
```

While warp 0 spins on global memory, the other `N-1` warps have nothing to do. That is the
structural intra-block cost: one warp of latency-bound work stalls the entire block's worth of
throughput-bound work.

Across blocks, the constraint is what one might call the *inclusive frontier*. Tile `K` cannot
publish `SCAN_TILE_INCLUSIVE` until it has read a `SCAN_TILE_INCLUSIVE` from some `j < K`,
plus every `SCAN_TILE_PARTIAL` in between. In the best case a single 32-wide window read finds a
terminating inclusive predecessor and the frontier advances 32 tiles per L2 round trip, giving a
serial chain of roughly `ceil(num_tiles / 32)` links. In the worst case predecessors are still
invalid and the window read has to be repeated. Either way, there is a genuine cross-block serial
dependency that scales with the number of tiles.

### Why there is a delay policy

Because the search spins on global loads, unthrottled polling competes for L2 bandwidth with the
producers it is waiting for. `single_pass_scan_operators.cuh` therefore carries a whole taxonomy of
back-off strategies: `no_delay_constructor_t`, `fixed_delay_constructor_t`,
`exponential_backoff_constructor_t`, `exponential_backon_constructor_t`, and jittered and
windowed variants of both. The default for a primitive or trivially copyable accumulator is
`fixed_delay_constructor_t<350, 450>`: sleep about 450 ns once on construction, which is the
assumed L2 write latency, then about 350 ns between retries. The delay is itself conditional on grid
size (`GridThreshold = 500`): below that many blocks, `__nanosleep` is replaced by a plain
`__threadfence_block()` that only prevents the compiler from hoisting the load out of the loop.
Per-algorithm tunings override the constructor, and `agent_batch_memcpy` uses two different ones in
the same kernel.

The existence and the size of this taxonomy is the clearest evidence that the polling traffic is a
real cost, not an implementation detail.

(cub-tile-prefix-lookahead)=

## How look-ahead works

### Tile state

`warpspeed::tile_state_t<AccumT>` is a `{scan_state state; AccumT value}` pair, over-aligned to
the next power of two so that the whole thing fits one native atomic access (16 bytes on SM90 and
newer, 8 otherwise), with a non-atomic fallback that uses a `STORE_CG` payload store followed by a
release store of the state. The array has exactly `num_tiles` entries; there is no padding, because
nothing ever indexes backwards.

The decisive difference from look-back is the state machine:

```c++
// cub/detail/warpspeed/look_ahead.cuh
enum scan_state : ::cuda::std::uint32_t
{
  empty          = 0,
  tile_aggregate = 1,
};
```

**There is no inclusive state, and no inclusive prefix is ever published.** Every block re-derives
the prefix for itself, from tile 0, every time. That single decision is what removes the cross-block
serial chain described above, and it is the reason the two mechanisms cannot be made to share a tile
state without the encoding being a parameter.

### Execution model

The look-ahead scan kernel is a persistent, warp-specialized kernel built on the `warpspeed`
framework. A block is partitioned into five *squads*, contiguous groups of warps each with one role,
described at compile time by `warpspeed::SquadDesc` and dispatched once at kernel entry by
`warpspeed::squadDispatch`:

| squad | warps | role |
| --- | --- | --- |
| `squad_reduce` | `reduce_and_scan_warps` | reduce the tile, publish its aggregate, stage thread and warp partials |
| `squad_scan_store` | `reduce_and_scan_warps` | combine partials with the block prefix, scan, store the tile |
| `squad_load` | 1 | bulk-copy the tile into shared memory |
| `squad_sched` | 1 | obtain the next tile index |
| `squad_lookahead` | 1 | resolve the prefix of all preceding tiles |

Tiles are acquired dynamically, through `clusterlaunchcontrol.try_cancel` on SM100 and an
`atomicAdd` counter on SM90. Every squad iterates over the same sequence of tile indices, but they
are decoupled by multi-stage, mbarrier-backed shared-memory resources, so at steady state different
squads are working on different tiles, up to `numStages` apart. Each resource has a producer phase
and a consumer phase, and the squads bound to each phase are declared in `setup_scan_resources`.
Writing them as producer, resource, consumer:

```text
within a block, for tile k:

  sched      --> smemNextBlockIdx      --> every squad
  load       --> smemInOut             --> reduce and scan_store
  reduce     --> smemThreadAndWarpAggr --> scan_store
  lookahead  --> smemAggrExclusiveCta  --> scan_store
  scan_store --> smemInOut             --> bulk store to global

across blocks:

  reduce of the owning block --> tile_state[k] in global --> lookahead warp of *every* block
```

Note that the tile's aggregate is published by `squad_reduce`, the prefix is consumed by
`squad_scan_store`, and the work of producing that prefix is done by a third squad entirely.
No single group performs all three operations.

### Resolving the prefix

`warpspeed::warpIncrementalLookahead` runs on one warp and maintains a monotone forward cursor,
`(idxTilePrev, aggrExclusiveCtaPrev)`, that **persists in registers across iterations of the
block's tile loop**. It never waits for an inclusive prefix, because none exists; it waits only for
aggregates, and folds them in order:

```c++
// cub/detail/warpspeed/look_ahead.cuh, condensed
while (idxTileCur < idxTileNext)
{
  warpLoadLookahead(laneIdx, regTmpStates, ptrTileStates, idxTileCur, idxTileNext, num_tiles);

  for (int idx = 0; idx < numTileStatesPerThread; ++idx)
  {
    // which lanes hold a ready aggregate
    const auto warp_has_aggregate_mask =
      __ballot_sync(0xffffffffu, regTmpStates[idx].state == scan_state::tile_aggregate);
    // the contiguous run of ready lanes starting at the low end
    const auto warp_right_aggregates_mask = warp_has_aggregate_mask & (~warp_has_aggregate_mask - 1);
    if (warp_right_aggregates_mask == 0) { break; }

    const auto n = ::cuda::std::popcount(warp_right_aggregates_mask);
    AccumT local_aggr = /* warp_redux_sm80 over the masked lanes, or WarpReduce::Reduce(value, op, n) */;

    aggrExclusiveCtaCur = idxTileCur == 0 ? local_aggr : scan_op(aggrExclusiveCtaCur, local_aggr);
    idxTileCur += n;
    if (n < 32) { break; }
  }
}
```

Only a *contiguous* run of ready tiles starting at the cursor can be consumed, because the fold must
respect scan order for a possibly non-commutative operator. Each round issues a coalesced 32-wide
(times `lookahead_items_per_thread`) vector load of tile states, so a not-yet-ready tile costs a
re-read of the window rather than a stall.

There is no back-off policy anywhere in this loop. The warp polls as fast as it can, because it has
nothing else to do, the entire tile-state array is small enough to stay resident in L2, and the loads
are coalesced.

### Critical path

Across blocks: there is none. Tile `K`'s aggregate is published as soon as `squad_reduce`
finishes reducing it, independently of every other tile. The chain of inclusive publications that
constrains look-back simply does not exist.

What replaces it is per-block work: folding `ceil(num_tiles / 32)` warp-reduced batches serially
through `scan_op`. That is ALU work on one warp, overlapped with the load, reduce, scan, and store
of other tiles that are in flight in the pipeline. The compute squads never block on global memory;
they block only on mbarriers guarding their shared-memory phase.

The cost of this arrangement is that the tile data is traversed twice in shared memory rather than
once in registers. `squad_reduce` reads the tile to produce the aggregate as early as possible, and
`squad_scan_store` reads it again to perform the scan. The per-thread and per-warp partials
computed during the reduction have to travel between two different warp groups, so they go through
`smemThreadAndWarpAggr` instead of staying in registers. Paying for a second shared-memory pass to
publish the aggregate earlier is the central trade of the design.

(cub-tile-prefix-comparison)=

## Commonalities and differences

### What they share

The overlap is real and load-bearing, which is what makes a shared abstraction plausible in the
first place.

- **The value contract is identical.** An associative binary operator, a per-tile aggregate, and an
  exclusive prefix over all preceding tiles, delivered to a group of threads that will apply it.
- **The storage shape is nearly identical.** A global array of `{state, value}` records carved out
  of `d_temp_storage`, zeroed by a separate init kernel, with a host-side size query.
- **A warp is the resolution unit in both.** Both examine a 32-wide window of tile states, both use
  a ballot to decide how much of that window is usable, and both collapse it with a segmented or
  partial warp reduce. `TailSegmentedReduce` with `SwizzleScanOp` and `warp_redux_sm80` over a
  ballot-derived mask are different mechanics for the same operation.
- **Both special-case tile 0** and both fold an optional initial value into it.
- **Both suppress publication where nobody will read it.** Look-back skips `SetInclusive` on the
  last tile; look-ahead skips the whole lookahead step on the first.
- **Both grew a run-to-run-deterministic variant, and both solved it the same way**, by pinning
  reductions to aligned 32-tile batches. See {ref}`cub-tile-prefix-determinism`.
- **Both ultimately broadcast a single value through shared memory** to the group that applies it.

### What differs

| axis | decoupled look-back | look-ahead |
| --- | --- | --- |
| tile state encoding | four statuses, 32 padding entries | two states, no padding |
| publishes an inclusive prefix | yes, mandatory; it is what bounds the search | never; the concept does not exist |
| search direction | backwards from `tile_idx - 1`, swizzled reduce | forwards from a persisted cursor |
| who publishes the aggregate | warp 0, inside the block scan | `squad_reduce`, an earlier pipeline stage |
| who resolves the prefix | the same warp 0, immediately after | `squad_lookahead`, a different warp, concurrently |
| who consumes the prefix | the same warp 0, then the whole block | `squad_scan_store`, a third group |
| state lifetime | stateless; a fresh functor per tile | stateful; a cursor that outlives the tile loop |
| blocking behaviour | blocks the entire block | blocks nothing but the lookahead warp |
| back-off | a tuned `DelayConstructor` taxonomy | none |
| tile assignment | static, `start_tile + blockIdx.x` | dynamic stealing, and monotonicity is *required* |
| passes over tile data | one, in registers | two, through shared memory |
| synchronization | `__syncthreads()`, `__any_sync`, `__all_sync` | mbarriers, per-squad named barriers, staged shared memory |
| hardware | anything (SM70 and newer for `__nanosleep`) | SM90 and newer, PTX ISA 8.6; SM100 for `clusterlaunchcontrol` |

### The four differences that actually constrain an abstraction

Most of the rows above are implementation detail that a trait or a policy can absorb. Four are
structural, and any proposed API has to answer them directly.

**1. The operations run on different groups.** In look-back, publish, search, and inclusive-publish
are the same warp at one program point. In look-ahead they are three different squads. A callback
signature that does not name the group cannot express the second case.

**2. There is a fourth operation with no obvious callback.** Somebody has to run
`warpIncrementalLookahead` once per tile. It is not "publish my aggregate", it is not "give me the
prefix", and it is not "publish my inclusive prefix". It is a background service that has to be
pumped, and it needs a warp of the block's budget that the algorithm author never writes code for.
This is what forces inversion of control: the abstraction has to own the tile loop, because the tile
loop is what drives the pump.

**3. The resolver is stateful across tiles.** `TilePrefixCallbackOp` is constructed inside the tile
loop, once per tile, and takes `tile_idx` as a constructor argument. `warpIncrementalLookahead`'s
cursor must survive from one tile to the next. The abstraction's object must therefore be
constructed *above* the tile loop and take `tile_idx` per call. That is a small but breaking change
to the look-back side.

**4. Look-back's ordering constraints must not be imposed on look-ahead.** In look-back,
`SetPartial` for tile `K` must happen before other blocks look back through `K`, and the search
must complete before `SetInclusive`. In look-ahead, publishing tile `K`'s aggregate and acquiring
tile `K`'s prefix are *independent and concurrent*, running in different squads. An abstraction
that expresses the prefix as a function of the aggregate, as `T operator()(T block_aggregate)`
does today, silently serializes look-ahead and destroys the pipeline.

(cub-tile-prefix-inventory)=

## What the current consumers need

Look-ahead serves one algorithm. Look-back serves eight CUB agents, running nine independent chains
between them, plus two Thrust CUDA-backend algorithms. Any abstraction that is meant to replace
`TilePrefixCallbackOp` has to cover all of them, so this section catalogues what each actually
uses.

### Summary

| consumer | tile state | accumulator | invocation | reads back |
| --- | --- | --- | --- | --- |
| `agent_scan` | `ScanTileState<AccumT>` | user accumulator | `BlockScan` callback | nothing |
| `agent_scan_by_key` | `ReduceByKeyScanTileState<AccumT, int>` | `KeyValuePair<int, AccumT>` | `BlockScan` callback | aggregate |
| `agent_select_if` | `ScanTileState<OffsetT>` via `tile_state_with_memory_order` | selection count | `BlockScan` callback | all three |
| `agent_unique_by_key` | `ScanTileState<OffsetT>` | selection count | `BlockScan` callback | all three |
| `agent_three_way_partition` | `ScanTileState<AccumPackT>` | packed pair of counters | `BlockScan` callback | all three |
| `agent_reduce_by_key` | `ReduceByKeyScanTileState<AccumT, OffsetT>` | `KeyValuePair<OffsetT, AccumT>` | `BlockScan` callback | all three, by field |
| `agent_rle` | `ReduceByKeyScanTileState<LengthT, OffsetT>` | `KeyValuePair<OffsetT, LengthT>` | manual, warp 0 only | exclusive and inclusive, as members |
| `agent_batch_memcpy` (buffers) | `ScanTileState<uint32_t>` | block-level buffer count | manual, first warp only | exclusive |
| `agent_batch_memcpy` (blocks) | `ScanTileState<BlockOffsetT>` | block-tile count | `BlockScan` callback | nothing |
| `thrust::set_operations` | `ScanTileState<Size>` | count | `BlockScan` callback | all three |
| `thrust::reduce_by_key` | `ReduceByKeyScanTileState<value_type, size_type>` | `KeyValuePair` | `BlockScan` callback | all three |

Two conventions hold across every multi-tile consumer. Tile 0 never constructs a
`TilePrefixCallbackOp`; it does a plain local block scan and publishes the result directly with
`SetInclusive(0, aggregate)`. And the last tile does not publish at all, since nothing follows it.
One accessor, `GetTileIdx()`, is defined but called by no agent; only the standalone example in
`cub/examples/device/example_device_decoupled_look_back.cu` uses it.

### The awkward cases

The summary table hides the constraints that will actually shape the API.

**A bare exclusive prefix is not enough.** Six of the eleven chains in the table read back the
exclusive prefix, the inclusive prefix, and the tile aggregate, all three. `agent_select_if` is
representative:

```c++
// cub/agent/agent_select_if.cuh
OffsetT num_tile_selections   = prefix_op.GetBlockAggregate();
OffsetT num_selections        = prefix_op.GetInclusivePrefix();
OffsetT num_selections_prefix = prefix_op.GetExclusivePrefix();
OffsetT num_rejected_prefix   = tile_offset - num_selections_prefix;
```

For compaction algorithms the exclusive prefix is a scatter offset, the inclusive prefix is the
running global count, and the aggregate is the count within this tile. The abstraction must return
all three, not a single value.

**Some consumers do not use `BlockScan` at all.** `agent_rle` performs a warp-level scan and
then calls the prefix functor by hand, on warp 0 only, reading its result through public data
members rather than the accessors:

```c++
// cub/agent/agent_rle.cuh
TilePrefixCallbackOpT prefix_op(
  tile_status, temp_storage.aliasable.scan_storage.prefix, ::cuda::std::plus<>{}, tile_idx);
unsigned int warp_id = ((WARPS == 1) ? 0 : threadIdx.x / WARP_THREADS);
if (warp_id == 0)
{
  prefix_op(tile_aggregate);
  if (threadIdx.x == 0) { temp_storage.tile_exclusive = prefix_op.exclusive_prefix; }
}
```

`agent_batch_memcpy` does the same thing for its buffer-count chain, guarded by
`threadIdx.x < warp_threads`. So the abstraction cannot assume the consuming group is the whole
block, and it cannot assume the value arrives through a block-scan callback.

**One kernel may need more than one chain.** `agent_batch_memcpy` runs two entirely independent
look-back chains simultaneously: one over the count of block-level buffers, one over the count of
block-level tiles, each with its own tile state, its own accumulator type, and its own delay
constructor (`buff_delay_constructor` and `block_delay_constructor`). Any design that assumes
one prefix mechanism per kernel is wrong. For a look-ahead backend this raises a resource question:
two chains would either need two lookahead warps or one warp servicing both cursors.

**Memory ordering is a functional requirement, not a tuning knob.** `agent_select_if` is the only
consumer of `tile_state_with_memory_order`, and it needs it for correctness:

```c++
// cub/agent/agent_select_if.cuh
static constexpr MemoryOrder memory_order =
  ((SelectionOpt == SelectImpl::SelectPotentiallyInPlace) && (!loads_via_smem))
    ? MemoryOrder::acquire_release
    : MemoryOrder::relaxed;
```

For potentially in-place compaction, all of a tile's loads must be ordered before the release store
that publishes its state, and the acquiring load of predecessor states must be ordered before the
compacted items are written. The agent additionally places a `__syncthreads()` immediately before
the scan so that the loads of *every* thread in the block, not just the publishing thread, precede
the release. The abstraction must let a consumer specify the memory order, and must preserve the
ability to insert a block-wide fence between "tile loaded" and "aggregate published".

**Accumulators are not always scalars.** Three families appear:

- `KeyValuePair` states driven by `ReduceBySegmentOp` or `ScanBySegmentOp`, where `key`
  counts segments and `value` carries a partial reduction (`reduce_by_key`, `scan_by_key`,
  `rle`). `ReduceByKeyScanTileState` packs these with a bespoke layout that reorders the fields
  depending on whether `sizeof(ValueT) == sizeof(KeyT)`, in order to fit a transaction word.
- Packed multi-counter states, as in `agent_three_way_partition`, where `AccumPackT` is either a
  `uint64_t` holding two 32-bit counters or a `pair_pack_t` struct with an element-wise
  `operator+`.
- Plain scalars everywhere else.

Look-ahead's `tile_state_t<AccumT>` is already generic over the payload and already has a
non-atomic fallback for payloads that do not fit a native atomic, so it may well subsume
`ReduceByKeyScanTileState` outright. That should be verified rather than assumed, because the
bespoke packing exists to keep specific type combinations inside one transaction word.

**Streaming variants bypass the mechanism for tile 0.** `reduce_by_key`, `rle`, `select_if`,
and `three_way_partition` all have streaming forms in which a partition's tile 0 is seeded from
`streaming_context.prefix()` rather than from a look-back, and in which the final partition writes
a global count. The abstraction needs a way to inject an externally supplied prefix for the first
tile instead of resolving one.

**Thrust is a client too.** `thrust/system/cuda/detail/set_operations.h` and
`thrust/system/cuda/detail/reduce_by_key.h` instantiate `cub::ScanTileState`,
`cub::ReduceByKeyScanTileState`, and `cub::TilePrefixCallbackOp` directly. These types are also
reachable from user code, and there is a public example and a public test for them
(`cub/examples/device/example_device_decoupled_look_back.cu`,
`cub/test/catch2_test_device_decoupled_look_back.cu`). Whatever is proposed here has to either
keep the existing spelling working or be staged behind a deprecation.

(cub-tile-prefix-proposed-api)=

## Proposed API

The proposal is three layers, ordered by increasing risk and decreasing independence. Layer 1 is
useful on its own and could land first. Layer 2 is the abstraction proper. Layer 3 is what makes
Layer 2 usable for a warp-specialized backend, and is where the performance risk concentrates.

:::{note}
All code in this section is exposition only. Names are placeholders.
:::

### Layer 1: the tile state

The lowest layer owns storage, allocation, initialization, publication, and observation. It knows
nothing about groups, warps, or how the prefix is resolved.

```c++
namespace cub::detail {

// Traits are a *type*, not a value: C++17 does not allow class-type NTTPs, and CUB
// already works around this elsewhere (see the note on agent policies in the
// device-scope developer guide).
struct lookback_traits
{
  static constexpr bool publishes_inclusive = true;
  static constexpr int  index_padding       = 32;  // OOB entries before tile 0
  static constexpr MemoryOrder order        = MemoryOrder::relaxed;
};

struct lookahead_traits
{
  static constexpr bool publishes_inclusive = false;
  static constexpr int  index_padding       = 0;
  static constexpr MemoryOrder order        = MemoryOrder::relaxed;
};

template <typename T, typename Traits>
struct tile_state
{
  // host
  static constexpr cudaError_t allocation_size(int num_tiles, size_t& bytes);
  cudaError_t init(int num_tiles, void* d_temp_storage, size_t bytes);

  // device, from the init kernel
  __device__ void initialize_status(int num_tiles);

  // device, publication
  __device__ void publish_aggregate(int tile_idx, T aggregate);
  __device__ void publish_inclusive(int tile_idx, T inclusive);  // no-op unless publishes_inclusive

  // device, observation; never blocks
  struct observation { tile_status status; T value; };
  __device__ observation load(int tile_idx);
};

}  // namespace cub::detail
```

`publishes_inclusive` collapses to nothing on the look-ahead side, so `publish_inclusive` is an
empty inline function there and the `SCAN_TILE_INCLUSIVE` status disappears from the encoding.
`index_padding` accounts for look-back's 32 `SCAN_TILE_OOB` entries, which look-ahead does not
need because it never indexes backwards. `order` covers `agent_select_if`'s acquire/release
requirement and subsumes `tile_state_with_memory_order`.

This layer alone unifies `ScanTileState`, `ReduceByKeyScanTileState`,
`warpspeed::tile_state_t`, both init kernel bodies, and both host-side temp-storage size queries.
It is a mechanical consolidation with no effect on the critical path, and it is the only layer that
can plausibly be adopted without touching any agent.

### Layer 2: the prefix service

This is the three-callback idea, plus the two things the three callbacks turn out to be missing: a
group parameter on every operation, and a fourth operation for the background work.

```c++
template <typename T, typename ScanOp, typename TileState>
struct prefix_service  // exposition only, a concept rather than a class
{
  // --- static description ---

  // Which execution roles this service needs beyond the caller's own group.
  // Empty for look-back; one warp for look-ahead.
  static constexpr auto roles = /* ... */;
  static constexpr int  shared_bytes(/* config */);

  // --- per tile ---

  // Make this tile's own aggregate visible to other blocks.
  template <typename Group>
  __device__ void publish_aggregate(Group producer, int tile_idx, T aggregate);

  // Obtain the aggregate of all tiles before tile_idx, valid on every thread of `consumer`.
  // May block.
  template <typename Group>
  __device__ T acquire_prefix(Group consumer, int tile_idx);

  // Make the aggregate of all tiles up to and including this one visible. May be a no-op.
  template <typename Group>
  __device__ void publish_inclusive(Group producer, int tile_idx, T inclusive);

  // Drive whatever background work the service needs to make progress on tile_idx.
  // Empty for look-back.
  template <typename Group>
  __device__ void advance(Group service, int tile_idx);
};
```

The mapping onto the two existing mechanisms is direct:

| operation | decoupled look-back | look-ahead |
| --- | --- | --- |
| `publish_aggregate` | `SetPartial`, on warp 0 | `storeTileAggregate`, on `squad_reduce` |
| `acquire_prefix` | the blocking sliding-window search, on warp 0 | read the staged value from `smemAggrExclusiveCta`, on `squad_scan_store` |
| `publish_inclusive` | `SetInclusive`, on warp 0 | nothing |
| `advance` | nothing | `warpIncrementalLookahead`, on `squad_lookahead` |

Five design points deserve argument.

**The service object is constructed above the tile loop.** `TilePrefixCallbackOp` is constructed
per tile and receives `tile_idx` in its constructor. Look-ahead's cursor cannot survive that, so
every operation takes `tile_idx` as a parameter instead. This is a breaking change to the
look-back spelling, and is the main reason the existing public types cannot simply be retrofitted.

**Ordering is a property of the backend, not of the interface.** The interface deliberately does not
say that `acquire_prefix` depends on `publish_aggregate`. In look-back the dependency exists and
is enforced inside the backend. In look-ahead the two are concurrent, in different squads, and any
interface that expresses the prefix as `T operator()(T aggregate)` forces a false dependency that
serializes the pipeline. This is why the current signature cannot be kept.

**The service returns only the exclusive prefix.** The other two values that consumers want are
derived rather than resolved: the tile aggregate comes from the algorithm's own reduce phase, and
the inclusive prefix is just `scan_op(exclusive, aggregate)`. Making the service responsible for
either would be wrong, because in look-ahead the service runs on a squad that never sees the tile
data at all. The driver assembles all three instead, which it can do because it is the thing that
called the reduce phase.

**The assembled triple must be dead-code-eliminable.** The driver hands the algorithm's finish phase
a `prefix_result<T>` with `exclusive`, `inclusive`, and `aggregate` members, because six
chains need all three. For `agent_scan`, which uses only `exclusive`, the other two must cost
nothing. With full inlining, scalar replacement should remove them; for the look-ahead backend that
also means removing the shared-memory round trip that would carry the tile aggregate from
`squad_reduce` to `squad_scan_store`. That round trip is cheap to add, since `reduce_tile`
already computes `regSquadAggr` and `smemThreadAndWarpAggr` already exists, but it must vanish
when unused. If measurement shows it does not, the fallback is to return an accessor object with
member functions rather than a struct with members.

**Multiple services must coexist.** `agent_batch_memcpy` needs two. Nothing in the interface
prevents that, but `roles` and `shared_bytes` have to compose, and a look-ahead backend has to
decide whether two chains get two warps or share one.

### Layer 3: the driver

Layer 2 alone is not enough, because of `advance`. Someone has to call it once per tile, on a
group that exists only because the service asked for it, and the algorithm author does not write
that code. The tile loop therefore has to belong to the abstraction. On the look-ahead side the loop
additionally owns stage and phase RAII, tile stealing, and squad dispatch, none of which the
algorithm can see.

So the algorithm is expressed as two phases and handed to a driver:

```c++
struct tile_context
{
  int    tile_idx;
  size_t tile_offset;
  int    valid_items;
  bool   is_first_tile;
  bool   is_last_tile;
};

template <typename T>
struct prefix_result
{
  T exclusive;  // aggregate of all preceding tiles
  T inclusive;  // ... including this one
  T aggregate;  // this tile only
};

struct my_algorithm  // exposition only
{
  using accum_t   = /* ... */;
  using carrier_t = /* intermediates that travel between the two phases */;

  // Produce this tile's aggregate. Must not publish anything itself.
  template <typename Group>
  __device__ accum_t reduce_tile(Group g, tile_context ctx, carrier_t& carrier);

  // Consume the resolved prefix and produce this tile's output.
  template <typename Group>
  __device__ void finish_tile(Group g, tile_context ctx, carrier_t& carrier,
                              prefix_result<accum_t> prefix);
};

template <typename Config, typename Algorithm>
__device__ void tile_prefix_driver(Config cfg, Algorithm algo);
```

The driver's contract is to call `reduce_tile`, publish its result, resolve the prefix, and call
`finish_tile`, for every tile assigned to the block, while keeping `advance` pumped. How it does
that differs completely between backends:

| | look-back driver | look-ahead driver |
| --- | --- | --- |
| groups | one group: the whole block | five squads |
| tile loop | `start_tile + blockIdx.x` | dynamic stealing, monotone per block |
| `carrier_t` storage | registers | shared memory |
| between the two phases | publish, acquire, publish inclusive, all on warp 0 | nothing; the squads are decoupled by mbarriers |
| `advance` | not called | once per tile on `squad_lookahead` |

#### The carrier is the crux

Today's look-back never materializes a carrier at all. `BlockScan`'s upsweep and downsweep are one
call, and the per-thread and per-warp partials live in registers inside it. Splitting the algorithm
into `reduce_tile` and `finish_tile` appears to force those partials out into something
addressable, which would be a regression.

It only appears to, because the two phases run on the same threads in the look-back backend. If the
driver calls them back to back in straight-line inlined code, `carrier_t` is a local object and
scalar replacement keeps it in registers. What is genuinely missing is a two-phase block scan: CUB
exposes `BlockScan::ExclusiveScan(items, items, op, prefix_op)` as one fused call, and
`BlockScanWarpScans` does the split internally but does not surface it. The proposal therefore
depends on adding something like:

```c++
// exposition only
typename BlockScanT::upsweep_state state;
AccumT aggregate = BlockScanT(storage).Upsweep(items, scan_op, state);
/* ... prefix resolution happens here ... */
BlockScanT(storage).Downsweep(items, items, scan_op, state, prefix);
```

with `BlockScan::ExclusiveScan(..., prefix_op)` reimplemented on top of it, so the existing fused
path is provably the same code. For the look-ahead backend, `upsweep_state` is what has to be
shared-memory backed, which is precisely what `smemThreadAndWarpAggr` already is.

Whether the carrier's storage is chosen by the backend or declared by the algorithm is an open
question; see {ref}`cub-tile-prefix-open-questions`.

#### What `agent_scan` would look like

```c++
// exposition only
struct scan_algorithm
{
  using accum_t   = AccumT;
  using carrier_t = typename BlockScanT::upsweep_state;

  template <typename Group>
  __device__ accum_t reduce_tile(Group g, tile_context ctx, carrier_t& carrier)
  {
    load_tile(g, ctx, items);
    return BlockScanT(storage).Upsweep(items, scan_op, carrier);
  }

  template <typename Group>
  __device__ void finish_tile(Group g, tile_context ctx, carrier_t& carrier,
                              prefix_result<accum_t> p)
  {
    BlockScanT(storage).Downsweep(items, items, scan_op, carrier, p.exclusive);
    store_tile(g, ctx, items);
  }
};
```

Neither phase mentions look-back or look-ahead, neither mentions squads, and the choice between the
two becomes what `ScanPolicy::algorithm` already is: a tuning decision.

(cub-tile-prefix-groups)=

## The group abstraction

Every operation in Layer 2 takes a `Group`, and Layer 3 hands one to each algorithm phase. What
that type should be is the last structural question, and it is not obvious, because the two backends
have genuinely different execution models: look-back runs everything on a homogeneous block and
carves out warp 0 by hand, while look-ahead partitions the block into named squads at kernel entry.

### What the abstraction needs from a group

Very little, as it turns out. Collecting every use across both mechanisms yields:

```c++
// exposition only
template <typename G>
concept tile_group = requires(const G g) {
  { G::size() }             -> std::same_as<int>;   // thread count, compile-time
  { g.thread_rank() }       -> std::same_as<int>;
  { g.warp_rank() }         -> std::same_as<int>;
  { g.is_leader() }         -> std::same_as<bool>;  // one thread in the whole group
  { g.is_leader_of_warp() } -> std::same_as<bool>;  // one thread per warp
  g.sync();
};
```

`warpspeed::Squad` already models all of this, under different names
(`threadCount` / `threadRank` / `warpRank` / `isLeaderThread` / `isLeaderThreadOfWarp` /
`syncThreads`). A trivial `block_group` adapter whose `sync()` is `__syncthreads()` and whose
`size()` is `BLOCK_THREADS` models it too, as does a `warp_group` for the warp-0 sub-group that
look-back's publish and acquire run on.

### Three options

**Reuse `warpspeed::Squad` for both backends.** A single squad spanning every warp of the
block is a legal configuration: `squadDispatch` handles `numSquads == 1` and admits every warp.
So look-back could be expressed as a one-squad kernel. This is rejected for two reasons. First,
`Squad`'s constructor calls `::cuda::ptx::elect_sync`, and the framework around it
(mbarrier-backed resources, `clusterInitSync`) is gated on SM90 and `__cccl_ptx_isa >= 860`.
Routing look-back through it would raise the hardware floor of eight agents that currently run
everywhere, which is not an acceptable price for interface tidiness. Second, `Squad::syncThreads`
issues a named barrier (`__barrier_sync_count(mSquadIdx + 1, threadCount())`), not
`__syncthreads()`. For a squad covering the whole block those are equivalent in effect, but they
are not the same instruction, and every look-back agent's synchronization would change.

**Use `cooperative_groups` or the cudax hierarchy types.** These are the standard spellings,
but neither expresses the thing that matters here: a compile-time, contiguous, named *role* within
the block, which is what `SquadDesc` is and what the look-ahead backend needs in order to size
shared memory and register barrier arrival counts before the kernel starts.

**Define a thin concept and let both model it.** This is the recommendation. The concept above is
small enough that `warpspeed::Squad` satisfies it by renaming, and small enough that a whole-block
adapter is a few lines with no runtime state. Crucially, the look-back backend keeps
`__syncthreads()` and keeps working on every architecture, while the look-ahead backend keeps
squads, named barriers, and the compile-time DCE that `squadDispatch` gives it.

### Describing roles

The remaining piece is how a service asks for the warps it needs. Layer 2's `roles` should be a
declarative request that the driver resolves:

```c++
// exposition only
struct role_request
{
  int warps;  // 0 means "borrow the calling group, do not allocate"
};

// look-back
static constexpr role_request roles{0};
// look-ahead
static constexpr role_request roles{1};
```

The look-back driver sees `warps == 0`, allocates nothing, and runs `publish_aggregate`,
`acquire_prefix`, and `publish_inclusive` on a `warp_group` carved from the calling block. The
look-ahead driver sees `warps == 1`, appends a `SquadDesc` to its layout, and calls `advance`
on it once per tile. The same mechanism composes for `agent_batch_memcpy`'s two chains: two
services, two requests, and the driver decides whether to satisfy them with two warps or by
multiplexing one.

This keeps `SquadDesc` where it belongs, as an implementation detail of the look-ahead driver
rather than something an algorithm author has to know about.

(cub-tile-prefix-risks)=

## Determinism, cost, risks, and validation

(cub-tile-prefix-determinism)=

### Determinism

Both mechanisms grew a run-to-run-deterministic variant independently, and both arrived at the same
solution: force every reduction to cover a fixed, 32-tile-aligned batch, so that the shape of the
reduction tree does not depend on which tiles happened to finish first.

- `TilePrefixCallbackOp::lookback_stable_reduction_order` designates tiles whose index is a
  multiple of 32 as *anchors*. Only anchors publish `SCAN_TILE_INCLUSIVE`; everything else stays
  at `SCAN_TILE_PARTIAL`. Every tile then waits until its anchor is inclusive and reduces exactly
  the window from that anchor.
- `warpspeed::warpIncrementalLookaheadStable` advances its cursor only in whole batches of 32,
  starting from multiples of 32, and only once all 32 aggregates in a batch are ready. The trailing
  partial batch is handled separately, and the cursor is left on the last multiple of 32.

Because the two implementations agree on the mechanism, determinism should be an abstraction-level
knob rather than something each backend reinvents. Concretely, the service should expose a
compile-time `stable_reduction_order` flag with a documented meaning ("the fold over predecessor
aggregates is associated identically on every run"), and each backend implements it however it can.
See {ref}`cub-determinism` for the user-facing contract this feeds into.

### Cost model

The two mechanisms trade global-memory traffic against critical-path length, in opposite directions.

**Look-back** reads at least one 32-wide window per tile, so at least `32 * num_tiles` state reads
across the grid, plus a re-read for every retry while predecessors are still invalid. The reads are
narrow and latency-critical, which is why they need a back-off policy.

**Look-ahead** walks each block's cursor from tile 0 to the last tile that block processes, so
roughly `num_blocks * num_tiles` state reads, plus re-reads of windows that were not yet complete.
The block count is occupancy-limited rather than proportional to the input, but it is still an order
of magnitude more traffic than look-back. It is affordable for three reasons: the reads are fully
coalesced vector loads, the tile-state array is small enough to stay resident in L2, and none of it
is on the critical path.

A rough sense of scale: one billion 4-byte elements at a 4096-element tile is about 244 thousand
tiles, so an 8-byte tile state is a 2 MB array that comfortably stays in L2. A few hundred resident
blocks each walking that array is on the order of hundreds of megabytes of L2 reads, against several
gigabytes of DRAM traffic for the data itself.

Where look-back should still win:

- **Small inputs**, where there are not enough tiles to fill look-ahead's pipeline and the inclusive
  frontier has too few links to matter.
- **Expensive operators.** Look-ahead re-does the `ceil(num_tiles / 32)`-step fold in *every*
  block, whereas look-back's chain of inclusive publications is computed once and shared. A costly
  `scan_op` therefore scales badly for look-ahead.
- **Anything below SM90**, where look-ahead cannot run at all.

These are the conditions the policy selector will have to encode, and they are a good argument for
keeping the choice a tuning decision rather than a hard-coded architecture check.

### Backend selection

Nothing here changes how the backend is chosen. `ScanPolicy` already carries an `algorithm`
member of type `ScanAlgorithm` (`lookback` or `lookahead`), alongside separate
`ScanLookbackPolicy` and `ScanLookaheadPolicy` sub-policies, and `dispatch_scan.cuh` branches
on it into `__invoke_lookback_algorithm` or `__invoke_lookahead_algorithm`. The host-side
differences in temp-storage layout and init kernel are already bridged by
`tile_state_kernel_arg_t`, a union over the two tile-state representations.

Generalizing that pattern to the other seven agents is mostly mechanical, and Layer 1 is what makes
it so: once both tile states expose the same allocation and initialization interface, the union and
the two `__invoke_*` paths collapse into one.

### Risks

In rough order of how likely they are to sink the proposal:

1. **The two-phase block scan may not be free.** The whole design rests on the claim that splitting
   `BlockScan`'s fused callback form into an upsweep and a downsweep produces identical code when
   both halves are inlined into the same threads. This is plausible, since `BlockScanWarpScans`
   already performs the split internally, but it must be measured, not assumed.
2. **Unused members of `prefix_result` may not disappear.** `agent_scan` uses only the
   exclusive prefix. If the inclusive prefix and the tile aggregate survive into the generated code,
   the look-ahead backend pays a shared-memory round trip per tile for nothing.
3. **Inversion of control may perturb register allocation.** The look-back agents are tuned tightly
   against occupancy, and the tuning database in `tuning_*.cuh` is keyed to the current agent
   shapes. A restructuring that changes register pressure invalidates tunings even if the SASS is
   otherwise equivalent.
4. **Look-ahead may not generalize to the harder accumulators.** It has only ever run with scalar
   accumulators. `KeyValuePair` states and packed multi-counter states are untested there, and the
   bespoke packing in `ReduceByKeyScanTileState` exists for a reason.
5. **Public surface.** `cub::ScanTileState`, `cub::ReduceByKeyScanTileState`, and
   `cub::TilePrefixCallbackOp` are reachable from user code, used by two Thrust algorithms, and
   covered by a public example and test. Layer 2 changes their shape (the service moves above the
   tile loop), so this needs a deprecation path rather than a rename.
6. **The warpspeed framework is young.** It has one consumer today. Making it the substrate for
   eight agents multiplies the cost of any design mistake in it.

(cub-tile-prefix-open-questions)=

### Open questions

1. Should a two-phase `BlockScan` (`Upsweep` returning the aggregate, `Downsweep` accepting a
   prefix) become part of the block-scope API, or stay in `detail`? If public, `upsweep_state`
   becomes a documented type.
2. Is `carrier_t`'s *storage* chosen by the backend or declared by the algorithm? Backend-chosen
   is cleaner but requires the algorithm to access it through a handle rather than directly.
3. Does `warpspeed::tile_state_t` subsume `ReduceByKeyScanTileState`, or does the bespoke field
   reordering buy a transaction-word fit that the generic version loses?
4. For `agent_batch_memcpy`'s two chains under a look-ahead backend: two lookahead warps, or one
   warp multiplexing two cursors?
5. Does look-ahead need a back-off policy once it serves more algorithms and more chains per kernel?
   Today it polls unthrottled, which is safe with one chain and one warp.
6. How do streaming contexts inject an externally supplied prefix for tile 0 in place of resolving
   one? A `seed` parameter on the driver seems natural but interacts with the first-tile fast path.
7. Can `GetTileIdx()` be dropped? No agent calls it; only the public example does.
8. Does the look-back backend keep static tile assignment, or does the driver always use dynamic
   stealing? Look-ahead requires monotone assignment; look-back does not care, but changing it would
   perturb tunings.

### Validation plan

None of this is code yet, but the acceptance criteria should be fixed before any is written, because
"zero overhead" is the entire premise.

**Codegen equivalence.** Use the `sass-diff` skill to compare generated SASS before and after, at a
fixed architecture, for every look-back agent that gets ported. The bar for a pure refactoring step
(Layer 1, and the two-phase `BlockScan` with the fused form reimplemented on top of it) is a clean
diff after normalization. For later steps the bar is a reviewed, explained diff.

**Benchmarks.** `cub.bench.scan.exclusive.sum.base` covers the look-back path;
`cub.bench.scan.exclusive.sum.lookahead` covers look-ahead. Note a gap: the look-ahead benchmark
guards itself off below `sm_100` and refuses to build for multiple architectures, so the SM90
look-ahead path (the one using the `atomicAdd` scheduler rather than `clusterlaunchcontrol`)
currently has no benchmark coverage. That should be closed before more algorithms depend on it.
Once compaction and by-key algorithms are ported, their benchmarks (`select.if`,
`partition.three_way`, `reduce.by_key`, `rle.encode`) join the set.

**Tests.** At minimum:

```bash
ci/util/build_and_test_targets.sh \
  --preset cub-cpp20 \
  --build-targets "cub.cpp20.test.device_scan cub.cpp20.test.device_decoupled_look_back" \
  --ctest-targets "cub.cpp20.test.device_scan|cub.cpp20.test.device_decoupled_look_back"
```

plus `catch2_test_device_scan_deterministic.cu` and
`test_device_scan_warpspeed_shifted_output.cu` for the determinism and look-ahead paths, and the
corresponding `device_select_if` / `device_reduce_by_key` / `device_run_length_encode` /
`device_three_way_partition` tests as each is ported. Thrust's `set_operations` and
`reduce_by_key` must keep building.

### Suggested staging

Each step should be independently mergeable and independently revertible.

1. **Layer 1 only.** Unify the tile states behind `tile_state<T, Traits>` with the existing types
   as aliases. No agent changes, no interface changes.
2. **Two-phase `BlockScan`.** Add `Upsweep` / `Downsweep` and reimplement the existing fused
   `ExclusiveScan(..., prefix_op)` on top of them. Gate on a clean SASS diff.
3. **Layers 2 and 3 for `cub::DeviceScan` only**, with both backends behind the existing
   `ScanPolicy::algorithm` switch. This is the real experiment: if look-back scan does not come out
   even, the design needs to change before anything else is touched.
4. **The compaction algorithms** (`select_if`, `unique_by_key`, `three_way_partition`), which
   share a shape and exercise the three-value result.
5. **The by-key and run-length algorithms** (`reduce_by_key`, `scan_by_key`, `rle`), which
   exercise multi-field accumulators and, in `rle`'s case, a non-`BlockScan` consumer.
6. **Batched memcpy**, the only consumer that exercises multiple concurrent services.
7. **Deprecate** the old public spellings once nothing in-tree uses them.
