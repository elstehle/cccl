# Proposal: refactor `multi_source_data_source` around the *single active source* invariant

## The current shape

`multi_source_data_source<SourceA, SourceB, OffsetT>` (in
`cub/detail/topk/tile_data_source.cuh`) is the runtime-switched two-source
adapter used by the keys and (currently) value channels of both top-k agents.
It composes two underlying `TileDataSource`s and picks between them at
construction via a runtime `bool pick_source_b`. Today it holds **both**
children alive for the lifetime of the instance:

```cpp
template <typename SourceA, typename SourceB, typename OffsetT = ::cuda::std::int64_t>
class multi_source_data_source {
public:
  struct TempStorage {
    typename SourceA::TempStorage a;   // <-- both arms always-alive in smem
    typename SourceB::TempStorage b;
  };

  using ScratchStorage = /* union over A/B::ScratchStorage, collapsed to
                            empty_storage_t when both children are empty */;

  // ...
  _CCCL_DEVICE _CCCL_FORCEINLINE void set_tile_base(OffsetT tile_base) {
    source_a.set_tile_base(tile_base);   // <-- redundant write to the
    source_b.set_tile_base(tile_base);   //     inactive arm every tile
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE full_load_handle submit_load(ScratchStorage& s) {
    // returns full_load_handle{ a{}, b{}, pick_b } -- BOTH handles alive
  }

private:
  SourceA source_a;            // <-- both child sources always-alive in registers
  SourceB source_b;
  bool pick_source_b;
};
```

Agent-side construction wires both children in lock-step (here from
`agent_batched_topk_last_filter::make_keys_source_for_segment`):

```cpp
_CCCL_DEVICE _CCCL_FORCEINLINE keys_source_t make_keys_source_for_segment(const per_segment_state_t& s) {
  key_source_input_t  key_src_input { s.d_keys_in,  storage.keys_source_state.a };
  key_source_buffer_t key_src_buffer{ s.in_key_buf, storage.keys_source_state.b };
  return keys_source_t{key_src_input, key_src_buffer, /*pick_b=*/s.load_from_candidates_buffer};
}
```

The agent drills `.a` / `.b` of the aggregate `TempStorage` directly, hands each
child its slot, then passes both pre-built children by value to the
`multi_source` ctor which copies them into `source_a` / `source_b`.

## The observation

`pick_source_b` is decided at the agent / multi_source ctor and **never
changes** for the rest of the multi-source's lifetime. Every call site already
respects this invariant — `set_tile_base`, `submit_load`, `gather_one`, and
every consumer's `complete_load`. Yet the type's representation behaves as if
both arms might be needed simultaneously:

1. **`TempStorage` is the sum of both children's TempStorages.** For
   `<direct, direct>` (today's config) both are `empty_storage_t`, so this
   costs nothing. For a future `<async_to_shared, direct>` (or
   `<sync_block_load, direct>`) it pessimistically reserves the inactive arm's
   mbarrier / BlockLoad state in smem.
2. **`set_tile_base` writes the `tile_base` member of both children every
   tile.** Per tile-iteration, that's one redundant register store for the
   inactive arm — usually small but it's unconditional and consumers can't
   elide it.
3. **Load handles carry both arms alive** with a `pick_b` tag — fine but
   structurally redundant: only one arm is ever read in `complete_load`.

The proposal is to collapse these three over-allocations into the *single
active source* invariant: at construction, decide on the arm; from then on
the two children alias one another at every level (TempStorage, the source
object itself, and the per-tile load handle).

## Design goals

1. **`SourceA::TempStorage` and `SourceB::TempStorage` alias in shared
   memory.** Only the active arm is materialized.
2. **Only the active source object exists.** The inactive child's ctor is
   never run; its slot is raw bytes.
3. **`set_tile_base` updates only the active source.**
4. **Load handles are a tagged union over the two children's handles.**
5. **The active arm is selected exactly once, at multi_source construction.**
   No "switching active arms mid-lifetime" semantics are added.
6. **`gather_one` continues to dispatch on the same runtime tag — no
   behavioural change.**
7. **The existing transitive `empty_storage_t` story is preserved**: when
   both children are empty, `TempStorage` and `ScratchStorage` both collapse
   to `empty_storage_t` so consumers (the agent's outer arena, the partition
   primitive's value-load slot) keep seeing the empty signal.
8. **No change to consumer API of the load handles** — `complete_load` /
   `gather_one` still take the same arguments and return the same values.

## Proposed API

### 1. `TempStorage` becomes a union (with transitive empty-collapse)

```cpp
template <typename SourceA, typename SourceB, typename OffsetT = ::cuda::std::int64_t>
class multi_source_data_source {
public:
  using value_t = typename SourceA::value_t;
  static_assert(::cuda::std::is_same_v<value_t, typename SourceB::value_t>,
                "multi_source_data_source requires both sources to share value_t");

private:
  static constexpr bool _temp_storage_is_empty =
       is_empty_storage_v<typename SourceA::TempStorage>
    && is_empty_storage_v<typename SourceB::TempStorage>;

  union _TempStorageInner {
    typename SourceA::TempStorage a;
    typename SourceB::TempStorage b;
  };
  // Uninitialized<> wrapper: caller can place this directly in __shared__ even
  // when an arm has a non-trivial ctor (e.g. async_to_shared's barrier).
  struct _TempStorageWrapped : CUB_NS_QUALIFIER::Uninitialized<_TempStorageInner> {};

public:
  using TempStorage =
    ::cuda::std::conditional_t<_temp_storage_is_empty, empty_storage_t, _TempStorageWrapped>;

  // ScratchStorage is unchanged from today's design (it already aliases through a
  // union and collapses to empty_storage_t when both children are empty).
  // ... existing _ScratchStorageInner / _ScratchStorageWrapped definitions ...
  using ScratchStorage =
    ::cuda::std::conditional_t<_scratch_storage_is_empty, empty_storage_t, _ScratchStorageWrapped>;
```

For `<direct, direct>`: `_temp_storage_is_empty` is `true`, so `TempStorage =
empty_storage_t` (was `struct{a, b}` of two empties → 1 byte; new shape is also
1 byte via EBO, but transitively detectable as empty).

For `<async_to_shared, direct>`: `_temp_storage_is_empty` is `false`,
`TempStorage` is `Uninitialized<union{barrier_state, empty}>` — sized to the
async source's barrier alone instead of the sum. Smem saving = `sizeof(SourceA::TempStorage)`
(typically tens of bytes; the inactive direct arm shrinks from 0 in the
struct-of-empties to truly 0 in the union).

### 2. The multi_source owns *one* in-place source via placement-new

Today the multi_source has `SourceA source_a; SourceB source_b;` as plain
members and the ctor copies pre-built children in. Under the refactor:

```cpp
private:
  // Raw-byte union over the two source types. The ctor placement-news the
  // active arm; the dtor placement-deletes it. The inactive arm is never
  // constructed -- its bytes are uninitialized and never read.
  union _ActiveSourceStorage {
    SourceA a;
    SourceB b;
    _CCCL_DEVICE _ActiveSourceStorage() {}    // raw bytes, no init
    _CCCL_DEVICE ~_ActiveSourceStorage() {}   // tagged dtor lives on the outer class
  };

  _ActiveSourceStorage active_source;
  bool pick_source_b;
```

The ctor takes the aggregate `TempStorage` (where the active arm's per-source
state lives) plus two factory callbacks. Only the active factory is called;
its result is placement-newed into the matching union arm. The active arm
binds its own internal references (e.g. async-shared's `loader(state.barrier)`)
to the chosen slot of the union'd `TempStorage`:

```cpp
public:
  // MakeA: callable `(SourceA::TempStorage&) -> SourceA`.
  // MakeB: callable `(SourceB::TempStorage&) -> SourceB`.
  //
  // Exactly one of MakeA / MakeB is invoked, decided by `pick_b`. The OTHER is
  // captured-but-never-called dead code that the optimizer eliminates.
  template <typename MakeA, typename MakeB>
  _CCCL_DEVICE _CCCL_FORCEINLINE
  multi_source_data_source(TempStorage& ts, bool pick_b, MakeA make_a, MakeB make_b)
      : pick_source_b(pick_b)
  {
    if constexpr (_temp_storage_is_empty) {
      // Both children publish empty TempStorage; pass on-stack empties of the
      // matching child type so the factory's ref binding works. Folded away.
      typename SourceA::TempStorage a_dummy{};
      typename SourceB::TempStorage b_dummy{};
      if (pick_b) {
        ::new (static_cast<void*>(&active_source.b)) SourceB(make_b(b_dummy));
      } else {
        ::new (static_cast<void*>(&active_source.a)) SourceA(make_a(a_dummy));
      }
    } else {
      auto& inner = ts.Alias();
      if (pick_b) {
        ::new (static_cast<void*>(&active_source.b)) SourceB(make_b(inner.b));
      } else {
        ::new (static_cast<void*>(&active_source.a)) SourceA(make_a(inner.a));
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE ~multi_source_data_source() {
    if (pick_source_b) { active_source.b.~SourceB(); }
    else               { active_source.a.~SourceA(); }
  }

  // Both copy AND move are implicitly deleted by `_ActiveSourceStorage` being
  // a union over types that may themselves have no copy/move (notably
  // `async_to_shared_data_source`, whose embedded `BlockLoadToShared` has
  // `= delete` copy and no implicit move -- see the discussion below). The
  // explicit `= delete` here is documentation, not strictly required.
  multi_source_data_source(const multi_source_data_source&)            = delete;
  multi_source_data_source(multi_source_data_source&&)                 = delete;
  multi_source_data_source& operator=(const multi_source_data_source&) = delete;
  multi_source_data_source& operator=(multi_source_data_source&&)      = delete;
```

> Note: the non-copyable / non-movable shape is not a stylistic choice — it
> falls out of the design constraint that this refactor is meant to *enable*
> `<async_to_shared, X>` keys-source configs (the headline future win). See
> the "non-copyable consequence" section below for the migration cost on
> existing agent call sites.

### 3. `set_tile_base`, `submit_load`, `gather_one` — single-arm dispatch

```cpp
  _CCCL_DEVICE _CCCL_FORCEINLINE void set_tile_base(OffsetT tile_base) {
    if (pick_source_b) { active_source.b.set_tile_base(tile_base); }
    else               { active_source.a.set_tile_base(tile_base); }
  }

  // Load handles become a tagged union -- only the active arm is initialized,
  // and complete_load reads only the active arm. Same in-register shape as
  // today's two-alives POD when both handles have the same size; smaller when
  // they don't.
  struct full_load_handle {
  private:
    union _H {
      typename SourceA::full_load_handle a;
      typename SourceB::full_load_handle b;
      _CCCL_DEVICE _H() {}
      _CCCL_DEVICE ~_H() {}
    };
  public:
    _H h;
    bool pick_b;

    template <int IPT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void complete_load(value_t (&out)[IPT]) {
      if (pick_b) { h.b.complete_load(out); }
      else        { h.a.complete_load(out); }
    }
  };
  // ... mirror partial_load_handle similarly ...

  _CCCL_DEVICE _CCCL_FORCEINLINE full_load_handle submit_load(ScratchStorage& s) {
    full_load_handle out;
    out.pick_b = pick_source_b;
    if constexpr (_scratch_storage_is_empty) {
      typename SourceA::ScratchStorage a_dummy{};
      typename SourceB::ScratchStorage b_dummy{};
      if (pick_source_b) { ::new (&out.h.b) typename SourceB::full_load_handle(active_source.b.submit_load(b_dummy)); }
      else               { ::new (&out.h.a) typename SourceA::full_load_handle(active_source.a.submit_load(a_dummy)); }
    } else {
      auto& inner = s.Alias();
      if (pick_source_b) { ::new (&out.h.b) typename SourceB::full_load_handle(active_source.b.submit_load(inner.b)); }
      else               { ::new (&out.h.a) typename SourceA::full_load_handle(active_source.a.submit_load(inner.a)); }
    }
    return out;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE value_t gather_one(int item_idx) const {
    return pick_source_b ? active_source.b.gather_one(item_idx)
                         : active_source.a.gather_one(item_idx);
  }
```

The branch on `pick_source_b` is constant within a kernel launch (set once at
ctor, never written again), so ptxas's R2UR uniformity heuristic should hoist
it. We get the same fold-the-dead-arm behaviour SASS shows for the existing
`if (pick_b) ... else ...` in `complete_load`.

## A worked example

The cleanest call site to migrate first is
`agent_batched_topk_last_filter::make_keys_source_for_segment` — it's
already a one-place factory.

**Before:**

```cpp
_CCCL_DEVICE _CCCL_FORCEINLINE keys_source_t make_keys_source_for_segment(const per_segment_state_t& s) {
  key_source_input_t  key_src_input { s.d_keys_in,  storage.keys_source_state.a };
  key_source_buffer_t key_src_buffer{ s.in_key_buf, storage.keys_source_state.b };
  return keys_source_t{key_src_input, key_src_buffer, /*pick_b=*/s.load_from_candidates_buffer};
}
```

The agent here:
- Manually drills `storage.keys_source_state.a` and `.b` (the aggregate's named members).
- Pre-builds *both* children.
- Hands them in by value to the multi-source ctor.
- The multi-source then copies both into its members; only one is ever used.

**After:**

```cpp
_CCCL_DEVICE _CCCL_FORCEINLINE keys_source_t make_keys_source_for_segment(const per_segment_state_t& s) {
  return keys_source_t{
    storage.keys_source_state,
    /*pick_b=*/s.load_from_candidates_buffer,
    /*make_a=*/[&](typename key_source_input_t::TempStorage& ts) {
      return key_source_input_t{s.d_keys_in, ts};
    },
    /*make_b=*/[&](typename key_source_buffer_t::TempStorage& ts) {
      return key_source_buffer_t{s.in_key_buf, ts};
    }};
}
```

What happens at runtime under `s.load_from_candidates_buffer == true` (the
"load from the back-buffer" case):
- The agent passes the aggregate `TempStorage` (a union) plus the `pick_b`
  flag and two factory lambdas.
- The multi_source's ctor enters the `pick_b == true` branch:
  - Grabs `storage.keys_source_state.Alias().b` (the buffer arm of the union).
  - Calls `make_b(...)` which constructs a `key_source_buffer_t{ s.in_key_buf, /*state=*/ts.b }`.
  - Placement-news the result into `active_source.b`.
  - Records `pick_source_b = true`.
- The `make_a` lambda is captured but never called. The `s.d_keys_in` access
  it would do is dead. The `ts.a` reference it would take is dead. The agent's
  `storage.keys_source_state.Alias().a` slot is **uninitialized bytes the
  whole pass** — which is fine because `a` and `b` of the union literally are
  the same bytes.

Then in the tile loop:
- `keys_source.set_tile_base(tile_base)` → `active_source.b.set_tile_base(tile_base)`.
  One register write, no dead one.
- `keys_source.submit_load(scratch)` → builds a `full_load_handle` with only
  the `b` arm alive (single placement-new), `pick_b = true`.
- `handle.complete_load(items)` → calls `h.b.complete_load(items)`. The `a`
  arm of the handle's union is uninitialized bytes; we never look at it.

If `s.load_from_candidates_buffer` flips to `false` for a different segment,
the agent recomputes per-segment state in
`agent_batched_topk_last_filter::run`, then calls
`make_keys_source_for_segment` again — *constructing a fresh
`multi_source`* over the new arm. This matches what the agent already does
today; the multi_source is rebuilt at every segment boundary in
`agent_batched_topk_last_filter::run` and at every per-tile call in
`agent_topk_filter_partition::drive_tile_loop`.

### Non-copyable consequence — why the type can't be relocated and how the agents migrate

#### Why the type ends up non-copyable / non-movable

The chain originates in `cub/block/block_load_to_shared.cuh`:

```cpp
_CCCL_DEVICE_API BlockLoadToShared(const BlockLoadToShared<...>&) = delete;
```

`BlockLoadToShared` explicitly deletes its copy ctor and declares no move
ctor. Per the standard, declaring a deleted copy ctor without an explicit
move ctor *also* suppresses the implicit move — so `BlockLoadToShared` is
neither copyable nor movable. That deletion propagates upward:

1. `async_to_shared_data_source` holds a `loader_t loader` member (=
   `BlockLoadToShared<BlockThreads>`). Its implicit copy and move ctors are
   therefore deleted too.
2. The refactored `_ActiveSourceStorage` union over `{SourceA, SourceB}`
   inherits the union-rule deletion: if any member is not trivially
   copyable, the union's implicit copy ctor is deleted (same for move).
3. The `multi_source_data_source` containing such a union picks up the
   deletion in turn.

This is the same reason today's multi-source can't *actually* instantiate
`<async_to_shared, X>` — its copy-by-value ctor parameters would fail to
compile. The refactor doesn't introduce non-copyability; it just exposes the
constraint the type already inherits from its building blocks.

For the all-trivially-copyable child case (today's `<direct, direct>`,
`<sync_block_load, direct>`, etc.) we *could* opt back in by writing custom
copy / move ctors that branch on `pick_source_b` and placement-new into the
new active arm. We choose not to: the whole point of the refactor is to make
the type compose cleanly when an async child arrives, and keeping the same
public API across "all-trivial" and "async present" cases avoids a silent
behaviour regression at the migration boundary.

#### Two agent-side patterns and their migration costs

Pattern A — **prvalue copy-elision sites**. The bulk of multi-source
construction in both agents goes through helpers like
`make_value_channel_sources` / `make_keys_source_for_segment`, written as a
single-return-site function:

```cpp
keys_source_t keys_source = make_keys_source_for_segment(state);
```

C++17's mandatory prvalue copy elision covers this case: the prvalue
returned by the helper is constructed directly into `keys_source`. No copy
ctor / move ctor is invoked, so the deleted ones never matter. Same for the
several `auto value_source = ...` locals in the per-tile bodies — they're
also prvalue initializations of fresh locals.

The migration cost is just lock-the-contract via a `static_assert` next to
the type so a future caller doesn't accidentally route through a copy /
move:

```cpp
static_assert(!::cuda::std::is_copy_constructible_v<multi_source_data_source<...>>);
static_assert(!::cuda::std::is_move_constructible_v<multi_source_data_source<...>>);
```

Pattern B — **reassignment at segment boundaries**. The single non-NRVO
site in the existing code is `agent_batched_topk_last_filter::run`:

```2570:2573:cub/cub/agent/agent_batched_topk.cuh
        partition.epilogue();
        state       = resolve_segment_state(resolve_queue_idx(tile_id));
        partition   = make_partition_for_segment(state);
        keys_source = make_keys_source_for_segment(state);
```

`keys_source = make_keys_source_for_segment(state)` is move-assignment
into an existing object — which the new type doesn't support. The selected
migration approach is **destroy-then-construct via explicit placement-new**:

```cpp
// at segment boundary in run():
partition.epilogue();
state = resolve_segment_state(resolve_queue_idx(tile_id));

// previously: partition = make_partition_for_segment(state);
//             keys_source = make_keys_source_for_segment(state);
// becomes:
partition.~partition_t();
::new (&partition) partition_t(make_partition_for_segment(state));

keys_source.~keys_source_t();
::new (&keys_source) keys_source_t(make_keys_source_for_segment(state));
```

The dtor call followed by placement-new in-place reuses the same stack /
register slot as the original local. The prvalue returned by
`make_*_for_segment(state)` is again constructed directly into that slot
under mandatory copy elision — no copy or move involved.

A small subtlety: `partition_t` itself does have a copy/move today (the
existing code uses copy-assignment on it without issue). We apply the same
destroy-then-construct treatment to both `partition` and `keys_source` at
the boundary so the segment-boundary block stays uniformly structured.
Leaving the existing `partition = ...` alone is technically fine; making
them symmetric is a readability call.

> Considered alternatives and why we ruled them out:
>
> * **`cuda::std::optional<keys_source_t>` with `.emplace(...)`.** Cleanest
>   reads but introduces a never-actually-empty representational state and
>   forces every read site through `*keys_source` / `keys_source->`. Costs
>   churn that the destroy-then-construct doesn't.
> * **Restructure `run()` so the multi-source is a per-iteration local.**
>   Smallest type-side cost but biggest agent-side rewrite. The other
>   segment-living object (`partition_t` with its `cand_reserve_open` flag)
>   does need to persist across tiles of the same segment, so the loop's
>   shape is constrained anyway.

The non-copyable refactor's biggest testable claim remains "SASS for our
`<direct, direct>` benchmarks is byte-identical to today" — the
destroy-then-construct pair lowers to the exact same writes as today's
move-assign would have, since `partition_t` and `keys_source_t` are both
trivially relocatable in that config (no real destructor work, no real
construction work beyond the field writes).

## Smem & register expectations

Configuration | TempStorage today | TempStorage after | Net smem delta
--- | --- | --- | ---
`multi_source<direct, direct>` (today's keys + values default) | `struct{empty,empty}` (1 B EBO) | `empty_storage_t` (1 B EBO) | **0** but newly detectable-as-empty across class boundaries
`multi_source<sync_block_load_direct, sync_block_load_direct>` (a low-cost alt for the keys load) | `struct{empty,empty}` | `empty_storage_t` | **0**
`multi_source<sync_block_load_vectorize, direct>` (mixed) | `struct{empty,empty}` | `empty_storage_t` | **0**
`multi_source<async_to_shared, direct>` (future keys-source variant) | `struct{barrier, empty}` ≈ sizeof barrier | `Uninitialized<union{barrier, empty}>` ≈ sizeof barrier | small (alignment fold-in)
`multi_source<async_to_shared, async_to_shared>` (both buffered + input via TMA) | `struct{barrier, barrier}` ≈ 2 × sizeof barrier | `Uninitialized<union{barrier, barrier}>` ≈ 1 × sizeof barrier | **~half** (the headline future win)

For registers / stack:
- `set_tile_base` goes from 2 unconditional writes to 1 conditional write.
  On `<direct, direct>` ptxas almost certainly already hoists/elides both,
  so SASS-identity is plausible.
- Load handle goes from `{ A::H a; B::H b; bool pick_b; }` to
  `{ union{A::H, B::H}; bool pick_b; }`. Same shape when both handles are
  pointer-sized; saves a few register slots when handles differ in width
  (e.g. async's handle carries an mbarrier token + span + loader ptr).
- The active source goes from `{ SourceA a; SourceB b; bool pick_b; }` to
  `{ union{A, B}; bool pick_b; }`. Mirror of the handle win, scaled up by
  whatever per-source state the children carry.

## Tradeoffs and open questions

1. **`pick_source_b` as a compile-time switch.** All agent call sites use a
   runtime `s.load_from_candidates_buffer` (or
   `load_from_candidates_buffer` member) flag. Keeping it runtime matches
   the user's intent. If a future agent specialises into "the buffer is
   never live" cases, it can construct the multi-source with a constant
   `false` and ptxas will fold the dead arm. No need to add a separate
   compile-time-pick template parameter.

2. **`SourceA` / `SourceB` are not default-constructible.** Today's
   `multi_source_data_source` works because its members are copy-initialized
   from caller-built sources. Placement-new is the cleanest way to keep
   "construct exactly once, in-place" semantics for non-default-constructible
   children. We'd have to add a paired in-place dtor on the multi-source,
   matched to `pick_source_b`. Same pattern the `_ActiveSourceStorage`
   union ctor/dtor would use.

3. **Non-copyable / non-movable multi_source.** Falls out of
   `BlockLoadToShared`'s `= delete` copy + no implicit move; not a
   stylistic choice. The two consequences on existing code are:
   * The factory helpers (`make_value_channel_sources`,
     `make_keys_source_for_segment`) keep working unchanged because they
     are prvalue-returning one-return-site functions that get C++17
     mandatory copy elision at the call site. A static_assert next to
     the type locks the contract.
   * The single move-assignment site at the segment boundary in
     `agent_batched_topk_last_filter::run` (line 2573) becomes
     destroy-then-construct via explicit placement-new. See the
     "Non-copyable consequence" section above for the exact pattern.

4. **What if `pick_source_b` could change?** The proposal explicitly bakes
   in "set once at ctor." If we ever need to swap arms mid-kernel (e.g. an
   adaptive policy reads from the input first then from the buffer), we
   destruct the current multi_source and construct a new one — same shape
   the agents already use across segment boundaries. We don't need to add
   `switch_arm()`-style operations.

5. **Validation plan.** Mirror the empty-storage refactor's playbook:
   - 253-record `(KeyT, ValueT)` resource snapshot on `pairs.base` and
     `keys.base` should be unchanged for the `<direct, direct>` config (the
     active config today).
   - SASS `cuobjdump` md5 should be byte-identical to the dev baseline for
     `pairs.base` + `keys.base`.
   - A new C2H test in `catch2_test_device_topk_tile_data_source.cu`
     covering: `multi_source<sync_block_load, direct>` with `pick_b=true`
     must NOT read from the input-arm's iterator (smuggle in a poisoning
     pointer for the inactive arm and verify no segfault / read).

6. **Backwards-source-ordering concern.** The handle types embed the active
   union; if a downstream consumer ever held a `partial_load_handle` past
   the multi_source's dtor, it would dangle (the active union and the
   active source share lifetime). All current consumers
   (`block_partition.cuh`, `block_filter.cuh`, agents' `drive_tile_loop`)
   complete the load within the same scope they submit it, so this is
   already an invariant we rely on — but it should be documented on the
   handle types.

7. **Should this generalize to N sources?** The current callers are all 2-
   way (`<input, buffer>`). Generalising to a variadic
   `multi_source<...>` is doable (the active union becomes a variadic
   `union`-of-tuples-of-tag) but buys nothing immediate. Out of scope.

## Migration plan

The migration mirrors the empty-storage convention's: change the type, then
sweep the agents:

1. **Land the new `multi_source_data_source` shape** (TempStorage union,
   active-source placement-new, handle tagged-union, factory-callback ctor)
   in `cub/detail/topk/tile_data_source.cuh`. Behavioural diff is gated on
   `_temp_storage_is_empty` so today's `<direct, direct>` config keeps
   landing on the empty-collapse path.

2. **Migrate the agent call sites** (search `multi_source_data_source` and
   `keys_source_state.a` / `.b` in both agents):
   - `agent_topk_filter_partition::run` (early_stop + buffered arms).
   - `agent_topk_filter_partition::make_value_channel_sources`.
   - `agent_topk_last_filter::run`.
   - `agent_topk_last_filter::make_value_channel_sources`.
   - `agent_batched_topk_filter_partition::process_tile_{early_stop,buffered,unbuffered}`.
   - `agent_batched_topk_last_filter::make_keys_source_for_segment` + the
     local `value_source` blocks inside `process_tile`.

   Each site replaces the `(a, b, pick_b)` ctor with the
   `(TempStorage&, pick_b, make_a, make_b)` ctor. The lambdas inline the
   per-side input-iterator / TempStorage slot binding the agent already
   does explicitly today.

3. **Rewrite the single segment-boundary reassignment in
   `agent_batched_topk_last_filter::run`** (the loop body at line 2573)
   as destroy-then-construct via explicit placement-new. Apply the same
   pattern to the adjacent `partition = make_partition_for_segment(state)`
   for symmetry so the boundary block stays uniformly structured.
   See the "Non-copyable consequence" section above for the exact form.

4. **Lock the contract with `static_assert`s** next to
   `multi_source_data_source` asserting `!is_copy_constructible_v` and
   `!is_move_constructible_v`. A future caller accidentally introducing a
   move-assignment site will fail at the static_assert rather than at a
   confusing template instantiation error inside the union.

5. **Update the C2H test** in
   `cub/test/catch2_test_device_topk_tile_data_source.cu` to exercise the
   factory-callback ctor and validate the active-arm-only invariant
   (poisoned inactive arm doesn't get touched). Existing tests on `direct`,
   `sync_block_load`, and the factory continue to compile unchanged.

6. **SASS-identity sanity check** on `pairs.base` + `keys.base` for
   `<direct, direct>`. If non-identical, investigate the diff against the
   `proposal_empty_storage_convention` baseline before continuing. The
   destroy-then-construct pair at the segment boundary should lower to the
   same writes as today's move-assign — both `partition_t` and
   `keys_source_t` are trivially relocatable in this config.

7. **(Optional follow-up) Wire a non-trivial keys source.** Once the
   refactor lands, an agent variant that uses `<async_to_shared, direct>`
   for the keys multi-source can be benchmarked to verify the
   smem-savings story. This is the headline future win — and the original
   reason the multi-source has to be non-copyable / non-movable.

## Summary

- **Single active source** is already an invariant agents respect at every
  call site, but the type doesn't express it. The refactor folds the
  invariant *into* the type:
  - `TempStorage` is a union (collapsed to `empty_storage_t` when both
    arms are empty).
  - The multi-source owns a single placement-new'd active child;
    constructed once in-place, destructed in-place when the multi-source
    leaves scope (or when the agent explicitly destroy-then-constructs
    over the slot at a segment boundary — see migration step 3).
  - `set_tile_base`, `submit_load`, `gather_one` all dispatch on a
    runtime tag bound at ctor.
  - Load handles are a tagged union; only the active arm is initialized.
- The agent's call sites switch from "build both children, hand both in"
  to "hand in two factories + the pick bit"; the multi-source picks the
  active factory and constructs only that child.
- The type is non-copyable / non-movable as a consequence of supporting
  `async_to_shared` children (whose embedded `BlockLoadToShared` has
  `= delete` copy and no implicit move). The two practical impacts:
  - Existing `make_*_for_segment` / `make_value_channel_sources`
    factories keep working via C++17 mandatory prvalue copy elision.
  - The one assignment site in `agent_batched_topk_last_filter::run`
    becomes destroy-then-construct via explicit placement-new.
- Behaviour on today's `<direct, direct>` config is unchanged (SASS-identity
  expected); the refactor pays off when any future keys-source variant
  introduces non-trivial per-source state (e.g. `<async_to_shared, direct>`
  — saves the inactive arm's mbarrier worth of smem).
- Migration footprint: one type change in `tile_data_source.cuh`, six
  agent call sites + one destroy-then-construct rewrite at the segment
  boundary in `agent_batched_topk_last_filter::run`, one new test case,
  static_asserts locking non-copy/non-move, one SASS-identity sanity
  check.
