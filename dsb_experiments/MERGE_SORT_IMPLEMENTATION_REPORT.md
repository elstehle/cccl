# CUB block/warp merge sort: implementation walkthrough, issues, and fixes

A self-contained deep dive into `cub::BlockMergeSort` / `cub::WarpMergeSort` as implemented on
NVIDIA/cccl `main` (@ `04a6df4fc7`, the base of PR #10733). Part 1 walks through the code with
excerpts so the algorithm can be followed end to end. Part 2 pinpoints the shortcomings, each
with a concrete failure trace. Part 3 presents the fixes as implemented on
`fix/block-merge-sort-partial-tile` (PR #10733, closes #5327).

All excerpts are from `cub/cub/block/block_merge_sort.cuh` unless noted.

---

# Part 1 — How the implementation works

## 1.1 One class, two collectives

Warp- and block-scope merge sort share a single implementation. `BlockMergeSortStrategy` holds
the whole algorithm; the derived classes only supply the thread index and the synchronization
primitive via CRTP:

```cpp
template <typename KeyT, typename ValueT, int NumThreads, int ItemsPerThread,
          typename SynchronizationPolicy, bool _Unroll = true>
class BlockMergeSortStrategy
{
  ...
  _CCCL_DEVICE _CCCL_FORCEINLINE void Sync() const
  {
    static_cast<const SynchronizationPolicy*>(this)->SyncImplementation();
  }
};
```

`BlockMergeSort` implements `SyncImplementation()` as `__syncthreads()`; `WarpMergeSort` as
`__syncwarp(member_mask)`. Everything below therefore applies to both scopes.

## 1.2 Data layout

The collective sorts a fixed-size **tile** of `ITEMS_PER_TILE = NumThreads × ItemsPerThread`
elements. Each thread holds `ItemsPerThread` (IPT) elements in registers in **blocked
arrangement**: thread `t` owns tile positions `[t·IPT, (t+1)·IPT)`.

The only shared state is one tile-sized scratch buffer — a union, because keys and values take
turns using it within a round:

```cpp
  union _TempStorage
  {
    KeyT keys_shared[ITEMS_PER_TILE + 1];
    ValueT items_shared[ITEMS_PER_TILE + 1];
  }; // union TempStorage
```

Note the `+ 1`: the merge deliberately prefetches one element past the end of a run (§1.5), and
this slot makes that read stay inside the allocation. Remember it — it is both load-bearing and,
in the current code, **never written** (§2.3).

## 1.3 Phase 1: thread-local sort

Each thread first sorts its own IPT registers with a stable odd-even transposition sort
(`cub/cub/thread/thread_sort.cuh`):

```cpp
  _CCCL_PRAGMA_UNROLL(Unroll ? ITEMS_PER_THREAD : 1)
  for (int i = 0; i < ITEMS_PER_THREAD; ++i)
  {
    for (int j = 1 & i; j < ITEMS_PER_THREAD - 1; j += 2)
    {
      if (compare_op(keys[j + 1], keys[j]))
      {
        swap(keys[j], keys[j + 1]);
        if constexpr (!KEYS_ONLY) { swap(items[j], items[j + 1]); }
      }
    }
  }
```

After this, the tile (viewed logically) consists of `NumThreads` sorted runs of length IPT,
laid out contiguously — the starting state for merging. Stability of this phase plus the
merge's tie rule (§1.5) is what makes `StableSort` possible.

## 1.4 Phase 2: log2(NumThreads) merge rounds

Each round doubles the sorted-run length by merging pairs of adjacent runs. Groups of
`target_merged_threads_number` threads cooperate on one merged pair:

```cpp
    for (int target_merged_threads_number = 2; target_merged_threads_number <= NumThreads;
         target_merged_threads_number *= 2)
    {
      const int merged_threads_number = target_merged_threads_number / 2;
      const int mask                  = target_merged_threads_number - 1;

      Sync();
      // store keys in shmem
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < ItemsPerThread; ++item)
      {
        int idx                       = ItemsPerThread * linear_tid + item;
        temp_storage.keys_shared[idx] = keys[item];
      }
      Sync();

      const int first_thread_idx_in_thread_group_being_merged = ~mask & linear_tid;
      const int start = ItemsPerThread * first_thread_idx_in_thread_group_being_merged;
      const int size  = ItemsPerThread * merged_threads_number;

      const int thread_idx_in_thread_group_being_merged = mask & linear_tid;
      const int diag = (::cuda::std::min) (valid_items, ItemsPerThread * thread_idx_in_thread_group_being_merged);
      ...
```

Worked example, `NumThreads = 4`, `IPT = 2` (tile = 8):

* **Round 1** (`target = 2`): groups {t0,t1} and {t2,t3}. Each group merges two runs of
  `size = 2`: group {t0,t1} covers positions `[0,4)` with run1 = `[0,2)`, run2 = `[2,4)`.
* **Round 2** (`target = 4`): one group {t0..t3} merging run1 = `[0,4)` with run2 = `[4,8)`.

Within a group, work is split by **diagonals**: thread `k` of the group is responsible for
producing merged output positions `[k·IPT, (k+1)·IPT)` of the group (its `diag = IPT·k`). Every
thread stores its full registers each round, so the whole tile is (re)written to shared memory
every round.

## 1.5 The Merge Path search and the serial merge

To find *which* elements of the two runs make up its IPT outputs, each thread runs a binary
search for the intersection of its diagonal with the "merge path" (Odeh, Green et al.):

```cpp
template <typename KeyIt1, typename KeyIt2, typename OffsetT, typename BinaryPred>
_CCCL_DEVICE _CCCL_FORCEINLINE OffsetT
MergePath(KeyIt1 keys1, KeyIt2 keys2, OffsetT keys1_count, OffsetT keys2_count, OffsetT diag, BinaryPred binary_pred)
{
  OffsetT keys1_begin = diag < keys2_count ? 0 : diag - keys2_count;
  OffsetT keys1_end   = (::cuda::std::min) (diag, keys1_count);

  while (keys1_begin < keys1_end)
  {
    const OffsetT mid = cub::MidPoint<OffsetT>(keys1_begin, keys1_end);
    const detail::it_value_t<KeyIt1> key1 = keys1[mid];
    const detail::it_value_t<KeyIt2> key2 = keys2[diag - 1 - mid];
    if (binary_pred(key2, key1)) { keys1_end = mid; } else { keys1_begin = mid + 1; }
  }
  return keys1_begin;
}
```

The return value `partition_diag` is "how many of my `diag` predecessors come from run 1"; the
rest (`diag − partition_diag`) come from run 2. The intuition: in the merge matrix of the two
runs, the merge follows a monotone staircase path; `diag` selects an anti-diagonal, and the
binary search finds where the staircase crosses it. **Precondition** (implicit):
`diag ≤ keys1_count + keys2_count` — an anti-diagonal beyond the matrix has no crossing. Keep
this in mind for §2.2.

The caller then converts the crossing into two slices and merges them sequentially:

```cpp
      const int keys1_beg_loc   = keys1_beg + partition_diag;
      const int keys2_beg_loc   = keys2_beg + diag - partition_diag;
      const int keys1_count_loc = keys1_end_loc - keys1_beg_loc;
      const int keys2_count_loc = keys2_end_loc - keys2_beg_loc;
      detail::serial_merge<_Unroll>(&temp_storage.keys_shared[0], keys1_beg_loc, keys2_beg_loc,
                                    keys1_count_loc, keys2_count_loc, keys, indices, compare_op, oob_default);
```

`serial_merge` is the hot loop — one output per iteration, and a **prefetch-style read pattern**:
after choosing an element, it immediately loads the *next* element of that run, unconditionally:

```cpp
  const int keys1_end = keys1_beg + keys1_count;
  const int keys2_end = keys2_beg + keys2_count;

  KeyT key1 = keys1_count != 0 ? keys_shared[keys1_beg] : oob_default;
  KeyT key2 = keys2_count != 0 ? keys_shared[keys2_beg] : oob_default;

  _CCCL_PRAGMA_UNROLL(Unroll ? ItemsPerThread : 1)
  for (int item = 0; item < ItemsPerThread; ++item)
  {
    const bool p  = (keys2_beg < keys2_end) && ((keys1_beg >= keys1_end) || compare_op(key2, key1));
    output[item]  = p ? key2 : key1;
    indices[item] = p ? keys2_beg++ : keys1_beg++;
    if (p) { key2 = keys_shared[keys2_beg]; }
    else   { key1 = keys_shared[keys1_beg]; }
  }
```

Three properties to note:

1. **Ties favor run 1** (`p` is false unless `compare_op(key2, key1)` strictly) — this is the
   stability rule: the left (earlier-input-position) run wins ties.
2. **The prefetch reads one past a run's end.** When the last element of run 1 is consumed,
   `keys1_beg` becomes `keys1_end` and `key1 = keys_shared[keys1_end]` is loaded even though it
   may never be used. The header documents this: *"One more item may be read but is not used."*
   For the final round's run 2, the run ends at `ITEMS_PER_TILE` — this is what the `+1` pad
   slot is for.
3. **When both runs are exhausted** (only possible if the slices sum to fewer than IPT — a
   partial-tile situation), `p` is false and the loop keeps emitting `key1` while walking
   `keys1_beg` further past the run end, up to IPT−1 extra positions. The emitted values are
   whatever those tile positions hold.

`indices[]` records where each output came from; for key-value sorts, the values are then moved
through the same (union'd) shared buffer:

```cpp
      if constexpr (!KEYS_ONLY)
      {
        Sync();
        for (int item = 0; item < ItemsPerThread; ++item)  // store values
        { temp_storage.items_shared[ItemsPerThread * linear_tid + item] = items[item]; }
        Sync();
        for (int item = 0; item < ItemsPerThread; ++item)  // gather values
        { items[item] = temp_storage.items_shared[indices[item]]; }
      }
```

## 1.6 The partial-tile path

The public API allows sorting only the first `valid_items` of the tile:

```cpp
  Sort(keys, compare_op, valid_items, oob_default);
```

with this documented contract: *"The value of `oob_default` is assigned to all elements that are
out of `valid_items` boundaries. It's expected that `oob_default` is ordered after any value in
the `valid_items` boundaries. The algorithm always sorts a fixed amount of elements, which is
equal to `ItemsPerThread * BLOCK_THREADS`."*

The implementation does two things. First, a **pre-pass** overwrites out-of-range registers —
note it starts at `item = 1` and builds a per-thread running max:

```cpp
    if constexpr (IS_LAST_TILE)
    {
      KeyT max_key = oob_default;
      _CCCL_PRAGMA_UNROLL(_Unroll ? ItemsPerThread : 1)
      for (int item = 1; item < ItemsPerThread; ++item)          // <-- item 0 never padded
      {
        if (ItemsPerThread * linear_tid + item < valid_items)
        { max_key = compare_op(max_key, keys[item]) ? keys[item] : max_key; }
        else
        { keys[item] = max_key; }
      }
    }
    // if first element of thread is in input range, stable sort items
    if (!IS_LAST_TILE || ItemsPerThread * linear_tid < valid_items)
    { detail::stable_odd_even_sort<_Unroll>(keys, items, compare_op); }
```

The running max is cleverer than it looks: it guarantees a thread's padding orders after *that
thread's own valid keys* even if the caller's `oob_default` doesn't order after anything — which
matters because the main internal caller passes an arbitrary key as `oob_default` (§1.7).

Second, every merge round **clamps all run boundaries** to `valid_items`:

```cpp
      const int diag = (::cuda::std::min) (valid_items, ItemsPerThread * thread_idx_in_thread_group_being_merged);

      const int keys1_beg = (::cuda::std::min) (valid_items, start);
      const int keys1_end = (::cuda::std::min) (valid_items, keys1_beg + size);
      const int keys2_beg = keys1_end;
      const int keys2_end = (::cuda::std::min) (valid_items, keys2_beg + size);
```

Because runs never extend past `valid_items`, out-of-range values can never be merged into the
valid region — the sorted valid prefix is correct for *any* `oob_default`. This is the actual
(weaker) guarantee the implementation delivers, and it is all the device-level user needs.

The full-tile overloads route through the same code with `IS_LAST_TILE = false` and
`valid_items = ITEMS_PER_TILE` (the clamps constant-fold away):

```cpp
  template <typename CompareOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Sort(KeyT (&keys)[ItemsPerThread], CompareOp compare_op)
  {
    ValueT items[ItemsPerThread];
    Sort<CompareOp, false>(keys, items, compare_op, ITEMS_PER_TILE, keys[0]);
  }
```

## 1.7 The in-repo callers

**`AgentBlockSort`** (`agent_merge_sort.cuh`, the block phase of `DeviceMergeSort`) loads the
last tile with a guarded `BlockLoad` and passes `keys_local[0]` — just *some* key, per thread,
possibly an uninitialized register for fully-out-of-range threads — as `oob_default`:

```cpp
      BlockMergeSortT(storage.block_merge).Sort(keys_local, items_local, compare_op, num_remaining, keys_local[0]);
```

It works because of the clamping (§1.6): the agent only ever stores back `num_remaining`
outputs, so it depends solely on the valid prefix.

**`AgentSubWarpSort`** (`agent_sub_warp_merge_sort.cuh`, DeviceSegmentedSort's small-segment
path) computes a *true* sentinel from radix traits and fills its loads with it:

```cpp
      bit_ordered_type default_key_bits = IS_DESCENDING ? traits::min_raw_binary_key(...)
                                                        : traits::max_raw_binary_key(...);
      ...
      warp_merge_sort.Sort(keys, values, BinaryOpT{}, segment_size, oob_default);
```

This caller satisfies the documented precondition; the previous one cannot (arbitrary types and
comparators have no sentinel). Hold that thought — it is the core of the fix design (§3.1).

---

# Part 2 — Where the problems are

The implementation is a **hybrid**: it half-pads (pre-pass) *and* fully clamps (rounds). Each
half covers for the other imperfectly, which creates four concrete issues.

## 2.1 The documented `oob_default` contract is not delivered

The docs say out-of-range elements are *assigned* `oob_default` — implying the output suffix
equals `oob_default`. Measured on B200 with `valid_items = N/2+3`, `oob_default = +inf`: the
suffix contains non-inf values for **every configuration with IPT ≥ 2**, warp and block scope
(harness `proto_wms_static.cu`; the `[oob-suffix=inf: 1/1/0]` column). Mechanism:

* the pre-pass never pads `keys[0]` of fully-out-of-range threads (loop starts at `item = 1`),
  so an arbitrary register value enters the shared tile at a position `≥ valid_items` on every
  round's full-tile store;
* degenerate groups (§2.2) additionally copy arbitrary tile positions around the `≥ valid_items`
  region;
* the final round's boundary thread — the one whose IPT outputs straddle `valid_items` — fills
  its tail outputs via the exhausted-runs walk (§1.5 property 3), i.e. from exactly those
  positions.

The *valid prefix* is always correct; only the documented suffix behavior is not. Severity:
functional only for callers that read the suffix; mostly a documentation-vs-implementation
mismatch.

## 2.2 Degenerate diagonals: contract-violating searches and negative counts

The per-round `diag` is a **group-relative** coordinate clamped by the **absolute**
`valid_items` — nothing ties it to what the group actually contains. For a group lying entirely
beyond `valid_items`, both runs are empty but `diag` can still be positive, violating
MergePath's precondition.

Concrete trace — `NumThreads = 4`, `IPT = 2` (tile 8), `valid_items = 3`, round 1, group
{t2,t3} (`start = 4`):

```
keys1_beg = min(3, 4) = 3     keys1_end = min(3, 3+2) = 3     -> keys1_count = 0
keys2_beg = 3                 keys2_end = min(3, 3+2) = 3     -> keys2_count = 0

t3: diag = min(3, 2*1) = 2            // 2 outputs requested from 0 available items!
MergePath(count1=0, count2=0, diag=2):
    keys1_begin = 2 - 0 = 2; keys1_end = min(2, 0) = 0        // begin > end: loop never runs
    returns 2                                                  // = partition_diag

keys1_beg_loc   = 3 + 2 = 5
keys1_count_loc = 3 - 5  = -2                                  // NEGATIVE count
serial_merge: keys1_count != 0  -> key1 = keys_shared[5]       // reads garbage at >= valid
    p = (3 < 3) && ... = false -> output[0] = keys_shared[5]   // garbage into registers
                                  output[1] = keys_shared[6]
```

Consequences: reads and value movement from arbitrary tile positions (bounded within the tile,
see §3.3, but semantically meaningless), `serial_merge` invoked with counts outside its domain,
and the garbage spreading that feeds §2.1. The dynamic `while` loop happens to "fail safe"
(returns without iterating), which is why nothing crashes — but it is precondition-violating
by construction.

## 2.3 Uninitialized reads — the #5327 initcheck failures

Three sources, all reads whose values never influence valid results:

1. **The `+1` pad slot is never written.** The final round's run 2 ends at `ITEMS_PER_TILE`, so
   the §1.5 prefetch reads `keys_shared[ITEMS_PER_TILE]` on **every sort, full tiles included**.
   The slot was allocated for this read but no code ever stores to it.
2. **Unpadded `keys[0]`** of fully-out-of-range threads (§2.1) — an uninitialized *register*
   whose value transits through shared memory.
3. **The agent's `oob_default = keys_local[0]`** — same class, at the call site.

Why this became CI-visible only recently: `compute-sanitizer --tool initcheck` tracks **global**
memory only. In shared memory all of this is invisible. But when a tuned configuration's
`TempStorage` exceeds the shared budget, cub's **vsmem** fallback places it in global memory —
and PR #5260 stopped zero-initializing temp buffers in the test harness, unmasking the reads.
That is issue #5327's DeviceMergeSort component.

## 2.4 Performance: everyone pays for boundaries, and empty rounds still run

* The clamped rounds cost every thread 4 extra `min`s per round plus a live `valid_items`
  register — yet clamping **cannot** make a round faster: rounds are barrier-separated and
  lockstep, so a round's duration is set by its fully-loaded threads while clamp-shrunk threads
  wait. The boundary handling is pure overhead on the partial-tile path.
* For a small `valid_items` (e.g. `DeviceMergeSort` on a small input = one nearly-empty tile),
  once the valid data forms a single sorted run, **every remaining round is an identity copy**
  plus two barriers — and all of them execute.

---

# Part 3 — The fixes (PR #10733)

## 3.1 Design: two overloads with genuinely different implementations

The two internal callers (§1.7) reveal two distinct caller profiles, and the fix gives each its
own optimal path instead of one hybrid:

| overload | implementation | contract | precondition | example caller |
|---|---|---|---|---|
| `Sort(..., valid_items, oob_default)` | **pad-then-sort-full**: pad all oob registers, then run the *unmodified full-tile path* | prefix sorted **and** suffix `== oob_default` | `oob_default` ordered after all valid keys, uniform | `AgentSubWarpSort` (radix-traits sentinel) |
| `Sort(..., valid_items)` — **new** | clamped rounds (+ degeneracy fix) | prefix sorted; suffix unspecified | none — arbitrary types/comparators | `AgentBlockSort` |

**Overload A** (with `oob_default`) becomes:

```cpp
    if constexpr (IS_LAST_TILE)
    {
      // Pad all out-of-range keys with oob_default. Since oob_default is ordered after every valid
      // key, sorting the padded full tile places the valid keys, sorted, in the first valid_items
      // positions, followed by copies of oob_default. Padding up front means the merge rounds need
      // no valid_items boundary handling at all.
      _CCCL_PRAGMA_UNROLL(_Unroll ? ItemsPerThread : 1)
      for (int item = 0; item < ItemsPerThread; ++item)
      {
        if (!(ItemsPerThread * linear_tid + item < valid_items)) { keys[item] = oob_default; }
      }
    }
    detail::stable_odd_even_sort<_Unroll>(keys, items, compare_op);
    MergeRounds<false>(keys, items, compare_op);
```

This fixes §2.1 (suffix `== oob_default` **by construction** — the docs become true), §2.2 (no
clamping → no degenerate searches on this path), removes the §2.4 boundary overhead, and pads
`item 0` (part of §2.3). Stability makes it robust even when a valid key compares equal to
`oob_default`: pads occupy later input positions, so the stable merge keeps valid keys first.
It is also *faster* than before — the rounds are the exact full-tile rounds.

**Overload B** (no sentinel) keeps the clamping idea — it is what makes "no sentinel" possible —
but fixes its degeneracy and leaves untouched what it doesn't need to touch:

```cpp
    // Threads holding at least one valid key pad their out-of-range registers with the running
    // max of their own valid keys (seeded from keys[0], which is valid for such threads)...
    // Threads lying entirely beyond valid_items are left untouched: their keys may hold
    // indeterminate values, which are copied around but never compared and never placed within
    // the valid prefix, because the merge rounds clamp every run to valid_items.
    if (ItemsPerThread * linear_tid < valid_items)
    {
      KeyT max_key = keys[0];                       // valid for such threads — no oob needed
      ... running-max pad loop (as before) ...
      detail::stable_odd_even_sort<_Unroll>(keys, items, compare_op);
    }
    MergeRounds<true>(keys, items, compare_op, valid_items);
```

`AgentBlockSort` switches to it, deleting the uninitialized-`oob_default` wart:

```cpp
-      BlockMergeSortT(storage.block_merge).Sort(keys_local, items_local, compare_op, num_remaining, keys_local[0]);
+      BlockMergeSortT(storage.block_merge).Sort(keys_local, items_local, compare_op, num_remaining);
```

`StableSort` gets the same pair of overloads; `WarpMergeSort` inherits everything via the
strategy class with zero changes.

## 3.2 The degeneracy fix and the identity-round early exit

Both round flavors now live in one driver, `MergeRounds<Clamped>`; the clamped instantiation
adds two things. First, the diagonal is clamped to what the group's runs actually contain,
restoring MergePath's precondition (fixes §2.2 — in the §2.2 trace, `diag` becomes 0, counts
become 0, and `serial_merge` stays in-domain):

```cpp
        // The diagonal is additionally clamped to the number of items actually present in this
        // group's two runs: a group lying entirely beyond valid_items would otherwise request a
        // diagonal larger than its (empty) runs, violating the MergePath precondition and
        // producing negative slice counts downstream.
        diag = (::cuda::std::min) (keys2_end - keys1_beg,
               (::cuda::std::min) (valid_items, ItemsPerThread * thread_idx_in_thread_group_being_merged));
```

Second, the identity rounds of §2.4 are skipped — once the input runs of a round already cover
all valid items, this and every later round would only copy:

```cpp
      if constexpr (Clamped)
      {
        // The tile's first `size` positions already form a single sorted run covering every valid
        // item: this and all remaining rounds would be identity copies. valid_items is uniform
        // across the thread block, so the break is uniform and barrier-safe.
        if (size >= valid_items) { break; }
      }
```

For a nearly-empty last tile this skips almost every round (small-input `DeviceMergeSort` is
exactly one such tile). The break is block-uniform because `valid_items` is required uniform.

## 3.3 The pad slot: written once per round, and `+1` is provably tight

The never-written pad slot (§2.3 item 1) gets one store, folded into the tile store:

```cpp
    if (linear_tid == 0)
    {
      temp_storage.keys_shared[ITEMS_PER_TILE] = keys[0];   // value never used; read must be initialized
    }
```

It must be per-round because the pairs path's items exchange aliases the same union storage.
The value is irrelevant — the write exists so the §1.5 prefetch never reads uninitialized
memory (this is the specific read initcheck flags under vsmem).

**Why `+1` suffices** (this was sharpened during review — an intermediate version of the PR
widened the pad to `+IPT` based on a loose bound): the exhausted-runs walk (§1.5 property 3) can
step up to IPT−1 positions past a run's end, but a run is only ever *truncated* by `valid_items`
within its group, and groups tile the range: `start + 2·size ≤ ITEMS_PER_TILE` with
`size ≥ ItemsPerThread`. So the walk past a truncated run at `keys1_end ≤ start + size` stays
`≤ start + size + IPT ≤ start + 2·size ≤ ITEMS_PER_TILE`. Untruncated runs are only ever read
one past the end. Maximum index read anywhere: exactly `ITEMS_PER_TILE`. The same argument shows
the value-gather indices (pre-increment positions) never exceed `ITEMS_PER_TILE − 1`, so the
items side needs no pad store at all.

## 3.4 Documentation

The `oob_default` overloads' docs now state the (delivered) guarantee and the hard
precondition: *"on output, elements beyond the `valid_items` boundary are equal to
`oob_default`. It is required that `oob_default` is ordered after any value in the
`valid_items` boundaries and that all threads provide the same `oob_default` and
`valid_items`."* The new overloads document the prefix-only contract explicitly. All new
entry points carry `versionadded:: 3.6`.

Note the one behavioral change worth a release note: a caller that passed a non-conforming
`oob_default` **and** read the suffix relied on undocumented behavior; under pad-then-sort the
documented precondition is genuinely required (such callers should move to the new no-sentinel
overload, as `AgentBlockSort` did).

## 3.5 Issue-to-fix summary and validation status

| issue | fix | where |
|---|---|---|
| §2.1 suffix ≠ `oob_default` despite docs | pad-then-sort-full: contract holds by construction | overload A |
| §2.2 degenerate searches, negative counts | `diag` clamped to run contents | `MergeRounds<true>` |
| §2.3 pad slot never written | one store per round from thread 0 | `StoreKeys` |
| §2.3 unpadded `keys[0]` / agent's uninit `oob_default` | A pads item 0; B leaves oob threads untouched & never compares them; agent uses B | overloads A/B, `AgentBlockSort` |
| §2.4 boundary overhead on every thread/round | A: no boundary logic at all; B: unavoidable (that's its job) | overload A |
| §2.4 identity rounds executed | uniform early exit once `size ≥ valid_items` | `MergeRounds<true>` |

Validation: a dual-build harness (`proto_merge_fix.cu`) compares baseline (`main` headers) vs
fixed (branch headers) on B200 — correctness at `valid ∈ {0, 1, N/2+3, N−1, N}` (prefix, and
suffix for A), stability vs `std::stable_sort` under heavy ties, `DeviceMergeSort` end-to-end
with a partial last tile, and latency slopes (full + partial) to confirm overload A got faster.
Expected: baseline reproduces the §2.1 suffix failures; fixed build reports zero. Alongside it:
cub's catch2 suites and the #5327 initcheck vsmem repro in CI.
