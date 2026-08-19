# BlockMergeSort partial-tile sorting: uninitialized reads, the real contract, and proposed fixes

A self-contained explainer — no prior knowledge of the implementation assumed. Written after the
behavior surfaced twice independently: as a failing (too-strict) correctness expectation in the
`MERGE_SORT_SEARCH_STATIC` benchmark harness (`proto_wms_static.cu`), and as the plausible root
cause of the `DeviceMergeSort` part of [NVIDIA/cccl#5327](https://github.com/NVIDIA/cccl/issues/5327)
(`compute-sanitizer --tool initcheck` failures in the vsmem tests after
[#5260](https://github.com/NVIDIA/cccl/pull/5260)).

## 1. Background: how the block/warp merge sort works

`cub::BlockMergeSort` (and `cub::WarpMergeSort`, which shares all of this code via
`BlockMergeSortStrategy` in `cub/block/block_merge_sort.cuh`) sorts a fixed-size **tile** of
`N = NumThreads × ItemsPerThread` elements. Each thread holds `ItemsPerThread` (IPT) consecutive
elements in registers ("blocked arrangement": thread `t` owns tile positions
`[t·IPT, (t+1)·IPT)`).

The algorithm:

1. **Thread-local sort**: each thread sorts its own IPT registers (odd-even transposition sort).
2. **log2(NumThreads) merge rounds**. Round `r` merges pairs of sorted runs produced by round
   `r−1`. Each round:
   - every thread **stores its full IPT registers** to a shared-memory tile
     (`keys_shared[N + 1]` — note the `+ 1`, it matters below);
   - each thread runs a **MergePath diagonal search** (a binary search) to find which slice of
     the two runs it is responsible for merging;
   - each thread runs **SerialMerge**: sequentially merges its slice back into its IPT registers.

After the last round, thread `t`'s registers hold sorted tile positions `[t·IPT, (t+1)·IPT)`.

### The partial-tile API

Callers can sort fewer than `N` elements:

```cpp
Sort(keys, compare_op, valid_items, oob_default);
```

The documentation promises: *"The value of `oob_default` is assigned to **all elements** that are
out of the `valid_items` boundaries. It's expected that `oob_default` is ordered after any value
in the `valid_items` boundaries."* So with `oob_default = +inf` you would expect the output to be
the `valid_items` sorted keys followed by `+inf` everywhere else.

This promise is **not** kept (§3), and the code additionally performs reads of memory nobody ever
wrote (§4). Both symptoms come from the same three code paths.

## 2. The three code paths that touch never-written data

### (a) The `+1` pad slot — read on *every* sort, even full tiles

`SerialMerge` is written so that after consuming the last element of a run it *pre-loads* the next
element unconditionally — the header even documents it: *"One more item may be read but is not
used."* For the final run of the final round, "one past the end" is `keys_shared[N]` — the
`+ 1` pad slot in `KeyT keys_shared[N + 1]`. That slot exists precisely to make this read legal,
**but nothing ever writes it**. So every sort performs a read of an uninitialized shared-memory
word (whose value provably never influences the output).

### (b) The out-of-bounds pre-pass skips `keys[0]`

Before sorting a partial tile, a pre-pass overwrites each thread's out-of-range registers with
`oob_default`:

```cpp
KeyT max_key = oob_default;
for (int item = 1; item < ItemsPerThread; ++item)   // <-- starts at 1
{
  if (ItemsPerThread * linear_tid + item < valid_items)
    max_key = ...;           // track max of valid keys
  else
    keys[item] = max_key;    // overwrite out-of-range slots
}
```

The loop starts at `item = 1`. For a thread whose *entire* range is out of bounds, `keys[0]`
keeps whatever the register happened to contain. Since every merge round stores the **full**
tile to shared memory, that unspecified value lands in the shared tile at a position
`≥ valid_items`, once per fully-out-of-range thread.

### (c) Clamped middle rounds run "impossible" searches

Merge rounds clamp all run boundaries by `valid_items`, but each thread's search diagonal is
computed as `diag = min(valid_items, IPT · thread_idx_in_group)` — clamped by the **absolute**
`valid_items` while being a **group-relative** coordinate. For groups that lie entirely beyond
`valid_items`, both runs are empty (`keys1_count = keys2_count = 0`) yet `diag` can still be
positive — violating the MergePath precondition `diag ≤ keys1_count + keys2_count`. The search
then returns `diag`, downstream slice arithmetic produces a **negative count**, and `SerialMerge`
happily reads `keys_shared[...]` at positions beyond `valid_items` — picking up the values from
(b), or values a previous degenerate round wrote there — and stores them back as that thread's
output. Net effect: the tile region beyond `valid_items` fills with arbitrary (but bounded to
the tile) values rather than `oob_default`, round by round.

None of this corrupts the *valid* region: the final round's runs are clamped to `[0, valid_items)`,
so every valid output position is merged only from valid inputs. But the *boundary* thread of the
final round — the one whose IPT outputs straddle `valid_items` — fills its tail outputs from
one-past-the-run reads (path (a) mechanics) at positions `≥ valid_items`, i.e. from the garbage of
paths (b)/(c) rather than from `oob_default`.

## 3. Symptom 1 — the functional contract gap (measured)

With `valid_items = N/2 + 3` and `oob_default = +inf`, output positions `≥ valid_items` are *not*
all `+inf` for **every configuration with IPT ≥ 2** (warp scope 64..384, block scope 512..2048;
`proto_wms_static.cu`, `[oob-suffix=inf: …0]` column). The sorted valid prefix is always correct.

So the *de facto* contract is weaker than documented:

> The first `valid_items` output positions contain the sorted valid keys. Output positions beyond
> `valid_items` are unspecified.

This is exactly (and only) what `DeviceMergeSort` relies on — its block-sort agent even passes
`keys_local[0]` as `oob_default`, which doesn't satisfy "ordered after any valid value" at all,
and only ever writes back `num_remaining` outputs.

## 4. Symptom 2 — the initcheck failures in cccl#5327

Three facts chain together:

1. **`initcheck` only tracks global memory.** Reads of uninitialized *shared* memory are invisible
   to it. On the normal path, everything in §2 happens in shared memory — silent.
2. **vsmem puts the tile in global memory.** When a configuration's `TempStorage` exceeds the
   shared-memory budget, cub's *virtual shared memory* fallback allocates it in global memory.
   Now the very same reads touch uninitialized **global** memory.
3. **PR #5260 stopped zero-initializing temp buffers in the test helpers.** Before it, the test
   harness's defensive initialization masked everything; after it, the buffers are genuinely
   uninitialized and `initcheck` starts reporting — which is when #5327 was filed, specifically
   for the **vsmem** DeviceMergeSort tests.

Additionally, path (b) has a register-level variant in the device agent: for threads entirely
beyond `num_remaining`, `keys_local[0]` is never loaded (guarded `BlockLoad`), and this
uninitialized register is passed as `oob_default` — so on the last tile even the slots the
pre-pass *does* fill are filled with uninitialized-derived data.

Everything is benign in the sense that no valid output ever depends on the uninitialized values —
they are either never used, or only steer comparisons whose outcome cannot displace a valid key.
But the reads are real, and the sanitizer is right to flag them.

(#5327 also names `DeviceReduceByKey` / `DeviceScanByKey`; those are different agents with,
presumably, their own by-design reads of unwritten temp storage unmasked the same way — not
covered by this analysis.)

## 5. Recommended design: two overloads, mirroring WarpBitonicSort

The doc sentence *"the algorithm always sorts a fixed amount of elements"* describes a
**pad-then-sort** design; the implementation instead half-pads *and* clamps every merge round by
`valid_items` — a hybrid that pays both costs and delivers neither contract cleanly. The clean
resolution is two overloads with genuinely different implementations (exactly the split
`WarpBitonicSort` already ships):

**A. `Sort(keys, op, valid_items, oob_default)` — pad-then-sort-full.** Pre-pass assigns
`oob_default` to *all* out-of-range registers (including item 0), then runs the completely
unmodified full-tile path: no `valid_items` in any round, no clamping, no degenerate searches.
Clamping never helped the critical path anyway — rounds are barrier-separated and lockstep, so
round time is set by the fully-loaded threads while the clamp-shrunk threads wait; the clamps are
pure overhead (~4 `min`s/thread/round + a live register). With this implementation the documented
contract ("suffix = `oob_default`") holds **by construction**, and stability makes it robust even
when valid keys compare equal to `oob_default` (pads occupy later input positions). Precondition
(unchanged from the docs): `oob_default` ordered after every valid key.

**B. `Sort(keys, op, valid_items)` — clamped, suffix unspecified.** Keeps the current clamped
implementation (plus fix (3) below) for callers with arbitrary types/comparators that have no
max sentinel — merge sort's core audience. Contract: sorted valid prefix; keys beyond
`valid_items` may be overwritten (WarpBitonicSort's wording). This is the overload
`AgentBlockSort` should call: it has no sentinel (today it passes `keys_local[0]`, possibly an
uninitialized register, as `oob_default`) and only consumes the valid prefix — switching it
deletes the agent-level uninitialized read as a side effect.

Migration note: pad-then-sort is a behavioral change for existing 4-arg callers that pass a
non-conforming `oob_default` *and* read the suffix — the clamped code protected the prefix for
any `oob_default`; overload A genuinely requires the (always-documented) precondition.

## 5b. Point fixes (needed regardless of the redesign)

Ordered smallest-first; (1)+(2) silence the sanitizer, (3) makes overload B's degenerate
searches well-defined, (4) is the documentation alternative if the redesign is not taken.

1. **Initialize the pad slot** — kills path (a), the always-on read. One predicated store next to
   the existing tile store, per round (and the same for `items_shared` in the key-value path):

   ```cpp
   // store keys in shmem (existing loop) ...
   if (linear_tid == 0)
   {
     temp_storage.keys_shared[ITEMS_PER_TILE] = keys[0];  // any deterministic value works;
   }                                                      // it is read but never used
   ```

2. **Never leak uninitialized registers into the tile** — kills path (b) and the agent variant:
   - in the pre-pass, also overwrite `keys[0]` for fully-out-of-range threads
     (`if (ItemsPerThread * linear_tid >= valid_items) keys[0] = oob_default;`), and
   - in `AgentBlockSort`, load the last tile with `BlockLoad`'s fill overload (or default-init
     `keys_local`) instead of passing possibly-uninitialized `keys_local[0]` as `oob_default`.

3. **Clamp the diagonal to its precondition** — kills path (c) and, together with (1)+(2), makes
   the documented "all out-of-range outputs become `oob_default`" contract actually hold:

   ```cpp
   const int diag = (::cuda::std::min) (keys1_count + keys2_count,
                    (::cuda::std::min) (valid_items, ItemsPerThread * thread_idx_in_thread_group_being_merged));
   ```

   With `diag` in-contract, the search result is well-defined, the slice counts are non-negative,
   and `SerialMerge` pads exhausted slices with `oob_default` (it already does — it just never got
   the chance in degenerate groups). Cost: one extra `min` per thread per round.

4. **Or: fix the documentation instead.** If out-of-range outputs are considered not worth paying
   for, change the doc to the de facto contract: *"the first `valid_items` positions contain the
   sorted valid keys; positions beyond `valid_items` are unspecified"* — and keep (1)+(2), which
   are needed for the sanitizer regardless. (Note (3) is still attractive: it is the only fix for
   the negative-count `SerialMerge` calls, which are UB-adjacent even if harmless today.)

### Verification

- Re-run the `proto_wms_static.cu` correctness mode: with (1)+(2)+(3) the informational
  `oob-suffix=inf` column should turn all-1 with unchanged valid-prefix results and unchanged
  dynamic/static byte-equality.
- Run the vsmem `DeviceMergeSort` tests under `compute-sanitizer --tool initcheck` (the #5327
  repro): the merge-sort reports should disappear with (1)+(2) alone.
- Perf sanity: all three fixes add O(1) work per round; re-run the latency mode to confirm the
  deltas stay within noise.

## 6. Relation to the MERGE_SORT_SEARCH_STATIC work

The statically-unrolled search (`WMS_STATIC_SWITCH_RESULTS.md`) always executes its fixed number
of probe steps, so — unlike the dynamic `while` loop, which simply doesn't iterate on degenerate
searches — it must (and does) clamp its probe indices into the padded tile. Fix (1) initializes
the pad slot those clamped probes can read, and fix (3) removes the degenerate searches
altogether; landing them together with the switch keeps the sanitizer clean for both search
algorithms.
