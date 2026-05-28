# `multi_source_data_source` by-reference variant: SASS-identical to dev

Third design iteration on top of the proposal in
`proposal_multi_source_active_source_refactor.md` and the two earlier
variants documented in `multi_source_active_source_refactor_results.md`.

## What changed vs the proposal

The simplification target: keep both child sources alive (same operational
shape as today's dev code) but make the multi-source compose with future
non-copyable / non-movable children (notably `async_to_shared_data_source`
via `BlockLoadToShared`). The minimal way to do that is to change the
ctor from **by-value** to **by-reference** and explicitly delete copy / move
on the multi-source itself.

### Multi-source diff vs dev

```diff
-  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE multi_source_data_source(SourceA a, SourceB b, bool pick_b)
+  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE multi_source_data_source(SourceA& a, SourceB& b, bool pick_b)
       : source_a(a)
       , source_b(b)
       , pick_source_b(pick_b)
   {}
+
+  multi_source_data_source(const multi_source_data_source&)            = delete;
+  multi_source_data_source(multi_source_data_source&&)                 = delete;
+  multi_source_data_source& operator=(const multi_source_data_source&) = delete;
+  multi_source_data_source& operator=(multi_source_data_source&&)      = delete;

   ...

 private:
-  SourceA source_a;
-  SourceB source_b;
+  SourceA& source_a;
+  SourceB& source_b;
   bool pick_source_b;
```

Everything else (TempStorage as `struct {a, b}`, ScratchStorage union,
tagged-union load handles, `set_tile_base` delegated to both arms,
`submit_load` dispatched on `pick_source_b`, `gather_one` ternary) is
unchanged.

### Agent diff

- `agent_batched_topk_filter_partition::process_tile_{early_stop,
  buffered, unbuffered}`: the value-source IIFE lambda now returns a
  prvalue (no NRVO via named local — that's what would break with a
  non-movable type). The lambda's children are *lifted to the enclosing
  scope* so the multi-source's references stay valid after the IIFE
  returns. Mandatory copy elision (C++17 §[class.copy.elision]) places
  the prvalue directly into `auto value_source`.

- `agent_batched_topk_last_filter`: same lambda fix in `process_tile`,
  plus the `make_keys_source_for_segment` helper (which returned by
  value and would dangle with by-ref ctor) is removed. The keys-source
  trio (`key_src_input`, `key_src_buffer`, `keys_source`) is constructed
  in-line in `run()`. The segment-boundary refresh changes from
  `keys_source = make_keys_source_for_segment(state);` to explicit
  destroy-then-construct of all three locals in place:

  ```cpp
  keys_source.~keys_source_t();
  key_src_input.~key_source_input_t();
  ::new (&key_src_input) key_source_input_t{state.d_keys_in, storage.keys_source_state.a};
  key_src_buffer.~key_source_buffer_t();
  ::new (&key_src_buffer) key_source_buffer_t{state.in_key_buf, storage.keys_source_state.b};
  ::new (&keys_source) keys_source_t{key_src_input, key_src_buffer, /*pick_b=*/state.load_from_candidates_buffer};
  ```

  For trivial children (today's `<direct, direct>` / `<sync_block_load,
  direct>`) the destructors / constructors lower to the same writes
  today's move-assign would have, and the multi-source's ref members
  bind to the same `key_src_input` / `key_src_buffer` addresses — just
  with re-initialized contents.

- `agent_topk_filter_partition::drive_tile_loop` and
  `agent_topk_last_filter::run`: `make_value_channel_sources` helper
  removed; the children are lifted to the helper / `run()`'s outer
  scope and the per-tile multi-source is constructed via an IIFE that
  returns a prvalue (mandatory copy elision into the outer
  `value_source` local).

### Lifetime story for future async support

The multi-source is non-copyable / non-movable, so the
`BlockLoadToShared`-via-`async_to_shared_data_source` chain (`= delete`
copy, no implicit move) propagates cleanly. The async child object
itself will live in the same scope as the multi-source — typically the
per-tile body for `filter_partition` agents and the per-segment scope
for `last_filter` (which is exactly where the destroy-then-construct
pattern already refreshes the trio at every segment boundary).

Concretely, an `<async_to_shared, direct>` keys-source would slot in
without changing the agent code at all — the agent's
`storage.keys_source_state.a` smem already exists and would hold the
async source's `loader_t::TempStorage` (carrying the mbarrier); the
async child binds its `loader(state.barrier)` reference there as it
already does in `async_to_shared_data_source`'s ctor. The multi-source
just stores a `SourceA&` to that async child and delegates as before.

## Verification — SASS-identical to dev

```
                          BAR.SYNC count        binary md5                         cuobjdump SASS md5
                          --------------        --------------                     -----------------------------
  pairs.base
    dev (e77b23ca23)    : 2196                  ...                                a61af970e6c69a5c46e323b1fc929cb9
    by_ref (5a31ba9b9e) : 2196                  3181b78f89e0499e0348daee706ae2e4   a61af970e6c69a5c46e323b1fc929cb9

  keys.base
    dev                 : 1243                  ...                                0a1c60ede9b3254e1f716ff2e760c775
    by_ref              : 1243                  14a53b98f70e7313f7abbcabf4d7e5df   0a1c60ede9b3254e1f716ff2e760c775
```

The two binary md5s differ (host-side ELF section diffs / debug
section / etc.), but the **disassembled SASS md5 matches byte-for-byte
both binaries** -- every device kernel emits identical instructions.
The runtime is therefore identical by construction.

## Resource snapshot — 138/138 kernels at zero delta

```
Register delta histogram (dev vs by_ref):
  delta (regs) | count
  -------------|------
       0       |  138
```

No register changes. No stack frame changes. No spill stores or loads.
No smem changes. Across every `(logical_kernel, KeyT, ValueT)` triple
covered by `pairs.base + keys.base`. (Full table:
`topk_perf_tracking/reports/full_resource_table_by_ref.md`.)

## Runtime spot check

`K=I64, V=I32, OffsetT=I32, Elements=2^28, SelectedElements=8,
Entropy=0.000` — the workload that the full active-source refactor
regressed to **1.065x** (3.18 ms vs dev's 2.99 ms):

```
=== Run 1 === GPU 3.003 ms (noise 0.11%)
=== Run 2 === GPU 3.004 ms (noise 0.12%)
=== Run 3 === GPU 3.003 ms (noise 0.12%)
```

Identical to dev within run-to-run noise (3.00 ms ± 0.01 ms), as
expected from byte-identical SASS.

## Comparison vs the earlier variants

| variant | what changed | SASS | resources | I64-pair runtime | I16-K-only runtime |
|---|---|---:|---:|---:|---:|
| dev (`e77b23ca23`) | baseline | -- | -- | 1.000x | 1.000x |
| active-source (`f90e824e98`) | placement-new active arm + tagged-union handles + factory ctor + non-copyable | byte-different | +12 regs worst (last_filter I64) | **1.012x mean, 1.065x worst** | 1.006x mean, 1.052x worst |
| factory-only (`dc13e0a249`) | factory-callback ctor only, both arms alive | byte-different | +8 regs (filter I16 K-only), +10 (finalize_filter I8 K-only); some -6 wins on finalize_filter I8 pairs | 1.003x mean, 1.019x worst | **1.008x mean, 1.054x worst** |
| **by-ref (`5a31ba9b9e`)** | by-ref ctor + non-copyable, both arms alive, agent-owned children | **byte-identical to dev** | **0 changes** | **1.000x** | **1.000x** |

The by-ref variant captures the proposal's durable contract — the type
composes with non-copyable / non-movable children for future
`async_to_shared` work — at zero immediate codegen cost.

## Recommendation

Land the by-ref variant. It:
- Preserves dev's resource / runtime baseline exactly (byte-identical SASS).
- Adds the non-copyable / non-movable contract that future async-shared
  keys sources need.
- Costs only the agent-side restructuring: removal of the `make_value_
  channel_sources` / `make_keys_source_for_segment` helpers in favour
  of in-line construction at the call site, plus one
  destroy-then-construct block at the single segment-boundary refresh
  in `agent_batched_topk_last_filter::run`. The lambda-IIFE pattern at
  the per-tile value-source sites is preserved (now returning a
  prvalue, with children lifted to the enclosing scope).
- Leaves the smem-saving "single active source" placement-new shape as
  an opt-in future refactor for the case where two async-children
  configurations actually arrive in tree.

## Branches and snapshots

- Branch: `exp/topk-multi-source-by-ref` (`5a31ba9b9e`).
- Snapshot: `topk_perf_tracking/snapshots/by_ref.json` (253 records).
- Full resource table: `topk_perf_tracking/reports/full_resource_table_by_ref.md`.
