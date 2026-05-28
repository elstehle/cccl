# Audit: user-defined ctor/dtor on `TempStorage` / `ScratchStorage` types

Searched all topk-related headers
(`cub/cub/{agent,detail/topk,device/dispatch,block,device}/**/*topk*`) for
`_CCCL_HOST_DEVICE` / `_CCCL_DEVICE` user-declared default ctors and
dtors on storage-shaped types. Five sites total: three could be dropped
without effect, two must stay.

## Dropped (commit `ea014f8ec3`)

| File | Type | Reason it's now redundant |
|------|------|---------------------------|
| `cub/cub/agent/agent_topk.cuh` | `union arms_t` inside `agent_topk_filter_partition::_TempStorage` | All variant members (`buffered_t`, `prefix_sum`, `early_stop_t`) are now built from trivially-default-constructible / -destructible pieces only: `OffsetT[]`, `key_source_*_t::TempStorage` (either `empty_storage_t` or `{ Uninitialized<...> barrier }`), and `*_storage_layout_t` (the prior `partition_storage_layout` refactor pushed all non-trivial inner unions inside `Uninitialized<>`). Implicit default ctor / dtor is now trivial. |
| `cub/cub/agent/agent_batched_topk.cuh` | `union arms_t` inside `agent_batched_topk_filter_partition::_TempStorage` | Same shape and same reasoning as the single-problem `arms_t`. |
| `cub/cub/device/dispatch/dispatch_topk.cuh` | `union all_modes_ts_t` (per-kernel smem alias of `agent_fp_t::TempStorage` and `agent_ub_t::TempStorage`) | Both variant arms are `Uninitialized<...>` and therefore trivial. The agents' `TempStorage` deliberately derives publicly from `Uninitialized<>` (e.g. `struct TempStorage : Uninitialized<_TempStorage> {}`) so the union here is a union-of-trivials. |

**Verification:**

- All 19 topk test binaries (sync + async tunings) still pass on
  `umb-b200-236` after the cleanup — same count as before
  (~245k assertions across ~370 cases).
- SASS byte-identical on representative binaries
  (`cub.test.device.topk_pairs.lid_0`,
  `cub.test.device.segmented_topk_pairs.lid_2.types_1`):
  `27a0d0736ce0fb06f805492e49db832b` and
  `ab43660bad91820654bd82253758c554` match exactly with and without the
  empty user-defined ctor/dtor. The user-declared `{}` ctor/dtor was
  semantically a no-op on these unions and the compiler treats it that
  way too — the cleanup is pure code clarity, zero codegen impact.

## Kept

| File | Type | Why |
|------|------|-----|
| `cub/cub/detail/topk/tile_data_source.cuh` ~line 587 | `union _H` inside `multi_source_data_source::full_load_handle` | Holds `SourceA::full_load_handle` / `SourceB::full_load_handle` in a tagged-union shape. When the configured `tile_load_kind` is `block_load_to_shared_async`, an arm type carries `BlockLoadToShared::CommitToken`, which is move-only with a *private* default ctor and *deleted* copy ctor. The union's implicit default ctor would be deleted (no accessible default for either arm) and the implicit dtor would be non-trivial / deleted depending on the configuration. The handle code expects to placement-new exactly one active arm via the tagged `from_a_t` / `from_b_t` ctors, so the union needs a no-op user-declared default ctor + dtor to remain instantiable; the active arm's dtor is then called explicitly from the outer `~full_load_handle()`. |
| `cub/cub/detail/topk/tile_data_source.cuh` ~line 659 | `union _H` inside `multi_source_data_source::partial_load_handle` | Same reasoning as `full_load_handle` -- partial-load handles wrap the same `SourceA/B::partial_load_handle` types, which carry the same `CommitToken` payload in the async configuration. |

These two sites are essential whenever the async tile-load kind is
selected. With non-async configurations they collapse to empty
no-op specials and have no effect.

## Out-of-scope (looked at, not topk)

`cub/cub/device/dispatch/kernels/kernel_transform.cuh:963`
(`~kernel_arg() noexcept {}`) and
`cub/cub/warp/specializations/warp_exchange_shfl.cuh:217`
(`~CompileTimeArray() {}`) are unrelated to the topk storage stack and
are presumably justified by their own surrounding patterns; this audit
didn't touch them.

## Files

- Cleanup commit: `ea014f8ec3` ("topk(cleanup): drop redundant union
  ctor/dtor on trivial-member unions"). Net diff: -7 lines / +4 lines
  (the +4 is the rewritten comment in
  `partition_storage_layout.cuh` that referenced
  `multi_source_data_source::ScratchStorage` as carrying ctor/dtor,
  which it no longer does -- that storage was wrapped in
  `Uninitialized<>` and made trivial in `f562ac6fe0`).
