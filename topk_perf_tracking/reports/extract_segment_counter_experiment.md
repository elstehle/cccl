## Experiment: extract `s.segment_counter` from `per_segment_state_t` to recover warp-aggregated atomics in the narrowed build

**Branch:** `exp/topk-narrow-segcount-extract-counter`
**Commits:**
- `39390228a6` — re-apply `num_segments_val_t` narrowing on top of widened `_CCCL_GRID_CONSTANT`.
- `a95355bc65` — extract `segment_counter` out of `per_segment_state_t` in both `agent_batched_topk_filter_partition` and `agent_batched_topk_last_filter`.

**Hypothesis:** When `segment_counter` lives inside `per_segment_state_t`, ptxas's register-to-uniform-register (R2UR) pass loses warp-uniformity tracking through the struct member-access chain (`state.segment_counter -> &state.segment_counter->num_ties_written_to_back`). Lifting the pointer out as a plain top-level local in `run()` and threading it through `make_partition_for_segment` / `process_tile_*` may restore the R2UR promotion and bring back NVCC's warp-aggregated atomic lowering.

**Result: rejected.** The extraction has **zero effect** on the generated SASS for `last_filter` (and `filter`). The narrow-only build and the extract build produce byte-identical atomic instruction sequences.

### Method

Three builds compared on `umbriel-b200-073`, B200, CTK 13.1, sm_100:

1. **dev** — `e6da604d0f` — current dev (`makeWarpUniform` on `resolve_queue_idx`, `_CCCL_GRID_CONSTANT` widened to all safe kernel params). 64-bit segment-count type.
2. **narrow-only** — `39390228a6` — `dev` + narrowing of `num_segments_val_t` to 32-bit (`SegmentCountT = uint32_t` for the common case). Identical to the reverted `931ba49866` plus the GRID_CONSTANT/makeWarpUniform follow-ups.
3. **extract** — `a95355bc65` — `narrow-only` + extract `segment_counter` from `per_segment_state_t` to a top-level local in `run()`, threaded through `make_partition_for_segment`, `process_tile_early_stop`, `process_tile_buffered`, `process_tile_unbuffered`, `dispatch_tile`, and `process_partial_for_segment`.

### SASS evidence — `last_filter` I32 / I32 (select::max), three kernel instantiations per build

```
                    ATOM_total  ATOM_predicated  VOTEU.ANY
dev (no narrow)     12 / 48 / 96   12 / 48 / 96    12 / 48 / 96   <- warp-aggregation active
narrow-only          12 / 48 / 96    0 /  0 /  0     0 /  0 /  0   <- warp-aggregation lost
extract              12 / 48 / 96    0 /  0 /  0     0 /  0 /  0   <- warp-aggregation lost
```

Representative atomic line, all three kernels, narrow-only **and** extract (byte-identical hex):
```
/*0e50*/  ATOM.E.ADD.STRONG.GPU PT, R43, desc[UR8][R14.64], R45 ;  /* 0x8000002d0e2b798a */
```

Representative atomic line in dev (predicated, hex differs):
```
/*1040*/  @P0 ATOM.E.ADD.STRONG.GPU PT, R28, desc[UR10][R14.64], R28 ;  /* 0x8000001c0e1c098a */
```

The `@P` predicate and the matching `VOTEU.ANY` are the signature of NVCC's warp-aggregated-atomic lowering. In both narrow-only and extract they are entirely absent — every lane issues its own ATOM.

### Runtime evidence — I32 / I32 pairs, sel=2^8, 2^28 elements (the entropy=0.000 stress case)

| Build | Ent=1.000 | Ent=0.201 | **Ent=0.000** |
|---|---:|---:|---:|
| dev (no narrow) | 577 µs | 486 µs | **970 µs** |
| narrow-only | 581 µs | 486 µs | **1.557 ms (+60%)** |
| extract | 581 µs | 486 µs | **1.557 ms (+60%)** |

Entropy=1.000 and 0.201 are runtime-neutral across the three (the atomics are not loaded heavily). Entropy=0.000 (all-equal keys → every classify is a tie → `back_grow_capped_reserve_op` hammers `num_ties_written_to_back`) is the warp-aggregation stress test, and the extract build is byte-for-byte the same as narrow-only.

### Interpretation

The warp-aggregation loss is **not** caused by ptxas tracking `segment_counter` as a struct member vs a top-level local. The data flow into the atomic site is the same in both representations:

```
LDCU d_segment_counters                  -> UR8 (kernel param, already grid-constant)
SHFL.IDX + CREDUX.MIN of queue_idx_lane0 -> UR (already warp-uniform via makeWarpUniform)
segment_counter = LDCU + offset          -> should be UR
&segment_counter->num_ties_written_to_back -> should be UR + const offset
ATOM ..., desc[UR8][...]                  -> should fire as warp-aggregated
```

This holds whether the pointer flows via a struct member or via a function parameter. ptxas's R2UR pass is deciding to *globally disable UR-tracking on this kernel* — the trigger is something in the narrowed kernel shape (the 32-bit `queue_idx_t` loop induction type, the 32-bit `atomicAdd<unsigned int>` on `large_segments_count`, or a register-allocation-pressure heuristic that fires once the compiler picks a different layout for the narrower types). Hints applied to individual variables don't help.

### Conclusion

- The extraction is a **no-op** at the codegen level. Land it only if we like the readability of "warp-uniform values live as top-level locals" as a convention; it does not buy any compiler optimisation back.
- The narrowing of `SegmentCountT` to 32-bit and warp-aggregated atomics on the per-segment counter are, on the current ptxas / CTK 13.1 combination, **mutually exclusive** for this kernel. C++-level workarounds (`makeWarpUniform`, `_CCCL_GRID_CONSTANT`, struct-member extraction) all failed.

### Next options to try (if we want to keep pursuing this)

1. **Partial narrowing.** Keep `queue_idx_t = uint64_t` (so the loop induction type and binary-search ops stay 64-bit) but use `uint32_t` for the *stored* `num_large_segments` cached field and `large_segments_count`. The register-pressure win of the latter would still apply; the queue-iteration type would match the unbroken-aggregation case.
2. ~~**Cast `queue_idx` to 64-bit at the atomic site.**~~ **TESTED, FALSIFIED — see follow-up below.**
3. **Inline-PTX the warp-aggregated atomic.** Use `cuda::std::atomic_ref` + a manual `__ballot_sync + popc + atomicAdd(by_lane_0)` pattern at the C++ level so ptxas doesn't have to discover the optimisation.
4. **Accept the trade-off.** The narrowing's resource win (1-2 regs in `last_filter`, smaller `num_large_segments` field, smaller counter pointee fields if we narrowed those) is small compared to the runtime hit on entropy=0 workloads. Stay on 64-bit `num_segments_val_t`.

### Follow-up experiment: 64-bit cast at the atomic site

**Commit:** `d63e211e3f topk(experiment): cast queue_idx to uint64_t at the last_filter atomic site`

**Hypothesis (option 2 above):** if the 32-bit `queue_idx_t` is what blocks R2UR from tracking the atomic pointer's warp-uniformity, then forcing the atomic-site address through an explicit 64-bit arithmetic chain — `reinterpret_cast<counter_t*>(reinterpret_cast<char*>(d_segment_counters) + static_cast<uint64_t>(queue_idx) * sizeof(counter_t))` — should restore the predicated `ATOM` + `VOTEU.ANY` lowering.

**Implementation:** In `agent_batched_topk_last_filter::resolve_segment_state`, alongside the existing `segment_counter = d_segment_counters + queue_idx` (32-bit-derived, used for the non-atomic `&segment_counter->kth_key_bits` pointer that flows into the `identify_candidates_op`), also compute a parallel `segment_counter_wide` (64-bit-derived). Thread `segment_counter_wide` to `make_partition_for_segment` and use it **only** at the atomic write sites (`&segment_counter_wide->num_selected_written`, `&segment_counter_wide->num_ties_written_to_back`).

**Result: rejected.** SASS is byte-identical to the previous extract build (and to narrow-only); runtime is also identical.

```
                                ATOM_total  ATOM_predicated  VOTEU.ANY
dev (no narrow)                 12 / 48 / 96   12 / 48 / 96    12 / 48 / 96
narrow-only                      12 / 48 / 96    0 /  0 /  0     0 /  0 /  0
extract (narrow + lift)          12 / 48 / 96    0 /  0 /  0     0 /  0 /  0
cast (narrow + lift + 64b-cast)  12 / 48 / 96    0 /  0 /  0     0 /  0 /  0   <- no change
```

Runtime, I32/I32 pairs, 2^28, sel=2^8:
- Ent=1.000: 583 µs (matches all narrow/extract/cast builds)
- Ent=0.201: 485 µs (matches all)
- **Ent=0.000: 1.564 ms** (matches narrow-only/extract; +60% vs dev 970 µs)

**Interpretation:** the 32-bit `queue_idx_t` in the pointer-arithmetic chain is **not** what blocks R2UR. ptxas resolves both `d_segment_counters + queue_idx` and `(char*)d_segment_counters + uint64_t{queue_idx} * sizeof(counter_t)` to the same address-computation pattern at the SASS level. The trigger is something else — most likely:

- the kernel-wide register-allocation pressure profile that narrowing induces, or
- a ptxas heuristic that ties R2UR enable/disable to a coarser kernel-shape signal (e.g. atomic-target type width, instantiated counter struct width, loop-induction type) that none of the C++-level hints we've tried touches.

### Revised recommendation

Given that **three** C++-level interventions (`makeWarpUniform` hints, `_CCCL_GRID_CONSTANT`, struct-extraction, atomic-site 64-bit cast) all failed to restore R2UR in the narrowed build, the C++ surface is exhausted. The remaining options are:

- **(3) Manually warp-aggregate the atomic.** Implement `__ballot_sync` + `popc` + lane-0 `atomicAdd` directly inside `back_grow_capped_reserve_op` and `selected_reserve_op_t`. Bypasses ptxas's auto-aggregation entirely and is robust against future ptxas heuristic shifts.
- **(4) Accept the trade-off and stay on 64-bit `num_segments_val_t`.**

Of these, **option (3)** is the most defensible path forward if we want the narrowing's resource win. It's also a clean improvement independent of the narrowing question, since it removes one source of optimisation fragility.

Option (4) is the right call if we're not willing to take on the maintenance cost of a hand-rolled warp-aggregated atomic.
