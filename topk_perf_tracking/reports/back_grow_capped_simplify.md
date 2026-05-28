# `back_grow_capped_reserve_op` simplification experiment

**Branch:** `exp/topk-back-grow-simplify` (pushed to remote)
**Commit:** `2663747355 topk: store region_start in back_grow_capped_reserve_op (was back_anchor)`

## What I changed

Reframed the op's state so its math collapses from two subtracts to one add. No fields were added or removed; one field was *re-purposed*.

```diff
 template <typename OffsetT>
 struct back_grow_capped_reserve_op
 {
   static constexpr bool may_grant_less = true;

   OffsetT* counter;
-  OffsetT back_anchor;            // end of the back region (= k_total)
+  OffsetT region_start;           // start of the back region (= k_total - cap)
   OffsetT cap;

   pair<OffsetT, OffsetT> operator()(OffsetT n) const
   {
     const OffsetT prev    = atomicAdd(counter, n);
-    const OffsetT writable = (cap > prev) ? cap - prev : 0;
-    const OffsetT granted  = (n < writable) ? n : writable;
-    const OffsetT base     = back_anchor - prev - granted;   // 2 subtracts
+    const OffsetT granted = (cap > prev) ? min(n, cap - prev) : 0;
+    const OffsetT base    = region_start + prev;             // 1 add
     return {base, granted};
   }
 };
```

Callers now construct it with `{counter, k_total - num_of_kth_needed, num_of_kth_needed}` instead of `{counter, k_total, num_of_kth_needed}`. The `k_total - num_of_kth_needed` subtraction happens once on segment-resolve, not per call. Items now fill the back region in **forward claim order** rather than backward — top-k cares only about set membership, so this is a no-op at the algorithm level.

Unit tests in `tile_data_source.cu` and `block_partition.cu` are updated. The 8-test, 120-assertion `tile_data_source` Catch2 suite passes.

## What I tried first and abandoned

Initial attempt was to *drop* `back_anchor` from the op entirely and shift the caller's `cand_iter` / `cand_val_iter` by `(k_total - num_of_kth_needed)` so the op could return `{prev, granted}` directly. This **regressed `last_filter` resources hard** (+120 regs aggregate, +48 bytes spill stores, +16 bytes stack frame across 20 instantiations) because shifting the value sink pointer breaks ptxas's pointer-deduplication: `value_channel_sinks_t{d_values_out, d_values_out}` (both equal) is one stored pointer; `value_channel_sinks_t{d_values_out, d_values_out + shift}` (different) is two. The "savings" from removing `back_anchor` were dwarfed by the duplicated value pointer. I reverted that approach.

The committed version keeps the value sinks identical and changes *only* the op's internal math.

## Resource impact (apples-to-apples on `umb-b200-263`, vs dev = `e6da604d0f`)

```
                       unchanged  improved  regressed  best Δ regs  worst Δ regs
initial_histogram      46          0         0          0            0
finalize_histogram     46          0         0          0            0
filter                 46          0         0          0            0    (unaffected: uses atomic_reserve_range)
finalize_filter        46          0         0          0            0
last_filter            43          0         3          0            +5
single_cta             23          0         0          0            0
```

| `last_filter` aggregate sum delta | Δ |
|---|---:|
| Σ registers (across 46 records) | **+14** |
| Σ stack frame | **−16** |
| Σ spill stores | **−48** |
| Σ spill loads | **−48** |

Spilling **completely eliminated** on `last_filter` for `I32`-key + `I64`-value (both selects): was `stack=8, sp_st=24, sp_ld=24`, now `0/0/0`. Three small regs regressions on keys-only `I8` (+5 regs, 55→60) and `F64` (+4 regs, 32→36) — both still far from any occupancy threshold.

`filter` (which uses `atomic_reserve_range_op`, not `back_grow_capped`) is unchanged across all 46 records, as expected — confirming the resource shift is attributable to this op specifically.

## Runtime impact (`umb-b200-263`, I32 OffsetT/OutOffsetT, 2^28 elements)

Pairs, all KeyT/ValueT in {I32, I64} × {I32, I64}, sel ∈ {2^8, 2^13}, ent ∈ {1.000, 0.201, 0.000}: 24 records. **Largest absolute Δ: 6 µs on a ~580 µs workload = 0.4%.** Everything else within ±0.2%. Mean delta indistinguishable from zero.

Keys-only, KeyT in {I8, I16, I32, I64, F32, F64}, sel=2^13, ent ∈ {1.000, 0.000}: 12 records. **All within ±0.3%.**

In particular, the entropy=0 (warp-aggregated-atomic stress) workloads are flat:

| Workload | dev | simplify | Δ% |
|---|---:|---:|---:|
| I32/I32, sel=2^8, ent=0.000 | 968.0 µs | 967.7 µs | 0.0% |
| I32/I32, sel=2^13, ent=0.000 | 965.6 µs | 965.5 µs | 0.0% |
| I32/I64, sel=2^8, ent=0.000 | 967.1 µs | 966.4 µs | -0.1% |
| I64/I64, sel=2^8, ent=0.000 | 3006 µs | 3006 µs | 0.0% |

## Net assessment

- **Cleaner code.** The op's per-call math is now `granted + 1 add`, down from `2 subtracts + 1 select chain`. Easier to read and reason about; obvious that it's two ops per call (1 atomic + 1 cap check + 1 base computation).
- **Spill elimination on I32/I64.** That single workload class saw stack + spills go from non-trivial (24 bytes spilled) to zero. This is a real codegen win even though it doesn't translate to measurable runtime on the workloads I tested (probably because those kernels weren't spill-bound at the bandwidth-limited operating point we're benchmarking).
- **Three small reg regressions.** I8 keys-only +5, F64 keys-only +4. Both still well under the 64-register-per-thread budget; no occupancy impact.
- **Runtime: neutral.** Both keys-only and pairs sweeps are within measurement noise.

The change is **safe to land** as a cleanup / minor improvement. It does not deliver a perf win on its own, but it removes one source of complexity from the op and clears up spill pressure on a specific workload class.

## What it doesn't do

The user's hypothesis ("just tell it how many items to allow → drop a field → fewer registers") didn't pan out because the op was already optimally aligned for a 3-field 64-bit-aligned struct: dropping a 4-byte field on a struct with an 8-byte-aligned leading pointer just adds padding and doesn't save real storage. The math simplification turns out to be the only lever left on this op's body without changing its return contract.

## Saved artifacts

- Branch: `exp/topk-back-grow-simplify` (commit `2663747355`).
- Snapshots: `topk_perf_tracking/snapshots/back_grow_simplify_v2.json` (the kept version) and `back_grow_simplify.json` (the abandoned v1, for reference).
- Sweep JSONs on the node at `/cccl_fork/topk_perf/{simp_v2,dev,simp_keys,dev_keys}.json`.
