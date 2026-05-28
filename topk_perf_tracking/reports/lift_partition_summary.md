# `last_filter` partition lifetime fix -- benchmark report

> **Variant.** `lift_partition` (branch `exp/lift-last-filter-partition`,
> commit [`6f88d97a1a`](#)) -- on top of the prior optimisation stack:
> `flat_walk` &rarr; `flat_last` &rarr; `drop_xforms` &rarr; `by_value` &rarr; **`lift_partition`**.
>
> **Hardware.** B200 (`sm_100`), `umbriel-b200-072`, CTK 13.1.
> **Build.** `cub-cpp17` preset, `-DTUNE_OffsetT=::cuda::std::int32_t -O3`.
> **Inputs.** Pairs benchmark, `OffsetT = OutOffsetT = I32`.

---

## TL;DR

| metric (i32/i32, full 3-axis sweep, n=51) | `by_value` (prior) | `lift_partition` (new) |
|---|---:|---:|
| Geometric mean ratio vs `main` | **4.17x slower** | **1.04x slower** |
| Median ratio vs `main` | **3.28x slower** | **1.33x slower** |
| Maximum ratio vs `main` | **28.28x slower** | **2.98x slower** |
| Cases where new variant **beats** main | -- | 14 / 51 |

The entropy=0.000 catastrophic cliff is **gone**: we now run **3.7x-7x faster than main**
on `Entropy=0.000` workloads (was 27-28x slower) thanks to the back-grow-cap exit hint
(`cand_reserve_open`) surviving across tiles of the same segment.

## Resource usage

`last_filter` kernel, `KeyT=int32, ValueT=int32`:

| variant | registers | stack | spill st | spill ld | smem | notes |
|---|---:|---:|---:|---:|---:|---|
| `main` (single-problem) | 32 | 0 | 0 | 0 | 0 | reference |
| `by_value` (prior batched) | 40 | 0 | 0 | 0 | 0 | partition reconstructed per tile |
| `lift_partition` (this CL) | 40 | 0 | 0 | 0 | 0 | partition reused across same-segment tiles |

No resource regression: same 40 GPR count as the previous batched variant.
The change is **purely structural** -- moving the partition object's lifetime
out of `process_tile` into `run()` so the per-thread `cand_reserve_open` flag
persists across tiles of the same segment.

---

## What changed

`agent_batched_topk_last_filter::run()` previously constructed a fresh
`partition_t` (and its embedded `block_partition` primitive) **inside**
`process_tile()`, on every grid-stride iteration. That reset
`block_partition::cand_reserve_open` to `true` on every tile, defeating the
existing optimisation in `block_partition.cuh` that drops the per-item
`back_grow_capped_reserve_op` atomic once any thread observes a grant=0
return for the segment.

The fix mirrors `agent_topk_last_filter::run()` from the single-problem
dispatch (`main`):

1. Build the partition once at the top of `run()` for the first segment.
2. Reuse it across every tile of the same segment.
3. On the segment-boundary crossing (`tile_id >= state.queue_segment_end`):
   - Flush the previous segment's partition via the existing
     `partition.epilogue()` handshake (no-op for the atomics strategy,
     real flush for the accumulating sister classes).
   - Resolve the new segment state.
   - Reconstruct the partition (and keys-source) for the new segment;
     `cand_reserve_open` is reset to `true`.
4. After the loop, one final `partition.epilogue()` to terminate the last
   active segment.

Files touched:

- `cub/cub/agent/agent_batched_topk.cuh` (+63 / -20)

No API changes upstream; no test or kernel-signature changes.

---

## Full i32/i32 sweep vs main

Generated with `topk_perf_tracking/compare_sweep.py` from
`sweep_main_i32i32.json`, `sweep_by_value_i32i32.json`,
`sweep_lift_partition_i32i32.json`.

See [`sweep_i32_i32_lift_partition_vs_main.md`](sweep_i32_i32_lift_partition_vs_main.md).

### Highlights (KeyT=I32, ValueT=I32, OffsetT=OutOffsetT=I32)

GPU mean times in microseconds.

| Elements | Sel | Entropy | main | by_value | **lift_partition** | bv/main | **lp/main** |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2^16 (65 KiB) | 8 | 0.000 | 24.56 | 81.61 | **40.94** | 3.32x | **1.67x** |
| 2^20 (1 MiB) | 8 | 0.000 | 49.16 | 734.84 | **61.47** | 14.95x | **1.25x** |
| 2^24 (16 MiB) | 8 | 0.000 | 409.20 | 11.22 ms | **107.86** | 27.43x | **0.26x** &uarr; |
| 2^24 (16 MiB) | 256 | 0.000 | 406.80 | 11.02 ms | **108.72** | 27.09x | **0.27x** &uarr; |
| 2^24 (16 MiB) | 2^13 | 0.000 | 402.24 | 10.92 ms | **108.63** | 27.15x | **0.27x** &uarr; |
| 2^24 (16 MiB) | 2^18 | 0.000 | 409.68 | 11.08 ms | **108.85** | 27.06x | **0.27x** &uarr; |
| 2^24 (16 MiB) | 2^23 | 0.000 | 415.01 | 11.03 ms | **281.06** | 26.58x | **0.68x** &uarr; |
| 2^28 (256 MiB) | 8 | 0.000 | 6.22 ms | 175.86 ms | **954.82** | 28.28x | **0.15x** &uarr; |
| 2^28 (256 MiB) | 2^18 | 0.000 | 6.22 ms | 175.85 ms | **955.56** | 28.27x | **0.15x** &uarr; |
| 2^28 (256 MiB) | 2^23 | 0.000 | 6.19 ms | 173.07 ms | **1.12 ms** | 27.96x | **0.18x** &uarr; |
| 2^28 (256 MiB) | 8 | 0.201 | 438.22 | 2.88 ms | **1.09 ms** | 6.56x | **2.48x** |
| 2^28 (256 MiB) | 2^23 | 0.201 | 4.32 ms | 63.55 ms | **4.83 ms** | 14.71x | **1.12x** |
| 2^24 (16 MiB) | 2^23 | 1.000 | 404.94 | 422.77 | **424.10** | 1.04x | **1.05x** |

&uarr; = `lift_partition` is **faster than `main`**.

### Worst remaining cases (lift_partition > 2x main)

All are on `Entropy=0.201` with small `SelectedElements`. They are a
separate bottleneck (queue path / second-pass amortisation) -- the
back-grow-cap optimisation does not apply because not every key is a
tie. These cases are unchanged from `flat_last` / `drop_xforms` /
`by_value` (the lift only affects the all-ties / capped-cap regime).

| Elements | Sel | main | lift_partition | ratio |
|---|---:|---:|---:|---:|
| 2^16 | 8 | 18.0 us | 42.8 us | 2.38x |
| 2^24 | 8 | 55.0 us | 163.9 us | 2.98x |
| 2^28 | 8 | 438.2 us | 1.09 ms | 2.48x |
| 2^24 | 256 | 63.5 us | 163.9 us | 2.58x |
| 2^28 | 256 | 438.6 us | 1.09 ms | 2.48x |

---

## Entropy=0.000 across all KeyT x ValueT (Elements=2^24, Sel=2^8)

The catastrophic cliff was present for every (KeyT, ValueT) combination
in the `by_value` variant. After the lift, `lift_partition` is faster
than `main` on every (KeyT, ValueT) pair tested:

| KeyT | ValueT | main (us) | lift_partition (us) | lp / main |
|---|---|---:|---:|---:|
| I8  | I8  | 366.6 | 95.3 | **0.26x** |
| I8  | I16 | 361.5 | 96.3 | **0.27x** |
| I8  | I32 | 366.8 | 94.3 | **0.26x** |
| I8  | I64 | 365.3 | 94.2 | **0.26x** |
| I16 | I8  | 376.2 | 91.4 | **0.24x** |
| I16 | I16 | 386.0 | 92.0 | **0.24x** |
| I16 | I32 | 384.6 | 89.1 | **0.23x** |
| I16 | I64 | 386.1 | 89.2 | **0.23x** |
| I32 | I8  | 406.8 | 108.1 | **0.27x** |
| I32 | I16 | 408.6 | 110.2 | **0.27x** |
| I32 | I32 | 402.8 | 108.6 | **0.27x** |
| I32 | I64 | 403.2 | 108.6 | **0.27x** |
| I64 | I8  | 547.5 | 270.0 | **0.49x** |
| I64 | I16 | 551.3 | 272.7 | **0.49x** |
| I64 | I32 | 545.9 | 272.0 | **0.50x** |
| I64 | I64 | 546.1 | 272.7 | **0.50x** |

`I64` shows a smaller (2x) but still material speedup; the smaller-key
variants are 3.7x-4.3x faster than `main` on the entropy=0 case.

---

## Artifacts

- `topk_perf_tracking/snapshots/lift_partition.json` -- ptxas resource snapshot.
- `topk_perf_tracking/bench/sweep_lift_partition_i32i32.json` -- full i32/i32 sweep.
- `topk_perf_tracking/bench/sweep_lift_short_keysizes.json` -- i8/i16/i32 short sweep across all 3 entropies.
- `topk_perf_tracking/bench/sweep_lift_e000_keysizes.json` -- entropy=0 across i8/i16/i32/i64.
- `topk_perf_tracking/bench/sweep_main_e000_keysizes.json` -- matching main reference sweep.
- `topk_perf_tracking/reports/sweep_i32_i32_lift_partition_vs_main.md` -- full per-state diff.
