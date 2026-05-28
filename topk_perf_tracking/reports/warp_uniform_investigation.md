# Attempt to regain warp-aggregated atomics on the narrowed build

## TL;DR

- **Landed on dev:** `makeWarpUniform` hint on `resolve_queue_idx` (commit `f27f8f371e`).
  Independently good: makes warp-uniformity of `queue_idx` explicit via `CREDUX`
  (`__reduce_min_sync`), which ptxas's `R2UR` pass actually recognises (unlike
  `SHFL.IDX`). On the dev branch (without narrowing), it's net **neutral**
  (GM 0.9965, median 1.00, max +1.4%, min -4.1% over the I32/I32 sweep). Future-proofs
  against ptxas heuristic changes.
- **Did not fix the narrow-build regression.** With narrowing on top, the kernel
  still emits unconditional atomics (24 unconditional `ATOMG` vs 24 `@P0`/`@P3`
  predicated in dev), still spills `total` to local memory, and still runs at
  ~6.0 ms vs dev's 0.28 ms on `I32/I32 / 2^24 / sel=2^23 / ent=0` (~21x slower).
- **Root cause is deeper** than just `queue_idx` not being uniform: ptxas's whole
  R2UR (register-to-uniform-register) promotion pass refuses to fire on the
  narrowed kernel, even when explicit `makeWarpUniform` hints are added at
  multiple points along the pointer chain.

## What worked: explicit uniformity hint on `resolve_queue_idx`

`resolve_queue_idx` previously ended in
```cpp
return __shfl_sync(0xffffffff, queue_idx_lane0, 0);
```

Per the docstring on `cub/detail/warpspeed/make_warp_uniform.cuh`:

> *Empirically, ptxas's `R2UR` heuristic on Blackwell does not recognize the result of
> `SHFL.IDX` with `srcLane=0` as warp-uniform, but does recognize the result of
> `__reduce_*_sync`.*

Routing the broadcast through `makeWarpUniform` (which lowers to `CREDUX.MIN` on
sm_90+) fixes that:

```cpp
return detail::warpspeed::makeWarpUniform(__shfl_sync(0xffffffff, queue_idx_lane0, 0));
```

SASS proof on the *dev branch* (without narrowing), I32/I32 last_filter:

| metric | dev (orig) | dev + warp_uniform |
|---|---:|---:|
| VOTEU.ANY / `@P` ATOMs | 24 / 24 | 24 / 24 |
| R2UR | 12+ | 12+ |
| CREDUX | 6 | **8** (added 2 from my hint) |
| LDL / STL | 0 | 0 |
| Runtime on the worst-case workload | 283 us | 281 us |

So on dev the hint is net-neutral (slightly better in the geomean, no regressions
worth caring about). It's a safe, robust landing.

## What didn't work: the same hint on the narrowed build

Applied the same hint on top of the narrowing change (commit `931ba49866`).
SASS on narrow + warp_uniform, same kernel:

| metric | narrow + warp_uniform | dev (reference) |
|---|---:|---:|
| VOTEU.ANY | **0** | 24 |
| `@P` predicated ATOMs | **0** | 24 |
| Unconditional ATOMs | **24** | 0 |
| R2UR | **0** | 12+ |
| CREDUX | 2 (from my hint) | 6 |
| LDL / STL | 3 | 0 |
| Runtime | **6.013 ms** | 0.281 ms |

The CREDUX from my hint is in the SASS (proof it reaches the right place), but
**no R2UR instructions at all**. ptxas's UR-promotion pass is globally disabled
on this kernel.

## Going deeper: also hint the segment-counter pointer

Tried also wrapping `s.segment_counter = d_segment_counters + queue_idx` with
`makeWarpUniform` (via u64 bit-cast):

```cpp
counter_t* raw = d_segment_counters + queue_idx;
auto raw_u64  = reinterpret_cast<::cuda::std::uint64_t>(raw);
raw_u64       = detail::warpspeed::makeWarpUniform(raw_u64);
s.segment_counter = reinterpret_cast<counter_t*>(raw_u64);
```

Result: CREDUX count went up (2 -> 7), but still **0 R2UR, 0 VOTEU**, atomics
remain unconditional, runtime still 6.013 ms. No fix.

## Why explicit hints aren't enough

Looking at the SASS prologue, the difference between dev and narrow is structural,
upstream of any user-controllable hint:

**Dev prologue:**
```
LDCU.64 UR10, c[0x0][0x358]     // *uniform* constant load -> UR
LDCU.64 UR4,  c[0x0][0x3f8]     // *uniform* constant load -> UR
LD.E.64 R2,   desc[UR10][R2.64] // global load into GPR
R2UR    UR14, R2                // *automatic* GPR -> UR promotion
R2UR    UR15, R3
ULEA    UR4, ...                // *uniform-LEA* pointer math, all in UR-space
```

**Narrow prologue:**
```
LDC.64  R2,  c[0x0][0x3f8]      // *per-thread* constant load -> GPR
LD.E    R36, desc[UR8][R36.64]  // global load into GPR
                                 // <-- no R2UR; loaded value stays in GPR
IMAD.WIDE.U32 R4, R36, 0x4, R2  // pointer math in regular-GPR-space
STL [R1], R0                     // ... and the loop-bound value gets spilled
```

Ptxas made a higher-level decision about this whole kernel: it's tracking values
in GPRs and not promoting to URs at all. The constant-memory loads use `LDC`
(per-thread) instead of `LDCU` (uniform) -- even though the source values are
constant-memory kernel parameters that are by definition warp-uniform.

What changed between dev and narrow that would tip ptxas's heuristic into the
GPR-only mode? My narrowing CL:

1. Added a new `typename SegmentCountT` template parameter to five kernels and two
   agents.
2. Changed `num_large_segments` field in the filter and last_filter agents from
   `NumSegmentsParameterT::value_type` (int64) to `SegmentCountT` (uint32).
3. Re-typed `large_segments_count_it`, `num_large_segments` locals, `queue_idx_t`
   from int64 to uint32 inside the kernels.

None of these are individually pathological -- they're type narrowings, no
semantic change. But the COMBINATION trips some ptxas heuristic that decides this
isn't a kernel worth doing UR-tracking on.

User-controllable workarounds I tried (and that did NOT help):

- Explicit `makeWarpUniform(queue_idx)`.
- Explicit `makeWarpUniform(segment_counter pointer)` via u64 bit-cast.

Workarounds I considered but didn't try (would be fragile / wide-reaching):

- Inline PTX hints (`asm volatile` directives).
- Restructuring all kernel params to use `_CCCL_GRID_CONSTANT` (already done).
- Wrapping every value with `makeWarpUniform` on entry.
- Splitting the narrowing into multiple smaller patches to bisect which specific
  change tips the heuristic.

The pragmatic conclusion is that recovering warp-aggregated atomics under the
narrowing needs **ptxas-side cooperation** (or a bisect-and-retry approach to
land only the narrowing pieces that don't trip the heuristic). It's not a quick
fix from the C++ source side.

## What landed

- `f27f8f371e topk(experiment): route resolve_queue_idx broadcast through makeWarpUniform`
  on `exp/topk-batched-large-segments-regressions`.
- Neutral on the dev branch (no regressions on the I32/I32 sweep; modest wins
  in some cells).
- The hint stays useful **regardless** of whether the narrowing is later re-tried:
  it makes the `queue_idx` warp-uniformity explicit instead of relying on
  ptxas's heuristic to recognise `SHFL.IDX` as uniform.

## What's still on the table

For a future re-attempt at the narrowing:

1. **Bisect the narrowing changes.** Land them one at a time and check at each
   step whether ptxas still emits warp-aggregated atomics for last_filter.
   Candidates for what trips the heuristic, in priority order:
   - The new `SegmentCountT` template parameter on the agent (changes the
     instantiation identity).
   - The `num_large_segments` field type change.
   - The kernel-local `num_large_segments` type change.
2. **Use the `__ldcs` / `__ldg` family** to nudge ptxas's view of which loads
   are constant. Probably ineffective for this issue but cheap to try.
3. **Consider keeping the dispatch-side narrowing only** (smaller
   `large_segments_ids` buffer, smaller `batched_topk_counters::large_segments_count`)
   without changing the agent template signatures. That captures the
   unconditional memory wins without provoking ptxas.

Branches:

- `tmp/perf-eval-warp-uniform` -- dev + warp-uniform hint + narrowing on top
  (regresses, kept for future SASS analysis).
- `exp/topk-batched-large-segments-regressions` -- dev + warp-uniform hint only
  (landed, neutral).
- `tmp/perf-eval-narrow-segcount` -- the original narrowing change (kept for
  reference / re-attempt).
