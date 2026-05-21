# Round 3 follow-up — lane-0-only `UpperBound` + explicit broadcast

Answers two questions raised after Round 2:

> "How is `makeWarpUniform` used in other places — and which lane's value is broadcasted?"

> "Should we invoke `UpperBound` only on the first thread and then broadcast using `makeWarpUniform`?"

## What `makeWarpUniform` is (and isn't)

`cub::detail::warpspeed::makeWarpUniform` is **not** a broadcast; it's a
**uniformity-preserving reduction** that depends on the caller already
having identical values across all lanes. The Hopper+ implementation
is literally `__reduce_min_sync(~0, x)` (CREDUX):

```cpp
[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t makeWarpUniform(::cuda::std::uint32_t x)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
    (return __reduce_min_sync(~0, x);),    // ← min across all 32 lanes
    (return x;));                          // ← no-op on pre-Hopper
}
```

What that means in practice:

| input on lanes 0..31    | `__reduce_min_sync` result | `__shfl_sync(~0, x, 0)` result |
|-------------------------|---------------------------:|-------------------------------:|
| all `v`                 |                       `v`  |                            `v` |
| `v, garbage, garbage…`  |  `min(v, garbage…)` (wrong)|                            `v` |

So **"which lane's value is broadcast" only has a well-defined answer
when the input is already uniform** — in which case both paths return
the same value. The 64-bit overload composes two 32-bit `makeWarpUniform`
halves on Hopper+ and falls back to `__shfl_sync(~0, x, 0)` on older
arch — the latter is a true broadcast from lane 0.

### The two existing callers

| call site                                                  | input | uniform by construction? |
|------------------------------------------------------------|------|--------------------------|
| `cub/detail/warpspeed/special_registers.cuh:39`            | `threadIdxX / 32` (warp id) | yes — all 32 threads of a warp share the same warp id |
| `cub/agent/agent_batched_topk.cuh:625, 1065, 1687` (Round 2) | `UpperBound(d_large_segments_tile_offsets, num_large_segments, global_tile_id) - 1` | yes — `global_tile_id` is identical across all lanes of the warp |

Both are pure ptxas hints. The "broadcast" framing was imprecise in the
Round-2 commit messages — the reduction works for them because the
input is already identical across lanes, not because `makeWarpUniform`
broadcasts anything.

## The lane-0-only experiment (commit `dad3fb2ceb`)

Yes: the proposed pattern is feasible, but `makeWarpUniform` is the
wrong primitive for it. Only lane 0 has the right answer; lanes 1..31
hold garbage/zero. `__reduce_min_sync` over that mix would return the
min of (lane 0's answer, garbage), which is wrong. We need a **true
broadcast**, i.e. `__shfl_sync(0xffffffff, value, 0)`.

The new pattern is:

```cpp
// Only lane 0 runs the binary search.
LargeSegmentTileOffsetT queue_idx_lane0 = 0;
if ((threadIdx.x & 31) == 0)
{
  queue_idx_lane0 =
    UpperBound(d_large_segments_tile_offsets, num_large_segments, global_tile_id) - 1;
}
// Broadcast lane 0's value to the rest of the warp.
const LargeSegmentTileOffsetT queue_idx =
  __shfl_sync(0xffffffff, queue_idx_lane0, 0);
```

Applied to all three batched agents in commit `dad3fb2ceb`.

### Results (commit `dad3fb2ceb`, vs Round-2 `makeWarpUniform` snapshot)

Keys-only:

| kernel    | KeyT  | makeWarpUniform REG | lane-0 broadcast REG | Δ |
|-----------|-------|--------------------:|---------------------:|--:|
| histogram | i8    | 55                  | 56                   | +1 |
| histogram | i16   | 49                  | 54                   | **+5** |
| filter    | i16   | 64                  | 63                   | −1 |
| filter    | f32   | 48                  | 52                   | **+4** |
| filter    | i32   | 48                  | 52                   | **+4** |
| filter    | **i128** | 50               | **40**               | **−10** ← biggest win |
| last_filt | f32   | 26                  | 29                   | +3 |
| last_filt | i32   | 30                  | 32                   | +2 |
| last_filt | i16   | 32                  | 32 (stack 0 → 8)     | 0 (new spill) |

Pairs:

| kernel    | KeyT | ValueT | makeWarpUniform REG | lane-0 broadcast REG | Δ |
|-----------|------|--------|--------------------:|---------------------:|--:|
| histogram | i8   |  —     | 55                  | 56                   | +1 |
| histogram | i16  |  —     | 49                  | 54                   | +5 |
| filter    | **i8** | **i8** | 112              | **104**              | **−8** |
| filter    | i8   | i16    | 108                 | 114                  | +6 |
| filter    | i8   | i32    | 108                 | 114                  | +6 |
| filter    | i8   | i64    | 108                 | 114                  | +6 |
| last_filt | i16  | i8     | 32 (stack 8)        | 40 (stack 0)         | +8 (lost 8B spill) |
| last_filt | i16  | i16    | 32 (stack 16)       | 40 (stack 0)         | +8 (lost 16B spill) |
| last_filt | i16  | i32    | 32 (stack 16)       | 40 (stack 0)         | +8 (lost 16B spill) |
| last_filt | i16  | i64    | 32 (stack 16)       | 40 (stack 0)         | +8 (lost 16B spill) |
| filter    | i32  | i8     | 40                  | 48                   | **+8** ← lost Round-2 win |
| filter    | i32  | i16    | 40                  | 48                   | **+8** |
| filter    | i32  | i32    | 40                  | 48                   | **+8** |
| filter    | i32  | i64    | 40                  | 48                   | **+8** |
| filter    | i64  | i8     | 52                  | 50                   | −2 |
| filter    | i64  | i16    | 52                  | 50                   | −2 |
| filter    | i64  | i32    | 52                  | 50                   | −2 |
| filter    | i64  | i64    | 52                  | 50                   | −2 |

### Cumulative across all three commits (vs initial `b23102c594` baseline)

Keys-only `Δ REG vs single-problem`:

| kernel    | KeyT  | initial Δ | round-2 Δ | **round-3 Δ** |
|-----------|-------|----------:|----------:|--------------:|
| filter    | i8    |  +90      |  +88      |  **+88** |
| filter    | i16   |  +27      |  +32      |  **+31** |
| filter    | f32   |  +20      |  +16      |  **+20** |
| filter    | i32   |  +20      |  +16      |  **+20** |
| filter    | f64   |  +20      |  +20      |  **+20** |
| filter    | i64   |  +20      |  +20      |  **+20** |
| filter    | **i128** | **+21** | **+19** |  **+9**  ← still by far the largest win |

Pairs `Δ REG vs single-problem`:

| kernel    | KeyT  | ValueT | initial Δ | round-2 Δ | **round-3 Δ** |
|-----------|------|--------|----------:|----------:|--------------:|
| filter    | **i8**  | **i8** | +82  | +80  | **+72** ← second-largest cumulative win |
| filter    | i8   | i16    | +74      | +76      | **+82** |
| filter    | i8   | i32    | +74      | +76      | **+82** |
| filter    | i8   | i64    | +74      | +76      | **+82** |
| filter    | i32  | i8     | +16      | **+8**   | +16    |
| filter    | i32  | i16    | +14      | **+8**   | +16    |
| filter    | i32  | i32    | +14      | **+8**   | +16    |
| filter    | i32  | i64    | +8       | +8       | +16    |
| filter    | i64  | *      | +20      | +20      | +18    |

## Why the mixed result?

The lane-0 broadcast changes two things simultaneously:

1. **Producer side:** only one lane runs the binary search. ptxas now
   sees a divergent block where ~10 regular registers are live for the
   duration of the search (`R7,R8,R9,R10,R11,R12,R14,R15` plus
   bookkeeping). The binary search itself **does not** get promoted to
   uniform registers despite running only on lane 0 — ptxas doesn't
   propagate "single-lane-active" uniformity through a `BSSY`-bounded
   block.

2. **Consumer side:** the broadcast result (`__shfl_sync` with
   `srcLane=0`) lands in a regular register. `R2UR` does fire here —
   four times in the i32×i32 filter (vs once with `makeWarpUniform`) —
   and downstream uses become `UISETP` against parameter-area scalars
   loaded with `LDCU` (we see `LDCU UR9, c[0x0][0x434]` for `num_items`
   and `LDCU.64 UR14, c[0x0][0x3f8]` for the tile-offsets pointer in
   the prologue, both uniform). The post-broadcast region is *more*
   uniform than the Round-2 version.

The configurations where this trade-off pays off:

- **`i128` keys-only filter (-10 regs)** — the per-thread "everything
  else" is small (1 item per thread, key_size=16), so the divergent
  search's 10-reg footprint isn't competing with anything else, and
  the now-uniform downstream code yields a big register saving.
- **`i8 × i8` pairs filter (-8 regs)** — narrow keys, narrow values:
  per-thread peak benefits significantly from moving even a handful of
  the segment-pointer + counter-field state to UR.
- **`i64 × *` pairs filter (-2 each)** — same logic, smaller magnitude.

The configurations where it backfires:

- **`i32 × *` pairs filter (+8 each, losing the Round-2 win)** — the
  divergent block's 10 regular regs now overlap with the per-thread
  unpacked-key working set, pushing peak up. Round-2's
  `makeWarpUniform`-on-redundant-search was the better choice here:
  it kept all 32 lanes "doing the same work" so the registers used
  during the search overlapped with the registers used after, and
  ptxas could schedule them as a single set.
- **`i16` histogram (+5)** — extra prologue overhead with no `R2UR`
  payoff in the histogram body.
- **`i8 × {i16,i32,i64}` pairs filter (+6 each)** — same pattern as
  the i32 pairs regression; the wider value type's per-thread work
  competes with the search's regular-reg footprint.

There's no single tuning knob that explains the win/loss pattern. It's
the interplay between (per-thread peak − search-block live set − useful
`R2UR` payload). When the search block is "free" (rest of the kernel
doesn't need those regs simultaneously), lane-0 broadcast wins. When
the search block competes with rest-of-kernel, it loses.

## SASS evidence for the wins

The `i128` keys-only filter dropped from 50 to 40 regs (-10). With
`__shfl_sync(0xffffffff, queue_idx_lane0, 0)`, the prologue now looks
like (snipped to show the uniform structure):

```sass
LDCU UR9,  c[0x0][0x434] ;          ; num_items into UR9
LDCU UR7,  c[0x0][0x3d8] ;          ; k_param into UR7
LDCU.64 UR14, c[0x0][0x3f8] ;       ; d_large_segments_tile_offsets into UR14
…
R2UR    UR5, R2 ;                   ; promote ceil_div result to UR5
UIADD3  UR6, UPT, UPT, URZ, -UR5, URZ ;
UIMAD   UR6, UR6, UR9, URZ ;        ; uniform-uniform arithmetic on the k-clip path
UISETP.NE.U32.AND UP2, UPT, UR9, URZ, UPT ;
UIMAD.WIDE.U32 UR4, UR5, UR6, UR4 ;
UIMAD.WIDE.U32 UR4, UR5, UR7, URZ ;
UISETP.GE.U32.AND UP0, UPT, UR4, UR9, UPT ;
…                                    ; the whole k-clip and segment-size derivation
                                     ;   stays in UR4..UR9.
```

That's the pattern we wanted Round-2 to produce but didn't get. The
broadcast eliminated the ambiguity in ptxas's analysis.

For `i32 × i32` pairs filter (which regressed), the divergent
binary-search block holds R7-R15 live across thread_search.cuh's loop
body simultaneously with the per-thread "16 unpacked bytes" working
set that's needed by the atomic-update loop further down, so the peak
moves up to 48 regs.

## Recommendation

The data shows the lane-0 broadcast is **not a strict improvement**
over `makeWarpUniform`. It produces the largest single-configuration
win we've seen (`i128` keys-only filter `-10` regs cumulative, putting
it at `Δ+9` vs single — i.e. essentially at parity), but it regresses
the `i32 × *` pairs filter back to its pre-Round-2 level (+8 vs
single) and degrades the wide-value `i8 × *` pairs filter further.

Three reasonable next moves:

1. **Revert commit `dad3fb2ceb`** and keep `makeWarpUniform` —
   trades the `i128` and `i8 × i8` wins for not regressing
   `i32 × *` pairs.
2. **Keep commit `dad3fb2ceb`** — trades the `i32 × *` and wide-value
   `i8 ×` losses for the `i128` and `i8 × i8` wins. Net "distance to
   single" sum is slightly worse than Round-2 but the worst-case
   regression (`+88` on `i8` keys-only) is unchanged, and we now have
   one configuration that's essentially at single-problem parity.
3. **Choose the pattern per agent / per template parameter** — e.g.
   gate on `items_per_thread <= 2` (`i64`, `i128`) for the lane-0
   broadcast and use `makeWarpUniform` everywhere else. This would
   keep all the wins without the regressions, but adds
   policy-dependent code to the agent.

My read: option 1 (revert) is the safest. Option 3 is the best
trade-off but more invasive — pick that one if we go to round 4.

(Either of options 1/3 still need suggestion 5 from FINDINGS.md —
clamping `items_per_thread` for narrow keys on the multi-CTA tuning —
to make a meaningful dent on the `i8` keys-only filter, which still
sits at +88 vs single across all three rounds. That's the next big
lever.)

## Files added this round

```
topk_register_pressure_investigation/
├── FOLLOWUP_round_3.md                                     (this file)
├── batched_i8_filter_max.after_lane0_broadcast.sass        (lane-0 broadcast, no improvement)
├── batched_i128_filter_max.after_lane0_broadcast.sass      (lane-0 broadcast, big -10 win)
├── batched_i32xi32_filter_max.after_lane0_broadcast.sass   (lane-0 broadcast, +8 regression vs Round-2)
├── keys.after_lane0_broadcast.raw
└── pairs.after_lane0_broadcast.raw
```
