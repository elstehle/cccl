# Round 2 follow-up — counter scalar refactor + uniform-`queue_idx`

Implements suggestions (1) and (6) from `FINDINGS.md`, in two separate commits
on `exp/topk-batched-large-segments-regressions`:

| commit  | summary                                                     |
|---------|-------------------------------------------------------------|
| `263bfc7e8d` | extract counter cross-pass scalars into a 16-B-aligned `counter_cross_pass_state` and widen the buffered-load flag from `bool` to `uint32_t`, so the entry-of-pass `LDG.E.128` no longer needs a `PRMT` to extract the bool byte |
| `979ba0e84b` | route the `UpperBound(d_large_segments_tile_offsets, …)` result in all three batched agents through `cub::detail::warpspeed::makeWarpUniform`, with a new `int64_t` overload that goes through two `__reduce_*_sync` (CREDUX) halves to give ptxas a uniform-promotion-friendly producer |

(Plus `9e502eb6dd` which snapshots the post-change SASS slices and dumps
into this directory for diffing.)

## TL;DR of the resource-usage moves

(All deltas vs the pre-change SASS captured at the start of the
investigation — `b23102c594`. `Δ REG` only; smem/stack/lmem/cmem are
unchanged everywhere.)

### Keys-only

| kernel    | KeyT  | initial REG | after both REG | Δ |
|-----------|-------|------------:|---------------:|--:|
| filter    | i8    |    122      |    120         | **−2** |
| filter    | i16   |     59      |     64         | **+5** ← regression |
| filter    | f32   |     52      |     48         | **−4** |
| filter    | i32   |     52      |     48         | **−4** |
| filter    | f64   |     52      |     52         |  0 |
| filter    | i64   |     52      |     52         |  0 |
| filter    | i128  |     52      |     50         | **−2** |
| last_filt | f64   |     27      |     28         | +1 (drift; max/min direction asymmetric) |
| histogram (all KeyT) |  | unchanged | unchanged |  0 |
| worker / last_filt (other KeyT) |  | unchanged | unchanged |  0 |

### Pairs

| kernel    | KeyT | ValueT | initial REG | after both REG | Δ |
|-----------|------|--------|------------:|---------------:|--:|
| filter    | i8   | i8     |    114      |    112         | **−2** |
| filter    | i8   | i16    |    106      |    108         | **+2** ← regression |
| filter    | i8   | i32    |    106      |    108         | **+2** ← regression |
| filter    | i8   | i64    |    106      |    108         | **+2** ← regression |
| filter    | i16  | any    |     64      |     64         |  0 |
| filter    | i32  | i8     |     48      |     40         | **−8** ← largest win |
| filter    | i32  | i16    |     46      |     40         | **−6** |
| filter    | i32  | i32    |     46      |     40         | **−6** |
| filter    | i32  | i64    |     40      |     40         |  0 (already at minimum) |
| filter    | i64  | any    |     52      |     52         |  0 |
| every histogram / worker / last_filt |  |  | unchanged | unchanged |  0 |

## Per-commit breakdown

### Commit 1 only — `counter_cross_pass_state` + `uint32_t` flag

| benchmark | rows moved | net effect |
|-----------|-----------:|-----------|
| keys      | 3 of 28    | filter i8 −2, filter i128 −2, last_filt f64 +1 |
| pairs     | 0 of 52    | no change |

The `PRMT` on the bool byte at the end of the counter load went away (the
batched filter now reads the flag with a direct 32-bit `ISETP` against
`RZ`), so the i8 / i128 keys-only filter shed a couple of registers each
from the byte-extracting overhead. That's the entire effect of this
commit on the static metrics — the per-`ValueT` pairs instantiations
already had the bool field accessed identically and didn't move.

**Critical caveat:** the loaded counter scalars are still **not**
`R2UR`-promoted on the batched filter. The `R2UR` count in the i8
keys-only filter SASS is `0` both before and after this commit. The
bool→uint32 change cleared one specific blocker (PRMT consumption of the
bool byte) but ptxas's heuristic needed more than that to fire — see
commit 2 for the rest.

### Commit 2 only — `makeWarpUniform(UpperBound(...) - 1)` in all 3 batched agents

| benchmark | rows moved | net effect |
|-----------|-----------:|-----------|
| keys      | 3 of 28    | filter f32 −4, filter i32 −4, filter i16 **+5** |
| pairs     | 7 of 52    | filter i32×i8 −8, filter i32×i16/i32 −6 each, filter i8×i8 −2, filter i8×i16/i32/i64 **+2** each |

The headline win is **filter i32 × * pairs going from 48/46/40 down to a
flat 40** — closing most of the gap vs the single-problem target of 32.
On the same instantiation, `R2UR` now actually fires (one `R2UR UR5, R2`
in the prologue) and the loaded counter scalars feed into a string of
`UISETP` (uniform-uniform integer set-predicate) instructions instead of
the regular `ISETP` chain we had before.

**Where it backfires:** on the `i8 × *` pairs filter and on the `i16`
keys-only filter, the `__shfl_sync`-driven warp broadcast adds prologue
overhead (the CREDUX still needs predication setup) but ptxas does
*not* fire `R2UR` because the per-thread peak is already dominated by
the 16 or 8 unpacked-key bytes — the marginal benefit of promoting the
~4 counter scalars to uniform registers is not enough to outweigh the
broadcast overhead in ptxas's cost model. Same story for `i16`
keys-only filter (`items_per_thread = 8`).

> The narrow-key paths use the Hopper+ tuning's `items_per_thread = 16`
> (i8) or `8` (i16) — the per-thread register footprint is dominated by
> the unpacked PRMT'd bytes carried live across the 16 atomic histogram
> updates. That live set is essentially incompressible without
> restructuring the load/classify/atomic flow itself. Suggestions (1)
> and (6) are orthogonal levers that work on the per-segment scalar
> path; for the narrow-key regression we need suggestion (5) — clamp
> `items_per_thread` for narrow keys on the multi-CTA tuning — or a
> structural change to that flow.

## SASS evidence: where R2UR fired vs where it didn't

Census of the `R2UR` opcode in the i8 keys-only filter and the i32×i32
pairs filter, both `max` direction, post-both-commits:

| kernel                       | R2UR count | LDCU count | unique R | unique UR |
|------------------------------|-----------:|-----------:|---------:|----------:|
| batched i8 keys-only filter  | **0**      | 76         |    111   |   8       |
| batched i32×i32 pairs filter | **1**      | 91         |     40   |  14       |
| single i8 keys-only filter   |  2         | 209        |     30   |  21       |
| single i32×i32 pairs filter  |  (not measured here) |  |  |  |

For the i32×i32 pairs filter, the `LDG.E.128 R32, desc[UR8][R68.64]`
that loads the counter struct is now followed by:

```sass
UISETP.NE.U32.AND UP2, UPT, UR9, URZ, UPT ;        // uniform-uniform compare
R2UR              UR5, R2 ;                        // promote a regular reg to UR
UISETP.GE.U32.AND UP0, UPT, UR4, UR9, UPT ;        // …and the rest of the early-stop / will_buffer
UISETP.GE.U32.AND UP1, UPT, UR4, UR9, UPT ;        //   chain is all UISETP/UR-only
```

That's the pattern the single-problem filter has by default and the
batched filter never used to have. With it, the per-segment scalar
state lives in the uniform-register file for the whole iteration body,
and ptxas can keep the regular-register file's peak low enough for the
filter kernel to run comfortably under 64 regs.

For the i8 keys-only filter the same `LDG.E.128 R32, ...` is followed
by per-thread `ISETP.NE.AND` against `RZ` and the loaded values stay in
`R32-R35`. No `R2UR` fires.

## Ranked distance-to-single after both commits

| rank | configuration                       | Δ REG (initial) | Δ REG (now) | change |
|-----:|-------------------------------------|----------------:|------------:|-------:|
| 1    | keys-only `filter` `i8`             | +90             | **+88**     | −2 |
| 2    | pairs `filter` `i8 × i8`            | +82             | **+80**     | −2 |
| 3    | pairs `filter` `i8 × {i16,i32,i64}` | +74             | **+76**     | **+2 worse** |
| 4    | pairs `filter` `i16 × *`            | +32             | +32         | 0 |
| 5    | keys-only `last_filt` `i8`          | +32             | +32         | 0 |
| 6    | pairs `last_filt` `i8 × *`          | +32             | +32         | 0 |
| 7    | keys-only `filter` `i16`            | +27             | **+32**     | **+5 worse** |
| 8    | keys-only `histogram` `i8`          | +27             | +27         | 0 |
| 9    | keys-only `filter` `i128`           | +21             | **+19**     | −2 |
| 10   | keys-only `filter` `f64`            | +20             | +20         | 0 |
| 11   | keys-only `filter` `i64`            | +20             | +20         | 0 |
| 12   | keys-only `filter` `f32`            | +20             | **+16**     | −4 |
| 13   | keys-only `filter` `i32`            | +20             | **+16**     | −4 |
| 14   | pairs `filter` `i64 × *`            | +20             | +20         | 0 |
| 15   | keys-only `histogram` `i16`         | +19             | +19         | 0 |
| 16   | pairs `histogram` `i16`             | +19             | +19         | 0 |
| 17   | pairs `filter` `i32 × i8`           | +16             | **+8**      | **−8** |
| 18   | pairs `filter` `i32 × i16/i32`      | +14             | **+8**      | **−6** |
| 19   | pairs `filter` `i32 × i64`          | +8              | +8          | 0 |
| 20-* | everything below +10                | unchanged       | unchanged   | 0 |

## What this tells us

- **`R2UR` is on the critical path** for closing the gap to single-problem.
  When it fires (i32 / f32 / i32×* configurations), the batched filter
  drops 4-8 registers. When it doesn't fire (i8, i16), the broadcast
  overhead is a net loss.
- **Suggestion (1) alone is insufficient** but **necessary** — it cleared
  one of the three friction patterns that block `R2UR` (PRMT on the
  bool field), without which suggestion (6) wouldn't be effective on
  the configurations where it does work today.
- **Suggestion (6) is a per-configuration win/loss** as currently
  implemented. The broadcast overhead is real and the heuristic for
  promotion depends on per-thread liveness elsewhere in the kernel.
- **The narrow-key configurations (i8, i16) are still the largest
  outliers** and won't close further without one of:
   - **suggestion 5** — clamp `items_per_thread` for narrow keys on
     the multi-CTA tuning (the Hopper+ tuning is the prime culprit;
     setting `items_per_thread = 4` for `key_size <= 2` would shrink
     the dominant per-thread live set from 16 → 4 unpacked bytes);
   - or a structural change to the load/classify/atomic flow itself
     (suggestions 3 / 4 in FINDINGS.md).

## What to do about the regressions

The +5 on `i16` keys-only filter and +2 on the three `i8 × {i16,i32,i64}`
pairs configurations come from the same source: the new
`makeWarpUniform` prologue work without ptxas firing `R2UR` to absorb
it. Two reasonable choices:

1. **Gate `makeWarpUniform` behind a `items_per_thread <= 4` policy
   check.** Cheap to add; preserves the i32 / f32 wins; recovers the
   i8 / i16 regressions to break-even. Recommended next step.
2. **Push harder on the narrow-key path itself** by clamping
   `items_per_thread` (suggestion 5) — that removes the underlying
   per-thread peak driver and would make the `makeWarpUniform` hint
   pay off there too. Bigger change, but it's the path with the
   biggest absolute headroom (those configurations are 80-90 regs
   above the single-problem target).

I have not applied either yet — they were not in the requested set —
but the diagnostic data is ready for a quick follow-up if you want.

## Files added this round

```
topk_register_pressure_investigation/
├── FOLLOWUP_round_2.md                              (this file)
├── batched_i8_filter_max.after_both_commits.sass    (i8 keys-only, post-both)
├── batched_i32_filter_max.after_both_commits.sass   (i32 keys-only, post-both)
├── batched_i32xi32_filter_max.after_both_commits.sass (i32×i32 pairs, biggest win)
├── keys.after_both_commits.raw / keys.compare.after_both_commits.txt
└── pairs.after_both_commits.raw / pairs.compare.after_both_commits.txt
```

The original `batched_i8_filter_max.sass` (pre-changes baseline) is
preserved alongside; diffing `batched_i8_filter_max.sass` against
`batched_i8_filter_max.after_both_commits.sass` shows the byte-level
SASS movement.
