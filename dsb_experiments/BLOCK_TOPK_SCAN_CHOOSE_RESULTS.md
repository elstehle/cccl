# A faster scan+choose for the block top-k radix pass — analysis, prototypes, results

Branch `exp/scan-choose-opt`. Code: `proto_scan_choose.cu` (isolated stage benchmark) and the
`AFSCAN` variant in `proto_air_ablate.cu` (end-to-end). Measured on NVIDIA B200 (sm_100),
CUDA 13.1, node `umb-b200-235`. Companion docs: `BLOCK_TOPK_AIR_ABLATION.md` (per-change
ablation; its "L8" section is the condensed form of this report),
`BLOCK_TOPK_AIR_OPT_RESULTS.md` (the wider optimization program).

## TL;DR

Restructuring the block scan so the **cross-warp aggregate is produced by `redux.sync.add`
*before* the shuffle scan runs** ("aggregate-first", AF) cuts the fused scan+choose stage from
**403 → 334 cyc at 256 bins (−25% net of peripherals)** and **558 → 525 at 2048 bins (−9%)**,
and translates end-to-end into new bests for the optimized top-k:
**f32+i32 1817 → 1689 (−7%)**, **u32+i32 1809 → 1637 (−9.5%)**, 4-pass worst case −270..−330 —
at +32 B shared memory, unchanged registers/occupancy, and a ~2-3% throughput dip at full
saturation (the one trade-off). All correctness runs pass. CUB's RAKING policy and a
single-warp serial scan were also measured and both lose at both bin widths.

## 1. Why this stage, and why it was stuck

The L0-vs-L6 phase profile (`proto_air_ablate prof`) showed that after all other optimizations,
the fused scan+choose is the optimized kernel's dominant invariant: **~377 cyc per pass, 42% of
the whole call, unchanged (±20) by every other change**. Its dependency structure under
`BLOCK_SCAN_WARP_SCANS` explains the number:

```
LDS count (33)
  → 5-shfl inclusive warp scan (~150, strict serial chain)
  → lane 31 posts the warp AGGREGATE (= the scan's LAST output)
  → __syncthreads (releases only after the slowest warp's full scan, ~+28)
  → per-thread fold of the 8 warp aggregates (LDS + adds, ~63, serial tail)
  → crossing test → state write → closing barrier
```

Every component is individually near its floor (a 32-lane prefix needs ≥5 dependent shuffle
steps; the barrier is 28 cyc; the fold needs the aggregates). The waste is *structural*: the
aggregate sits at the **end** of the scan chain, so the barrier releases late and the fold
becomes a serial tail that nothing overlaps.

## 2. The insight

The warp aggregate — the only cross-warp datum — **does not need the scan**. It is a plain warp
sum, and sm_80+ hardware computes that in one instruction: `redux.sync.add.u32` ≈ 22 cyc (from
the original building-block microbenchmarks). Producing it first inverts the structure:

```
LDS count (33)
  → redux aggregate (22) → post → __syncthreads   (barrier releases ~120 cyc EARLIER)
  → in-warp 5-shfl scan (~150)  ⟍  these two run interleaved: the fold's loads/adds are
  → fold of 8 aggregates (~63)  ⟋  independent of the shuffles and fill their stall slots
  → crossing test → state write → closing barrier
```

The scan itself is not made faster — the fold tail and most of the barrier wait are moved into
the scan's shadow. In code:

```cpp
const unsigned cnt  = hist[tid];
const unsigned wsum = __reduce_add_sync(0xffffffffu, cnt);   // aggregate WITHOUT the scan
if (lane == 0) { agg[warp] = wsum; }
__syncthreads();                                             // early release
unsigned base = 0;                                           // cross-warp base: independent of
#pragma unroll                                               // the shuffles; the scheduler
for (int w = 0; w < 7; ++w) { base += (w < warp) ? agg[w] : 0u; } // interleaves it into their
unsigned incl = cnt;                                         // stall slots
#pragma unroll
for (int d = 1; d < 32; d <<= 1)
{
  const unsigned o = __shfl_up_sync(~0u, incl, d);
  if (lane >= d) { incl += o; }
}
incl += base;                                                // fused crossing test follows (L1)
```

A subtlety that makes this legal at all: barriers gate *issue order per warp*, so the only way
to overlap the fold with the scan is for the barrier to come **before** both — which requires
the aggregate to exist before the scan. `redux.sync` is exactly the primitive that breaks the
false dependency.

## 3. What else was considered

* **Exploit k ≤ 16 sparsity** (the crossing bucket lies within the first ≤ k nonzero buckets):
  needs cumulative sums, not positions — no cheap ballot shortcut exists.
* **Binary search on the bucket index with block-wide counts**: ~8 rounds × ~80 cyc — worse
  than any scan.
* **Fine-scan only in the crossing warp** (coarse phase first): serializes the fine scan
  *after* the coarse result; speculative scanning in all warps is free — which is AF.
* **Shorter shuffle chains via masked-group redux**: group sums via masked `redux` + two-level
  combine ≈ 170 cyc chain vs 150 — no win; the 5-step depth is a real floor for a 32-prefix.
* Measured controls: **`BLOCK_SCAN_RAKING`** (two barriers + serial raking: worse for latency)
  and a **single-warp serial scan** (the hist_narrow approach: one warp does everything while
  7 wait — confirmed loser again).

## 4. Methodology

`proto_scan_choose.cu` isolates the stage: a histogram is built in shared memory once (not
timed) from realistic bin streams (top digits of twiddled normal floats; `tie_heavy`; uniform
bins), then the scan+choose implementation is chained with the slope method (chains 1..32, min
of 24 reps). Serialization between calls is enforced by adding a **laundered zero** derived
from the previous call's crossing bucket to every histogram index, so call *n*'s first LDS
depends on call *n−1*'s result. A **no-scan floor probe** (histogram LDS + state write +
closing barrier only) measures the peripherals so the scan's *net* cost can be compared.
Each implementation's crossing state (bucket, candidates, selected) is validated against a
host-computed reference per pattern. End-to-end integration reuses the ablation suite
(`AirAblate<..., AFSCAN=true>`), inheriting its full correctness matrix.

## 5. Results

### Isolated stage (slope cyc/call; identical across input patterns)

| implementation | (a) 256 bins, 1 bin/thread | net of floor | (b) 2048 bins, 8 padded bins/thread | net |
|---|---|---|---|---|
| cub WARP_SCANS fused (incumbent) | 403 | 280 | 558 | 377 |
| **aggregate-first (AF)** | **334** | **211 (−25%)** | **525** | **344 (−9%)** |
| cub RAKING fused (control) | 477 | 354 | 720 | 539 |
| 1-warp serial scan (reference) | 428 | 305 | — | — |
| no-scan floor (peripherals) | 123 | — | 181 | — |

### End-to-end (ablation suite, L6 vs L6+AF; all correctness runs pass)

| config | random | tie_heavy | pivot_tie40 | sorted | G elem/s | regs | smem B | blk/SM |
|---|---|---|---|---|---|---|---|---|
| f32+i32 L6 | 1817 | 1601 | 3221 | 1751 | 440 | 48 | 8368 | 5 |
| **f32+i32 L6+AF** | **1689** | **1486** | **2952** | **1639** | 426 | 48 | 8400 | 5 |
| u32+i32 L6 | 1809 | 1594 | 3237 | 1738 | 450 | 48 | 8368 | 5 |
| **u32+i32 L6+AF** | **1637** | **1430** | **2907** | **1580** | 439 | 48 | 8400 | 5 |

The isolated per-pass gains translate almost exactly (f32 random: 2 passes × −69 ≈ −128
measured; u32 pivot: 4 × −82 ≈ −330 measured) — a good indication the isolation harness is
faithful.

## 6. Insights

1. **The win is structural, not arithmetical.** AF executes strictly *more* work (an extra
   redux) yet is 69 cyc faster per pass: it removes a false dependency (aggregate ← scan) that
   forced ~120 cyc of barrier wait + fold to run serially after the scan.
2. **Why (b) gains less than (a):** with 8 bins/thread, the local tree-sum (~100 cyc of loads
   and adds) gates *both* the aggregate and the scan. The barrier can only move as early as the
   tree allows, shrinking the hideable window from ~120 to ~35 cyc. Wide-digit configurations
   keep the pass-count advantage as their only lever — consistent with the earlier radix-width
   sweep where R11 lost overall.
3. **The one cost is saturated-throughput (~2-3%):** at one block per SM the redux's issue
   slots are free (the warp would be stalling anyway); at full occupancy they compete with
   other resident blocks. Latency and throughput optima genuinely diverge here — a policy
   split, not a bug.
4. **Scan-strategy controls confirm the frontier:** RAKING (+74/+162 vs incumbent) and 1-warp
   (+25) lose; among structures that keep all warps scanning, the shuffle chain + one early
   barrier + hidden fold is the best found, and its remaining depth (LDS 33 + redux 22 + bar 28
   + scan 150 + test/state ≈ 250 + closing bar) is within ~25% of the measured 334 — little
   headroom left short of changing the algorithm.

## 7. Recommendation & applicability

Integrate AF as the scan for the latency-optimized path (or default, if ~3% saturated
throughput is acceptable). It composes with the L1 fusion (same fused crossing test), applies
to **all key types and both keys-only/pairs modes**, scales its benefit with the number of
executed passes (best for f64 keys and tie floods, least for f16), costs +32 B shared memory
and nothing in registers/occupancy, and requires **sm_80+** (`redux.sync`); older architectures
keep the `BlockScan` path. A natural implementation shape upstream is a
`BLOCK_SCAN_WARP_SCANS_AGGREGATE_FIRST`-style specialization or a local scan inside
`block_topk_air`/`block_topk_sieve_air`, gated on `__CUDA_ARCH__ >= 800`.

Updated best-known latency for (256 threads, N=1024, K=16, pairs, B200):
**f32 1689 cyc / u32 1637 cyc** — 35-37% below the shipping header (2601/2514 on this node).

## 8. Should this be a new `BlockScanAlgorithm` specialization?

Yes — as an **explicit, opt-in policy** (e.g. `BLOCK_SCAN_WARP_SCANS_AGGREGATE_FIRST`), not as a
silent change to `BLOCK_SCAN_WARP_SCANS`. The reasoning:

**Why it fits the BlockScan policy model.**
* The `BlockScanAlgorithm` enum exists precisely to encode structure/trade-off variants
  (RAKING vs RAKING_MEMOIZE vs WARP_SCANS); AF is a fourth point on that frontier
  (latency-optimal for redux-eligible scans).
* All the machinery the specialization needs already exists in CUB:
  `is_warp_redux_op_supported<ScanOp, T>` and `cub::detail::warp_redux`
  (`warp_redux.cuh`, already used by `WarpReduce` and `WarpReduceBatched`), and
  `BlockScanWarpScans`' `warp_aggregates[WARPS]` storage — AF reorders who writes it and when,
  with **no TempStorage growth** (unlike in the top-k prototype, where the +32 B bought a
  storage array BlockScan already has).
* Bonus for some callers: the **block aggregate becomes available early** (right after the
  first barrier) instead of after the full scan — `ExclusiveSum(..., block_aggregate)` users
  on a latency path get that for free.

**Why opt-in, with three hard conditions:**
1. **Eligibility-gated with graceful fallback.** The restructure only wins when the warp
   aggregate is much cheaper than the warp scan — i.e. when (ScanOp, T) maps to a `redux.sync`
   instruction (Sum/Min/Max/And/Or/Xor on 32-bit integers) on sm_80+. For a generic operator
   the aggregate costs a full 5-shfl warp reduction — as expensive as the scan it would
   bypass, so nothing is gained. The specialization should `if constexpr`-dispatch on
   `is_warp_redux_op_supported` (+ arch) and otherwise compile to plain WARP_SCANS. This is
   exactly the dispatch pattern `warp_reduce_shfl.cuh` already uses.
2. **Documented as latency-oriented.** At full occupancy the extra redux competes for issue
   slots and cost ~2-3% throughput in our end-to-end context. CUB's device-wide algorithms
   (scan, radix-rank, partition) are throughput-tuned around WARP_SCANS — making AF the
   default inside WARP_SCANS would silently regress them. As a named policy the trade is the
   caller's informed choice, mirroring how `BLOCK_SCAN_RAKING_MEMOIZE` documents its own trade.
3. **Bit-exactness holds on the eligible set by construction.** Integer Sum/Min/Max are
   associative and commutative, so the redux-computed aggregate equals the scan's last lane
   exactly; float never qualifies (`redux.sync` has no float variants), so no
   summation-order-changes-rounding hazard can arise.

**Scope notes for the implementation:** the full BlockScan surface (exclusive/inclusive,
block-aggregate out, `BlockPrefixCallbackOp`, ITEMS_PER_THREAD > 1) composes with the reorder
without structural issues; for ITEMS_PER_THREAD > 1 the thread-local reduction gates both paths
and the gain decays (measured at the 2048-bin/8-items point: −9% net vs −25%), which the docs
should state. Test cost is the standard new-algorithm matrix.

**Suggested staging:** land the win first as a `detail::` scan inside `block_topk_air` /
`block_topk_sieve_air` (smallest blast radius, covers the motivating users including PR #9066's
sieve), then promote to the public `BLOCK_SCAN_WARP_SCANS_AGGREGATE_FIRST` policy — the
promotion is mechanical given the enum and the existing redux-eligibility infrastructure, and
the top-k usage becomes a one-line policy switch.

Reproduce: `./run_remote.sh proto_scan_choose.cu` and
`./proto_air_ablate [correct|lat|thr|res]` on branch `exp/scan-choose-opt`.
