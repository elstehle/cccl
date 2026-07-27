# WarpMergeSort latency optimization — cycle breakdowns, prototypes, findings

Single warp (32 threads), keys-only `float`, sizes 32,64,96,128,160,192,256,320,384 (= IPT
1,2,3,4,5,6,8,10,12). Measured on **NVIDIA B200 (sm_100)**, CUDA 13.1, node `umb-b200-261`.
Latency = chain-length-slope cycles/call. Merge sort is **data-dependent** (does less work on
already-sorted input), so per the `DEVICE_SIDE_BENCHMARKING_ISSUE.md` methodology we feed fresh
random input per call, serialize the chain through the previous output (a real RAW dependency),
and subtract a generate-only control so the reported number is the sort's marginal cost. Code:
`proto_wms_bench.cu` (real `cub::WarpMergeSort` baseline + instrumented mirror), `proto_wms_opt.cu`
(policy-templated ablation). All variants validated == `cub` == `std::sort` on every size.

## TL;DR — recommendation

**Statically unroll the MergePath diagonal search** (like `StaticUpperBound` in
`block_run_length_decode.cuh`). It is the single largest lever: **−420 to −960 cyc (15-25%)** on
every size, because the data-dependent `while`-loop binary search diverges across the warp (each
lane runs a different number of iterations) and its reconvergence stall sits on the critical
path. A fixed trip count keeps the warp converged. Combined with an optimal per-thread sorting
network (helps only at large IPT), the mirror gets **−10% (IPT 8) to −32% (IPT 1)**:

| size | IPT | baseline | **MP+NET** | speedup | Δregs | smem |
|---|---|---|---|---|---|---|
| 32  | 1  | 2219 | **1511** | −32% | +15 | 0 |
| 128 | 4  | 3613 | **2992** | −17% | +9  | 0 |
| 256 | 8  | 5143 | **4628** | −10% | +7  | 0 |
| 384 | 12 | 6658 | **5504** | −17% | +12 | 0 |

**Bank-conflict padding is a measured *loss* (−negative, +57..+581 cyc)** in this regime — see §5.

## 1. Baseline latency (real `cub::WarpMergeSort`, slope cyc/call)

| size | 32 | 64 | 96 | 128 | 160 | 192 | 256 | 320 | 384 |
|---|---|---|---|---|---|---|---|---|---|
| IPT | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 |
| cyc | 2236 | 3460 | 4008 | 4569 | 5161 | 5541 | 6460 | 7376 | 8269 |

Cost is roughly linear in IPT above IPT=2 (~470 cyc/item) with a ~2000-cyc floor (5 merge rounds
× fixed per-round overhead). The instrumented mirror reproduces these within ~1% at small IPT.

## 2. Where the time goes (instrumented mirror, clock64 phase stamps)

Approximate — the 17 stamps add `__syncwarp`s that inflate the divergent MergePath phase — but
the shape is unambiguous (cycles, min of 32 reps):

| size | thread-sort | STS (store) | MergePath | SerialMerge | note |
|---|---|---|---|---|---|
| 32  (IPT 1)  | 44  | 220 | 1819 | 504  | MergePath + SerialMerge = 90% |
| 128 (IPT 4)  | 69  | 140 | 2471 | 957  | |
| 256 (IPT 8)  | 221 | 181 | 3030 | 1865 | thread-sort now non-trivial |
| 384 (IPT 12) | 460 | 199 | 3506 | 2640 | thread-sort ~7% |

Three structural facts:

1. **The two shared-memory-resident merge phases (MergePath + SerialMerge) dominate — ~80-90%.**
   Both are *dependent chains of shared loads*: MergePath's next probe address depends on the
   previous probe's comparison; SerialMerge's next read depends on which side advanced. The
   primitive is **shared-memory-latency bound, not compute or bandwidth bound.**
2. **MergePath is disproportionately expensive** relative to its ~log₂(size) probes because the
   `while` loop's trip count is data-dependent → the warp diverges and pays reconvergence.
3. **Thread-local StableOddEvenSort grows as O(IPT²)** (44→460 cyc), crossing STS around IPT=6
   and becoming a real term at IPT≥8.

STS (the store to shared) is minor and flat; the per-round barrier + store latency is fixed.

## 3. Per-change ablation (clean slope, one policy at a time)

`proto_wms_opt.cu` toggles each optimization independently on the mirror (no stamps → no
perturbation). Δ vs baseline, cyc/call:

| size | IPT | base | +MP (static search) | +NET (Batcher) | +PAD (bank pad) | **MP+NET** |
|---|---|---|---|---|---|---|
| 32  | 1  | 2219 | **−708** | +0   | +57  | 1511 (**−32%**) |
| 64  | 2  | 2651 | **−614** | −10  | +195 | 2033 (−23%) |
| 96  | 3  | 3103 | **−710** | −2   | +291 | 2390 (−23%) |
| 128 | 4  | 3613 | **−603** | −19  | +255 | 2992 (−17%) |
| 160 | 5  | 4077 | **−869** | −23  | +367 | 3206 (−21%) |
| 192 | 6  | 4379 | **−818** | −38  | +369 | 3525 (−20%) |
| 256 | 8  | 5143 | **−421** | −93  | +235 | 4628 (−10%) |
| 320 | 10 | 5919 | **−963** | −145 | +581 | 4828 (−18%) |
| 384 | 12 | 6658 | **−941** | −215 | +578 | 5504 (−17%) |

## 4. MergePath static unroll — the big win

The baseline diagonal search is a data-dependent `while` loop:

```cpp
while (b < e) {
  const int mid = (b + e) >> 1;
  if (keys2[diag-1-mid] < keys1[mid]) e = mid; else b = mid + 1;
}
```

Different lanes hit `b >= e` after different iteration counts, so the warp diverges and stalls on
reconvergence — on top of the dependent shared-load chain. The search range per merge round is
`IPT * 2^round`, a **compile-time** bound (the round loop unrolls), so the iteration count is
statically known. Replacing the loop with a fixed `ceil(log2(range+1))` trip count (predicated
body, exactly the `StaticUpperBound` pattern) keeps all lanes converged:

```cpp
template <int RANGE>
__device__ int mp_static(const float* sh, int k1b, int k2b, int c1, int c2, int diag) {
  int b = diag < c2 ? 0 : diag - c2;
  int e = min(diag, c1);
  #pragma unroll
  for (int i = 0; i <= cub::Log2<RANGE + 1>::VALUE; ++i) {   // fixed count => no divergence
    const int mid = (b + e) >> 1;
    const bool go = b < e;
    const bool up = go && (sh[k2b + diag-1-mid] < sh[k1b + mid]);
    e = up ? mid : e;
    b = (go && !up) ? mid + 1 : b;
  }
  return b;
}
```

**−420 to −960 cyc, every size.** It doesn't remove the dependent-load chain (inherent to binary
search) but it removes the divergence and loop overhead riding on top of it. Applicability: any
merge round whose run length is a compile-time bound — true for the fixed-tile warp/block merge
sort. The device (`dispatch_merge`) partitioning search has runtime bounds and would keep the
loop, or bound-and-cap.

## 5. Bank-conflict padding — a measured negative result

The blocked layout `keys_shared[IPT*lane + item]` is strided by IPT, so for IPT≥2 the STS and the
SerialMerge LDS have bank conflicts. Padding one dead word every 32 elements (`idx + idx/32`) to
rotate banks **made every size slower (+57 to +581 cyc).** The reason is instructive and matches
§2: in the single-warp latency regime there is no throughput pressure, so **bank conflicts (a
bandwidth phenomenon) are not on the critical path** — the bottleneck is dependent-access
*latency*. Padding only adds an address-computation term (`idx>>5`, an extra add) to every shared
access and shifts nothing that matters. Conflict-avoidance would help the *throughput* regime
(many warps saturating the SM) but not latency; it should be a throughput-only policy.

## 6. Per-thread sorting network — small, IPT-dependent

`StableOddEvenSort` (odd-even transposition) does ~IPT²/2 compares serially. A Batcher odd-even
mergesort network (branchless `fminf`/`fmaxf`) does fewer for large IPT. Measured (+NET column):
negligible for IPT≤6 (thread-sort is a tiny fraction there), growing to −93/−145/−215 at
IPT=8/10/12. Note the Batcher network here pads non-power-of-2 IPT up to the next power of two
with +∞ sentinels (Batcher is only correct for power-of-2 lengths); a production impl would use a
per-size *optimal* comparator network (e.g. 39 comparators for n=12 vs odd-even's 66) and capture
a larger share. Even the padded version never regresses. Register cost: the branchless network
raises the register count (part of the +7..+16 in §7).

## 7. Resources (lat kernel, base vs MP+NET)

No shared-memory change (both use the `TILE`-word exchange); no spills at any size. The static
search (unrolled, more live index state) and the branchless network raise registers by **+7
(IPT 8) to +16 (IPT 1)** — from ~46-59 to ~60-71. At single-warp occupancy this is free; for a
throughput kernel packing many warps it is the one cost to weigh, and argues for gating NET / the
static-search unroll depth behind a latency-vs-occupancy policy.

## 8. Recommendation & next steps

1. **Static-unrolled MergePath** — the clear win, all sizes, zero smem, portable to any
   compile-time-tile merge sort. Recommended for `block_merge_sort.cuh`'s `MergePath` when the
   run length is a compile-time bound (warp/block merge sort's fixed tile); the device merge
   partitioning keeps the runtime loop. This is the first thing to upstream.
2. **Optimal per-thread networks** for large IPT — modest, worth it where IPT≥8 and registers
   allow; use true optimal networks (not padded Batcher) for non-power-of-2 IPT.
3. **Do not** add bank-conflict padding for the latency regime; reserve conflict-avoidance for
   the throughput path.
4. Next: port the static MergePath into the real `block_merge_sort.cuh` and confirm the win on
   `cub::WarpMergeSort`/`cub::BlockMergeSort` directly (the mirror tracks cub within ~1%, so the
   ~15-25% is expected to carry), then measure the throughput regime and the pairs (key+value)
   variant, where the extra value-exchange rounds change the phase balance.

Reproduce: build `proto_wms_bench.cu` / `proto_wms_opt.cu` with the standard nvcc line and run
`[correct|lat|prof]` / `[correct|lat|res]`.
