# WarpRadixSort (ballot-ranked LSD) vs WarpMergeSort vs WarpBitonicSort

Comparison of a user-provided warp-scope radix sort (`warp_radix_sort.cuh`: stable LSD,
ballot-based ranking, RadixBits=5 → **7 passes for 32-bit keys**, pointer interface with a
double buffer) against `cub::WarpMergeSort` and `cub::detail::WarpBitonicSort`. Keys-only float,
single-warp scope, sizes 32..384 (IPT 1..12). Node **umb-b200-239, B200 (sm_100), CUDA 13.3.1**;
harness `proto_warp_radix.cu`, methodology as `WARP_MERGE_SORT_RESULTS.md` §13 (merge & radix:
back-to-back random-input chain, generate-only control subtracted; bitonic: in-place dependency
chain + `sink()`; radix pays its reg→smem→reg wrapping, i.e. the same "registers in, registers
out" contract). All three pass correctness vs `std::sort` on every size; the merge/bitonic
columns reproduce the §13 study within ~1%.

## Latency (single-warp slope, cyc/call)

| size | 32 | 64 | 96 | 128 | 160 | 192 | 256 | 320 | 384 |
|---|---|---|---|---|---|---|---|---|---|
| IPT | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 |
| WarpMergeSort | 2256 | 3479 | 4033 | 4596 | 5185 | 5566 | 6489 | 7400 | 8295 |
| WarpBitonicSort | **660** | **1080** | **1710** | **1964** | **3021** | **3792** | **5468** | **7510*** | **8934*** |
| **WarpRadixSort** | 1383 | 3676 | 5219 | 6866 | 8353 | 9925 | 12307 | 15317 | 18188 |
| radix vs best | 2.1× | 3.4× | 3.1× | 3.5× | 2.8× | 2.6× | 2.3× | 2.1× | 2.2× |

\*merge retakes the lead from bitonic at IPT ≥ 10 (§13); radix trails both everywhere.

Radix is almost perfectly linear at **~1530 cyc per item-per-thread** on top of a ~1400-cyc
floor — 3× the merge slope. The cost model explains it exactly: 7 passes × IPT strided
iterations, each iteration a **dependent chain of warp-collective ops** (5 ballots for the
histogram in phase 1; 5 ballots feeding two masks + popc + shfl + LDS/STS in phase 3, where the
running scatter-pointer update serializes consecutive iterations of a pass). Warp-collective
issue latency, not memory, is the bottleneck. One bright spot: **at size 32 (IPT 1) radix beats
merge sort by 1.6×** (1383 vs 2256) — with one item per lane the ballot ranking is a genuinely
cheap ranking machine, and only bitonic (660) is faster.

## Throughput (one occupancy wave, Gelem/s)

| size | 32 | 64 | 96 | 128 | 160 | 192 | 256 | 320 | 384 |
|---|---|---|---|---|---|---|---|---|---|
| WarpMergeSort | 64 | 107 | 132 | 119 | 159 | **162** | 97 | **181** | **148** |
| WarpBitonicSort | **260** | **210** | **177** | **174** | 152 | 151 | **143** | 124 | 124 |
| **WarpRadixSort** | 41 | 35 | 36 | 36 | 37 | 37 | 38 | 37 | 38 |
| radix vs best | 0.16× | 0.17× | 0.20× | 0.21× | 0.23× | 0.23× | 0.26× | 0.21× | 0.25× |

Radix saturates at a **flat ~37 Gelem/s ceiling, 4-6× below the alternatives**. The flatness is
the tell: work per element is constant (~7×15/32 warp-collective ops each), and with a full wave
of warps the ballot/shuffle pipes saturate — a per-SM issue-throughput ceiling that IPT cannot
amortize. The 2×N-word double buffer (vs N for merge, 0 for bitonic) additionally costs
occupancy.

## Verdict and where this design *would* pay

For 32-bit keys at warp scope, **ballot-ranked LSD radix is not competitive on either axis**:
the 7-pass structure multiplies a per-iteration collective-op chain that costs more than the
comparison sorts' entire work. But the structure has real niches:

1. **Short keys.** Passes scale with `ceil(KeyBits/5)`: 16-bit keys → 4 passes (~1.75× faster),
   8-bit → 2 passes (~3.5×). At IPT 1 float it already beats merge; a `__half` variant at small
   IPT would plausibly approach bitonic while being **stable** — which bitonic is not.
2. **Stability at warp scope** currently costs a merge sort; radix is the only other stable
   option here and needs no comparison operator.
3. **Key-value pairs**: the rank/scatter separation extends to values for one extra store/load
   per pass, while bitonic's pair cost grows per compare-exchange stage.

## Implementation review notes (independent of performance)

* **Missing `__syncwarp()` between scatter passes**: phase 3's pass p+1 reads shared memory
  written by other lanes in pass p (and phase 3 reads what phase 1 read) with only ballot/shfl
  rendezvous in between. Ballots synchronize *execution*, but per the CUDA memory model they are
  not shared-memory fences; the tests pass on sm_100, but a `__syncwarp()` per pass boundary is
  required for correctness-by-contract.
* The bit budget is awkward for 32-bit keys: 7×5 = 35 bit-slots for 32 bits — the last pass
  processes 2 meaningful bits at full cost. No better schedule exists under RadixBits ≤ 5, which
  is another way of saying the design wants shorter keys.
* Nice properties worth keeping: single fused histogram scan for all passes (phase 1), one
  5-ballot group feeding both the rank mask and the bin mask in phase 3, zero shared memory
  beyond the key buffers.

Reproduce: `nvcc -std=c++17 -arch=sm_100 -O3 -I<cub> -I<libcudacxx> -I<thrust>
proto_warp_radix.cu` → `./proto_warp_radix [correct|lat|thr]`.
