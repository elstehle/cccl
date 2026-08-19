# Stable warp sort via WarpBitonicSort + wrappers vs the hand-rolled network

Does the real `WarpBitonicSort` (branch `exp/sub-warp-bitonic-sort` @ 7d9813a2a3) with a stability
wrapper beat the hand-rolled stable network from the hybrid-block-sort prototype — and what does
stability cost on top of the unstable original? Harness `proto_stable_wrap.cu`,
**umbriel-b200-022, B200 (sm_100, 148 SMs)**, full warp (LW=32), sizes 32..256 (IPT 1..8), float
keys / int values, input order = striped position. All methods pass correctness; the three stable
ones pass exact stability vs `std::stable_sort` under heavy ties.

Methods: **mine** = hand-rolled network, separate rank channel (the hybrid prototype's bootstrap);
**cub+packed** = `WarpBitonicSort<u64>` on an order-preserving bit-twiddled `(key32 << 32) | rank`
pack, single integer compare (the "builtin type + builtin comparator" fast path);
**cub+wrapper** = `WarpBitonicSort<{KeyT,rank}>` with a generic two-call lexicographic comparator
(arbitrary types, no `==`/twiddle assumed); **cub-unstable** = `WarpBitonicSort<float>` as-is.

## Latency (single-warp slope cyc/sort)

| | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| keys mine | 1559 | 3765 | 9542 | 23346 |
| keys **cub+packed** | **625** | 961 | 2343 | **4731** |
| keys cub+wrapper | 711 | 1082 | **1904** | 5872 |
| keys cub-unstable | 638 | **879** | 1959 | 5449 |
| pairs mine | 1554 | 3773 | 9615 | 23348 |
| pairs **cub+packed** | **625** | **1090** | 3109 | 8475 |
| pairs cub+wrapper | 890 | 1561 | 3574 | 9612 |
| pairs cub-unstable | 609 | 1187 | **2239** | **6110** |

## Throughput (one occupancy wave, Gelem/s)

| | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| keys mine | 110 | 90 | 68 | 50 |
| keys cub+packed | 195 | 157 | 120 | 100 |
| keys cub-unstable | 257 | 205 | 169 | 139 |
| pairs mine | 101 | 83 | 62 | 47 |
| pairs cub+packed | 130 | 102 | 77 | 60 |
| pairs cub-unstable | 203 | 163 | 136 | 113 |

(cub+wrapper trails cub+packed on every throughput row: 120/94/74/58 keys, 97/68/46/35 pairs.)

## Findings

1. **The real WarpBitonicSort beats the hand-rolled network 2.4×–4.9×** (latency, growing with
   size) — the projected ~3× implementation-quality gap confirmed and exceeded. Every future
   network use goes through `WarpBitonicSort`; the hand-rolled one retires.
2. **Stability via the twiddle-pack is nearly free in the latency regime**: keys 625 vs 638 at
   size 32 (free), +9% at 64, and at 256 packed is even *faster* than the unstable float sort
   (4731 vs 5449 — the single u64 compare/select codegen appears to beat the float
   compare path at scale; logged as a pleasant codegen anomaly). Pairs: +3% to +39%.
3. **In the throughput regime stability costs real bandwidth**: the u64 pack doubles key shuffle
   traffic → ~25–30% keys, ~35–45% pairs below unstable. Still 1.3–2× above the hand-rolled
   network everywhere.
4. **Packed beats the generic wrapper** at 6 of 8 latency points (up to 31%) and on every
   throughput row (up to 65%) — the production shape is a `stable_sort` helper that picks the
   twiddle-pack for radix-twiddleable KeyT + builtin comparators and falls back to the
   (key, rank) struct + two-call comparator otherwise. Both are pure wrappers over the existing
   `WarpBitonicSort`; a `STABLE` template parameter is API sugar over exactly this dispatch.
5. **API-hardening note for the branch**: calling the keys-only `Sort(keys, op)` on a
   `ValueT != NullType` instantiation dereferences the internal `nullptr` values pointer
   (manifested as silently zeroed keys). It should be rejected at compile time
   (`static_assert(keys_only)`) or made to ignore values.
6. **Hybrid re-projection** (from V4 = capped-bootstrap hybrid, HYBRID_BLOCK_SORT_RESULTS.md):
   substituting cub+packed for the hand-rolled bootstrap gives estimated block-sort latencies of
   ~4.0k / ~6.0k / ~10.1k for tiles 512/1024/2048 vs stock's 5.9k / 7.8k / 11.7k — i.e. the
   capped hybrid projects to **win at every IPT** (−32% / −24% / −14%), not just IPT 1. Needs the
   real measurement: wire `WarpBitonicSort` + packed wrapper into V4's bootstrap slot (headers
   now available on-node at /cccl_bit).

## Measurement-methodology notes (for the archive)

Two harness traps caught during this study: (a) partial output consumption (2 of IPT elements)
let the compiler eliminate register-only cub sorts — consume every output through the sink;
(b) the first "elimination" was actually finding 5: keys-only calls on ValueT=int instantiations
were nullptr-UB, caught only once keys-only correctness coverage was added. Rule: every
(method × PAIRS) instantiation that gets timed must also be correctness-tested.

Reproduce: `nvcc -std=c++17 -arch=sm_100 -O3 -I<branch>/cub -I<branch>/libcudacxx/include
-I<branch>/thrust proto_stable_wrap.cu && ./proto_stable_wrap [correct|lat|thr|all]`.
