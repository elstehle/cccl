# WarpBitonicSort max-throughput sweep: unstable vs stable (twiddle-pack), sizes 4..512

Max achievable compute throughput of `WarpBitonicSort` (branch `exp/sub-warp-bitonic-sort`),
unstable vs stable-via-u64-twiddle-pack, segment sizes 4..512. **umbriel-b200-022, B200
(sm_100, 148 SMs)**; harness `proto_wbs_maxthr.cu`. Sizes 4-32 = sub-warp logical warps (IPT 1),
64-512 = LW 32 with IPT 2-16. Data generated in registers and fully sink-consumed: these are
**pure compute/issue ceilings** (no memory traffic) — a real pipeline adds its load/store on top.
Grids = {1, 4, 16} occupancy waves (64 iters/thread) to expose and amortize the tail effect;
block 128 and 256 both measured; best of 5 event timings. All 32 timed instantiations pass
correctness, stable-pairs additionally vs `std::stable_sort` under heavy ties.

## Peak throughput (16 waves, best block, Gelem/s)

| size | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|
| unstable keys | 395 | **558** | 400 | 291 | 220 | 180 | 149 | 123 |
| stable keys | 377 | 432 | 299 | 216 | 165 | 126 | 107 | 80 |
| unstable pairs | 379 | 463 | 312 | 226 | 172 | 142 | 121 | 92 |
| stable pairs | 355 | 296 | 194 | 138 | 104 | 81 | 65 | 47 |
| stability tax, keys | −5% | −23% | −25% | −26% | −25% | −30% | −28% | −35% |
| stability tax, pairs | −6% | −36% | −38% | −39% | −39% | −43% | −46% | −49% |

## Findings

1. **The peak is at size 8, not size 4**: 558 Gelem/s unstable keys. At size 4 the network is so
   cheap that harness generation/consumption overhead bounds the number (which also masks the
   stability tax there, −5%); size 8 is the genuine sweet spot. From 8 upward throughput falls
   with the O(log²n) per-element work — but slower than the stage count grows (45/6 stages from
   512/8 vs only 4.5× throughput drop), because register-local exchange stages get cheaper than
   shuffle stages as IPT grows.
2. **Tail effects are real and size-dependent — the wave sweep was warranted**: 1→16 waves gains
   **+25-31% at size 8**, +18% at 16, shrinking to +5-9% at 512. Single-wave numbers (as in the
   earlier studies) understate small-segment throughput substantially; multi-wave saturated
   numbers are the honest ceiling.
3. **Block size is irrelevant at saturation**: 128 vs 256 agree within ~1-3% at 16 waves
   (256 slightly better at 1 wave). Occupancy in blk/SM halves but threads/SM match.
4. **The stability tax has two regimes.** Through size 64 it is pure shuffle traffic: ~25% keys
   (the u64 pack doubles key words per exchange), ~36-39% pairs. From size 128 up, **register
   pressure compounds it**: stable configs lose occupancy (keys: 8→5→3 blk/SM at 256/512;
   pairs: 6→4→3), reaching −35% keys / −49% pairs at 512. The first spill appears exactly at
   the corner: stable pairs @ 512 = 80 regs + 8 B spill.
5. **Context**: even the worst stable-pairs number (47 Gelem/s @ 512) exceeds the measured
   WarpMergeSort throughput at comparable sizes, and unstable keys at small segments (395-558)
   is ~2-5× everything measured in the warp-sort studies — `WarpBitonicSort` is the throughput
   ceiling-setter for warp-scope sorting on B200.
6. Productization notes for the `STABLE` mode this feeds: (a) document the tax as
   latency-cheap / bandwidth-priced (~25% keys, ~40% pairs at saturation); (b) for ≤22-bit keys
   a 32-bit (key<<10|rank) pack would eliminate the extra shuffle word entirely — worth a
   specialization; (c) the register growth at IPT ≥ 8 argues for capping stable network use at
   ≤ 256 elements per segment (consistent with the hybrid block-sort gate).

Reproduce: `nvcc -std=c++17 -arch=sm_100 -O3 -I<branch>/cub -I<branch>/libcudacxx/include
-I<branch>/thrust proto_wbs_maxthr.cu && ./proto_wbs_maxthr [correct|thr|res|all]`.
