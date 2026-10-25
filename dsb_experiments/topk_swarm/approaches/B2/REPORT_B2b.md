# B2 (instance opt-B2b): atomics based iterative extraction, packed key-index rounds

Author opt-B2b, 2026-10-25, branch exp/topk-swarm/B2b. B2 ran as two optimizer instances by a launch accident. This instance owns the variants B2_packed, B2_valrounds, B2_spec, B2_spec2, B2_spec3 and B2_spec4 and the cards named below. The other instance (holder opt-B2) owns B2_baseline, B2_twolevel, B2_tiecheck, B2_redux8 and B2_trunc32 and used B2_packed as its control, its report is approaches/B2/REPORT.md. Every number here is B200 (sm_100 build, nvcc 13.4.92, nodes umbriel-b200-091 and umbriel-b200-047), latency in cycles per call net of the harness control, throughput in Gelem/s, random normal input, with the control named in the same lock section. Cards: B2_packed_b200_20261025_1 (baseline against B1, N 1024 to 8192), B2_packed_n4kill_b200_20261025_1 (N4 kill experiment against the value-rounds port), B2_spec_b200_20261025_1 and B2_spec3_b200_20261025_1 (iterations, negative), B2_packed_b200_20261025_2 (N 16384 and 32768 against B1), B2_packed_b200_20261025_3 (128-thread blocks against B1). Raw tables under state/results/opt-B2b/.

## 1. What it is

k rounds of a block-wide maximum. Every thread packs its items into 32-bit words that order by (twiddled key, smaller index first), sorts them descending in registers (Batcher odd-even merge sort from a compile-time table), and in every round posts its head into one shared slot with a fire-and-forget atomicMax, waits at one barrier, reads the slot back and, if it won, shifts its list by one with predicated selects. The k slots are the answer, sorted, and the indices are unpacked from them at the end. bf16 packs the full 16-bit key with a 16-bit inverted index. fp32 packs the top 32 - log2 N key bits with the inverted index (idea N4), then one counting pass compares the tie class of the k-th truncated key with its members among the slots and re-ranks the class by the dropped key bits with the same rounds when they differ (0.1 to 4 percent of random inputs at N 1024 to 8192, host simulation in tie_sim.cpp, 2000 seeds per cell). Source harness/adapters/B2_packed.cuh.

## 2. Contract

Block scope, 128, 256 or 512 threads, blocked input in registers, N = Threads x IPT with IPT a power of two up to 128 (bf16) or 64 (fp32), N up to 65536, k in [1, min(N/2, 256)], keys bf16 or fp32, int32 indices derived from the blocked position, index-only output, both requirements with the same binary. Output sorted by key descending then index ascending. Temp storage k + 1 words. Barriers: 1 after the slot init plus k rounds, fp32 adds 1 for the tie count and, on the tie path, 1 plus the re-ranked rounds. Assumes the caller separates calls by a block barrier. -0.0 is normalized to +0.0 before packing (one FADD per fp32 item, one HADD2.BF16 per bf16 word pair), NaN is twiddled bitwise (positive NaN above +inf, negative below -inf), a negative NaN with all mantissa bits set at the last index packs to the empty-slot word and is never selected. Not in the focus set, documented only.

## 3. Where the time goes

Cost model that held across every cell: cycles = fixed(IPT) + k x round(IPT).

| threads | IPT (N) | round bf16 | round fp32 | fixed bf16 | fixed fp32 |
|---|---|---|---|---|---|
| 256 | 4 (1024) | 83.5 | 85.8 | 45 | 310 |
| 256 | 16 (4096) | 110 | 103 | 274 | 650 |
| 256 | 32 (8192) | 112 to 148 | 102 | 960 | 1650 |
| 256 | 64 (16384) | 162 to 205 | 136 to 227 | 1280 | 2000 |
| 128 | 8 (1024) | 69.5 | 73.5 | 70 | 320 |
| 128 | 32 (4096) | 85 to 91 | 71 to 86 | 540 | 950 |

The round is the dependent chain LDS (read back), ISETP, SEL (new head), CREDUX.MAX, IMAD.U32, one predicated ATOMS.MAX RZ per warp, BAR.SYNC, then the remaining IPT - 1 selects of the shift. The CREDUX is ptxas: it aggregates the warp's 32 same-address fire-and-forget atomics into one redux plus one elected atomic (T26, T27, the mechanism behind T10 and T11's 74-cycle round). The round grows with IPT through the shift selects (about 2 cycles per item at two warps per scheduler) and, at IPT 32 and above, through instruction cache pressure of the unrolled code (the slope rises with k at fixed IPT). The fixed cost is the per-thread sort (5, 63, 191, 543 compare-exchanges at IPT 4 to 64, issue bound) plus the pack, and for fp32 the twiddle, the tie count compares and one more barrier round trip. At IPT 4 the whole prologue hides behind the harness control's own loads and barrier (45 cycles net).

B1 is nearly k-flat, so B2 wins below a crossover k and loses by k x round above it. Crossovers on the linear model (bf16, 256 threads): about k 22 at N 1024, 34 at N 4096, 36 to 40 at N 8192, 45 to 50 at N 16384 (fp32 19, 33, 50, 45). At 128 threads about k 28 at N 1024 and 64 at N 4096.

Measured against B1 in the same lock section (latency, bf16): N 1024 k 4 379 vs 1767 (4.7x), k 16 1380 vs 1882 (1.36x), k 64 5397 vs 2017 (0.37x), k 256 21388 vs 2129 (0.10x). N 4096 k 16 2057 vs 3833 (1.86x), k 64 7290 vs 4411. N 8192 k 16 2758 vs 5868 (2.13x), k 64 9886 vs 6591. N 16384 k 16 3869 vs 9568 (2.47x), k 64 13717 vs 10342. fp32: N 1024 k 16 1680 vs 1905 (1.13x), N 4096 k 16 2299 vs 3710 (1.61x), N 8192 k 16 3306 vs 5971 (1.81x), N 16384 k 16 4243 vs 10386 (2.45x), k 64 loses at every N (0.35x to 0.80x). Throughput follows: bf16 N 1024 k 4 1274 vs 365 Gelem/s, k 16 491 vs 347, k 64 143 vs 321, N 4096 k 16 800 vs 530, N 8192 k 16 906 vs 561, N 16384 k 16 881 vs 525, at 8 blocks per SM for bf16 up to IPT 16 (21 to 32 registers) against B1's 4 to 6.

128-thread blocks (4 warps): bf16 N 1024 k 16 1182 vs B1 at 128 threads 2265 (1.92x), N 4096 k 16 1907 vs 5228 (2.74x), k 64 6254 vs 6074, throughput 663 vs 461 and 932 vs 593. The round drops from 83.5 to 69.5 (cross-session observation, same adapter, canaries identical): a 20-cycle barrier instead of 28 (T11) and 4 aggregated CREDUX per round instead of 8 outweigh the doubled IPT. B2 wants small blocks.

N 32768 (IPT 128): the 1471-entry sort table is not unrolled by nvcc 13.4, the 128-word list goes to local memory (12800-byte stack frame, 11833 LDL) and the call costs about 1M cycles (T34). The sorted per-thread list is the wrong structure there anyway, the other instance's two-level list (B2_twolevel) does 4774 cycles at bf16 N 32768 k 16 and also wins at N 8192 and 16384 for k 4 (961 vs 1420, 988 vs 1919 against B2_packed in its cards).

## 4. Ideas tried, with verdicts

1. Packed bf16 words with inverted index, k atomic rounds (B2_packed, baseline). Win at k <= 16 to 32 for every N up to 16384 (1.4x to 4.9x in latency, 1.4x to 3.5x in throughput), loss at k >= 64. Keep.
2. fp32 through idea N4 (truncated key plus index in one word, exact tie finish). Kill experiment against the July atomic_adaptive design ported to the harness (B2_valrounds, value-only rounds plus value-based index resolution, not_guaranteed only): N 1024 k 16 1680 vs 2103 (20 percent fewer cycles, the bar was 15), k 4 650 vs 958, N 4096 k 16 2301 vs 3067, N 8192 k 16 3319 vs 4980. Part 1 (host simulation): finish on 0.5 to 2.3 percent of inputs at N 1024 to 4096 k 16, 4 percent at N 8192, tie classes at most 5, 20 to 50 percent at N 32768 with 17 key bits. Verdict: N4 kept, the value-rounds epilogue retired for fp32 at N <= 16384. The other instance's B2_tiecheck improves the N4 tie count further (1572 vs 1680 at N 1024 k 16, 2761 vs 3306 at N 8192 k 16 against B2_packed).
3. Redux off the dependent chain, posting from the lanes that hold the warp's stale maximum and second maximum with per-lane addresses so that ptxas does not aggregate (B2_spec). Round 138 vs 83.5 at IPT 4 (N 1024 k 16 2252 vs 1380): the barrier then waits for 256 shared atomic operations instead of 8. Verdict: negative, the aggregation is the cheap form (T27).
4. The same with one atomic per warp and a two-level redux refresh (B2_spec2): failed the oracle on sorted input (the one-round-deferred tail shift leaves the winner's second item stale when a lane wins twice in a row). Not measured. Verdict: unsound as written, superseded.
5. One STS per warp of a speculative warp maximum into a per-warp word, 8-way max after the barrier, three redux per round whose inputs do not depend on the read back, no atomics, no init barrier (B2_spec3). Round 137 vs 83.5 (N 1024 k 16 2200 vs 1380). ptxas moves CREDUX across BAR.SYNC in both directions and places the UR to R move right after each CREDUX, so in most rounds one refresh redux lands after the barrier and in front of the store, whatever the PTX order says. Redux throughput is not the limit (T33, per scheduler at 5.7 cycles). Verdict: negative on nvcc 13.4, the single aggregated CREDUX of the baseline is the shortest chain ptxas will schedule at 256 threads.
6. The atomic form of 5 (B2_spec4): the predicated red compiles to a divergent branch with BSSY and BSYNC and the CREDUX moves stall before it. Not measured, same reasoning. Verdict: dropped.
7. 128-thread blocks (same adapter): round 69.5 at IPT 8, 1182 vs 1380 at bf16 N 1024 k 16 across sessions, 1.9x to 2.7x against B1 at 128 threads in the same session. Verdict: positive, the block size is a free parameter B2 should set to 128 at N 1024 to 4096.
8. Rolled round loop when K x IPT > 4096: 260 cycles per round at IPT 32 k 256 against 112 to 148 unrolled. Verdict: negative, k 256 loses to B1 either way.

## 5. Ideas not yet tried

- Value rounds with a min-index ticket for fp32 prefer_smaller_index: dominated by measurement, the value-rounds path without any index pass (B2_valrounds, not_guaranteed) is already 20 to 38 percent slower than the N4 packed rounds that give prefer_smaller_index for free, so an index pass on top cannot win at N <= 16384. It would matter only where the N4 finish runs often (N 32768, 17 key bits), where B2 needs a different per-thread structure first.
- Two-level per-thread list (sorted front plus unsorted remainder with refill): built and measured by the other instance (B2_twolevel), wins from N 8192 at k 4 and owns N 16384 and 32768.
- Partial sort of the top k items per thread when k < IPT (prologue cut at IPT 32 and above), per-stage unrolled network tables (fixes IPT 128, T34), 64-thread blocks (2 warps, IPT 16 at N 1024, the model is close to even with 128), the fp32 tie count riding on the last round's barrier.
- A one-redux refresh for the speculative post (merge c2 and c3 by the known up flag) would still hand ptxas a redux to misplace, not worth a batch.
- sm_100f variant: nothing in the round benefits from redux.sync.max.f32 since keys are packed integers, not built.

## 6. Determinism and ordering support

prefer_smaller_index is inherent: the inverted index is the low part of the packed word, the same binary runs for both requirements, and the measured cycles agree within 1 percent in every cell of every batch (bf16 N 1024 k 16 1380 and 1380, N 8192 k 16 2758 and 2771, fp32 N 4096 k 16 2299 and 2296, raw files). The measured cost of prefer_smaller_index is therefore zero. The exact oracle check passes on every battery pattern including signed_zero (after the -0.0 normalization, one instruction per word) and the fp32 tie finish on the tie patterns. The output is sorted by key descending then index ascending at no extra cost (slot r is the r-th largest), so B2 also covers the sorted-output requirement inherently. prefer_larger_index would be the non-inverted index in the low bits, not built.

## 7. Open items

- The other instance's B2_twolevel and B2_tiecheck should become the B2 baseline for N >= 8192 and for fp32, the cards exist, the two code lines should be merged into one adapter before the final evaluation.
- 128-thread blocks: confirm in the batched device kernel (several blocks per SM share the schedulers) and measure 64 threads once.
- The prefer_smaller_index cells of the baseline card have no control (B1 does not support the requirement, harness 0.3's --control-req arrived after the batch), their values are in the raw file and equal the not_guaranteed cells, later cards carry them with B2_packed as control.
- GB200 not measured by this instance (T16 says SM cycle costs transfer, the final evaluation decides).
- Whether any CUDA C++ spelling pins a redux's issue point relative to BAR.SYNC in ptxas, which would reopen the speculative post at a modeled 70 cycles per round.

## 8. Pointers

Adapters harness/adapters/B2_packed.cuh, B2_valrounds.cuh, B2_spec.cuh, B2_spec2.cuh, B2_spec3.cuh, B2_spec4.cuh (exp/topk-swarm/B2b, commits 8f15348fc3 to the head of the branch). Findings files and the tie class simulation under approaches/B2/ (findings_*.json, tie_sim.cpp). Catalogue entries T27 (aggregation and its cost) and T34 (unroll limit), with T26 (curated from the spec card) and T33 (hardware researcher, redux throughput). Reference material: reference/BLOCK_TOPK_RESULTS.md sections 1 and 3 (atomic_adaptive, atomic_iter_h), ideas/queue/B2_atomics_extraction.md, ideas/queue/N4_truncated_key_packing.md.
