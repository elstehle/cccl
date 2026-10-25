# B5: Hierarchical warp then block composition
status: queued (revived 2026-10-25 by the measured redux cost, to be built by W2's optimizer in wave 1 right after the W2 round is confirmed)
scope: block
origin: `hier_extract` and `bitonic_hier` in `reference/BLOCK_TOPK_RESULTS.md` (lost at k 16, N 1024, fp32: 5024 and 5112 against 2331, with value-only warp rounds of 82 cycles that needed ballot and elect)
hypothesis: each of the 8 warps runs packed W2 on its 1 over 8 of the row, writes its k words to shared memory, one barrier, and warp 0 runs packed W2 once more over the 8 k candidates. The July composition lost because a value-only warp round (redux, ballot, elect, 82 cycles) cost the same as a block atomic round (81). With a packed unique word the warp round is the redux plus a compare and a select, about 30 cycles (`TECHNIQUES.md` T2, T4, T17: redux 22.1, no ballot, no ffs), while the block round stays at 74 (T9), so two warp phases of k rounds (about 60 k) undercut one block phase (74 k) and B5 beats B2 wherever the per-lane list shift stays inside the redux latency. Index-only output and prefer_smaller_index are inherent in the packed word, fp32 goes through N4.
cells it should win: block N 1024 to 4096, k 4 to 32, bf16 packed and fp32 with N4, latency and throughput (one barrier per call against k in B2 and about 9 in B1)

cost model (measured 2026-10-25 for redux, ballot, barrier and the block round, July placeholders for prologue and list costs):
- Warp phase: items per lane IPT_w = N over 256 (the same as the block primitive's items per thread), packed W2 about 100 + k round(IPT_w) with round about 30 for IPT_w at most 8 and 14 + 2 IPT_w above (W2 card).
- Exchange: 8 k words to smem (one STS per selected lane), one barrier 28, warp 0 reads 8 k words (8 k over 32 per lane), about 100 for k 16.
- Merge: packed W2 by warp 0 over 8 k words at IPT 8 k over 32, about 100 + k round(k over 4): k 16 gives IPT 4, round 30, 580. k 32 gives IPT 8, round 30, 1060. k 64 gives IPT 16, round 46, 3044.
- Totals: N 1024 k 16 about 580 + 100 + 580 = 1260 against B2 1640, B1 1700, N6 1500. N 1024 k 32 about 1060 + 100 + 1060 = 2220 against B1 1700 and B2 3000. N 4096 k 16 about 836 + 100 + 580 = 1520 against B2 2000. N 8192 k 16 about 1350 + 100 + 580 = 2030 against B2 2600 and B4 about 2600. N 1024 k 8 about 340 + 100 + 340 = 780 against B2 940.
- Crossover with B1 at N 1024 near k 26 on the model. Against B2 the ratio is about 60 k over 74 k plus the exchange, so B5 wins from k 8 upward until IPT_w exceeds 8 and the warp round grows faster than the block round.
- Alternative merges: N2 over 8 sorted runs of k (3 levels of 1 + log2 k stages) when the warp phase emits sorted runs, which W2 does for free, so for k 32 the merge is about 3 (1 + 5) stages at IPT 8, 4, 2, about 1600, no better than W2's 1060.

focus cells: 1. block N 1024 k 16 bf16 (the July cell, now the predicted flip), 2. block N 1024 k 8 bf16, 3. block N 4096 k 16 fp32 with N4 (Quest at 64K with a 256 budget), 4. block N 2048 k 32 bf16, 5. block N 1024 k 16 bf16 throughput.
boundary cells: block N 1024 k 64 (B1 expected), block N 16384 k 16 (IPT_w 64, warp round about 142, B4 expected).

explore first: (1) build from W2's confirmed packed round: W2 per warp plus exchange plus W2 merge at N 1024 k 16 bf16 against B2 and B1 in one session, latency and throughput. (2) the merge by warp 0 against a second level where 2 warps merge 4 k each then one warp merges 2 k (one more barrier, half the merge IPT). (3) fp32 through N4 with the tie pass once after the merge. (4) register and smem footprint against B2 (B2 is 19 registers and 192 bytes), since throughput at 8 blocks per SM decides the batched case.

determinism and ordering: packed form inherent (deterministic set, prefer_smaller_index by the inverted index) and the merge output is sorted. Value-only fp32 would need the election per round, use N4 instead.

main risk: everything rests on the W2 packed round being about 30 cycles with the list shift overlapped. If round(IPT_w) measures above about 45 cycles, B5 returns to the July verdict and B2 keeps its cells. Second risk: the merge over 8 k words puts 8 k over 32 items per lane in one warp, so at k 64 and above the merge alone costs what B1 costs in total.

cheapest kill experiment: explore item 1, one build, three kernels, one lock section, as soon as W2's slope exists. If B5 is not at least 1.15x faster than B2 at N 1024 k 16 bf16, close the card with the number.

references: `reference/BLOCK_TOPK_RESULTS.md` sections 1, 2, 4, `TECHNIQUES.md` T2, T4, T9, T17, `ideas/queue/W2_redux_extraction.md`, `ideas/queue/B2_atomics_extraction.md`, `ideas/queue/N4_truncated_key_packing.md`
