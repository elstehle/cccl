# N2: Level-parallel bitonic top-k (sort all k-tiles at once, halve per level)
status: queued (wave 1 candidate, same optimizer as W1)
scope: warp, subwarp
origin: new here, the structure is the bitonic top-k of Shanbhag et al. (SIGMOD 2018) applied to a register-striped warp. Seed from the lead: pruned bitonic networks that compute only the top k lanes.
hypothesis: W1's eager loop processes tiles sequentially and sorts every tile fully, so it costs (N over k) full sorts of k. Instead sort all N over k tiles concurrently (the same stage instructions act on every register), then merge pairs of tiles level by level with one compare stage plus log2 k merge stages, discarding the lower half each level. Pair tiles across registers first (the compare is register to register, no shuffle) and inside a slot last. Total stages about log2 k (log2 k + 1) over 2 at full IPT plus log2(N over k) times (1 + log2 k) at halving IPT, against W1's (N over k) times that. Same registers as W1 (the input is live anyway), same determinism and sorted output, native float compare, pairs by moving the index alongside.
cells it should win: warp N 64 to 256 with k 32 to 64, sub-warp N up to 8 times the logical warp size with k equal to the logical warp size, both types, both requirements, latency and throughput (no smem). Everything W1 holds above N 2 MaxK.

cost model (placeholders July 2026 B200: shuffle stage s(1) 44, s(2) 51, s(4) 70, s(8) 152, s(16) about 300 extrapolated, register compare-exchange about 3 cycles per item):
- k 32 (one slot per tile): 15 s(IPT) for the concurrent sorts, then log2(IPT) cross-slot levels of (register compare + 5 shuffle stages at IPT over 2, 4, ...). N 256 k 32: 15 times 152 + 5 (70 + 51 + 44) = 3100 (W1 eager model 7100, measured about 5000). N 128 k 32: 1050 + 475 = 1525 (W1 3435 model, about 2500 measured). N 64 k 32: 985 (W1 1585).
- k 16 (two tiles per slot): 10 s(IPT) + log2(IPT) (1 + 4 stages) + one in-slot level (5 stages at s(1)). N 256 k 16: 1520 + 660 + 220 = 2400 (W2 1670, N1 1800).
- k 64 (tile spans two slots): 15 shuffle stages plus 6 register stages for the sort, levels of (register compare + 5 shuffle + 1 register stage). N 512 k 64 bf16: about 15 s(16) + 5 (152 + 70 + 51) = 5900 (W3 2400, N1 2950), so N2 loses at IPT 16 and above unless s(16) is far better than extrapolated.
- Pairs (index as value): one more shuffle per shuffle stage, expect 1.5 to 1.7 times, unmeasured. bf16 with the inverted index in the key word: no extra cost.

focus cells: 1. warp N 128 k 32 bf16, 2. warp N 256 k 32 fp32, 3. warp N 64 k 32 fp32, 4. sub-warp LW 8 N 64 k 8 bf16, 5. warp N 256 k 64 bf16.
boundary cells: warp N 512 k 64 (W3 or N1 expected), warp N 256 k 16 (W2 expected).

explore first: (1) sort all tiles plus one level at N 64 k 32 against two W1 tiles, keys only, same session. (2) cross-slot pairing order (register compares first) against in-slot pairing. (3) pairs against bf16 index-in-key. (4) min and max operator (#12160) for fp32. (5) sub-warp k equal to the logical warp size. (6) register count against IPT, spills kill the throughput row.

determinism and ordering: inherent (deterministic, sorted). prefer_smaller_index by (key, index) compare or bf16 index-in-key.

main risk: the first phase runs all 15 sort stages at full IPT, and s(IPT) grows faster than linearly (152 at IPT 8 against 44 at IPT 1), so above IPT 8 the concurrent sorts cost more than the levels save. The win over W1 is a restructuring, not a new asymptotic class against N1 or W3.

cheapest kill experiment: explore item 1 (one build, two kernels, one cell). If N2 is not at least 1.3x faster than W1 at N 64 k 32, the restructuring does not pay and W1 keeps its card as is.

references: `ideas/queue/W1_bitonic_topk.md`, `reference/WARP_MERGE_SORT_RESULTS.md` section 13 (stage costs), `cub/cub/warp/warp_bitonic_topk.cuh` (the `reverse` trick that keeps alternate tiles reverse-sorted)
