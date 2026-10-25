# W1: Bitonic network top-k (warp and sub-warp)
status: queued (wave 1)
scope: warp, subwarp
origin: PR #12222 (eager, cherry-picked here as `cub/cub/warp/warp_bitonic_topk.cuh`), PR #9281 (adds the `buffered` algorithm: tracks the current k-th key as a threshold, ballot-compacts candidates into a MaxK buffer, merges when full, head `yong-wang/cccl@c9530c03`), sub-warp `WarpBitonicSort` on fork branch `exp/sub-warp-bitonic-sort`, issue #12160 (min and max operator, packed half2), #9747 strategy 1, issue #9749
hypothesis: register-resident, shuffle-only, data-oblivious network. Wins where the input is one or two MaxK tiles (N at most 2 MaxK) and where determinism and sorted output come for free. The eager loop sorts every incoming tile fully, so its cost grows as N over MaxK times a full sort of MaxK, which is why it should lose to W2 at small k and to N1, W3 and N2 at large N. Index-only output lets the index ride as the value or, for bf16, inside the key word.
cells it should win: warp N 32 to 64 with k 8 to 32, sub-warp N equal to the logical warp size times 1 or 2 with k up to that size, both requirements, latency regime. Throughput regime at any N where no shared memory matters (occupancy).

cost model (cycles per call, placeholders are July 2026 B200 measurements at one design point, to be re-measured by the hardware researcher):
- Stage cost s(IPT) for one shuffle compare-exchange stage over 32 lanes, keys only: s(1) 44, s(2) 51, s(4) 70, s(8) 152 (from `WarpBitonicSort` 660, 1080, 1964, 5468 cycles over 15, 21, 28, 36 stages, `reference/WARP_MERGE_SORT_RESULTS.md` section 13). Pairs with a 32-bit value add one shuffle per stage, expect 1.5 to 1.7 s (unmeasured, measure first).
- Eager, MaxK = M, L = log2 M, tiles T = N over M: cost = sort(M) + (T minus 1) times (sort(M) + c&r(M) + merge(M)), with sort(M) = L(L+1)/2 stages, merge(M) = L stages, c&r = M over LW in-register compares. For M 32 at warp scope: 660 + (N/32 minus 1) times about 925 on the model. Measured 2500 for N 128 in `reference/BLOCK_TOPK_RESULTS.md` section 2 (`bitonic_hier`), so the stages pipeline better than the model says. k below 32 costs the same as k 32 at warp scope because MaxK must be a multiple of the logical warp size.
- Buffered (PR #9281): one compare per item plus one ballot and popc per item slot, plus merges only when the candidate buffer fills. Expected candidates on random input about k times (1 + ln(N over k)), so merges are about ln(N over k). Sorted ascending input makes every item a candidate (the known pathology).

focus cells (bf16 and fp32 keys, int32 index, index-only output, both requirements, both regimes):
1. warp N 32 k 8, 2. warp N 64 k 32, 3. warp N 128 k 16, 4. sub-warp LW 8 N 16 k 4 (DBRX, gpt-oss-20b routing shape), 5. sub-warp LW 16 N 32 k 8.
boundary cells: warp N 256 k 32 (expect N2 or N1 to win by 1.5x or more), warp N 128 k 8 (expect W2 to win by 2x or more).

explore first: (1) reproduce the PR numbers on the focus cells as the reference, keys only and pairs. (2) sub-warp logical warps 4, 8, 16 with MaxK a multiple of the logical warp size. (3) index-only output: index as the value (one extra shuffle per stage) against bf16 key with inverted index packed into one 32-bit word (zero extra shuffles, prefer_smaller_index free). (4) min and max operator instead of the bool comparator for fp32 (#12160, 2 instructions per compare-exchange instead of 3). (5) buffered variant for N at least 8 k on random input, with its candidate count measured, and on sorted input. (6) hand the level-parallel restructuring to N2 and the two-row bf16x2 packing to N5 (same code family, same optimizer if the lead agrees).

determinism and ordering: inherent (deterministic set, sorted). prefer_smaller_index by comparing (key, index), or for bf16 by packing key and inverted index into one word, free up to N 65536.

main risk: the eager loop is a sequential chain of full tile sorts, so every doubling of N adds a sort of MaxK. At N 128 and k 16 the model says 2500 to 3400 cycles against about 1700 for W2 packed bf16. If that holds, W1 is confined to N at most 2 MaxK and to the sorted-output pass.

cheapest kill experiment: W1 eager at warp N 128 k 16 bf16 index-only against W2 packed in the same session. If W2 wins by 1.5x or more, W1 keeps only cells 1, 2, 4, 5 and the sorted-output pass. One build, two kernels, one lock section.

references: `reference/SUBWARP_SORT_RESULTS.md`, `reference/WARP_MERGE_SORT_RESULTS.md` section 13, `reference/BLOCK_TOPK_RESULTS.md` sections 1 and 2, knowledge base section 4, `ideas/queue/N2_level_parallel_bitonic.md`, `ideas/queue/N5_two_row_bf16x2.md`
