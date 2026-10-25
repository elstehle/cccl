# W4: Merge based warp top-k
status: queued (last, crossover mapping only)
scope: warp
origin: `reference/WARP_MERGE_SORT_RESULTS.md`, `reference/SUBWARP_SORT_RESULTS.md`, `cub/cub/warp/warp_merge_sort.cuh`
hypothesis: per-lane sorted runs plus merge-path merges that keep only the top k. Expected to lose to W1 and N2 at small N and to W3 and N1 at large N. Kept to find the crossover at large N with k close to N, where a full sort is needed anyway, and because it is the only stable warp primitive. Second use: the k-truncated multiway merge of M sorted per-chunk results (DSA and sampling chunk merges, see `ideas/WORKLOADS.md`), where the inputs are already sorted runs and the merge is O(k log M).
cells it should win: warp N 2048 to 4096, k at least N over 2, sorted or stable output. Chunk-merge cells with sorted inputs: M 4 to 8 runs of k, k 128 to 2048.

cost model (placeholders July 2026 B200: `WarpMergeSort` about 2000 + 470 IPT baseline, tuned static merge path 15 to 30 percent less, compiler sensitive):
- Per-lane sort of IPT items (network) + 5 merge levels per 32 lanes, each a static binary merge-path search (log S steps of 2 loads) plus a serial merge of IPT items. Truncation to k only idles lanes beyond k over IPT in a run unless the kept items are re-packed, so model the truncated cost as the untruncated merge sort until runs exceed k, then a compaction (one ballot prefix and smem move per level) and half the items per lane per further level.
- N 2048 k 1024 fp32: about 2000 + 470 times 64 (per-lane sort of 64 items is already register-heavy) at the untruncated levels, so above 30k cycles on the model against W3 at d 4 of about 2000 + 176 times 64 = 13k. W4 is predicted to lose everywhere except where the output must be sorted or stable.
- Chunk merge of M sorted runs of k: one merge-path split per lane into M runs (M binary searches) plus a serial merge of k over 32 items per lane, about M log2 k times 2 loads plus k over 32 steps, so about 400 + 20 k over 32 cycles for M 4 and k 2048, against a fresh selection over M k items.

focus cells: 1. warp N 2048 k 1024 fp32, 2. warp N 4096 k 2048 bf16, 3. chunk merge of 4 sorted runs of 2048 (DSA V3.2 merge shape, as one warp or one block), 4. warp N 512 k 256 fp32.
boundary cells: warp N 512 k 64 (W3 and N2 expected to win), warp N 128 k 32 (N2 expected).

explore first: (1) truncated merge with re-packing, measure the compaction per level. (2) static unroll of the merge-path search (WMS static switch study). (3) k-truncated multiway merge of M sorted runs with k 256 to 2048. (4) crossover table against W3, N1 and N2.

determinism and ordering: inherent, sorted, stable.

main risk: the untruncated prefix of the merge tree is a full merge sort on N and the tuned variant regressed under CUDA 13.3 (`WARP_MERGE_SORT_RESULTS.md` round 3). The crossover may not exist inside the grid.

cheapest kill experiment: W4 against W3 at N 2048 k 1024 fp32 in one session. If W4 is more than 1.5x slower, keep only the sorted-input chunk merge (explore 3) and the phase 3b sorted-output row.

references: `reference/WARP_MERGE_SORT_RESULTS.md` sections 4, 10, 13, 14
