# W4: Merge based warp top-k
status: queued (last, crossover mapping only)
scope: warp
origin: `reference/WARP_MERGE_SORT_RESULTS.md`, `reference/SUBWARP_SORT_RESULTS.md`, `cub/cub/warp/warp_merge_sort.cuh`
hypothesis: per-lane sorted runs plus merge-path merges that keep only the top k. Expected to lose to W1 at small N. Kept to find the crossover at large N and large k where the bitonic network's log squared depth hurts.
cells it should win: warp N 2048 to 4096, k 256 to 2048
cost model: per-lane sort plus log(lanes) merge levels, each a binary search plus serial merge, truncated to k
explore first: (1) truncated merge (only the top k of each merge survive). (2) static unroll of the merge-path search (see the WMS static switch study). (3) crossover table against W1 and W3.
determinism and ordering: inherent, sorted, stable.
references: `reference/WARP_MERGE_SORT_RESULTS.md`
