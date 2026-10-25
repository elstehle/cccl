# B6: Sort when k is large
status: queued (wave 2)
scope: block
origin: `cub/cub/block/block_radix_sort.cuh`, `cub/cub/block/block_merge_sort.cuh`, the hybrid block sort study on `exp/device-side-perf`
hypothesis: for k above roughly N over 4 (expert-choice and mixture-of-depths capacity routing), a full sort and truncation beats selection. Find the crossover against B1 per key width, with index-only output (sort pairs, emit indices).
cells it should win: block N 1024 to 8192, k from N over 8 to N over 2
cost model: radix sort passes over N (bf16 two 8-bit passes) or merge sort depth
explore first: (1) BlockRadixSort on packed (key, index) for bf16, one 32-bit sort. (2) crossover table against B1 and B3. (3) partial sort (only the top k ranks materialized).
determinism and ordering: inherent, sorted, stable options.
references: `/Users/estehle/Downloads/cccl-exp-perf/dsb_experiments/HYBRID_BLOCK_SORT_RESULTS.md`
