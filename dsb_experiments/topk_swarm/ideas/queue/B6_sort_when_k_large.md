# B6: Sort when k is large
status: queued (wave 2, crossover mapping)
scope: block
origin: `cub/cub/block/block_radix_sort.cuh`, `cub/cub/block/block_merge_sort.cuh`, the hybrid block sort study on `exp/device-side-perf`
hypothesis: for k above roughly N over 4, sort and truncate beats selection. The workload check (`ideas/WORKLOADS.md`) moves the large k over N regime: capacity routing is k over N = c over E (1 over 64 to 1 over 8) and mixture-of-depths is 1 over 8, so those are selection cells. The cells with k at least N over 4 are the chunk merges (top k over M chunk results, k over N = 1 over M for M 2 to 8), H2O style eviction (20 percent of the cache), SnapKV on short prompts, and the top-2 of 8 and top-4 of 16 routers at sub-warp scope. B1's cost is k-flat except for the epilogue scatter, so the sort must beat 2 passes plus a scatter of k items, which the model doubts. B6 earns its place through the sorted-output pass (phase 3b) and through the sorted-input chunk merge, where a merge of M sorted runs is cheaper than a fresh selection.
cells it should win: block N 1024 to 8192, k from N over 4 to N over 2, sorted output, and the chunk merge of sorted runs

cost model (placeholders July 2026 B200, block sort numbers to be measured by this optimizer first):
- B1 at large k: 14 IPT + 2 (85 IPT + 410) + scatter of k indices by ticket (k atomics across banks, about k over 8 cycles) + 150. N 1024 k 512 bf16 about 1800.
- `BlockRadixSort` on packed 32-bit (bf16 key, 16-bit index) with 4-bit digits: 8 passes of rank (upsweep and scan over 16 bins) plus exchange through smem. Even at 300 cycles per pass that is 2400 above B1. A 16-bit key-only sort (4 passes) with the index as the value adds a second exchange per pass. Needs the real numbers, the hybrid block sort study has them for other shapes.
- `BlockMergeSort`: log2(256) merge levels plus per-thread sort, each level a merge-path search and a serial merge through smem, typically slower than the radix sort at 32-bit keys.
- Merge of M sorted runs of k (chunk merge): one merge-path partition per thread into M runs plus a serial merge of k over 256 items per thread, O(k log M) work against 2 B1 passes over M k items. For M 4 k 2048 about 2000 against B1 on 8192 items about 6900.

focus cells: 1. block N 1024 k 512 bf16, 2. block N 4096 k 1024 fp32 (H2O 20 percent and SnapKV 4096 budget on 16K prompts, sort also gives the order), 3. block N 8192 k 2048 fp32 (DSA V3.2 merge of 4 chunk results), 4. block N 2048 k 512 bf16.
boundary cells: block N 4096 k 256 (B1 expected to win by 2x), block N 1024 k 128 (B1 expected).

explore first: (1) `BlockRadixSort` on packed (key, index) for bf16, one 32-bit sort, against B1 at k N over 2, N 1024, one cell, same session. (2) the merge of M sorted runs of k against B1 on M k items (the chunk merge, M 4 and 8). (3) only if (1) is within 20 percent: partial sort (materialize the top k ranks only) and the crossover table against B1 and B3.

determinism and ordering: inherent, sorted, stable options.

main risk: B1 is nearly k-flat and already 2 passes on bf16, so a full sort has to do 2 passes of rank and exchange over N to tie. The large k over N workload regime is narrower than the plan assumed.

cheapest kill experiment: explore item 1. If the sort is more than 1.2x slower than B1 at N 1024 k 512 bf16, B6 keeps only the chunk merge (explore 2) and the phase 3b sorted-output row.

references: `/Users/estehle/Downloads/cccl-exp-perf/dsb_experiments/HYBRID_BLOCK_SORT_RESULTS.md`, `ideas/WORKLOADS.md` section on the large k over N regime
