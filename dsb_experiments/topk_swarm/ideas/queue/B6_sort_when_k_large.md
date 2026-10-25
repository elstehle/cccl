# B6: Sort when k is large
status: queued (wave 2, one cell, then the sorted-output pass only)
scope: block
origin: `cub/cub/block/block_radix_sort.cuh`, `cub/cub/block/block_merge_sort.cuh`, the hybrid block sort study on `exp/device-side-perf`
hypothesis: for k above roughly N over 4 a full sort and truncation beats selection. Wave 1 moved the goalposts: B3's packed block scan emission (T29) made the k dependence of the block select almost disappear, so the incumbent at k at least N over 4 is B3_tuned at 2550 cycles for N 2048 k 1024 bf16 (B1_reference 3470) and 6321 for N 8192 k 1024 (B1 11060), with prefer_smaller_index free. A block radix sort of N packed 32-bit words needs 8 passes of 4-bit rank and exchange at about 350 to 400 cycles each at 256 threads by 8 items, about 3000 at N 2048, above the select. B6 keeps two uses: the sorted-output pass (phase 3b) where the sort's output order is the requirement, and the k-truncated merge of M sorted chunk results (the chunk-merge cells of `ideas/WORKLOADS.md` section 7) where the inputs arrive sorted from W1, W2 or B6 itself.
cells it should win: block N 1024 to 8192 with k at least N over 4 only under the sorted-output requirement, and the chunk merge of sorted runs

what it must beat (B200, 2026-10-25, B3_tuned_b200_20261025_2, bf16 then fp32): N 2048 k 1024 2550 and 2508, N 4096 k 256 3947 and 3987, N 8192 k 1024 6321 and 6693, N 16384 k 1024 11009 and 11328, N 32768 k 2048 19253 and 20804. B1_reference at N 16384 k 4096 21465 and 23343, N 32768 k 4096 40026 and 47125 (B3 not measured there, predicted about 12k and 21k by its k-flat structure). To matter B6 has to be within 1.2x of B3_tuned under not_guaranteed, or it is the sorted-output row only.

cost model (measured constants: packed block scan about 300 (T29), barrier 28, the block sort numbers are this optimizer's first measurement):
- `BlockRadixSort` on the packed 32-bit word (bf16 key high, inverted index low, so the sort is keys-only and the tie-break is inherent): 8 passes of 4 bits, each a per-thread digit count, a block scan over 16 bins and an exchange through shared memory (two barriers per pass), about 350 to 400 per pass at IPT 8, about 3000 at N 2048 and 10k at N 8192 (IPT 32, exchange traffic grows with IPT). A 16-bit keys-only sort with the index as the value (4 passes) exchanges two words per pass and lands near the same number. fp32 keys with a 32-bit index are a 64-bit sort word or a pairs sort, 1.5x to 2x.
- Partial sort (materialize the top k ranks only): saves the exchange of the non-selected items in the last passes, not the rank passes, at most 20 percent.
- Chunk merge of M sorted runs of k: one merge-path partition per thread into M runs plus a serial merge of k over 256 items per thread, O(k log2 M) work: M 4 k 2048 about 2000 against B3_tuned over the 8192 candidates at about 6300.
- B3_tuned's cost at k = N over 2 is two radix passes plus the scan emission, k-flat, and it already gives index order. The sort cannot win that race on random input, only on the requirement.

focus cells: 1. block N 2048 k 1024 bf16 (the kill cell, against B3_tuned 2550), 2. block N 8192 k 2048 fp32 sorted output (DSA V3.2 merge of 4 chunk results), 3. chunk merge of 4 sorted runs of 2048 as one block against B3_tuned on 8192 items, 4. block N 4096 k 1024 bf16 sorted output.
boundary cells: block N 4096 k 256 (B3 3947 and N6 4467 expected to win by 2x), block N 1024 k 128 (B2 and B1_idx expected).

explore first: (1) `BlockRadixSort` on the packed bf16 word at N 2048 k 1024 against B3_tuned in one session, not_guaranteed. (2) the merge of 4 sorted runs of 2048 against B3_tuned on 8192 items. (3) only if (1) is within 1.2x: the partial sort and the crossover table against B3 and B1_idx.

determinism and ordering: inherent, sorted, stable with the index in the word.

main risk: B3's emission removed the one k-dependent cost a sort could have beaten. The large k over N workload regime is also narrower than the plan assumed (`ideas/WORKLOADS.md` section 3).

cheapest kill experiment: explore item 1. If the sort is more than 1.2x slower than B3_tuned at N 2048 k 1024 bf16, B6 keeps the chunk merge (explore 2) and the phase 3b sorted-output row and gets no further GPU time in wave 2.

references: `/Users/estehle/Downloads/cccl-exp-perf/dsb_experiments/HYBRID_BLOCK_SORT_RESULTS.md`, `TECHNIQUES.md` T29, cards B3_tuned_b200_20261025_2 and B1_reference_b200, `ideas/WORKLOADS.md` sections 3 and 7
