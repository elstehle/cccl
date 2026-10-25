# W4: Merge based warp top-k
status: queued (last, crossover mapping and the sorted-input multiway merge only)
scope: warp
origin: `reference/WARP_MERGE_SORT_RESULTS.md`, `reference/SUBWARP_SORT_RESULTS.md`, `cub/cub/warp/warp_merge_sort.cuh`
hypothesis: per-lane sorted runs plus merge-path merges that keep only the top k. Wave 1 settled the small and medium k cells for W2 (26-cycle rounds) and left k near N over 2 at N 2048 to 4096 to the bucket selects: W2 at N 2048 k 1024 costs 150107 (bf16) and 154812 (fp32), W1_reference 188007 and 131669, both far above a merge sort (about 30k on the July model), and N6 at warp scope is predicted near 5k to 8k there. W4's remaining uses are the stable or sorted output requirement at large k and the k-truncated multiway merge of M sorted chunk results, where the inputs are already sorted runs and the merge is O(k log2 M).
cells it should win: warp N 2048 to 4096 with k at least N over 2 under the sorted or stable requirement, and the chunk merge of sorted runs (M 4 to 8 runs of k 128 to 2048)

what it must beat (B200, 2026-10-25): under not_guaranteed the bar is N6 warp and W3, predicted about 5k to 8.5k at N 2048 to 4096 k 1024 (bf16, fp32 the same for N6), against which a merge sort at about 30k loses by 4x to 6x. Under sorted output the bar is W2 (150107 and 154812 at N 2048 k 1024, its output is sorted) and W1_reference (188007 and 131669), which W4 beats by 4x to 5x on the model. For the chunk merge the bar is B3_tuned over the M k candidates (6321 at N 8192 k 1024 for M 8 k 1024, 19253 at N 32768 k 2048).

cost model (July placeholders: `WarpMergeSort` about 2000 + 470 IPT baseline, tuned static merge path 15 to 30 percent less, compiler sensitive. Measured constants where they exist: shuffle 4.9 per further shuffle in flight T17, T20 for the branch codegen on aarch64 hosts, T34 and T36 for unrolled networks and register arrays):
- Per-lane sort of IPT items (network, under 500 compare-exchanges per unrolled loop per T34) + 5 merge levels per 32 lanes, each a static binary merge-path search plus a serial merge of IPT items. Truncation to k only idles lanes beyond k over IPT in a run unless the kept items are re-packed, so model the truncated cost as the untruncated merge sort until runs exceed k, then a compaction and half the items per lane per further level.
- N 2048 k 1024 fp32: about 2000 + 470 times 64 at the untruncated levels, above 30k.
- Chunk merge of M sorted runs of k: one merge-path split per lane into M runs (M binary searches) plus a serial merge of k over 32 items per lane, about M log2 k times 2 loads plus k over 32 steps, so about 400 + 20 k over 32 cycles for M 4 and k 2048 at warp scope, or the block form in B6.

focus cells: 1. warp N 2048 k 1024 fp32 sorted output (against W2 and N6 warp in one session), 2. chunk merge of 4 sorted runs of 2048 as one warp against B3_tuned on 8192 items, 3. warp N 4096 k 2048 bf16 sorted output.
boundary cells: warp N 512 k 256 (W2 12362 and N6 warp expected to win), warp N 2048 k 64 (W5 and N7 expected).

explore first: (1) k-truncated multiway merge of M sorted runs with k 256 to 2048 (the one predicted win). (2) truncated merge sort with re-packing at N 2048 k 1024, static merge path, against N6 warp and W2. (3) crossover table for the sorted-output pass (phase 3b).

determinism and ordering: inherent, sorted, stable.

main risk: the untruncated prefix of the merge tree is a full merge sort on N and the tuned variant regressed under CUDA 13.3. No not_guaranteed cell is predicted for W4 at all after wave 1.

cheapest kill experiment: explore item 2 at N 2048 k 1024 fp32 against N6 warp in one session. If W4 is more than 1.5x slower, keep only the sorted-input chunk merge and the phase 3b row.

references: `reference/WARP_MERGE_SORT_RESULTS.md` sections 4, 10, 13, 14, `TECHNIQUES.md` T17, T20, T34, T36, cards W2_packed_b200_20261025_1 and W1_reference_b200, `ideas/queue/N6_equiwidth_bucket_select.md`, `ideas/queue/B6_sort_when_k_large.md`
