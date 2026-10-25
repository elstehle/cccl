# W3: Warp radix select
status: queued (wave 2)
scope: warp
origin: `reference/WARP_RADIX_SORT_RESULTS.md` (ballot-ranked LSD), radix select as in `block_topk_air` at warp scope
hypothesis: digit histograms by `match.any` or ballot counts over the warp, k-selection over digit buckets, narrow digits so bf16 takes two passes and fp32 three to four. Depth independent of k, so it should win at larger N per warp and larger k where W1's network and W2's rounds grow.
cells it should win: warp N 512 to 4096, k 32 to 1024
cost model: passes times (histogram by ballots over items per lane plus a warp scan over buckets plus a filter)
explore first: (1) ballot-count histogram with 4-bit and 5-bit digits. (2) candidate compaction between passes in registers via ballot prefix. (3) packed bf16 digits. (4) index-only output with a final rank of the k survivors.
determinism and ordering: set deterministic by construction. prefer_smaller_index via a final index pass over the pivot bucket or the warp-redundant rank idea (knowledge base 4.5). Sorted via a final W1 pass over k.
references: knowledge base section 4, `cub/cub/block/specializations/block_topk_air.cuh`
