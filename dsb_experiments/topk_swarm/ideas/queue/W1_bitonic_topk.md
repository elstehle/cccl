# W1: Bitonic network top-k (warp and sub-warp)
status: queued (wave 1)
scope: warp, subwarp
origin: PR #12222 (eager, cherry-picked on this branch as `cub/cub/warp/warp_bitonic_topk.cuh`), PR #9281 (buffered threshold-compaction variant, fetch `refs/pull/9281/head`), `WarpBitonicSort` sub-warp support on fork branch `exp/sub-warp-bitonic-sort`, issue #12160 (min and max operator, packed half2), #9747 strategy 1, issue #9749
hypothesis: register-resident, shuffle-only, data-oblivious network. Wins small N per lane at small k where network depth is low. Index-only output lets the index ride as the value or, for keys of 16 bits or less, inside the key word.
cells it should win: warp N 32 to 512, k 4 to 32, both requirements, latency regime especially
cost model: sort of MaxK tile plus one merge per further tile, each O(log^2) compare-exchange stages of shuffles
explore first: (1) reproduce PR numbers on the focus set as the reference. (2) Sub-warp logical warps 4, 8, 16 with MaxK a multiple of the logical warp size. (3) Packed arrangement for bf16: two keys per 32-bit register in blocked order inside the register, striped across the logical warp, compared with packed min and max (`max.bf16x2` natively, or integer SIMD on a twiddled variant). (4) Min and max operator instead of the bool comparator (#12160) for fp32. (5) The buffered variant for N much larger than k. (6) Index-only output: drop key materialization at the end.
determinism and ordering: inherent (deterministic, sorted). prefer_smaller_index by comparing (key, index), or for bf16 by packing key and inverted index into one 32-bit word, free up to N 65536.
references: `reference/SUBWARP_SORT_RESULTS.md`, `reference/WARP_MERGE_SORT_RESULTS.md` (bitonic vs merge crossover), knowledge base section 4
