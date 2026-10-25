# B1: AIR radix select, the block incumbent
status: queued (wave 1)
scope: block
origin: `cub/cub/block/specializations/block_topk_air.cuh` (PR #11060 merged), striped branch `enh/block-topk-striped` (worktree `/Users/estehle/Downloads/cccl-block-topk-striped`, rebased copy `enh/block-topk-striped-rebased`), knowledge base section 4, `reference/BLOCK_TOPK_AIR_OPT_RESULTS.md`
hypothesis: the production block primitive. Its optimizer continues the open items of knowledge base 4.7 on the striped base and re-tunes for the focus set: bf16 keys need two 8-bit passes, index-only output removes the key scatter.
cells it should win: block N 1024 to 32768, k 64 to 4096, throughput regime
cost model: passes times (histogram with shared atomics plus scan and choose plus filter), 2 passes for bf16 at 8-bit digits, 4 for fp32, each pass with several barriers
explore first: (1) reproduce main and striped numbers on the focus set as the reference. (2) digit width per key width (bf16 at 8 or 16 bits, fp32 at 8, 11, 16). (3) index-only output path (no key memoization or scatter). (4) raw digit extractor and `NormalizeSignedZero` (knowledge base 4.4). (5) deterministic warp-redundant rank for prefer_smaller_index (knowledge base 4.5). (6) the compile-time k and striped histogram follow-ups.
determinism and ordering: set deterministic with the warp-redundant rank. Sorted via post-sort of k.
references: knowledge base 4.2 to 4.7, `reference/BLOCK_TOPK_AIR_OPT_RESULTS.md`
