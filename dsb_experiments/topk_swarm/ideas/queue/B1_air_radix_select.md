# B1: AIR radix select, the block incumbent
status: queued (wave 1)
scope: block
origin: `cub/cub/block/specializations/block_topk_air.cuh` (PR #11060 merged), striped branch `enh/block-topk-striped` (worktree `/Users/estehle/Downloads/cccl-block-topk-striped`, rebased copy `enh/block-topk-striped-rebased`), knowledge base section 4, `reference/BLOCK_TOPK_AIR_OPT_RESULTS.md`
hypothesis: the production block primitive. Its optimizer continues the open items of knowledge base 4.7 on the striped base and re-tunes for the focus set: bf16 keys take two 8-bit passes with no early exit to hope for (the first digit is sign plus 7 exponent bits and separates little on logit-like data), index-only output removes the key half of the pair epilogue. Pass count is the lever, not the pass body, whose histogram and scan phases sit at measured floors.
cells it should win: block N 1024 to 4096, k 32 to 4096, both types, throughput regime, and every cell where B2, B3 and B4 fall back (sorted, tie-heavy, adversarial input)

cost model (placeholders July 2026 B200 at 256 threads by 4 items, `reference/BLOCK_TOPK_AIR_OPT_RESULTS.md` section 6: twiddle 54, histogram 340 per pass, scan 360 per pass, state broadcast 50 per pass, pair epilogue 250, about 9 barriers per 2-pass call, barrier 28):
- IPT = N over 256. Histogram phase is MIO-issue bound and scales with items: H(IPT) about 85 IPT. Scan and state are IPT-independent: 410 per pass. Twiddle about 14 IPT. Epilogue: pairs 250, index-only expected about 150 (scatter indices by ticket, no key gather), to be measured.
- cost about 14 IPT + P (85 IPT + 410) + 150, with P = 2 for bf16, 2 to 4 for fp32 (early exit when the pivot bucket count equals k remaining).
- N 1024 bf16: about 1700 (measured fp32 pairs 1779 to 1857 at 2 passes). N 8192 bf16: about 6900. N 32768 bf16: about 24.5k, fp32 at 3 passes about 35.7k. Above N 8192 the histogram sweep dominates and B3, B4 and N6 attack exactly that term.
- Wider digits lose at N 1024 (10 to 12 bits measured slower on every axis, scan superlinear in bins). 16-bit digits for bf16 are rejected (`ideas/rejected/R2_wide_first_digit.md`).

focus cells: 1. block N 1024 k 64 bf16, 2. block N 4096 k 256 fp32 (sampling chunk, Quest pages), 3. block N 8192 k 2048 fp32 (DSA chunk merge shape, k over N 1 over 4), 4. block N 2048 k 16 bf16 (NSA at 128K, scores per block), 5. block N 32768 k 2048 fp32 (DSA V3.2 chunk).
boundary cells: block N 1024 k 16 bf16 (B2 packed expected to win by 10 percent or more), block N 32768 k 256 fp32 (B3 or B4 expected to win by 1.5x or more).

explore first: (1) reproduce main and striped numbers on the focus cells as the reference. (2) index-only output path: threshold broadcast, ballot prefix per slot, per-warp atomic base, scatter indices only, measure against the pair epilogue. (3) raw digit extractor and `NormalizeSignedZero` (knowledge base 4.4). (4) deterministic warp-redundant rank for prefer_smaller_index (knowledge base 4.5 approach C, about plus 1 percent). (5) digit schedule per type: bf16 8+8, fp32 8 by 4 against 11+11+10 against 12+12+8, with the pass count distribution on the harness inputs reported. (6) the FINISH knob (small-candidate finish by a warp primitive) as a tail bound, measure the common-case cost. (7) N6 (equi-width bins over the value range) as a drop-in replacement of the digit extractor in pass 1, hand over if it needs its own optimizer.

determinism and ordering: set deterministic with the warp-redundant rank. Sorted via post-sort of k.

main risk: none for existence, it is the incumbent and the fallback. The risk is for the study: every B1 gain must come from fewer passes or a cheaper epilogue, since histogram and scan are at their measured floors, and the pass count on bf16 is fixed at 2.

cheapest kill experiment (for the sub-ideas): pass count histogram of the fp32 digit schedules on the harness random input at N 1024 and 32768, host side, before any GPU time. If 8-bit digits give 2 passes in more than 90 percent of rows, explore item 5 collapses to one row.

references: knowledge base 4.2 to 4.7, `reference/BLOCK_TOPK_AIR_OPT_RESULTS.md` sections 1, 2, 6, `ideas/queue/N6_equiwidth_bucket_select.md`
