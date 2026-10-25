# B5: Hierarchical warp then block composition
status: queued (wave 2, predicted negative, one confirming cell only)
scope: block
origin: `hier_extract` and `bitonic_hier` in `reference/BLOCK_TOPK_RESULTS.md` (lost at k 16, N 1024, fp32: 5024 and 5112 against 2331)
hypothesis: each warp runs a warp primitive on its share, then the 8 k candidates are merged by one warp. It pays only if the warp phase is much cheaper per unit of work than the block phase. The July measurement says a warp extraction round (82) costs the same as a block atomic round (81), so two phases of k rounds cost twice one flat phase. Packed bf16 does not change the ratio because both scopes pack. The remaining hopes are the throughput regime (no block barriers, better overlap across resident blocks) and a warp primitive whose cost is not round-based (N1, N2) where the block has no equivalent.
cells it should win: block N 1024 to 4096, k 32 to 128, bf16, throughput regime, if anywhere

cost model (placeholders July 2026 B200):
- Warp phase on N over 8 items per warp (IPT_w = N over 256, the same items per thread as the block primitive): W2 packed k (82 + 2 IPT_w), N1 bf16 about 610 + 128 IPT_w, N2 about sort(k) at IPT_w plus log2(N over 8k) levels.
- Exchange: 8 k words to smem, one barrier, about 100.
- Merge of 8 k candidates by one warp: W2 packed k (82 + 2 times 8 k over 32), N1 on 8 k items, or N2's last 3 levels on pre-sorted runs (3 (1 + log2 k) stages).
- N 1024 k 16 bf16: W2 phases 16 times 90 + 100 + 16 times 90 = 3000 against B2 1640. N1 phases: 1120 + 100 + 1120 = 2340 against B1 1700 and B2 1640. N 1024 k 32: N2 phases about 1620 + 100 + 1620 = 3340 against B1 1700. Negative everywhere on the model.
- Throughput: no `BAR.SYNC` in the warp phase and one in the exchange, against 9 in B1. Whether that buys anything at one wave per SM is the only open question.

focus cells: 1. block N 1024 k 32 bf16 throughput, 2. block N 4096 k 64 bf16 throughput, 3. block N 1024 k 16 bf16 latency (the July cell, as the control).
boundary cells: block N 8192 k 16 (B4 and B2 expected), block N 1024 k 128 (B1 expected).

explore first: (1) one cell only until something wins: N1 per warp plus N1 merge at N 1024 k 32 bf16, latency and throughput, against B1 and B2 in the same session. (2) only if (1) is within 20 percent: N2 per warp plus N2 merge. (3) barrier-free variant where warp 0 owns the merge and the other warps exit (throughput only).

determinism and ordering: inherits from the warp primitive, merge is deterministic.

main risk: the measured equality of warp and block round cost makes the composition a 2x loss by construction. The throughput argument is unproven and the harness's register and occupancy effects did not transfer to device kernels three times before (knowledge base 2.5).

cheapest kill experiment: explore item 1, one build, three kernels, one lock section. If B5 is more than 1.2x slower than the better of B1 and B2 in both regimes, close the card with the number.

references: `reference/BLOCK_TOPK_RESULTS.md` sections 1, 2, 4, `ideas/queue/N1_bit_serial_select.md`, `ideas/queue/N2_level_parallel_bitonic.md`
