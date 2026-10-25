# W3: Warp radix select (ballot histograms, 2 to 5 bit digits)
status: queued (wave 2, N1 is its 1-bit degenerate case and should share the optimizer)
scope: warp
origin: `reference/WARP_RADIX_SORT_RESULTS.md` (ballot-ranked LSD), radix select as in `block_topk_air` at warp scope
hypothesis: digit histograms by ballot over the warp, one lane per bucket counts with popc over the combined bit masks, a shuffle suffix scan over the buckets finds the one holding rank k, the prefix narrows, no shared memory and no atomics. Depth is independent of k, so it wins at large N per warp and large k where W1 and W2 grow. Every digit width pays the same 16 (bf16) or 32 (fp32) ballots per item in total, wider digits only amortize the per-pass fixed cost against a longer scan.
cells it should win: warp N 512 to 4096, k 32 to 2048, both types

cost model (placeholders July 2026 B200: ballot 23 latency, shuffle 30, shuffle stage 44 at IPT 1, popc and LOP3 about 2 cycles issue):
- Pass with d-bit digit over IPT items per lane: d IPT ballots pipelined (23 + about 4 d IPT) + bucket counts (2^d buckets, one or more per lane, about 6 IPT) + suffix scan over 2^d buckets (d shuffle steps, 44 each) + crossing test, broadcast, prefix update (about 50). pass(d) about 73 + 4 d IPT + 6 IPT + 44 d.
- Total for B key bits: (B over d) passes. bf16 (B 16): total(d) about (16 over d)(73 + 6 IPT) + 64 IPT + 704. d 4: about 1000 + 88 IPT. d 2: about 1290 + 112 IPT. fp32 doubles everything: d 4 about 2000 + 176 IPT.
- Against N1 (d 1, no scan): N1 about 610 + 128 IPT for bf16. Model crossover at IPT about 10, so W3 wins above N 320 and N1 below. Both are far below W1 at N 256 and above.
- Candidate compaction between passes (explore 2) does not reduce ballots per item unless the surviving items are moved to fewer slots, so model it as a win only for IPT at least 8.
- Index-only output: after the last pass, items matching the full prefix are the pivot ties, items above the prefix are selected. Emission by ballot and popc prefix in slot-major order gives indices in increasing index order, so prefer_smaller_index costs one ballot per slot and nothing else.

focus cells: 1. warp N 512 k 64 bf16, 2. warp N 1024 k 128 fp32, 3. warp N 2048 k 256 bf16 (NSA scored blocks at 128K context are N 2048 k 16, so also run that), 4. warp N 1024 k 16 fp32 (PEER sub-key retrieval, NSA), 5. warp N 4096 k 1024 bf16.
boundary cells: warp N 256 k 16 (expect N1, W5 or W2), warp N 4096 k 2048 (expect W4 or a sort).

explore first: (1) ballot histogram with 4-bit and 5-bit digits, measure pass cost against IPT 4, 16, 64. (2) candidate compaction into fewer slots via ballot prefix when survivors fit in one slot. (3) digit schedule for bf16 (4+4+4+4 against 5+5+6) and fp32 (8 by 4 against 6 by 5 plus 2). (4) index-only emission in slot-major order with the pivot tie count. (5) share the kernel skeleton with N1 (d 1) and report the d sweep as one table.

determinism and ordering: set deterministic by construction. prefer_smaller_index via the slot-major ballot emission. Sorted via a final N2 or W1 pass over k.

main risk: bf16 router and attention keys have few distinct values per binade (128), so the pivot bucket is often a tie class larger than k remaining and the pass count never shrinks by early exit. The scan is a shuffle chain of d steps on the critical path of every pass. If the measured pass cost is above 1.5x the model, W3 loses the k 32 to 128 band to W5 and the rest to N2.

cheapest kill experiment: one 4-bit pass at IPT 8 and IPT 32, slope over passes, against the N1 round at the same IPT. If pass(4 bit) is more than 4 times round(N1), W3 has no cell N1 does not cover and the two merge into one card.

references: knowledge base section 4, `cub/cub/block/specializations/block_topk_air.cuh`, `ideas/queue/N1_bit_serial_select.md`
