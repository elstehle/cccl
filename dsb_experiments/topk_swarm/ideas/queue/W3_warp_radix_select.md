# W3: Warp radix select (ballot histograms, 2 to 5 bit digits)
status: queued (wave 2, N1 is its 1-bit degenerate case and shares the optimizer)
scope: warp
origin: `reference/WARP_RADIX_SORT_RESULTS.md` (ballot-ranked LSD), radix select as in `block_topk_air` at warp scope
hypothesis: digit histograms by ballot over the warp, one lane per bucket counts with popc over the combined bit masks, a shuffle suffix scan over the buckets finds the one holding rank k, the prefix narrows, no shared memory and no atomics. Depth is independent of k, so it wins at large N per warp and large k where W2's rounds and the networks grow. Every digit width pays the same 16 (bf16) or 32 (fp32) ballots per item in total, wider digits only amortize the per-pass fixed cost against a longer scan.
cells it should win: warp N 1024 to 4096, k 128 to 2048, both types. Below k 128 the packed W2 round wins even at N 4096 on the model, and k 32 to 64 at N 512 to 1024 goes to W5.

cost model (measured 2026-10-25, `TECHNIQUES.md` T17: dependent ballot 26.7, about 6 cycles per further independent ballot including its predicate, __popc 23, shuffle 4.9 per further shuffle. July placeholders: shuffle stage 44 at IPT 1):
- Pass with d-bit digit over IPT items per lane: d IPT ballots pipelined (21 + 6 d IPT) + bucket counts (one bucket per lane for d at most 5, one LOP3 of d masks and a popc per item, about 2 IPT + 10) + suffix scan over 2^d buckets (d shuffle steps, 44 each) + crossing test, broadcast, prefix update (about 50). pass(d) about 81 + 6 d IPT + 2 IPT + 44 d.
- Total for B key bits: (B over d) passes. bf16 (B 16): total(d) about (16 over d)(81 + 2 IPT) + 96 IPT + 704. d 4: about 1030 + 104 IPT. d 2: about 1350 + 112 IPT. d 5 (5+5+6): about 950 + 100 IPT. fp32 doubles: d 4 about 2060 + 208 IPT.
- Examples, bf16 d 4: N 512 2700, N 1024 4360, N 4096 14.3k. fp32 d 4: N 1024 8700, N 4096 28.7k.
- Against W2 packed (100 + k round(IPT), round 14 + 2 IPT above IPT 8): W3 wins at N 1024 above k about 55 for bf16 and about 110 for fp32 with N4, at N 4096 above k about 50 bf16. Against N1 (840 + 134 IPT): W3 wins above IPT 12. Against W5 at k 64 (2210 + 10 IPT): W5 wins at N 512 to 1024, W3 at N 2048 and above only for k above 64 where W5 needs a sample.
- Candidate compaction between passes does not reduce ballots per item unless survivors move to fewer slots, so model it as a win only for IPT at least 16 after pass 1.
- Index-only output: items above the final prefix are selected, items equal to it are the tie class. Emission by ballot and popc prefix in slot-major order gives increasing index order, so prefer_smaller_index costs one ballot per slot and nothing else.

focus cells: 1. warp N 1024 k 128 fp32, 2. warp N 2048 k 256 bf16, 3. warp N 4096 k 1024 bf16, 4. warp N 4096 k 512 fp32, 5. warp N 2048 k 128 bf16.
boundary cells: warp N 1024 k 32 (W2 expected to win 2x), warp N 4096 k 2048 (W4 or a sort expected).

explore first: (1) ballot histogram with 4-bit and 5-bit digits, pass cost against IPT 16, 64, 128. (2) candidate compaction into fewer slots via ballot prefix after pass 1. (3) digit schedule for bf16 (4+4+4+4 against 5+5+6) and fp32 (8 by 4 against 6 by 5 plus 2). (4) index-only emission in slot-major order with the pivot tie count. (5) the d sweep with N1 (d 1, 2, 4, 5) in one table.

determinism and ordering: set deterministic by construction. prefer_smaller_index via the slot-major ballot emission. Sorted via a final N2 or W1 pass over k.

main risk: bf16 keys have 128 distinct values per binade, so the pivot bucket is often a tie class larger than k remaining and the pass count never shrinks by early exit. The scan is a shuffle chain of d steps on the critical path of every pass. The per-item ballot cost (6 cycles per ballot, 16 or 32 ballots per item) is a floor no digit width changes, so W3 cannot beat a compare-based filter (W5, B4 style) wherever that filter's candidate count stays small.

cheapest kill experiment: one 4-bit pass at IPT 32 and IPT 128, slope over passes, against W2 packed at k 128 and W5 at k 64 in the same session. If W3 at N 1024 k 128 is not within 10 percent of the better of the two, its cells shrink to N 2048 and above with k 256 and above.

references: knowledge base section 4, `cub/cub/block/specializations/block_topk_air.cuh`, `TECHNIQUES.md` T17, `ideas/queue/N1_bit_serial_select.md`, `ideas/queue/W5_threshold_filter.md`
