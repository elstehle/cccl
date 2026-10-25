# N1: Bit-serial ballot select (the 1-bit row of W3, kept for sub-warp rows at large k)
status: queued (wave 2, inside the W3 and N6-warp optimizer, no cells of its own at warp scope)
scope: warp, subwarp
origin: new (seed from the lead: one ballot count per key bit). Radix select with a 1-bit digit, where the "histogram" is a single popcount and the "scan" is one compare.
hypothesis: walk the twiddled key from the most significant bit down. Per round: each lane ballots (still a candidate and bit set) for each of its IPT items, sums the popcounts, and compares the count c with the number still needed r. If c is at least r the k-th key has the bit set, so candidates without it drop out. Otherwise all c items with the bit set are selected, r decreases by c, and candidates with the bit set drop out. 16 rounds for bf16, 32 for fp32, no shared memory, no scan, cost independent of k. At sub-warp scope one full-warp ballot serves every logical warp of the hardware warp at once (the per-row count is the popcount of the row's byte of the mask), which is the form T37 recommends for sub-warp collectives, so N1's cost per hardware warp does not grow with the number of rows.
cells it should win: sub-warp LW 8 and LW 4 rows at N 128 with k 64 (W2 sw_v2 measured 5261 and 11526 there). At warp scope none: the measured W2 round (26 cycles, T38) and the W5 bound take every cell up to k 64, N6 warp takes k 128 and above.

what it must beat (B200, 2026-10-25): W2_packed_sw_v2 bf16 LW 8 N 128 k 64 5261, LW 4 N 128 k 64 11526, LW 16 N 128 k 64 2619, LW 8 N 128 k 16 1249. At warp scope W2_packed_v3 N 256 k 64 1868, N 128 k 64 1783 (v1). To matter N1 has to be 1.5x under W2 sw_v2 at LW 8 N 128 k 64 (at most 3500) and it is predicted at about 3000.

cost model (measured 2026-10-25, `TECHNIQUES.md` T17 and T4: dependent ballot 26.7, about 6 cycles per further independent ballot including its predicate, __popc 23, __ffs 40, `fns` emulated at 177 so never used):
- Round(IPT) about 27 + 6 (IPT minus 1) ballots + 10 for the first popc latency with the rest overlapped + 2 IPT adds + 15 compare and mask update = 46 + 8 IPT. IPT 4: 78, IPT 8: 110, IPT 16: 174, IPT 32: 302.
- Emission: IPT ballots in slot-major order (21 + 6 IPT) plus popc prefixes (30) plus 50, about 100 + 6 IPT. No `fns`.
- bf16 warp: 16 rounds plus emission about 840 + 134 IPT (N 128 1380, N 256 1910, N 512 2980), fp32 about 1570 + 262 IPT. Against W2_packed_v3 (N 256 k 64 1868, N 512 k 64 2704) N1 wins nothing at warp scope.
- Sub-warp, G rows per hardware warp at LW 32 over G: the same ballots count every row, the per-row counts are G popcounts over bytes (pipelined), the per-row prefix update is G selects, so the round is about 46 + 8 IPT + 4 G with IPT the items per lane. LW 8 N 128 (IPT 16, G 4): about 190 per round, 3040 bf16 for the four rows against W2 sw_v2 5261 at k 64 (1.7x) and 1249 at k 16 (W2 wins). LW 4 N 128 (IPT 32, G 8): about 330 per round, 5300 against 11526 at k 64 (2.2x).
- Early termination (c equals r exactly) is rare on bf16 logit data because of ties, model without it.

focus cells: 1. sub-warp LW 8 N 128 k 64 bf16, 2. sub-warp LW 4 N 128 k 64 bf16, 3. sub-warp LW 8 N 128 k 64 fp32, 4. warp N 256 k 128 bf16 (the one warp cell inside the W3 sweep), 5. sub-warp LW 16 N 128 k 64 bf16 (W2 2619, expected to lose).
boundary cells: sub-warp LW 8 N 128 k 16 (W2 1249 expected to win 2x), warp N 512 k 64 (W5 and N6 warp expected).

explore first: (1) round cost at IPT 8, 16 and 32 as a slope over rounds, one kernel, with G 1, 4 and 8 rows. (2) per-item candidate flags in one register per lane (IPT bits, one LOP3 per round) against IPT predicate registers. (3) emission by popc prefixes in slot-major order per row. (4) the d sweep together with W3 (d 1, 2, 4, 5) for the knowledge base.

determinism and ordering: set deterministic by construction. prefer_smaller_index free by slot-major emission. Sorted via W1 over k.

main risk: 16 dependent rounds at 6 cycles per ballot leave N1 no warp cell, and at sub-warp scope W2 sw_v2 already holds k at most 16. The k 64 sub-warp cells are two named cells at most (no router asks for k 64 of 128).

cheapest kill experiment: N1 against W2_packed_sw_v2 at LW 8 N 128 k 64 bf16 in one session. If N1 is not at least 1.5x faster (at most 3500 cycles), it has no cell and stays as the d 1 row of W3's table only.

references: `ideas/queue/W3_warp_radix_select.md`, `ideas/queue/W2_redux_extraction.md`, `TECHNIQUES.md` T4, T17, T37, T38, cards W2_packed_sw_v2 and W2_packed_v3, `reference/WARP_RADIX_SORT_RESULTS.md` (ballot ranking at IPT 1 beats merge sort)
