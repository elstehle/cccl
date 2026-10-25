# N1: Bit-serial ballot select (1-bit radix select without histogram or scan, warp and sub-warp)
status: queued (wave 2, the d 1 row of W3's table, narrowed by the measured ballot and redux costs)
scope: warp, subwarp
origin: new (seed from the lead: one ballot count per key bit). Radix select with a 1-bit digit, where the "histogram" is a single popcount and the "scan" is one compare.
hypothesis: walk the twiddled key from the most significant bit down. Per round: each lane ballots (still a candidate and bit set) for each of its IPT items, sums the popcounts, and compares the count c with the number still needed r. If c is at least r the k-th key has the bit set, so candidates without it drop out. Otherwise all c items with the bit set are selected, r decreases by c, and candidates with the bit set drop out. 16 rounds for bf16, 32 for fp32, no shared memory, no atomics, no shuffle scan, cost independent of k, trivially sub-warp through the ballot mask. Index-only emission: the final tie class gives exactly r items in slot-major ballot order, which is increasing index order for a striped arrangement, so prefer_smaller_index is free.
cells it should win: warp N 256 k 128 bf16 and N 128 k 64 bf16, and that is about all. The measured packed W2 round (26 cycles up to IPT 8, `TECHNIQUES.md` T19) beats 16 ballot rounds up to k about 69 at N 256, above N 384 W3 amortizes the ballots better, and fp32 doubles the rounds. The card stays as the d 1 row of W3's sweep.

cost model (measured 2026-10-25, `TECHNIQUES.md` T17 and T4: dependent ballot 26.7, about 6 cycles per further independent ballot including its predicate, __popc 23, __ffs 40, `fns` emulated at 177 so never used):
- Round(IPT) about 27 + 6 (IPT minus 1) ballots + 10 for the first popc latency with the rest overlapped + 2 IPT adds + 15 compare and mask update = 46 + 8 IPT. IPT 1: 54, IPT 4: 78, IPT 8: 110, IPT 16: 174.
- Emission: IPT ballots in slot-major order (21 + 6 IPT) plus popc prefixes (30) plus 50, about 100 + 6 IPT. No `fns`.
- bf16: 16 rounds plus emission about 840 + 134 IPT. N 128: 1380, N 256: 1910, N 512: 2980. fp32: about 1570 + 262 IPT, N 256: 3670.
- Against W2 packed (100 + 26 k at IPT at most 8, measured round): N1 wins at k above about 69 at N 256 and 49 at N 128, so only the k 128 cell at N 256 and the k 64 cell at N 128 remain. Against W5 (2210 + 10 IPT at k 17 to 64): N1 wins at N 256 k 64 by 15 percent and loses at N 512. Against W3 at 4-bit digits (1028 + 104 IPT): N1 wins below IPT 12 (N 384) by having no scan, W3 above.
- Early termination (c equals r exactly) is rare on bf16 logit data because of ties, model without it.

focus cells: 1. warp N 256 k 64 bf16, 2. warp N 256 k 128 bf16, 3. warp N 128 k 64 bf16, 4. warp N 256 k 64 fp32 (expected to lose to W2 with N4, kept as the type control), 5. sub-warp LW 8 N 64 k 32 bf16 (4 rows per warp).
boundary cells: warp N 256 k 32 (W2 expected to win), warp N 512 k 64 (W5 or W3 expected to win).

explore first: (1) round cost at IPT 1, 4, 8, 16 as a slope over rounds, one kernel, against the measured ballot pipelining of T17. (2) per-item candidate flags in one register per lane (IPT bits, one LOP3 per round) against IPT predicate registers. (3) emission by popc prefixes in slot-major order. (4) the d sweep together with W3 (d 1, 2, 4, 5) in one table, which is the real deliverable of this card.

determinism and ordering: set deterministic by construction. prefer_smaller_index free by slot-major emission. Sorted via N2 or W1 over k.

main risk: the measured ballot pipelining (6 cycles per further ballot, not 4) already moved N1 from the k 32 band to the k 64 band, and 16 dependent rounds leave no slack. The remaining band (N 128 to 256, k 64 to 128, bf16) is narrow and W5's top-2 bound covers its upper half.

cheapest kill experiment: round cost at IPT 8 against the W3 4-bit pass at IPT 8 in one session, and W2 packed at N 256 k 64. If round(8) is above 120 cycles or W2 is within 10 percent at k 64, N1 has no cell of its own and stays only as the d 1 row of W3's table.

references: `ideas/queue/W3_warp_radix_select.md`, `ideas/queue/W2_redux_extraction.md`, `TECHNIQUES.md` T4, T17, `reference/WARP_RADIX_SORT_RESULTS.md` (ballot ranking at IPT 1 beats merge sort)
