# R9: Exact rank by pairwise comparison through shuffles for N at most 32
status: rejected on the cost model (2026-10-25, algo-researcher)
scope: warp, subwarp
seed: own idea for the one-item-per-lane routing cells
idea: with one item per lane, each lane counts the lanes holding a larger key by 31 shuffle-and-compare steps, the rank is exact, items with rank below k are the answer and prefer_smaller_index falls out of the comparison.
reason: 31 dependent shuffles at about 35 cycles each is about 1100 cycles, against N1 at 16 rounds of about 40 (640), W2 at k times 82 (656 at k 8) and a sort32 at 660. The GVR kernel uses the quadratic rank only for its crossing bin of up to 288 candidates in shared memory where the alternative is another narrowing level, not as a primary primitive.
what survives: nothing at warp scope. The ballot-ranked radix sort study already showed that at IPT 1 ballot ranking is the cheap ranking machine (`reference/WARP_RADIX_SORT_RESULTS.md`), which is N1.
