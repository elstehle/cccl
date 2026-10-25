# R6: `match.any` pre-aggregation of histogram atomics
status: rejected, confirmed by measurement (2026-10-25, algo-researcher, hardware researcher T5: `match.any.sync` costs 51 cycles with one distinct value, 62 at 2, 83 at 4, 128 at 8, 215 at 16 and 392 at 32 distinct values on B200 and GB200, never under 10)
scope: block
seed: own idea while sharpening B1, the W3 card's original "match.any histograms"
idea: before the shared atomic, group the lanes with the same digit by `match.any.sync`, let the leader add the popcount, so same-address atomics shrink by the multiplicity.
reason: the fire-and-forget same-address burst is already combined by the MIO at about one operation per cycle and the histogram phase measured contention-insensitive (`reference/BLOCK_TOPK_RESULTS.md` section 1, `reference/BLOCK_TOPK_AIR_OPT_RESULTS.md` section 6). Pre-reducing with `redux.sync` before an atomic made rounds slower (148 against 74) for the same reason. `match.any` is a slow instruction on every architecture measured so far and would add a dependent op in front of work the MIO does for free. Expected negative.
what survives: nothing. The cost grows with the number of distinct values in the warp, so it is worst exactly where a histogram has work to do.
