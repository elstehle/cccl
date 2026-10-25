# R6: `match.any` pre-aggregation of histogram atomics
status: rejected on expectation, one number from the hardware researcher would settle it (2026-10-25, algo-researcher)
scope: block
seed: own idea while sharpening B1, the W3 card's original "match.any histograms"
idea: before the shared atomic, group the lanes with the same digit by `match.any.sync`, let the leader add the popcount, so same-address atomics shrink by the multiplicity.
reason: the fire-and-forget same-address burst is already combined by the MIO at about one operation per cycle and the histogram phase measured contention-insensitive (`reference/BLOCK_TOPK_RESULTS.md` section 1, `reference/BLOCK_TOPK_AIR_OPT_RESULTS.md` section 6). Pre-reducing with `redux.sync` before an atomic made rounds slower (148 against 74) for the same reason. `match.any` is a slow instruction on every architecture measured so far and would add a dependent op in front of work the MIO does for free. Expected negative.
what survives: a request to the hardware researcher for the `match.any.sync` cost on B200 and GB200 (one T entry). If it is below 10 cycles the idea returns as a B1 variant.
