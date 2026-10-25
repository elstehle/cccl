# B5: Hierarchical warp then block composition
status: rejected (2026-10-25, closed in wave 1 by opt-W2's measurement against B2's adapter, see `ideas/rejected/R13_B5_hierarchical.md`)
scope: block

Two packed W2 phases and one exchange cost 1194 cycles at N 1024 k 16 bf16 against B2_packed's 1380 (1.16x, the card's threshold by one percent) with 0.73x the throughput, and B2 wins or ties every other cell (cards B5_hier_b200_20261025_1 and _2). B2 owns block extraction at small k, the packed W2 round owns the warp.
