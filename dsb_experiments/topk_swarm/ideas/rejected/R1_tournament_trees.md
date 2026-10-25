# R1: Tournament trees over per-thread sorted runs
status: rejected (2026-10-25, algo-researcher)
scope: warp, block
seed: the lead's list (tournament trees over per-thread sorted runs)
idea: each thread sorts its items into a run, a loser tree over the run heads yields the global maximum, and after each extraction only the path from the winner's leaf to the root is re-evaluated (log2 of the thread count nodes).
reason: on a SIMT warp the full `redux.sync` max costs 22 cycles while a 5-step shuffle tree costs 150 (`reference/BLOCK_TOPK_RESULTS.md` section 1), so the "path only" update is slower than recomputing the whole reduction. At block scope the hierarchical form (warp pre-reduction then one atomic per warp) measured 148 cycles per round against 74 for the flat fire-and-forget atomic burst, because the MIO combiner already does the tree for free. B2 and W2 are the tournament tree with the hardware as the tree.
evidence: `proto_topk_micro.cu` composed block-wide max table (classic 290, atomic256 74, hybrid8 275, hybrid8 with redux 148), July 2026 B200, one design point.
what survives: nothing separate. The per-thread sorted run as the source of the next candidate is already in B2 and W2 (shift-down list or head pointer).
