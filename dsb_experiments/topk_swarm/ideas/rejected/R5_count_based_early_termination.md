# R5: Count-based early termination in radix passes
status: rejected as a new idea, already present (2026-10-25, algo-researcher)
scope: block, warp
seed: the lead's list (count-based early termination in radix passes)
idea: stop the pass loop as soon as the pivot bucket's count equals the number of keys still needed, or switch to a cheaper finisher when the candidate count is small.
reason: `block_topk_air` already exits when the bucket count equals k remaining (the AIR sweep's "pass count that matters, 2, via early exit", `reference/BLOCK_TOPK_AIR_OPT_RESULTS.md` section 6), and the small-candidate switch is the measured FINISH knob of the same study (a tail bound at a 52 cycle common-case cost). Nothing in a histogram pass can terminate before all items are counted, so the only lever is the pass count, which is the subject of B1 explore items 5 and 6, of B3 (sweep only the candidates) and of N6 (one discriminating pass). For bit-serial selection (N1) the exact-count termination is listed but is rare on bf16 data because of ties.
what survives: B1 explore items 5 and 6, N1's note on early termination.
