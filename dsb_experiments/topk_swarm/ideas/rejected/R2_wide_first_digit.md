# R2: Bucket select with a wide first digit for 8-bit and 16-bit keys
status: rejected for 16-bit keys, not an idea for 8-bit keys (2026-10-25, algo-researcher)
scope: block
seed: the lead's list (bucket select with a wide first digit for 8-bit and 16-bit keys)
idea: select bf16 keys in one radix pass with a 16-bit digit (65536 bins), fp8 keys in one pass with 256 bins.
reason: the scan over the bins is on the critical path of every pass and grows superlinearly with the bin count. At N 1024 the AIR sweep measured 10-bit digits at 2063, 11-bit at 2404 and 12-bit at 3342 cycles against 1779 for 8-bit, with shared memory at 9 to 37 KB and occupancy falling, and the promised pass saving never materialized (`reference/BLOCK_TOPK_AIR_OPT_RESULTS.md` section 6). A 65536-bin histogram needs 256 bins per thread in the scan alone, several times a whole 8-bit pass, plus 64 KB of counters even at one byte each. For fp8 keys a single 8-bit pass is simply B1 with one pass and belongs to B1's type sweep in the full evaluation.
what survives: the goal (one discriminating pass for bf16) is pursued by N6 with 256 equi-width bins over the value bracket, which gets the discrimination from the data range instead of from more bins.
