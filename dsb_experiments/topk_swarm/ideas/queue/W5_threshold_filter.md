# W5: Threshold filter with verification (GVR style, warp and block)
status: queued (wave 2)
scope: warp, block
origin: `reference/GVR_V2_ALGORITHM_REPORT.md` (TRT-LLM self-sampling GVR), new at warp and block scope
hypothesis: guess a threshold from a cheap sample (per-lane maxima sorted by a small bitonic network, or the top-k of a sub-sample), one pass keeps candidates at or above it, finish exactly on the candidates with W1 or W3, count to verify, retry lower on underflow, fall back on overflow. For N much larger than k the filter pass is one compare per item and the finish is tiny.
cells it should win: N at least 16 times k, random input, both scopes
cost model: sample cost plus one compare per item plus compaction plus finish on about 2k to 4k candidates, plus the retry probability times the retry cost
explore first: (1) sample choice and its candidate-count distribution (measure, not model). (2) compaction of candidates via ballot prefix into registers or smem. (3) the retry and fallback paths and their cost on sorted and tie-heavy input. (4) threshold reuse across rows in the batched setting (note for the device level, not measured here).
determinism and ordering: exact by counting. Ties at the threshold resolved in the finishing stage. Ordering from the finishing primitive.
references: `reference/GVR_V2_ALGORITHM_REPORT.md` sections 2.2 to 2.6
