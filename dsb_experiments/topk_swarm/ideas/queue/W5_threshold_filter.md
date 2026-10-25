# W5: Threshold filter with verification (exact lane-maxima bound first, sampled guess second)
status: queued (wave 2)
scope: warp (block scope is B4)
origin: `reference/GVR_V2_ALGORITHM_REPORT.md` (TRT-LLM self-sampling GVR), new at warp scope
hypothesis: a warp knows 32 lane maxima after IPT compares. Their k-th largest (k at most 32) is an exact lower bound on the k-th largest of the row (k elements are at least that large), so there is no underflow and no retry. One compare per item keeps candidates at or above it, a ballot prefix compacts them to one per lane, and a sort32 or W2 finish on at most 32 candidates is exact. For k 33 to 64 keep the top 2 per lane (64 values, sort64). Beyond 64 the bound needs a sampled guess (GVR style) with the count verification and a retry rung, or fall back to W3. Candidate count on random input with IPT items per lane: N times (1 minus (1 minus k over 32) to the power 1 over IPT), about k plus a few: N 256 k 8 gives 9, N 128 k 16 gives 20, N 512 k 16 gives 22. At k 32 the bound is the minimum of the lane maxima and admits about a third of the row, so k 32 already needs the top-2 bound. The filter pass is compare-based, so its per-item cost (2 compares and one sixth of a ballot) is below any radix or bit-serial pass, which is why W5 takes over from W2 where k rounds grow and from W3 everywhere its candidate count stays small.
cells it should win: warp N 512 to 1024 with k 32 to 64, and N 2048 to 4096 with k 16 to 64 (NSA at 128K, EAGLE-2 rerank), random input, both types (compare based, no twiddle)

cost model (measured 2026-10-25, `TECHNIQUES.md` T17: dependent ballot 26.7, 6 cycles per further ballot, __popc 23. July placeholders: sort32 660, sort64 1080, shuffle 30):
- Lane maxima: IPT compares (about 2 IPT). Bound: sort32 of the maxima 660 (top-2 per lane: 2 IPT compares plus sort64 1080), broadcast 30.
- Filter and compaction: IPT compares, IPT ballots pipelined (21 + 6 IPT), popc prefix and a predicated register move (about 30), a guard ballot for overflow, about 80 + 8 IPT.
- Finish: sort32 660, or W2 packed at IPT 1 (100 + 30 k, so 340 at k 8 and 580 at k 16, cheaper than the sort below k 20), or sort64 1080 for the top-2 bound. Index-only emission: candidates carry their index in a second register through the compaction, or for bf16 inside the word.
- Total k at most 16 with the W2 finish: about 1100 + 10 IPT + 30 k. N 256 k 16: 1660 against W2 580. N 2048 k 16: 2220 against W2 2370 and W3 8700 (fp32). k 17 to 64 (top-2 bound, sort64 finish): about 2250 + 12 IPT. N 512 k 32: 2440 against W2 1570. N 512 k 64: 2440 against W2 3040, N1 2980, W3 2700. N 1024 k 64: 2630 against W2 5090, W3 4360. N 4096 k 64: 3800 against W2 17k, W3 14.3k.
- Overflow (sorted blocked input, tie floods): candidates exceed 32 or 64, fall back to W3 on the full row, cost is the wasted bound plus the fallback. Striped arrangement makes sorted input benign.

focus cells: 1. warp N 512 k 64 bf16 (EAGLE-2 rerank is N about 600 k 60), 2. warp N 1024 k 64 fp32, 3. warp N 2048 k 16 fp32 (NSA at 128K), 4. warp N 1024 k 32 bf16, 5. warp N 4096 k 64 fp32.
boundary cells: warp N 256 k 16 (W2 expected to win 2x or more), warp N 1024 k 128 (bound needs a sample, W3 expected).

explore first: (1) host simulation of the candidate count of the lane-maxima bound and the top-2 bound on random, sorted and tie-heavy bf16 input in striped arrangement, using the harness generator. (2) compaction into one register per lane via ballot prefix, measured against IPT. (3) the finish per k (W2 packed at IPT 1 against sort32 and sort64). (4) sampled guess with verification for k above 64, retry rung from a second order statistic. (5) note for the batched device level: the threshold of chunk j bounds chunk j+1 of the same row only, never another row (GVR V1 lost 3.6 to 6x on stale hints).

determinism and ordering: exact by counting. Ties at the bound are inside the candidate set and resolved by the finish (slot-major ballot order gives prefer_smaller_index, the packed W2 finish gives it inherently). Sorted if the finish sorts.

main risk: the bound costs a sort32 or sort64 before any item is filtered, so W5 cannot win below k 32 at N up to 1024 against the 30-cycle W2 round, and bf16 ties at the bound can push candidates above 32 or 64 on logit-like data. If the overflow rate on realistic bf16 input exceeds a few percent, the fallback cost dominates and W5 keeps fp32 only.

cheapest kill experiment: explore item 1, minutes, no GPU. If the 99th percentile exceeds 64 candidates at k 32 on bf16 logits at N 512, W5 is dead for bf16. Then one GPU cell: W5 against W2 packed at N 512 k 64 bf16.

references: `reference/GVR_V2_ALGORITHM_REPORT.md` sections 2.2 to 2.7, `TECHNIQUES.md` T17, `ideas/queue/B4_exact_bound_prefilter.md`, `ideas/queue/W2_redux_extraction.md`
