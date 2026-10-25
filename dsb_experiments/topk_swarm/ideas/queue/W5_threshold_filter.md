# W5: Threshold filter with verification (exact lane-maxima bound first, sampled guess second)
status: queued (wave 2)
scope: warp (block scope is B4)
origin: `reference/GVR_V2_ALGORITHM_REPORT.md` (TRT-LLM self-sampling GVR), new at warp scope
hypothesis: a warp knows 32 lane maxima after IPT compares. Their k-th largest (k at most 32) is an exact lower bound on the k-th largest of the row (k elements are at least that large), so there is no underflow and no retry. One compare per item keeps candidates at or above it, a ballot prefix compacts them to one per lane, and a sort32 or N1 finish on at most 32 candidates is exact. For k 33 to 64 keep the top 2 per lane (64 values, sort64). Beyond 64 the bound needs a sampled guess (GVR style) with the count verification and a retry rung, or fall back to W3. Candidate count on random input with IPT items per lane: N times (1 minus (1 minus k over 32) to the power 1 over IPT), which is about k plus a few: N 256 k 8 gives 9, N 128 k 16 gives 20, N 512 k 16 gives 22. At k 32 the bound is the minimum of the lane maxima and admits about a third of the row, so k 32 already needs the top-2 bound.
cells it should win: warp N 128 to 1024, k 16 to 64, random input, both types (compare based, no twiddle)

cost model (placeholders July 2026 B200: sort32 660, sort64 1080, ballot 23, shuffle 30):
- Lane maxima: IPT compares (about 2 IPT). Bound: sort32 of the maxima 660 (top-2 per lane: 2 IPT compares plus sort64 1080), broadcast 30.
- Filter and compaction: IPT compares, IPT ballots pipelined plus popc and a predicated register move, about 22 IPT, plus a guard ballot for overflow (more than 32 or 64 candidates).
- Finish: sort32 660 or N1 on 32 items about 640 or W2 at IPT 1 about 82 k. Index-only emission: candidates carry their index in a second register through the compaction, or for bf16 inside the word.
- Total k at most 16: about 1350 + 24 IPT. N 256 k 16 about 1540 against W2 1670 and N1 1630. k 17 to 64: about 2200 + 26 IPT. N 512 k 32 about 2620 against N1 2660 and W3 2400. A three-way tie band, measurement decides.
- Overflow (sorted or tie-heavy input): candidates exceed the buffer, fall back to W3 or N1 on the full row, cost is the wasted bound plus the fallback.

focus cells: 1. warp N 256 k 16 bf16 (Kimi K3 style routing at k 16), 2. warp N 512 k 16 fp32 (NSA selection at 32K context), 3. warp N 1024 k 32 fp32, 4. warp N 128 k 16 bf16, 5. warp N 1024 k 64 bf16 (EAGLE-2 rerank is N about 600 k 60).
boundary cells: warp N 256 k 8 (W2 expected to win), warp N 1024 k 128 (bound needs a sample, W3 or N1 expected).

explore first: (1) measure the candidate count distribution of the lane-maxima bound and the top-2 bound on random, sorted and tie-heavy bf16 input in striped arrangement (striped decorrelates sorted input, blocked does not). (2) compaction into one register per lane via ballot prefix, measure against IPT. (3) the finish choice (sort32, N1, W2) per k. (4) sampled guess with verification for k above 64, retry rung from a second order statistic. (5) note for the batched device level: the threshold of chunk j is a valid lower bound for chunk j+1 of the same row only, never across rows (GVR V1 lost 3.6 to 6x on stale hints).

determinism and ordering: exact by counting. Ties at the bound are inside the candidate set and resolved by the finish (slot-major ballot order gives prefer_smaller_index). Sorted if the finish sorts.

main risk: the bound costs a sort32 before any item is filtered, so W5 can never beat W2 at k 8, and bf16 ties at the bound can push candidates above 32 on logit-like data (128 distinct values per binade). If the overflow rate on realistic bf16 input exceeds a few percent, the fallback cost dominates.

cheapest kill experiment: host simulation of the candidate count for the two bounds on the harness input generator (random fp32, random bf16 from a logit range, sorted, tie-heavy) at N 128 to 1024, no GPU needed. If the 99th percentile exceeds 64 candidates at k 16 on bf16 logits, W5 is dead for bf16 and keeps fp32 only.

references: `reference/GVR_V2_ALGORITHM_REPORT.md` sections 2.2 to 2.7, `ideas/queue/B4_exact_bound_prefilter.md`
