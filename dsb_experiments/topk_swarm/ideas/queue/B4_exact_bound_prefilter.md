# B4: Exact-bound prefilter then small selection (block)
status: queued (wave 2)
scope: block
origin: new
hypothesis: any k elements give a lower bound on the k-th largest, their minimum. A cheap exact bound is the k-th largest of the per-thread maxima, one fast selection on N equal to the thread count. On random input the expected candidate count above the bound is close to k, so the finish runs on about k to 2k candidates with W1 or B2. An exact fallback to B1 handles overflow (sorted or adversarial input). Also the sampled GVR guess at block scope when the exact bound is too loose.
cells it should win: block N 4096 to 32768, k 4 to 256, random input
cost model: one per-thread max over items per thread plus one selection on 256 maxima plus one compare per item plus compaction plus finish on about 2k
explore first: (1) measure the candidate-count distribution for the exact bound on random, sorted and tie-heavy input in blocked and striped arrangement. (2) the finish primitive for about 2k candidates. (3) the overflow fallback and its cost. (4) the second-level bound (k-th largest of per-warp top-k) when k exceeds the thread count.
determinism and ordering: exact by counting. Finishing primitive decides ordering.
references: `reference/GVR_V2_ALGORITHM_REPORT.md` section 2.3 (register-resident bracket without a sample)
