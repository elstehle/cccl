# B4: Exact-bound prefilter then small selection (block)
status: queued (wave 2, candidate for promotion to wave 1 at large N)
scope: block
origin: new
hypothesis: any k elements give a lower bound on the k-th largest, their minimum. The cheap exact bound is the k-th largest of the 256 per-thread maxima (k at most 256), or of the 256 m per-thread top-m values for k up to 256 m. On random striped input the candidate count above the bound is about k plus a few (the quantile argument: F(bound) = (1 minus k over 256) to the power 1 over IPT, so candidates about N k over (256 IPT) = k). The sweep is one max and one compare per item against three to four instructions and an MIO atomic per item per pass in B1, so B4 wins where IPT is large. The finish runs on about k to 2k compacted candidates with B1, B2 or a warp primitive. Overflow (too many candidates, sorted blocked input, tie floods) falls back to B1 exactly. Compare based: fp32 and bf16 native, no twiddle, so it is also the natural pairs-free index-only path.
cells it should win: block N 8192 to 32768, k 16 to 2048, random input, both types

cost model (placeholders July 2026 B200: B1 pass terms, B2 round 80, ballot 23, barrier 28):
- Per-thread top-m: m IPT compares, about 2 m IPT cycles (m 1 for k at most 256, m 8 for k 2048).
- Bound: selection of rank k among 256 m values. k at most 32: B2 rounds on 256 items, k times 75, or per-warp sort32 and a one-warp merge of 8 runs, about 1600. k 33 to 256: B1 on 256 m items, 2 passes with IPT m, about 1000 to 1400. So bound(k) about min(75 k, 1000 to 1400), measure both.
- Sweep and compaction: 1 compare per item plus ballot prefix per slot plus per-warp atomic base plus predicated STS of (key, index), about 15 IPT + 100, one barrier.
- Finish on about k to 2k candidates: B1 2 passes on candidates (about 450 + 85 C over 256 per pass), or sort32 when C at most 32, or B2 rounds when k at most 16.
- Totals: N 8192 k 64 about 64 + 1200 + 580 + 900 = 2700 against B1 6900 and B3 4100. N 32768 k 256 about 256 + 1400 + 2000 + 1100 = 4800 against B1 24.5k and B3 15.5k. N 32768 k 2048 (m 8): 2000 + 1400 + 2000 + 2500 = 7900 against B1 above 25k. N 1024 k 16: 8 + 1200 + 160 + 700 = 2100 against B2 1640 and B1 1700 (B4 loses at small N because the bound costs a selection on 256 items).

focus cells: 1. block N 32768 k 256 fp32, 2. block N 8192 k 64 bf16, 3. block N 32768 k 2048 fp32 (DSA V3.2 chunk), 4. block N 16384 k 1024 fp32 (DSA V4 Pro chunk), 5. block N 8192 k 16 bf16.
boundary cells: block N 2048 k 64 bf16 (B1 expected to tie or win), block N 32768 k 4096 (m 16, bound selection on 4096 items, B3 expected).

explore first: (1) host simulation of the candidate count distribution of the top-m bound on random, sorted, tie-heavy and bf16 logit inputs in blocked and striped arrangement (striped makes sorted input benign, blocked makes it the worst case), no GPU. (2) the bound primitive per k range (B2 rounds against B1 on 256 m values against per-warp sort plus merge). (3) compaction cost against IPT. (4) the finish per candidate count. (5) overflow guard and fallback cost (the fallback is the full B1, the waste is the bound plus the sweep).

determinism and ordering: exact by counting. The finishing primitive decides tie-break and ordering, index-only output is natural because candidates carry (key, index).

main risk: the bound selection on 256 m values costs 1000 to 1600 cycles before any item is touched, so B4 cannot win below N 4096, and the candidate count on bf16 logit data (ties at the bound) may be several k, which pushes the finish toward a full B1 on candidates. Register pressure: the per-thread top-m for m 8 is 8 live keys and indices on top of the input.

cheapest kill experiment: the host simulation of explore item 1 (minutes, no node). If the 99th percentile candidate count exceeds 4 k on random striped bf16 logits at N 8192 k 64, B4 is dead for bf16 and keeps fp32 only. Then one GPU cell: B4 against B3 at N 32768 k 256 fp32.

references: `reference/GVR_V2_ALGORITHM_REPORT.md` section 2.3 (register-resident bracket without a sample), `ideas/queue/W5_threshold_filter.md` (same bound at warp scope), `ideas/queue/N3_streaming_running_threshold.md`
