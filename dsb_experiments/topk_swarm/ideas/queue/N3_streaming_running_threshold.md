# N3: Streaming block selection with a running exact threshold (chunked input, DSA style)
status: queued (wave 2, last, one cell, streaming contract only)
scope: block
origin: new at block scope, the block analog of PR #9281's `buffered` warp algorithm. Seeds from the lead: DSA style chunked selection with merge, threshold reuse.
hypothesis: when a row arrives in chunks (a DSA query over 128K scores, a vocabulary over 32 chunks), the exact k-th value of everything seen so far is a lower bound for the rest of the same row. Initialize the retained top-k from the first chunk, sweep every further chunk with one compare per item, compact the candidates into a buffer, and merge (top-k of retained plus buffer) only when the buffer fills. On random order about k (1 + ln(N over C)) items ever become candidates. The row is read once and nothing is materialized per chunk, which is the "insert chunk, keep k" interface a device agent over many chunks per query could use. On resident input it is B4 with the bound learned progressively instead of in one shot, and wave 1 priced every piece it is made of.
cells it should win: none in the resident grid. Streaming cells only: N 32768 and above per row fed as chunks of 1024 to 4096, k 256 to 2048, random order.

what it must beat (B200, 2026-10-25): B2_twolevel N 32768 bf16 k 16 4774, k 64 22947, B1_idx N 32768 k 256 18686 and 21788, B3_tuned N 32768 k 256 19167 and 20071, k 2048 19253 and 20804, and B4 (predicted 6k to 9k at N 32768 k 64 to 256). To matter N3 has to be within 1.2x of B4 on resident input, or it is a contract note.

cost model (measured constants from wave 1: B1 pass over C items about 85 C over 256 + 410, B3 compaction sweep about 12 instructions per item slot plus a 300-cycle packed scan plus padded loads (T29, T31, B3 cards), B1_idx at N 1024 about 1770 for the init selection):
- Init: top-k of the first chunk C by B1_idx or N6 (1770 at C 1024, about 3400 at C 4096 for k 2048).
- Sweep per chunk: 1 compare per item, count, predicated store of (key, index) into the candidate buffer, about 4 instructions per item slot, plus one packed scan per chunk for the positions (300) or a per-warp atomic base (T22 says per-slot aggregation is the wrong shape, per-item tickets are right).
- Merge: B1 passes on retained k plus buffer B (B equal to k): 2 (85 times 2 k over 256 + 410), about 1160 at k 256 and 3550 at k 2048.
- N 32768 k 256 resident, C 1024: 1770 + 31 chunks times (31 slots times 4 plus 300) = 1770 + 13100 + ln 32 times 256 candidates over 256 per merge gives 4 merges times 1160 = 4640, about 19.5k against B4 6k to 9k and B1_idx 18.7k. The per-chunk scan is the killer, with per-item tickets instead (about 1 cycle per selected item) the sweep is about 2000 and the total about 8400, still above B4.
- Sorted ascending chunk order: every item is a candidate, N over B merges, catastrophic. A fixed pseudo-random chunk order defuses sorted input, not adversarial input. Budget guard: above 8 k candidates switch to B1 on the remainder.

focus cells: 1. block N 32768 k 256 fp32 fed as 32 chunks of 1024 (against B4 and B3_tuned on resident input), 2. block N 32768 k 2048 fp32 as 8 chunks of 4096 (DSA V3.2 chunk), 3. block N 16384 k 1024 fp32 as 4 chunks (DSA V4 Pro).
boundary cells: block N 8192 k 16 (B2 expected), block N 32768 k 4096 (merge of 8192 items each time, B3 expected).

explore first: (1) one cell against B4 and B3_tuned on resident input, N 32768 k 256 fp32, same session, after B4 exists. (2) the chunk sweep with per-item tickets against the packed scan. (3) candidate count on random, sorted and bf16 logit input (host simulation first).

determinism and ordering: exact. The merge primitive decides tie-break and order, candidates carry (key, index).

main risk: on resident input B4 computes the bound in one shot and sweeps once, so N3 pays the same sweep plus ln(N over C) merges and a position bookkeeping per chunk. N3 earns its place only if the device agent wants the streaming contract, which the lead decides.

cheapest kill experiment: explore item 1. If N3 is more than 1.2x slower than B4 on resident input, retire it to the streaming note in `ideas/WORKLOADS.md` and the knowledge base, no further GPU time.

references: `ideas/queue/B4_exact_bound_prefilter.md`, PR #9281 `buffered` algorithm, `reference/GVR_V2_ALGORITHM_REPORT.md` section 2.4, `TECHNIQUES.md` T22, T29, T31, cards B3_tuned_b200_20261025_2 and B2_twolevel, `ideas/WORKLOADS.md` section 7
