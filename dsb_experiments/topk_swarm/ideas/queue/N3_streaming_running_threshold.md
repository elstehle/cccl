# N3: Streaming block selection with a running exact threshold (chunked input, DSA style)
status: queued (wave 2, low priority, streaming contract only)
scope: block
origin: new at block scope, the block analog of PR #9281's `buffered` warp algorithm. Seeds from the lead: DSA style chunked selection with merge, threshold reuse.
hypothesis: when a row arrives in chunks (global or shared memory, a DSA query over 128K scores, a vocabulary over 32 chunks), the exact k-th value of everything seen so far is a lower bound for the rest of the same row. Initialize the retained top-k from the first chunk (B1 or B2), then sweep every further chunk with one compare per item, ballot-compact the candidates into a buffer, and merge (top-k of retained plus buffer, B1 on 2k items or N2 for small k) only when the buffer fills. On random order about k (1 + ln(N over C)) items ever become candidates, so merges are about ln(N over C). The row is read once, nothing is materialized per chunk, and the primitive's interface is "insert chunk, keep k", which is what a device agent over many chunks per query can feed. It replaces the independent per-chunk top-k plus a final merge over M k candidates.
cells it should win: none in the resident grid on the model (B4 is cheaper when all N is resident). Streaming cells: N 32768 and above per row fed as chunks of 1024 to 4096, k 256 to 2048, random order.

cost model (placeholders July 2026 B200 as B1 and B4):
- Init: top-k of the first chunk C, B1 at IPT C over 256 (about 1700 at C 1024, 3900 at C 4096 for k 2048).
- Sweep per chunk: 1 compare per item, 1 ballot per slot, popc prefix, per-warp atomic base, predicated STS of (key, index), about 10 cycles per item slot, no barrier until a merge.
- Merge: B1 on retained k plus buffer B (B equal to k), 2 passes at IPT 2k over 256, about 2 (450 + 85 times 2k over 256), 1200 for k 256, 3900 for k 2048.
- N 32768 k 256 resident, C 1024: 1700 + 1240 + 890 candidates over 256 per merge gives 4 merges times 1200 = 7700, against B4 4800, B3 15.5k, B1 24.5k. N 32768 k 2048, C 4096: 3900 + 1120 + 4260 candidates gives 2 merges times 3900 = 12.8k against B4 7900.
- Sorted ascending chunk order: every item is a candidate, N over B merges, catastrophic. A fixed pseudo-random chunk order defuses sorted input, not adversarial input. Budget guard: above 8 k candidates switch to B1 on the remainder.

focus cells: 1. block N 32768 k 256 fp32 fed as 32 chunks of 1024, 2. block N 32768 k 2048 fp32 as 8 chunks of 4096 (DSA V3.2 chunk), 3. block N 16384 k 1024 fp32 as 4 chunks (DSA V4 Pro), 4. block N 32768 k 64 bf16 as 32 chunks.
boundary cells: block N 8192 k 16 (B2 and B4 expected), block N 32768 k 4096 (merge of 8192 items each time, B1 expected).

explore first: (1) one cell against B4 on resident input, N 32768 k 256 fp32, same session. (2) the chunk sweep cost per item slot. (3) merge primitive per k (N2 for k at most 64, B1 above). (4) candidate count on random, sorted and bf16 logit input (host simulation first).

determinism and ordering: exact. The merge primitive decides tie-break and order, candidates carry (key, index).

main risk: on resident input B4 computes the bound in one shot and sweeps once, so N3 pays the same sweep plus ln(N over C) merges. N3 only earns its place if the streaming contract matters to the device agent, which the lead decides.

cheapest kill experiment: explore item 1. If N3 is more than 1.2x slower than B4 on resident input, retire it to the streaming note in `ideas/WORKLOADS.md` and the knowledge base, no further GPU time.

references: `ideas/queue/B4_exact_bound_prefilter.md`, PR #9281 `buffered` algorithm, `reference/GVR_V2_ALGORITHM_REPORT.md` section 2.4 (one streaming pass with a candidate buffer), `ideas/WORKLOADS.md` chunk merge section
