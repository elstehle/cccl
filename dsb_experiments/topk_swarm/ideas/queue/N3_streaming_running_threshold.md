# N3: Streaming block selection with a running exact threshold (chunked input, DSA style)
status: queued (wave 2b, one cell after B4's final build lands, streaming contract plus one candidate cell at N 32768 k 2048)
scope: block
origin: new at block scope, the block analog of PR #9281's `buffered` warp algorithm. Seeds from the lead: DSA style chunked selection with merge, threshold reuse.
hypothesis: when a row arrives in chunks (a DSA query over 128K scores, a vocabulary over 32 chunks), the exact k-th value of everything seen so far is a lower bound for the rest of the same row. Initialize the retained top-k from the first chunk, sweep every further chunk with one compare per item, compact the candidates into a buffer with per-item tickets, and merge (top-k of retained plus buffer, a radix select over 2k items) only when the buffer fills. On random order about k (1 + ln(N over C)) items ever become candidates. The row is read once, which is the "insert chunk, keep k" interface a device agent over many chunks per query could use. On resident input it is B4 with the bound learned progressively, and B4 is cheaper wherever its one-shot bound is tight. B4's bound is loose by construction at k above the thread count (the thread-max bound certifies k at most 256, above that B4 keeps the top m per thread and its passes over 27 to 51 live slots cost it: B4_v3 at N 32768 k 2048 is 27640 and 30776 cycles against B3_tuned's 19253 and 20804), and N3's merges are k-sized selections that do not care how large k is relative to the thread count. That makes N 32768 k 2048 (the DSA V3.2 chunk cell) N3's one candidate cell on resident input.
cells it should win: N 32768 k 2048 on resident input if the model holds, and the streaming cells (N 32768 and above per row fed as chunks of 1024 to 4096, k 256 to 2048, random order)

what it must beat (B200, 2026-10-25): at N 32768 k 2048 B3_tuned 19253 (bf16) and 20804 (fp32), B4_v3 27640 and 30776 (B4_v2 22721 and 24570, B4's final build pending), B1_idx at 512 threads 17991 and 19695 at k 1024 (k 2048 not measured). At N 32768 k 256 B4_v3 8109 and 6741 (chained 2.5x to 3x over B3_tuned and B1_idx), B2_twolevel 4774 at k 16. To matter N3 has to be at least 1.3x under B3_tuned at N 32768 k 2048 (at most 14.8k bf16, 16k fp32) and within 1.2x of B4 at k 256 (at most 9.7k bf16, 8.1k fp32).

cost model (measured constants: B1 or N6 pass over C items about 85 C over 256 + 410 (B1 cards), B1_idx at N 4096 k 1024 4705 and 5702 for the init selection, per-item tickets about one cycle per selected item (T21, T22), B3's packed scan emission 300 (T29), B4_v3's sweeps about 10 cycles per item slot at IPT 128 (B4 card)):
- Init: top-k of the first chunk C by B1_idx or N6, about 1770 at C 1024 k 256, about 5500 at C 4096 k 2048.
- Sweep per chunk: 1 compare per item, count, predicated store of (key, index) into the candidate buffer at a per-item ticket, about 4 to 5 instructions per item slot, about 10 cycles per slot at two warps per scheduler (B4 measured that rate), so about 1300 cycles for the 28 non-initial chunks of 1024 at N 32768 or 1100 for 7 chunks of 4096.
- Merge: a radix select over retained k plus buffer B (B equal to k): 2 passes at IPT 2k over 256, about 1160 at k 256 and 3550 at k 2048, plus the scan emission.
- N 32768 k 256, C 1024: 1770 + 1300 + ln 32 times 256 over 256 gives 4 merges times 1160 = 4640, about 8000 against B4_v3 6741 to 8109 and B3_tuned 19167 to 20071: a tie with B4 at best, B4 owns the cell.
- N 32768 k 2048, C 4096: 5500 + 1100 + about 2 merges of 3550 + 300, about 14k against B3_tuned 19253 and 20804 and B4_v3 27640 and 30776: 1.4x to 1.5x under B3_tuned on the model, 2x under B4_v3.
- Sorted ascending chunk order: every item is a candidate, N over B merges, catastrophic. A fixed pseudo-random chunk order defuses sorted input, not adversarial input. Budget guard: above 8 k candidates switch to B3_tuned on the remainder.

focus cells: 1. block N 32768 k 2048 fp32 fed as 8 chunks of 4096 (DSA V3.2 chunk, against B3_tuned and B4 in one session), 2. block N 32768 k 2048 bf16, 3. block N 16384 k 1024 fp32 as 4 chunks (DSA V4 Pro, B4_v3 9089 and 9800, B3_tuned 11328 and 11009).
boundary cells: block N 32768 k 256 fp32 (B4 6741 expected to win), block N 8192 k 16 (B2 expected).

explore first: (1) one session: N3 against B3_tuned and B4's final build at N 32768 k 2048 fp32 and bf16 on resident input, chunk 4096. (2) the chunk sweep with per-item tickets against a packed scan per chunk. (3) candidate count on random, sorted and bf16 logit input (host simulation first). (4) the merge primitive: B1_idx passes against N6 passes against B3_tuned's emission over the 2k words.

determinism and ordering: exact. The merge primitive decides tie-break and order, candidates carry (key, index), the B3 scan emission gives index order for free.

main risk: B4's final build may fix its k 2048 regression (group-guarded loads, v4 is measured) and B3_tuned at 512 threads (T45) cuts its own numbers by 1.12x to 1.15x, so the 1.4x margin of the model can vanish before N3 is built. Below k 1024 B4 owns every cell and N3 is a contract note.

cheapest kill experiment: explore item 1. If N3 is not at least 1.3x under B3_tuned at N 32768 k 2048 fp32 (at most 16k) in the same session, retire it to the streaming note in `ideas/WORKLOADS.md` and the knowledge base, no further GPU time.

references: `ideas/queue/B4_exact_bound_prefilter.md`, PR #9281 `buffered` algorithm, `reference/GVR_V2_ALGORITHM_REPORT.md` section 2.4, `TECHNIQUES.md` T21, T22, T29, T45, cards B3_tuned_b200_20261025_2, B4_v3_b200_20261025_1, B2_twolevel, N8_b1idx, `ideas/WORKLOADS.md` section 7
