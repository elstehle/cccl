# B3: Compacting radix select (block)
status: queued (wave 1)
scope: block
origin: new, cousin of cluster idea A3 (knowledge base 3.4)
hypothesis: after pass 1 the items in buckets above the pivot bucket are final (write their indices now), the pivot bucket holds about N over 256 candidates on uniform digits regardless of k, and only those need passes 2 and later. Compact them (key and index) into shared memory via ballot prefix and per-warp atomic bases, so later passes sweep about N over 256 items instead of N. Removes the dominant term of B1 above N 4096 and makes the pass count almost free for fp32. Index-only output falls out: selected-above indices are emitted during compaction, the pivot survivors at the end.
cells it should win: block N 4096 to 32768, k 4 to 4096 (k-independent after pass 1), fp32 more than bf16, random and sorted input (compaction does not care about order, only the bucket size matters)

cost model (placeholders July 2026 B200 as B1: histogram 85 IPT, scan plus state 410 per pass, barrier 28, ballot 23, atomics across banks 32 per 256):
- Pass 1 over N as B1: 14 IPT + 85 IPT + 410.
- Compaction sweep: per slot one compare of the digit against the pivot bucket (above, equal, below), one ballot for equal and one for above, popc prefix, per-warp atomicAdd for the two bases, predicated STS of (key, index) for equal and of index for above. About 15 IPT + 100 (two atomics per warp per slot pipelined, one barrier).
- Passes 2 and later over C about N over 256 (plus the tie class at the pivot digit): one or two items per thread of the first warps, H about 85 times C over 256, so pass about 420 + 85 C over 256. For fp32 two or three such passes.
- Total bf16 N 32768: about 1800 + 11300 + 2000 + 450 = 15.5k against B1 24.5k. fp32 N 32768: about 16k against B1 35.7k (3 passes). N 1024 bf16: about 750 + 160 + 430 + 150 = 1500 against B1 1700, so a modest win even at the small end.
- Writing the selected-above indices needs their count from the scan (available) and k remaining stays exact. Positions by ticket within class, deterministic positions need the warp-redundant rank (knowledge base 4.5).

focus cells: 1. block N 32768 k 256 fp32 (DSA chunk at smaller k, Quest at 128K), 2. block N 8192 k 64 bf16, 3. block N 32768 k 2048 bf16 (DSA V3.2 chunk), 4. block N 4096 k 32 fp32, 5. block N 16384 k 1024 fp32 (SnapKV, DSA V4 Pro chunk).
boundary cells: block N 1024 k 16 bf16 (B2 expected to win), block N 2048 k 1024 fp32 (pivot bucket of 8 items but 1024 above-pivot writes, B1 expected to tie).

explore first: (1) compaction sweep cost at IPT 4, 32, 128 against the saved histogram sweep of pass 2 (same session, one cell each). (2) register-resident compaction (candidates fit one per thread) against smem compaction. (3) wider first digit (10 or 11 bits) only if the pass-1 scan cost stays below the saved compaction volume, else keep 8. (4) index-only emission of the above-pivot class during compaction. (5) tie class at the pivot digit larger than the smem buffer (bf16 logit data): fall back to B1 sweeps for that pass, measure the branch cost.

determinism and ordering: as B1. Ticket positions are arrival order, the warp-redundant rank gives index order.

main risk: the compaction sweep is itself a pass over N with two ballots per slot and predicated stores, so at N 1024 it may cost what it saves. On bf16 logit-like data the pivot tie class can be large (128 distinct values per binade), so the compacted set is not N over 256 but the tie class, and pass 2 is not small. Shared memory grows by the candidate buffer (8 bytes per candidate).

cheapest kill experiment: B3 against B1 at N 8192 k 64 bf16 and fp32, random input, same session. If B3 is not at least 1.3x faster at N 8192, the idea is confined to N 32768 and fp32 and merges into B1 as an option.

references: knowledge base 3.4 idea 3, 4.5, `reference/BLOCK_TOPK_AIR_OPT_RESULTS.md` sections 2 and 6
