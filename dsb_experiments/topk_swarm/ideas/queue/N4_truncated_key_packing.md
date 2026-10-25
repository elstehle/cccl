# N4: Truncated-key packing for fp32 extraction (key prefix and index in one 32-bit word, exact tie finish)
status: queued (wave 1 candidate, same optimizer as B2 and W2)
scope: warp, subwarp, block
origin: new. Generalizes the bf16 packed word of `atomic_iter_h` (`reference/BLOCK_TOPK_RESULTS.md`) to fp32.
hypothesis: the packed 32-bit word (16-bit key, 16-bit inverted index) is what makes bf16 extraction free of tie machinery and gives prefer_smaller_index inherently. For fp32 pack the top 32 minus ceil(log2 N) bits of the twiddled key with the index: 24 key bits for N at most 256, 20 for N at most 4096, 16 for N at most 65536. Run the W2 or B2 rounds on the packed word. The result is the exact top-k by (truncated key, index). It equals the top-k by full key except inside the tie class of the k-th truncated key, which is tiny on random data (expected other members about N over 2^(key bits minus 9) per binade: 0.5 at N 1024 with 20 bits). One cheap tie pass fixes it exactly: ballot or count the items whose truncated key equals the k-th, compare their full keys, keep the right ones. The pass is branch-free and always runs, so the cost is round-based plus a constant, and the fp32 epilogue of B2 (about 890 cycles at N 1024) and the index pass of W2 disappear.
cells it should win: every W2 and B2 cell with fp32 keys and N at most 4096, both requirements. fp32 becomes as cheap as bf16 for extraction.

cost model (placeholders July 2026 B200: W2 round 82 + 2 IPT, B2 round 80 + 2 IPT, ballot 23, barrier 28):
- Packing: twiddle (3 instructions) plus shift and or with the inverted index, 2 more per item, once.
- Rounds: unchanged from the packed bf16 path, k (82 + 2 IPT) at warp, k (80 + 2 IPT) at block.
- Tie pass, warp: IPT ballots (truncated key equals t_k) pipelined, about 23 + 8 IPT, then a full-key compare among the tie members (expected 1 to 3, worst case a sort32 if more than 32) about 50. Block: 1 compare per item, ballot per slot, per-warp atomicAdd, one barrier, then the same small finish, about 150 + 2 IPT.
- B2 fp32 N 1024 k 16: about 150 + 16 times 88 + 170 = 1730 against `atomic_adaptive` 2331 measured, a 1.35x gain on the model. W2 fp32 N 256 k 8: about 100 + 8 times 98 + 140 = 1020 against value rounds plus index pass about 1100 to 1300.
- When the tie class exceeds the finish capacity (bf16-like clustered fp32 data, many equal prefixes), fall back to full-key rounds on the class members only. Choose the split by N at compile time, and by the measured tie class on data patterns when N is small but values cluster.

focus cells: 1. block N 1024 k 16 fp32 (both requirements), 2. block N 4096 k 32 fp32, 3. warp N 256 k 8 fp32, 4. warp N 512 k 16 fp32, 5. sub-warp LW 8 N 128 k 8 fp32.
boundary cells: block N 32768 k 16 fp32 (16 key bits, tie classes of about 8, finish grows), block N 1024 k 128 fp32 (rounds dominate, B1 expected anyway).

explore first: (1) B2 packed rounds with 20-bit keys at N 1024 k 16 against `atomic_adaptive` in the same session. (2) tie pass cost and tie class size distribution on the harness inputs (host simulation first). (3) the same for W2 at N 256 k 8 and N 512 k 16. (4) 16-bit keys at N 8192 to 32768 and the finish for tie classes of 8 to 64 (one warp sort64).

determinism and ordering: inherent for prefer_smaller_index after the tie pass (the pass orders by full key then index). Sorted output by key is the round order except inside the tie class, which the pass sorts.

main risk: fp32 data whose values cluster within a few binades at N 4096 (softmax probabilities, attention scores after scaling) have tie classes of tens of items, so the finish is a small selection, not a compare. Correctness is unaffected, cost is. The pass count of B1 on such data also grows, so the comparison stays fair.

cheapest kill experiment: host simulation of the tie class size at 20 and 16 key bits on the harness fp32 patterns at N 1024 to 32768, minutes. Then explore item 1. If the tie pass costs more than 300 cycles at block scope or the gain over `atomic_adaptive` is below 15 percent at N 1024 k 16, drop N4 and keep B2's value-rounds epilogue.

references: `reference/BLOCK_TOPK_RESULTS.md` sections 3 and 5 (packed half variant, 16-bit cliff), `ideas/queue/B2_atomics_extraction.md`, `ideas/queue/W2_redux_extraction.md`
