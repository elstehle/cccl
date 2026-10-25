# N10: fp32 keys-only network with index recovery by ballot rank (the prefer_smaller_index route for the W1 family)
status: queued (wave 2, small, same optimizer as N5)
scope: warp, subwarp
origin: new, from the wave 1 W1 cards: the fp32 route that wins not_guaranteed is the min max pairs network (W1_minmax_pairs, 35 cycles per stage at IPT 1), the route that gives prefer_smaller_index is the 64-bit index-in-key word, which costs 13 percent at N 32, 24 to 30 percent at N 64, 16 percent at N 128 k at most 16 and 82 percent at N 128 k 64 in latency and 9 to 10 percent in throughput against the pairs (W1_eager_same card), and loses 0.7x to 0.9x to the pairs at sub-warp scope (W1_eager_same_sub, W1_packed_sub). The branch-free struct comparator costs 1.03x to 1.24x fp32 and 1.35x to 1.50x bf16 (W1_kvcmp, T35). For bf16 the one-word packing settles it at zero cost, fp32 has no 32-bit word for key plus index.
hypothesis: run the fp32 keys-only min max network (no index moves at all, the cheapest fp32 network measured: N 32 531, N 64 720, N 128 k 16 1649 cycles), then recover the indices of the k selected keys exactly: the k-th key t sits at a known position after the network (one shuffle), every lane compares its IPT original keys against t (IPT compares, the original keys are still in registers or reloaded), one ballot per slot in slot-major order (27 + 6 IPT pipelined), popc prefixes give each item above t its rank in index order and the items equal to t take the first k minus count_above ranks in index order. That is exact prefer_smaller_index for about 110 + 8 IPT cycles, with the network itself one register per element and one shuffle per stage less than the pairs. Output indices by the prefix, sorted order lost unless the ranks are written into the network's positions (one more shuffle per selected item).
cells it should win: warp N 64 to 128 and sub-warp LW 4 to 16 N 16 to 128, fp32, prefer_smaller_index, both regimes, wherever the W1 family is the chosen primitive (the sorted-output pass, k 32 at N 64, throughput at k 64)

what it must beat (B200, 2026-10-25, cycles): the u64 word (W1_eager_same fp32) N 32 601, N 64 934, N 128 k 16 1915, k 64 2216, N 256 k 32 3826, throughput 163 to 184 Gelem per s. The struct comparator (W1_kvcmp) N 32 721, N 64 1081, N 128 k 32 2120. Sub-warp: the u64 word LW 8 N 128 k 8 5083 against min max pairs 4189 (W1_minmax_pairs_sub) and the CUB pairs 4867. To matter N10 has to be at least 10 percent under the u64 word at N 64 and 128 and under the pairs plus 10 percent, else the u64 word stays the route.

cost model (measured: min max keys network stage 35 cycles at IPT 1, dependent ballot 27 and 6 per further ballot (T17), popc 23, shuffle 30):
- Network: the keys-only min max numbers (531, 720, 1649 at N 32, 64, 128 k 16, 4301 at N 256), about 5 to 10 percent under the pairs since the index select per stage goes.
- Recovery: 30 (broadcast t) + 2 IPT compares + 21 + 6 IPT ballots + popc prefixes 30 + emission 30, about 110 + 8 IPT: 120 at N 32, 125 at N 64, 140 at N 128, 175 at N 256.
- Totals: N 32 about 650 (u64 601, loses 8 percent), N 64 about 840 (u64 934, 10 percent better), N 128 k 16 about 1790 (u64 1915, 6 percent), k 64 about 1360 (u64 2216, 1.6x, the two-register tile cell), N 256 about 4480 (u64 3826, loses, the u64 word's shuffle advantage wins at IPT 8). Sub-warp LW 8 N 128 k 8 about 4189 + 240 = 4430 (u64 5083, 13 percent better).
- Throughput: the pairs route is 9 to 10 percent above the u64 word, the recovery costs one ballot pass, so about plus 8 percent over the u64 word.

focus cells: 1. warp N 64 k 32 fp32 prefer_smaller_index, 2. warp N 128 k 16 fp32, 3. warp N 128 k 64 fp32, 4. sub-warp LW 8 N 128 k 8 fp32 (gpt-oss and Qwen3 routing shape), 5. sub-warp LW 4 N 32 k 8 fp32.
boundary cells: warp N 32 (u64 word 601 expected to win), warp N 256 k 32 (u64 word 3826 expected to win).

explore first: (1) the recovery pass alone as a slope over IPT at IPT 1, 4, 8 (one kernel), then the full route at N 64 k 32 and N 128 k 16 against W1_eager_same under prefer_smaller_index in one session. (2) the tie handling when more than k minus count_above items equal t (bf16 does not use this route, fp32 random has 1.0 ties at rank k, the oracle patterns force it). (3) sub-warp LW 8 and LW 4 with full-warp ballots serving every logical warp at once (T37's form).

determinism and ordering: exact prefer_smaller_index by construction (rank in index order). Output unsorted by default, sorted at one shuffle per selected item.

main risk: a 6 to 13 percent gain is inside T28's schedule swing for these networks, so the verdict needs several cells and the SASS. The route matters only where the W1 family is the chosen primitive, which after wave 1 is the sorted-output pass and a few small cells.

cheapest kill experiment: explore item 1. If the full route is not at least 10 percent under the u64 word at N 64 k 32 and N 128 k 16 fp32 in one session, the u64 word stays the fp32 prefer_smaller_index route for the W1 family and the card closes.

references: `TECHNIQUES.md` T17, T28, T35, T37, cards W1_eager_same, W1_eager_same_sub, W1_minmax_pairs, W1_minmax_pairs_sub, W1_kvcmp, W1_packed_sub, `ideas/queue/W1_bitonic_topk.md`, `ideas/queue/N5_two_row_bf16x2.md`
