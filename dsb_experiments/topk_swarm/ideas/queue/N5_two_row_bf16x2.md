# N5: Two rows per lane with bf16x2 SIMD compare-exchange (batched throughput for networks)
status: queued (wave 2, throughput regime, same optimizer as W1 and N2)
scope: warp, subwarp
origin: new here, motivated by issue #12160 (half2 min and max network 1.8x over scalar half at equal array length) and the batched MoE router workload (thousands of rows per launch with the same N and k).
hypothesis: put two independent rows in the high and low halves of every 32-bit key register and run the same data-oblivious network (W1 or N2) on both at once with `max.bf16x2` and `min.bf16x2`. Every stage of the network compares element i with element j of the same row, so the two halves never interact and no cross-half stage exists (unlike packing two keys of one row). Keys only: 2 SIMD instructions and 1 shuffle per stage for two rows against 3 instructions and 1 shuffle per row. With indices: a packed compare mask (`set` on bf16x2 or an integer compare on the twiddled halves) plus two LOP3 selects move two 16-bit indices in one value register, 5 instructions and 2 shuffles for two rows against 10 and 4. Up to 2x on the network cost in the throughput regime, where the batched router has rows to spare.
cells it should win: throughput regime, bf16 keys, sub-warp and warp N 64 to 256 per row with k 16 to 64 (the k range where networks beat W2), 2 rows per lane, index-only output with 16-bit indices (N at most 65536)

cost model (measured 2026-10-25, `TECHNIQUES.md` T7, T8, T18: `max.bf16x2` and `min.bf16x2` are one HMNMX2 at 6 cycles, `__hgt2_mask` and its siblings give per-half 0xffff masks in one HSET2 at 6 cycles, integer u16x2 and s16x2 min and max are native at 6 cycles but the integer per-half compare masks are emulated at 20 to 24 cycles and u8x4 max at 28, so the index path stays on native bf16 compares. July placeholders: stage s(IPT) as in W1 and N2, shuffle 30 of the 44 per stage at IPT 1):
- Per stage for two rows: keys only, max plus min (2 ALU at 6 cycles each, 1 shuffle) against 2 (compare plus 2 selects, 1 shuffle) for two scalar rows: ALU 2 against 6, shuffles 1 against 2. With indices: HSET2 mask plus 2 LOP3 selects plus max and min, 5 ALU and 2 shuffles against 10 and 4.
- Network cost per row pair: N2 at k 32 N 128 (IPT 4) keys only about 1525 for one row, about 1525 times 0.55 per row for two rows if shuffles dominate (shuffle count halves, ALU thirds). Pairs about 0.5 per row.
- Latency regime: no gain for a single row (same stage count), a loss if the packed compare needs more instructions than the scalar one. Throughput regime: the gain is the per-row instruction count, modulated by registers (packing halves the key registers for two rows, which helps occupancy).
- Packing and unpacking: `prmt` per pair of items at load and at emission, about 1 instruction per item.

focus cells: 1. sub-warp LW 8, 2 rows per lane, N 128 k 16 bf16 throughput (gpt-oss-120b and Qwen3 routing shape at k 16), 2. warp N 128 k 32 bf16 throughput, 3. warp N 256 k 32 bf16 throughput, 4. warp N 64 k 32 bf16 throughput.
boundary cells: warp N 128 k 32 bf16 latency (expect no gain), warp N 256 k 8 bf16 throughput (W2 expected to win regardless).

explore first: (1) `WarpBitonicSort` bf16x2 two rows against scalar bf16 at N 64, keys only, throughput, same session (the #12160 result on Blackwell). (2) the index path: packed compare mask and two LOP3 against scalar selects, instruction count from SASS first. (3) N2 two-row variant at k 32. (4) native bf16x2 only for the mask path (integer per-half compare masks are emulated, T18), twiddled u16x2 min and max only for a keys-only variant where no mask is needed, with the signed zero semantics checked by the oracle.

determinism and ordering: inherent per row (deterministic, sorted). prefer_smaller_index needs the index to move with the key (the 5-instruction path), the index-in-key trick has no room with two rows per word.

main risk: the gain exists only where networks are the right algorithm, and the model says that is k 32 to 64 at N at most 256, a narrow band. Packed compare masks for the index path may cost more than two scalar selects. Latency regime gains nothing.

cheapest kill experiment: explore item 1 (keys only, two kernels, one lock section). If two-row bf16x2 is not at least 1.4x the per-row throughput of scalar bf16 at N 64 k 32, drop the card and record the number.

references: issue #12160 (SASS counts and the half2 benchmark), `ideas/queue/W1_bitonic_topk.md`, `ideas/queue/N2_level_parallel_bitonic.md`
