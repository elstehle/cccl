# N5: Two rows per lane with bf16x2 SIMD compare-exchange (batched throughput for networks at k 32 to 64)
status: queued (wave 2, throughput regime only, same optimizer as the W1 fp32 route card N10)
scope: warp, subwarp
origin: new here, motivated by issue #12160 and the batched MoE router workload. Wave 1 fixed the comparison point: the one-word bf16 network (W1_eager_same) runs at 263 to 275 Gelem/s from N 32 to 256 for k up to 32 and 224 to 237 at k 64, and the packed W2 at 439 (N 128 k 16), 332 (N 256 k 32), 178 (N 256 k 64) and 135 (N 128 k 64). So networks already win throughput at k 64 and lose it at k at most 32.
hypothesis: put two independent rows in the high and low halves of every 32-bit key register and run the same data-oblivious network on both at once with `max.bf16x2` and `min.bf16x2` (one HMNMX2 each, 6 cycles, T7). Every stage compares element i with element j of the same row, so the halves never interact. Keys only: 2 SIMD instructions and 1 shuffle per stage for two rows against the one-word network's ISETP and SEL plus 1 shuffle per row, so half the shuffles and a third of the ALU per row. With indices: a per-half mask by one HSET2 (`__hgt2_mask`, 6 cycles, T18) and two LOP3 selects move two 16-bit indices in one value register, 5 ALU and 2 shuffles for two rows. Up to 2x per row where the network is shuffle or issue bound, which W1_eager_same's flat 263 to 275 Gelem/s curve says it is.
cells it should win: throughput regime, bf16 keys, warp and sub-warp N 64 to 256 per row with k 32 to 64 (Nemotron 3 Ultra routing at k 22 to 32 in prefill, EAGLE-2 children at k 16 is the boundary), two rows per lane

what it must beat (B200, 2026-10-25, throughput Gelem per s): W1_eager_same bf16 N 64 k 32 275, N 128 k 32 268, k 64 237, N 256 k 32 263, k 64 224, sub-warp W1_eager_same_sub LW 8 N 128 k 64 282, LW 4 N 16 k 4 735. W2_packed_v3 bf16 N 256 k 32 332, k 64 178, N 512 k 64 159, W2_packed v1 N 128 k 16 439, k 64 135, W2 sw_v2 LW 8 N 128 k 16 485, k 64 147. To matter N5 has to reach 1.4x the one-word network (at least 370 at N 128 k 32, 330 at k 64) and pass W2 at k 32 (332 at N 256).

cost model (measured: HMNMX2 and HSET2 6 cycles (T7, T18), integer per-half compare masks emulated at 20 to 24 so never used, shuffle 4.9 per further shuffle in flight (T17), one-word packed exchange at IPT 1 measured 38 cycles per stage (W1_eager_same)):
- Per stage for two rows: keys only HMNMX2 max plus min and 1 SHFL against 2 (ISETP, SEL, SHFL) for two one-word rows: shuffles halve, ALU drops from 4 to 2. With indices: HSET2, 2 LOP3, 2 HMNMX2 and 2 SHFL for two rows against 2 (ISETP, SEL, SHFL) with the index inside the word, so the index path costs the shuffle saving back and N5 pays off only keys-only or with a 16-bit index rebuilt by the ballot rank pass of N10.
- Latency regime: the stage chain is unchanged, no gain, possibly a loss from the pack and unpack (one PRMT per pair of items at load and emission).
- Throughput: the one-word network is issue bound at 263 to 275 Gelem/s independent of N, so halving shuffles and ALU per row predicts 1.6x to 1.9x per row for keys only, less with indices. Registers halve for the keys, which helps the 8 blocks per SM the network already has.
- Rows must share N and k, which the batched router has by construction (prefill rows of one layer).

focus cells (throughput): 1. warp N 128 k 32 bf16, 2. warp N 256 k 32 bf16 (against W2 332), 3. warp N 128 k 64 bf16, 4. sub-warp LW 8 N 128 k 32 bf16 two rows per lane.
boundary cells: warp N 128 k 32 bf16 latency (expect no gain), warp N 256 k 16 bf16 throughput (W2 559 expected to win).

explore first: (1) the one-word bf16 eager network (W1_eager_same source) with two rows per word, keys only, at N 128 k 32, throughput against W1_eager_same and W2_packed_v3 in one session (the #12160 result on Blackwell). (2) the index path: HSET2 mask and two LOP3 against the N10 ballot rank recovery after a keys-only network. (3) sub-warp LW 8 two rows per lane. (4) registers and occupancy of both kernels (the one-word network is at 8 blocks per SM already).

determinism and ordering: inherent per row (deterministic, sorted). prefer_smaller_index needs the index to move with the key or the N10 recovery pass, the index-in-key trick has no room with two rows per word.

main risk: the gain exists only where networks are the right algorithm, which after wave 1 is k 32 to 64 at N at most 256 in the throughput regime, a narrow band, and the index path eats the shuffle saving. Latency regime gains nothing.

cheapest kill experiment: explore item 1 (keys only, two kernels, one lock section). If two-row bf16x2 is not at least 1.4x the per-row throughput of the one-word network at N 128 k 32, drop the card and record the number.

references: issue #12160, `TECHNIQUES.md` T7, T17, T18, cards W1_eager_same, W1_eager_same_sub, W2_packed_v3, W2_packed_sw_v2, `ideas/queue/N10_fp32_keys_only_index_recovery.md`
