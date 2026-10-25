# B2: Atomics based iterative extraction (block)
status: queued (wave 1)
scope: block
origin: `reference/BLOCK_TOPK_RESULTS.md` (`atomic_adaptive`, `atomic_iter_h`), `reference/proto_topk.cu`, #9747 strategy 4
hypothesis: k rounds of a fire-and-forget shared `atomicMax` of the thread's current candidate into a per-round slot, a barrier, a branchless shift-down. The MIO combines same-address bursts, a block round costs about 74 to 81 cycles on B200. For bf16 pack key and inverted index into the atomic word so the slots are the sorted answer, output is indices by one mask and prefer_smaller_index is inherent. Rounds are N-independent, so B2 holds small k up to the N where the per-thread list update dominates the round.
cells it should win: block N 1024 to 8192, k 4 to 32, bf16 first, latency regime, prefer_smaller_index at no extra cost

cost model (measured 2026-10-25 by the hardware researcher, `TECHNIQUES.md` T9 to T11: the 256-thread fire-and-forget atomicMax round with barrier and readback is 74.4 cycles on B200 and 74.5 on GB200 with zero completion mismatches, the uncontended shared atomicMax with return is 55 (July placeholder 115), with-return atomics serialize at one lane per cycle, 64-bit shared max is a CAS loop at 4540 per round. July placeholders from `reference/BLOCK_TOPK_RESULTS.md`: marginal round with shift-down about 90 at IPT 4, prologue 150, fp32 epilogue about 890 at N 1024):
- IPT = N over 256. Round(IPT) about 80 + 2 IPT (predicated shift of the per-thread list, or a head pointer into a per-thread sorted run at about 5 cycles when IPT is large). Prologue: twiddle and pack 2 IPT plus a per-thread sort of IPT items (network up to IPT 8, about 60 cycles, partial sort or a shift-list above).
- Packed bf16: cost about 150 + k (80 + 2 IPT) + 50 (read k slots). N 1024 k 16 about 1640 (measured `atomic_iter_h` 1511 at fp16, 19 registers). N 8192 k 16 about 2600 against B1 6900. N 8192 k 64 about 10k against B1 6900, so the crossover at N 8192 is near k 40.
- fp32: value rounds plus index resolution (count items at or above slot k minus 1, ticket scatter, flood path on ties), measured 890 at N 1024 and growing with IPT. N4 (truncated key 20 bits plus 12-bit index for N at most 4096) removes it and makes fp32 cost the bf16 cost with a rare exact fallback.
- Crossover with B1 at N 1024 bf16: about k 18 on the model. Two slots per round need two barriers, so no gain expected (confirm cheaply).

focus cells: 1. block N 1024 k 16 bf16, 2. block N 4096 k 16 fp32 (Quest with a 256-token budget at 64K context is N 4096 k 16), 3. block N 8192 k 32 bf16, 4. block N 2048 k 8 fp32 (beam search and draft candidates per chunk), 5. block N 1024 k 64 bf16.
boundary cells: block N 1024 k 128 (B1 expected to win 2x), block N 32768 k 16 (IPT 128, list update, B4 expected to win).

explore first: (1) packed bf16 pairs, index-only output, k 4 to 64, slope over k at IPT 4, 16, 32. (2) fp32: N4 truncated packing with the exact fallback, against the value-rounds-plus-epilogue path for prefer_smaller_index. (3) per-thread structure for IPT at least 16 (sorted run with head pointer against shift list). (4) where rounds stop paying against B1, B4 and the revived B5 (packed W2 per warp at about 30 cycles per round, then one more W2 over 8 k) per N. (5) ticket atomics in the epilogue at the measured 55 cycles with return, and whether a warp-aggregated ticket (one atomic per warp per slot) beats per-item tickets since with-return atomics serialize at one lane per cycle.

determinism and ordering: packed form inherent and sorted. fp32 value-only form needs the index pass for ties, N4 restores inherent tie-break for N at most 4096.

main risk: the round is barrier-bound (28 of about 74) and the shift-down grows with IPT, so B2 has no answer above N 8192 and none above k 32 to 64. The GB200 risk is retired (74.5 cycles, T9). The new risk is B5: if the packed W2 warp round is really about 30 cycles, two warp phases of k rounds (about 60 k plus 100) undercut one block phase of k rounds (74 k) at every k, and B2 keeps only the cells where the warp phase's per-lane list at IPT 16 and above costs more than the block barrier. Key 0 collides with the empty-slot sentinel (handled in the u32 edition, keep the fix).

cheapest kill experiment: round slope over k at N 1024 and N 8192 for packed bf16 in one kernel against the B1 reference. If round(IPT 32) exceeds 150 cycles, B2 keeps N at most 4096 only.

references: `reference/BLOCK_TOPK_RESULTS.md` sections 1, 3, 4, 5, `reference/BLOCK_TOPK_AIR_OPT_RESULTS.md` section 5 (u32 edition, zero-key fix), `ideas/queue/N4_truncated_key_packing.md`
