# B2 (atomics based iterative extraction), slice B2b: large items per thread, redux posting, tie check, stale posting

Report of the second B2 optimizer instance (opt-B2b), B200 (umbriel-b200-091, sm_100, nvcc 13.4.92, 2026-10-25). The lead started opt-B2 twice; after the collision this instance kept the B2_packed baseline card and the N4 kill batch (`cards/B2_packed_b200_20261025_1.json`, `cards/B2_packed_n4kill_b200_20261025_1.json`) and took the slice described here, the other instance (opt-B2) keeps B2_baseline, B2_detail (its own copy) and B2_trunc32. Branch `exp/topk-swarm/B2b`, worktree `cccl-topk-swarm-B2b`, holder `opt-B2b`, raw batches under `state/results/opt-B2b/`. Every number below is from a batch whose control ran in the same lock section unless marked as context.

A second collision happened inside this slice: from about 13:13Z two opt-B2b instances were alive on the same worktree and node work dir (see `NOTE_duplicate_instance.md`). This report and the cards B2_spec, B2_twolevel, B2_redux8, B2_tiecheck are from the instance that measured tl1, rx1, tc1 and the B2_spec batch; the other instance's line (B2_spec2, B2_spec3) is referenced in section 5 and not measured here.

Cards of this slice: `B2_spec_b200_20261025_1.json` (72 cells), `B2_twolevel_b200_20261025_1.json` (144 cells), `B2_redux8_b200_20261025_1.json` (96), `B2_tiecheck_b200_20261025_1.json` (36). Catalogue entry contributed: T26.

## 1. What it is

B2 posts each thread's current candidate (a 32-bit word: twiddled key over inverted index) into one shared slot per round with a fire-and-forget `atomicMax`, one block barrier per round, reads the slot back and advances the one thread whose head won. k rounds give the k largest in order; the index half makes prefer_smaller_index inherent. The baseline is `B2_packed` (bf16 packed word, fp32 through idea N4: top 32 - log2 N key bits plus the index, exact tie finish), measured against B1 in the baseline card: wins at k below about 20 to 50 depending on N and type, N up to 8192.

This slice asked four questions about the round and the per-thread list:

1. `B2_twolevel`: can a two-level per-thread list (sorted head of 4, unsorted rest, refill when the head runs empty) replace the full sorted list whose prologue (Batcher network over IPT words) and per-round shift (IPT - 1 selects) grow with IPT, and so carry B2 to N 16384 and 32768?
2. `B2_redux8`: does an explicit warp pre-reduction (one `redux.sync.max` and one atomic per warp, 8 atomics per round instead of 256) help the throughput regime where several blocks share the shared memory atomic unit?
3. `B2_tiecheck`: can the fp32 tie check after the rounds (count the tie class members, atomicAdd, barrier, read back) become one `__syncthreads_or` over the heads?
4. `B2_spec`: can the warp reduction that ptxas inserts before the atomic (CREDUX on every round's dependent chain) be taken off the chain by posting from the lanes that hold the warp's stale maximum and second maximum, with the refresh in the shadow of the barrier?

Verdicts: 1 yes for k <= 16 at IPT >= 32 and for every k at IPT >= 64 and in the throughput regime, no for the latency regime at IPT 16 to 32 with k 16 to 64 (a divergent branch per round), fix in flight. 2 no, strictly worse (double reduction). 3 yes, 75 to 570 cycles per fp32 call at no cost, adopt. 4 no, strictly worse, and it explains why: the aggregated post is the floor.

## 2. Contract

As B2_packed (block scope, 256 threads measured, blocked registers in, k indices to shared memory out, both requirements with one binary, output sorted by key then index, bf16 N <= 65536 by the 16-bit index half, fp32 N <= 65536 by log2 N index bits with the exact tie finish, NaN above +inf or below -inf by twiddle, -0.0 equals +0.0). B2_twolevel extends the measured range to N 32768 (IPT 128). B2_packed should be declared unsupported above IPT 64 for bf16 too (section 3.5). Full contract objects in the findings files next to this report.

## 3. Where the time goes

3.1 The round. B2_packed's round at IPT 4 is 10 SASS instructions (BAR.SYNC, LDS, ISETP, SEL, SEL, CREDUX.MAX, SEL, SEL, IMAD.U32, ATOMS.MAX) and 83 cycles: about 28 for the barrier (T9), 22 for the redux (T1), the rest read back, compare and atomic issue. ptxas rewrites the 32 same-address atomics of a warp into one CREDUX plus one elected ATOMS, so the round never issues a 256-way burst; the redux is on the dependent chain and cannot be removed (section 4.4, T26). The shift of the per-thread list (IPT - 1 SEL) is scheduled before the read back and runs under the barrier wait up to IPT 16; above that it shows (B2_packed slope 112 at IPT 16, 148 at IPT 32, 205 at IPT 64 for k 16 to 64).

3.2 The prologue. B2_packed sorts IPT words with a Batcher network: 272 cycles net at IPT 16, 968 at IPT 32, 1264 at IPT 64 (bf16). The two-level partial sort (groups of 4 sorted, merged in a tree keeping the 4 largest) costs 38, 400, 234 at the same IPT, and about 4380 at IPT 128 where the full sort does not fit in registers at all (B2_packed: a 12800-byte local stack frame, 985k cycles per call).

3.3 The two-level round. 48 instructions at IPT 16 against 37, but with a divergent branch (`if my head ran empty`) that puts a BSSY/BSYNC region into every round and makes the aggregation after it use VOTEU.ANY, CREDUX, UFLO. Latency slope 159 against 112 at IPT 16, 140 against 113 at IPT 32 (k 4 to 16), 189 against 164 at IPT 64. In the throughput regime the same round is cheaper than the full shift because it issues fewer instructions.

3.4 The fp32 tail. B2_packed's count pass after the rounds (IPT compares, atomicAdd, barrier, read back, compare) costs 75 to 108 cycles at IPT 4, 136 to 227 at IPT 16, 262 to 570 at IPT 32 of serial time; B2_tiecheck's `__syncthreads_or` over the heads replaces it.

3.5 Code generation cliffs. Full unrolling of the rounds above IPT 16 is instruction-cache bound: the fully unrolled k 64 kernel at IPT 32 costs 227 cycles per round (two-level) and 148 (packed), the k 256 kernels with partial unrolling 111 and 261 (the packed one is `#pragma unroll 1`, which loses the shift overlap). At IPT 128 B2_packed's 128-word list goes to local memory with 0 bytes of "spill" reported by ptxas (12800-byte stack frame): the resource field to watch above IPT 64 is the stack frame, not the spill count.

## 4. Ideas tried, with verdicts

4.1 Two-level per-thread list (`B2_twolevel`, 160 cells, N 1024 to 32768, k 4 to 256, both types and requirements, control B2_packed). Latency ratio (control over variant) bf16: IPT 4 0.99 to 1.00 (same structure); IPT 16 1.07, 0.80, 0.75, 0.83 for k 4, 16, 64, 256; IPT 32 1.48, 1.05, 0.73, 1.71; IPT 64 1.94, 1.19, 1.04, 2.55; IPT 128 the control is in local memory. fp32: IPT 16 1.07, 0.86, 0.73, 0.86; IPT 32 1.09, 0.99, 0.72, 1.49; IPT 64 1.30, 0.95, 1.04, 2.56. Throughput: wins at every IPT >= 16 and every k (bf16 1.09x to 3.92x, fp32 1.00x to 2.34x). Against B1 (context, other session): bf16 N 16384 k 16 3.0x, N 32768 k 4 and 16 4.4x and 4.1x (4479 and 4774 cycles against 19745 and 19709), fp32 N 16384 k 16 2.3x, N 32768 k 16 2.5x; B1 keeps k 64 at N >= 16384. Verdict: adopt per (IPT, k): full sorted list up to IPT 16 and at IPT 32 for k 16 to 64, two-level above; B2 now reaches N 32768 for k <= 16 where the approach card expected B4 and B1. The latency penalty at IPT 16 to 32 is the refill branch's reconvergence (section 4.5).

4.2 Explicit warp pre-reduction (`B2_redux8`, 96 cells, control B2_twolevel). Latency 0.54x to 0.97x (median 0.70x), throughput 0.57x to 0.96x (median 0.82x), no cell even. The SASS shows two CREDUX per round: the hand-written redux, and the predicated lane 0 atomic aggregated again by ptxas (VOTEU, CREDUX, UFLO, ATOMS) inside a BSSY/BSYNC region. Verdict: retired; this also explains T11's observation that the explicit form was slower.

4.3 Head-only tie check (`B2_tiecheck`, 36 fp32 cells, control B2_packed). Latency saving 75, 108, 220 cycles at IPT 4 (k 4, 16, 64), 227, 154, 136 at IPT 16, 570, 551, 262 at IPT 32 (1.02x to 1.38x), throughput 0.98x to 1.17x, registers unchanged (31 to 32, 64, 128), both requirements within 1 percent. The argument: the list is sorted and every remaining word is at most slot k - 1, so only the head can be in the tie class of the k-th truncated key; a block-wide OR decides the finish. Verdict: adopt into the fp32 path (B2_twolevel already carries it).

4.4 Stale warp maximum posting (`B2_spec`, 72 cells, control B2_packed). Loses in every shape: latency cost 1.08x to 2.46x (median 1.48x), throughput 1.06x to 2.58x. The round grew from 10 to 22 instructions with two CREDUX (the refresh of the warp maximum and second maximum) scheduled before the atomic, and the atomic itself became an un-aggregated 32-lane ATOMS with per-lane addresses (a dummy word for the non-posting lanes was needed, because a predicated atomic from two or three lanes is still aggregated by ptxas). 138 against 83 cycles per round at IPT 4. Registers 47 against 32 at IPT 16, 75 against 48 at IPT 32, a 20-byte spill at fp32 N 8192 k 16. Verdict: retired; the aggregated post is the floor of the block round (T26), the remaining lever for B2 is fewer rounds, not cheaper rounds.

4.5 Warp-uniform refill (`B2_twolevel_any`): the refill test as one `__any_sync` and a uniform branch, so the common round carries one VOTE and no reconvergence point. Built and measured after this report was first written; see section 4b if present, else the batch files `*_any1_*` (control B2_twolevel) and `*_any2_*` (control B2_packed) under `state/results/opt-B2b/`.

## 5. Ideas not yet tried

- Unroll policy per IPT: unroll the rounds by 8 above IPT 16 for every k (section 3.5). Predicted from the measured slopes: bf16 N 8192 k 64 from 13551 to about 7500 cycles (B2_packed 9869).
- Head of 8 at IPT >= 64, and a shared-memory resident rest at IPT 128 to get below 128 registers (two blocks per SM) and remove the fp32 spills (96 to 944 bytes at N 32768).
- Fewer rounds: two slots per round would need two barriers (rejected on the model in the approach card); B5 (packed W2 warp phase at 26 cycles per round, then one block phase over 8 k candidates) is the structural answer and belongs to W2's optimizer.
- The other opt-B2b instance's line: `B2_spec2` (one aggregated atomic per warp whose value is known before the read back) and `B2_spec3` (no shared atomics, one word per warp and a read back of 8 words). Both test whether the refresh redux can overlap the barrier when its consumers are placed after the read back; T26 records the question as open.
- B2_packed's fp32 prologue is 76 cycles cheaper than B2_twolevel's at IPT 4 (same list); port that code shape.

## 6. Determinism and ordering support

Inherent and free in every variant of this slice: the index is the low half of the word, prefer_smaller_index and not_guaranteed run the same binary and measured within 2 percent in every batch (tl1 max 1.9 percent, tc1 0.9 percent, iter1 1.7 percent). Output is sorted by key descending then index ascending for bf16, and for fp32 outside the finished tie class. Set-deterministic by construction.

## 7. Open items

- Measure `B2_twolevel_any` (in flight) and, if the latency penalty disappears, the unroll policy on top of it; then one combined adapter choosing the list per (IPT, k).
- Same-session comparison with B1 and B4 at N 16384 and 32768 (both B1 and the two-level list sit at 254 to 255 registers there).
- Cut B2_packed's bf16 support to IPT <= 64 (stack frame cliff).
- Whether `red.shared.max` from one lane via inline PTX escapes the second aggregation (relevant to the other instance's spec2 and spec3).

## 8. Pointers

- `harness/adapters/B2_packed.cuh`: the baseline of this slice (packed bf16, N4 fp32, both requirements), card `B2_packed_b200_20261025_1.json`.
- `harness/adapters/B2_detail.cuh` (namespace `b2b`): negate-form twiddles, Batcher network, the one-level round, the two-level partial sort and rounds (divergent, warp-uniform and redux-posting forms), the unroll policy.
- `harness/adapters/B2_twolevel.cuh`, `B2_twolevel_any.cuh`, `B2_redux8.cuh`, `B2_tiecheck.cuh`, `B2_spec.cuh`: the variants of this slice. `B2_baseline.cuh`, `B2_trunc32.cuh`: this instance's first adapters, superseded as baselines, kept as material.
- `approaches/B2/findings_*_b200.json`: the findings files behind each card. `n4_tie_class_sim.cpp`, `.txt`, `tie_sim.cpp`: the N4 tie class host simulation.
- Raw batches: `state/results/opt-B2b/B2_spec_vs_B2_packed_focus_iter1_20261025_131354.json`, `B2_twolevel_vs_B2_packed_focus_tl1_20261025_131927.json`, `B2_redux8_vs_B2_twolevel_focus_rx1_20261025_132206.json`, `B2_tiecheck_vs_B2_packed_focus_tc1_20261025_132327.json`. Per-cell logs and SASS dumps on the node under `/work/topk_swarm/work/opt-B2b/{logs,gen}`.
- Catalogue: T26 (aggregation of predicated atomics, barrier shadow).
