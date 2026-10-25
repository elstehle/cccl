# B1_det2: the deterministic AIR path with the class-indexed ticket (opt-B1det, 2026-10-25)

Bounded follow-up to the B1 report (section 10, item 3) and T57. Node umbriel-b200-068 (B200 sm_100, 148 SMs), nvcc 13.4.92, harness 0.8, one session (16:49 to 16:52 UTC) with three lock sections, then a phase profile session (16:59 UTC). Source commit d22434f718 (this branch, `exp/topk-swarm/B1det`), the stamp commit 5152f285f8 changes nothing without `B1_PROFILE`. Latency in cycles per call (chain slope net of the generate-only control), throughput in Gelem/s, random normal rows, 256 threads unless stated. Cards `cards/B1_det2_b200_20261025_1.json` (against B1_det) and `_2.json` (against B1_idx under not_guaranteed), batches under `state/results/opt-B1det/`.

## 1. What changed

`B1_det` drew the emission ticket of the deterministic epilogue with `atomicAdd_block(&selected_offset[0], 1)` for the items above the k-th key. The address is provably uniform and the increment constant, so ptxas warp-aggregates the atomic (T23) and puts two special register reads on every item's store chain. The T23 empty-asm fix is a no-op (T39). `B1_det2` spells the ticket as B1_idx does (T57): one atomic per item under the union of the two classes on `selected_offset[item_class]`, with `item_class = !select_all_candidates && is_candidate`, then a branch on the class for the store (class 0 writes `out[ticket]`, class 1 drops the ticket and writes `out[total_selected + rank]` when taken). Everything else is B1_det: prologue -0.0 normalization, two radix passes, the warp-redundant rank over the exact ties. The header `block_topk_b1_det2.cuh` is a copy of `block_topk_b1.cuh` in its own namespace so that variant and control share a translation unit, `diff` shows the namespace, the ticket hunk and the compiled-out stamps.

SASS of the latency kernel at fp32 N 1024 k 16 (two copies of the primitive per kernel): B1_det 1304 instructions with 8 `S2R SR_LTMASK`, 5 `S2R SR_LANEID`, 8 `VOTEU.ANY`, 8 `FLO.U32`, 24 `POPC`, 16 `SHFL.IDX`, B1_det2 1240 with none of the aggregation (0, 1, 0, 0, 8, 8) and 8 plain `ATOMS.ADD R3, [R3+0x84c], R12` on a register address selected by two predicated IMADs. Registers identical in the latency kernel, 8 fewer in the throughput kernel at IPT 4.

## 2. The same-kernel A/B (card 1, B1_det2 against B1_det, both prefer_smaller_index)

| cell | bf16 det2 / det | fp32 det2 / det |
|---|---|---|
| N 1024 k 16 | 1868 / 2173 (minus 14.0 percent) | 1802 / 2160 (minus 16.6) |
| N 1024 k 256 | 2177 / 2440 (minus 10.8) | 2142 / 2432 (minus 11.9) |
| N 4096 k 64 | 4776 / 5750 (minus 16.9) | 4633 / 5693 (minus 18.6) |
| N 16384 k 256 | 15242 / 19077 (minus 20.1) | 13706 / 17643 (minus 22.3) |
| throughput N 1024 k 16 | 362 / 314 (plus 15.6) | 392 / 344 (plus 13.8) |
| throughput N 16384 k 256 | 276 / 226 (plus 22.0) | 302 / 245 (plus 23.4) |

The fp32 rows have no tie at rank k, so the saving there is the ticket alone: 358 cycles at N 1024 k 16 (the B1_det card attributed 344 to the aggregation by comparing two different kernels), about 90 per item slot at IPT 4, 49 to 80 at IPT 16, 37 to 62 at IPT 64. The attribution of T23 and T57 holds inside the AIR kernel.

## 3. The price of prefer_smaller_index (card 2, B1_det2 against B1_idx under not_guaranteed)

Percent is variant over control for latency and control over variant for throughput, "before" is the verified card B1_det_b200_20261025_1 against the same control on the same node.

| key | N | k | latency after | before | throughput after | before |
|---|---|---|---|---|---|---|
| fp32 | 1024 | 16 / 64 / 256 | minus 0.2 / plus 0.0 / minus 0.4 | 19.2 / 19.3 / 12.9 | 5.2 / 4.3 / 3.2 | 19.7 / 19.4 / 8.4 |
| fp32 | 4096 | 16 / 64 / 256 | plus 1.0 / minus 0.1 / minus 0.7 | 22.8 / 23.9 / 25.6 | 11.1 / 9.3 / 6.5 | 29.7 / 28.1 / 26.4 |
| fp32 | 16384 | 16 / 64 / 256 | 9.6 / 6.3 / 3.5 | not measured | 82 / 77 / 74 | not measured |
| fp32 | 32768, 512 threads | 16 / 64 / 256 | 9.2 / 6.9 / 5.0 | unsupported | 9.1 / 7.8 / 4.6 | unsupported |
| bf16 | 1024 | 16 / 64 / 256 | 5.2 / 7.8 / 10.3 | 23.2 / 25.6 / 22.3 | 12.6 / 13.6 / 14.7 | 30.2 / 32.0 / 26.5 |
| bf16 | 4096 | 16 / 64 / 256 | 12.2 / 24.3 / 22.5 | 34.3 / 51.1 / 48.2 | 22.0 / 39.1 / 34.9 | 41.6 / 61.6 / 57.6 |
| bf16 | 16384 | 16 / 64 / 256 | 12.1 / 25.3 / 36.7 | not measured | 71 / 86 / 100 | not measured |
| bf16 | 32768, 512 threads | 16 / 64 / 256 | 27.4 / 51.2 / 53.1 | unsupported | 22.9 / 42.5 / 42.5 | unsupported |

Registers (latency kernel, det2 against idx): 40 against 32 at IPT 4, 64 against 47 to 48 at IPT 16, 206 to 209 against 128 at IPT 64. Throughput occupancy 6 against 8 blocks per SM at fp32 IPT 4 (8 against 8 at bf16), 4 against 5 at IPT 16, 1 against 2 at IPT 64, which is the 74 to 100 percent throughput price at N 16384. At 512 threads and N 32768 the 128-register cap spills the tie arrays (stack 216 bytes, 592 spill bytes, 74 STL and 74 LDL in the latency kernel), B1_idx at 512 threads has 100 to 104 registers and no spill. N 32768 at 256 threads is unsupported (IPT 128 exceeds the 64-bit tie mask).

## 4. Where the remaining cost sits (phase stamps, T56 method)

`approaches/B1/profile_det_phases.cu` with `-DB1_PROFILE`: thread 0 stamps `clock64` after the prologue, after each radix pass, after the tie ballots and their barrier, after the base scan and ranks, and after the epilogue behind an extra barrier. One block, 32 random normal rows, median cycles, the same header with the deterministic path on (det2) and off (B1_idx's code) on the same rows. Log `state/results/opt-B1det/prof/profile_165917.log`.

| cell (bf16 unless stated) | rows with a tie phase | tie ballots and barrier | tie scan and ranks | prologue and pass 1, det2 minus idx | epilogue, det2 minus idx |
|---|---|---|---|---|---|
| N 1024 k 16 (IPT 4) | 9 of 32 | 192 | 334 | plus 14 | minus 42 |
| N 1024 k 256 | 12 of 32 | 192 | 334 | minus 4 | minus 119 |
| N 4096 k 64 (IPT 16) | 21 of 32 | 433 | 571 | plus 67 | minus 236 |
| N 16384 k 256 (IPT 64) | 29 of 32 | 1546 | 1459 | plus 554 | minus 681 |
| fp32 N 1024 k 16 | 0 of 32 | not run | not run | minus 13 | minus 34 |
| fp32 N 16384 k 16 | 0 of 32 | not run | not run | plus 328 | plus 463 |

So the remaining price is the tie phase proper, 526 cycles per run at IPT 4, 1004 at IPT 16 and 3005 at IPT 64, split about evenly between the per-slot ballots with their table stores and barrier and the per-slot base shuffles and rank popcounts. Both halves are per item slot, not per tie: at IPT 64 they are 64 VOTE, 64 STS, 64 SHFL.IDX and 64 POPC per thread at two warps per scheduler, issue bound on the VOTE and SHFL pipes (T33). The phase runs whenever the ties do not exactly fill the remaining slots, 28 percent of the bf16 rows at N 1024 k 16 and 91 percent at N 16384 k 256, never on fp32 random rows. The second term at IPT 64 is the -0.0 normalization (330 to 550 cycles) and the third the registers of the two per-slot arrays (occupancy and the 512-thread spill). The B1_det2 epilogue is cheaper than B1_idx's bounded tie store for bf16 and pays part of the phase back. Run fraction times phase cost plus normalization minus the epilogue difference reproduces the harness price within 25 percent at every profiled cell (about 100 against 93 cycles at bf16 N 1024 k 16, 740 against 934 at N 4096 k 64, 3150 against 4093 at N 16384 k 256).

## 5. What to do next (not measured, batch budget of three honored with two)

- `B1_det3`, idea card `ideas/queue/B1_det3_candidate_rank_lookup.md`: store the full ballot word per warp and slot in the tie table instead of its popcount, write the per-slot bases of the shuffle scan to shared memory, then let only the threads that hold a candidate loop over the set bits of their candidate mask and compute `base + popc(ballot & lt_mask)` from two shared loads, writing the output directly when the rank is below the remaining k. No `tie_rank` or `ballots` register array, no per-slot shuffles or popcounts, the `taken` mask and the class 1 branch of the epilogue go. Expected: the tie scan half (334 to 1459 cycles) falls to the shuffle scan alone (about 150 to 300), the latency-kernel registers return to about the B1_idx count plus the mask, two blocks per SM at IPT 64 and no spill at 512 threads.
- A one-ballot any-candidate test per warp before the per-slot ballot loop, so warps without a tie (5 or 6 of 8 on random rows) write zeros and skip it. Saves issue pressure, not the barrier wait.
- The normalization at IPT 64 (330 to 550 cycles, 3 to 4 percent) could move into the digit extractor of the first pass only (T23's knowledge base 4.4 note), not tried.

## 6. Pointers

Cards `cards/B1_det2_b200_20261025_1.json` and `_2.json`. Batches `state/results/opt-B1det/B1_det2_vs_B1_det_focus_d2a_20261025_165020.json`, `B1_det2_vs_B1_idx_focus_d2b_20261025_165100.json`, `B1_det2_vs_B1_idx_focus_d2c_20261025_165139.json` (512 threads), profile `state/results/opt-B1det/prof/`. Code `harness/adapters/B1_det2.cuh`, `approaches/B1/block_topk_b1_det2.cuh`, `approaches/B1/profile_det_phases.cu`, findings `approaches/B1/findings_det2_b200.json` and `findings_det2_price_b200.json`. Catalogue entry source `approaches/B1/techniques/T_det_ticket.md`. Prior: `approaches/B1/REPORT.md` sections 4, 7 and 10, T21 to T23, T39, T56, T57, knowledge base 13.5 to 13.7.

# B1_det3: the per-candidate rank lookup (opt-B1det3, 2026-10-25)

Bounded follow-up to section 5 and the idea card `ideas/queue/B1_det3_candidate_rank_lookup.md`. Node umbriel-b200-018 (B200 sm_100, 148 SMs), nvcc 13.4.92, harness 0.8, one session (17:54 to 17:56 UTC) with four lock sections, phase profiles on umbriel-b200-091 (17:19 UTC) and 018 (17:29, 17:35, 17:54 UTC). Source commit e1c6ed6816 (`exp/topk-swarm/B1det3`). Latency in cycles per call, throughput in Gelem/s, random normal rows, 256 threads unless stated. Cards `cards/B1_det3_b200_20261025_1.json` (against B1_det2) and `_2.json` (against B1_idx under not_guaranteed), batches under `state/results/opt-B1det3/` (the `d4` labels, the `d3` labels are the intermediate commit 4497544a4f before the address pin of section 10).

## 7. What changed

`B1_det2`'s tie phase is per item slot: a ballot per slot and warp with its popcount in a table, one barrier, then per slot a base shuffle and a rank popcount, with one ballot word and one rank word per slot in registers across the barrier and into the epilogue. `B1_det3` (`approaches/B1/block_topk_b1_det3.cuh`, a copy of the det2 header in its own namespace, adapter `harness/adapters/B1_det3.cuh`) is per candidate:

- One candidate bit per slot in a 32-bit or 64-bit mask (the compiler builds it as a LOP3 tree). The test is `key == kth_prefix`, every bit is in the prefix when the phase runs.
- Above 8 items per thread (`tie_ballots::warp_sparse`): the warp ORs its masks (one REDUX.OR per word), zeros its row of the (warp, slot) ballot table, and ballots only the slots whose bit is set in the union, a warp-uniform loop over the set bits per 32-bit half. Up to 8 (`tie_ballots::per_slot`): one ballot per slot as det2, the word stored instead of its popcount. One block barrier. Warps without a candidate (the union, or one any-vote) skip the rest.
- Each warp with a candidate sums the popcounts of its slots over the warps and scans the slots by shuffles, the rounds of 32 slots loaded, summed and scanned side by side. Above 8 items per thread the bases go to the warp's row of a base table (warp sync, no block barrier) and the threads with a candidate loop over their set bits per 32-bit half: `rank = base[warp][slot] + popc(ballot[warp][slot] & lt_mask)`, two shared loads and a popcount, `out[total_selected + rank]` written when the rank is below the remaining k. Up to 8 the base comes by one shuffle per slot from its owning lane and the ballot word by a load with a compile-time offset, one predicated lookup per slot.
- The epilogue keeps the class-indexed ticket (T57) with no taken mask and no class 1 branch, on a 32-bit shared address taken once and pinned (section 10).

Temp storage 2144 + 2 x 4 x warps x IPT bytes: 2400 at IPT 4, 3168 at IPT 16, 6240 at IPT 64, 10368 at 512 threads and IPT 64 (det2 2256 to 4192, the base table is the difference). Barriers unchanged (21 bf16, 33 fp32 in the latency kernel).

## 8. The kill experiment (phase profile, T56 method)

`approaches/B1/profile_det3_phases.cu`: det2, det3 with per-slot ballots, det3 with warp-sparse ballots and the not_guaranteed path of the det2 header in one binary on the same 32 rows, with the digests of the three deterministic results compared (identical in every row at every cell). The idea card's threshold was the tie scan half below 300 cycles per run at bf16 N 4096 k 64 (from 571) and the IPT 64 registers at most 160. The scan half in the lookup form is 404 + 225 for the scan and the lookup by warp 0's stamps (the sparse form's inner stamps are warp 0's view, since the other warps run ahead to the epilogue), the whole-call measure is what counts: `post` is the last pass stamp to the end stamp behind a barrier (tie phase plus epilogue), medians over the rows with a tie phase, final commit, node 018, log `state/results/opt-B1det3/prof/profile_175423.log`.

| cell (bf16) | rows with a tie phase | det2 post | det3 post (adapter form) | not_guaranteed post | premium det2, det3 | registers det2, det3, idx (profiler kernel) |
|---|---|---|---|---|---|---|
| N 1024 k 16 (IPT 4, per slot) | 9 of 32 | 793 | 929 | 406 | 387, 523 | 40, 32, 32 |
| N 1024 k 256 | 12 | 1175 | 1161 | 660 | 515, 501 | 32, 32, 30 |
| N 4096 k 64 (IPT 16, sparse) | 21 | 2508 | 1651 | 1281 | 1227, 370 | 48 (24 spill), 48, 40 |
| N 4096 k 256 | 25 | 3066 | 2454 | 1980 | 1086, 474 | 48 (24 spill), 48, 40 |
| N 16384 k 256 (IPT 64) | 29 | 7445 | 5882 | 4714 | 2731, 1168 | 255, 100, 95 |
| N 32768 k 64, 512 threads | 20 | 10425 | 4654 | 3165 | 7260, 1489 | 128 (168 spill), 102, 98 |

fp32 rows never run the phase, there det3's post is 377 to 385 against det2's 421 and the not_guaranteed 453 at N 1024 k 16, and 2670 against 3719 and 3073 at N 16384 k 16. Verdict: passed at the named cell on the whole-call measure (the premium on tie rows minus 70 percent, 56 to 79 percent at the larger cells), registers 100 to 102 at IPT 64 against the 160 limit, and at IPT 4 a push: the lookup costs about what det2's four shuffles cost and the cheaper epilogue pays 19 cycles back on every row.

The first build (run 1, umbriel-b200-091, `prof/run1_umbriel-b200-091/`) showed three things the SASS explained: the 64-bit `__ffsll` in both loops is a ten-instruction emulation per iteration (negate, mask, FLO, two subtracts, select), the two scan rounds at IPT 64 ran back to back (round 2's loads after round 1's base store), and at IPT 4 the sparse ballots (248 cycles against 149 for det2's four ballots) and the base table round trip (store, warp sync, load, 937 against 793 on tie rows) cost more than they saved. Hence the 32-bit half loops, the interleaved rounds (tiescan 947 to 694 at IPT 64), and the per-slot form with the shuffled base up to IPT 8.

## 9. The harness (cards 1 and 2)

Card 1, det3 against det2, both prefer_smaller_index, same lock section. Card 2, det3 against B1_idx under not_guaranteed (`--control-req`), the price of the requirement. Before is card `B1_det2_b200_20261025_2` (node 068), the chained det2 against idx on node 018 (det2 from card 1's control, idx from card 2's) agrees with it within 1 point everywhere.

| key | N | k | det3 against det2, latency | throughput | price after, latency | before | price after, throughput | before |
|---|---|---|---|---|---|---|---|---|
| bf16 | 1024 | 16 / 64 / 256 | minus 0.8 / 1.2 / 2.0 | plus 2.3 / 3.0 / 4.9 | 4.4 / 6.5 / 7.9 | 5.2 / 7.8 / 10.3 | 10.1 / 10.3 / 9.3 | 12.6 / 13.6 / 14.7 |
| bf16 | 4096 | 16 / 64 / 256 | minus 1.6 / 4.1 / 4.7 | plus 9.7 / 18.6 / 17.4 | 10.5 / 19.1 / 17.0 | 12.2 / 24.3 / 22.5 | 11.1 / 17.4 / 14.8 | 22.0 / 39.1 / 34.9 |
| bf16 | 16384 | 16 / 64 / 256 | minus 5.2 / 11.0 / 14.9 | plus 59 / 65 / 70 | 6.8 / 12.2 / 16.4 | 12.1 / 25.3 / 36.7 | 7.7 / 12.4 / 17.2 | 71 / 86 / 100 |
| bf16 | 32768, 512 threads | 16 / 64 / 256 | minus 10.0 / 18.1 / 18.9 | plus 11 / 21 / 20 | 15.6 / 24.1 / 24.4 | 27.4 / 51.2 / 53.1 | 10.6 / 18.2 / 18.8 | 22.9 / 42.5 / 42.5 |
| fp32 | 1024 | 16 / 64 / 256 | minus 1.1 / 1.8 / 1.0 | plus 0.2 / 0.4 / 0.8 | minus 1.3 / 1.9 / 1.4 | minus 0.2 / 0.0 / 0.4 | 5.0 / 4.0 / 2.4 | 5.2 / 4.3 / 3.2 |
| fp32 | 4096 | 16 / 64 / 256 | plus 1.4 / minus 0.1 / minus 1.3 | plus 4.5 / 5.3 / 5.5 | plus 2.4 / minus 0.2 / minus 1.8 | 1.0 / minus 0.1 / minus 0.7 | 6.5 / 3.8 / 1.1 | 11.1 / 9.3 / 6.5 |
| fp32 | 16384 | 16 / 64 / 256 | minus 4.8 / 2.7 / 2.3 | plus 68 / 65 / 66 | 5.0 / 3.9 / 1.0 | 9.6 / 6.3 / 3.5 | 8.5 / 7.4 / 5.0 | 82 / 77 / 74 |
| fp32 | 32768, 512 threads | 16 / 64 / 256 | minus 2.0 / 1.2 / 1.0 | plus 3.5 / 3.3 / 1.6 | 8.6 / 6.1 / 3.9 | 9.2 / 6.9 / 5.0 | 5.3 / 4.4 / 2.9 | 9.1 / 7.8 / 4.6 |

Registers (latency kernel, det3 against det2 against idx): 40, 40, 32 at IPT 4, 48, 64, 47 to 48 at IPT 16, 128, 206 to 209, 128 at IPT 64, 102, 128 (592 spill bytes), 100 to 104 at 512 threads. Throughput kernel 32, 48, 127 to 128, 97 to 100 with 8, 5, 2, 1 blocks per SM, the same as idx (det2 6 to 8, 4, 1, 1). No stack frame and no spill anywhere except the latency kernel at IPT 16: 20 (bf16) and 72 (fp32) spill bytes, 3 and 8 STL, all in the kernel prologue and the clock bookkeeping outside the timed chain. SASS of the latency kernel at bf16 N 4096 k 64: 1896 instructions against det2's 2256.

## 10. The rematerialized shared window base

The first harness batch (`d3`, commit 4497544a4f, the same tie phase) came out slower than det2 in latency at N 4096 (plus 2.2 to 8.3 percent, also on fp32 rows that never run the tie phase) while the one-block profile showed det3 ahead by 700 cycles at the same cell. The SASS of the harness latency kernel explained it: at the 48-register target ptxas recomputed the CTA's shared window base in front of every ticket atomic (`S2UR SR_CgaCtaId`, `UMOV`, `ULEA`, `UIADD3` per item slot, 37 S2UR in the det3 kernel against 2 in det2's, which keeps the base in one uniform register at 64 registers), because the temp storage is a generic reference and the conversion is a pure function of a kernel parameter and a special register. Taking the counters' shared address once with `__cvta_generic_to_shared` changed nothing (the conversion itself was rematerialized), pinning the 32-bit address through `asm volatile("" : "+r"(addr))` and issuing the ticket as `atom.relaxed.cta.shared.add.u32` on it brought the count to 6 and the N 4096 cells from plus 2.4 to minus 4.1 percent (bf16 k 64) and from plus 6.6 to minus 0.1 (fp32 k 64) against the same control, 300 to 1100 cycles per call. Catalogue entry T66, source `approaches/B1/techniques/t_shared_base_remat.md`, the lookup itself is T65 (`t_candidate_lookup.md`).

## 11. What remains, and what to do next

- The tie phase is still per slot in two places: the mask build (about 3.5 instructions per slot, 230 at IPT 64) and the scan (136 instructions for two rounds, latency bound on 16 loads and two shuffle chains). On tie rows the premium is 370 at N 4096 k 64, 1168 at N 16384 k 256 and 1489 at 512 threads, and it runs on 66 to 91 percent of the bf16 rows there, so bf16 still pays 12 to 24 percent in latency at N 16384 and 32768. A cheaper mask build would be the next lever (the per-slot compare is shared with the epilogue, which recomputes it).
- The -0.0 normalization is now the largest fp32 term: 1000 to 1250 cycles per call at 512 threads and IPT 64 on rows without ties (fp32 N 32768 k 16, 12944 against 11914), about 450 at 256 threads (fp32 N 16384 k 16, 9416 against 8968). Section 5's idea, folding it into the first pass's digit extraction, is untested. At 16 warps per SM the two instructions per key are issue bound.
- The latency kernel at IPT 16 sits on ptxas's 48-register target with 20 to 72 spill bytes outside the chain. Harmless now, but any growth of the tie phase will spill into the chain. The throughput kernel at 48 is what gives 5 blocks per SM.
- At IPT 4 the lookup form is a push on tie rows and the cheaper epilogue wins on the rest, net minus 1 to 2 percent against det2. Nothing more to gain there without touching the normalization.
- det3 is faster than B1_idx's not_guaranteed path at fp32 N 1024 and 4096 in latency (minus 1.3 to 1.9 percent at N 1024): the class branch epilogue is cheaper than the bounded tie store. B1_idx could adopt it.
- Flood patterns (all_equal, tie_heavy, quantized_random) pass the oracle, their loop counts (IPT iterations per thread) are not timed.

## 12. Pointers

Cards `cards/B1_det3_b200_20261025_1.json` and `_2.json`. Batches `state/results/opt-B1det3/B1_det3_vs_B1_det2_focus_d4a_20261025_175424.json`, `_d4d_20261025_175553.json` (512 threads), `B1_det3_vs_B1_idx_focus_d4b_20261025_175501.json`, `_d4c_20261025_175537.json`, the intermediate `d3` batches at 4497544a4f, profiles `state/results/opt-B1det3/prof/` (`run1_umbriel-b200-091`, `run2_umbriel-b200-018`, `run3_umbriel-b200-018`, final `profile_175423.log`, `build.log`, `prof.sass.gz`), harness kernel SASS `state/results/opt-B1det3/*.sass.gz`. Code `harness/adapters/B1_det3.cuh`, `approaches/B1/block_topk_b1_det3.cuh`, `approaches/B1/profile_det3_phases.cu`, findings `approaches/B1/findings_det3_b200.json` and `findings_det3_price_b200.json`, catalogue entry sources `approaches/B1/techniques/t_candidate_lookup.md` and `t_shared_base_remat.md`.
