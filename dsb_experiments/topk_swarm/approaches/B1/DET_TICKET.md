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
