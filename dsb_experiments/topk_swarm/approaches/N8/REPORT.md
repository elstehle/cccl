# N8: block size as a tuning dimension of the block primitives

Optimizer opt-N8, wave 2, 2026-10-25. Eight measurement batches, six on one B200 node (umbriel-b200-020, sm_100, nvcc 13.4.92, harness 0.4) and the two-node repeat of section 4b on umbriel-b200-068 and -020, every cell with its control in the same lock section and a passing oracle (13 cases per shape), 904 measured cells in all. Branch `exp/topk-swarm/N8`, worktree `cccl-topk-swarm-N8`, holder `opt-N8`, raw batches under `state/results/opt-N8/`. Numbers are cycles per call (latency, chain slope net of the generate-only control) or Gelem per s (throughput, one occupancy wave), bf16 first and fp32 second where both are given. The comparison across block sizes is across cells of one lock section (the harness pairs variant and control at one thread count), the comparison against the control is inside one process.

Cards: `N8_b2packed_b200_20261025_1.json` (B2_packed at 64, 128, 256 threads, N 1024 to 4096, 240 cells), `N8_b2packed_b200_20261025_2.json` (256 and 512 threads, N 8192 to 32768, 60 cells), `B2_twolevel_b200_20261025_2.json` (128 and 256 threads, N 8192 to 32768, 144 cells), `N8_b3tuned_b200_20261025_1.json` (B3_tuned at 256 and 512 threads against B1_idx, N 16384 and 32768, 128 cells), `N8_b1idx_b200_20261025_1.json` (B1_idx at 256, 512, 1024 threads, N 4096 to 32768, 144 cells), `B1_n6b_b200_20261025_2.json` (N6b at 256 and 512 threads, 48 cells), `N8_b2packed_b200_20261025_3.json` (the two-node repeat of the 128-thread cells, 192 cells on umbriel-b200-068 and -020). Catalogue entries T43, T44, T45, T46 (correction to T43 with the hardware researcher's barrier and round rows), T50 (the node split).

## 1. What it is

Not a new primitive: the wave 1 block primitives (B2_packed, B2_twolevel, B1_idx, B3_tuned, B1_n6b) measured at block sizes other than the 256 threads of the focus grid. Three thin adapters widen the support predicates and inherit the primitives unchanged: `N8_b2packed` (B2_packed with 64 threads allowed and the list capped at 64 words, T34), `N8_b1idx` (B1_idx with 64 and 1024 threads), `N8_b3tuned` (B3_tuned with 128, 512 and 1024 threads). B2_twolevel and B1_n6b already accepted 128 and 512 threads. The harness launches any block size that divides N without a change (64 and 1024 both worked), so no request went to the evaluation engineer. The cells file `harness/cells_n8.json` carries the thread counts 64 to 1024 and the k per N.

The hypothesis of the idea card had two halves: fewer warps per block make the B2 round cheaper (a 2-warp block predicted at 58 cycles per round and 1060 cycles at N 1024 k 16, against 1182 at 128 threads), and more threads per block halve the per-thread sweep of the radix passes at N 16384 and 32768 where the 256-thread kernels sit at 255 registers.

## 2. Contract

Unchanged per primitive (see the wave 1 reports). The only contract change is the thread count: B2_packed 64 to 512 threads with IPT at most 64 for both key types, B1_idx 64 to 1024, B3_tuned 128 to 1024. prefer_smaller_index stays free for B2 (within 1.0 percent of not_guaranteed at every block size) and for B3 (within 0.3 percent at 512 threads), and unsupported by B1_idx.

## 3. Where the time goes

3.1 The B2 round against warps per block. Round slopes from consecutive k at N 1024 bf16: 83.5 cycles at 8 warps and IPT 4, 69.5 at 4 warps and IPT 8, 77 at 2 warps and IPT 16 (fp32 the same within 5). At N 2048: 91, 78, 80 at IPT 8, 16, 32. The six rounds fit about 48 + 4 W + IPT cycles with W warps (T43): halving the warps saves the barrier and one aggregated post per warp (about 4 cycles per warp), doubling the list costs one cycle per word because the IPT - 1 predicated selects of the shift are issued in program order before the next round's post. At 16 warps the fit undershoots: 175 cycles at IPT 16 (N 8192 at 512 threads) against 128 predicted, the 44-cycle barrier (T11) and 16 serialized posts on one slot.

3.2 The B2 prologue against warps per block. The sort of IPT words costs 43, 74, 102 cycles for sort4, sort8, sort16 (256, 128, 64 threads at N 1024), 92, 200, 585 for sort8, sort16, sort32 at N 2048, and about 3080 for sort64 at 512 threads and N 32768. A 4-warp block issues the sort with one warp per scheduler and is 1.4x to 1.8x faster per thread than an 8 or 16-warp block on the same network (540 against 950 for sort32).

3.3 The radix pass against threads per block. B1_idx's pass is a dependent sweep of IPT predicated shared atomics per thread, a 256-bin scan and two barriers; the index-only epilogue is one shared atomic with return per selected item, serialized per warp. At N 32768, 512 threads halve the sweep (64 items), run 16 warps per SM at the same one block per SM, and split the epilogue over twice the warps: 1.24x to 1.36x bf16 and 1.32x to 1.45x fp32 in latency, growing with k, with 100 to 103 registers and 0 spill against 248 to 250 at 256 threads. 1024 threads quarter the sweep but run the scan on a quarter of the threads with 32-warp barriers and a 64-register cap: 2 to 12 percent behind 512 at N 32768, far behind at smaller N.

3.4 Throughput at equal threads per SM. 32 blocks of 64 threads, 16 of 128 and 8 of 256 all put 2048 threads on an SM, and the 64-thread B2 processes 1.2x to 1.66x the elements per second of the 256-thread one at N 1024 to 2048 (B1_idx 1.2x to 1.6x at the same N). The gain goes away where the register tier drops the block count (fp32 at 128 registers and above). The per-SM work is identical, so the effect is the barrier: a 2-warp block's round waits for one other warp and 32 independent blocks interleave, an 8-warp block idles 8 warps until its slowest arrives (T44).

## 4. Ideas tried, with verdicts

| item | batch | verdict |
|---|---|---|
| B2_packed at 64 threads (kill cell bf16 N 1024 k 16, threshold at most 1030 cycles) | b2t | killed for latency: 1422 against 1182 at 128 threads and 1380 at 256 (0.83x and 0.97x), fp32 1724 against 1494 and 1680; loses to 128 threads at all 27 (type, N, k) points, 0.64x to 0.97x. The round did not shrink (77 against 69.5, T43). |
| B2_packed at 64 threads, throughput regime | b2t | wins: 1.36x to 1.66x over 256 threads at bf16 N 1024 (820 against 495 at k 16), 1.18x to 1.50x at N 2048, 1.19x to 1.25x at N 4096, fp32 1.20x to 1.56x at N 1024; 128 threads 1.29x to 1.35x. Against B1_idx at 256 threads the 64-thread B2 wins throughput up to k 32 at N 1024 and k 16 at N 2048 and 4096 (T44). |
| B2_packed at 128 against 256 threads (new N 2048 cells) | b2t | 128 wins k 16 and 32 (1.07x and 1.12x bf16, 1.06x and 1.14x fp32), ties k 8. The 128-thread card's N 1024 and 4096 numbers reproduce to the cycle, with the N 4096 k 64 cells at this node's 7831 and 8263 (verify-B2's node dependence). |
| B2_twolevel at 128 against 256 threads, N 8192 to 32768 | tl | 256 from N 16384 (128 threads 0.34x to 0.89x, IPT 128 and 256 at 4 warps, 1.8 to 9 KB of spill at N 32768), tie at N 8192 (0.94x to 1.06x, fp32 k 4 1.46x for 128). Throughput 128 wins only the 1-block 256-thread cells at k 64 (1.43x at bf16 N 32768). |
| B2_packed at 512 threads, N 8192 to 32768 | b2big | loses every cell with 16 or more rounds (0.67x to 0.93x, round 175 to 203 cycles at 16 warps), wins k 4 at N 8192 (1.15x, 1.21x); at N 32768 (the one size at which the full list runs) 0.72x to 0.92x of the two-level list at 256 threads except bf16 k 4 (1.16x). |
| B1_idx at 512 threads, N 16384 and 32768 (threshold 1.15x under 256 threads at N 32768 k 64) | rx, bs | passes: 1.24x bf16 and 1.35x fp32 at N 32768 k 64 (13762 and 14154 against 17088 and 19070), 1.24x to 1.45x over all k, throughput 1.27x and 1.46x; N 16384 1.19x to 1.35x latency and 1.02x to 1.15x throughput; N 8192 1.05x to 1.17x latency but 0.70x to 0.84x throughput; N 4096 a tie in latency and 0.63x to 0.69x in throughput (T45). |
| B1_idx at 1024 threads | bs | never best: 0.93x to 0.99x (bf16) and 0.88x to 0.93x (fp32) of 512 threads at N 32768 in latency, 0.96x to 1.02x in throughput, 64 registers with 20 bytes of spill; 0.57x to 0.82x of 256 threads at N 4096 and 8192. |
| B3_tuned at 512 threads | rx, bs | 1.12x to 1.15x latency and 1.00x to 1.12x throughput over 256 threads, spills stay (64 bytes bf16, 844 fp32 at N 32768 under the 128-register cap); 1024 threads slower than 512 everywhere. Against B1_idx at 512 threads B3 keeps only a latency edge at k 1024 (1.03x to 1.22x) and loses every throughput cell. |
| B1_n6b at 512 threads | n6 | 1.05x to 1.19x latency and 1.14x to 1.24x throughput at N 16384 (where it does not spill), 0.88x to 1.08x at N 32768 (3.9 KB of spill left of 7.3 KB); behind B1_idx at 512 threads by 1.4x to 1.9x at N 32768. New: N6b at N 16384 k 16 is 1.48x and 1.58x ahead of B1_idx at 256 threads, its largest margin measured. |
| B3_tuned against B1_idx in one session (the final-evaluation pairing asked in PRIORITIES.md) | rx, bs | at 256 threads B3 wins from k at least N over 16 (1.08x to 1.46x latency) and loses throughput below k 1024; at 512 threads B3 wins latency at k 1024 only (1.03x to 1.22x), nothing in throughput; prefer_smaller_index remains B3's structural advantage at this N. |

Not iterated: no shape was within 10 percent of a win that an iteration could reach (the 64-thread B2 misses its threshold by 38 percent, the 512-thread radix passes it by 9 to 30 percent). The sixth batch went to the 512-thread B2 question instead of a second node, because the lease tool returned the same node (the only other free B200 was expiring in 30 minutes).

## 4b. Two-node repeat of the 128-thread B2 cells (added after verify-B2's refutation of card B2_packed_b200_20261025_3)

The lead asked for the N 4096 k 64 cells at 128 threads (plus the k 16 and k 32 neighbours, both key types, both requirements) on two different B200 nodes, with the 256-thread siblings and the B1_reference control in the same lock section and the hardware researcher's 100 ms nvidia-smi trace in the background. Card `N8_b2packed_b200_20261025_3.json` carries both nodes: umbriel-b200-068 (batch `N8_b2packed_vs_B1_reference_n8_ns068_20261025_152940.json`, 15:29Z) and umbriel-b200-020 (`N8_b2packed_vs_B1_reference_n8_ns020_20261025_154336.json`, 15:43Z), 48 shapes each (N 2048 and 4096, k 16, 32, 64, 128 and 256 threads, both types, both requirements), same binaries (source commit 07f4c8ff6d), same control. Clock traces `smi_ns068_umbriel-b200-068.csv` and `smi_ns020_umbriel-b200-020.csv`: SM clock 1965 MHz on every busy sample on both nodes, memory 3996 MHz, no event reason beyond idle, canaries identical.

Result: 94 of 96 cells agree between the nodes to 0.7 percent (latency 0.974x to 1.007x, throughput 0.997x to 1.002x, controls within 0.6 percent). The two exceptions are exactly the refuted cells, the 128-thread N 4096 k 64 kernels: bf16 7831 on -020 against 6056 on -068 (1.293x, 5008 SASS instructions, 78 KB of straight-line code, 48 registers, 0 spill), fp32 8264 against 6205 (1.332x, 6552 instructions, 102 KB, 128 registers). Their throughput cells (258 and 208 Gelem per s), their controls (6071 against 6062, 7092 against 7039) and their 256-thread siblings (7290 and 7696 on both, 3064 and 3728 instructions at two warps per scheduler) agree. The largest unaffected one-warp-per-scheduler kernel is 64 KB (fp32 k 32, 4120 instructions, 0.987x), so the threshold lies between 64 and 78 KB of code at 128 threads. The chain minima on -020 are higher by the same factor at every chain length (a steady per-call cost, not a cold-cache one-off). With card 3 (-047: 6254 and 6219, two repeats) this makes three nodes, two of which agree. Catalogue entry T50; the mechanism is with the hardware researcher (their code-size probe found nothing on -047 and -091, the -020 run is pending).

The k 64 conclusion of card 3 restated: on -047 and -068 the 128-thread B2 at N 4096 k 64 is 1.17x to 1.24x under its 256-thread form and ties B1_reference at 128 threads (1.00x bf16, 1.13x fp32), on -020 it is 0.93x and loses that control (0.78x, 0.86x). Against B1 at 256 threads, the B1 shape that matters, it loses on every node (0.73x to 0.75x against B1_reference, further against B1_idx), so the block size does not move B2's crossover with B1 at k 64 and the k 64 row of the 128-thread recommendation is dropped. New node-invariant cells: k 32 at N 4096 at 128 threads 3205 against 3801 (1.19x bf16) and 3394 to 3439 against 4066 to 4070 (1.18x fp32), k 32 and 64 at N 2048 1.12x to 1.19x; the 128-thread recommendation holds for k up to 32 at N 2048 to 4096 on both nodes. prefer_smaller_index equals not_guaranteed within 0.4 percent on both nodes. Throughput is node invariant in all 48 cells including the two affected kernels (whose throughput kernels are 39 and 51 KB).

## 5. Block size per primitive and cell (the recommendation for the final evaluation)

Latency regime (single block), best thread count and its ratio over the 256-thread form, bf16 and fp32:

| N | k | B2_packed | B2_twolevel | B1_idx | B3_tuned | N6b |
|---|---|---|---|---|---|---|
| 1024 | 4 to 16 | 128 (1.06x to 1.17x) | not measured | 256 (128: 0.78x, 64: 0.54x) | 256 (not swept) | 256 (not swept) |
| 1024 | 32 to 64 | 128 (1.19x to 1.20x, but B1_idx owns these) | | 256 | 256 | 256 |
| 2048 | 8 | 128 or 256 (0.99x, 0.97x) | | 256 | | |
| 2048 | 16 to 32 | 128 (1.06x to 1.14x) | | 256 | | |
| 4096 | 16 to 32 | 128 (1.08x to 1.19x, both nodes) | | 256 or 512 (0.98x to 1.02x) | 256 | |
| 4096 | 64 | 256 (128 node dependent: 0.93x on umbriel-b200-020, 1.17x to 1.24x on -047 and -068, T50; B1 owns the cell on every node) | | 256 or 512 (1.04x, 1.08x) | 256 (512: 0.91x) | |
| 8192 | 4 | 512 (1.15x, 1.21x) | 128 or 256 (0.94x, 1.46x) | 512 (1.13x, 1.05x) | 256 or 512 (1.05x, 0.97x) | |
| 8192 | 16 to 64 | 256 (512: 0.83x) | 128 or 256 (0.91x to 1.07x) | 512 (1.10x to 1.17x) | 256 or 512 | |
| 16384 | 4 to 16 | 256 (512: 0.80x) | 256 (128: 0.34x to 0.70x) | 512 (1.19x to 1.20x) | 512 (1.12x to 1.15x) | 512 (1.05x bf16, 0.96x fp32) |
| 16384 | 64 to 1024 | B1 cells | 256 | 512 (1.19x to 1.35x) | 512 (1.12x to 1.15x) | 512 (1.06x to 1.19x) |
| 32768 | 4 | 512 only (3869 bf16, beats two-level 4490) | 256 | 512 (1.25x, 1.32x) | 512 | |
| 32768 | 16 | 512 only (6227, two-level at 256 is 4895) | 256 | 512 (1.24x to 1.25x, 1.32x) | 512 (1.12x, 1.14x) | 256 or 512 (1.08x, 0.99x) |
| 32768 | 64 to 1024 | B1 cells | 256 | 512 (1.24x to 1.45x), 1024 second | 512 (1.12x to 1.15x) | 256 (fp32 512: 0.88x to 0.91x) |

Throughput regime (one wave), best thread count and its ratio over 256 threads:

| N | B2_packed | B2_twolevel | B1_idx | B3_tuned | N6b |
|---|---|---|---|---|---|
| 1024 | 64 (1.36x to 1.66x bf16, 1.20x to 1.56x fp32) | | 64 (1.2x to 1.6x) | not swept | not swept |
| 2048 | 64 bf16 (1.18x to 1.50x), 128 fp32 at k 8 (128 registers at 64) | | 64 (1.18x) | | |
| 4096 | 64 bf16 (1.19x to 1.25x), 256 fp32 (64: 254 registers) | | 256 (512: 0.63x to 0.69x) | 256 | |
| 8192 | 256 at k 16 and above, 512 at k 4 (1.09x, 1.35x) | 128 or 256 (1.00x to 1.12x) | 256 (512: 0.70x to 0.84x) | 256 | |
| 16384 | 256 (512: 0.85x to 1.01x, k 4 1.25x) | 256 (128: 0.57x to 0.98x) | 512 (1.02x to 1.15x) | 512 (1.00x to 1.12x) | 512 (1.09x to 1.24x) |
| 32768 | 512 only (two-level at 256 is better at k 16 and above) | 256 (128 at k 64: 1.43x) | 512 (1.27x to 1.46x), 1024 within 8 percent | 512 (1.06x to 1.08x) | 512 (0.98x to 1.21x) |

Block owners after this study, latency: k at most 16 B2 at 128 threads for N 1024 to 4096 (1182, 1474, 1907 bf16 against B1_idx at its best shape 1767, 2310, 3371), B2_twolevel at 256 threads for N 8192 to 32768 (946, 994, 4490 at k 4; 2648, 3362, 4895 at k 16, against B1_idx at 512 threads 4627, 7567, 13530 at k 16); k 32 at N 1024 B1_idx at 256 threads (1851 against B2 at 128 threads 2280); k 64 to 256 B1_idx at 256 threads up to N 4096 and at 512 threads from N 8192 (N 32768 k 64: 13762 and 14154 against the previous incumbent 17088 and 19070, B3 at 512 threads 17013 and 17088, N6b 25108 and 20392); k 1024 at N 16384 and 32768 B3_tuned at 512 threads by 3 to 22 percent in latency and B1_idx at 512 threads in throughput. For a batched agent (throughput): 64-thread blocks for B2 and B1_idx at N at most 2048, 256 at N 4096 to 8192, 512 from N 16384.

## 6. Determinism and ordering support

Unchanged per primitive: B2 inherent (same binary for both requirements at every block size, within 1 percent), B3 inherent (within 0.3 percent at 512 threads), B1_idx not_guaranteed only at every block size (B1_det was not swept), N6b measured under not_guaranteed only. The block size changes which warp posts or scatters an item, not the result set, and B2's sorted output is unchanged.

## 7. Open items

- The 128-thread N 4096 k 64 cells are now measured on two nodes (section 4b, T50). Still single-node: the 64-thread k 64 kernels of card 1 (7736 and 11320 instructions) and the 512-thread k 64 kernels of card 2, both on -020, and the 512-thread B1_idx (radix kernels were node invariant in wave 1).
- The hardware researcher measured the barrier (15.9, 20.0, 28.0, 44.5 cycles at 2, 4, 8, 16 warps) and the list-free aggregated round (barrier plus a flat 46), T46 corrects T43 accordingly: the block size lever on the B2 round is the barrier alone and the list shift costs it back. The 16-warp round (175 against 91.5 list-free) stays open.
- 384 and 768-thread blocks for B1_idx at N 4096 to 8192, where 512 ties or loses; the harness takes any multiple of 32 that divides N.
- The device-agent question: the one-wave throughput regime says 64-thread blocks at N at most 2048 and 512 from N 16384, with the input reloaded from L1 or L2 per call; a `segmented_topk` run with the block size as a parameter would confirm it for a kernel that loads each tile once.
- B1_det and B2_twolevel_any were not swept. A 512-thread B1_det doubles its per-warp tie table and halves its slots.
- B3's compaction threshold is per items per thread; with the block size free it may want to be per N.

## 8. Pointers

- Adapters: `harness/adapters/N8_b2packed.cuh`, `N8_b1idx.cuh`, `N8_b3tuned.cuh` (inheriting wrappers, the primitives are the wave 1 files). Cells: `harness/cells_n8.json`. Node helpers: `approaches/N8/node.sh`. Analysis: `approaches/N8/analyze.py <batch.json>` prints the cross-block-size table with resources and ratios.
- Findings files: `approaches/N8/findings_{b2packed,b2packed_big,twolevel,b3tuned,b1idx,n6b}_b200.json`. Catalogue entry sources: `approaches/N8/techniques/`.
- Raw batches (umbriel-b200-020 unless named, 2026-10-25): `N8_b2packed_vs_B1_reference_n8_ns068_20261025_152940.json` (umbriel-b200-068) and `..._ns020_20261025_154336.json` (the two-node repeat, 48 shapes each) with the clock traces `smi_ns068_umbriel-b200-068.csv` and `smi_ns020_umbriel-b200-020.csv`; `state/results/opt-N8/N8_b2packed_vs_N8_b1idx_n8_b2t_20261025_145019.json` (120 shapes), `N8_b3tuned_vs_B1_idx_n8_rx_20261025_145325.json` (64), `B2_twolevel_vs_B1_idx_n8_tl_20261025_145650.json` (72), `B1_n6b_vs_B1_idx_n8_n6_20261025_150045.json` (24), `N8_b1idx_vs_N8_b3tuned_n8_bs_20261025_150621.json` (72), `N8_b2packed_vs_N8_b1idx_n8_b2big_20261025_151404.json` (36, 6 unsupported). Per-cell logs and ptxas output on the node under `/work/topk_swarm/work/opt-N8/{logs,gen}`.
- Source commits: 2e1fc9520309 (batches b2t, rx, tl, n6), d8ce5463c450 (bs, b2big, the 1024-thread extension of the wrappers), 07f4c8ff6d49 (ns068, ns020, the k 32 cells and the clock-trace helper).
