# B1 reference measurements: block_topk_air as on the branch

Reference numbers for the block incumbent, measured by the evaluation engineer with the common
harness (`harness/README.md`), adapter `harness/adapters/B1_reference.cuh`. Not an approach report:
the B1 optimizer writes `REPORT.md` next to this file. Every number below has its A/A control, its
canaries and its raw file in the card named in section 3, and the board (`RESULTS_BOARD.md`) is the
place to compare variants against these rows.

## 1. What was measured

- `cub::detail::block_topk_air<KeyT, 256, IPT, int>::select_pairs<max, full tile>` with int32
  source indices as values, the selected indices copied from the first k blocked slots to the
  harness output, keys dropped (index-only output through the pairs path).
- Focus set: bf16 and fp32 keys, random normal input (bf16 rounded, so bf16 rows tie at the
  pivot), requirement `not_guaranteed`. `prefer_smaller_index` is unsupported by the primitive
  (atomic ticket order at the k-th key) and those cells are reported as unsupported.
- Coarse grid: N 1024, 4096, 16384, 32768 at 256 threads (IPT 4 to 128), k 4 to N/4 in powers of
  four (k = 4096 at N 16384 is k = N/4, the grid stops at N/2).
- Latency: chain-slope cycles per call net of the generate-only control (fresh input per call,
  data-dependent chain). Throughput: one occupancy wave, Gelem/s, with occupancy and registers.

## 2. Tables

See the per-arch sections appended by `tools/summarize_batch.py --md` below.

## 3. Cards and raw files

Listed per arch below.

### b200 (umbriel-b200-020 CUDA 13.4.92 commit b6bfae9a37bb 2026-10-25, harness 0.6)

Card: `cards/B1_reference_b200_20261025_1.json`, raw file: `state/results/eval-engineer/b200/B1_reference_vs_B1_reference_focus_focus3_20261025_154557.json`. Columns: latency cycles per call net of the control, the A/A control, raw and no-op slopes, fit R^2, throughput Gelem/s with its control and the harness ceiling, occupancy in blocks per SM, registers, spill bytes, local memory, mean tie multiplicity at rank k over the 16 inputs. Measured with harness 0.6 (input rotation per repetition, control with the variant's shared-memory footprint), which replaced the harness 0.2 measurement of the same grid (that batch is kept under `state/results/eval-engineer/b200/` with label focus2).

| key | N | k | req | lat cyc | ctrl | raw | noop | r2 | thr Gelem/s | ctrl | ceiling | occ | regs | spill B | local | ties@k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 1024 | 4 | not_guaranteed | 1770 | 1770 | 1909 | 139 | 1.0000 | 367.7 | 367.7 | 5155 | 8 | 40 | 0 |  | 1.4 |
| bf16 | 1024 | 16 | not_guaranteed | 1892 | 1892 | 2031 | 139 | 1.0000 | 349.6 | 349.6 | 5149 | 8 | 40 | 0 |  | 1.4 |
| bf16 | 1024 | 64 | not_guaranteed | 2025 | 2025 | 2164 | 139 | 1.0000 | 323.0 | 323.0 | 5157 | 8 | 40 | 0 |  | 2.1 |
| bf16 | 1024 | 256 | not_guaranteed | 2136 | 2135 | 2281 | 145 | 1.0000 | 297.5 | 297.5 | 5155 | 8 | 40 | 0 |  | 2.3 |
| bf16 | 4096 | 4 | not_guaranteed | 3390 | 3393 | 3942 | 552 | 1.0000 | 573.9 | 574.8 | 7561 | 5 | 63 | 0 |  | 1.2 |
| bf16 | 4096 | 16 | not_guaranteed | 3571 | 3573 | 4124 | 553 | 1.0000 | 533.9 | 533.4 | 7595 | 4 | 64 | 0 |  | 1.7 |
| bf16 | 4096 | 64 | not_guaranteed | 4140 | 4124 | 4690 | 551 | 1.0000 | 485.1 | 485.3 | 7602 | 4 | 64 | 0 |  | 3.5 |
| bf16 | 4096 | 256 | not_guaranteed | 4976 | 4976 | 5524 | 548 | 1.0000 | 406.4 | 406.1 | 7573 | 4 | 64 | 0 |  | 5.1 |
| bf16 | 4096 | 1024 | not_guaranteed | 5677 | 5676 | 6237 | 561 | 1.0000 | 328.7 | 328.8 | 7389 | 4 | 63 | 0 |  | 5.4 |
| bf16 | 16384 | 4 | not_guaranteed | 9952 | 9937 | 12475 | 2523 | 1.0000 | 526.1 | 528.1 | 2266 | 2 | 122 | 0 |  | 1.2 |
| bf16 | 16384 | 16 | not_guaranteed | 10018 | 10023 | 12542 | 2524 | 1.0000 | 527.1 | 528.6 | 2258 | 2 | 122 | 0 |  | 1.7 |
| bf16 | 16384 | 64 | not_guaranteed | 10161 | 10166 | 13077 | 2916 | 1.0000 | 510.8 | 509.9 | 1956 | 2 | 127 | 0 |  | 4.4 |
| bf16 | 16384 | 256 | not_guaranteed | 12718 | 12727 | 15629 | 2910 | 1.0000 | 445.1 | 444.9 | 1950 | 2 | 128 | 0 |  | 10.4 |
| bf16 | 16384 | 1024 | not_guaranteed | 17468 | 17467 | 20381 | 2913 | 1.0000 | 333.5 | 333.5 | 1990 | 2 | 128 | 0 |  | 17.7 |
| bf16 | 16384 | 4096 | not_guaranteed | 22078 | 22039 | 24604 | 2526 | 1.0000 | 256.1 | 256.1 | 2255 | 2 | 128 | 0 |  | 19.9 |
| bf16 | 32768 | 4 | not_guaranteed | 20416 | 20402 | 25024 | 4608 | 1.0000 | 376.0 | 376.0 | 1986 | 1 | 254 | 0 |  | 1.4 |
| bf16 | 32768 | 16 | not_guaranteed | 20401 | 20424 | 25005 | 4604 | 1.0000 | 376.0 | 376.5 | 1982 | 1 | 254 | 0 |  | 1.6 |
| bf16 | 32768 | 64 | not_guaranteed | 20841 | 20842 | 25446 | 4606 | 1.0000 | 371.9 | 372.1 | 1984 | 1 | 254 | 0 |  | 4.1 |
| bf16 | 32768 | 256 | not_guaranteed | 23050 | 23040 | 27657 | 4607 | 1.0000 | 340.4 | 340.4 | 1982 | 1 | 255 | 0 |  | 11.2 |
| bf16 | 32768 | 1024 | not_guaranteed | 30132 | 30142 | 34767 | 4636 | 1.0000 | 270.6 | 270.8 | 1967 | 1 | 255 | 0 |  | 18.6 |
| bf16 | 32768 | 4096 | not_guaranteed | 41270 | 41259 | 45899 | 4629 | 1.0000 | 205.4 | 205.4 | 1975 | 1 | 255 | 0 |  | 54.0 |
| fp32 | 1024 | 4 | not_guaranteed | 1789 | 1789 | 1934 | 145 | 0.9999 | 381.1 | 380.9 | 5386 | 6 | 44 | 0 |  | 1.0 |
| fp32 | 1024 | 16 | not_guaranteed | 1891 | 1891 | 2036 | 145 | 0.9997 | 355.5 | 355.6 | 5376 | 6 | 40 | 0 |  | 1.0 |
| fp32 | 1024 | 64 | not_guaranteed | 2038 | 2037 | 2183 | 145 | 0.9996 | 326.8 | 326.7 | 5385 | 6 | 40 | 0 |  | 1.0 |
| fp32 | 1024 | 256 | not_guaranteed | 2312 | 2312 | 2463 | 151 | 0.9997 | 280.8 | 280.7 | 5386 | 6 | 44 | 0 |  | 1.0 |
| fp32 | 4096 | 4 | not_guaranteed | 3117 | 3111 | 4039 | 923 | 0.9999 | 571.1 | 570.7 | 2220 | 4 | 64 | 0 |  | 1.0 |
| fp32 | 4096 | 16 | not_guaranteed | 3699 | 3699 | 4624 | 926 | 0.9996 | 521.4 | 521.9 | 2219 | 4 | 64 | 0 |  | 1.0 |
| fp32 | 4096 | 64 | not_guaranteed | 4718 | 4732 | 5648 | 929 | 1.0000 | 435.3 | 435.6 | 2219 | 4 | 64 | 0 |  | 1.0 |
| fp32 | 4096 | 256 | not_guaranteed | 5485 | 5481 | 6412 | 927 | 0.9993 | 373.1 | 373.4 | 2218 | 4 | 64 | 0 |  | 1.0 |
| fp32 | 4096 | 1024 | not_guaranteed | 6472 | 6476 | 7401 | 928 | 1.0000 | 293.3 | 293.1 | 2162 | 4 | 64 | 0 |  | 1.0 |
| fp32 | 16384 | 4 | not_guaranteed | 10053 | 10060 | 14617 | 4564 | 1.0000 | 325.9 | 325.9 | 1144 | 1 | 169 | 0 |  | 1.0 |
| fp32 | 16384 | 16 | not_guaranteed | 10731 | 10729 | 15287 | 4556 | 0.9996 | 310.2 | 310.3 | 1141 | 1 | 169 | 0 |  | 1.0 |
| fp32 | 16384 | 64 | not_guaranteed | 11284 | 11304 | 17135 | 5851 | 0.9999 | 279.4 | 279.3 | 789 | 1 | 169 | 0 |  | 1.0 |
| fp32 | 16384 | 256 | not_guaranteed | 14064 | 14049 | 19891 | 5827 | 1.0000 | 237.8 | 238.1 | 788 | 1 | 169 | 0 |  | 1.0 |
| fp32 | 16384 | 1024 | not_guaranteed | 19019 | 19011 | 24867 | 5848 | 1.0000 | 190.0 | 190.0 | 793 | 1 | 169 | 0 |  | 1.0 |
| fp32 | 16384 | 4096 | not_guaranteed | 24514 | 24494 | 29114 | 4600 | 1.0000 | 162.8 | 162.9 | 1139 | 1 | 169 | 0 |  | 1.0 |
| fp32 | 32768 | 4 | not_guaranteed | 20171 | 20204 | 28848 | 8677 | 1.0000 | 326.6 | 326.5 | 1075 | 1 | 255 | 992 | 112B stl74 ldl82 | 1.0 |
| fp32 | 32768 | 16 | not_guaranteed | 20156 | 20193 | 28827 | 8671 | 0.9998 | 323.9 | 324.3 | 1075 | 1 | 255 | 992 | 112B stl74 ldl82 | 1.0 |
| fp32 | 32768 | 64 | not_guaranteed | 24153 | 24127 | 32830 | 8678 | 0.9999 | 289.0 | 289.0 | 1074 | 1 | 255 | 992 | 112B stl74 ldl82 | 1.0 |
| fp32 | 32768 | 256 | not_guaranteed | 28023 | 28018 | 36696 | 8673 | 1.0000 | 260.5 | 260.5 | 1075 | 1 | 255 | 1024 | 112B stl70 ldl80 | 1.0 |
| fp32 | 32768 | 1024 | not_guaranteed | 36260 | 36236 | 44937 | 8676 | 1.0000 | 213.6 | 213.4 | 1074 | 1 | 255 | 1024 | 112B stl70 ldl80 | 1.0 |
| fp32 | 32768 | 4096 | not_guaranteed | 48001 | 48036 | 56680 | 8679 | 1.0000 | 167.6 | 167.6 | 1072 | 1 | 255 | 1024 | 112B stl70 ldl80 | 1.0 |

### gb200 (gb200-nvl4-ts2-105 CUDA 13.4.92 commit 2151128eaa40 2026-10-25, harness 0.6)

Card: `cards/B1_reference_gb200_20261025_1.json`, raw file: `state/results/eval-engineer/gb200/B1_reference_vs_B1_reference_focus_focus3_20261025_160627.json`. Same columns as the b200 section. Measured with harness 0.6, replacing the harness 0.2 measurement of the same grid (kept under `state/results/eval-engineer/gb200/` with label focus2).

| key | N | k | req | lat cyc | ctrl | raw | noop | r2 | thr Gelem/s | ctrl | ceiling | occ | regs | spill B | local | ties@k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 1024 | 4 | not_guaranteed | 1770 | 1770 | 1909 | 139 | 1.0000 | 391.6 | 391.5 | 5558 | 8 | 40 | 0 |  | 1.4 |
| bf16 | 1024 | 16 | not_guaranteed | 1892 | 1892 | 2031 | 139 | 1.0000 | 372.5 | 372.3 | 5553 | 8 | 40 | 0 |  | 1.4 |
| bf16 | 1024 | 64 | not_guaranteed | 2025 | 2025 | 2164 | 139 | 1.0000 | 343.9 | 344.1 | 5529 | 8 | 40 | 0 |  | 2.1 |
| bf16 | 1024 | 256 | not_guaranteed | 2135 | 2136 | 2280 | 145 | 1.0000 | 317.0 | 316.9 | 5539 | 8 | 40 | 0 |  | 2.3 |
| bf16 | 4096 | 4 | not_guaranteed | 3380 | 3380 | 3934 | 554 | 1.0000 | 612.1 | 612.0 | 8124 | 5 | 63 | 0 |  | 1.2 |
| bf16 | 4096 | 16 | not_guaranteed | 3563 | 3577 | 4116 | 554 | 1.0000 | 568.9 | 569.4 | 8133 | 4 | 64 | 0 |  | 1.7 |
| bf16 | 4096 | 64 | not_guaranteed | 4138 | 4140 | 4690 | 552 | 1.0000 | 516.9 | 516.3 | 8137 | 4 | 64 | 0 |  | 3.5 |
| bf16 | 4096 | 256 | not_guaranteed | 4967 | 4963 | 5521 | 554 | 1.0000 | 432.1 | 432.4 | 8125 | 4 | 64 | 0 |  | 5.1 |
| bf16 | 4096 | 1024 | not_guaranteed | 5676 | 5685 | 6234 | 558 | 1.0000 | 349.9 | 350.0 | 7767 | 4 | 63 | 0 |  | 5.4 |
| bf16 | 16384 | 4 | not_guaranteed | 9623 | 9621 | 12151 | 2528 | 1.0000 | 563.7 | 563.7 | 2414 | 2 | 122 | 0 |  | 1.2 |
| bf16 | 16384 | 16 | not_guaranteed | 9681 | 9698 | 12207 | 2526 | 1.0000 | 562.9 | 563.8 | 2397 | 2 | 122 | 0 |  | 1.7 |
| bf16 | 16384 | 64 | not_guaranteed | 10001 | 10024 | 12921 | 2920 | 1.0000 | 545.7 | 546.5 | 2049 | 2 | 127 | 0 |  | 4.4 |
| bf16 | 16384 | 256 | not_guaranteed | 12661 | 12647 | 15576 | 2914 | 1.0000 | 474.5 | 476.4 | 2065 | 2 | 128 | 0 |  | 10.4 |
| bf16 | 16384 | 1024 | not_guaranteed | 17453 | 17439 | 20359 | 2906 | 1.0000 | 355.4 | 356.3 | 2094 | 2 | 128 | 0 |  | 17.7 |
| bf16 | 16384 | 4096 | not_guaranteed | 22024 | 21997 | 24558 | 2534 | 1.0000 | 273.4 | 272.9 | 2259 | 2 | 128 | 0 |  | 19.9 |
| bf16 | 32768 | 4 | not_guaranteed | 19759 | 19752 | 24370 | 4612 | 1.0000 | 401.2 | 402.0 | 2104 | 1 | 254 | 0 |  | 1.4 |
| bf16 | 32768 | 16 | not_guaranteed | 19787 | 19795 | 24393 | 4606 | 1.0000 | 401.2 | 403.3 | 2107 | 1 | 254 | 0 |  | 1.6 |
| bf16 | 32768 | 64 | not_guaranteed | 20230 | 20254 | 24836 | 4607 | 1.0000 | 397.8 | 398.5 | 2099 | 1 | 254 | 0 |  | 4.1 |
| bf16 | 32768 | 256 | not_guaranteed | 22923 | 22899 | 27531 | 4608 | 1.0000 | 362.1 | 363.3 | 2108 | 1 | 255 | 0 |  | 11.2 |
| bf16 | 32768 | 1024 | not_guaranteed | 30038 | 30053 | 34672 | 4634 | 1.0000 | 288.9 | 288.9 | 2105 | 1 | 255 | 0 |  | 18.6 |
| bf16 | 32768 | 4096 | not_guaranteed | 41109 | 41119 | 45740 | 4630 | 1.0000 | 219.0 | 219.2 | 2103 | 1 | 255 | 0 |  | 54.0 |
| fp32 | 1024 | 4 | not_guaranteed | 1789 | 1789 | 1934 | 145 | 0.9999 | 405.8 | 405.8 | 5795 | 6 | 44 | 0 |  | 1.0 |
| fp32 | 1024 | 16 | not_guaranteed | 1891 | 1891 | 2036 | 145 | 0.9997 | 378.7 | 378.8 | 5794 | 6 | 40 | 0 |  | 1.0 |
| fp32 | 1024 | 64 | not_guaranteed | 2037 | 2037 | 2183 | 145 | 0.9996 | 348.0 | 347.9 | 5787 | 6 | 40 | 0 |  | 1.0 |
| fp32 | 1024 | 256 | not_guaranteed | 2312 | 2312 | 2463 | 151 | 0.9997 | 299.0 | 299.2 | 5789 | 6 | 44 | 0 |  | 1.0 |
| fp32 | 4096 | 4 | not_guaranteed | 3111 | 3122 | 4043 | 932 | 0.9999 | 609.8 | 610.3 | 2370 | 4 | 64 | 0 |  | 1.0 |
| fp32 | 4096 | 16 | not_guaranteed | 3697 | 3693 | 4629 | 932 | 0.9996 | 556.2 | 555.6 | 2368 | 4 | 64 | 0 |  | 1.0 |
| fp32 | 4096 | 64 | not_guaranteed | 4734 | 4729 | 5663 | 929 | 1.0000 | 464.8 | 464.7 | 2372 | 4 | 64 | 0 |  | 1.0 |
| fp32 | 4096 | 256 | not_guaranteed | 5481 | 5485 | 6413 | 932 | 0.9993 | 397.9 | 397.9 | 2368 | 4 | 64 | 0 |  | 1.0 |
| fp32 | 4096 | 1024 | not_guaranteed | 6465 | 6458 | 7402 | 937 | 1.0000 | 312.6 | 312.7 | 2305 | 4 | 64 | 0 |  | 1.0 |
| fp32 | 16384 | 4 | not_guaranteed | 9746 | 9755 | 14307 | 4561 | 1.0000 | 348.0 | 348.1 | 1216 | 1 | 169 | 0 |  | 1.0 |
| fp32 | 16384 | 16 | not_guaranteed | 10450 | 10446 | 15016 | 4565 | 0.9996 | 331.2 | 333.4 | 1214 | 1 | 169 | 0 |  | 1.0 |
| fp32 | 16384 | 64 | not_guaranteed | 11120 | 11098 | 16999 | 5879 | 0.9999 | 299.2 | 298.4 | 815 | 1 | 169 | 0 |  | 1.0 |
| fp32 | 16384 | 256 | not_guaranteed | 14005 | 13981 | 19854 | 5848 | 1.0000 | 254.5 | 255.0 | 816 | 1 | 169 | 0 |  | 1.0 |
| fp32 | 16384 | 1024 | not_guaranteed | 18981 | 18947 | 24853 | 5872 | 1.0000 | 203.2 | 203.1 | 822 | 1 | 169 | 0 |  | 1.0 |
| fp32 | 16384 | 4096 | not_guaranteed | 24372 | 24424 | 28967 | 4595 | 1.0000 | 173.7 | 173.5 | 1215 | 1 | 169 | 0 |  | 1.0 |
| fp32 | 32768 | 4 | not_guaranteed | 19465 | 19456 | 28140 | 8675 | 1.0000 | 349.7 | 349.9 | 1145 | 1 | 255 | 992 | 112B stl74 ldl82 | 1.0 |
| fp32 | 32768 | 16 | not_guaranteed | 19452 | 19442 | 28136 | 8684 | 0.9998 | 346.8 | 346.7 | 1145 | 1 | 255 | 992 | 112B stl74 ldl82 | 1.0 |
| fp32 | 32768 | 64 | not_guaranteed | 23437 | 23399 | 32111 | 8674 | 0.9999 | 308.1 | 309.9 | 1146 | 1 | 255 | 992 | 112B stl74 ldl82 | 1.0 |
| fp32 | 32768 | 256 | not_guaranteed | 27845 | 27834 | 36518 | 8673 | 1.0000 | 277.3 | 277.8 | 1146 | 1 | 255 | 1024 | 112B stl70 ldl80 | 1.0 |
| fp32 | 32768 | 1024 | not_guaranteed | 36064 | 36072 | 44744 | 8681 | 1.0000 | 228.1 | 227.5 | 1145 | 1 | 255 | 1024 | 112B stl70 ldl80 | 1.0 |
| fp32 | 32768 | 4096 | not_guaranteed | 47912 | 47921 | 56596 | 8684 | 1.0000 | 178.3 | 178.4 | 1144 | 1 | 255 | 1024 | 112B stl70 ldl80 | 1.0 |
