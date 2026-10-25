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

### gb200 (gb200-nvl4-ts2-100 CUDA 13.4.92 commit fc5de9f42987 2026-10-25)

Card: `cards/B1_reference_gb200_20261025_1.json`, raw file: `state/results/eval-engineer/gb200/B1_reference_vs_B1_reference_focus_focus2_20261025_104524.json`. Columns: latency cycles per call net of the control, the A/A control, raw and no-op slopes, fit R^2, throughput Gelem/s with its control and the harness ceiling, occupancy in blocks per SM, registers, spill bytes, mean tie multiplicity at rank k over the 16 inputs.

| key | N | k | req | lat cyc | ctrl | raw | noop | r2 | thr Gelem/s | ctrl | ceiling | occ | regs | spill B | ties@k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 1024 | 4 | not_guaranteed | 1768 | 1766 | 1907 | 139 | 1.0000 | 394.6 | 394.9 | 5561 | 8 | 40 | 0 | 1.4 |
| bf16 | 1024 | 16 | not_guaranteed | 1882 | 1882 | 2021 | 139 | 1.0000 | 375.3 | 375.5 | 5541 | 8 | 40 | 0 | 1.4 |
| bf16 | 1024 | 64 | not_guaranteed | 2017 | 2017 | 2156 | 139 | 1.0000 | 346.8 | 346.7 | 5561 | 8 | 40 | 0 | 2.1 |
| bf16 | 1024 | 256 | not_guaranteed | 2129 | 2129 | 2270 | 141 | 1.0000 | 319.3 | 319.3 | 5538 | 8 | 40 | 0 | 2.3 |
| bf16 | 4096 | 4 | not_guaranteed | 3662 | 3658 | 3934 | 271 | 0.9996 | 617.5 | 617.2 | 8197 | 5 | 63 | 0 | 1.2 |
| bf16 | 4096 | 16 | not_guaranteed | 3835 | 3837 | 4106 | 271 | 0.9997 | 573.0 | 573.0 | 8203 | 4 | 63 | 0 | 1.7 |
| bf16 | 4096 | 64 | not_guaranteed | 4430 | 4429 | 4701 | 271 | 0.9997 | 521.2 | 521.1 | 8180 | 4 | 64 | 0 | 3.5 |
| bf16 | 4096 | 256 | not_guaranteed | 5248 | 5248 | 5519 | 271 | 0.9998 | 436.1 | 436.4 | 8197 | 4 | 64 | 0 | 5.1 |
| bf16 | 4096 | 1024 | not_guaranteed | 5765 | 5763 | 6208 | 443 | 0.9998 | 353.0 | 353.1 | 7696 | 4 | 64 | 0 | 5.4 |
| bf16 | 16384 | 4 | not_guaranteed | 9573 | 9572 | 12122 | 2549 | 1.0000 | 565.2 | 564.8 | 2436 | 2 | 122 | 0 | 1.2 |
| bf16 | 16384 | 16 | not_guaranteed | 9653 | 9660 | 12201 | 2548 | 1.0000 | 566.7 | 566.0 | 2430 | 2 | 122 | 0 | 1.7 |
| bf16 | 16384 | 64 | not_guaranteed | 10359 | 10349 | 12903 | 2544 | 1.0000 | 547.2 | 547.0 | 2432 | 2 | 127 | 0 | 4.4 |
| bf16 | 16384 | 256 | not_guaranteed | 12973 | 12967 | 15521 | 2548 | 1.0000 | 477.4 | 477.5 | 2436 | 2 | 128 | 0 | 10.4 |
| bf16 | 16384 | 1024 | not_guaranteed | 17749 | 17737 | 20291 | 2542 | 1.0000 | 357.7 | 358.1 | 2433 | 2 | 128 | 0 | 17.7 |
| bf16 | 16384 | 4096 | not_guaranteed | 21421 | 21433 | 24573 | 3151 | 1.0000 | 274.9 | 274.9 | 2061 | 2 | 128 | 0 | 19.9 |
| bf16 | 32768 | 4 | not_guaranteed | 19712 | 19726 | 24347 | 4635 | 1.0000 | 402.6 | 403.4 | 2463 | 1 | 254 | 0 | 1.4 |
| bf16 | 32768 | 16 | not_guaranteed | 19704 | 19671 | 24351 | 4647 | 1.0000 | 402.9 | 403.4 | 2463 | 1 | 254 | 0 | 1.6 |
| bf16 | 32768 | 64 | not_guaranteed | 20215 | 20161 | 24844 | 4629 | 1.0000 | 399.5 | 400.3 | 2464 | 1 | 254 | 0 | 4.1 |
| bf16 | 32768 | 256 | not_guaranteed | 22829 | 22781 | 27458 | 4630 | 1.0000 | 365.2 | 365.5 | 2462 | 1 | 255 | 0 | 11.2 |
| bf16 | 32768 | 1024 | not_guaranteed | 30055 | 30033 | 34693 | 4638 | 1.0000 | 290.3 | 290.8 | 2460 | 1 | 255 | 0 | 18.6 |
| bf16 | 32768 | 4096 | not_guaranteed | 39907 | 39860 | 45773 | 5866 | 1.0000 | 220.3 | 220.4 | 1809 | 1 | 255 | 0 | 54.0 |
| fp32 | 1024 | 4 | not_guaranteed | 1785 | 1785 | 1930 | 145 | 0.9998 | 409.0 | 408.7 | 5780 | 6 | 44 | 0 | 1.0 |
| fp32 | 1024 | 16 | not_guaranteed | 1905 | 1905 | 2050 | 145 | 0.9984 | 381.5 | 381.7 | 5792 | 6 | 44 | 0 | 1.0 |
| fp32 | 1024 | 64 | not_guaranteed | 2038 | 2038 | 2184 | 145 | 0.9994 | 350.8 | 350.6 | 5795 | 6 | 44 | 0 | 1.0 |
| fp32 | 1024 | 256 | not_guaranteed | 2313 | 2313 | 2460 | 147 | 0.9997 | 301.2 | 301.2 | 5793 | 6 | 44 | 0 | 1.0 |
| fp32 | 4096 | 4 | not_guaranteed | 3143 | 3141 | 4062 | 919 | 0.9988 | 613.8 | 614.5 | 2280 | 4 | 64 | 0 | 1.0 |
| fp32 | 4096 | 16 | not_guaranteed | 3707 | 3702 | 4624 | 917 | 1.0000 | 559.2 | 560.1 | 2276 | 4 | 64 | 0 | 1.0 |
| fp32 | 4096 | 64 | not_guaranteed | 4674 | 4676 | 5586 | 912 | 0.9998 | 467.3 | 467.2 | 2279 | 4 | 64 | 0 | 1.0 |
| fp32 | 4096 | 256 | not_guaranteed | 5304 | 5317 | 6235 | 931 | 0.9981 | 400.8 | 400.6 | 2281 | 4 | 64 | 0 | 1.0 |
| fp32 | 4096 | 1024 | not_guaranteed | 6414 | 6425 | 7348 | 934 | 0.9999 | 314.9 | 314.8 | 2336 | 4 | 64 | 0 | 1.0 |
| fp32 | 16384 | 4 | not_guaranteed | 9734 | 9725 | 14320 | 4586 | 0.9999 | 349.7 | 350.3 | 1228 | 1 | 169 | 0 | 1.0 |
| fp32 | 16384 | 16 | not_guaranteed | 10423 | 10426 | 15010 | 4587 | 0.9996 | 332.6 | 332.1 | 1230 | 1 | 169 | 0 | 1.0 |
| fp32 | 16384 | 64 | not_guaranteed | 12127 | 12140 | 16718 | 4591 | 1.0000 | 299.8 | 298.9 | 1232 | 1 | 169 | 0 | 1.0 |
| fp32 | 16384 | 256 | not_guaranteed | 15240 | 15248 | 19814 | 4574 | 1.0000 | 254.8 | 254.8 | 1230 | 1 | 169 | 0 | 1.0 |
| fp32 | 16384 | 1024 | not_guaranteed | 20290 | 20295 | 24860 | 4570 | 1.0000 | 203.8 | 203.8 | 1226 | 1 | 169 | 0 | 1.0 |
| fp32 | 16384 | 4096 | not_guaranteed | 23217 | 23224 | 29024 | 5807 | 1.0000 | 174.6 | 174.5 | 781 | 1 | 169 | 0 | 1.0 |
| fp32 | 32768 | 4 | not_guaranteed | 19391 | 19374 | 28103 | 8712 | 0.9999 | 349.7 | 349.7 | 1239 | 1 | 255 | 992 | 1.0 |
| fp32 | 32768 | 16 | not_guaranteed | 18903 | 18908 | 27612 | 8710 | 0.9998 | 347.3 | 347.7 | 1236 | 1 | 255 | 992 | 1.0 |
| fp32 | 32768 | 64 | not_guaranteed | 22753 | 22780 | 31441 | 8688 | 0.9999 | 308.3 | 309.5 | 1237 | 1 | 255 | 992 | 1.0 |
| fp32 | 32768 | 256 | not_guaranteed | 27353 | 27322 | 36046 | 8693 | 1.0000 | 278.0 | 278.4 | 1235 | 1 | 255 | 1024 | 1.0 |
| fp32 | 32768 | 1024 | not_guaranteed | 36084 | 36059 | 44784 | 8700 | 1.0000 | 228.5 | 228.5 | 1237 | 1 | 255 | 1024 | 1.0 |
| fp32 | 32768 | 4096 | not_guaranteed | 46922 | 46922 | 56603 | 9681 | 1.0000 | 179.1 | 179.2 | 772 | 1 | 255 | 1024 | 1.0 |
