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

### b200 (umbriel-b200-068 CUDA 13.4.92 commit fc5de9f42987 2026-10-25)

Card: `cards/B1_reference_b200_20261025_1.json`, raw file: `state/results/eval-engineer/b200/B1_reference_vs_B1_reference_focus_focus2_20261025_104523.json`. Columns: latency cycles per call net of the control, the A/A control, raw and no-op slopes, fit R^2, throughput Gelem/s with its control and the harness ceiling, occupancy in blocks per SM, registers, spill bytes, mean tie multiplicity at rank k over the 16 inputs.

| key | N | k | req | lat cyc | ctrl | raw | noop | r2 | thr Gelem/s | ctrl | ceiling | occ | regs | spill B | ties@k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 1024 | 4 | not_guaranteed | 1768 | 1768 | 1907 | 139 | 1.0000 | 367.6 | 367.4 | 5154 | 8 | 40 | 0 | 1.4 |
| bf16 | 1024 | 16 | not_guaranteed | 1882 | 1882 | 2021 | 139 | 1.0000 | 349.5 | 349.5 | 5151 | 8 | 40 | 0 | 1.4 |
| bf16 | 1024 | 64 | not_guaranteed | 2017 | 2017 | 2156 | 139 | 1.0000 | 322.8 | 323.0 | 5156 | 8 | 40 | 0 | 2.1 |
| bf16 | 1024 | 256 | not_guaranteed | 2129 | 2129 | 2270 | 141 | 1.0000 | 297.4 | 297.4 | 5154 | 8 | 40 | 0 | 2.3 |
| bf16 | 4096 | 4 | not_guaranteed | 3650 | 3650 | 3921 | 271 | 0.9996 | 574.3 | 574.3 | 7611 | 5 | 63 | 0 | 1.2 |
| bf16 | 4096 | 16 | not_guaranteed | 3845 | 3847 | 4116 | 271 | 0.9996 | 533.5 | 533.8 | 7601 | 4 | 63 | 0 | 1.7 |
| bf16 | 4096 | 64 | not_guaranteed | 4416 | 4412 | 4687 | 271 | 0.9998 | 485.6 | 484.8 | 7618 | 4 | 64 | 0 | 3.5 |
| bf16 | 4096 | 256 | not_guaranteed | 5251 | 5248 | 5522 | 271 | 0.9998 | 405.9 | 406.1 | 7607 | 4 | 64 | 0 | 5.1 |
| bf16 | 4096 | 1024 | not_guaranteed | 5774 | 5778 | 6201 | 427 | 0.9998 | 328.8 | 328.7 | 7201 | 4 | 64 | 0 | 5.4 |
| bf16 | 16384 | 4 | not_guaranteed | 9598 | 9591 | 12134 | 2536 | 1.0000 | 527.0 | 528.5 | 2261 | 2 | 122 | 0 | 1.2 |
| bf16 | 16384 | 16 | not_guaranteed | 9679 | 9675 | 12212 | 2533 | 1.0000 | 526.2 | 527.7 | 2264 | 2 | 122 | 0 | 1.7 |
| bf16 | 16384 | 64 | not_guaranteed | 10376 | 10376 | 12915 | 2540 | 1.0000 | 510.1 | 510.4 | 2262 | 2 | 127 | 0 | 4.4 |
| bf16 | 16384 | 256 | not_guaranteed | 12995 | 12987 | 15532 | 2537 | 1.0000 | 444.9 | 444.9 | 2269 | 2 | 128 | 0 | 10.4 |
| bf16 | 16384 | 1024 | not_guaranteed | 17750 | 17745 | 20283 | 2533 | 1.0000 | 333.5 | 333.5 | 2265 | 2 | 128 | 0 | 17.7 |
| bf16 | 16384 | 4096 | not_guaranteed | 21465 | 21481 | 24554 | 3089 | 1.0000 | 256.1 | 256.1 | 1958 | 2 | 128 | 0 | 19.9 |
| bf16 | 32768 | 4 | not_guaranteed | 19745 | 19784 | 24377 | 4632 | 1.0000 | 375.9 | 375.8 | 2290 | 1 | 254 | 0 | 1.4 |
| bf16 | 32768 | 16 | not_guaranteed | 19709 | 19704 | 24345 | 4636 | 1.0000 | 376.1 | 375.7 | 2291 | 1 | 254 | 0 | 1.6 |
| bf16 | 32768 | 64 | not_guaranteed | 20170 | 20209 | 24804 | 4634 | 1.0000 | 372.2 | 372.3 | 2286 | 1 | 254 | 0 | 4.1 |
| bf16 | 32768 | 256 | not_guaranteed | 22880 | 22896 | 27493 | 4614 | 1.0000 | 340.5 | 340.2 | 2290 | 1 | 255 | 0 | 11.2 |
| bf16 | 32768 | 1024 | not_guaranteed | 30048 | 30137 | 34676 | 4628 | 1.0000 | 270.6 | 270.9 | 2290 | 1 | 255 | 0 | 18.6 |
| bf16 | 32768 | 4096 | not_guaranteed | 40026 | 39994 | 45761 | 5735 | 1.0000 | 205.4 | 205.5 | 1737 | 1 | 255 | 0 | 54.0 |
| fp32 | 1024 | 4 | not_guaranteed | 1785 | 1785 | 1930 | 145 | 0.9998 | 381.1 | 380.8 | 5399 | 6 | 44 | 0 | 1.0 |
| fp32 | 1024 | 16 | not_guaranteed | 1905 | 1904 | 2050 | 145 | 0.9984 | 355.3 | 355.4 | 5381 | 6 | 44 | 0 | 1.0 |
| fp32 | 1024 | 64 | not_guaranteed | 2039 | 2039 | 2184 | 145 | 0.9994 | 326.6 | 326.6 | 5394 | 6 | 44 | 0 | 1.0 |
| fp32 | 1024 | 256 | not_guaranteed | 2313 | 2313 | 2460 | 147 | 0.9997 | 280.7 | 280.7 | 5398 | 6 | 44 | 0 | 1.0 |
| fp32 | 4096 | 4 | not_guaranteed | 3152 | 3151 | 4055 | 904 | 0.9989 | 571.6 | 571.4 | 2125 | 4 | 64 | 0 | 1.0 |
| fp32 | 4096 | 16 | not_guaranteed | 3705 | 3703 | 4605 | 901 | 1.0000 | 521.1 | 521.4 | 2122 | 4 | 64 | 0 | 1.0 |
| fp32 | 4096 | 64 | not_guaranteed | 4673 | 4668 | 5574 | 900 | 0.9999 | 435.6 | 435.7 | 2122 | 4 | 64 | 0 | 1.0 |
| fp32 | 4096 | 256 | not_guaranteed | 5305 | 5305 | 6221 | 916 | 0.9982 | 372.9 | 372.9 | 2126 | 4 | 64 | 0 | 1.0 |
| fp32 | 4096 | 1024 | not_guaranteed | 6419 | 6415 | 7338 | 919 | 0.9999 | 293.2 | 293.1 | 2179 | 4 | 64 | 0 | 1.0 |
| fp32 | 16384 | 4 | not_guaranteed | 9728 | 9742 | 14307 | 4579 | 0.9999 | 325.7 | 325.8 | 1144 | 1 | 169 | 0 | 1.0 |
| fp32 | 16384 | 16 | not_guaranteed | 10463 | 10448 | 15029 | 4566 | 0.9996 | 309.8 | 309.8 | 1145 | 1 | 169 | 0 | 1.0 |
| fp32 | 16384 | 64 | not_guaranteed | 12135 | 12164 | 16707 | 4572 | 1.0000 | 279.2 | 279.4 | 1146 | 1 | 169 | 0 | 1.0 |
| fp32 | 16384 | 256 | not_guaranteed | 15245 | 15237 | 19809 | 4565 | 1.0000 | 237.9 | 237.9 | 1145 | 1 | 169 | 0 | 1.0 |
| fp32 | 16384 | 1024 | not_guaranteed | 20290 | 20284 | 24853 | 4563 | 1.0000 | 190.0 | 190.1 | 1142 | 1 | 169 | 0 | 1.0 |
| fp32 | 16384 | 4096 | not_guaranteed | 23343 | 23323 | 29012 | 5669 | 1.0000 | 162.8 | 162.7 | 754 | 1 | 169 | 0 | 1.0 |
| fp32 | 32768 | 4 | not_guaranteed | 19469 | 19469 | 28155 | 8686 | 0.9999 | 326.5 | 326.4 | 1152 | 1 | 255 | 992 | 1.0 |
| fp32 | 32768 | 16 | not_guaranteed | 18955 | 18974 | 27655 | 8701 | 0.9998 | 324.0 | 324.1 | 1152 | 1 | 255 | 992 | 1.0 |
| fp32 | 32768 | 64 | not_guaranteed | 22847 | 22841 | 31523 | 8677 | 0.9999 | 288.5 | 288.2 | 1152 | 1 | 255 | 992 | 1.0 |
| fp32 | 32768 | 256 | not_guaranteed | 27420 | 27444 | 36115 | 8695 | 1.0000 | 260.2 | 260.2 | 1152 | 1 | 255 | 1024 | 1.0 |
| fp32 | 32768 | 1024 | not_guaranteed | 36111 | 36107 | 44803 | 8692 | 1.0000 | 213.3 | 213.4 | 1150 | 1 | 255 | 1024 | 1.0 |
| fp32 | 32768 | 4096 | not_guaranteed | 47125 | 47082 | 56633 | 9508 | 1.0000 | 167.4 | 167.4 | 741 | 1 | 255 | 1024 | 1.0 |

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
