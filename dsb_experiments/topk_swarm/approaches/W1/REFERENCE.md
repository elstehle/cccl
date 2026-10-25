# W1 reference measurements: WarpBitonicTopK (PR #12222) as on the branch

Reference numbers for the warp bitonic top-k, measured by the evaluation engineer with the common
harness (`harness/README.md`), adapter `harness/adapters/W1_reference.cuh`. Not an approach report:
the W1 optimizer writes `REPORT.md` next to this file. Every number below has its A/A control, its
canaries and its raw file in the card named in section 3.

## 1. What was measured

- `cub::detail::WarpBitonicTopK<KeyT, MaxK, 32, int>::TopK(keys, idx, greater, k)` with
  MaxK = round_up(k, 32), int32 source indices as values, the indices of the first k striped
  positions copied to the harness output, keys dropped.
- Focus set: bf16 (compared with `__hgt`) and fp32 keys, random normal input, requirement
  `not_guaranteed`. `prefer_smaller_index` is unsupported by the primitive (ties are not index
  resolved) and those cells are reported as unsupported.
- Coarse grid: N 32, 128, 512, 2048, 4096 at 32 lanes (IPT 1 to 128), k 4 to N/2 in powers of
  four. k = 4 and k = 16 share MaxK = 32 and therefore the same numbers.
- Latency: chain-slope cycles per call net of the generate-only control (in-place chain, the
  network is data-oblivious). Throughput: one occupancy wave of 256-thread blocks with 8
  independent warps each, Gelem/s.

## 2. Tables

See the per-arch sections appended by `tools/summarize_batch.py --md` below.

## 3. Cards and raw files

Listed per arch below.

### b200 (umbriel-b200-068 CUDA 13.4.92 commit fc5de9f42987 2026-10-25)

Card: `cards/W1_reference_b200_20261025_1.json`, raw file: `state/results/eval-engineer/b200/W1_reference_vs_W1_reference_focus_focus2_20261025_104654.json`. Columns: latency cycles per call net of the control, the A/A control, raw and no-op slopes, fit R^2, throughput Gelem/s with its control and the harness ceiling, occupancy in blocks per SM, registers, spill bytes, mean tie multiplicity at rank k over the 16 inputs.

| key | N | k | req | lat cyc | ctrl | raw | noop | r2 | thr Gelem/s | ctrl | ceiling | occ | regs | spill B | ties@k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 32 | 4 | not_guaranteed | 778 | 778 | 826 | 48 | 1.0000 | 125.5 | 125.5 | 1565 | 8 | 26 | 0 | 1.1 |
| bf16 | 32 | 16 | not_guaranteed | 778 | 778 | 826 | 48 | 1.0000 | 125.5 | 125.4 | 1566 | 8 | 26 | 0 | 1.0 |
| bf16 | 128 | 4 | not_guaranteed | 2267 | 2267 | 2315 | 48 | 1.0000 | 105.8 | 105.8 | 5432 | 8 | 36 | 0 | 1.1 |
| bf16 | 128 | 16 | not_guaranteed | 2267 | 2267 | 2315 | 48 | 1.0000 | 105.8 | 105.8 | 5431 | 8 | 36 | 0 | 1.0 |
| bf16 | 128 | 64 | not_guaranteed | 2562 | 2562 | 2610 | 48 | 1.0000 | 88.2 | 88.2 | 5470 | 8 | 34 | 0 | 1.1 |
| bf16 | 512 | 4 | not_guaranteed | 12606 | 12606 | 12654 | 48 | 1.0000 | 98.0 | 97.5 | 7634 | 6 | 63 | 0 | 1.2 |
| bf16 | 512 | 16 | not_guaranteed | 12606 | 12606 | 12654 | 48 | 1.0000 | 97.3 | 97.2 | 7633 | 6 | 63 | 0 | 1.3 |
| bf16 | 512 | 64 | not_guaranteed | 15765 | 15765 | 15813 | 48 | 1.0000 | 75.9 | 77.4 | 7635 | 6 | 60 | 0 | 1.6 |
| bf16 | 512 | 256 | not_guaranteed | 22889 | 22889 | 22937 | 48 | 1.0000 | 53.8 | 53.6 | 7634 | 4 | 64 | 0 | 1.1 |
| bf16 | 2048 | 4 | not_guaranteed | 76068 | 76064 | 76116 | 48 | 1.0000 | 78.1 | 78.2 | 8551 | 2 | 164 | 0 | 1.1 |
| bf16 | 2048 | 16 | not_guaranteed | 76030 | 76030 | 76078 | 48 | 1.0000 | 78.0 | 78.2 | 8548 | 2 | 163 | 0 | 1.6 |
| bf16 | 2048 | 64 | not_guaranteed | 98410 | 98415 | 98458 | 48 | 1.0000 | 61.2 | 60.5 | 8548 | 2 | 165 | 0 | 2.1 |
| bf16 | 2048 | 256 | not_guaranteed | 145656 | 145651 | 145704 | 48 | 1.0000 | 40.8 | 41.6 | 8550 | 2 | 168 | 0 | 4.2 |
| bf16 | 2048 | 1024 | not_guaranteed | 188007 | 188009 | 188055 | 48 | 1.0000 | 24.0 | 24.0 | 5400 | 1 | 244 | 0 | 1.1 |
| bf16 | 4096 | 4 | not_guaranteed | 155795 | 155807 | 155844 | 49 | 1.0000 | 59.9 | 59.9 | 8869 | 1 | 255 | 436 | 1.2 |
| bf16 | 4096 | 16 | not_guaranteed | 155785 | 155819 | 155834 | 49 | 1.0000 | 59.9 | 59.9 | 8871 | 1 | 255 | 584 | 1.7 |
| bf16 | 4096 | 64 | not_guaranteed | 202023 | 202019 | 202072 | 49 | 1.0000 | 46.1 | 46.1 | 8863 | 1 | 255 | 372 | 3.5 |
| bf16 | 4096 | 256 | not_guaranteed | 304895 | 304893 | 304944 | 49 | 1.0000 | 31.2 | 31.2 | 5567 | 1 | 255 | 600 | 5.1 |
| bf16 | 4096 | 1024 | not_guaranteed | 397814 | 397841 | 397863 | 49 | 1.0000 | 23.4 | 23.4 | 4493 | 1 | 255 | 1544 | 5.4 |
| fp32 | 32 | 4 | not_guaranteed | 583 | 583 | 631 | 48 | 1.0000 | 197.8 | 197.8 | 1565 | 8 | 19 | 0 | 1.0 |
| fp32 | 32 | 16 | not_guaranteed | 583 | 583 | 631 | 48 | 1.0000 | 197.8 | 197.8 | 1565 | 8 | 19 | 0 | 1.0 |
| fp32 | 128 | 4 | not_guaranteed | 2041 | 2041 | 2089 | 48 | 1.0000 | 179.2 | 179.2 | 5338 | 8 | 32 | 0 | 1.0 |
| fp32 | 128 | 16 | not_guaranteed | 2041 | 2041 | 2089 | 48 | 1.0000 | 179.2 | 179.2 | 5339 | 8 | 32 | 0 | 1.0 |
| fp32 | 128 | 64 | not_guaranteed | 2152 | 2152 | 2200 | 48 | 1.0000 | 156.8 | 156.8 | 5325 | 8 | 33 | 0 | 1.0 |
| fp32 | 512 | 4 | not_guaranteed | 9970 | 9970 | 10018 | 48 | 1.0000 | 171.1 | 171.1 | 7635 | 6 | 56 | 0 | 1.0 |
| fp32 | 512 | 16 | not_guaranteed | 9970 | 9970 | 10018 | 48 | 1.0000 | 171.1 | 171.2 | 7634 | 6 | 56 | 0 | 1.0 |
| fp32 | 512 | 64 | not_guaranteed | 10538 | 10538 | 10586 | 48 | 1.0000 | 138.2 | 138.2 | 7629 | 6 | 59 | 0 | 1.0 |
| fp32 | 512 | 256 | not_guaranteed | 13920 | 13920 | 13968 | 48 | 1.0000 | 98.4 | 98.5 | 7629 | 4 | 61 | 0 | 1.0 |
| fp32 | 2048 | 4 | not_guaranteed | 49347 | 49340 | 49395 | 48 | 1.0000 | 132.2 | 132.2 | 8553 | 2 | 151 | 0 | 1.0 |
| fp32 | 2048 | 16 | not_guaranteed | 49343 | 49345 | 49391 | 48 | 1.0000 | 131.8 | 132.2 | 8548 | 2 | 151 | 0 | 1.0 |
| fp32 | 2048 | 64 | not_guaranteed | 66434 | 66437 | 66482 | 48 | 1.0000 | 108.0 | 109.5 | 8550 | 2 | 162 | 0 | 1.0 |
| fp32 | 2048 | 256 | not_guaranteed | 97283 | 97277 | 97331 | 48 | 1.0000 | 82.8 | 84.5 | 6779 | 2 | 162 | 0 | 1.0 |
| fp32 | 2048 | 1024 | not_guaranteed | 131669 | 131662 | 131717 | 48 | 1.0000 | 70.8 | 70.8 | 6595 | 2 | 168 | 0 | 1.0 |
| fp32 | 4096 | 4 | not_guaranteed | 130476 | 130474 | 130524 | 48 | 1.0000 | 88.1 | 88.1 | 6895 | 1 | 255 | 432 | 1.0 |
| fp32 | 4096 | 16 | not_guaranteed | 130474 | 130477 | 130522 | 48 | 1.0000 | 88.1 | 88.1 | 7006 | 1 | 255 | 432 | 1.0 |
| fp32 | 4096 | 64 | not_guaranteed | 169722 | 169722 | 169770 | 48 | 1.0000 | 68.4 | 68.4 | 7127 | 1 | 255 | 336 | 1.0 |
| fp32 | 4096 | 256 | not_guaranteed | 251271 | 251273 | 251319 | 48 | 1.0000 | 46.6 | 46.6 | 7384 | 1 | 255 | 296 | 1.0 |
| fp32 | 4096 | 1024 | not_guaranteed | 337437 | 337429 | 337485 | 48 | 1.0000 | 34.3 | 34.3 | 7830 | 1 | 255 | 440 | 1.0 |

### gb200 (gb200-nvl4-ts2-100 CUDA 13.4.92 commit fc5de9f42987 2026-10-25)

Card: `cards/W1_reference_gb200_20261025_1.json`, raw file: `state/results/eval-engineer/gb200/W1_reference_vs_W1_reference_focus_focus2_20261025_104644.json`. Columns: latency cycles per call net of the control, the A/A control, raw and no-op slopes, fit R^2, throughput Gelem/s with its control and the harness ceiling, occupancy in blocks per SM, registers, spill bytes, mean tie multiplicity at rank k over the 16 inputs.

| key | N | k | req | lat cyc | ctrl | raw | noop | r2 | thr Gelem/s | ctrl | ceiling | occ | regs | spill B | ties@k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 32 | 4 | not_guaranteed | 778 | 778 | 826 | 48 | 1.0000 | 134.7 | 134.7 | 1684 | 8 | 26 | 0 | 1.1 |
| bf16 | 32 | 16 | not_guaranteed | 778 | 778 | 826 | 48 | 1.0000 | 134.8 | 134.7 | 1685 | 8 | 26 | 0 | 1.0 |
| bf16 | 128 | 4 | not_guaranteed | 2267 | 2267 | 2315 | 48 | 1.0000 | 113.5 | 113.6 | 5829 | 8 | 36 | 0 | 1.1 |
| bf16 | 128 | 16 | not_guaranteed | 2267 | 2267 | 2315 | 48 | 1.0000 | 113.6 | 113.6 | 5819 | 8 | 36 | 0 | 1.0 |
| bf16 | 128 | 64 | not_guaranteed | 2562 | 2562 | 2610 | 48 | 1.0000 | 94.6 | 94.6 | 5875 | 8 | 34 | 0 | 1.1 |
| bf16 | 512 | 4 | not_guaranteed | 12586 | 12586 | 12634 | 48 | 1.0000 | 104.5 | 104.8 | 8197 | 6 | 63 | 0 | 1.2 |
| bf16 | 512 | 16 | not_guaranteed | 12586 | 12586 | 12634 | 48 | 1.0000 | 104.3 | 104.3 | 8190 | 6 | 63 | 0 | 1.3 |
| bf16 | 512 | 64 | not_guaranteed | 15736 | 15736 | 15784 | 48 | 1.0000 | 82.3 | 82.5 | 8197 | 6 | 60 | 0 | 1.6 |
| bf16 | 512 | 256 | not_guaranteed | 22862 | 22862 | 22910 | 48 | 1.0000 | 57.4 | 57.6 | 8169 | 4 | 64 | 0 | 1.1 |
| bf16 | 2048 | 4 | not_guaranteed | 78450 | 78472 | 78498 | 48 | 1.0000 | 83.6 | 83.9 | 9178 | 2 | 164 | 0 | 1.1 |
| bf16 | 2048 | 16 | not_guaranteed | 78320 | 78297 | 78368 | 48 | 1.0000 | 83.8 | 83.9 | 9177 | 2 | 163 | 0 | 1.6 |
| bf16 | 2048 | 64 | not_guaranteed | 101505 | 101482 | 101553 | 48 | 1.0000 | 65.3 | 64.2 | 9178 | 2 | 165 | 0 | 2.1 |
| bf16 | 2048 | 256 | not_guaranteed | 149793 | 149806 | 149841 | 48 | 1.0000 | 44.5 | 44.4 | 9183 | 2 | 168 | 0 | 4.2 |
| bf16 | 2048 | 1024 | not_guaranteed | 193109 | 193229 | 193157 | 48 | 1.0000 | 24.8 | 24.8 | 5811 | 1 | 244 | 0 | 1.1 |
| bf16 | 4096 | 4 | not_guaranteed | 159869 | 159912 | 159918 | 49 | 1.0000 | 62.2 | 62.2 | 9530 | 1 | 255 | 436 | 1.2 |
| bf16 | 4096 | 16 | not_guaranteed | 160011 | 160002 | 160060 | 49 | 1.0000 | 62.1 | 62.1 | 9525 | 1 | 255 | 584 | 1.7 |
| bf16 | 4096 | 64 | not_guaranteed | 207833 | 207870 | 207882 | 49 | 1.0000 | 47.9 | 47.9 | 9537 | 1 | 255 | 372 | 3.5 |
| bf16 | 4096 | 256 | not_guaranteed | 314188 | 314063 | 314237 | 49 | 1.0000 | 32.4 | 32.4 | 5962 | 1 | 255 | 600 | 5.1 |
| bf16 | 4096 | 1024 | not_guaranteed | 409199 | 409266 | 409248 | 49 | 1.0000 | 24.3 | 24.3 | 4789 | 1 | 255 | 1544 | 5.4 |
| fp32 | 32 | 4 | not_guaranteed | 605 | 605 | 653 | 48 | 1.0000 | 203.3 | 203.3 | 1683 | 8 | 22 | 0 | 1.0 |
| fp32 | 32 | 16 | not_guaranteed | 605 | 605 | 653 | 48 | 1.0000 | 203.3 | 203.2 | 1689 | 8 | 22 | 0 | 1.0 |
| fp32 | 128 | 4 | not_guaranteed | 2038 | 2038 | 2086 | 48 | 1.0000 | 191.9 | 192.1 | 5727 | 8 | 33 | 0 | 1.0 |
| fp32 | 128 | 16 | not_guaranteed | 2038 | 2038 | 2086 | 48 | 1.0000 | 191.8 | 192.1 | 5736 | 8 | 33 | 0 | 1.0 |
| fp32 | 128 | 64 | not_guaranteed | 2255 | 2255 | 2303 | 48 | 1.0000 | 164.3 | 164.5 | 5735 | 8 | 33 | 0 | 1.0 |
| fp32 | 512 | 4 | not_guaranteed | 10281 | 10281 | 10329 | 48 | 1.0000 | 183.3 | 182.9 | 8182 | 6 | 63 | 0 | 1.0 |
| fp32 | 512 | 16 | not_guaranteed | 10281 | 10281 | 10329 | 48 | 1.0000 | 183.0 | 183.0 | 8189 | 6 | 63 | 0 | 1.0 |
| fp32 | 512 | 64 | not_guaranteed | 11015 | 11015 | 11063 | 48 | 1.0000 | 150.2 | 149.7 | 8180 | 6 | 61 | 0 | 1.0 |
| fp32 | 512 | 256 | not_guaranteed | 15481 | 15481 | 15529 | 48 | 1.0000 | 105.9 | 106.0 | 8181 | 4 | 62 | 0 | 1.0 |
| fp32 | 2048 | 4 | not_guaranteed | 50216 | 48859 | 50264 | 48 | 1.0000 | 159.0 | 159.2 | 9162 | 3 | 150 | 0 | 1.0 |
| fp32 | 2048 | 16 | not_guaranteed | 50193 | 49096 | 50241 | 48 | 1.0000 | 159.1 | 159.2 | 9171 | 3 | 150 | 0 | 1.0 |
| fp32 | 2048 | 64 | not_guaranteed | 67955 | 68274 | 68003 | 48 | 1.0000 | 132.5 | 124.5 | 9152 | 2 | 165 | 0 | 1.0 |
| fp32 | 2048 | 256 | not_guaranteed | 100867 | 100438 | 100915 | 48 | 1.0000 | 98.1 | 98.1 | 7169 | 2 | 162 | 0 | 1.0 |
| fp32 | 2048 | 1024 | not_guaranteed | 141984 | 136712 | 142032 | 48 | 1.0000 | 74.1 | 73.8 | 7336 | 2 | 168 | 64 | 1.0 |
| fp32 | 4096 | 4 | not_guaranteed | 132315 | 132289 | 132363 | 48 | 1.0000 | 93.6 | 93.6 | 7375 | 1 | 255 | 604 | 1.0 |
| fp32 | 4096 | 16 | not_guaranteed | 132312 | 132298 | 132360 | 48 | 1.0000 | 93.6 | 93.5 | 7464 | 1 | 255 | 604 | 1.0 |
| fp32 | 4096 | 64 | not_guaranteed | 159561 | 159556 | 159609 | 48 | 1.0000 | 72.3 | 72.3 | 7498 | 1 | 255 | 524 | 1.0 |
| fp32 | 4096 | 256 | not_guaranteed | 226711 | 226726 | 226759 | 48 | 1.0000 | 48.2 | 48.2 | 7880 | 1 | 255 | 476 | 1.0 |
| fp32 | 4096 | 1024 | not_guaranteed | 309566 | 309446 | 309614 | 48 | 1.0000 | 35.5 | 35.5 | 8682 | 1 | 255 | 536 | 1.0 |
