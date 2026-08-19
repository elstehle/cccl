# Hybrid BlockMergeSort: stable warp-bitonic phase + cross-warp MergePath rounds

Does replacing BlockMergeSort's thread-local sort + five intra-warp merge rounds with ONE stable
(rank-augmented) warp bitonic sort pay off? Prototype `proto_hybrid_block_sort.cu` (self-contained
mirror; no cub includes), **umbriel-b200-022, B200 (sm_100, 148 SMs)**, 256 threads,
IPT ∈ {1,2,4,8} (tiles 256..2048), keys-only float and float+int pairs. Variants:
V0 = stock mirror (thread sort + 8 dynamic-MergePath rounds), V1 = V0 with statically-unrolled
search, V2 = hybrid (stable warp bitonic + 3 cross-warp dynamic rounds), V3 = hybrid + static.
All variants pass correctness **and stability** (vs `std::stable_sort` on (key, index); heavy
ties and all-equal patterns) at every size.

## Latency (single-block slope cyc/sort, gen-subtracted)

| | tile 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| keys V0 (stock) | 4986 | **5851** | **7814** | **11684** |
| keys V1 (static) | 4524 | 6269 | 9111 | 13199 |
| keys V2 (hybrid) | **4141** | 6808 | 13600 | 29571 |
| keys V3 (hyb+stat) | 4146 | 7344 | 14569 | 31005 |
| pairs V0 | 4882 | **6262** | **8860** | **14795** |
| pairs V2 (hybrid) | **4111** | 7016 | 14194 | 31310 |

## Throughput (one occupancy wave, Gelem/s) and resources

| | tile 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| keys V0 | 36.2 | **59.6** | **83.3** | **85.3** |
| keys V2 | **47.1** | 51.3 | 49.4 | 46.7 |
| pairs V0 | 34.4 | **53.8** | **62.4** | **57.7** |
| pairs V2 | **41.8** | 47.3 | 45.4 | 40.3 |

Registers/occupancy at IPT 1 (thr kernels): V0 = 46 regs / 5 blk/SM; **V2 = 32 regs / 8 blk/SM**.
V1 (static) = 78-80 regs / 3 blk/SM at all sizes. No spills anywhere.

## Findings

1. **The hybrid is a clean three-way win at IPT 1** (tile 256): latency **−16.9%** keys /
   **−15.8%** pairs, throughput **+30%** keys / +22% pairs, and it is *leaner* than stock
   (32 vs 46 registers, occupancy 8 vs 5 blk/SM) — the warp phase needs no indices array and no
   search state. Both regimes and the resource budget improve together, which none of the other
   levers in this project achieved.
2. **The crossover is at IPT ≈ 2, far left of the predicted 6–8** — and the miss decomposes into
   two causes worth separating:
   * *Implementation quality*: this prototype's warp phase is a naive full bitonic network
     (all `Σ min(log2 s, 5)` shuffle substages × IPT items × 2-3 channels ≈ 480+ shuffles at
     IPT 8). A cub-`WarpBitonicSort`-quality structure (thread-local phase for register-distance
     stages, single-channel where possible) is ~3× cheaper by the earlier full-warp
     measurements (unstable 128-elem sort: 1964 cyc there vs ≈11k implied here). With that,
     IPT 2 plausibly flips to a win and IPT 4 becomes borderline.
   * *Fundamental*: the O(N log²N) network work at warp tiles of 128–256 elements is deep into
     merge-wins territory regardless (the study's own bitonic/merge crossover, shifted left by
     the stable rank channel). IPT 8 stays with merge rounds no matter the implementation.
3. **Static-search anomaly in the mirror**: V1 loses latency at IPT ≥ 2 here (+7..+17%), while
   the real-cub port (`WMS_STATIC_SWITCH_RESULTS.md`) measured −10..−14% for tiles ≤ 512 on
   umbriel-b200-017. Differences: this mirror has no valid_items clamping at all (leaner dynamic
   baseline), possibly a different container toolkit, and the mirror's register allocation
   differs (V1 here: 78-80 regs). The static-switch decision should rest on the real-header
   measurements; this discrepancy is logged as further evidence of the switch family's
   codegen/toolkit sensitivity (third such observation after CUDA 13.1→13.3.1).
4. Static search never helps the hybrid (V3 ≥ V2 everywhere): with only three large cross-warp
   rounds left, the search is a small fraction and the register cost dominates.

## Recommendation

* Productize the hybrid **gated at IPT = 1 (possibly 2 after a proper warp phase)**: for
  256-thread blocks that is tile ≤ 256(-512) — precisely the small-tile regime where the
  partial-tile fix work (PR #10733) already concentrates, and where DeviceMergeSort's small
  inputs land. Larger IPT keeps the existing merge rounds unconditionally.
* Before any PR: replace the naive network with cub's `WarpBitonicSort` structure + rank channel
  (or literally extend cub's implementation with a stable mode and call it), re-measure the
  crossover, and reconcile the static-search mirror anomaly on real headers on this node.
* Design note carried from the analysis (untested): partial tiles in the hybrid can be handled
  without a sentinel by a static out-of-bounds flag that dominates the comparator
  (`(is_oob, key, rank)` lexicographic; comp is never invoked on out-of-range data) — cleaner
  than the current running-max pre-pass and it keeps each warp run's valid prefix contiguous for
  the clamped cross-warp rounds.

Reproduce: `nvcc -std=c++17 -arch=sm_100 -O3 proto_hybrid_block_sort.cu -o proto_hybrid_block_sort
&& ./proto_hybrid_block_sort [correct|lat|thr|res|all]`.
