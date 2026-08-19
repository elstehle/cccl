# Hybrid BlockMergeSort: stable warp-bitonic phase + cross-warp MergePath rounds

Does replacing BlockMergeSort's thread-local sort + five intra-warp merge rounds with ONE stable
(rank-augmented) warp bitonic sort pay off? Prototype `proto_hybrid_block_sort.cu` (self-contained
mirror; no cub includes), **umbriel-b200-022, B200 (sm_100, 148 SMs)**, 256 threads,
IPT ∈ {1,2,4,8} (tiles 256..2048), keys-only float and float+int pairs. Variants:
V0 = stock mirror (thread sort + 8 dynamic-MergePath rounds), V1 = V0 with statically-unrolled
search, V2 = hybrid (stable warp bitonic over the WHOLE warp tile + cross-warp dynamic rounds),
V3 = hybrid + static, V4 = **capped hybrid**: the network bootstrap is fixed at 64-element chunks
(sub-warp segments of 64/IPT lanes × IPT items; ≡ V2 at IPT ≤ 2) and MergePath rounds take over
from run length 64 — including the re-appearing intra-warp rounds at IPT ≥ 4. All variants pass
correctness **and stability** (vs `std::stable_sort` on (key, index); heavy ties and all-equal
patterns) at every size; V4 reproduces V2 exactly at IPT 1–2 (built-in cross-check).

## Latency (single-block slope cyc/sort, gen-subtracted)

| | tile 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| keys V0 (stock) | 4986 | 5851 | 7814 | **11684** |
| keys V1 (static) | 4524 | 6269 | 9111 | 13199 |
| keys V2 (hybrid) | 4141 | 6808 | 13600 | 29575 |
| keys V3 (hyb+stat) | 4146 | 7344 | 14569 | 31009 |
| keys V4 (cap 64, own net) | 4141 | 6808 | 12104 | 21887 |
| keys **V5 (cap 64, cub+packed)** | **3065 (−38%)** | **4260 (−27%)** | **7479 (−4%)** | 13084 (+12%) |
| pairs V0 | 4882 | 6262 | 8860 | **14795** |
| pairs V2 (hybrid) | 4111 | 7016 | 14194 | 31301 |
| pairs V4 (cap 64, own net) | 4111 | 7015 | 12726 | 24201 |
| pairs **V5 (cap 64, cub+packed)** | **3158 (−35%)** | **4816 (−23%)** | **8662 (−2%)** | 15472 (+5%) |

## Throughput (one occupancy wave, Gelem/s) and resources

| | tile 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| keys V0 | 36.2 | 59.5 | **83.4** | **85.3** |
| keys V2 | 47.2 | 51.3 | 49.5 | 46.7 |
| keys V4 | 47.1 | 51.3 | 56.9 | 63.0 |
| keys **V5** | **56.1 (+55%)** | **69.6 (+17%)** | 76.5 (−8%) | 76.2 (−11%) |
| pairs V0 | 34.4 | 53.7 | **62.5** | **57.7** |
| pairs V2 | 41.7 | 47.3 | 45.4 | 40.3 |
| pairs V4 | 41.8 | 47.3 | 50.6 | 48.2 |
| pairs **V5** | **47.2 (+37%)** | **55.4 (+3%)** | 55.5 (−11%) | 50.6 (−12%) |

V5 is also the **leanest** variant where it wins: 31–32 regs / 8 blk/SM at IPT ≤ 2 vs stock's
46–57 regs / 4–5 blk/SM. No spills anywhere.

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
5. **Capping the bootstrap at 64 elements (V4) strictly improves the hybrid at IPT ≥ 4** —
   −11% (IPT 4) and −26% (IPT 8) latency vs the whole-tile bootstrap, +15%/+35% throughput —
   confirming the "network only in its sweet spot, merge path from there" structure.
6. **V5 (capped bootstrap via the real `WarpBitonicSort` + u64 twiddle-pack, sub-warp logical
   warps of 64/IPT lanes) is the definitive design** (`STABLE_WRAP_RESULTS.md` has the
   component study). Final verdict vs stock:
   * **IPT 1: −38% latency, +55% throughput, 31 vs 46 regs, occupancy 8 vs 5** — a triple win
     larger than any single optimization measured in this project.
   * **IPT 2: −27% latency, +17% throughput, 32 vs 57 regs** — clear win on all axes.
   * IPT 4: ≈neutral (−4%/−2% latency, −8%/−11% throughput) — a latency-vs-throughput policy
     point, not a default.
   * IPT 8: stock wins (+12%/+5% latency, −11% throughput) — keep merge rounds.
   The earlier projection (−32/−24/−14) was too optimistic at IPT ≥ 4: sub-warp segment network
   cost does not scale down as favorably as the full-warp standalone numbers suggested.

## Recommendation

* Productize the **capped hybrid in its V5 form, gated at IPT ≤ 2** (unconditional win on
  latency, throughput, and registers), with IPT 4 available as a latency-leaning policy option
  and IPT ≥ 8 unchanged. Components: `WarpBitonicSort` (sub-warp branch) + a stable wrapper
  helper (twiddle-pack for radix-twiddleable keys + builtin comparators, (key, rank) struct +
  two-call comparator otherwise — a `STABLE` template parameter is sugar over this dispatch).
* PR sequencing: (1) `WarpBitonicSort` stable wrapper/mode + the keys-only-on-ValueT
  static_assert hardening, on top of the sub-warp branch; (2) hybrid `BlockMergeSort` policy
  using it; (3) the `MERGE_SORT_SEARCH_STATIC` switch stands separately on its real-header
  numbers (the mirror anomaly at IPT ≥ 2 remains logged; decisions there rest on
  `WMS_STATIC_SWITCH_RESULTS.md`).
* Non-float KeyT coverage (twiddle for ints/doubles, struct fallback for custom types) and
  partial-tile handling (the oob-flag comparator design note below) are the two open items for
  the productization pass.
* Design note carried from the analysis (untested): partial tiles in the hybrid can be handled
  without a sentinel by a static out-of-bounds flag that dominates the comparator
  (`(is_oob, key, rank)` lexicographic; comp is never invoked on out-of-range data) — cleaner
  than the current running-max pre-pass and it keeps each warp run's valid prefix contiguous for
  the clamped cross-warp rounds.

Reproduce: `nvcc -std=c++17 -arch=sm_100 -O3 proto_hybrid_block_sort.cu -o proto_hybrid_block_sort
&& ./proto_hybrid_block_sort [correct|lat|thr|res|all]`.
