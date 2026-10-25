# Sub-warp WarpBitonicSort (exp/sub-warp-bitonic-sort) vs sub-warp WarpMergeSort

Does the sub-warp support added to `WarpBitonicSort` (branch `exp/sub-warp-bitonic-sort`,
`cub::detail::WarpBitonicSort<KeyT, ItemsPerThread, LogicalWarpThreads>`) beat
`cub::WarpMergeSort<KeyT, IPT, LW>` for small inputs? **Yes — on every tested configuration,
on both latency and throughput.**

Setup: keys-only float, `LW ∈ {4,8,16} × IPT ∈ {1,2,4}` (segment sizes 4..64), one hardware
warp running `32/LW` concurrent sub-sorts (the realistic deployment; all sub-warps active in the
throughput wave too). Node **umbriel-b200-017, B200 (sm_100), 148 SMs**; harness
`proto_subwarp_sort.cu`; methodology as the full-warp study (`WARP_MERGE_SORT_RESULTS.md` §13):
bitonic = in-place dependency chain + `sink()`; merge = random-input chain, generate-only
control subtracted; throughput = one occupancy wave, element-normalized. Both primitives pass
per-segment correctness (merge blocked / bitonic striped arrangements handled) in all 9 configs.

## Results (latency: slope cyc/call for the whole warp of concurrent sub-sorts)

| LW | IPT | seg size | LAT merge | LAT bitonic | bitonic speedup | THR merge | THR bitonic | speedup |
|---|---|---|---|---|---|---|---|---|
| 4  | 1 | 4  | 797  | **258**  | **3.1×** | 187 | **378** | 2.0× |
| 4  | 2 | 8  | 1105 | **422**  | 2.6× | 317 | **634** | 2.0× |
| 4  | 4 | 16 | 1318 | **641**  | 2.1× | 413 | **540** | 1.3× |
| 8  | 1 | 8  | 1149 | **351**  | **3.3×** | 157 | **484** | **3.1×** |
| 8  | 2 | 16 | 1518 | **546**  | 2.8× | 220 | **425** | 1.9× |
| 8  | 4 | 32 | 2046 | **1251** | 1.6× | 277 | **339** | 1.2× |
| 16 | 1 | 16 | 1517 | **497**  | 3.1× | 106 | **350** | **3.3×** |
| 16 | 2 | 32 | 2181 | **698**  | 3.1× | 150 | **278** | 1.9× |
| 16 | 4 | 64 | 2875 | **2201** | 1.3× | 185 | **217** | 1.2× |

## Reading

* **Bitonic wins everywhere**: 1.3-3.3× latency, 1.2-3.3× throughput. The margin is largest at
  IPT=1 (pure shuffle network, zero shared memory, no MergePath/SerialMerge machinery) and
  narrows as IPT grows — the same O(N log²N)-vs-O(N log N) trend as the full-warp study, whose
  crossover sat at IPT≈10; at sub-warp segment sizes (≤64) merge never gets close enough to
  cross.
* Throughput tells the same story more sharply for merge: its per-logical-warp `TempStorage`
  multiplies with `blockDim/LW` sub-warps (e.g. 32 storages per 128-thread block at LW=4),
  costing occupancy exactly where bitonic needs none.
* The sub-warp plumbing in the branch adds no visible overhead: bitonic's LW=4/IPT=1 (258 cyc)
  scales cleanly from the full-warp 660 @ size 32.
* Caveats: keys-only float, random input; merge sort remains the choice where **stability** or
  non-power-of-2 logical widths are required (bitonic branch requires power-of-two LW).

**Recommendation:** for warp- and sub-warp-scope sorting of small segments (≤64 elements),
prefer the sub-warp `WarpBitonicSort` — this measurement supports the PR. A follow-up worth a
row in the PR: key-value pairs, where bitonic's per-stage value shuffles shift the balance
somewhat.

Reproduce: `nvcc -std=c++17 -arch=sm_100 -O3 -I<branch cub> -I<libcudacxx> -I<thrust>
proto_subwarp_sort.cu && ./proto_subwarp_sort`.
