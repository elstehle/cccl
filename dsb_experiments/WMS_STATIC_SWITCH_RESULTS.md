# MERGE_SORT_SEARCH_STATIC — a configurable static-MergePath switch in the real cub headers

Productization of the WarpMergeSort study's top finding (`WARP_MERGE_SORT_RESULTS.md` §4): the
statically-unrolled MergePath diagonal search, packaged as an opt-in template parameter on the
**real** `cub::WarpMergeSort` and `cub::BlockMergeSort` (patched headers in `wms_static/cub/`,
shadow-included ahead of the repo). Node **umbriel-b200-017, B200 (sm_100), 148 SMs** (container
toolkit version not captured before the node rotated; the dynamic-path baselines reproduce the
earlier CUDA 13.1/13.3.1 measurements within ~1%); harness `proto_wms_static.cu`.

## The API

```cpp
enum MergeSortSearchAlgorithm
{
  MERGE_SORT_SEARCH_DYNAMIC, // current while-loop search; lowest register footprint (default)
  MERGE_SORT_SEARCH_STATIC,  // fixed log2 trip count; lower latency, higher registers
};

cub::WarpMergeSort <KeyT, IPT, LOGICAL_WARP_THREADS, ValueT, SearchAlgorithm>
cub::BlockMergeSort<KeyT, BLOCK_DIM_X, IPT, ValueT, BLOCK_DIM_Y, BLOCK_DIM_Z, SearchAlgorithm>
```

Trailing + defaulted → source-compatible; the default path's codegen is untouched (its round loop
and `MergePath` are verbatim, only factored into a `MergeRound` member). Both classes share
`BlockMergeSortStrategy`, so the switch lives in one place; `AgentBlockSort` (DeviceMergeSort's
block phase) instantiates `BlockMergeSort` directly, so device-level tuning policies get the hook
for free. The static path drives the merge rounds through a compile-time `index_sequence` so round
`r` runs `StaticMergePath<IPT·2^(r-1)>`: a fixed `ceil(log2(MaxRange+1))+1`-step predicated binary
search. Probe indices are clamped into `[0, count]` — cub's partial-tile clamping passes
contract-degenerate searches (`diag > keys1_count + keys2_count`) to middle rounds, where the
dynamic `while` simply never iterates but a fixed-trip loop still issues its probe loads.

## Correctness (13 configs × {full tile, valid_items=1, valid_items=N/2+3} × pairs)

All PASS, and **dynamic and static produce byte-identical outputs in every test** — including the
degenerate partial-tile rounds — which is the strongest bug-compatibility evidence available.

A discovery about the *stock* contract along the way: with `valid_items < N` and
`oob_default = +inf`, output positions `>= valid_items` are **not** reliably `oob_default` (fails
for stock dynamic at `valid_items = N/2+3` for every IPT ≥ 2, warp and block). The oob pre-pass
skips `keys[0]` of fully-out-of-range threads, and middle rounds spread non-oob values into the
tail via one-past-run reads. The guaranteed property is the **sorted valid prefix** — which is all
`DeviceMergeSort` relies on (it even passes `keys_local[0]` as `oob_default`). The doc wording
("The value of oob_default is assigned to all elements that are out of valid_items boundaries")
overpromises; worth a docs clarification upstream, independent of this switch.

## Latency (slope cyc/call, random-input chain, generate-only control subtracted)

| scope | size | IPT | dynamic | static | delta |
|---|---|---|---|---|---|
| warp  | 32   | 1  | 2256 | **1812** | **−19.7%** |
| warp  | 64   | 2  | 3479 | **2639** | **−24.1%** |
| warp  | 96   | 3  | 4033 | **3114** | −22.8% |
| warp  | 128  | 4  | 4596 | **3849** | −16.2% |
| warp  | 160  | 5  | 5185 | **4120** | −20.5% |
| warp  | 192  | 6  | 5566 | **4515** | −18.9% |
| warp  | 256  | 8  | 6489 | **5568** | −14.2% |
| warp  | 320  | 10 | 7400 | **6119** | −17.3% |
| warp  | 384  | 12 | 8295 | **6972** | −15.9% |
| block | 256  | 1  | 5473 | **4715** | **−13.9%** |
| block | 512  | 2  | 6969 | **6239** | −10.5% |
| block | 1024 | 4  | 8940 | **8523** | −4.7% |
| block | 2048 | 8  | 12604 | **12477** | −1.0% |

* Warp-scope dynamic baselines reproduce the study (§1: 2236..8269) within ~1% — fourth node,
  third toolkit, same numbers.
* **The static-search win survives the port into the real headers and this toolkit**, and at
  IPT ≥ 8 exceeds the 13.1 prototype's MP-only deltas (−921/−1281/−1323 vs −421/−963/−941). The
  earlier "tuned variants regressed under 13.3" observation therefore traces to the *other*
  stacked optimizations (Batcher network / prefetch-shift), not the static search.
* Block scope (new data): the win concentrates in small tiles. At 256 threads the tile-2048 sort
  is barrier/serial-merge dominated and the search is a small slice; at tile 256–512 the search
  still matters (−10..−14%).

## Throughput (one occupancy wave, fixed workload, Gelem/s) — the cost side

| scope | size | 32 | 64 | 96 | 128 | 160 | 192 | 256 | 320 | 384 |
|---|---|---|---|---|---|---|---|---|---|---|
| warp dynamic | | 63 | **107** | **131** | 119 | **159** | **161** | 96 | **181** | **148** |
| warp static  | | **70** | 98 | 119 | 117 | 140 | 150 | 95 | 163 | 142 |

| scope | tile | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| block dynamic | | **34.4** | **52.9** | **79.7** | **84.8** |
| block static  | | 22.1 | 39.4 | 58.8 | 74.4 |

Static **regresses throughput** almost everywhere (warp −2..−12%, block −12..−36%; the one
exception is warp IPT 1, +11%). The mechanism is pure occupancy, as the resource table shows.

## Resources (thr kernel = 128 thr/block warp scope, 256 thr/block block scope)

| scope | IPT | regs dyn→stat | occupancy dyn→stat (blk/SM) | spills | smem Δ |
|---|---|---|---|---|---|
| warp  | 1  | 40→71 | 12→7 | 0/0 | 0 |
| warp  | 2–12 | 21–32→71–80 | 16→6-7 | 0/0 | 0 |
| block | 1–8 | 19–26→78–88 | 8→2-3 | 0/0 | 0 |

The register cost is much larger than the 13.1 prototype's +7..+16 because the prototype's
*baseline* already fully unrolled the round loop, while stock cub keeps it rolled (very lean,
19–32 regs). The static path **must** unroll the rounds to give each search a compile-time range,
and that unrolling — not the search itself — carries most of the register/occupancy cost. No
spills anywhere; shared memory unchanged.

## Verdict

1. **The switch design is validated**: default `MERGE_SORT_SEARCH_DYNAMIC` (throughput/occupancy
   unchanged by construction, and measured lean), opt-in `MERGE_SORT_SEARCH_STATIC` for the
   latency regime — warp scope −14..−24% on every size, block scope −10..−14% for tiles ≤ 512.
   Flipping the default would regress throughput users; keeping it opt-in regresses no one.
2. **Guidance for callers**: choose static for latency-critical warp-scope sorts and small block
   tiles at low residency (e.g. cooperative/persistent kernels, few warps per SM); stay dynamic
   for throughput kernels packing many warps. Warp IPT 1 is the rare both-axes win.
3. **Possible middle ground (unprototyped)**: a rolled-round variant using one flat
   `StaticMergePath<IPT·NumThreads/2>` bound for every round would keep the round loop rolled
   (dynamic-like registers) at ~30% more search steps — worth a look if block-scope throughput
   with static search ever matters.
4. **Upstreaming**: patch is API-preserving and lives in `wms_static/cub/{block,warp}/*.cuh`
   as a diff against the branch headers. Re-validate per CUDA release (the family of tuned
   variants has demonstrated toolkit sensitivity even though this one reproduced across 13.1→this
   toolkit); add the §-above docs clarification for the partial-tile oob contract.

Reproduce: `nvcc -std=c++17 -arch=sm_100 -O3 -Iwms_static -I../cub -I../libcudacxx/include
-I../thrust proto_wms_static.cu -o proto_wms_static && ./proto_wms_static [correct|lat|thr|res]`.
