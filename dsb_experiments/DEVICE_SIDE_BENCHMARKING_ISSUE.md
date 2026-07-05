# [CUB] A robust device-side (block/warp) benchmarking framework + 3 exemplary benchmarks

## Summary

CUB ships many *device-side* primitives (warp- and block-level: `WarpReduce`, `WarpScan`,
`WarpMergeSort`, `WarpBitonicSort`, `WarpBitonicTopK`, `BlockReduce`, `BlockRadixSort`,
`block_topk`, …). These are invoked from user-written kernels, so their quality is judged by
three different axes that a *device-level* (whole-GPU) benchmark cannot capture:

* **(A) Throughput** — the primitive is used in a kernel that saturates the GPU; we care about
  aggregate elements/second.
* **(B) Latency** — the primitive is on a critical path (few warps resident, or a dependent
  chain); we care about the time of a *single* invocation.
* **(C) Resource usage** — registers / shared memory / spills, which determine achievable
  occupancy and therefore the performance of the *surrounding* kernel.

We already have the seed of a device-side harness
(`nvbench_helper/nvbench_helper/device_side_benchmark.cuh`, introduced in #6431 and extended by
#8391 and #8607), but it has correctness and robustness gaps (documented below with data). This
issue specifies a consolidated, principled framework and asks for **three exemplary benchmarks**
implemented on top of it:

1. **`WarpBitonicSort`** — with an apples-to-apples comparison against **`WarpMergeSort`**.
2. **`WarpBitonicTopK`** (warp top-k).
3. **`block_topk`** (block top-k).

Reference prototypes for every claim below live in `dsb_experiments/` on the `exp/device-side-perf`
branch (`proto_latency.cu`, `proto_resource.cu`, `proto_throughput.cu`). All numbers were measured
on a single **NVIDIA B200** (148 SMs, 64 warps/SM, 65536 regs/SM), CUDA 13.1, `-arch=sm_100`.

---

## 1. Motivation: what is wrong with the current device-side benchmarks

`device_side_benchmark.cuh` currently offers two unrelated kernel styles:

* **Style 1** (used by `reduce/warp_reduce_*`): a statically **unrolled** loop
  `data = action(data)` (dependency chain), grid sized to a **compiler-chosen occupancy wave**
  (`cudaOccupancyMaxActiveBlocksPerMultiprocessor * numSMs`), and **no element normalization**
  (`state.add_element_count` is never called; the benchmark reports *raw kernel time*).
* **Style 2** (used by `bitonic_sort/*`): a **rolled** loop (`#pragma unroll 1`) that generates
  fresh random data *inside the timed region* and calls the primitive, with a `Latency`/`Throughput`
  mode enum (single warp vs. one occupancy wave), element-normalized in throughput mode.

Concrete problems, each backed by a measurement:

### P1 — Throughput workload depends on compiler-chosen occupancy (not regression-safe)
`warp_reduce_base.cuh` sets `grid = maxActiveBlocksPerSM * numSMs` and reports raw time. The
*amount of work* therefore changes whenever the compiler’s register allocation changes the
occupancy, so a “regression” can be pure occupancy drift, and an actual regression can be masked
by an occupancy increase. This is precisely the fragility that motivated #8607.

### P2 — “Latency” mode actually measures partially-pipelined throughput
Style 2’s latency mode uses independent iterations (fresh data each loop). Independent iterations
**pipeline**: the hardware overlaps the tail of one call with the head of the next, so the reported
“latency” is *below* the true single-call critical path. Measured with `clock64()` on one warp
(prototype `proto_latency.cu`, cycles per call):

| len (float) | true critical-path (dependency chain) | current-style (independent iters) | **underestimate** |
|---|---|---|---|
| 64  | 915  | 922  | ~1%  |
| 128 | 2258 | 2157 | ~4.5% |
| 256 | ~6900 | 4617 | **~33%** |

The error grows with primitive size (more independent instructions → more overlap). A latency
benchmark must **serialize** calls (see §3B).

### P3 — Data generation is inside the timed region (Style 2)
Generating random inputs each iteration adds LCG + `%clock` work to the measured time. For small
primitives this is a non-trivial fraction of the signal; it also makes the measurement clock-rate
dependent.

### P4 — No resource-usage regime at all
Neither style reports registers/shared-memory/spills or occupancy, even though for device-side
primitives this is often *the* deciding factor (a primitive that costs 20 extra registers can halve
the occupancy — and the throughput — of the user’s kernel).

### P5 — Two incompatible kernel/mode conventions
The `Mode` enum and `run_bench` helper are duplicated (copied) between `bitonic_common.cuh` and
`bitonic_sort/warp_topk_pairs.cu`. There is no single, reusable device-side harness.

---

## 2. Findings from the prototypes (evidence base for the design)

All reproducible; passes were bit-identical run-to-run for the deterministic metrics.

**Latency via `clock64()` on a single warp is essentially noise-free.** The dependency-chain and
independent variants returned identical cycle counts across repeated passes (0% run-to-run drift),
versus ~0.1–0.4 % noise for nvbench wall-clock timing of the same kernel. Cycles are also
**clock-rate invariant**, removing GPU-clock/throttle sensitivity from the metric.

**A dependency chain is required, and it must respect data-obliviousness.**
* Data-**oblivious** primitives (bitonic sort/top-k, reduce, scan) do identical work regardless of
  input, so an in-place “sort→sort→…” chain is a valid, fair critical-path latency and needs no
  per-call input generation.
* Data-**dependent** primitives (merge sort, radix `block_topk`) do *less* work on already-sorted
  or degenerate input, so a naive in-place chain measures the best case. They need **varied input
  per call**; the input generation then unavoidably enters the timed region and the measurement
  becomes a “back-to-back random-input” latency rather than a pure critical path. The framework must
  make this distinction explicit (§3B).

**Explicitly unrolling *independent* calls does not create ILP.** Cycling through 4–8 independent
register arrays did **not** reduce per-call latency (it was equal or worse due to register
pressure): the compiler emits each inlined primitive body sequentially and will not interleave them.
So cross-call overlap only appears opportunistically in *rolled* loops with a cheap loop-carried
scalar — which is exactly the trap P2 falls into. Conclusion: use an explicit **dependency chain**
to *guarantee* serialization; don’t rely on unrolling shape.

**The exemplary comparison the framework should make easy** (float, single warp, `clock64`
cyc/call, random input):

| len | WarpBitonicSort | WarpMergeSort | speedup | Bitonic smem | Merge smem |
|---|---|---|---|---|---|
| 64  | 922  | 3436 | **3.7×** | 0 B | 260 B |
| 128 | 2157 | 4587 | **2.1×** | 0 B | 516 B |
| 256 | 4617 | 6497 | **1.4×** | 0 B | 1028 B |

`WarpBitonicSort` is both faster and shared-memory-free — a decision a device-level benchmark could
never surface.

**Resource usage is cleanly obtainable at runtime** via `cudaFuncGetAttributes`
(`numRegs`, `sharedSizeBytes`, `localSizeBytes` = spills) and
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` — no `cuobjdump`/`nvcc -Xptxas -v` post-processing
required. Measured footprints:
* `WarpBitonicSort<float>` standalone: 14–25 regs, **0 smem, 0 spill**, full occupancy at block=256.
* `block_topk<float>`: 27–31 regs, **2064–8208 B smem** (scales with block×IPT); full occupancy on
  B200 but smem-limited on smaller GPUs / when the user kernel also needs smem.
* **Carrier delta** (a realistic kernel with the primitive vs. a NoOp): a *stable* +5 registers
  attributable to `WarpBitonicSort` — a better “what will it cost me” number than the raw
  benchmark-kernel register count (which is inflated by the harness).

**Throughput with a fixed workload + element normalization is stable and portable** (prototype
`proto_throughput.cu`): 223 / 196 / 160 G elem/s for len 64 / 128 / 256, reproducible to 3
decimals, and consistent with the existing nvbench `bitonic_sort` throughput bench (176 G elem/s @
0.35% noise for len=128). Because the result is elements/second over a *fixed* element count, it is
invariant to occupancy drift (fixes P1).

---

## 3. Proposed framework

Consolidate everything into one reusable header, e.g.
`nvbench_helper/nvbench_helper/device_side_benchmark.cuh` (extend the existing file; remove the
duplicated `Mode`/`run_bench` from the bitonic benches). Provide **one entry point per regime**,
each taking a user *action functor* (the primitive-under-test) plus compile-time shape parameters.

### Common building blocks
* `sink(...)` — keep results live via an unprovable `get_sreg_smid() == -1` guard (already present).
* An **action functor** convention: `operator()(KeyT (&keys)[IPT] [, ValueT (&vals)[IPT]] , args...)`
  that performs exactly one invocation of the primitive on register-resident data. (This is what the
  bitonic benches already do with `full_op_t` / `partial_op_t`.)
* A **NoOp action** with the identical signature, for the resource baseline (§3C).

### (A) Throughput harness  `run_throughput<Action, KeyT, ValueT, IPT>(state, axes...)`
* **Fixed workload**: choose the total number of warp-/block-invocations from a compile-time/runtime
  constant (e.g. `2^26` warp-sorts), *independent* of occupancy. Reach it with an inner `outer` loop
  count so the grid can still be one occupancy wave for realism, but the *work* is pinned. (This is
  the #8607 idea — `grid = fixed_elems / block` — combined with Style-2 element normalization.)
* **Element-normalized**: always `state.add_element_count(total_elements)` and report `Elem/s`.
* Data-oblivious primitives: generate inputs *once* per thread outside the hot path and re-sort with
  a modest `UNROLL`; do **not** regenerate per call (avoids P3). Data-dependent primitives: load
  varied inputs from a pre-generated global buffer (as `bitonic_sort/warp_topk_pairs.cu::iterator`
  already does) so generation is not timed.
* **`LaunchBoundsMode` axis** `{partial, full}` (from #8607): `partial` = `__launch_bounds__(block)`
  (primitive dominates, low occupancy); `full` = `__launch_bounds__(block, maxBlocksForFullOcc)`
  (primitive is register-squeezed for full occupancy). This is the throughput-under-pressure signal.

### (B) Latency harness  `run_latency<Action, KeyT, ValueT, IPT>(state, axes...)`
* **Single warp** (`grid=1, block=32`) or single block for block-level primitives.
* **Serialize with a dependency chain** of `CHAIN` statically-unrolled calls where call *n* consumes
  call *n-1*’s output, so no cross-call overlap is possible (fixes P2).
  * *Data-oblivious* primitives: in-place chain, inputs generated once (no P3).
  * *Data-dependent* primitives: feed varied input per call from registers/global and accept the
    “back-to-back random-input latency” interpretation; **document which mode a given bench uses.**
* **Measure in cycles with `clock64()`** around the chain, report `cycles/call` as an nvbench
  summary (clock-invariant, ~0 noise). Keep the nvbench wall-clock number too for continuity.
* **Acceptance gate (SASS):** the implementation MUST verify with `cuobjdump -sass` that the chain’s
  calls are serialized (the output registers of call *n* feed call *n+1*; no interleaving/ILP across
  calls). Provide the exact `cuobjdump` command and the reasoning in a comment. (`proto_latency.cu`
  shows the DEP vs. independent gap that this gate protects against.)

### (C) Resource-usage harness  `report_resources<Action, ...>(state, ...)`  (nvbench summaries)
Report all of the following as nvbench summaries so they are tracked over time:
1. **Standalone footprint** — `cudaFuncGetAttributes` on a canonical primitive-only kernel:
   `regs`, `static_smem`, `spill_bytes`. (Ballpark, as requested.)
2. **Carrier delta** — a user-supplied *carrier* kernel instantiated with the real action and with
   the NoOp action; report `Δregs`, `Δsmem`. This attributes cost to the primitive inside a
   representative kernel (answers the user’s “embed and compare before/after” idea). The carrier is a
   template so it can be reused/overridden per primitive.
3. **Occupancy** — `cudaOccupancyMaxActiveBlocksPerMultiprocessor` → achieved warps/SM and % of peak
   for the benchmarked block size.
4. **Pressure sweep / register budget** — instantiate the primitive kernel under
   `__launch_bounds__(block, minBlocks)` for increasing `minBlocks` and report the **largest
   occupancy at which it still compiles with 0 spills** (a single “occupancy-friendliness” number),
   plus flag any `spill_bytes > 0`.
5. **(bonus regime) Occupancy-limited performance** — reuse the throughput/latency harness under the
   `full` launch-bounds mode to report perf *at forced occupancy*, directly connecting resource cost
   to the performance a user would actually get. This is the most decision-relevant resource metric
   and generalizes #8607’s `partial`/`full` split.

> Note on ballpark-ness: register counts are allocated holistically by the compiler, so (1) will
> differ from what a real kernel sees; (2) and (5) exist precisely to bound that uncertainty from
> both sides (isolated footprint vs. footprint-in-context vs. footprint-under-occupancy-constraint).

---

## 4. Exemplary benchmarks to implement

Directory: `cub/benchmarks/bench/` (targets auto-register as `cub.bench.<path>.<file>.base`).

### 4.1 `bench/warp_sort/` — `WarpBitonicSort` vs `WarpMergeSort`
* Same axes for both so results are directly comparable: `KeyT ∈ {int16,int32,float,int64}`,
  keys-only and pairs, `len ∈ {32,64,96,128,160,192,224,256}`, `mode ∈ {throughput, latency}`,
  `LaunchBoundsMode ∈ {partial, full}` (throughput only).
* `WarpBitonicSort` action is data-oblivious → latency uses the in-place dependency chain.
* `WarpMergeSort` action is data-dependent → latency uses per-call varied input (documented).
* Emit the resource summaries (§3C) for both; the win is the head-to-head table in §2.

### 4.2 `bench/warp_topk/` — `WarpBitonicTopK`
* Largely already present (`bitonic_sort/warp_topk_pairs.cu`); **port it onto the shared harness**
  (drop the duplicated `Mode`/`run_topk`), add the `clock64` latency metric and the §3C resource
  summaries. Axes: `KeyT`, `ValueT` (keys-only + pairs), `len`, `max_k`, `k`, `mode`.

### 4.3 `bench/block_topk/` — `block_topk`
* Block-level, data-dependent, **uses shared memory** — exercises the smem/occupancy parts of §3C.
* Axes: `KeyT`, `ValueT`, `BlockDim ∈ {128,256,512}`, `ItemsPerThread`, `k`, `mode`,
  `LaunchBoundsMode`. Latency = single block with per-call varied input; throughput = fixed
  workload of block-tiles, element-normalized.

---

## 5. Deliverables / acceptance criteria

* [ ] A single reusable device-side harness header with the three entry points (§3), replacing the
      duplicated `Mode`/`run_bench`/`run_topk` code.
* [ ] The three benchmark groups (§4), each exposing `throughput`, `latency` (cycles/call summary),
      and resource summaries.
* [ ] `WarpBitonicSort` vs `WarpMergeSort` comparison reproduces the §2 ranking (bitonic faster and
      smem-free) on the CI reference GPU.
* [ ] SASS acceptance gate for every latency bench: documented `cuobjdump -sass` check proving no
      cross-call ILP for the dependency-chain (oblivious) benches; documented input-variation
      rationale for the data-dependent ones.
* [ ] Run-to-run noise: `cycles/call` summary noise < 1%; throughput `Elem/s` noise < 2% with the
      recommended flags (`--stopping-criterion entropy --throttle-threshold 90 …`).
* [ ] Docs: a short `docs/` section describing the three regimes and how to read the summaries.

---

## 6. Open questions / challenges

1. **Data-dependent latency semantics.** For merge sort / radix top-k there is no single “latency”;
   we measure back-to-back random-input latency. Should we also add a *worst-case* input axis
   (e.g. reverse-sorted, all-equal) to bound the range? (Recommended: at least one adversarial
   pattern per data-dependent bench.)
2. **`clock64` vs `%globaltimer`.** `clock64` is per-SM cycle count (what we want for latency).
   Confirm it is not affected by clock gating on sm_100 during the tiny measured window (our
   measurements were stable, but worth a note).
3. **Cross-GPU comparability of cycles.** Cycles are clock-invariant but not micro-architecture
   invariant; the CI reference GPU must be pinned for regression thresholds.
4. **Launch-bounds pressure sweep is GPU-relative.** On B200 these light primitives never spill even
   at max occupancy; the “register budget” number is most useful on smaller GPUs or for heavier
   primitives. Report it, but thresholds should be per-arch.
5. **Carrier realism.** The carrier kernel’s surrounding work is synthetic; the Δregs is a proxy.
   Should the carrier be user-overridable per primitive so teams can plug in their real kernel shape?
6. **nvbench integration for cycles.** Cleanest is a custom summary (we compute cycles in-kernel and
   `state.add_summary(...)`); confirm this is the preferred nvbench pattern vs. a fully custom
   measurement.

---

## References
* #6431 — original `WarpReduce` device-side benchmarks (`device_side_benchmark.cuh`).
* #7692 / #8544 — `WarpReduceBatched` implementation + docs.
* #8607 — `WarpReduceBatched` benchmarks: fixed workload + `partial`/`full` launch-bounds modes
  (the throughput-robustness ideas adopted here).
* #8391 — `WarpBitonicSort` (+ Style-2 `Mode` latency/throughput harness).
* #9281 — `WarpBitonicTopK`.
* Prototypes: `dsb_experiments/proto_latency.cu`, `proto_resource.cu`, `proto_throughput.cu`
  (branch `exp/device-side-perf`).
