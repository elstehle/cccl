# [CUB] Latency-optimized block-wide Top-K — implementation exploration spec

## 0. What this document is

This is a **problem spec for an AI agent**, not a design. It fixes the *target* and the *rules of
engagement*, and deliberately leaves the *algorithm* open. The agent is expected to work from first
principles: research, microbenchmark the relevant hardware primitives, prototype several genuinely
different approaches, measure them rigorously, and recommend one.

Companion document: `DEVICE_SIDE_BENCHMARKING_ISSUE.md` in this directory defines *how* to measure
device-side latency/throughput/resources correctly (single-block `clock64` timing, dependency-chain
+ chain-length **slope** method, timer fencing, spill detection). **Reuse that methodology here** —
do not reinvent the measurement harness, and do not trust a naive "read clock / do work / read clock".

## 1. Target (fixed)

Implement a **block-cooperative top-K** device function with this exact shape:

* **Block size:** 256 threads (8 warps).
* **Input:** `N = 1024` elements per block (⇒ 4 items/thread), starting in registers or shared
  memory (the agent decides the residency; loading from global is out of scope for the latency
  measurement — inputs are already on-chip).
* **K:** 16 (select the 16 **largest**).
* **Output:** the 16 largest **values and their source indices** (arg-top-k), written to shared
  memory or registers. Keys-only is an acceptable secondary variant.
* **Primary dtype:** `float` (32-bit). Note this is the *harder* case for the packed-reduction trick
  (see §5); `__half`/`__nv_bfloat16` (16-bit) is a valuable secondary target and may admit a much
  cheaper implementation — call this out if so.
* **Semantics (relaxed — exploit these):**
  * **Non-deterministic ties:** when values are equal, *any* valid choice is correct. No stable /
    index-deterministic tie-break required.
  * **Unsorted output:** the 16 results may be in any order.
  * Correctness = the output is exactly a valid top-16 **set**: the multiset of output values equals
    the multiset of the 16 largest input values, and each output index points to an input position
    holding that value.
* **Optimize for:** **latency** on **B200 (sm_100)** — the primitive sits on a critical path with
  few resident warps. Report throughput and resource usage too, but rank by latency.

The two relaxations are not incidental — they *enable* faster designs (no ordering, no
tie determinism). The agent should actively use them (e.g., a plain `atomicMax` reduction is only
correct-enough *because* ties are non-deterministic).

## 2. Rules of engagement (how to work)

1. **Research & microbenchmark first.** Before committing to an approach, measure the latency (in
   cycles, single-warp/single-block, via the §-companion methodology) of the candidate building
   blocks on B200: `redux.sync.max.u32`, warp-shuffle reduction, `atomicMax`/`atomicAdd` to shared
   memory (and how it scales with contention: 1 word vs. banked words), `__syncthreads()`, LDS/STS.
   These numbers drive the design.
2. **Reason about the critical path and interleaving.** Identify the dependency structure of each
   approach (what is serial vs. parallel), and whether independent work can be interleaved to hide
   latency. Example tension to resolve: iterative max-extraction is **O(K) serial rounds** (each
   round removes the previous winner), so its latency ≈ `K × round_latency` — for K=16 that is 16
   dependent reductions. Radix-select and sorting networks instead have depth independent of K.
   Which wins at (N=1024, K=16) on B200 is an empirical question — answer it with data.
3. **Prototype several approaches** (§4), standalone (like the `proto_*.cu` files here: compile with
   `nvcc -std=c++17 -arch=sm_100 -O3 -I<cub> -I<libcudacxx/include> -I<thrust> --extended-lambda`).
4. **Measure rigorously** with the companion methodology; **validate correctness** against a
   reference (§6).
5. **Iterate**, then recommend.

## 3. Deliverables

* **A few (≥3) working prototype implementations** of block top-K covering *genuinely different*
  algorithmic strategies (not parameter tweaks of one idea).
* For **each** prototype: latency (cycles/call, slope-method), throughput, registers/shared-mem/
  spills, and a correctness pass on all data patterns (§6).
* A **comparison table** + a short **recommendation** with the reasoning (why the winner is fastest
  for this specific (256, 1024, 16, float, B200) point, and how sensitive that is to dtype/K/N).
* The building-block **microbenchmark numbers** that informed the design.

## 4. Candidate approaches to explore (starting points, not a menu to pick one from)

The agent should treat these as seeds and is encouraged to invent/combine. At minimum, explore the
redux-based approach (§5), the atomics-based approach and the redux/atomics synthesis (§6), and one
sort/network-based baseline.

* **(A) Iterative packed-reduction ("redux") extraction** — §5.
* **(B) Radix-select** — CUB already has this as `cub::detail::block_topk` /
  `block_topk_air` (`cub/cub/block/specializations/block_topk_air.cuh`): MSB→LSB radix-digit
  histograms (atomic counters in shared memory) narrow to the bucket containing the K-th key, then
  scatter via atomic counters. Depth is O(key_bits / radix_bits) *independent of K*. **Use it as a
  baseline** (it is the existing "atomics-based" design) and as the thing to beat for latency at K=16.
* **(C) redux ↔ atomics synthesis** — §7 (the main question posed below).
* **(D) Sort / network baseline** — partial bitonic top-K (`cub::detail::WarpBitonicTopK`,
  block-wide bitonic, or `WarpBitonicSort` per warp + merge). Data-oblivious ⇒ fixed depth, no K-serial
  dependency; good latency-comparison anchor.
* **(E) Hierarchical hybrids** — e.g. each of the 8 warps computes its top-16 of its 128 elements
  independently (no block sync), producing 8×16 = 128 candidates, then a single final top-16 over
  those 128. Trades one expensive block-wide phase for cheap parallel warp-local phases + one merge.

## 5. Summary of the redux-based warp approach (from the attached MoE routing kernel)

The reference (TensorRT-LLM `RoutingKernelTopK`, a warp-wide arg-top-k) works as follows:

* **Pack (value, index) into one comparable integer** (`TopKRedType`). The float value's bits are
  order-preserving-twiddled (`cub::Traits<T>::TwiddleIn`, the same monotonic bit mapping radix sort
  uses) and placed in the **high** bits; the index (as `maxIdx − idx`) goes in the **low** 16 bits.
  Result: a single unsigned integer whose natural `max` yields the largest value, breaking ties
  toward the smallest index. `TypeCmp` is `uint64` for 32-bit values, `uint32` for 16-bit values.
* **Reduce with one instruction where possible.** The warp-wide max of the packed word is the core
  op. For **32-bit** packed words (i.e. 16-bit dtypes) on sm_100+ it uses the hardware
  `redux.sync.max.u32` — a *single* warp reduction instruction. For **64-bit** packed words (i.e.
  `float`) `redux` does not apply, so it falls back to a shuffle-based `cg::reduce` (≈5 shuffles).
  **This is the key dtype cliff: the trick is ~free for 16-bit, but costs a shuffle tree for float.**
* **Iterate K times to extract the top-K.** Each round: reduce to get the current global-max packed
  word; unpack it into `out[kk]/outIdx[kk]`; then **remove** that element from the candidate pool so
  the next round finds the next-largest. Removal is a branchless per-thread shift: the owning thread
  advances its local cursor (its matched slot is replaced by `−inf`, the rest shift down).
* **Many items per thread** (`N`>4): sort each group of 4 with a tiny sorting network, process in
  chunks of 4, funnel per-chunk partial top-Ks into a small buffer, then a final `reduceTopK` over
  the buffer. So the structure is *local sort/partial-topk → hierarchical merge → iterative extract*.
* **Cost shape:** O(K) **serial** reduction rounds (round *n*+1 depends on round *n*'s removal) plus
  O(N) local work. For K=16 that is 16 dependent reductions on the critical path.

Notes for adapting it to *our* target: (i) the reference is **warp-wide** (one warp / row); we need
**block-wide** (256 threads cooperating on one 1024-element top-16), so a cross-warp combine is
required. (ii) Our **non-deterministic-tie** relaxation means the `maxIdx − idx` tie-break bits are
**not needed** — the agent can reclaim those bits or simplify packing. (iii) The **O(K)-serial**
critical path is the thing to attack for latency.

## 6. The atomics-based approach, and the redux ↔ atomics question

Two "atomics-based" families to consider, plus the synthesis the user specifically wants explored:

* **Radix-select with atomic histograms (existing `block_topk_air`).** Atomics are used to build
  per-digit histograms and to scatter survivors. Depth independent of K. This is the incumbent.
* **Atomic max/threshold reductions.** Maintain state in shared memory updated with atomics.

**The key question: can the atomics-based approach perform the same *logic* as the redux approach —
and can the two be combined?** Yes, and the agent should prototype it:

* The redux approach's essential primitive is a **block-wide max of packed (value,index) words**. That
  reduction can be implemented directly with **`atomicMax` on a single shared packed word** that all
  256 threads update — a commutative/associative reduction that needs **no `__syncthreads` barrier
  and no shuffle tree**. Because our ties are **non-deterministic**, a plain `atomicMax` is
  already correct (we don't care which tied element wins) — the relaxation is exactly what makes this
  clean. Then iterate K=16 times, removing the winner each round (broadcast the winning packed word;
  the owner invalidates its slot). This replicates redux's iterative-extract logic at **block scope**
  via atomics instead of shuffle+sync.
* **Combining the two** is the likely sweet spot: do the fast intra-warp reduction with
  `redux`/shuffle (8 warp-local maxes), then combine the 8 partials into the block result with a
  handful of `atomicMax`es to shared memory (contention 8, not 256) — atomics replace the awkward
  cross-warp `__syncthreads`+shared-tree combine. Evaluate contention scaling (1 shared word vs.
  several banked words reduced at the end).
* Also worth trying: use atomics/radix to **cheaply narrow** 1024→a small candidate set (e.g. a
  threshold pass), then finish with redux iterative extraction on the survivors — i.e. radix for the
  K-independent narrowing, redux for the final exact selection.

Open sub-questions for the agent to answer empirically: does `atomicMax`-to-shared beat a
shuffle+sync reduction at block scope on B200? At what contention does it stop scaling? Does removing
`__syncthreads` from the K-round loop (via atomics) shorten the critical path enough to matter?

## 7. Data generation & benchmark reuse

Reuse CUB's segmented-topk data generators — they already model the distributions that matter for
top-K behavior. See `cub/benchmarks/bench/segmented_topk/variable/common.cuh::gen_data` and its
patterns: `random` (normal), `quantized_random`, `relu_quantized`, **`tie_heavy`** (many equal keys),
**`pivot_tie`** (bimodal 1.0/2.0). The tie-heavy / pivot patterns are essential here: they stress the
tie handling that our non-deterministic relaxation depends on, and they are where a buggy "unique
winner" assumption would break. (The segmented benchmark uses large K; here K=16 — reuse the
*patterns*, not its K axis.) The `nvbench_helper` `generate(elements, entropy)` path is also
available for quick random inputs.

## 8. Correctness validation

For each prototype and each data pattern: compute a reference top-16 on the host (or with
`thrust::sort`) and compare **as a set** — sort the 16 output values and the 16 reference values and
require equality (values); require each output index to point to an input equal to its output value.
Do **not** require matching order or matching tie choices. Include the adversarial `tie_heavy` /
`pivot_tie` inputs (e.g. >16 elements equal to the pivot) to confirm the relaxed semantics hold.

## 9. Evaluation setup

* Node **`umb-b200-250`** (same container setup as the `dsb_experiments/` work: repo at
  `/cccl_fork/cccl`, branch `exp/device-side-perf`, cmake at `/cccl/cmake/cmake-4.3.2-linux-x86_64`,
  CUDA 13.1, `cuobjdump`/`ncu` available). Sync prototypes via git (push here, pull in the container)
  or `scp` + `docker cp`.
* Build standalone prototypes with:
  `nvcc -std=c++17 -arch=sm_100 -O3 -I/cccl_fork/cccl/cub -I/cccl_fork/cccl/libcudacxx/include -I/cccl_fork/cccl/thrust --extended-lambda proto_X.cu -o proto_X`
* Lock nothing for cycle measurements (`clock64` is clock-invariant); for wall-clock/throughput use
  the nvbench flags from the companion doc.

## 10. Hints / things to watch (do not treat as constraints)

* **dtype cliff:** float (64-bit packed) loses the single-instruction `redux`; 16-bit keeps it.
  Consider whether a float value can be packed with its index into 32 bits with acceptable precision
  loss (not for exact top-k), or whether the value alone (32-bit) is reduced first and the index
  resolved second.
* **K-serial critical path:** the O(K) dependent rounds are the main latency risk for iterative
  designs; sorting-network / radix depth is K-independent. Interleaving multiple extractions per
  round (e.g. warp-local partial sorts producing many candidates at once) can cut the serial depth.
* **`__syncthreads` count:** each block-wide barrier is a fixed latency add on the critical path;
  minimize them (atomics or warp-specialization can remove per-round barriers).
* **8 warps, 4 items/thread:** favors warp-local work + a cheap 8-way cross-warp combine.
* Use the companion doc's **slope method** so per-call latency is free of timer/fill/drain/barrier
  overhead, and **single-block bracketing** (`__syncthreads(); t0; …; __syncthreads(); t1;`) for the
  block-level timing.

## References
* Attached reference: TensorRT-LLM `RoutingKernelTopK` (warp redux arg-top-k) — summarized in §5.
* `cub/cub/block/specializations/block_topk_air.cuh` — existing block radix-select (§6 baseline).
* `cub/benchmarks/bench/segmented_topk/variable/common.cuh` — data-generation patterns (§7).
* `dsb_experiments/DEVICE_SIDE_BENCHMARKING_ISSUE.md` + `proto_*.cu` — measurement methodology to reuse.
* `redux.sync` PTX; `cub::Traits<T>::TwiddleIn/Out` (order-preserving float↔uint bit mapping).
