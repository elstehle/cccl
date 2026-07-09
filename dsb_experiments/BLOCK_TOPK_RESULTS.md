# Block top-K (256 threads, N=1024, K=16, float, B200) — exploration results & recommendation

> **Addendum (follow-up project):** `BLOCK_TOPK_AIR_OPT_RESULTS.md` optimizes the radix-select
> baseline itself (`block_topk_air`) to 1857 cyc random / 437 G elem/s, overtaking
> `atomic_adaptive` on random/sorted inputs and throughput. See its §3 for the reconciled
> per-workload guidance; the tie-flood and 16-bit conclusions below stand.

Companion to `BLOCK_TOPK_LATENCY_SPEC.md`. All numbers measured on **NVIDIA B200 (sm_100)**,
CUDA 13.1, `-O3 -arch=sm_100`, node `umb-b200-250`. Latency = **chain-length slope** (marginal
cycles/call, per `DEVICE_SIDE_BENCHMARKING_ISSUE.md` §3B) with single-block `__syncthreads`
bracketing and a data-dependent chain link (`v[0] = fmaf(prev_out, 0, v0[0])`, ~50 cyc/call,
identical across all prototypes). Code: `proto_topk.cu` (9 prototypes + 4 diagnostics),
`proto_topk_micro.cu` (building blocks). Every prototype passes the §8 set-semantics validation
on all 8 data patterns × seeds (28 runs each), including `pivot_tie40` (40 tied 2.0s > K) and
`all_equal`.

## TL;DR — recommendation

**Use `atomic_adaptive`** for float: iterative extraction where each of the 16 rounds is one
fire-and-forget `atomicMax` of the thread's current (twiddled u32) candidate **value** into
`slot[round]` + one `__syncthreads` + a branchless predicated shift-down — no indices, no
bookkeeping, no winner branch in the loop. Indices are resolved once at the end by a value-based
scatter, with an adaptive three-way epilogue (fast path / early-flood path / post-hoc-boundary
path). It **strictly dominates the incumbent radix-select (`block_topk_air`) on every pattern**:

| pattern      | atomic_adaptive | air (incumbent) | speedup |
|---|---|---|---|
| random       | **2331** cyc | 2499 | 1.07× |
| sorted_asc   | **2331** cyc | 2437 | 1.05× |
| tie_heavy    | **1415** cyc | 2316 | 1.64× |
| pivot_tie40  | **1996** cyc | 4372 | 2.19× |
| throughput   | **397 G elem/s** | 330 | 1.20× |
| smem         | **280 B** | 4240 B | 15× less |
| regs / spills| 38 / 0 | 31 / 0 | — |

For **16-bit dtypes** the same skeleton gets radically simpler and faster: pack
`(twiddled half ≪ 16) | index` into one u32 → every round is a *full arg-max*, the slots array
itself is the output, zero tie machinery: **`atomic_iter_h` = 1511 cyc flat on every pattern**
(only 219 cyc over the bare round skeleton) and 533 G elem/s at 19 registers. This is the
16-bit cliff of §5 inverted: on B200 the winning primitive is not `redux.sync` but the
MIO-combined shared `atomicMax`; the u32-packability of 16-bit keys is what removes *all*
extraction logic.

## 1. Building-block microbenchmarks (`proto_topk_micro.cu`, slope cyc/op)

Warp scope (1 warp): `redux.sync.max.u32` **22.4**; single `shfl_xor` 30; 5-shfl max tree u32
**150** / u64 **190**; `ballot_sync` 23; ballot+ffs+elect 71; `__syncwarp` 1; STS+LDS 33.

Shared atomics: `atomicMax` u32 uncontended round-trip **115** (u64 167); contention 2/8/32 →
117/129/92; **256 threads on one word: 255** (u32) — i.e. ~1 op/cycle aggregate; 256 threads
across 32 banks **32**. `__syncthreads` (256 thr) **28**.

Composed block-wide max, per round (the §6/§7 question, 256 threads):

| method | u32 | u64 |
|---|---|---|
| classic: warp shfl-tree → smem → barrier → 8-way combine | 290 | 368 |
| **atomic256: all threads `atomicMax` one slot → barrier → read** | **74** | 4519 (!) |
| hybrid8: warp shfl-tree → 8× lane0 atomicMax → barrier | 275 | 698 |
| hybrid8 with `redux.sync` instead of shfl-tree | 148 | — |

Three load-bearing facts fall out:

1. **Fire-and-forget u32 shared atomics with same-address bursts are nearly free** (SASS:
   `ATOMS.MAX RZ, …` — no return wait; the MIO combines the burst). A complete block-wide
   max round costs ~74–81 cyc, of which barrier (28) + LDS readback (33) are ~75%.
2. **The 64-bit packed word is dead at block scope**: u64 same-slot bursts collapse (4519
   cyc/round), and even the shuffle path pays 368/round. For float, reduce the **32-bit value
   alone** and resolve indices separately — legal precisely because ties are non-deterministic.
3. **Pre-reducing with `redux.sync` before the atomic makes rounds *slower*** (148 vs 74): it
   inserts a 22-cyc dependent op in front of work the MIO combiner does for free. The §7
   synthesis question is answered: atomics don't just match the redux logic, they replace it;
   the profitable combination is *atomics for the reduction + registers for the sorted
   per-thread streams*, not redux+atomics.

Also: a warp-scope extraction round (`redux.sync` + `__ballot` + elect) costs **82 cyc** —
*no cheaper* than a block-scope atomic round (81), because redux→ballot serialize. Hence
hierarchical (per-warp then merge) designs, which need 2×16 rounds, cannot beat one flat
16-round block phase. Measured: `hier_extract` = 5024 cyc ≈ 2× its own 2625-cyc skeleton.

## 2. Prototype comparison (slope cyc/call; throughput; resources)

| prototype | random | tie_heavy | pivot_tie40 | sorted | G elem/s | regs | smem B | spill |
|---|---|---|---|---|---|---|---|---|
| **atomic_adaptive** (recommended) | **2331** | **1415** | **1996** | **2331** | **397** | 38 | 280 | 0 |
| air = `block_topk_air` (baseline B) | 2499 | 2316 | 4372 | 2437 | 330 | 31 | 4240 | 0 |
| hist_narrow (tuned radix, 1-warp scan) | 2831 | 2381 | 4659 | 3160 | 341 | 48 | 2452 | 0 |
| atomic_iter (in-loop count + break) | 4398 | **815** | **1215** | 4419 | 142 | 29 | 204 | 0 |
| hybrid redux+atomic (§7 synthesis, C) | 5345 | 1001 | 1410 | 5331 | 126 | 28 | 204 | 0 |
| hier_extract (E: warp redux ×16 → merge) | 5024 | 5023 | 5023 | 5024 | 263 | 32 | 1152 | 0 |
| bitonic_hier (D: WarpBitonicTopK ×2) | 5112 | 5112 | 5112 | 5112 | 140 | 32 | 1152 | 0 |
| redux_iter (A: pure shfl+smem, u64) | 7906 | 7862 | 7998 | 8024 | 98 | 32 | 256 | 0 |
| — | | | | | | | | |
| **atomic_iter_h (`__half`, packed u32)** | **1511** | **1511** | **1511** | **1511** | **533** | 19 | 192 | 0 |
| DIAG: bare 16-round block-atomic skeleton | 1292 | — | — | — | 713 | 10 | 192 | 0 |
| DIAG: rounds + prologue (float, no epilogue) | 1443 | — | — | — | 534 | 17 | 192 | 0 |
| DIAG: 2×16 warp redux round skeleton | 2625 | — | — | — | 255 | 13 | 132 | 0 |

Occupancy: all fit ≥5 blocks/SM at 256 threads (adaptive: 6, limited by 38 regs); none spill
(`-Xptxas -v`: 0 spill stores/loads on all latency kernels).

## 3. The recommended design (`atomic_adaptive`)

Prologue (~150 cyc): twiddle 4 floats to order-preserving u32 (`TwiddleIn` bit trick), sort the
4-element working copy descending with a 5-comparator network; reset 16 slots + counters; 1 barrier.

Rounds (16 × ~90 cyc): each thread posts its current candidate `s[0]` with
`atomicMax(&slot[r], s[0])` (compiles to `ATOMS.MAX RZ` — fire-and-forget), `__syncthreads()`,
reads `M = slot[r]`, and does a **branchless predicated shift-down** (`w = s[0]==M; s[i] = w ?
s[i+1] : s[i]`). No indices, no ballots, no divergent winner branch, no in-loop count. All
instances of a tied max leave their lists in the same round, so round maxima strictly decrease
and each round's slot value is the next distinct value in descending order.

Epilogue (value-based index resolution — the key idea: *membership in the top-16 is a pure
value predicate*, `> M*` certain, `== M*` boundary):
* **Early flood check** (after round 2, ~156 cyc always): count items ≥ `slot[1]`; if ≥ K the
  boundary is already inside {slot[0], slot[1]} → resolve over 2 slots and skip 14 rounds.
  This is what turns tie floods from 3725 cyc into 1415 (`tie_heavy`) / 1996 (`pivot_tie40`).
* **Fast path** (common): count items ≥ `slot[15]`; if exactly K, those items *are* the top-16 —
  one atomic ticket per selected item, done.
* **Flood path** (boundary multiplicity): participating items (≥ `slot[15]`, warp-uniform skip
  elsewhere) vote into a 16-bin per-round histogram; a 16-lane prefix scan finds the round where
  the cumulative count crosses K → boundary value M* *and* tier-A size nA in one shot; a single
  scatter pass packs certain items at 0..nA-1 and boundary ties from nA up, dropping the surplus
  (any tie subset is valid under the relaxed semantics).

Cost accounting (measured by ablation): skeleton 1292 → +prologue/shift 1443 → +check/count/fast
scatter 2331. The two relaxations in the spec are both load-bearing: unordered output lets slots
double as the result, and non-deterministic ties are what make value-only reduction + arbitrary
tie subsets legal.

## 4. Why the alternatives lose (at this design point)

* **redux_iter (A)** — 16 serial rounds × 368-cyc classic u64 block max ⇒ ~7900. The K-serial
  chain is affordable only because atomic rounds are 4.5× cheaper than shuffle+smem rounds.
* **bitonic_hier (D)** — data-oblivious and flat (5112 everywhere) but two chained
  `WarpBitonicTopK<32>` passes cost ~2500 each; the network depth (sort32 + 3× merge per 128
  keys) simply exceeds 16 cheap rounds. Its niche is zero-smem warp-scope problems.
* **hier_extract (E)** — killed by the 82-cyc warp round (redux→ballot dependency): two 16-round
  phases ≈ 5024. Hierarchy only pays if the per-warp phase is ≪ the block phase; it isn't.
* **hybrid (C, redux + atomicMax)** — strictly worse than pure atomics (5345 vs 4398): the warp
  pre-reduction adds latency and saves traffic the MIO combiner already eliminates.
* **atomic_iter (in-loop count + break)** — the flood specialist (815/1215!) because it exits
  after ~2-3 rounds, but the per-round count LDS + dependent branch + second atomic burst
  triples round cost (~185 extra/round ⇒ 4398 on random). `atomic_adaptive` is precisely this
  trade resolved: branchless rounds + one early check.
* **hist_narrow / air (B)** — radix selection has K-independent depth but pays ~1200-1400/pass
  (histogram + find-bucket + barriers) and 2 passes minimum on float; its latency is also the
  most data-dependent (4372/4659 when ties force 4 passes). air remains the right *throughput/
  generality* design (any K, any dtype, valid_items), but at K=16 latency it loses everywhere.

## 5. Sensitivity (dtype / K / N)

* **dtype**: 16-bit keys (half/bf16) → use `atomic_iter_h`: packed u32 arg-max rounds, 1511 cyc
  flat, no epilogue machinery at all. 32-bit float → `atomic_adaptive`. 64-bit keys: value-only
  u64 rounds are unusable (4519/round burst collapse) — radix-select (air) likely wins there.
* **K**: rounds scale linearly (~90 cyc/round marginal, measured skeleton slope), epilogue is
  K-independent: adaptive(K) ≈ 900 + 90·K. air is roughly K-flat (~2500). Crossover ≈ K ≈ 18-24;
  at K ≤ 16 iterative wins, K ≥ 32 radix wins. K > 16 also needs >16 slots (still trivial smem).
* **N (at fixed block)**: rounds are N-independent; N enters via items-per-thread (local sort +
  shift list). IPT=4 is the sweet spot; IPT ≥ 8 needs the TRT-LLM funnel (sort chunks of 4,
  merge partial top-Ks) before the rounds — prologue grows ~linearly, rounds unchanged.
* **Block size**: the round cost is barrier-dominated (28 of ~90); larger blocks raise barrier
  latency moderately; the atomic burst is combining-limited, not thread-count-limited.

## 6. Correctness & methodology notes

* Set-semantics validation: sorted output values == sorted reference top-16 (multiset), all 16
  indices distinct and `in[idx] == value`; half variant validated against the half-rounded
  reference. Patterns: random ×8 seeds, quantized_random ×8, relu_quantized, tie_heavy,
  pivot_tie4/40 (random placement), all_equal, sorted_asc — 28 runs/prototype, all PASS.
* Latency: slope over chains {1,2,4,8,16}, min of 24 reps, R² checked ≈ 1; data-dependent
  chaining (per the companion doc, "back-to-back varied input" interpretation); block-level
  `__syncthreads` bracketing; consume-fence before the end clock.
* SASS gate (`cuobjdump -sass`, `lat_kernel<ProtoAtomicAdaptive>`): chain serialized through
  `FFMA` on the previous output feeding the twiddle; rounds are `ATOMS.MAX RZ` (fire-and-forget);
  ticket atomics keep returns; no `STL/LDL`.
* Throughput: fixed workload (2048 blocks × 33 calls × 1024 elems), element-normalized, best of 5.
* Reproduce: `./run_remote.sh proto_topk_micro.cu` and `proto_topk.cu` +
  `./proto_topk [correct|lat|thr|res]` (node umb-b200-250, container `quizzical_keldysh`,
  repo `/cccl_fork/cccl`, branch `exp/device-side-perf`).

## 7. Productization notes (CUB)

* Natural home: a `block_topk` latency specialization alongside `block_topk_air`, selected for
  small K (≤ ~24) and 32-bit arithmetic keys; 16-bit keys route to the packed variant.
* Needs from CUB: `Traits<T>::TwiddleIn/Out` (exists), `Uninitialized` temp storage (~72 B:
  16 slots + 16 histogram bins + 6 counters), K as a compile-time bound for the slot array.
* The `k < tile_items` guard, `valid_items` handling, and custom decomposers would follow air's
  interface; the algorithm itself only assumes: arithmetic key twiddleable to u32, K ≤ warp
  count × something reasonable (slots in smem), non-deterministic ties acceptable (document!).
  For deterministic-tie or sorted-output requirements, fall back to air / bitonic.
