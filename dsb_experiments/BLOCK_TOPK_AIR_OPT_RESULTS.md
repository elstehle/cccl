# Optimizing `block_topk_air` — phase profile, SASS/ncu analysis, and prototypes

Follow-up project to `BLOCK_TOPK_RESULTS.md`. Same design point (BLOCK=256, N=1024, K=16,
float+index pairs), same methodology (slope latency, set-semantics validation incl. a new
`neg_zero` pattern for the −0.0 restoration path), measured on **NVIDIA B200 (sm_100)**,
CUDA 13.1 (node `umb-b200-240`, container `elastic_gauss`). Code: `proto_air_opt.cu`
(`correct|lat|prof|thr|res` modes). All variants pass 30/30 correctness runs.

## TL;DR

Two algorithm-preserving optimization layers take `block_topk_air` from **2499 → 1857 cyc**
(−26%) on random and improve **every** pattern and throughput (+33%), with zero spills:

| variant | random | tie_heavy | pivot_tie40 | sorted | G elem/s | regs | smem B |
|---|---|---|---|---|---|---|---|
| air_ref (header today) | 2499 | 2316 | 4372 | 2437 | 329 | 31 | 4240 |
| **air_fused (v2: barrier diet)** | **1938** | 1754 | 3420 | 1869 | 405 | 40 | 6368 |
| **air_pair (v6: v2 + packed state + pair epilogue)** | **1857** | **1618** | **3349** | **1789** | **437** | 40 | 8336 |
| air_wscan (v3: explicit warp scan) | 2062 | 1838 | 3491 | 1995 | 382 | 39 | 6324 |
| air_eager (v4: scatter in passes) | 2898 | 2099 | 4019 | 2799 | 295 | 48 | 6368 |
| air_wspec (v5: warp-spec + split barriers) | 2653 | 2116 | 4379 | 2862 | 293 | 57 | 6548 |
| air_reimpl (profiling proxy) | 2589 | 2405 | 4554 | 2523 | 338 | 32 | 5344 |

The wins are pure structure — same radix selection, same passes, same semantics, same results:

* **v2, per pass (5 barriers → 3):** unroll the pass loop (compile-time shifts/masks);
  double-buffer the histograms so the *init phase and its barrier vanish* (the pass-p+1 buffer
  is zeroed during pass p's histogram phase — it was last read a barrier earlier); fuse
  choose_bucket into the scan (`exclusive = inclusive − input`, crossing test on registers) so
  the *histogram writeback, the choose phase, and their barrier vanish*.
* **v6, cross-pass + epilogue:** pack the pass state `(bucket | candidates | selected)` into
  **one** 32-bit word — the per-pass state broadcast becomes 1 dependent shared load instead of
  3; scatter keys+values together as 8-byte pairs and gather once — the epilogue collapses from
  `[setup | scatter keys | gather keys | scatter values | gather values]` (4 barriers) to
  `[scatter pairs | gather pairs]` (2 barriers); scatter counters preset in the prologue with
  class-1 positions computed as `total_selected + zero_based_ticket`, so the setup phase and the
  post-loop barrier disappear. The pair exchange unions with the (then-dead) histograms, so
  TempStorage grows only 4240 → 8336 B.

Dynamic barriers per call (2-pass input): ~16 → ~9.

## 1. Where the time goes (in-kernel clock64 phase profile)

Faithful reimplementation (parity: within 3.6% of the header on every pattern, identical stage
and barrier structure), stamps at every barrier boundary, min over 24 reps × 4 chained calls:

| stage (random input, 2 passes) | cycles | share | notes |
|---|---|---|---|
| twiddle + −0.0 normalize | 54 | 2% | |
| p0: init histograms + bar | 30 | 1% | (p1's init costs 112 — the store waits on prior-phase reads) |
| p0: histogram (1024 atomics) + bar | 339 | 13% | |
| p0: BlockScan + writeback + bar | 375 | 15% | largest single stage |
| p0: choose bucket + bar | 86 | 3% | |
| p1: init / histogram / scan / choose | 112 / 409 / 375 / 89 | 38% | p1 histogram ≈ p0 despite ~17 surviving candidates (see below) |
| scatter setup + bar | 94 | 4% | |
| scatter (ticket atomics) + bar | 243 | 9% | |
| gather keys + bar | 217 | 8% | |
| scatter values + bar / gather values | 74 / 104 | 7% | pairs-only cost |
| **total** | **~2590** | | epilogue alone = 732 cyc = 28% |

Three findings that drove the variants:

1. **The histogram phase is *not* contention-bound.** `pivot_tie40` funnels all 1024 adds into
   2 bins yet costs the same 339 cyc as random (~30 bins); and pass 1 costs ~400 with only ~17
   filtered adds. The phase is bound by the *dependent chain into it* — 3 serial broadcast loads
   of the pass state, prefix update, then issue+drain — not by atomic throughput. Hence: pack
   the state into one word (v6) rather than optimize the atomics (a `match_any` warp-aggregation
   was analyzed and judged a wash: it trades ~200 cyc of MIO for ~200 cyc of ALU).
2. **Scan + writeback is the largest stage and is mostly ceremony.** The scan's product is one
   crossing bucket + its exclusive prefix; the 256-word writeback, the extra barrier, and the
   choose re-reads exist only to communicate what each thread already holds in registers after
   `InclusiveSum`. Fusing them (v2) removes ~180/pass. The scan itself (~300) is near the floor:
   any 256-bin prefix structure pays a ~5-step dependent shuffle chain (~150) + a barrier + a
   cross-warp fold, whichever way it is spelled (confirmed by v3: an explicit warp-scan
   replacement is *slower* by ~120).
3. **The epilogue is 28% of the call.** Keys and values take separate scatter+gather trips
   because the exchange buffer is unioned to save smem. Scattering (key,value) pairs (STS.64)
   halves the trips (v6) — and the exchange can still union with the histograms, which are dead
   by then, so the smem cost is only +4 KB, not +8 KB.

ncu corroborates: shared-memory bank conflicts are negligible (4–6 per call); warp stalls are
dominated by **short scoreboard** (2.1 stalls/issue-active — waiting on shared loads) with
barrier stalls second (1.18). The kernel is dependent-LDS-chain bound, exactly what v2/v6 attack.
SASS: no local memory (0 spill) in any variant; histogram adds are fire-and-forget `ATOMS.ADD RZ`.

## 2. Warp specialization and mbarriers: explored, and why they lose here

Per the project brief, v4/v5 explored overlap-based designs. Both are **negative results with
clear mechanics** (they pass all correctness runs — the machinery works; it just doesn't pay):

* **v4 (eager scatter, +960 cyc vs v2):** scatter items the moment they become certainly-selected,
  inside the next pass's histogram phase. This moves the scatter's *ticket-return latency*
  (~115 cyc `ATOMS.ADD` round-trip) from a single epilogue into **every pass boundary**, where
  the barrier converts the slowest thread's latency into whole-block time. The epilogue it
  saves (~250) is far cheaper than the 2×~400 it adds. Lesson: with barriers every ~300 cycles,
  latency you "hide" inside a phase is only hidden if it is *shorter than that phase's other
  work for the same thread* — tickets aren't.
* **v5 (split named barriers + specialized scan warp, +715 cyc vs v2):** warps 1–7 post their
  histogram atomics, `barrier.cta.arrive` without blocking, and eager-scatter while warp 0
  (`barrier.cta.sync`, i.e. waits for all histogram traffic) finds the k-th bucket alone via a
  conflict-free stride-9 single-warp scan. Two structural problems: (a) the only work available
  to overlap is the scatter, which v4 already showed is a net loss to relocate; (b) freeing 7
  warps requires a 1-warp scan, and a 1-warp serial scan of 256 bins is slower than the
  all-thread BlockScan it replaces (v3 measured that gap at ~120/pass even with all threads).
  The split-barrier pattern itself validated fine (correctness across all patterns) — it is the
  *absence of overlappable critical-path work* that makes it unprofitable, not the mechanism.
  mbarriers proper (`mbarrier.arrive/try_wait`) would add generation management on top of the
  same structure and were not pursued further.

The general lesson for this primitive: its critical path is a chain of
*histogram → scan → state broadcast* dependencies with no independent work anywhere — every
overlap trick just relocates latency onto a barrier that then waits for it.

## 3. Reconciliation with BLOCK_TOPK_RESULTS.md (the overall K=16 float picture)

Optimized air changes the project-1 leaderboard: **air_pair (1857) overtakes atomic_adaptive
(2331) on random and sorted inputs and on throughput (437 vs 397 G elem/s)**; atomic_adaptive
keeps the tie-flood crown (1415/1996 vs 1618/3349) and `atomic_iter_h` (1511 flat) remains the
16-bit winner. Updated guidance for the (256, 1024, 16) point:

| workload | best | cyc |
|---|---|---|
| float, random-like (common case) | **air_pair** | 1857 |
| float, tie-flood-heavy | atomic_adaptive | 1415–1996 |
| float, worst measured pattern | air_pair 3349 vs adaptive 4050* | — |
| 16-bit keys | atomic_iter_h | 1511 flat |
| K ≳ 24, or 64-bit keys | air_pair (K-independent depth) | — |

*adaptive's pivot_tie40 = 1996 with its early flood check; its no-flood worst case is ~4050.
air_pair's pivot40 = 3349 (4 passes). Neither dominates the other across all patterns; air_pair
has the better common case and the better tail-vs-K behavior, adaptive the better tie behavior.

## 4. Productization notes (changes to `block_topk_air.cuh`)

Ordered by value/effort; all preserve the algorithm, API, and semantics:

1. **Fuse scan+choose** (v2, header-ready): `InclusiveSum` already yields the inclusive sum;
   compute `exclusive = inclusive − input`, run the crossing test immediately, drop
   `compute_bin_offsets`' writeback and `choose_bucket` entirely. Removes 1 barrier + ~260 smem
   accesses per pass. No interface change.
2. **Double-buffered histogram init** (v2): `histogram[2][num_buckets]`; zero the other buffer
   inside the histogram phase. Removes 1 barrier per pass; +1 KB TempStorage (unions with the
   exchange stage, so often free).
3. **Packed pass state** (v6): one `uint32` `(bucket | candidates | selected)` instead of the
   3-field struct. (Bit budget at RadixBits=8, tile 1024: 8+11+5? use 8+12+12 in a u32 for
   k ≤ 4096, or simply two u32s if k is unconstrained — the win is one load, and even 2 packed
   words beat 3 dependent ones.)
4. **Pair exchange for the pairs variant** (v6): scatter `(KeyT, ValueT)` together, gather once;
   union with the histograms. Saves 2 barriers + one full item pass; TempStorage for
   float+int pairs grows 4240 → 8336 B. Keys-only select_keys keeps the current layout (no gain).
5. **Preset scatter counters + computed class-1 base** (v6): zero `cntA/cntB` in the prologue;
   class-1 position = `total_selected + ticket`. Removes the setup phase and post-loop barrier.
6. **Compile-time bit range** (v2): when `begin_bit`/`end_bit` are compile-time (the common
   instantiation), unroll the pass loop for immediate shifts/masks.

Costs to weigh: registers 31 → 40 (maxblk/SM 8 → 6 at 256 threads; irrelevant for
latency-critical single/few-block uses, and the throughput measurement *improved* 33% despite
it); pairs TempStorage +4 KB. A `latency_optimized` policy knob (or simply K/bits-compile-time
specialization) could gate items 4–6 if the footprint matters.

## 5. What was measured and how

* Parity gate: `air_reimpl` mirrors the header stage-for-stage and lands within 3.6% on every
  pattern — the profile attributes real header behavior, not an artifact.
* Latency: chain-length slope (chains 1..16, min of 24 reps, block-bracketed `clock64`).
* Correctness: set semantics on 9 patterns (incl. `pivot_tie40`, `all_equal`, and `neg_zero`
  which exercises the −0.0 flip-back path), 30 runs per variant, all PASS.
* Throughput: fixed workload, 2048 blocks × 33 calls, element-normalized, best of 5.
* SASS: `cuobjdump -sass` — 0 spill (`STL/LDL` absent), fire-and-forget histogram atomics.
* ncu: `smsp__average_warps_issue_stalled_{barrier,short_scoreboard}_per_issue_active`,
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}`.
* Reproduce: `./run_remote.sh proto_air_opt.cu` then `./proto_air_opt [correct|lat|prof|thr|res]`.
