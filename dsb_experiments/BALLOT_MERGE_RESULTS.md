# Stable warp-striped merge: rank-augmented bitonic vs ballot-routed merge path vs smem co-rank

Can warp shuffles/ballots implement an efficient **stable** merge of two sorted runs held
warp-striped in registers (no shared memory, no blocked rearrangement)? Three designs, identical
contracts (striped in, striped-2 out, ties: run A first, within-run order preserved), measured on
**umbriel-b200-022, B200 (sm_100, 148 SMs)**; harness `proto_ballot_merge.cu` (standalone, no cub
includes). Runs of length S each, S ∈ {4,8,16,32} (one element per lane per run, 32/S segments
concurrently per hardware warp); float keys, pairs variant carries an int value. All methods pass
correctness + stability (exact match vs `std::merge` on identity values) across random,
heavy-ties, all-equal, and fully-skewed patterns.

## The three designs

* **mp_smem** — baseline: runs staged to shared memory; each output position binary-searches its
  merge-path crossing (co-rank), then picks its element. Stable via strict/non-strict predicate
  asymmetry. Represents the "existing approach" adapted to striped I/O.
* **rank_bitonic** — the bitonic merger made stable by **rank augmentation**: compare
  `(key, input_rank)` lexicographically (ranks 0..S−1 for A, S..2S−1 for B, static per slot).
  All augmented keys distinct ⇒ the oblivious network's unique output IS the stable merge.
  log2(2S) compare-exchange stages; key+rank (+value) shuffled per stage; first stage (distance S)
  is register-local.
* **ballot_route** — ballot-driven merge path: one warp ballot per level resolves the
  mid-diagonal crossings of ALL sub-problems at once (the monotone crossing predicate's
  candidates are distributed across lanes; every lane decodes every crossing from the uniform
  ballot mask with popc — orchestration is pure ALU). The crossing *count* enables stable
  routing: piece A stays in place, piece B shifts — order-preserving concatenation, so stability
  is structural, with no rank payload. 1 ballot + ~5 shuffles at level 1, 2 ballots + ~6 shuffles
  per level after (~46 collectives at S=32 vs rank_bitonic's ~21 shuffles).

## Latency (slope cyc per warp-call = 32/S concurrent segment merges, gen-subtracted)

| | S=4 | S=8 | S=16 | S=32 |
|---|---|---|---|---|
| keys mp_smem | 825 | 1051 | 1257 | 1516 |
| keys **rank_bitonic** | **164** | **237** | **283** | **341** |
| keys ballot_route | 376 | 521 | 662 | 909 |
| pairs mp_smem | 770 | 943 | 1153 | 1410 |
| pairs **rank_bitonic** | **170** | **247** | **456** | **518** |
| pairs ballot_route | 389 | 520 | 811 | 1199 |

## Throughput (one occupancy wave, Gelem/s)

| | S=4 | S=8 | S=16 | S=32 |
|---|---|---|---|---|
| keys mp_smem | 392 | 315 | 272 | 239 |
| keys **rank_bitonic** | **755** | **574** | **444** | **371** |
| keys ballot_route | 360 | 256 | 202 | 159 |
| pairs mp_smem | 342 | 295 | 250 | 237 |
| pairs **rank_bitonic** | **630** | **461** | **338** | **268** |
| pairs ballot_route | 325 | 226 | 174 | 142 |

## Reading

1. **Rank-augmented bitonic wins everything**: 4.4–5× lower latency than the smem baseline
   (keys), 2.7× lower than pairs at S=32, and 1.1–2× higher throughput everywhere. Stability via
   rank augmentation costs one extra shuffle channel and a two-term compare — far cheaper than
   any search-based alternative at these sizes. A stable 64-element merge in **341 cyc**
   (518 pairs) is ~4× faster than the lean smem co-rank baseline, and the real
   `WarpMergeSort`-style machinery it would replace is heavier still.
2. **The ballot-routed design is a principled negative** (documented as such): it beats the smem
   baseline on latency (~1.7–2.2×) — the ballot-bisection tree and ALU-only orchestration work
   exactly as designed, and all correctness/stability tests pass — but it is strictly dominated
   by rank_bitonic on both axes. The mechanism is the WarpRadixSort lesson again: warp
   collectives are issue-limited, and ~46 collectives per merge lose to ~21 regardless of how
   clever their arrangement is. In the throughput regime it even falls below the smem baseline
   (collective pipes saturate; smem has bandwidth to spare at these sizes).
3. **Pairs asymmetry**: the ballot design's value routing is relatively cheaper (+32% at S=32 vs
   rank_bitonic's +52%) — the predicted advantage exists but is far too small to overcome the
   2.3× base gap. Prediction registered before the run: wrong on the overall pairs winner.
4. Pairs ≈ keys at S ≤ 8 for rank_bitonic (170 vs 164): at small S the extra value shuffles hide
   under issue slots; the cost only materializes at S ≥ 16.

## Recommendation

* For a warp-scope **stable merge** (and by extension a stable small sort), **rank augmentation
  on the bitonic network is the design to productize** — e.g. a `WarpStableMerge<KeyT, ValueT>`,
  or a stable mode on `WarpBitonicSort` (sort `(key, static input rank)`). This directly attacks
  the main reason callers are pushed off the 2–3×-faster bitonic path onto `WarpMergeSort` in the
  small-tile regime: stability.
* Caveats to carry into productization: rank augmentation needs `2S` distinct ranks (≤10 bits at
  warp tiles — packable into spare key bits for integer keys, free for argsort-shaped workloads
  where values are the ranks); the two-term comparator invokes `compare_op` twice per exchange
  for arbitrary comparators (once for arithmetic keys); and the O(S log S) network will lose to
  O(S) merges at larger per-thread element counts — the IPT > 1 regime (multi-register runs) is
  unmeasured and is where the known bitonic/merge crossover (IPT ≈ 7–10) should reappear.
* The ballot-bisection idea is worth remembering where a *crossing count* (not a merge) is the
  actual product — e.g. computing merge-path partitions for a consumer that is not a
  compare-exchange network — but not as a merge engine.

Reproduce: `nvcc -std=c++17 -arch=sm_100 -O3 proto_ballot_merge.cu -o proto_ballot_merge &&
./proto_ballot_merge [correct|lat|thr|all]` (no cub includes required).
