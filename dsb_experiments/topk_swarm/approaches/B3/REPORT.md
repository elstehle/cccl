# B3: compacting radix select at block scope

Optimizer opt-B3, wave 1, 2026-10-25. Seven measurement batches on B200 nodes (umbriel-b200-091,
-020, -047, sm_100, nvcc 13.4.92), 96 cells each (12 (N, k) points, bf16 and fp32, both unsorted
requirements, latency and throughput), every cell with its control in the same lock section and
a passing oracle. Cards `cards/B3_*_b200_20261025_*.json`, raw batches under
`state/results/opt-B3/`, code on branch `exp/topk-swarm/B3` (`approaches/B3/block_topk_compact.cuh`,
adapters `harness/adapters/B3_*.cuh`, cells `harness/cells_b3.json`).

## 1. What it is

Radix select in the style of `block_topk_air` (one 8-bit or 9-bit digit per pass, histogram by
fire-and-forget shared atomics, fused block scan and pivot choice, pass state in shared memory)
with two structural changes:

1. After the first pass the items strictly above the pivot bucket are final. Their indices are
   written to the output at once, and the pivot bucket's items (key and index) are compacted into
   shared memory in index order. The later passes sweep the compacted candidates, which each
   thread holds in registers, instead of the whole tile. When the pivot bucket does not fit the
   buffer (N/4 items), the primitive falls back to sweeping the register tile as AIR does.
2. Every output position comes from one block scan of per-thread class counts packed into one
   word (items above the pivot prefix in the high 16 bits, ties in the low 16 bits), in blocked
   order, never from atomic tickets. The compacted candidates and the ties at the k-th key are
   therefore in index order and `prefer_smaller_index` is the same code as `not_guaranteed`.

Keys are mapped to an unsigned word whose ascending order is the key's descending order (the
complemented twiddle), so every pass is a smallest-k selection with digits read as plain shifts.
-0.0 is folded into +0.0 before ranking (fp32: one FADD with +0.0, bf16: compare and select).

The final adapter `B3_tuned` picks the configuration by tile size: from 16 items per thread for
32-bit keys and 32 for 16-bit keys it compacts after a 9-bit first digit (sign plus the whole
exponent, 512 bins), with a padded candidate layout and predicated stores. Below that it uses 8-bit
digits and no compaction, which leaves AIR's passes plus the scan-based index emission.

## 2. Contract

Block scope, 256 threads, blocked input in registers, N = 256 x IPT with IPT 1 to 128
(N <= 32768, 16-bit count and index fields), k in [1, N), keys fp32 or bf16 (fp16 has traits),
int32 indices derived from the tile position. Output: k source indices in the caller's shared
array, unsorted (above-pivot items in index order first, then the candidates in index order).
Requirements: `not_guaranteed`, `prefer_smaller_index`, `prefer_larger_index` (rank reversed), all
the same cost. Temp storage: two histograms (256 or 512 bins), the scan storage, the pass state and
the candidate buffer, 3152 bytes at N 1024 fp32 and 69712 at N 32768 fp32 (B1: 4256 to 147488).
Sync: `__syncthreads` inside (1 init, 3 per pass, 2 for the compaction, 1 for the final scan),
assumes the caller separates consecutive calls that reuse the temp storage by a barrier. NaN: a
positive NaN orders above +inf and is selected first, a negative NaN below -inf, never produced by
the focus set. Full tiles only.

## 3. Where the time goes

Measured by differences between variants in the same lock section (the harness measures whole
calls, there is no in-kernel phase profile here).

- Pass 1 over the tile (twiddle, histogram atomics, scan, state) is common to B1 and B3 and is
  the bulk of every call at small k: B3 at N 8192 k 64 fp32 is 6083 cycles against B1's 6992,
  of which pass 1 is roughly 3.5k.
- The compaction sweep (two counts per item, one packed scan, two predicated stores per item) and
  the candidate loads cost about what one filtered AIR pass over the tile costs. Compaction pays
  only when it removes at least two later tile sweeps, which is why it wins from 16 items per
  thread for fp32 (passes 2 to 4 saved) and only from 32 for bf16 (one pass saved), and why it
  loses 10 to 18 percent at N 1024.
- The candidate loads were the single largest implementation cost of the baseline: a per-thread
  stride of N/1024 words gave 32-way bank conflicts at N 32768, 1.34x (bf16) and 1.33x (fp32) once
  padded (T31).
- The index emission is one block scan (about 300 cycles) plus one predicated store per selected
  item, independent of k. B1's ticket pair epilogue grows with k by 5k to 13k cycles between
  k 64 and k = N/16 (T29), which is the whole 1.3x to 2.0x at large k.
- The filtered histogram atomics of the later passes sit in reconvergence regions (BSSY, BRA,
  BSYNC per item) at 32 items per thread and above, about four instructions per item per pass,
  for B1 as much as for B3, and inline PTX predication does not change that (T32). The compaction
  shrinks the number of such items from IPT to about N/1024 per thread.
- Registers: B3 holds one 32-bit word per item (no index array, no memoized key copy) and still
  needs 255 registers at IPT 128 and 128 at IPT 64, because the three code paths (early exit,
  compaction, fallback) and the unrolled sweeps keep the tile live. Throughput at bf16 N 8192 and
  16384 loses 2 blocks per SM to the predicated asm form (103 and 128 registers).

## 4. Results against B1_reference (B200, same lock section, card B3_tuned_b200_20261025_2)

Latency ratio B1 over B3 (larger is better), bf16 and fp32:

| N | k | bf16 | fp32 | | N | k | bf16 | fp32 |
|---|---|---|---|---|---|---|---|---|
| 1024 | 16 | 1.00 | 0.94 | | 8192 | 64 | 1.05 | 1.15 |
| 1024 | 64 | 1.06 | 0.99 | | 8192 | 1024 | 1.75 | 1.87 |
| 2048 | 16 | 0.99 | 0.96 | | 16384 | 64 | 0.94 | 1.16 |
| 2048 | 1024 | 1.36 | 1.35 | | 16384 | 1024 | 1.61 | 1.79 |
| 4096 | 32 | 1.10 | 1.03 | | 32768 | 256 | 1.20 | 1.37 |
| 4096 | 256 | 1.33 | 1.33 | | 32768 | 2048 | 1.82 | 2.00 |

Throughput (Gelem/s ratio B3 over B1): 0.90x to 1.07x at small k below N 8192, 1.07x to 1.26x
at small k from N 8192 on for fp32, 0.83x and 0.91x at bf16 N 8192 and 16384 small k (occupancy),
1.16x to 1.76x at k >= N/16. `prefer_smaller_index` 0.997x to 1.003x of `not_guaranteed` everywhere.

Kill experiment of the idea card (N 8192 k 64, random, same session, threshold 1.3x): 1.051x
(bf16) and 1.149x (fp32) in latency, 0.833x and 1.068x in throughput. The idea as stated is
retired at N 8192 and its own clause holds: confined to N 32768 and fp32 (1.366x at N 32768
k 256 fp32) and to be merged into B1 as an option. The large-k wins belong to the emission.

## 5. Ideas tried, with verdicts (chain of batches, each against the previous variant)

| batch | variant | change | verdict |
|---|---|---|---|
| 1 | baseline | AIR pass, compaction into N/4 buffer, scan positions, three paths | compaction path 0.80x to 0.99x of B1 at small k, fallback path 1.12x to 1.70x at k = N/16 (card B3_baseline) |
| 2 | padded | candidate stride N/1024 + 1 words, predicated loads | 1.33x to 1.34x at N 32768, 1.05x to 1.29x at N 8192 to 16384, neutral below (card B3_padded) |
| 3 | straight | guarded memory ops as predicated inline PTX | 1.00x to 1.10x on compaction cells, 0.88x to 0.98x on bf16 fallback cells (registers), ptxas keeps branching around the atomics (card B3_straight, T32) |
| 4 | digit9 | 9-bit first digit | 1.15x to 1.28x at k = N/16 from N 8192 (compaction instead of fallback), 0.82x to 0.99x at N 1024 to 4096 (card B3_digit9, T30) |
| 5 | lean | plain C++ single-store ifs, select increments, no asm | rejected, 0.92x to 1.06x, registers up to 215, identical BSSY counts (card B3_lean) |
| 6 | tuned | size policy (no compaction and 8-bit digits below the threshold), live-bin scans | 1.13x to 1.23x at N 1024 to 2048, fp32 N 4096 0.875x without compaction so the threshold is 16 for fp32, live-bin scans 1.000x (card B3_tuned_1) |
| 7 | tuned | final against B1_reference | table above (card B3_tuned_2) |

Host model `approaches/B3/model.py` reproduces the pass, compaction and rank logic and passed
1584 oracle runs before the first GPU batch (three digit widths, eleven patterns, both key
types, both requirements, all three code paths). It also produced the pivot bucket statistics
that explain the 8-bit digit's binade pairing (T30).

## 6. Ideas not yet tried

1. Compaction after an N6 first pass (T25): the fixed-point first digit leaves 1 to 7 candidates
   on random normal input instead of 14 percent of N, so the compaction sweep would be followed
   by passes over one item per thread at every N, and the compaction would pay from IPT 4 on.
   The scan emission already fits N6's structure. This is the combination to build in wave 2.
2. Striped candidate layout with a block-uniform slot count: thread t holds compacted items t,
   t + 256, ..., and the later passes loop over ceil(c1/256) slots (a uniform branch out of the
   unrolled loop) instead of N/1024 predicated slots. Loads are conflict free without padding.
   Index order for the tie ranks then needs the warp-redundant rank of knowledge base 4.5
   approach C (one ballot and popcount per slot and warp, a (slot, warp) table, one barrier,
   bases from a warp scan over the table), which also replaces the final block scan. Expected
   5 to 10 percent at N 16384 and 32768, nothing below.
3. Unconditional histogram atomics into a per-thread dummy word for the filtered passes (T10,
   T32) to remove the reconvergence regions, estimated a wash at IPT 64, cheap to measure.
4. The predicated asm store form costs bf16 two blocks per SM at 32 and 64 items per thread, the
   C++ form with compaction at those sizes was not measured (batch 2 had 64 registers there).
5. Packed bf16 compaction sweep with `__hgt2_mask` (T18) classifying two keys per instruction and
   the candidate word already packed (key16 << 16 | index), halving the sweep's compares.
6. A 10-bit first digit (pivot bucket 10 percent of N on the host model) with four bins per
   thread in the scan.
7. 128 and 512 threads, partial tiles, fp16 and fp8 keys (the primitive is generic, untested).

## 7. Determinism and ordering support

Set deterministic and index deterministic by construction: positions come from block scans in
blocked order, the compacted candidates keep index order, and the ties at the k-th key are ranked
in index order by the same packed scan in the final emission (`prefer_smaller_index`), or by the
reversed rank (`prefer_larger_index`). Measured cost of `prefer_smaller_index`: 0.997x to 1.003x of
`not_guaranteed` in all 48 cell pairs of every batch (same code path). Sorted output is not
supported inherently (post-sort of k). Output positions are deterministic as well (above-pivot
items in index order, then the candidates in index order).

## 8. Open items

- The final evaluation should compare B3_tuned against B1_idx (T21), not the pairs reference: the
  gap at large k then shrinks to the ticket serialization of the index-only ticket epilogue.
- bf16 throughput at N 8192 and 16384 small k (0.83x, 0.91x) is a register tier problem of the
  asm store form, see idea 4.
- The host model and the GPU agree on which path each cell takes, but the GPU numbers at N 16384
  and above carry the few percent of control subtraction tolerance the B1 card describes.
- `k_by_n` cells only: the card's N 8192 and 2048 points are outside the focus grid, the focus
  cells N 1024, 4096, 16384, 32768 are covered at two k each.

## 9. Pointers

- Primitive: `approaches/B3/block_topk_compact.cuh`. Adapters: `harness/adapters/B3_baseline.cuh`,
  `B3_padded.cuh`, `B3_straight.cuh`, `B3_digit9.cuh`, `B3_lean.cuh`, `B3_tuned.cuh`.
- Host model and tools: `approaches/B3/model.py` (`--stats` for the pivot bucket table),
  `approaches/B3/summarize.py` (one batch), `approaches/B3/crosstab.py` (the iteration chain).
- Findings files: `approaches/B3/findings_*.json`. Catalogue entries: T29 to T32
  (`approaches/B3/techniques/`).
- Cards: `cards/B3_baseline_b200_20261025_1.json`, `B3_padded_...`, `B3_straight_...`,
  `B3_digit9_...`, `B3_lean_...`, `B3_tuned_b200_20261025_1.json` (against digit9),
  `B3_tuned_b200_20261025_2.json` (against B1_reference).
- Raw batches: `state/results/opt-B3/*.json` (labels base1, pad2, str3, dig4, lean5, tun6, fin7).
