# W1 bitonic network top-k, report of the second optimizer instance (opt-W1b)

W1 was run by two optimizer instances after a duplicate launch. opt-W1 (worktree `cccl-topk-swarm-W1`,
report `REPORT.md`) owns the packed key-index word through the CUB network (`W1_packed`), its
level-parallel form (`W1_level`) and the sub-warp reference. This instance (opt-W1b, worktree
`cccl-topk-swarm-W1b`, branch `exp/topk-swarm/W1b`) owns a disjoint slice: the N2 kill experiment on
identical network code and identical policies (eager against level-parallel, keys-only words and
pairs), the fp32 native min and max pairs route against the index-in-key u64 route with the cost of
`prefer_smaller_index` on each, the comparator route through the unchanged primitive, and the
sub-warp grid for these variants. Every number below is B200 (sm_100 build, umbriel-b200-020), CUDA
13.4.92, 2026-10-25, oracle battery passed before any number was read, with its control in the same
lock section. Batches are under `state/results/opt-W1b/`, cards under
`cards/W1_{level_same,level_pairs,eager_same,minmax_pairs,baseline,kvcmp,eager_same_sub,minmax_pairs_sub,level_same_sub}_b200_20261025_1.json`.

## 1. What it is

A register-resident bitonic network over a logical warp of LW lanes (4 to 32) with M registers per
tile (MaxK = M x LW, MaxK = max(LW, next_pow2(k))), striped layout (position i x LW + lane), one
shuffle per 32-bit word per stage, no shared memory, no barrier. The top-k loop over the T = N / MaxK
tiles is either eager (sort tile 0, then for each further tile sort it in reverse, keep the larger
per position, merge: the structure of PR #12222) or level-parallel (sort all tiles at once, halve
the tile count per level: idea card N2). The network is one template
(`approaches/W1/w1b_detail.cuh`, `w1b::net`) over an element policy:

- `policy_u32`: bf16 key twiddled into the high half, inverted 16-bit index in the low half, one
  word per element, integer order = (key descending, index ascending). A compare-exchange is one
  `ISETP.GT.U32` with the lane's direction folded into the predicate XOR input plus one `SEL`.
- `policy_u64`: fp32 key twiddled into the high word, inverted 32-bit index low, two shuffles per
  stage, `ISETP`, `ISETP.EX` and two `SEL`.
- `policy_pair_f32`, `policy_pair_bf16`: native (key, index) pairs, two shuffles per stage, key
  path `nk = keep_min ? fminf : fmaxf` (compiles to `FMNMX` plus a predicated duplicate), the index
  follows the key that changed (`FSETP.NEU`, `SEL`). Equal keys keep their position in both lanes.

Adapters: `W1_eager_same` and `W1_level_same` (packed words, both requirements), `W1_minmax_pairs`
and `W1_level_pairs` (pairs, not_guaranteed), `W1_baseline` (the unchanged CUB primitive: pairs for
not_guaranteed, a (key, index) struct with a short-circuit composite comparator for
prefer_smaller_index), `W1_kvcmp` (the same comparator branch-free), and `_sub` forms at sub-warp
scope (LW 4, 8, 16).

## 2. Contract

Input: N keys striped over the logical warp in registers, int32 source indices (the packed policies
fold them into the word). Output: the k selected source indices in the caller's array, sorted by key
descending (and index ascending for the packed policies). Requirements: the packed policies give
`prefer_smaller_index` for free (same instantiation, same cycles in all 52 warp and 76 sub-warp cell
pairs), the pairs policies give `not_guaranteed` only. Keys bf16 and fp32. `-0.0` is canonicalized
to `+0.0` in the packed policies and compares equal in the pairs policies. NaN: the packed policies
order a NaN by its sign bit (deterministic), the pairs policies let a NaN lose every exchange (never
selected). Limits: MaxK divides N, the level-parallel loop needs N / MaxK a power of two, bf16 packed
needs N <= 65536. Temp storage none, synchronization shuffles only, no stack, no spills in any cell. (Amendment after verification: two sub-warp latency cells, fp32 and bf16 LW4 N128 k64, spill 352 and 48 bytes at 168 registers, the numbers stand.)

## 3. Where the time goes

At IPT 1 a stage is shuffle latency plus the first dependent ALU operations: 35 cycles per stage for
fp32 min max pairs (531 cycles for the 15 stages of N 32), 38 for the packed u32 (574), 39 for the
reference (583), 40 for the packed u64 (601). The floor is the 30-cycle shuffle (T1, T17) plus one
or two 4 to 6 cycle ALU steps (T7). Below that only fewer dependent stages help, not a cheaper
exchange.

Above IPT 1 the eager loop's tile sorts are independent and ptxas overlaps them with the merge
chain: packed bf16 costs 574, 761, 1211, 2371 cycles at N 32, 64, 128, 256 (k <= 32), 190 to 1160
cycles per added 32-element tile against 574 for the first. The chain of T - 1 dependent 5-stage
merges is the critical path. The level-parallel loop does not shorten it enough to matter (4.1).

Instruction counts of the N 128 k 32 latency kernels (SASS, harness included): packed bf16 728 (150
SHFL), packed fp32 1176 (300 SHFL), min max pairs fp32 1136 (300 SHFL, 306 FMNMX), reference fp32
1152 (300 SHFL, 306 FSETP, 314 SEL), reference bf16 1656 (300 SHFL, 424 HSETP2, 176 PLOP3), CUB with
the short-circuit struct comparator 3448 (300 BSSY, 300 BSYNC, 305 BRA), branch-free 1840.

The schedule matters as much as the count: the same fp32 pairs network measures 1649 cycles at N 128
k 8 and k 16 and 2004 at k 32 (MaxK 32 in all three, the only source difference the epilogue
predicate), T28. Differences below 20 percent between network variants are not proven by one cell.

## 4. Ideas tried, with verdicts

### 4.1 N2 level-parallel restructuring (kill experiment, as the card states it)

Setup: `W1_level_same` against `W1_eager_same` (identical network code and policies, keys-only
words) and `W1_level_pairs` against `W1_minmax_pairs` (identical code, pairs), N 64, 128, 256 at k 32
and 64, bf16 and fp32, plus sub-warp LW 8 and 16 at N 64 and 128 (4 to 16 tiles).

Verdict at the kill cell N 64 k 32: killed. Keys-only words 0.96x (bf16 793 against 761) and 1.00x
(fp32 931 against 934), pairs 0.96x (bf16 1420 against 1369, fp32 753 against 720). At N = 2 MaxK the
two loops are the same dependency DAG, so no difference was possible there; from four tiles on:
packed words 0.95x to 1.08x at warp scope, 0.91x to 1.17x at sub-warp (the 1.17x is the single
16-tile fp32 cell), throughput within 3 to 8 percent, 2 to 30 more registers. Pairs gain 10 to 22
percent latency at N 128 to 256 (fp32 N 256 k 32 3739 against 4545) because that loop is
shuffle-issue bound, but keep all tiles live (67 against 35 registers at bf16 N 128) and lose the
gain in throughput (119 against 125 Gelem/s). The 1.3x threshold is reached nowhere. The shuffle
count is identical by construction, the lever was dependency depth only, and the eager loop's
independent tile sorts already give ptxas that overlap.

### 4.2 bf16 index-in-key (packed u32) against the reference

`W1_eager_same` bf16 against `W1_reference` (not_guaranteed, same cycles under
prefer_smaller_index): N 32 574 against 779 (1.36x), N 64 k 32 761 against 1164 (1.53x), N 128 k 16
1211 against 2267 (1.87x), N 256 k 32 2371 against 5640 (2.38x), N 256 k 64 2800 against 4658
(1.66x). Throughput 254 against 125 Gelem/s at N 32, 263 against 102 at N 256 (2.0x to 2.6x), flat
over N while the reference falls. Half the shuffles (150 against 300 per N 128 kernel), 728 against
1656 instructions, 21 to 43 registers against 26 to 43. Sub-warp (4.6) 1.0x to 2.1x latency and
1.8x to 2.8x throughput. This is the W1 answer for bf16 and the W1 answer to the tie-break
requirement: index-in-key, exact, at zero cost.

### 4.3 fp32 routes: native min max pairs against index-in-key u64

Three fp32 routes measured against each other in one lock section (not_guaranteed unless noted):

| cell (fp32, LW 32) | reference | min max pairs | u64 packed (ng = psi) | struct cmp branch-free (psi) |
|---|---|---|---|---|
| N 32 k 8 | 583 | 531 | 601 | 721 |
| N 64 k 32 | 925 | 720 | 934 | 1081 |
| N 128 k 16 | 2041 | 1649 | 1918 | 2120 |
| N 128 k 32 | 2041 | 2004 (T28) | 1917 | 2120 |
| N 128 k 64 | 2152 | 1217 | 2216 | 1821 |
| N 256 k 32 | 4606 | 4545 | 3826 | 4860 |
| N 256 k 64 | 5104 | 4547 | 4650 | 5902 |
| throughput N 128 k 32, Gelem/s | 179 | 183 | 167 | 101 |

Min max pairs win latency up to N 128 (1.13x to 1.30x against the u64 word, 1.82x at N 128 k 64) and
throughput everywhere (9 to 10 percent over u64, 1 to 5 percent over the reference); the u64 word
wins latency only at N 256 k <= 32 (1.12x to 1.19x over the pairs). The exchange `keep_min ? fminf :
fmaxf` compiles to FMNMX plus a predicated duplicate (the compiler does not use the predicate
operand of FMNMX), still the cheapest key path at 35 cycles per stage.

The price of `prefer_smaller_index` on fp32, against the best not_guaranteed route: 13 percent at N
32, 24 to 30 percent at N 64, 16 percent at N 128 k <= 16, 82 percent at N 128 k 64 (2216 against
1217), a gain of 11 to 16 percent at N 256 k <= 32, and 9 to 10 percent throughput everywhere. A
caller that does not need the tie-break should take the pairs at N <= 128.

### 4.4 prefer_smaller_index through the unchanged CUB primitive

A (key, index) struct as the key with a composite comparator written with `||` and `&&`
(`W1_baseline`): exact result, 4.4x to 6.4x the cycles of the not_guaranteed reference for fp32
(N 128 k 32: 12899 against 2041) and 1.8x to 2.4x for bf16, 0.4x to 0.6x the throughput, 3448
against 1152 instructions with a BSSY, BSYNC and BRA per comparator call. cicc keeps the short
circuit as control flow and ptxas wraps every exchange in a convergence region.

### 4.5 Branch-free composite comparator (T35)

The same comparator with `|` and `&` (`W1_kvcmp`): fp32 3.5x to 6.1x faster than 4.4 (N 128 k 32:
2120, no BSSY, BSYNC or BRA, 1840 instructions), bf16 1.3x to 1.6x (no branches in either bf16
form, the gain is the predicate chain, PLOP3 455 to 189). Against the not_guaranteed reference the
fixed struct route costs 1.03x to 1.24x latency for fp32 and 1.35x to 1.50x for bf16 and 30 to 45
percent throughput (6 against 8 blocks per SM, two compares per exchange). It is never the best
route: index-in-key is 1.2x (fp32) to 3.5x (bf16) faster for the same result. The reusable rule is
the catalogue entry: a network comparator must never short-circuit.

### 4.6 Sub-warp logical warps (LW 4, 8, 16, N 8 to 128, k 4 to 64)

Control `W1_baseline_sub`, the CUB pairs network at the same LW, not_guaranteed.

- bf16 one-word packed: LW 4 1.27x to 2.07x latency, LW 8 1.05x to 1.91x, LW 16 1.01x to 1.51x (one
  loss, N 128 k 64 0.89x), throughput 1.8x to 2.8x from N 16 on (LW 4 N 16 k 4: 735 against 411
  Gelem/s, the highest throughput of the study). prefer_smaller_index free in all 76 cells.
- fp32 u64 packed: a loss, 0.71x to 1.0x latency at LW 4 to 16 (two exceptions at N 8 and N 32 k 16),
  0.79x to 1.0x throughput, up to 167 registers against 128. At a small LW the network is mostly
  in-register exchanges, where the 64-bit compare costs and the one-shuffle advantage is gone.
- fp32 min max pairs: LW 4 1.17x to 2.24x latency (N 16 k 4 318 against 713, N 128 k 16 4240 against
  8580), LW 8 0.90x to 1.27x, LW 16 0.89x to 1.40x (N 64 and 128 at 0.89x to 0.93x), throughput
  0.85x to 1.04x. The sub-warp fp32 route is min max pairs at LW 4 and 8, the CUB pairs network at
  LW 16.
- bf16 pairs: dropped (0.64x to 0.97x from N 32 on at LW 8 and 16).
- Level-parallel at sub-warp: 0.91x to 1.17x, killed as at warp scope.

## 5. Ideas not yet tried

- The predicate operand of `FMNMX` and `IMNMX` (one instruction for `keep_min ? min : max`); no CUDA
  C or PTX form was found, a `selp` on the result costs what the predicated duplicate costs.
- A level-parallel loop that pairs adjacent registers so consumed tiles retire early (keep the pairs
  latency gain without the register growth).
- A one-word fp32 route for sub-warp N <= 128: key alone in 32 bits, index recovered after the
  network by a ballot match (smaller lane first gives the tie-break).
- bf16x2 (two keys per register, N5) for a keys-only network: with an index per key it has the
  shuffle count of index-in-key and no cheap tie-break.
- The buffered threshold-compaction variant of PR #9281 for N >= 8 k, and the sm_100f build
  (nothing in these networks uses it).

## 6. Determinism and ordering

Every variant is a fixed network: deterministic and sorted output for free. `prefer_smaller_index`:
free in the packed policies (identical instantiation and identical cycles in every measured pair of
cells), not supported by the pairs policies, 2x to 6x through the CUB primitive with a short-circuit
comparator (4.4), 1.03x to 1.5x with the branch-free one (4.5), and on fp32 the switch from the pairs
route to the u64 route costs 13 to 30 percent latency up to N 128 and 9 to 10 percent throughput
(4.3).

## 7. Open items

- The LW 16 fp32 min max loss (0.89x to 0.93x) and the N 128 k 32 pairs swing are schedule effects
  (T28); a rescheduled build (different epilogue, dummy dependence) would settle whether the min max
  form is the LW 16 route as well.
- T >= 32 tiles (N >= 1024 at warp scope) were not measured for the level-parallel form; the trend
  from T 2 to 16 does not suggest the threshold is reached before the registers run out.
- A 48-bit fp32 route (truncated key) changes the result and was not tried; N4 covers it.
- GB200 was not measured in this slice (B200 only, the shared harness has gb200 nodes).

## 8. Pointers

- Code: `approaches/W1/w1b_detail.cuh` (network, policies, packing, adapter body),
  `harness/adapters/W1_{eager_same,level_same,minmax_pairs,level_pairs,baseline,kvcmp}{,_sub}.cuh`.
- Node workflow `approaches/W1/w1b_node.sh`, SASS opcode histogram `approaches/W1/sass_hist.awk`.
- Findings files `approaches/W1/findings_*_b200.json`, catalogue drafts
  `approaches/W1/tech_schedule_fragility.md` (T28) and `tech_comparator_branches.md` (T35).
- Batches `state/results/opt-W1b/*.json` (12 measured, build JSONs beside them), node work
  directories `/work/topk_swarm/work/opt-W1b/` on umbriel-b200-020 (all binaries), -047 and -091
  (batch-2 and smoke builds only).
- Related: `approaches/W1/REFERENCE.md` (eval-engineer's reference numbers), `REPORT.md` (opt-W1).
