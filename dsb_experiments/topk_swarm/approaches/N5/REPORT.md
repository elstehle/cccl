# N5: two bf16 rows per lane in bf16x2 words, one SIMD compare-exchange for both (throughput regime)

Optimizer: opt-N5. Branch `exp/topk-swarm/N5`, worktree `cccl-topk-swarm-N5`. Three measurement batches on
B200 umbriel-b200-018 (sm_100 build, nvcc 13.4.92, 2026-10-25, harness 0.5 with rows per group), every
oracle passed (13 battery cases per cell, both requirements), canaries 32.001 cycles per op and 3779 to
3781 us in every lock section. Batches under `state/results/opt-N5/`, cards `cards/N5_*_b200_20261025_*.json`.
Catalogue entry contributed: T53 (the bf16x2 exchange codegen forms). Adopted: T7, T17, T18, T28, T34, T36,
T37, T40, T42, T47, T52.

## 1. What it is

The W1 eager bitonic top-k network (opt-W1's template recursion, the eager tile loop of
`cub::detail::WarpBitonicTopK`, the min and max exchange of T42) run on 32-bit words that hold two bf16
keys, one per 16-bit half. Every stage of a sorting network compares position i with position j of the same
sequence, so the two halves never interact: `max.bf16x2` and `min.bf16x2` (one HMNMX2 each, T7) are the
compare-exchange for both halves at once and one shuffle moves both. The network is keys only. Indices are
recovered afterwards from the k-th key of every half: the keys strictly greater than it are all selected and
placed by one inclusive lane scan, the keys equal to it fill the remaining slots, in index order when the
requirement asks for it. The two halves are filled in two ways:

- `N5_rows`: two independent rows of N keys per call (the harness 0.5 `packed_rows` arrangement, row 0 in
  the low half of word j = item j LW + lane, row 1 in the high half). This is the card's primitive, the
  batched MoE router shape where every row of a layer shares N and k. Output: row r's k indices at
  `out_idx[r K, (r + 1) K)`.
- `N5_split`: one row of N keys split into its even and odd items (the `packed_striped` arrangement hands
  the bf16x2 words over as loaded), the network runs top-MaxK on both sets at once, then one more bitonic
  merge on one-key words (set B mirrored by a lane xor LW - 1 shuffle, position-wise max, log2(MaxK)
  stages) gives the row's top-k keys. A drop-in for the single-row one-word route, no API change.

Variants (policies in `approaches/N5/n5_detail.cuh`): exchange `ops_bf16` (the SIMD calls inside the
ternary, predicated HMNMX2 pair), `ops_sel` (both results then a select, HMNMX2 pair plus one predicated
move), `ops_pred` (predicate inside one asm block, worst); recovery `recover_exact` (greater keys by lane
scan, equal keys by one ballot per word and half in index order), `recover_fast` (both classes by the lane
scan, the exact ballot rank only when the tie class overflows the remaining slots under
prefer_smaller_index, a hardware-warp-uniform branch); network `net_w1` (W1's recursion) and `net_dir`
(the direction as one bit test on precomputed lane-bit parity words, `n5_net.cuh`, no effect). Adapters:
`harness/adapters/N5_rows.cuh` (bf16, exact), `N5_rows_f.cuh` (bf16, fast, the main line), `N5_rows_fs.cuh`
(sel, fast), `N5_rows_fd.cuh` (dir), the same four for `N5_split`, and `_sub` wrappers at sub-warp scope.
The first batch ran `N5_rows` and `N5_split` with `ops_sel` (commit 3ede7791e3), the second with `ops_bf16`
(30f5b8f821), the third the final commit 8f613a317f.

## 2. Contract

bf16 keys only (two per word, fp32 does not fit). Input in registers, `packed_rows` (two rows) or
`packed_striped` (one row), N = LW x IPT with MaxK = round_up(k, LW) dividing N (rows) or N / 2 (split) and
the tile count a power of two, k in [1, N / 2], logical warps 8, 16, 32 (4 compiles, not measured). Output: k
indices per row in shared memory, unsorted (greater keys in lane order, then the tie class), both unsorted
requirements with the same code, prefer_smaller_index exact by construction (the tie class is emitted in
index order whenever it is capped). Temp storage none. Sync: shuffles, ballots and one any-vote over the
full hardware warp, so the primitive requires the whole hardware warp in the call (the harness convention,
also T37, T40 and T52). -0.0 equals +0.0: the network orders -0.0 below +0.0 (max.bf16x2 semantics, T18) but
the recovery classifies by IEEE comparison, and every key the network ranks at or above position k - 1 is
either greater than or equal to the k-th key, so the selected set is the oracle's set in every case (the
signed_zero battery pattern passes under both requirements). NaN: never selected, a NaN loses every min and
max and both set instructions, so with fewer than k non-NaN keys output slots stay unwritten. Denormals are
compared as values. Registers: 30 to 32 at N 32 (occupancy 8 blocks of 256), 36 to 48 at N 64 to 128
(occupancy 5 to 6), 48 to 64 at N 256 (occupancy 4 to 5), no spills, no stack, no local memory in any cell.

## 3. Where the time goes

Throughput kernel at N 128 k 32, one call per warp, SASS instruction counts (the control is the T42
one-word min max network `W1x_vimnmx_eager`, 336 instructions for one row):

| part | one-word control | N5_rows, two rows (b2 build) | N5_split, one row |
|---|---|---|---|
| shuffle stages | 75 | 75 | 41 plus 1 mirror shuffle plus 5 merge stages |
| exchange instructions | 153 VIMNMX (2 per stage, predicated pair) | 153 HMNMX2 (2 per stage) | 82 HMNMX2 |
| direction predicates | 5 ISETP, 24 LOP3 (hoisted in P registers) | about 1 to 2 per stage (ISETP 67, LOP3 47) | ISETP 51, LOP3 30 |
| index path | in the word (free), 1 LOP3 plus 1 STS per output | 8 HSET2, scan 5 SHFL plus 15, emission about 20 per word | 4 HSET2, scan, emission |
| total | 336 | 528 (exact) or about 500 executed (fast) | 336 (exact), about 330 (fast) |

Per row the two-row kernel issues about 250 to 265 instructions against 336, 1.3x, and the measured
throughput gain is 1.28x to 1.33x at this cell. The pipe picture of T47 explains why the gain is not the 2x
of the shuffle and exchange count: min, max, LOP3 and HSET2 all issue at half rate and the stage is pipe
bound from two warps per scheduler, so the extra direction predicate per stage (a LOP3 into a predicate
register that the if-converted branch rebuilds every stage, where the u32 select form keeps the predicates
hoisted) and the recovery's HSET2 and LOP3 work land on the same pipe as the exchange. The index recovery is
the other half: about 20 to 25 instructions per packed word (two items) in the fast form, 90 to 100 per row
at N 128 and about 180 at N 256, a third of the per-row budget. The split variant pays the one-word final
merge (about 20 instructions on half-empty words) and its network covers only half a row per SIMD lane, so
it saves shuffles (48 against 75) but not exchange instructions (82 HMNMX2 against 153 VIMNMX for one row
is 1.9x fewer, then the recovery adds about 95), and lands at 1.03x to 1.09x at N 128 and 1.16x to 1.26x at
N 256 against the T42 control, 1.4x to 1.7x against the card's W1_eager_same.

Latency (chain slope, one warp): the two-row call costs 1.3x to 1.6x one one-word row (two rows in 774 to
2367 cycles against 512 to 1718), that is 0.65x to 0.8x per row, and the split call costs 1.1x to 1.5x the
one-word row below N 256 and 0.92x at N 256 k at most 32 (1570 against 1718: the shorter network wins once
the recovery is amortized). Neither is the latency primitive, as the card predicted.

## 4. Ideas tried, with verdicts

### 4.1 Exchange codegen (T53)

Three source forms of the same exchange, static SASS of a 32-stage chain (`approaches/N5/micro/
exchange_forms.cu`): the u32 reference 128 instructions (predicated VIMNMX pair), bf16x2 with the two asm
calls inside the ternary 144 (predicated HMNMX2 pair, plus about half an instruction per stage of predicate
rebuilding), bf16x2 with both results computed then selected 152 (HMNMX2 pair plus a predicated IMAD.MOV),
u16x2 through `__vmaxu2` and `__vminu2` the same 144 and 152, the predicate inside one asm block far worse
(424 against 352 in the N 128 k 32 split kernel). Negating the partner operand (`neg.bf16x2` or an xor of the
sign bits) is never folded into HMNMX2. Measured on the two-row kernel: select form 445 Gelem per s, ternary
form 458 at N 128 k 32 (control 358 in both sections, batches b1 and b2), at the price of 16 more registers
(32 to 48) and three fewer blocks per SM. Verdict: the ternary form, adopted in `ops_bf16`. At sub-warp
instantiations (LW 16, IPT 8) ptxas turns the same source into two predicated HMNMX2 into different
registers plus two predicated moves (four per stage), so the select form (`N5_rows_fs_sub`) is the sub-warp
A/B of batch 3.

### 4.2 Direction words (`net_dir`, N5_rows_fd)

Precomputing the lane-bit parities once per call so every stage direction is a single bit test did not
change the instruction count (600 against 592 at N 128 k 32, ISETP 90 in both): the extra predicate work is
the if-conversion's, not the direction arithmetic's. Not measured on the GPU, dropped.

### 4.3 Index recovery forms (batch 2, same session)

Exact (ballot rank for the tie class in every call) against fast (both classes by the lane scan, the tie
class capped only when it overflows the remaining slots, a uniform branch): rows N 128 k 32 458 against 477
Gelem per s (1.28x against 1.33x over the control), N 128 k 64 406 against 436, N 256 k 32 477 against 511,
N 256 k 64 417 against 452, N 32 395 against 393, N 64 407 against 405. The fast form wins 4 to 8 percent
from N 128 on and ties below, where the scan's fixed cost dominates. Both requirements run the same code
and measure the same throughput to 0.1 percent in every cell (prefer_smaller_index costs nothing, the
overflow branch is not taken on random input, the battery patterns that force it pass).

### 4.4 Two rows against the split of one row

Same session, same control: rows 1.24x to 1.33x at N 32 to 256 k at most 32 and 1.44x to 1.82x at k 64,
split 0.95x to 1.09x at N 64 to 128, 1.24x to 1.26x at N 256 and 1.54x to 1.66x at k 64 (fast forms, batch
2). The split's one-word final merge and half-filled SIMD words cost it 15 to 25 points. The two-row form is
the primitive the card asked for and the one that clears the thresholds with margin.

### 4.5 What the card's thresholds say

Card (and `ideas/PRIORITIES.md` item 7): 1.4x the one-word network's throughput at N 128 k 32, at least 370
Gelem per s, and pass W2 at k 32 (332 at N 256). Against the card's reference numbers (W1_eager_same, 268 at
N 128 k 32, re-measured at 268 in the same session as batch 1): N5_rows_f 477 is 1.78x, N5_split_f 387 is
1.44x, both above 370. Against the T42 min max one-word network that appeared after the card was written
(358 in session): 1.33x and 1.08x. W2 at N 256 k 32: 511 and 457 against 332, and the same-session
comparison with `W2_packed_v3` is in batch 3. The verdict against the card's letter is pass for both forms,
against the stronger one-word control the two-row form passes 1.3x and the split form does not reach 1.4x.

### 4.6 Sub-warp scope (batch 3)

LW 8 and 16, N 32 to 128, k 4 to 64, against `W1_eager_same_sub` (opt-W1b's one-word network at the same
logical warp), one lease, consecutive lock sections. Two-row form in the select exchange (`N5_rows_fs_sub`):
LW 8 1.23x to 1.72x and LW 16 1.37x to 1.93x in throughput (N 128: 713, 632, 554, 483 against 506, 413, 354,
281 at LW 8 and 608, 608, 522, 464 against 363, 336, 285, 240 at LW 16 for k 4, 16, 32, 64), and from N 64 on
the two-row call is also faster than the control's one-row call in latency (1.14x to 2.06x per call, so 2x
to 4x per row), the one scope where the card's "no latency gain" is wrong: the sub-warp control is a chain
of short in-register and shuffle stages whose count the SIMD word halves. The ternary exchange form
(`N5_rows_f_sub`, preceding section) is 5 to 6 percent slower at N 128 through the four-instruction phi
shape of section 4.1 and carries 14 more registers at LW 16. The split form (`N5_split_f_sub`) reaches 1.07x
to 1.61x in throughput and 1.16x to 2.63x in latency, 10 to 20 points below the two-row form. The card's
sub-warp must-beat (LW 8 N 128 k 64 282) is passed at 483 (rows) and 450 (split), and the W2 sub-warp card
values at LW 8 N 128 (485 at k 16, 147 at k 64) at 632 and 483.

### 4.7 The price of the requirement in the fast form

The exact form compiles the same kernel for both requirements and measured identical throughput to 0.1
percent (batches b1, b2). The fast form compiles different fallbacks (the index-ordered ballot rank for
prefer_smaller_index, the capped prefix for not_guaranteed) and although neither runs on random input the
prefer_smaller_index kernel carries 3 to 16 more registers and measures 2 to 4 percent lower throughput in
most cells (up to 9 percent at LW 8 N 128, and plus 8 percent latency at N 256 k at most 32), a T28 schedule
and register effect of code that is never executed. The not_guaranteed numbers are the ones in the tables,
the prefer_smaller_index numbers are in the batch JSONs and the cards carry both.

## 5. Ideas not yet tried

- T51's truncated final merge (stop at stride k, take the k-th key with a redux): for two bf16 rows the
  per-half minimum needs two integer reductions on twiddled halves (about 10 instructions) against the
  5 saved stages (about 18), a 2 to 3 percent gain at k 32 on the two-row form, more at k 64.
- A one-class "at least t" pass as in T52's one-ballot form, replacing the two HSET2 per word by one when the
  group holds exactly k keys at least t (always on random fp32, mostly on random bf16): saves one HSET2 and
  one count per word, 4 to 8 instructions per word.
- `__launch_bounds__` or `-maxrregcount 32` to restore 8 blocks per SM for the ternary form (T47 says the
  stage is pipe bound from 2 warps per scheduler, so the expected gain is small).
- Twiddled u16x2 keys (`VIMNMX.U16x2`, T47 same pipe and rate) with integer compares in the recovery, which
  removes the HSET2 but needs the -0.0 canonicalization at the input (about 6 instructions per word).
- fp32 keys: two fp32 keys do not fit one register, the fp32 analogue of this card is N10's keys-only network.
- The latency regime: no gain is expected and none was found, the card said so.

## 6. Determinism and ordering

Deterministic (fixed network, fixed recovery order). Output unsorted: the greater class in lane order, the
tie class in lane order (fast path) or index order (overflow and exact paths). Sorted output would cost the
index-in-key route's own emission (one shuffle per selected key to move the recovered index to the sorted
position) and is not provided. prefer_smaller_index is exact at zero cost (section 4.3).

## 7. Open items

- The direction predicate rebuilt per stage by the if-conversion (about 0.5 instruction per stage on the
  half-rate pipe) is the one codegen cost left between the bf16x2 exchange and the u32 reference shape. A
  ptxas peephole that folds `selp` of two `max.bf16x2`/`min.bf16x2` results into a predicated pair, as it
  does for `max.u32`/`min.u32`, would remove it (toolchain, not code).
- Register growth of the ternary form (32 to 48 at N 128) and the occupancy drop to 5 blocks per SM: a
  maxrregcount build would show whether any throughput hides there.
- Sub-warp codegen: the four-instruction phi form at LW 16 (section 4.1), and the harness's sub-warp kernels
  carry a `BRA.DIV` fast and slow copy of the whole call body plus a `WARPSYNC.COL` per shuffle on the slow
  copy (also in the W1 control), so sub-warp instruction counts are twice the executed path.
- GB200 not measured (B200 only, T16 says the SM costs transfer, T20 says the compare networks' host codegen
  difference does not apply to explicitly written exchanges).

## 8. Pointers

- Code: `approaches/N5/n5_detail.cuh` (SIMD primitives, exchange policies, recovery policies, both adapter
  bodies), `n5_net.cuh` (direction-word network copy), `micro/exchange_forms.cu` (codegen micro),
  `node.sh` (node workflow), `sass_hist.awk`, `harness/adapters/N5_*.cuh`.
- Catalogue: `tech_bf16x2_exchange.md` (T53 as appended).
- Batches (`state/results/opt-N5/`): b1 (ops_sel, commit 3ede7791e3): `N5_rows_vs_W1x_vimnmx_eager_focus_b1_20261025_153333.json`,
  `N5_split_vs_W1x_vimnmx_eager_focus_b1_20261025_153411.json`, `W1_eager_same_vs_W1x_vimnmx_eager_focus_b1_20261025_153444.json`;
  b2 (ops_bf16, 30f5b8f821): `N5_rows_vs_W1x_vimnmx_eager_focus_b2_20261025_154953.json`, `N5_rows_f_..._155030`,
  `N5_split_..._155108`, `N5_split_f_..._155140`; b3 (8f613a317f): `N5_rows_f_sub_vs_W1_eager_same_sub_focus_b3_20261025_160143.json`,
  `N5_rows_fs_sub_..._160248`, `N5_split_f_sub_..._160353`, `N5_rows_f_vs_W1x_vimnmx_eager_focus_b3_20261025_160452.json`,
  `N5_split_f_..._160530`, `N5_rows_f_vs_W2_packed_v3_focus_b3w2_20261025_160604.json`.
- Findings files `approaches/N5/findings_*.json`, cards `cards/N5_*_b200_20261025_*.json`.
- Related: `approaches/W1/CODEGEN_EXPERIMENT.md` (the control's codegen), `ideas/queue/N5_two_row_bf16x2.md`,
  `ideas/queue/N10_fp32_keys_only_index_recovery.md`, T51 and T52 (opt-N10's keys-only route and recovery).

## 9. Result tables

Generated from the batch JSONs by `state/opt-N5/tables.py` (throughput in Gelem per s, latency in cycles per
call, ratio is variant over control for throughput and control over variant for latency, registers and
occupancy of the variant's throughput kernel, not_guaranteed rows, prefer_smaller_index measured identical
in every cell).

### 9.1 Warp scope, final commit (batch b3)

N5_rows_f against W1x_vimnmx_eager (throughput, not_), batch N5_rows_f_vs_W1x_vimnmx_eager_focus_b3_20261025_160452
| cell | N5_rows_f | W1x_vimnmx_eager | ratio | registers | occupancy |
|---|---|---|---|---|---|
| N 32 k 4 | 387 | 270 | 1.43x | 29 | 8 |
| N 32 k 16 | 388 | 270 | 1.44x | 29 | 8 |
| N 64 k 4 | 401 | 328 | 1.22x | 38 | 6 |
| N 64 k 16 | 401 | 327 | 1.23x | 38 | 6 |
| N 64 k 32 | 401 | 331 | 1.21x | 38 | 6 |
| N 128 k 4 | 480 | 356 | 1.35x | 47 | 5 |
| N 128 k 16 | 480 | 356 | 1.35x | 47 | 5 |
| N 128 k 32 | 480 | 358 | 1.34x | 47 | 5 |
| N 128 k 64 | 434 | 302 | 1.43x | 40 | 6 |
| N 256 k 4 | 513 | 363 | 1.41x | 59 | 4 |
| N 256 k 16 | 512 | 363 | 1.41x | 59 | 4 |
| N 256 k 32 | 511 | 367 | 1.39x | 59 | 4 |
| N 256 k 64 | 452 | 248 | 1.82x | 48 | 5 |

N5_split_f against W1x_vimnmx_eager (throughput, not_), batch N5_split_f_vs_W1x_vimnmx_eager_focus_b3_20261025_160530
| cell | N5_split_f | W1x_vimnmx_eager | ratio | registers | occupancy |
|---|---|---|---|---|---|
| N 64 k 4 | 309 | 328 | 0.94x | 30 | 8 |
| N 64 k 16 | 309 | 328 | 0.94x | 30 | 8 |
| N 64 k 32 | 309 | 331 | 0.94x | 30 | 8 |
| N 128 k 4 | 384 | 356 | 1.08x | 37 | 6 |
| N 128 k 16 | 384 | 356 | 1.08x | 37 | 6 |
| N 128 k 32 | 384 | 358 | 1.07x | 37 | 6 |
| N 128 k 64 | 357 | 302 | 1.18x | 39 | 6 |
| N 256 k 4 | 457 | 363 | 1.26x | 40 | 6 |
| N 256 k 16 | 456 | 363 | 1.26x | 40 | 6 |
| N 256 k 32 | 455 | 367 | 1.24x | 40 | 6 |
| N 256 k 64 | 411 | 248 | 1.65x | 48 | 5 |

N5_rows_f against W2_packed_v3 (throughput, not_), batch N5_rows_f_vs_W2_packed_v3_focus_b3w2_20261025_160604
| cell | N5_rows_f | W2_packed_v3 | ratio | registers | occupancy |
|---|---|---|---|---|---|
| N 128 k 16 | 480 | 447 | 1.07x | 47 | 5 |
| N 128 k 32 | 480 | 252 | 1.90x | 47 | 5 |
| N 128 k 64 | 434 | 136 | 3.19x | 40 | 6 |
| N 256 k 16 | 512 | 578 | 0.89x | 59 | 4 |
| N 256 k 32 | 511 | 331 | 1.54x | 59 | 4 |
| N 256 k 64 | 452 | 178 | 2.54x | 48 | 5 |

N5_rows_f against W1x_vimnmx_eager (latency, not_), batch N5_rows_f_vs_W1x_vimnmx_eager_focus_b3_20261025_160452
| cell | N5_rows_f | W1x_vimnmx_eager | ratio | registers | occupancy |
|---|---|---|---|---|---|
| N 32 k 4 | 855 | 512 | 0.60x | 37 | 32 |
| N 32 k 16 | 855 | 512 | 0.60x | 37 | 32 |
| N 64 k 4 | 1037 | 695 | 0.67x | 44 | 32 |
| N 64 k 16 | 1037 | 695 | 0.67x | 44 | 32 |
| N 64 k 32 | 1037 | 695 | 0.67x | 44 | 32 |
| N 128 k 4 | 1434 | 1017 | 0.71x | 56 | 32 |
| N 128 k 16 | 1432 | 1017 | 0.71x | 57 | 32 |
| N 128 k 32 | 1433 | 1017 | 0.71x | 57 | 32 |
| N 128 k 64 | 1433 | 912 | 0.64x | 57 | 32 |
| N 256 k 4 | 2300 | 1718 | 0.75x | 64 | 32 |
| N 256 k 16 | 2315 | 1718 | 0.74x | 64 | 32 |
| N 256 k 32 | 2315 | 1718 | 0.74x | 64 | 32 |
| N 256 k 64 | 2113 | 1322 | 0.63x | 64 | 32 |

N5_split_f against W1x_vimnmx_eager (latency, not_), batch N5_split_f_vs_W1x_vimnmx_eager_focus_b3_20261025_160530
| cell | N5_split_f | W1x_vimnmx_eager | ratio | registers | occupancy |
|---|---|---|---|---|---|
| N 64 k 4 | 1062 | 695 | 0.65x | 32 | 32 |
| N 64 k 16 | 1062 | 695 | 0.65x | 32 | 32 |
| N 64 k 32 | 1062 | 695 | 0.65x | 32 | 32 |
| N 128 k 4 | 1211 | 1017 | 0.84x | 46 | 32 |
| N 128 k 16 | 1210 | 1017 | 0.84x | 46 | 32 |
| N 128 k 32 | 1210 | 1017 | 0.84x | 46 | 32 |
| N 128 k 64 | 1218 | 912 | 0.75x | 46 | 32 |
| N 256 k 4 | 1604 | 1718 | 1.07x | 54 | 32 |
| N 256 k 16 | 1606 | 1718 | 1.07x | 54 | 32 |
| N 256 k 32 | 1607 | 1718 | 1.07x | 54 | 32 |
| N 256 k 64 | 1621 | 1322 | 0.82x | 56 | 32 |

N5_rows_f against W2_packed_v3 (latency, not_), batch N5_rows_f_vs_W2_packed_v3_focus_b3w2_20261025_160604
| cell | N5_rows_f | W2_packed_v3 | ratio | registers | occupancy |
|---|---|---|---|---|---|
| N 128 k 16 | 1432 | 468 | 0.33x | 57 | 32 |
| N 128 k 32 | 1433 | 909 | 0.63x | 57 | 32 |
| N 128 k 64 | 1433 | 1775 | 1.24x | 57 | 32 |
| N 256 k 16 | 2315 | 568 | 0.25x | 64 | 32 |
| N 256 k 32 | 2315 | 966 | 0.42x | 64 | 32 |
| N 256 k 64 | 2113 | 1868 | 0.88x | 64 | 32 |

### 9.2 Sub-warp scope (batch b3)

N5_rows_fs_sub against W1_eager_same_sub (throughput, not_), batch N5_rows_fs_sub_vs_W1_eager_same_sub_focus_b3_20261025_160248
| cell | N5_rows_fs_sub | W1_eager_same_sub | ratio | registers | occupancy |
|---|---|---|---|---|---|
| LW 8 N 32 k 4 | 705 | 513 | 1.38x | 32 | 8 |
| LW 8 N 32 k 16 | 578 | 471 | 1.23x | 40 | 6 |
| LW 8 N 64 k 4 | 726 | 515 | 1.41x | 40 | 6 |
| LW 8 N 64 k 16 | 641 | 436 | 1.47x | 40 | 6 |
| LW 8 N 64 k 32 | 574 | 348 | 1.65x | 40 | 6 |
| LW 8 N 128 k 4 | 713 | 506 | 1.41x | 64 | 4 |
| LW 8 N 128 k 16 | 632 | 413 | 1.53x | 64 | 4 |
| LW 8 N 128 k 32 | 554 | 354 | 1.57x | 64 | 4 |
| LW 8 N 128 k 64 | 483 | 281 | 1.72x | 62 | 4 |
| LW 16 N 32 k 4 | 459 | 310 | 1.48x | 30 | 8 |
| LW 16 N 32 k 16 | 460 | 337 | 1.37x | 30 | 8 |
| LW 16 N 64 k 4 | 556 | 333 | 1.67x | 32 | 8 |
| LW 16 N 64 k 16 | 555 | 337 | 1.65x | 32 | 8 |
| LW 16 N 64 k 32 | 477 | 280 | 1.70x | 40 | 6 |
| LW 16 N 128 k 4 | 608 | 363 | 1.67x | 40 | 6 |
| LW 16 N 128 k 16 | 608 | 336 | 1.81x | 40 | 6 |
| LW 16 N 128 k 32 | 522 | 285 | 1.83x | 60 | 4 |
| LW 16 N 128 k 64 | 464 | 240 | 1.93x | 48 | 5 |

N5_rows_f_sub against W1_eager_same_sub (throughput, not_), batch N5_rows_f_sub_vs_W1_eager_same_sub_focus_b3_20261025_160143
| cell | N5_rows_f_sub | W1_eager_same_sub | ratio | registers | occupancy |
|---|---|---|---|---|---|
| LW 8 N 32 k 4 | 670 | 513 | 1.31x | 40 | 6 |
| LW 8 N 32 k 16 | 571 | 471 | 1.21x | 40 | 6 |
| LW 8 N 64 k 4 | 713 | 515 | 1.39x | 48 | 5 |
| LW 8 N 64 k 16 | 617 | 436 | 1.42x | 48 | 5 |
| LW 8 N 64 k 32 | 534 | 348 | 1.54x | 47 | 5 |
| LW 8 N 128 k 4 | 713 | 506 | 1.41x | 64 | 4 |
| LW 8 N 128 k 16 | 601 | 413 | 1.46x | 64 | 4 |
| LW 8 N 128 k 32 | 520 | 354 | 1.47x | 64 | 4 |
| LW 8 N 128 k 64 | 438 | 281 | 1.56x | 64 | 4 |
| LW 16 N 32 k 4 | 477 | 310 | 1.54x | 40 | 6 |
| LW 16 N 32 k 16 | 478 | 337 | 1.42x | 40 | 6 |
| LW 16 N 64 k 4 | 556 | 333 | 1.67x | 40 | 6 |
| LW 16 N 64 k 16 | 556 | 337 | 1.65x | 40 | 6 |
| LW 16 N 64 k 32 | 480 | 280 | 1.71x | 48 | 5 |
| LW 16 N 128 k 4 | 571 | 363 | 1.57x | 54 | 4 |
| LW 16 N 128 k 16 | 571 | 336 | 1.70x | 54 | 4 |
| LW 16 N 128 k 32 | 490 | 285 | 1.72x | 62 | 4 |
| LW 16 N 128 k 64 | 440 | 240 | 1.84x | 60 | 4 |

N5_split_f_sub against W1_eager_same_sub (throughput, not_), batch N5_split_f_sub_vs_W1_eager_same_sub_focus_b3_20261025_160353
| cell | N5_split_f_sub | W1_eager_same_sub | ratio | registers | occupancy |
|---|---|---|---|---|---|
| LW 8 N 32 k 4 | 580 | 513 | 1.13x | 32 | 8 |
| LW 8 N 32 k 16 | 503 | 471 | 1.07x | 32 | 8 |
| LW 8 N 64 k 4 | 679 | 515 | 1.32x | 40 | 6 |
| LW 8 N 64 k 16 | 571 | 436 | 1.31x | 48 | 5 |
| LW 8 N 64 k 32 | 492 | 348 | 1.41x | 40 | 6 |
| LW 8 N 128 k 4 | 724 | 506 | 1.43x | 60 | 4 |
| LW 8 N 128 k 16 | 620 | 413 | 1.50x | 53 | 4 |
| LW 8 N 128 k 32 | 502 | 354 | 1.42x | 58 | 4 |
| LW 8 N 128 k 64 | 450 | 281 | 1.60x | 48 | 5 |
| LW 16 N 32 k 4 | 381 | 310 | 1.23x | 32 | 8 |
| LW 16 N 32 k 16 | 383 | 337 | 1.14x | 32 | 8 |
| LW 16 N 64 k 4 | 448 | 333 | 1.35x | 40 | 6 |
| LW 16 N 64 k 16 | 448 | 337 | 1.33x | 40 | 6 |
| LW 16 N 64 k 32 | 401 | 280 | 1.43x | 38 | 6 |
| LW 16 N 128 k 4 | 523 | 363 | 1.44x | 47 | 5 |
| LW 16 N 128 k 16 | 523 | 336 | 1.56x | 47 | 5 |
| LW 16 N 128 k 32 | 451 | 285 | 1.58x | 40 | 6 |
| LW 16 N 128 k 64 | 385 | 240 | 1.61x | 48 | 5 |

N5_rows_fs_sub against W1_eager_same_sub (latency, not_), batch N5_rows_fs_sub_vs_W1_eager_same_sub_focus_b3_20261025_160248
| cell | N5_rows_fs_sub | W1_eager_same_sub | ratio | registers | occupancy |
|---|---|---|---|---|---|
| LW 8 N 32 k 4 | 1003 | 980 | 0.98x | 55 | 32 |
| LW 8 N 32 k 16 | 1083 | 1103 | 1.02x | 58 | 32 |
| LW 8 N 64 k 4 | 1709 | 1953 | 1.14x | 64 | 32 |
| LW 8 N 64 k 16 | 1581 | 2015 | 1.27x | 64 | 32 |
| LW 8 N 64 k 32 | 1665 | 2434 | 1.46x | 64 | 32 |
| LW 8 N 128 k 4 | 3039 | 3879 | 1.28x | 80 | 24 |
| LW 8 N 128 k 16 | 2600 | 4522 | 1.74x | 92 | 20 |
| LW 8 N 128 k 32 | 2562 | 5274 | 2.06x | 93 | 20 |
| LW 8 N 128 k 64 | 2716 | 5243 | 1.93x | 95 | 20 |
| LW 16 N 32 k 4 | 1030 | 766 | 0.74x | 57 | 32 |
| LW 16 N 32 k 16 | 1032 | 767 | 0.74x | 57 | 32 |
| LW 16 N 64 k 4 | 1344 | 1589 | 1.18x | 61 | 32 |
| LW 16 N 64 k 16 | 1345 | 1591 | 1.18x | 61 | 32 |
| LW 16 N 64 k 32 | 1414 | 1746 | 1.23x | 63 | 32 |
| LW 16 N 128 k 4 | 2263 | 3186 | 1.41x | 69 | 28 |
| LW 16 N 128 k 16 | 2265 | 3242 | 1.43x | 69 | 28 |
| LW 16 N 128 k 32 | 2045 | 3818 | 1.87x | 71 | 28 |
| LW 16 N 128 k 64 | 2197 | 4182 | 1.90x | 72 | 28 |

N5_split_f_sub against W1_eager_same_sub (latency, not_), batch N5_split_f_sub_vs_W1_eager_same_sub_focus_b3_20261025_160353
| cell | N5_split_f_sub | W1_eager_same_sub | ratio | registers | occupancy |
|---|---|---|---|---|---|
| LW 8 N 32 k 4 | 848 | 980 | 1.16x | 46 | 32 |
| LW 8 N 32 k 16 | 873 | 1103 | 1.26x | 46 | 32 |
| LW 8 N 64 k 4 | 1151 | 1953 | 1.70x | 56 | 32 |
| LW 8 N 64 k 16 | 1144 | 2015 | 1.76x | 59 | 32 |
| LW 8 N 64 k 32 | 1327 | 2434 | 1.83x | 56 | 32 |
| LW 8 N 128 k 4 | 1867 | 3879 | 2.08x | 72 | 28 |
| LW 8 N 128 k 16 | 1822 | 4522 | 2.48x | 72 | 28 |
| LW 8 N 128 k 32 | 2009 | 5274 | 2.63x | 72 | 28 |
| LW 8 N 128 k 64 | 2078 | 5243 | 2.52x | 72 | 28 |
| LW 16 N 32 k 4 | 944 | 766 | 0.81x | 44 | 32 |
| LW 16 N 32 k 16 | 945 | 767 | 0.81x | 45 | 32 |
| LW 16 N 64 k 4 | 1098 | 1589 | 1.45x | 51 | 32 |
| LW 16 N 64 k 16 | 1099 | 1591 | 1.45x | 51 | 32 |
| LW 16 N 64 k 32 | 1127 | 1746 | 1.55x | 52 | 32 |
| LW 16 N 128 k 4 | 1508 | 3186 | 2.11x | 60 | 32 |
| LW 16 N 128 k 16 | 1506 | 3242 | 2.15x | 59 | 32 |
| LW 16 N 128 k 32 | 1548 | 3818 | 2.47x | 61 | 32 |
| LW 16 N 128 k 64 | 1755 | 4182 | 2.38x | 61 | 32 |

### 9.3 Development batches (throughput, b1 select-form exchange at 3ede7791e3, b2 ternary form at 30f5b8f821)

N5_rows against W1x_vimnmx_eager (throughput, not_), batch N5_rows_vs_W1x_vimnmx_eager_focus_b1_20261025_153333
| cell | N5_rows | W1x_vimnmx_eager | ratio | registers | occupancy |
|---|---|---|---|---|---|
| N 32 k 4 | 358 | 270 | 1.33x | 29 | 8 |
| N 32 k 16 | 359 | 270 | 1.33x | 29 | 8 |
| N 64 k 4 | 418 | 327 | 1.28x | 32 | 8 |
| N 64 k 16 | 418 | 328 | 1.28x | 32 | 8 |
| N 64 k 32 | 418 | 331 | 1.26x | 32 | 8 |
| N 128 k 4 | 446 | 356 | 1.25x | 32 | 8 |
| N 128 k 16 | 444 | 356 | 1.25x | 32 | 8 |
| N 128 k 32 | 445 | 358 | 1.24x | 32 | 8 |
| N 128 k 64 | 405 | 302 | 1.34x | 40 | 6 |
| N 256 k 4 | 458 | 363 | 1.26x | 40 | 6 |
| N 256 k 16 | 457 | 363 | 1.26x | 40 | 6 |
| N 256 k 32 | 457 | 367 | 1.24x | 40 | 6 |
| N 256 k 64 | 409 | 248 | 1.65x | 48 | 5 |

N5_split against W1x_vimnmx_eager (throughput, not_), batch N5_split_vs_W1x_vimnmx_eager_focus_b1_20261025_153411
| cell | N5_split | W1x_vimnmx_eager | ratio | registers | occupancy |
|---|---|---|---|---|---|
| N 64 k 4 | 318 | 328 | 0.97x | 30 | 8 |
| N 64 k 16 | 318 | 328 | 0.97x | 30 | 8 |
| N 64 k 32 | 318 | 330 | 0.96x | 30 | 8 |
| N 128 k 4 | 386 | 356 | 1.08x | 31 | 8 |
| N 128 k 16 | 385 | 356 | 1.08x | 31 | 8 |
| N 128 k 32 | 385 | 358 | 1.08x | 31 | 8 |
| N 128 k 64 | 346 | 302 | 1.14x | 40 | 6 |
| N 256 k 4 | 422 | 363 | 1.16x | 32 | 8 |
| N 256 k 16 | 422 | 363 | 1.16x | 32 | 8 |
| N 256 k 32 | 421 | 367 | 1.15x | 32 | 8 |
| N 256 k 64 | 377 | 248 | 1.52x | 48 | 5 |

W1_eager_same against W1x_vimnmx_eager (throughput, not_), batch W1_eager_same_vs_W1x_vimnmx_eager_focus_b1_20261025_153444
| cell | W1_eager_same | W1x_vimnmx_eager | ratio | registers | occupancy |
|---|---|---|---|---|---|
| N 64 k 16 | 273 | 328 | 0.83x | 24 | 8 |
| N 64 k 32 | 274 | 331 | 0.83x | 25 | 8 |
| N 128 k 16 | 268 | 356 | 0.75x | 32 | 8 |
| N 128 k 32 | 267 | 358 | 0.75x | 32 | 8 |
| N 128 k 64 | 236 | 302 | 0.78x | 32 | 8 |
| N 256 k 16 | 263 | 363 | 0.72x | 32 | 8 |
| N 256 k 32 | 263 | 367 | 0.71x | 32 | 8 |
| N 256 k 64 | 223 | 248 | 0.90x | 32 | 8 |

N5_rows against W1x_vimnmx_eager (throughput, not_), batch N5_rows_vs_W1x_vimnmx_eager_focus_b2_20261025_154953
| cell | N5_rows | W1x_vimnmx_eager | ratio | registers | occupancy |
|---|---|---|---|---|---|
| N 32 k 4 | 395 | 270 | 1.47x | 32 | 8 |
| N 32 k 16 | 395 | 270 | 1.46x | 32 | 8 |
| N 64 k 4 | 407 | 328 | 1.24x | 40 | 6 |
| N 64 k 16 | 407 | 328 | 1.24x | 40 | 6 |
| N 64 k 32 | 407 | 331 | 1.23x | 40 | 6 |
| N 128 k 4 | 459 | 356 | 1.29x | 48 | 5 |
| N 128 k 16 | 458 | 356 | 1.29x | 48 | 5 |
| N 128 k 32 | 458 | 358 | 1.28x | 48 | 5 |
| N 128 k 64 | 406 | 302 | 1.34x | 40 | 6 |
| N 256 k 4 | 478 | 363 | 1.32x | 64 | 4 |
| N 256 k 16 | 477 | 363 | 1.31x | 64 | 4 |
| N 256 k 32 | 477 | 367 | 1.30x | 64 | 4 |
| N 256 k 64 | 417 | 248 | 1.68x | 48 | 5 |

N5_rows_f against W1x_vimnmx_eager (throughput, not_), batch N5_rows_f_vs_W1x_vimnmx_eager_focus_b2_20261025_155030
| cell | N5_rows_f | W1x_vimnmx_eager | ratio | registers | occupancy |
|---|---|---|---|---|---|
| N 32 k 4 | 393 | 270 | 1.45x | 30 | 8 |
| N 32 k 16 | 394 | 270 | 1.46x | 30 | 8 |
| N 64 k 4 | 405 | 328 | 1.24x | 36 | 6 |
| N 64 k 16 | 405 | 328 | 1.24x | 36 | 6 |
| N 64 k 32 | 404 | 330 | 1.22x | 36 | 6 |
| N 128 k 4 | 477 | 356 | 1.34x | 47 | 5 |
| N 128 k 16 | 477 | 356 | 1.34x | 47 | 5 |
| N 128 k 32 | 477 | 358 | 1.33x | 47 | 5 |
| N 128 k 64 | 436 | 302 | 1.44x | 40 | 6 |
| N 256 k 4 | 514 | 363 | 1.42x | 59 | 4 |
| N 256 k 16 | 513 | 363 | 1.41x | 59 | 4 |
| N 256 k 32 | 511 | 367 | 1.39x | 59 | 4 |
| N 256 k 64 | 452 | 248 | 1.82x | 48 | 5 |

N5_split against W1x_vimnmx_eager (throughput, not_), batch N5_split_vs_W1x_vimnmx_eager_focus_b2_20261025_155108
| cell | N5_split | W1x_vimnmx_eager | ratio | registers | occupancy |
|---|---|---|---|---|---|
| N 64 k 4 | 320 | 328 | 0.98x | 28 | 8 |
| N 64 k 16 | 320 | 328 | 0.98x | 28 | 8 |
| N 64 k 32 | 320 | 331 | 0.97x | 28 | 8 |
| N 128 k 4 | 370 | 356 | 1.04x | 40 | 6 |
| N 128 k 16 | 370 | 356 | 1.04x | 40 | 6 |
| N 128 k 32 | 370 | 358 | 1.03x | 40 | 6 |
| N 128 k 64 | 354 | 302 | 1.17x | 38 | 6 |
| N 256 k 4 | 428 | 363 | 1.18x | 40 | 6 |
| N 256 k 16 | 427 | 363 | 1.18x | 40 | 6 |
| N 256 k 32 | 426 | 367 | 1.16x | 40 | 6 |
| N 256 k 64 | 384 | 248 | 1.54x | 48 | 5 |

N5_split_f against W1x_vimnmx_eager (throughput, not_), batch N5_split_f_vs_W1x_vimnmx_eager_focus_b2_20261025_155140
| cell | N5_split_f | W1x_vimnmx_eager | ratio | registers | occupancy |
|---|---|---|---|---|---|
| N 64 k 4 | 312 | 328 | 0.95x | 30 | 8 |
| N 64 k 16 | 312 | 328 | 0.95x | 30 | 8 |
| N 64 k 32 | 312 | 330 | 0.94x | 30 | 8 |
| N 128 k 4 | 387 | 356 | 1.09x | 37 | 6 |
| N 128 k 16 | 387 | 356 | 1.09x | 37 | 6 |
| N 128 k 32 | 387 | 358 | 1.08x | 37 | 6 |
| N 128 k 64 | 360 | 302 | 1.19x | 39 | 6 |
| N 256 k 4 | 459 | 363 | 1.26x | 40 | 6 |
| N 256 k 16 | 458 | 363 | 1.26x | 40 | 6 |
| N 256 k 32 | 457 | 367 | 1.24x | 40 | 6 |
| N 256 k 64 | 412 | 248 | 1.66x | 48 | 5 |
