# W1: bitonic network top-k at warp and sub-warp scope (approach report, Instance A)

Author opt-W1 (Instance A of the two W1 optimizer instances that shared one agent id on
2026-10-25). This report covers W1_packed, W1_level and their sub-warp forms, the N2 kill
experiment as W1_level against W1_packed, and the sub-warp controls. Instance B (opt-W1b) owns the
same-code eager against level comparison with min and max stages, the fp32 native min and max pairs
variant and the composite comparator study, report `approaches/W1/REPORT_W1b.md` and cards
`W1_eager_same`, `W1_level_pairs`, `W1_minmax_pairs`, `W1_kvcmp`, `W1_baseline` (B200). Every
number below is from GB200 (gb200-nvl4-ts2-99, sm_100, nvcc 13.4.92, 2026-10-25), variant and
control in one lock section, random normal input, cards in `cards/`.

## 1. What it is

A data-oblivious register-resident bitonic network that selects the k largest of N keys held by a
logical warp of LW lanes in a striped arrangement (word i of lane l is item i times LW plus l) and
writes the k source indices, sorted. Two structures on the same words:

- `W1_packed`: the eager loop of PR #12222 (`cub::detail::WarpBitonicTopK`): sort the first MaxK
  tile, then for every further tile sort it, keep the elementwise larger of retained and incoming
  (the first half-cleaner of their concatenation) and merge the retained tile. Keys only, on packed
  words.
- `W1_level`: the level-parallel restructuring of idea card N2 (`approaches/W1/warp_level_topk.cuh`,
  my own code): sort all N over MaxK tiles at once with the stages interleaved, then halve the tile
  count per level with an elementwise max and one bitonic merge of log2(MaxK) stages. Same
  registers, same determinism, 25 dependent stages at N 128 where the eager has 75.

The packed word (`approaches/W1/w1_common.cuh`): bf16 keys become one 32-bit word, order
preserving key bits in the high half and the inverted 16-bit index in the low half, fp32 keys one
64-bit word with the inverted 32-bit index below. The network moves one word per item and the
integer word order is exactly (key descending, index ascending), so `prefer_smaller_index` is a
property of the data, not of the code. Minus zero is mapped to plus zero before packing.

Files: `harness/adapters/W1_packed.cuh`, `W1_level.cuh`, `W1_level_n64.cuh` (shuffle ablation),
`W1_reference_sub.cuh` (the pairs reference at LW 4, 8, 16, the sub-warp control), `W1_packed_sub.cuh`,
`W1_level_sub.cuh`, shared templates in `approaches/W1/w1_adapters.cuh`.

## 2. Contract

Input: N keys in registers, striped, N a multiple of LW (W1_level also needs N over MaxK to be a
power of two), bf16 or fp32 (any 16-bit or 32-bit key with an order preserving bit image works).
k in [1, N] with MaxK = round_up(k, LW) at most N, k below MaxK pays the MaxK network. bf16 needs N
at most 65536 (16-bit index). Output: k indices in the caller's array, sorted by (key descending,
index ascending), both unsorted requirements at the same cost, keys never materialized (one more
unpack per output if wanted). No shared memory, no barrier, shuffles inside the logical warp only.
NaN: a NaN key orders by its sign bit (positive NaN above every finite key), deterministic. Partial
tiles: not in this contract (the CUB primitive has valid_items overloads, the level network would
need a pad word per missing item, which is one select per word).

## 3. Where the time goes

Per shuffle stage (one SHFL at 30 cycles, T1) the bf16 word costs two predicated VIMNMX (T7, 6
cycles) in the level network and ISETP, ISETP, PLOP3, SEL in the CUB comparator path (ptxas does
not emit VIMNMX for the bool comparator on u32 words, contrary to the integer remark in issue
#12160). The fp32 u64 word costs two SHFL and two dependent ISETP plus two SEL in both forms. The
reference pairs network costs two SHFL (key and index), two FSETP or `__hgt`, PLOP3, two SEL and,
for bf16, pack and unpack per stage.

Measured per call, warp scope (latency cycles, throughput Gelem/s, control in the same section):

| cell | W1_reference (pairs) | W1_packed | W1_level | level over reference |
|---|---|---|---|---|
| bf16 N 32 k 8 | 778, 135 | 626, 244 | same network as packed | 1.24x, 1.81x |
| bf16 N 64 k 32 | 1164, 121 | 905, 273 | 695, 356 | 1.67x, 2.94x |
| bf16 N 128 k 16 | 2267, 114 | 1367, 264 | 883, 370 | 2.57x, 3.25x |
| bf16 N 128 k 64 | 2562, 95 | 1598, 233 | 916, 326 | 2.80x, 3.43x |
| bf16 N 256 k 32 | not measured | 2840, 272 | 1115, 398 | |
| fp32 N 32 k 8 | 605, 204 | 597, 201 | same network as packed | 1.01x, 0.99x |
| fp32 N 64 k 32 | 846, 205 | 990, 166 | 1194, 216 | 0.71x, 1.05x |
| fp32 N 128 k 16 | 2038, 192 | 2408, 156 | 1131, 212 | 1.80x, 1.10x |
| fp32 N 128 k 64 | 2255, 165 | 2583, 136 | 1671, 152 | 1.35x, 0.92x |
| fp32 N 256 k 32 | not measured | 4584, 154 | 1790, 199 | |

The level over reference column chains two same-session sections through W1_packed, which measured
to the cycle in both. The eager cost per added MaxK tile is about 710 to 740 cycles (fp32 990 at
N 64 to 2408 at N 128), the level cost per doubling of N is 190 to 230 cycles at bf16 (695, 883,
1115) and 60 to 650 at fp32 (1194, 1131, 1728): the restructuring turns a chain of tile sorts into
one sort at higher items per lane, and the independent shuffles of one stage pipeline at about 5
cycles each (T17), far below the July placeholder of s(8) 152 against s(4) 70.

Sub-warp (W1_level_sub against W1_packed_sub against W1_reference_sub, bf16 latency): LW 8 N 16 k 4
302 against 422 against 570, LW 16 N 32 k 8 466 against 809 against 829, LW 16 N 128 k 4 888
against 3504 against 3980, LW 8 N 128 k 8 1074 against 4050 against 5050. fp32: LW 8 N 16 k 4 523
against 600 against 516, LW 16 N 128 k 4 2297 against 4033 against 3281. Throughput gains of the
level form 1.1x to 1.5x at LW 8 and 16, none at LW 4 where 32 words per lane cap the occupancy.

Registers: W1_packed 20 to 27 (bf16) and 24 to 42 (fp32) at warp N 32 to 128, W1_level 35 to 56
and 37 to 64 at N 64 to 256, no spills, no stack frame. The sub-warp forms reach 96 to 168 registers
in the latency kernel at LW 4 N 128.

## 4. Ideas tried, with verdicts

1. Index in the key word, keys-only network (W1_packed, cards `W1_packed_gb200_20261025_1`,
   `W1_packed_sub_gb200_20261025_1`). bf16: 1.24x to 1.74x latency and 1.8x to 2.5x throughput over
   the pairs reference at warp scope, 1.0x to 2.1x at sub-warp scope, `prefer_smaller_index` free.
   fp32 through a u64 word: 0.85x at N at least 64 (the price of the exact tie-break in the eager
   form), 0.7x to 1.1x at sub-warp scope. Kept for bf16, kept for fp32 only as the carrier of the
   level network.
2. N2 level-parallel restructuring (W1_level, cards `N2_level_gb200_20261025_1`,
   `N2_level_sub_gb200_20261025_1`). Kill experiment at the stated cell warp N 64 k 32: bf16 1.30x
   (threshold met exactly), fp32 0.83x (missed), because at two tiles the eager's two independent
   sorts already overlap in ptxas's schedule. From four tiles on: 1.55x to 2.55x bf16 and 1.48x to
   2.65x fp32 at warp N 128 and 256, 1.2x to 4.2x at sub-warp scope, throughput 1.1x to 1.5x.
   Verdict: N2 survives and replaces the eager loop for N at least 4 MaxK, the eager keeps N at
   most 2 MaxK for fp32. The researcher's model held in shape and overstated the stage cost at high
   items per lane.
3. Local-array fix (same variant, commits ccc1166d9c to 851d41cc5c): the first level network put
   the word array in local memory in the latency kernel at N 256 and at 32 words per lane (80 to
   544 bytes stack frame, 0 spill bytes, 14871 cycles instead of 3565 at fp32 N 256 k 64). Lambdas
   capturing the array and `#pragma unroll` loops with array indexing were replaced by template
   recursion with every index a compile-time constant. Catalogue entry T36 (with T34).
4. 64-bit shuffle split by hand (W1_level_n64 against W1_level): 1.00x in every cell, the CUDA
   13.4 header already splits the 64-bit shuffle without asm volatile. Retired, the policy
   parameter stays in the code with the split as default.
5. Narrower tiles than the logical warp for k below 32 at warp scope (N2 explore item 2): modeled
   on the measured stage costs, no gain (the saved sort stages are given back by the extra in-slot
   level), not built. Section 5 has the numbers.

## 5. Ideas not yet tried

- fp32 keys only with index recovery: one 32-bit shuffle and one FMNMX per stage (the min and max
  form of issue #12160, which the level network already uses for the integer words), then the
  k-th key from position k minus 1 and a ballot rank pass over the input (strictly larger items,
  then the first m equal items in position order, which gives `prefer_smaller_index` for free).
  Model at warp N 128: network near the bf16 cost (883) plus about 200 cycles of recovery against
  1131 with the u64 word, more in throughput where the u64 word doubles the shuffles.
- Sub-tile width below LW for k below LW at warp scope: at N 128 k 16 with 16-lane tiles, phase 1
  is 10 stages at IPT 8 instead of 15 at IPT 4, the levels are two register levels of 4 merge
  stages and one in-slot level (shuffle max plus 4 stages), total about 880 cycles on the measured
  per-stage costs against 883 now. At k 8 with 8-lane tiles the two in-slot levels (3 stages each
  plus the shuffle max) cost what the 9 saved sort stages save. Not worth a batch at warp scope,
  the sub-warp forms are this idea with LW as the tile width.
- Phase 1 in two halves at LW 4 (throughput): keep only 16 words live through the first sorts and
  levels, then one more level, to recover the occupancy that the 32 live words cost (0.91x to
  1.08x throughput at LW 4 N 128).
- bf16x2 two rows per lane (N5) on the level network for the throughput regime, where the bf16
  network is ALU bound (T33 puts the SHFL pipe far above the measured rate).
- The buffered variant of PR #9281 (threshold ballot compaction) for N at least 8 MaxK against the
  level network, random and sorted input. Not built, the level network already removes most of
  the eager's chain cost that the buffered form targets.
- A `prefer_larger_index` form: the index goes in without inversion, one instruction less per item.

## 6. Determinism and ordering

Inherent. The network is deterministic and the output is sorted by (key descending, index
ascending) in the first k striped positions. `prefer_smaller_index` costs nothing: in every cell of
every card the psi number equals the ng number to the cycle (same code, the word order carries the
tie-break), for example bf16 N 128 k 16 1367 and 1367 (W1_packed), 883 and 883 (W1_level), fp32
N 128 k 16 2408 and 2408, 1131 and 1131. The price of the exact tie-break for fp32 is the u64 word
itself: 15 to 19 percent against the not_guaranteed pairs reference in the eager form at N 64 and
128, and none left once the level network wins by 1.35x to 1.80x against that reference at N 128.
Minus zero and plus zero compare equal (canonicalized before packing), verified by the oracle's
signed_zero pattern under the exact requirement.

## 7. Open items

- The latency chain of an unchanged network moves by up to 25 percent with k at fixed MaxK (the
  output store predicate is the only difference): W1_packed bf16 N 64 1150 at k 16 against 905 at
  k 32, W1_level_sub fp32 LW 16 N 128 2297 at k 8 against 3282 at k 16. This is T28's schedule
  fragility (reported by Instance B on their min max pairs network, here also on the CUB
  comparator network). Rule adopted: measure every (N, k) separately, quote the conservative
  control.
- fp32 at two tiles: the level form loses 0.83x against the eager with the same instruction mix.
  Schedule, not algorithm, not pursued.
- The harness summary and the card schema show spill bytes only, a local array shows as a stack
  frame with zero spill (T34, T36). Asked eval-engineer to surface stack bytes and STL plus LDL.
- No same-session comparison with W2 yet at warp N 64 to 256, k 16 to 64, where PRIORITIES.md
  expects W2 to own k at most 32. The level network's 883 cycles at N 128 k 16 bf16 and 1115 at
  N 256 k 32 are the numbers W2 has to beat there.
- Control for psi cells: W1_reference does not support the requirement, the psi cells of the
  W1_packed cards carry an A/A control from a second lock section of the same lease. Asked
  eval-engineer for a `--control-req` option so the control can be the variant's own ng path in
  one section. W1_level's psi cells have W1_packed as a real control.
- Sub-warp LW 4 throughput, see section 5.

## 8. Pointers

- Cards: `cards/W1_packed_gb200_20261025_1.json`, `cards/W1_packed_sub_gb200_20261025_1.json`,
  `cards/N2_level_gb200_20261025_1.json`, `cards/N2_level_sub_gb200_20261025_1.json`.
- Batches (state/results/opt-W1/): `W1_packed_vs_W1_reference_focus_base_ng_20261025_131236.json`,
  `W1_packed_vs_W1_packed_focus_base_psi_20261025_131315.json`,
  `W1_packed_sub_vs_W1_reference_sub_focus_base_ng_20261025_131358.json`,
  `W1_packed_sub_vs_W1_packed_sub_focus_base_psi_20261025_131552.json`,
  `W1_level_vs_W1_packed_focus_kill_n2_20261025_131737.json` (first level build, local array at
  N 256, superseded), `W1_level_sub_vs_W1_packed_sub_focus_kill_n2_20261025_131831.json` (same),
  `W1_level_n64_vs_W1_packed_focus_fix3_20261025_134309.json`,
  `W1_level_vs_W1_level_n64_focus_fix3_20261025_134401.json`,
  `W1_level_vs_W1_packed_focus_final3_20261025_134828.json`,
  `W1_level_sub_vs_W1_packed_sub_focus_fix3_20261025_134920.json`. Three leases in all (baseline,
  kill experiment, fixes), each with several lock sections, every section with its control.
- Code: worktree `/Users/estehle/Downloads/cccl-topk-swarm-W1`, branch `exp/topk-swarm/W1`, commit
  851d41cc5c. `approaches/W1/warp_level_topk.cuh` (the level network), `w1_common.cuh` (packing,
  word operations, shuffle policies), `w1_adapters.cuh` (adapter templates), `node.sh` (node
  workflow), `summarize_threads.py` (batch summary that keeps the logical warp size),
  `sass_count.awk` (opcode histogram per kernel), `findings_*.json` (card findings),
  `technique_local_array.md` (T36 as appended).
- Catalogue: adopted T3, T7, T17, T20, T28, T33, T34, contributed T36.
- Idea cards: `ideas/queue/W1_bitonic_topk.md`, `ideas/queue/N2_level_parallel_bitonic.md` (verdict
  in section 4), `ideas/queue/N5_two_row_bf16x2.md` (section 5).
- Reference numbers: `approaches/W1/REFERENCE.md` (eval-engineer), cards `W1_reference_*`.
