# N10: fp32 keys-only network with index recovery by ballot rank

Optimizer opt-N10, worktree `cccl-topk-swarm-N10`, branch `exp/topk-swarm/N10`. Idea card
`ideas/queue/N10_fp32_keys_only_index_recovery.md`. Every number is B200 (sm_100 build,
umbriel-b200-020, x86 docker), CUDA 13.4.92, 2026-10-25, oracle battery passed on every cell before
any number was read, control in the same lock section. Batches under `state/results/opt-N10/`.

## 1. What it is

The fp32 prefer_smaller_index route for the W1 family without an index in the network. The fp32 key
is twiddled into an order-preserving u32 word (sign flip, -0.0 mapped to +0.0) and the words run
through the min and max exchange network of opt-W1's template recursion (`approaches/W1/
warp_level_topk.cuh`, `word_ops<uint32_t>`, T42): one shuffle and two predicated VIMNMX per shuffle
stage, one VIMNMX pair per in-register stage, no index moves. The network's only job is the k-th
largest word t. Three forms of getting it were built:

- `N10_u32`: the eager tile loop with the complete final merge, t is position k - 1 (one shuffle).
- `N10_u32_trunc`: the final merge of tile 0 stops after the half-cleaner stages with stride at
  least k, positions [0, k) then hold the k largest words as a bitonic sequence and t is their
  minimum, one `redux.sync.min` over the full warp with the identity in the other lanes (T37, T40).
- `N10_u32_ltrunc`: the same truncated merge on the level tile order (all tiles sorted at once,
  tile count halved per level).

Then every lane compares its original words (kept in registers, IPT of them) against t and the
indices are recovered by ballot rank in slot-major index order (slot i of lane l is source index
i x LW + l), so prefer_smaller_index is exact by construction and the same code serves
not_guaranteed:

- two-class recovery (`N10_u32*`): one "above t" and one "equal to t" ballot per slot, popc
  prefixes give every item above t a slot in index order and the items equal to t take the next
  slots in index order until k is reached.
- one-ballot recovery (`N10_ge1_*`): one "at least t" ballot per slot. When the group holds
  exactly k words at least t (no excess ties at rank k, the random fp32 case) the prefix rank is the
  slot. Otherwise a warp-uniform branch (one any-vote at sub-warp scope) runs the two-class pass
  with the "above" ballots added.

Controls: `W1_eager_same` (the fp32 index-in-key u64 word in opt-W1b's code, the card's must-beat),
`W1_minmax_pairs` (the best not_guaranteed fp32 route, pairs with fmaxf and fminf),
`N10_ctrl_u64rec` (the u64 word in the same recursion as N10, isolating the code structure) and
`W1x_vimnmx_eager` on bf16 (the identical u32 network with the index packed, a network-only
reference measured in the same session).

## 2. Contract

Input: N fp32 keys striped over a logical warp of LW lanes (4, 8, 16, 32) in registers, the index of
item i of lane l is i x LW + l (the harness's `idx` array is not read). MaxK = max(LW, next_pow2(k))
divides N and N / MaxK is a power of two (the recursion's static_assert). Output: the k selected
source indices in the caller's array, unsorted (items above t in index order, then the tied items in
index order). Requirements: not_guaranteed and prefer_smaller_index, same code, same cycles (within
0.04 percent in every measured pair of cells). Temp storage none. Synchronization: shuffles, ballots
and one redux on the full warp mask, no barrier. Contract assumption: all lanes of the hardware warp
call the primitive together, the logical warps of a sub-warp instantiation run in lockstep (the
assumption T37 and T40 make, and the harness guarantees). NaN: a NaN twiddles above +inf (positive
sign) or below -inf (negative sign) and is selected or dropped deterministically by its sign bit,
focus-set inputs contain no NaN. -0.0 and +0.0 are one word. The truncated merge applies to a power
of two k, other k take the complete merge and the shuffle. Sorted output is not available from the
truncated forms (the complete-merge form has the sorted keys in position but the recovery does not
place indices by rank).

## 3. Where the time goes

Measured decomposition at warp scope, fp32, prefer_smaller_index, latency cycles per call (batch 1,
umbriel-b200-020, all in one session):

| cell | network only (bf16 W1x_vimnmx_eager, same u32 network) | + twiddle, shuffle t, two-class recovery (`N10_u32`) | truncated merge (`N10_u32_trunc`) | level order (`N10_u32_ltrunc`) |
|---|---|---|---|---|
| N 32 k 4 | 512 | 570 | 504 | 504 |
| N 32 k 16 | 512 | 570 | 442 | 442 |
| N 64 k 32 | 695 | 815 | 661 | 678 |
| N 128 k 16 | 1017 | 1238 | 1106 | 953 |
| N 128 k 64 | 912 | 1110 | 933 | 952 |
| N 256 k 16 | 1718 | 2105 | 1966 | 1422 |
| N 256 k 64 | 1322 | 1704 | 1566 | 1435 |

- The network is the W1x min and max network (T42) and costs what it costs on bf16 words: 512, 695,
  1017, 1718 cycles at N 32 to 256 for k at most 32, one shuffle and two predicated VIMNMX per
  shuffle stage. The twiddle of the fp32 key is three ALU operations per item before the network.
- The two-class recovery after a complete merge costs 58, 120, 221, 387 cycles at IPT 1, 2, 4, 8
  (the difference of the two first columns): about 48 cycles per slot from IPT 4 on, well above the
  pipelined-ballot model (27 + 6 per further ballot, T17), because the SASS interleaves the second
  ballot of each slot with the popc and add chains of the earlier slots (two VOTE, four POPC, two
  LOP3, four IADD, one ISETP, one IMAD and one predicated STS per slot, about 19 instructions, issue
  bound on one warp).
- Truncating the final merge saves 30 to 35 cycles per skipped shuffle stage (two stages at k 4,
  four at k 16, five at k 32 and six at MaxK 64) and replaces the shuffle of position k - 1 by a
  redux.min of the same cost: 66, 128, 154, 132, 177, 139, 138 cycles in the rows above.
- The level order gains nothing at one or two tiles (same dependency graph, 0 to 17 cycles of
  schedule noise) and 13 percent at four tiles (N 128, 953 against 1106), 27 percent at eight
  (N 256, 1422 against 1966), as the W1x codegen experiment predicted for M 1.
- The one-ballot recovery (batch 2, `N10_ge1_trunc` against `N10_u32_trunc`): plus 58 to 65 cycles
  at IPT 1 and 2 (the branch, the popc sum and the reconvergence cost more than the two ballots
  they save), minus 53 to 81 at IPT 4 and minus 104 to 129 at IPT 8. On the level order at N 128 the
  two recoveries tie (925 to 992 against 918 to 1013), at N 256 the one-ballot form wins 10 to 12
  percent latency (1240 to 1304 against 1386 to 1480) and 23 percent throughput (339 to 344
  against 276 to 280 Gelem/s).
- Price of excess ties at rank k (the two-class pass behind the branch, patterns pivot_tie and
  all_equal against random): N 64 k 16 781 against 747, N 128 k 16 1149 against 1053, N 256 k 16
  2126 against 1862 cycles, that is 34 to 264 cycles or 5 to 14 percent when every item ties, 45 to
  166 cycles above the branch-free two-class form. Random fp32 never takes it (ties at rank k 1.0
  in every measured cell).
- Registers: 30 to 64 in the latency kernel at warp scope (W1_eager_same 24 to 59), 23 to 40 in the
  throughput kernel, occupancy 8 blocks per SM up to N 128 and 6 at N 256 for the ge1 forms (8 for
  the two-class forms). Sub-warp N 128: 79 (LW 16), 118 (LW 8) and 168 (LW 4, 32 items per lane)
  registers in the latency kernel, throughput occupancy 4, 2 and 1 blocks, one 32-byte stack frame
  with 11 STL at LW 4 N 128 k 4 in the level form. No local memory anywhere else (T36 check: 0
  stack, 0 STL, 0 LDL in every other cell).

## 4. Ideas tried, with verdicts

Numbers are prefer_smaller_index latency cycles per call unless marked, the control in the same lock
section. Cards `cards/N10_*_b200_20261025_1.json`.

### 4.1 Keys-only u32 network plus two-class recovery after a complete merge (`N10_u32`)

Against the u64 word: 0.95x (N 32), 0.87x (N 64), 0.65x (N 128 k 16), 0.50x (N 128 k 64), 0.55x
(N 256 k 16), 0.37x (N 256 k 64), throughput 1.33x to 1.77x. Both must-beat cells pass (815 at N 64
k 32, 1238 at N 128 k 16). Verdict: the idea card's hypothesis holds, and the measured recovery is
58, 120, 221, 387 cycles at IPT 1, 2, 4, 8 against the card's 110 + 8 IPT model: cheaper at IPT 1,
dearer from IPT 4 on (section 3).

### 4.2 Truncated final merge with redux threshold (`N10_u32_trunc`)

Saves 66 to 177 cycles per call (2 to 10 percent) in every cell against 4.1, 30 to 35 cycles per
skipped shuffle stage, the redux costs what the position shuffle cost. Against the u64 word 0.84x to
0.34x, against the not_guaranteed pairs 0.95x to 0.34x, throughput 1.40x to 1.82x over the u64 word.
Verdict: adopted. Catalogue entry T51.

### 4.3 Level tile order (`N10_u32_ltrunc`, `N10_ge1_ltrunc`)

Nothing at one or two tiles (same dependency graph, 0 to 17 cycles of noise), 13 percent at four
tiles (N 128), 27 to 31 percent at eight (N 256), 28 to 37 percent at sub-warp LW 8 and 16 N 128.
Throughput: plus 1.5 percent at N 128, minus 4 percent at N 256 with the two-class recovery, plus 24
percent at N 256 with the one-ballot recovery (a schedule effect of the eager one-ballot kernel, not
a property of the order). Verdict: adopted from four tiles on. This is the M 1 latency case the W1x
codegen experiment flagged for reopening R12, confirmed on fp32 words.

### 4.4 One-ballot recovery with the tie pass behind a uniform branch (`N10_ge1_trunc`)

Against the two-class pass: plus 51 to 65 cycles at IPT 1 and 2 (the branch and the popc sum cost
more than the saved ballots), minus 53 to 81 at IPT 4, minus 102 to 129 at IPT 8, throughput plus 1
to 11 percent up to N 128 and minus 5 percent at N 256 (occupancy 8 to 6). Tie cost (pivot_tie and
all_equal against random): plus 34, 96, 264 cycles at N 64, 128, 256 for the one-ballot form, zero
for the two-class form. Verdict: adopted from IPT 4 on. Catalogue entry T52 carries both forms.

### 4.5 The combined adapter (`N10_auto`, `N10_auto_sub`)

Truncated merge always, level order from four tiles, two-class recovery up to IPT 2 and one-ballot
recovery above, every switch a compile-time function of N, k and LW. Measured on a second node
(umbriel-b200-028), it reproduces the best measured form of every cell to the cycle. Warp scope
against the u64 word: 0.84x, 0.74x (N 32 k 4, 16), 0.80x, 0.75x, 0.71x (N 64 k 4, 16, 32), 0.52x,
0.50x, 0.48x, 0.43x (N 128 k 4 to 64), 0.34x, 0.33x, 0.32x, 0.29x (N 256 k 4 to 64), throughput
1.40x to 2.1x. Against the not_guaranteed pairs 0.99x to 0.27x. Sub-warp against the u64 word:
LW 16 0.65x to 0.24x, LW 8 0.69x to 0.18x (N 128 k 8: 1409 against 5299), LW 4 0.87x to 0.23x,
N 8 1.06x to 1.23x (a loss). Against the sub-warp pairs: LW 16 and LW 8 0.82x to 0.25x, LW 4 N 32
k 4 and 8 1.20x and 1.28x (a loss), LW 4 N 128 0.73x to 0.35x. Verdict: the fp32
prefer_smaller_index route of the W1 family at warp scope for N 32 to 256 and at sub-warp scope from
N 32 (LW 8, 16) or N 128 (LW 4) on, and the fastest fp32 route measured under either requirement
there.

### 4.6 Controls that explain the gain

`N10_ctrl_u64rec` (the u64 word in the same recursion): 0.77x to 0.37x of it, so the keys-only
network plus recovery wins with the code structure held fixed. `W1x_vimnmx_eager` on bf16 (the same
u32 network, index packed): N10's network costs exactly that, the whole fp32 overhead is the twiddle
and the recovery. The W1 family's fp32 u64 word is slower than the bf16 u32 word by 1.6x to 2.2x at
N 128 to 256 (two shuffles and a compare-select chain per stage), N10 removes that gap and the tie
problem at once.

## 5. Ideas not yet tried

- An asm barrier between the ballot block and the popc block of the two-class pass, to reach the
  pipelined-ballot cost (would save up to 100 cycles at IPT 4 on that form, which the combined
  adapter no longer uses there).
- Sorted output by one shuffle per selected item into the network positions (needs the complete
  merge, 4.1), for the sorted-output pass.
- Sub-warp LW 4: a two-step shuffle-xor min inside the segment instead of eight pipelined full-warp
  redux, and two slots per ballot word (16 free lanes per vote at LW 4 and 8).
- Register cut at IPT 16 and 32 (sub-warp N 128): recompute the twiddled originals after the
  network or run the recovery slot by slot without the ballot arrays.
- Two rows per call (the N5 rows arrangement) for throughput, the recovery is per row.
- sm_100f: redux.sync.min.f32 would allow the native float min and max network without the twiddle.
- GB200 cross-check of the recovery (the network is host independent by T42).

## 6. Determinism and ordering

Deterministic: a fixed network and a rank pass with no atomics. prefer_smaller_index is exact by
construction (items above t in index order, then tied items in index order until k) and costs the
same as not_guaranteed in every measured cell (within 0.04 percent, same instantiation). The price
of the requirement relative to the best not_guaranteed fp32 route of wave 1 is negative at every
measured warp cell (0.99x to 0.27x of the pairs) and at every sub-warp cell except LW 4 N 32 k 4 and
8 (1.20x, 1.28x) and N 8 (1.27x to 1.53x). Output order: class then index, not sorted by key.
prefer_larger_index would be the mirror (lanes above me in the rank) and was not built.

## 7. Open items

- The two-class pass is issue bound at about 50 cycles per item per lane in the measured SASS, the
  pipelined-ballot model predicts half, and whether a scheduling barrier recovers it is untested.
- N 8 and LW 4 N 32 lose to the pairs network, the route table should keep the pairs there under
  not_guaranteed and N10 under prefer_smaller_index.
- Registers at sub-warp N 128 (118 at LW 8, 168 at LW 4 in the latency kernel, throughput occupancy
  2 and 1 blocks per SM) are a register-tier risk for batched router kernels, the device kernel has
  to be measured (BRIEF section 4).
- The eager one-ballot kernel's throughput loss at N 256 (272 against 339 for the level order at the
  same occupancy) is unexplained, a schedule effect of the T28 kind.
- The card schema's input_pattern enum has no all_equal value, the all_equal cells of the tie
  sections are quoted in the findings and dropped from the cards.

## 8. Pointers

- Code: `approaches/N10/n10_detail.cuh` (twiddle, common_net with the truncated merge, eager_net,
  level_net, recover_two_class, recover_fast_path, adapter_impl, auto_adapter),
  `harness/adapters/N10_{u32,u32_trunc,u32_ltrunc,ge1_trunc,ge1_ltrunc,auto}{,_sub}.cuh`, the
  control `N10_ctrl_u64rec{,_sub}.cuh`. Network stage machinery from `approaches/W1/
  warp_level_topk.cuh` and `w1_common.cuh` (unchanged).
- Node workflow `approaches/N10/n10_node.sh`, batch scripts under the scratchpad (in the batch
  logs on the nodes as `batch1.sh` to `batch3.sh`), host model of the recovery and the truncation
  `approaches/N10/model.py`.
- Findings `approaches/N10/findings_*_b200.json` (generated by `make_findings.py` from
  `findings_spec_b12.py` and `findings_spec_b3.py`), catalogue drafts `tech_truncated_merge.md`
  (T51) and `tech_ballot_rank_recovery.md` (T52).
- Batches `state/results/opt-N10/*b1_*`, `*b2_*`, `*b3_*` (b1 and b2 on umbriel-b200-020, b3 on
  umbriel-b200-028), smoke batches `*smoke*` (oracle only).
- Cards `cards/N10_{u32,u32_trunc,u32_ltrunc,ge1_trunc,ge1_ltrunc,ge1_trunc_sub,ge1_ltrunc_sub,auto,auto_sub}_b200_20261025_1.json`.
- Related: `approaches/W1/REPORT_W1b.md` (the fp32 routes of wave 1), `approaches/W1/
  CODEGEN_EXPERIMENT.md` (T42 and the level order), `ideas/queue/N10_fp32_keys_only_index_recovery.md`
  (now `approaches/N10/CARD.md`), TECHNIQUES.md T1, T4, T7, T17, T28, T36, T37, T40, T42, T51, T52.
