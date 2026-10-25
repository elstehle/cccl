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

FILL

## 4. Ideas tried, with verdicts

FILL

## 5. Ideas not yet tried

FILL

## 6. Determinism and ordering

FILL

## 7. Open items

FILL

## 8. Pointers

FILL
