# W2: redux based iterative extraction on packed words (warp, sub-warp, and the B5 block composition)

Optimizer: opt-W2. Branch `exp/topk-swarm/W2`, worktree `cccl-topk-swarm-W2`. Cards under `cards/W2_*.json`
and `cards/B5_*.json`, batches under `state/results/opt-W2/`. Status: in progress, sections are filled as
batches land.

## 1. What it is

Every item becomes one unique 32-bit word `key_field | inverted_index` that orders exactly like the pair
(key descending, index ascending). bf16 keeps all 16 twiddled key bits above a 16-bit inverted index. fp32
keeps the top `32 - ceil(log2 N)` bits of the twiddled key (idea N4) above the index. Each lane packs its
items, selects its L largest in sorted order with a per-lane network (L = next power of two of min(k, items
per lane)), and the group runs k rounds of `redux.sync.max` over the list heads. The lane whose head equals
the maximum is the unique winner and shifts its list by one with predicated selects. The winner index is
decoded from the uniform maximum and stored by lane r mod T. For fp32 a tie pass after the rounds restores
the exact result inside the tie class of the k-th truncated key.

Files: `harness/adapters/W2_impl.cuh` (the primitive, policies v1 to v3), `W2_packed.cuh` (warp baseline),
`W2_packed_v2.cuh`, `W2_packed_v3.cuh`, `W2_packed_sw.cuh` (sub-warp), `B5_hier.cuh` (block composition),
`approaches/W2/n4_tie_sim.py` (N4 host simulation), `approaches/W2/sass_funcs.py` (SASS statistics aid).

## 2. Contract

See the adapter headers. Input blocked in registers, output k indices in shared memory in descending key
order, no temp storage at warp and sub-warp scope, `Warps x k` words at block scope. Both tie-break
requirements run the same code (the packed order is prefer_smaller_index by construction). NaN behavior in
`W2_impl.cuh`. k in [1, N/2], N a power of two multiple of the group size up to 65536 (bf16) or 32768 (fp32).

## 3. Where the time goes

(filled from the batches)

## 4. Ideas tried, with verdicts

(filled from the batches)

## 5. Ideas not yet tried

(filled at the end)

## 6. Determinism and ordering

Inherent: the packed word is unique, every round is an exact arg-max, the output is sorted by (key
descending, index ascending). prefer_smaller_index costs nothing extra, measured in the grid batch.

## 7. Open items

(filled at the end)

## 8. Pointers

`ideas/queue/W2_redux_extraction.md` (approach card, copied to `approaches/W2/CARD.md`), `N4_truncated_key_packing.md`,
`B5_hierarchical.md`, `TECHNIQUES.md` T1, T3, T7, T17, T19, T20, `reference/BLOCK_TOPK_RESULTS.md` sections 1 and 3.
