# W2: redux based iterative extraction on packed words (warp, sub-warp, and the B5 block composition)

Optimizer: opt-W2. Branch `exp/topk-swarm/W2`, worktree `cccl-topk-swarm-W2`. All numbers below are B200 (sm_100 build,
nvcc 13.4.92, 2026-10-25) from the cards `cards/W2_packed_b200_20261025_1.json` (nodes umbriel-b200-068 and 091),
`W2_packed_v2_b200_20261025_1.json`, `W2_packed_v3_b200_20261025_1.json`, `W2_packed_sw_b200_20261025_1.json` (091),
`W2_packed_sw_v2_b200_20261025_1.json` (047), `B5_hier_b200_20261025_1.json` (091, control B1) and
`B5_hier_b200_20261025_2.json` (047, control B2_packed). Batches under `state/results/opt-W2/`. Commits dfeeab4138
(baseline sources) and 28239d1a4b (v2, v3, sw_v2). Eight measurement batches were run, the stop rule of the launch prompt.

## 1. What it is

Every item becomes one unique 32-bit word `key_field | inverted_index` that orders exactly like the pair (key
descending, index ascending). bf16 keeps all 16 twiddled key bits above a 16-bit inverted index. fp32 keeps the top
`32 - ceil(log2 N)` twiddled key bits above the index (idea N4). Each lane packs its items and selects its L largest in
sorted order with a per-lane network (L = next power of two of min(k, items per lane)), then the group runs k rounds
of `redux.sync.max` over the list heads. The lane whose head equals the maximum is the unique winner and shifts its
list by one with predicated selects. The winner index is decoded from the uniform maximum and stored by lane r mod T,
off the critical path. For fp32 one more redux after the rounds (the largest word not emitted) decides whether the tie
class of the k-th truncated key was cut, and a slow path of value rounds over the class members restores the exact
result in that case (0 to 3 percent of random inputs at N up to 4096, `approaches/W2/n4_tie_sim.py`).

Files: `harness/adapters/W2_impl.cuh` (the primitive, policies v1 to v3 and the two sub-warp policies),
`W2_packed.cuh` (warp baseline), `W2_packed_v2.cuh` (next-word tie check), `W2_packed_v3.cuh` (plus odd-even merge
sort blocks), `W2_packed_sw.cuh` (sub-warp baseline, masked redux), `W2_packed_sw_v2.cuh` (full-warp redux per logical
warp), `W2_packed_sw_v3.cuh` (built, not measured), `B5_hier.cuh` (block composition), `B2_packed.cuh` (opt-B2's
adapter, copied unchanged from branch exp/topk-swarm/B2 commit 8212d9e0d2 as the B5 control),
`approaches/W2/n4_tie_sim.py` and `n4_tie_sim.out` (N4 host simulation), `approaches/W2/sass_funcs.py` (SASS counts),
`approaches/W2/build_baseline.sh` and `build_variants.sh` (node-side build scripts), `approaches/W2/findings/*.json`
(the findings behind every card), `approaches/W2/cards/` (copies of the cards).

## 2. Contract

Input blocked in registers (lane t holds items t x IPT to t x IPT + IPT - 1), output k source indices in shared memory
in descending key order with ties by smaller index (the fp32 slow path may reorder the last slots of the tie class).
No temp storage at warp and sub-warp scope, Warps x k words plus 4 counters at block scope. k in [1, N/2], N a power
of two multiple of the group size up to 65536 (bf16) or 32768 (fp32), T in {4, 8, 16, 32} or a block of 128 to 512.
Both tie-break requirements run the same code. Sync: redux.sync on the full warp mask (one per round plus one for
fp32), no barriers at warp scope, one barrier per call at block scope (three for fp32). NaN: a NaN with the sign bit
clear ranks above +inf, with the sign bit set below -inf, the all-ones pattern may collide with the empty-list
sentinel. The oracle battery (random, tie_heavy, sorted, pivot_tie, all_equal, signed_zero, inf_denorm, quantized)
passed on every one of the 680 cells measured across the seven cards.

## 3. Where the time goes

Measured against the harness's generate-only control (subtracted), warp scope, bf16, v3 unless stated:

| cell | cycles | split |
|---|---|---|
| N 32 k 4 (IPT 1) | 124 (v1) | 4 rounds at 25 plus about 25 of emission and chain |
| N 256 k 8 (IPT 8, list 8) | 330 | sort8 plus 8 rounds at 25 plus about 100 |
| N 512 k 16 (IPT 16, list 16) | 820 | one sort16 plus 16 rounds at 25 (400) plus about 300 of prologue |
| N 1024 k 16 (IPT 32, list 16) | 1485 | two sort16 and one top-16 merge (about 900) plus 16 rounds (400) plus emission |
| N 2048 k 16 (IPT 64, list 16) | 2377 | four sort16 and three merges (about 1800) plus the same 16 rounds |
| N 1024 k 64 (IPT 32, list 32) | 6628 | 64 rounds at about 89 (every round shifts the whole 32-deep list) plus prologue |

The round is 24.9 to 29.6 cycles inside the full primitive (slope over k at N 256, baseline batch), the same as the
isolated T19 number, and it stays there at any N as long as k is at most 32 because the compiler drops the shifts of
list positions that cannot reach the head before the last round (T38). The N dependence is entirely in the per-lane
prologue: about 30 cycles per item for bf16 at IPT 16 to 64 with the odd-even blocks. Above the list depth every round
shifts the whole list and costs the T19 figures (50 at L 16, 89 at L 32, 147 at L 64), which is where W2 hands over
to W5 and N1 (k 64 and above at N 512 and above).

fp32 adds, with the v2 check, 80 to 115 cycles at IPT 8 (N 256), 77 at N 1024 k 16, for the 32-bit pack (3 more
instructions per item), the spare tracking in the merges and one redux. The v1 counting pass cost 234 to 676.

Registers (throughput kernel, 256 threads): bf16 19 at N 32, 32 at N 128 to 256, 40 at N 512, 63 at N 1024 k 16, 100
at N 2048 k 16 (occupancy 8, 8, 6, 4, 2). fp32 v3: 40, 64, 113, 186 (occupancy 6, 4, 2, 1). No spills at warp scope
except fp32 N 2048 (160 to 232 bytes with v2 and v3, 0 to 56 with v1) and 20 bytes at fp32 N 1024 k 32 with v2.

## 4. Ideas tried, with verdicts

1. Packed round inside the full primitive (the approach card's kill experiment, explore item 1). Verdict: confirmed.
   Round 24.9 to 29.6 cycles at IPT 8 against the 40-cycle kill threshold, registers 31 to 42, no spills. W2_packed
   beats W1_reference in every bf16 warp cell of the grid up to k 256: N 32 k 4 124 against 778, N 128 k 16 476 against
   2267, N 512 k 16 889 against 12606, N 2048 k 16 2629 against 76075, N 512 k 256 12362 against 22889. The only warp
   cell W1 wins is fp32 N 2048 k 1024 (131668 against 154812).
2. N4 truncated-key packing for fp32 (attached idea, kill experiment: host simulation of the tie class, then the pass
   cost). Verdict: confirmed at warp scope. Tie class of the k-th truncated key on normal fp32 data: mean 1.00 to 1.06,
   slow path 0.0 to 1 percent at N 256 to 2048 with 21 to 24 key bits, 3 percent at N 4096 k 16 with 20 bits, 10 to 25
   percent only at 16 bits (N 32768). The v1 counting pass cost 234 to 304 cycles at N 256 and up to 1319 at N 2048,
   more than the card's 150, and it held fp32 at occupancy 1 to 2. The v2 next-word check (one redux after the rounds
   over max(list head, largest word dropped in the prologue merge)) brings the fp32 overhead to 80 to 115 cycles at
   N 256 and 77 at N 1024 k 16 and the registers to the bf16 level plus 8 to 50, gains over v1 of 1.45x (N 256 k 4),
   1.30x (k 8, k 16), 1.85x (N 512 k 4), 1.21x (N 1024 k 16), throughput 1.13 to 1.26x from the occupancy. Negative:
   fp32 N 1024 k 8 (0.74x), N 2048 k 16 (0.85x) and k 32 (0.83x) got slower under v2 because the slow path's IPT
   twiddled keys and member flags are provisioned in the latency kernel (spills at N 2048), see open items.
3. Odd-even merge sort blocks in the per-lane prologue (v3, 19 comparators at 8 against 24, 63 at 16 against 80).
   Verdict: pays 3 to 21 percent on bf16 latency (N 256 k 8 1.11x, N 512 k 16 1.08x, N 1024 k 8 to 32 1.08x, N 2048
   k 16 1.11x, k 32 1.21x) and 3 to 10 percent on throughput, 0 to 9 percent on fp32, nothing at k 4 (sort4 is 5
   comparators either way). The comparator count transfers one to one into VIMNMX count and cycles.
4. Sub-warp scope with a masked redux.sync (W2_packed_sw baseline). Verdict: wrong primitive. nvcc 13.4 lowers a
   redux.sync whose member mask differs between the logical warps of one hardware warp to a BSSY, WARPSYNC, BRA, BSYNC
   loop over the distinct masks, one serialized pass per group of about 85 cycles: rounds of 174, 327 and 649 cycles
   at 2, 4 and 8 groups per warp (T37).
5. One full-warp redux per logical warp with the identity in the other lanes (sw_v2). Verdict: 2.2x to 7.1x in
   latency over the masked form, 1.1x to 1.7x in throughput, round 35 cycles at 2 and 4 groups, 88 at 8 groups. bf16
   T 8 N 32 k 4 295 cycles, T 8 N 128 k 8 (gpt-oss, Qwen3 routers) 802, T 16 N 128 k 16 782, T 4 N 8 k 4 369.
6. B5 hierarchical composition (attached idea, kill experiment: 1.15x over B2 at block N 1024 k 16 bf16). Verdict:
   marginal pass on latency, fail on throughput, close for the focus set. Against B2_packed in one lock section: N 1024
   k 16 bf16 1194 against 1380 cycles (1.16x) and 361 against 494 Gelem/s (0.73x). B5 wins latency only at N 1024
   k 8 to 32 (1.10 to 1.16x) and N 16384 k 4 (1.32x), B2 wins every other cell in latency and every cell but one in
   throughput (B2 is 21 registers, B5 28 to 48, and B5's merge runs on warp 0 while 7 warps wait). Against B1 (AIR)
   B5 is 1.58x at N 1024 k 16 bf16, 2.05x at N 16384 k 16, and loses from k 32 (N 1024) or k 64. B5's one barrier per
   call against B2's k barriers does not show in throughput because B2's rounds keep all 8 warps issuing.
7. prefer_smaller_index (measured as a requirement, not an idea): identical cycles to not_guaranteed in every cell of
   every card (the packed order is the tie-break), see section 6.

## 5. Ideas not yet tried

- Outlined or reload-based fp32 slow path, to remove the provisioned IPT arrays that regress fp32 N 1024 k 8 and
  N 2048 k 16 to 32 under v2 and v3 (the fast path never touches them). Expected to recover the 500 to 1150 cycles
  lost there and the 160 to 232 bytes of spill.
- k + 1 rounds instead of spare tracking (the (k + 1)-th word is the next word for free at one extra 25-cycle round,
  removing one min and max per merged item in the prologue).
- Two-level per-lane list for k above 32 at IPT 32 and above (approach card explore item 2): sorted top L plus an
  unsorted remainder refilled by a lane max when the top drains. The rounds at k 64 cost 89 to 147 cycles because the
  whole list shifts, a 16-deep top with refills would keep them near 30 at an occasional refill.
- A selection network for the L largest of IPT items that skips the full sort of each block, and a half-cleaner merge,
  for the IPT 32 to 128 prologue (now 60 to 75 percent of the cell at N 1024 to 2048 k 16).
- Two candidates per round (approach card explore item 5, a second redux over the lanes whose head was not the winner).
- sw_v3 (next-word check and odd-even blocks at sub-warp scope, built, not measured), and a 2-step shuffle tree at
  T 4 against the 8-redux round (88 cycles).
- An sm_100f variant is not useful for this approach: the packed word is an integer and redux.sync.u32 is what runs,
  T2's fp32 redux would only serve a value-only fp32 variant, which N4 made unnecessary.
- N 4096 warp cells (IPT 128): built (k 4 to 64, umbriel-b200-020, gen dir of batch grid_b) and not measured within
  the batch budget. The k 256 and 1024 translation units at IPT 128 compile for more than 20 minutes (T34 territory)
  and were dropped.
- GB200 cards: every lease of this round landed on B200 nodes. The primitive uses min, max, selects and redux only, so
  T20's compiler difference should not apply, unverified.

## 6. Determinism and ordering support

Inherent. The packed word is unique, every round is an exact arg-max, the output is sorted by key descending with
ties by smaller index. prefer_smaller_index runs the same code as not_guaranteed and measured identical to the cycle in
all 28 bf16 and 28 fp32 cell pairs of the warp grid (against W1_reference's not_guaranteed path through
`--control-req`, 124 against 124 at N 32 k 4 up to 150107 against 150109 at N 2048 k 1024) and in all 60 cell pairs
of the B5 against B2 batch (both adapters inherit it from the word). Measured cost of prefer_smaller_index: 0 cycles.
Sorted output is inherent and free, except that the fp32 slow path writes the tie class members of the last slots in
(full key, index) order, which is still the sorted order.

## 7. Open items

- The v2 and v3 fp32 regressions at N 1024 k 8 and N 2048 k 16 to 32 (section 4 item 2), a codegen issue of the slow
  path, not of the algorithm. Until fixed, v1 is the better fp32 variant at those three cells and v3 everywhere else.
- N 4096 warp cells and a GB200 card (section 5).
- B5 against B2 was measured with B2_packed at commit 8212d9e0d2 copied into this worktree, the B2 card of opt-B2 may
  carry a newer variant.
- Measurement hygiene: the variant batches on umbriel-b200-047 ran twice each (two batch files 38 to 43 seconds apart
  under the same holder and container, cause not found, the pairs agree to the cycle in latency and within 1.2 percent
  in throughput, one cell 5.5 percent), and the trailing `lease.py release` of the first run freed the lease at
  14:02:00 UTC while the second was still measuring under the flock. The cards use the second file of each pair and say
  so. Measurement commands should not chain the release after the batch in one shell command.
- The B5 fp32 tie pass at block scope still runs the v1 count with two extra barriers (384 cycles at N 1024 k 16).

## 8. Pointers

Approach card `ideas/queue/W2_redux_extraction.md` (copied to `approaches/W2/CARD.md`), idea cards
`ideas/queue/N4_truncated_key_packing.md` and `B5_hierarchical.md`, catalogue entries adopted T1, T3, T7, T11, T17,
T19, T20, T33, contributed T37 (sub-warp redux lowering) and T38 (list-depth pruning), reference
`reference/BLOCK_TOPK_RESULTS.md` sections 1 and 3, `harness/micro/intrinsics_micro.cu` row `W2PackedList`.
