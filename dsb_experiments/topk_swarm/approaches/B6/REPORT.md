# B6: sort when k is large. Optimizer report (opt-B6, wave 2, 2026-10-25)

All numbers are B200 (umbriel-b200-020, NVIDIA B200 sm_100, 148 SMs), nvcc 13.4.92, build
`-arch=sm_100`, harness 0.6, 256 threads, random normal input (bf16 rounded), requirement
`not_guaranteed` unless stated, latency in cycles per call net of the generate-only control
(chain slope), throughput in Gelem per s over one occupancy wave. Every number sits next to its
control from the same lock section in a card under `cards/B6_*_b200_20261025_*.json`, raw
batches under `state/results/opt-B6/`. Code is on branch `exp/topk-swarm/B6` (worktree
`/Users/estehle/Downloads/cccl-topk-swarm-B6`), source commit 971a398f42: adapters
`harness/adapters/B6_radix.cuh` and `B6_pairs.cuh`, cells `harness/cells_b6.json`, node helpers
`approaches/B6/node.sh`, findings files `approaches/B6/findings_*.json`, catalogue entry sources
`approaches/B6/techniques/`.

Verdict in one line: the kill cell of the idea card fails by a wide margin. A full
`cub::BlockRadixSort` of the packed key-index word is 2.18x slower than B3_tuned and 1.89x slower
than B1_idx at bf16 N 2048 k 1024 (threshold 1.2x of the better control), 5.37x and 4.71x at
fp32, and the sweep over N 1024 to 8192 and k from N/8 to N/2 finds no crossover (2.1x to 2.35x
bf16, 5.0x to 6.5x fp32). B6 is not a selection primitive. It keeps the sorted-output row of
phase 3b, where its number is the sort itself, and the chunk merge of sorted runs stays with W4.

## 1. What it is

Every item becomes one unsigned word that orders like (key descending, index ascending): for
bf16 a 32-bit word with the twiddled 16-bit key in the high half and the source index in the low
half, for fp32 a 64-bit word with the twiddled 32-bit key high and the index low. -0.0 is folded
onto +0.0 before twiddling. `cub::BlockRadixSort` sorts the words descending, keys only, over the
key bits alone (`begin_bit` 16 or 32): the LSD sort is stable and the blocked input is in index
order, so equal keys come out in ascending index order without sorting the index bits, 4 passes
of 4 bits for bf16 and 8 for fp32 (T55). `SortDescendingBlockedToStriped` leaves rank r in item
r / 256 of thread r mod 256, and the top k ranks are stored to the k output slots with one
conflict-free store per item, items whose whole slice lies beyond k dropped at compile time.

`B6_pairs` is the pairs form for comparison: the twiddled key (16 or 32 bits) as the sort key,
the index as the value.

## 2. Contract

Block scope, 128, 256 or 512 threads (256 measured), blocked input in registers, N = Threads x IPT
with IPT >= 1 and N <= 32768 (the 16-bit rank counters of `BlockRadixRank`), k in [1, N) at
compile time, keys bf16 or fp32, int32 indices from the blocked position. Requirements
`not_guaranteed` and `prefer_smaller_index` at the same cost (same binary). Output: k indices in
the caller's shared array, slot r holds rank r (key descending, index ascending), so the output is
sorted at no extra cost. Temp storage: `BlockRadixSort`'s, max(9216 bytes of rank counters at 256
threads, (N + N/32) x word bytes of exchange buffer): 9216 at bf16 N 2048, 16896 at fp32 N 2048,
33792 at bf16 N 8192, 67584 at fp32 N 8192. Sync: `__syncthreads` inside, about 7 per pass
(ranking 2, the 256-thread BlockScan 2, exchange 2, between passes 1), 26 BAR.SYNC in the latency
kernel with two inlined calls. The caller separates consecutive calls by a block barrier. The
input keys are left intact. NaN: twiddled bitwise, a positive NaN orders above +inf and a negative
NaN below -inf, never produced by focus-set inputs. `prefer_larger_index` is not built (it would
sort the inverted index bits as well).

## 3. Where the time goes

Per-pass cost of the stock sort (whole call over passes), bf16 32-bit words (4 passes) and fp32
64-bit words (8 passes):

| IPT (N at 256 threads) | bf16 per pass | fp32 per pass | bf16 call | fp32 call | B3_tuned bf16, fp32 (k = N/2) |
|---|---|---|---|---|---|
| 4 (1024) | 1070 | 1221 | 4279 | 9767 | 1879, 1946 |
| 8 (2048) | 1388 | 1686 | 5550 | 13487 | 2550, 2509 |
| 16 (4096) | 2075 | 2651 | 8068 | 21207 | 3427, 3476 |
| 32 (8192) | 3376 | 4064 | 13640 | 32192 | 6154, 4935 |

Least squares: 740 + 82 x IPT (bf16) and 815 + 100 x IPT (fp32) cycles per pass (T54).

- The fixed 740 to 815 per pass is the barrier chain (six BAR.SYNC at 28 cycles plus the arrival
  skew of 8 warps) and the 256-thread BlockScan over 9 packed counters per thread (SHFL.UP warp
  scans, an aggregate round trip through shared memory, two barriers), about 2.5 times the one
  packed scan of B3's emission (T29, about 300).
- The per-item 82 to 100 cycles per pass are shared-memory wavefronts summed over the 8 warps,
  not a dependent latency. Per item and pass: one LDS.U16 and one STS.U16 on the thread's digit
  counter (a chain through the aliasing counter pointers, the compiler cannot hoist item i+1's
  load above item i's store), one LDS.U16 for the rank, one scatter STS to a random rank (3 to
  4 way bank conflicts at 32 random addresses, the one-word-per-32 padding does not remove them)
  and one gather LDS. 8 warps x IPT x 4 to 5 wavefronts at one per cycle is 32 to 40 x IPT
  cycles per pass, the conflicts double it. At IPT 32 the per-item part is 2600 of 3376.
- The 64-bit word costs 15 to 20 percent more per pass than the 32-bit word (STS.64 and LDS.64
  in the exchange, SHF.R.U64 for the digit) and twice the passes.
- SASS of the bf16 N 2048 latency kernel (two inlined sort calls plus the harness): 1008
  instructions, 26 BAR.SYNC, 15 SHFL.UP, 56 LDS.U16, 24 STS.U16, 96 STS, 47 LDS, 0 ATOMS, 0 BSSY,
  0 STL, 0 LDL, stack 0. The pass loop is a runtime `while` over `begin_bit`, so the kernel is
  small and still 2.2x slower: the instruction count says nothing about a wavefront-bound kernel.
- Registers (latency kernel) bf16 48, 56, 79, 128 and fp32 48, 64, 112, 206 at IPT 4 to 32, no
  spill. Throughput occupancy bf16 5, 4, 2, 2 and fp32 5, 4, 2, 1 blocks per SM (B3_tuned 6, 6,
  3, 2 and 6, 5, 2, 2 in its own card). The fp32 N 8192 kernel is at the register wall.
- Throughput: bf16 114 to 205 Gelem per s against B3_tuned 327 to 559 (0.35x to 0.44x), fp32
  45 to 68 against 332 to 471 (0.14x to 0.18x). The wavefronts are SM-wide, so the throughput
  loss matches the latency loss and occupancy does not help.

Why a radix sort cannot win a selection cell: a radix select (B1, B3) does 2 to 4 passes of one
histogram sweep over the tile with no data movement and one emission, a radix sort does 4 to 8
passes each with a ranking sweep and a full exchange of every item through shared memory. Even at
k = N/2 the select never touches the non-selected half again after pass 1, the sort moves
everything 4 to 8 times. The idea card charged a pass at 350 to 400 cycles, which is the cost of
one block scan, not of a ranking pass with an exchange.

## 4. Ideas tried, with verdicts

| variant | control | latency (cycles), B200 | throughput (Gelem per s) | verdict |
|---|---|---|---|---|
| B6_radix (packed word, key bits only, striped output), the kill cell bf16 N 2048 k 1024 | B3_tuned | 5550 against 2550 (2.18x slower), fp32 13487 against 2509 (5.37x) | 159 against 450, fp32 59 against 456 | killed, threshold 1.2x missed by 1.8x and 4.5x (card B6_radix_b200_20261025_1) |
| same | B1_idx (`--control-req not_guaranteed` for the deterministic cells) | 5550 against 2933 (1.89x), fp32 13499 against 2869 (4.71x) | 159 against 342, 59 against 337 | killed against the weaker control as well (card B6_radix_b200_20261025_2) |
| B6_radix at N 1024 k 512, N 4096 k 512 and 2048, N 8192 k 2048 and 4096 | B3_tuned | bf16 2.10x to 2.35x, fp32 5.02x to 6.52x slower, k-flat on both sides | 0.31x to 0.44x, fp32 0.13x to 0.20x | no crossover anywhere in the block range (same card) |
| B6_pairs (pairs sort, index as value) | B6_radix | bf16 6878 against 5550 (1.24x slower), fp32 13398 against 13496 (0.99x, tie) | 121 against 159, 62.3 against 59.4 | the packed word wins for 16-bit keys through the saved value exchange, the 64-bit word buys nothing for fp32 (card B6_pairs_b200_20261025_1, T55) |
| prefer_smaller_index against not_guaranteed | same binary | identical to the cycle (5550.8 against 5550.5) | identical | free, inherent through the stability of the sort |

Not built, by the plan's stop rule (the kill cell failed): the crossover table against B1_idx,
the partial-sort variant (it could save at most the non-selected part of the last exchange, a few
percent of one pass in 4 to 8), 512 threads at N 4096 to 8192 (halves the per-item part of the
pass, about 1.5x on the sort at N 8192 by the fit, adds 16 cycles per barrier, cannot reach a
2.2x gap), wider digits (the raking segment grows with the digit width, the fixed part is already
2.5 scans). The one extra batch beyond the kill cell is the N sweep at k = N/2 (and k = N/8 at
N 4096 and 8192), which section 6 needs.

## 5. Ideas not yet tried

1. A ranking pass without the shared-memory counter chain (a per-thread register histogram of the
   16 digits over IPT items, one packed block scan as in T29) and a conflict-free exchange
   (warp-level striped layouts, two half exchanges): removes the per-item wavefront cost and most
   of the fixed cost. Even at the B3 scan's 300 cycles plus 20 per item per pass it would give
   about 1840 cycles at bf16 N 2048, 0.72x of B3_tuned's selection. Worth building only for the
   sorted-output pass, never for selection.
2. The chunk merge of M sorted runs of k (explore item 2 of the card, the DSA merge cells of
   `ideas/WORKLOADS.md` section 7): one merge-path partition per thread into M runs and a serial
   merge of k / 256 items per thread, O(k log2 M). It needs sorted inputs, which only W1, W2 and
   this sort produce, and belongs with W4's sorted-input merge. The bound it has to beat at the
   DSA V3.2 merge cell (fp32 N 8192 k 2048): 32.2k to 32.5k for the full sort and 4935 to 6477 for
   B3_tuned unsorted.
3. A 32-bit truncated word for fp32 (top 32 - log2 N key bits plus the index, B2's N4 idea) with a
   tie class check after the sort: halves the exchange bytes for fp32, at 5x behind the select not
   worth it for selection, possibly for the sorted-output pass.

## 6. Sorted output through this family (for the phase 3b pass)

B6's output is sorted by key descending then index ascending at zero extra cost, so its number is
the sorted-output number, and the premium over B3_tuned's unsorted selection is the whole gap:
bf16 2400 cycles at N 1024 k 512, 3000 at N 2048 k 1024, 4640 at N 4096 k 2048, 7490 at N 8192
k 4096, fp32 7820, 10980, 17730, 27240.

The alternative route is B3_tuned plus a post-sort of the k selected words with the same
`BlockRadixSort` (the sort of k words at 256 threads is the N = k row of section 3, the selected
keys assumed available):

| cell | full sort | B3_tuned + sort of k | better |
|---|---|---|---|
| bf16 N 2048 k 1024 | 5550 | 2550 + 4279 = 6829 | full sort by 19 percent |
| bf16 N 4096 k 2048 | 8068 | 3427 + 5550 = 8977 | full sort by 10 percent |
| bf16 N 8192 k 4096 | 13640 | 6154 + 8068 = 14222 | full sort by 4 percent |
| bf16 N 4096 k 1024 | 8068 | 3427 + 4279 = 7706 | post-sort by 5 percent |
| bf16 N 8192 k 2048 | 13505 | 6328 + 5550 = 11878 | post-sort by 14 percent |
| fp32 N 2048 k 1024 | 13487 | 2509 + 9767 = 12276 | post-sort by 10 percent |
| fp32 N 4096 k 2048 | 21207 | 3476 + 13487 = 16963 | post-sort by 25 percent |
| fp32 N 8192 k 4096 | 32192 | 4935 + 21207 = 26142 | post-sort by 23 percent |

So for 16-bit keys at k = N/2 the full sort is the cheaper sorted route (the sort of k words at
IPT 4 still pays the 740-cycle fixed cost of every pass), at k at most N/4 and for fp32 at every
k the post-sort route is cheaper. The post-sort route additionally needs the selected keys (a
gather from a shared copy of the tile or from B3's candidate buffer), unmeasured, so its real cost
is higher than the sums above by one gather. The third route, W4's merge of sorted chunk runs,
is unmeasured at block scope. The phase 3b pass should run all three in one session.

## 7. Determinism and ordering support

Set deterministic, index deterministic and sorted by construction: the stable LSD sort over the
key bits with the blocked input in index order gives (key descending, index ascending) exactly.
`prefer_smaller_index` is the same binary as `not_guaranteed` and measured identical to the cycle
in every pair, the exact oracle passes on tie_heavy (1.1 to 10.4 ties at the k-th key on the bf16
cells), signed_zero (2k zeros of both signs, after the -0.0 fold), pivot_tie and all_equal.
`prefer_larger_index` needs the inverted index bits sorted as well (8 passes for bf16) or the
stable order reversed, not built.

## 8. Open items

- The idea card `ideas/queue/B6_sort_when_k_large.md` should move to the rejected or closed state
  for the selection cells with this verdict, keeping the sorted-output row (phase 3b) and pointing
  the chunk merge at W4 (the lead moves queue cards).
- No verification yet. The three cards are unverified and the kill verdict rests on one node, one
  session, four batches with flat canaries. Radix kernels were node invariant in wave 1 and the
  margin is 1.8x, so a rerun is cheap insurance, not a doubt.
- The gather of the selected keys for the post-sort route and the W4 merge are the two unmeasured
  parts of section 6.

## 9. Pointers

- Cards (B200, 2026-10-25, author opt-B6, commit 971a398f42): `cards/B6_radix_b200_20261025_1.json`
  (against B3_tuned, kill cell plus the N sweep, 48 cells), `cards/B6_radix_b200_20261025_2.json`
  (against B1_idx, 8 cells), `cards/B6_pairs_b200_20261025_1.json` (B6_pairs against B6_radix,
  8 cells).
- Raw batches: `state/results/opt-B6/B6_radix_vs_B3_tuned_b6_kill_b3_20261025_155509.json`,
  `B6_radix_vs_B1_idx_b6_kill_b1_20261025_155518.json`,
  `B6_pairs_vs_B6_radix_b6_kill_pairs_20261025_155526.json`,
  `B6_radix_vs_B3_tuned_b6_sortcost_20261025_155741.json`, plus the `.build.json` files. Per-cell
  JSON, ptxas output and SASS dumps on the node under `/work/topk_swarm/work/opt-B6/{logs,gen}`.
- Catalogue entries: T54 (block radix sort pass cost model, sort never a selection primitive),
  T55 (key bits only with the index in the word, stability gives the tie order). Sources under
  `approaches/B6/techniques/`.
- Code: `harness/adapters/B6_radix.cuh`, `harness/adapters/B6_pairs.cuh`, `harness/cells_b6.json`,
  `approaches/B6/node.sh` (ship, build, measure, fetch under the holder name opt-B6).
- Prior material: `ideas/queue/B6_sort_when_k_large.md`, `ideas/WORKLOADS.md` sections 3 and 7,
  `cccl-exp-perf/dsb_experiments/HYBRID_BLOCK_SORT_RESULTS.md` (stock BlockMergeSort at 2048
  fp32 keys 11684 cycles and pairs 14795, the same band as this radix sort), `cub/cub/block/
  block_radix_sort.cuh` and `block_radix_rank.cuh`.
