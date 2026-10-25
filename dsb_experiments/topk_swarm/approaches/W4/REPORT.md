# W4: k-truncated merge of sorted runs. Optimizer report (opt-W4, wave 2b, 2026-10-25)

All numbers are B200 (umbriel-b200-091, NVIDIA B200 sm_100), nvcc 13.4.92, build `-arch=sm_100`,
harness 0.8, random normal input (bf16 rounded), latency in cycles per call net of the
generate-only control (chain slope), throughput in Gelem per s over one occupancy wave, variant
and control in the same lock section. Cards under `cards/W4_*_b200_20261025_*.json`, raw batches
under `state/results/opt-W4/`. Code on branch `exp/topk-swarm/W4` (worktree
`/Users/estehle/Downloads/cccl-topk-swarm-W4`): `harness/adapters/W4_core.cuh` (the merge tree,
host compilable), `W4_impl.cuh` (device drivers), `W4_chunk4.cuh`, `W4_chunk8.cuh`,
`W4_wmerge.cuh` (adapters), `approaches/W4/host_test.cpp` (host emulation against std::sort),
`cells_w4.json`, `node.sh`, `findings_*.json`.

Verdict in one line: both bars of the card are met. The chunk merge of 4 sorted runs of 2048
(the DSA V3.2 merge shape) emits the sorted top 2048 of the 8192 candidates in 4266 cycles (bf16)
and 6571 (fp32) against B3_tuned's unsorted selection at 6501 and 7100 in the same lock section
(1.52x and 1.08x), 3.2x and 5.0x under B6's full sort, and about 3x under select plus post-sort,
so it is the cheapest sorted route for every merge cell measured (M 4 and 8, k 512 to 2048) and
beats the unsorted selection at M 4. The warp truncated merge sort gives sorted output at N 2048
k 1024 in 19786 (bf16) and 42598 (fp32) cycles against W2's 150078 and 154821 (7.6x and 3.6x, the
bar was 2x), takes every k 256 cell from W2 (1.24x to 3.0x), and loses to W5_v4 at k 64 by 1.9x to
3.4x as the card expected. Three builds in one afternoon took the warp form from 112403 to 42598
cycles at fp32 N 2048 k 1024 with the same algorithm (T62 and section 4).

## 1. What it is

Every item is one unsigned word whose integer order is (key descending, index ascending): fp32
keys become a 64-bit word (order preserving twiddle of the key in the high half, the bit-inverted
source index in the low half), bf16 keys a 32-bit word (16 twiddled key bits above the inverted
16-bit index). Minus zero is folded onto plus zero before the twiddle. The all-zero word is below
every real word and stands for an absent item.

The merge tree takes R sorted runs of RunLen words and merges adjacent pairs level by level,
every merged run truncated to out = min(k, 2 x in) items, so no level carries more than k items
per run and the last level emits the k largest of the union, sorted. A pair's output is split
evenly over the threads assigned to it: thread j owns output positions [j x OPT, (j + 1) x OPT).
It finds the merge-path split of its end diagonal with a binary search of a compile-time step
count (predicated body, the group stays converged, the static search of
`reference/WARP_MERGE_SORT_RESULTS.md` section 4), takes the start split from its predecessor
(one shared-memory exchange and barrier at block scope, one `__shfl_up_sync` at warp scope), and
so knows exactly how many items it consumes from each run. It merges them in chunks of C (16, or
OPT when smaller): up to C words from each side are loaded with the loads beyond the consumed
count predicated off (the register preset to the filler), the C largest of the 2C are taken with
one half-cleaner stage (also counting the wins of the first run) and a bitonic merge of log2(C)
stages on static register indices, the chunk is stored, and both heads advance by the counted
wins. The merge-path comparator takes the first run on ties, which is the word order because the
first run's indices are smaller, and the chunk network compares full words, so the result is
exact under both requirements. Runs live in shared memory in a layout swizzled inside aligned
groups of 16 (64-bit) or 32 (32-bit) words: the threads of a pair stride through a run at about
OPT / 2 words per thread, which unswizzled maps 16 threads onto two or four banks.

Two forms share the tree:

- `W4_chunk4` and `W4_chunk8` (block, 256 threads): the blocked tile of N keys is M = 4 or 8
  consecutive runs of N / M items, each sorted by (key descending, index ascending) as an exact
  per-chunk top-k emits them (the harness provides such input through `--runs M`, harness 0.8).
  Thread t stages its IPT items as words into run buffer 0, then log2(M) levels with 2 barriers
  each (split exchange, level boundary), 2 log2(M) + 1 barriers per call.
- `W4_wmerge` (warp, 32 lanes): lane l holds items [l x IPT, (l + 1) x IPT) (blocked), sorts them
  in registers with a Batcher odd-even merge sort (template recursion as B2, 543 exchanges at
  IPT 64), writes its run to buffer 0, then five merge levels truncated at k with one `__syncwarp`
  between levels and no barriers.

Level buffers alternate by parity (level l reads buffer l mod 2 and writes the other, the last
level writes indices), so the temp storage is two run buffers.

## 2. Contract

Keys bf16 or fp32, int32 indices derived from the blocked position, requirements not_guaranteed
and prefer_smaller_index by the same code (the word order is the tie-break), output k indices in
the harness's shared output array sorted by (key descending, index ascending), slot r holds rank
r, keys never materialized. NaN: twiddled bitwise, a positive NaN orders above +inf, a negative
NaN below -inf, the all-ones negative NaN collides with the filler word (focus-set inputs contain
no NaN). The input tile is not modified.

`W4_chunk<M>`: block of 256 threads, blocked input whose M runs of N / M consecutive items arrive
sorted (not checked, an unsorted run gives a wrong result), N / M a power of two and a multiple of
IPT, k a power of two with 2k <= N such that every level splits evenly (`w4::tree_valid`), N at
most 32768 (fp32) or 65536 (bf16), and the temp storage plus the harness's double-buffered k output
slots inside the 227 KB opt-in limit (`w4::chunk_temp_bytes`, so fp32 M 8 at k = N / 4 from N 16384
and bf16 M 8 at N 32768 k 8192 are reported as unsupported, phase 3 found them as failed launches
at 263680 and 329216 bytes before this check). Temp storage two word buffers plus 256 ints: 66960 bytes for fp32 N 8192 M 4 (50464 bf16), 75264 for fp32 N 8192 M 8, 38400 for fp32 N 4096 M 8.
Sync: 2 log2(M) + 1 `__syncthreads` per call, the caller separates consecutive calls by a block
barrier (the harness does).

`W4_wmerge`: one warp, blocked input of N = 32 x IPT keys with IPT a power of two in [2, 64] (the
per-lane network stays under the local-memory threshold of T34), k a power of two in [32, N / 2].
Temp storage two word buffers: 39936 bytes for fp32 N 2048 k 1024, 31744 for fp32 N 2048 k 64, 20064 for bf16 N 2048 k 1024, 9824 for bf16 N 512 k 256 (8 warps x the fp32 N 2048 storage exceed the 227 KB block limit of the harness throughput regime, those cells are latency only). Sync: six `__syncwarp` and five `__shfl_up_sync`
per call, no barriers, the caller separates consecutive calls by a `__syncwarp`.

Host test: `approaches/W4/host_test.cpp` runs the per-thread phases of the core in a loop over the
threads with the barriers emulated, on both word widths, six patterns (random, tie_heavy,
all_equal, signed_zero, sorted_asc, relu) and three seeds per shape, and demands the exact
reference order. 234 cases pass before the first node build.

## 3. Where the time goes

Every number here is from umbriel-b200-091 (B200), latency in cycles per call net of the
generate-only control, variant and control in one lock section, both requirements identical to
the cycle in every cell (the word order is the tie-break, the code is the same).

### 3.1 Block chunk merge (W4_chunk4, W4_chunk8), cards `W4_chunk4_b200_20261025_1`, `W4_chunk8_b200_20261025_1`

| cell (runs x run length, k) | W4 | B3_tuned (same section) | ratio | B6_radix (same section) | W4 throughput / B3 |
|---|---|---|---|---|---|
| bf16 N 8192, 4 x 2048, k 2048 | 4266 | 6501 | 1.52x | 13511 | 525 / 426 Gelem per s |
| fp32 N 8192, 4 x 2048, k 2048 | 6571 | 7100 | 1.08x | 32578 | 319 / 338 |
| bf16 N 8192, 8 x 1024, k 1024 | 5194 | 6318 | 1.22x | (B6 card 13505) | 448 / 449 |
| fp32 N 8192, 8 x 1024, k 1024 | 8151 | 6714 | 0.82x | (B6 card 32513) | 280 / 360 |
| bf16 N 4096, 8 x 512, k 512 | 3869 | 3790 | 0.98x | (B6 card 8068) | 369 / 516 |
| fp32 N 4096, 8 x 512, k 512 | 5230 | 4310 | 0.82x | (B6 card 21207) | 291 / 413 |

Phase profile (`approaches/W4/phase_probe.cu`, one block, 256 calls on 16 run-sorted random
inputs, clock64 stamps with a barrier after each phase, the stage phase also absorbs the input
loads): bf16 M 4 stage 1245, level 0 search plus split exchange 1497 and merge 1066, level 1 931
and 557 (5298 with the probe syncs). fp32 M 4: 2454, 1016 and 2693, 1083 and 1490 (8737). fp32 M 8:
2643, 1029 and 2759, 956 and 1508, 997 and 822 (10716). So at 256 threads the two or three static
searches with their exchange barrier are about 1000 cycles each (12 steps of two dependent loads,
about 85 cycles per step including the barrier) and are the largest phase for bf16, the chunk
merges are issue bound (every thread runs a 48 or 20 exchange network, 2 VIMNMX per exchange on
the bf16 word, 2 ISETP and 4 SEL on the fp32 word, T42 and T7) and cost 2.5x more for fp32, and
the third level of M 8 adds a flat 1800 cycles. Registers 116 to 123 (M 8 at N 4096: 64), no spill,
no stack, 13 and 17 BAR.SYNC in the latency kernel, 2928 to 4152 SASS instructions for two calls.
Temp storage 50 to 67 KB (M 4) and 30 to 75 KB (M 8), occupancy 2 (N 8192) or 4 blocks per SM.

### 3.2 Warp truncated merge sort (W4_wmerge), card `W4_wmerge_b200_20261025_1`

The final build (v4) against the sorted incumbents of PRIORITIES.md section 4:

| cell | W4_wmerge | control (same section) | ratio | throughput W4 / control |
|---|---|---|---|---|
| bf16 N 2048 k 1024 | 19786 | W2_packed 150078 | 7.59x | 102.7 / 16.1 |
| fp32 N 2048 k 1024 | 42598 | W2_packed 154818 | 3.63x | latency only (temp storage) |
| bf16 N 2048 k 256 | 13538 | W2_packed 40422 | 2.99x | 189 / 60 |
| fp32 N 2048 k 256 | 29581 | W2_packed 42915 | 1.45x | latency only |
| bf16 N 1024 k 256 | 8408 | W2_packed 21859 | 2.60x | 160 / 58 |
| fp32 N 1024 k 256 | 17190 | W2_packed 21315 | 1.24x | 70 / 58 |
| bf16 N 512 k 256 | 4875 | W2_packed 12361 | 2.54x | 145 / 53 |
| fp32 N 512 k 256 | 9327 | W2_packed 12616 | 1.35x | 62 / 53 |
| bf16 N 1024 k 64 | 5416 | W5_v4 2926 | 0.54x | 329 / 484 |
| bf16 N 2048 k 64 | 9036 | W5_v4 4177 | 0.46x | 363 / 711 |
| fp32 N 1024 k 64 | 11342 | W5_v4 4773 | 0.42x | 138 / 425 |
| fp32 N 2048 k 64 | 19146 | W5_v4 5532 | 0.29x | latency only |

Phase profile of v3 (the build before the layout change, one warp, 256 calls): bf16 N 2048 k 1024
prepare (words, Batcher sort of 64, run store) 5750, then per level search 509 to 792 and merge
3153 to 3178 (4 chunks of 16 outputs, about 790 per chunk), last level 1686, total 23437. fp32:
prepare 29442, merges 8595 to 8820 per level (2150 per chunk), total 73921. bf16 N 2048 k 64:
prepare 6553, merges 1891, 897, 407, 201, 133 (the truncation shrinks the outputs per lane to 32,
16, 8, 4, 2), searches about 500 each, total 12623. Profile of the final build (v4): bf16 N 2048 k 1024 prepare 6439, searches 451 to 1122,
merges 1864, 2389, 2932, 3581 and 1178 (total 21986). fp32: prepare 15697, merges 4657, 5424, 6513,
8791 and 2446 (47067). bf16 N 2048 k 64: prepare 6404 then merges 1140, 529, 310, 213, 65 (10914).
fp32 N 2048 k 64: prepare 14429 then 2530, 1264, 616, 314, 104 (21471). The merge of the same 64
outputs per lane grows from level 0 to level 3 (1864 to 3581 bf16): the skewed layout spreads the
runs over the banks, not the lanes of one pair, and at level 3 the 16 lanes of a pair read one run
at positions 32 words apart (16-way conflicts) where at level 0 only 2 lanes share a run.

Three things set the warp cost. The chunk merge is instruction bound at one warp: per chunk of
16 outputs the thread issues 32 (bf16) or 64 (fp32) shared loads with their address arithmetic, a
48-exchange network and 16 or 32 stores, and with one warp nothing hides the dependent latencies,
so a chunk costs 2 to 3 cycles per instruction. The fp32 per-lane sort of 64 words is 29k cycles
in the probe for 543 exchanges (3258 instructions, 9 cycles each): at 128 registers of words plus
the harness state the kernel sits at 255 registers and ptxas serializes the network. The searches
are a minor term (about 10 percent), unlike at block scope.

## 4. Ideas tried, with verdicts

1. v1, the first build (one session, controls B3_tuned, W2_packed, W5_v4). Block: bf16 N 8192
   k 2048 4251 against 6497, fp32 8123 against 7115. Warp: bf16 N 512 k 256 8340 against 12361,
   fp32 15987 against 12616, the fp32 N 2048 cells failed in the throughput regime (8 warps x 39 KB
   of temp storage above the 227 KB block limit). Verdict: the block merge beats the unsorted
   selection on bf16 at once, the warp form is slow and the fp32 word is the problem on both.
2. v2, swizzle on the absolute buffer position, fp32 key and index halves in separate arrays with
   the level 0 index implicit from the position (keys-only staging and probes), no per-run pad.
   The v1 swizzle was applied to the run-relative position, so at warp scope the 16 lane pairs of
   level 0 read 16 runs at the same relative position into the same banks (16-way conflict), while
   at block scope the pairs sit in different warps and nothing changed. Block fp32 8123 to 6571
   (1.08x over B3_tuned, temp storage 100 KB to 67 KB), bf16 4251 to 4266. Warp bf16 N 512 k 256
   8340 to 7160, N 2048 k 1024 measured for the first time: bf16 59199 (2.54x over W2), fp32 112403
   (1.38x, the bar missed). Kept.
3. v3, the chunk loop rolled (`#pragma unroll 1`, the body's register indices static) and a tree
   sum of the win flags (T62). bf16 N 2048 k 1024 59199 to 21783 (SASS 21080 to 10216 instructions,
   the 200-byte stack frame gone), fp32 112403 to 71573 (39048 to 19320), fp32 N 2048 k 256 83688 to
   56320, cells with one chunk per level unchanged. The fully unrolled four chunks per level made
   20k to 40k instruction kernels that paid instruction fetch, the same mechanism as T48. Kept.
4. W4_wmerge2, the per-lane sort in two halves of 32 with one more (search-free) merge level, as
   a cure for the 29k-cycle fp32 sort: against W4_wmerge (v3) 0.80x to 0.91x in 8 of 10 cells and
   1.31x and 1.06x at fp32 N 2048 k 256 and k 1024 (43137 against 56379, 67635 against 71743). The
   extra level costs more than the smaller network saves except where the 64-word fp32 sort is at
   its worst. Rerun against the final build (v4) in one section: 0.91x to 0.98x in all 10 cells (bf16 N 2048 k 1024 20779 against 19780, fp32 43326 against 42596), card `W4_wmerge2_b200_20261025_1`. Not adopted as the default.
5. v4, the warp form's layout and loads: linear addressing with an odd run stride (consecutive runs
   one bank apart, every access one instruction with an immediate offset) instead of the xor
   swizzle (two extra instructions per access), unconditional chunk loads into a 16-word zero pad
   after every run instead of loads predicated by the exact consumed count, so no end split, no
   split exchange, no shuffle. bf16 N 512 k 256 7095 to 4875, N 2048 k 1024 21783 to 19786, fp32
   N 512 k 256 14074 to 9327, N 1024 k 256 22221 to 17190, registers 168 to 255 down to 64 to 128.
   fp32 N 2048 k 1024 71573 to 42598 (3.63x over W2, the bar of 2x passed on both key types), fp32 N 2048 k 256 56320 to 29581 (now 1.45x over W2 where v3 lost 0.76x). At one warp the two or three address instructions per swizzled access and the predicate per load were the cost, not the bank conflicts they addressed. The block form keeps the swizzle and the exact counts (its lanes stride through
   one run, where the xor is what spreads them, and its loads are half the traffic at 8 warps).
6. W4_chunk4q, a 4-ary merge path for the block form (three probes per side per step, six loads
   in flight, 6 steps instead of 12): against W4_chunk4 in one section at N 8192 k 2048 bf16 5035 against 4250
   (0.84x), fp32 7387 against 6581 (0.89x), against B3_tuned 1.29x and 0.96x. 36 probe loads per
   thread instead of 24 through a shared memory pipe that 8 warps share, plus the select cascade and
   three address computations per step, cost more than the six saved dependent steps. The k-ary
   search loses at block scope at a range of 2048 as it lost at warp scope at small ranges in July
   (`reference/WARP_MERGE_SORT_RESULTS.md` section 14.1). Card `W4_chunk4q_b200_20261025_1`, not
   adopted.

## 5. Ideas not yet tried

- The fp32 per-lane sort at IPT 64 (29k of 72k cycles at N 2048 k 1024): sort in four pieces of
  16 and merge them with two lane-local levels (needs the tree to let one thread own several
  pairs), or a key-only u32 network with the index as payload and a stable tie rule, or an
  `sm_100f` build where the key half could stay a float and use FMNMX.
- A 4-way merge path per level for M 8 (two pairs materialized together), halving the level
  count and the barriers of the block form.
- STS.128 staging with a quad-preserving swizzle, and the level 1 words of fp32 as one 64-bit
  array (one LDS.64 per item instead of two LDS.32).
- N 4096 at warp scope (IPT 128): an in-lane merge level of two sorted 64-runs before the five
  cross-lane levels, so that the per-lane network stays at 64 (T34).
- The throughput regime of the warp form: the fp32 N 2048 cells exceed 227 KB at 8 warps per
  block, a 16-bit index half (N at most 65536) would bring fp32 N 2048 k 256 inside the limit.
- Runs shorter than N / M (a per-chunk top-k that returns fewer than k valid items): per-run
  lengths in the merge path and the pads.

## 6. The cheapest sorted route per cell (for the phase 3b sorted-output pass)

Cycles per call, B200, the W4 numbers from this session (umbriel-b200-091), the others from their
cards (B6_radix_b200_20261025_1 on umbriel-b200-020 for the sort of k words, W2_packed and
W1_reference on umbriel-b200-068, W5_v4 on umbriel-b200-028, B3_tuned from this session where
measured and from B3_tuned_b200_20261025_2 otherwise). Select plus post-sort is the unsorted
selection plus the B6 sort of the k selected words (and still needs an unmeasured gather of the
selected keys).

Block scope (sorted top-k of a tile, or of M sorted chunk results):

| cell | W4 merge of sorted runs | select + post-sort | full sort (B6) | cheapest |
|---|---|---|---|---|
| bf16 N 8192 k 2048, 4 runs | 4266 | B3 6501 + 5550 = 12051 | 13511 | W4 by 2.8x |
| fp32 N 8192 k 2048, 4 runs | 6571 | B3 7100 + 13487 = 20587 | 32578 | W4 by 3.1x |
| bf16 N 8192 k 1024, 8 runs | 5194 | B3 6318 + 4279 = 10597 | 13505 | W4 by 2.0x |
| fp32 N 8192 k 1024, 8 runs | 8151 | B3 6714 + 9767 = 16481 | 32513 | W4 by 2.0x |
| bf16 N 4096 k 512, 8 runs | 3869 | B3 3790 + 4279 = 8069 | 8068 | W4 by 2.1x |
| fp32 N 4096 k 512, 8 runs | 5230 | B3 4310 + 9767 = 14077 | 21207 | W4 by 2.7x |
| bf16 N 2048 k 1024, unsorted tile | no sorted runs | B3 2550 + 4279 = 6829 | 5550 | B6 (B6 report) |
| fp32 N 2048 k 1024, unsorted tile | no sorted runs | B3 2509 + 9767 = 12276 | 13487 | post-sort (B6 report) |

When the chunk results arrive sorted (W1, W5, W2 or B6 produce them sorted, B1 to B4 and N6 do
not), the merge of sorted runs is the cheapest sorted route at every merge cell by 2x to 3x and
at M 4 it also beats the unsorted radix select. For a single unsorted tile the B6 report's table
stands (B6's full sort for 16-bit keys at k = N/2, select plus post-sort otherwise).

Warp scope (sorted output of one row):

| cell | W4_wmerge | W2 (sorted by construction) | W5_v4 (sorted) | W1_reference | cheapest sorted |
|---|---|---|---|---|---|
| bf16 N 2048 k 1024 | 19786 | 150078 | not supported | 187659 | W4 by 7.6x |
| fp32 N 2048 k 1024 | 42598 | 154818 | not supported | 131668 | W4 by 3.1x over W1 |
| bf16 N 2048 k 256 | 13538 | 40422 | not supported | 145656 | W4 by 3.0x |
| fp32 N 2048 k 256 | 29581 | 42915 | not supported | 97060 | W4 by 1.45x |
| bf16 N 1024 k 256 | 8408 | 21859 | not supported | | W4 by 2.6x |
| fp32 N 1024 k 256 | 17190 | 21315 | not supported | | W4 by 1.24x |
| bf16 N 512 k 256 | 4875 | 12361 | not supported | | W4 by 2.5x |
| fp32 N 512 k 256 | 9327 | 12616 | not supported | | W4 by 1.35x |
| bf16 N 1024 k 64 | 5416 | 7308 | 2936 | | W5 |
| bf16 N 2048 k 64 | 9036 | 10071 | 4163 | | W5 |
| fp32 N 1024 k 64 | 11342 | 6774 | 4737 | | W5 |
| fp32 N 2048 k 64 | 19146 | 10188 | 5975 | | W5 |

So the sorted-output crossover at warp scope sits between k 64 and k 256: W5_v4 (sorted by its
finish word) up to k 64 at N 1024 and 2048, and W2 below that where W2 wins the unsorted cell
(k at most 32, N at most 512), W4_wmerge from k 256 up. Against the unsorted incumbents the
sorted premium is what W4 costs over N6 at warp scope (unmeasured here, N6 warp was running in
parallel), at k 1024 the premium against W2's own unsorted number is negative since W2 is sorted
by construction and 7.6x slower.

## 7. Determinism and ordering support

Inherent. The word order is (key descending, index ascending), the merge path takes the first run
on ties (its indices are smaller) and the chunk network compares whole words, so the output is the
exact prefer_smaller_index result in sorted order and both requirements run the same binary:
every pair of cells measured identical to the cycle (4266 against 4258, 19786 against 19787,
6571 against 6585). Minus zero equals plus zero (folded before the twiddle, verified by the
signed_zero pattern under the exact requirement). prefer_larger_index would store the index
uninverted (one instruction less per item) and swap the tie rule of the merge path to the second
run, not built.

## 8. Open items

- One lease on umbriel-b200-091 (holder opt-W4, 16:30 to 17:20Z), four build-and-measure
  rounds (v1 to v4) of 26 run.py batches in all (5, 7, 5 and 9) plus three phase-probe runs,
  every oracle passed, every fit R squared 1.0000, canary 32.001 cycles per op at the start and
  end of every batch.
- The fp32 warp sort (section 5) and the fp32 N 2048 throughput cells (temp storage).
- The chunk merge's contract needs the runs sorted and full. A device agent that produces chunk
  results with B1 to B4 or N6 (unsorted) would need W5, W2 or a sort for the chunks first, or
  the unsorted B3_tuned over the M k candidates (7100 against 6571 for fp32, 6501 against 4266
  for bf16 at M 4).
- No GB200 card. No sm_100f variant.
- The chunk cards were built from the v2 batches (commit 6f4b9c2190), the block code of v4 is the
  same algorithm (the rolled loop does not apply at one chunk per level, the exact-count path and
  the swizzle are unchanged), W4_chunk4q is its own card.

## 9. Pointers

- Cards: `cards/W4_chunk4_b200_20261025_1.json`, `cards/W4_chunk8_b200_20261025_1.json`,
  `cards/W4_wmerge_b200_20261025_1.json`, `cards/W4_wmerge2_b200_20261025_1.json`,
  `cards/W4_chunk4q_b200_20261025_1.json`.
- Batches under `state/results/opt-W4/`: `*_wsort_*`, `*_wk64_*`, `*_chunk4_*`, `*_chunk4b6_*`,
  `*_chunk8_*` (v1), `*_v2_*`, `*_v3_*`, `*_v4_*`.
- Code: worktree `/Users/estehle/Downloads/cccl-topk-swarm-W4`, branch `exp/topk-swarm/W4`,
  final commit 0ba868b00d. `harness/adapters/W4_core.cuh`, `W4_impl.cuh`, `W4_chunk4.cuh`,
  `W4_chunk8.cuh`, `W4_chunk4q.cuh`, `W4_wmerge.cuh`, `W4_wmerge2.cuh`, `approaches/W4/host_test.cpp`,
  `phase_probe.cu`, `celltab.py`, `cells_w4.json`, `node.sh`, `findings_*.json`,
  `techniques/t_rolled_chunk_loop.md` (T62).
- Catalogue: adopted T7, T28, T29, T34, T36, T42, T48, T49, T54, T55, contributed T62.
- Reference: `reference/WARP_MERGE_SORT_RESULTS.md` (static merge path, section 4, k-ary search,
  section 14.1), `approaches/B6/REPORT.md` section 6 (the sorted-output premium), card
  `ideas/queue/W4_merge_topk.md`.
