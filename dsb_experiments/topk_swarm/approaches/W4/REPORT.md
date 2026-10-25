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

SECTION_VERDICT

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
most 32768 (fp32) or 65536 (bf16). Temp storage two word buffers plus 256 ints: SECTION_TS_CHUNK.
Sync: 2 log2(M) + 1 `__syncthreads` per call, the caller separates consecutive calls by a block
barrier (the harness does).

`W4_wmerge`: one warp, blocked input of N = 32 x IPT keys with IPT a power of two in [2, 64] (the
per-lane network stays under the local-memory threshold of T34), k a power of two in [32, N / 2].
Temp storage two word buffers: SECTION_TS_WMERGE. Sync: six `__syncwarp` and five `__shfl_up_sync`
per call, no barriers, the caller separates consecutive calls by a `__syncwarp`.

Host test: `approaches/W4/host_test.cpp` runs the per-thread phases of the core in a loop over the
threads with the barriers emulated, on both word widths, six patterns (random, tie_heavy,
all_equal, signed_zero, sorted_asc, relu) and three seeds per shape, and demands the exact
reference order. 234 cases pass before the first node build.

SECTION_REST
