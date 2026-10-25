# W5: threshold filter with a certified lower bound, compaction and a packed W2 finish (warp), with N7 (two-level per-lane list for W2)

Optimizer: opt-W5. Branch `exp/topk-swarm/W5`, worktree `cccl-topk-swarm-W5`. Cards `cards/W5_base_b200_20261025_1.json`,
`cards/W5_v2_b200_20261025_1.json`, `cards/N7_twolevel_b200_20261025_1.json`. Batches under `state/results/opt-W5/`. All
measurements on umbriel-b200-028 (B200, sm_100, nvcc 13.4.92, 2026-10-25), every oracle passed, every latency fit
R squared 1.0000, canary 32.001 cycles per op at the start and end of every batch, controls in the same lock section.

## 1. What it is

W5. Every lane holds IPT items of the row in a striped arrangement (lane t has items t, t + 32, ...). A certified lower
bound on the k-th largest key is built from grouped lane statistics: the 32 lanes form G groups of 32 / G adjacent lanes,
every lane selects its J largest keys with a small per-lane network, the group's J largest are formed with log2(32 / G)
xor-shuffle merge steps, and the threshold is the warp minimum (one redux) over the groups of the group's J-th largest.
Because G x J >= k, at least k keys are at or above the threshold, so the filter never under-selects and there is no
retry. One compare per item against the threshold gives the per-lane candidate count, log2(IPT + 1) ballots of the count
bits give the exclusive prefix over lanes and the total, every lane writes its candidates as one run into a shared
buffer of 32 x D words, and after a __syncwarp lane t reads positions t, t + 32, ..., so each lane holds at most D
candidates. The candidates are packed into W2's unique word (bf16: twiddled key above the inverted source index, fp32:
the top 32 - log2(32 D) twiddled key bits above the inverted buffer position, idea N4), sorted per lane with an odd-even
network, and the packed W2 rounds (one redux per round, T19, T41) emit the k largest words. fp32 keeps W2's next-word
check after the rounds and resolves the tie class of the k-th truncated key on full keys read from the buffer. When more
than 32 x D candidates pass (tie floods, periodic patterns, never on random input) an exact bit-serial radix select over
the register tile with ballot-ticket emission in index order takes over.

(G, J, D) per k, from the host simulation `sim_gj.py` (4000 trials per cell, striped lanes, word-space ranks, candidate
counts nearly independent of N): k <= 16 G = next_pow2(k), J 1, D 4 (k <= 8) or 8. k 17 to 32: 16x2, D 8 (80 candidates
on average, 198 to 212 max). k 33 to 64: 16x4, D 8 (126 mean, 241 to 259 max). k 65 to 128: 16x8, D 16 (208 to 214 mean,
354 to 388 max). `seed_check.cu` confirms that none of the 16 measured inputs of any cell overflows (worst 185 of 256 at
k 64, 284 of 512 at k 128).

Variants. `W5_base`: packs every item into the W2 word before the filter, serial count and position chains. `W5_v2`:
filter and bound on raw float keys (bf16 widened by a shift), the pack only for the at most D candidates per lane after
the gather, tree count, IPT / 8 scatter position chains, pair-tree top 2.

N7. W2's rounds on a two-level per-lane list: a sorted head of 8 words (positions 0 to 7 of the register tile) and an
unsorted remainder (the rest). The prologue sorts block 0 and merges every further block of 8 into the head with a
write-back merge (elementwise max with the reversed sorted block, the mins written back into the block, half-cleaner on
the head), so the remainder holds exactly the words not in the head. Rounds run in blocks of 4: at the top of block b
every lane tests whether its head was below full at the end of block b - 1 and still has remainder words, one
__any_sync decides, and the refill (every remainder block sorted again and merged into the head with write-back) runs
under that warp-uniform branch after the rounds of block b. Invariant: at the start of every block a lane has at least
4 valid head words or an empty remainder, so no lane runs dry inside a block, and the vote overlaps the block's rounds.
Files: `harness/adapters/W5_impl.cuh`, `W5_base.cuh`, `W5_impl_v2.cuh`, `W5_v2.cuh`, `N7_impl.cuh`, `N7_twolevel.cuh`,
`approaches/W5/sim_candidates.py`, `sim_gj.py`, `seed_check.cu`, `findings/*.json`.

## 2. Contract

W5: warp, 32 lanes, striped input in registers, N a power of two from 256 to 65536 (bf16) or 32768 (fp32), k in
[4, 128] with 2k <= N, keys bf16 or fp32, int32 indices derived from the position, both unsorted requirements by the
same code (the finish word and the fallback tickets are prefer_smaller_index by construction), output k indices in
descending key order on the filter path (fp32: by truncated key, the tie class of the k-th key rewritten in (key, index)
order), ticket order on the fallback. Temp storage 32 x D words (bf16) or pairs (fp32), 512 bytes to 4 KB per warp.
Sync: redux, ballot, shuffles over the full warp mask and one __syncwarp, no barriers. NaN as W2 (documented in
`W5_impl.cuh`), minus zero equals plus zero.

N7: as W2_packed_v3 (blocked input, no temp storage, redux.sync and one vote per 4 rounds), IPT >= 16, k a multiple of 4
in [4, 256].

## 3. Where the time goes (B200, cycles per call, control in the same lock section)

W5_base (card W5_base_b200_20261025_1, control W2_packed_v3, bf16 then fp32):
N 512 k 8 961 and 1036 (W2 571 and 678), k 16 1195 and 1536 (820 and 1119), k 32 1661 and 2428 (1468 and 2227), k 64
2628 and 4194 (2704 and 4711). N 1024 k 8 1401 and 1385 (986 and 1815), k 16 1618 and 1914 (1485 and 1547), k 32 2162
and 2865 (2481 and 3059), k 64 3219 and 4819 (7308 and 6774). N 2048 k 8 2317 and 2095 (1656 and 2735), k 16 2530 and
2616 (2377 and 4391), k 32 3093 and 3853 (4383 and 7354), k 64 5335 and 6626 (10071 and 10187). prefer_smaller_index
identical to not_guaranteed (2162 and 2865 at N 1024 k 32). Throughput (Gelem per s) bf16 N 1024 k 32 672 against 480,
k 64 458 against 209, N 2048 k 64 628 against 255, fp32 N 2048 k 32 810 against 516, k 64 531 against 230.

Structure of the base cost: the round is 27 cycles with the list of 4 or 8 at every N (slopes over k), the fixed part is
about 750 at IPT 16, 1190 at IPT 32 and 2100 at IPT 64 (about 27 cycles per item per lane). SASS of the bf16 throughput
kernel at N 1024 k 8: 1624 instructions against 664 for W2 (about 600 of them the cold fallback), the bf16 pack 8
instructions per item, the count and the scatter position two serial IADD chains of IPT steps.

W5_v2 (card W5_v2_b200_20261025_1, controls W5_base and W2_packed_v3 in two lock sections): the filter on raw float
keys with the pack after the gather removes about 300 instructions per call and moves the latency by at most 5 percent
either way (bf16 N 1024 k 32 2279 against base 2162 and W2 2481, N 2048 k 64 5104 against 5335 and 10071, fp32 N 2048
k 16 2592 against 2616 and 4392), while the throughput rises 4 to 34 percent over base (bf16 N 2048 k 8 1770 against
1323 Gelem per s). The fixed cost still grows by 23 cycles per lane item.

Phase profile of v2 (`approaches/W5/phase_probe.cu`, one warp, dependent clock stamps per phase, mean over 256 chained
calls on the 16 measured inputs, cycles): bf16 N 1024 k 32: loads 260, per-lane top 2 188, group merge and threshold 79,
compare and count 314, ballot prefix 187, scatter 341, syncwarp and gather 446, sort 70, rounds 863 (total 2748). bf16
N 2048 k 32: loads 450, top 2 326, threshold 75, count 836, prefix 193, scatter 465, gather 452, sort 74, rounds 840.
bf16 N 512 k 32: 159, 86, 79, 130, 162, 158, 435, 74, 847. So the compare-and-count and the scatter phases cost about
10 cycles per lane item each for 3 to 5 instructions per item, and the gather after the __syncwarp costs 430 to 520
cycles for 8 loads at every N, which is the time the 32 scattered stores take to drain before the first load may issue.
The SASS explains the first two: every per-item float compare produces a predicate, there are seven predicate
registers, and ptxas serializes compare and consume per item with P2R moves. v3 (sign-bit flags, integer count tree,
unconditional stores to a per-lane dummy slot) removes every per-item predicate.

N7, first two builds (batches n7a and n7b, not carded): with the block loop unrolled the refill network was inlined once
per block of rounds (3737 min max instructions in the throughput kernel against 382 for W2, 11368 in the latency
kernel) and the round cost 70 to 185 cycles (bf16 N 1024 k 64 9903 against W2 7308). With the loop rolled the kernel is
1000 instructions but the round still cost 70, 106 and 190 cycles at IPT 16, 32 and 64 (bf16 N 1024 k 64 7182 against
7308, N 512 k 64 4655 against 2704): the lookahead test "fewer than 2 B valid head words" with H 8 and B 4 fires as soon
as any lane has won once since its last refill, so the warp-uniform refill (sorting and merging every remainder block,
about 70 instructions per block of 8) ran in nearly every block, and the per-round cost scales with the tile. The third
build uses blocks of 2 (test: fewer than 4 valid, five wins of one lane).

## 4. Ideas tried, with verdicts

1. Baseline W5_base against W2_packed_v3 (batch base1): wins k 64 everywhere from N 512 (2.27x at bf16 N 1024 k 64,
   1.89x at N 2048), k 32 at N 2048 (1.42x and 1.91x), fp32 N 2048 k 16 (1.68x, focus cell 1 passed), every throughput
   cell from k 16; loses IPT 16 below k 64 and k 8 to 16 at N 1024 and 2048 bf16. The must-beat cell bf16 N 1024 k 32 is
   2162 against 2481 (1.15x), threshold 1900 not reached. Keep as the control of v2.
2. Grouped bound family (G, J) (host simulation): the min over 16 lane pairs of the pair's 2nd largest (80 candidates at
   k 32) replaces the exact 32-lane bound (122) at one shuffle step, and the exact k-th of the certified set (36) is not
   worth k redux rounds. Adopted.
3. N7 with an unrolled block loop (batch n7a): 9903 at bf16 N 1024 k 64 against W2 7308, the round at 70 to 185 cycles,
   because the refill network was inlined once per block of rounds (3737 min max instructions in the throughput kernel
   against 382 for W2, 11368 instructions in the latency kernel). Rejected, fixed by rolling the block loop (n7b).
4. N7 rolled with blocks of 4 (batch n7b): 7182 against 7308 at bf16 N 1024 k 64, the round still 70 to 190 cycles
   scaling with IPT, because the lookahead test fires after one win and the refill runs in nearly every block.
   Rejected, fixed by blocks of 2 (n7c).
5. W5_v2 raw-key filter (batches v2a, v2b): latency within 5 percent of base either way, throughput up 4 to 34 percent,
   the fixed cost unchanged at 23 cycles per lane item. Kept as the control of v3, the lesson is in the phase profile.

## 5. Ideas not yet tried

- Second rung of the fallback: the exact k-th largest of the certified set, then refilter, which keeps tie floods on the
  fast structure.
- Two bf16 keys per word in the filter (HSET2 compares, T18) for the bf16 pack-free path.
- N 4096 cells (IPT 128) against W1_reference, where W2 does not compile in reasonable time.
- An sm_100f variant with redux.sync.max.f32 for the bound (saves the twiddle of the per-lane bound, 18 cycles, T2),
  not worth a card.

## 6. Determinism and ordering

Exact by construction: the finish orders by the unique word, the fallback emits ties in index order. Both requirements
run the same code and measured identical cycles.

## 7. Open items

See the findings of each card.

## 8. Pointers

`ideas/queue/W5_threshold_filter.md`, `ideas/queue/N7_two_level_list_warp.md`, `TECHNIQUES.md` T1, T3, T4, T7, T17,
T18, T19, T21, T32, T36, T41, `reference/GVR_V2_ALGORITHM_REPORT.md` sections 2.2 to 2.7, cards W2_packed_v3.
