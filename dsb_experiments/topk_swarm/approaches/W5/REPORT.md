# W5: threshold filter with a certified lower bound, compaction and a packed W2 finish (warp), with N7 (two-level per-lane list for W2)

Optimizer: opt-W5. Branch `exp/topk-swarm/W5`, worktree `cccl-topk-swarm-W5`. Cards `cards/W5_base_b200_20261025_1.json`,
`cards/W5_v2_b200_20261025_1.json`, `cards/W5_v3_b200_20261025_1.json`, `cards/W5_v4_b200_20261025_1.json` (the W5
result), `cards/N7_twolevel_b200_20261025_1.json`. Batches under `state/results/opt-W5/`. All
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

N7, third build (card N7_twolevel_b200_20261025_1, batch n7c, blocks of 2): bf16 N 1024 k 64 4623 against W2 7308
(1.58x), k 128 8382 against 11872, N 2048 k 32 3009 against 4381, k 64 5371 against 10071, k 128 9451 against 21324
(2.26x), fp32 N 1024 k 64 3980 against 6774, N 2048 k 16 2267 against 4391, k 32 2843 against 7353 (2.59x), k 64 5357
against 10187. Losses at IPT 16 (0.72x to 0.79x) and bf16 N 1024 k 16 (0.92x). The round is 45 to 74 cycles against 27
for W2's pruned list: the refill is now rare (the per-round cost no longer scales with IPT) and the rest is block control,
one vote, two taken branches and runtime round indices per block of 2 rounds (SASS in the card, T49). The kill
criterion of the N7 card (2x under W2 at bf16 N 1024 k 64, at most 3300, within 10 percent of W5) is not met: 4623 is
1.58x and 36 percent behind W5_v2's 3397 and W5_v3's 3303 in the same session.

W5_v3 (card W5_v3_b200_20261025_1, controls W5_v2 and W2_packed_v3): predicate-free flags (sign bit of a - t),
integer count tree, unconditional stores with a per-lane dummy slot. Phase profile (`phase_probe3.cu`) at bf16 N 1024
k 32: loads 258, top 2 162, threshold 80, count 98 (v2 314), prefix 216, scatter 332 (v2 341), gather 440 (v2 446), sort
74, rounds 837, total 2497 (v2 2748). Harness: bf16 N 1024 k 8 1173 against v2 1358, k 32 2116 against 2279 and W2
2481 (1.17x), k 64 3303 against 7308 (2.21x), N 2048 k 16 2018 against 2377 (1.18x), k 32 2675 against 4381 (1.64x),
k 64 4427 against 10071 (2.27x), fp32 N 2048 k 16 2256 against 4391 (1.95x), k 32 3296 against 7354 (2.23x). Registers
128 at N 1024 and 249 to 255 at N 2048 (v2 72 and 128 to 168), throughput 1 to 15 percent under v2 and 1.31x to 2.64x
over W2. The scatter costs about 10 cycles per shared store instruction at every IPT and whether or not the stores are
predicated, and the bf16 gather costs 430 to 530 cycles against 170 to 220 for fp32 (W2's twiddle with its compare per
gathered word and the position predicate), which v4 attacks.

W5_v4 (card W5_v4_b200_20261025_1, controls W5_v3 and W2_packed_v3, both requirements): minus zero normalized at load
by one FADD, compare-free twiddle, unconditional gather loads masked by the sign of p - tot. Against v3 the bf16 cells
drop 190 to 400 cycles (N 1024 k 32 1837 against 2116, k 64 2936 against 3303, N 2048 k 32 2485 against 2675), fp32
is unchanged. Against W2_packed_v3: bf16 N 1024 k 16 1337 against 1485 (1.11x), k 32 1837 against 2481 (1.35x), k 64
2936 against 7308 (2.49x), N 2048 k 16 1861 against 2377 (1.28x), k 32 2485 against 4381 (1.76x), k 64 4163 against
10071 (2.42x), fp32 N 1024 k 16 1774 against 1547 (0.87x), k 32 2741 against 3059 (1.12x), k 64 4737 against 6774
(1.43x), N 2048 k 16 2293 against 4391 (1.92x), k 32 3296 against 7353 (2.23x), k 64 5975 against 10188 (1.70x).
prefer_smaller_index identical to not_guaranteed in all 12 cell pairs. Throughput 1.31x to 2.79x over W2 everywhere.
Registers 64, 127 to 128 and 254 to 255 at N 512, 1024, 2048.

Verdicts against the thresholds of PRIORITIES.md. W5: the kill cell bf16 N 1024 k 32 is 1837, under 1900 (1.35x
against the required 1.3x), fp32 N 2048 k 16 is 2293, under 3300 (1.92x), no measured input overflows the finish,
so W5 passes. It takes bf16 from k 16 at N 1024 and N 2048, fp32 from k 32 at N 1024 and from k 16 at N 2048, and
every throughput cell from k 16 at N 512 and above. W2_packed_v3 keeps k 8, fp32 N 1024 k 16 and the latency cells at
N 512 (W5_v4 N 512 k 32 1612 against 1468 across sessions). N7: 4623 at bf16 N 1024 k 64 against the 3300 threshold,
1.58x against the required 2x, 36 percent behind W5 in the same session, not passed.

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
6. N7 rolled with blocks of 2 (batch n7c, carded): 1.58x at bf16 N 1024 k 64, the round 45 to 74 cycles, kill criterion
   (2x, within 10 percent of W5) not met. N7 is closed in this form, W5 owns the k 64 band at N 1024 to 2048, N7 keeps a
   claim on fp32 N 1024 k 32 and N 2048 k 16 (2227 and 2267 across sessions against W5_v3's 2737 and 2256) and on the
   k 128 cells where W5 has no measurement yet.
7. W5_v3 predicate-free filter (batches v3a, v3b): 2 to 27 percent latency gain over v2 from the count phase (314 to 98
   cycles), the kill cell at 1.17x (2116 against 1900), registers up, throughput slightly down. Kept as the control of v4.
8. W5_v4 predicate-free gather (batches v4a, v4b): 13 to 15 percent off every bf16 cell, the kill cell at 1.35x (1837),
   W5 passes both thresholds. The W5 result.

## 5. Ideas not yet tried

- The shared store instructions of the scatter cost about 10 cycles each on one warp (32 per call at IPT 32, 64 at
  IPT 64, the same whether predicated or unconditional): two candidates per 64-bit store when the lane's run start is
  even, or a fused count-and-scatter into per-lane private runs with a prefix-free gather, would halve or remove them.
- The per-lane top 2 (160 cycles at IPT 32) by a cheaper pair tree, the ballot prefix (216) by fewer count bits.
- Registers: v3 and v4 hoist the flag and address arithmetic of every item before the stores (128 at N 1024, 255 at
  N 2048 against 72 and 128 for v2), a per-chunk ordering pin (T24) would give the throughput regime its occupancy back.
- N7: two blocks per loop iteration with the refill out of line (two copies, one back-edge per 4 rounds), or a head of
  16 with blocks of 4, to bring the round from 45 to about 35 cycles, and a same-session A/B against W5 at fp32 N 1024
  k 32 and N 2048 k 16 where N7 looked faster across sessions.

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

- Measurement batches: eleven run.py invocations in eight node scripts (base1 with its prefer_smaller_index run, n7a,
  v2a and v2b, n7b, n7c, v3a and v3b, v4a and v4b), three more than the eight the launch prompt allowed, the extra
  ones being the two N7 rebuilds and the decisive v4 (the kill cell flipped from 1.17x to 1.35x on it).
- The fp32 kernels at N 2048 sit at 254 to 255 registers from v3 on (0 spill), occupancy 1 in the throughput regime.
- N 4096 (IPT 128), the k 128 band and a GB200 card are not measured. The seed check says the configured bounds hold
  there (284 of 512 at k 128).
- The phase profilers (`phase_probe.cu`, `phase_probe3.cu`) run outside the harness under the GPU lock and are the
  only numbers in this report that are not harness batches.

## 8. Pointers

`ideas/queue/W5_threshold_filter.md`, `ideas/queue/N7_two_level_list_warp.md`, `TECHNIQUES.md` T1, T3, T4, T7, T12,
T17, T18, T19, T21, T28, T32, T36, T41 (adopted), T48 and T49 (contributed), `reference/GVR_V2_ALGORITHM_REPORT.md`
sections 2.2 to 2.7, cards W2_packed_v3, W5_base, W5_v2, W5_v3, N7_twolevel.
