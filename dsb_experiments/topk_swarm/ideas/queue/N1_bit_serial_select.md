# N1: Bit-serial ballot select (1-bit radix select without histogram or scan, warp and sub-warp)
status: queued (wave 2, share the optimizer with W3)
scope: warp, subwarp
origin: new (seed from the lead: one ballot count per key bit). Radix select with a 1-bit digit, where the "histogram" is a single popcount and the "scan" is one compare.
hypothesis: walk the twiddled key from the most significant bit down. Per round: each lane ballots (still a candidate and bit set) for each of its IPT items, sums the popcounts, and compares the count c with the number still needed r. If c is at least r the k-th key has the bit set, so candidates without it drop out. Otherwise all c items with the bit set are selected, r decreases by c, and candidates with the bit set drop out. 16 rounds for bf16, 32 for fp32, no shared memory, no atomics, no shuffle scan, cost independent of k, trivially sub-warp through the ballot mask. Index-only emission: selected items are known per round, the final tie class (items equal to the k-th key) gives exactly r items in slot-major ballot order, which is increasing index order for a striped arrangement, so prefer_smaller_index is free.
cells it should win: warp N 128 to 512, k 32 to 256, bf16 first. Sub-warp rows with k above 8. fp32 where W3 is not available or at N at most 256.

cost model (placeholders July 2026 B200: ballot 23 latency, about 4 cycles issue per further ballot, popc and add 2 each, shuffle stage 44):
- Round(IPT) about 23 + 4 (IPT minus 1) + 4 IPT + 15 = 38 + 8 IPT (ballots pipelined, popc sum, compare, branch-free mask update of IPT candidate flags).
- Emission at the end: IPT ballots in slot-major order plus popc prefix, about 15 IPT + 50. `fns` (find n-th set bit) can replace the per-lane prefix for the tie class, hardware researcher measures it.
- bf16: about 16 (38 + 8 IPT) + 15 IPT + 50 = 660 + 143 IPT. N 128: 1230, N 256: 1800, N 512: 2950. fp32: about 1270 + 270 IPT, N 256: 3430.
- Against W3 at 4-bit digits (about 1000 + 88 IPT for bf16): N1 wins below IPT about 10 (N 320) by having no scan, W3 wins above by sharing the bucket counts. Against W2 (100 + k (82 + 2 IPT)): N1 wins at k above about 16 for N 256. Against W5 (1350 + 24 IPT for k at most 16, 2200 + 26 IPT for k 17 to 64): a tie band at N 256 to 512, k 16 to 32.
- Early termination (c equals r exactly, all candidates selected) is rare on bf16 logit data because of ties, model without it.

focus cells: 1. warp N 256 k 32 bf16, 2. warp N 512 k 64 bf16, 3. warp N 256 k 16 fp32, 4. sub-warp LW 8 N 128 k 16 bf16 (4 rows per warp), 5. warp N 128 k 32 fp32.
boundary cells: warp N 256 k 8 (W2 expected to win), warp N 2048 k 64 (W3 expected to win).

explore first: (1) round cost at IPT 1, 4, 8, 16 as a slope over rounds, one kernel. (2) emission with and without `fns`. (3) branch-free candidate flag update against a per-item candidate mask register per lane (IPT bits in one register, one LOP3 per round). (4) sub-warp with 4 and 8 rows per warp. (5) the d sweep together with W3 (d 1, 2, 4, 5) in one table.

determinism and ordering: set deterministic by construction. prefer_smaller_index free by slot-major emission. Sorted via N2 or W1 over k.

main risk: 16 dependent rounds, each a ballot latency plus a mask update, with the per-item ballot count equal to W3's. If ballots do not pipeline at 4 cycles issue (VOTE throughput on Blackwell is unmeasured), round cost at IPT 8 goes from about 100 to about 200 and W3 wins everywhere above N 128.

cheapest kill experiment: round cost at IPT 8 against the W3 4-bit pass at IPT 8 in one session. If round(8) is above 110 cycles or W3's pass is below 3.5 rounds, N1 has no cell of its own and becomes the d 1 row of W3's table.

references: `ideas/queue/W3_warp_radix_select.md`, `reference/WARP_RADIX_SORT_RESULTS.md` (ballot ranking at IPT 1 beats merge sort), `reference/BLOCK_TOPK_RESULTS.md` section 1
