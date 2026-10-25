# Priorities for the queue (algo-researcher, 2026-10-25, before any card has numbers)

Basis: the cost models in the cards, built on the July 2026 B200 placeholders (one design point each) and the workload check in `WORKLOADS.md`. Everything here is a prediction to be overturned by the first result cards. One line of reasoning per card.

## Wave 1 (five optimizers, in the order I would fill them)

1. B1 with N6 as its first experiment. The incumbent and every fallback, and N6 (equi-width bins over the value bracket) is the only idea that cuts the bf16 pass count below 2, a 15 to 40 percent lever at every block N on the model. Same code, one extractor swap, host simulation first.
2. W2 with N4 attached. Owns the 2026 router shape (N 256 to 896, k 6 to 16) on the model at about 100 + k (82 + 2 IPT). N4 (truncated key plus index in one word) makes fp32 as cheap as bf16 and removes the index pass. Its kill experiment is one slope.
3. B2 with N4 attached. Owns block k at most 16 to 32 up to N 8192 on the model, prefer_smaller_index for free in packed form, and N4 removes the 890-cycle fp32 epilogue that made it lose to the optimized AIR in July.
4. B3. The only wave 1 card with a clean win predicted at N 32768 (15.5k against 24.5k for bf16, 35.7k for fp32), k-independent after pass 1, robust to sorted input. Honest kill at N 8192.
5. W1 with N2 as its restructuring. W1 eager is the reference that PR #12222 needs, but the model says it loses above N 2 MaxK to everything. N2 (sort all tiles at once, halve per level) is the same code family and predicted 1.3 to 2x faster than eager at N 64 to 256 with k 32. Run N2's kill experiment first so the optimizer's time goes to the faster structure.

Recommended changes to wave 1 as planned (decision 10): keep the five approach ids, attach N6 to B1, N4 to W2 and B2, N2 to W1, so no sixth optimizer is needed. If a sixth node is available, give N6 its own optimizer because it also has a warp variant (one pass against W3 and W5 at N 512 k 16) and its result decides the shape of wave 2.

## Wave 2 (in order)

1. B4. Predicted winner at block N 8192 and above on random input (4.8k against 15.5k for B3 at N 32768 k 256), compare based so fp32 is native, and its kill experiment is a host simulation. Promote to wave 1 if a node frees up.
2. W5. The exact lane-maxima bound makes it retry-free for k at most 32 (top 2 per lane to 64), predicted to tie W2 and N1 at N 256 k 16 and to beat W3 at N 512 k 32. Host simulation of the candidate count first, bf16 ties are its risk.
3. W3 with N1 as its d 1 row. Same ballots per item for every digit width, the width only trades per-pass fixed cost against the scan. N1 wins below IPT 10, W3 above, both beat W1 and W2 at k 32 and above. One optimizer, one table.
4. N6 at warp scope, if not done in wave 1. One bracket pass with 5 ballots per item against W3's 4 passes, predicted 1400 against 2400 at N 512 k 16 fp32.
5. B6, one cell first. The large k over N regime is narrower than the plan assumed (see `WORKLOADS.md` section 3) and B1 is nearly k-flat, so B6's value is the sorted-input chunk merge and the phase 3b sorted output. Kill at N 1024 k 512 bf16.
6. N5, throughput only. Two rows per lane with bf16x2 min and max, up to 2x on network cost where networks are the right algorithm (k 32 to 64, N at most 256). Kill with the keys-only sort at N 64.
7. B5, one cell. Predicted 1.4 to 2x slower than the flat block primitives because the warp round costs the block round. One confirming measurement with N1 per warp, then close.
8. N3, one cell against B4. Streaming contract only, predicted 1.3 to 1.6x slower than B4 on resident input. Keep if the device agent wants "insert chunk, keep k".
9. W4, last. Crossover mapping and the sorted-input multiway merge of M chunk results, which is its one predicted win.

## Where I expect the winner to flip (the cells the evaluation engineer should make dense)

Warp scope, bf16 and fp32:
- N 256: k 8 W2, k 16 a three-way band (W2, W5, N1 within 10 percent on the model), k 32 and above N1, with W3 taking over above N 320 (IPT 10).
- N at most 64: W1 or N2 at k 16 to 32 (one or two tiles), W2 at k 8 and below.
- fp32 against bf16: with N4, W2 and B2 lose their fp32 penalty, so the type flips nothing in extraction cells. Without N4, W5 and N2 (native compares) take the fp32 rows that W2 holds for bf16.
- prefer_smaller_index: free for packed W2, B2, N4 and for slot-major emission in N1 and W3, about one compare per stage for W1 and N2 (or free with bf16 index-in-key), plus 1 percent for B1. Expect no flips except where W1 pairs lose to N2 packed.

Block scope:
- N 1024 bf16: k 16 B2 (about 1640 against 1700 for B1 and about 1500 for N6), k 32 and above B1 or N6. The B2 to B1 crossover sits near k 18 on the model and moves to k 40 at N 8192.
- N 8192 and above: B4 on random input, B3 where the input is sorted or the bound is loose, N6 as the one-pass robust option, B1 as the fallback everyone pays on tie floods.
- k at least N over 4 (the chunk merges, H2O, SnapKV at 4096): B1 against B6, B1 expected unless the inputs are sorted runs (then W4 or B6's multiway merge).
- Latency against throughput: B2 at 19 registers and 192 bytes of smem and B1 at 32 registers both reach 8 blocks per SM, so the regime should not flip the winner at block scope. At warp scope the data-oblivious networks (W1, N2, N5) have no smem and win throughput at small N even where W2 wins latency (the July sort studies showed the same pattern).
- Arch: the B2 round depends on the MIO combining same-address bursts, unverified on GB200. If GB200 rounds are 2x slower, every B2 cell flips to B1 or N6 there.

## What I ask of the other roles

- Evaluation engineer: bf16 inputs from a logit-like distribution with the tie multiplicity at rank k reported (`WORKLOADS.md` section 8), the named cells of `named_cells.json`, and dense grids on the flip bands above.
- Hardware researcher: ballot issue rate and `redux.sync` f32 availability (W2, N1, W3), `match.any` cost (`rejected/R6`), `fns` cost (N1), `set` or compare on bf16x2 and integer SIMD min and max (N5), the shared atomic burst on GB200 (B2).
- Lead: attach N2, N4, N6 to the wave 1 optimizers as described, and read `WORKLOADS.md` section 9 for the five corrections to PLAN.md 3.1.
