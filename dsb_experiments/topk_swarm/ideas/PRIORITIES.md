# Priorities for the queue (algo-researcher, 2026-10-25, revised with the hardware researcher's measured intrinsic costs, before any result card)

Basis: the cost models in the cards. Intrinsic costs are now measured on B200 and GB200 (`TECHNIQUES.md` T1 to T18, 2026-10-25), primitive compositions (sort stages, AIR phases, rounds) are still July 2026 placeholders at one design point. Everything here is a prediction to be overturned by the first result cards. One line of reasoning per card.

What the measured costs changed since the first version of this file: (a) a packed W2 round needs no ballot and no ffs (the word is unique, the winner lane recognizes itself), and the hardware researcher measured the full list round at 25.8 cycles up to IPT 8, 37.2 at IPT 16 and 89.4 at IPT 32 (T19) against 74.4 for a block atomic round, which widens W2 to k 32 at N up to 512 and k 64 at N 256, and revives B5 at 52 k against 74 k. (b) Ballots pipeline at 6 cycles each, not 4, which pushes N1 from the k 32 band to k 64 to 128 and W3 to k 128 and above. (c) `redux.sync` f32 is sm_100f only, so fp32 goes through N4 or a 40-cycle twiddled round. (d) `match.any` is 51 to 392 cycles and `fns` is emulated, both closed. (e) bf16x2 min, max and per-half masks are 6-cycle single instructions, N5's model holds. (f) The B2 round is identical on GB200, that risk is retired.

## Wave 1 (five optimizers, in the order I would fill them)

1. W2 with N4 attached, and B5 as its second deliverable. The measured packed round (26 cycles up to IPT 8, T19) makes W2 the predicted owner of every router cell (N 256 to 512 at k 4 to 32, N 1024 at k 16) and of B5's warp phase. The open item is the list above IPT 8 (89 cycles at IPT 32), where a two-level list could keep the round flat. The optimizer's first slope decides three cards at once (W2, N4's host, B5).
2. B1 with N6 as its first experiment. The incumbent and every fallback, and N6 (equi-width bins over the value bracket) is the only idea that cuts the bf16 pass count below 2, a 15 to 40 percent lever at every block N on the model. Same code, one extractor swap, host simulation first.
3. B2 with N4 attached. Owns block k at most 16 to 32 up to N 8192 against B1 on the model, prefer_smaller_index free in packed form, and N4 removes the 890-cycle fp32 epilogue. Its new competitor is B5 (about 60 k + 200 against 74 k + 200), so B2's optimizer should run the B5 composition as soon as W2's round is confirmed.
4. B3. The only wave 1 card with a clean win predicted at N 32768 (15.5k against 24.5k for bf16, 35.7k for fp32), k-independent after pass 1, robust to sorted input. Honest kill at N 8192.
5. W1 with N2 as its restructuring. W1 eager is the reference that PR #12222 needs, but the model says it loses above N 2 MaxK to everything. N2 (sort all tiles at once, halve per level) is the same code family and predicted 1.3 to 2x faster than eager at N 64 to 256 with k 32. With W2 at 30 cycles per round even N2 holds only N at most 64 at k 32 and the sorted-output pass, so this optimizer gets the smallest budget.

Recommended changes to wave 1 as planned (decision 10): keep the five approach ids, attach N4 to W2 and B2, N6 to B1, N2 to W1, and let W2's optimizer build B5 (two W2 phases and one smem exchange) right after confirming the round, instead of waiting for wave 2. If a sixth node is available, give N6 its own optimizer because it also has a warp variant and its result decides the shape of wave 2.

## Wave 2 (in order)

1. B4. Predicted winner at block N 8192 and above on random input (4.8k against 15.5k for B3 at N 32768 k 256), compare based so fp32 is native, and its kill experiment is a host simulation. Promote to wave 1 if a node frees up.
2. W5. Compare-based filter with an exact bound, now the predicted owner of k 32 to 64 at N 512 to 1024 and of k 16 at N 2048 and above (NSA at 128K, EAGLE-2 rerank), since its per-item cost is below any ballot-based pass and W2's rounds grow with k. Host simulation of the candidate count first, bf16 ties are its risk.
3. N6 at warp scope, if not done in wave 1. One bracket pass with 5 ballots per item against W3's 4 passes, and against W5's filter at N 512 k 16 fp32.
4. W3 with N1 as its d 1 row. Narrowed to N 1024 and above with k 128 and above by the measured ballot cost, one optimizer, one d sweep table. N1's own band is N 128 to 256 with k 64 to 128, bf16.
5. B6, one cell first. The large k over N regime is narrower than the plan assumed (`WORKLOADS.md` section 3) and B1 is nearly k-flat, so B6's value is the sorted-input chunk merge and the phase 3b sorted output. Kill at N 1024 k 512 bf16.
6. N5, throughput only. bf16x2 min, max and masks confirmed at 6 cycles, up to 2x on network cost where networks are the right algorithm (N at most 128, k 32 to 64 after W2's revision, a narrow band). Kill with the keys-only sort at N 64.
7. N3, one cell against B4. Streaming contract only, predicted 1.3 to 1.6x slower than B4 on resident input. Keep if the device agent wants "insert chunk, keep k".
8. W4, last. Crossover mapping and the sorted-input multiway merge of M chunk results, which is its one predicted win.

B5 left this list: it runs inside W2's and B2's wave 1 work (see above).

## Where I expect the winner to flip (the cells the evaluation engineer should make dense)

Warp scope, bf16 (fp32 the same with N4):
- N 128 to 256: W2 up to k 64 (measured round 26), N1 only at N 256 k 128. The W2 to N1 crossover is near k 69 at N 256 on the model.
- N 512: W2 up to k 32, a W2 and W5 tie at k 64 (2470 against 2440).
- N 1024: W2 at k 16 (1520 against W5 1720), W5 from k 32 (2630 against W2 2950, the measured 89-cycle round at IPT 32), W3 from k 128 (bf16) or k 256 (fp32). A two-level list in W2 (explore item 2) would move this band back to W2.
- N 2048 to 4096: W5 from k 16 (W2's round is issue bound at 2.8 cycles per item above IPT 8), W3 from k 128 to 256, W4 only at k near N over 2.
- N at most 64: W2 everywhere except sorted output, since the measured round at IPT 1 and 2 is 26 cycles and a sort32 alone is 660 (July placeholder). N2 and W1 keep the sorted-output pass and the throughput cells where W2's 45 registers at IPT 8 cost occupancy.
- prefer_smaller_index: free for packed W2, B2, N4 and for slot-major emission in N1 and W3, about one compare per stage for W1 and N2 (or free with bf16 index-in-key), plus 1 percent for B1. Expect no flips.

Block scope:
- N 1024 bf16: k 16 B5 (about 1140 with the measured round) against B2 (1640) against N6 (1500) against B1 (1700), the densest band of the study. k 32 and above B1 or N6. B2 against B1 crosses near k 18 on the model, B5 against B1 near k 27.
- N 4096: k 16 B5 (1310) or B2 (2000), k 32 to 256 N6 or B1, the B4 bound starts to pay at k 64.
- N 8192 and above: B4 on random input, B3 where the input is sorted or the bound is loose, N6 as the one-pass robust option, B1 as the fallback everyone pays on tie floods.
- k at least N over 4 (the chunk merges, H2O, SnapKV at 4096): B1 against B6, B1 expected unless the inputs are sorted runs (then W4 or B6's multiway merge).
- Latency against throughput: B2 at 19 registers and B1 at 32 registers both reach 8 blocks per SM, so the regime should not flip the block winner. At warp scope the data-oblivious networks (W1, N2, N5) have no smem and win throughput at small N even where W2 wins latency.
- Arch: B2's round is identical on GB200 (T9), redux, ballot and shuffle costs agree to the cycle, so no arch flip is expected in the focus set.

## What I ask of the other roles

- Evaluation engineer: done for the bf16 generator and the tie statistics (confirmed 2026-10-25). Remaining: the named cells of `named_cells.json` in the final evaluation, and dense grids on the flip bands above, in particular block N 1024 k 8 to 32 and warp N 256 to 1024 k 16 to 64.
- Hardware researcher: all requests answered, including the packed W2 list round (T19). Nothing open from my side. A measurement of the two-level list (sorted top 8 per lane plus an unsorted remainder with refill) at IPT 32 would settle whether W2 keeps N 1024 at k 32, if cheap.
- Lead: attach N4 to W2 and B2, N6 to B1, N2 to W1, let W2's optimizer build B5 in wave 1, and read `WORKLOADS.md` section 9 for the five corrections to PLAN.md 3.1.
