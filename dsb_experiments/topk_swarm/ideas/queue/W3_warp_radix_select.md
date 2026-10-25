# W3: Warp bucket select, the radix-digit passes (fallback of the one-pass bracket select, with N1 as its 1-bit row)
status: queued (wave 2, one optimizer with N6 at warp scope and N1)
scope: warp
origin: `reference/WARP_RADIX_SORT_RESULTS.md` (ballot-ranked LSD), radix select as in `block_topk_air` at warp scope
hypothesis: digit histograms by ballot over the warp, one lane per bucket counts with popc over the combined bit masks, a shuffle suffix scan over the buckets finds the one holding rank k, the prefix narrows. Every digit width pays the same 16 (bf16) or 32 (fp32) ballots per item in total, so after the measured ballot cost (6 cycles per further ballot, T17) a 4-bit pass costs about 81 + 26 IPT and bf16 needs 4 of them. The one-pass bracket select of N6 at warp scope (32 equal-width bins over an exact bracket, 5 ballots per item) reaches the same discrimination in one pass on random and logit-like data, so W3's digit passes are the fallback for rows where the bracket fails (heavy tails, bf16 tie classes that fill the pivot bin) and the finish for the pivot bin's tie class on the key bits, as in T25 at block scope. Its own cells are the ones where no bound exists: k above N over 4 at N 2048 and above.
cells it should win: as the fallback, every warp cell N 512 and above with k at least 128 on adversarial input. On random input only k above N over 4 at N 2048 to 4096 (W4's cells), and the d sweep is knowledge for the final table.

what it must beat (B200, 2026-10-25): W2_packed (v1 card) N 512 k 256 12362 bf16 and 12614 fp32, N 2048 k 256 40417 and 42931, k 1024 150107 and 154812. W1_reference N 512 k 256 22889 and 13920, N 2048 k 1024 188007 and 131669, N 4096 k 1024 397814 and 337437. These are weak incumbents: W3 at 4-bit digits is predicted 3x to 10x under them. The real bar is N6 at warp scope, predicted about 2000 at N 512 k 64, 2750 at N 1024 k 128 and 4700 at N 2048 k 256, and N7 at about 8300 at N 2048 k 256. W3 matters where it beats N6 warp or where N6 falls back.

cost model (measured 2026-10-25, `TECHNIQUES.md` T17: dependent ballot 26.7, about 6 cycles per further independent ballot including its predicate, __popc 23, shuffle 4.9 per further shuffle in flight. July placeholder: shuffle stage 44 at IPT 1):
- Pass with d-bit digit over IPT items per lane: d IPT ballots pipelined (21 + 6 d IPT) + bucket counts (one bucket per lane for d at most 5, one LOP3 and a popc per item, about 2 IPT + 10) + suffix scan over 2^d buckets (d shuffle steps, 44 each) + crossing test, broadcast, prefix update (about 50). pass(d) about 81 + 6 d IPT + 2 IPT + 44 d.
- bf16 total(d) about (16 over d)(81 + 2 IPT) + 96 IPT + 704. d 4: 1030 + 104 IPT (N 512 2700, N 1024 4360, N 2048 7700, N 4096 14.3k). fp32 d 4: 2060 + 208 IPT (N 1024 8700, N 2048 15.4k, N 4096 28.7k). N1 (d 1): 840 + 134 IPT, cheaper than d 4 below IPT 12 and the natural fallback pass for sub-warp rows (one ballot serves all rows of the hardware warp).
- Index-only emission by ballot and popc prefix in slot-major order gives increasing index order, so prefer_smaller_index costs one ballot per slot and nothing else.
- The measured W2 round (26 cycles, T38) means W2 holds every k up to 64 at N up to 512 and k 16 up to N 2048, so no W3 cell below k 128 exists on random input, and N7 extends W2 to k 128 at N 1024.

focus cells: 1. warp N 2048 k 1024 bf16 (k over N 1 over 2, no bound), 2. warp N 4096 k 1024 fp32, 3. warp N 2048 k 256 bf16 (against N6 warp in the same session), 4. warp N 1024 k 128 fp32, 5. sub-warp LW 8 N 128 k 64 bf16 with the d 1 row (W2 sw_v2 is 5261 there).
boundary cells: warp N 1024 k 32 (W2 2481, N7 and W5 expected to win), warp N 512 k 64 (N6 warp 2000 and W5 2740 expected).

explore first: (1) one 4-bit pass and one 1-bit pass at IPT 16, 64 and 128 as slopes over passes, one kernel, against N6 warp's bracket pass in the same session. (2) digit schedule for bf16 (4+4+4+4 against 5+5+6) and fp32 (8 by 4 against 6 by 5 plus 2) on the fallback rows. (3) the key-bit finish for a bf16 tie class in the pivot bin (restricted digits over the tracked min and max, T25). (4) the d sweep table with N1 for the knowledge base.

determinism and ordering: set deterministic by construction. prefer_smaller_index via the slot-major ballot emission. Sorted via a W1 pass over k.

main risk: on random input W3 has no cell of its own below k over N of 1 over 4, so the card's value is the fallback and the knowledge, and an optimizer can sink time into digit schedules that never run. bf16 tie classes larger than k remaining defeat early exit at every digit width.

cheapest kill experiment: explore item 1 at N 2048 k 256 bf16: if N6 warp's one bracket pass plus finish beats four 4-bit passes by 1.5x or more on the harness random input, W3 is the fallback only and gets no further cells of its own.

references: knowledge base section 4, `TECHNIQUES.md` T17, T25, T38, cards W2_packed_b200_20261025_1 and W1_reference, `ideas/queue/N6_equiwidth_bucket_select.md`, `ideas/queue/N1_bit_serial_select.md`, `ideas/queue/W5_threshold_filter.md`
