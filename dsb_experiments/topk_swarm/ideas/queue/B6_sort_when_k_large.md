# B6: Sort when k is large
status: rejected (2026-10-25, killed by opt-B6's kill batch, see `ideas/rejected/R17_B6_sort_when_k_large.md`)
scope: block

A stock `BlockRadixSort` pass costs about 740 + 82 x IPT cycles on 32-bit words and 815 + 100 x IPT on 64-bit words at 256 threads (T54), so the full sort is 2.1x to 2.35x behind B3_tuned on bf16 and 5.0x to 6.5x on fp32 at every N 1024 to 8192 and k from N over 8 to N over 2, with no crossover. The sorted-output premium numbers (`approaches/B6/REPORT.md` section 6) are the phase 3b row, the chunk merge of sorted runs moved to W4's card, and the keys-only sort with the index in the low bits (T55) is the way to sort pairs when a sort is wanted.
