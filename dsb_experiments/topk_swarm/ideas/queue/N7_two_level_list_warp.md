# N7: Two-level per-lane list for the packed W2 rounds at warp scope
status: rejected (2026-10-25, failed its 2x threshold in opt-W5's batches, see `ideas/rejected/R18_N7_two_level_list_warp.md`)
scope: warp, subwarp

The best build reaches 1.58x over W2_packed_v3 at bf16 N 1024 k 64 (4623 against 7308, threshold 3300) and sits 36 percent behind W5 in the same session (3397 for W5_v2, 2936 for W5_v4). The round costs 45 to 74 cycles against 27 for the pruned list because the block control (one vote, two taken branches, runtime round indices) costs about 18 cycles per round at blocks of 2, and a refilled head must hold four blocks of rounds or the refill fires after one win (T48, T49). The throughput regime still gains 1.26x to 2.64x over W2 at k 32 to 128, which the W5 card covers better.
