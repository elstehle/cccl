# N2: Level-parallel bitonic top-k
status: rejected (2026-10-25, killed on a clean comparison in wave 1, see `ideas/rejected/R12_N2_level_parallel_bitonic.md`)
scope: warp, subwarp

The restructuring gains nothing with identical network code (W1_level_same 0.95x to 1.08x, W1_level_same_sub 0.91x to 1.17x, W1_level_pairs 0.96x to 1.22x with more registers). The 1.5x to 2.5x of opt-W1's W1_level network over CUB's network on the same packed word is the exchange implementation, not the level structure, and is tracked in `ideas/PRIORITIES.md` (the W1 codegen experiment) and in the W1 card.
