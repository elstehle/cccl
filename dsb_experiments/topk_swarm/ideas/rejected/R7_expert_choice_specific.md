# R7: An expert-choice specific primitive
status: rejected, no primitive-level structure (2026-10-25, algo-researcher)
scope: block
seed: the lead's list (anything specific to expert-choice routing)
idea: exploit that expert-choice routing selects a fixed capacity k = n c over e tokens per expert over the same n tokens for every expert.
reason: the per-expert selection is an ordinary top-k over a column of the token-by-expert score matrix with k over N = c over E, which is 1 over 64 to 1 over 8 for c 1 to 2 and E 16 to 128 (Zhou et al. 2022, capacity factors 0.5, 1, 2, experts 16 to 128), and 1 over 8 for mixture-of-depths (12.5 percent capacity). Those are B1 cells, k-flat, and the columnwise layout (scores are produced token-major) is the device agent's transpose problem, not a primitive's. Nothing is shared between experts' selections except N and k. The plan's "N over 8 to N over 2" regime for capacity routing is corrected in `ideas/WORKLOADS.md`.
what survives: the named cells `ec_*` and `mod_*` in `ideas/named_cells.json`, and the note that the large k over N regime comes from chunk merges and H2O style eviction instead.
