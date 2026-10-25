# W2: Redux based iterative extraction (warp)
status: queued (wave 1)
scope: warp, subwarp
origin: `reference/BLOCK_TOPK_RESULTS.md` section 1 (warp round 82 cycles) and `hier_extract`, #9747 strategy 2
hypothesis: k rounds of `redux.sync` max plus ballot plus elect. For bf16 pack key and index into the 32-bit word so every round is an exact arg-max with no tie machinery, the `atomic_iter_h` trick at warp scope. Small k only, cost linear in k.
cells it should win: warp and sub-warp, N 32 to 256, k 4 to 16, bf16 first, prefer_smaller_index for free in packed form
cost model: k rounds times (redux 22 cycles plus ballot plus elect plus a predicated shift), the redux to ballot dependency serializes
explore first: (1) packed bf16 arg-max rounds, measure cycles per round. (2) fp32: `redux.sync` f32 min and max on sm_100 if available, else value rounds plus an index resolution pass. (3) Sub-warp via masked redux. (4) Two candidates per round (extract top 2 per round with a second redux on the masked set) to halve rounds. (5) Where k stops paying against W1.
determinism and ordering: packed form inherent and sorted, prefer_smaller_index by inverting the index. Value-only form gives set membership for distinct values, ties need an index pass.
references: `reference/proto_topk_micro.cu`, `reference/proto_topk.cu` (`atomic_iter_h`), knowledge base 2.7
