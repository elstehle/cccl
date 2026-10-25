# B5: Hierarchical warp then block composition
status: queued (wave 2)
scope: block
origin: `hier_extract` and `bitonic_hier` in `reference/BLOCK_TOPK_RESULTS.md` (lost at k 16, N 1024, fp32)
hypothesis: each warp runs W1, W2 or W3 on its share, then the W times k candidates are merged with a bitonic merge or a second warp top-k. Lost once because the warp round was as expensive as the block atomic round. Re-measured per cell with packed bf16 keys, where the warp phase gets much cheaper, and with W1 as the warp primitive.
cells it should win: block N 1024 to 8192, k 4 to 32, bf16
cost model: warp phase cost plus one smem exchange plus a merge of 8k candidates
explore first: (1) W1 per warp plus bitonic merge of 8k. (2) W2 packed per warp plus one more W2 over 8k. (3) no barrier between phases where a warp can own the merge.
determinism and ordering: inherits from the warp primitive, merge is deterministic.
references: `reference/BLOCK_TOPK_RESULTS.md` section 4
