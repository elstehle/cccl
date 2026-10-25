# B2: Atomics based iterative extraction (block)
status: queued (wave 1)
scope: block
origin: `reference/BLOCK_TOPK_RESULTS.md` (`atomic_adaptive`, `atomic_iter_h`), `reference/proto_topk.cu`, #9747 strategy 4
hypothesis: k rounds of a fire-and-forget shared `atomicMax` of the thread's current candidate into a per-round slot, a barrier, a branchless shift-down. The MIO combines same-address bursts, a block round costs about 74 to 81 cycles on B200. For bf16 pack key and index into the atomic word so the slots are the answer and output is indices by construction.
cells it should win: block N 1024 to 8192, k 4 to 64, bf16 especially, latency regime
cost model: k rounds times (atomic burst plus barrier plus readback about 80 cycles) plus an epilogue for index resolution in the fp32 case
explore first: (1) packed bf16 pairs, index-only output, k 4 to 64. (2) fp32 keys: value rounds plus deterministic index resolution (min-index ticket) for prefer_smaller_index. (3) two slots per round. (4) where rounds stop paying against B1 and B5. (5) GB200 round cost (MIO behavior may differ).
determinism and ordering: packed form inherent and sorted. fp32 form needs the index pass for ties.
references: `reference/BLOCK_TOPK_RESULTS.md` sections 1, 3, 4
