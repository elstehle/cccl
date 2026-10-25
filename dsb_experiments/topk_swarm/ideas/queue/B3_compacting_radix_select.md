# B3: Filtering or stream-compacting radix select (block)
status: queued (wave 1)
scope: block
origin: new, cousin of cluster idea A3 (knowledge base 3.4)
hypothesis: after each pass compact the surviving candidates into shared memory or into a dense register set via ballot and prefix, so later passes touch candidates only. Pays when the first pass removes most items (large N over k), with more passes (fp32, fp64), and for pairs where moving the index is the cost.
cells it should win: block N 8192 to 32768, k 4 to 256, fp32 more than bf16
cost model: pass 1 over N plus compaction plus later passes over the candidate count
explore first: (1) candidate compaction into smem after pass 1, measure the compaction cost against the saved pass work. (2) register-resident compaction via ballot prefix for small candidate counts. (3) interaction with digit width (a wider first digit shrinks candidates faster). (4) index-only output from the compacted set.
determinism and ordering: as B1.
references: knowledge base 3.4 idea 3, `reference/BLOCK_TOPK_AIR_OPT_RESULTS.md`
