# GitHub issue tree for the top-k cooperative primitives study

Draft of 2026-10-25 20:30 UTC, revised 20:45 UTC against the saved GitHub references (`state/github_refs/`: #9747, #9749, #12160, PR #12222, PR #9281, PR #8391, the label list), from the knowledge base (`TOPK_KNOWLEDGE_BASE.md`, version 26) and the approach reports under `approaches/`. Nothing in this document has been created on GitHub. Every performance number that the phase 3 evaluation will replace is written as `<phase 3>` and is filled from `WINNERS_b200.md` and `WINNERS_gb200.md` before any issue is created. Numbers that stand as measured carry hardware, CUDA version, date and card, and come from verified cards unless the line says otherwise. B200 means NVIDIA B200 (sm_100), nodes umbriel-b200-018, 028, 047, 068 and 091, CUDA 13.4.92, build `-arch=sm_100`, 2026-10-25, latency in cycles per call net of the generate-only control, throughput in Gelem/s for one occupancy wave, variant and control in one lock section. umbriel-b200-020 is the retired outlier node (knowledge base 2.10, T58).

Part A is for the user: the tree, the table, the existing issues to update, and the lines not to pursue. Part B is for an agent with `gh`: the ordered manifest with one block per issue and the exact creation and linking steps.

Conventions for both parts: no em dashes, no semicolons in prose, no AI attribution in any body, no performance numbers in code comments, short bodies that point to the fork branch (`https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm`) with the report and card paths. The knowledge base page is private to the user and is never linked from a public body. The knowledge base markdown itself is committed to the fork branch at the end of phase 4 as `KNOWLEDGE_BASE.md` (without the private link), which is what the "KB n" section references in the bodies resolve to, and the user may replace the fork pointers before creating issues. Labels come from the repository's list and nothing else: `cub` on every issue, `future-productization` on the approach parents, `bench` on the harness and methodology parent and its sub-issues, `docs` where the sub-issue is documentation. Bodies mirror the section order of #9749 (Summary with scope, comparison method and status, Problem coverage with segment size, k and types, Output guarantees as the determinism, tie-break and output-ordering tables, then the owner), with the words yes, no, planned and TBD in the tables where #9747 uses symbols.

---

## Part A. The tree for the user

### A.1 The tree

```
Top-k cooperative primitives (strategy list of #9747: 1, 2, 4 kept, 5 and 6 are the cluster and multi-CTA lines outside this study, no 3, new numbers from 7)
|
+-- TK-P01  W2  packed redux extraction at warp and sub-warp scope           [strategy 2]
|     +-- TK-S01.1 fp32 route v4 (the v2 regressions, key loads pinned)
|     +-- TK-S01.2 sub-warp form with one full-warp redux per logical warp
|     +-- TK-S01.3 N 4096 cells and the compile time of the k 256 and 1024 translation units
|     +-- TK-S01.4 two candidates per round at 8 items per lane and below
|     +-- TK-S01.5 GB200 card
|     (do not pursue: redux pre-reduction before a block round, R14)
|
+-- TK-P02  W5  threshold filter with a certified lower bound and a packed finish   [new strategy 7]
|     +-- TK-S02.1 the scatter store count (two candidates per 64-bit store, prefix-free gather)
|     +-- TK-S02.2 the per-lane top 2 and the ballot prefix
|     +-- TK-S02.3 register pin for the throughput regime
|     +-- TK-S02.4 N 4096 and the k 128 band
|     +-- TK-S02.5 a second fallback rung for tie floods
|     +-- TK-S02.6 the signed-zero contract of a threshold on twiddled keys (the v3 lesson)
|     (do not pursue: N7 two-level per-lane list for the W2 rounds, R18)
|
+-- TK-P03  W1  bitonic networks: min and max route, sub-warp, two-row form     [strategy 1, updates #9749]
|     +-- TK-S03.1 min and max customization point in WarpBitonicTopK (updates #12160)
|     +-- TK-S03.2 sub-warp support (logical warps 4, 8, 16)
|     +-- TK-S03.3 the bf16 index-in-key word through the network
|     +-- TK-S03.4 N5, two bf16 rows per lane in bf16x2 words (throughput)
|     +-- TK-S03.5 the level tile order at one word per tile, the two-word loss
|     +-- TK-S03.6 aarch64 cicc branch emission on compare networks (toolchain report)
|     +-- TK-S03.7 the fp32 u64 word kept for sorted output and N 8
|     (do not pursue: N2 level-parallel restructuring as such, R12)
|
+-- TK-P04  N10 fp32 keys-only network with index recovery by ballot rank          [new strategy 8]
|     +-- TK-S04.1 an asm barrier between the ballot block and the prefix block
|     +-- TK-S04.2 sorted output by one shuffle per selected item
|     +-- TK-S04.3 logical warp 4: shuffle-xor min inside the segment, two slots per ballot word
|     +-- TK-S04.4 register cut at sub-warp N 128 for batched router kernels
|     +-- TK-S04.5 rows-per-group throughput form
|     +-- TK-S04.6 sm_100f variant with redux.sync.min.f32
|
+-- TK-P05  N6  equi-width bucket select at warp and block scope, with N9            [new strategy 9]
|     +-- TK-S05.1 warp fixed costs below 64 items per lane
|     +-- TK-S05.2 warp register footprint at 64 and 128 items per lane
|     +-- TK-S05.3 compaction before the second bracket pass (k 1024)
|     +-- TK-S05.4 atom32f at fp32 N 4096 and the unmeasured variants
|     +-- TK-S05.5 N9, the packed finish as the block N6 finish from k 64
|     +-- TK-S05.6 block N6 at 128 items per thread
|     +-- TK-S05.7 the bracket scale guard audit (T60) across N6, B4 and N9
|     +-- TK-S05.8 host models of the device arithmetic as test fixtures (T60, T63)
|     (do not pursue: W3 and N1 as primitives of their own, N3 streaming on resident data)
|
+-- TK-P06  B2  packed atomics extraction, two-level list, block size            [strategy 4]
|     +-- TK-S06.1 two-level list with the rest in shared memory (128 items per thread and above)
|     +-- TK-S06.2 the warp-uniform refill variant and the fp32 k 16 cells
|     +-- TK-S06.3 the unroll policy above k 64 and the node class of large unrolled kernels
|     +-- TK-S06.4 block size per regime: 128 threads for latency, 64 for one-wave throughput
|     +-- TK-S06.5 a combined adapter choosing threads and list per (N, k)
|     +-- TK-S06.6 the fp32 truncated word at N 32768 (17 key bits)
|     (do not pursue: B5 hierarchical composition R13, redux pre-reduction R14, speculative post R15, 64 threads for latency)
|
+-- TK-P07  B4  exact-bound prefilter, compaction and a small finish                [new strategy 10]
|     +-- TK-S07.1 packed bf16 sweeps
|     +-- TK-S07.2 the fallback out of the fast path's register allocation
|     +-- TK-S07.3 a custom block scan
|     +-- TK-S07.4 bounds for k above 256 and the 64-word finish for k 1024 to 2048
|     +-- TK-S07.5 a 2048-word buffer with its fallback rate
|     +-- TK-S07.6 128 and 512 threads
|     +-- TK-S07.7 N6 and B4 as one primitive
|     +-- TK-S07.8 the deterministic path through the per-candidate lookup
|     +-- TK-S07.9 same-session pairings with B1_idx at 512 threads and B2_twolevel at k 16
|
+-- TK-P08  B3  compacting radix select                                          [new strategy 11]
|     +-- TK-S08.1 512 threads from N 16384
|     +-- TK-S08.2 the compaction threshold per N
|     +-- TK-S08.3 the fp32 fallback cells and the pass-loop sensitivity
|     +-- TK-S08.4 the k at least N over 16 band against B4 and B1_idx at 512 threads
|     +-- TK-S08.5 select plus post-sort as the fp32 sorted route for one tile
|     (do not pursue: B6 full sort as a selection primitive, R17)
|
+-- TK-P09  B1  AIR radix select: index-only epilogue, deterministic ticket and lookup, block size   [no entry in #9747, the shipped block primitive]
|     +-- TK-S09.1 the index-only ticket epilogue (T21) in block_topk_air
|     +-- TK-S09.2 the class-indexed ticket of the deterministic path (T57, T61)
|     +-- TK-S09.3 the per-candidate tie lookup and the pinned shared base (T65, T66)
|     +-- TK-S09.4 512 threads from N 8192, 64 threads for one-wave throughput (T44, T45)
|     +-- TK-S09.5 the minus-zero normalization folded into the first digit extraction
|     +-- TK-S09.6 the striped arrangement branch and the raw extractor
|     +-- TK-S09.7 the N6 first pass for fp32 N 4096 and bf16 and fp32 N 16384 k 16
|     +-- TK-S09.8 B1_idx adopting the class-branch epilogue
|     +-- TK-S09.9 the mask build and the scan of the tie phase, still per slot
|     (do not pursue: per-slot ballot tickets R16, the empty-asm ticket fix T39, 1024 threads, normalization under not_guaranteed)
|
+-- TK-P10  W4  k-truncated merge of sorted runs: sorted output and chunk merges        [new strategy 12]
|     +-- TK-S10.1 the fp32 per-lane sort of 64 words
|     +-- TK-S10.2 a hybrid run layout for the deeper levels
|     +-- TK-S10.3 a 4-way merge path for eight runs and 128-bit staging
|     +-- TK-S10.4 N 4096 at warp scope
|     +-- TK-S10.5 fp32 N 2048 in the throughput regime (16-bit index half)
|     +-- TK-S10.6 runs shorter than N over M
|     +-- TK-S10.7 the chunk-merge contract in the device agents (sorted runs only)
|     (do not pursue: the 4-ary merge path, the two-half per-lane sort)
|
+-- TK-P11  Determinism and ordering features across the families                [cross-cutting]
|     +-- TK-S11.1 the measured price of prefer_smaller_index per family
|     +-- TK-S11.2 the sorted-output routes (W4 and B6 tables, W5's sorted finish, W2 at small k)
|     +-- TK-S11.3 prefer_larger_index (not built anywhere)
|     +-- TK-S11.4 signed zero and NaN behavior per family
|     +-- TK-S11.5 the deterministic block primitive in the batched backends
|
+-- TK-P12  Key type specializations                                            [cross-cutting]
|     +-- TK-S12.1 the bf16 packed key-plus-inverted-index word
|     +-- TK-S12.2 the fp32 routes (truncated word, keys-only network, u64 word)
|     +-- TK-S12.3 fp8 widened to 16-bit lanes (T8)
|     +-- TK-S12.4 fp16 (the bf16 word with a different twiddle)
|     +-- TK-S12.5 fp64 (untested, 64-bit words)
|
+-- TK-P13  Harness and methodology items worth upstreaming                       [infrastructure]
      +-- TK-S13.1 the device-side benchmarking methodology
      +-- TK-S13.2 the sorted-output check in the tests
      +-- TK-S13.3 the signed-zero battery in the tests
      +-- TK-S13.4 the oracle self-test at k equal to N over 2
      +-- TK-S13.5 node pre-flight and the outlier rule for performance work (T58)
      +-- TK-S13.6 phase stamps between barriers as the profiling method (T56)
      +-- TK-S13.7 the harness version policy for cross-version comparisons
```

### A.2 The table

One line of rationale and evidence per issue. "KB" is the knowledge base section, cards are under `cards/` without the `.json` suffix, reports under `approaches/`. Status: V means every number comes from verified cards, U means unverified cards, P means pending the phase 3 numbers, N means a note on a parent and no issue.

| Id | Title | Parent | Rationale and evidence | Status |
|---|---|---|---|---|
| TK-P01 | W2 packed redux extraction | strategy 2 | The packed key-plus-inverted-index word makes every round an exact arg max at 24.9 to 29.6 cycles, W2 beats the warp bitonic reference in every bf16 cell up to k 256 (6x at N 32 to 29x at N 2048 k 16), owns the warp up to k 32 and the k 16 cells everywhere (KB 12, 16, cards `W2_packed`, `_v2`, `_v3`, `_sw_v2`, report `W2/REPORT.md`) | V, phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25: 42 of 89 warp latency cells, median 61 percent ahead of the runner-up and 825 percent ahead of the warp bitonic reference, eight of them close calls (runner-up within 15 percent) |
| TK-S01.1 | fp32 route v4 | TK-P01 | The v2 fp32 regressions at N 1024 k 8 and N 2048 k 16 and 32 come from ptxas sinking three key loads into the selection network (KB 12.4, 12.5, verifier notes) | V |
| TK-S01.2 | sub-warp form | TK-P01 | One full-warp redux per logical warp with the identity in the other lanes is 4x to 7x faster than a masked redux (T37, card `W2_packed_sw_v2`, KB 12.4) | V |
| TK-S01.3 | N 4096 cells | TK-P01 | The k 256 and 1024 translation units compile for more than 20 minutes, the cells are unmeasured (KB 12.5, 16) | open |
| TK-S01.4 | two candidates per round | TK-P01 | Built as sw_v3, about 1.2x on the model at 8 items per lane and below, never measured (KB 12.5) | open |
| TK-S01.5 | GB200 card | TK-P01 | W2 was measured on B200 only, T16 says the SM costs transfer (KB 12.7) | open |
| TK-P02 | W5 threshold filter with verification | strategy 7 | A grouped exact lower bound plus compaction plus the W2 finish takes bf16 k 16 to 64 at N 1024 to 2048 and fp32 from k 32 at N 1024 and k 16 at N 2048 from W2 (bf16 N 1024 k 32 1837 against 2481, k 64 2936 against 7308, fp32 N 2048 k 16 2293 against 4391, card `W5_v4`, KB 18) | V, phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25: 21 of 89 warp latency cells and 34 of 87 throughput cells, every k 16 to 64 cell at N 2048 and 4096, the flip from W2 at N 1024 (bf16 k 16, fp32 k 8), three latency cells among the close calls (runner-up within 15 percent) |
| TK-S02.1 | scatter store count | TK-P02 | A shared store instruction costs about 10 cycles on one warp whether or not its lanes are active, 32 per call at 32 items per lane (KB 18.3, 18.5) | V |
| TK-S02.2 | top 2 and ballot prefix | TK-P02 | 160 and 216 cycles of a 1837-cycle call at bf16 N 1024 k 32 by the phase probe (KB 18.3) | V |
| TK-S02.3 | register pin | TK-P02 | 128 registers at N 1024 and 255 at N 2048 since v3 hoists the flag and address arithmetic, occupancy 1 in the throughput regime (KB 18.3, T24) | V |
| TK-S02.4 | N 4096 and k 128 | TK-P02 | Unmeasured, the seed check says the bounds hold at k 128 (KB 18.7), N7 holds measured k 128 cells (KB 18.4) | open |
| TK-S02.5 | second fallback rung | TK-P02 | The exact k-th of the certified set then a refilter keeps tie floods on the fast structure (KB 18.5) | open |
| TK-S02.6 | signed-zero contract | TK-P02 | W5_v3 dropped minus-zero items at a plus-zero threshold, caught only by a 32-seed battery (KB 2.10, card `W5_v3` refuted on contract) | V |
| TK-P03 | W1 bitonic networks | strategy 1 | The min and max exchange on the packed word is 1.15x to 1.6x per network (T42, card `W1x_codegen`, verified on second hosts of both archs), the family keeps N at most 64 at k 16 to 32, the bf16 k 64 cells at N 128 and 256 and the sorted-output pass (KB 11, 16) | V, phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25: one warp latency cell (bf16 N 128 k 64, 912 cycles, N6w_atom32 within 0.2 percent, a close call, runner-up within 15 percent) and the sub-warp N 8 k 4 cells at 4 lanes, the N 32 k 16 cells went to W2 and N10 |
| TK-S03.1 | min and max customization point | TK-P03 | A bool comparator object hides the min and max form from the compiler, the CUB network needs the customization point (T42, KB 2.12, issue #12160) | V |
| TK-S03.2 | sub-warp support | TK-P03 | Logical warps 4, 8, 16 at N 8 to 128 are the router shapes (KB 10.2), the W1 sub-warp cards are verified (KB 11.4) | V |
| TK-S03.3 | bf16 index-in-key word | TK-P03 | 1.36x to 2.38x latency and 2.0x to 2.6x throughput over the pairs network, prefer_smaller_index free (cards `W1_eager_same` and siblings, KB 11.4) | V |
| TK-S03.4 | N5 two-row form | TK-P03 | 1.22x to 1.44x per call at k at most 32 and 1.43x to 1.82x at k 64 over the one-word network in one-wave throughput, a latency win at sub-warp scope only (cards `N5_rows_f` and siblings, KB 20, T53) | V |
| TK-S03.5 | level order | TK-P03 | 1.15x to 1.54x only with one word per tile in latency, 0.93x to 1.0x with two (KB 11.4, 11.7, R12), confirmed on fp32 words by N10 (KB 19.3) | V |
| TK-S03.6 | aarch64 cicc branches | TK-P03 | The aarch64 cicc emits branches around every compare-exchange swap where x86 emits predicates, 1.17x to 1.26x on the CUB control (T20, KB 2.10) | V |
| TK-S03.7 | fp32 u64 word | TK-P03 | N10 replaces it under both requirements at N 32 to 256, it keeps the sorted-output pass and N 8 (KB 11.7, 19) | V |
| TK-P04 | N10 fp32 exact route | strategy 8 | A keys-only u32 network with the final merge truncated at stride k and ballot rank recovery gives exact prefer_smaller_index at 661 against the u64 word's 934 at N 64 k 32 and 960 against 1918 at N 128 k 16, faster than the best not_guaranteed route in every warp cell (cards `N10_auto` and eight siblings, verified on a second node, KB 19, T51, T52) | V, phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25: two warp fp32 latency cells (N 32 k 16 and N 128 k 64, both close calls (runner-up within 15 percent)) and the sub-warp fp32 k 16 and 64 cells at 16 lanes, the other fp32 small-N cells went to W2, prefer_smaller_index free for every warp family |
| TK-S04.1 | asm barrier | TK-P04 | The two-class pass is issue bound at about 50 cycles per item per lane, the pipelined-ballot model predicts half (KB 19.3, 19.5) | V |
| TK-S04.2 | sorted output | TK-P04 | The truncated forms give no sorted output, one shuffle per selected item with the complete merge would (KB 19.2, 19.5) | open |
| TK-S04.3 | logical warp 4 | TK-P04 | LW 4 N 32 k 4 and 8 lose to the pairs network by 1.20x and 1.28x, eight pipelined full-warp redux cost about 88 cycles (KB 19.4, 19.5) | V |
| TK-S04.4 | register cut at sub-warp N 128 | TK-P04 | 118 registers at LW 8 and 168 at LW 4 in the latency kernel, throughput occupancy 2 and 1 (KB 19.3, 19.7) | V |
| TK-S04.5 | rows-per-group form | TK-P04 | The recovery is per row, the N5 rows arrangement applies (KB 19.5) | open |
| TK-S04.6 | sm_100f variant | TK-P04 | redux.sync.min.f32 allows the native float network without the twiddle (T2, KB 19.5) | open |
| TK-P05 | N6 equi-width bucket select with N9 | strategy 9 | At warp scope an exact per-lane bound, a 32-bin shared atomic histogram and a packed finish take k 64 to 1024 at N 256 to 2048 from W2 (bf16 N 512 k 64 1879 against 2704, N 2048 k 1024 17741 against 152618, card `N6_w_atom32w2`, verified on both archs, KB 22, T59). At block scope N6 wins where it saves a pass and N9 is its finish from k 64 (KB 13, 21.4) | V, phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25: N6w_atom32 20 of 89 warp latency cells (k 48 to 1024 at N 256 to 4096, median 21 percent ahead of the runner-up), N9_w2fin2 13 of 77 block latency cells at N 1024 to 8192, all but one close calls (runner-up within 15 percent) |
| TK-S05.1 | warp fixed costs | TK-P05 | Below 64 items per lane the shuffle scan (about 190), the pivot search (about 80) and the pivot bin's min and max chain dominate (KB 22.3, 22.5) | V |
| TK-S05.2 | warp register footprint | TK-P05 | 255 registers at 64 items per lane and a 4 to 5 KB stack at 128, 60 to 80 thousand cycles at N 4096 (KB 22.3) | V |
| TK-S05.3 | compaction before the second pass | TK-P05 | At k 1024 the second bracket pass sweeps the whole tile for at most N over 8 candidates (KB 22.5) | open |
| TK-S05.4 | atom32f at fp32 N 4096 | TK-P05 | 2.8x to 3.1x over atom32 at fp32 N 4096 in the card's own data, which may give N6w an N 4096 cell under W2, unmeasured in one section, atom128 and W3_d5c built and unmeasured (KB 22.4, 22.7) | open |
| TK-S05.5 | N9 as the block finish | TK-P05 | 9 to 17 percent at k 64 to 256 under not_guaranteed and 13 to 41 percent under prefer_smaller_index, nothing at k 16 (cards `N9_w2fin2`, verified, KB 21.4) | V |
| TK-S05.6 | block N6 at 128 items per thread | TK-P05 | Register bound in two of three builds, 1.5 to 8.6 KB of spill (KB 13.7) | V |
| TK-S05.7 | bracket scale guard audit | TK-P05 | A width that overflows gives a plus-zero scale and NaN images for infinite keys (T60), the block N6, B4 and N9 brackets are in scope (KB 2.10, 13.7) | V |
| TK-S05.8 | host models as fixtures | TK-P05 | The host models of the device arithmetic found T60 and T63 before any GPU time (KB 2.10) | V |
| TK-P06 | B2 packed atomics extraction | strategy 4 | k rounds of a fire-and-forget shared atomic on the packed word, 83.5 cycles per round at 256 threads and 69.5 at 128, beats B1 at k 4 to 16 at every N to 16384 and the two-level list carries it to N 32768 at k 4 and 16 for bf16 (cards `B2_packed_1` to `_3`, `B2_twolevel_1`, KB 14, 17) | V, phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25: the 128-thread form 12 of 77 block latency cells (k 4 to 32 at N 1024 to 8192), the two-level list 7 (k 4 and 16 at N 4096 to 32768, 11.4x the reference at bf16 N 16384 k 4), the N 1024 and 4096 cells close calls (runner-up within 15 percent) |
| TK-S06.1 | rest in shared memory | TK-P06 | At 128 items per thread and above the two-level list spills and 128 threads cannot be used, N 65536 is out of reach (KB 14.5, 17.4) | open |
| TK-S06.2 | warp-uniform refill variant | TK-P06 | `B2_twolevel_any` exists as batches only, within a few percent at 256 threads (KB 14.7) | open |
| TK-S06.3 | unroll policy above k 64 | TK-P06 | Fully unrolled 128-thread kernels above about 64 KB of SASS were the node-dependent class (T50, T58), B2 above k 64 already rolls (KB 2.10, 14.5) | V |
| TK-S06.4 | block size per regime | TK-P06 | 128 threads for latency at k at most 32 and N 1024 to 4096, 64 threads for one-wave throughput at N at most 2048 (1.36x to 1.66x), the regimes disagree (cards `N8_b2packed_1` to `_3`, verified, KB 17, T43 to T46) | V |
| TK-S06.5 | combined adapter | TK-P06 | Threads and list per (N, k) from the N8 recommendation tables (KB 17.4) | open |
| TK-S06.6 | fp32 truncated word at N 32768 | TK-P06 | 17 key bits leave the finish running on 20 to 50 percent of random inputs (KB 2.12, 14) | V |
| TK-P07 | B4 exact-bound prefilter | strategy 10 | The minimum of the per-thread maxima admits about 1500 candidates at every N from 4096, B4_v4 owns block N 8192 to 32768 for k 64 to 1024 (fp32 N 32768 k 256 7026 against B3_tuned 18278 under harness 0.8, 2.60x, bf16 2.26x, N 8192 k 64 1.32x and 1.38x, cards `B4_v4_1` and `_2`, verified on two nodes, KB 21) | V, phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25: 32 of 77 block latency cells, N 8192 to 32768 from k 32 or 64 to k 1024, median 160 percent ahead of the reference and 52 percent ahead of the runner-up, confirmed on a second node, five of them close calls (runner-up within 15 percent) |
| TK-S07.1 | packed bf16 sweeps | TK-P07 | The three register sweeps are about 37 cycles per item of the tile and the only lever left there (KB 21.3, 21.5, T18) | V |
| TK-S07.2 | fallback out of the fast path | TK-P07 | 255 registers at 64 items per thread and 1.4 to 2.5 KB of spill at 128 from the fallback's pass bodies in the same function (KB 21.3) | V |
| TK-S07.3 | custom block scan | TK-P07 | Four block scans at 350 to 430 cycles each, a shuffle scan with one barrier is about 230 (KB 21.3, 21.5) | open |
| TK-S07.4 | bounds for k above 256 | TK-P07 | The half-group bound leaves 3300 items at N 32768, the pivot bin holds 28 to 62 items at k 1024 to 2048 and a second pass runs in half of the rows, B4 loses at k 2048 (KB 21.4, 21.5) | V |
| TK-S07.5 | 2048-word buffer | TK-P07 | Overflows 0.3 to 10 percent of the rows against a 16-slot pass and emission (KB 21.4) | open |
| TK-S07.6 | 128 and 512 threads | TK-P07 | Accepted by the adapter and unmeasured, the bound is the 128th order statistic at 128 threads (KB 21.5) | open |
| TK-S07.7 | N6 and B4 as one primitive | TK-P07 | N6's first pass for 16 items per thread and below, B4's compaction above, the warp finish in both (KB 21.5) | open |
| TK-S07.8 | deterministic path | TK-P07 | 4 to 17 percent today from the fallback's index passes, T65 names B4's emission as in scope (KB 21.6, 13.6) | V |
| TK-S07.9 | same-session pairings | TK-P07 | B1_idx at 512 threads and B2_twolevel at k 16 were compared by card numbers only (KB 21.7) | P |
| TK-P08 | B3 compacting radix select | strategy 11 | Compacting the pivot bucket after pass 1 pays from 16 items per thread for fp32 and 32 for bf16, B3_tuned owns k at least N over 16 from N 2048 and keeps k 2048 against B4 (cards `B3_tuned_1` and `_2`, KB 15, 16) | V, phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25: 11 of 77 block latency cells (k 1024 at N 4096, k 4096 at N 16384 and 32768 with B4_v4 2.0x to 2.7x behind) and 15 of 75 deterministic latency cells, four of them close calls (runner-up within 15 percent) |
| TK-S08.1 | 512 threads | TK-P08 | 1.12x to 1.15x latency and 1.00x to 1.12x throughput at N 16384 and 32768, the spills stay (card `N8_b3tuned`, verified, KB 17.4) | V |
| TK-S08.2 | compaction threshold per N | TK-P08 | The threshold is per items per thread, with the block size free it may want to be per N (KB 17.5) | open |
| TK-S08.3 | fp32 fallback cells | TK-P08 | Commit 50039daee3 restructured the pass loop and moved the fp32 fallback cells by 7 to 10 percent (KB 2.10, 15) | V |
| TK-S08.4 | the k at least N over 16 band | TK-P08 | Against B1_idx at 512 threads B3 keeps a latency edge at k 1024 only, against B4 it keeps k 2048 (KB 15.7, 21.4) | P |
| TK-S08.5 | select plus post-sort | TK-P08 | The fp32 sorted route for one unsorted tile by the B6 table, the gather of the selected keys unmeasured (KB 16, 10.5) | U |
| TK-P09 | B1 AIR radix select | none in #9747 (the shipped block primitive, strategies 5 and 6 build on it) | The incumbent block primitive, the index-only epilogue saves 5 to 19 percent, the class-indexed ticket makes the deterministic path free on fp32 and the per-candidate lookup takes the bf16 price to 4 to 24 percent (cards `B1_idx`, `B1_det2_1` and `_2`, `B1_det3_1` and `_2`, verified, KB 13) | V, phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25: B1_idx 20 and its 512-thread form 9 block throughput cells, B1_det2 2 deterministic latency and 16 deterministic throughput cells, the B1_idx cells ties with B1_det2 and close calls (runner-up within 15 percent), B1_det3 a deterministic variant at B1_det2's speed (within 2 percent in latency at N 1024 in the det3 list), no winner changes hands, its 110 cross-node cells (umbriel-b200-068) kept as evidence pending the same-node rerun |
| TK-S09.1 | index-only ticket epilogue | TK-P09 | One independent atomic ticket per selected item removes the exchange buffer, three barriers and the memoized key copy (T21, card `B1_idx`, KB 13.4) | V |
| TK-S09.2 | class-indexed ticket | TK-P09 | The constant-address ticket cost 10.8 to 22.3 percent of the deterministic call through ptxas's warp aggregation (T57, T61, cards `B1_det2`, KB 13.6) | V |
| TK-S09.3 | per-candidate lookup and pinned base | TK-P09 | Registers back at B1_idx's, two blocks per SM at N 16384, no spill at 512 threads, the shared window base rematerialized until pinned (T65, T66, cards `B1_det3`, KB 13.6) | V |
| TK-S09.4 | block size | TK-P09 | 512 threads from N 8192 in latency and N 16384 in throughput (1.24x to 1.45x at N 32768), 64 threads in one-wave throughput at N 1024 to 2048 (T44, T45, card `N8_b1idx`, KB 17) | V |
| TK-S09.5 | normalization in the digit extraction | TK-P09 | The minus-zero normalization is the largest fp32 term of the deterministic path, 1000 to 1250 cycles per call at 512 threads and 64 items per thread (KB 13.7) | open |
| TK-S09.6 | striped arrangement and raw extractor | TK-P09 | Branch `enh/block-topk-striped` validated 2026-10-11 and 2026-10-19, the raw extractor and NormalizeSignedZero open (KB 4.3, 4.4, 6) | V |
| TK-S09.7 | N6 first pass | TK-P09 | fp32 N 4096 k 16 to 256 by 11 to 24 percent and N 16384 k 16 by 1.48x and 1.58x, loses throughput by occupancy (cards `B1_n6_2`, `B1_n6b_1` and `_2`, KB 13.4, 17.4) | V |
| TK-S09.8 | class-branch epilogue in B1_idx | TK-P09 | det3 is 1.3 to 1.9 percent faster than B1_idx's not_guaranteed path at fp32 N 1024 (KB 13.7) | V |
| TK-S09.9 | per-slot mask build and scan | TK-P09 | About 3.5 instructions per slot and 136 instructions for two scan rounds, on 66 to 91 percent of the bf16 rows at N 4096 and above (KB 13.6) | V |
| TK-P10 | W4 k-truncated merge | strategy 12 | The chunk merge of 4 sorted runs of 2048 is 1.52x bf16 and 1.08x fp32 under B3_tuned's unsorted selection and 3x to 5x under a full sort, the warp truncated merge sort owns sorted output from k 256 (cards `W4_chunk4`, `W4_wmerge`, verified with the sorted order checked, KB 23, 16) | V, phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25: the sorted-runs track, at 4 runs the unsorted reference costs 1.27x to 6.19x the merge, at 8 runs 0.71x to 3.71x with the one loss at fp32 N 8192 k 4096, warp sorted output W4_wmerge at k 256 and up and W5_v4 at k 4 to 64 from N 512 (3b) |
| TK-S10.1 | fp32 per-lane sort | TK-P10 | 15.7k of 42.6k cycles at fp32 N 2048 k 1024 in the final build (KB 23.3, 23.5) | V |
| TK-S10.2 | hybrid run layout | TK-P10 | The merge of the same 64 outputs grows from 1864 to 3581 cycles from level 0 to 3 through 16-way conflicts, 15 to 20 percent predicted at k 1024 (KB 23.3, 23.5) | open |
| TK-S10.3 | 4-way merge path, 128-bit staging | TK-P10 | For eight runs the third level adds a flat 1800 cycles (KB 23.3, 23.5) | open |
| TK-S10.4 | N 4096 at warp scope | TK-P10 | An in-lane merge level keeps the per-lane network at 64 (T34, KB 23.5) | open |
| TK-S10.5 | fp32 N 2048 throughput | TK-P10 | Eight warps times 39 KB of temp storage exceed the 227 KB limit, a 16-bit index half brings k 256 inside (KB 23.2, 23.5) | open |
| TK-S10.6 | runs shorter than N over M | TK-P10 | Per-run lengths in the merge path and the pads (KB 23.5) | open |
| TK-S10.7 | chunk-merge contract | TK-P10 | The merge needs sorted and full runs, which W1, W2, W5 and B6 produce and B1 to B4 and N6 do not (KB 23.7) | V |
| TK-P11 | Determinism and ordering | cross-cutting | The measured price of prefer_smaller_index per family and the sorted-output routes (KB 2.12, 16) | V, phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25: prefer_smaller_index free within 0.4 percent for W2, W5, N6w, N10, W3, W4, the W1 family, B2 and B3, a median 7 percent on B4_v4, 15 percent on N9_w2fin2 and 7 percent on B1_det2 in latency, the sorted routes per scope in TK-S11.2 (3b) |
| TK-S11.1 | price per family | TK-P11 | W2, B2, B3, W5, W4, N6 at warp scope and the bf16 W1 word zero, N10 negative on fp32, B1 free on fp32 and 4 to 24 percent on bf16, B4 4 to 17 percent, N5 zero to 4 percent (KB 2.12) | V |
| TK-S11.2 | sorted-output routes | TK-P11 | The merge of sorted runs by 2x to 3x at every merge cell, B6 only for bf16 single tiles at k equal to N over 2, W5 up to k 64 and W4 from k 256 at warp scope (KB 16) | V, B6 U |
| TK-S11.3 | prefer_larger_index | TK-P11 | Not built in any family, the mirror forms are listed per report (KB 11.5, 19.6, 21.6, 23.6) | open |
| TK-S11.4 | signed zero and NaN | TK-P11 | Every family folds minus zero onto plus zero or orders it below, NaN handling differs per family and N6 does not support it (KB 2.10, section contracts) | V |
| TK-S11.5 | deterministic block primitive in the backends | TK-P11 | The backend selector sends deterministic requests to the cluster backend, B1_det3 is a deterministic block primitive at near zero cost on fp32 (KB 1.1, 13.6) | P |
| TK-P12 | Key type specializations | cross-cutting | bf16 wins only through the packed word, fp32 has three routes, fp8, fp16 and fp64 are unmeasured in the study (KB 2.12, 10.4) | V, `<phase 3>` for the full evaluation |
| TK-S12.1 | bf16 packed word | TK-P12 | W2, B2, W1, N5 and the N4 truncation all rest on it (KB 2.12) | V |
| TK-S12.2 | fp32 routes | TK-P12 | The truncated word (N4) for W2 and B2, the keys-only network (N10) for the W1 family, the u64 word for sorted output (KB 12, 14, 19, 11.7) | V |
| TK-S12.3 | fp8 | TK-P12 | 8-bit SIMD min and max are a 28-cycle emulation on sm_100, widen to 16-bit lanes with prmt (T8, KB 2.10) | open |
| TK-S12.4 | fp16 | TK-P12 | The bf16 word with a different twiddle, unmeasured (KB 10.1) | open |
| TK-S12.5 | fp64 | TK-P12 | Untested, the striped B1 raised f64 keys from 64 to 74 registers (KB 4.4), W4 and B6 use 64-bit words (KB 23, 10.5) | open |
| TK-P13 | Harness and methodology | infrastructure | The harness found three node and harness effects and four correctness bugs that random input never would have (KB 2.10, 10.4, 10.6) | V |
| TK-S13.1 | benchmarking methodology | TK-P13 | Chain slope net of a generate-only control with the variant's shared memory, one-wave throughput, canaries, lock sections, the oracle before any number (KB 10.1, 10.4, `harness/README.md`) | V |
| TK-S13.2 | sorted-output check | TK-P13 | The harness had no order check until verify-W4, B3_tuned fails 20 of 44 cases as the negative control (KB 2.10) | V |
| TK-S13.3 | signed-zero battery | TK-P13 | Five seeds plus the boundary and pivot_tie_zero patterns, after W5_v3 passed a single seed with a broken contract (KB 2.10) | V |
| TK-S13.4 | oracle self-test | TK-P13 | A false negative at k equal to N over 2 in harness 0.3 and earlier (KB 2.10) | V |
| TK-S13.5 | node pre-flight and outlier rule | TK-P13 | One B200 ran 128-thread kernels above about 64 KB of SASS 1.29x to 2.04x slower with every synthetic identical (T58, KB 2.10, 10.6), and both GB200 hosts run the 128-thread k 64 configuration 3 to 15 percent slower than the healthy B200s, a generation property with an open mechanism (KB 2.12) | V |
| TK-S13.6 | phase stamps | TK-P13 | clock64 stamps between phases found every structural cost of B4, N3 and the deterministic AIR path that the instruction count hid (T56, KB 2.10) | V |
| TK-S13.7 | harness version policy | TK-P13 | A card is judged under its harness version, the 0.6 control change moved shared-memory-heavy controls by 9 to 11 percent (KB 9, 2.10, `cards/README.md`) | V |

### A.3 Existing issues and PRs to update instead of creating

- **#9747, the strategy list.** Its body lists 1 (warp bitonic, TK-P03), 2 (warp redux, TK-P01), 4 (atomics, TK-P06), 5 (cluster radix selection, the cluster backend of knowledge base section 3, not part of this study) and 6 (multi-CTA AIR, knowledge base section 5, not part of this study), and no 3. The block radix select itself (B1, TK-P09) has no entry, which the comment says. New numbers from 7: 7 W5 (TK-P02), 8 N10 (TK-P04), 9 N6 with N9 (TK-P05), 10 B4 (TK-P07), 11 B3 (TK-P08), 12 W4 (TK-P10), plus the three cross-cutting parents without a number. The update is one comment with the extended list in the issue's own format (one line per strategy, comparison method, where it is expected to do well, owner TBD), each line pointing to its parent issue, and the winner matrix pointer on the fork branch (`dsb_experiments/topk_swarm/WINNERS_b200.md` on the fork branch, B200 umbriel-b200-047, harness 0.10, 2026-10-25, block, warp and sub-warp scopes, and `WINNERS_gb200.md`, gb200-nvl4-ts2-99, harness 0.10, 2026-10-26, warp and sub-warp scopes plus the block cells at N 1024 and 4096). The "Implementation overview" the issue leaves as TBD can be filled from the phase 3 winners files.
- **#9749, warp bitonic top-k (strategy 1).** Its body is the template of every parent body below. One comment: the codegen attribution of the W1 contradiction (three multiplying factors, T42, card `W1x_codegen` verified on second hosts of both archs), the min and max route for the bf16 word, the sub-warp cards, the N5 two-row form, the T20 aarch64 observation, and the cells the family keeps after wave 2 (N at most 64 at k 16 to 32, bf16 k 64 at N 128 and 256, the sorted-output pass up to the W4 crossover). TK-P03 links to it as the parent and the sub-issues hang under TK-P03, not under #9749.
- **#12160, the min and max operator.** One comment with the measured evidence: two predicated VIMNMX per stage against two ISETP and a SEL, 1.15x to 1.6x latency and 1.10x to 1.24x throughput on the same network (T42, KB 2.12), the bool comparator hides the form, host independent, 32-bit words only, and the note that ptxas does not emit VIMNMX for a bool comparator on u32 words contrary to the issue's integer remark (KB 11.3). TK-S03.1 is the implementation issue and links to it.
- **PR #12222, the WarpBitonicTopK eager form (open, "partially addresses #9749", follow-up PRs announced for an alternative implementation and an iterator-based overload).** Its follow-up list gets the sub-warp support (TK-S03.2), the min and max customization point (TK-S03.1), the index-in-key word for 16-bit keys (TK-S03.3), the N5 arrangement (TK-S03.4) and the pointer to the N10 route for fp32 under prefer_smaller_index (TK-P04).
- **PR #9281, the earlier WarpBitonicTopK PR with the eager and buffered algorithms (open).** A note that the measured level network removes most of the eager chain cost it targets and that W5 is the measured threshold-plus-compaction form at warp scope (KB 11.5, 18).

### A.4 Do not pursue (notes on the parents, no issues)

PR #8391 (WarpBitonicSort, closed, merged) needs no update, its crossover against the warp merge sort (N at most 224) stands and the study used the sort inside the W1 family.

- B5, hierarchical warp then block composition: 1.16x over B2 at one cell and 0.73x of its throughput, loses everywhere else, one warp works while seven wait (R13, card `B5_hier_2`, verified). Note on TK-P06.
- B6, sort and truncate for selection: 2.1x to 6.5x behind B3_tuned at every k from N over 8 to N over 2 (R17, cards `B6_radix_1` and `_2`, unverified by rerun, the margin is the evidence). Its sorted-output premium and the keys-only sort of the packed word survive as the post-sort route in TK-S08.5 and TK-S11.2. Note on TK-P08.
- N2, the level-parallel restructuring by itself: 0.96x to 1.08x with identical network code (R12, verified). The one-word-per-tile case lives on as TK-S03.5. Note on TK-P03.
- N3, streaming selection on resident data: 2.1x to 2.7x over its bar (card `N3_stream`, unverified by rerun). The streaming contract is a separate workload needing a harness arrangement that loads chunks inside the call, recorded in KB 2.12. Note on TK-P05 and TK-P13.
- N7, the two-level per-lane list for the W2 rounds: 1.58x against a 2x bar and 36 percent behind W5 (R18, card `N7_twolevel`, verified). Note on TK-P02.
- Redux pre-reduction before a block round (R14) and the speculative post (R15): ptxas already aggregates the plain round and moves CREDUX across BAR.SYNC, both lose (cards `B2_redux8`, `B2_spec`, `B2_spec3`, verified). Note on TK-P06.
- Per-slot ballot aggregation of tickets (R16) and the empty-asm ticket fix (T39): the first serializes the slots, the second is a no-op (cards `B1_idxw`, verified, `B1_det` verifier notes). Note on TK-P09.
- 64-thread blocks for latency, 1024-thread blocks for radix select, 512-thread B2 with 16 or more rounds, 128-thread two-level list from N 16384 (section 17, verified). Notes on TK-P06 and TK-P09.
- W3 and N1 as primitives of their own: W3 is the fallback behind N6 at warp scope and the k 1024 row, N1 has no cell (cards `W3_d4c`, `N1_bitserial`, verified on both archs). Note on TK-P05.
- The 4-ary merge path and the two-half per-lane sort of W4 (cards `W4_chunk4q`, `W4_wmerge2`, verified). Note on TK-P10.

---

## Part B. The manifest for an agent with gh

### B.0 How to run this manifest

Pre-flight, before any creation:

1. Replace every `<phase 3>` in this file from `WINNERS_b200.md` and `WINNERS_gb200.md` (tools/winners.py output, node and harness version per cell). If a `<phase 3>` cannot be filled, leave the sentence out, never invent a number. The strategy mapping of A.3 is final (read from the saved body of #9747).
2. The labels below are from the repository's list (`state/github_refs/labels.json`): `cub` on every issue, `future-productization` on the approach parents, `bench` on TK-P13 and its sub-issues, `docs` on the documentation sub-issues. Create nothing, substitute nothing.
3. Every pointer in the bodies is a path under the fork branch `https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm`. Ask the user whether those pointers stay or are replaced (for example by an upstream branch) before creating anything, the knowledge base page is private and is not linked.
4. "KB n" in a Pointers section means section n of the knowledge base. At the end of phase 4 the knowledge base markdown is committed to the fork branch as `dsb_experiments/topk_swarm/KNOWLEDGE_BASE.md`, a copy of the master file at that time with the private artifact link removed from its conventions section, so every "KB n" reference resolves to that path. Check that the file exists on the branch before creating issues, and never add the private link back into the copy.

Creation order: TK-P01 to TK-P13 first (parents), in that order, then every TK-S block in the order written. Record the number GitHub assigns to each parent, the sub blocks refer to parents by their TK id.

Creating one issue (the body is the fenced block of the issue, written to a file first so that the markdown survives the shell):

```
gh issue create --repo NVIDIA/cccl --title "<title>" --label "<label1>" --label "<label2>" --body-file body_TK-P01.md
```

Linking a sub-issue to its parent after both exist (REST, the sub_issues endpoint takes the sub-issue's numeric id, not its number):

```
SUB_ID=$(gh api repos/NVIDIA/cccl/issues/<sub number> --jq .id)
gh api --method POST repos/NVIDIA/cccl/issues/<parent number>/sub_issues -f sub_issue_id=$SUB_ID
```

The GraphQL form, if the REST endpoint is not enabled for the repository:

```
PARENT=$(gh api graphql -f query='{repository(owner:"NVIDIA",name:"cccl"){issue(number:<parent number>){id}}}' --jq .data.repository.issue.id)
CHILD=$(gh api graphql -f query='{repository(owner:"NVIDIA",name:"cccl"){issue(number:<sub number>){id}}}' --jq .data.repository.issue.id)
gh api graphql -f query='mutation($p:ID!,$c:ID!){addSubIssue(input:{issueId:$p,subIssueId:$c}){issue{number}}}' -f p=$PARENT -f c=$CHILD
```

Updates to existing issues (A.3) are comments, one per issue, posted after the parents exist so that they can link to them:

```
gh issue comment 9747 --repo NVIDIA/cccl --body-file comment_9747.md
```

Every body below ends with a Pointers section and an owner line, which the agent keeps verbatim (the owner is TBD unless the user names one). In the guarantee tables "yes" means supported and measured, "planned" means reachable by a listed follow-up, "no" means not available, "TBD" means not determined, the user may replace them by the legend symbols of #9747. "KB" is the knowledge base, sections as numbered there. Dependencies name the TK ids that should exist first (for links in the body), nothing blocks creation otherwise.

---

### TK-P01

- Title: `[CUB] Warp and sub-warp top-k by packed redux extraction (strategy 2)`
- Labels: `cub`, `future-productization`
- Parent: none (strategy 2 of #9747)
- Dependencies: none

```markdown
## Summary

- A warp-level top-k for 16-bit keys that runs k rounds of `redux.sync` max on a 32-bit word holding the twiddled key above the bit-inverted source index. The word orders like (key descending, index ascending), so every round is an exact arg max with no ballot, election or tie machinery, the winner lane recognizes itself, and `prefer_smaller_index` costs nothing. fp32 keys go through a truncated word (the top 32 minus log2 N key bits above the index) with an exact tie finish on the next word. Measured as W2 of the cooperative primitives study.
- **Scope**: Warp of 32 lanes and logical warps of 4, 8 and 16 (one full-warp redux per logical warp with the identity in the other lanes), N 32 to 2048 (N 4096 unmeasured), k 4 to 256, bf16 and fp32 keys with int32 indices, both unsorted requirements by the same code, sorted output by construction. Per-lane sorted list of depth min(k, items per lane) with the shifts written as selects so the compiler prunes the positions that cannot reach the head.
- **Comparison method**: The study harness: latency as the chain-length slope of clock64 cycles per call net of a generate-only control launched with the variant's shared memory, throughput as one occupancy wave in Gelem/s, variant and control in one lock section, a host oracle with a 23-case battery before any number, canaries at both ends of every batch, every card rebuilt from its commit by a verifier on another session or node. B200, CUDA 13.4.92, 2026-10-25.
- **Status**: Prototype adapters `W2_packed`, `W2_packed_v2`, `W2_packed_v3`, `W2_packed_sw_v2` on branch `exp/topk-primitives-swarm` (elstehle/cccl), verified cards `W2_packed_b200_20261025_1`, `_v2`, `_v3`, `_sw_v2`. The round costs 24.9 to 29.6 cycles inside the primitive at 8 items per lane. Against `cub::WarpBitonicTopK` in one session: bf16 N 512 k 16 820 against 12606 cycles, N 2048 k 16 2377 against 68874 (card `W2_packed_v3`, B200 umbriel-b200-047 and 091). Final cells (phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25, one session): W2_packed_v3 wins 42 of 89 warp latency cells under not_guaranteed (median 825 percent faster than cub::WarpBitonicTopK as the reference and 61 percent ahead of the runner-up) and 26 of 70 under prefer_smaller_index at identical cycles: every k 4 to 32 cell at N 32 to 512 except fp32 N 32 k 16 and fp32 N 512 k 32, bf16 N 1024 k 8, fp32 N 1024 k 16 (a 1.2 percent tie with W5_v4), bf16 N 2048 and 4096 k 4, and every MoE router cell of the named workloads (Qwen3 N 128 k 8 262 cycles, DeepSeek-V3 N 256 k 8 467, Kimi K2 N 384 k 8 441, Kimi K3 N 896 k 16 1314). Representative: bf16 N 256 k 8 329 cycles (W5_v4 101 percent behind, reference 5950), bf16 N 512 k 16 829 (W5_v4 29 percent behind), fp32 N 256 k 16 660 (N6w_atom32 61 percent behind). Throughput: 21 of 87 cells at k 4 to 16 up to N 512 (bf16 N 128 k 4 1022 Gelem/s, N 512 k 4 1374). The cells with a runner-up within 15 percent (bf16 N 32 k 16, N 512 k 32, N 1024 k 8, N 2048 k 4, fp32 N 256 k 32, N 512 k 24, N 1024 k 16 and the PEER sub-key cell at the same fp32 N 1024 k 16 shape) are close calls (runner-up within 15 percent). On the tie_heavy input as generated W2_packed_v3 keeps 16 of the 22 close calls that were rerun and N6w_atom32 takes six (bf16 N 512 k 32, fp32 N 256 k 32 and N 512 k 24 under both requirements), evidence about that layout rather than a verdict, so for those six the choice stays with the implementer unless the shuffled variant confirms the flip.

## Problem coverage

Owns the warp latency cells up to k 32 for bf16 and up to k 64 for fp32 from N 128 on, and the k 16 cells at every N measured. Hands over to W5 (threshold filter) from k 16 at N 1024 and 2048 for bf16, to N6 at warp scope from k 64 at N 256 to 2048, and keeps the one-wave throughput cells up to k 16. The round is the finish inside W5 and N6, so it stays load bearing above those crossovers.

### Segment / input size
- **Supported range**: N 32 to 2048 per warp (N 4096 built, unmeasured), 1 to 64 items per lane.
- **Expected sweet spot**: N 128 to 2048 at small k, the 2026 router shapes of N 256 to 1024.

### k
- **Supported range**: k 4 to 256, compile-time k, k at most N over 2.
- **Expected sweet spot**: k at most 32 (bf16) and at most 64 (fp32), the per-lane list depth min(k, items per lane) sets the round cost.

### Types
- **Supported**: bf16 (packed key-plus-inverted-index word), fp32 (truncated word plus an exact tie finish), int32 indices.
- **Expected sweet spot**: 16-bit keys. fp16 unmeasured (the bf16 word with a different twiddle), fp8 through widening (T8), fp64 not applicable to the packed word.

## Output guarantees

### Determinism (set membership)
| `determinism` | Supported | Notes |
|---|---|---|
| `not_guaranteed` | yes | The reference requirement of every measurement. |
| `run_to_run` | yes | A fixed sequence of redux rounds, no atomics. |
| `gpu_to_gpu` | yes | The word order (key descending, index ascending) is the result. |

### Tie-break (only meaningful with `gpu_to_gpu`)
| `tie_break` | Supported | Notes |
|---|---|---|
| `unspecified` | yes | Resolved by the word order. |
| `prefer_smaller_index` | yes | Zero cost, identical cycles in 56 pairs, the same binary. |
| `prefer_larger_index` | planned | The uninverted index in the word, one instruction less, not built. |

### Output ordering (result sequence)
| `output_ordering` | Supported | Notes |
|---|---|---|
| `unsorted` | yes | Trivially. |
| `sorted` | yes | By construction, each round emits the next largest word. |
| `stable_sorted` | yes | The word order is a stable order over (key, index). |

## Pointers

Knowledge base sections 12 and 16 (fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm), report `approaches/W2/REPORT.md`, techniques T19, T37, T38, T41 in `TECHNIQUES.md`, idea card `ideas/queue/W2_redux_extraction.md`.

Owner: TBD
```

### TK-S01.1

- Title: `[CUB] W2 fp32 route: pin the key loads above the selection network (v4)`
- Labels: `cub`
- Parent: TK-P01
- Dependencies: TK-P01

```markdown
## Summary

- The v2 fp32 form regresses at N 1024 k 8 and N 2048 k 16 and 32 because ptxas sinks three key loads into the selection network and wraps the emission stores in reconvergence regions. The fix pins the key registers at the start of the call with an empty asm and stores the uniform winner from all lanes.
- **Scope**: The fp32 truncated-word adapter at 32 and 64 items per lane, the three regression cells.
- **Comparison method**: The study harness, v4 against v3 and v2 in one lock section.
- **Status**: Proposed in the W2 report, not built. Regressions verified by verify-W2 (card `W2_packed_v2`, B200, CUDA 13.4.92, 2026-10-25).

## Problem coverage

Restores the fp32 k 16 to 32 cells at N 2048 where `W2_packed` (v1) is faster than v3 today.

## Output guarantees

Unchanged from the parent.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 12.4, 12.5, 12.7. Technique T24.

Owner: TBD
```

### TK-S01.2

- Title: `[CUB] W2 at sub-warp scope: one full-warp redux per logical warp`
- Labels: `cub`
- Parent: TK-P01
- Dependencies: TK-P01

```markdown
## Summary

- Logical warps of 4, 8 and 16 lanes share one hardware warp. A masked redux per logical warp lowers to a WARPSYNC loop and costs 174 to 649 cycles per round, one full-warp redux per logical warp with the identity in the other lanes costs 34 to 88 (T37, card `W2_packed_sw_v2`, verified, B200, CUDA 13.4.92, 2026-10-25).
- **Scope**: The router shapes of KB 10.2: N 8 to 128 at logical warps 4, 8 and 16, bf16 and fp32.
- **Comparison method**: The study harness against the masked form and against the W1 sub-warp cards.
- **Status**: Prototype verified. The named router cell (logical warp 8, N 128, k 8) reads 802 cycles bf16 and 1184 fp32 against 3877 and 4189 for the W1 family. Final cells (phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25, the lane count chosen per cell): W2_packed_sw_v2 at 16 lanes wins every bf16 latency cell from N 32 (N 32 k 4 206 cycles against the sub-warp bitonic baseline 820, N 128 k 4 342 against 3980, k 16 785 against 4297, k 64 2593 against 3704) and the fp32 k 4 cells (N 32 363, N 128 578), its own 8-lane form the runner-up by 48 to 58 percent at k 4 and 16. Throughput goes to fewer lanes: 4 lanes at bf16 N 32 k 4 (258 Gelem/s) and fp32 N 8 k 4 (100), 8 lanes at bf16 N 32 k 16 (228), N 128 k 4 (1071) and fp32 N 32 k 4 (477), 16 lanes at fp32 N 128 k 4 (717), the other lane counts within 1 to 13 percent on four of these cells, which are close calls (runner-up within 15 percent). Lane-count close calls are the implementer's choice by the lane count the surrounding kernel already has. The router cell at logical warp 8, N 128, k 8 was not in the pass: `<phase 3>` (lane-count addendum).

## Problem coverage

MoE routing at the 2026 flagship shapes.

## Output guarantees

As the parent, the contract requires all lanes of the hardware warp in the call.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 12.4, 16 (sub-warp paragraph), T37.

Owner: TBD
```

### TK-S01.3

- Title: `[CUB] W2 at N 4096: compile time of the k 256 and k 1024 instantiations`
- Labels: `cub`
- Parent: TK-P01
- Dependencies: TK-P01

```markdown
## Summary

- The N 4096 translation units at k 256 and 1024 compile for more than 20 minutes with the fully unrolled rounds and pruned list, so the cells are unmeasured.
- **Scope**: Compile-time control of the unrolled round count, or a rolled form above a k threshold that keeps the list pruning for the head.
- **Comparison method**: Compile time per translation unit, then the study harness for the cells.
- **Status**: Open, no build.

## Problem coverage

Warp N 4096 is the one unmeasured warp row of the winners matrix.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 12.5, 16, T38, T41, T48.

Owner: TBD
```

### TK-S01.4

- Title: `[CUB] W2: two candidates per round at 8 items per lane and below`
- Labels: `cub`
- Parent: TK-P01
- Dependencies: TK-P01

```markdown
## Summary

- Emit two candidates per redux round when the list is shallow, about 1.2x on the model. Built as `sw_v3`, never measured.
- **Scope**: Warp and sub-warp forms at 8 items per lane and below.
- **Comparison method**: The study harness against `W2_packed_v3` in one lock section.
- **Status**: Built, unmeasured.

## Problem coverage

The small-N router cells where the rounds are the whole call.

## Output guarantees

Unchanged if both candidates are drawn in word order.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 12.5.

Owner: TBD
```

### TK-S01.5

- Title: `[CUB] W2: GB200 card`
- Labels: `cub`
- Parent: TK-P01
- Dependencies: TK-P01

```markdown
## Summary

- W2 was measured on B200 only. T16 says the SM cycle costs transfer and T20 says the aarch64 compare-network effect does not touch explicitly written exchanges, a card closes the question.
- **Scope**: The focus cells on one GB200 under the frozen harness version.
- **Comparison method**: The study harness, same adapters, same controls.
- **Status**: GB200 pass (gb200-nvl4-ts2-99, harness 0.10, 2026-10-26, same cells and controls as the B200 pass): W2_packed_v3 wins the same 42 of 89 warp latency cells as on B200 with the GB200 values within 2 percent of the B200 values on every W2 cell but bf16 N 4096 k 4 (0.994), throughput a uniform 7.5 percent higher on both the winners and the W1_reference control. At sub-warp scope the 16-lane W2_packed_sw_v2 form takes the named cells GB200 ran at every lane count (gpt-oss-20b bf16 N 32 k 4 206 cycles, DBRX fp32 N 16 k 4 328, DeepSeek group top 2 N 32 k 2 302, MoBA N 64 k 4 469), pending the B200 lane-count addendum for the comparison.

## Problem coverage

Arch coverage of the winner matrix.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 12.7, T16, T20.

Owner: TBD
```

---

### TK-P02

- Title: `[CUB] Warp top-k by a certified threshold filter with a packed finish (strategy 7)`
- Labels: `cub`, `future-productization`
- Parent: none (new strategy 7)
- Dependencies: TK-P01 (the finish is W2's round)

```markdown
## Summary

- A warp-level top-k for 32 items per lane and above that builds an exact lower bound on the k-th largest key from grouped lane statistics (G groups certify their top J, G times J at least k), keeps the items at or above it with one compare each, compacts them through shared memory with log2(IPT + 1) ballots for the prefix, and finishes with the packed redux rounds on at most D candidates per lane. No retry, an exact bit-serial fallback for tie floods. Measured as W5 of the study.
- **Scope**: Warp of 32 lanes, N 256 to 65536 (bf16) or 32768 (fp32), k 4 to 128, bf16 and fp32 keys, temp storage 512 bytes to 4 KB per warp, one `__syncwarp`, no barriers. Measured at N 512 to 2048 and k 8 to 64.
- **Comparison method**: The study harness (see TK-P01), W5 against `W2_packed_v3` in one lock section, phase probes with clock64 stamps for the attribution.
- **Status**: Prototype adapters `W5_base` to `W5_v4` on `exp/topk-primitives-swarm`, cards `W5_base`, `W5_v2`, `W5_v4` verified to the cycle on umbriel-b200-028 (B200, CUDA 13.4.92, 2026-10-25), `W5_v3` refuted on contract (signed zeros, see TK-S02.6). Against `W2_packed_v3`: bf16 N 1024 k 16 1337 against 1485, k 32 1837 against 2481, k 64 2936 against 7308, N 2048 k 16 1861 against 2377, k 64 4163 against 10071, fp32 N 2048 k 16 2293 against 4391, throughput 1.31x to 2.79x from k 16 at N 512. Final cells (phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25): W5_v4 wins 21 of 89 warp latency cells under not_guaranteed (median 2320 percent ahead of the reference, 51 percent ahead of the runner-up) and 20 of 70 under prefer_smaller_index at identical cycles: bf16 N 1024 k 16 to 64 (k 16 1353 cycles against W2_packed_v3 1521, k 32 1839 against 2526, k 64 2957 against N6w_atom32 3592), fp32 N 1024 k 8, 24 and 32, k 16 to 64 at N 2048 and 4096 under both key types (fp32 N 2048 k 16 1987 against W2 4515, bf16 N 2048 k 64 3877 against W4_wmerge 8419, fp32 N 4096 k 64 9269 against W2 19956) and fp32 k 4 at N 2048 and 4096. It does not take fp32 N 1024 k 48 and 64 (N6w_atom32 3454 and 3709 against 3675 and 4713). Throughput: 34 of 87 cells, every N 1024 to 4096 cell from k 4 to 64 (bf16 N 2048 k 16 1312 Gelem/s, 52 percent ahead of W2, fp32 N 2048 k 64 500, 120 percent ahead) plus bf16 N 512 k 16 and 24 and fp32 N 512 k 4 and 16. bf16 N 1024 k 16, fp32 N 1024 k 32 and fp32 N 4096 k 4 in latency and the N 512 and N 4096 k 4 throughput cells are close calls (runner-up within 15 percent). On the tie_heavy input as generated W2_packed_v3 or N6w_atom32 take 14 of W5_v4's 19 rerun close calls (W2 the bf16 N 1024 k 16 latency cell and the N 512 k 16 and N 4096 k 4 throughput cells, N6w_atom32 the fp32 N 1024 k 32 latency cell and the k 24 to 64 throughput cells at N 512 and 1024), since W5's exact lower bound is the thread-max bound that the layout-aligned pattern degenerates (T67), evidence about that layout rather than a verdict, so the choice stays with the implementer unless the shuffled variant confirms the flip.
- **Caveat, sorted output**: the phase 3b sorted check (B200 umbriel-b200-047, harness 0.10, 2026-10-26) refutes W5_v4's fp32 sorted output under prefer_smaller_index as built: at k 16 and 64 of N 512, 2048 and 4096 equal keys inside the top k come out in buffer-position order instead of index order, because the fp32 finish word replaces the low 7 to 9 twiddled key bits by the inverted buffer position and the tie rewrite covers only the class that straddles the k-th slot. The selected set is exact, the bf16 route is sorted by construction, and the unsorted winners above stand. The fix is in the emission, the tie rewrite generalized to every class run inside the top k, which also closes a latent ordering hole under not_guaranteed when two distinct keys agree in the kept bits (not observed in the pass, a host simulation puts such a pair in 1 to 16 percent of rows at k 16 to 64). W5_v4's sorted fp32 output is a sub-item until then.

## Problem coverage

Takes the warp bf16 cells from k 16 at N 1024 and 2048 and the fp32 cells from k 32 at N 1024 and k 16 at N 2048, keeps the sorted-output route up to k 64 (its finish word is sorted), loses N 512 and k 8 to W2 and k 64 at N 256 to 512 and k 128 and above to N6 at warp scope (TK-P05).

### Segment / input size
- **Supported range**: N 256 to 65536 (bf16) or 32768 (fp32), measured at N 512 to 2048.
- **Expected sweet spot**: 32 items per lane and above, N 1024 and 2048.

### k
- **Supported range**: k 4 to 128 with 2k at most N.
- **Expected sweet spot**: k 16 to 64, where the pruned per-lane list of W2 stops paying and the bound still certifies a short candidate list (80 at k 32, 126 at k 64 on average).

### Types
- **Supported**: bf16 and fp32 keys, int32 indices.
- **Expected sweet spot**: both, fp32 within 3 percent of bf16 on this structure.

## Output guarantees

### Determinism (set membership)
| `determinism` | Supported | Notes |
|---|---|---|
| `not_guaranteed` | yes | The reference requirement of every measurement. |
| `run_to_run` | yes | A data-dependent candidate count but a fixed exact result. |
| `gpu_to_gpu` | yes | The finish word and the fallback tickets are index ordered. |

### Tie-break (only meaningful with `gpu_to_gpu`)
| `tie_break` | Supported | Notes |
|---|---|---|
| `unspecified` | yes | Resolved by the word order. |
| `prefer_smaller_index` | yes | Zero cost, identical cycles and instruction streams in 12 pairs (v4). The v3 form broke it at signed zeros. |
| `prefer_larger_index` | planned | Not built, the mirror form is listed in the report. |

### Output ordering (result sequence)
| `output_ordering` | Supported | Notes |
|---|---|---|
| `unsorted` | yes | Trivially. |
| `sorted` | yes | Descending key on the filter path (fp32 by truncated key with the tie class rewritten), ticket order on the fallback. |
| `stable_sorted` | yes | On the filter path through the word order. |

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 18 and 16, report `approaches/W5/REPORT.md`, techniques T48, T49, host simulation `approaches/W5/sim_gj.py`, phase probes `phase_probe.cu` and `phase_probe3.cu`.

Owner: TBD
```

### TK-S02.1

- Title: `[CUB] W5: halve the shared store count of the candidate scatter`
- Labels: `cub`
- Parent: TK-P02
- Dependencies: TK-P02

```markdown
## Summary

- Each shared store instruction of the scatter costs about 10 cycles on one warp whether predicated or not, 32 per call at 32 items per lane (332 cycles by the v3 phase probe at bf16 N 1024 k 32, where the whole v4 call is 1837 in the harness). Two candidates per 64-bit store when the lane's run start is even, or a fused count-and-scatter into per-lane private runs with a prefix-free gather, would halve or remove them.
- **Scope**: The scatter and gather phases of W5_v4.
- **Comparison method**: Phase probe plus the study harness against W5_v4.
- **Status**: Open. Phase costs verified by the verifier's rerun of the probes (B200, CUDA 13.4.92, 2026-10-25).

## Problem coverage

Every W5 cell, the largest W5-specific cost left.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 18.3, 18.5, T12.

Owner: TBD
```

### TK-S02.2

- Title: `[CUB] W5: cheaper per-lane top 2 and ballot prefix`
- Labels: `cub`
- Parent: TK-P02
- Dependencies: TK-P02

```markdown
## Summary

- The per-lane top 2 costs about 160 cycles at 32 items per lane and the ballot prefix 216, the next two items after the scatter. A pair tree for the top 2 and fewer count bits for the prefix (a per-lane count above 15 means the 16 by 2 bound is loose and can go to the fallback).
- **Scope**: W5_v4 at k 32 and 64.
- **Comparison method**: Phase probe plus the study harness.
- **Status**: Open.

## Problem coverage

Every W5 cell.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 18.3, 18.5.

Owner: TBD
```

### TK-S02.3

- Title: `[CUB] W5: register pressure of the hoisted flags, occupancy in the throughput regime`
- Labels: `cub`
- Parent: TK-P02
- Dependencies: TK-P02

```markdown
## Summary

- Since v3 the unconditional scatter lets ptxas hoist the flag and address arithmetic of every item, 128 registers at N 1024 and 255 at N 2048 against 72 and 128 in v2, one block per SM in the throughput regime at N 2048. A per-chunk ordering pin would give the occupancy back.
- **Scope**: W5_v4 at N 1024 and 2048.
- **Comparison method**: The study harness, throughput regime.
- **Status**: Open.

## Problem coverage

The batched throughput cells at N 2048.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 18.3, 18.5, T24.

Owner: TBD
```

### TK-S02.4

- Title: `[CUB] W5 at N 4096 and k 128`
- Labels: `cub`
- Parent: TK-P02
- Dependencies: TK-P02

```markdown
## Summary

- N 4096 (128 items per lane) and the k 128 band are unmeasured. The seed check says the configured bound holds at k 128 (284 of 512 slots at worst). N7's measured k 128 cells and N6 at warp scope hold the band today.
- **Scope**: Cells N 4096 against W1_reference and N 1024 to 2048 at k 128 against N6w and W4.
- **Comparison method**: The study harness in one lock section with those controls.
- **Status**: Open.

## Problem coverage

The upper edge of W5's band.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 18.5, 18.7, 22.

Owner: TBD
```

### TK-S02.5

- Title: `[CUB] W5: a second fallback rung for tie floods`
- Labels: `cub`
- Parent: TK-P02
- Dependencies: TK-P02

```markdown
## Summary

- When more than 32 times D candidates pass (tie floods, periodic patterns), W5 falls back to an exact bit-serial radix select over the register tile. A second rung, the exact k-th largest of the certified set and a refilter, would keep such inputs on the fast structure.
- **Scope**: The overflow path, measured on tie_heavy, pivot_tie and all_equal.
- **Comparison method**: The study harness on the battery patterns.
- **Status**: Open. The fallback is exact and passes the battery.

## Problem coverage

bf16 inputs with heavy ties at rank k (KB 10.2).

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 18.5.

Owner: TBD
```

### TK-S02.6

- Title: `[CUB] Threshold logic on twiddled keys must treat signed zeros explicitly`
- Labels: `cub`
- Parent: TK-P02
- Dependencies: TK-P02

```markdown
## Summary

- W5_v3's flag was the sign bit of a minus t on raw keys while the threshold came back from a twiddle that maps minus zero to plus zero, so items equal to minus zero were dropped when the threshold was plus zero. One signed_zero seed passed under not_guaranteed because exactly k plus zeros remained, 32 seeds under prefer_smaller_index failed in every cell. v4's load normalization fixes it, the lesson is a contract test.
- **Scope**: Every threshold or tie test on fp32 and bf16 keys derived from a bin edge or a twiddled word.
- **Comparison method**: The 23-case oracle battery of the harness with five signed_zero seeds, signed_zero_boundary and pivot_tie_zero (TK-S13.3).
- **Status**: Found by verify-W5 (card `W5_v3` refuted on contract, B200, CUDA 13.4.92, 2026-10-25), the battery rerun passed with zero failures on W5_v4, W2, B2_tiecheck and both references.

## Problem coverage

Correctness, not performance.

## Output guarantees

The exact index set under prefer_smaller_index with mixed-sign zeros.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.10, 18.6.

Owner: TBD
```

---

### TK-P03

- Title: `[CUB] Warp bitonic top-k: min and max route, sub-warp support and the two-row form (strategy 1)`
- Labels: `cub`, `future-productization`
- Parent: none (strategy 1 of #9747, the family of #9749 and PR #12222)
- Dependencies: none

```markdown
## Summary

- The register-resident bitonic network of `cub::WarpBitonicTopK` (eager form) and its measured variants: the packed key-plus-inverted-index word for 16-bit keys, the min and max compare-exchange on that word (two predicated VIMNMX per stage against two ISETP and a SEL), logical warps of 4 to 16 lanes, and the two-row bf16x2 form for throughput. Measured as W1, N2 and N5 of the study, with a controlled codegen experiment that attributes the gains.
- **Scope**: Warp and sub-warp, N 8 to 4096, k 4 to 1024, bf16, fp16 (unmeasured) and fp32 keys, sorted and deterministic output by construction.
- **Comparison method**: The study harness (TK-P01). The codegen experiment held the algorithm fixed and changed one factor per kernel: the exchange, the loop structure against the CUB header, the tile order, on B200 and GB200.
- **Status**: Cards verified on both archs. The min and max exchange is 1.15x to 1.6x in latency and 1.10x to 1.24x in throughput on the same network and loop (N 256 k 64 bf16 1322 against 2104 cycles), the recursion's code structure another 1.03x to 1.41x over the CUB header, the level order 1.15x to 1.54x only with one word per tile (card `W1x_codegen_b200_20261025_1`, verified by verify-W1x on second hosts of both archs, B200 and GB200, CUDA 13.4.92, 2026-10-25). The bf16 index-in-key word is 1.36x to 2.38x over the pairs network. The family keeps N at most 64 at k 16 to 32, the bf16 k 64 cells at N 128 and 256, and the sorted-output pass up to the W4 crossover. Final cells (phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25): in the same-session matrix the W1 family holds one warp latency cell, bf16 N 128 k 64 (W1x_vimnmx_eager 912 cycles against N6w_atom32 914 and W1_level 916, a close call, runner-up within 15 percent), and one throughput cell, bf16 N 2048 k 1024 (156 Gelem/s against W1_level 151, a close call). At bf16 N 32 k 16 the level network (512) is 13 percent behind W2_packed_v3 (452, a close call), at N 32 k 4 4.1x behind (512 against 124), at N 256 k 64 W1x_vimnmx_eager (1322) is 2 percent behind N6w_atom32 (1296, a close call). At sub-warp scope W1_eager_same_sub at 4 lanes wins N 8 k 4 under both key types (200 and 219 cycles, its own 8-lane form 10 to 12 percent behind, a close call) and is 24 to 644 percent behind W2_packed_sw_v2 or N10_auto_sub elsewhere. On the tie_heavy input as generated N6w_atom32 takes the bf16 N 128 k 64 cell from W1x_vimnmx_eager (708 cycles against the network's random 912) and the bf16 N 2048 k 1024 throughput cell, and between W1x_vimnmx_eager and N6w_atom32 the choice stays with the implementer unless the shuffled variant confirms the flip.

## Problem coverage

Small N routing (N at most 64), the sorted-output pass at small k, the throughput regime through the two-row form, and the network inside N10 (fp32 exact route) and N5.

### Segment / input size
- **Supported range**: N 8 to 4096 at warp scope, N 8 to 128 at logical warps 4, 8, 16, registers run out at N 4096 (255 with spills).
- **Expected sweet spot**: N at most 64, and N 128 to 256 at k 64 for bf16 through the min and max word.

### k
- **Supported range**: k 1 to MaxK with MaxK a multiple of the logical warp size, measured k 4 to 1024.
- **Expected sweet spot**: k 16 to 64 at small N, small MaxK over the warp size.

### Types
- **Supported**: any comparable type (the CUB network), the packed bf16 word, fp32 as pairs, as the u64 word or through N10 (TK-P04).
- **Expected sweet spot**: 16-bit keys in the packed word, 32-bit words for the min and max exchange.

## Output guarantees

### Determinism (set membership)
| `determinism` | Supported | Notes |
|---|---|---|
| `not_guaranteed` | yes | The reference requirement of every measurement. |
| `run_to_run` | yes | Fixed data-independent network. |
| `gpu_to_gpu` | yes | Through the index-in-key word (bf16, u64 for fp32) or through N10 for fp32. |

### Tie-break (only meaningful with `gpu_to_gpu`)
| `tie_break` | Supported | Notes |
|---|---|---|
| `unspecified` | yes | Lane position of the network with keys only, as #9749 says. |
| `prefer_smaller_index` | yes | bf16 word zero (identical cycles in 128 pairs), fp32 u64 word plus 13 to 30 percent over the pairs, N10 negative. |
| `prefer_larger_index` | planned | The uninverted index, not built. |

### Output ordering (result sequence)
| `output_ordering` | Supported | Notes |
|---|---|---|
| `unsorted` | yes | Trivially. |
| `sorted` | yes | The network yields sorted order. |
| `stable_sorted` | yes | Through the index-in-key word only. |

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 11, 16, 2.12, report `approaches/W1/REPORT_W1b.md`, `approaches/W1/CODEGEN_EXPERIMENT.md`, techniques T20, T37, T42, T53, issues #9749 and #12160, PRs #12222 and #9281.

Owner: TBD
```

### TK-S03.1

- Title: `[CUB] WarpBitonicTopK: a min and max customization point for the compare-exchange`
- Labels: `cub`
- Parent: TK-P03 (updates #12160)
- Dependencies: TK-P03

```markdown
## Summary

- A bool comparator object hides the min and max form from the compiler: ptxas emits ISETP plus SEL for the CUB comparator on u32 words and two predicated VIMNMX only when the exchange is written as min and max. The network needs a customization point that lets a key type supply its min and max (packed unsigned words, fp32 with FMNMX).
- **Scope**: `cub::WarpBitonicTopK` and `WarpBitonicSort`, 32-bit words first (no 64-bit VIMNMX exists, the fp32 u64 word gains nothing).
- **Comparison method**: The codegen experiment of the parent, one factor per kernel.
- **Status**: Measured 1.15x to 1.59x per network (T42, verified on second hosts of both archs, B200 and GB200, CUDA 13.4.92, 2026-10-25), host independent. Not implemented in the header.

## Problem coverage

Every W1 family cell on 32-bit words, and N5 and N10, which use the form already in their own code.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.12, 11.4, 11.7, T42, T47, issue #12160.

Owner: TBD
```

### TK-S03.2

- Title: `[CUB] WarpBitonicTopK at sub-warp scope (logical warps 4, 8, 16)`
- Labels: `cub`
- Parent: TK-P03 (follow-up of PR #12222)
- Dependencies: TK-P03

```markdown
## Summary

- MaxK a multiple of the logical warp size instead of 32, shuffles with the full warp mask (the xor partner stays inside the segment), the harness guarantees that all lanes of the hardware warp call together. Measured as the `_sub` cards of W1 and W1b.
- **Scope**: Logical warps 4, 8 and 16 at N 8 to 128, bf16 and fp32.
- **Comparison method**: The study harness against the full-warp form and against W2's sub-warp form.
- **Status**: Prototype verified (cards `W1_eager_same_sub`, `W1_minmax_pairs_sub`, B200, CUDA 13.4.92, 2026-10-25). W2's sub-warp form is 2x to 7x faster wherever both were measured, the network leads only at N 8 (logical warp 8, k 4: 220 against 235). Final cells (phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25): W1_eager_same_sub at 4 lanes wins the N 8 k 4 latency cells (bf16 200 cycles against the sub-warp bitonic baseline 405, fp32 219 against 342, its own 8-lane form 10 to 12 percent behind, a close call, runner-up within 15 percent). In every other sub-warp latency cell the best W1 form is 24 percent (bf16 N 32 k 16 at 16 lanes, 767 against W2_packed_sw_v2 619) to 644 percent (bf16 N 128 k 4) behind the winner. Throughput: no W1 cell, the family is 23 percent behind N5_rows_fs_sub at bf16 N 8 k 4 (53 against 65 Gelem/s).

## Problem coverage

Routing at N 8 and the fp32 pairs at logical warp 4 N 32 where N10 loses.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 11.4, 16 (sub-warp paragraph), branch `exp/sub-warp-bitonic-sort`.

Owner: TBD
```

### TK-S03.3

- Title: `[CUB] WarpBitonicTopK: the bf16 index-in-key word`
- Labels: `cub`
- Parent: TK-P03 (follow-up of PR #12222)
- Dependencies: TK-P03

```markdown
## Summary

- For 16-bit keys pack the twiddled key above the bit-inverted index in one 32-bit word and run the network on it: one shuffle per stage instead of two, prefer_smaller_index free, sorted by construction.
- **Scope**: bf16 and fp16 keys with indices below 65536 (the index half).
- **Comparison method**: The study harness against the pairs network.
- **Status**: 1.36x to 2.38x latency and 2.0x to 2.6x throughput over the pairs network, identical cycles under both requirements (cards `W1_eager_same` and siblings, verified, B200, CUDA 13.4.92, 2026-10-25).

## Problem coverage

Every bf16 cell of the family.

## Output guarantees

prefer_smaller_index exact at zero cost, sorted output.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 11.4, 2.12.

Owner: TBD
```

### TK-S03.4

- Title: `[CUB] N5: two bf16 rows per lane in bf16x2 words for the throughput regime`
- Labels: `cub`
- Parent: TK-P03
- Dependencies: TK-P03

```markdown
## Summary

- Two independent rows of N keys per call, one per 16-bit half of a 32-bit word. max.bf16x2 and min.bf16x2 are the compare-exchange for both halves at once and one shuffle moves both, the indices are recovered from the k-th key of every half by a lane scan.
- **Scope**: bf16 only, the batched MoE router shape where every row shares N and k, warp and sub-warp.
- **Comparison method**: The study harness, one-wave throughput, against the one-word min and max network and W2.
- **Status**: 1.22x to 1.44x per call at k at most 32 and 1.43x to 1.82x at k 64 over the one-word network, 1.54x over W2 at N 256 k 32 and 2.5x to 3.2x at k 64, loses k 16 to W2, no latency gain at warp scope, a latency win at sub-warp scope from N 64 (cards `N5_rows_f` and siblings, verified on the cards' node, B200 umbriel-b200-018, CUDA 13.4.92, 2026-10-25). The exchange must be spelled with the SIMD calls inside the ternary (T53). Final cells (phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25): N5_rows_f wins 12 of 87 warp throughput cells under not_guaranteed, all bf16: N 32 k 4 and 16 (360 and 362 Gelem/s, W2_packed_v3 at 226 and 179, the reference 121), N 128 k 16 (473, W2 9 percent behind, a close call) and k 64 (427, N6w_atom32 22 percent behind), N 256 k 24 to 64 (449 to 506, N6w_atom32 10 to 15 percent behind, a close call), N 512 k 32 and 256 (521 and 361, a close call), N 4096 k 256 and 1024 (150 and 122, a close call). It loses k 4 to 16 at N 128 to 512 to W2 or W5_v4 (12 to 161 percent behind) and every N 1024 to 4096 cell at k 4 to 64 to W5_v4 (12 to 630 percent behind). Sub-warp: N5_rows_fs_sub wins bf16 N 128 k 16 and 64 at 16 lanes (604 and 458) and N 8 k 4 at 8 lanes (65). The cells marked as close calls have a runner-up within 15 percent. On the tie_heavy input as generated N6w_atom32 takes nine of N5_rows_f's 14 close calls (bf16 N 256 k 32 to 64, N 512 k 32 and 256, N 4096 k 256) and W3_d4c the N 4096 k 1024 cell, N5 keeps four, so the choice stays with the implementer unless the shuffled variant confirms the flip.

## Problem coverage

Batched bf16 routing in the throughput regime from k 32.

## Output guarantees

prefer_smaller_index exact, zero in the exact form, 2 to 4 percent in the fast form from never-executed fallback code, unsorted output.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 20, report `approaches/N5/REPORT.md`, T47, T53.

Owner: TBD
```

### TK-S03.5

- Title: `[CUB] WarpBitonicTopK: the level tile order at one word per tile`
- Labels: `cub`
- Parent: TK-P03
- Dependencies: TK-P03

```markdown
## Summary

- Sorting all tiles at once and merging level by level gains nothing with identical network code and two words per tile (N2 killed, 0.96x to 1.08x), and 1.15x to 1.54x in latency with one word per tile, confirmed on fp32 words by N10 (13 percent at four tiles, 27 percent at eight). The two-word loss at four tiles is a schedule effect still unexplained.
- **Scope**: The eager loop of the header at one word per tile (k at most the logical warp size).
- **Comparison method**: The codegen experiment and the N10 cards.
- **Status**: Measured, verified (cards `W1x_codegen`, `N10_u32_ltrunc`, B200, CUDA 13.4.92, 2026-10-25). Not in the header.

## Problem coverage

The latency cells with k at most 32 at N 128 and 256.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 11.4, 11.7, 19.3, R12 in 10.5.

Owner: TBD
```

### TK-S03.6

- Title: `[Toolchain] aarch64 cicc emits branches around compare-exchange swaps where x86 emits predicates`
- Labels: `cub`
- Parent: TK-P03
- Dependencies: none

```markdown
## Summary

- On the aarch64 host the device front end emits branches around every compare-exchange swap of the CUB comparator network where the x86 front end emits predicates. ptxas re-predicates both but leaves more instructions and a different register count, 1.17x to 1.26x on the CUB control at GB200 (T20). Explicitly written min and max exchanges are not affected.
- **Scope**: nvcc 13.4.92, sm_100 builds on aarch64 hosts, the CUB bitonic network.
- **Comparison method**: Same commit, same flags, both hosts, the codegen experiment's control kernel.
- **Status**: Measured and verified on second hosts of both archs (card `W1x_codegen`, B200 and GB200, CUDA 13.4.92, 2026-10-25). To be reported to the compiler team with the SASS.

## Problem coverage

Every compare network compiled on an aarch64 host.

## Output guarantees

Not applicable.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.10, 11.4, T20, `approaches/W1/w1x_sass_b200.txt` and `w1x_sass_gb200.txt`.

Owner: TBD
```

### TK-S03.7

- Title: `[CUB] WarpBitonicTopK fp32: the u64 word for sorted output and N 8`
- Labels: `cub`
- Parent: TK-P03
- Dependencies: TK-P03, TK-P04

```markdown
## Summary

- The fp32 index-in-key u64 word is 1.6x to 2.2x slower than the bf16 word at N 128 to 256 (two shuffles and a compare-select chain per stage) and N10 replaces it under both requirements at N 32 to 256. It keeps the sorted-output pass and N 8, where N10 loses to both network routes.
- **Scope**: The route table of the family for fp32.
- **Comparison method**: The N10 cards against `W1_eager_same` and the pairs.
- **Status**: Measured, verified (KB 19.4). The fp32 run of the codegen experiment (structure and order factors of the u64 word) is still owed.

## Problem coverage

Sorted fp32 output at small k, N 8.

## Output guarantees

prefer_smaller_index through the word, sorted by construction.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 11.7, 19, 16.

Owner: TBD
```

---

### TK-P04

- Title: `[CUB] fp32 warp top-k with exact tie-break: keys-only network and index recovery by ballot rank (strategy 8)`
- Labels: `cub`, `future-productization`
- Parent: none (new strategy 8)
- Dependencies: TK-P03

```markdown
## Summary

- fp32 keys twiddled to an order-preserving u32 word run through the min and max network without any index, the final merge stops at stride k and one redux.sync.min yields the k-th largest word, then every lane compares its original words against it and the indices are recovered by ballot rank in slot-major index order. prefer_smaller_index is exact by construction and the same code serves not_guaranteed. Measured as N10 of the study.
- **Scope**: Warp of 32 lanes and logical warps 4, 8, 16, N 32 to 256 at warp scope and 8 to 128 at sub-warp scope, k 4 to 64, fp32 keys, output unsorted (items above the threshold in index order, then the tied items in index order).
- **Comparison method**: The study harness (TK-P01), controls the fp32 u64 index-in-key word, the min and max pairs network under not_guaranteed, the u64 word in N10's own recursion (code structure held fixed) and the identical u32 network on bf16 words (network-only cost).
- **Status**: Nine cards verified by verify-N10 on a second node, every headline latency identical to the cycle (B200 umbriel-b200-020, 028 and 068, CUDA 13.4.92, 2026-10-25). `N10_auto` against the u64 word: 661 against 934 at N 64 k 32, 960 against 1918 at N 128 k 16, 1341 against 4650 at N 256 k 64, throughput 1.40x to 2.11x, and 0.99x to 0.27x of the best not_guaranteed route in all 13 warp cells in both regimes. Sub-warp router shape (logical warp 8, N 128, k 8) 1409 against 5299. Final cells (phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25): in the same-session matrix N10_auto wins two warp fp32 latency cells, N 32 k 16 (442 cycles, W2_packed_v3 9 percent behind, reference 583) and N 128 k 64 (944, N6w_atom32 12 percent behind, reference 2152), both close calls (runner-up within 15 percent), and two throughput cells (N 32 k 16 268 Gelem/s, N 4096 k 256 76, the latter a close call). The other fp32 small-N cells went to W2_packed_v3 (N10 12 to 386 percent behind at N 32 to 256). Sub-warp: N10_auto_sub at 16 lanes wins the fp32 k 16 and 64 latency cells (N 32 k 16 500 against the sub-warp bitonic baseline 927, N 128 k 16 1133 against W2_packed_sw_v2 1226, a close call, N 128 k 64 1262 against 4195) and the fp32 N 128 k 16 and 64 throughput cells at 8 lanes (403 and 320, a close call). prefer_smaller_index costs nothing in any N10 cell (identical cycles), nor in any other warp family. The router shape at logical warp 8 was not in the pass. On the tie_heavy input as generated N6w_atom32 takes N10's fp32 N 128 k 64 latency and N 4096 k 256 throughput close calls while N10 keeps fp32 N 32 k 16, and since W2 and N6w are also free of the deterministic requirement, the choice stays with the implementer unless the shuffled variant confirms the flip.

## Problem coverage

The fp32 prefer_smaller_index route of the W1 family at warp scope for N 32 to 256 and at sub-warp scope from N 32 (logical warps 8 and 16) or N 128 (logical warp 4), and the fastest fp32 route measured under either requirement there. Not served: sorted output, N 8, logical warp 4 at N 32 k 4 and 8.

### Segment / input size
- **Supported range**: N 32 to 256 at warp scope, N 8 to 128 at logical warps 4, 8, 16, MaxK dividing N with a power-of-two tile count.
- **Expected sweet spot**: N 64 to 256 at warp scope, N 32 to 128 at sub-warp scope (the router shapes).

### k
- **Supported range**: k 1 to N, the truncated merge for a power of two k, measured k 4 to 64.
- **Expected sweet spot**: k 16 to 64, where the saved shuffle stages and the recovery balance.

### Types
- **Supported**: fp32 keys only (the twiddled u32 word), int32 indices.
- **Expected sweet spot**: fp32 under prefer_smaller_index, where it is also the fastest route under not_guaranteed.

## Output guarantees

### Determinism (set membership)
| `determinism` | Supported | Notes |
|---|---|---|
| `not_guaranteed` | yes | The reference requirement of every measurement. |
| `run_to_run` | yes | A fixed network and a rank pass, no atomics. |
| `gpu_to_gpu` | yes | Exact by construction. |

### Tie-break (only meaningful with `gpu_to_gpu`)
| `tie_break` | Supported | Notes |
|---|---|---|
| `unspecified` | yes | Items above the threshold then the tied items, both in index order. |
| `prefer_smaller_index` | yes | The same code and cycles, 0.99x to 0.27x of the best not_guaranteed route. |
| `prefer_larger_index` | planned | The mirror rank (lanes above me), not built. |

### Output ordering (result sequence)
| `output_ordering` | Supported | Notes |
|---|---|---|
| `unsorted` | yes | Trivially. |
| `sorted` | planned | Not from the truncated forms, one shuffle per selected item with the complete merge (TK-S04.2). |
| `stable_sorted` | no | Not available. |

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 19, 16, report `approaches/N10/REPORT.md`, techniques T51, T52, code `approaches/N10/n10_detail.cuh`.

Owner: TBD
```

### TK-S04.1

- Title: `[CUB] N10: reach the pipelined-ballot cost in the two-class recovery`
- Labels: `cub`
- Parent: TK-P04
- Dependencies: TK-P04

```markdown
## Summary

- The two-class pass is issue bound at about 48 cycles per slot from 4 items per lane because ptxas interleaves the second ballot of each slot with the popc and add chains of the earlier slots. An asm barrier between the ballot block and the prefix block would let the ballots issue back to back, up to 100 cycles at 4 items per lane.
- **Scope**: The two-class recovery used at 1 and 2 items per lane (the one-ballot form above).
- **Comparison method**: The study harness against `N10_u32_trunc` in one lock section, SASS check of consecutive VOTE.
- **Status**: Open. The issue-bound cost is verified (KB 19.3).

## Problem coverage

Small-N cells and the tie path.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 19.3, 19.5, T17, T52.

Owner: TBD
```

### TK-S04.2

- Title: `[CUB] N10: sorted output by one shuffle per selected item`
- Labels: `cub`
- Parent: TK-P04
- Dependencies: TK-P04

```markdown
## Summary

- The truncated forms give no sorted output. The complete-merge form has the sorted keys in network positions, one shuffle per selected item moves the recovered index to its rank.
- **Scope**: The sorted-output pass for fp32 at warp scope.
- **Comparison method**: The study harness against the u64 word and W4_wmerge.
- **Status**: Open.

## Problem coverage

Sorted fp32 output at N 32 to 256.

## Output guarantees

Sorted by key, ties by index.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 19.2, 19.5.

Owner: TBD
```

### TK-S04.3

- Title: `[CUB] N10 at logical warp 4: segment-local minimum and two slots per ballot word`
- Labels: `cub`
- Parent: TK-P04
- Dependencies: TK-P04

```markdown
## Summary

- At logical warp 4 the group minimum is eight pipelined full-warp redux (about 88 cycles) and every ballot carries a segment LOP3, so N 32 k 4 and 8 lose to the pairs network by 1.20x and 1.28x. A two-step shuffle-xor min inside the segment (about 72 cycles) and two slots per ballot word (16 free lanes per vote) are the candidates.
- **Scope**: Logical warp 4 at N 32 and 128.
- **Comparison method**: The study harness against `N10_auto_sub` and the pairs.
- **Status**: Open. The losses are verified (card `N10_auto_sub`).

## Problem coverage

The small sub-warp router cells.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 19.4, 19.5, T40.

Owner: TBD
```

### TK-S04.4

- Title: `[CUB] N10: register footprint at sub-warp N 128 for batched router kernels`
- Labels: `cub`
- Parent: TK-P04
- Dependencies: TK-P04

```markdown
## Summary

- 118 registers at logical warp 8 and 168 at logical warp 4 in the latency kernel at N 128 (32 items per lane), throughput occupancy 2 and 1 blocks per SM. Recompute the twiddled originals after the network or run the recovery slot by slot without the ballot arrays.
- **Scope**: Sub-warp N 128, the device router kernel has to be measured.
- **Comparison method**: The study harness plus the device benchmark of the router shape.
- **Status**: Open.

## Problem coverage

Batched routing at N 128.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 19.3, 19.7.

Owner: TBD
```

### TK-S04.5

- Title: `[CUB] N10: rows-per-group form for throughput`
- Labels: `cub`
- Parent: TK-P04
- Dependencies: TK-P04, TK-S03.4

```markdown
## Summary

- Two independent rows per call as N5 does for bf16x2, the recovery is per row. The natural throughput follow-up of the fp32 route.
- **Scope**: Warp scope, the batched router shape.
- **Comparison method**: The study harness, one-wave throughput.
- **Status**: Open.

## Problem coverage

Batched fp32 routing.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 19.5, 20.

Owner: TBD
```

### TK-S04.6

- Title: `[CUB] N10: sm_100f variant with redux.sync.min.f32`
- Labels: `cub`
- Parent: TK-P04
- Dependencies: TK-P04

```markdown
## Summary

- The family build has a float redux, which allows the native float min and max network without the twiddle. The mandated sm_100 build cannot emit it and the family macro exists only in the device pass, so the host needs its own define (T2).
- **Scope**: An sm_100f variant of the N10 adapter.
- **Comparison method**: The study harness, sm_100 against sm_100f in one lock section.
- **Status**: Open.

## Problem coverage

Deployments that build for the family target.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.10, 19.5, T2.

Owner: TBD
```

---

### TK-P05

- Title: `[CUB] Equi-width bucket select over an exact value bracket at warp and block scope, with the packed finish (strategy 9)`
- Labels: `cub`, `future-productization`
- Parent: none (new strategy 9)
- Dependencies: TK-P01, TK-P09

```markdown
## Summary

- An exact lower bound on the k-th largest key (the minimum over lanes or threads of their m-th largest key) brackets the values, one pass of equal-width bins over the bracket leaves a pivot bin of a few items, and a packed redux finish resolves it. At warp scope the histogram is one shared atomic per item into a per-warp 32-word table, at block scope a 256-bin fixed-point image replaces the first radix pass of the AIR select and N9 adds the packed finish on the pivot bin. Measured as N6 (block), N6w (warp) and N9 of the study, with W3 (warp radix select) as the fallback and N1 (bit-serial) as the one-bit row.
- **Scope**: Warp: N 256 to 2048 (N 4096 register bound), k 64 to 1024, bf16 and fp32, 256 to 640 bytes of shared memory per warp, no barriers. Block: 256 threads, N 1024 to 32768 (128 items per thread register bound), k 16 to 256.
- **Comparison method**: The study harness (TK-P01), the warp form against `W2_packed_v3` and the ballot-histogram form in one lock section, the block form against `B1_idx` and `B1_n6b`. Host simulation of the pivot-bin statistics and a host model of the device arithmetic for the bracket.
- **Status**: Warp: seven cards verified on B200 and GB200 to the cycle (B200 umbriel-b200-028, GB200 gb200-nvl4-ts2-105, CUDA 13.4.92, 2026-10-25). `N6w_atom32` against `W2_packed_v3`: bf16 N 512 k 64 1879 against 2704, k 256 2564 against 14404, N 1024 k 128 4010 against 11872, N 2048 k 256 15982 against 40044, k 1024 17741 against 152618, fp32 N 2048 k 1024 15880 against 151563, throughput 506 against 170 at bf16 N 512 k 64. W2 keeps k 16, N 2048 k 64 and N 4096. Block: N6 wins where it saves a pass (fp32 N 4096 k 16 to 256 by 11 to 24 percent, N 16384 k 16 by 1.48x and 1.58x) and loses throughput by occupancy, N9's finish gains 9 to 17 percent at k 64 to 256 and 13 to 41 percent under prefer_smaller_index (cards `B1_n6_2`, `B1_n6b_1` and `_2`, `N9_w2fin2`, verified). Final cells (phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25): N6w_atom32 wins 20 of 89 warp latency cells under not_guaranteed (median 454 percent ahead of the reference, 21 percent ahead of the runner-up) and 18 of 70 under prefer_smaller_index at identical cycles: k 48 and 64 at N 256 (bf16 1270 and 1296, W1x_vimnmx_eager 2 to 3 percent behind, a close call, runner-up within 15 percent), N 512 from k 48 (bf16) or k 32 (fp32) up to k 256 (bf16 k 64 1911 with W5_v4 30 percent behind, k 256 2626 with W3_d4c 54 percent behind, fp32 k 64 1899 with N10_auto 23 percent behind), fp32 N 1024 k 48 and 64 (3454, a close call, and 3709, W5_v4 6 and 27 percent behind), fp32 N 2048 k 256 (15540, W3_d4c 19 percent behind), N 4096 k 256 and 1024 under both key types (bf16 63856 and 59810, fp32 59723 and 66140, W3_d4c 13 to 29 percent behind, the fp32 k 1024 cell a close call), and the named EAGLE-2 rerank (N 1024 k 64 3709) and sampling merge (N 512 k 64 1899) cells. It does not take bf16 N 1024 k 48 and 64 or the N 2048 k 64 cells (W5_v4), bf16 N 2048 k 256 (W4_wmerge) or k 1024 at N 2048 (W3_d4c, a close call). Throughput: 15 of 87 cells, fp32 from k 24 at N 256 and 512 and N 128 k 64, bf16 N 512 k 48 and 64 (a close call), N 2048 k 256 under both key types. On the tie_heavy input as generated N6w_atom32 keeps 17 of its 21 close calls and loses fp32 N 4096 k 1024 to W3_d4c and the bf16 N 512 k 64 throughput cell to N5_rows_f, so for those four the choice stays with the implementer unless the shuffled variant confirms the flip.

## Problem coverage

Warp k 64 to 1024 at N 256 to 2048 under both key types (the k 128 and above band has no other owner), the block fp32 N 4096 band and N 16384 k 16 for both types, and under prefer_smaller_index the block k 64 to 256 cells through N9. W3 is the fallback (digit passes when the bracket pass does not finish) with no cell of its own, N1 has no cell.

### Segment / input size
- **Supported range**: warp N 256 to 4096 (N 4096 register bound), block N 1024 to 32768 (128 items per thread register bound).
- **Expected sweet spot**: warp N 256 to 2048, block fp32 N 4096 and N 16384 at k 16.

### k
- **Supported range**: warp k 64 to 1024 (exact per-lane bound for k at most 256, the full range above), block k 16 to 256.
- **Expected sweet spot**: warp k 64 to 1024 (the k 128 and above band has no other owner), block k 64 to 256 through the N9 finish.

### Types
- **Supported**: bf16 and fp32 (widened to fp32 for the image), int32 indices. NaN unsupported at warp scope.
- **Expected sweet spot**: both types cost the same under the bracket image.

## Output guarantees

### Determinism (set membership)
| `determinism` | Supported | Notes |
|---|---|---|
| `not_guaranteed` | yes | The reference requirement of every measurement. |
| `run_to_run` | yes | Exact selection, the shared atomics only count. |
| `gpu_to_gpu` | yes | The emission ranks by index, the finish orders by (key, inverted index). |

### Tie-break (only meaningful with `gpu_to_gpu`)
| `tie_break` | Supported | Notes |
|---|---|---|
| `unspecified` | yes | Index order on every exit path. |
| `prefer_smaller_index` | yes | Warp: zero (identical cycles). Block: through the N9 finish, 13 to 41 percent faster than the warp-redundant rank. |
| `prefer_larger_index` | planned | Not built, the mirror form is listed in the report. |

### Output ordering (result sequence)
| `output_ordering` | Supported | Notes |
|---|---|---|
| `unsorted` | yes | Trivially. |
| `sorted` | no | Post-sort of k. |
| `stable_sorted` | no | Not available. |

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 22, 13, 21.4, 16, reports `approaches/N6w/REPORT.md`, `approaches/B1/REPORT.md`, `approaches/B4/REPORT.md` section 4.3, techniques T25, T30, T59, T60, T63, host models `approaches/N6w/model_device.py`, `approaches/N6w/sim_warp_bins.md`.

Owner: TBD
```

### TK-S05.1

- Title: `[CUB] N6 at warp scope: the fixed costs below 64 items per lane`
- Labels: `cub`
- Parent: TK-P05
- Dependencies: TK-P05

```markdown
## Summary

- Below 64 items per lane the call is dominated by dependent fixed costs, not per-item issue: the 5-step shuffle suffix scan (about 190 cycles), the pivot ballot, popc and shuffle (about 80), the serial min and max chain over the pivot bin's images (about 10 per item), two syncwarps and about n over 2 redux rounds. A ballot scan over the count bits (about 70), an approximate reciprocal for the scale, tree-reduced min and max accumulators and a redux-based pivot search are the levers.
- **Scope**: `N6w_atom32` at N 256 to 1024.
- **Comparison method**: The study harness against `N6w_atom32` in one lock section.
- **Status**: Open. Trimming 13 instructions per item changed nothing (card `N6_w_atom32f`, verified), the attribution stands.

## Problem coverage

Every warp N6 cell below N 2048.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 22.3, 22.5.

Owner: TBD
```

### TK-S05.2

- Title: `[CUB] N6 at warp scope: the register footprint at 64 and 128 items per lane`
- Labels: `cub`
- Parent: TK-P05
- Dependencies: TK-P05

```markdown
## Summary

- 255 registers at 64 items per lane with a few bytes of spill, and at 128 a 4 to 5 KB stack frame with 600 to 800 bytes of spill, 60 to 80 thousand cycles at N 4096 against W2's 6.5 to 48 thousand. A two-half item loop over the tile, or a packed 16-bit bf16 tile with the image recomputed, decides whether N6 can own N 2048 to 4096 at all.
- **Scope**: N 2048 and 4096 at warp scope, both key types.
- **Comparison method**: The study harness against W2_packed_v3 and the current N6w_atom32.
- **Status**: Open. The footprint is verified (KB 22.3).

## Problem coverage

Warp N 2048 k 64 (W2's today) and all of N 4096.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 22.3, 22.5, T34, T36.

Owner: TBD
```

### TK-S05.3

- Title: `[CUB] N6 at warp scope: compaction before the second bracket pass`
- Labels: `cub`
- Parent: TK-P05
- Dependencies: TK-P05

```markdown
## Summary

- At k 1024 the full-range bracket leaves 175 to 300 candidates and the second bracket pass sweeps the whole tile with six ballots per item for at most N over 8 of them. The W3 compaction code applies as is once the bracket pass takes a compacted (image, index) array.
- **Scope**: k 1024 at N 2048 and the k 256 cells where W3 is within 1.10x.
- **Comparison method**: The study harness against `N6w_atom32` and `W3_d4c`.
- **Status**: Open.

## Problem coverage

The k 1024 row, where W3 is 1.54x ahead today.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 22.4, 22.5.

Owner: TBD
```

### TK-S05.4

- Title: `[CUB] N6 at warp scope: the trimmed finish at fp32 N 4096 and the unmeasured variants`
- Labels: `cub`
- Parent: TK-P05
- Dependencies: TK-P05

```markdown
## Summary

- In the card's own data the trimmed finish (atom32f) is 2.8x to 3.1x faster than atom32 at fp32 N 4096 (19839 against 62282 cycles at k 128) and 2.1x in throughput, which would put N6 under W2_packed_v3 at fp32 N 4096 k 128 (48554). The batch against W2 was built and never measured. atom128 (four times smaller pivot bins), W3 with 5-bit digits and W3 without compaction are built and unmeasured as well.
- **Scope**: fp32 N 4096 and the variants named.
- **Comparison method**: The study harness, each against its control in one lock section.
- **Status**: Open lead found by verify-N6w (B200 and GB200, CUDA 13.4.92, 2026-10-25).

## Problem coverage

An N 4096 cell for N6 under W2 for fp32.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 22.4, 22.7.

Owner: TBD
```

### TK-S05.5

- Title: `[CUB] N9: the packed redux finish as the block N6 finish from k 64`
- Labels: `cub`
- Parent: TK-P05
- Dependencies: TK-P05, TK-P09

```markdown
## Summary

- When the pivot bin after the first pass holds at most 32 items and the pass did not finish, the items above the bin are scattered by ticket and warp 0 resolves the bin by redux rounds on (key, inverted index) words instead of a second pass over the tile. Nothing at k 16 (the block N6 already exits after one pass in 97 percent of those rows), 9 to 17 percent at k 64 to 256 under not_guaranteed and 13 to 41 percent under prefer_smaller_index, where it replaces the warp-redundant rank.
- **Scope**: The block N6 at N 1024 and 4096, k 64 and above, both key types.
- **Comparison method**: The study harness against `B1_n6b` in one lock section, the ticket spelled on a class-indexed counter (T57).
- **Status**: Verified (cards `N9_w2fin2`, `N9_w2fin`, B200 umbriel-b200-047 and 018, CUDA 13.4.92, 2026-10-25). By the chain through the N6 card it puts N6 ahead of B1_idx at bf16 N 1024 and 4096 k 64 (1851 against 1878, 3359 against 3838). Final cells (phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25): N9_w2fin2 wins 13 of 77 block latency cells under not_guaranteed and 6 of 75 under prefer_smaller_index, at N 1024 (bf16 k 32 to 64, fp32 k 24 to 256), N 4096 k 64 under both key types and fp32 k 256, and fp32 N 8192 k 32. Margins over B1_idx 1 to 43 percent (bf16 N 1024 k 32 1805 cycles against 1825, fp32 N 4096 k 64 3363 against 4660, fp32 N 8192 k 32 4234 against 6059), over B1_n6b 1 to 26 percent, over B1_det2 1 to 43 percent, the deterministic cells 16 to 59 percent ahead of B1_det and 2 to 31 percent ahead of B1_det2. Every cell but fp32 N 1024 k 48 (B1_idx 18 percent behind) is a close call (runner-up within 15 percent). No throughput cell. On the tie_heavy input as generated B1_n6b takes the N 1024 and 4096 close calls by its exact-count exit (T68) and B1_det2 the deterministic ones, N9_w2fin2 is faster on random input by 1 to 9 percent, and the choice stays with the implementer unless the shuffled variant confirms the flip.

## Problem coverage

Block k 64 to 256 at N 1024 to 4096, and the deterministic block cells there.

## Output guarantees

prefer_smaller_index exact through the word order of the finish and the index-ordered scatter above the bin.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 21.4, 13.7, T57, code `approaches/B4/block_topk_n6_w2.cuh`.

Owner: TBD
```

### TK-S05.6

- Title: `[CUB] Block N6 at 128 items per thread: the register problem`
- Labels: `cub`
- Parent: TK-P05
- Dependencies: TK-P05

```markdown
## Summary

- At N 32768 and 256 threads the block N6 is register bound in two of three builds of the same algorithm (1.5 KB, 8.6 KB and 7.3 KB of spill), and the build that did not spill badly ran 1.4 to 1.8x faster than the radix path at k 16 and cannot be rebuilt. 512 threads halve the spill but do not remove it. A shared-memory-resident rest or a recomputed bf16 image is the lever.
- **Scope**: N 32768, both key types, 256 and 512 threads.
- **Comparison method**: The study harness against `B1_idx` at 512 threads.
- **Status**: Open. The register problem is verified (cards `B1_n6_2`, `B1_n6b_2`).

## Problem coverage

Block N 32768 k 16, where B1_idx at 512 threads and the two-level B2 list own the cell today.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 13.7, 17.4, T24, T36.

Owner: TBD
```

### TK-S05.7

- Title: `[CUB] Bracket scale guard audit across the N6, B4 and N9 brackets (T60)`
- Labels: `cub`
- Parent: TK-P05
- Dependencies: TK-P05, TK-P07

```markdown
## Summary

- A bracket whose width overflows gives a scale of plus zero, and inf times zero puts infinite keys into the wrong bin through a NaN image, so the bin map stops being monotone. Guard the scale (not the width) with a tiny positive fallback. The warp N6 had the bug at its first commit, the block N6, B4 and N9 brackets are named as in scope and their batteries passed inf_denorm at the measured cells.
- **Scope**: Every equi-width or fixed-point image over a value range in the study's primitives.
- **Comparison method**: The host model of the device arithmetic on the oracle battery, then the GPU battery.
- **Status**: Fixed in the warp N6 from v2 on (verified), audit of the others open.

## Problem coverage

Correctness with infinities and the largest finite values.

## Output guarantees

Monotone image for every input the contract admits.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.10, 13.7, 22.7, T60, T63.

Owner: TBD
```

### TK-S05.8

- Title: `[CUB] Host models of the device arithmetic as test fixtures for bracket code`
- Labels: `cub`
- Parent: TK-P05
- Dependencies: TK-P05

```markdown
## Summary

- Two correctness bugs of bracket code (T60, the degenerate scale, and T63, the threshold edge stepped by ulps of the edge instead of the magnitude of the base) were found by host models that reproduce the device arithmetic on the oracle battery before any GPU time. Such a model should accompany any bracket or image code that ships.
- **Scope**: `approaches/N6w/model_device.py`, `approaches/N3/model.py`, their patterns.
- **Comparison method**: Not applicable, a test fixture.
- **Status**: Both models exist in the study tree.

## Problem coverage

Correctness.

## Output guarantees

Not applicable.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.10, T60, T63.

Owner: TBD
```

---

### TK-P06

- Title: `[CUB] Block top-k by packed shared atomics extraction, the two-level list and the block size (strategy 4)`
- Labels: `cub`, `future-productization`
- Parent: none (strategy 4 of #9747)
- Dependencies: TK-P01

```markdown
## Summary

- k rounds of a fire-and-forget shared atomicMax on the packed key-plus-inverted-index word into a per-round slot, with a per-thread sorted list, a barrier and a read back per round. ptxas aggregates the warp-uniform-address atomic into one CREDUX plus one elected atomic per warp, so the round floor is the barrier plus a flat post (about 48 plus 4 per warp plus the list shift). A two-level per-thread list (sorted head of 4, unsorted rest) carries it to N 32768 at k 4 and 16 for bf16. Measured as B2 and N8 of the study.
- **Scope**: Block of 64 to 512 threads, N 1024 to 32768, k 4 to 64 (the crossover with the radix select), bf16 through the packed word and fp32 through the truncated word with an exact tie finish, sorted output by construction.
- **Comparison method**: The study harness (TK-P01) against `B1_reference` and `B1_idx` in one lock section, a two-node repeat with clock traces for the node-dependent cells.
- **Status**: Cards verified (B200 umbriel-b200-047, 068, 091, 018, CUDA 13.4.92, 2026-10-25), the two k 64 cells of the 128-thread card node dependent (T50, T58). B2 beats B1 at k 4 to 16 at every N from 1024 to 16384 (1.36x to 5.1x), the two-level list 4490 and 4895 against B1_idx's 13479 and 13505 at bf16 N 32768 k 4 and 16 (fp32 holds k 4 and is even at k 16). Block size: 128 threads for latency at k at most 32 and N 1024 to 4096 (1182 against 1380 at bf16 N 1024 k 16), 64 threads for one-wave throughput at N at most 2048 (1.36x to 1.66x), 256 from N 8192, 512 only for k 4 at N 8192. Final cells (phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25): the 128-thread B2 wins 12 of 77 block latency cells under not_guaranteed and 12 of 75 under prefer_smaller_index at identical cycles, k 4 to 24 at N 1024, k 16 at N 4096 and 8192 and bf16 k 32 at N 8192 (bf16 N 1024 k 4 340 cycles against the 128-thread reference 2067, N 8192 k 16 2120 against B2_twolevel 2655 and the reference 8868). The two-level list wins 7 and 7 cells, k 4 at N 4096 and 16384 under both key types, bf16 N 16384 k 16 and bf16 N 32768 k 4 and 16 (N 16384 k 4 934 against the reference 10673, N 32768 k 4 1865 and k 16 4437 against B4_v4 6423, the reference 19604 and 19583), B2_packed at 256 threads fp32 N 16384 k 16 (4173, B4_v4 5 percent behind). Throughput: the 128-thread form 9 and 11 cells at N 1024 (bf16 k 4 1586 Gelem/s against the reference 490) and the two-level list 9 and 9 cells (bf16 N 16384 k 4 1921, N 32768 k 4 1372). Most N 1024 and N 4096 cells, where B2_packed at 256 threads or B1_det2 follow within 1 to 15 percent, are close calls (runner-up within 15 percent). On the tie_heavy input as generated the 128-thread form keeps 21 of its 27 close calls (the 256-thread form takes fp32 N 4096 and 8192 k 16, N9_w2fin2 and B1_det2 three throughput cells at N 1024) and the two-level list keeps all ten, and where the runner-up is the same kernel at 256 threads the block size is the implementer's choice by the surrounding kernel.

## Problem coverage

Block k 4 to 16 at every N to 32768 for bf16 (fp32 to k 4 at N 32768), the one-wave throughput cells at small N.

### Segment / input size
- **Supported range**: N 1024 to 32768 at 64 to 512 threads, N 65536 out of reach until the rest lives in shared memory.
- **Expected sweet spot**: N 1024 to 4096 at 128 threads, N 8192 to 32768 at 256 threads with the two-level list.

### k
- **Supported range**: k 1 to N over 2 at compile time, measured k 4 to 64.
- **Expected sweet spot**: k 4 to 16 (bf16 to N 32768, fp32 to N 16384 and k 4 at N 32768), the crossover with the radix select is near k 22 to 50 depending on N.

### Types
- **Supported**: bf16 (packed word), fp32 (truncated word plus exact tie finish), int32 indices.
- **Expected sweet spot**: bf16, fp32 pays 270 to 670 fixed cycles more.

## Output guarantees

### Determinism (set membership)
| `determinism` | Supported | Notes |
|---|---|---|
| `not_guaranteed` | yes | The reference requirement of every measurement. |
| `run_to_run` | yes | The max of a round is unique in the word, atomics do not change the result. |
| `gpu_to_gpu` | yes | The word order. |

### Tie-break (only meaningful with `gpu_to_gpu`)
| `tie_break` | Supported | Notes |
|---|---|---|
| `unspecified` | yes | Resolved by the word order. |
| `prefer_smaller_index` | yes | Zero (within 1 percent at every block size, the same binary). |
| `prefer_larger_index` | planned | Not built, the mirror form is listed in the report. |

### Output ordering (result sequence)
| `output_ordering` | Supported | Notes |
|---|---|---|
| `unsorted` | yes | Trivially. |
| `sorted` | yes | By construction, the k slots hold the words in descending order. |
| `stable_sorted` | yes | Through the word order. |

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 14, 17, 16, 2.12, reports `approaches/B2/REPORT_B2b.md`, `approaches/N8/REPORT.md`, techniques T10, T11, T26, T27, T38, T43 to T46, T50.

Owner: TBD
```

### TK-S06.1

- Title: `[CUB] B2 two-level list with the rest in shared memory for 128 items per thread and above`
- Labels: `cub`
- Parent: TK-P06
- Dependencies: TK-P06

```markdown
## Summary

- At 128 items per thread the two-level list holds 255 registers and spills, at 256 it spills 1.8 to 9 KB, so 128 threads cannot be used at N 16384 and above and N 65536 is out of reach. The unsorted rest in shared memory is the lever the block size cannot replace.
- **Scope**: N 16384 to 65536 at 128 and 256 threads.
- **Comparison method**: The study harness against the current two-level list and B1_idx at 512 threads.
- **Status**: Open. The spills are verified (card `B2_twolevel_2`).

## Problem coverage

Block k 4 to 16 at N 32768 and 65536.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 14.5, 17.4.

Owner: TBD
```

### TK-S06.2

- Title: `[CUB] B2: the warp-uniform refill variant and the fp32 k 16 cells`
- Labels: `cub`
- Parent: TK-P06
- Dependencies: TK-P06

```markdown
## Summary

- `B2_twolevel_any` (one vote per round, the refill under a uniform branch) exists as batches only and was within a few percent at 256 threads. At fp32 k 16 the two-level list is even with the full list, which is the simpler code.
- **Scope**: N 8192 to 32768.
- **Comparison method**: The study harness against `B2_twolevel` in one lock section.
- **Status**: Open, a card is missing.

## Problem coverage

Block fp32 k 16 at large N.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 14.7.

Owner: TBD
```

### TK-S06.3

- Title: `[CUB] B2: the unroll policy above k 64 and the node class of large unrolled kernels`
- Labels: `cub`
- Parent: TK-P06
- Dependencies: TK-P06

```markdown
## Summary

- Fully unrolled 128-thread kernels above about 64 KB of SASS ran 1.29x to 2.04x slower on one B200 than on three others with every smaller kernel identical to the cycle (T50, T58). B2 above k 64 already rolls its rounds, the policy per (N, k) and the loss of the list pruning in a rolled loop are the open parts.
- **Scope**: k 64 and above at 128 and 256 threads.
- **Comparison method**: The study harness on two nodes, SASS size per kernel in the batch JSON.
- **Status**: Open. The node class is characterized (KB 2.10).

## Problem coverage

Robustness of the latency numbers across GPUs.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.10, 14.5, T38, T41, T48, T50, T58.

Owner: TBD
```

### TK-S06.4

- Title: `[CUB] B2 block size per regime: 128 threads for latency, 64 for one-wave throughput`
- Labels: `cub`
- Parent: TK-P06
- Dependencies: TK-P06

```markdown
## Summary

- The round fits about 48 plus 4 per warp plus the list depth per thread, so 4 warps are the latency shape at N 1024 to 4096 for k at most 32 and 2 warps gain nothing. At equal threads per SM, 32 two-warp blocks interleave their barrier waits and give 1.36x to 1.66x in one-wave throughput over 8 eight-warp blocks (B1_idx 1.2x to 1.6x), gone where the register tier drops the block count. A device agent that serves one row per block should take the throughput shape, the harness reloads the tile per call, so a `segmented_topk` run with the block size as a parameter confirms it for a kernel that loads each tile once.
- **Scope**: N 1024 to 4096, both key types, the device agent.
- **Comparison method**: The study harness (verified cards `N8_b2packed_1` to `_3`, B200, CUDA 13.4.92, 2026-10-25) and the segmented top-k device benchmark.
- **Status**: Measured and verified, the device run open.

## Problem coverage

Block small-k cells and the batched throughput regime.

## Output guarantees

Unchanged at every block size.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 17, T43, T44, T46.

Owner: TBD
```

### TK-S06.5

- Title: `[CUB] B2: a combined adapter choosing threads and list per (N, k)`
- Labels: `cub`
- Parent: TK-P06
- Dependencies: TK-P06, TK-S06.4

```markdown
## Summary

- The N8 recommendation tables give the block size and the list form per cell (full list at 128 threads to N 4096, two-level at 256 from N 8192, 512 for k 4 at N 8192). One adapter should pick them at compile time from N and k.
- **Scope**: The policy layer of the B2 primitive.
- **Comparison method**: The study harness against the hand-picked forms.
- **Status**: Open.

## Problem coverage

Every B2 cell.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 17.4, 14.5.

Owner: TBD
```

### TK-S06.6

- Title: `[CUB] B2 fp32 truncated word at N 32768: the 17-bit key prefix`
- Labels: `cub`
- Parent: TK-P06
- Dependencies: TK-P06

```markdown
## Summary

- The truncated word keeps 32 minus log2 N key bits, 17 at N 32768, so the exact tie finish runs on 20 to 50 percent of random inputs there (0.1 to 4 percent below N 16384). A wider key prefix with a block-local index, or the two-level list's rest holding the full key, would cut it.
- **Scope**: fp32 at N 32768.
- **Comparison method**: Host simulation of the finish rate plus the study harness.
- **Status**: Open. The rate is from the N4 host simulation (2000 seeds) and the B2 cards.

## Problem coverage

Block fp32 k 4 to 16 at N 32768.

## Output guarantees

Unchanged (the finish is exact).

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.12, 14, idea card `ideas/queue/N4_truncated_key_packing.md`.

Owner: TBD
```

---

### TK-P07

- Title: `[CUB] Block top-k by an exact-bound prefilter, compaction and a small finish (strategy 10)`
- Labels: `cub`, `future-productization`
- Parent: none (new strategy 10)
- Dependencies: TK-P09, TK-P05

```markdown
## Summary

- The minimum over the threads of the per-thread maximum is an exact lower bound on the k-th largest key for k up to the thread count. On random input about 1500 items of the tile sit at or above it at every N from 4096 on, so the compare sweep compacts them into shared memory and the selection runs over a few candidates per thread: one pass over a fixed-point image (the N6 digit), a warp finish on the pivot bin, an exact fallback over the register tile when the buffer overflows or the count falls short. Measured as B4 of the study.
- **Scope**: Block of 256 threads (128 and 512 accepted, unmeasured), striped tile of N 1024 to 32768, k 16 to 2048 at run time, bf16 and fp32, temp storage 22.9 KB (bf16) and 39.3 KB (fp32) at N 32768 for k up to 256. The striped arrangement is a requirement (a sorted blocked tile falls back).
- **Comparison method**: The study harness (TK-P01) against `B3_tuned` and `B1_idx` in one lock section, clock64 phase stamps for the attribution (T56), a host simulation of the candidate counts.
- **Status**: Eight cards verified on two nodes (B200 umbriel-b200-047 and 018, CUDA 13.4.92, 2026-10-25), three under the harness-version policy with the current ratios recorded. `B4_v4` against `B3_tuned` under harness 0.8: fp32 N 32768 k 256 7026 against 18278 (2.60x), bf16 8145 against 18369 (2.26x), N 16384 k 64 2.00x and 2.09x, N 8192 k 64 1.38x and 1.32x, a tie at N 4096, a loss at N 1024 and at k 2048 (0.70x to 0.81x), throughput 832 and 941 against 400 and 345 Gelem/s at N 32768. The base build against `B1_idx` in one session 2.34x fp32 and 1.94x bf16 at N 32768 k 256, v4 about 3.1x and 2.15x by the chain through that card. The cost is about 3300 cycles of finish plus 37 per item of the tile. Final cells (phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25, confirmed on umbriel-b200-068 with no winner flips and ratios within 2.3 percent): B4_v4 wins 32 of 77 block latency cells under not_guaranteed (median 160 percent ahead of the reference and 52 percent ahead of the runner-up) and 32 of 75 under prefer_smaller_index, every N 8192 and above from k 64 (N 8192) or k 32 (N 16384 and 32768) to k 1024 under both key types and fp32 N 32768 k 16 (bf16 N 8192 k 64 4600 cycles against N9_w2fin2 4836, N 16384 k 256 5829 against B1_idx at 512 threads 8932 and B3_tuned 11448, N 32768 k 256 7183 against 14361 and 17831, fp32 N 32768 k 256 6058 against 15380 and 15761, fp32 N 16384 k 64 4763 against B3_tuned 8294). It does not win k 4096 (B3_tuned), bf16 k 16 at N 16384 and 32768 (B2_twolevel) or the N 4096 cells (B3_tuned, N9_w2fin2, B1_idx at 512 threads). Throughput: 24 and 27 cells at the same N. The five not_guaranteed latency cells with a runner-up within 15 percent (bf16 N 8192 k 64 and 128, N 16384 k 1024, N 32768 k 32, fp32 N 8192 k 64) hold on random input and go to B3_tuned on the tie_heavy input as generated by 10 to 170 percent depending on N, with the fallback as the cause, see the caveat.
- **Caveat, tie_heavy input**: the phase 3 wins above are on random input. On the harness's tie_heavy pattern ((j mod 64) / 64, unshuffled, measured on B200 umbriel-b200-068, harness 0.10, 2026-10-26) the period 64 divides the 256-thread stripe, so four threads hold only zeros, the minimum of the per-thread maxima is the tile minimum, every item is a candidate and B4_v4 takes its tile fallback at every N above the 4096-word buffer. Against B3_tuned on the same cells and input: fp32 N 8192 k 32 11404 against 10401 cycles (1.10x behind), fp32 N 16384 k 32 19155 against 15520 (1.23x behind and 4.1x its own random time), bf16 N 32768 k 32 55587 against 20404 (2.72x behind, 2020 bytes of spill), one tile pass more under prefer_smaller_index. At N 1024 to 4096 the tile fits the buffer and B4_v4 stays within 4 percent of its random time. The trigger is the weak-thread bound on a periodic input aligned with the stripe, not the tie density (T67), TK-S07.2 bounds the loss (a fallback at B3_tuned's tile cost), and a lazy second bound, the k-th largest thread maximum computed only when the count exceeds the buffer, is a candidate sub-item. A shuffled copy of the pattern, pending with the evaluation engineer, separates layout from ties.

## Problem coverage

Block N 8192 to 32768 for k 64 to 1024 under both key types, 1.8x to 2.3x ahead of the 512-thread radix select by card numbers. Hands N 4096 and below to B2 and B1, k 2048 to B3, k 16 at large N to the two-level B2 list for bf16.

### Segment / input size
- **Supported range**: N 1024 to 32768 at 256 threads (128 and 512 accepted, unmeasured), striped tile required.
- **Expected sweet spot**: N 8192 to 32768, where the bound admits about 1500 of N candidates.

### k
- **Supported range**: k 1 to N at run time, buffer sizes per compile-time hint (4096 words to k 256, 8192 to k 1024, 16384 above).
- **Expected sweet spot**: k 64 to 1024, a loss at k 2048 where the pivot bin needs a second pass.

### Types
- **Supported**: bf16 and fp32, int32 indices.
- **Expected sweet spot**: both, fp32 slightly ahead by ratio.

## Output guarantees

### Determinism (set membership)
| `determinism` | Supported | Notes |
|---|---|---|
| `not_guaranteed` | yes | The reference requirement of every measurement. |
| `run_to_run` | yes | Exact selection with a data-dependent fallback that gives the same set. |
| `gpu_to_gpu` | yes | The pivot set resolved by the warp finish, the tie class rule or index passes. |

### Tie-break (only meaningful with `gpu_to_gpu`)
| `tie_break` | Supported | Notes |
|---|---|---|
| `unspecified` | yes | Bin order then ticket order. |
| `prefer_smaller_index` | yes | Plus 4 to 17 percent (register pressure from the fallback's index passes), TK-S07.8 targets it. |
| `prefer_larger_index` | planned | One flipped comparison in the finish and the index digit, not built. |

### Output ordering (result sequence)
| `output_ordering` | Supported | Notes |
|---|---|---|
| `unsorted` | yes | Trivially. |
| `sorted` | no | Post-sort of k. |
| `stable_sorted` | no | Not available. |

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 21, 16, 2.12, report `approaches/B4/REPORT.md`, `sim_bound.md`, `profiles/`, techniques T56, T57, T32, T25.

Owner: TBD
```

### TK-S07.1

- Title: `[CUB] B4: packed bf16 sweeps`
- Labels: `cub`
- Parent: TK-P07
- Dependencies: TK-P07

```markdown
## Summary

- The three register sweeps (count, compaction store, image) cost 4700 to 6200 cycles per call at 128 items per thread, about 10 cycles per item slot for 7 to 10 instructions at two warps per scheduler, and fewer instructions per item is the only lever left there. HMNMX2 maxima and HSET2 compares on the packed_striped arrangement halve the count and max sweeps and hold 128 keys in 64 registers, predicted about 1000 cycles at N 32768 and two blocks per SM at 64 items per thread.
- **Scope**: bf16 at N 8192 to 32768.
- **Comparison method**: The study harness against `B4_v4`.
- **Status**: Open. The sweep cost is attributed by the profile (verified cards).

## Problem coverage

Every bf16 B4 cell.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 21.3, 21.5, T18.

Owner: TBD
```

### TK-S07.2

- Title: `[CUB] B4: the fallback out of the fast path's register allocation`
- Labels: `cub`
- Parent: TK-P07
- Dependencies: TK-P07

```markdown
## Summary

- 255 registers from 64 items per thread and 1.4 to 2.5 KB of spill at 128 come from the fallback's key-pass and index-pass bodies living in the same function as the fast path, one block per SM against B1_idx's two. A noinline tile fallback, or one that re-reads the keys from the caller's memory, takes them out of the allocation.
- **Scope**: N 16384 and 32768, both requirements.
- **Comparison method**: The study harness, throughput regime in particular.
- **Status**: Open.

## Problem coverage

The throughput cells at N 16384 and the deterministic price.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 21.3, 21.5, 21.6.

Owner: TBD
```

### TK-S07.3

- Title: `[CUB] B4: a custom block scan for the count and pass scans`
- Labels: `cub`
- Parent: TK-P07
- Dependencies: TK-P07

```markdown
## Summary

- Four block scans at 350 to 430 cycles each (cub::BlockScan) sit in the structure cost after v2. A warp shuffle scan with warp totals in shared memory and one barrier is about 230.
- **Scope**: The count scan and the pass scan of B4, also B3's and N3's.
- **Comparison method**: The phase stamps plus the study harness.
- **Status**: Open.

## Problem coverage

Every B4 cell, about 500 cycles per call.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 21.3, 21.5, T29.

Owner: TBD
```

### TK-S07.4

- Title: `[CUB] B4: bounds for k above 256 and the 64-word finish for k 1024 to 2048`
- Labels: `cub`
- Parent: TK-P07
- Dependencies: TK-P07

```markdown
## Summary

- Above the thread count the bound is the minimum over group maxima (half-group form with a run-time count check, 3300 candidates at N 32768), the pivot bin holds 28 to 62 items at k 1024 to 2048 and a second pass runs in half of the rows, which is where B4 loses to B3 at k 2048. The exact per-thread m-th largest at small items per thread and a W2 finish on a 64-word buffer (list depth 2) are the candidates.
- **Scope**: k 1024 and 2048 at N 16384 and 32768.
- **Comparison method**: The host simulation of the bound plus the study harness against `B3_tuned`.
- **Status**: Open. The k 2048 loss is verified (0.70x to 0.81x).

## Problem coverage

The k 2048 row (B3's today) and the k 1024 margin.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 21.4, 21.5, `approaches/B4/sim_bound.md`.

Owner: TBD
```

### TK-S07.5

- Title: `[CUB] B4: a 2048-word candidate buffer with its fallback rate`
- Labels: `cub`
- Parent: TK-P07
- Dependencies: TK-P07

```markdown
## Summary

- A buffer of 4096 words never overflowed on random, quantized or outlier rows, 2048 overflows 0.3 to 10 percent of the rows. Eight candidate slots instead of sixteen would cut the pass and the emission, against the fallback rate.
- **Scope**: k up to 256 at every N.
- **Comparison method**: Host simulation of the overflow rate plus the study harness on random and battery patterns.
- **Status**: Open.

## Problem coverage

Every B4 cell with k at most 256.

## Output guarantees

Unchanged (the fallback is exact).

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 21.4, 21.5.

Owner: TBD
```

### TK-S07.6

- Title: `[CUB] B4 at 128 and 512 threads`
- Labels: `cub`
- Parent: TK-P07
- Dependencies: TK-P07, TK-S09.4

```markdown
## Summary

- The adapter accepts 128 and 512 threads, unmeasured. At 128 threads the bound is the 128th order statistic (twice the candidates per thread) and the barriers are cheaper, at 512 the radix incumbent gained 1.24x to 1.45x.
- **Scope**: N 8192 to 32768.
- **Comparison method**: The study harness, the same protocol as the N8 block-size study.
- **Status**: Open.

## Problem coverage

The B4 band against B1_idx at 512 threads.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 21.5, 17.

Owner: TBD
```

### TK-S07.7

- Title: `[CUB] N6 and B4 as one block primitive`
- Labels: `cub`
- Parent: TK-P07
- Dependencies: TK-P07, TK-P05

```markdown
## Summary

- N6's first pass over the tile for 16 items per thread and below (B4's compaction does not pay at N 1024), B4's compaction above, the warp finish in both. One primitive with a compile-time switch on the tile size.
- **Scope**: Block N 1024 to 32768.
- **Comparison method**: The study harness against the two separate adapters.
- **Status**: Open.

## Problem coverage

The whole block range above k 16.

## Output guarantees

As the two parents.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 21.5, 13, 22.

Owner: TBD
```

### TK-S07.8

- Title: `[CUB] B4: the deterministic path through the per-candidate lookup`
- Labels: `cub`
- Parent: TK-P07
- Dependencies: TK-P07, TK-S09.3

```markdown
## Summary

- prefer_smaller_index costs 4 to 17 percent in B4 today, all of it register pressure from the fallback's index passes. The per-candidate rank lookup of B1_det3 (T65) names B4's emission as in scope, and the pinned shared address of T66 applies to its tickets.
- **Scope**: The deterministic path of B4 at every N.
- **Comparison method**: The study harness, B4 under prefer_smaller_index against B4 under not_guaranteed.
- **Status**: Open. The current price is verified.

## Problem coverage

The deterministic block cells at N 8192 and above.

## Output guarantees

Exact index order among ties.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 21.6, 13.6, T65, T66.

Owner: TBD
```

### TK-S07.9

- Title: `[CUB] B4: same-session pairings with B1_idx at 512 threads and with B2_twolevel at k 16`
- Labels: `cub`
- Parent: TK-P07
- Dependencies: TK-P07

```markdown
## Summary

- B4 against B1_idx exists in one session for the base build only, the later builds chain through it, and the k 16 cells against the two-level B2 list were compared by card numbers (B2 ahead for bf16, B4 for fp32 at N 32768).
- **Scope**: N 16384 and 32768, k 16 to 256.
- **Comparison method**: The phase 3 evaluation under one harness version on one node.
- **Status**: Phase 3 pass (B200 umbriel-b200-047, harness 0.10, 2026-10-25), latency under not_guaranteed: against B1_idx at 512 threads B4_v4 is 47 to 53 percent ahead at bf16 N 16384 k 64 to 256 (5402 to 5829 cycles against 7928 to 8932), 84 to 103 percent at fp32 N 16384, 93 to 100 percent at bf16 N 32768 (6888 to 7183 against 13291 to 14361) and 135 to 154 percent at fp32 N 32768 (5772 to 6058 against 13576 to 15380). At k 16 the two-level B2 list wins bf16 N 16384 (3225 against B4_v4 4970) and N 32768 (4437 against 6423), B2_packed wins fp32 N 16384 (4173 against 4368, a close call, runner-up within 15 percent), B4_v4 wins fp32 N 32768 k 16 (5327 against B2_twolevel 8664).

## Problem coverage

The ownership of block k 16 and k 64 at large N.

## Output guarantees

Not applicable.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 21.7, 17.

Owner: TBD
```

---

### TK-P08

- Title: `[CUB] Block top-k by a compacting radix select (strategy 11)`
- Labels: `cub`, `future-productization`
- Parent: none (new strategy 11)
- Dependencies: TK-P09

```markdown
## Summary

- After the first radix pass the indices above the pivot bucket are written out and the pivot bucket (about N over 256 items) is compacted into shared memory, so the later passes sweep candidates only, and a packed block scan emits the result in index order. Measured as B3 of the study.
- **Scope**: Block of 256 and 512 threads, N 1024 to 32768, k 4 to 4096, bf16 and fp32.
- **Comparison method**: The study harness (TK-P01) against `B1_reference` and `B1_idx` in one lock section, both block sizes.
- **Status**: Seven cards verified (B200 umbriel-b200-047 and 068, CUDA 13.4.92, 2026-10-25). Compaction pays only when it removes at least two tile sweeps, from 16 items per thread for fp32 and 32 for bf16, never more than 1.37x at small k. B3_tuned owns k at least N over 16 from N 2048 (1.3x to 2.0x over the reference through the k-flat scan emission), keeps k 2048 against B4 (B4 0.70x to 0.81x there) and gains 1.12x to 1.15x at 512 threads. Final cells (phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25): B3_tuned wins 11 of 77 block latency cells under not_guaranteed and 15 of 75 under prefer_smaller_index (its scan emission is free of the requirement): k 1024 at N 4096 (bf16 3996 cycles against B1_idx at 512 threads 4193 and B4_v4 5795, a close call (runner-up within 15 percent), fp32 4241 against 5211 and 6268), k 4096 at N 16384 and 32768 (bf16 10489 and 17986 against 11423 and 19918, close calls, fp32 9171 and 16298 against 13132 and 21664, B4_v4 2.0x to 2.7x behind), bf16 N 1024 k 256 (a close call), named cells, and under prefer_smaller_index bf16 N 1024 k 32 and 64 and k 256 at N 4096. On the tie_heavy input as generated B3_tuned keeps its k 1024 and 4096 close calls against B1_idx at 512 threads, and B1_n6b or B1_det2 take the N 1024 ones from it, where the choice stays with the implementer unless the shuffled variant confirms the flip. It loses k 1024 at N 16384 and 32768 to B4_v4 (13 to 114 percent behind) and k 256 from N 8192 (31 to 160 percent behind). Throughput: 2 and 11 cells (bf16 N 4096 k 1024 a close call, deterministic bf16 cells at N 1024 to 32768 and fp32 k 4096).

## Problem coverage

Block k 2048 and 4096 at N 16384 and 32768, the k at least N over 16 band at N 2048 to 4096, and the deterministic cells at those sizes at zero cost.

### Segment / input size
- **Supported range**: N 1024 to 32768 at 256 and 512 threads.
- **Expected sweet spot**: N 2048 to 32768 where compaction removes at least two tile sweeps (from 16 items per thread for fp32 and 32 for bf16).

### k
- **Supported range**: k 1 to N at compile time, measured k 4 to 4096.
- **Expected sweet spot**: k at least N over 16 (the k-flat scan emission), k 2048 and 4096 against B4.

### Types
- **Supported**: bf16 and fp32, int32 indices.
- **Expected sweet spot**: both.

## Output guarantees

### Determinism (set membership)
| `determinism` | Supported | Notes |
|---|---|---|
| `not_guaranteed` | yes | The reference requirement of every measurement. |
| `run_to_run` | yes | Radix passes with a deterministic compaction. |
| `gpu_to_gpu` | yes | The packed scan emits in index order. |

### Tie-break (only meaningful with `gpu_to_gpu`)
| `tie_break` | Supported | Notes |
|---|---|---|
| `unspecified` | yes | Index order from the scan. |
| `prefer_smaller_index` | yes | Zero (0.997x to 1.003x, within 0.3 percent at 512 threads). |
| `prefer_larger_index` | planned | Not built, the mirror form is listed in the report. |

### Output ordering (result sequence)
| `output_ordering` | Supported | Notes |
|---|---|---|
| `unsorted` | yes | Trivially. |
| `sorted` | planned | Select plus a post-sort of the k words (TK-S08.5). |
| `stable_sorted` | planned | Through the stable keys-only sort of the packed word (T55) in the post-sort route. |

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 15, 17, 16, report `approaches/B3/REPORT.md`, techniques T29, T31, T30.

Owner: TBD
```

### TK-S08.1

- Title: `[CUB] B3 at 512 threads from N 16384`
- Labels: `cub`
- Parent: TK-P08
- Dependencies: TK-P08

```markdown
## Summary

- 512 threads give 1.12x to 1.15x in latency and 1.00x to 1.12x in throughput at N 16384 and 32768, the spills stay under the 128-register cap (64 bytes bf16, 844 fp32), 1024 threads are slower again. Against B1_idx at 512 threads B3 keeps a latency edge at k 1024 only.
- **Scope**: N 16384 and 32768.
- **Comparison method**: The study harness (card `N8_b3tuned`, verified on a second node, B200, CUDA 13.4.92, 2026-10-25).
- **Status**: Measured and verified, the block size is a policy change.

## Problem coverage

The large-N cells of B3.

## Output guarantees

prefer_smaller_index within 0.3 percent at 512 threads.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 17.4, 15.7, T45.

Owner: TBD
```

### TK-S08.2

- Title: `[CUB] B3: the compaction threshold per N instead of per items per thread`
- Labels: `cub`
- Parent: TK-P08
- Dependencies: TK-P08, TK-S08.1

```markdown
## Summary

- The compaction threshold is per items per thread, with the block size free it may want to be per N.
- **Scope**: The policy of B3 at 256 and 512 threads.
- **Comparison method**: The study harness across block sizes.
- **Status**: Open.

## Problem coverage

The B3 band at matched block sizes.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 17.5.

Owner: TBD
```

### TK-S08.3

- Title: `[CUB] B3: the fp32 fallback cells move with the pass-loop structure`
- Labels: `cub`
- Parent: TK-P08
- Dependencies: TK-P08

```markdown
## Summary

- Commit 50039daee3 restructured the pass loop and moved the fp32 fallback cells by 7 to 10 percent, so a finding on one commit does not transfer. The cells need a schedule-stable form or a per-commit measurement.
- **Scope**: fp32 at N 16384 and 32768 where the fallback runs.
- **Comparison method**: The study harness per commit.
- **Status**: Recorded (KB 2.10, 15).

## Problem coverage

Robustness of the fp32 B3 numbers.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.10, 15.4, T28.

Owner: TBD
```

### TK-S08.4

- Title: `[CUB] B3: the k at least N over 16 band against B4 and B1_idx at 512 threads`
- Labels: `cub`
- Parent: TK-P08
- Dependencies: TK-P08, TK-P07

```markdown
## Summary

- B4 takes N 8192 to 32768 for k 64 to 1024 and B1_idx at 512 threads shrinks B3's latency edge to k 1024, so B3's band narrows to k 2048, the N 2048 to 4096 cells and the deterministic cells. The final boundaries come from one session.
- **Scope**: N 2048 to 32768, k 64 to 4096.
- **Comparison method**: The phase 3 evaluation.
- **Status**: Phase 3 pass (B200 umbriel-b200-047, harness 0.10, 2026-10-25), latency under not_guaranteed: at k 4096 B3_tuned leads B1_idx at 512 threads by 9 and 11 percent (bf16 N 16384 and 32768, a close call, runner-up within 15 percent) and by 43 and 33 percent (fp32), with B4_v4 2.0x to 2.7x behind. At k 1024 it wins N 4096 (bf16 5 percent ahead of B1_idx at 512 threads, a close call, fp32 23 percent) and loses N 16384 and 32768 to B4_v4 by 13 and 76 percent (bf16) and 25 and 114 percent (fp32). At k 256 it wins nothing: N 4096 goes to B1_idx at 512 threads (bf16 3959 against 3994, a close call) and N9_w2fin2 (fp32 4003 against 4165, a close call), N 8192 to 32768 to B4_v4 by 31 to 160 percent.

## Problem coverage

The ownership of the large-k block cells.

## Output guarantees

Not applicable.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 15.7, 21.4, 17.4.

Owner: TBD
```

### TK-S08.5

- Title: `[CUB] Select plus post-sort as the sorted route for one unsorted tile`
- Labels: `cub`
- Parent: TK-P08
- Dependencies: TK-P08, TK-P10

```markdown
## Summary

- For a single unsorted tile the cheapest sorted output is B3_tuned plus a sort of the k selected words with the keys-only packed-word sort (T55) for fp32 at every k and for bf16 at k at most N over 4 (fp32 N 2048 k 1024: 2509 plus 9767 against 13487 for the full sort), the full sort only for bf16 at k equal to N over 2. The gather of the selected keys is unmeasured.
- **Scope**: Block sorted output of one tile, N 1024 to 8192.
- **Comparison method**: The phase 3b sorted-output pass with the order check.
- **Status**: Card numbers (B6 cards unverified by rerun, B200 umbriel-b200-020, CUDA 13.4.92, 2026-10-25). `<phase 3>`.

## Problem coverage

The sorted-output pass at block scope.

## Output guarantees

Sorted by key descending then index ascending through the stable sort.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 16 (sorted-output tables), 10.5 (R17), T54, T55.

Owner: TBD
```

---

### TK-P09

- Title: `[CUB] Block AIR radix select: index-only epilogue, deterministic ticket and lookup, block size (the shipped block primitive, no entry in #9747)`
- Labels: `cub`, `future-productization`
- Parent: none (#9747 has no entry for the block radix select, strategies 5 and 6 build on it)
- Dependencies: none

```markdown
## Summary

- The shipped block top-k (`block_topk_air`, the baseline backend of `DeviceBatchedTopK`) with the measured improvements of the study: an index-only ticket epilogue, the deterministic path with a class-indexed ticket and a per-candidate tie lookup, the pinned shared window base, 512 threads at large N, and the N6 first pass where it saves a pass. Measured as B1, B1_det2, B1_det3 and N8 of the study.
- **Scope**: Block of 256 and 512 threads (64 and 1024 measured and rejected), N 1024 to 32768, k 4 to 4096, bf16 and fp32 on the striped base.
- **Comparison method**: The study harness (TK-P01) against `B1_reference` (the shipped pairs path) in one lock section, same-kernel A/Bs for the epilogue spellings, clock64 phase stamps for the tie phase.
- **Status**: Cards verified (B200 umbriel-b200-047, 068, 018, CUDA 13.4.92, 2026-10-25). The index-only epilogue saves 5 to 19 percent and removes the exchange buffer and three barriers (card `B1_idx`). 512 threads give 1.24x to 1.45x at N 32768 with 100 to 103 registers and no spill (card `N8_b1idx`). The deterministic path: the constant-address ticket cost 10.8 to 22.3 percent of the call through the compiler's warp aggregation (card `B1_det2`, the same-kernel A/B), and the per-candidate lookup takes the price of prefer_smaller_index to minus 1.3 to plus 8.5 percent on fp32 and 4.4 to 24.4 percent on bf16 with B1_idx's registers (cards `B1_det3`). Final cells (phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25): B1_idx wins 20 of 84 block throughput cells under not_guaranteed at N 1024 to 8192 (bf16 N 1024 k 32 to 256, N 4096 k 64, N 8192 k 64 to 256, fp32 N 1024 k 48 to 256, N 4096 k 16, 64 and 1024, N 8192 k 16), every one a tie with B1_det2 at the same cycles and a close call (runner-up within 15 percent), its 512-thread form 9 throughput cells at N 16384 and 32768 (bf16 N 16384 k 64 to 4096, N 32768 k 4096, fp32 k 1024 and 4096) and one latency cell (bf16 N 4096 k 256, 3959 against B3_tuned 3994, a close call). B1_det2 wins 2 deterministic latency cells (fp32 N 1024 k 32 and 64, a close call) and 16 of 74 deterministic throughput cells at N 1024 to 8192, and 11 not_guaranteed throughput cells as the tie partner. Across the deterministic block cells the winner is a median 104 percent ahead of B1_det in latency and 60 percent in throughput. The B1_idx ties with B1_det2 are the same selection with and without the deterministic tie-break, so the deterministic column decides them: B1_det2 where prefer_smaller_index is wanted (a median 7 percent on the cells both ran), B1_idx otherwise. The 512-thread form keeps its close calls on the tie_heavy input as generated except bf16 N 4096 k 256 (B1_n6b) and N 16384 k 4096 (B3_tuned), where the choice stays with the implementer unless the shuffled variant confirms the flip. The B1_det3 addendum was measured on umbriel-b200-068 under the random label and sits outside the one-node matrices (110 prefer_smaller_index block cells, `state/phase3/det3_cells_b200.md`): read across nodes, B1_det3 is a deterministic variant at B1_det2's speed, within 2 percent of det2 in latency at N 1024, and no winner changes hands on its account. The cells where det3 reads ahead of a 047 winner go to the same-node rerun on the pass node before any matrix changes, and an earlier count of 22 deterministic cells won was a cross-node artifact.

## Problem coverage

The block fallback everyone pays on tie floods, the owner of block k 64 to 256 at N 1024 to 4096 (with N6 and N9), the incumbent at 512 threads at N 16384 and 32768 for k up to 256 where B4 does not apply, and the deterministic block primitive at every N.

### Segment / input size
- **Supported range**: N 1024 to 32768 at 256 and 512 threads (the shipped primitive covers the baseline backend's segment sizes).
- **Expected sweet spot**: every N as the fallback, N 1024 to 4096 for k 64 to 256 and N 16384 to 32768 at 512 threads where B4 does not apply.

### k
- **Supported range**: k 1 to N at run time, measured k 4 to 4096.
- **Expected sweet spot**: k 32 to 1024, nearly k-flat passes with an epilogue that grows with k.

### Types
- **Supported**: bf16, fp32 (the radix digits per key width), fp8 as a single pass and fp64 by contract, unmeasured in the study.
- **Expected sweet spot**: fp32, bf16 is not cheaper (two 8-bit passes without early exit).

## Output guarantees

### Determinism (set membership)
| `determinism` | Supported | Notes |
|---|---|---|
| `not_guaranteed` | yes | The reference requirement of every measurement. |
| `run_to_run` | yes | Ticket order among ties under not_guaranteed is run dependent, the set is not. |
| `gpu_to_gpu` | yes | The deterministic path (B1_det3). |

### Tie-break (only meaningful with `gpu_to_gpu`)
| `tie_break` | Supported | Notes |
|---|---|---|
| `unspecified` | yes | Ticket order. |
| `prefer_smaller_index` | yes | fp32 minus 2 to plus 9 percent, bf16 4 to 24 percent (the per-slot mask build and scan on tie rows). |
| `prefer_larger_index` | planned | The rank test flips, not built. |

### Output ordering (result sequence)
| `output_ordering` | Supported | Notes |
|---|---|---|
| `unsorted` | yes | Trivially. |
| `sorted` | no | Post-sort of k. |
| `stable_sorted` | no | Not available. |

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 13, 17, 4, 16, reports `approaches/B1/REPORT.md`, `approaches/B1/DET_TICKET.md`, `approaches/N8/REPORT.md`, techniques T21, T23, T39, T45, T57, T61, T65, T66, PR #11060, branch `enh/block-topk-striped`.

Owner: TBD
```

### TK-S09.1

- Title: `[CUB] block_topk_air: the index-only ticket epilogue`
- Labels: `cub`
- Parent: TK-P09
- Dependencies: TK-P09

```markdown
## Summary

- One independent atomic ticket per selected item instead of the pairs exchange epilogue removes the exchange buffer, three barriers and the memoized key copy: 5 to 19 percent, temp storage independent of N, no spill at 128 items per thread (T21).
- **Scope**: The not_guaranteed path of the block top-k, indices-only output.
- **Comparison method**: The study harness against `B1_reference` (card `B1_idx`, verified, B200, CUDA 13.4.92, 2026-10-25).
- **Status**: Measured and verified, not in the header.

## Problem coverage

Every block cell where the radix select runs.

## Output guarantees

Ties in ticket order under not_guaranteed.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 13.4, 2.12, T21.

Owner: TBD
```

### TK-S09.2

- Title: `[CUB] block_topk_air deterministic path: the class-indexed ticket`
- Labels: `cub`
- Parent: TK-P09
- Dependencies: TK-P09, TK-S09.1

```markdown
## Summary

- A ticket drawn by atomicAdd on a constant shared address from an unrolled item loop is warp-aggregated by ptxas (two special register reads and an election per site on every item's store chain). Spelling it as one atomic per item on a class-indexed counter removes it: 10.8 to 22.3 percent of the deterministic call (358 cycles at fp32 N 1024 k 16) and the whole fp32 price at N 1024 and 4096 (T57, T61).
- **Scope**: The deterministic epilogue of the block top-k and every epilogue that assigns slots by shared tickets.
- **Comparison method**: The same-kernel A/B, cards `B1_det2_1` and `_2` (verified on the cards' node, B200 umbriel-b200-068, CUDA 13.4.92, 2026-10-25).
- **Status**: Measured and verified, not in the header.

## Problem coverage

The deterministic block cells.

## Output guarantees

Exact index order among ties.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 13.6, 2.10, T23, T39, T57, T61.

Owner: TBD
```

### TK-S09.3

- Title: `[CUB] block_topk_air deterministic path: per-candidate tie lookup and the pinned shared base`
- Labels: `cub`
- Parent: TK-P09
- Dependencies: TK-P09, TK-S09.2

```markdown
## Summary

- The tie phase per item slot (a ballot, a table store, a base shuffle, a popcount and two registers per slot) becomes per candidate: ballot words into a (warp, slot) table, bases by one shuffle scan per warp, rank by two shared loads and a popcount per candidate (T65). Registers equal B1_idx's at every items per thread, two blocks per SM at N 16384, no spill at 512 threads, the premium on tie rows down 57 to 79 percent. The shared window base must be pinned, ptxas rematerializes it in front of every shared atomic at a tight register target (T66, 300 to 1100 cycles per call).
- **Scope**: The deterministic path at 16 items per thread and above (per-slot ballots stay up to 8).
- **Comparison method**: Cards `B1_det3_1` and `_2` (verified on a second node, B200 umbriel-b200-018 and 068, CUDA 13.4.92, 2026-10-25), phase stamps.
- **Status**: Measured and verified, not in the header.

## Problem coverage

The deterministic block cells at every N, bf16 above all.

## Output guarantees

Exact index order among ties, the three deterministic forms digest-identical in every row.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 13.6, 2.10, T65, T66, `approaches/B1/DET_TICKET.md` sections 7 to 12.

Owner: TBD
```

### TK-S09.4

- Title: `[CUB] block_topk_air block size: 512 threads from N 8192, 64 threads for one-wave throughput`
- Labels: `cub`
- Parent: TK-P09
- Dependencies: TK-P09

```markdown
## Summary

- 512 threads halve the dependent histogram sweep, run 16 warps per SM at the same one block per SM and split the ticket epilogue over twice the warps: 1.24x to 1.45x in latency at N 32768 and 1.27x to 1.46x in throughput, 1.19x to 1.35x at N 16384, 1.05x to 1.17x at N 8192 (latency only), 1024 threads never better than 512 beyond the noise. At N 1024 to 2048 the one-wave throughput regime wants 64-thread blocks (1.2x to 1.6x), the latency regime 256.
- **Scope**: The tuning policy of the block top-k per N and regime.
- **Comparison method**: Card `N8_b1idx` (verified on a second node, B200, CUDA 13.4.92, 2026-10-25), the device benchmark for the batched regime.
- **Status**: Measured and verified, the policy change open.

## Problem coverage

Block N 8192 to 32768 for the radix path, the batched throughput at small N.

## Output guarantees

Unchanged, B1_det not swept (its tie table doubles at 512 threads, det3 holds 102 registers there).

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 17, T44, T45.

Owner: TBD
```

### TK-S09.5

- Title: `[CUB] block_topk_air: fold the minus-zero normalization into the first digit extraction`
- Labels: `cub`
- Parent: TK-P09
- Dependencies: TK-P09

```markdown
## Summary

- The normalization is two instructions per key in the prologue, 25 to 31 cycles at N 1024 and, at 512 threads and 64 items per thread, 1000 to 1250 cycles per call on rows without ties, now the largest fp32 term of the deterministic path. Folding it into the digit extractor of the first pass only is untested.
- **Scope**: The deterministic path and the not_guaranteed path under the signed-zero contract.
- **Comparison method**: The study harness against `B1_det3`.
- **Status**: Open.

## Problem coverage

fp32 at N 16384 and 32768.

## Output guarantees

Both zeros one tie class.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 13.7, 4.4, T23.

Owner: TBD
```

### TK-S09.6

- Title: `[CUB] block_topk_air: the striped arrangement branch, the raw extractor and NormalizeSignedZero`
- Labels: `cub`
- Parent: TK-P09
- Dependencies: TK-P09

```markdown
## Summary

- Branch `enh/block-topk-striped` (striped arrangement plus striped histogram plus E2) was validated on B200 on 2026-10-11 and its rebased copy on 2026-10-19, with the barrier elision guarded by a static assert. The raw extractor and NormalizeSignedZero are its open items. The study's block primitives all run on the striped base.
- **Scope**: The shipped block top-k.
- **Comparison method**: The segmented top-k benchmarks of the repository plus the study harness.
- **Status**: Branch validated, no PR, the f64 register caveat recorded (64 to 74 registers at 256 by 4).

## Problem coverage

The baseline backend of DeviceBatchedTopK.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 4.3, 4.4, 6, `CUB_EMPTY_TEMP_STORAGE_DESIGN.md`.

Owner: TBD
```

### TK-S09.7

- Title: `[CUB] block_topk_air: the N6 first pass for fp32 N 4096 and N 16384 k 16`
- Labels: `cub`
- Parent: TK-P09
- Dependencies: TK-P09, TK-P05

```markdown
## Summary

- A fixed-point image over the tile's value bracket in place of the first radix digit leaves 1.1 to 7.3 candidates after one 256-bin pass and is exact after two passes in 97 to 100 percent of random rows where the fp32 8-bit schedule needs three. N6 wins where it saves a pass (fp32 N 4096 k 16 to 256 by 11 to 24 percent, N 16384 k 16 by 1.48x and 1.58x) and loses where it does not and in throughput by occupancy.
- **Scope**: fp32 at N 4096 and both types at N 16384 k 16.
- **Comparison method**: Cards `B1_n6_2`, `B1_n6b_1` and `_2` (verified, B200, CUDA 13.4.92, 2026-10-25).
- **Status**: Measured and verified, a digit schedule choice in the header.

## Problem coverage

The fp32 N 4096 band and the N 16384 k 16 cells.

## Output guarantees

Unchanged, N9 (TK-S05.5) carries the deterministic path.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 13.4, 17.4, 2.12, T25, T30.

Owner: TBD
```

### TK-S09.8

- Title: `[CUB] block_topk_air: adopt the class-branch epilogue of B1_det3 in the not_guaranteed path`
- Labels: `cub`
- Parent: TK-P09
- Dependencies: TK-P09, TK-S09.3

```markdown
## Summary

- det3 is 1.3 to 1.9 percent faster than B1_idx's not_guaranteed path at fp32 N 1024 and cheaper by 42 to 681 cycles for bf16 in the epilogue: the class branch beats the bounded tie store.
- **Scope**: The not_guaranteed epilogue of the block top-k.
- **Comparison method**: The study harness, B1_idx with the new epilogue against B1_idx.
- **Status**: Open, the measurement is in the det3 cards (verified).

## Problem coverage

Every block radix cell.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 13.6, 13.7.

Owner: TBD
```

### TK-S09.9

- Title: `[CUB] block_topk_air deterministic path: the mask build and the scan are still per slot`
- Labels: `cub`
- Parent: TK-P09
- Dependencies: TK-P09, TK-S09.3

```markdown
## Summary

- After the lookup the tie phase is still per slot in the mask build (about 3.5 instructions per slot, 230 at 64 items per thread) and the scan (136 instructions for two rounds), and it runs on 66 to 91 percent of the bf16 rows at N 4096 and above, which is the remaining 12 to 24 percent bf16 price at N 16384 and 32768. A cheaper mask build is the next lever, the per-slot compare is shared with the epilogue, which recomputes it.
- **Scope**: The deterministic path at 16 items per thread and above, bf16.
- **Comparison method**: Phase stamps plus the study harness against `B1_det3`.
- **Status**: Open.

## Problem coverage

The deterministic bf16 block cells at large N.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 13.6, 13.7, T65.

Owner: TBD
```

---

### TK-P10

- Title: `[CUB] k-truncated merge of sorted runs at warp and block scope for sorted output and chunk merges (strategy 12)`
- Labels: `cub`, `future-productization`
- Parent: none (new strategy 12)
- Dependencies: TK-P01, TK-P08

```markdown
## Summary

- On the packed word (key descending, index ascending) a merge tree takes R sorted runs and merges adjacent pairs level by level, every merged run truncated to k, the merge-path split per thread a static binary search with the start split from the predecessor, the chunks of 16 outputs one half-cleaner stage plus a bitonic merge on static register indices. The block form merges M sorted chunk results (the DSA and vocabulary merge shapes), the warp form sorts per lane and merges five levels for sorted output of one row. Measured as W4 of the study.
- **Scope**: Block of 256 threads with M 4 or 8 sorted runs, N 4096 to 8192 (N to 32768 fp32 and 65536 bf16 by contract), k 512 to 2048. Warp of 32 lanes, N 512 to 2048 (2 to 64 items per lane), k 32 to 1024. bf16 as a 32-bit word, fp32 as a 64-bit word. The runs must arrive sorted and full.
- **Comparison method**: The study harness (TK-P01) with the sorted-runs input of harness 0.8, against `B3_tuned`, `B6_radix`, `W2_packed` and `W5_v4` in one lock section, the sorted order checked on every cell by the exact check added in verification.
- **Status**: Five cards verified on the cards' node within 0.6 percent with the sorted order checked (B200 umbriel-b200-091, CUDA 13.4.92, 2026-10-25). Block: 4 runs of 2048 to the sorted top 2048 in 4266 cycles bf16 and 6571 fp32 against B3_tuned's unsorted 6501 and 7100 (1.52x, 1.08x) and B6's full sort 13511 and 32578, 8 runs 1.22x bf16 and 0.82x fp32 at N 8192 k 1024. Warp: bf16 N 2048 k 1024 19786 against W2's sorted 150078 (7.6x), fp32 42598 against 154818 (3.6x), 1.24x to 3.0x over W2 at k 256, 0.29x to 0.54x of W5_v4 at k 64. Final cells (phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25): the sorted-runs track at block scope, W4_chunk4 on 4 sorted runs and W4_chunk8 on 8, N 8192 to 32768, k 256 to 4096, against B1_reference's unsorted selection of the same cell. At 4 runs the reference costs 1.27x (fp32 N 8192 k 4096) to 6.19x (fp32 N 16384 k 256) the merge in latency and the merge delivers 1.02x to 3.17x the throughput. At 8 runs 0.71x to 3.71x in latency and 0.66x to 2.81x in throughput, the one loss at fp32 N 8192 k 4096 (18527 cycles against 13238). Both requirements identical. Warp sorted output (phase 3b, B200 umbriel-b200-047, harness 0.10, 2026-10-26): W4_wmerge is the sorted route at k 256 and 1024 from N 512 (fp32) or N 2048 (bf16), for example fp32 N 2048 k 256 32001 cycles against the unsorted N6w_atom32 15540, and at fp32 N 512 and 2048 k 64 under prefer_smaller_index where W5_v4's fp32 output is refuted, W5_v4 is the sorted route at k 4 to 64 from N 512 otherwise, and the W1 network at N 32 and 128.

## Problem coverage

The cheapest sorted route at every merge cell by 2x to 3x when the chunk results arrive sorted (W1, W2, W5 and B6 produce them sorted, B1 to B4 and N6 do not), ahead of the unsorted radix select at four runs, and sorted output at warp scope from k 256 (W5 up to k 64, W2 below that where it wins the unsorted cell).

### Segment / input size
- **Supported range**: block runs of N over M consecutive sorted items, N 4096 to 65536 (bf16) or 32768 (fp32) with M 4 or 8, warp N 64 to 2048 (2 to 64 items per lane).
- **Expected sweet spot**: the merge cells of the chunked workloads (4 sorted runs of 2048 to the top 2048), warp N 512 to 2048 for sorted output.

### k
- **Supported range**: k a power of two with 2k at most N, block k 512 to 2048 measured, warp k 32 to N over 2.
- **Expected sweet spot**: k at least 256 at warp scope, k equal to the run length at block scope.

### Types
- **Supported**: bf16 as a 32-bit word, fp32 as a 64-bit word, int32 indices.
- **Expected sweet spot**: bf16 (the 64-bit word costs 2.5x per exchange).

## Output guarantees

### Determinism (set membership)
| `determinism` | Supported | Notes |
|---|---|---|
| `not_guaranteed` | yes | The reference requirement of every measurement. |
| `run_to_run` | yes | A fixed merge tree on sorted runs. |
| `gpu_to_gpu` | yes | The word order, the merge path takes the first run on ties. |

### Tie-break (only meaningful with `gpu_to_gpu`)
| `tie_break` | Supported | Notes |
|---|---|---|
| `unspecified` | yes | Resolved by the word order. |
| `prefer_smaller_index` | yes | Zero (identical to the cycle in every pair). |
| `prefer_larger_index` | planned | The uninverted index and the second run on ties, not built. |

### Output ordering (result sequence)
| `output_ordering` | Supported | Notes |
|---|---|---|
| `unsorted` | yes | Trivially. |
| `sorted` | yes | Slot r holds rank r, it is the primitive. |
| `stable_sorted` | yes | Through the word order. |

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 23, 16 (sorted-output tables), report `approaches/W4/REPORT.md` section 6, technique T62, `approaches/W4/host_test.cpp`, `reference/WARP_MERGE_SORT_RESULTS.md`.

Owner: TBD
```

### TK-S10.1

- Title: `[CUB] W4 warp: the fp32 per-lane sort of 64 words`
- Labels: `cub`
- Parent: TK-P10
- Dependencies: TK-P10

```markdown
## Summary

- The Batcher sort of 64 fp32 words per lane (543 exchanges on 64-bit words) costs 15.7k of 42.6k cycles at N 2048 k 1024 in the final build and 29k in the one before, at 255 registers ptxas serializes the network. Sort in four pieces of 16 with two lane-local merge levels, a key-only u32 network with the index as payload and a stable tie rule, or an sm_100f build with the key half as a float.
- **Scope**: fp32 at 64 items per lane.
- **Comparison method**: The phase probe plus the study harness against `W4_wmerge`.
- **Status**: Open. The two-half sort (`W4_wmerge2`) was tried and lost 0.91x to 0.98x.

## Problem coverage

Every fp32 warp sorted cell.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 23.3, 23.5, T34.

Owner: TBD
```

### TK-S10.2

- Title: `[CUB] W4 warp: a hybrid run layout for the deeper levels`
- Labels: `cub`
- Parent: TK-P10
- Dependencies: TK-P10

```markdown
## Summary

- The merge of the same 64 outputs per lane grows from 1864 to 3581 cycles from level 0 to 3 because the layout spreads runs over banks and not over the lanes of one pair, 16-way conflicts at level 3. A layout per level predicted at 15 to 20 percent at k 1024.
- **Scope**: The warp form at k 256 and 1024.
- **Comparison method**: The phase probe plus the study harness.
- **Status**: Open.

## Problem coverage

The warp sorted cells at large k.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 23.3, 23.5.

Owner: TBD
```

### TK-S10.3

- Title: `[CUB] W4 block: a 4-way merge path for eight runs and 128-bit staging`
- Labels: `cub`
- Parent: TK-P10
- Dependencies: TK-P10

```markdown
## Summary

- For eight runs the third level adds a flat 1800 cycles and the fp32 form loses to the unsorted selection (0.82x). A 4-way merge path per level (two pairs materialized together) halves the levels and the barriers, STS.128 staging with a quad-preserving swizzle and the fp32 level 1 words as one 64-bit array cut the exchange traffic. The 4-ary search itself lost (TK-P10 note).
- **Scope**: The block form at M 8.
- **Comparison method**: The study harness against `W4_chunk8` and `B3_tuned`.
- **Status**: Open.

## Problem coverage

The eight-run merge cells.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 23.3, 23.5.

Owner: TBD
```

### TK-S10.4

- Title: `[CUB] W4 warp at N 4096`
- Labels: `cub`
- Parent: TK-P10
- Dependencies: TK-P10

```markdown
## Summary

- At 128 items per lane an in-lane merge level of two sorted 64-runs before the five cross-lane levels keeps the per-lane network at 64 and under the local-memory threshold.
- **Scope**: Warp N 4096.
- **Comparison method**: The study harness against W1_reference and W2 where they compile.
- **Status**: Open.

## Problem coverage

The unmeasured warp N 4096 row.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 23.5, T34.

Owner: TBD
```

### TK-S10.5

- Title: `[CUB] W4 warp: fp32 N 2048 in the throughput regime`
- Labels: `cub`
- Parent: TK-P10
- Dependencies: TK-P10

```markdown
## Summary

- Eight warps times 39 KB of temp storage exceed the 227 KB block limit, so the fp32 N 2048 cells are latency only. A 16-bit index half (N at most 65536) brings k 256 inside the limit.
- **Scope**: fp32 at N 2048.
- **Comparison method**: The study harness, throughput regime.
- **Status**: Open.

## Problem coverage

The batched sorted-output cells for fp32.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 23.2, 23.5.

Owner: TBD
```

### TK-S10.6

- Title: `[CUB] W4 block: runs shorter than N over M`
- Labels: `cub`
- Parent: TK-P10
- Dependencies: TK-P10

```markdown
## Summary

- A per-chunk top-k may return fewer than k valid items. Per-run lengths in the merge path and the pads make the merge accept them.
- **Scope**: The block chunk merge.
- **Comparison method**: Host test plus the study harness.
- **Status**: Open.

## Problem coverage

Chunk merges of variable-length results.

## Output guarantees

Unchanged.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 23.5, 23.7.

Owner: TBD
```

### TK-S10.7

- Title: `[CUB] Chunk-merge contract in the device agents: sorted runs from W1, W2, W5 or B6 only`
- Labels: `cub`
- Parent: TK-P10
- Dependencies: TK-P10

```markdown
## Summary

- The merge needs sorted and full runs. A device agent whose chunk results come from the radix selects or N6 (unsorted) needs W5, W2 or a sort for the chunks first, or the unsorted B3_tuned over the M k candidates (7100 against 6571 for fp32 and 6501 against 4266 for bf16 at M 4).
- **Scope**: The multi-chunk device agents (vocabulary, DSA, beam, draft trees of KB 10.2).
- **Comparison method**: The phase 3b pass and the device benchmarks.
- **Status**: Contract recorded, no agent built.

## Problem coverage

Every workload that exceeds one block.

## Output guarantees

Sorted and exact through the merge when the inputs are.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 23.7, 10.2 (chunk merges), 5.

Owner: TBD
```

---

### TK-P11

- Title: `[CUB] Top-k determinism and ordering features across the cooperative primitives`
- Labels: `cub`
- Parent: none (cross-cutting)
- Dependencies: TK-P01 to TK-P10

```markdown
## Summary

- What each family guarantees under `cuda::execution::tie_break` and `output_ordering`, with the measured price, so that the requirements API can route a request to the cheapest primitive that satisfies it. From the cooperative primitives study, waves 1 and 2.
- **Scope**: prefer_smaller_index, prefer_larger_index, sorted output, signed zero and NaN behavior, per family and per scope.
- **Comparison method**: The study harness with both requirements in one lock section (`--control-req` lets a deterministic variant run against a not_guaranteed control), the exact index-set oracle under prefer_smaller_index, the exact order check for sorted output.
- **Status**: Measured and verified (B200, CUDA 13.4.92, 2026-10-25). prefer_smaller_index: zero for W2, B2, B3, W5, W4, N6 at warp scope and the bf16 W1 word (same binary, identical cycles), negative for fp32 in the W1 family through N10 (0.99x to 0.27x of the pairs), free on fp32 and 4 to 24 percent on bf16 for the block AIR path (B1_det3), 4 to 17 percent for B4, zero to 4 percent for N5. Sorted output: the merge of sorted runs is the cheapest sorted route at every merge cell by 2x to 3x, the full sort only for bf16 single tiles at k equal to N over 2, at warp scope W2 at k at most 32 and N at most 512, W5 up to k 64, W4 from k 256. prefer_larger_index is built nowhere. Final cells (phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25): prefer_smaller_index against not_guaranteed on the same adapter and cell, latency ratio: W2_packed_v3, W5_v4, N6w_atom32, N10_auto, W3_d4c, W4_wmerge and the W1 family 0.998 to 1.002 on 70 warp cell pairs, the sub-warp winners identical, B2_packed, the 128-thread B2, B2_twolevel and B3_tuned 0.996 to 1.004 on the block pairs, B4_v4 median 1.07 (1.05 to 1.47), N9_w2fin2 median 1.15 (1.03 to 1.43), B1_det2 median 1.07 (0.99 to 1.34), B1_n6b median 1.40 (1.14 to 1.69). The deterministic block winner is a median 104 percent ahead of B1_det in latency and 60 percent in throughput. Sorted output (phase 3b): the routes per scope are in TK-S11.2 and knowledge base 16, with W5_v4's fp32 prefer_smaller_index sorted output refuted as built, N5 unsorted by contract and B6_radix unsupported at fp32 N 32768.

## Problem coverage

The routing decisions of the requirements API for every cell of the winner matrix.

### Segment / input size
- **Supported range**: every N of the families above.
- **Expected sweet spot**: not applicable, a routing table.

### k
- **Supported range**: every k of the families above.
- **Expected sweet spot**: not applicable.

### Types
- **Supported**: bf16 and fp32 as measured, the other types with the full evaluation.
- **Expected sweet spot**: not applicable.

## Output guarantees

### Determinism (set membership)
| `determinism` | Supported | Notes |
|---|---|---|
| `not_guaranteed` | yes | Every family. |
| `run_to_run` | per family | See the price table in the body. |
| `gpu_to_gpu` | per family | See the price table. |

### Tie-break (only meaningful with `gpu_to_gpu`)
| `tie_break` | Supported | Notes |
|---|---|---|
| `unspecified` | yes | See the body. |
| `prefer_smaller_index` | per family | See the price table. |
| `prefer_larger_index` | no | Built nowhere, TK-S11.3. |

### Output ordering (result sequence)
| `output_ordering` | Supported | Notes |
|---|---|---|
| `unsorted` | yes | Trivially. |
| `sorted` | per family | See the sorted-output routes. |
| `stable_sorted` | per family | The word-order families only. |

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.12 (the price bullet), 16 (sorted-output tables), the determinism subsections of sections 11 to 23, `TOPK_KNOWLEDGE_BASE.md` 1.2 for the requirements API.

Owner: TBD
```

### TK-S11.1

- Title: `[CUB] The measured price of prefer_smaller_index per top-k family`
- Labels: `cub`, `docs`
- Parent: TK-P11
- Dependencies: TK-P11

```markdown
## Summary

- A table per family and scope, from the cards: the zero-cost families through the packed word or the index-ordered scan, the fp32 W1 route through N10 at negative cost, the block AIR path after the ticket and lookup fixes, B4 and N5. Kept current with the phase 3 numbers.
- **Scope**: Every family of the study.
- **Comparison method**: Both requirements in one lock section.
- **Status**: Verified numbers in KB 2.12, final (phase 3 pass, B200 umbriel-b200-047, harness 0.10, 2026-10-25): the price of prefer_smaller_index as the latency ratio of the same adapter under both requirements, zero (0.996 to 1.004) for W2, W5, N6w, N10, W3, W4, the W1 family, B2 in all three forms and B3_tuned, a median 7 percent for B4_v4 (5 to 47 percent), 15 percent for N9_w2fin2 (3 to 43 percent), 7 percent for B1_det2 (minus 1 to 34 percent) and 40 percent for B1_n6b (14 to 69 percent), throughput losses in the same order (B4_v4 median 5 percent, N9 26 percent, B1_det2 40 percent). B1_det3's deterministic cells are cross-node evidence (umbriel-b200-068 against the 047 matrices, 110 cells): a deterministic variant at B1_det2's speed, within 2 percent of det2 in latency at N 1024, no winner changes hands, the same-node rerun pending.

## Problem coverage

The routing table of the requirements API.

## Output guarantees

The table of the parent.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.12, 13.6, 19.6, 21.6.

Owner: TBD
```

### TK-S11.2

- Title: `[CUB] The sorted-output routes per scope`
- Labels: `cub`, `docs`
- Parent: TK-P11
- Dependencies: TK-P11, TK-P10

```markdown
## Summary

- Block: the k-truncated merge of sorted runs for merge cells (2x to 3x), the B6 full sort only for bf16 single tiles at k equal to N over 2, select plus post-sort otherwise (gather unmeasured). Warp: W2 at small k, W5's sorted finish up to k 64, W4 from k 256. The phase 3b pass runs the three routes in one session under the order check.
- **Scope**: Block and warp sorted output.
- **Comparison method**: Phase 3b.
- **Status**: Phase 3b pass (B200 umbriel-b200-047, harness 0.10 with the sorted check, 2026-10-26): the cheapest sorted route per cell is W5_v4 at warp scope from N 512 at k 4 to 64 (fp32 under not_guaranteed only, its fp32 prefer_smaller_index output is refuted as built), W4_wmerge at k 256 and 1024 and at fp32 N 512 and 2048 k 64 under prefer_smaller_index, the W1 level network at N 32 to 128 and where W4_wmerge does not apply, W1_eager_same_sub at sub-warp scope, and at block scope B6_radix as the only sorted route from an unsorted row (2.0x to 95x the unsorted winner of the cell, unsupported at fp32 N 32768) with the merge of presorted runs the cheapest route when the input arrives as runs. The price of sorted output over the unsorted winner of each cell is tabulated in knowledge base 16.

## Problem coverage

`output_ordering::sorted`.

## Output guarantees

Sorted by key descending, ties by index ascending on every route.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 16 (sorted-output tables), 23, 10.5 (R17).

Owner: TBD
```

### TK-S11.3

- Title: `[CUB] prefer_larger_index in the cooperative primitives`
- Labels: `cub`
- Parent: TK-P11
- Dependencies: TK-P11

```markdown
## Summary

- Not built in any family. The mirror forms are known: the uninverted index in the packed word (W2, B2, W1, W4, one instruction less), the mirror rank in N10, one flipped comparison in B4's finish and index digit, the flipped rank test in B1, the second run on ties in W4's merge path, the inverted index bits sorted in B6.
- **Scope**: Every family.
- **Comparison method**: The study harness with the exact oracle for the mirrored order.
- **Status**: Open.

## Problem coverage

`tie_break::prefer_larger_index`.

## Output guarantees

Exact mirrored order.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 11.5, 12.2, 14.2, 19.6, 21.6, 23.6.

Owner: TBD
```

### TK-S11.4

- Title: `[CUB] Signed zero and NaN behavior per top-k family`
- Labels: `cub`, `docs`
- Parent: TK-P11
- Dependencies: TK-P11

```markdown
## Summary

- Every family folds minus zero onto plus zero before the twiddle or orders it below (N5's network) with the recovery classifying by IEEE comparison, which the signed-zero battery checks under both requirements. NaN handling differs: twiddled bitwise (W2, B2, W4), by sign bit (N10), unsupported (N6 at warp scope, the all-ones pattern in B4), never selected (N5). A contract table per family for the documentation.
- **Scope**: Every family.
- **Comparison method**: The 23-case battery.
- **Status**: Recorded per section, the battery passed with zero failures on the fp32 threshold and tie adapters (KB 2.10).

## Problem coverage

Correctness contracts.

## Output guarantees

As the table.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.10, the contract subsections of sections 11 to 23, TK-S13.3.

Owner: TBD
```

### TK-S11.5

- Title: `[CUB] The deterministic block primitive in the batched backends`
- Labels: `cub`
- Parent: TK-P11
- Dependencies: TK-P11, TK-S09.3

```markdown
## Summary

- The backend selector of DeviceBatchedTopK sends every deterministic or explicit tie-break request to the cluster backend. B1_det3 is a deterministic block primitive at near zero cost on fp32 and 4 to 24 percent on bf16, so the baseline backend could serve deterministic requests for segments that fit a block.
- **Scope**: `policy_selector_from_types` and the baseline backend.
- **Comparison method**: The segmented top-k benchmarks under the requirements.
- **Status**: Open, depends on TK-S09.3 landing in the header. The block cells: B1_det3 reads at B1_det2's speed in the cross-node addendum (110 prefer_smaller_index cells on umbriel-b200-068, knowledge base 16), no winner changes hands, evidence pending the same-node rerun.

## Problem coverage

Deterministic batched top-k below the cluster threshold.

## Output guarantees

Exact index order among ties.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 1.1, 1.2, 13.6.

Owner: TBD
```

---

### TK-P12

- Title: `[CUB] Key type specializations of the top-k cooperative primitives`
- Labels: `cub`
- Parent: none (cross-cutting)
- Dependencies: TK-P01 to TK-P10

```markdown
## Summary

- Which key width takes which route. bf16 (and fp16) win only through the packed key-plus-inverted-index word, never through the compare itself. fp32 has three routes (the truncated word with an exact tie finish, the keys-only network with ballot rank recovery, the u64 index-in-key word). fp8, fp16 and fp64 are unmeasured in the study and belong to the full evaluation.
- **Scope**: fp8 e4m3, bf16, fp16, fp32, fp64 keys with int32 indices, every family.
- **Comparison method**: The study harness, the phase 3 full evaluation adds the missing types.
- **Status**: Measured (B200, CUDA 13.4.92, 2026-10-25): bf16 is not cheaper than fp32 for the radix select and slower in every warp bitonic cell with the native compare (reference cards), cheaper by 60 to 115 cycles for W2 and 270 to 670 for B2 through the word, N6's bracket makes the two types cost the same. Full evaluation `<phase 3>`.

## Problem coverage

The type dispatch of every primitive.

### Segment / input size
- **Supported range**: every N of the families above.
- **Expected sweet spot**: not applicable, a dispatch table.

### k
- **Supported range**: every k of the families above.
- **Expected sweet spot**: not applicable.

### Types
- **Supported**: fp8 e4m3, bf16, fp16, fp32, fp64 keys with int32 indices, as the family allows.
- **Expected sweet spot**: 16-bit keys through the packed word, fp32 through its three routes.

## Output guarantees

### Determinism (set membership)
| `determinism` | Supported | Notes |
|---|---|---|
| `not_guaranteed` | yes | Every family. |
| `run_to_run` | per family | As the family. |
| `gpu_to_gpu` | per family | As the family. |

### Tie-break (only meaningful with `gpu_to_gpu`)
| `tie_break` | Supported | Notes |
|---|---|---|
| `unspecified` | yes | As the family. |
| `prefer_smaller_index` | per family | As the family. |
| `prefer_larger_index` | no | As the family. |

### Output ordering (result sequence)
| `output_ordering` | Supported | Notes |
|---|---|---|
| `unsorted` | yes | Trivially. |
| `sorted` | per family | As the family. |
| `stable_sorted` | per family | As the family. |

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.12 (bf16 bullet), 10.4, 2.10 (T8, T18), the contracts of sections 11 to 23.

Owner: TBD
```

### TK-S12.1

- Title: `[CUB] The bf16 packed key-plus-inverted-index word`
- Labels: `cub`, `docs`
- Parent: TK-P12
- Dependencies: TK-P12

```markdown
## Summary

- The twiddled 16-bit key above the bit-inverted 16-bit index orders like (key descending, index ascending), makes every extraction round an exact arg max and the network's compare a 32-bit min and max, and gives prefer_smaller_index for free. W2, B2, W1, N5 and the N4 truncation rest on it, indices below 65536.
- **Scope**: 16-bit keys in every family.
- **Comparison method**: The cards of the families.
- **Status**: Measured and verified, the single most valuable idea of the study (KB 2.12).

## Problem coverage

Every bf16 and fp16 cell.

## Output guarantees

prefer_smaller_index exact, sorted where the family sorts.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.12, T19, T42.

Owner: TBD
```

### TK-S12.2

- Title: `[CUB] The fp32 routes: truncated word, keys-only network, u64 word`
- Labels: `cub`, `docs`
- Parent: TK-P12
- Dependencies: TK-P12, TK-P04

```markdown
## Summary

- The truncated word (top 32 minus log2 N key bits plus the index, exact tie finish on the next word) for W2 and B2, within 62 to 98 cycles of bf16 at warp scope and 20 to 38 percent cheaper than value rounds with an index epilogue at block scope. The keys-only network with ballot rank recovery (N10) for the W1 family under either requirement. The u64 word for sorted output and N 8.
- **Scope**: fp32 at warp and block scope.
- **Comparison method**: The cards of W2, B2 and N10.
- **Status**: Measured and verified. The 17-bit prefix at N 32768 is the open edge (TK-S06.6).

## Problem coverage

Every fp32 cell.

## Output guarantees

Exact under both requirements on every route.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.12, 12, 14, 19, 11.7.

Owner: TBD
```

### TK-S12.3

- Title: `[CUB] fp8 keys widened to 16-bit lanes`
- Labels: `cub`
- Parent: TK-P12
- Dependencies: TK-P12

```markdown
## Summary

- 8-bit SIMD min and max are a 10-instruction emulation at 28 cycles on sm_100 (native only on the sm_107f and sm_120f families), so fp8 keys widen to 16-bit lanes with prmt and run the u16x2 or bf16x2 network and the packed word (T8). For the radix select one 8-bit pass is the whole selection.
- **Scope**: fp8 e4m3 in every family.
- **Comparison method**: The full evaluation of phase 3.
- **Status**: Unmeasured. `<phase 3>`.

## Problem coverage

fp8 scores.

## Output guarantees

As the 16-bit routes.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.10 (T8), 10.1.

Owner: TBD
```

### TK-S12.4

- Title: `[CUB] fp16 keys`
- Labels: `cub`
- Parent: TK-P12
- Dependencies: TK-P12

```markdown
## Summary

- The bf16 word with a different twiddle, every 16-bit route applies, unmeasured in the study.
- **Scope**: fp16 in every family.
- **Comparison method**: The full evaluation.
- **Status**: Unmeasured. `<phase 3>`.

## Problem coverage

fp16 scores.

## Output guarantees

As bf16.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 10.1, 2.12.

Owner: TBD
```

### TK-S12.5

- Title: `[CUB] fp64 keys`
- Labels: `cub`
- Parent: TK-P12
- Dependencies: TK-P12

```markdown
## Summary

- Untested in the study. The striped block top-k raised f64 keys from 64 to 74 registers at 256 by 4 (one block per SM less), W4 and B6 use 64-bit words already (8 radix passes for the sort), the packed and truncated words do not apply.
- **Scope**: fp64 in the block families and W4.
- **Comparison method**: The full evaluation.
- **Status**: Unmeasured. `<phase 3>`.

## Problem coverage

fp64 scores.

## Output guarantees

Radix and merge routes only.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 4.4, 23, 10.5.

Owner: TBD
```

---

### TK-P13

- Title: `[CUB] Device-side benchmarking and test methodology from the top-k study, items worth upstreaming`
- Labels: `cub`, `bench`
- Parent: none (infrastructure)
- Dependencies: none

```markdown
## Summary

- The harness and the verification protocol of the cooperative primitives study found three node and harness effects and four correctness bugs that random input and whole-call timing never would have. The items below are the parts worth keeping in the repository's benchmark and test infrastructure.
- **Scope**: `bench/`, the CUB tests, the performance CI, and the documentation of how a block or warp primitive is measured.
- **Comparison method**: Not applicable.
- **Status**: All items exist in `dsb_experiments/topk_swarm/harness/` on the fork branch, versions 0.2 to 0.10, with `README.md` section 3.1 as the record of what each version changed and how it moved the numbers. One classification item from phase 3b: an oracle kernel that cannot launch (B6_radix at fp32 N 32768 needs 270 KB of dynamic shared memory against the 227 KB opt-in limit) is counted as oracle_failed where an unlaunchable regime is counted as unsupported, and the two should read the same.

## Problem coverage

Measurement validity and correctness coverage of every top-k primitive.

### Segment / input size
- **Supported range**: not applicable.
- **Expected sweet spot**: not applicable.

### k
- **Supported range**: not applicable.
- **Expected sweet spot**: not applicable.

### Types
- **Supported**: not applicable.
- **Expected sweet spot**: not applicable.

## Output guarantees

### Determinism (set membership)
| `determinism` | Supported | Notes |
|---|---|---|
| `not_guaranteed` | yes | Not applicable, infrastructure. |
| `run_to_run` | TBD |  |
| `gpu_to_gpu` | TBD |  |

### Tie-break (only meaningful with `gpu_to_gpu`)
| `tie_break` | Supported | Notes |
|---|---|---|
| `unspecified` | yes | Not applicable. |
| `prefer_smaller_index` | TBD |  |
| `prefer_larger_index` | TBD | Not applicable. |

### Output ordering (result sequence)
| `output_ordering` | Supported | Notes |
|---|---|---|
| `unsorted` | yes | Trivially. |
| `sorted` | TBD |  |
| `stable_sorted` | TBD |  |

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 10.4, 10.6, 2.10, 9, `harness/README.md`, `cards/README.md`, `reference/DEVICE_SIDE_BENCHMARKING_ISSUE.md`.

Owner: TBD
```

### TK-S13.1

- Title: `[CUB] Device-side benchmarking methodology for block and warp primitives`
- Labels: `cub`, `bench`, `docs`
- Parent: TK-P13
- Dependencies: TK-P13

```markdown
## Summary

- Latency as the chain-length slope of clock64 cycles per call net of a generate-only control launched with the variant's dynamic shared memory, throughput as one occupancy wave with occupancy and registers reported, variant and control in one lock section, a drift canary at both ends, the oracle before any number, loads streamed past L1 with strided input subsets per chain length (harness 0.10), and a version policy for comparing across harness versions.
- **Scope**: A bench utility or a documented pattern for `bench/`.
- **Comparison method**: Not applicable.
- **Status**: Exists in the study harness. The 0.6 control change moved shared-memory-heavy controls by 9 to 11 percent and the 0.10 streaming change removed the chain-length dependence of the control, both recorded.

## Problem coverage

Every microbenchmark of a cooperative primitive.

## Output guarantees

Not applicable.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 10.1, 10.4, 2.10, `harness/README.md` sections 2 and 3.1, `reference/DEVICE_SIDE_BENCHMARKING_ISSUE.md`.

Owner: TBD
```

### TK-S13.2

- Title: `[CUB] An exact sorted-output check in the top-k tests`
- Labels: `cub`, `bench`
- Parent: TK-P13
- Dependencies: TK-P13

```markdown
## Summary

- The harness checked the selected set and the exact index set but not the order of the k slots until verify-W4 added an exact key-descending index-ascending check over all k slots. Every W4 variant and B6 pass it, B3_tuned fails 20 of 44 cases as the negative control. It is the `--check-sorted` flag of harness 0.10.
- **Scope**: Every primitive that claims `output_ordering::sorted`.
- **Comparison method**: Not applicable.
- **Status**: In the harness, not in the CUB tests.

## Problem coverage

Correctness of the sorted contract.

## Output guarantees

Not applicable.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.10, 23.

Owner: TBD
```

### TK-S13.3

- Title: `[CUB] The signed-zero battery in the top-k tests`
- Labels: `cub`, `bench`
- Parent: TK-P13
- Dependencies: TK-P13

```markdown
## Summary

- Five signed_zero seeds varying the share of plus zeros among the 2k zeros, signed_zero_boundary with the k-th rank inside mixed-sign zeros, pivot_tie_zero, 23 battery cases per cell, both requirements in one batch (the set check under not_guaranteed, the exact check under prefer_smaller_index). Added after a threshold primitive passed a single seed with a broken contract (W5_v3).
- **Scope**: Every top-k test with float keys.
- **Comparison method**: Not applicable.
- **Status**: In harness 0.7 and later, passed with zero failures on every fp32 threshold and tie adapter of the study.

## Problem coverage

The signed-zero contract.

## Output guarantees

Not applicable.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.10, `harness/README.md` section 2.

Owner: TBD
```

### TK-S13.4

- Title: `[CUB] Oracle self-tests must not share the degenerate input of the case they test`
- Labels: `cub`, `bench`
- Parent: TK-P13
- Dependencies: TK-P13

```markdown
## Summary

- Harness 0.3 and earlier recorded a false negative of the oracle self-test at k equal to N over 2, because the self-test's signed-zero input then has 2k equal to N zeros and no non-top element for its wrong-set case. The per-cell oracle was intact. A self-test needs its own input that keeps a non-top element at every k.
- **Scope**: Test fixtures.
- **Comparison method**: Not applicable.
- **Status**: Fixed in harness 0.4.

## Problem coverage

Test validity.

## Output guarantees

Not applicable.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.10.

Owner: TBD
```

### TK-S13.5

- Title: `[CUB] Node pre-flight and the outlier rule for performance work`
- Labels: `cub`, `bench`
- Parent: TK-P13
- Dependencies: TK-P13

```markdown
## Summary

- One B200 ran 128-thread kernels above about 64 KB of SASS 1.29x to 2.04x slower than three others from the same binary with every smaller kernel and every synthetic identical to the cycle, hosts identical in every inventory item, and was drained by the cluster for a failed reboot minutes after the last measurement. Rules: name the node on every latency card of such a kernel, reproduce on a second GPU before a number enters a winner table, pre-flight the evaluation node, ratios within one session stay valid.
- **Scope**: The performance CI and any cross-node comparison.
- **Comparison method**: Not applicable.
- **Status**: Recorded as T58, a pre-flight script exists in the study tree. Open item from the phase 3 pre-flights: both GB200 hosts (gb200-nvl4-ts2-99 and ts2-105, within 1.6 percent of each other on 25 cells) run the 128-thread k 64 configuration with 32 to 64 items per thread 3 to 15 percent slower than the healthy B200s while every other 128-thread cell matches within 0.5 percent, so this is a generation property of one warp per scheduler with a long register-resident list and not a node or code size, the same configuration that deviated on umbriel-b200-020. The mechanism is open, the GB200 pass values stand as measured.

## Problem coverage

Validity of latency numbers across GPUs and across the B200 and GB200 generations.

## Output guarantees

Not applicable.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.10, 10.6, T50, T58, `harness/` pre-flight script.

Owner: TBD
```

### TK-S13.6

- Title: `[CUB] Phase stamps between barriers as the profiling method for block primitives`
- Labels: `cub`, `bench`, `docs`
- Parent: TK-P13
- Dependencies: TK-P13

```markdown
## Summary

- Thread 0 writes clock64 between the phases of a primitive behind a macro that compiles to nothing otherwise, median over calls on one block, the same rows for every variant. It found every structural cost of B4 (per-slot branches, scans and barriers, tickets with return), N3 (rolled sweeps, per-slot atomics) and the deterministic AIR path (the tie phase per slot) that the instruction count and the whole-call time hid.
- **Scope**: A documented pattern for primitive authors.
- **Comparison method**: Not applicable.
- **Status**: Three profilers exist in the study tree (`approaches/B4/profile_phases.cu`, `approaches/N3/profile_phases.cu`, `approaches/B1/profile_det3_phases.cu`).

## Problem coverage

Attribution inside a call.

## Output guarantees

Not applicable.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 2.10, T56.

Owner: TBD
```

### TK-S13.7

- Title: `[CUB] A harness version policy for numbers compared across versions`
- Labels: `cub`, `bench`, `docs`
- Parent: TK-P13
- Dependencies: TK-P13

```markdown
## Summary

- A card is judged under the harness version it was measured with, because variant and control shared one build and one lock section. When a later version changes a control's net number, the verifier records the current-version ratios next to the card's and the card stays verified if the conclusions hold under both. No comparisons across versions, the final evaluation re-measures everything under one version. The table of version effects is the reference for verifiers.
- **Scope**: Any benchmark whose control can change between versions.
- **Comparison method**: Not applicable.
- **Status**: Policy in `cards/README.md`, the version table in `harness/README.md` section 3.1.

## Problem coverage

Comparability of results over time.

## Output guarantees

Not applicable.

## Pointers

Fork branch https://github.com/elstehle/cccl/tree/exp/topk-primitives-swarm/dsb_experiments/topk_swarm (paths below are relative to it). KB 9, 2.10.

Owner: TBD
```

---

## Closing notes for the agent

- Post the A.3 comments after the parents exist and link each parent by number.
- After every sub-issue creation run the linking step, then verify with `gh api repos/NVIDIA/cccl/issues/<parent number>/sub_issues --jq 'length'` that the count matches the tree.
- Do not open issues for the A.4 lines, they are paragraphs in the parent bodies if the user wants them visible (append a "Not pursued" section to the parent with the A.4 line, nothing else).
- Report the created numbers back as a table (TK id, number, title) and stop.

## Phase 3 placeholder sources

Every `<phase 3>` in this file and where its number comes from, so the fill is mechanical when the passes land. Files: `WINNERS_b200.md` (all three scopes at 43ce95b3a9), `WINNERS_gb200.md` (the GB200 pass), the phase 3b sorted-output tables (appended to the winners files by `tools/winners.py` after the 3b pass), `state/close_cells_b200.json` and the tie_heavy rerun (a second input distribution read as robustness, not a tie-breaker), and `state/phase3/winners_b200.json`, whose rows carry `winner`, `value`, `runner_up`, `margin_runner_up`, `reference`, `margin_reference`, `reference2` and `margin_reference2` (the deterministic cells) and an `all` list with every adapter's value in the cell. "Available" means the block pass of 2026-10-25 21:15 UTC already holds the number, "pending" names the pass that will. A close call (margin over the runner-up below 15 percent) stays marked: nothing settles it by speed, and the choice falls to the tie_heavy flip list, registers and spills, the deterministic cost and code size, as knowledge base 16 defines.

| Placeholder | Where | Source file | Matrix or table | Cells or column | Status |
|---|---|---|---|---|---|
| header sentence | Part A, first paragraph | none | the instruction that the placeholders exist | not a fill target | remove the sentence after the fill |
| pre-flight step 1 | Part B.0 | none | the instruction to fill | not a fill target | keep |
| #9747 comment | Part A.3, the winner matrix pointer | `WINNERS_b200.md`, `WINNERS_gb200.md` | the file paths on the fork branch | one sentence naming both files and the node, harness and date | filled |
| TK-P01 table row and body | A.2, B TK-P01 "Final winner cells and margins" | `WINNERS_b200.md` | warp latency and throughput, both requirements | the cells W2_packed_v3 wins (expected k 4 to 16 at every N and the N 512 latency cells), its margins over the runner-up and W1_reference | filled from the warp pass (43ce95b3a9), close calls marked |
| TK-S01.2 | B "Final cells" | `WINNERS_b200.md` | sub-warp latency and throughput | the logical warp 4, 8, 16 cells, W2_packed_sw_v2 against W1_baseline_sub, the router cell logical warp 8 N 128 k 8 | filled from the sub-warp pass, the router cell pending the lane-count addendum |
| TK-S01.5 | B "carries the GB200 pass" | `WINNERS_gb200.md` | warp latency and throughput | the W2 cells, with the B200 values beside them | filled from the GB200 pass |
| TK-P02 table row and body | A.2, B TK-P02 "Final cells" | `WINNERS_b200.md` | warp latency and throughput, both requirements | bf16 N 1024 and 2048 k 16 to 64, fp32 N 1024 k 32 to 64 and N 2048 k 16 to 64, W5_v4 against W2_packed_v3 and N6w_atom32 | filled from the warp pass |
| TK-P03 table row and body | A.2, B TK-P03 "Final cells" | `WINNERS_b200.md` | warp and sub-warp latency, bf16 | N 32 to 64 at k 16 to 32, bf16 N 128 and 256 k 64 (W1x_vimnmx_eager, W1_level), the sub-warp N 8 cells | filled from the warp and sub-warp passes |
| TK-S03.2 | B "Final cells" | `WINNERS_b200.md` | sub-warp latency and throughput | the W1 sub-warp adapters against W1_baseline_sub, every logical warp | filled from the sub-warp pass |
| TK-S03.4 | B "Final cells" | `WINNERS_b200.md` | warp throughput, bf16 | N5_rows_f (throughput only by design) against W1_reference and W2_packed_v3 at N 32 to 256, k 4 to 64 | filled from the warp pass |
| TK-P04 table row and body | A.2, B TK-P04 "Final cells" | `WINNERS_b200.md` | warp and sub-warp latency and throughput, prefer_smaller_index and not_guaranteed, fp32 | N 32 to 256 at warp scope, N 32 to 128 at sub-warp scope, N10_auto and N10_auto_sub, margins over the runner-up and the references | filled from the warp and sub-warp passes, the router shape pending the lane-count addendum |
| TK-P05 table row and body | A.2, B TK-P05 "Final cells" | `WINNERS_b200.md` | warp latency and throughput (N6w_atom32), block latency and throughput (N9_w2fin2) | warp k 64 to 1024 at N 256 to 2048, block N 1024 to 4096 k 24 to 256 (available: N9_w2fin2 wins 13 not_guaranteed and 6 deterministic latency cells, many of them close calls) | filled from both passes |
| TK-S05.5 | B "Final cells" | `WINNERS_b200.md`, `winners_b200.json` | block latency, both requirements, N 1024 and 4096 | the N9_w2fin2 cells at k 24 to 256 with the margins over B1_idx, B1_n6b, B1_det2 and B1_det (the `all` list gives B1_n6b's value where it is not the runner-up) | filled |
| TK-P06 table row and body | A.2, B TK-P06 "Final cells" | `WINNERS_b200.md` | block latency and throughput, both requirements | N8_b2packed@T128 at N 1024 to 8192 k 4 to 32 and B2_twolevel at N 4096 to 32768 k 4 to 16, margins over the runner-up and B1_reference@T128 or B1_reference | filled |
| TK-P07 table row and body | A.2, B TK-P07 "Final cells" | `WINNERS_b200.md` | block latency and throughput, both requirements | B4_v4 at N 8192 to 32768 k 32 to 1024 (32 of 77 and 32 of 75 latency cells, 24 and 27 throughput cells), the medians plus 160 and plus 52 percent, the k 2048 and N 4096 cells it does not win | filled |
| TK-S07.9 | B "Status" | `winners_b200.json` | block latency, the `all` lists | B4_v4 against B1_idx@T512 at N 16384 and 32768 k 64 to 256 and against B2_twolevel at k 16 (B2_twolevel wins k 16 at N 16384 and 32768 under both key types, B4_v4 is the runner-up at N 32768) | filled |
| TK-P08 table row and body | A.2, B TK-P08 "Final cells" | `WINNERS_b200.md` | block latency and throughput, both requirements | B3_tuned at k 1024 to 4096 (11 not_guaranteed and 15 deterministic latency cells, 13 deterministic throughput cells), margins over B1_idx@T512 and B4_v4 | filled |
| TK-S08.4 | B "Status" | `winners_b200.json` | block latency, the `all` lists | B3_tuned against B4_v4 at k 1024 and 4096 and against B1_idx@T512 at k 256 to 4096, N 4096 to 32768 | filled |
| TK-S08.5 | B "Status" | phase 3b tables | sorted output, block, one unsorted tile | B3_tuned plus the post-sort against B6_radix and W4 at N 2048 to 8192 k equal to N over 4 and N over 2 | pending phase 3b |
| TK-P09 table row and body | A.2, B TK-P09 "Final cells" | `WINNERS_b200.md`, the B1_det3 addendum | block latency and throughput, both requirements | B1_idx (20 throughput cells) and B1_idx@T512 (9) at N 4096 to 16384, B1_det2 (11 and 16 throughput cells, 2 latency cells), the B1_det margins of the deterministic cells, then the B1_det3 cells when the addendum lands | filled, det3 from the cross-node addendum pending the same-node rerun |
| TK-P10 table row and body | A.2, B TK-P10 "Final cells" | `WINNERS_b200.md`, phase 3b tables | the sorted-runs table (block, W4_chunk4 at runs 4 and W4_chunk8 at runs 8, N 8192 to 32768 k 256 to 4096 against B1_reference, available, one loss at W4_chunk8 fp32 N 8192 k 4096), the warp sorted-output table (W4_wmerge against W2 and W5_v4, pending) | 18 runs 4 and 16 runs 8 cells per regime and requirement, the warp sorted cells at N 512 to 2048 k 64 to 1024 | filled |
| TK-P11 table row and body | A.2, B TK-P11 "Final cells" | `WINNERS_b200.md`, phase 3b tables | the prefer_smaller_index matrices (block available, warp and sub-warp pending), the sorted-output tables | per family the deterministic winner and its margin over B1_det (`margin_reference2`), the sorted routes per scope | filled |
| TK-S11.1 | B "Status" | `winners_b200.json` and the warp pass | the prefer_smaller_index and not_guaranteed matrices | per family the ratio of its deterministic cell to its not_guaranteed cell (the `all` lists give both values of one adapter), block available | filled, det3 cross-node |
| TK-S11.2 | B "Status" | phase 3b tables | sorted output, block and warp | the cheapest sorted route per cell with the margins | filled from the 3b pass |
| TK-S11.5 | B "Status" | `WINNERS_b200.md`, the B1_det3 addendum | block latency and throughput, prefer_smaller_index | the deterministic block cells and B1_det3's place in them | filled from the cross-node addendum |
| TK-P12 table row and body | A.2, B TK-P12 "Full evaluation" | the type extension of the evaluation (fp8 e4m3, fp16, fp64, pair output, knowledge base 10.1) | per type the winner per cell | the fp8, fp16 and fp64 columns | pending, not in the phase 3 plan files yet |
| TK-S12.3, TK-S12.4, TK-S12.5 | B "Status" | the type extension | per type | the fp8, fp16 and fp64 cells of every family that supports the type | pending the type extension |
| tie_heavy outcome | every "p" cell quoted above | the tie_heavy rerun of `state/close_cells_b200.json` | the close-cell tables | 264 close calls (block 144, warp 97, sub-warp 23) stay marked, the tie_heavy flip list of knowledge base 16 is their first secondary column | filled with the tie_heavy pass |
