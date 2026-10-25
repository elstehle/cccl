# W1x: what makes opt-W1's level network faster than CUB's on the same packed bf16 word

Author exp-W1codegen, 2026-10-25. One hour experiment on the W1 contradiction (ideas/PRIORITIES.md
section 2): opt-W1 measured its level network 1.5x to 2.55x faster than cub::detail::WarpBitonicTopK
on the same packed bf16 word (GB200, cards N2_level_gb200), opt-W1b measured the level order against
the eager order with identical network code at 0.95x to 1.08x (B200, card W1_level_same, idea N2
rejected as R12). Card: cards/W1x_codegen_b200_20261025_1.json. Catalogue: T42.

## 1. Kernels

All on the bf16 u32 word of approaches/W1/w1_common.cuh (twiddled key high, inverted index low),
warp scope, striped registers, k indices out. Three factors, one changed at a time:

| kernel | code structure | exchange | tile order |
|---|---|---|---|
| (a) W1_packed | cub::detail::WarpBitonicTopK (static_for lambdas, bool comparator) | CUB: two compares, PLOP3, select | eager |
| (e) W1x_cub_eager | opt-W1 template recursion (warp_level_topk.cuh, eager_topk) | CUB form (cub_ops) | eager |
| (b) W1x_cub_level | opt-W1 template recursion (level_topk) | CUB form (cub_ops) | level |
| (c) W1x_vimnmx_eager | opt-W1 template recursion (eager_topk) | min and max (word_ops) | eager |
| (d) W1_level | opt-W1 template recursion (level_topk) | min and max (word_ops) | level |
| W1_eager_same | opt-W1b (w1b_detail.cuh, unrolled loops) | one compare, select | eager |

(b), (c) and (e) are new. They come from two small parameterizations of opt-W1's source: an Ops
policy on level_topk (word_ops is the original, cub_ops writes the three exchange forms of the CUB
network on the word) and an eager driver (eager_topk) that reuses level_topk's stage machinery in
the source order of the CUB loop (sort tile 0, then per tile: sort, keep larger, merge). (e) was
added because (a) and (b) differ in two things (order and code structure), so (e) is the control
that isolates the code structure: same algorithm and same exchange form as (a), opt-W1's code.
The recursion was not flattened, the factor under test is the recursion family against the CUB
header, and (e) measures exactly that. W1_eager_same was added as the same-session link to
opt-W1b's family.

## 2. Method

B200 umbriel-b200-047 (x86 docker, sm_100, nvcc 13.4.92), one lease, five consecutive run.py lock
sections, each kernel against W1_packed as the control in its own section. W1_packed measured
identically to the cycle in all five sections (1205, 1267, 2417, 2965) and the canaries were
32.001 cycles per op and 3778 to 3780 us in every section, so the sections are one session.
bf16, not_guaranteed, random input, N 128 and 256, k 32 (MaxK 32, one word per tile) and k 64
(MaxK 64, two words per tile), latency and throughput, oracle battery (13 cases) passed on every
cell for every kernel before any number. Batches under state/results/exp-W1codegen/. GB200
gb200-nvl4-ts2-99 (aarch64 enroot, same nvcc) got the same build commands for (a), (c), (d), (e)
in a second lease (results under state/results/exp-W1codegen/gb200/).

## 3. Numbers (B200, cycles per call, control W1_packed in every section)

| cell | (a) W1_packed | (e) cub_eager | (b) cub_level | (c) vimnmx_eager | (d) W1_level | W1_eager_same |
|---|---|---|---|---|---|---|
| N 128 k 32 | 1205 | 1173 | 1016 | 1017 | 883 | 1211 |
| N 128 k 64 | 1267 | 1127 | 1076 | 912 | 916 | 1338 |
| N 256 k 32 | 2417 | 2121 | 1528 | 1718 | 1115 | 2371 |
| N 256 k 64 | 2965 | 2104 | 1801 | 1322 | 1419 | 2800 |

Throughput (Gelem per s, 256 threads, occupancy 8 blocks per SM for every kernel):

| cell | (a) | (e) | (b) | (c) | (d) | W1_eager_same |
|---|---|---|---|---|---|---|
| N 128 k 32 | 282 | 308 | 317 | 358 | 344 | 268 |
| N 128 k 64 | 249 | 264 | 266 | 303 | 303 | 236 |
| N 256 k 32 | 285 | 296 | 306 | 368 | 370 | 263 |
| N 256 k 64 | 230 | 227 | 223 | 249 | 251 | 223 |

SASS of the latency kernel (two network copies per kernel, cuobjdump -sass, histogram files
w1x_sass_b200.txt and w1x_sass_gb200.txt), B200 build:

| cell | kernel | instructions | SHFL | VIMNMX | ISETP | PLOP3 | SEL | LOP3 | registers |
|---|---|---|---|---|---|---|---|---|---|
| N 128 k 32 | (a) | 736 | 150 | 6 | 255 | 20 | 157 | 34 | 27 |
| | (e) | 792 | 150 | 6 | 281 | 0 | 157 | 80 | 41 |
| | (b) | 768 | 150 | 6 | 256 | 0 | 157 | 80 | 41 |
| | (c) | 632 | 150 | 306 | 24 | 0 | 7 | 40 | 40 |
| | (d) | 632 | 150 | 306 | 24 | 0 | 7 | 40 | 40 |
| | W1_eager_same | 728 | 150 | 6 | 230 | 40 | 162 | 27 | 28 |
| N 128 k 64 | (a) | 832 | 180 | 16 | 296 | 10 | 187 | 26 | 31 |
| | (e) | 928 | 180 | 16 | 295 | 0 | 187 | 104 | 38 |
| | (b) | 928 | 180 | 16 | 294 | 0 | 187 | 103 | 39 |
| | (c) | 752 | 180 | 376 | 33 | 0 | 7 | 40 | 41 |
| | (d) | 752 | 180 | 376 | 33 | 0 | 7 | 40 | 41 |
| N 256 k 32 | (a) | 1384 | 310 | 14 | 529 | 20 | 321 | 51 | 36 |
| | (e) | 1488 | 310 | 14 | 576 | 0 | 321 | 120 | 54 |
| | (b) | 1416 | 310 | 14 | 509 | 0 | 321 | 120 | 57 |
| | (c) | 1152 | 310 | 634 | 28 | 0 | 11 | 44 | 54 |
| | (d) | 1152 | 310 | 634 | 28 | 0 | 11 | 44 | 56 |
| N 256 k 64 | (a) | 1616 | 380 | 40 | 615 | 10 | 391 | 41 | 39 |
| | (e) | 1800 | 380 | 40 | 593 | 0 | 391 | 193 | 50 |
| | (b) | 1776 | 380 | 40 | 591 | 0 | 391 | 184 | 51 |
| | (c) | 1408 | 380 | 800 | 37 | 0 | 11 | 44 | 56 |
| | (d) | 1408 | 380 | 800 | 40 | 0 | 11 | 45 | 56 |

BSSY 0, BRA 5 (the harness's), STL and LDL 0, stack 0 in every kernel. IMNMX, FSETP and FSEL
do not occur. Throughput kernels use 27 to 32 registers for every kernel.

## 4. Attribution (latency ratios, B200)

| factor, held fixed | N 128 k 32 | N 128 k 64 | N 256 k 32 | N 256 k 64 |
|---|---|---|---|---|
| exchange, eager recursion (e over c) | 1.15x | 1.24x | 1.23x | 1.59x |
| exchange, level recursion (b over d) | 1.15x | 1.17x | 1.37x | 1.27x |
| tile order, CUB exchange (e over b) | 1.15x | 1.05x | 1.39x | 1.17x |
| tile order, min max exchange (c over d) | 1.15x | 1.00x | 1.54x | 0.93x |
| code structure, same algorithm and exchange (a over e) | 1.03x | 1.12x | 1.14x | 1.41x |
| all three (a over d) | 1.36x | 1.38x | 2.17x | 2.09x |
| exchange and structure only (a over c) | 1.18x | 1.39x | 1.41x | 2.24x |

The factors multiply within a few percent (N 256 k 32: 1.14 x 1.39 x 1.37 = 2.17, N 128 k 32:
1.03 x 1.15 x 1.15 = 1.36).

Throughput ratios: exchange (e over c) 1.17x, 1.15x, 1.24x, 1.10x, tile order (c over d) 0.96x,
1.00x, 1.01x, 1.01x, code structure (a over e) 1.09x, 1.06x, 1.04x, 0.99x. In throughput the
exchange is the whole gain and the order is nothing, which fits an ALU-issue-bound kernel: the min
max form removes 1.7 ISETP and 1 SEL per stage and adds 2 VIMNMX, 104 to 160 instructions fewer
per network copy.

What each factor is:

- Exchange (T42). keep_max ? max(k, other) : min(k, other) compiles to two predicated VIMNMX.U32,
  one 6-cycle step after the shuffle. The CUB comparator form compiles to ISETP, a second ISETP
  with the first as predicate input, then SEL, three dependent steps. Per stage that is about
  36 against 44 cycles on the critical path, and the ratio grows with the number of dependent
  stages (1.15x at 30 stages, 1.59x at 39 stages with two words per tile, where the in-register
  exchange also goes from compare and two selects to one min and one max).
- Tile order. (c) and (d) have identical opcode histograms in every cell, so (c over d) is a pure
  scheduling effect of the source order. The level order pays only with one word per tile (M 1,
  k 32): 1.15x at four tiles and 1.54x at eight, close to the dependency-depth model (30 against
  25 dependent stages at N 128, 50 against 30 at N 256). With two words per tile (M 2, k 64) it is
  1.00x at two tiles (the same DAG) and 0.93x at four tiles, a schedule loss of the T28 kind.
  The same reorder in opt-W1b's code (W1_level_same against W1_eager_same) measured 0.95x to 1.08x,
  so R12's "the level order gains nothing" was a property of that implementation (phase 1 written
  tile by tile through pointer offset loops) and not of the order: in the stage-interleaved
  recursion the order gains 1.15x to 1.54x at M 1 in latency, and nothing in throughput.
- Code structure. (a) and (e) run the same algorithm with the same exchange expressions, (e) in
  template recursion with every array index a compile-time constant and the direction bits as
  lane functions, (a) through the CUB header (static_for lambdas, pointer offsets, a bool
  comparator evaluated twice per stage, group_reverse built per stage from lane shifts). (e) has
  more instructions (792 against 736 at N 128 k 32, LOP3 80 against 34) yet is 1.03x to 1.41x
  faster, so this factor is the schedule, not the count. W1_eager_same (opt-W1b's eager loop, one
  compare and select per stage) sits with (a) in every cell (1211, 1338, 2371, 2800 against 1205,
  1267, 2417, 2965).

## 5. GB200 against B200

Same build commands on the aarch64 host (T20). (c), (d) and (e) compile to the same SASS (same
instruction counts and histograms) and measure the same cycles to the cycle in every cell. (a),
the CUB header, does not: 1521, 1598, 2840, 3539 cycles against 1205, 1267, 2417, 2965 on B200
(1.26x, 1.26x, 1.17x, 1.19x slower), 792 against 736 instructions at N 128 k 32 (ISETP 345 against
255, PLOP3 5 against 20), 920 against 832, 1488 against 1384, 1768 against 1616, BRA 5 on both
(no branches, the aarch64 cicc produces a longer predicate chain instead). Throughput of (a) 264,
233, 272, 227 against 282, 249, 285, 230 Gelem per s. This is why opt-W1's GB200 ratios
(W1_level over W1_packed 1.72x, 1.74x, 2.55x, 2.49x) exceed the B200 ratios of the same pair
(1.36x, 1.38x, 2.17x, 2.09x): the remainder is the toolchain penalty on the CUB header path, and
it is 1.17x to 1.26x here against the 3 to 4 percent T20 measured on the fp32 pairs reference. The
explicitly written exchanges (both the min max form and the written-out compare and select form)
are host independent.

## 6. Resolution

Both W1 instances were right about what they held fixed. opt-W1's 1.5x to 2.55x over CUB on the
same word is the product of three factors, the min max exchange (1.15x to 1.6x), the code
structure of the recursion against the CUB header (1.03x to 1.4x on B200, 1.3x to 1.7x on an
aarch64 build), and the level order (1.15x to 1.54x only at one word per tile, 0.93x to 1.0x at
two). opt-W1b changed the order alone inside an implementation whose phase 1 did not expose the
overlap, and saw none of the three. The prediction in PRIORITIES.md (exchange and code structure,
not the level order) holds for throughput exactly and for latency at M 2. At M 1 in latency the
order is a real third lever on top of the exchange, worth 1.15x at four tiles and 1.54x at eight.

## 7. What the W1 bf16 route should be

- Adopt the min max exchange on the packed u32 word (T42) in whichever loop W1 ships. It is the
  only factor that pays in throughput (1.10x to 1.24x) and it pays 1.15x to 1.6x in latency.
- Write the network in the recursion form (every index a compile-time constant, direction bits
  as lane functions, the exchange written out, not a bool comparator), T34, T36 and this card:
  1.03x to 1.41x over the CUB header on B200 with the same algorithm, and the same code on both
  hosts. For the CUB primitive itself this means a min max customization point next to the
  comparator (the comparator form cannot express it).
- Loop: the eager loop with the min max exchange (c) is the one-code-path answer, 1.18x to 2.24x
  over W1_packed on B200 with the same registers as the level form. Where k is at most 32
  (one word per tile) and N is at least 4 MaxK and latency matters, the level order adds 1.15x
  (N 128) to 1.54x (N 256) on top at the same throughput, which is the case for reopening R12 for
  M 1 latency cells only. At k 64 the eager loop is the better one (912 against 916, 1322 against
  1419).
- The GB200 cards of W1 (N2_level_gb200) overstate the gap over CUB by 1.17x to 1.26x because of
  the aarch64 build of the control. The B200 numbers here are the ones to quote against CUB.

## 8. Open questions

- Why the level order loses 7 percent at M 2 with four tiles (1419 against 1322) when its
  dependent chain is shorter (33 against 39 stage slots). A schedule dump comparison (T28) or a
  perturbation of the source order of the in-register stages would settle it. Not done in the
  hour.
- Whether opt-W1b's level loop can be made to show the M 1 order gain by interleaving phase 1
  stage by stage in source order (their sort_all sorts tile by tile). If yes, R12's evidence base
  changes to "implementation", if no, something else in w1b_detail.cuh hides it.
- The fp32 u64 word was not in scope. Its exchange is compare and select in both families (no
  64-bit VIMNMX), so the exchange factor is zero there and the remaining gap of opt-W1's fp32
  cards (2.1x to 2.65x on GB200) must be code structure, order and the aarch64 control. That is a
  separate one-hour run with the same adapters and keys fp32 after a cub_ops<uint64_t> check.
- The prefer_smaller_index cells were not re-run (same code on these words, the earlier cards show
  identical cycles).

## 9. Pointers

- Code: worktree /Users/estehle/Downloads/cccl-topk-swarm-W1x, branch exp/topk-swarm/W1x.
  approaches/W1/w1_common.cuh (cub_ops), warp_level_topk.cuh (Ops parameter, eager_topk),
  w1_adapters.cuh (Ops and Eager parameters on level_packed_adapter, rec_net),
  harness/adapters/W1x_cub_level.cuh, W1x_vimnmx_eager.cuh, W1x_cub_eager.cuh, w1x_node.sh,
  w1x_sass_hist.awk, w1x_sass_b200.txt, w1x_sass_gb200.txt, findings_W1x_codegen_b200.json,
  tech_minmax_exchange.md (T42 as appended).
- Batches (state/results/exp-W1codegen/): W1_level_vs_W1_packed_focus_codegen_20261025_144714.json,
  W1x_vimnmx_eager_vs_W1_packed_focus_codegen_20261025_144735.json,
  W1x_cub_level_vs_W1_packed_focus_codegen_20261025_144753.json,
  W1x_cub_eager_vs_W1_packed_focus_codegen_20261025_144803.json,
  W1_eager_same_vs_W1_packed_focus_codegen_20261025_144813.json, and under gb200/ the three GB200
  batches (W1_level 145046, W1x_vimnmx_eager 145053, W1x_cub_eager 145100).
- Card: cards/W1x_codegen_b200_20261025_1.json (built from the W1x_vimnmx_eager batch, the
  kernel that did not exist before, with the whole attribution in its findings, make_card.py
  takes one adapter per card).
- Related: REPORT.md (opt-W1), REPORT_W1b.md (opt-W1b), ideas/rejected/R12_N2_level_parallel_bitonic.md,
  ideas/PRIORITIES.md section 2, TECHNIQUES.md T7, T20, T28, T34, T35, T36, T42.
