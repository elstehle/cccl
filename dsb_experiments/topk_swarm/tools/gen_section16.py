"""Regenerates knowledge base section 16 (the winners of the phase 3 pass) from the winners JSON.

Inputs: the winners file written by tools/winners.py (--winners, default state/phase3/winners_b200.json
next to this tools directory) and the knowledge base markdown (--kb), from which the phase 3b card
tables at the end of the section are carried over until the phase 3b pass lands.
The GB200 winners file (--winners-gb200) adds the GB200 matrices and the comparison tables when present.
Only the random patterns enter the winner matrices. The tie_heavy rows render as their own matrices plus
a flip list against the random winners once their prose is written, and the phase 3b sorted-output rows
(SORTED_PATTERNS, set when that pass lands) replace the card tables at the end of the section.
Output: the section text from its heading line through the line before section 17 (--out, default
stdout), plus a tally of cells, head-to-head cells, close calls and winners per scope on stderr.
The tables are generated, the paragraphs that read them are the PROSE strings below and are edited by
hand when a pass lands. The knowledge base is spliced by hand from the output.
"""
import argparse, collections, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--winners', default=os.path.join(HERE, '..', 'state', 'phase3', 'winners_b200.json'))
ap.add_argument('--kb', default=os.path.join(HERE, '..', 'KNOWLEDGE_BASE.md'))
ap.add_argument('--winners-gb200', default=os.path.join(HERE, '..', 'state', 'phase3', 'winners_gb200.json'))
ap.add_argument('--out', default='-')
ap.add_argument('--winners-addendum', default='', help='a reduction that includes the random-label batches measured on another node, rendered as addendum tables with the node named')
ap.add_argument('--results-3b', default=os.path.join(HERE, '..', 'state', 'results', 'eval-engineer', 'phase3', 'b200'))
args = ap.parse_args()
JSON, KB, OUT, JSON_GB = args.winners, args.kb, args.out, args.winners_gb200
RND = ('random', 'random@runs4', 'random@runs8')
KS = [2, 4, 8, 10, 12, 16, 22, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096]
NS = {'block': [1024, 2048, 4096, 8192, 16384, 32768],
      'warp': [32, 64, 128, 256, 512, 1024, 2048, 4096],
      'subwarp': [8, 16, 32, 64, 128]}
LANES = [4, 8, 16]
PROV = 0.15

# prose that depends on the pass reports, filled by hand when a pass lands
PROSE = {
    'warp_header': 'Warp scope (the second lease session on the same node, measured 20:50 UTC, 57 batches with numbers across the warp and sub-warp scopes and 21 empty by construction since the named cells carry only `not_guaranteed` and the sub-warp scope has no flip bands, {cells} cells all head-to-head, 9 adapters, zero oracle failures, `W1_reference` the control in every section, committed at 43ce95b3a9). Close calls: {prov} of {cells} cells. N 32 to 4096 from the focus grid, N 256 to 2048 from the flip bands, the named cells add N 64, 160, 384, 768 and 896. `N5_rows_f` runs in the throughput regime only by design (two rows per call, section 20). `W1_reference` costs 1.3x (fp32 N 32 k 16) to 65x (bf16 N 4096 k 4) the winner\'s latency, median 9x and the middle half 5.5x to 18x, as its linear cost per element predicts, so the `ref` margins at this scope are large everywhere and the runner-up margin is the one that ranks.',
    'subwarp_header': 'Sub-warp scope (the same session, {cells} cells all head-to-head, 4 adapters at 4, 8 and 16 lanes, `W1_baseline_sub` the control in every section). The lane count is a tuning dimension chosen per cell like the block size at block scope: every adapter ran at every lane count the cell allows (lanes at most N) and the label carries the winner\'s lane count as `@T4`, `@T8` or `@T16`, the runner-up is often the same adapter at another lane count. Close calls: {prov} of {cells} cells. N 8, 32 and 128 from the focus grid at k 4 to 64, the named cells add N 16 and 64 and k 2. `N5_rows_fs_sub` runs in the throughput regime only by design. The named sub-warp cells ran at 4 lanes only (their per-cell lists hold no other lane count), while 10.2 specifies logical warps of 8 and 16 for Mixtral, DBRX, gpt-oss and the DeepSeek group select, so those rows are the 4-lane answer and not the shape\'s, an addendum at the specified lane counts on both arches is pending with the evaluation engineer and the rows stay marked until the matrices regenerate.',
    'warp_reading': 'What the warp pass says. W2_packed_v3 takes 42 of 89 `not_guaranteed` latency cells (median +825 percent over W1_reference and +61 percent over the runner-up): every k 4 to 32 cell at N 32 to 512 except fp32 N 32 k 16 (N10_auto, a close call) and fp32 N 512 k 32 (N6w_atom32), bf16 N 1024 k 8 and fp32 N 1024 k 16 (both close calls, the fp32 one a 1.2 percent tie with W5_v4), bf16 N 2048 k 4 (a close call) and bf16 N 4096 k 4, and every MoE router cell of the named table. W5_v4 takes 21 (median +2320 percent, +51 percent): bf16 N 1024 k 16 to 64 (k 16 a close call), fp32 N 1024 k 8, 24 and 32 (k 32 a close call), k 16 to 64 at N 2048 and 4096 under both key types, and fp32 k 4 at N 2048 and 4096 (the latter a close call). N6w_atom32 takes 20 (median +454 percent, +21 percent): k 48 and 64 at N 256 (bf16 and fp32 k 64 close calls), N 512 from k 48 (bf16) or k 32 (fp32) up to k 256, fp32 N 1024 k 48 (a close call) and 64, fp32 N 2048 k 256, and k 256 and 1024 at N 4096 under both key types (fp32 k 1024 a close call against W3_d4c at +13 percent). W3_d4c takes k 1024 at N 2048 under both key types (close calls against N6w_atom32), W4_wmerge bf16 N 2048 k 256, N10_auto fp32 N 32 k 16 and N 128 k 64 (both close calls, W2 takes the other fp32 small-N cells N10 held on the cards), W1x_vimnmx_eager bf16 N 128 k 64 (a close call). `prefer_smaller_index` repeats the ranking: every warp adapter runs the same code under both requirements (the deterministic over unsorted ratio per adapter between 0.998 and 1.002 on the 70 cell pairs), so the 70 cells differ from the `not_guaranteed` matrix only by what was measured, W2_packed_v3 26, W5_v4 20, N6w_atom32 18. Throughput: W5_v4 takes 34 of 87 (median +674 percent, +43 percent), every N 1024 to 4096 cell from k 4 to 64 (bf16 and fp32 N 1024 k 64 and N 4096 k 4 close calls), bf16 N 512 k 16 and 24 (close calls) and fp32 N 512 k 4 and 16, W2_packed_v3 21 at k 4 to 16 up to N 512 where N5_rows_f and N10_auto do not take the cell, N6w_atom32 15 (fp32 from k 24 at N 256 and 512 and N 128 k 64, bf16 N 512 k 48 and 64 close calls, N 2048 k 256 under both key types), N5_rows_f 12 bf16 cells (N 32 k 4 and 16, N 128 k 16 a close call and k 64, N 256 k 24 to 64 all close calls, N 512 k 32 and 256 close calls, N 4096 k 256 and 1024 close calls), W3_d4c fp32 k 1024 at N 2048 and 4096, N10_auto fp32 N 32 k 16 and N 4096 k 256 (a close call), W1x_vimnmx_eager bf16 N 2048 k 1024 (a close call). The four cells where the `prefer_smaller_index` throughput winner differs are N5_rows_f cells that N6w_atom32 or W3_d4c take by 0.3 to 11 percent, inside the close-call band. Representative latency cells (`not_guaranteed`): bf16 N 256 k 8 W2_packed_v3 329 (W5_v4 +101 percent, reference 5950), bf16 N 512 k 16 W2_packed_v3 829 (W5_v4 +29 percent), fp32 N 256 k 16 W2_packed_v3 660 (N6w_atom32 +61 percent), fp32 N 512 k 64 N6w_atom32 1899 (N10_auto +23 percent), fp32 N 1024 k 16 W2_packed_v3 1768 (W5_v4 +1.2 percent), fp32 N 2048 k 16 W5_v4 1987 (W2_packed_v3 +127 percent), bf16 N 1024 k 64 W5_v4 2957 (N6w_atom32 +21 percent), fp32 N 4096 k 1024 N6w_atom32 66140 (W3_d4c +13 percent). Named workloads: W2_packed_v3 takes every MoE router latency cell (Step-3 N 64 k 4 179 cycles, GLM-4.5 N 160 k 8 258, Qwen3 N 128 k 8 262, ERNIE-4.5 N 64 k 8 287, the DeepSeek-V3 final top 8 N 128 k 8 314, Kimi K2 N 384 k 8 441, DeepSeek-V3 N 256 k 8 467, Qwen3.5 N 512 k 10 602, LongCat N 768 k 12 952, Nemotron 3 Ultra N 512 k 22 1070, Kimi K3 N 896 k 16 1314, the runner-up N6w_atom32 or W5_v4 from +21 to +265 percent) and their throughput except Nemotron (W5_v4 554, a close call), W5_v4 takes the N 2048 cells (NSA 128K context 1986 cycles, W2 +127 percent) and the PEER sub-key throughput at N 1024 k 16 (970) where the latency is W2\'s by 1.2 percent, N6w_atom32 the k 64 cells (EAGLE-2 rerank N 1024 k 64 3709, the sampling merge N 512 k 64 1899). Against the wave 2 card picture (2.12): the W2 to W5 flip sits at N 1024 (bf16 k 16, fp32 k 8), W2 to N6w at k 48 (N 256 and 512) or k 32 (fp32 N 512), N6w yields the N 1024 bf16 k 48 and 64 cells and the N 2048 k 64 cells to W5, and N10 keeps two close-call fp32 cells. The {prov} close calls stay marked in the final matrix.',
    'addendum': 'Addendum rows kept out of the one-node matrices above, by the rulings of 2026-10-26: the random matrices stay one-node ({main_node}), the 3b sorted-check batches are their own pattern and never enter them, and the two addenda measured under the random label on {other} are shown here with the node named, {n_cells} cells. Each row names the main winner of the pass, the extra adapters with their values and the nodes the cell mixes, and the margin of the first extra adapter over the main winner, which crosses nodes and is read as evidence and not as a verdict (the only scale for a cross-node margin is the B4_v4 confirmation, ratios 0.965 to 1.043 between the two nodes). In the mixed reduction B1_det3 wins {det3_w} of the {psi_n} prefer_smaller_index block cells (fp32 N 1024 k 32 and 64 latency within 1.5 percent of B1_det2, bf16 and fp32 N 1024 throughput at k 48 to 64, bf16 N 8192 k 64 and 128 throughput 2.6 to 6.1 percent over B4_v4) and none under not_guaranteed. On the sorted-runs track B1_det3@runs4 is a reference (an unsorted selection over the presorted runs), not a candidate. The named sub-warp cells at every lane count, latency, with the GB200 single-host winner beside each as the clean evidence: {sub_txt}. A same-node rerun of both addenda on the pass node is requested, version 31 replaces these rows if it lands.',
    'tie_header': 'Robustness under heavy ties, the tie_heavy pass on the tie_heavy input as generated (period 64, unshuffled). B200 {nodes}, harness 0.10, the close cells of the random matrices rerun with the tie_heavy pattern of the harness battery, v[j] = (j mod 64) / 64 without a shuffle, so 64 distinct values each N / 64 times and the top k one tie class for every k up to N / 64, {n_rows} cells: {scopes}, {n_prov} of them with a runner-up within 15 percent. This is a second input distribution and not a tie-breaker for the random matrices, whose close-call marks stand. The matrices below name the best adapter on this input per cell in the format of the random matrices, with the same controls. The tie_heavy manifest carries {n_extra} adapter labels the random pass did not run (listed at the end of the reading paragraph), which stay in these matrices and are kept out of the flip list.',
    'tie_reading': 'What the tie_heavy pass says, on the input as generated. Of the {n_both} cells with a random counterpart, {n_same} keep their winner and {n_flip} change it. The pattern is layout as much as ties: at 128, 256 and 512 threads the period 64 divides the stripe, so every striped thread holds 64 copies of one value and four threads of 256 hold only zeros, while the blocked tile of B3_tuned sees every value once per thread. B4_v4\'s exact-bound prefilter degenerates on it: the minimum of the per-thread maxima is the tile minimum, every item is a candidate, the count exceeds the 4096-word buffer from N 8192 and the tile fallback runs, {b4_8192}x behind B3_tuned at fp32 N 8192 k 32 ({b4v_8192} against {b3v_8192} cycles), {b4_16384}x at fp32 N 16384 k 32 ({b4v_16384} against {b3v_16384}, {b4r_16384}x its own random time), {b4_32768}x at bf16 N 32768 k 32 ({b4v_32768} against {b3v_32768}, with spill), and within 4 percent of its random time at N 1024 to 4096 where the tile fits the buffer. The guard is correct and the fallback\'s cost is the loss ({t_b4}, TK-S07.2). B1_n6b takes the bf16 N 1024 k 32 to 256 cells ({n6b_lo} to {n6b_hi} cycles, {n6b_rlo}x to {n6b_rhi}x of its random time) by its exact-count exit: the fixed-point image bins each of the 64 values separately, 16 items per bin at N 1024, every grid k but 24 is a multiple of 16, so the pivot bin is exact after one pass, and at k 24 it costs {n6b_k24}x its random time ({t_n6}). Reading flags: a ratio across the two inputs mixes a pattern-wide slowdown (strictly_above_k is 0 in every N 16384 cell, so every radix select runs its maximal pass count, B3_tuned 1.85x and the B1_reference control 1.45x their random time) with each adapter\'s own premium, the period 64 aligns with the stripe, k against the multiplicity 16 decides the N 1024 rows, the prefer_smaller_index rows of B1_n6b, N9_w2fin2 and B4_v4 carry spills (896 to 2020 bytes), and the tie_heavy rows come from umbriel-b200-068 while the random rows come from 047 (B4_v4 agrees across the two within 2.3 percent). A shuffled variant of the same multiset, requested from the evaluation engineer on the spare 047 lease after phase 3b, would separate layout from ties, and its outcome goes here. Adapter labels measured on tie_heavy input only, kept out of the flip list: {extra}.',
    'sorted_header': 'Sorted output, the phase 3b pass (B200 umbriel-b200-047, harness 0.10 with the sorted check in the oracle, 18 batches on 2026-10-26, the focus grid, every adapter that declares sorted output or has a documented post-sort: W5_v4, W1_level, W1x_vimnmx_eager and W4_wmerge at warp scope, W1_eager_same_sub at sub-warp scope, B6_radix at block scope, with the W4_chunk4 merge of four presorted runs measured in the same pass and reported in the sorted-runs table above since its input contract differs, and the N5 forms measured but not sorted routes since their contract is unsorted output). The sorted check is the harness battery with the order verified, keys non-increasing along the slots under both requirements and equal keys in ascending index order under prefer_smaller_index. Each cell names the fastest adapter whose sorted output passed, the next passing adapters, the price of sorted output as the margin over the random-pass winner of the same cell (the unsorted matrices above), and the adapters the check refuted or that do not support the shape. These 3b tables replace the card tables that stood here until version 29.',
    'sorted_reading': 'What the 3b pass says. Warp: W5_v4 is the fastest sorted route from N 512 at k 4 to 64 under not_guaranteed and for bf16 under both requirements (bf16 N 512 k 16 1067 cycles against the unsorted W2_packed_v3 829, N 2048 k 64 4323 against its own unsorted 3877, N 4096 k 64 10051 against 7789), the W1 level network at N 32 and 128 and at k 256 and 1024 where W4_wmerge does not apply, and W4_wmerge at k 256 and 1024 from N 512 (fp32) or N 2048 (bf16) and at fp32 N 512 and 2048 k 64 under prefer_smaller_index. W5_v4\'s fp32 sorted output under prefer_smaller_index is refuted as built: at k 16 and 64 of N 512, 2048 and 4096 the check finds equal keys inside the top k in buffer-position order instead of index order (quantized_random and relu_quantized rows, slots 13 and 14 at N 512), because the fp32 finish word replaces the low 7 to 9 twiddled key bits by the inverted buffer position and the tie rewrite covers only the class that straddles the k-th slot. The selected set is exact, the bf16 route is exact and sorted by construction, and W5_v4\'s unsorted winners stand. The fix is in the emission, the tie rewrite generalized to every class run inside the top k, which also closes a latent ordering hole under not_guaranteed when two distinct keys agree in the kept bits (not observed in the pass, a host simulation puts such a pair in 1 to 16 percent of rows at k 16 to 64). Sub-warp: the W1 network is the only sorted route (W1_eager_same_sub, 220 to 5335 cycles), the N5 forms are unsorted by contract. Block: B6_radix is the only sorted route from an unsorted row, a full block sort whose price over the unsorted winner runs from {b6_min} to {b6_max}, unsupported at fp32 N 32768 where its oracle kernel needs 270 KB of dynamic shared memory against the 227 KB opt-in limit (a harness item: an unlaunchable oracle kernel is counted as a failure where an unlaunchable regime counts as unsupported), and the merge of four presorted runs stays the cheapest sorted route where the input arrives as runs (the sorted-runs table). Status per adapter in the pass (cells, the N5 failures are the contract mismatch and the B6 failures the unlaunchable shape): {status}.',
    'gb200_header': 'GB200 pass, warp and sub-warp scope (gb200-nvl4-ts2-99, harness 0.10, one session ending 05:31 UTC on 2026-10-26, 57 batches, zero oracle failures, `WINNERS_gb200.md` and `state/phase3/winners_gb200.json` from `tools/winners.py`, committed at 11f4d5c71e, the batches of gb200-nvl4-ts2-105 taken before the network outage set aside in a superseded bucket). Same cells, adapters and controls as the B200 pass, `W1_reference` and `W1_baseline_sub` measured in the same lock section on the same host, so the `ref` margins are GB200 against GB200. Close calls: {prov_g} of {cells_g} cells across the scopes measured. The named sub-warp cells ran at every lane count the cell allows on this host (4 lanes only on the B200 side until the addendum lands), which is where the two tables differ most. The GB200 block pass covers the focus and flip grids at N 1024 and 4096, the named block workloads and the sorted-runs track.',
    'gb200_reading': 'What the GB200 pass says against the B200 pass. The winner is the same adapter at the same thread or lane count in {n_same} of the {n_both} cells both hosts measured. Among the warp and sub-warp differences: six named sub-warp cells where GB200 ran every lane count and the 16-lane W2_packed_sw_v2 form wins (gpt-oss-20b bf16 N 32 k 4 206 cycles against the 4-lane form\'s 581, DBRX fp32 N 16 k 4 328 against N10_auto_sub at 4 lanes 536, the DeepSeek group top 2 at N 32 k 2 302 against 650, MoBA N 64 k 4 469 against 1099, and the group top 2 and MoBA throughput cells at 8 and 16 lanes), eight sub-warp throughput cells at N 128 that are lane-count ties within W2_packed_sw_v2 or N10_auto_sub on one host or the other (bf16 k 4, fp32 k 16 and 64, under both requirements) with one real flip among them, bf16 N 128 k 16, where N5_rows_fs_sub at 16 lanes drops from 604 Gelem/s on B200 to 418 on GB200 and W2_packed_sw_v2 at 8 lanes takes the cell (519 against 483 on B200), and three warp throughput cells that are B200 ties within 5 percent (N5_rows_f against N6w_atom32 at bf16 N 512 k 64, W1_level against W1x_vimnmx_eager at bf16 N 2048 k 1024). Values on the same-winner cells, GB200 over B200: warp latency median 1.000 with 126 of 159 cells within 2 percent, the exceptions the long-list cells at N 2048 to 4096 from k 64 up (W5_v4 bf16 N 4096 k 64 1.104, W3_d4c k 1024 at N 2048 1.08, N6w_atom32 at N 4096 k 256 and 1024 1.04, the generation property of 2.12 and T58), warp throughput median 1.075 with the middle half 1.072 to 1.077 and the `W1_reference` control at the same 1.078, so the uniform gain is the host and not the adapters (one exception, the Kimi K2 router cell at 0.88), sub-warp latency median 1.000 (the Mixtral N 8 k 2 cell at 0.743 the one outlier, 217 against 292 cycles), sub-warp throughput 0.675 to 1.074 with the 16-lane forms at N 128 lower on GB200 (W2 fp32 k 4 0.675, N5 bf16 k 64 0.878) and the 8-lane forms at N 32 higher (1.07).',
    'subwarp_reading': 'What the sub-warp pass says. Latency: W2_packed_sw_v2 at 16 lanes takes every bf16 cell from N 32 (N 32 k 4 206 cycles, N 128 k 4 342, k 16 785, k 64 2593, its own 8-lane form the runner-up at +54 to +58 percent for k 4 and 16, W1_eager_same_sub at 16 lanes at +24 and +61 percent for N 32 k 16 and N 128 k 64) and the fp32 k 4 cells (N 32 363, N 128 578), N10_auto_sub at 16 lanes takes the fp32 k 16 and 64 cells (N 32 k 16 500, N 128 k 16 1133 a close call against W2 at +8 percent, N 128 k 64 1262), and W1_eager_same_sub at 4 lanes takes N 8 k 4 under both key types (200 and 219, its own 8-lane form at +10 to +12 percent, a close call). More lanes per row won every random latency cell where the lane count was free, the 16-lane form leads the 8-lane form of the same adapter by 40 to 58 percent. The named cells are 4-lane rows: W1_eager_same_sub at N 8 (Mixtral k 2 292 a close call, the DeepSeek group select k 4 219), N10_auto_sub at N 16 k 4 (DBRX 536) and N 64 k 4 (MoBA 1099, a close call over W2 at +5 percent), W2_packed_sw_v2 at N 32 (gpt-oss-20b bf16 k 4 581, the DeepSeek group top 2 650). Throughput inverts the lane count: W2_packed_sw_v2 at 4 lanes takes bf16 N 32 k 4 and fp32 N 8 k 4 and every named throughput cell (its 8-lane form within 2 percent on the bf16 focus cell), at 8 lanes bf16 N 32 k 16 and N 128 k 4 and fp32 N 32 k 4 (the other lane counts within 1 to 13 percent), at 16 lanes fp32 N 128 k 4 (717 Gelem/s), N5_rows_fs_sub at 16 lanes the bf16 N 128 k 16 and 64 cells (604 and 458) and at 8 lanes bf16 N 8 k 4, N10_auto_sub the fp32 N 128 k 16 and 64 cells at 8 lanes and N 32 k 16 at 16 lanes, every one a close call against its own other lane count or W2. `prefer_smaller_index` names the same winner as `not_guaranteed` on every sub-warp cell measured under both. `W1_baseline_sub` is 1.04x (fp32 N 32 k 16 throughput) to 11.6x (bf16 N 128 k 4 latency) the winner. The {prov} close calls are mostly same-adapter lane-count ties, which the implementer decides by the lane count the surrounding kernel already has.',
}

R_all = json.load(open(JSON))['rows']
R = [r for r in R_all if r['pattern'] in RND]
TIE = [r for r in R_all if r['pattern'].startswith('tie_heavy')]
import glob
def load_3b(results_dir):
    # the phase 3b batches (label phase3_b200_3b, the harness run with the sorted check): per cell and
    # adapter the oracle verdict and the value, plus the per-adapter unsupported and failed lists
    cells = collections.defaultdict(dict)
    status = collections.defaultdict(collections.Counter)
    texts = collections.defaultdict(set)
    for f in sorted(glob.glob(os.path.join(results_dir, '*_3b_*.json'))):
        if f.endswith('.build.json'):
            continue
        d = json.load(open(f))
        if not str(d.get('label', '')).endswith('_3b'):
            continue
        base = d['adapter']['id']
        scope = d['adapter']['scope']
        for c in d['cells']:
            aid = base + (f"@runs{c['runs']}" if c.get('runs') else '')
            ok = bool(c['raw']['variant'].get('oracle_passed'))
            cells[(scope, c['regime'], c['requirement'], c['key_type'], c['n'], c['k'])][aid] = ('pass' if ok else 'fail', c['value'], c.get('control_value'), c.get('control_name'))
            status[base]['pass' if ok else 'fail'] += 1
        for u in d.get('unsupported', []):
            status[base]['unsupported'] += 1
        for x in d.get('failed', []):
            status[base]['failed'] += 1
            t0 = str((x.get('failures') or [''])[0])
            texts[base].add(t0)
            # the failed list names the cell, both regimes of the cell carry the verdict
            m = re.search(r'__(bf16|fp32)_T(\d+)_N(\d+)_K(\d+)_(not_guaranteed|prefer_smaller_index)$', str(x.get('cell', '')))
            if m:
                kind = 'unsupported' if 'exceeds the device limit' in t0 else 'refuted'
                for regime in ('latency', 'throughput'):
                    cells[(scope, regime, m.group(5), m.group(1), int(m.group(3)), int(m.group(4)))].setdefault(base, (kind, None, None, None))
    return cells, status, texts
SORTED, SORTED_STATUS, SORTED_TEXTS = load_3b(args.results_3b)
ADD = [r for r in json.load(open(args.winners_addendum))['rows'] if r['pattern'] in RND] if args.winners_addendum and os.path.exists(args.winners_addendum) else []
MAIN_NODE = collections.Counter(n for r in R for n in r['nodes']).most_common(1)[0][0] if R else ''
# adapters whose contract is unsorted output are not sorted routes (N5 by its card), and the
# oracle kernel that cannot launch is an unsupported shape, not a failure (B6 at fp32 N 32768)
SORTED_EXCLUDE = ('N5_rows_f', 'N5_rows_fs_sub', 'W4_chunk4')
G = [r for r in json.load(open(JSON_GB))['rows'] if r['pattern'] in RND] if os.path.exists(JSON_GB) else []
def fmt(v):
    if v is None:
        return ''
    if v >= 10000:
        return f'{v/1000:.1f}k'
    if v >= 100:
        return f'{v:.0f}'
    return f'{v:.1f}'

def pct(m):
    return '' if m is None else f'{m*100:+.0f}%'

def canon(label):
    # the N8 block-size study measured B2_packed at other thread counts under its own id, the
    # winners tool merges the two names in newer reductions, older ones keep both
    return label.replace('N8_b2packed', 'B2_packed')

def strip_t(label, scope):
    # labels keep their thread or lane count suffix, it is a tuning dimension chosen per cell
    return label

def is_prov(r):
    return r['margin_runner_up'] is not None and r['margin_runner_up'] < PROV

def cell(r):
    sc = r['scope']
    prov = ' p' if is_prov(r) else ''
    parts = [f"{strip_t(r['winner'], sc)} {fmt(r['value'])}{prov}"]
    if r['runner_up']:
        parts.append(f"{strip_t(r['runner_up'], sc)} {pct(r['margin_runner_up'])}")
    parts.append(f"ref {pct(r['margin_reference'])}")
    if r['reference2']:
        parts.append(f"det {pct(r['margin_reference2'])}")
    return ', '.join(parts)

def rows_of(scope, regime, req, key, rows=None, pattern='random'):
    return [r for r in (R if rows is None else rows) if r['scope'] == scope and r['regime'] == regime and r['requirement'] == req
            and r['pattern'] == pattern and r['key_type'] == key and not r['workload']]

def matrix(scope, regime, req, key, rows=None, pattern='random'):
    sel = {(r['n'], r['k']): r for r in rows_of(scope, regime, req, key, rows, pattern)}
    ks = [k for k in KS if any((n, k) in sel for n in NS[scope])]
    ns = [n for n in NS[scope] if any((n, k) in sel for k in ks)]
    out = ['| N | ' + ' | '.join(f'k {k}' for k in ks) + ' |', '|---|' + '---|' * len(ks)]
    for n in ns:
        out.append(f'| {n} | ' + ' | '.join(cell(sel[(n, k)]) if (n, k) in sel else '' for k in ks) + ' |')
    return '\n'.join(out), len(sel)

unit = {'latency': 'cycles per call', 'throughput': 'Gelem/s for one occupancy wave'}
def scope_parts(scope, title, rows=None, pattern='random'):
    parts = []
    for regime in ('latency', 'throughput'):
        for req in ('not_guaranteed', 'prefer_smaller_index'):
            for key in ('bf16', 'fp32'):
                t, n = matrix(scope, regime, req, key, rows, pattern)
                if n == 0:
                    continue
                parts.append(f'{title}, {regime}, `{req}`, {key}, {unit[regime]} ({n} cells):\n\n{t}')
    return parts

have = {s: any(r['scope'] == s for r in R) for s in ('warp', 'subwarp', 'block')}
for s in ('warp', 'subwarp'):
    if have[s]:
        assert PROSE[s + '_header'] and PROSE[s + '_reading'], f'write the {s} prose first'

# sorted runs track (block)
runs = [r for r in R if r['pattern'] in ('random@runs4', 'random@runs8')]
by = collections.defaultdict(dict)
for r in runs:
    by[(r['key_type'], r['n'], r['k'])][(r['pattern'], r['regime'], r['requirement'])] = r
psi_dev = []
rrows = ['| key | N | k | runs 4, W4_chunk4 latency (B1_reference, margin) | runs 4 throughput (B1_reference, margin) | runs 8, W4_chunk8 latency (B1_reference, margin) | runs 8 throughput (B1_reference, margin) |', '|---|---|---|---|---|---|---|']
def cellr(d, pat, reg):
    r = d.get((pat, reg, 'not_guaranteed'))
    if r is None:
        return ''
    return f"{fmt(r['value'])} ({fmt(r['reference_value'])}, {pct(r['margin_reference'])})"
for (key, n, k) in sorted(by):
    d = by[(key, n, k)]
    for pat in ('random@runs4', 'random@runs8'):
        for reg in ('latency', 'throughput'):
            a = d.get((pat, reg, 'not_guaranteed'))
            b = d.get((pat, reg, 'prefer_smaller_index'))
            if a and b:
                psi_dev.append(abs(a['value'] - b['value']) / a['value'])
    rrows.append(f"| {key} | {n} | {k} | {cellr(d,'random@runs4','latency')} | {cellr(d,'random@runs4','throughput')} | {cellr(d,'random@runs8','latency')} | {cellr(d,'random@runs8','throughput')} |")
runs_table = '\n'.join(rrows)
psi_note = f'{max(psi_dev)*100:.1f}'
n_runs = len(runs)

# named workloads, every scope
SCOPE_ORDER = {'warp': 0, 'subwarp': 1, 'block': 2}
def named_table(rows):
    wl = [r for r in rows if r['workload']]
    wrows = ['| scope | workload | key | N | k | regime | requirement | best | runner-up, margin | reference, margin |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(wl, key=lambda r: (SCOPE_ORDER[r['scope']], r['workload'], r['key_type'], r['regime'], r['requirement'])):
        sc = r['scope']
        lab = 'sub-warp' if sc == 'subwarp' else sc
        prov = ' p' if is_prov(r) else ''
        ru = f"{strip_t(r['runner_up'], sc)} {pct(r['margin_runner_up'])}" if r['runner_up'] else ''
        wrows.append(f"| {lab} | {r['workload']} | {r['key_type']} | {r['n']} | {r['k']} | {r['regime']} | {r['requirement']} | {strip_t(r['winner'], sc)} {fmt(r['value'])}{prov} | {ru} | {r['reference']} {pct(r['margin_reference'])} |")
    return '\n'.join(wrows)
wl_table = named_table(R)

def wins_table(rows):
    w = collections.Counter((r['scope'], r['regime'], strip_t(r['winner'], r['scope'])) for r in rows)
    return '\n'.join(['| scope | regime | adapter | cells won |', '|---|---|---|---|'] + [
        f"| {'sub-warp' if sc == 'subwarp' else sc} | {reg} | {ad} | {c} |"
        for (sc, reg, ad), c in sorted(w.items(), key=lambda x: (SCOPE_ORDER[x[0][0]], x[0][1], -x[1]))])

def flip_table(tie, rnd):
    # tie_heavy cells against the random cell of the same shape. Only adapters the random pass
    # measured in that scope take part, so the tie_heavy winner here is the best of those and a
    # flip is never an adapter the random pass never saw. Values of both inputs, and the random
    # winner's value on tie_heavy input where it ran.
    key = lambda r: (r['scope'], r['regime'], r['requirement'], r['key_type'], r['n'], r['k'], r['workload'])
    rm = {(key(r), r['pattern']): r for r in rnd}
    seen = collections.defaultdict(set)
    for r in rnd:
        for a in r['all']:
            seen[r['scope']].add(canon(a['adapter']).split('@runs')[0])
    rows = ['| scope | regime | requirement | key | N | k | pattern | random winner | tie_heavy winner | random winner on tie_heavy | margin |', '|---|---|---|---|---|---|---|---|---|---|---|']
    same = flips = 0
    extra = collections.Counter()
    for t in sorted(tie, key=lambda r: (SCOPE_ORDER[r['scope']], r['regime'], r['requirement'], r['pattern'], r['key_type'], r['n'], r['k'])):
        b = rm.get((key(t), t['pattern'].replace('tie_heavy', 'random')))
        if b is None:
            continue
        cands = [a for a in t['all'] if canon(a['adapter']).split('@runs')[0] in seen[t['scope']]]
        for a in t['all']:
            if canon(a['adapter']).split('@runs')[0] not in seen[t['scope']]:
                extra[a['adapter']] += 1
        if not cands:
            continue
        best = dict(cands[0])
        best['adapter'] = canon(best['adapter'])
        if canon(b['winner']) == best['adapter']:
            same += 1
            continue
        flips += 1
        tv = {a['adapter']: a['value'] for a in t['all']}
        lab = 'sub-warp' if t['scope'] == 'subwarp' else t['scope']
        rw = tv.get(b['winner'])
        m = ''
        if rw is not None:
            m = pct((rw - best['value']) / best['value']) if t['regime'] == 'latency' else pct((best['value'] - rw) / rw)
        rows.append(f"| {lab} | {t['regime']} | `{t['requirement']}` | {t['key_type']} | {t['n']} | {t['k']} | {t['pattern']} | {b['winner']} {fmt(b['value'])} | {best['adapter']} {fmt(best['value'])} | {fmt(rw) if rw is not None else 'not run'} | {m} |")
    return '\n'.join(rows), same, flips, extra

def addendum_tables(add, main):
    # cells of the addendum reduction whose candidate list holds an adapter the main reduction never
    # measured in that cell, with the nodes named: the det3 rows and the lane-count rows of the
    # named sub-warp cells, measured on another node than the main pass
    key = lambda r: (r['scope'], r['regime'], r['requirement'], r['pattern'], r['key_type'], r['n'], r['k'], r['workload'])
    mm = {key(r): r for r in main}
    rows = ['| scope | regime | requirement | key | N | k | workload | pattern | main winner (' + MAIN_NODE + ') | addendum adapters, value, nodes of the cell | margin of the first over the main winner |', '|---|---|---|---|---|---|---|---|---|---|---|']
    n_cells = 0
    for a in sorted(add, key=lambda r: (SCOPE_ORDER[r['scope']], r['regime'], r['requirement'], r['pattern'], r['key_type'], r['n'], r['k'], r['workload'])):
        m = mm.get(key(a))
        have = {canon(x['adapter']) for x in m['all']} if m else set()
        extra = [x for x in a['all'] if canon(x['adapter']) not in have]
        if not extra:
            continue
        n_cells += 1
        best = extra[0]
        mw = f"{m['winner']} {fmt(m['value'])}" if m else 'no main cell'
        marg = ''
        if m:
            marg = pct((m['value'] - best['value']) / best['value']) if a['regime'] == 'latency' else pct((best['value'] - m['value']) / m['value'])
        lab = 'sub-warp' if a['scope'] == 'subwarp' else a['scope']
        extras = ', '.join(f"{x['adapter']} {fmt(x['value'])}" for x in extra[:3])
        if '@runs' in a['pattern']:
            marg = (marg + ', ' if marg else '') + 'reference only, unsorted output on the presorted runs'
        rows.append(f"| {lab} | {a['regime']} | `{a['requirement']}` | {a['key_type']} | {a['n']} | {a['k']} | {a['workload']} | {a['pattern']} | {mw} | {extras}, {' and '.join(a['nodes'])} | {marg} |")
    return '\n'.join(rows), n_cells

def compare_tables(gb, b200):
    # per scope, regime and requirement: cells both hosts measured, same winner, the GB200 over B200
    # value ratio on the same-winner cells, and the cells whose winner differs
    key = lambda r: (r['scope'], r['regime'], r['requirement'], r['pattern'], r['key_type'], r['n'], r['k'], r['workload'])
    bm = {key(r): r for r in b200}
    groups = collections.defaultdict(list)
    diffs = []
    extra_adapters = collections.Counter()
    for g0 in gb:
        b = bm.get(key(g0))
        if b is None:
            continue
        have = {canon(x['adapter']) for x in b['all']}
        for x in g0['all']:
            if canon(x['adapter']) not in have:
                extra_adapters[x['adapter']] += 1
        cands = [x for x in g0['all'] if canon(x['adapter']) in have]
        if not cands:
            continue
        # the GB200 cell seen through the adapters the B200 cell measured, labels canonical
        g = dict(g0)
        g['winner'], g['value'] = canon(cands[0]['adapter']), cands[0]['value']
        b = dict(b)
        b['winner'] = canon(b['winner'])
        groups[(g['scope'], g['regime'], g['requirement'])].append((g, b))
        if g['winner'] != b['winner']:
            diffs.append((g, b))
    crow = ['| scope | regime | requirement | cells | same winner | GB200 over B200, median | min | max | within 2 percent |', '|---|---|---|---|---|---|---|---|---|']
    for (sc, reg, req) in sorted(groups, key=lambda x: (SCOPE_ORDER[x[0]], x[1], x[2])):
        pairs = groups[(sc, reg, req)]
        same = [(g['value'] / b['value']) for g, b in pairs if g['winner'] == b['winner']]
        lab = 'sub-warp' if sc == 'subwarp' else sc
        crow.append(f"| {lab} | {reg} | `{req}` | {len(pairs)} | {len(same)} | {statistics.median(same):.3f} | {min(same):.3f} | {max(same):.3f} | {sum(1 for v in same if abs(v - 1) <= 0.02)} |")
    drow = ['| scope | regime | requirement | key | N | k | workload | GB200 winner | B200 winner | B200 winner on GB200 | GB200 winner on B200 |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for g, b in sorted(diffs, key=lambda x: (SCOPE_ORDER[x[0]['scope']], x[0]['regime'], x[0]['requirement'], x[0]['key_type'], x[0]['n'], x[0]['k'])):
        gv = {canon(a['adapter']): a['value'] for a in g['all']}
        bv = {canon(a['adapter']): a['value'] for a in b['all']}
        lab = 'sub-warp' if g['scope'] == 'subwarp' else g['scope']
        drow.append(f"| {lab} | {g['regime']} | `{g['requirement']}` | {g['key_type']} | {g['n']} | {g['k']} | {g['workload']} | {g['winner']} {fmt(g['value'])}{' p' if is_prov(g) else ''} | {b['winner']} {fmt(b['value'])}{' p' if is_prov(b) else ''} | {fmt(gv[b['winner']]) if b['winner'] in gv else 'not run'} | {fmt(bv[g['winner']]) if g['winner'] in bv else 'not run'} |")
    total = sum(len(v) for v in groups.values())
    per_scope = {}
    for sc in ('warp', 'subwarp', 'block'):
        pairs = [pr for (s_, reg, req), v in groups.items() if s_ == sc for pr in v]
        if not pairs:
            continue
        lat = [g['value'] / b['value'] for g, b in pairs if g['winner'] == b['winner'] and g['regime'] == 'latency']
        thr = [g['value'] / b['value'] for g, b in pairs if g['winner'] == b['winner'] and g['regime'] == 'throughput']
        per_scope[sc] = {'both': len(pairs), 'same': sum(1 for g, b in pairs if g['winner'] == b['winner']),
                         'lat_med': statistics.median(lat) if lat else None, 'thr_med': statistics.median(thr) if thr else None,
                         'lat_in2': sum(1 for v in lat if abs(v - 1) <= 0.02), 'lat_n': len(lat),
                         'pairs': collections.Counter((g['winner'], b['winner']) for g, b in pairs if g['winner'] != b['winner'])}
    per_scope['extra_adapters'] = extra_adapters
    return '\n'.join(crow), '\n'.join(drow), total, total - len(diffs), len(diffs), per_scope

# tallies
def tally(scope, regime, req, pattern='random'):
    sel = [r for r in R if r['scope'] == scope and r['regime'] == regime and r['requirement'] == req and r['pattern'] == pattern]
    return len(sel), collections.Counter(strip_t(r['winner'], scope) for r in sel)
stats = {}
for scope in ('warp', 'subwarp', 'block'):
    for regime in ('latency', 'throughput'):
        for req in ('not_guaranteed', 'prefer_smaller_index'):
            stats[(scope, regime, req)] = tally(scope, regime, req)
prov = {s: sum(1 for r in R if r['scope'] == s and is_prov(r)) for s in ('warp', 'subwarp', 'block')}
cells = {s: sum(1 for r in R if r['scope'] == s) for s in ('warp', 'subwarp', 'block')}
h2h = {s: sum(1 for r in R if r['scope'] == s and r['candidates'] > 1 and not r['workload']) for s in ('warp', 'subwarp', 'block')}
h2h_all = {s: sum(1 for r in R if r['scope'] == s and r['candidates'] > 1) for s in ('warp', 'subwarp', 'block')}

b4 = [r for r in R if r['scope'] == 'block' and r['regime'] == 'latency' and r['requirement'] == 'not_guaranteed' and r['pattern'] == 'random' and r['winner'] == 'B4_v4']
b4_med_ref = statistics.median(r['margin_reference'] for r in b4)
b4_med_ru = statistics.median(r['margin_runner_up'] for r in b4 if r['margin_runner_up'] is not None)
S = lambda scope, regime, req: stats[(scope, regime, req)]
TIE_T_IDS = ('T67', 'T68')
B2_128 = 'N8_b2packed@T128' if any(r['winner'] == 'N8_b2packed@T128' for r in R) else 'B2_packed@T128'

# the phase 3b card tables kept from the current knowledge base
kb = open(KB).read()
a = kb.index('Sorted output, the phase 3b routes on card numbers')
ends = [kb.index(m, a) for m in ('What the card matrices say', 'The pre-phase-3 picture', '\n## 17.') if m in kb[a:]]
sorted_block = kb[a:min(ends)].rstrip()

out = []
out.append('## 16. Winners (the phase 3 pass on B200, warp, sub-warp and block)')
out.append('')
out.append('What this is. The phase 3 final evaluation: one arch (B200), one node (umbriel-b200-047), one harness version (0.10), one lease session per scope group on 2026-10-25 (block, then warp and sub-warp), random input on the focus grid plus the flip bands, the named workload cells in their own table, `WINNERS_b200.md` and `state/phase3/winners_b200.json` from `tools/winners.py`. Every cell names the best adapter and its value (latency in cycles per call net of the generate-only control, throughput in Gelem/s for one occupancy wave), then the runner-up with the margin (latency: how much slower the runner-up is, throughput: how much faster the best is), then the margin over the reference of the scope measured in the winner\'s own lock section at the winner\'s thread or lane count (`ref`: `B1_reference` at block scope, `W1_reference` at warp scope, `W1_baseline_sub` at sub-warp scope), and in the block `prefer_smaller_index` cells the margin over `B1_det`, the wave 1 deterministic path, as the second reference (`det`). "p" marks a close call, a margin over the runner-up below 15 percent. Nothing in the study settles a close call by speed: same-node reruns reproduce to 0.1 percent, so the ranking on the node is stable and the margin is real, but below the 15 percent the study treats as a decision, so either adapter is an acceptable choice on speed and the choice falls to the other columns, the tie_heavy flip list below (robustness on the layout-aligned tie input), register count and spills, the deterministic tie-break cost (2.12) and code size, with cross-node agreement as the only measurement evidence added (the 48-cell B4_v4 confirmation on umbriel-b200-068, ratios 0.965 to 1.043 and no winner flips, is that kind of evidence). The close-call count per scope is a headline number of the pass. Pending after this pass: the GB200 block cells at N 1024 and 4096, phase 3b (sorted output, whose card tables stay at the end of this section until then), the shuffled variant of the tie_heavy input. These numbers are not comparable with the wave 1 and 2 card numbers of sections 11 to 23 (harness 0.10 streams every load past L1 and its throughput ceilings are the L2 rate, 2.10). The card-based matrices that stood here before the pass are gone, their numbers stay in the approach sections and the ledger (8) records when each part was replaced.')
out.append('')
if have['warp']:
    out.append(PROSE['warp_header'].format(cells=cells['warp'], prov=prov['warp']))
    out.append('')
    for p in scope_parts('warp', 'Warp'):
        out.append(p)
        out.append('')
if have['subwarp']:
    out.append(PROSE['subwarp_header'].format(cells=cells['subwarp'], prov=prov['subwarp']))
    out.append('')
    for p in scope_parts('subwarp', 'Sub-warp'):
        out.append(p)
        out.append('')
out.append(f'Block scope (71 batches plus the W4_chunk8 addendum, {cells["block"]} cells of which {h2h_all["block"]} head-to-head on random input and {len(runs)} on the sorted-runs track, 13 adapters, zero oracle failures, committed at e62f286f62 and c0e67ea6db). The reference behind `ref` is the control of the winner\'s own section (`tools/winners.py` takes the winner\'s control value), so a cell won at 512 or 128 threads carries `B1_reference@T512` or `@T128`, and the 512-thread reference is up to 28 percent above the 256-thread one on some throughput cells (verify-B4\'s self-test), which is why `ref` margins are not comparable across block sizes. Close calls: {prov["block"]} of the {h2h_all["block"]} head-to-head cells. Adapters at a thread count other than 256 carry `@T128` or `@T512`, the reference then runs at the same thread count. `B1_det3` had no card at the phase 3 go and is not in this pass, the deterministic cells use `B1_det2`. Four `W4_chunk4` batches were quarantined (a manifest mistake fed 8 runs to the 4-run adapter, which violates its contract and failed the oracle) and re-measured as `W4_chunk8`, whose 64 cells are the runs 8 columns of the sorted-runs table. Four notes from the algorithm researcher\'s cold review of these tables (no monotonicity break, no requirement gap, no duplicate spread above 0.5 percent): the B2_packed runner-up values at N 16384 and 32768 are a broken-control artifact, since at 64 and 128 items per thread its fully unrolled list leaves the register file (T34, 14.4), so the plus 68 to 100 percent margins of the two-level list over it at N 16384 and the about 530x at bf16 N 32768 k 4 in the per-cell lists are not competitive margins and the next capable form there is B4_v4 or B1_idx at 512 threads. B4_v4\'s margin over B3_tuned at fp32 N 16384 k 64 and 256 reads 1.74x and 1.78x here against 2.23x on the cards, the difference is the harness 0.10 load path (B3_tuned is a large-shared-memory control whose net dropped, B4_v4 is striped and rose toward its 0.4 value), both same-session under 0.10 so the ranking stands. The 136 single-candidate rows are the W4 sorted-runs track and not head-to-head winners, and one of them is a loss: W4_chunk8 at fp32 N 8192 k 4096 is 28.6 percent slower than B1_reference\'s unsorted selection (18527 against 13238 cycles, 34 percent lower throughput), the k 4096 merge touches the whole row. B4_v4 has two verified cards (`B4_v4_b200_20261025_1` and `_2`, rerun by verify-B4 on umbriel-b200-018 under harness 0.8), and verify-B4 confirmed its phase 3 cells on a second node under harness 0.10 (umbriel-b200-068, 48 cells across N 8192 to 32768 and k 32 to 1024 in all four quadrants, no winner flips, the B4_v4 ratio 068 over 047 between 0.983 and 1.023 with median 1.000, runner-ups 0.965 to 1.040, B1_reference 0.995 to 1.043, `state/phase3/b4v4_confirm_umbriel-b200-068.md`): the closest margin, bf16 N 32768 k 32 over B2_twolevel, narrowed from 14.0 to 8.6 percent and held, and the fp32 N 8192 k 32 near tie stays with N9_w2fin2 by 0.6 percent on both nodes, both cells close calls.')
out.append('')
for p in scope_parts('block', 'Block'):
    out.append(p)
    out.append('')
out.append(f'Sorted runs, block (`--runs 4` and `--runs 8`, four or eight sorted runs of N over M merged to the sorted top k by `W4_chunk4` and `W4_chunk8`, the only adapters with that contract, so these are a track and not head-to-head winners, `B1_reference` on the same unsorted-selection cell as the control, both requirements identical within {psi_note} percent, {n_runs} cells). The one loss of the track is W4_chunk8 at fp32 N 8192 k 4096, 28.6 percent behind the unsorted reference:')
out.append('')
out.append(runs_table)
out.append('')
out.append('Named workload cells of the pass (10.2 names the workloads):')
out.append('')
out.append(wl_table)
out.append('')
out.append('Cells won per adapter (every pattern and requirement):')
out.append('')
out.append(wins_table(R))
out.append('')
if have['warp']:
    out.append(PROSE['warp_reading'].format(prov=prov['warp']))
    out.append('')
if have['subwarp']:
    out.append(PROSE['subwarp_reading'].format(prov=prov['subwarp']))
    out.append('')
out.append(f"What the block pass says. B4_v4 takes {S('block','latency','not_guaranteed')[1]['B4_v4']} of {S('block','latency','not_guaranteed')[0]} `not_guaranteed` latency cells (median {b4_med_ref*100:+.0f} percent over the reference and {b4_med_ru*100:+.0f} percent over the runner-up) and {S('block','latency','prefer_smaller_index')[1]['B4_v4']} of {S('block','latency','prefer_smaller_index')[0]} `prefer_smaller_index` latency cells, every N 8192 and above from k 32 (N 16384) or k 64 (N 8192) up to k 1024, with the two-level B2 list keeping k 4 and 16 at N 16384 and 32768 and B3_tuned keeping k 4096. N 1024 to 4096 is shared by the 128-thread B2 ({S('block','latency','not_guaranteed')[1][B2_128]} latency cells, k at most 16, and k 32 at N 8192) and N9_w2fin2, the N6 first pass with the packed finish ({S('block','latency','not_guaranteed')[1]['N9_w2fin2']} cells, k 24 to 256, many of them close calls against B1_idx, B1_n6b and B1_det2). Under `prefer_smaller_index` B3_tuned takes {S('block','latency','prefer_smaller_index')[1]['B3_tuned']} latency cells (its scan emission is free of the requirement) and the 128-thread B2 and N9_w2fin2 keep the small N. Throughput: B4_v4 {S('block','throughput','not_guaranteed')[1]['B4_v4']} and {S('block','throughput','prefer_smaller_index')[1]['B4_v4']} cells at large N, B1_idx {S('block','throughput','not_guaranteed')[1]['B1_idx']} and B1_idx at 512 threads {S('block','throughput','not_guaranteed')[1]['B1_idx@T512']} under `not_guaranteed` and B1_det2 {S('block','throughput','prefer_smaller_index')[1]['B1_det2']} under `prefer_smaller_index` at mid N, the 128-thread B2 and the two-level list at small k. Representative fp32 `not_guaranteed` latency: N 1024 k 16 N8_b2packed at 128 threads 1476 (B2_packed plus 10 percent, the reference at 128 threads 2321), N 1024 k 64 N9_w2fin2 1850 (B1_idx plus 2.8 percent, reference 2010), N 4096 k 16 the 128-thread B2 2237 (reference 5012), N 4096 k 256 N9_w2fin2 4003 (B3_tuned plus 4.1 percent, reference 5334), N 16384 k 256 B4_v4 5198 (N9_w2fin2 plus 75 percent, reference 13360), N 32768 k 16 B4_v4 5327 (B2_twolevel plus 63 percent, reference 19415). The sorted-runs track is W4_chunk4's and W4_chunk8's alone (1.3x to 6.2x under the unsorted reference selection at four runs, one loss at eight runs and k 4096), the phase 3b pass sets it against the other sorted routes. The B4_v4 cells at N 8192 and above are confirmed on a second node under harness 0.10 (no winner flips, ratios within 2.3 percent). The {prov['block']} close calls stay marked in the final matrix, the tie_heavy flip list below is the first of their secondary columns.")
out.append('')
if G:
    assert PROSE['gb200_header'] and PROSE['gb200_reading'], 'write the GB200 prose first'
    out.append(PROSE['gb200_header'].format(cells_g=len(G), prov_g=sum(1 for r in G if is_prov(r))))
    out.append('')
    for scope, title in (('warp', 'GB200 warp'), ('subwarp', 'GB200 sub-warp'), ('block', 'GB200 block')):
        if any(r['scope'] == scope for r in G):
            for p in scope_parts(scope, title, G):
                out.append(p)
                out.append('')
    if any(r['workload'] for r in G):
        out.append('Named workload cells of the GB200 pass:')
        out.append('')
        out.append(named_table(G))
        out.append('')
    out.append('Cells won per adapter in the GB200 pass:')
    out.append('')
    out.append(wins_table(G))
    out.append('')
    ctab, dtab, n_both, n_same, n_diff, per_scope = compare_tables(G, R)
    out.append(f'GB200 against B200, the cells both hosts measured ({n_both}, same winner in {n_same}, each GB200 cell seen through the adapters the B200 cell measured, the ratio is the GB200 value over the B200 value of the same winner):')
    out.append('')
    out.append(ctab)
    out.append('')
    out.append(f'The {n_diff} cells whose winner differs between the hosts (values in the unit of the regime, "not run" where the other host did not measure that lane count):')
    out.append('')
    out.append(dtab)
    out.append('')
    blk = per_scope.get('block')
    def pair_text(c):
        return ', '.join(f'{gw} on GB200 against {bw} on B200 in {n} cells' for (gw, bw), n in c.most_common(3))
    blk_s = (f"At block scope (the N 1024 and 4096 grids, the named block workloads and the sorted-runs track) the winner is the same in {blk['same']} of {blk['both']} comparable cells, the differences {pair_text(blk['pairs'])}, with the latency ratio median {blk['lat_med']:.3f} ({blk['lat_in2']} of {blk['lat_n']} same-winner cells within 2 percent) and the throughput ratio median {blk['thr_med']:.3f}." if blk else '')
    ex = per_scope.get('extra_adapters')
    if ex:
        blk_s += ' The comparison sees each GB200 cell through the adapters the B200 cell measured, the GB200 pass also ran ' + ', '.join(f'{a} ({n} cells)' for a, n in ex.most_common(4)) + ', which the GB200 matrices above show.'
    out.append(PROSE['gb200_reading'].format(n_same=n_same, n_both=n_both) + (' ' + blk_s if blk_s else ''))
    out.append('')
if ADD and PROSE['addendum']:
    atab, n_add = addendum_tables(ADD, R)
    psi_b = [r for r in ADD if r['scope'] == 'block' and r['pattern'] == 'random' and r['requirement'] == 'prefer_smaller_index']
    det3_w = sum(1 for r in psi_b if r['winner'].startswith('B1_det3'))
    sub_named = [r for r in ADD if r['scope'] == 'subwarp' and r['workload'] and r['regime'] == 'latency']
    g_named = {r['workload']: r for r in G if r['scope'] == 'subwarp' and r['workload'] and r['regime'] == 'latency'}
    sub_txt = ', '.join(f"{r['workload']} {r['winner']} {fmt(r['value'])} (GB200 {g_named[r['workload']]['winner']} {fmt(g_named[r['workload']]['value'])})" if r['workload'] in g_named else f"{r['workload']} {r['winner']} {fmt(r['value'])}" for r in sub_named)
    other = sorted({n for r in ADD for n in r['nodes'] if n != MAIN_NODE})
    out.append(PROSE['addendum'].format(n_cells=n_add, main_node=MAIN_NODE, other=' and '.join(other), det3_w=det3_w, psi_n=len(psi_b), sub_txt=sub_txt))
    out.append('')
    out.append(atab)
    out.append('')
elif ADD:
    print('addendum reduction present but its prose is empty, the addendum part is omitted', file=sys.stderr)
if TIE and PROSE['tie_header'] and PROSE['tie_reading']:
    _ftab, _ns, _nf, _extra = flip_table(TIE, R)
    out.append(PROSE['tie_header'].format(n_rows=len(TIE), n_extra=len(_extra), nodes=' and '.join(sorted(set(r['node'] for r in TIE))),
                                          n_prov=sum(1 for r in TIE if is_prov(r)), scopes=', '.join(f"{'sub-warp' if sc == 'subwarp' else sc} {n}" for sc, n in sorted(collections.Counter(r['scope'] for r in TIE).items(), key=lambda x: SCOPE_ORDER[x[0]]))))
    out.append('')
    for scope, title in (('warp', 'tie_heavy input, warp'), ('subwarp', 'tie_heavy input, sub-warp'), ('block', 'tie_heavy input, block')):
        if any(r['scope'] == scope and r['pattern'] == 'tie_heavy' for r in TIE):
            for p in scope_parts(scope, title, TIE, 'tie_heavy'):
                out.append(p)
                out.append('')
    ftab, n_same, n_flip, extra = flip_table(TIE, R)
    out.append(f'tie_heavy against random input on the same cell ({n_same + n_flip} cells with a random counterpart, same winner in {n_same}, a different winner in {n_flip}). Only adapters the random pass measured in the scope take part in this list, so a flip is never an adapter the random pass never saw, and the margin is the tie_heavy winner over the random winner on tie_heavy input:')
    out.append('')
    out.append(ftab)
    out.append('')
    def tv(pat, regime, req, key, n, k, adapter):
        for r in (TIE if pat == 'tie_heavy' else R):
            if r['scope'] == 'block' and r['pattern'] == pat and r['regime'] == regime and r['requirement'] == req and r['key_type'] == key and r['n'] == n and r['k'] == k and not r['workload']:
                for a in r['all']:
                    if a['adapter'] == adapter:
                        return a['value']
        return None
    tb = {}
    for n_, key_ in ((8192, 'fp32'), (16384, 'fp32'), (32768, 'bf16')):
        b4 = tv('tie_heavy', 'latency', 'not_guaranteed', key_, n_, 32, 'B4_v4')
        b3 = tv('tie_heavy', 'latency', 'not_guaranteed', key_, n_, 32, 'B3_tuned')
        tb[n_] = (b4, b3)
    b4r = tv('random', 'latency', 'not_guaranteed', 'fp32', 16384, 32, 'B4_v4')
    n6b = [(tv('tie_heavy', 'latency', 'not_guaranteed', 'bf16', 1024, k_, 'B1_n6b'), tv('random', 'latency', 'not_guaranteed', 'bf16', 1024, k_, 'B1_n6b')) for k_ in (32, 48, 64, 256)]
    n6b = [(a, b) for a, b in n6b if a and b]
    k24 = (tv('tie_heavy', 'latency', 'not_guaranteed', 'fp32', 1024, 24, 'B1_n6b'), tv('random', 'latency', 'not_guaranteed', 'fp32', 1024, 24, 'B1_n6b'))
    out.append(PROSE['tie_reading'].format(n_rows=len(TIE), n_both=n_same + n_flip, n_same=n_same, n_flip=n_flip, t_b4=TIE_T_IDS[0], t_n6=TIE_T_IDS[1],
                                           b4_8192=f'{tb[8192][0] / tb[8192][1]:.2f}', b4v_8192=f'{tb[8192][0]:.0f}', b3v_8192=f'{tb[8192][1]:.0f}',
                                           b4_16384=f'{tb[16384][0] / tb[16384][1]:.2f}', b4v_16384=f'{tb[16384][0]:.0f}', b3v_16384=f'{tb[16384][1]:.0f}', b4r_16384=f'{tb[16384][0] / b4r:.1f}',
                                           b4_32768=f'{tb[32768][0] / tb[32768][1]:.2f}', b4v_32768=f'{tb[32768][0]:.0f}', b3v_32768=f'{tb[32768][1]:.0f}',
                                           n6b_lo=fmt(min(a for a, b in n6b)), n6b_hi=fmt(max(a for a, b in n6b)), n6b_rlo=f'{min(a / b for a, b in n6b):.2f}', n6b_rhi=f'{max(a / b for a, b in n6b):.2f}',
                                           n6b_k24=f'{k24[0] / k24[1]:.2f}' if k24[0] and k24[1] else 'n/a',
                                           extra=', '.join(sorted(extra))))
    out.append('')
elif TIE:
    print('tie_heavy rows present but their prose is empty, the tie_heavy part is omitted', file=sys.stderr)
def sorted_tables():
    # per scope, regime and requirement: the fastest adapter that passed the sorted check, its value,
    # the random-pass winner of the same cell for the price of sorted output, and the refuted or
    # unsupported entries of the cell
    rnd = {(r['scope'], r['regime'], r['requirement'], r['key_type'], r['n'], r['k']): r for r in R if r['pattern'] == 'random' and not r['workload']}
    parts = []
    for scope in ('warp', 'subwarp', 'block'):
        for regime in ('latency', 'throughput'):
            for req in ('not_guaranteed', 'prefer_smaller_index'):
                keys = sorted(k for k in SORTED if k[0] == scope and k[1] == regime and k[2] == req)
                if not keys:
                    continue
                lab = 'sub-warp' if scope == 'subwarp' else scope
                rows = [f'| key | N | k | sorted output, best passing | next | price over the unsorted winner | refuted or unsupported |', '|---|---|---|---|---|---|---|']
                for k in keys:
                    ad = {a: v for a, v in SORTED[k].items() if a.split('@')[0] not in SORTED_EXCLUDE}
                    if not ad:
                        continue
                    ok = sorted(((v[1], a) for a, v in ad.items() if v[0] == 'pass'), reverse=(regime != 'latency'))
                    bad = [f'{a} {v[0]}' for a, v in ad.items() if v[0] in ('refuted', 'unsupported')]
                    r = rnd.get(k)
                    price = ''
                    if ok and r:
                        price = pct((ok[0][0] - r['value']) / r['value']) if regime == 'latency' else pct((r['value'] - ok[0][0]) / ok[0][0])
                        price += f" over {r['winner']} {fmt(r['value'])}"
                    best = f'{ok[0][1]} {fmt(ok[0][0])}' if ok else 'none'
                    nxt = ', '.join(f'{a} {fmt(v)}' for v, a in ok[1:3])
                    rows.append(f'| {k[3]} | {k[4]} | {k[5]} | {best} | {nxt} | {price} | {", ".join(bad)} |')
                parts.append(f'Sorted output, {lab}, {regime}, `{req}`, {unit[regime]} ({len(rows) - 2} cells):\n\n' + '\n'.join(rows))
    return parts

if SORTED and PROSE['sorted_header'] and PROSE['sorted_reading']:
    out.append(PROSE['sorted_header'])
    out.append('')
    for p in sorted_tables():
        out.append(p)
        out.append('')
    st = ', '.join(f"{a} {c.get('pass', 0)} passed, {c.get('failed', 0)} failed, {c.get('unsupported', 0)} unsupported" for a, c in sorted(SORTED_STATUS.items()))
    rnd6 = {(r['scope'], r['regime'], r['requirement'], r['key_type'], r['n'], r['k']): r for r in R if r['pattern'] == 'random' and not r['workload'] and r['scope'] == 'block'}
    b6 = []
    for k, ad in SORTED.items():
        v = ad.get('B6_radix')
        r = rnd6.get(k)
        if v and v[0] == 'pass' and r and k[1] == 'latency':
            b6.append((v[1] / r['value'], k[3], k[4], k[5], r['winner']))
    b6.sort()
    out.append(PROSE['sorted_reading'].format(status=st, b6_min=f'{b6[0][0]:.1f}x ({b6[0][1]} N {b6[0][2]} k {b6[0][3]} against {b6[0][4]})', b6_max=f'{b6[-1][0]:.0f}x ({b6[-1][1]} N {b6[-1][2]} k {b6[-1][3]} against {b6[-1][4]})'))
    out.append('')
elif SORTED:
    print('3b batches present but their prose is empty, the card tables stay', file=sys.stderr)
    out.append(sorted_block)
else:
    out.append(sorted_block)
out.append('')
out.append('The pre-phase-3 picture (the card matrices of wave 1 and 2 and the paragraphs that read them) was removed from this section when the passes landed, the card numbers stay in sections 11 to 23 and the ledger (8) has the dates.')
text = '\n'.join(out) + '\n'
if OUT == '-':
    sys.stdout.write(text)
else:
    open(OUT, 'w').write(text)
print('section 16:', text.count('\n'), 'lines, scopes present', have, file=sys.stderr)
if TIE:
    _f, _s, _n, _x = flip_table(TIE, R)
    print(f'== tie_heavy: rows {len(TIE)}, with a random counterpart {_s + _n}, same winner {_s}, different {_n}, nodes {sorted(set(r["node"] for r in TIE))}, adapters not in the random pass {dict(_x)}', file=sys.stderr)
if SORTED:
    print(f'== 3b: cells {len(SORTED)}, status per adapter ' + '; '.join(f'{a} {dict(c)}' for a, c in sorted(SORTED_STATUS.items())), file=sys.stderr)
if G:
    ctab, dtab, n_both, n_same, n_diff, per_scope = compare_tables(G, R)
    print(f'== GB200: cells {len(G)}, both hosts {n_both}, same winner {n_same}, different {n_diff}, close calls {sum(1 for r in G if is_prov(r))}', file=sys.stderr)
for s in ('warp', 'subwarp', 'block'):
    if not have[s]:
        continue
    print(f'== {s}: cells {cells[s]}, head-to-head {h2h_all[s]}, close calls {prov[s]}', file=sys.stderr)
    for regime in ('latency', 'throughput'):
        for req in ('not_guaranteed', 'prefer_smaller_index'):
            n, c = stats[(s, regime, req)]
            print(f'   {regime} {req}: {n} cells, ' + ', '.join(f'{a} {k}' for a, k in c.most_common(8)), file=sys.stderr)
