"""Regenerates knowledge base section 16 (the winners of the phase 3 pass) from the winners JSON.

Inputs: the winners file written by tools/winners.py (--winners, default state/phase3/winners_b200.json
next to this tools directory) and the knowledge base markdown (--kb), from which the phase 3b card
tables at the end of the section are carried over until the phase 3b pass lands.
Output: the section text from its heading line through the line before section 17 (--out, default
stdout), plus a tally of cells, head-to-head cells, provisional cells and winners per scope on stderr.
The tables are generated, the paragraphs that read them are the PROSE strings below and are edited by
hand when a pass lands. The knowledge base is spliced by hand from the output.
"""
import argparse, collections, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--winners', default=os.path.join(HERE, '..', 'state', 'phase3', 'winners_b200.json'))
ap.add_argument('--kb', default=os.path.join(HERE, '..', 'KNOWLEDGE_BASE.md'))
ap.add_argument('--out', default='-')
args = ap.parse_args()
JSON, KB, OUT = args.winners, args.kb, args.out
KS = [2, 4, 8, 10, 12, 16, 22, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096]
NS = {'block': [1024, 2048, 4096, 8192, 16384, 32768],
      'warp': [32, 64, 128, 256, 512, 1024, 2048, 4096],
      'subwarp': [8, 16, 32, 64, 128]}
LANES = [4, 8, 16]
PROV = 0.15

# prose that depends on the pass reports, filled by hand when a pass lands
PROSE = {
    'warp_header': 'Warp scope (the second lease session on the same node, measured 20:50 UTC, 57 batches with numbers across the warp and sub-warp scopes and 21 empty by construction since the named cells carry only `not_guaranteed` and the sub-warp scope has no flip bands, 316 cells all head-to-head, 9 adapters, zero oracle failures, `W1_reference` the control in every section, committed at 43ce95b3a9). Provisional: 97 of 316 cells. N 32 to 4096 from the focus grid, N 256 to 2048 from the flip bands, the named cells add N 64, 160, 384, 768 and 896. `N5_rows_f` runs in the throughput regime only by design (two rows per call, section 20). `W1_reference` costs 1.3x (fp32 N 32 k 16) to 65x (bf16 N 4096 k 4) the winner\'s latency, median 9x and the middle half 5.5x to 18x, as its linear cost per element predicts, so the `ref` margins at this scope are large everywhere and the runner-up margin is the one that ranks.',
    'subwarp_header': 'Sub-warp scope (the same session, 60 cells all head-to-head, 4 adapters at 4, 8 and 16 lanes, `W1_baseline_sub` the control in every section). The lane count is a tuning dimension chosen per cell like the block size at block scope: every adapter ran at every lane count the cell allows (lanes at most N) and the label carries the winner\'s lane count as `@T4`, `@T8` or `@T16`, the runner-up is often the same adapter at another lane count. Provisional: 23 of 60 cells. N 8, 32 and 128 from the focus grid at k 4 to 64, the named cells add N 16 and 64 and k 2. `N5_rows_fs_sub` runs in the throughput regime only by design. The named sub-warp cells ran at 4 lanes only (their per-cell lists hold no other lane count), while 10.2 specifies logical warps of 8 and 16 for Mixtral, DBRX, gpt-oss and the DeepSeek group select, so those rows are the 4-lane answer and not the shape\'s, an addendum at the specified lane counts on both arches is pending with the evaluation engineer and the rows stay marked until the matrices regenerate.',
    'warp_reading': 'What the warp pass says. W2_packed_v3 takes 42 of 89 `not_guaranteed` latency cells (median +825 percent over W1_reference and +61 percent over the runner-up): every k 4 to 32 cell at N 32 to 512 except fp32 N 32 k 16 (N10_auto, provisional) and fp32 N 512 k 32 (N6w_atom32), bf16 N 1024 k 8 and fp32 N 1024 k 16 (both provisional, the fp32 one a 1.2 percent tie with W5_v4), bf16 N 2048 k 4 (provisional) and bf16 N 4096 k 4, and every MoE router cell of the named table. W5_v4 takes 21 (median +2320 percent, +51 percent): bf16 N 1024 k 16 to 64 (k 16 provisional), fp32 N 1024 k 8, 24 and 32 (k 32 provisional), k 16 to 64 at N 2048 and 4096 under both key types, and fp32 k 4 at N 2048 and 4096 (the latter provisional). N6w_atom32 takes 20 (median +454 percent, +21 percent): k 48 and 64 at N 256 (bf16 and fp32 k 64 provisional), N 512 from k 48 (bf16) or k 32 (fp32) up to k 256, fp32 N 1024 k 48 (provisional) and 64, fp32 N 2048 k 256, and k 256 and 1024 at N 4096 under both key types (fp32 k 1024 provisional against W3_d4c at +13 percent). W3_d4c takes k 1024 at N 2048 under both key types (provisional against N6w_atom32), W4_wmerge bf16 N 2048 k 256, N10_auto fp32 N 32 k 16 and N 128 k 64 (both provisional, W2 takes the other fp32 small-N cells N10 held on the cards), W1x_vimnmx_eager bf16 N 128 k 64 (provisional). `prefer_smaller_index` repeats the ranking: every warp adapter runs the same code under both requirements (the deterministic over unsorted ratio per adapter between 0.998 and 1.002 on the 70 cell pairs), so the 70 cells differ from the `not_guaranteed` matrix only by what was measured, W2_packed_v3 26, W5_v4 20, N6w_atom32 18. Throughput: W5_v4 takes 34 of 87 (median +674 percent, +43 percent), every N 1024 to 4096 cell from k 4 to 64 (bf16 and fp32 N 1024 k 64 and N 4096 k 4 provisional), bf16 N 512 k 16 and 24 (provisional) and fp32 N 512 k 4 and 16, W2_packed_v3 21 at k 4 to 16 up to N 512 where N5_rows_f and N10_auto do not take the cell, N6w_atom32 15 (fp32 from k 24 at N 256 and 512 and N 128 k 64, bf16 N 512 k 48 and 64 provisional, N 2048 k 256 under both key types), N5_rows_f 12 bf16 cells (N 32 k 4 and 16, N 128 k 16 provisional and k 64, N 256 k 24 to 64 all provisional, N 512 k 32 and 256 provisional, N 4096 k 256 and 1024 provisional), W3_d4c fp32 k 1024 at N 2048 and 4096, N10_auto fp32 N 32 k 16 and N 4096 k 256 (provisional), W1x_vimnmx_eager bf16 N 2048 k 1024 (provisional). The four cells where the `prefer_smaller_index` throughput winner differs are N5_rows_f cells that N6w_atom32 or W3_d4c take by 0.3 to 11 percent, inside the provisional band. Representative latency cells (`not_guaranteed`): bf16 N 256 k 8 W2_packed_v3 329 (W5_v4 +101 percent, reference 5950), bf16 N 512 k 16 W2_packed_v3 829 (W5_v4 +29 percent), fp32 N 256 k 16 W2_packed_v3 660 (N6w_atom32 +61 percent), fp32 N 512 k 64 N6w_atom32 1899 (N10_auto +23 percent), fp32 N 1024 k 16 W2_packed_v3 1768 (W5_v4 +1.2 percent), fp32 N 2048 k 16 W5_v4 1987 (W2_packed_v3 +127 percent), bf16 N 1024 k 64 W5_v4 2957 (N6w_atom32 +21 percent), fp32 N 4096 k 1024 N6w_atom32 66140 (W3_d4c +13 percent). Named workloads: W2_packed_v3 takes every MoE router latency cell (Step-3 N 64 k 4 179 cycles, GLM-4.5 N 160 k 8 258, Qwen3 N 128 k 8 262, ERNIE-4.5 N 64 k 8 287, the DeepSeek-V3 final top 8 N 128 k 8 314, Kimi K2 N 384 k 8 441, DeepSeek-V3 N 256 k 8 467, Qwen3.5 N 512 k 10 602, LongCat N 768 k 12 952, Nemotron 3 Ultra N 512 k 22 1070, Kimi K3 N 896 k 16 1314, the runner-up N6w_atom32 or W5_v4 from +21 to +265 percent) and their throughput except Nemotron (W5_v4 554, provisional), W5_v4 takes the N 2048 cells (NSA 128K context 1986 cycles, W2 +127 percent) and the PEER sub-key throughput at N 1024 k 16 (970) where the latency is W2\'s by 1.2 percent, N6w_atom32 the k 64 cells (EAGLE-2 rerank N 1024 k 64 3709, the sampling merge N 512 k 64 1899). Against the wave 2 card picture (2.12): the W2 to W5 flip sits at N 1024 (bf16 k 16, fp32 k 8), W2 to N6w at k 48 (N 256 and 512) or k 32 (fp32 N 512), N6w yields the N 1024 bf16 k 48 and 64 cells and the N 2048 k 64 cells to W5, and N10 keeps two provisional fp32 cells. The 97 provisional cells are where the tie_heavy pass may change this matrix.',
    'subwarp_reading': 'What the sub-warp pass says. Latency: W2_packed_sw_v2 at 16 lanes takes every bf16 cell from N 32 (N 32 k 4 206 cycles, N 128 k 4 342, k 16 785, k 64 2593, its own 8-lane form the runner-up at +54 to +58 percent for k 4 and 16, W1_eager_same_sub at 16 lanes at +24 and +61 percent for N 32 k 16 and N 128 k 64) and the fp32 k 4 cells (N 32 363, N 128 578), N10_auto_sub at 16 lanes takes the fp32 k 16 and 64 cells (N 32 k 16 500, N 128 k 16 1133 provisional against W2 at +8 percent, N 128 k 64 1262), and W1_eager_same_sub at 4 lanes takes N 8 k 4 under both key types (200 and 219, its own 8-lane form at +10 to +12 percent, provisional). More lanes per row won every random latency cell where the lane count was free, the 16-lane form leads the 8-lane form of the same adapter by 40 to 58 percent. The named cells are 4-lane rows: W1_eager_same_sub at N 8 (Mixtral k 2 292 provisional, the DeepSeek group select k 4 219), N10_auto_sub at N 16 k 4 (DBRX 536) and N 64 k 4 (MoBA 1099, provisional over W2 at +5 percent), W2_packed_sw_v2 at N 32 (gpt-oss-20b bf16 k 4 581, the DeepSeek group top 2 650). Throughput inverts the lane count: W2_packed_sw_v2 at 4 lanes takes bf16 N 32 k 4 and fp32 N 8 k 4 and every named throughput cell (its 8-lane form within 2 percent on the bf16 focus cell), at 8 lanes bf16 N 32 k 16 and N 128 k 4 and fp32 N 32 k 4 (the other lane counts within 1 to 13 percent), at 16 lanes fp32 N 128 k 4 (717 Gelem/s), N5_rows_fs_sub at 16 lanes the bf16 N 128 k 16 and 64 cells (604 and 458) and at 8 lanes bf16 N 8 k 4, N10_auto_sub the fp32 N 128 k 16 and 64 cells at 8 lanes and N 32 k 16 at 16 lanes, every one provisional against its own other lane count or W2. `prefer_smaller_index` names the same winner as `not_guaranteed` on every sub-warp cell measured under both. `W1_baseline_sub` is 1.04x (fp32 N 32 k 16 throughput) to 11.6x (bf16 N 128 k 4 latency) the winner. The 23 provisional cells are mostly same-adapter lane-count ties.',
}

R = json.load(open(JSON))['rows']
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

def rows_of(scope, regime, req, key):
    return [r for r in R if r['scope'] == scope and r['regime'] == regime and r['requirement'] == req
            and r['pattern'] == 'random' and r['key_type'] == key and not r['workload']]

def matrix(scope, regime, req, key):
    sel = {(r['n'], r['k']): r for r in rows_of(scope, regime, req, key)}
    ks = [k for k in KS if any((n, k) in sel for n in NS[scope])]
    ns = [n for n in NS[scope] if any((n, k) in sel for k in ks)]
    out = ['| N | ' + ' | '.join(f'k {k}' for k in ks) + ' |', '|---|' + '---|' * len(ks)]
    for n in ns:
        out.append(f'| {n} | ' + ' | '.join(cell(sel[(n, k)]) if (n, k) in sel else '' for k in ks) + ' |')
    return '\n'.join(out), len(sel)

unit = {'latency': 'cycles per call', 'throughput': 'Gelem/s for one occupancy wave'}
def scope_parts(scope, title):
    parts = []
    for regime in ('latency', 'throughput'):
        for req in ('not_guaranteed', 'prefer_smaller_index'):
            for key in ('bf16', 'fp32'):
                t, n = matrix(scope, regime, req, key)
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
wl = [r for r in R if r['workload']]
wrows = ['| scope | workload | key | N | k | regime | requirement | best | runner-up, margin | reference, margin |', '|---|---|---|---|---|---|---|---|---|---|']
for r in sorted(wl, key=lambda r: (SCOPE_ORDER[r['scope']], r['workload'], r['key_type'], r['regime'], r['requirement'])):
    sc = r['scope']
    lab = 'sub-warp' if sc == 'subwarp' else sc
    prov = ' p' if is_prov(r) else ''
    ru = f"{strip_t(r['runner_up'], sc)} {pct(r['margin_runner_up'])}" if r['runner_up'] else ''
    wrows.append(f"| {lab} | {r['workload']} | {r['key_type']} | {r['n']} | {r['k']} | {r['regime']} | {r['requirement']} | {strip_t(r['winner'], sc)} {fmt(r['value'])}{prov} | {ru} | {r['reference']} {pct(r['margin_reference'])} |")
wl_table = '\n'.join(wrows)

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
wins = collections.Counter((r['scope'], r['regime'], strip_t(r['winner'], r['scope'])) for r in R)
wins_rows = ['| scope | regime | adapter | cells won |', '|---|---|---|---|'] + [
    f"| {'sub-warp' if sc == 'subwarp' else sc} | {reg} | {ad} | {c} |"
    for (sc, reg, ad), c in sorted(wins.items(), key=lambda x: (SCOPE_ORDER[x[0][0]], x[0][1], -x[1]))]

def _check(scope, regime, req, adapter, n):
    assert stats[(scope, regime, req)][1][adapter] == n, (scope, regime, req, adapter, stats[(scope, regime, req)][1][adapter], n)
if have['warp']:
    for a, n in (('W2_packed_v3', 42), ('W5_v4', 21), ('N6w_atom32', 20)):
        _check('warp', 'latency', 'not_guaranteed', a, n)
    for a, n in (('W2_packed_v3', 26), ('W5_v4', 20), ('N6w_atom32', 18)):
        _check('warp', 'latency', 'prefer_smaller_index', a, n)
    for a, n in (('W5_v4', 34), ('W2_packed_v3', 21), ('N6w_atom32', 15), ('N5_rows_f', 12)):
        _check('warp', 'throughput', 'not_guaranteed', a, n)
    assert (stats[('warp', 'latency', 'not_guaranteed')][0], stats[('warp', 'throughput', 'not_guaranteed')][0], prov['warp'], cells['warp']) == (89, 87, 97, 316)
if have['subwarp']:
    assert (prov['subwarp'], cells['subwarp']) == (23, 60)
b4 = [r for r in R if r['scope'] == 'block' and r['regime'] == 'latency' and r['requirement'] == 'not_guaranteed' and r['pattern'] == 'random' and r['winner'] == 'B4_v4']
b4_med_ref = statistics.median(r['margin_reference'] for r in b4)
b4_med_ru = statistics.median(r['margin_runner_up'] for r in b4 if r['margin_runner_up'] is not None)
S = lambda scope, regime, req: stats[(scope, regime, req)]

# the phase 3b card tables kept from the current knowledge base
kb = open(KB).read()
a = kb.index('Sorted output, the phase 3b routes on card numbers')
b = kb.index('What the card matrices say')
sorted_block = kb[a:b].rstrip()

out = []
out.append('## 16. Winners (the phase 3 pass on B200, warp, sub-warp and block)')
out.append('')
out.append('What this is. The phase 3 final evaluation: one arch (B200), one node (umbriel-b200-047), one harness version (0.10), one lease session per scope group on 2026-10-25 (block, then warp and sub-warp), random input on the focus grid plus the flip bands, the named workload cells in their own table, `WINNERS_b200.md` and `state/phase3/winners_b200.json` from `tools/winners.py`. Every cell names the best adapter and its value (latency in cycles per call net of the generate-only control, throughput in Gelem/s for one occupancy wave), then the runner-up with the margin (latency: how much slower the runner-up is, throughput: how much faster the best is), then the margin over the reference of the scope measured in the winner\'s own lock section at the winner\'s thread or lane count (`ref`: `B1_reference` at block scope, `W1_reference` at warp scope, `W1_baseline_sub` at sub-warp scope), and in the block `prefer_smaller_index` cells the margin over `B1_det`, the wave 1 deterministic path, as the second reference (`det`). "p" marks a provisional cell, a margin over the runner-up below 15 percent, which the tie_heavy pass decides. Pending after this pass: tie_heavy, the GB200 pass (`WINNERS_gb200.md`), phase 3b (sorted output, whose card tables stay at the end of this section until then). These numbers are not comparable with the wave 1 and 2 card numbers of sections 11 to 23 (harness 0.10 streams every load past L1 and its throughput ceilings are the L2 rate, 2.10). The card-based matrices that stood here before the pass are gone, their numbers stay in the approach sections and the ledger (8) records when each part was replaced.')
out.append('')
if have['warp']:
    out.append(PROSE['warp_header'])
    out.append('')
    for p in scope_parts('warp', 'Warp'):
        out.append(p)
        out.append('')
if have['subwarp']:
    out.append(PROSE['subwarp_header'])
    out.append('')
    for p in scope_parts('subwarp', 'Sub-warp'):
        out.append(p)
        out.append('')
out.append(f'Block scope (71 batches plus the W4_chunk8 addendum, {cells["block"]} cells of which {h2h_all["block"]} head-to-head on random input and {len(runs)} on the sorted-runs track, 13 adapters, zero oracle failures, committed at e62f286f62 and c0e67ea6db). The reference behind `ref` is the control of the winner\'s own section (`tools/winners.py` takes the winner\'s control value), so a cell won at 512 or 128 threads carries `B1_reference@T512` or `@T128`, and the 512-thread reference is up to 28 percent above the 256-thread one on some throughput cells (verify-B4\'s self-test), which is why `ref` margins are not comparable across block sizes. Provisional: {prov["block"]} of the {h2h_all["block"]} head-to-head cells. Adapters at a thread count other than 256 carry `@T128` or `@T512`, the reference then runs at the same thread count. `B1_det3` had no card at the phase 3 go and is not in this pass, the deterministic cells use `B1_det2`. Four `W4_chunk4` batches were quarantined (a manifest mistake fed 8 runs to the 4-run adapter, which violates its contract and failed the oracle) and re-measured as `W4_chunk8`, whose 64 cells are the runs 8 columns of the sorted-runs table. Four notes from the algorithm researcher\'s cold review of these tables (no monotonicity break, no requirement gap, no duplicate spread above 0.5 percent): the B2_packed runner-up values at N 16384 and 32768 are a broken-control artifact, since at 64 and 128 items per thread its fully unrolled list leaves the register file (T34, 14.4), so the plus 68 to 100 percent margins of the two-level list over it at N 16384 and the about 530x at bf16 N 32768 k 4 in the per-cell lists are not competitive margins and the next capable form there is B4_v4 or B1_idx at 512 threads. B4_v4\'s margin over B3_tuned at fp32 N 16384 k 64 and 256 reads 1.74x and 1.78x here against 2.23x on the cards, the difference is the harness 0.10 load path (B3_tuned is a large-shared-memory control whose net dropped, B4_v4 is striped and rose toward its 0.4 value), both same-session under 0.10 so the ranking stands. The 136 single-candidate rows are the W4 sorted-runs track and not head-to-head winners, and one of them is a loss: W4_chunk8 at fp32 N 8192 k 4096 is 28.6 percent slower than B1_reference\'s unsorted selection (18527 against 13238 cycles, 34 percent lower throughput), the k 4096 merge touches the whole row. B4_v4 has two verified cards (`B4_v4_b200_20261025_1` and `_2`, rerun by verify-B4 on umbriel-b200-018 under harness 0.8), and verify-B4 confirmed its phase 3 cells on a second node under harness 0.10 (umbriel-b200-068, 48 cells across N 8192 to 32768 and k 32 to 1024 in all four quadrants, no winner flips, the B4_v4 ratio 068 over 047 between 0.983 and 1.023 with median 1.000, runner-ups 0.965 to 1.040, B1_reference 0.995 to 1.043, `state/phase3/b4v4_confirm_umbriel-b200-068.md`): the closest margin, bf16 N 32768 k 32 over B2_twolevel, narrowed from 14.0 to 8.6 percent and held, and the fp32 N 8192 k 32 near tie stays with N9_w2fin2 by 0.6 percent on both nodes, both cells already provisional for the tie_heavy pass.')
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
out.append('\n'.join(wins_rows))
out.append('')
if have['warp']:
    out.append(PROSE['warp_reading'])
    out.append('')
if have['subwarp']:
    out.append(PROSE['subwarp_reading'])
    out.append('')
out.append(f"What the block pass says. B4_v4 takes {S('block','latency','not_guaranteed')[1]['B4_v4']} of {S('block','latency','not_guaranteed')[0]} `not_guaranteed` latency cells (median {b4_med_ref*100:+.0f} percent over the reference and {b4_med_ru*100:+.0f} percent over the runner-up) and {S('block','latency','prefer_smaller_index')[1]['B4_v4']} of {S('block','latency','prefer_smaller_index')[0]} `prefer_smaller_index` latency cells, every N 8192 and above from k 32 (N 16384) or k 64 (N 8192) up to k 1024, with the two-level B2 list keeping k 4 and 16 at N 16384 and 32768 and B3_tuned keeping k 4096. N 1024 to 4096 is shared by the 128-thread B2 ({S('block','latency','not_guaranteed')[1]['N8_b2packed@T128']} latency cells, k at most 16, and k 32 at N 8192) and N9_w2fin2, the N6 first pass with the packed finish ({S('block','latency','not_guaranteed')[1]['N9_w2fin2']} cells, k 24 to 256, many of them provisional against B1_idx, B1_n6b and B1_det2). Under `prefer_smaller_index` B3_tuned takes {S('block','latency','prefer_smaller_index')[1]['B3_tuned']} latency cells (its scan emission is free of the requirement) and the 128-thread B2 and N9_w2fin2 keep the small N. Throughput: B4_v4 {S('block','throughput','not_guaranteed')[1]['B4_v4']} and {S('block','throughput','prefer_smaller_index')[1]['B4_v4']} cells at large N, B1_idx {S('block','throughput','not_guaranteed')[1]['B1_idx']} and B1_idx at 512 threads {S('block','throughput','not_guaranteed')[1]['B1_idx@T512']} under `not_guaranteed` and B1_det2 {S('block','throughput','prefer_smaller_index')[1]['B1_det2']} under `prefer_smaller_index` at mid N, the 128-thread B2 and the two-level list at small k. Representative fp32 `not_guaranteed` latency: N 1024 k 16 N8_b2packed at 128 threads 1476 (B2_packed plus 10 percent, the reference at 128 threads 2321), N 1024 k 64 N9_w2fin2 1850 (B1_idx plus 2.8 percent, reference 2010), N 4096 k 16 the 128-thread B2 2237 (reference 5012), N 4096 k 256 N9_w2fin2 4003 (B3_tuned plus 4.1 percent, reference 5334), N 16384 k 256 B4_v4 5198 (N9_w2fin2 plus 75 percent, reference 13360), N 32768 k 16 B4_v4 5327 (B2_twolevel plus 63 percent, reference 19415). The sorted-runs track is W4_chunk4's and W4_chunk8's alone (1.3x to 6.2x under the unsorted reference selection at four runs, one loss at eight runs and k 4096), the phase 3b pass sets it against the other sorted routes. The B4_v4 cells at N 8192 and above are confirmed on a second node under harness 0.10 (no winner flips, ratios within 2.3 percent). The {prov['block']} provisional cells are where the tie_heavy pass and the final matrix may differ from this one.")
out.append('')
out.append(sorted_block)
out.append('')
out.append('The pre-phase-3 picture (the card matrices of wave 1 and 2 and the paragraphs that read them) was removed from this section when the passes landed, the card numbers stay in sections 11 to 23 and the ledger (8) has the dates.')
text = '\n'.join(out) + '\n'
if OUT == '-':
    sys.stdout.write(text)
else:
    open(OUT, 'w').write(text)
print('section 16:', text.count('\n'), 'lines, scopes present', have, file=sys.stderr)
for s in ('warp', 'subwarp', 'block'):
    if not have[s]:
        continue
    print(f'== {s}: cells {cells[s]}, head-to-head {h2h_all[s]}, provisional {prov[s]}', file=sys.stderr)
    for regime in ('latency', 'throughput'):
        for req in ('not_guaranteed', 'prefer_smaller_index'):
            n, c = stats[(s, regime, req)]
            print(f'   {regime} {req}: {n} cells, ' + ', '.join(f'{a} {k}' for a, k in c.most_common(8)), file=sys.stderr)
