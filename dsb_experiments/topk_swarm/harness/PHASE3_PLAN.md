# Phase 3 plan: final evaluation

Status: plan and scripts ready, no measurement started. The lead gives the go.

Everything below is driven by `harness/phase3_adapters.json` (the adapter manifest), generated into
node scripts by `tools/phase3.py`, and reduced to the winners matrix by `tools/winners.py`. The
scripts run on the Mac and drive the node over ssh exactly like every batch so far: ship, build
outside the lock, pinned lease, batches in order with canaries, fetch after every adapter, release.

## 1. What is measured

Every winner and runner-up per scope from `RESULTS_BOARD.md` and knowledge base section 16, under
the current harness (0.10 at the time of writing: streaming loads, strided input subsets, control
with the variant's shared memory, 23-case oracle battery with the signed-zero cases), on one healthy
B200, both requirements, both regimes, canaries first and last of every batch, each scope's
reference adapter as the control in the same lock section.

Adapters and source commits (all on `exp/topk-primitives-swarm`, commit of the last change to the
adapter header, the approach sources it includes ship with it):

| scope | adapter | commit | control | includes |
|---|---|---|---|---|
| block | B2_packed | a3f4d37ebf | B1_reference (bb116b9775) | |
| block | N8_b2packed at 128 threads | 2e1fc95203 | B1_reference | |
| block | B2_twolevel | 10d232003a | B1_reference | |
| block | B1_idx at 256 and at 512 threads | f577f3815a | B1_reference | approaches/B1 |
| block | B1_det2 | d22434f718 | B1_reference | approaches/B1 |
| block | B3_tuned | f269717c45 | B1_reference | approaches/B3 |
| block | B4_v4 | b1f8ce8d24 | B1_reference | approaches/B4 |
| block | B1_n6b | 02fdf9dbdc | B1_reference | approaches/B1 |
| block | N9_w2fin2 | d55e2cd430 | B1_reference | approaches/B4 |
| block | W4_chunk4, runs 4 and runs 8, N 8192 and above | ea9daa2283 | B1_reference | |
| warp | W2_packed_v3 | dfeeab4138 | W1_reference (bb116b9775) | |
| warp | W5_v4 | dfd5a1ee14 | W1_reference | |
| warp | W1x_vimnmx_eager | babb055d8a | W1_reference | approaches/W1 |
| warp | W1_level | ccc1166d9c | W1_reference | approaches/W1 |
| warp | N10_auto | e2bb6eccd5 | W1_reference | approaches/N10 |
| warp | N5_rows_f, throughput only | 30f5b8f821 | W1_reference | approaches/N5 |
| warp | N6w_atom32 | fdb4777590 | W1_reference | |
| warp | W3_d4c | b01bb21347 | W1_reference | |
| warp | W4_wmerge | ea9daa2283 | W1_reference | |
| subwarp | W2_packed_sw_v2 | 28239d1a4b | W1_baseline_sub (3fa70dd18f) | |
| subwarp | W1_eager_same_sub | 3fa70dd18f | W1_baseline_sub | |
| subwarp | N10_auto_sub | e2bb6eccd5 | W1_baseline_sub | approaches/N10 |
| subwarp | N5_rows_fs_sub, throughput only | 8f613a317f | W1_baseline_sub | approaches/N5 |

The references do not support `prefer_smaller_index`, so every prefer_smaller_index batch runs with
`--control-req not_guaranteed` (the reference's not_guaranteed path in the same lock section, control
name `<reference>@not_guaranteed`), the not_guaranteed batch of the same cells gives the reference
under its own requirement. The card's `commit` for every phase 3 card is the swarm branch HEAD at
shipping time (written to `harness/SOURCE_COMMIT` by the script), `clone_commit` the node's clone.

Cells: `cells_focus.json` (block 84 cells, warp 76, sub-warp 68, both requirements), `cells_flipbands.json`
(block 84, warp 76, no sub-warp band), `ideas/named_cells.json` (block 24, warp 19, sub-warp 6,
`moe_step3` at N 48 skipped as not a multiple of 32), bf16 and fp32, int32 indices, index-only
output, random input. Per adapter: block 192, warp 171, sub-warp 74 cells, each a translation unit
with variant and control and one latency plus one throughput measurement. Second pass: the
`tie_heavy` pattern on every cell where the winner's margin over the runner-up is below 15 percent
(`tools/winners.py --close` writes the list, `tools/phase3.py --pass tie_heavy --cell-list` runs it).

## 2. Batch order and estimated node time

Per cell under the lock (measured on the 0.6 to 0.8 batches: canary, oracle battery of 27 cases,
latency two sweeps of five chain lengths for variant and no-op, same for the control, throughput
six launches for each of the four kernels): about 1 s at N up to 1024, 2 to 3 s at N 4096, 5 to
10 s at N 16384 and 32768 (block) and N 2048 and 4096 (warp, where W1 kernels spill). Compile per
translation unit with 14 parallel jobs: 5 to 10 s for AIR-sized kernels, up to 16 s for W2_packed,
60 s for W1 at N 4096.

| step | batches (run.py calls) | cells | compile, outside the lock | lock time |
|---|---|---|---|---|
| block random pass (12 adapter entries, 3 cells files, 2 requirements) | 66 | about 2200 | about 45 min | about 95 min |
| warp random pass (9 adapters) | 54 | about 1540 | about 40 min | about 60 min |
| sub-warp random pass (4 adapters) | 26 | about 300 | about 10 min | about 10 min |
| tie_heavy pass on the close cells (expected 20 to 40 percent of cells) | about 60 | about 1000 | reuses the binaries | about 40 min |
| GB200 pass: warp and sub-warp on every cell, block at N 1024 and 4096 | about 110 | about 2700 | about 60 min on 144 cores | about 90 min |
| phase 3b sorted output on the focus set, 9 adapters, `--check-sorted` | 18 | about 900 | about 15 min | about 25 min |

Compile and measurement of one node do not overlap: compile everything first (no lease needed,
the node is held for us), then one lease for the measurement. The block random pass alone needs a
lease of about 2 hours including fetches, the warp plus sub-warp passes about 1.5 hours, so the
B200 random pass is two lease sessions on the same node (block, then warp and sub-warp), each with
its own canaries, which the brief allows since every number carries its control from its own lock
section and the matrix compares adapters within a scope. The tie_heavy pass is a third session.

Order: block first (longest kernels, most sensitive to node health), then warp, then sub-warp, then
the tie_heavy pass on the close cells of all three scopes, then the GB200 pass, then 3b on the B200.
Within a scope the adapters run in the manifest order with the reference measured in every batch.

## 3. Node choice rule

- B200 passes: one computelab node other than umbriel-b200-020 (retired, T58), with at least 3.5
  hours of lease left when the measurement starts, held for the evaluation engineer alone for the
  whole session (`lease.py acquire --node <name> --min-remaining 3h`). The hardware researcher runs
  a short pre-flight on it first (canary cycles, the T58 large-kernel check, compute mode Default,
  no co-tenant job on the host if that can be seen). The node must have the harness compiled before
  the lease starts, so the broker names it at least 90 minutes ahead.
- GB200 pass: one dlcluster node with the same rule, after the B200 random pass is reduced.
- The reference control is measured in every batch, so a node drift inside a session shows as a
  moving control, and the canaries bracket every batch. If the canary or the control moves by more
  than 3 percent between batches, the session stops and the broker provides a replacement node,
  and the scope restarts on it (no mixing of nodes inside a scope).
- No cross-node and no cross-arch comparison in the matrix: the winners table carries the node and
  the harness version per cell, and the GB200 pass gets its own `WINNERS_gb200.md`.

## 4. Output: the winners matrix

`tools/winners.py --results state/results/eval-engineer/phase3/<arch> --arch <arch> --out WINNERS_<arch>.md --json state/phase3/winners_<arch>.json --close state/phase3/close_cells_<arch>.json`

One table per scope, regime, requirement and input pattern. Per cell (key type, N, k, workload
name for named cells): the winning adapter and its value, the runner-up and the margin over it, the
reference (the winner's control from its own lock section) and the margin over it, the number of
candidates, the winner's registers, the node. Margins: latency `(other - winner) / winner`,
throughput `(winner - other) / other`. Adapters at a non-default thread count or with sorted runs
are labeled `id@T<threads>` and `id@runs<M>`. A summary table counts cells won per adapter, scope
and regime. The JSON has every candidate's value per cell for the heatmaps. The close-cells file is
the `--cell-list` input of the tie_heavy pass.

Cards: one card per adapter and arch from the phase 3 batches (`make_card.py` with the batch files of
that adapter, `--source-commit` the shipped HEAD), findings written by the evaluation engineer from
the matrix, verification by the usual verifiers on the same node from the same binaries.

## 5. What the broker provides

1. One B200 node (not umbriel-b200-020) dedicated to the evaluation engineer for the whole B200
   session: named 90 minutes before the lease, so the compile finishes first, then a lease of at
   least 3.5 hours for the block pass and a second one for warp, sub-warp and tie_heavy (or one
   8 hour hold if the cluster grants it). No other holder queued on it during the session.
2. The hardware researcher's pre-flight result for that node before the lease.
3. Later, one GB200 node under the same rule for the GB200 pass (about 2.5 hours including
   compile on 144 cores), and the B200 node again (or an equivalent) for phase 3b.
4. A replacement node on short notice if the canaries or the control drift during a session.

## 6. Harness items done for phase 3

- `run.py --cell-list` (explicit cells from the close-cells file), `--check-sorted` (the oracle also
  verifies the output order for 3b), `--named-cells`, `--runs`, `--control-req`, batch meta carries
  `harness_version` and `source_commit`.
- `tools/phase3.py` (script generator), `tools/winners.py` (matrix), `tools/compare_batches.py`
  (node and version comparisons for the verifiers), `tools/summarize_batch.py`.
- Open: `moe_step3` (N 48) needs a partial-tile contract or padding to 64 before it can be a named
  cell. N5's two-row adapters measure throughput only by design. W4_chunk4 only runs on sorted-runs
  input (runs 4 and 8), so it appears in the block tables at N 8192 and above with the `@runs` label
  and competes against adapters measured on random input of the same cells only where the manifest
  puts both on the same cells file (the sorted-runs input is a different cell in the matrix and is
  reported as such, `input_pattern` plus `runs`).

## 6.1 Adjustments at the go (lead, after commit 91fac66081)

- B1_det3 had no card on the swarm branch at the go and is skipped.
- B1_det (the pre-fix deterministic AIR path, both requirements) is a second control of every
  prefer_smaller_index batch of the block pass (`psi_second_control` in the manifest): each such
  cell is measured twice, once against `B1_reference@not_guaranteed` and once against `B1_det`,
  and `tools/winners.py --second-reference B1_det` prints B1_det's value and the margin over it as
  an extra reference column so the deterministic price is in the matrix. The variant's duplicate
  measurement from the B1_det batches is not a candidate. This adds 36 batches and about 50
  minutes to the block pass.
- The harness is pinned at 0.10 for the whole of phase 3, every batch records the version and the
  node, and `WINNERS_<arch>.md` prints both.

## 6.2 Execution log and addenda (2026-10-25)

- Block pass: umbriel-b200-047, 17:47Z to 19:02Z, 71 batches plus the W4_chunk8 addendum (the manifest
  had W4_chunk4 on 8-run input, a contract mismatch the oracle caught, those batches are quarantined under
  `state/results/eval-engineer/phase3/b200/rejected_contract_mismatch`). Compile and measurement
  overlapped (two build chains, measurement starts per adapter on its build marker), which the
  cycle and event-bracketed metrics tolerate. B1_det ran as a candidate of the prefer_smaller_index
  cells instead of as a second control of 36 batches (compile budget), `winners.py --second-reference`.
- Warp and sub-warp pass: umbriel-b200-047, 19:09Z to 20:50Z, 57 batches with numbers (named cells carry
  not_guaranteed only, the sub-warp scope has no flip bands).
- Addenda after the go: B1_det3 as a prefer_smaller_index candidate (random and random@runs4), and the
  named sub-warp cells at every valid lane count (4, 8, 16: the first pass ran them at 4 lanes only,
  run.py now expands a named sub-warp cell without an explicit lane count to all fitting sizes).
- Phase 3b runs the `--check-sorted` oracle over the full battery, which is how verify-W4 found that
  W2_packed fp32 (N4 truncation) and W5_v4 (tie_heavy, quantized_random) are not sorted.
- Network outage: the Mac's tunnel to the clusters dropped at 21:26Z with the tie_heavy pass at 7 of 51
  batches on umbriel-b200-035 and the GB200 pass at 8 of 141 on gb200-nvl4-ts2-105. `state/phase3/resume.sh`
  waits for name resolution and runs `tools/phase3_resume.py`: reuse a ready dedicated node per arch,
  otherwise request one through the broker, then per arch in order GB200 warp and sub-warp, named
  sub-warp addendum, GB200 block (N 1024 and 4096); tie_heavy, B1_det3 addendum, named sub-warp
  addendum, 3b. Builds skip existing binaries, measure scripts skip batches already measured
  (`run.py --skip-if-done`), batches resumed on a different node carry a `note` in their meta and the
  matrices print the node per cell and per table.

- Resumed run (2026-10-26, tunnel back at 03:46Z): tie_heavy pass on umbriel-b200-068 (51 batches,
  388 cells, done 06:36Z), B1_det3 addendum on 068 (done 06:42Z), named sub-warp addendum at 8 and
  16 lanes on 068 (done 06:50Z), phase 3b on the successor umbriel-b200-047 (18 batches, done
  06:3xZ), GB200 warp and sub-warp pass on gb200-nvl4-ts2-99 (57 batches, done 05:31Z), named
  sub-warp addendum (05:40Z) and block cells at N 1024 and 4096 (done 06:27Z) on the same node.
  The partial pre-outage batches of 035 and ts2-105 are under `superseded_outage/` and are not
  matrix inputs, so no table mixes nodes within a pass. Final matrices: `WINNERS_b200.md` 1480
  cells (random, tie_heavy, and the sorted-runs tracks), `WINNERS_gb200.md` 666 cells.

- Matrix integrity (2026-10-26 07:00Z to 07:10Z, curator review): exact batch labels, the 3b pass as
  its own pattern (`random@sorted`), the one-node random tables kept on umbriel-b200-047 with the
  068 addenda (B1_det3, 8 and 16 lane named sub-warp cells) as `random@addendum:umbriel-b200-068`
  tables, B1_det3 as a reference column on the sorted-runs track, N8_b2packed reported as
  B2_packed@T128, the 3b section classifying unlaunchable and by-design-unsorted adapters as
  unsupported. The det3 cell list for the curator is `state/phase3/det3_cells_b200.md`.
- Shuffled tie_heavy run (harness 0.11, pattern `tie_heavy_shuffled`, label `phase3_b200_tieshuf`):
  umbriel-b200-094, 07:17Z to 08:29Z, block adapters on the 90 close cells at N 8192 and above plus
  bf16 N 1024, both requirements, 21 of 27 batches measured when the lead's 08:30Z stop came (the
  W4 chunk merges on shuffled ties and the B1_det3 sorted-runs reference were cut, nothing of the
  prediction test). The same-node rerun of the two addenda on 047 did not happen (047 never came
  back), so the addendum tables keep their 068 node name.

## 7. Go procedure

1. Lead gives the go, broker names the B200 node and starts the pre-flight.
2. `python3 tools/phase3.py --node <node> --exec-prefix "<prefix>" --arch b200 --pass random --scopes block --out state/phase3/phase3_b200_block.sh` and the same with `--scopes warp,subwarp`, review, run the block script (it ships, builds, acquires the pinned lease, measures, fetches, releases), then the warp and sub-warp script.
3. `tools/winners.py` on the fetched batches, review `WINNERS_b200.md`, then the tie_heavy pass with `--pass tie_heavy --cell-list state/phase3/close_cells_b200.json`.
4. GB200 pass with `--pass gb200`, then `WINNERS_gb200.md`.
5. Phase 3b with `--pass 3b` on the B200, reduced with `tools/winners.py --pattern random` restricted to the sorted adapters (the sorted check is in every cell's oracle result).
6. Cards per adapter and arch, board, knowledge base winners section.
