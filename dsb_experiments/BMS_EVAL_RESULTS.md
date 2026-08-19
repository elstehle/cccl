# BlockMergeSort partial-tile eval: upstream/main vs PR #10733's two overloads

Three-way performance evaluation for `fix/block-merge-sort-partial-tile` (review-addressed state:
+1 padding, identity-round early exit, unified round driver): **BASE** = today's `upstream/main`
`Sort(..., valid_items, oob_default)` (old clamped implementation), **FIX-A** = branch same
signature (pad-then-sort-full, contract-delivering), **FIX-B** = branch `Sort(..., valid_items)`
(no sentinel, clamped + early exit). Node **umbriel-b200-037, B200 (sm_100, 148 SMs)**; harness
`proto_bms_eval.cu` (real headers, dual build, `merge_fix/` shadow over main).
`BlockMergeSort<float, 256, IPT[, int]>`, IPT 1..8, `oob_default = +inf`. Correctness gate
(sorted valid prefix) passes everywhere on both builds; FIX-A additionally delivers
suffix == oob_default at every (tile, valid) — BASE does not for mid/near-full partials
(the documented stock gap, reconfirmed against today's main).

## Latency (single-block slope cyc/sort, gen-subtracted; keys / pairs)

| tile, valid | BASE-A | FIX-A | FIX-B | B vs BASE-A |
|---|---|---|---|---|
| 256, 256 (full) | 5395 / 5663 | 5583 / 5708 | 5923 / 6114 | +10% / +8% |
| 256, 32 | 3310 / 3536 | 5485 / 5618 | **3010 / 3141** | **−9% / −11%** |
| 512, 512 | 5951 / 6356 | 6068 / 6321 | 6257 / 6540 | +5% / +3% |
| 512, 64 | 3606 / 3958 | 5901 / 6206 | **3172 / 3375** | **−12% / −15%** |
| 1024, 1024 | 7948 / 8989 | 7907 / 8913 | 8398 / 9030 | +6% / +0.5% |
| 1024, 128 | 5161 / 5905 | 7872 / 8995 | **4506 / 4800** | **−13% / −19%** |
| 2048, 2048 | 11688 / 14979 | 11807 / 14951 | 12554 / 15314 | +7% / +2% |
| 2048, 1027 | 10873 / 14388 | 12379 / 16763 | **11348 / 13537** | +4% / **−6%** |
| 2048, 256 | 8553 / 11419 | 12806 / 17954 | **7003 / 8023** | **−18% / −30%** |

## Throughput (8 waves, Gelem/s of VALID elements; keys / pairs)

| tile, valid | BASE-A | FIX-A | FIX-B |
|---|---|---|---|
| 256, 256 | 38.5 / 36.1 | **39.4 / 37.1** | 35.4 / 33.5 |
| 256, 32 | 9.4 / 8.3 | 5.1 / 4.8 | **12.0 / 10.8** |
| 512, 512 | 64.1 / 57.4 | **64.8 / 57.8** | 61.1 / 56.9 |
| 512, 64 | 14.6 / 12.6 | 8.4 / 7.5 | **17.6 / 16.4** |
| 1024, 1024 | 84.7 / 61.0 | **85.0 / 64.8** | 84.5 / 60.0 |
| 1024, 128 | 19.5 / 13.0 | 10.4 / 7.6 | **26.6 / 17.6** |
| 2048, 2048 | 83.6 / 53.9 | **85.8 / 54.7** | 84.6 / 53.1 |
| 2048, 1027 | 51.7 / 29.4 | 36.4 / 22.9 | **59.2 / 32.5** |
| 2048, 256 | 17.5 / 9.4 | 8.2 / 5.1 | **27.9 / 14.9** |

Resources: all three comparable (no spills; B occasionally leaner, e.g. 48 vs 63 regs keys @ tile
1024; pairs @ 2048: 90-95 regs, occ 2 blk/SM for all).

## Reading

1. **Full and near-full tiles: the fix is free.** All three implementations agree within ±3-7%
   latency and ±3% throughput at valid = N and N−3; FIX-A is even marginally the throughput
   leader at full tiles. No regression for the dominant use case.
2. **Overload A now pays for its contract on partial tiles — by design.** Pad-then-sort always
   sorts the full tile, so at valid = N/8 it is ~1.5-1.6× slower than the old implementation and
   ~half the valid-normalized throughput. That is the price of "suffix == oob_default" actually
   holding (it never held before). Callers that don't need the suffix have overload B.
3. **Overload B beats even the old implementation on partial tiles** — the early exit pays:
   −9..−18% keys / −11..−30% pairs latency at valid = N/8 and up to +60% valid-normalized
   throughput (27.9 vs 17.5 @ 2048/256). Small-partial tiles are exactly the small-input
   `DeviceMergeSort` shape, and the PR already routes `AgentBlockSort` through B.
4. **Guidance for the PR text**: full tiles unchanged; A = contract (needs a sentinel, pays
   full-tile cost regardless of valid_items); B = performance for prefix-only consumers,
   strictly better than the status quo on partials.
5. Secondary observation: FIX-A latency *rises* as valid shrinks at large tiles (12806 vs 11807
   keys @ 2048) — the +inf-saturated suffix creates massive tie regions whose data-dependent
   search/merge paths are slower than random data. An implementation detail of pad-then-sort
   worth knowing, dwarfed by point 2's structural cost.
6. BASE full-tile numbers differ from the earlier wms_static study's stock measurements
   (5395/5951/7948/11688 today-main/node-037 vs 5473/6969/8940/12604 exp-branch/node-017) —
   consistent with main having moved and the family's known toolkit/codegen sensitivity;
   within-run comparisons are unaffected.

Reproduce: build `proto_bms_eval.cu` twice — `-I<main>` (BASE) and `-DFIXED=1 -Imerge_fix
-I<main>` (FIX) — and run `[correct|lat|thr|res|all]`.
