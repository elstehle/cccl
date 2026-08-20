# DeviceMergeSort: nvbench sweep + kernel-level attribution, main vs PR #10733

End-to-end evaluation of `fix/block-merge-sort-partial-tile` (PR #10733, rebased state) against
same-day `upstream/main` using the official cub nvbench benchmarks
(`cub.bench.merge_sort.{keys,pairs}.base`, full default axes: 7-8 key types × value types ×
OffsetT × 2^16..2^28 × entropy {1.0, 0.201}), plus Nsight Compute on the block-sort kernel.
Node **umb-b200-235, B200 (sm_100, 148 SMs)**; scripts `bms_bench_run.sh`, `bms_profile_run.sh`,
`bms_summary.py`. (Note: pairs_fix was re-run cleanly after its first pass overlapped with a
profiling session on the GPU.)

## Aggregate over all 640 configurations (GPU time, PR vs main)

| | n | mean | median | < −2% | −2..−0.5% | ±0.5% | +0.5..2% | > +2% |
|---|---|---|---|---|---|---|---|---|
| keys | 128 | **−0.06%** | +0.08% | 4 | 17 | 98 | 7 | 2 |
| pairs | 512 | **−0.17%** | +0.05% | 38 | 43 | 348 | 76 | 7 |

Mean slightly PR-favorable, median ≈ 0: **no systematic regression**; 70-77% of configs within
±0.5% (mostly within nvbench's noise floor).

## The tails, and where they come from

* **Wins (up to −5.7%) at 2^24-2^28 on expensive-comparator / small types**: C32 keys −2.3..−3.1%,
  C32 pairs −4.1..−5.7%, I8 keys/pairs −1.3..−4.8%, F64 pairs −0.8..−1.2%. Mechanism: the PR's
  full-tile path (`MergeRounds<false>`) contains **no `valid_items` clamping at all**, whereas
  main computes `min(valid_items, ...)` run boundaries with a runtime parameter on every round —
  removal pays most where each search step's comparator is expensive (complex compares) or tiles
  are large in elements (small types).
* **Losses (up to +3.0%, mostly +1-2.5%) concentrated at 2^16** (a few at 2^20), noise 1-1.6%.
  Mechanism: at 2^16 the sort is 12-24 tiles, so the single last tile is a large fraction of the
  work, and the benchmark's power-of-two sizes leave it *nearly full* — the exact shape where the
  collective-level eval measured overload B at +6-7% vs the old clamped implementation
  (BMS_EVAL_RESULTS.md). +7% × ~1/12 of tiles ≈ the observed +0.5-2.5%. The flip side — B's
  −9..−30% latency / up to +60% throughput on genuinely partial tiles — is not exercised by
  power-of-two benchmark sizes.

## Kernel-level attribution (ncu, F32 keys, 2^28, 61681×256 blocks)

| DeviceMergeSortBlockSortKernel | main | PR | delta |
|---|---|---|---|
| Elapsed cycles | 4,224,869 | 4,245,727 | **+0.49%** |
| Duration | 2.16 ms | 2.17 ms | ≈noise |
| Registers/thread | 40 | 40 | 0 |
| Occupancy theor./achieved | 75% / 74.10% | 75% / 74.09% | 0 |

The block-sort kernel is **structurally unchanged** by the PR at device scale: identical register
allocation and occupancy, +0.49% cycles. Consistency check: +0.49% × block-sort's ~¼ share of
end-to-end time ≈ the +0.12% observed at F32 2^28 in the sweep — the attribution closes.

## Verdict for the PR

1. No systematic performance change across 640 configurations (mean −0.06%/−0.17%, median ≈ 0);
   the block-sort kernel itself carries identical resources and sub-1% cycles.
2. Real wins on expensive-comparator and small key types at scale (up to −5.7%) from the
   clamp-free full-tile path — a side benefit of the correctness restructuring.
3. Small-size (2^16) costs of +1-3% on nearly-full last tiles are the one measurable price;
   they are bounded by tile-count dilution at larger sizes, borderline vs noise at these, and
   accompanied by large improvements on genuinely partial last tiles that this power-of-two
   sweep cannot show (see BMS_EVAL_RESULTS.md for the valid_items spectrum).

Reproduce: `bms_bench_run.sh` (clones/configures/builds/runs/compares; requires
-DCCCL_ENABLE_CUB=ON -DCCCL_ENABLE_BENCHMARKS=ON and Thrust enabled — benchmarks depend on
nvbench_helper), `bms_profile_run.sh` (nvbench --profile under ncu/nsys), `bms_summary.py`
(aggregate stats from nvbench JSONs).
