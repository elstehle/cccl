# Benchmark bug: `direction_param` silently picks `select::min`

> **TL;DR.** All prior batched TopK benchmark numbers in
> `topk_perf_tracking/bench/` were taken against a binary that silently
> ran `select::min` (top-k *smallest*) for the batched dispatch even
> though the source asks for `select::max` (top-k *largest*) and
> `main`'s dispatch correctly ran `select::max`. This is the root cause
> of the entropy=0.201 / large-N "structural gap" I previously
> attributed to grid-not-shrinking-between-passes -- the two pipelines
> were simply computing different things.
>
> After the one-line fix, on the I32/I32 / 2^28 / sel=2^8 / ent=0.201
> workload `lift_part` runs at **521.15 us** vs `main`'s **438.72 us**
> (1.19x) -- the same workload was previously measured at **1.08 ms**
> for batched / **438 us** for main (2.48x).

## How we found it

While instrumenting the batched filter / last_filter agents with
per-pass `printf`s (gated by `-DCUB_DETAIL_BATCHED_TOPK_DEBUG_PRINTF=1`)
on the `I32/I32, 2^28, sel=2^8, ent=0.201` workload, the per-pass state
showed:

```
[batched_topk filter_run_first pass=1 ... in_len=268435456 curr_len=6105069 load_from_cand=0]
[batched_topk filter_run_first pass=2 ... in_len=268435456 curr_len=4304842 load_from_cand=0]
[batched_topk last_filter_run_first pass=3 ... in_len=268435456 ...]
```

`curr_len` shrinking from `6,105,069` -> `4,304,842` between passes
1 and 2, but `in_len` and the grid both staying at the full input
size. The pass-N kernel is **scanning the full input** even though
the kth bucket has narrowed to a few million candidates.

Instrumenting `main`'s `agent_topk` similarly (different macro:
`CUB_DETAIL_TOPK_DEBUG_PRINTF=1`) gave a very different picture for
the same workload:

```
[topk histogram pass=0] num_items=268435456
[topk choose_bucket pass=0] kth_bucket=79 prev=247 cur=256 -> counter.k=9 counter.len=9
[topk filter_hist pass=1] current_k=9 current_len=9 ... is_last_pass=0
[topk filter_hist pass=2] current_k=9 current_len=0 ... is_last_pass=1
[topk last_filter   pass=3] current_len=0 counter.len=0 ...
```

Same workload, completely different bucket distribution:

- `main` finds the kth bucket at **bin 79**, with **9 items** in it
  (247 already selected from preceding bins).
- `batched` finds the kth bucket at **bin 0**, with **6,105,069 items**
  in it (0 already selected).

Both pipelines use the same `extract_bin_op_t<KeyT, SelectDirection,
BitsPerPass, ...>`, same `bits_per_pass=11`, same `total_bits=32`.
The only way the histograms diverge for the same input is if
`SelectDirection` differs. Inspecting the kernel mangled names in the
nsys trace confirmed it:

| pipeline | SelectDirection in kernel template |
|---|---|
| `main`'s `DeviceTopKKernel` (and friends) | `(select)1` = `select::max` |
| `batched`'s 7 large-segment kernels | **`(select)0` = `select::min`** |

The benchmark source asks for `select::max` in both cases.

## The root cause

In `cub/cub/detail/segmented_params.cuh`:

```43:55:cub/cub/detail/segmented_params.cuh
template <typename T, T... Options>
struct uniform_discrete_param
{
  using value_type          = T;
  using supported_options_t = supported_options<T, Options...>;

  T value;

  _CCCL_HOST_DEVICE constexpr uniform_discrete_param(T v)
      : value(v)
  {}

  uniform_discrete_param() = default;
```

The default constructor leaves `value` **uninitialized** -- even when
the template only allows a single Option (e.g. `select_direction_static<
select::max>` aliases to `uniform_discrete_param<select, select::max>`,
which constrains `value` to a single legal value but doesn't initialise
it to that value).

In `cub/benchmarks/bench/topk/pairs.cu` (and `keys.cu`):

```cpp
cub::detail::batched_topk::select_direction_static<cub::detail::topk::select::max> direction_param{};
```

The `{}` invokes the default constructor -> `direction_param.value` is
**uninitialised garbage**.

The dispatch reads it at runtime:

```cpp
const auto direction_value = select_directions.get_param(num_segments_val_t{0});
if (direction_value == detail::topk::select::min)
{
  // instantiate kernels with select::min ...
}
else
{
  _CCCL_ASSERT(direction_value == detail::topk::select::max, "select_directions value not in the supported list");
  // instantiate kernels with select::max ...
}
```

Stack memory for `direction_param.value` happens to be zero in
practice on this configuration -> compares equal to `select::min`
(which has enum value 0) -> the `if` branch fires and **the batched
pipeline silently runs `select::min`**.

The `_CCCL_ASSERT` does *not* fire because the value reads as a valid
enum (`select::min == 0`). It only fires when the value is neither
`min` nor `max`.

## The fix

One-liner per bench file: pass the value explicitly.

```cpp
// cub/benchmarks/bench/topk/pairs.cu (and keys.cu)
cub::detail::batched_topk::select_direction_static<cub::detail::topk::select::max> direction_param{
  cub::detail::topk::select::max};
```

This matches the pattern of the other params in the same file
(`segment_sizes_param`, `k_param`, `num_segments_param`,
`total_items_param`), which all pass the value to the constructor.

A longer-term fix would be in `uniform_discrete_param` itself --
when the template's `Options...` pack has size 1, the default
constructor should initialise `value` to that single option (or the
default constructor should be `= delete`d, forcing callers to pass
the value). I have not made that change here -- it should be a
separate CL, since it might affect other call sites.

## Impact

The kernel-by-kernel trace on `I32/I32, 2^28, sel=2^8, ent=0.201`,
captured with `nsys profile --trace=cuda` and `nvbench --profile`
(single dispatch), reads as follows.

### `main` (always `select::max`, both before and after the fix)

- kernels: **4**
- sum: **438.72 us**
- wall-time span: **448.51 us**

| # | start (us) | dur (us) | kernel |
|---:|---:|---:|---|
| 1 | 0.00 | 232.19 | `DeviceTopKHistogramKernel` |
| 2 | 232.45 | 202.37 | `DeviceTopKKernel` (pass 1) |
| 3 | 435.07 | 2.08 | `DeviceTopKKernel` (pass 2) |
| 4 | 446.43 | 2.08 | `DeviceTopKLastFilterKernel` |

### `lift_part` (this CL) **before** the bench fix -- silently `select::min`

- kernels: **7**
- sum: **1076.87 us**
- wall-time span: **1079.01 us**

| # | start (us) | dur (us) | kernel |
|---:|---:|---:|---|
| 1 | 0.00 | 235.65 | `device_segmented_topk_histogram_kernel` |
| 2 | 235.94 | 4.22 | `device_segmented_topk_finalize_histogram_kernel` |
| 3 | 240.42 | **237.86** | `device_segmented_topk_filter_kernel` (pass 1) |
| 4 | 478.82 | 4.22 | `device_segmented_topk_finalize_filter_kernel` |
| 5 | 483.58 | **232.67** | `device_segmented_topk_filter_kernel` (pass 2) |
| 6 | 716.51 | 4.03 | `device_segmented_topk_finalize_filter_kernel` |
| 7 | 720.80 | **358.21** | `device_segmented_topk_last_filter_kernel` |

### `lift_part` (this CL) **after** the bench fix -- correctly `select::max`

- kernels: **7**
- sum: **521.15 us** (**-555.72 us / -52%**)
- wall-time span: **523.30 us**

| # | start (us) | dur (us) | kernel |
|---:|---:|---:|---|
| 1 | 0.00 | 234.40 | `device_segmented_topk_histogram_kernel` |
| 2 | 234.66 | 4.13 | `device_segmented_topk_finalize_histogram_kernel` |
| 3 | 239.07 | 228.00 | `device_segmented_topk_filter_kernel` (pass 1) |
| 4 | 467.62 | 2.88 | `device_segmented_topk_finalize_filter_kernel` |
| 5 | 471.04 | **29.02** | `device_segmented_topk_filter_kernel` (pass 2) |
| 6 | 500.32 | 1.95 | `device_segmented_topk_finalize_filter_kernel` |
| 7 | 502.53 | **20.77** | `device_segmented_topk_last_filter_kernel` |

### Per-phase side-by-side (after bench fix)

| phase | main (us) | lift_part (us) | delta | factor |
|---|---:|---:|---:|---:|
| pass 0 histogram          | 232.19 | 234.40 + 4.13 = **238.53**  | +6.34 | 1.03x |
| pass 1 (filter + finalize)| 202.37 | 228.00 + 2.88 = **230.88**  | +28.51 | 1.14x |
| pass 2 (filter + finalize)|   2.08 | 29.02 + 1.95 = **30.97**    | +28.89 | 14.9x |
| pass 3 last_filter        |   2.08 | **20.77**                   | +18.69 | 9.99x |
| **sum**                   | **438.72** | **521.15**              | **+82.43 (+19%)** | **1.19x** |

After the fix, the per-pass behaviour matches main:

- Both find the kth bucket at bin 79 with 9 candidates in pass 0.
- Both early-stop in pass 1 (after writing 247 already-selected items
  + 9 ties).
- Both make pass 2 and `last_filter` near-no-ops (empty=true).

The remaining gap (~82 us / ~19%) is the structural launch overhead
of the 7-kernel batched pipeline vs the 4-kernel single-problem
pipeline:

- 3 extra finalize-kernel launches (4.13 + 2.88 + 1.95 = 8.96 us).
- ~25 us of additional overhead in batched's pass 1 (filter +
  early-stop write), which is doing the same logical work as main's
  pass 1 (`DeviceTopKKernel`) but in a different shape.
- ~30 us in batched's pass 2 + ~20 us in last_filter even though
  they're empty -- per-CTA `resolve_segment_state` + the smem
  histogram init runs unconditionally. Main's pass 2 / last_filter
  short-circuit earlier.

These are all **launch-overhead / structural costs**, not
algorithm-shape costs (which is what we'd be in for if the input
weren't shrinking between passes -- which is what the broken bench
made it look like).

## What this means for the prior numbers

All the sweeps under `topk_perf_tracking/bench/sweep_*_i32i32.json`
were measuring `MaxPairs(main)` vs `MinPairs(batched, accidentally)`.
That likely also explains the worst-case ratios reported in
`reports/sweep_i32_i32_lift_partition_vs_main.md` and the earlier
catastrophic ent=0.000 numbers attributed to `cand_reserve_open` not
surviving across tiles.

The `cand_reserve_open` lifetime fix in this branch is still
valid (it's a correctness-relevant improvement for true entropy=0
all-ties cases on `MaxPairs`), but its measured impact in the
previous sweeps may have been partially attributable to the
direction mismatch. I will rerun the sweep with the bench fix in
place and compare.

## Artifacts

- `topk_perf_tracking/profile/profile_main_v2_2p28_ent0p201.nsys-rep`
- `topk_perf_tracking/profile/profile_lift_part_v2_2p28_ent0p201.nsys-rep`
  (also `profile_lift_part_i32_2p28_sel2p8_ent0p201.nsys-rep` = pre-fix)
- `topk_perf_tracking/profile/trace_{main_v2,lift_part_v2}_2p28_ent0p201.csv`
- `topk_perf_tracking/profile/debug_printf_*_2p28_ent0p201*.log`
- branches: `tmp/perf-eval-debug-printf` (batched-side printfs +
  bench fix), `tmp/perf-eval-debug-printf-main` (main-side printfs)
- final commit on `exp/topk-batched-large-segments-regressions`:
  `27f6a88e63 topk(bench): pass select::max explicitly to direction_param (was silently MIN)`
