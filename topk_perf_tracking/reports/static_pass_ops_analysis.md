# Sibling extract_bin / identify_candidates ops -- 3-option analysis

Three sibling op variants were prototyped, built on B200/sm_100/CTK 13.1.115,
and measured against the existing `flat_walk` baseline (`exp/flat-cta-walk`).

## Implementations

All three add to `cub/cub/device/dispatch/dispatch_topk_common.cuh` as
sibling structs that take the radix pass as a template parameter so
`start_bit` and `mask` become `constexpr`. The filter kernel
(`device_segmented_topk_filter_kernel`) gains an `int Pass` template
parameter and the dispatcher launches one specialisation per filter
pass via a small `filter_kernel_ptr_t[8]` table indexed at runtime.

| variant | branch / commit | what changes |
|---|---|---|
| **A** `extract_bin_op_static_t` + `identify_candidates_op_static_t` (pointer) | `tmp/perf-eval-static-pass` (`e14f6cad58`) | `Pass` is template parameter; per-pass scalars are constexpr; identify_candidates still holds a pointer to `kth_key_bits` |
| **B** Option A + explicit `BFE` PTX | (not committed -- see "Option B" below) | force `BFE.U32` for the extract-bin via inline asm |
| **C** `identify_candidates_op_static_value_t` (value) | `tmp/perf-eval-static-value` (`c3fc33038b`) | Option A *plus* identify_candidates holds the dereferenced `kth_key_bits` *value* (loaded at ctor) instead of the pointer |

## Result at a glance (mean wall-clock GPU time, 8 workloads, single radix-pass sweep)

`main` is the single-problem dispatch (`cub::DeviceTopK::MaxPairs`).
`flat_walk` is the previous experiment (the chunked->flat run() rewrite).
All four batched variants instantiate the same pipeline outside the filter kernel.

|   |    main |    flat_walk |    A (static pointer)    |    C (static value)    |
|---|---:|---:|---:|---:|
| Mean GPU time (us) | 72.52 | **105.87** | 107.15 (+1.2%) | 108.01 (+2.0%) |
| Mean speedup vs `main` | 1.000x | **1.460x slower** | 1.479x slower | 1.491x slower |
| Mean speedup vs `flat_walk` | 0.685x | 1.000x | 0.988x | **0.980x** |

Per workload (us, lower is better):

| KeyT | ValueT |  main | flat_walk | A | C | A/flat | C/flat |
|---|---|---:|---:|---:|---:|---:|---:|
| I16 | I64 | 62.10 | 104.92 | 110.69 | 107.20 | 1.055x | 1.022x |
| I16 | I8  | 58.53 | 101.41 | 101.98 | 103.20 | 1.006x | 1.018x |
| I32 | I64 | 63.81 |  90.92 |  91.93 |  96.12 | 1.011x | 1.057x |
| I32 | I8  | 63.51 |  90.61 |  91.65 |  95.58 | 1.012x | 1.055x |
| I64 | I64 | 100.42 | 137.94 | 138.44 | 138.92 | 1.004x | 1.007x |
| I64 | I8  | 100.51 | 137.90 | 138.88 | 139.58 | 1.007x | 1.012x |
| I8 | I64 | 65.68 |  90.09 |  90.31 |  90.15 | 1.002x | 1.001x |
| I8 | I8  | 65.59 |  93.16 |  93.34 |  93.37 | 1.002x | 1.002x |

**None of A, B, C help end-to-end. A and C regress by 1-5% over `flat_walk` on
the multi-pass key types.** I8 doesn't move because its filter kernel never
runs (`num_passes = 1`).

## Filter-kernel register usage (sm_100, ptxas verbose), in-use Pass values only

The Pass-templated variants generate 8 specialisations per (KeyT, ValueT, select).
Filter pass values actually used at runtime are 1..num_passes-1; we only quote
those here.

| KeyT | (ValueT=I64) | flat_walk | A | C | notes |
|---|---|---:|---:|---:|---|
| I32 (uses Pass=1,2)  | regs | 40 | 40 | 40 | same |
| I32 | stack | 0 | 0 | 0 | |
| I32 | spill_st/ld | 0/0 | 0/0 | 0/0 | |
| I16 (uses Pass=1) | regs | 40 | 50 | 40 | A regressed +10 regs |
| I16 | stack | 0 | 0 | 8 | C added small stack |
| I16 | spill_st/ld | 0/0 | 0/0 | 24/24 | C added small spill |
| I64 (uses Pass=1..5) | regs | 40 | 40 | 40 | same |
| I64 | stack | 0 | 0 | 0 | |
| I64 | spill_st/ld | 0/0 | 0/0 | 0/0 | |

## Why none of them help (the diagnostics)

### Option A -- compile-time `Pass`

The dynamic ops are constructed in the kernel with
`_CCCL_GRID_CONSTANT const int pass`, which is just `__grid_constant__` in
PTX. ptxas already constant-propagates that across the kernel body, so the
`pass`, `start_bit`, `mask` reads inside `operator()` are already folded to
immediates. Making them `constexpr` template parameters doesn't add new
constant-propagation opportunities -- it only changes how ptxas schedules
the surrounding code, sometimes for the worse (I16 went from 40 to 50 regs).

### Option B -- explicit `BFE` PTX

The pointer-variant filter kernel for KeyT=int32, ValueT=int64, Pass=1
disassembles to **0 `BFE.U32` instructions, 53 `LOP3.LUT` instructions**.
ptxas is deliberately preferring `LOP3.LUT` because on Blackwell it can fuse
the shift + mask + extra logic operand into a single 3-input instruction
(visible in our earlier int8 SASS analysis at e.g. `LOP3.LUT R28, R13, 0x7f,
R27, 0x60, !PT`, which computes a 3-operand boolean). `BFE` is a 2-operand
shift+mask; replacing the `LOP3.LUT` with `BFE` would *cost* an additional
instruction in the cases where ptxas is fusing.

The CCCL bit-utilities header `<cuda/__bit/bitfield.h>` confirms this --
its `cuda::bitfield_extract` deliberately *skips* the inline-asm `BFE`
on SM >= 70 and falls back to plain `(value >> start) & mask`, exactly
because ptxas's lowering on those archs is at least as good.

Verdict: forcing `BFE` would not help on sm_100 and could regress.
(Not committed, since the experiment was concluded by inspection rather
than by a build/measure cycle.)

### Option C -- value-holding `kth_key_bits`

The intuition was that the per-item `*kth_key_bits` deref in the
identify-candidates inner loop was 16 LSU hits per tile per thread. The SASS
proves otherwise: counting `LDG.E.*` instructions in the in-use int/long
Pass=1 filter kernel:

|   | LDG.E.* count |
|---|---:|
| Option A (pointer) | 13 |
| Option C (value) | 13 |

ptxas was already CSE-ing the `*kth_key_bits` deref to a single load per
tile body. Moving the load to the op's ctor just relocates the same one
load; the value variant additionally keeps the loaded value live in a
register through more of the body, which is why I16 picked up a small
stack frame and a few spill bytes.

The 16-LDG-per-tile reading I'd earlier inferred from the original chunked
SASS was an artefact of the unrolled chunk body -- the *same* LDG appeared
once per *tile body*, not once per item, and the inner item loop reused the
already-loaded register. The pessimistic "P5: hoist `*kth_key_bits` out of
the j-loop" item from the original register-pressure proposal therefore
also doesn't apply: ptxas already does it.

## What this changes about the followups

The original "savings from hoisting `*kth_key_bits`" line in the
int8/int8 register-pressure write-up is wrong; striking it would be
honest. (Caveat: this verification was on int32/int64 keys, not on the
int8/int8 instantiation. int8's filter never runs at the benchmark
configuration; verifying on int8 would require a separate run.)

The 53 `LOP3.LUT`s in the in-use filter kernels are not low-hanging
register-pressure fruit -- they are the result of ptxas's chosen lowering
for the extract-bin / identify-candidates body. Replacing them with `BFE`
or rearranging the high-level shape (Options A/B/C) doesn't shake them
loose. The remaining 1.46x gap vs `main` for the multi-pass keys is more
likely structural -- per-segment state resolution + per-pass iterator
wrapping in the batched dispatch -- than something locally fixable in the
filter agent's identify/extract ops.

## Artifacts

- `topk_perf_tracking/snapshots/{static_a,static_c}.json` -- per-kernel
  register-usage snapshots for Options A and C.
- `topk_perf_tracking/bench/bench_{static_a,static_c}.json` -- raw nvbench
  output for the same 8 workloads used for `flat_walk`.
- `topk_perf_tracking/raw_logs/{static_a,static_c}__pairs.log` -- ptxas
  verbose build logs.
- `tmp/perf-eval-static-pass` (Option A) and `tmp/perf-eval-static-value`
  (Option C) on origin; not landed.
