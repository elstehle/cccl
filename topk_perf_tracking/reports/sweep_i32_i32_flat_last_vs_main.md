# I32 keys, I32 values: full workload sweep, `flat_last` vs `main`

Axes (per your earlier instructions):

- `KeyT = I32`, `ValueT = I32`, `OffsetT = I32`, `OutOffsetT = I32`
- `SelectedElements` = `2^[3, 8, 13, 18, 23]`
- `Elements` = `2^[16, 20, 24, 28]`
- `Entropy` = `{1.000, 0.201, 0.000}`

5 × 4 × 3 = 60 grid points; 51 non-skipped (SelectedElements >= Elements is auto-skipped by the bench harness).

nvbench config: `--timeout 5 --skip-time 15e-6 --stopping-criterion entropy --throttle-threshold 90 --throttle-recovery-delay 0.15`.
GPU: B200 (`umbriel-b200-072`, container `bold_mahavira`), CTK 13.1.115.

## Headline

| | flat_last / main |
|---|---:|
| geomean (51 workloads) | **4.18×** |
| min | 1.03× (best case) |
| max | 28.28× (worst case) |

The geomean is heavily dominated by the low-entropy workloads. Bucketed:

### Geomean flat_last / main, by entropy

| Entropy | n | min ratio | **geomean** | max ratio |
|---|---:|---:|---:|---:|
| 1.000 (uniform random) | 17 | 1.03× | **1.30×** | 1.73× |
| 0.201 (moderate)     | 17 | 1.23× | **3.40×** | 14.71× |
| 0.000 (all equal)    | 17 | 3.29× | **16.47×** | 28.28× |

### Geomean flat_last / main, by Elements

| Elements | n | min ratio | **geomean** | max ratio |
|---|---:|---:|---:|---:|
| 2^16 (65K)   | 9 | 1.52× | **2.17×** | 3.32× |
| 2^20 (1M)    | 12 | 1.24× | **3.45×** | 15.48× |
| 2^24 (16M)   | 15 | 1.04× | **4.94×** | 27.43× |
| 2^28 (256M)  | 15 | 1.03× | **6.08×** | 28.28× |

### Geomean flat_last / main, Elements × Entropy

| Elements \\ Entropy | 1.000 | 0.201 | 0.000 |
|---|---:|---:|---:|
| 2^16 (65K)  | 1.67× | 1.86× | 3.30× |
| 2^20 (1M)   | 1.34× | 2.02× | **15.10×** |
| 2^24 (16M)  | 1.24× | 3.59× | **27.06×** |
| 2^28 (256M) | **1.13×** | 7.04× | **28.21×** |

The data has a clean two-axis story:

- **Down the Entropy axis**: at uniform random (1.000) the batched
  dispatch is competitive with `main` (1.13-1.67× geomean). At
  all-equal (0.000) it falls off a cliff (3.30-28.21× geomean).
- **Down the Elements axis at high entropy**: as input grows, the gap
  *closes* (1.67× -> 1.13× at entropy 1.000). The batched fixed-overhead
  amortises better at scale.
- **Down the Elements axis at low entropy**: the gap *opens* (3.30× ->
  28.21× at entropy 0.000). Whatever the batched dispatch is doing for
  all-equal-keys scales badly with input size.

## Is the low-entropy gap from my changes?

No -- the pre-existing chunked baseline (`dev`) is the same shape on
low-entropy. Geomean flat_last / dev:

| Entropy | dev -> flat_walk | dev -> flat_last | flat_walk -> flat_last |
|---|---:|---:|---:|
| 1.000 | 0.841× | 0.844× | 1.004× |
| 0.201 | 0.927× | 0.860× | 0.927× |
| 0.000 | 0.969× | 0.944× | 0.974× |

So:

- At entropy 1.000, the **filter** flat-walk does most of the lift
  (-16% vs dev). The **last_filter** flat-walk on top of it is a wash
  (1.004×), because at high entropy the filter kernel dominates and
  last_filter is small.
- At entropy 0.201, the **last_filter** flat-walk pays off (-7.3%
  on top of flat_walk's -7.3%, compounding to -14% vs dev).
- At entropy 0.000, both flat-walks help only modestly (-3% / -2.6%).
  The kernels we're touching are not the bottleneck on this path --
  most of the time is being spent elsewhere in the pipeline (almost
  certainly the histogram + finalize kernels and the per-pass
  candidate-buffer write-throughs that all-equal keys saturate).

## Per-workload table

Full table is in `topk_perf_tracking/reports/sweep_i32_i32_all.md`.
Selected highlights below.

### Where flat_last is close to main (entropy 1.000)

| Elements | SelectedElems | main (us) | flat_last (us) | flat_last/main |
|---:|---:|---:|---:|---:|
| 268,435,456 | 8,388,608 | 4280.34 | 4424.78 | **1.034×** |
| 16,777,216 | 8,388,608 | 404.94 | 422.69 | **1.044×** |
| 16,777,216 | 262,144 | 195.56 | 216.41 | **1.107×** |
| 268,435,456 | 262,144 | 615.46 | 676.00 | **1.098×** |
| 16,777,216 | 8,192 | 65.59 | 91.46 | **1.394×** |

### Where flat_last is catastrophically slower than main (entropy 0.000)

| Elements | SelectedElems | main (us) | flat_last (us) | flat_last/main |
|---:|---:|---:|---:|---:|
| 268,435,456 | 256 | 6219.99 | 175850.00 | **28.28×** |
| 16,777,216 | 256 | 406.80 | 11023.69 | **27.09×** |
| 268,435,456 | 8,388,608 | 6189.74 | 173066.21 | **27.96×** |
| 1,048,576 | 8 | 49.16 | 734.89 | **14.95×** |

These are pre-existing in dev (28.27× there too), not introduced by
the flat-walk changes.

## What this implies for next steps

1. **The flat-walks have done what they can on the filter / last_filter
   register-pressure axis.** At high entropy (1.000), `flat_last` is
   already within 1.04-1.67× of `main`. The remaining gap there is
   small.
2. **The low-entropy cliff is the real perf problem and it's NOT in
   the kernels I've been touching.** Geomean 16-28× vs `main` on
   all-equal-key inputs. The path is probably the histogram /
   finalize-histogram kernels (which all-equal-keys make a worst case
   for: every key lands in the same bucket every pass) plus the
   candidate-buffer write-through, all of which `main`'s single-problem
   dispatch handles differently.
3. **Profiling-driven decomposition is the next step.** A
   per-kernel `nsys nvprof --print-gpu-trace` or `ncu --metrics ...`
   on one of the 27× cases would tell us which kernel(s) account for
   the extra ~170 ms, which `main` does not pay.

## Artifacts

- Full per-workload table: `topk_perf_tracking/reports/sweep_i32_i32_all.md`
- Raw sweeps: `topk_perf_tracking/bench/sweep_{main,dev,flat_walk,flat_last}_i32i32.json`
- Comparison tool: `topk_perf_tracking/compare_sweep.py`
