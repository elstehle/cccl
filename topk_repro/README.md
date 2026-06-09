# Segmented top-k benchmarking & profiling repro kit

Self-contained scripts to (re)generate the segmented-topk latency numbers and the
per-kernel time breakdown, so results can be reproduced on any GPU node after code
changes — e.g. to compare a tuning/code modification against the committed reference.

Benchmark under test: `cub.bench.segmented_topk.variable.keys.base`
(source: `cub/benchmarks/bench/segmented_topk/variable/keys.cu`).

## Requirements

- NVIDIA GPU + CUDA Toolkit (results were taken on a **B200**, `CUDA_ARCH=100`, CUDA 13.1).
- **CMake >= 4.0** (the benchmarks pull in rapids-cmake, which requires it).
- Ninja, Python 3, and internet access for the first configure (CPM fetches deps).
- For the per-kernel breakdown: **Nsight Systems `nsys`** on `PATH`.
  - If your `nsys` is the stripped one bundled with Nsight Compute, it may emit only a
    `.qdstrm`; the script then locates a `QdstrmImporter` automatically. That importer
    needs `libdw` — on Debian/Ubuntu: `apt-get install -y libdw1`.

## Quick start

```bash
# from the repo root
export CUDA_ARCH=100          # B200; change for other GPUs
./topk_repro/build.sh         # configure + build the benchmark (~1-2 min)
./topk_repro/run_sweep.sh     # GPU-time sweep  -> topk_repro/out/baseline_sweep.txt
./topk_repro/run_breakdown.sh # per-kernel breakdown -> topk_repro/out/breakdown_ns32_1048576_k2048.md
```

Or everything at once: `./topk_repro/run_all.sh`.

## What each script does

| Script | Output | Notes |
|---|---|---|
| `build.sh` | `build/topk_repro/bin/...` | auto-detects CMake >= 4.0; honors `CMAKE`, `BUILD_DIR`, `CUDA_ARCH`, `PRESET`, `FORCE` |
| `run_sweep.sh` | `out/baseline_sweep.txt`, `out/sweep.json` | GPU time vs `MaxSegmentSize × K × NumSegments` (pattern `random` by default) |
| `run_breakdown.sh` | `out/breakdown_ns<NS>_<SEG>_k<K>.md`, `out/pat_*.sqlite`, `out/totals.json` | nsys per-kernel-instance averages over steady-state iterations, all 5 patterns |
| `parse_sweep.py` | — | nvbench JSON → table (used by `run_sweep.sh`) |
| `kernel_breakdown.py` | — | nsys SQLite(s) + totals → Markdown (used by `run_breakdown.sh`) |

## Reference numbers (committed)

`reference/` holds the golden outputs captured on the B200 at commit `f326b019f6`:

- `reference/baseline_sweep.txt` — the latency sweep (pattern `random`).
- `reference/breakdown_ns32_1M_k2048.md` — per-kernel breakdown for **NS=32, 1M, K=2048**
  across all 5 patterns (the headline config).

After a code change, regenerate and diff:

```bash
./topk_repro/run_breakdown.sh
git diff --no-index topk_repro/reference/breakdown_ns32_1M_k2048.md \
                    topk_repro/out/breakdown_ns32_1048576_k2048.md
```

## Methodology notes

- nvbench measures **cold** GPU time (L2 flushed before every sample), so the per-kernel
  durations reflect cold-input behavior. The `~132 MB` L2-flush memset and nvbench harness
  kernels are excluded from the breakdown; the 3 small dispatch memsets are summed.
- The radix-select launch sequence is **data-independent** (fixed by the 32-bit key at
  11 bits/pass): `worker → histogram → finalize_histogram → filter×2 → finalize_filter×2 →
  last_filter`. The breakdown averages each launch position over all iterations; it asserts
  every iteration contained exactly that sequence (`bad=0`).
- Axis labels: `MaxSegmentSize{ct}`, `K{ct}` are compile-time; only the values compiled into
  the benchmark are valid (see `keys.cu`). `NumSegments`, `Pattern` are runtime axes.

## Customizing the config

All knobs are env vars, e.g. a different breakdown config or GPU:

```bash
NS=8 SEG=524288 K=1024 ./topk_repro/run_breakdown.sh
CUDA_ARCH=90 ./topk_repro/build.sh                       # H100
NSEGS=32 KS=2048 SEG_SIZES=1048576 PATTERN=pivot_tie ./topk_repro/run_sweep.sh
```
