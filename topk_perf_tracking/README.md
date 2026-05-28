# TopK Kernel Resource Tracking

Tracks register, stack, spill, and shared-memory usage of CUB TopK kernels
across versions of the implementation. Builds the
`cub.bench.topk.pairs.base` benchmark on a B200 (sm_100) with PTXAS verbose
output, parses the build log, and produces per-kernel snapshots + comparison
reports.

## Layout

- `record_snapshot.sh` — drives a full snapshot run: pushes the requested
  ref to `origin/tmp/perf-eval-<label>`, configures + builds the pairs
  benchmark on `umbriel-b200-068` inside the `clever_hellman` container,
  pulls the log back, and stores a parsed JSON snapshot. Hostname /
  container / paths are all overridable through `TOPK_*` env vars
  (see the script header for the full list and the known-good node table).
- `build_snapshot.py` — parses a PTXAS verbose build log and emits a
  snapshot JSON (used by `record_snapshot.sh`).
- `compare_snapshots.py` — generates a side-by-side per-kernel Markdown
  table for one logical kernel (e.g. `filter`).
- `summarize.py` — produces a compact Markdown summary across all logical
  kernels and metrics.
- `analyze_liveness.py` — joins an nvdisasm life-ranges dump with an
  nvdisasm line-info dump and prints peak-liveness source lines plus
  surrounding SASS context.
- `find_kernel.sh` — finds the mangled symbol for a given
  (KeyT, ValueT, select, logical_name).
- `find_nvdisasm_index.sh` — brute-forces the `--cuda-function-index`
  value that `nvdisasm` uses for a given symbol substring.
- `snapshots/` — JSON snapshots, one per labelled run.
- `raw_logs/` — raw PTXAS build logs.
- `sass/` — extracted SASS dumps (lrm + lineinfo) used by `analyze_liveness.py`.
- `reports/` — generated Markdown comparison reports.

## Build context

- Default host: `umbriel-b200-068`, Docker container `clever_hellman`
  (image `nvidia/cuda:13.1.1-devel-ubuntu24.04`).
  Migration history (same image / CTK / GPU at each step, data compares
  cleanly across nodes):
  - `umb-b200-261` / `brave_rosalind`     until 2026-05-26
  - `umbriel-b200-072` / `bold_mahavira`  until 2026-05-26
  - `umbriel-b200-068` / `clever_hellman` current
- Default repo path inside container: `/cccl/cccl`
  (NFS-mounted on `umbriel-b200-068` -- has ~250 MB free, so build
  trees must live elsewhere. On previous nodes this was
  `/cccl_fork/cccl` on the overlay disk.)
- Default build root: `/cccl_fork/topk_perf/build_<label>`
  (on the local overlay disk, ~490 GB free)
- CMake: `/cccl/cmake/cmake-4.3.2-linux-x86_64/bin/cmake`
- CTK: 13.1.115, sm_100, `--preset cub-cpp17`
- CUDA flags: `-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t` (filters to
  32-bit `OffsetT`/`OutOffsetT`; keeps all four `KeyT`×`ValueT` combos)
- Target: `cub.bench.topk.pairs.base` (defaults `CUB_BENCH_TOPK_USE_BATCHED=1`
  on the dev branch, so the dev snapshot exercises the segmented-batch path
  through a single segment)

## Portability / surviving a node lease change

What lives where:

| artifact | where | lost if B200 node disappears? | lost if `/raid` disappears? |
|---|---|---|---|
| `snapshots/*.json` | here, committed to git | no | no (also on `origin`) |
| `reports/*.md` | here, committed to git | no | no (also on `origin`) |
| `raw_logs/*.log` | here, committed to git | no | no (also on `origin`) |
| `sass/`, `sass_flat/` (~700 MB) | here, gitignored | no | yes -- regen from cubin |
| build trees, container cubins | container only | yes -- regen | no |
| topk source branches | `origin/tmp/perf-eval-*` on GitHub | no | no |

So the **canonical baselines (snapshots + reports + raw_logs) are committed
to git** under this directory. If the B200 node lease ends, only the
multi-hundred-MB SASS dumps under `sass*/` are forfeit, and even those
can be re-extracted by running `record_snapshot.sh` against the new
node and re-running the `nvdisasm` step.

To switch to a new B200 node, set environment variables for
`record_snapshot.sh` -- the script reads them with sensible defaults:

```bash
TOPK_HOST=umb-b200-NEW              \
TOPK_CONTAINER=other_container_name \
TOPK_REPO_DIR=/path/in/container/to/cccl \
TOPK_BUILD_DIR_BASE=/path/in/container/scratch \
TOPK_CMAKE_BIN=/path/to/cmake-4.x   \
TOPK_NINJA_JOBS=24                  \
./topk_perf_tracking/record_snapshot.sh my_label HEAD
```

Existing snapshots produced on the old node remain valid as baselines so
long as the same CTK + GPU arch + CMake preset is used on the new node.
The `metadata.gpu_arch` and `metadata.ctk_version` fields in each
snapshot make this auditable.

## Logical kernel mapping

Kernels are named differently on main vs the batched-topk dev branch. We
map both to a shared "logical name" so they can be compared:

| logical name | main kernel | dev kernel |
|---|---|---|
| `filter` | `DeviceTopKKernel` | `device_segmented_topk_filter_kernel` |
| `last_filter` | `DeviceTopKLastFilterKernel` | `device_segmented_topk_last_filter_kernel` |
| `initial_histogram` | `DeviceTopKHistogramKernel` | `device_segmented_topk_histogram_kernel` |
| `finalize_filter` | *(none)* | `device_segmented_topk_finalize_filter_kernel` |
| `finalize_histogram` | *(none)* | `device_segmented_topk_finalize_histogram_kernel` |
| `single_cta` | *(none)* | `device_segmented_topk_kernel` |

The dev branch instantiates each `(KeyT, ValueT)` for both `select::min`
and `select::max`. The two variants produce identical register / stack /
smem / spill numbers, so the comparison de-duplicates by `(KeyT, ValueT)`
and keeps `select::max`.

## Usage

### Record a new snapshot

From the local workspace, drive a build on the B200 node:

```bash
# Snapshot the local HEAD as label "exp_r25"
./topk_perf_tracking/record_snapshot.sh exp_r25

# Or snapshot a specific commit
./topk_perf_tracking/record_snapshot.sh exp_r25 HEAD~2
```

This produces `snapshots/exp_r25.json` and `raw_logs/exp_r25__pairs.log`.

### Re-generate comparison reports

```bash
# Filter kernel (focus of round 1)
python3 topk_perf_tracking/compare_snapshots.py \
  topk_perf_tracking/snapshots/main.json \
  topk_perf_tracking/snapshots/dev.json \
  --logical-name filter \
  --out topk_perf_tracking/reports/filter_main_vs_dev.md

# Compact roll-up across all logical kernels and metrics
python3 topk_perf_tracking/summarize.py \
  topk_perf_tracking/snapshots/main.json \
  topk_perf_tracking/snapshots/dev.json \
  --out topk_perf_tracking/reports/summary_main_vs_dev.md
```

`compare_snapshots.py` accepts more than two snapshots; each non-baseline
gets its own absolute column plus a `Δ vs baseline` column.

### Drill into peak register usage for a specific kernel

Requires the dev build to be built once with `-lineinfo` added to
`CMAKE_CUDA_FLAGS`:

```bash
# 1. inside the container, build with -lineinfo (in addition to the usual flags)
.../cmake -G Ninja -B .../build_dev_lineinfo --preset cub-cpp17 \
    -DCCCL_ENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=100 \
    -DCMAKE_CUDA_FLAGS="-Xptxas=-v -lineinfo -DTUNE_OffsetT=::cuda::std::int32_t"
ninja cub.bench.topk.pairs.base -j 16

# 2. extract the sm_100 cubin
cuobjdump --extract-elf pairs.sm_100.cubin <pairs.cu.o>

# 3. find the nvdisasm function index for your (KeyT, ValueT, select) combo
bash topk_perf_tracking/find_nvdisasm_index.sh pairs.sm_100.cubin \
    "device_segmented_topk_filter_kernel.*policy_selector_from_typesIaal" 0 700

# 4. dump life ranges + line info (cannot be combined in one invocation)
nvdisasm --print-life-ranges -lrm count --cuda-function-index N pairs.sm_100.cubin \
    > sass/<label>.lrm.txt
nvdisasm --print-line-info        --cuda-function-index N pairs.sm_100.cubin \
    > sass/<label>.lineinfo.txt

# 5. compute peak-liveness report
python3 topk_perf_tracking/analyze_liveness.py \
    --lrm      sass/<label>.lrm.txt \
    --lineinfo sass/<label>.lineinfo.txt \
    --kernel-substring "device_segmented_topk_filter_kernelINS2_26policy_selector_from_typesIaal" \
    --top 15 --window 4 \
    --out reports/<label>_peak_liveness.md
```

Caveat: `nvdisasm --print-life-ranges` on the *full* cubin currently
trips an `Invalid register count : '255'` bug on at least one
`single_cta` instantiation; always pass `--cuda-function-index` to
restrict scope.

## Snapshot schema

```json
{
  "metadata": {
    "label": "main",
    "branch": "main",
    "commit_sha": "...",
    "commit_subject": "...",
    "datetime": "...",
    "gpu_arch": "sm_100",
    "ctk_version": "13.1.115",
    "build_target": "cub.bench.topk.pairs.base",
    "build_flags": "-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t",
    "build_seconds": 54.0,
    "record_count": 48
  },
  "kernels": [
    {
      "kernel_name": "device_segmented_topk_filter_kernel",
      "logical_name": "filter",
      "key_t": "signed char",
      "value_t": "signed char",
      "select": "max",
      "registers": 64,
      "stack_frame": 0,
      "spill_stores": 0,
      "spill_loads": 0,
      "smem_bytes": 1028,
      "barriers": 1,
      "cmem": {"0": 412, "2": 80},
      "arch": "sm_100",
      "compile_time_ms": 78.624,
      "demangled": "...",
      "mangled": "..."
    }
  ]
}
```
