# Harness contract

The common harness for every top-k primitive in the study. It owns input generation, the host
oracle, the latency driver, the throughput driver, the drift canary, resource reporting and the
JSON output. An approach plugs in as an adapter header. Nothing in an adapter measures anything,
and nothing in the harness knows an approach.

Files:

- `topk_harness.cuh`: the harness, one header, no dependencies beyond the CUDA toolkit and CCCL.
- `adapters/<approach>_<variant>.cuh`: one adapter per approach variant (`B1_reference.cuh`,
  `W1_reference.cuh` are the references).
- `cells_focus.json`: the focus-set cell grid (BRIEF.md section 3, PLAN.md section 8 tier 2).
- `run.py`: node-side driver (generate, compile outside the lock, measure inside the lock, merge).
- `canary_main.cu`: the standalone drift canary run first and last in a batch.
- `../tools/make_card.py`: batch JSON plus findings file to a validated card.
- `../tools/board.py`: regenerates `RESULTS_BOARD.md` from the cards.

## 1. Adapter contract

An adapter is a header `harness/adapters/<id>.cuh` where `<id>` is `<approach>_<variant>`
(`B1_reference`, `W1_packed16`, ...). It declares two things named after the file:

```cpp
struct <id>_info
{
  static constexpr const char* id       = "<id>";
  static constexpr topk::scope scope    = topk::scope::block;   // subwarp | warp | block
  static constexpr bool data_dependent  = true;                 // picks the latency chain mode, see 3
  static constexpr const char* contract = "one line";           // optional
  // Compile-time predicate. Must only use its arguments (never instantiate the primitive here).
  static constexpr bool supported(topk::key_type kt, int threads, int n, int k, topk::requirement req);
};

template <typename KeyT, int Threads, int N, int K, topk::requirement Req>
struct <id>
{
  static constexpr int items_per_thread                = N / Threads;
  static constexpr topk::arrangement input_arrangement = topk::arrangement::blocked; // see 1.1
  static constexpr topk::output_form output            = topk::output_form::indices; // or pairs
  using key_storage = KeyT[items_per_thread];        // uint32_t[items_per_thread / P] when packed
  using TempStorage = ...;                           // any trivially usable type, may be empty

  // Runs the primitive exactly once on the calling group's input and writes the result.
  //   keys, idx : this thread's items in the declared arrangement, may be clobbered
  //   ts        : this group's TempStorage in shared memory (the harness allocates it)
  //   out_idx   : K ints in shared memory, the adapter writes the K selected source indices in
  //               any order and from any threads of the group
  //   out_keys  : K keys in shared memory, written only when output == pairs, else nullptr
  __device__ __forceinline__ static void
  run(key_storage& keys, int (&idx)[items_per_thread], TempStorage& ts, int* out_idx, KeyT* out_keys);
};
```

Rules:

- `Threads` is the cooperating group: the block size for block scope, the logical warp size for
  warp (32) and sub-warp (4, 8, 16) scope. `N = Threads x items_per_thread`. The harness launches
  exactly `Threads` threads for a block adapter, one 32-thread warp for warp and sub-warp adapters
  in the latency regime, and 256 threads (several independent groups) in the throughput regime.
- `supported()` must be a constant expression over its arguments only. run.py generates a cell
  for every grid point, the generated TU calls `supported()` and does not instantiate the adapter
  when it returns false (so a static_assert inside the primitive can never break a batch). Say in
  the header why a combination is unsupported, and say it in the card's findings as well. A key
  type the primitive cannot handle is reported this way, never worked around silently.
- `run` may synchronize internally as it needs. It may assume the harness separates consecutive
  calls that reuse `ts` and `out_idx` by a group barrier (block: `__syncthreads`, warp and
  sub-warp: `__syncwarp` on the member mask). It must not assume anything about the state of
  `ts` or `out_idx` on entry. The harness double-buffers `out_idx`.
- Writing the K indices to `out_idx` is part of the measured cost for every approach, as a
  consumer needs them somewhere. Keys are never materialized for index-only output.
- NaN behavior is documented in the header comment, focus-set inputs contain no NaN.
- No performance numbers in code comments. Numbers go in cards and reports.

### 1.1 Input arrangement

`input_arrangement` selects how the harness loads a group's N keys into `key_storage` and what
`idx[i]` is. `t` is the thread's index inside the group, `T = Threads`, `IPT = items_per_thread`:

| arrangement | `keys[i]` is input item | `key_storage` |
|---|---|---|
| `blocked` | `t * IPT + i` | `KeyT[IPT]` |
| `striped` | `i * T + t` | `KeyT[IPT]` |
| `packed_striped` | word `j = i / P` of the thread is input word `j * T + t`, holding items `(j * T + t) * P + s` for `s < P` in its P lanes, low lane first | `uint32_t[IPT / P]`, `P = key_traits<KeyT>::packed_per_word` (2 for 16-bit keys) |

`idx[i]` is always the input item index of `keys[i]` (for packed, of lane `i % P` of word
`i / P`). An adapter may ignore `idx` and derive indices from positions. Blocked loads use the
widest aligned vector (16 bytes, else 8, else scalar), as a real kernel would, striped and packed
loads are per element or per word and coalesce by construction.

### 1.2 Output form and how index-only output is checked

`output == indices`: the adapter writes K source indices to `out_idx`. The oracle looks the keys
up in the host copy of the input and checks (see 2) without ever seeing a device key.
`output == pairs`: the adapter also writes the K selected keys to `out_keys[i]` such that
`out_keys[i]` is the key at `out_idx[i]`, the oracle additionally checks `out_keys[i] == input[out_idx[i]]`
(float `==`, so a returned `-0.0` for a `+0.0` input is accepted).

### 1.3 Requirements

`Req` is the tie-break requirement of the cell. `not_guaranteed`: any valid top-k set.
`prefer_smaller_index`: among equal keys the smaller index wins, the result is unique.
`prefer_larger_index` is defined symmetrically and is not in the focus set. `-0.0 == +0.0` under
all requirements. An adapter that implements a requirement by a different code path uses `Req`
at compile time, one that does not support it returns false from `supported()`.

## 2. Oracle

Every cell runs the oracle before any number, on the cell's own pattern (`--seeds`, default 4
seeds) plus a fixed battery: `tie_heavy`, `sorted_asc`, `sorted_desc`, `pivot_tie`, `all_equal`,
`signed_zero`, `inf_denorm`, `quantized_random`, `relu_quantized` (and two `random` seeds when the
cell pattern is something else). All groups of the launch are checked. A single failure means the
cell gets no number and the failure text lands in the JSON. Checks per output:

1. All K indices are in `[0, N)` and distinct.
2. `not_guaranteed`: the multiset of input keys at the returned indices equals the multiset of the
   K largest input keys (comparison with `==`, so signed zeros are equal).
3. `prefer_smaller_index`: the returned index set equals the first K of the input sorted by
   (key descending, index ascending), exactly.
4. pairs: every returned key equals the input key at its index.

The oracle self-tests on the host at every run (the exact reference passes, a wrong set fails, a
duplicate fails) and the result is recorded (`oracle_selftest` in the JSON).

Every measured input also reports its tie structure at rank k (`input_tie_stats` per adapter,
`ties_at_k_mean` and `ties_at_k_max` per cell: how many keys equal the k-th largest, over the 16
measurement inputs). bf16 has 128 values per binade, so bf16 rows tie at the pivot routinely and
the numbers say so.

Patterns (values generated in fp32, then rounded to the key type, so bf16 inputs carry the ties
rounding introduces): `random` normal(0, 1); `quantized_random`, `relu_quantized` as in
`segmented_topk/variable/common.cuh`; `tie_heavy` `(j % 64) / 64`; `sorted_asc` and `sorted_desc`
the sorted normal sample; `pivot_tie` all 1.0 with `max(1, min(k/4, N/8))` values 2.0 at the tail
(boundary flood of equal keys); `all_equal`; `signed_zero` negative keys with 2k zeros of
alternating sign at random positions; `inf_denorm` normal sample with +-inf, denormals and the
largest finite values mixed in.

## 3. Latency regime

Single warp (`<<<1, 32>>>`) or single block (`<<<1, Threads>>>`), `clock64` cycles, chain-length
slope: the kernel runs a chain of 1, 2, 4, 8 and 16 serialized calls, takes the min over at least
24 repetitions (`--reps`), two sweeps, and the host fits a least-squares line. The slope is the
marginal cycles per call, the intercept and R^2 are recorded. Bracketing: group barrier, `t0`,
chain, consume of the last output with a memory clobber, group barrier, `t1`.

Serialization: call `n` consumes `dep = out_idx[0]` of call `n - 1` (read after the group barrier)
masked with a kernel argument that is zero at run time but unknown to the compiler.

- `data_dependent = true` (radix select, atomics, threshold filters: the work depends on the
  data): call `n` loads input `(n + dep) & 15` out of 16 distinct inputs of the cell's pattern,
  fresh data every call, the load address carries the dependency.
- `data_dependent = false` (sorting networks: the work is fixed): in-place chain on the register
  resident input, the first key is xor-perturbed with `dep`. The input degenerates over the chain,
  which does not matter for a data-oblivious primitive (the oracle runs separately on pristine
  input).

Generate-only control: the identical kernel with a no-op adapter (same arrangement, same loads or
perturbation, every thread folds its inputs into one word and stores it to `out_idx[lane % K]` so
no load is dead, same barrier and read-back) is measured with
the same method, and `cycles_per_call = slope(adapter) - slope(noop)`. Both slopes, the minima per
chain length and the fit quality are in the JSON (`slope_raw`, `slope_noop`, `chain_minima`,
`r2_raw`). The fold costs about one LOP3 per three items and is the residual error of the
subtraction.

## 4. Throughput regime

Grid = `cudaOccupancyMaxActiveBlocksPerMultiprocessor x SMs` blocks (one occupancy wave) of
`Threads` threads for block adapters and 256 threads for warp and sub-warp adapters (several
independent groups per block). Every group runs `calls = clamp(2^21 / N, 32, 4096)` calls
(`--calls` overrides), each on one of the 16 inputs of the pattern loaded from global memory
(L1 or L2 resident), independent across calls. Gelem/s = `grid x groups x calls x N / time`, best
of 5 event timings after one warm-up launch. Occupancy, registers, shared memory and spills are
reported next to the number. The no-op adapter's throughput is reported as
`harness_ceiling_gelem_per_s`: if an adapter approaches it the harness loads, not the primitive,
set the number.

## 5. Canary

`canary_main` and every cell binary run two fixed kernels at start and end: a single-warp shuffle
chain in cycles per op (`lat_cycles_per_op`, drift here means a measurement problem, cycles are
clock invariant) and a fixed FMA kernel on the whole GPU in microseconds (`thr_us`, drift here
means clocks or contention). Latency cells carry the cycles canary, throughput cells the
microseconds canary, in `canary_start` and `canary_end`.

## 6. Resources

Per regime from the harness kernel that contains the primitive: registers, static plus dynamic
shared memory, local bytes (`cudaFuncGetAttributes`), occupancy in blocks per SM. run.py refines
`spill_bytes` from `nvcc -Xptxas -v` (spill stores plus spill loads of that kernel symbol) and
adds `stl_count`, `ldl_count`, `bar_sync_count` and `sass_instructions` from `cuobjdump -sass`.
The kernel symbol comes from `cudaFuncGetName`, so the match is exact. The numbers include the
harness (loads, barrier, output write), which is the same for variant and control.

## 7. Cells, JSON, run.py

`cells_focus.json` lists key types, requirements, patterns, regimes, k values and per scope the
thread counts and N values. A cell is one combination with `k <= N/2` and `N >= Threads`, and
an optional `k_by_n` map restricts the k values per N (`cells_flipbands.json` uses it for the dense
grids on the flip bands of `ideas/PRIORITIES.md`, phase 3). `--named-cells ideas/named_cells.json`
runs the algorithm researcher's named workload cells instead of the grid: each entry gives scope,
key type, n, k, requirement and regime (`both` expands to the two focus values), the thread count
is the scope's first entry in the cells file unless the entry has `threads` (sub-warp cells get the
smallest logical warp that divides n). Non power of two N is fine as long as N is a multiple of the
thread count, otherwise the cell is skipped with a message (partial tiles are not in the contract
yet). Named cells carry `workload_name`, `input_pattern` `named_workload` and the generating
pattern in `named_pattern`. run.py expands the scope of the adapter, generates one TU per cell
(variant and control in the same process), compiles them in parallel with

```
nvcc -std=c++17 -O3 -arch=sm_100 -I<repo>/cub -I<repo>/libcudacxx/include -I<repo>/thrust -I<harness> -Xptxas -v
```

(plain nvcc, no cmake, x86 and aarch64 alike), runs static analysis, then inside one exclusive
`flock` section on the GPU lock file: canary, every cell binary (variant then control, all
patterns), canary. The batch file `<out>/<batch>.json` holds meta (node, GPU, CUDA, nvcc, commit,
holder), canaries, `cells[]` whose objects match `cells/schema.json` items plus a `raw` member
with everything else, `failed[]`, `unsupported[]` and the build table. A/A controls (control
omitted) measure the same adapter twice in one process and are marked `control_is_aa`.

On a node (see `nodes/README.md` for the lease):

```
tar czf - -C $SWARM harness | ssh <node> '<exec_prefix> tar xzf - -C <root>/work/<holder>'
ssh <node> '<exec_prefix> bash -c "cd <root>/work/<holder>/harness && export PATH=/usr/local/cuda/bin:\$PATH && \
  python3 run.py --repo <root>/cccl --root <root> --holder <holder> --gpu <i> --node-name <node> --arch-name <arch> \
  --adapter W1_packed16 --control W1_reference --keys bf16 --label wave1 > <root>/work/<holder>/run.log 2>&1"'
ssh <node> '<exec_prefix> tar czf - -C <root>/results/<holder> .' | tar xzf - -C $SWARM/state/results/<holder>
```

`--control-req <requirement>` instantiates the control under a different requirement than the
variant, in the same process and lock section: a `prefer_smaller_index` variant can be measured
against the reference's `not_guaranteed` path when the reference does not support the tie-break
(the control name then reads `W1_reference@not_guaranteed`). Without it, cells the control does
not support come out without a control value and `make_card.py` drops them unless a second A/A
batch of the variant is merged through a second `--batch`. `--root` derives
`--work <root>/work/<holder>`, `--out <root>/results/<holder>` and `--lock <root>/locks/gpu<i>.lock`. `--node-name` and `--arch-name` come from the lease line
(inside a container the hostname is the container id, so pass them). Filters: `--keys`, `--req`, `--n`, `--k`, `--threads`,
`--patterns`, `--regimes`, `--oracle-only`, `--no-battery`. `--build-only` and `--measure-only`
split the phases, `--dry-run` lists the cells. Compiles never run under the lock.

Card: `python3 $SWARM/tools/make_card.py --batch <batch.json> --findings <findings.json> --out $SWARM/cards/<approach>_<variant>_<arch>_<yyyymmdd>_<n>.json`,
then `python3 $SWARM/tools/board.py`.

## 8. Known limits of the references

- `B1_reference` (`block_topk_air`): not_guaranteed only (atomic ticket order at the boundary),
  index-only output goes through the pairs path (keys are exchanged too and dropped at the end).
  bf16 keys use the 16-bit radix traits (two 8-bit passes). N up to 32768 at 256 threads is
  accepted, the exchange buffer is allocated as dynamic shared memory by the harness.
- `W1_reference` (`WarpBitonicTopK`): not_guaranteed only (ties are not index resolved), MaxK is
  k rounded up to 32, so k = 4 and k = 16 pay the k = 32 network. bf16 compared with `__hgt`.
- Neither reference supports `prefer_smaller_index`. Those cells are listed under `unsupported`
  in the batch and in the card's findings. The oracle path for that requirement is exercised by
  the host self-test until an adapter supports it.
