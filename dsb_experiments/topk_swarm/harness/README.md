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

| `packed_rows` (rows per group) | word `j` of the thread is item `j * T + t` of every row: row `r` of the call sits in key lane `r` of the word (row 0 in the low bits) | `uint32_t[IPT]`, requires `static constexpr int rows = P` |

`idx[i]` is always the input item index of `keys[i]` (for packed_striped, of lane `i % P` of word
`i / P`, for packed_rows of word `i`, the same index for every row). An adapter may ignore `idx` and derive indices from positions. Blocked loads use the
widest aligned vector (16 bytes, else 8, else scalar), as a real kernel would, striped and packed
loads are per element or per word and coalesce by construction.

### 1.1.1 Rows per group

An adapter that processes R independent rows of N keys per call (N5's bf16x2 network, two rows
per lane) declares `static constexpr int rows = R` and the `packed_rows` arrangement (R must equal
the keys per word, 2 for 16-bit keys). The harness then generates R rows per input (row 0 of a
multi-row input equals the single-row input of the same seed, row r uses seed plus 1000003 r),
loads them packed per item, expects `R x K` output slots with row r's k indices in
`out_idx[r * K, (r + 1) * K)`, each in `[0, N)`, checks every row on its own in the oracle, reports
tie statistics per row, and counts `R x N` elements per call in Gelem/s. Cells carry `rows`.
`N0_rows_naive` under `adapters/` is a slow but exact two-row example (k shuffle arg-max rounds per
row) that exercises the path.

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
`signed_zero` (5 seeds), `signed_zero_boundary` (2), `pivot_tie_zero` (3), `inf_denorm`,
`quantized_random`, `relu_quantized` (and two `random` seeds when the cell pattern is something
else), 23 cases per cell plus the cell's own seeds.

Signed zeros (harness 0.7, after a threshold primitive dropped -0.0 items at a +0.0 threshold and
passed the old single signed_zero seed under not_guaranteed because exactly k plus zeros remained):
the five `signed_zero` seeds vary the share of +0.0 among the 2k zeros (one half, one quarter,
three quarters, none, all), so with fewer than k plus zeros a primitive that drops -0.0 fails under
not_guaranteed too, and under prefer_smaller_index the exact index check demands the smallest
indices regardless of sign. `signed_zero_boundary` puts k/2 positive keys above k zeros (one
quarter plus) so the k-th rank sits inside the zeros with both signs present. `pivot_tie_zero` is
pivot_tie with 2k signed zeros at the pivot below the 1.0 tail. A cell instantiated under
not_guaranteed is checked for set validity only, the exact check applies to the
prefer_smaller_index cell of the same shape, so a batch must contain both requirements to cover
both. All groups of the launch are checked. A single failure means the
cell gets no number and the failure text lands in the JSON. Checks per output:

1. All K indices are in `[0, N)` and distinct.
2. `not_guaranteed`: the multiset of input keys at the returned indices equals the multiset of the
   K largest input keys (comparison with `==`, so signed zeros are equal).
3. `prefer_smaller_index`: the returned index set equals the first K of the input sorted by
   (key descending, index ascending), exactly.
4. pairs: every returned key equals the input key at its index.

The oracle self-tests on the host at every run (the exact reference passes, a wrong set fails, a
duplicate fails, any k of tied zeros pass under not_guaranteed and only the smallest indices under
prefer_smaller_index) and the result is recorded (`oracle_selftest` in the JSON). A false self-test
sets the cell status to `selftest_failed`, run.py lists it under `failed`, and make_card.py drops
the cell unless `--allow-selftest-failure`. Harness 0.3 and earlier recorded a false negative at
k == N/2 (the self-test's own input had no non-zero key left for its "wrong" set, the oracle was
intact), fixed in 0.4, cards built from such batches keep those cells with a note.

Sorted runs (`--runs M`, harness 0.8, for chunk-merge primitives whose contract requires sorted
input runs): after the key-type rounding every input row is split into M consecutive runs of
about N / M items and each run is stably sorted by (key descending, index ascending), the order
an exact per-chunk top-k emits. The transform applies to the cell pattern and to every battery
pattern (ties, signed zeros and floods stay valid tests of a merge), the cell JSON and the card
cell carry `runs`, `input_pattern` stays the generating pattern, and cell names get a `_runsM`
suffix. The control adapter sees the same input.

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
largest finite values mixed in; `signed_zero_boundary` and `pivot_tie_zero` as described above.

## 3. Latency regime

Single warp (`<<<1, 32>>>`) or single block (`<<<1, Threads>>>`), `clock64` cycles, chain-length
slope: the kernel runs a chain of 1, 2, 4, 8 and 16 serialized calls, takes the min over at least
24 repetitions (`--reps`), two sweeps, and the host fits a least-squares line. The slope is the
marginal cycles per call, the intercept and R^2 are recorded. Bracketing: group barrier, `t0`,
chain, consume of the last output with a memory clobber, group barrier, `t1`.

Serialization: call `n` consumes `dep = out_idx[0]` of call `n - 1` (read after the group barrier)
masked with a kernel argument that is zero at run time but unknown to the compiler.

- `data_dependent = true` (radix select, atomics, threshold filters: the work depends on the
  data): call `n` loads input `(n * (16 / chain) + dep) & 15` out of 16 distinct inputs of the
  cell's pattern, fresh data every call, the load address carries the dependency. Every chain
  length sees a deterministic, evenly strided subset of the inputs (chain 16 all of them), the same
  in every repetition, so the min over repetitions compares identical work. All harness loads
  stream past L1 (`ld.global.cg`, since 0.10): a real kernel loads each tile once, and L1 residency
  of the 16 inputs had made the control depend on the chain length and on the variant's
  shared-memory carve-out.
- `data_dependent = false` (sorting networks: the work is fixed): in-place chain on the register
  resident input, the first key is xor-perturbed with `dep`. The input degenerates over the chain,
  which does not matter for a data-oblivious primitive (the oracle runs separately on pristine
  input).

Generate-only control: the identical kernel with a no-op adapter (same arrangement, same loads or
perturbation, every thread folds its inputs into one word and stores it to `out_idx[lane % K]` so
no load is dead, same barrier and read-back, launched with the variant's dynamic shared memory
size so both kernels see the same L1 carve-out and, in the throughput regime, the same occupancy) is measured with
the same method, and `cycles_per_call = slope(adapter) - slope(noop)`. Both slopes, the minima per
chain length and the fit quality are in the JSON (`slope_raw`, `slope_noop`, `chain_minima`,
`r2_raw`). The fold costs about one LOP3 per three items and is the residual error of the
subtraction.

### 3.1 Harness versions and their effect on numbers

Cards record the harness version in their raw files. Known effects when comparing cards made with
different versions (same adapter, same node):

- 0.2 to 0.5: blocked loads vectorized (0.2), rows and control-req added (0.3 to 0.5). Data
  dependent chains used input `n` for call `n`, so a chain of 16 touched 16 inputs and left L1
  where a chain of 8 stayed resident: fits bent at N 8192 and above for 16 inputs over about
  128 KB (fp32 N 4096 and up, bf16 N 8192 and up).
- 0.6 to 0.9: input rotation with the repetition, control launched with the variant's dynamic
  shared memory (`dyn_floor`). Large-smem controls (B3 at fp32 N 16384 and above) read 9 to 11
  percent higher, so nets with such controls dropped by that much. Striped block kernels at bf16
  N 4096 lost 7 to 9 percent of net because the control's per-call cost rose with the chain length
  (per-repetition working set against the smaller L1) while the min over repetitions picked cached
  repetitions for short chains. Kernels whose pass count depends on the input saw the min over
  repetitions pick the cheaper half of the inputs at chain 8 under the rotation. Large-smem blocked
  kernels were not affected by the chain-length dependence: B3_tuned fp32 N 32768 k 256 had a flat
  control of 10424 to 10734 cycles per call across chain lengths under 0.8 (8382 to 8656 under 0.4,
  the `dyn_floor` difference), because their working set never fit L1 in the first place
  (verify-B4).
- 0.10: loads stream past L1 in both regimes, deterministic strided input subsets per chain
  length, no rotation. Controls are L2 bound and larger, the same at every chain length and for
  every shared-memory footprint (validated on a GB200: the no-op per-call cost at chain 1 to 16 is
  719, 691, 678, 672, 670 for a striped bf16 N 4096 kernel and 2599 to 2589 for blocked fp32
  N 16384, fits R^2 at or above 0.9996). Against 0.6 on the same arch, B1_reference nets move by
  under 1 percent at N 1024, by -3.5 percent (bf16) and -2 percent (fp32) at N 4096, and by +12
  percent (bf16) and +3 percent (fp32) at N 16384, where the variant's own loads had partly hit L1
  under 0.6 and now stream like the control's. The striped B4_v4 at bf16 N 4096 reads its 0.4
  value again (3808 against 3818, not the 3461 of 0.8). Throughput ceilings drop to the L2
  streaming rate (about 4 to 4.6 Telem/s at N 1024, 1.2 Telem/s for fp32 at N 16384), which is
  what a kernel sees for a tile it loads once.

Verifiers rerun a card with the harness version it was made with when they want to reproduce it to
the cycle (the card's `harness/` files are in the commit named in the card), and with the current
version when they want a comparable number.

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

A regime whose launch does not fit the device (typically the throughput block of 8 warps times a
large per-warp temp storage plus the double-buffered output exceeding the 227 KB opt-in limit) is
reported as unsupported for that regime in the adapter's `error` field, the other regime's numbers
stand, and the cell binary does not fail (harness 0.9).

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
adds `stl_count`, `ldl_count`, `bar_sync_count`, `bssy_count` and `sass_instructions` from
`cuobjdump -sass`. Sub-warp kernels (logical warps of 4, 8 or 16 lanes) carry `BSSY` and `BSYNC`
convergence regions around their partial-mask shuffles and syncs, in variant and control alike
and scaling with the network size, warp-scope kernels have none: they do not bias a same-harness
comparison but they are one more reason not to compare sub-warp cycles with warp-scope cycles.
The kernel symbol comes from `cudaFuncGetName`, so the match is exact. A register array that
NVVM could not promote shows up as a stack frame with STL and LDL in the SASS while
`spill_bytes` stays 0, so `stack_bytes`, `stl_count` and `ldl_count` are carried into the card
resources and `summarize_batch.py` and the run.py summary flag them as `LOCAL`. `sass_instructions`
is there for the same reason: two cells that run the same network with a different k (for example
k 16 and k 32 at MaxK 32, where only a store predicate differs) can compile to different code and
differ by 20 percent in cycles, which the instruction count and register count show and a
cycle-only table hides. Compare them before reading a k dependence into a data-oblivious primitive. The numbers include the
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
git rev-parse HEAD > $SWARM/harness/SOURCE_COMMIT      # in the optimizer's worktree, records the adapter source commit
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
`--patterns`, `--regimes`, `--oracle-only`, `--no-battery`. `--cell-list <file>` runs an
explicit list of cells (`[{scope, key_type, threads, n, k, requirement}]`, the format
`tools/winners.py --close` writes for the tie_heavy follow-up pass). `--check-sorted` (phase 3b)
makes the oracle also verify the output order: keys non-increasing along the slots, and under an
index tie-break the indices in tie-break order among equal keys. `--build-only` and `--measure-only`
split the phases, `--dry-run` lists the cells. A cell binary that exits with an error (seen once as
CUDA error 46, device busy, before the oracle) is retried once after 5 seconds (`--retries`), the
batch records `retries` per run. Compiles never run under the lock.

Source commit: a card's `commit` is the commit of the adapter source the kernels were built from
(the optimizer's worktree HEAD), never the node's CCCL clone commit (that one goes to the card's
optional `clone_commit`). Write it before shipping the harness, `git rev-parse HEAD > harness/SOURCE_COMMIT`
in the worktree, run.py records it in the batch as `source_commit`, or pass `--source-commit` to
run.py or to make_card.py. make_card.py refuses to build a card without one.

Card: `python3 $SWARM/tools/make_card.py --batch <batch.json> --findings <findings.json> --source-commit <adapter HEAD> --out $SWARM/cards/<approach>_<variant>_<arch>_<yyyymmdd>_<n>.json`,
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
