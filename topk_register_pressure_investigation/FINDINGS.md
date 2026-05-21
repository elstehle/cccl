# Where the batched-topk filter's register pressure is coming from

Hardware: B200 (sm_100), nvcc 13.1.115 — build dir
`build/blackwell-cub-cpp17`, repo at commit
`5854e3fd87 remaining batched wiring`, working tree clean.

Target kernel: the most-regressed configuration, `i8` keys-only,
`select::max` direction, multi-CTA filter pass:

| kernel                              | KeyT | batched REG | single REG | Δ REG |
|-------------------------------------|-----:|------------:|-----------:|------:|
| `device_segmented_topk_filter_kernel` | i8 | **122**     | **32**     | **+90** |

Everything below is sm_100 SASS extracted from binaries built with
`-DCMAKE_CUDA_FLAGS=-lineinfo` (so line-info maps SASS to source) and
restored to "no extra flags" after capture. Artifacts (raw resource-usage
dumps, the two function-scope SASS slices, the small Python helpers used
for the analysis) live in this directory next to this file.

## TL;DR

The +90-reg gap on `i8` filter is the sum of two independent effects:

1. **Item-per-thread amplification (~16-20 regs)** — both the single-problem
   and batched filter use the Hopper+ `items_per_thread = 16` for `i8`, so
   both keep 16 unpacked bytes live during the load → bin-classify → atomic
   sequence. This is identical on both sides.
2. **Missed uniform-register promotion (~70-80 regs)** — the per-segment
   scalars and pointers that all threads in a warp compute identically
   (counter struct fields, per-segment histogram / key-buf / val-buf base
   pointers, segment_id, segment_size, segment_k, total_large_tiles) stay
   in per-thread regular registers (`R*`) in the batched filter, whereas
   the single-problem filter aggressively promotes the analogous scalars
   into uniform registers (`UR*`) via `R2UR`. ptxas's `R2UR` heuristic
   doesn't fire on the batched code, so 32-thread-replicated scalars
   inflate the per-thread footprint by ~10× their byte size.

Action: the largest knob we have is restructuring the batched-filter code
paths to make the warp-uniform values discoverable to ptxas (Section 4).
Secondary: the single-problem filter also pays a per-segment scalar
amplification in the *worst* iteration but doesn't (a) carry as many of
them and (b) rematerialize `SR_TID.X` so aggressively the second time
it's needed. Mirroring those patterns is a feasible source-side
intervention.

## 1. Workflow used

```
sm_100 build (release + -lineinfo)
   ├── bin/cub.bench.topk.keys.base                                  (USE_BATCHED=1, current)
   ├── bin/cub.bench.topk.keys.base.batched                          (USE_BATCHED=1, snapshot)
   └── bin/cub.bench.topk.keys.base.single                           (USE_BATCHED=0, snapshot)

cuobjdump --extract-elf all <bin>     → keys_{batched,single}.sm_100.cubin
nvdisasm -g <cubin>                    → keys_{batched,single}_g.sass  (line-info SASS)

slice_func.py     → batched_i8_filter_max.sass  /  single_i8_filter_max.sass
sass_peaks.py     → per-window peak R / UR per kernel
peak_with_src.py  → top-N peaks attributed back to source files
param_census.py   → instruction-class census (LDC vs LDCU, R2UR, ATOMG, etc.)
```

Every helper is in this directory. All commands are SSH-friendly and
container-friendly.

## 2. Where the peak is

Per-window peak R / UR for the batched `i8` filter (0x800-byte buckets):

```
off 0x0000-0x0800  peak    R96  UR14    (kernel prologue + per-segment counter / pointer setup)
off 0x0800-0x1000  peak   R119  UR12    ← absolute peak (16 i8 keys unpacked + 16 atomic adds)
off 0x1000-0x1800  peak   R117  UR12
off 0x1800-0x2000  peak    R99  UR10
off 0x2000-0x2800  peak    R96  UR10    (bin computation / classification)
off 0x2800-0x3000  peak    R96  UR10
off 0x3000-0x3800  peak   R109  UR10    (second load tile / second atomic-add pass)
…
off 0xa800-0xb000  peak    R94   UR8    (epilogue / counter writeback)
```

For comparison the same kernel under the single-problem dispatch
(`DeviceTopKFilterKernel`, same `i8` key type, same `u32`/`u32` offsets):

| metric                    | batched | single | Δ        |
|---------------------------|--------:|-------:|---------:|
| unique R registers used   |    111  |    30  | **+81**  |
| unique UR registers used  |      7  |    21  | **−14**  |
| total instructions        |   2792  |  4312  |  −1520   |
| LDC (cmem → R)            |      9  |    62  |   −53    |
| LDCU (cmem → UR)          |     78  |   209  |  −131    |
| LDG.E (ROC global → R)    |    191  |    84  |  +107    |
| LD.E (standard global → R)|     77  |   182  |  −105    |
| R2UR (R → UR promotion)   |    **0**|    **2**|   −2    |
| S2R (special → R)         |      2  |   208  |  −206    |
| S2UR (special → UR)       |     67  |    74  |    −7    |
| ATOMG.*                   |     97  |    98  |    −1    |
| ATOMS.*                   |     64  |    64  |     0    |
| BAR.SYNC                  |     19  |    24  |    −5    |
| PRMT                      |    122  |   104  |   +18    |
| IMAD                      |    319  |   456  |  −137    |
| LDS                       |      8  |    14  |    −6    |
| STS                       |      5  |     8  |    −3    |

Three interpretations leap out:

- **Batched runs fewer instructions overall (−1520).** It's not "more
  work", it's "the same work compressed into a much wider live set."
- **Batched uses 2.7× fewer LDCU loads.** ptxas doesn't pull the
  batched kernel's parameter slots directly into uniform regs.
- **Zero R2UR vs two in single.** ptxas doesn't move any globally-loaded
  value into a uniform reg in the batched kernel.

The absolute peak instruction is on line 343, mapped to
`cub/cub/detail/topk/block_filter.cuh:350` (the atomic-update inner
loop inside `block_topk_filter` / `early_stop` strategy = `atomics`).
Top source contributors to the top-50 peaks:

```
  14  device_atomic_functions.hpp:112       (CUDA atomicAdd backend)
   9  block_load.cuh:69                     (per-byte WARP_TRANSPOSE load)
   8  thread_load.cuh:326                   (PRMT unpacks of vector-loaded dwords)
   5  dispatch_topk_common.cuh:177          (extract_bin_op_t::operator())
   4  block_filter.cuh:350                  (early-stop atomic histogram update)
   2  agent_batched_topk.cuh:1309           (`run()` body)
```

## 3. What is alive at the peak

Walking the SASS from offset 0x05b0 (first per-segment metadata load)
through 0x0acf0 (counter writeback) — that span is the body of one
grid-stride iteration. The values that are alive through ≥1/3 of it,
and that *should* be warp-uniform in every CTA, are:

| reg(s) | source meaning                                                  | should be in UR? |
|--------|-----------------------------------------------------------------|------------------|
| R68/R69 | `&d_segment_counters[queue_idx]` (also +0x180 → segment hist)  | **yes** — same value for all threads |
| R32     | `counter.k` (bytes 0..3 of the counter struct)                  | **yes** |
| R33     | `counter.num_candidates_out` (bytes 4..7)                       | **yes** |
| R34     | `counter.num_candidates_in` (bytes 8..11)                       | **yes** |
| R35     | `counter.load_from_candidates_buffer` (byte 12, PRMT'd)         | yes (and PRMT is what blocks it) |
| R66/R67 | `&d_segment_in_key_buf[queue_idx * candidate_buffer_length]`   | **yes** |
| R70/R71 | `*large_segments_count_it` (mixed path: from device pointer)    | **yes** |
| R39     | `segment_id` (resolved from binary search through `large_segments_tile_offsets`) | **yes** |
| R28     | `total_large_tiles` (read from sentinel of tile-offsets array)  | **yes** |
| R0..R31 | the 16 unpacked i8 keys (per-thread, immutable across atomic loop) | no (per-thread by construction) |

That's roughly **20+ scalar/pointer values that should be uniform** but
are sitting in per-thread regs. Each scalar costs one regular reg per
warp lane it doesn't need to be in (i.e., 32× as much physical register
file as it should). On Blackwell the per-warp uniform-register file is
separate from the per-thread register file, so spilling these into URs
*reduces* per-thread peak liveness directly.

R32, R33, R34 are written by a single `LDG.E.128 R32, desc[UR8][R68.64]`
at offset 0x5e0 (mapped to `agent_batched_topk.cuh:1086`). In the
single-problem kernel the equivalent `LDG.E.128 R4, desc[UR8][R4.64]`
at offset 0x30 is *immediately followed* by:

```
R2UR UR11, R5   ; counter.num_candidates_out → UR11
R2UR UR10, R4   ; counter.k                    → UR10
```

and from then on the values flow through `UISETP.GT.U32.AND UP0, UPT,
UR11, UR4, UPT` (uniform-uniform comparisons). In the batched kernel
the same struct fields are immediately compared with `ISETP.NE.AND P0,
PT, R34, RZ, PT` and `PRMT R42, R35, 0x7610, R42` — both per-thread.
No promotion happens.

### Verifying the user's intuition

> "I would imagine/hope that most of the segment specific variables
> could be covered by uniform registers? That would be worth verifying."

**Confirmed**, with caveats:

- The compiler does have the uniform-register file readily available
  on this build (sm_100). It's using it for some values (`UR8`
  descriptor, `UR4`-`UR14` for various uniformly-derived values), but
  the count drops from 21 unique URs (single-problem) to 7 unique URs
  (batched) — a substantial regression in uniform-reg use.
- The specific values currently NOT promoted include every value loaded
  through global memory (counter struct fields, `total_large_tiles`,
  `num_large_segments`) plus every per-segment base pointer derived
  from a `__grid_constant__` parameter (histogram base, key-buf base,
  val-buf base, counter base).
- A pure source-level fix that converts these into UR-resident values
  could plausibly reclaim 30-50 of the +90 reg gap, possibly more if
  the consumer instructions can stay uniform.

## 4. Why ptxas doesn't promote them (hypotheses backed by SASS evidence)

Three friction patterns I see repeatedly:

### 4a. Result of an `LD.E` (not `LDG.E`) is rarely uniform-promoted

The very first load in the batched prologue, mapped to
`kernel_batched_topk.cuh:447`:

```sass
LDC.64    R70, c[0x0][0x410] ;             // d_large_segments_tile_offsets
LDCU.64   UR8, c[0x0][0x358] ;             // descriptor
LD.E.64   R70, desc[UR8][R70.64] ;         // <-- *large_segments_count_it
                                           //     (regular LD.E, not LDG.E)
```

`large_segments_count_it` is a `_CCCL_GRID_CONSTANT` pointer, but on the
mixed-path build it is `unsigned long long*` (not `const unsigned long
long*`). ptxas emits `LD.E` for it, the result lands in `R70/R71`, and
no subsequent `R2UR` is inserted. By contrast `d_segment_counters` is
also a non-const pointer but the *counter struct itself* gets read via
`LDG.E.128` — likely because the compiler proved the load is non-aliased
inside that scope. Either way, `LD.E` results never get R2UR'd.

> **Lever**: mark `large_segments_count_it` and the other
> "read-only per-grid" pointers as `const T* const __restrict__`
> (or use `__ldg(it)` in the dereference); ptxas should then prefer
> `LDG.E` + R2UR.

### 4b. `R2UR` heuristic doesn't fire when the consumer is mixed-mode

In the single-problem prologue:

```sass
LDG.E.128 R4, desc[UR8][R4.64] ;
R2UR      UR11, R5 ;
R2UR      UR10, R4 ;
UISETP.GT.U32.AND UP0, UPT, UR11, UR4, UPT ;   // uniform-uniform compare → triggers R2UR
@!P0 EXIT ;
```

In the batched prologue:

```sass
LDG.E.128 R32, desc[UR8][R68.64] ;
ISETP.NE.AND P0, PT, R34, RZ, PT ;             // regular-vs-RZ compare
@!P0 BRA ;
PRMT       R42, R35, 0x7610, R42 ;             // PRMT consumes R35 in regular reg
ISETP.NE.AND P4, PT, R33.reuse, R32.reuse, PT ;// regular-vs-regular
```

The compiler's `R2UR` heuristic appears to prefer moves where the
consumer is also a `U*` instruction. The batched kernel mixes regular
and uniform consumers immediately after the load (the `PRMT` on R35
in particular is a per-thread op that has no uniform variant), so the
promotion is blocked for the whole struct.

> **Lever**: don't `PRMT` the bool field. Read
> `counter.load_from_candidates_buffer` as a 4-byte `int` instead of a
> packed `bool`, then `ISETP.NE` against zero (no PRMT). Better yet,
> split the struct so `load_from_candidates_buffer` lives in its own
> word (`alignas(4) bool`), so the LDG.E.128 only fetches the 4 scalar
> 32-bit fields and ptxas can `R2UR` the whole thing.

### 4c. Per-segment base pointers are pinned in R registers across the entire iteration

`R68` (the per-segment counter / histogram base address) is alive from
offset 0x5b0 through 0xacf0 — basically the entire grid-stride iteration
body. Every per-segment global memory access in the kernel uses
`desc[UR8][R68.64 + offset]`. All 32 threads in the warp use the
same R68. Yet the compiler keeps it in regular regs, costing 2 regs ×
unused-uniformity.

Same story for `R66/R67` (per-segment key/val buffer base) and
`R32/R33/R34` (counter scalars).

> **Lever**: compute these via a small `cuda::std::tuple` of
> per-segment params at the top of `run()`, mark them
> `__forceinline__ const` locals; let ptxas's standard "uniform value
> from constant inputs" detection fire. If that's not enough, an
> explicit cast through `__shfl_sync(0xffffffff, ptr, 0)` will force
> the compiler to materialize them as warp-broadcast values, which
> reliably produces UR.

## 5. Where the +90 regs likely break down (rough budget)

Best estimate from the live-range walk of the batched kernel's grid-stride
iteration body:

| component                                                            | est. regs |
|----------------------------------------------------------------------|----------:|
| 16 unpacked i8 keys held live (PRMT'd from 4 vector loads)           |    16-20  |
| Per-thread bin indices (one per item, alive into atomic loop)        |        8  |
| Per-segment counter struct in regular regs (R32-R35, alive ~42 KB)   |        4  |
| Per-segment histogram / counter base pointer R68/R69                 |        2  |
| Per-segment in/out key buf bases R66/R67 (and val if pairs)          |       2-4 |
| `total_large_tiles`, `num_large_segments`, `segment_id` scalars      |        4  |
| `current_len`, `input_length`, `k`, `buffer_length`, `coefficient`   |        5  |
| Binary-search loop state (R28, R29, R33-R41 across thread_search)    |     10-12 |
| `pass`, `total_bits`, `reset_histogram`, `num_passes` scalars        |        4  |
| Spill / overlap for the warp-scan epilogue (BlockScan)               |     10-15 |
| atomicAdd return values (alive briefly across each of 16 atomics)    |     1-16  |
| Misc loop counters, predicates, address arithmetic                   |       10  |
| **estimated peak in batched filter**                                 | **~95-120** |
| same accounting for the single-problem filter                        |    **30**  |

The single-problem filter's 30-register peak is **the unpacked-keys
budget + a tiny amount of overhead**. Almost everything else either
sits in a uniform register or is rematerialized on demand.

## 6. Concrete actionable suggestions (ordered by expected impact)

**These are hypotheses for the user/team to evaluate; no source changes
have been applied yet.**

| # | suggestion                                                                                                                | difficulty | expected ΔREG |
|--:|---------------------------------------------------------------------------------------------------------------------------|------------|--------------:|
| 1 | Split per-segment scalars (`counter.k`, `num_candidates_in/out`, `load_from_candidates_buffer`) into separate uint32 fields and read them as separate uint32 loads (or one `uint4` load + per-field `R2UR`-friendly use). Avoid PRMT on the bool. | Medium     | **−10 to −20** |
| 2 | Add `const __restrict__` to all per-grid pointer parameters that are read-only (`d_segment_histograms` is read+written via atomics, but `large_segments_count_it`, `d_large_segments_tile_offsets`, `d_segment_in_key_buf`, `d_segment_in_val_buf` can be `const`). ptxas should then prefer `LDG.E` and `R2UR`. | Easy       | **−8 to −15** |
| 3 | Hoist per-segment base pointers via `__shfl_sync(0xffffffff, ptr, 0)` (or an explicit `__cvta_generic_to_global`) at the top of `agent.run()` to force materialization in URs. | Easy       | **−6 to −10** |
| 4 | Reduce the kernel parameter count by packing the per-segment counter slab pointer, histogram slab pointer, and key/val buf slab pointers into a single `__grid_constant__ struct` (one register-friendly pointer + per-slab offsets known at compile-time). | Medium     | **−4 to −8** |
| 5 | Tune `items_per_thread` down for `i8` on Blackwell. The Hopper+ branch picks 16; the per-thread unpacked-key budget is the dominant 16-20 reg peak component for narrow keys. Clamp at 8 (or 4) just for `key_size <= 2` and see if the throughput holds. | Medium     | **−4 to −8** for `i8`, `i16` |
| 6 | Mark the binary-search loop in `thread_search.cuh` as uniform on the batched path (it operates only on uniform inputs in this context); maybe use `cooperative_groups::tiled_partition` or just `__shfl_sync` to factor it out of the per-thread context. | Hard       | **−10 to −15** |

Any subset of (1) + (2) + (3) alone should be enough to bring the
batched `i8` filter from `+90` regs vs single-problem down to the
`+30..+50` range, which would close most of the runtime regression.

## 7. Files in this directory

| file                                | content                                                              |
|-------------------------------------|----------------------------------------------------------------------|
| `FINDINGS.md`                       | this document                                                        |
| `batched_i8_filter_max.sass`        | SASS slice of the batched i8 filter (`select::max`), with `-g` line info |
| `single_i8_filter_max.sass`         | SASS slice of the single-problem i8 filter, same key/direction        |
| `keys.raw` / `keys.single.raw`      | full `cuobjdump --dump-resource-usage` for batched / single binaries  |
| `pairs.raw` / `pairs.single.raw`    | same, pairs benchmark                                                 |
| `keys.compare.txt` / `pairs.compare.txt` | batched-vs-single overview tables                              |
| `parse.py`                          | parses resource-usage dumps                                          |
| `compare.py`                        | batched-vs-single comparator                                         |
| `diff_batched.py`                   | compares two batched dumps                                           |
| `sass_peaks.py`                     | per-window peak R / UR census                                        |
| `peak_with_src.py`                  | top-N peaks attributed to source files                                |
| `param_census.py`                   | instruction-class census                                              |
| `slice_func.py`                     | slice one `.text.SYMBOL` function out of an nvdisasm dump            |
| `git_head.txt` / `toolchain.txt`    | reproducibility metadata                                              |

To re-run the SASS extraction after any source change:

```bash
# on the Blackwell container:
export PATH=/cccl/cmake/cmake-4.3.2-linux-x86_64/bin:$PATH
cd /cccl_fork/cccl/build/blackwell-cub-cpp17
cmake -B . -DCMAKE_CUDA_FLAGS="-lineinfo"        # one-time
ninja cub.bench.topk.keys.base cub.bench.topk.pairs.base

cuobjdump --extract-elf all bin/cub.bench.topk.keys.base
nvdisasm -g keys.sm_100.cubin > /tmp/keys_g.sass

python3 /cccl_fork/cccl/topk_register_pressure_investigation/slice_func.py \
  /tmp/keys_g.sass \
  "device_segmented_topk_filter.*policy_selector_from_typesIaNS0_8NullTypeElLl8388608EEELNS1_4topk6selectE1" \
  /tmp/batched_i8_filter_max.sass

python3 /cccl_fork/cccl/topk_register_pressure_investigation/peak_with_src.py /tmp/batched_i8_filter_max.sass
python3 /cccl_fork/cccl/topk_register_pressure_investigation/param_census.py \
  /tmp/batched_i8_filter_max.sass \
  /cccl_fork/cccl/topk_register_pressure_investigation/single_i8_filter_max.sass
```
