# `device_segmented_topk_filter_kernel`: register-pressure analysis (int8/int8)

Target kernel:
`device_segmented_topk_filter_kernel<policy_selector_from_types<int8_t, int8_t, ...>, select::max, ...>`

Baseline: dev (`tmp/perf-eval-baseline`, commit `6dae99c414`, sm_100, CTK 13.1.115).

## Methodology

1. Rebuilt `cub.bench.topk.pairs.base` on the B200 container with
   `-Xptxas=-v -lineinfo -DTUNE_OffsetT=::cuda::std::int32_t`. `-lineinfo`
   embeds DWARF line-table info into the cubin without changing
   optimization (same `Used 64 registers, 1028 bytes smem` PTXAS report).
2. Located the cubin symbol for `KeyT=int8, ValueT=int8, select::max`
   (`policy_selector_from_typesIaalLl8388608E ... selectE1`).
3. Used `nvdisasm` twice on the cubin:
   - `--print-life-ranges -lrm count --cuda-function-index 611`
     produces an SASS dump where each instruction has trailing columns
     `// | regs | preds | uregs | upreds |` giving the count of live
     general / predicate / uniform-reg / uniform-predicate registers.
   - `--print-line-info --cuda-function-index 611` produces the same
     SASS annotated with `//## File "...", line N` markers (the two
     options are mutually exclusive in `nvdisasm`, but their outputs
     agree on instruction offsets so we cross-reference by offset).
4. Sidenote: `nvdisasm --print-life-ranges` on the full cubin aborts with
   `Invalid register count : '255'` because one of the `single_cta`
   instantiations (index 608, `K=int8, V=short`) trips a tool bug; we
   work around it by always passing `--cuda-function-index`.

Function index 611 was found by probing — `cuobjdump --dump-elf-symbols`
indices do not match `nvdisasm`'s. A helper `find_kernel.sh` shell
script finds the mangled symbol name given (KeyT, ValueT, select,
logical_name); pair with `cuobjdump --dump-sass --function "<sym>"` for
plain SASS, or feed an index to `nvdisasm` for life ranges / line info.

## Findings

### Peak live-register count: 60 (cap is 64)

PTXAS reports `Used 64 registers` for this instantiation. nvdisasm's
liveness analysis reports a **peak of 60 simultaneously live general
registers** (plus 3 predicates, 2 uniform registers, 0 uniform
predicates). The remaining 4 of the 64 allocated registers are tied up
by 64-bit register-pair alignment + a few transient scratch values.

### Liveness sits at 50-58 regs for ~70% of the kernel

| live regs | % of instrs | bar |
|---:|---:|---|
| 40 | 0.3% | |
| 45 | 0.9% | # |
| 50 | 5.8% | ########### |
| **53** | **10.1%** | **####################** |
| **54** | **10.5%** | **#####################** |
| **55** | **10.6%** | **#####################** |
| **56** | **11.0%** | **#####################** |
| 57 | 8.8% | ################# |
| 58 | 6.1% | ############ |
| 59 | 3.8% | ####### |
| 60 | 0.8% | # |

It is not a brief spike. **41% of instructions execute with 53-56
registers alive**, so even a moderate drop in the persistent state
would cap the allocation a lot lower than 64.

### Top contributors to peak-liveness instructions

| count | file | line | what it is |
|---:|---|---:|---|
| 91 | `cub/device/dispatch/dispatch_topk_common.cuh` | 179 | `(bits < *kth_key_bits) ? ... : (bits == *kth_key_bits) ? ...` |
| 61 | `cub/device/dispatch/dispatch_topk_common.cuh` | 177 | `bits = (bits >> start_bit) << start_bit;` |
| 18 | `cub/device/dispatch/dispatch_topk_common.cuh` | 180 | candidate-class ternary |
| 16 | `cub/thread/thread_load.cuh` | 326 | scalar `LDG.E.U8` of `*kth_key_bits` |
| 14 | `cub/detail/topk/block_filter.cuh` | 350 | `sel_iter[r.first] = sel_xform(keys[j])` |
| 13 | `cub/device/dispatch/dispatch_topk_common.cuh` | 43 | `topk_index_gather_op::operator()` (value gather) |
| 12 | `cub/block/block_load.cuh` | 69 | per-thread `BlockLoad` |
| 12 | `cub/detail/topk/block_partition.cuh` | 694, 697 | `cand_iter[r.first] = cand_xform(keys[j])`, value sink |
| 12 | `cub/detail/topk/block_partition.cuh` | 674 | `sel_iter[r.first] = sel_xform(keys[j])` |

All of these are inside `block_partition::partition`'s unrolled
`for (j = 0..ITEMS_PER_THREAD-1)` loop: classify each loaded key as
selected/candidate/rejected, then conditionally append key+value to a
queue.

### What lives at the peak

Looking at the actual SASS window around the peak instruction:

```
0x23b0..0x24a0   LDG.E.U8 R45,R27,R22,R24,R26,R44,R46,R48,R50,R52,
                          R54,R56,R58,R60,R23,R25   // 16 separate u8 LDGs
0x24b0           WARPSYNC.ALL
0x24c0           BAR.SYNC.DEFER_BLOCKING 0x0
0x24d0   LDG.E.U8 R47, desc[UR6][R36.64+0x10]       // *kth_key_bits
0x24e0   LOP3.LUT R28, R13, 0x7f, R45, 0x60, !PT    // classify keys[0]
...      (selected-path: atomic reserve + STG)
...      (candidate-path: value gather + STG)
0x2700   LDG.E.U8 R47, desc[UR6][R36.64+0x10]       // RELOAD *kth_key_bits
0x2720   LOP3.LUT R28, R13, 0x7f, R27, 0x60, !PT    // classify keys[1]
0x2730   BSSY.RECONVERGENT B5, `(.L_x_42)
* 0x2740   LOP3.LUT R29, R47, 0xffff, RZ, 0xc0, !PT  // peak: 60 regs
  0x2750   LOP3.LUT R28, R28, 0xff, RZ, 0xc0, !PT
  0x2760   ISETP.GT.U32.AND P2, PT, R28, R29, PT
```

What's alive at the 60-reg peak, in rough buckets:

| bucket | regs | notes |
|---|---:|---|
| `keys[0..15]` from `BlockLoad` | ~14-16 | the compiler unrolls the 16-iter j-loop, so all not-yet-consumed `keys[j]` stay live across iterations |
| `per_segment_state_t` fields | ~30 | see breakdown below |
| current-iteration scratch | ~3-5 | extracted `bits`, masked `*kth_key_bits`, candidate-class flags |
| stack pointer (`R1`), CTA-id, lane masks, etc. | ~5 | always-alive |

#### `per_segment_state_t` breakdown (`agent_batched_topk.cuh:1399`)

| field | dtype | regs |
|---|---|---:|
| `empty`, `early_stop`, `will_buffer`, `load_from_candidates_buffer` | 4× `bool` | 1-4 (often promoted to predicates) |
| `d_keys_in`, `d_keys_out` | `constant_iterator<ptr,offset>` | 2-4 each |
| `d_values_in`, `d_values_out` (`[[maybe_unused]]`) | `constant_iterator<ptr,offset>` | 2-4 each — still materialized |
| `in_key_buf`, `out_key_buf` | `int8_t*` | 2 each (64-bit) |
| `in_val_buf`, `out_val_buf` (`[[maybe_unused]]`) | `int8_t*` | 2 each — still materialized |
| `segment_histogram` | `uint32_t*` | 2 |
| `segment_counter` | `counter_t*` | 2 |
| `pass` | `int` | 1 |
| `current_k`, `current_len`, `input_length_actual`, `num_full_tiles`, `partial_items`, `segment_tiles_input` | 6× 32-bit | 6 |
| `slab_base`, `queue_segment_end` | 2× `LargeSegmentTileOffsetT` (int32) | 2 |

Rough total: **24-34 regs** are tied up in the segment-state struct
through the entire tile body. Two iterator fields (`d_values_in`,
`d_values_out`) and two pointer fields (`in_val_buf`, `out_val_buf`)
are `[[maybe_unused]]` for keys-only configurations but still get
materialized — and even in the int8/int8 pairs case they remain
copies of values reachable elsewhere.

### What's *not* the bottleneck

- **Smem**: only 1028 B (a single small histogram slab + a sliver of
  partition scratch). The smem budget is fine and not where we should
  spend register-pressure relief.
- **Spilling**: int8/int8 has zero stack frame, zero spill stores,
  zero spill loads. The compiler successfully kept the kernel
  register-resident, just at a very high water mark.
- **`*kth_key_bits` is not loaded into registers permanently** — it's
  reloaded each j-iteration (`LDG.E.U8 R47, [R36+0x10]` at 0x24d0 and
  0x2700 — that's an LDG **per item**, 16 redundant loads per tile per
  thread). This wastes memory bandwidth/latency, not registers
  (only one register slot is used for it at a time). Worth fixing for
  perf, but not for register pressure.

## Proposals

Ordered roughly by expected register-saving impact.

### P1. Spill `per_segment_state_t` (or its cold half) to `__shared__`

By far the biggest pool of constant-but-alive registers. The 4 booleans
+ 6 32-bit counters + 2 32-bit tile offsets + 4 `constant_iterator`
fields + 6 64-bit pointers total roughly 24-34 registers tied up for
the entirety of every tile-body call.

The git history of `tmp/perf-eval-baseline` shows you already tried this
direction (`d37f5e639f topk(experiment): move filter agent
per_segment_state to smem`, `0879b72ff9 topk(experiment): hybrid
hot-register / cold-smem segment state in filter agent`,
`fdcc4a325c topk(experiment): UR-promote per-segment metadata in filter
agent`). The currently-checked-out tip is a Revert chain on top of those,
so the production state is the "everything in registers" version. The
register data confirms this is where the biggest wins are. Two
sub-strategies:

- **Cold-state-in-smem**: keep only the per-tile-hot fields in
  registers (e.g. `pass`, `current_k`, `current_len`, the in/out key
  buffer pointers and `segment_counter`) and store the rest in a small
  `__shared__` block read by lane-0 on segment enter, broadcast via
  uniform registers when only constant-across-warp values are needed.
  Smem cost: ~80 B per segment slab; reg savings: ~15-20.
- **All-state-in-smem-plus-UR-broadcast**: store the entire
  `per_segment_state_t` once per segment in smem and use `LDS.U` /
  uniform reads in the inner loop. Reg savings: ~30; perf risk:
  uniform load latency in the hot path. Promising for int8 (8-bit
  classifier is short, so the uniform-load latency hides) but might
  regress the int/long path that already spills.

### P2. Remove `[[maybe_unused]]` value-channel fields from `per_segment_state_t`

For the keys-only case `d_values_in`, `d_values_out`, `in_val_buf`,
`out_val_buf` are unused — the attribute marker prevents the warning
but does not delete the storage, and ptxas can't eliminate them
across the kernel body. Estimated savings on the keys-only path:
4-12 registers. Path: introduce an empty `value_channel_state_t`
specialization for `keys_only=true` and EBO/conditionally-empty it
inside `per_segment_state_t` instead of `[[maybe_unused]]`. Even on
the pairs path, several of these are derivable from each other and
keeping all four materialized is wasteful — `in_val_buf` and
`out_val_buf` can be computed from the segment-id + base pointers
already kept for the key buffers, so the structurally-empty version
is preferable when feasible.

### P3. Pack int8 keys[] post-`BlockLoad` (4 keys per 32-bit register)

The compiler keeps all 16 keys live across the unrolled j-loop. For
`KeyT=int8` you can collapse them to 4 packed 32-bit registers and
unpack with `BFE` / `PRMT` inside the classifier. Reg savings: up
to 12 (16 -> 4). This is specific to 1-byte keys (where the
classifier inputs are smaller than the storage register), but that's
the exact case at issue here.

If a manual pack is too invasive, **forcing `BLOCK_LOAD_VECTORIZE`**
in the policy for `sizeof(KeyT) == 1` may achieve the same effect —
vectorized loads land 4 bytes per register and the unpacking happens
naturally. Worth a separate experiment recording.

Note: this proposal does *not* help the int/short/long key cases
(where each key already occupies a full 32-bit register or more), so
it should be evaluated alongside P1/P2.

### P4. Reduce items_per_thread for the int8 policy

`items_per_thread = 16` for int8 keys at this policy. The 16-byte
unroll is exactly proportional to the keys[] regs. Dropping to 12 or
8 cuts `keys[]` registers proportionally (~4-8 regs) at the cost of
more tiles per launch. Worth a quick policy tweak + bench-run to see
if the throughput hit is acceptable.

### P5. Hoist `*kth_key_bits` out of the j-loop *(perf, not register pressure)*

`identify_candidates_op_t::operator()(T key) const` reads
`*kth_key_bits` once per call, and the compiler does not CSE the
load across calls because each call has different `key`. The SASS
shows `LDG.E.U8 R47, [R36+0x10]` repeated 16x per tile per thread.
`*kth_key_bits` is written by `device_segmented_topk_finalize_filter_kernel`
between filter passes, so within a single filter kernel invocation
it is constant per CTA. Load it once into a thread-local, pass into
the classifier:

```cpp
// before the j-loop in block_partition::partition
const auto kth = *identify_op.kth_key_bits;

// inside the operator():
candidate_class operator()(T key, unsigned_bits_t kth) const
{
  ...
  return (bits < kth) ? selected : (bits == kth) ? candidate : rejected;
}
```

Reg cost: 1 (already 1 in the current SASS, just slightly longer-lived).
Perf savings: 15 redundant `LDG` per tile per thread.

### P6. Sanity-check P2-P5 do not regress the spilling paths (int/short)

The biggest tracked regressions versus main are not the int8 reg
peak — they're the heavy spilling on `KeyT=int, ValueT=long`
(140 B / 192 B spill_stores/loads on a 32-reg cap). The same
analysis pipeline (`record_snapshot.sh` + `analyze_liveness.py`)
should be re-run on that instantiation; the structural lever
(per_segment_state in regs) is shared, so P1 likely helps there
too, but the keys[]-packing argument of P3 does not transfer.

## Files added

- `topk_perf_tracking/sass/int8_int8.sass`        — full cuobjdump SASS for the kernel
- `topk_perf_tracking/sass/int8_int8.lrm_v2.txt`  — SASS + life-range counts
- `topk_perf_tracking/sass/int8_int8.lineinfo_v2.txt` — SASS + `//## File ..., line N`
- `topk_perf_tracking/analyze_liveness.py`        — joins lrm + lineinfo, dumps the peak-line tables and SASS context above
- `topk_perf_tracking/find_kernel.sh`             — locates the mangled symbol given (KeyT, ValueT, select, logical_name)
- `topk_perf_tracking/find_nvdisasm_index.sh`     — brute-force probe to find the nvdisasm `--cuda-function-index` of a given symbol
