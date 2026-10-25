# Instruction and intrinsic catalogue for selection work on sm_100 class GPUs

Author: hw-researcher. Written 2026-10-25. Every availability statement below was checked against the PTX ISA 9.4 reference (the version shipped with CUDA 13.4 Update 2, `docs.nvidia.com/cuda/parallel-thread-execution`, downloaded 2026-10-25 and read in full text) or the CUDA Math API 13.4 reference, not from memory. Where a statement comes from somewhere else, the source is named. Measured costs are not in this file, they live in `TECHNIQUES.md` as `T` entries and in `state/results/hw-researcher/`. The column "micro row" names the measurement in `harness/micro/intrinsics_micro.cu` that prices the primitive.

## 1. Toolchain facts that gate everything

PTX ISA versions per CUDA release, read from the page titles of the archived PTX documentation (`docs.nvidia.com/cuda/archive/<cuda>/parallel-thread-execution`, 2026-10-25):

| CUDA | PTX ISA | What it added that matters here |
|---|---|---|
| 12.8 | 8.7 | sm_100, sm_100a (8.6): `redux.sync` on `.f32` with `.abs` and `.NaN` for sm_100a, `clusterlaunchcontrol`, `ldmatrix` `.m16n16` and `.b8`, `stmatrix` `.m16n8` and `.b8`, `.f32x2` arithmetic. sm_120 (8.7). |
| 12.9 | 8.8 | Family targets `sm_100f`, `sm_103f`, `sm_120f` (code runs on every later member of the family). sm_103. Three-input `min` and `max` for `.f32` (sm_100+). 256-bit `ld` and `st` (`.v8.b32`, `.v4.b64`, sm_100+). `redux.sync.f32` promoted to sm_100f. |
| 13.0 | 9.0 | sm_88, sm_110 (renamed from sm_101). `.blocksareclusters`. |
| 13.1 | 9.1 | Nothing for selection. |
| 13.2 | 9.2 | `min` and `max` on `.u8x4` and `.s8x4`, only on sm_107f and sm_120f families. |
| 13.3 | 9.3 | Nothing for selection (`clmad`, fabric instructions, mbarrier `.layout`). |
| 13.4 | 9.4 | sm_107, sm_107f, sm_107a (Rubin, compute capability 10.7). `set` on packed `.u8x4`, `.s8x4`, `.u16x2`, `.s16x2` (sm_107f only). `ldmatrix` `.s8.s4`. |

Compile targets and what they unlock on B200 and GB200 (both compute capability 10.0):

| nvcc target | Feature set | Runs on | Device macros in the device pass |
|---|---|---|---|
| `-arch=sm_100` (the swarm's mandated flag) | Baseline. No `redux.sync.f32`, no `setmaxnreg`, no `.b8` `ldmatrix`. Does have three-input `max.f32`, `max.u16x2`, `max.bf16x2`, `match.sync`, `elect.sync`, `bmsk`, `fns`, `prmt`, `ldmatrix`/`stmatrix` `.m8n8.b16`, `cp.async.bulk`, `mbarrier`, 256-bit loads. | every sm_10x, sm_11x, sm_12x device | `__CUDA_ARCH__ == 1000` |
| `-arch=sm_100f` | Baseline plus the sm_10x family features: `redux.sync.f32` (`.abs`, `.NaN`), `setmaxnreg`, `ldmatrix` `.m16n16`/`.m8n16`/`.b8`, `stmatrix` `.m16n8`/`.b8`, `clusterlaunchcontrol` multicast. | compute capability 10.0, 10.3, 10.7 (programming guide Table 28) | `__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000` (CCCL checks `>= 1000 && < 1100` in `libcudacxx/include/nv/detail/__target_macros`) |
| `-arch=sm_100a` | Everything of sm_100f plus sm_100 specific features (tcgen05 kinds, and so on). Nothing extra for selection. | compute capability 10.0 only | `__CUDA_ARCH_FEAT_SM100_ALL`, `__CUDA_ARCH_SPECIFIC__ == 1000` |

Two consequences. First, the only selection-relevant primitive that the mandated `-arch=sm_100` build cannot use is `redux.sync` on fp32 keys. An approach that wants it must ship an sm_100f build and a fallback (twiddle to u32, `redux.sync.max.u32`, untwiddle), and measure both. Second, the family and arch macros exist only in the device compilation pass. Host code that decides whether to launch a family-specific kernel needs its own define (`run_micro.sh` passes `-DMICRO_FAMILY=1` for the sm_100f build), otherwise the kernel compiles but is never launched.

CUDA C intrinsics do not expose everything. `__reduce_max_sync` and friends are integer only (programming guide, "Warp Reduce Functions", compute capability 8.x and later). `redux.sync.f32`, `bmsk`, `max.u16x2` with `.relu`, `max.bf16x2` as a bit pattern, `elect.sync`, `ldmatrix`, `stmatrix`, `movmatrix`, `cp.async.bulk`, `mbarrier.try_wait` and the packed `set` forms need inline PTX or the `cuda::ptx` wrappers in `libcudacxx/include/cuda/__ptx/instructions/` (present for `elect_sync`, `ldmatrix`, `stmatrix`, `shfl_sync`, `prmt`, `bmsk`, `cp_async_bulk`, `mbarrier_*`, `setmaxnreg`, `clusterlaunchcontrol`, not for `redux.sync`).

## 2. Catalogue

Column "use" says which selection job the primitive could do. Column "micro row" is the measurement name in `intrinsics_micro.cu`, "none" means catalogue only.

### 2.1 Warp reductions and votes

| Primitive | PTX ISA (CUDA) | Minimum target | CUDA C access | Use in selection | Micro row |
|---|---|---|---|---|---|
| `redux.sync.{add,min,max}.{u32,s32}`, `redux.sync.{and,or,xor}.b32` | 7.0 (11.0) | sm_80 | `__reduce_{add,min,max,and,or,xor}_sync(mask, v)` | Warp max of twiddled keys or of packed key-index words (W2 rounds), threshold broadcast, candidate counts (`add` of 0/1), digit histogram bins via `add` | `redux.sync.max.u32`, `W2 round ...` |
| `redux.sync.{min,max}{.abs}{.NaN}.f32` | 8.6 (12.8) for sm_100a, on sm_100f from 8.8 (12.9) | sm_100a or sm_100f or higher in the sm_10x family | inline PTX `redux.sync.max.f32 d, a, mask;` only | Warp max of raw fp32 keys without the twiddle. Semantics: `+0.0 > -0.0`, without `.NaN` NaN inputs are ignored and the result is canonical NaN only if all inputs are NaN, with `.NaN` any NaN input gives NaN. `.abs` reduces magnitudes. | `redux.sync.max.f32 (sm_100f)`, `redux.sync.max.abs.NaN.f32 (sm_100f)`, control `redux.sync.max.u32 on twiddled f32` |
| `vote.sync.ballot.b32`, `vote.sync.{all,any,uni}.pred` | 6.0 (9.0) | sm_30 | `__ballot_sync`, `__all_sync`, `__any_sync`, `__uni_sync` | Winner lane after a max (`ballot(v == m)` then `ffs`), candidate masks for compaction, early exit (`all` lanes done) | `ballot_sync`, `ballot + ffs + winner select` |
| `match.any.sync.{b32,b64}`, `match.all.sync.{b32,b64}` | 6.0 (9.0) | sm_70 | `__match_any_sync`, `__match_all_sync` | Lanes holding an equal digit (W3 histograms without shared memory), duplicate detection for tie handling, `match.all` as a one-instruction "all equal" test. Cost grows with the number of distinct values in the warp, so it is measured at 1, 2, 4, 8, 16 and 32 classes. | `match_any <U> classes` |
| `elect.sync d\|p, mask` | 8.0 (12.0) | sm_90 | inline PTX or `cuda::ptx::elect_sync` | Deterministic leader for the one-lane work after a reduction (the atomic, the shared memory write). Portable alternative: `__ffs(__activemask()) - 1`. | `redux chain + elect.sync per step` against `redux chain + ffs(activemask) leader per step` |
| `shfl.sync.{up,down,bfly,idx}.b32` | 6.0 (9.0) | sm_30 | `__shfl_*_sync` | Bitonic and merge network stages, broadcasts, segmented (sub-warp) exchanges through the `c` operand's segment mask. 32-bit only, a 64-bit value costs two. | `shfl_xor (single)`, `shfl_idx data-dependent lane`, `warp max 5-shfl tree u32/u64/f32` |
| `%lanemask_lt`, `%lanemask_le`, `%lanemask_gt`, `%lanemask_ge`, `%lanemask_eq` | 2.0 | sm_20 | `cuda::ptx::get_sreg_lanemask_lt()` or inline PTX | Free rank masks: `popc(ballot & lanemask_lt)` is the exclusive rank of a lane among candidates, no shuffle needed | none (one `S2R`) |

### 2.2 Scalar and SIMD compare-exchange

| Primitive | PTX ISA (CUDA) | Minimum target | CUDA C access | Use in selection | Micro row |
|---|---|---|---|---|---|
| `max.{u32,s32,u64,s64,u16,s16}`, `min` likewise, `max.relu.s32` | 1.0, `.relu` 8.0 (11.8) | all, `.relu.s32` sm_90 | `max()`, `min()` | Compare-exchange of twiddled keys. `SASS IMNMX`. | `alu max u32 (2-input)` |
| `max.f32 d, a, b, c` and `min.f32 d, a, b, c` (three inputs), with `.NaN` and `.abs`, no `.xorsign` | 8.8 (12.9) | sm_100 | inline PTX, or `fmaxf(fmaxf(a, b), c)` if ptxas fuses (checked in SASS, `FMNMX3`) | A bitonic compare-exchange against two candidates in one instruction, threshold clamps `max(x, t_lo, ...)`. Integer `max` has no three-input PTX form: `__vimax3_s32`, `__vimax3_u32`, `__vimin3_s32`, `__vimin3_u32` are CUDA Math API functions (13.4 reference, no stated compute capability) that compile to two PTX `max` and may or may not fuse (`VIMNMX3` exists in the Blackwell SASS table). | `alu max.f32 3-input ptx`, `alu fmaxf(fmaxf(v,a),b)`, `alu __vimax3_u32`, `alu __vimax3_s32`, `alu max(max(v,a),b) u32` |
| `max.u16x2`, `max{.relu}.s16x2`, `min` likewise | 8.0 (11.8) | sm_90 | `__vmaxu2`, `__vmaxs2`, `__vminu2`, `__vmins2` (Math API SIMD), `__vimax3_u16x2`, `__vimax3_s16x2`, `__vimin3_*_s16x2`, `__vibmax_u16x2` (returns the per-half predicates), `__vimax_s16x2_relu` | Two 16-bit keys per register (bf16 or fp16 twiddled to u16, or fp8 widened) with one compare-exchange per pair: the packed arrangement of W1 and the packed merge of B5. The predicate-returning `__vibmax_u16x2` gives the swap decision for values without a second compare. Native since sm_90 (`VIMNMX`). | `simd __vmaxu2 u16x2`, `simd max.u16x2 ptx`, `simd __vmaxs2 s16x2`, `simd __vimax3_u16x2` |
| `max.u8x4`, `max{.relu}.s8x4`, `min` likewise | 9.2 (13.2) | sm_107f or sm_120f or higher in the same family, not on sm_100 or sm_103 | `__vmaxu4`, `__vmaxs4`, `__vminu4`, `__vmins4` exist on every target but are emulated where the PTX form is missing | Four 8-bit keys per register (fp8 twiddled). On the hardware of this study the intrinsic is an emulation sequence, so the row measures what fp8 keys pay on sm_100. Native on Rubin. | `simd __vmaxu4 u8x4` |
| `max{.NaN}{.xorsign.abs}.bf16x2`, `.f16x2`, `.bf16`, `.f16`, `min` likewise | 7.0 (11.0), `.xorsign.abs` 7.2 | sm_80, `.xorsign.abs` sm_86 | `__hmax2`, `__hmax2_nan`, `__hmin2`, `__hmin2_nan` on `__nv_bfloat162` and `__half2` (`cuda_bf16.h`, `cuda_fp16.h`), or inline PTX on the `b32` bit pattern | Two bf16 or fp16 keys per register compared natively, no twiddle. Semantics per the PTX reference: `+0.0 > -0.0`, a NaN input loses to the other operand unless `.NaN` is given. `SASS HMNMX2`. The Blackwell SASS table also lists `VHMNMX` (SIMD fp16 three-input min and max), whether `__hmax2(__hmax2(a, b), c)` reaches it is a SASS question. | `simd __hmax2 bf16x2`, `simd max.bf16x2 ptx`, `simd max.f16x2 ptx` |
| `set.CmpOp.{u8x4,s8x4,u16x2,s16x2}` (per-lane masks `0xff` or `0xffff`) | 9.4 (13.4) | sm_107f or higher in the same family, not on sm_100 or sm_103 | inline PTX | Vectorized threshold test over packed keys, `popc` of the mask counts survivors. Rubin only, see `RUBIN_NOTES.md`. | none |
| `set`, `setp`, `selp`, `slct` on scalars | 1.0 | all | C comparisons and `?:` | The compare and select pairs that the knowledge base counts per key per pass (2.1). | none (part of every chain) |
| `lop3.b32` (three-input logic) | 4.3 (7.5) | sm_50 | compiler folds `a ^ (m \| HIGH_BIT)` shapes into one `LOP3` (knowledge base 2.1) | Twiddle and untwiddle in one logic op, mask merges | none |

### 2.3 Bit manipulation and packing

| Primitive | PTX ISA (CUDA) | Minimum target | CUDA C access | Use in selection | Micro row |
|---|---|---|---|---|---|
| `prmt.b32{.f4e,.b4e,.rc8,.ecl,.ecr,.rc16}` | 2.0 | sm_20 | `__byte_perm(a, b, sel)` | Assemble `(key16 << 16) \| index16` words, split them, swap halves, byte reverse, pick a 16-bit lane of a packed register, all in one `PRMT` | `alu prmt (__byte_perm)` |
| `cvt.pack.sat.{u16,s16,u8,s8,u4,s4,u2,s2}.{u32,s32} d, a, b{, c}` | 6.5 (10.2) | sm_72, sub-byte sm_75 | inline PTX (no intrinsic found) | Pack two 32-bit values into a `u16x2` register with saturation in one instruction (`SASS I2IP`), the other way to build packed words | none (unverified in SASS) |
| `bmsk.{clamp,wrap}.b32 d, pos, width` | 7.6 (11.6) | sm_70 | inline PTX or `cuda::ptx::bmsk_clamp`, `bmsk_wrap` | Segment masks for logical warps (`bmsk(start, width)`), lane range masks for ranks, replaces `((1 << w) - 1) << p` with its shift-by-32 edge cases | `alu bmsk.clamp` |
| `fns.b32 d, mask, base, offset` | 6.0 (9.0) | sm_30 | `__fns(mask, base, offset)` | The lane with rank `r` among candidates (`fns(ballot, 0, r + 1)`), the lane holding the k-th element in a compaction, all without a loop | `alu __fns` |
| `popc`, `clz`, `bfind`, `brev`, `bfe`, `bfi`, `szext` | 1.0 to 7.6 | all, `szext` sm_70 | `__popc`, `__clz`, `__brev`, `__ffs` (`FLO` plus adds) | Ranks, histogram counts, digit extraction. `szext` sign or zero extends a field, an alternative to `prmt` for unpacking 16-bit halves. | `alu __popc`, `alu __ffs` |
| `shf.{l,r}.{clamp,wrap}` (funnel shift) | 3.1 | sm_32 | `__funnelshift_l`, `__funnelshift_r` and `_lc`, `_rc` | Digit extraction across a 64-bit key pair, rotating packed words | none |

### 2.4 Shared memory, barriers and atomics

| Primitive | PTX ISA (CUDA) | Minimum target | CUDA C access | Use in selection | Micro row |
|---|---|---|---|---|---|
| `bar.sync` / `barrier.sync` with thread count, named barriers 0 to 15 | 1.0, `.aligned` 6.0 | all | `__syncthreads()`, `__syncthreads_count/and/or`, `asm("bar.sync id, count;")` | Block rounds. A named barrier over a subset (128 of 256 threads) lets two half blocks run independent rounds. | `__syncthreads (128/256/512 thr)` |
| `bar.warp.sync` | 6.0 | sm_30 | `__syncwarp()` | Warp-local shared memory handoffs (transpose, warp atomic round) | `syncwarp` |
| `atom{.sem}{.scope}.shared{::cta,::cluster}.{max,min,add,inc,dec,and,or,xor,exch,cas}.{u32,s32,u64,s64,b32,b64}`, `red` likewise without return | 1.2, 64-bit min and max 3.1, `::cluster` 7.8 | sm_12, 64-bit min and max sm_32, `::cluster` sm_90 | `atomicMax` and friends, unused result becomes `red` (`SASS REDS` or `ATOMS ... RZ`) | B2 rounds: fire-and-forget `max` into a slot, barrier, read back. No 16-bit integer atomics and no packed integer forms exist (16-bit atomics are `add` on `f16`, `bf16` and their pairs only), so a packed key-index word must be 32 bits wide. DSMEM atomics (`::cluster`) for cluster merges. | `atomicMax smem ...` rows (with return, uncontended to 256-way, banked), `fire-and-forget` rows (issue cost as a delta over an `IADD` chain), `warp max atomic32`, `blockmax atomic256`, `blockmax hybrid8` |
| `atom.global` vector forms `.v2/.v4/.v8` on `f16`, `bf16`, `f16x2`, `bf16x2`, `.v2/.v4.f32` with `add`, `min`, `max` | 8.1 (12.1) | sm_90 | `atomicMax` overloads do not cover them, inline PTX | Global memory only, not shared. Listed so that nobody expects a shared memory vector atomic. | none |
| `ld.shared`, `st.shared` (`LDS`, `STS`), 128-bit forms | 1.0 | all | pointers, `uint4` | Staging, slot read back | `STS+LDS round trip` |
| `ldmatrix.sync.aligned.m8n8.{x1,x2,x4}{.trans}.shared.b16` | 6.5 (10.2), `::cta` 7.8 | sm_75 | inline PTX or `cuda::ptx::ldmatrix` | Loads four 8x8 blocks of 16-bit values with an optional transpose: converts a blocked arrangement of packed 16-bit keys to striped (or back) in one instruction per four registers. `.m16n16`, `.m8n16` shapes and `.b8` type need sm_100a or sm_100f. | `transpose 4 words: stmatrix.x4 + syncwarp + ldmatrix.x4.trans` against `4 STS + syncwarp + 4 LDS`, `ldmatrix.x4 (address-dependent chain)` |
| `stmatrix.sync.aligned.{m8n8,m16n8}.{x1,x2,x4}{.trans}.shared.{b16,b8}` | 7.8 (11.8), `.m16n8` and `.b8` 8.6 | sm_90, `.m16n8` and `.b8` sm_100a or sm_100f | inline PTX or `cuda::ptx::stmatrix` | The store half of the transpose above | same rows |
| `movmatrix.sync.aligned.m8n8.trans.b16 d, a` | 7.8 (11.8) | sm_75 | inline PTX | Transposes an 8x8 16-bit matrix across the warp in registers, no shared memory (`SASS MOVM`). One instruction per register, a candidate for the blocked to striped change of 16-bit keys with no smem traffic. | none (not yet in the micro, follow-up) |
| `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [dst], [src], size, [mbar]`, `.shared::cta` destination | 8.0 (12.0), `.shared::cta` 8.6 | sm_90 | inline PTX or `cuda::ptx::cp_async_bulk` | Fill a block's shared memory with a segment through the TMA unit (`SASS UBLKCP`), one thread issues, no register staging. 16-byte aligned, size a multiple of 16. | `fill smem 4KB/16KB: cp.async.bulk + mbarrier` against `LDG.128 + STS + bar (control)` |
| `mbarrier.init`, `.inval`, `.arrive`, `.arrive.expect_tx`, `.try_wait.parity`, `.test_wait` | 7.0 (11.0), `try_wait` and `expect_tx` 7.8 and 8.0 | sm_80, `try_wait` sm_90 | `cuda::barrier` or inline PTX or `cuda::ptx::mbarrier_*` | Completion of the bulk copy, producer and consumer handoff in warp-specialized blocks. The phase parity must be tracked by the waiter. | part of the fill rows |
| `ld.global{.nc}.v8.b32` and `.v4.b64` (256-bit loads), `st` likewise | 8.8 (12.9) | sm_100 | no intrinsic known, inline PTX, 32-byte alignment presumed | Eight 32-bit keys per load instruction for block tiles | none (unverified) |
| `st.async`, `red.async` to a remote CTA's shared memory with mbarrier completion | 8.1 (12.1) | sm_90 | `cuda::ptx::st_async`, `red_async` | Cluster-level merges (B5 across CTAs, the knowledge base 3.4 cluster ideas) | none |

### 2.5 Register and scheduling control

| Primitive | PTX ISA (CUDA) | Minimum target | CUDA C access | Use in selection | Micro row |
|---|---|---|---|---|---|
| `setmaxnreg.{inc,dec}.sync.aligned.u32 imm` | 8.0 (12.0) | sm_90a, sm_100a, sm_100f and later families | inline PTX or `cuda::ptx::setmaxnreg` | Warp-specialized blocks hand registers from producer warps to the selection warps. Needs a warpgroup, a launch with the maximum register count set, and 24 to 256 in multiples of 8. Not usable in the one-warp latency harness. | none |
| `clusterlaunchcontrol.try_cancel`, `.query_cancel` | 8.6 (12.8) | sm_100, `.multicast::cluster::all` sm_100a or sm_100f | `cuda::ptx::clusterlaunchcontrol_*` | Persistent batched kernels claim not yet launched clusters (work stealing). Relevant to the device-level batched kernel, not to a warp or block primitive. | none |
| `griddepcontrol` (programmatic dependent launch) | not re-verified in this pass | sm_90 per earlier reading | `cudaGridDependencySynchronize` | Overlaps the next kernel's prologue with the previous kernel's tail in the batched setting. Out of scope for primitives. | none |
| `%clock64`, `%smid` | 1.0 | all | `clock64()`, `cuda::ptx::get_sreg_smid()` | The timer and the anti-DCE sink of the harness | none |

## 3. Semantics worth remembering

- Floating point `max` and `min` in PTX, in `redux.sync.f32` and in `max.bf16x2` all treat `+0.0 > -0.0`. The study's oracle treats minus zero and plus zero as equal under the tie-break requirement. A value-only max therefore returns `+0.0` when both zeros are present, and the index pass that compares `key == max` with IEEE equality matches both zero keys, so the smaller index wins naturally. Radix and atomics approaches that twiddle must normalize `-0.0` to `+0.0` themselves (knowledge base 2.1 and 4.4).
- Without `.NaN`, a NaN input to `max.f32`, `max.bf16x2` or `redux.sync.max.f32` loses to the other operand and the result is canonical NaN only when all inputs are NaN. With `.NaN` any NaN input poisons the result. The focus set has no NaN, but a contract that documents NaN behavior can rely on either form.
- `redux.sync.add.s32` truncates to 32 bits. `match.any` returns the mask of lanes with an equal value, which for twiddled keys is also the mask of equal keys.
- `elect.sync` elects deterministically (the same lane for the same mask every time), so its choice is reproducible, unlike `__activemask()` based choices after divergence.
- A packed word `(twiddled_key16 << 16) | (0xffff - index)` makes `redux.sync.max.u32` and a shared `atomicMax` an exact arg max that prefers the smaller index for equal keys, up to 65536 indices per segment. With `index` stored directly it prefers the larger index. Both are deterministic. The 64-bit packed word for 32-bit keys is possible for `redux` (two instructions through a hi then lo cascade) and for atomics (`atom.max.u64`), but the July measurements show the 64-bit shared atomic collapsing under same-slot bursts, which the re-measurement in `TECHNIQUES.md` revisits.
- Shared memory atomics have no 16-bit integer forms, no packed integer forms and no vector forms. Vector atomics (`.v2`, `.v4`, `.v8`) exist for global memory only.

## 4. Excluded on purpose

`tcgen05.*` and tensor memory, `wgmma`, `mma` (no selection use beyond fused scoring), `multimem.*` and `fabric.*` (multi-GPU), `cp.async.bulk.tensor` and `tensormap.*` (the plain bulk copy suffices for a segment fill), `applypriority` and `createpolicy` (cache hints, a throughput regime question for the batched kernel), `nanosleep`, `trap`, `brkpt`, `pmevent`, texture and surface instructions.

## 5. Open questions the hardware answered (2026-10-25, B200 and GB200, CUDA 13.4.92, numbers in `TECHNIQUES.md`)

1. Three-input forms: `__vimax3_u32`, `__vimax3_s32`, `max(max(v, a), b)`, `fmaxf(fmaxf(v, a), b)`, the PTX `max.f32 d, a, b, c` and `__vimax3_u16x2` all reach `VIMNMX3`, `FMNMX3` or `VIMNMX3.U16x2`, nested `__hmax2` reaches `VHMNMX.BF16_V2`, and `max(v + a, b)` reaches `VIADDMNMX`. All cost the same 6 cycles of dependent latency as the two-input form, so the third input and the fused add are free.
2. `__vmaxu4` on sm_100 is 10 instructions (`LOP3` x5, `SHF`, `IMAD.IADD`, `PRMT`) and 28 cycles of dependent latency against 6 for the native `VIMNMX.U16x2`. Widen fp8 keys to 16 bits before any compare network on Blackwell.
3. `redux.sync.max.f32` compiles to `CREDUX.MAX.F32` on the sm_100f build and costs 21 cycles, the same as the u32 form (22), against 40 for twiddle plus u32 redux plus untwiddle.
4. `match.any` costs 50 cycles with one distinct value and about 11 more per additional distinct value, 391 cycles with 32 distinct values. Narrow digits only.
5. Transposes of four 16-bit-pair registers: `movmatrix` 45 cycles, `stmatrix` plus `ldmatrix.trans` 63, scalar `STS` plus `LDS` 83.
6. Dependent fills from L2: 4 KB in 460 (LDG plus STS) against 490 (cp.async.bulk), 16 KB in 637 against 595 on GB200. Latency bound either way, the bulk path saves registers and instructions, not time.
7. The July fire-and-forget facts hold on CUDA 13.4.92 and on GB200: atomic256 round 74 cycles, u64 round 4500 (the 64-bit shared max is an `ATOMS.CAST.SPIN.64` CAS loop), redux before the atomic 149. New: the uncontended u32 atomic with return is 55 cycles, not 115, and a completion check found 0 stale reads after the barrier in 64 by 31 rounds.
8. Methodology: an smid-guard sink inside the timed region hides chains shorter than about 80 cycles, and ptxas reassociates chains of one associative op. The microbenchmark hoists the smid read and measures ALU latencies with IMAD-interleaved pair chains (entries T14 and T15 of `TECHNIQUES.md`).
