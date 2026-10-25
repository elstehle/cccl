# Rubin generation notes for selection primitives

Author: hw-researcher. Written 2026-10-25 from public NVIDIA documentation only. Nothing in part A is measured. Part B lists what has been measured, which is nothing on Rubin hardware, because none is available to this study and CUDA 13.4 calls the platform support a developer preview.

## A. What the public documentation says

Sources, all fetched 2026-10-25: PTX ISA 9.4 (`docs.nvidia.com/cuda/parallel-thread-execution`, the version shipped with CUDA 13.4 Update 2), CUDA Toolkit 13.4 Update 2 release notes, the nvcc 13.4 user guide (GPU architecture table), the CUDA Programming Guide 13.4 appendix "Compute Capabilities", and the CUDA Binary Utilities 13.4 instruction set reference.

### A.1 Target names and compatibility

- Rubin is compute capability 10.7. The nvcc 13.4 architecture table lists `sm_107`, `sm_107f`, `sm_107a` under "Rubin support". The cuBLAS 13.4 notes say "Added support for the NVIDIA Rubin (compute capability 10.7) GPU architecture". PTX ISA 9.4 adds the `sm_107`, `sm_107f` and `sm_107a` targets.
- Rubin is in the sm_10x family together with sm_100 (B200, GB200) and sm_103 (B300). PTX ISA 9.4 Table 70 "Architecture Families" lists the sm_10x family as `sm_100f, sm_103f, sm_107f, future targets in sm_10x family`. The programming guide Table 28 says code built for `compute_100f` runs on compute capability 10.0, 10.3 and 10.7, `compute_103f` on 10.3 and 10.7, and `compute_107f` on 10.7 only.
- Consequence for this study: a primitive that needs a family-specific feature of sm_100f (the only one in our catalogue is `redux.sync` on `.f32`) can be built once for `sm_100f` and will run unchanged on B300 and Rubin. A primitive built for `sm_100a` runs on B200 and GB200 only.
- sm_110 is not Rubin. PTX ISA 9.0 renamed `sm_101` to `sm_110` (an sm_11x family of its own, Table 70). sm_120 and sm_121 are the consumer and embedded Blackwell parts with 100 KB of shared memory per block.
- The CUDA 13.4 release notes say that Vera Rubin platform support is available as a developer preview and is not intended for benchmarking, performance analysis or production deployment. Any Rubin number produced before that changes would not be citable.

### A.2 Shared memory

- Compute capability 10.7 has 328 KB of shared memory per SM and 327 KB per thread block (programming guide Table 31), against 228 KB and 227 KB on 10.0, 10.3 and 11.0, and 100 KB on 12.x. The carveout list for 10.7 is 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 and 328 KB (Table 32).
- The 328 KB configuration is opt-in: "For devices of compute capability 10.7, kernels that use the 328 KB shared-memory configuration must explicitly enable `cudaSharedMemoryModeAllowOversizedSharedMemory` by setting either the `cudaFuncAttributeSharedMemoryMode` function attribute or the `cudaLaunchAttributeSharedMemoryMode` launch attribute" (Table 32, footnote 4).
- Registers per SM stay at 64 K 32-bit registers and 255 per thread (Table 31). I could not read the merged cells of Table 30 (resident warps and blocks per SM) reliably for the 10.7 column, so the warp and block limits for Rubin should be read from `cudaGetDeviceProperties` on hardware before anyone relies on them.
- Relevance: the cluster agent of `DeviceBatchedTopK` keeps segments resident in shared memory and re-reads them per pass (knowledge base 2.1 and 3). A 44 percent larger shared memory per SM raises the segment size that stays resident in one CTA and lowers the number of CTAs per cluster for a given segment, which is the regime where cluster barrier cost dominates (knowledge base 2.3). PTX ISA 9.4 also adds the `.minperctamemory` directive and the `%perctamemoryoffset` and `%perctamemorysize` special registers (release notes 9.4), whose relation to the oversized configuration is not described in the parts I read.

### A.3 New instructions that could matter for selection, by target

| Instruction or form | PTX ISA | Targets | Why it matters for top-k |
|---|---|---|---|
| `min.u8x4`, `min.s8x4`, `max.u8x4`, `max.s8x4` (with `.relu` for signed) | 9.2 | `sm_107f` or higher in the family, `sm_120f` or higher in the family | Native 4-lane 8-bit min and max. On sm_100 and sm_103 these forms do not exist, so `__vmaxu4` is emulated there. fp8 keys twiddled to u8 can run compare-exchange networks four keys per register on Rubin. |
| `set.CmpOp.type` with `.u8x4`, `.s8x4`, `.u16x2`, `.s16x2` | 9.4 | `sm_107f` or higher in the family | Packed compares that write per-lane masks of `0xffff` or `0xff`. A threshold filter over packed 16-bit or 8-bit keys becomes one instruction per 2 or 4 keys, and `popc` of the mask word counts candidates. Not available on B200, GB200 or B300. |
| `add`, `sub`, `mul`, `fma` on `.f16x2`, `.bf16x2`, `.f32x2` mixed precision | 9.4 | `sm_107f` or higher in the family | Two-wide fp32 arithmetic. Only relevant where a selection fuses with the scoring that produces the keys. |
| `cvt` with `.pzo` and `.rz` for the FP8, FP6 and FP4 packed types, `.ue5m3x2` | 9.4 | not read | Key conversion, not selection. Listed for completeness. |
| `ldmatrix` with `.s8.s4` types | 9.4 | `sm_90a`, `sm_100f`, `sm_110f`, `sm_120f` or higher in the family | 4-bit to 8-bit expansion on load. Not a selection use. |
| `spcompress`, `spdecompress` | 9.4 | not read | Sparsity metadata compression of a vector. The SASS table lists `SPARSIFY` (generate metadata and compress a vector) and `DECOMPRESS`. A compaction of surviving candidates (approach B3) could in principle use a hardware compress, but the PTX semantics are tied to the 2:4 sparsity format and I have not read them. Open question. |
| `prefetch` with `.valid_addr` and `.L1::32B`, `createpolicy.range.fabric`, `applypriority.async.bulk`, `fabric.*`, mbarrier `.multicast::cluster::32b` | 9.3 and 9.4 | various | Memory system and multi-GPU features, not selection primitives. |

Already family-specific on sm_100f and therefore available on Rubin with the same binary: `redux.sync` on `.f32` with `.abs` and `.NaN` (PTX 8.6, `sm_100a`, on `sm_100f` from PTX 8.8), `setmaxnreg`, the `.m16n16` and `.m8n16` shapes and `.b8` type of `ldmatrix` and `.m16n8` and `.b8` of `stmatrix`, `clusterlaunchcontrol`.

### A.4 SASS opcodes shared by Blackwell and Rubin

The CUDA Binary Utilities 13.4 instruction set reference has one table "Blackwell and Rubin Instruction Set" for compute capabilities 10.0, 10.3, 10.7, 11.0 and 12.0, so it cannot tell a Rubin-only opcode from a Blackwell one. Opcodes in that table that are relevant to selection and that an optimizer should look for in `cuobjdump -sass` output: `REDUX` and `CREDUX` (reduction of a vector register into a uniform register, coupled form), `VIMNMX` and `VIMNMX3` (SIMD integer min and max, 2 and 3 input), `VIADDMNMX` (SIMD integer add fused with min or max), `VHMNMX` (SIMD fp16 3-input min and max), `FMNMX` and `FMNMX3`, `HMNMX2`, `IMNMX`, `MATCH`, `VOTEU`, `SHFL`, `PRMT`, `BMSK`, `FLO`, `POPC`, `LDSM` and `STSM` (matrix load and store from shared memory), `MOVM` (move matrix with transposition), `ATOMS` and `REDS` (shared memory atomics with and without return), `UBLKCP` (bulk copy), `SYNCS` (the mbarrier unit), `LDGMC` (reducing load), `I2IP` and `F2IP` (convert and pack). Which of these the compiler emits for which PTX on sm_100 is what the microbenchmark's SASS dump answers, see `INTRINSICS.md` and `TECHNIQUES.md`.

### A.5 What I would plan for, and what not

- Build family-specific primitives for `sm_100f`, not `sm_100a`, when they need `redux.sync.f32`. The binary then covers B200, GB200, B300 and Rubin. The swarm's mandated `-arch=sm_100` build does not get the `.f32` form at all, so an approach that wants it must document the family-specific build as part of its contract and provide the twiddle-plus-`u32` fallback for `sm_100`.
- Treat 8-bit SIMD min and max and packed `set` as Rubin-only tuning knobs for fp8 keys and for threshold filters. On the hardware of this study they are emulated or absent, and the measured cost on sm_100 (see the `simd __vmaxu4 u8x4` row of the microbenchmark) is what the focus set pays.
- Do not plan around a Rubin shared memory size until the oversized configuration has been measured for occupancy and latency, and do not plan around Rubin cycle counts at all: cycles are not microarchitecture invariant (benchmarking issue, section 6, item 3) and the generation changes the SM.

## B. What is measured

Nothing on Rubin. All numbers of this study come from B200 (sm_100, x86) and GB200 (sm_100, aarch64) nodes, recorded in `state/results/hw-researcher/` and in `TECHNIQUES.md` with arch, CUDA version, node and date. If a Rubin node becomes available to the swarm, the first batch to run is `harness/micro/run_micro.sh`, built additionally with `-arch=sm_107f`, and its output is compared to the sm_100 rows cell by cell, never across architectures inside one card.
