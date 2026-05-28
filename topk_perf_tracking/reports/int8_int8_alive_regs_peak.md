# Alive registers at peak liveness

- peak offset: `0x2740`
- nvdisasm reported alive (general regs): **60**
- this script computed alive set size: **40**
- discrepancy comes from approximate operand classification (stores, atomics, branches treat the first reg as USE rather than DEF, so transient scratch may be missed).

## Alive registers (sorted by first-def offset)

| reg | def offset | last use offset | live span | inner source | def instruction |
|---|---|---|---|---|---|
| R30 | 0x40 | 0x8c9b0 | 0x8c970 | `kernel_batched_topk.cuh:639` | `LD.E.64 R30, desc[UR6][R30.64]` |
| R31 | 0x40 | 0x8c9b0 | 0x8c970 | `kernel_batched_topk.cuh:639` | `LD.E.64 R30, desc[UR6][R30.64]` |
| R0 | 0x130 | 0x8c9e0 | 0x8c8b0 | `agent_batched_topk.cuh:1872` | `IMAD.U32 R0, RZ, RZ, UR4` |
| R13 | 0x970 | 0x8c350 | 0x8b9e0 | `agent_batched_topk.cuh:1442` | `SHF.L.U32 R13, R20, R13, RZ` |
| R11 | 0x9b0 | 0x8c930 | 0x8bf80 | `agent_topk_common.cuh:134` | `VIMNMX R11, PT, PT, RZ, R15, !PT` |
| R15 | 0x9f0 | 0x8c940 | 0x8bf50 | `dispatch_topk_common.cuh:78` | `SHF.L.U32 R15, R20, R15, RZ` |
| R17 | 0xa20 | 0x41d60 | 0x41340 | `agent_batched_topk.cuh:1887` | `VIMNMX.U32 R17, PT, PT, R19, R17, PT` |
| R37 | 0x16f0 | 0x8ba60 | 0x8a370 | `agent_batched_topk.cuh:1468` | `LDC.64 R36, c[0x0][0x400]` |
| R36 | 0x1710 | 0x8ba60 | 0x8a350 | `agent_batched_topk.cuh:1468` | `IMAD.WIDE.U32 R36, R16, 0x280, R36` |
| R3 | 0x1760 | 0x4a3d0 | 0x48c70 | `agent_batched_topk.cuh:1454` | `LD.E R3, desc[UR6][R24.64]` |
| R32 | 0x1a80 | 0x8b270 | 0x897f0 | `agent_batched_topk.cuh:1496` | `SEL R32, R26, RZ, P1` |
| R34 | 0x1ae0 | 0x8b4b0 | 0x899d0 | `agent_batched_topk.cuh:1497` | `SEL R34, R24, RZ, P3` |
| R6 | 0x1af0 | 0x86380 | 0x84890 | `agent_batched_topk.cuh:1491` | `IMAD.X R6, RZ, RZ, R9, P1` |
| R12 | 0x1b30 | 0x39550 | 0x37a20 | `ceil_div.h:71` | `VIMNMX.U32 R12, PT, PT, R20, R9, PT` |
| R9 | 0x1b40 | 0x8b430 | 0x898f0 | `agent_batched_topk.cuh:1492` | `SEL R9, R10, RZ, P3` |
| R10 | 0x1b80 | 0x8b440 | 0x898c0 | `agent_batched_topk.cuh:1492` | `SEL R10, R22, RZ, P3` |
| R39 | 0x1bc0 | 0x8caf0 | 0x8af30 | `agent_batched_topk.cuh:1469` | `LDC.64 R38, c[0x0][0x408]` |
| R38 | 0x1c80 | 0x8cae0 | 0x8ae60 | `agent_batched_topk.cuh:1469` | `IMAD.WIDE.U32 R38, R21, 0x4, R38` |
| R21 | 0x1d50 | 0xb1c0 | 0x9470 | `agent_batched_topk.cuh:1946` | `IMAD.IADD R21, R19, 0x1, -R3` |
| R19 | 0x1d90 | 0x26d50 | 0x24fc0 | `agent_batched_topk.cuh:1951` | `IMAD.IADD R19, R20, 0x1, -R21` |
| R20 | 0x2360 | 0x4910 | 0x25b0 | `agent_batched_topk.cuh:1442` | `S2R R20, SR_TID.X` |
| R27 | 0x23c0 | 0x28b0 | 0x4f0 | `block_load.cuh:69` | `LDG.E.U8 R27, desc[UR6][R28.64+0x1]` |
| R22 | 0x23d0 | 0x2b10 | 0x740 | `block_load.cuh:69` | `LDG.E.U8 R22, desc[UR6][R28.64+0x2]` |
| R24 | 0x23e0 | 0x2d70 | 0x990 | `block_load.cuh:69` | `LDG.E.U8 R24, desc[UR6][R28.64+0x3]` |
| R26 | 0x23f0 | 0x2ff0 | 0xc00 | `block_load.cuh:69` | `LDG.E.U8 R26, desc[UR6][R28.64+0x4]` |
| R44 | 0x2400 | 0x3250 | 0xe50 | `block_load.cuh:69` | `LDG.E.U8 R44, desc[UR6][R28.64+0x5]` |
| R46 | 0x2410 | 0x34b0 | 0x10a0 | `block_load.cuh:69` | `LDG.E.U8 R46, desc[UR6][R28.64+0x6]` |
| R48 | 0x2420 | 0x3710 | 0x12f0 | `block_load.cuh:69` | `LDG.E.U8 R48, desc[UR6][R28.64+0x7]` |
| R50 | 0x2430 | 0x3970 | 0x1540 | `block_load.cuh:69` | `LDG.E.U8 R50, desc[UR6][R28.64+0x8]` |
| R52 | 0x2440 | 0x3bd0 | 0x1790 | `block_load.cuh:69` | `LDG.E.U8 R52, desc[UR6][R28.64+0x9]` |
| R54 | 0x2450 | 0x3e30 | 0x19e0 | `block_load.cuh:69` | `LDG.E.U8 R54, desc[UR6][R28.64+0xa]` |
| R56 | 0x2460 | 0x4090 | 0x1c30 | `block_load.cuh:69` | `LDG.E.U8 R56, desc[UR6][R28.64+0xb]` |
| R58 | 0x2470 | 0x42f0 | 0x1e80 | `block_load.cuh:69` | `LDG.E.U8 R58, desc[UR6][R28.64+0xc]` |
| R60 | 0x2480 | 0x4550 | 0x20d0 | `block_load.cuh:69` | `LDG.E.U8 R60, desc[UR6][R28.64+0xd]` |
| R23 | 0x2490 | 0x47b0 | 0x2320 | `block_load.cuh:69` | `LDG.E.U8 R23, desc[UR6][R28.64+0xe]` |
| R25 | 0x24a0 | 0x49e0 | 0x2540 | `block_load.cuh:69` | `LDG.E.U8 R25, desc[UR6][R28.64+0xf]` |
| R43 | 0x25c0 | 0x27d0 | 0x210 | `agent_batched_topk.cuh:1442` | `IMAD.SHL.U32 R43, R40, 0x10, RZ` |
| R47 | 0x2700 | 0x2740 | 0x40 | `dispatch_topk_common.cuh:179` | `LDG.E.U8 R47, desc[UR6][R36.64+0x10]` |
| R28 | 0x2720 | 0x2750 | 0x30 | `dispatch_topk_common.cuh:177` | `LOP3.LUT R28, R13, 0x7f, R27, 0x60, !PT` |
| R29 | 0x2740 | 0x2760 | 0x20 | `dispatch_topk_common.cuh:179` | `LOP3.LUT R29, R47, 0xffff, RZ, 0xc0, !PT` |

## Source line origin of alive registers

| count | file | line |
|---|---|---|
| 15 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/block/block_load.cuh` | 69 |
| 3 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/agent/agent_batched_topk.cuh` | 1442 |
| 2 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/agent/agent_batched_topk.cuh` | 1492 |
| 2 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh` | 179 |
| 2 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/kernels/kernel_batched_topk.cuh` | 639 |
| 2 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/agent/agent_batched_topk.cuh` | 1468 |
| 2 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/agent/agent_batched_topk.cuh` | 1469 |
| 1 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/agent/agent_batched_topk.cuh` | 1872 |
| 1 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/agent/agent_batched_topk.cuh` | 1454 |
| 1 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/agent/agent_batched_topk.cuh` | 1491 |
| 1 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/agent/agent_topk_common.cuh` | 134 |
| 1 | `/cccl_fork/cccl/lib/cmake/libcudacxx/../../../libcudacxx/include/cuda/__cmath/ceil_div.h` | 71 |
| 1 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh` | 78 |
| 1 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/agent/agent_batched_topk.cuh` | 1887 |
| 1 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/agent/agent_batched_topk.cuh` | 1951 |
| 1 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/agent/agent_batched_topk.cuh` | 1946 |
| 1 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh` | 177 |
| 1 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/agent/agent_batched_topk.cuh` | 1496 |
| 1 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/agent/agent_batched_topk.cuh` | 1497 |

## Inline stack for each alive register

### R30  def @ 0x40  `LD.E.64 R30, desc[UR6][R30.64]`
  - `kernel_batched_topk.cuh:639`

### R31  def @ 0x40  `LD.E.64 R30, desc[UR6][R30.64]`
  - `kernel_batched_topk.cuh:639`

### R0  def @ 0x130  `IMAD.U32 R0, RZ, RZ, UR4`
  - `agent_batched_topk.cuh:1872`

### R13  def @ 0x970  `SHF.L.U32 R13, R20, R13, RZ`
  - `agent_batched_topk.cuh:1442`

### R11  def @ 0x9b0  `VIMNMX R11, PT, PT, RZ, R15, !PT`
  - `agent_topk_common.cuh:134`

### R15  def @ 0x9f0  `SHF.L.U32 R15, R20, R15, RZ`
  - `dispatch_topk_common.cuh:78`

### R17  def @ 0xa20  `VIMNMX.U32 R17, PT, PT, R19, R17, PT`
  - `agent_batched_topk.cuh:1887`

### R37  def @ 0x16f0  `LDC.64 R36, c[0x0][0x400]`
  - `agent_batched_topk.cuh:1468`

### R36  def @ 0x1710  `IMAD.WIDE.U32 R36, R16, 0x280, R36`
  - `agent_batched_topk.cuh:1468`

### R3  def @ 0x1760  `LD.E R3, desc[UR6][R24.64]`
  - `agent_batched_topk.cuh:1454`

### R32  def @ 0x1a80  `SEL R32, R26, RZ, P1`
  - `agent_batched_topk.cuh:1496`

### R34  def @ 0x1ae0  `SEL R34, R24, RZ, P3`
  - `agent_batched_topk.cuh:1497`

### R6  def @ 0x1af0  `IMAD.X R6, RZ, RZ, R9, P1`
  - `agent_batched_topk.cuh:1491`

### R12  def @ 0x1b30  `VIMNMX.U32 R12, PT, PT, R20, R9, PT`
  - `ceil_div.h:71`

### R9  def @ 0x1b40  `SEL R9, R10, RZ, P3`
  - `agent_batched_topk.cuh:1492`

### R10  def @ 0x1b80  `SEL R10, R22, RZ, P3`
  - `agent_batched_topk.cuh:1492`

### R39  def @ 0x1bc0  `LDC.64 R38, c[0x0][0x408]`
  - `agent_batched_topk.cuh:1469`

### R38  def @ 0x1c80  `IMAD.WIDE.U32 R38, R21, 0x4, R38`
  - `agent_batched_topk.cuh:1469`

### R21  def @ 0x1d50  `IMAD.IADD R21, R19, 0x1, -R3`
  - `agent_batched_topk.cuh:1946`

### R19  def @ 0x1d90  `IMAD.IADD R19, R20, 0x1, -R21`
  - `agent_batched_topk.cuh:1951`

### R20  def @ 0x2360  `S2R R20, SR_TID.X`
  - `agent_batched_topk.cuh:1442`

### R27  def @ 0x23c0  `LDG.E.U8 R27, desc[UR6][R28.64+0x1]`
  - `block_load.cuh:69`

### R22  def @ 0x23d0  `LDG.E.U8 R22, desc[UR6][R28.64+0x2]`
  - `block_load.cuh:69`

### R24  def @ 0x23e0  `LDG.E.U8 R24, desc[UR6][R28.64+0x3]`
  - `block_load.cuh:69`

### R26  def @ 0x23f0  `LDG.E.U8 R26, desc[UR6][R28.64+0x4]`
  - `block_load.cuh:69`

### R44  def @ 0x2400  `LDG.E.U8 R44, desc[UR6][R28.64+0x5]`
  - `block_load.cuh:69`

### R46  def @ 0x2410  `LDG.E.U8 R46, desc[UR6][R28.64+0x6]`
  - `block_load.cuh:69`

### R48  def @ 0x2420  `LDG.E.U8 R48, desc[UR6][R28.64+0x7]`
  - `block_load.cuh:69`

### R50  def @ 0x2430  `LDG.E.U8 R50, desc[UR6][R28.64+0x8]`
  - `block_load.cuh:69`

### R52  def @ 0x2440  `LDG.E.U8 R52, desc[UR6][R28.64+0x9]`
  - `block_load.cuh:69`

### R54  def @ 0x2450  `LDG.E.U8 R54, desc[UR6][R28.64+0xa]`
  - `block_load.cuh:69`

### R56  def @ 0x2460  `LDG.E.U8 R56, desc[UR6][R28.64+0xb]`
  - `block_load.cuh:69`

### R58  def @ 0x2470  `LDG.E.U8 R58, desc[UR6][R28.64+0xc]`
  - `block_load.cuh:69`

### R60  def @ 0x2480  `LDG.E.U8 R60, desc[UR6][R28.64+0xd]`
  - `block_load.cuh:69`

### R23  def @ 0x2490  `LDG.E.U8 R23, desc[UR6][R28.64+0xe]`
  - `block_load.cuh:69`

### R25  def @ 0x24a0  `LDG.E.U8 R25, desc[UR6][R28.64+0xf]`
  - `block_load.cuh:69`

### R43  def @ 0x25c0  `IMAD.SHL.U32 R43, R40, 0x10, RZ`
  - `agent_batched_topk.cuh:1442`

### R47  def @ 0x2700  `LDG.E.U8 R47, desc[UR6][R36.64+0x10]`
  - `dispatch_topk_common.cuh:179`

### R28  def @ 0x2720  `LOP3.LUT R28, R13, 0x7f, R27, 0x60, !PT`
  - `dispatch_topk_common.cuh:177`

### R29  def @ 0x2740  `LOP3.LUT R29, R47, 0xffff, RZ, 0xc0, !PT`
  - `dispatch_topk_common.cuh:179`

