# SASS register-liveness analysis

- Total instructions analyzed: **2701**
- Peak live-register count: **38**
- Instructions at peak: **206**
- Distinct source lines at peak: **19**

## Top 10 source lines contributing to peak liveness

| count | file | line |
|---|---|---|
| 36 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh` | 177 |
| 26 | `/usr/local/cuda/targets/x86_64-linux/include/device_atomic_functions.hpp` | 112 |
| 21 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh` | 43 |
| 18 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh` | 179 |
| 18 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/detail/topk/block_partition.cuh` | 674 |
| 15 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/detail/topk/tile_data_source.cuh` | 277 |
| 12 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/detail/topk/block_filter.cuh` | 350 |
| 9 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/agent/agent_batched_topk.cuh` | 1442 |
| 8 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh` | 113 |
| 8 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/detail/topk/block_partition.cuh` | 694 |

## SASS context around each peak instruction (±3)

### peak @ 0x1b80  live_regs=38 `/usr/local/cuda/targets/x86_64-linux/include/device_atomic_functions.hpp:112`

```
    0x  1b50  R= 37  P=3  UR=14  UP=0   ISETP.EQ.U32.AND P1, PT, R22, R27, PT
    0x  1b60  R= 36  P=3  UR=14  UP=0   @P1 ATOMG.E.ADD.STRONG.GPU PT, R23, desc[UR16][R20.64+0x180], R23
    0x  1b70  R= 37  P=2  UR=14  UP=0   IMAD R28, R2, 0x200, R37
  * 0x  1b80  R= 38  P=2  UR=14  UP=0   S2R R27, SR_LTMASK
    0x  1b90  R= 38  P=2  UR=14  UP=0   IMAD.SHL.U32 R28, R28, 0x10, RZ
    0x  1ba0  R= 38  P=2  UR=14  UP=0   SHFL.IDX PT, R23, R23, R22, 0x1f
    0x  1bb0  R= 38  P=2  UR=14  UP=0   LOP3.LUT R22, R27, UR13, RZ, 0xc0, !PT
```

### peak @ 0x1b90  live_regs=38 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/agent/agent_batched_topk.cuh:1442`

```
    0x  1b60  R= 36  P=3  UR=14  UP=0   @P1 ATOMG.E.ADD.STRONG.GPU PT, R23, desc[UR16][R20.64+0x180], R23
    0x  1b70  R= 37  P=2  UR=14  UP=0   IMAD R28, R2, 0x200, R37
    0x  1b80  R= 38  P=2  UR=14  UP=0   S2R R27, SR_LTMASK
  * 0x  1b90  R= 38  P=2  UR=14  UP=0   IMAD.SHL.U32 R28, R28, 0x10, RZ
    0x  1ba0  R= 38  P=2  UR=14  UP=0   SHFL.IDX PT, R23, R23, R22, 0x1f
    0x  1bb0  R= 38  P=2  UR=14  UP=0   LOP3.LUT R22, R27, UR13, RZ, 0xc0, !PT
    0x  1bc0  R= 37  P=2  UR=14  UP=0   LDCU UR13, c[0x0][0x3a8]
```

### peak @ 0x1ba0  live_regs=38 `/usr/local/cuda/targets/x86_64-linux/include/device_atomic_functions.hpp:112`

```
    0x  1b70  R= 37  P=2  UR=14  UP=0   IMAD R28, R2, 0x200, R37
    0x  1b80  R= 38  P=2  UR=14  UP=0   S2R R27, SR_LTMASK
    0x  1b90  R= 38  P=2  UR=14  UP=0   IMAD.SHL.U32 R28, R28, 0x10, RZ
  * 0x  1ba0  R= 38  P=2  UR=14  UP=0   SHFL.IDX PT, R23, R23, R22, 0x1f
    0x  1bb0  R= 38  P=2  UR=14  UP=0   LOP3.LUT R22, R27, UR13, RZ, 0xc0, !PT
    0x  1bc0  R= 37  P=2  UR=14  UP=0   LDCU UR13, c[0x0][0x3a8]
    0x  1bd0  R= 38  P=2  UR=14  UP=0   POPC R27, R22
```

### peak @ 0x1bb0  live_regs=38 `/usr/local/cuda/targets/x86_64-linux/include/device_atomic_functions.hpp:112`

```
    0x  1b80  R= 38  P=2  UR=14  UP=0   S2R R27, SR_LTMASK
    0x  1b90  R= 38  P=2  UR=14  UP=0   IMAD.SHL.U32 R28, R28, 0x10, RZ
    0x  1ba0  R= 38  P=2  UR=14  UP=0   SHFL.IDX PT, R23, R23, R22, 0x1f
  * 0x  1bb0  R= 38  P=2  UR=14  UP=0   LOP3.LUT R22, R27, UR13, RZ, 0xc0, !PT
    0x  1bc0  R= 37  P=2  UR=14  UP=0   LDCU UR13, c[0x0][0x3a8]
    0x  1bd0  R= 38  P=2  UR=14  UP=0   POPC R27, R22
    0x  1be0  R= 37  P=2  UR=14  UP=0   IMAD.IADD R27, R23, 0x1, R27
```

### peak @ 0x1bd0  live_regs=38 `/usr/local/cuda/targets/x86_64-linux/include/device_atomic_functions.hpp:112`

```
    0x  1ba0  R= 38  P=2  UR=14  UP=0   SHFL.IDX PT, R23, R23, R22, 0x1f
    0x  1bb0  R= 38  P=2  UR=14  UP=0   LOP3.LUT R22, R27, UR13, RZ, 0xc0, !PT
    0x  1bc0  R= 37  P=2  UR=14  UP=0   LDCU UR13, c[0x0][0x3a8]
  * 0x  1bd0  R= 38  P=2  UR=14  UP=0   POPC R27, R22
    0x  1be0  R= 37  P=2  UR=14  UP=0   IMAD.IADD R27, R23, 0x1, R27
    0x  1bf0  R= 37  P=3  UR=14  UP=0   IADD3 R22, P1, PT, R27, UR14, RZ
    0x  1c00  R= 38  P=3  UR=13  UP=0   IMAD.X R23, RZ, RZ, UR15, P1
```

### peak @ 0x1c00  live_regs=38 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/detail/topk/block_filter.cuh:350`

```
    0x  1bd0  R= 38  P=2  UR=14  UP=0   POPC R27, R22
    0x  1be0  R= 37  P=2  UR=14  UP=0   IMAD.IADD R27, R23, 0x1, R27
    0x  1bf0  R= 37  P=3  UR=14  UP=0   IADD3 R22, P1, PT, R27, UR14, RZ
  * 0x  1c00  R= 38  P=3  UR=13  UP=0   IMAD.X R23, RZ, RZ, UR15, P1
    0x  1c10  R= 38  P=2  UR=14  UP=0   LDCU.64 UR14, c[0x0][0x3c0]
    0x  1c20  R= 38  P=2  UR=14  UP=0   STG.E.U8 desc[UR16][R22.64], R26
    0x  1c30  R= 38  P=2  UR=14  UP=0   VIADD R26, R28.reuse, UR13
```

### peak @ 0x1c10  live_regs=38 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh:43`

```
    0x  1be0  R= 37  P=2  UR=14  UP=0   IMAD.IADD R27, R23, 0x1, R27
    0x  1bf0  R= 37  P=3  UR=14  UP=0   IADD3 R22, P1, PT, R27, UR14, RZ
    0x  1c00  R= 38  P=3  UR=13  UP=0   IMAD.X R23, RZ, RZ, UR15, P1
  * 0x  1c10  R= 38  P=2  UR=14  UP=0   LDCU.64 UR14, c[0x0][0x3c0]
    0x  1c20  R= 38  P=2  UR=14  UP=0   STG.E.U8 desc[UR16][R22.64], R26
    0x  1c30  R= 38  P=2  UR=14  UP=0   VIADD R26, R28.reuse, UR13
    0x  1c40  R= 38  P=2  UR=13  UP=0   @P0 IMAD.WIDE.U32 R22, R28, 0x4, R16
```

### peak @ 0x1c20  live_regs=38 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/detail/topk/block_filter.cuh:350`

```
    0x  1bf0  R= 37  P=3  UR=14  UP=0   IADD3 R22, P1, PT, R27, UR14, RZ
    0x  1c00  R= 38  P=3  UR=13  UP=0   IMAD.X R23, RZ, RZ, UR15, P1
    0x  1c10  R= 38  P=2  UR=14  UP=0   LDCU.64 UR14, c[0x0][0x3c0]
  * 0x  1c20  R= 38  P=2  UR=14  UP=0   STG.E.U8 desc[UR16][R22.64], R26
    0x  1c30  R= 38  P=2  UR=14  UP=0   VIADD R26, R28.reuse, UR13
    0x  1c40  R= 38  P=2  UR=13  UP=0   @P0 IMAD.WIDE.U32 R22, R28, 0x4, R16
    0x  1c50  R= 38  P=2  UR=13  UP=0   @P0 LD.E R26, desc[UR16][R22.64]
```

### peak @ 0x1c30  live_regs=38 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/agent/agent_batched_topk.cuh:1442`

```
    0x  1c00  R= 38  P=3  UR=13  UP=0   IMAD.X R23, RZ, RZ, UR15, P1
    0x  1c10  R= 38  P=2  UR=14  UP=0   LDCU.64 UR14, c[0x0][0x3c0]
    0x  1c20  R= 38  P=2  UR=14  UP=0   STG.E.U8 desc[UR16][R22.64], R26
  * 0x  1c30  R= 38  P=2  UR=14  UP=0   VIADD R26, R28.reuse, UR13
    0x  1c40  R= 38  P=2  UR=13  UP=0   @P0 IMAD.WIDE.U32 R22, R28, 0x4, R16
    0x  1c50  R= 38  P=2  UR=13  UP=0   @P0 LD.E R26, desc[UR16][R22.64]
    0x  1c60  R= 37  P=3  UR=13  UP=0   IADD3 R22, P1, PT, R26, UR14, RZ
```

### peak @ 0x1c40  live_regs=38 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/detail/topk/tile_data_source.cuh:277`

```
    0x  1c10  R= 38  P=2  UR=14  UP=0   LDCU.64 UR14, c[0x0][0x3c0]
    0x  1c20  R= 38  P=2  UR=14  UP=0   STG.E.U8 desc[UR16][R22.64], R26
    0x  1c30  R= 38  P=2  UR=14  UP=0   VIADD R26, R28.reuse, UR13
  * 0x  1c40  R= 38  P=2  UR=13  UP=0   @P0 IMAD.WIDE.U32 R22, R28, 0x4, R16
    0x  1c50  R= 38  P=2  UR=13  UP=0   @P0 LD.E R26, desc[UR16][R22.64]
    0x  1c60  R= 37  P=3  UR=13  UP=0   IADD3 R22, P1, PT, R26, UR14, RZ
    0x  1c70  R= 37  P=3  UR=12  UP=0   IMAD.X R23, RZ, RZ, UR15, P1
```

