# SASS register-liveness analysis

- Total instructions analyzed: **36018**
- Peak live-register count: **60**
- Instructions at peak: **296**
- Distinct source lines at peak: **17**

## Top 15 source lines contributing to peak liveness

| count | file | line |
|---|---|---|
| 91 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh` | 179 |
| 61 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh` | 177 |
| 18 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh` | 180 |
| 16 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/thread/thread_load.cuh` | 326 |
| 14 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/detail/topk/block_filter.cuh` | 350 |
| 13 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh` | 43 |
| 12 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/block/block_load.cuh` | 69 |
| 12 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/detail/topk/block_partition.cuh` | 694 |
| 12 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/detail/topk/block_partition.cuh` | 697 |
| 12 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/detail/topk/block_partition.cuh` | 674 |
| 8 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/block/block_load.cuh` | 226 |
| 6 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/detail/topk/tile_data_source.cuh` | 277 |
| 6 | `/cccl_fork/cccl/lib/cmake/libcudacxx/../../../libcudacxx/include/cuda/__iterator/transform_output_iterator.h` | 261 |
| 6 | `/cccl_fork/cccl/lib/cmake/libcudacxx/../../../libcudacxx/include/cuda/__iterator/transform_output_iterator.h` | 93 |
| 4 | `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh` | 113 |

## SASS context around each peak instruction (±3)

### peak @ 0x2740  live_regs=60 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh:179`

```
    0x  2710  R= 58  P=2  UR=2  UP=0   BSYNC.RECONVERGENT B5
    0x  2720  R= 59  P=2  UR=2  UP=0   LOP3.LUT R28, R13, 0x7f, R27, 0x60, !PT
    0x  2730  R= 59  P=2  UR=2  UP=0   BSSY.RECONVERGENT B5, `(.L_x_42)
  * 0x  2740  R= 60  P=2  UR=2  UP=0   LOP3.LUT R29, R47, 0xffff, RZ, 0xc0, !PT
    0x  2750  R= 60  P=2  UR=2  UP=0   LOP3.LUT R28, R28, 0xff, RZ, 0xc0, !PT
    0x  2760  R= 60  P=3  UR=2  UP=0   ISETP.GT.U32.AND P2, PT, R28, R29, PT
    0x  2770  R= 59  P=3  UR=2  UP=0   @P2 BRA `(.L_x_43)
```

### peak @ 0x2750  live_regs=60 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh:177`

```
    0x  2720  R= 59  P=2  UR=2  UP=0   LOP3.LUT R28, R13, 0x7f, R27, 0x60, !PT
    0x  2730  R= 59  P=2  UR=2  UP=0   BSSY.RECONVERGENT B5, `(.L_x_42)
    0x  2740  R= 60  P=2  UR=2  UP=0   LOP3.LUT R29, R47, 0xffff, RZ, 0xc0, !PT
  * 0x  2750  R= 60  P=2  UR=2  UP=0   LOP3.LUT R28, R28, 0xff, RZ, 0xc0, !PT
    0x  2760  R= 60  P=3  UR=2  UP=0   ISETP.GT.U32.AND P2, PT, R28, R29, PT
    0x  2770  R= 59  P=3  UR=2  UP=0   @P2 BRA `(.L_x_43)
    0x  2780  R= 54  P=2  UR=2  UP=0   S2R R29, SR_LANEID
```

### peak @ 0x2760  live_regs=60 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh:179`

```
    0x  2730  R= 59  P=2  UR=2  UP=0   BSSY.RECONVERGENT B5, `(.L_x_42)
    0x  2740  R= 60  P=2  UR=2  UP=0   LOP3.LUT R29, R47, 0xffff, RZ, 0xc0, !PT
    0x  2750  R= 60  P=2  UR=2  UP=0   LOP3.LUT R28, R28, 0xff, RZ, 0xc0, !PT
  * 0x  2760  R= 60  P=3  UR=2  UP=0   ISETP.GT.U32.AND P2, PT, R28, R29, PT
    0x  2770  R= 59  P=3  UR=2  UP=0   @P2 BRA `(.L_x_43)
    0x  2780  R= 54  P=2  UR=2  UP=0   S2R R29, SR_LANEID
    0x  2790  R= 54  P=2  UR=3  UP=0   VOTEU.ANY UR4, UPT, PT
```

### peak @ 0x2890  live_regs=60 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/detail/topk/block_filter.cuh:350`

```
    0x  2860  R= 59  P=2  UR=4  UP=0   IMAD.IADD R47, R43, 0x1, R28
    0x  2870  R= 58  P=2  UR=4  UP=0   @!P1 LDC R43, c[0x0][0x3a8]
    0x  2880  R= 59  P=3  UR=4  UP=0   IADD3 R28, P2, PT, R47, UR4, RZ
  * 0x  2890  R= 60  P=3  UR=3  UP=0   IMAD.X R29, RZ, RZ, UR5, P2
    0x  28a0  R= 60  P=2  UR=4  UP=0   LDCU.64 UR4, c[0x0][0x3c0]
    0x  28b0  R= 60  P=2  UR=4  UP=0   STG.E.U8 desc[UR6][R28.64], R27
    0x  28c0  R= 57  P=2  UR=4  UP=0   @P1 LD.E R42, desc[UR6][R40.64+0x4]
```

### peak @ 0x28a0  live_regs=60 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh:43`

```
    0x  2870  R= 58  P=2  UR=4  UP=0   @!P1 LDC R43, c[0x0][0x3a8]
    0x  2880  R= 59  P=3  UR=4  UP=0   IADD3 R28, P2, PT, R47, UR4, RZ
    0x  2890  R= 60  P=3  UR=3  UP=0   IMAD.X R29, RZ, RZ, UR5, P2
  * 0x  28a0  R= 60  P=2  UR=4  UP=0   LDCU.64 UR4, c[0x0][0x3c0]
    0x  28b0  R= 60  P=2  UR=4  UP=0   STG.E.U8 desc[UR6][R28.64], R27
    0x  28c0  R= 57  P=2  UR=4  UP=0   @P1 LD.E R42, desc[UR6][R40.64+0x4]
    0x  28d0  R= 57  P=2  UR=4  UP=0   @!P1 IMAD.U32 R43, R49, 0x10, R43
```

### peak @ 0x28b0  live_regs=60 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/detail/topk/block_filter.cuh:350`

```
    0x  2880  R= 59  P=3  UR=4  UP=0   IADD3 R28, P2, PT, R47, UR4, RZ
    0x  2890  R= 60  P=3  UR=3  UP=0   IMAD.X R29, RZ, RZ, UR5, P2
    0x  28a0  R= 60  P=2  UR=4  UP=0   LDCU.64 UR4, c[0x0][0x3c0]
  * 0x  28b0  R= 60  P=2  UR=4  UP=0   STG.E.U8 desc[UR6][R28.64], R27
    0x  28c0  R= 57  P=2  UR=4  UP=0   @P1 LD.E R42, desc[UR6][R40.64+0x4]
    0x  28d0  R= 57  P=2  UR=4  UP=0   @!P1 IMAD.U32 R43, R49, 0x10, R43
    0x  28e0  R= 57  P=2  UR=4  UP=0   @!P1 VIADD R42, R43, 0x1
```

### peak @ 0x29a0  live_regs=60 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh:179`

```
    0x  2970  R= 58  P=2  UR=2  UP=0   BSYNC.RECONVERGENT B5
    0x  2980  R= 59  P=2  UR=2  UP=0   LOP3.LUT R27, R13, 0x7f, R22, 0x60, !PT
    0x  2990  R= 59  P=2  UR=2  UP=0   BSSY.RECONVERGENT B5, `(.L_x_44)
  * 0x  29a0  R= 60  P=2  UR=2  UP=0   LOP3.LUT R28, R47, 0xffff, RZ, 0xc0, !PT
    0x  29b0  R= 60  P=2  UR=2  UP=0   LOP3.LUT R27, R27, 0xff, RZ, 0xc0, !PT
    0x  29c0  R= 60  P=3  UR=2  UP=0   ISETP.GT.U32.AND P2, PT, R27, R28, PT
    0x  29d0  R= 59  P=3  UR=2  UP=0   @P2 BRA `(.L_x_45)
```

### peak @ 0x29b0  live_regs=60 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh:177`

```
    0x  2980  R= 59  P=2  UR=2  UP=0   LOP3.LUT R27, R13, 0x7f, R22, 0x60, !PT
    0x  2990  R= 59  P=2  UR=2  UP=0   BSSY.RECONVERGENT B5, `(.L_x_44)
    0x  29a0  R= 60  P=2  UR=2  UP=0   LOP3.LUT R28, R47, 0xffff, RZ, 0xc0, !PT
  * 0x  29b0  R= 60  P=2  UR=2  UP=0   LOP3.LUT R27, R27, 0xff, RZ, 0xc0, !PT
    0x  29c0  R= 60  P=3  UR=2  UP=0   ISETP.GT.U32.AND P2, PT, R27, R28, PT
    0x  29d0  R= 59  P=3  UR=2  UP=0   @P2 BRA `(.L_x_45)
    0x  29e0  R= 54  P=2  UR=2  UP=0   S2R R29, SR_LANEID
```

### peak @ 0x29c0  live_regs=60 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh:179`

```
    0x  2990  R= 59  P=2  UR=2  UP=0   BSSY.RECONVERGENT B5, `(.L_x_44)
    0x  29a0  R= 60  P=2  UR=2  UP=0   LOP3.LUT R28, R47, 0xffff, RZ, 0xc0, !PT
    0x  29b0  R= 60  P=2  UR=2  UP=0   LOP3.LUT R27, R27, 0xff, RZ, 0xc0, !PT
  * 0x  29c0  R= 60  P=3  UR=2  UP=0   ISETP.GT.U32.AND P2, PT, R27, R28, PT
    0x  29d0  R= 59  P=3  UR=2  UP=0   @P2 BRA `(.L_x_45)
    0x  29e0  R= 54  P=2  UR=2  UP=0   S2R R29, SR_LANEID
    0x  29f0  R= 54  P=2  UR=3  UP=0   VOTEU.ANY UR4, UPT, PT
```

### peak @ 0x2c00  live_regs=60 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh:179`

```
    0x  2bd0  R= 58  P=2  UR=2  UP=0   BSYNC.RECONVERGENT B5
    0x  2be0  R= 59  P=2  UR=2  UP=0   LOP3.LUT R22, R13, 0x7f, R24, 0x60, !PT
    0x  2bf0  R= 59  P=2  UR=2  UP=0   BSSY.RECONVERGENT B5, `(.L_x_46)
  * 0x  2c00  R= 60  P=2  UR=2  UP=0   LOP3.LUT R27, R47, 0xffff, RZ, 0xc0, !PT
    0x  2c10  R= 60  P=2  UR=2  UP=0   LOP3.LUT R22, R22, 0xff, RZ, 0xc0, !PT
    0x  2c20  R= 60  P=3  UR=2  UP=0   ISETP.GT.U32.AND P2, PT, R22, R27, PT
    0x  2c30  R= 58  P=3  UR=2  UP=0   @P2 BRA `(.L_x_47)
```

### peak @ 0x2c10  live_regs=60 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh:177`

```
    0x  2be0  R= 59  P=2  UR=2  UP=0   LOP3.LUT R22, R13, 0x7f, R24, 0x60, !PT
    0x  2bf0  R= 59  P=2  UR=2  UP=0   BSSY.RECONVERGENT B5, `(.L_x_46)
    0x  2c00  R= 60  P=2  UR=2  UP=0   LOP3.LUT R27, R47, 0xffff, RZ, 0xc0, !PT
  * 0x  2c10  R= 60  P=2  UR=2  UP=0   LOP3.LUT R22, R22, 0xff, RZ, 0xc0, !PT
    0x  2c20  R= 60  P=3  UR=2  UP=0   ISETP.GT.U32.AND P2, PT, R22, R27, PT
    0x  2c30  R= 58  P=3  UR=2  UP=0   @P2 BRA `(.L_x_47)
    0x  2c40  R= 54  P=2  UR=2  UP=0   S2R R29, SR_LANEID
```

### peak @ 0x2c20  live_regs=60 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/device/dispatch/dispatch_topk_common.cuh:179`

```
    0x  2bf0  R= 59  P=2  UR=2  UP=0   BSSY.RECONVERGENT B5, `(.L_x_46)
    0x  2c00  R= 60  P=2  UR=2  UP=0   LOP3.LUT R27, R47, 0xffff, RZ, 0xc0, !PT
    0x  2c10  R= 60  P=2  UR=2  UP=0   LOP3.LUT R22, R22, 0xff, RZ, 0xc0, !PT
  * 0x  2c20  R= 60  P=3  UR=2  UP=0   ISETP.GT.U32.AND P2, PT, R22, R27, PT
    0x  2c30  R= 58  P=3  UR=2  UP=0   @P2 BRA `(.L_x_47)
    0x  2c40  R= 54  P=2  UR=2  UP=0   S2R R29, SR_LANEID
    0x  2c50  R= 54  P=2  UR=3  UP=0   VOTEU.ANY UR4, UPT, PT
```

### peak @ 0x4c90  live_regs=60 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/thread/thread_load.cuh:326`

```
    0x  4c60  R= 57  P=2  UR=2  UP=0   PRMT R60, R40.reuse, 0x7770, RZ
    0x  4c70  R= 58  P=2  UR=2  UP=0   PRMT R23, R40.reuse, 0x7771, RZ
    0x  4c80  R= 59  P=2  UR=2  UP=0   PRMT R25, R40.reuse, 0x7772, RZ
  * 0x  4c90  R= 60  P=2  UR=2  UP=0   PRMT R27, R40, 0x7773, RZ
    0x  4ca0  R= 60  P=2  UR=2  UP=0   PRMT R24, R28, 0x7610, R24
    0x  4cb0  R= 59  P=2  UR=2  UP=0   PRMT R26, R29, 0x7610, R26
    0x  4cc0  R= 58  P=2  UR=2  UP=0   BRA `(.L_x_74)
```

### peak @ 0x4ca0  live_regs=60 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/block/block_load.cuh:226`

```
    0x  4c70  R= 58  P=2  UR=2  UP=0   PRMT R23, R40.reuse, 0x7771, RZ
    0x  4c80  R= 59  P=2  UR=2  UP=0   PRMT R25, R40.reuse, 0x7772, RZ
    0x  4c90  R= 60  P=2  UR=2  UP=0   PRMT R27, R40, 0x7773, RZ
  * 0x  4ca0  R= 60  P=2  UR=2  UP=0   PRMT R24, R28, 0x7610, R24
    0x  4cb0  R= 59  P=2  UR=2  UP=0   PRMT R26, R29, 0x7610, R26
    0x  4cc0  R= 58  P=2  UR=2  UP=0   BRA `(.L_x_74)
    0x  4cd0  R= 42  P=2  UR=2  UP=0   S2R R20, SR_TID.X
```

### peak @ 0x4e00  live_regs=60 `/cccl_fork/cccl/lib/cmake/cub/../../../cub/cub/block/block_load.cuh:69`

```
    0x  4dd0  R= 57  P=2  UR=2  UP=0   LD.E.U8 R60, desc[UR6][R28.64+0xc]
    0x  4de0  R= 58  P=2  UR=2  UP=0   LD.E.U8 R23, desc[UR6][R28.64+0xd]
    0x  4df0  R= 59  P=2  UR=2  UP=0   LD.E.U8 R25, desc[UR6][R28.64+0xe]
  * 0x  4e00  R= 60  P=2  UR=2  UP=0   LD.E.U8 R27, desc[UR6][R28.64+0xf]
    0x  4e10  R= 58  P=2  UR=2  UP=0   BRA `(.L_x_74)
    0x  4e20  R= 41  P=2  UR=3  UP=0   LDCU UR4, c[0x0][0x388]
    0x  4e30  R= 41  P=2  UR=3  UP=1   ULOP3.LUT UP0, URZ, UR4, 0x3, URZ, 0xc0, !UPT
```

