# Resource usage report: `filter`

## Build context

| label | branch | sha | subject | target | flags | record count |
|---|---|---|---|---|---|---|
| `dev` | `tmp/perf-eval-baseline` | `6dae99c414` | Revert "topk(experiment): drop multi_source in filter agent, templa... | `cub.bench.topk.pairs.base` | `-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t` | 176 |
| `flat_walk` | `exp/flat-cta-walk` | `1fc82b9cbf` | topk(experiment): flatten filter agent run() to grid-stride-per-tile | `cub.bench.topk.pairs.base` | `-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t` | 176 |

## `registers` (regs) per (KeyT, ValueT)

| KeyT | ValueT | dev | flat_walk | Δ vs dev |
|---|---|---|---|---|
| int | int | 32 | 40 | +8 |
| int | long | 32 | 40 | +8 |
| int | short | 32 | 40 | +8 |
| int | signed char | 32 | 40 | +8 |
| long | int | 40 | 40 | 0 |
| long | long | 40 | 40 | 0 |
| long | short | 40 | 40 | 0 |
| long | signed char | 40 | 40 | 0 |
| short | int | 40 | 40 | 0 |
| short | long | 40 | 40 | 0 |
| short | short | 40 | 40 | 0 |
| short | signed char | 40 | 40 | 0 |
| signed char | int | 63 | 40 | -23 |
| signed char | long | 63 | 40 | -23 |
| signed char | short | 63 | 40 | -23 |
| signed char | signed char | 64 | 40 | -24 |

## `smem_bytes` (smem) per (KeyT, ValueT)

| KeyT | ValueT | dev | flat_walk | Δ vs dev |
|---|---|---|---|---|
| int | int | 8196 | 8196 | 0 |
| int | long | 8196 | 8196 | 0 |
| int | short | 8196 | 8196 | 0 |
| int | signed char | 8196 | 8196 | 0 |
| long | int | 8196 | 8196 | 0 |
| long | long | 8196 | 8196 | 0 |
| long | short | 8196 | 8196 | 0 |
| long | signed char | 8196 | 8196 | 0 |
| short | int | 8196 | 8196 | 0 |
| short | long | 8196 | 8196 | 0 |
| short | short | 8196 | 8196 | 0 |
| short | signed char | 8196 | 8196 | 0 |
| signed char | int | 1028 | 1028 | 0 |
| signed char | long | 1028 | 1028 | 0 |
| signed char | short | 1028 | 1028 | 0 |
| signed char | signed char | 1028 | 1028 | 0 |

## `stack_frame` (stack) per (KeyT, ValueT)

| KeyT | ValueT | dev | flat_walk | Δ vs dev |
|---|---|---|---|---|
| int | int | 24 | 0 | -24 |
| int | long | 24 | 0 | -24 |
| int | short | 24 | 0 | -24 |
| int | signed char | 16 | 0 | -16 |
| long | int | 0 | 0 | 0 |
| long | long | 0 | 0 | 0 |
| long | short | 0 | 0 | 0 |
| long | signed char | 0 | 0 | 0 |
| short | int | 32 | 0 | -32 |
| short | long | 32 | 0 | -32 |
| short | short | 32 | 0 | -32 |
| short | signed char | 24 | 0 | -24 |
| signed char | int | 0 | 24 | +24 |
| signed char | long | 0 | 32 | +32 |
| signed char | short | 0 | 24 | +24 |
| signed char | signed char | 0 | 24 | +24 |

## `spill_stores` (sp.st) per (KeyT, ValueT)

| KeyT | ValueT | dev | flat_walk | Δ vs dev |
|---|---|---|---|---|
| int | int | 36 | 0 | -36 |
| int | long | 140 | 0 | -140 |
| int | short | 36 | 0 | -36 |
| int | signed char | 28 | 0 | -28 |
| long | int | 0 | 0 | 0 |
| long | long | 0 | 0 | 0 |
| long | short | 0 | 0 | 0 |
| long | signed char | 0 | 0 | 0 |
| short | int | 80 | 0 | -80 |
| short | long | 80 | 0 | -80 |
| short | short | 80 | 0 | -80 |
| short | signed char | 56 | 0 | -56 |
| signed char | int | 0 | 28 | +28 |
| signed char | long | 0 | 44 | +44 |
| signed char | short | 0 | 28 | +28 |
| signed char | signed char | 0 | 28 | +28 |

## `spill_loads` (sp.ld) per (KeyT, ValueT)

| KeyT | ValueT | dev | flat_walk | Δ vs dev |
|---|---|---|---|---|
| int | int | 88 | 0 | -88 |
| int | long | 192 | 0 | -192 |
| int | short | 88 | 0 | -88 |
| int | signed char | 80 | 0 | -80 |
| long | int | 0 | 0 | 0 |
| long | long | 0 | 0 | 0 |
| long | short | 0 | 0 | 0 |
| long | signed char | 0 | 0 | 0 |
| short | int | 120 | 0 | -120 |
| short | long | 120 | 0 | -120 |
| short | short | 120 | 0 | -120 |
| short | signed char | 96 | 0 | -96 |
| signed char | int | 0 | 20 | +20 |
| signed char | long | 0 | 36 | +36 |
| signed char | short | 0 | 20 | +20 |
| signed char | signed char | 0 | 20 | +20 |

