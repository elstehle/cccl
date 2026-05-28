# Resource usage report: `last_filter`

## Build context

| label | branch | sha | subject | target | flags | record count |
|---|---|---|---|---|---|---|
| `flat_walk` | `exp/flat-cta-walk` | `1fc82b9cbf` | topk(experiment): flatten filter agent run() to grid-stride-per-tile | `cub.bench.topk.pairs.base` | `-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t` | 176 |
| `flat_last` | `exp/flat-cta-walk-plus-last` | `d4ca353952` | topk(experiment): flatten last_filter agent run() to grid-stride-pe... | `cub.bench.topk.pairs.base` | `-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t` | 176 |

## `registers` (regs) per (KeyT, ValueT)

| KeyT | ValueT | flat_walk | flat_last | Δ vs flat_walk |
|---|---|---|---|---|
| int | int | 40 | 40 | 0 |
| int | long | 40 | 40 | 0 |
| int | short | 40 | 40 | 0 |
| int | signed char | 40 | 40 | 0 |
| long | int | 40 | 40 | 0 |
| long | long | 40 | 40 | 0 |
| long | short | 40 | 40 | 0 |
| long | signed char | 40 | 40 | 0 |
| short | int | 64 | 40 | -24 |
| short | long | 64 | 40 | -24 |
| short | short | 64 | 40 | -24 |
| short | signed char | 64 | 40 | -24 |
| signed char | int | 64 | 64 | 0 |
| signed char | long | 64 | 64 | 0 |
| signed char | short | 64 | 64 | 0 |
| signed char | signed char | 64 | 64 | 0 |

## `smem_bytes` (smem) per (KeyT, ValueT)

| KeyT | ValueT | flat_walk | flat_last | Δ vs flat_walk |
|---|---|---|---|---|
| int | int | 0 | 0 | 0 |
| int | long | 0 | 0 | 0 |
| int | short | 0 | 0 | 0 |
| int | signed char | 0 | 0 | 0 |
| long | int | 0 | 0 | 0 |
| long | long | 0 | 0 | 0 |
| long | short | 0 | 0 | 0 |
| long | signed char | 0 | 0 | 0 |
| short | int | 0 | 0 | 0 |
| short | long | 0 | 0 | 0 |
| short | short | 0 | 0 | 0 |
| short | signed char | 0 | 0 | 0 |
| signed char | int | 0 | 0 | 0 |
| signed char | long | 0 | 0 | 0 |
| signed char | short | 0 | 0 | 0 |
| signed char | signed char | 0 | 0 | 0 |

## `stack_frame` (stack) per (KeyT, ValueT)

| KeyT | ValueT | flat_walk | flat_last | Δ vs flat_walk |
|---|---|---|---|---|
| int | int | 16 | 0 | -16 |
| int | long | 16 | 0 | -16 |
| int | short | 16 | 0 | -16 |
| int | signed char | 8 | 0 | -8 |
| long | int | 8 | 0 | -8 |
| long | long | 0 | 0 | 0 |
| long | short | 8 | 0 | -8 |
| long | signed char | 8 | 0 | -8 |
| short | int | 0 | 24 | +24 |
| short | long | 0 | 24 | +24 |
| short | short | 0 | 24 | +24 |
| short | signed char | 0 | 16 | +16 |
| signed char | int | 0 | 0 | 0 |
| signed char | long | 0 | 0 | 0 |
| signed char | short | 0 | 0 | 0 |
| signed char | signed char | 0 | 0 | 0 |

## `spill_stores` (sp.st) per (KeyT, ValueT)

| KeyT | ValueT | flat_walk | flat_last | Δ vs flat_walk |
|---|---|---|---|---|
| int | int | 56 | 0 | -56 |
| int | long | 84 | 0 | -84 |
| int | short | 56 | 0 | -56 |
| int | signed char | 8 | 0 | -8 |
| long | int | 8 | 0 | -8 |
| long | long | 0 | 0 | 0 |
| long | short | 8 | 0 | -8 |
| long | signed char | 8 | 0 | -8 |
| short | int | 0 | 68 | +68 |
| short | long | 0 | 68 | +68 |
| short | short | 0 | 68 | +68 |
| short | signed char | 0 | 20 | +20 |
| signed char | int | 0 | 0 | 0 |
| signed char | long | 0 | 0 | 0 |
| signed char | short | 0 | 0 | 0 |
| signed char | signed char | 0 | 0 | 0 |

## `spill_loads` (sp.ld) per (KeyT, ValueT)

| KeyT | ValueT | flat_walk | flat_last | Δ vs flat_walk |
|---|---|---|---|---|
| int | int | 72 | 0 | -72 |
| int | long | 92 | 0 | -92 |
| int | short | 72 | 0 | -72 |
| int | signed char | 24 | 0 | -24 |
| long | int | 8 | 0 | -8 |
| long | long | 0 | 0 | 0 |
| long | short | 8 | 0 | -8 |
| long | signed char | 8 | 0 | -8 |
| short | int | 0 | 60 | +60 |
| short | long | 0 | 60 | +60 |
| short | short | 0 | 60 | +60 |
| short | signed char | 0 | 12 | +12 |
| signed char | int | 0 | 0 | 0 |
| signed char | long | 0 | 0 | 0 |
| signed char | short | 0 | 0 | 0 |
| signed char | signed char | 0 | 0 | 0 |

