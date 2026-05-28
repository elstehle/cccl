# Resource usage report: `last_filter`

## Build context

| label | branch | sha | subject | target | flags | record count |
|---|---|---|---|---|---|---|
| `main` | `main` | `ac941aa0f1` | [libcu++] Add some missing any_resource tests (#7064) | `cub.bench.topk.pairs.base` | `-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t` | 48 |
| `dev` | `tmp/perf-eval-baseline` | `6dae99c414` | Revert "topk(experiment): drop multi_source in filter agent, templa... | `cub.bench.topk.pairs.base` | `-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t` | 176 |

## `registers` (regs) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | 32 | 40 | +8 |
| int | long | 32 | 40 | +8 |
| int | short | 32 | 40 | +8 |
| int | signed char | 32 | 40 | +8 |
| long | int | 32 | 40 | +8 |
| long | long | 32 | 40 | +8 |
| long | short | 32 | 40 | +8 |
| long | signed char | 30 | 40 | +10 |
| short | int | 32 | 64 | +32 |
| short | long | 32 | 64 | +32 |
| short | short | 32 | 64 | +32 |
| short | signed char | 32 | 64 | +32 |
| signed char | int | 32 | 64 | +32 |
| signed char | long | 32 | 64 | +32 |
| signed char | short | 32 | 64 | +32 |
| signed char | signed char | 32 | 64 | +32 |

## `smem_bytes` (smem) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
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

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | 0 | 16 | +16 |
| int | long | 0 | 16 | +16 |
| int | short | 0 | 16 | +16 |
| int | signed char | 0 | 8 | +8 |
| long | int | 0 | 8 | +8 |
| long | long | 0 | 0 | 0 |
| long | short | 0 | 8 | +8 |
| long | signed char | 0 | 8 | +8 |
| short | int | 0 | 0 | 0 |
| short | long | 0 | 0 | 0 |
| short | short | 0 | 0 | 0 |
| short | signed char | 0 | 0 | 0 |
| signed char | int | 0 | 0 | 0 |
| signed char | long | 0 | 0 | 0 |
| signed char | short | 0 | 0 | 0 |
| signed char | signed char | 0 | 0 | 0 |

## `spill_stores` (sp.st) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | 0 | 56 | +56 |
| int | long | 0 | 84 | +84 |
| int | short | 0 | 56 | +56 |
| int | signed char | 0 | 8 | +8 |
| long | int | 0 | 8 | +8 |
| long | long | 0 | 0 | 0 |
| long | short | 0 | 8 | +8 |
| long | signed char | 0 | 8 | +8 |
| short | int | 0 | 0 | 0 |
| short | long | 0 | 0 | 0 |
| short | short | 0 | 0 | 0 |
| short | signed char | 0 | 0 | 0 |
| signed char | int | 0 | 0 | 0 |
| signed char | long | 0 | 0 | 0 |
| signed char | short | 0 | 0 | 0 |
| signed char | signed char | 0 | 0 | 0 |

## `spill_loads` (sp.ld) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | 0 | 72 | +72 |
| int | long | 0 | 92 | +92 |
| int | short | 0 | 72 | +72 |
| int | signed char | 0 | 24 | +24 |
| long | int | 0 | 8 | +8 |
| long | long | 0 | 0 | 0 |
| long | short | 0 | 8 | +8 |
| long | signed char | 0 | 8 | +8 |
| short | int | 0 | 0 | 0 |
| short | long | 0 | 0 | 0 |
| short | short | 0 | 0 | 0 |
| short | signed char | 0 | 0 | 0 |
| signed char | int | 0 | 0 | 0 |
| signed char | long | 0 | 0 | 0 |
| signed char | short | 0 | 0 | 0 |
| signed char | signed char | 0 | 0 | 0 |

