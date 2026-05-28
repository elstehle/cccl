# Resource usage report: `filter`

## Build context

| label | branch | sha | subject | target | flags | record count |
|---|---|---|---|---|---|---|
| `main` | `main` | `ac941aa0f1` | [libcu++] Add some missing any_resource tests (#7064) | `cub.bench.topk.pairs.base` | `-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t` | 48 |
| `dev` | `tmp/perf-eval-baseline` | `6dae99c414` | Revert "topk(experiment): drop multi_source in filter agent, templa... | `cub.bench.topk.pairs.base` | `-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t` | 176 |

## `registers` (regs) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | 32 | 32 | 0 |
| int | long | 32 | 32 | 0 |
| int | short | 32 | 32 | 0 |
| int | signed char | 32 | 32 | 0 |
| long | int | 32 | 40 | +8 |
| long | long | 32 | 40 | +8 |
| long | short | 32 | 40 | +8 |
| long | signed char | 32 | 40 | +8 |
| short | int | 32 | 40 | +8 |
| short | long | 32 | 40 | +8 |
| short | short | 32 | 40 | +8 |
| short | signed char | 32 | 40 | +8 |
| signed char | int | 40 | 63 | +23 |
| signed char | long | 40 | 63 | +23 |
| signed char | short | 40 | 63 | +23 |
| signed char | signed char | 32 | 64 | +32 |

## `smem_bytes` (smem) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | 16400 | 8196 | -8204 |
| int | long | 16400 | 8196 | -8204 |
| int | short | 16400 | 8196 | -8204 |
| int | signed char | 16400 | 8196 | -8204 |
| long | int | 16400 | 8196 | -8204 |
| long | long | 16400 | 8196 | -8204 |
| long | short | 16400 | 8196 | -8204 |
| long | signed char | 16400 | 8196 | -8204 |
| short | int | 16400 | 8196 | -8204 |
| short | long | 16400 | 8196 | -8204 |
| short | short | 16400 | 8196 | -8204 |
| short | signed char | 16400 | 8196 | -8204 |
| signed char | int | 3088 | 1028 | -2060 |
| signed char | long | 3088 | 1028 | -2060 |
| signed char | short | 3088 | 1028 | -2060 |
| signed char | signed char | 3088 | 1028 | -2060 |

## `stack_frame` (stack) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | 0 | 24 | +24 |
| int | long | 0 | 24 | +24 |
| int | short | 0 | 24 | +24 |
| int | signed char | 0 | 16 | +16 |
| long | int | 0 | 0 | 0 |
| long | long | 0 | 0 | 0 |
| long | short | 0 | 0 | 0 |
| long | signed char | 0 | 0 | 0 |
| short | int | 0 | 32 | +32 |
| short | long | 0 | 32 | +32 |
| short | short | 0 | 32 | +32 |
| short | signed char | 0 | 24 | +24 |
| signed char | int | 0 | 0 | 0 |
| signed char | long | 0 | 0 | 0 |
| signed char | short | 0 | 0 | 0 |
| signed char | signed char | 0 | 0 | 0 |

## `spill_stores` (sp.st) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | 0 | 36 | +36 |
| int | long | 0 | 140 | +140 |
| int | short | 0 | 36 | +36 |
| int | signed char | 0 | 28 | +28 |
| long | int | 0 | 0 | 0 |
| long | long | 0 | 0 | 0 |
| long | short | 0 | 0 | 0 |
| long | signed char | 0 | 0 | 0 |
| short | int | 0 | 80 | +80 |
| short | long | 0 | 80 | +80 |
| short | short | 0 | 80 | +80 |
| short | signed char | 0 | 56 | +56 |
| signed char | int | 0 | 0 | 0 |
| signed char | long | 0 | 0 | 0 |
| signed char | short | 0 | 0 | 0 |
| signed char | signed char | 0 | 0 | 0 |

## `spill_loads` (sp.ld) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | 0 | 88 | +88 |
| int | long | 0 | 192 | +192 |
| int | short | 0 | 88 | +88 |
| int | signed char | 0 | 80 | +80 |
| long | int | 0 | 0 | 0 |
| long | long | 0 | 0 | 0 |
| long | short | 0 | 0 | 0 |
| long | signed char | 0 | 0 | 0 |
| short | int | 0 | 120 | +120 |
| short | long | 0 | 120 | +120 |
| short | short | 0 | 120 | +120 |
| short | signed char | 0 | 96 | +96 |
| signed char | int | 0 | 0 | 0 |
| signed char | long | 0 | 0 | 0 |
| signed char | short | 0 | 0 | 0 |
| signed char | signed char | 0 | 0 | 0 |

