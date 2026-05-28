# Resource usage report: `finalize_filter`

## Build context

| label | branch | sha | subject | target | flags | record count |
|---|---|---|---|---|---|---|
| `main` | `main` | `ac941aa0f1` | [libcu++] Add some missing any_resource tests (#7064) | `cub.bench.topk.pairs.base` | `-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t` | 48 |
| `dev` | `tmp/perf-eval-baseline` | `6dae99c414` | Revert "topk(experiment): drop multi_source in filter agent, templa... | `cub.bench.topk.pairs.base` | `-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t` | 176 |

## `registers` (regs) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | ? | 52 | — |
| int | long | ? | 52 | — |
| int | short | ? | 52 | — |
| int | signed char | ? | 52 | — |
| long | int | ? | 52 | — |
| long | long | ? | 52 | — |
| long | short | ? | 52 | — |
| long | signed char | ? | 52 | — |
| short | int | ? | 64 | — |
| short | long | ? | 64 | — |
| short | short | ? | 64 | — |
| short | signed char | ? | 64 | — |
| signed char | int | ? | 120 | — |
| signed char | long | ? | 120 | — |
| signed char | short | ? | 120 | — |
| signed char | signed char | ? | 116 | — |

## `smem_bytes` (smem) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | ? | 8208 | — |
| int | long | ? | 8208 | — |
| int | short | ? | 8208 | — |
| int | signed char | ? | 8208 | — |
| long | int | ? | 8208 | — |
| long | long | ? | 8208 | — |
| long | short | ? | 8208 | — |
| long | signed char | ? | 8208 | — |
| short | int | ? | 8208 | — |
| short | long | ? | 8208 | — |
| short | short | ? | 8208 | — |
| short | signed char | ? | 8208 | — |
| signed char | int | ? | 2048 | — |
| signed char | long | ? | 2048 | — |
| signed char | short | ? | 2048 | — |
| signed char | signed char | ? | 2048 | — |

## `stack_frame` (stack) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | ? | 0 | — |
| int | long | ? | 0 | — |
| int | short | ? | 0 | — |
| int | signed char | ? | 0 | — |
| long | int | ? | 0 | — |
| long | long | ? | 0 | — |
| long | short | ? | 0 | — |
| long | signed char | ? | 0 | — |
| short | int | ? | 0 | — |
| short | long | ? | 0 | — |
| short | short | ? | 0 | — |
| short | signed char | ? | 0 | — |
| signed char | int | ? | 0 | — |
| signed char | long | ? | 0 | — |
| signed char | short | ? | 0 | — |
| signed char | signed char | ? | 0 | — |

## `spill_stores` (sp.st) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | ? | 0 | — |
| int | long | ? | 0 | — |
| int | short | ? | 0 | — |
| int | signed char | ? | 0 | — |
| long | int | ? | 0 | — |
| long | long | ? | 0 | — |
| long | short | ? | 0 | — |
| long | signed char | ? | 0 | — |
| short | int | ? | 0 | — |
| short | long | ? | 0 | — |
| short | short | ? | 0 | — |
| short | signed char | ? | 0 | — |
| signed char | int | ? | 0 | — |
| signed char | long | ? | 0 | — |
| signed char | short | ? | 0 | — |
| signed char | signed char | ? | 0 | — |

## `spill_loads` (sp.ld) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | ? | 0 | — |
| int | long | ? | 0 | — |
| int | short | ? | 0 | — |
| int | signed char | ? | 0 | — |
| long | int | ? | 0 | — |
| long | long | ? | 0 | — |
| long | short | ? | 0 | — |
| long | signed char | ? | 0 | — |
| short | int | ? | 0 | — |
| short | long | ? | 0 | — |
| short | short | ? | 0 | — |
| short | signed char | ? | 0 | — |
| signed char | int | ? | 0 | — |
| signed char | long | ? | 0 | — |
| signed char | short | ? | 0 | — |
| signed char | signed char | ? | 0 | — |

