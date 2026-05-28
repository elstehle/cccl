# Resource usage report: `finalize_histogram`

## Build context

| label | branch | sha | subject | target | flags | record count |
|---|---|---|---|---|---|---|
| `main` | `main` | `ac941aa0f1` | [libcu++] Add some missing any_resource tests (#7064) | `cub.bench.topk.pairs.base` | `-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t` | 48 |
| `dev` | `tmp/perf-eval-baseline` | `6dae99c414` | Revert "topk(experiment): drop multi_source in filter agent, templa... | `cub.bench.topk.pairs.base` | `-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t` | 176 |

## `registers` (regs) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | ? | 32 | — |
| int | long | ? | 32 | — |
| int | short | ? | 32 | — |
| int | signed char | ? | 32 | — |
| long | int | ? | 40 | — |
| long | long | ? | 40 | — |
| long | short | ? | 40 | — |
| long | signed char | ? | 40 | — |
| short | int | ? | 32 | — |
| short | long | ? | 32 | — |
| short | short | ? | 32 | — |
| short | signed char | ? | 32 | — |
| signed char | int | ? | 32 | — |
| signed char | long | ? | 32 | — |
| signed char | short | ? | 32 | — |
| signed char | signed char | ? | 32 | — |

## `smem_bytes` (smem) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | ? | 8192 | — |
| int | long | ? | 8192 | — |
| int | short | ? | 8192 | — |
| int | signed char | ? | 8192 | — |
| long | int | ? | 8192 | — |
| long | long | ? | 8192 | — |
| long | short | ? | 8192 | — |
| long | signed char | ? | 8192 | — |
| short | int | ? | 8192 | — |
| short | long | ? | 8192 | — |
| short | short | ? | 8192 | — |
| short | signed char | ? | 8192 | — |
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

