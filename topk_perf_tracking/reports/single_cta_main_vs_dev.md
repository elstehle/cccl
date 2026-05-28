# Resource usage report: `single_cta`

## Build context

| label | branch | sha | subject | target | flags | record count |
|---|---|---|---|---|---|---|
| `main` | `main` | `ac941aa0f1` | [libcu++] Add some missing any_resource tests (#7064) | `cub.bench.topk.pairs.base` | `-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t` | 48 |
| `dev` | `tmp/perf-eval-baseline` | `6dae99c414` | Revert "topk(experiment): drop multi_source in filter agent, templa... | `cub.bench.topk.pairs.base` | `-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t` | 176 |

## `registers` (regs) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | ? | 254 | — |
| int | long | ? | 196 | — |
| int | short | ? | 255 | — |
| int | signed char | ? | 255 | — |
| long | int | ? | 183 | — |
| long | long | ? | 186 | — |
| long | short | ? | 181 | — |
| long | signed char | ? | 181 | — |
| short | int | ? | 255 | — |
| short | long | ? | 196 | — |
| short | short | ? | 255 | — |
| short | signed char | ? | 255 | — |
| signed char | int | ? | 255 | — |
| signed char | long | ? | 128 | — |
| signed char | short | ? | 255 | — |
| signed char | signed char | ? | 255 | — |

## `smem_bytes` (smem) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | ? | 33808 | — |
| int | long | ? | 33808 | — |
| int | short | ? | 33808 | — |
| int | signed char | ? | 33808 | — |
| long | int | ? | 33808 | — |
| long | long | ? | 33808 | — |
| long | short | ? | 33808 | — |
| long | signed char | ? | 33808 | — |
| short | int | ? | 33808 | — |
| short | long | ? | 33808 | — |
| short | short | ? | 33808 | — |
| short | signed char | ? | 33808 | — |
| signed char | int | ? | 33808 | — |
| signed char | long | ? | 33808 | — |
| signed char | short | ? | 33808 | — |
| signed char | signed char | ? | 33808 | — |

## `stack_frame` (stack) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | ? | 0 | — |
| int | long | ? | 0 | — |
| int | short | ? | 56 | — |
| int | signed char | ? | 32 | — |
| long | int | ? | 0 | — |
| long | long | ? | 0 | — |
| long | short | ? | 0 | — |
| long | signed char | ? | 0 | — |
| short | int | ? | 128 | — |
| short | long | ? | 0 | — |
| short | short | ? | 904 | — |
| short | signed char | ? | 1424 | — |
| signed char | int | ? | 0 | — |
| signed char | long | ? | 0 | — |
| signed char | short | ? | 672 | — |
| signed char | signed char | ? | 136 | — |

## `spill_stores` (sp.st) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | ? | 0 | — |
| int | long | ? | 0 | — |
| int | short | ? | 52 | — |
| int | signed char | ? | 28 | — |
| long | int | ? | 0 | — |
| long | long | ? | 0 | — |
| long | short | ? | 0 | — |
| long | signed char | ? | 0 | — |
| short | int | ? | 124 | — |
| short | long | ? | 0 | — |
| short | short | ? | 912 | — |
| short | signed char | ? | 1432 | — |
| signed char | int | ? | 0 | — |
| signed char | long | ? | 0 | — |
| signed char | short | ? | 732 | — |
| signed char | signed char | ? | 192 | — |

## `spill_loads` (sp.ld) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | ? | 0 | — |
| int | long | ? | 0 | — |
| int | short | ? | 52 | — |
| int | signed char | ? | 28 | — |
| long | int | ? | 0 | — |
| long | long | ? | 0 | — |
| long | short | ? | 0 | — |
| long | signed char | ? | 0 | — |
| short | int | ? | 124 | — |
| short | long | ? | 0 | — |
| short | short | ? | 1012 | — |
| short | signed char | ? | 1568 | — |
| signed char | int | ? | 0 | — |
| signed char | long | ? | 0 | — |
| signed char | short | ? | 732 | — |
| signed char | signed char | ? | 192 | — |

