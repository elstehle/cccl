# Resource usage report: `initial_histogram`

## Build context

| label | branch | sha | subject | target | flags | record count |
|---|---|---|---|---|---|---|
| `main` | `main` | `ac941aa0f1` | [libcu++] Add some missing any_resource tests (#7064) | `cub.bench.topk.pairs.base` | `-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t` | 48 |
| `dev` | `tmp/perf-eval-baseline` | `6dae99c414` | Revert "topk(experiment): drop multi_source in filter agent, templa... | `cub.bench.topk.pairs.base` | `-Xptxas=-v -DTUNE_OffsetT=::cuda::std::int32_t` | 176 |

## `registers` (regs) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | 31 | 30 | -1 |
| int | long | 31 | 30 | -1 |
| int | short | 31 | 30 | -1 |
| int | signed char | 31 | 30 | -1 |
| long | int | 31 | 30 | -1 |
| long | long | 31 | 30 | -1 |
| long | short | 31 | 30 | -1 |
| long | signed char | 31 | 30 | -1 |
| short | int | 31 | 32 | +1 |
| short | long | 31 | 32 | +1 |
| short | short | 31 | 32 | +1 |
| short | signed char | 31 | 32 | +1 |
| signed char | int | 32 | 26 | -6 |
| signed char | long | 32 | 26 | -6 |
| signed char | short | 32 | 26 | -6 |
| signed char | signed char | 32 | 26 | -6 |

## `smem_bytes` (smem) per (KeyT, ValueT)

| KeyT | ValueT | main | dev | Δ vs main |
|---|---|---|---|---|
| int | int | 16400 | 8240 | -8160 |
| int | long | 16400 | 8240 | -8160 |
| int | short | 16400 | 8240 | -8160 |
| int | signed char | 16400 | 8240 | -8160 |
| long | int | 16400 | 8240 | -8160 |
| long | long | 16400 | 8240 | -8160 |
| long | short | 16400 | 8240 | -8160 |
| long | signed char | 16400 | 8240 | -8160 |
| short | int | 16400 | 8240 | -8160 |
| short | long | 16400 | 8240 | -8160 |
| short | short | 16400 | 8240 | -8160 |
| short | signed char | 16400 | 8240 | -8160 |
| signed char | int | 3088 | 1072 | -2016 |
| signed char | long | 3088 | 1072 | -2016 |
| signed char | short | 3088 | 1072 | -2016 |
| signed char | signed char | 3088 | 1072 | -2016 |

## `stack_frame` (stack) per (KeyT, ValueT)

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

## `spill_stores` (sp.st) per (KeyT, ValueT)

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

## `spill_loads` (sp.ld) per (KeyT, ValueT)

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

