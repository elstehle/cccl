# Resource usage summary: `main` (baseline) -> `dev` (target)

## Build context

| label | branch | sha | subject |
|---|---|---|---|
| `main` | `main` | `ac941aa0f1` | [libcu++] Add some missing any_resource tests (#7064) |
| `dev` | `tmp/perf-eval-baseline` | `6dae99c414` | Revert "topk(experiment): drop multi_source in filter agent, template PickB a... |

## Per-kernel deltas (target - baseline, summed across all (KeyT, ValueT))

| kernel | registers | smem_bytes | stack_frame | spill_stores | spill_loads | instances (base/target) |
|---|---|---|---|---|---|---|
| `filter` | +165 | -106688 | +208 | +536 | +904 | 16/16 |
| `last_filter` | +322 | 0 | +80 | +228 | +284 | 16/16 |
| `initial_histogram` | -28 | -105984 | 0 | 0 | 0 | 16/16 |
| `finalize_filter` | 0 | 0 | 0 | 0 | 0 | 0/16 |
| `finalize_histogram` | 0 | 0 | 0 | 0 | 0 | 0/16 |
| `single_cta` | 0 | 0 | 0 | 0 | 0 | 0/16 |

## Per-kernel max-impact (KeyT, ValueT) configurations

For each (logical_name, metric), the (KeyT, ValueT) combination with the largest signed delta.

| kernel | metric | KeyT | ValueT | baseline | target | Δ |
|---|---|---|---|---|---|---|
| `filter` | `registers` | signed char | signed char | 32 | 64 | +32 |
| `filter` | `smem_bytes` | int | int | 16400 | 8196 | -8204 |
| `filter` | `stack_frame` | short | int | 0 | 32 | +32 |
| `filter` | `spill_stores` | int | long | 0 | 140 | +140 |
| `filter` | `spill_loads` | int | long | 0 | 192 | +192 |
| `last_filter` | `registers` | short | int | 32 | 64 | +32 |
| `last_filter` | `stack_frame` | int | int | 0 | 16 | +16 |
| `last_filter` | `spill_stores` | int | long | 0 | 84 | +84 |
| `last_filter` | `spill_loads` | int | long | 0 | 92 | +92 |
| `initial_histogram` | `registers` | signed char | int | 32 | 26 | -6 |
| `initial_histogram` | `smem_bytes` | int | int | 16400 | 8240 | -8160 |

