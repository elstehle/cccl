# Resource report: `main` vs dev (full type matrix)

- main:  `dev_full.json` (253 ptxas records)
- dev:   `dev_with_fixes.json` (253 ptxas records)
- Aggregation: per `(logical_kernel, KeyT, ValueT)` group, take the worst-case
  (max) value across `OffsetT`/`OutOffsetT` variants. Dev's `select::min` and
  `select::max` are byte-identical here, so only `max` is shown.
- ValueT `(K only)` denotes the keys-only kernel (`NullType`).

## Side-by-side: `regs / stack / sp_st / sp_ld / smem (bytes)`

Format `MAIN -> DEV (delta)` when they differ; a bare number means equal.
An asterisk (*) marks a logical kernel that exists only on the dev side
(`finalize_filter`, `finalize_histogram`, `single_cta`).

### `initial_histogram`

| KeyT | ValueT | regs | stack | sp_st | sp_ld | smem (B) |
|---|---|---|---|---|---|---|
| I8 | I8 | 26 | 0 | 0 | 0 | 1072 |
| I8 | I16 | 26 | 0 | 0 | 0 | 1072 |
| I8 | I32 | 26 | 0 | 0 | 0 | 1072 |
| I8 | I64 | 26 | 0 | 0 | 0 | 1072 |
| I8 | (K only) | 26 | 0 | 0 | 0 | 1072 |
| I16 | I8 | 32 | 0 | 0 | 0 | 8240 |
| I16 | I16 | 32 | 0 | 0 | 0 | 8240 |
| I16 | I32 | 32 | 0 | 0 | 0 | 8240 |
| I16 | I64 | 32 | 0 | 0 | 0 | 8240 |
| I16 | (K only) | 32 | 0 | 0 | 0 | 8240 |
| I32 | I8 | 30 | 0 | 0 | 0 | 8240 |
| I32 | I16 | 30 | 0 | 0 | 0 | 8240 |
| I32 | I32 | 30 | 0 | 0 | 0 | 8240 |
| I32 | I64 | 30 | 0 | 0 | 0 | 8240 |
| I32 | (K only) | 30 | 0 | 0 | 0 | 8240 |
| I64 | I8 | 30 | 0 | 0 | 0 | 8240 |
| I64 | I16 | 30 | 0 | 0 | 0 | 8240 |
| I64 | I32 | 30 | 0 | 0 | 0 | 8240 |
| I64 | I64 | 30 | 0 | 0 | 0 | 8240 |
| I64 | (K only) | 30 | 0 | 0 | 0 | 8240 |
| I128 | (K only) | 30 | 0 | 0 | 0 | 1072 |
| F32 | (K only) | 30 | 0 | 0 | 0 | 8240 |
| F64 | (K only) | 32 | 0 | 0 | 0 | 8240 |

### `finalize_histogram`  (* dev-only)

| KeyT | ValueT | regs | stack | sp_st | sp_ld | smem (B) |
|---|---|---|---|---|---|---|
| I8 | I8 | 32 | 0 | 0 | 0 | 2048 |
| I8 | I16 | 32 | 0 | 0 | 0 | 2048 |
| I8 | I32 | 32 | 0 | 0 | 0 | 2048 |
| I8 | I64 | 32 | 0 | 0 | 0 | 2048 |
| I8 | (K only) | 32 | 0 | 0 | 0 | 2048 |
| I16 | I8 | 32 | 0 | 0 | 0 | 8192 |
| I16 | I16 | 32 | 0 | 0 | 0 | 8192 |
| I16 | I32 | 32 | 0 | 0 | 0 | 8192 |
| I16 | I64 | 32 | 0 | 0 | 0 | 8192 |
| I16 | (K only) | 32 | 0 | 0 | 0 | 8192 |
| I32 | I8 | 32 | 0 | 0 | 0 | 8192 |
| I32 | I16 | 32 | 0 | 0 | 0 | 8192 |
| I32 | I32 | 32 | 0 | 0 | 0 | 8192 |
| I32 | I64 | 32 | 0 | 0 | 0 | 8192 |
| I32 | (K only) | 32 | 0 | 0 | 0 | 8192 |
| I64 | I8 | 40 | 0 | 0 | 0 | 8192 |
| I64 | I16 | 40 | 0 | 0 | 0 | 8192 |
| I64 | I32 | 40 | 0 | 0 | 0 | 8192 |
| I64 | I64 | 40 | 0 | 0 | 0 | 8192 |
| I64 | (K only) | 40 | 0 | 0 | 0 | 8192 |
| I128 | (K only) | 32 | 0 | 0 | 0 | 2048 |
| F32 | (K only) | 32 | 0 | 0 | 0 | 8192 |
| F64 | (K only) | 40 | 0 | 0 | 0 | 8192 |

### `filter`

| KeyT | ValueT | regs | stack | sp_st | sp_ld | smem (B) |
|---|---|---|---|---|---|---|
| I8 | I8 | 40 -> 64 (+24) | 24 -> 0 (-24) | 28 -> 0 (-28) | 24 -> 0 (-24) | 1028 |
| I8 | I16 | 40 -> 64 (+24) | 24 -> 0 (-24) | 28 -> 0 (-28) | 24 -> 0 (-24) | 1028 |
| I8 | I32 | 40 -> 64 (+24) | 24 -> 0 (-24) | 28 -> 0 (-28) | 24 -> 0 (-24) | 1028 |
| I8 | I64 | 40 -> 64 (+24) | 32 -> 0 (-32) | 44 -> 0 (-44) | 40 -> 0 (-40) | 1028 |
| I8 | (K only) | 40 | 0 | 0 | 0 | 1028 |
| I16 | I8 | 40 -> 56 (+16) | 0 | 0 | 0 | 8196 |
| I16 | I16 | 40 -> 58 (+18) | 0 | 0 | 0 | 8196 |
| I16 | I32 | 40 -> 58 (+18) | 0 | 0 | 0 | 8196 |
| I16 | I64 | 40 -> 58 (+18) | 0 | 0 | 0 | 8196 |
| I16 | (K only) | 40 -> 32 (-8) | 0 | 0 | 0 | 8196 |
| I32 | I8 | 40 | 0 | 0 | 0 | 8196 |
| I32 | I16 | 40 | 0 | 0 | 0 | 8196 |
| I32 | I32 | 40 | 0 | 0 | 0 | 8196 |
| I32 | I64 | 40 | 0 | 0 | 0 | 8196 |
| I32 | (K only) | 32 | 0 | 0 | 0 | 8196 |
| I64 | I8 | 40 | 0 | 0 | 0 | 8196 |
| I64 | I16 | 40 | 0 | 0 | 0 | 8196 |
| I64 | I32 | 40 | 0 | 0 | 0 | 8196 |
| I64 | I64 | 40 | 0 | 0 | 0 | 8196 |
| I64 | (K only) | 32 | 0 | 0 | 0 | 8196 |
| I128 | (K only) | 40 -> 39 (-1) | 0 | 0 | 0 | 1028 |
| F32 | (K only) | 32 | 0 | 0 | 0 | 8196 |
| F64 | (K only) | 38 -> 36 (-2) | 0 | 0 | 0 | 8196 |

### `finalize_filter`  (* dev-only)

| KeyT | ValueT | regs | stack | sp_st | sp_ld | smem (B) |
|---|---|---|---|---|---|---|
| I8 | I8 | 116 | 0 | 0 | 0 | 2048 |
| I8 | I16 | 120 -> 118 (-2) | 0 | 0 | 0 | 2048 |
| I8 | I32 | 120 -> 118 (-2) | 0 | 0 | 0 | 2048 |
| I8 | I64 | 120 -> 118 (-2) | 0 | 0 | 0 | 2048 |
| I8 | (K only) | 104 -> 106 (+2) | 0 | 0 | 0 | 2048 |
| I16 | I8 | 64 | 0 | 0 | 0 | 8208 |
| I16 | I16 | 64 | 0 | 0 | 0 | 8208 |
| I16 | I32 | 64 | 0 | 0 | 0 | 8208 |
| I16 | I64 | 64 | 0 | 0 | 0 | 8208 |
| I16 | (K only) | 63 -> 64 (+1) | 0 | 0 | 0 | 8208 |
| I32 | I8 | 52 | 0 | 0 | 0 | 8208 |
| I32 | I16 | 52 | 0 | 0 | 0 | 8208 |
| I32 | I32 | 52 | 0 | 0 | 0 | 8208 |
| I32 | I64 | 52 | 0 | 0 | 0 | 8208 |
| I32 | (K only) | 53 -> 48 (-5) | 0 | 0 | 0 | 8208 |
| I64 | I8 | 52 -> 54 (+2) | 0 | 0 | 0 | 8208 |
| I64 | I16 | 52 -> 54 (+2) | 0 | 0 | 0 | 8208 |
| I64 | I32 | 52 -> 54 (+2) | 0 | 0 | 0 | 8208 |
| I64 | I64 | 52 -> 54 (+2) | 0 | 0 | 0 | 8208 |
| I64 | (K only) | 54 -> 50 (-4) | 0 | 0 | 0 | 8208 |
| I128 | (K only) | 52 -> 50 (-2) | 0 | 0 | 0 | 2048 |
| F32 | (K only) | 50 -> 48 (-2) | 0 | 0 | 0 | 8208 |
| F64 | (K only) | 54 -> 50 (-4) | 0 | 0 | 0 | 8208 |

### `last_filter`

| KeyT | ValueT | regs | stack | sp_st | sp_ld | smem (B) |
|---|---|---|---|---|---|---|
| I8 | I8 | 64 | 0 | 0 | 0 | 0 |
| I8 | I16 | 64 | 0 | 0 | 0 | 0 |
| I8 | I32 | 64 | 0 | 0 | 0 | 0 |
| I8 | I64 | 64 | 0 | 0 | 0 | 0 |
| I8 | (K only) | 55 | 0 | 0 | 0 | 0 |
| I16 | I8 | 54 | 0 | 0 | 0 | 0 |
| I16 | I16 | 56 | 0 | 0 | 0 | 0 |
| I16 | I32 | 56 | 0 | 0 | 0 | 0 |
| I16 | I64 | 56 | 0 | 0 | 0 | 0 |
| I16 | (K only) | 55 | 0 | 0 | 0 | 0 |
| I32 | I8 | 40 | 0 | 0 | 0 | 0 |
| I32 | I16 | 40 | 0 | 0 | 0 | 0 |
| I32 | I32 | 40 | 0 | 0 | 0 | 0 |
| I32 | I64 | 40 | 0 -> 8 (+8) | 0 -> 24 (+24) | 0 -> 24 (+24) | 0 |
| I32 | (K only) | 32 | 0 | 0 | 0 | 0 |
| I64 | I8 | 40 | 8 -> 0 (-8) | 8 -> 0 (-8) | 8 -> 0 (-8) | 0 |
| I64 | I16 | 40 | 0 | 0 | 0 | 0 |
| I64 | I32 | 40 | 0 | 0 | 0 | 0 |
| I64 | I64 | 40 | 0 | 0 | 0 | 0 |
| I64 | (K only) | 32 | 0 | 0 | 0 | 0 |
| I128 | (K only) | 32 | 0 | 0 | 0 | 0 |
| F32 | (K only) | 32 | 0 | 0 | 0 | 0 |
| F64 | (K only) | 32 | 0 | 0 | 0 | 0 |

### `single_cta`  (* dev-only)

| KeyT | ValueT | regs | stack | sp_st | sp_ld | smem (B) |
|---|---|---|---|---|---|---|
| I8 | I8 | 255 | 136 | 192 | 192 | 33808 |
| I8 | I16 | 255 | 672 | 732 | 732 | 33808 |
| I8 | I32 | 255 | 0 | 0 | 0 | 33808 |
| I8 | I64 | 128 | 0 | 0 | 0 | 33808 |
| I8 | (K only) | 255 | 0 | 0 | 0 | 33808 |
| I16 | I8 | 255 | 1424 | 1432 | 1568 | 33808 |
| I16 | I16 | 255 | 904 | 912 | 1012 | 33808 |
| I16 | I32 | 255 | 128 | 124 | 124 | 33808 |
| I16 | I64 | 196 | 0 | 0 | 0 | 33808 |
| I16 | (K only) | 255 | 72 | 72 | 72 | 33808 |
| I32 | I8 | 255 | 32 | 28 | 28 | 33808 |
| I32 | I16 | 255 | 56 | 52 | 52 | 33808 |
| I32 | I32 | 254 | 0 | 0 | 0 | 33808 |
| I32 | I64 | 196 | 0 | 0 | 0 | 33808 |
| I32 | (K only) | 246 | 0 | 0 | 0 | 33808 |
| I64 | I8 | 181 | 0 | 0 | 0 | 33808 |
| I64 | I16 | 181 | 0 | 0 | 0 | 33808 |
| I64 | I32 | 183 | 0 | 0 | 0 | 33808 |
| I64 | I64 | 186 | 0 | 0 | 0 | 33808 |
| I64 | (K only) | 128 | 0 | 0 | 0 | 33808 |
| I128 | (K only) | 128 | 0 | 0 | 0 | 33808 |
| F32 | (K only) | 166 | 0 | 0 | 0 | 33808 |
| F64 | (K only) | 128 | 0 | 0 | 0 | 33808 |

## Summary statistics

- `(logical, K, V)` triples present on both sides: **138**
- Dev-only triples (no main counterpart -- `finalize_*`, `single_cta`): **0**
- Main-only triples (should be 0): **0**

### Register delta histogram (dev - main, common kernels)

| delta (regs) | count |
|---:|---:|
| -8 | 1 |
| -5 | 1 |
| -4 | 2 |
| -2 | 6 |
| -1 | 1 |
| 0 | 113 |
| +1 | 1 |
| +2 | 5 |
| +16 | 1 |
| +18 | 3 |
| +24 | 4 |

### Worst register regressions (top 12 dev-over-main)

| logical | KeyT | ValueT | main regs | dev regs | delta |
|---|---|---|---:|---:|---:|
| `filter` | I8 | I8 | 40 | 64 | +24 |
| `filter` | I8 | I16 | 40 | 64 | +24 |
| `filter` | I8 | I32 | 40 | 64 | +24 |
| `filter` | I8 | I64 | 40 | 64 | +24 |
| `filter` | I16 | I16 | 40 | 58 | +18 |
| `filter` | I16 | I32 | 40 | 58 | +18 |
| `filter` | I16 | I64 | 40 | 58 | +18 |
| `filter` | I16 | I8 | 40 | 56 | +16 |
| `finalize_filter` | I8 | (K only) | 104 | 106 | +2 |
| `finalize_filter` | I64 | I8 | 52 | 54 | +2 |
| `finalize_filter` | I64 | I16 | 52 | 54 | +2 |
| `finalize_filter` | I64 | I32 | 52 | 54 | +2 |

### Stack / spill activity

Triples where any of `stack_frame`, `spill_stores`, `spill_loads` is non-zero on either side. Dev-only kernels (`single_cta`, `finalize_*`) are shown when they spill.

| logical | KeyT | ValueT | main stack/sp_st/sp_ld | dev stack/sp_st/sp_ld |
|---|---|---|---|---|
| `filter` | I8 | I8 | 24 / 28 / 24 | 0 / 0 / 0 |
| `filter` | I8 | I16 | 24 / 28 / 24 | 0 / 0 / 0 |
| `filter` | I8 | I32 | 24 / 28 / 24 | 0 / 0 / 0 |
| `filter` | I8 | I64 | 32 / 44 / 40 | 0 / 0 / 0 |
| `last_filter` | I32 | I64 | 0 / 0 / 0 | 8 / 24 / 24 |
| `last_filter` | I64 | I8 | 8 / 8 / 8 | 0 / 0 / 0 |
| `single_cta` | I8 | I8 | 136 / 192 / 192 | 136 / 192 / 192 |
| `single_cta` | I8 | I16 | 672 / 732 / 732 | 672 / 732 / 732 |
| `single_cta` | I16 | I8 | 1424 / 1432 / 1568 | 1424 / 1432 / 1568 |
| `single_cta` | I16 | I16 | 904 / 912 / 1012 | 904 / 912 / 1012 |
| `single_cta` | I16 | I32 | 128 / 124 / 124 | 128 / 124 / 124 |
| `single_cta` | I16 | (K only) | 72 / 72 / 72 | 72 / 72 / 72 |
| `single_cta` | I32 | I8 | 32 / 28 / 28 | 32 / 28 / 28 |
| `single_cta` | I32 | I16 | 56 / 52 / 52 | 56 / 52 / 52 |

### Notable smem differences

(no smem differences across common kernels)
