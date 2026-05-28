# Absolute resource numbers per snapshot

Snapshots: `dev`, `active_source`, `factory_only`.

Format `regs/stack` (worst-case across OffsetT/OutOffsetT variants; `select=max` only -- min/max are byte-identical in dev tracking).

## Pair kernels (`pairs.base`)

### `initial_histogram` -- pairs

| KeyT | ValueT | dev regs/stack | active_source regs/stack | factory_only regs/stack |
|---|---|---:|---:|---:|
| I8 | I8 | 26/0 | 26/0 | 26/0 |
| I8 | I16 | 26/0 | 26/0 | 26/0 |
| I8 | I32 | 26/0 | 26/0 | 26/0 |
| I8 | I64 | 26/0 | 26/0 | 26/0 |
| I16 | I8 | 28/0 | 28/0 | 28/0 |
| I16 | I16 | 28/0 | 28/0 | 28/0 |
| I16 | I32 | 28/0 | 28/0 | 28/0 |
| I16 | I64 | 28/0 | 28/0 | 28/0 |
| I32 | I8 | 30/0 | 30/0 | 30/0 |
| I32 | I16 | 30/0 | 30/0 | 30/0 |
| I32 | I32 | 30/0 | 30/0 | 30/0 |
| I32 | I64 | 30/0 | 30/0 | 30/0 |
| I64 | I8 | 30/0 | 30/0 | 30/0 |
| I64 | I16 | 30/0 | 30/0 | 30/0 |
| I64 | I32 | 30/0 | 30/0 | 30/0 |
| I64 | I64 | 30/0 | 30/0 | 30/0 |

### `finalize_histogram` -- pairs

| KeyT | ValueT | dev regs/stack | active_source regs/stack | factory_only regs/stack |
|---|---|---:|---:|---:|
| I8 | I8 | 32/0 | 32/0 | 32/0 |
| I8 | I16 | 32/0 | 32/0 | 32/0 |
| I8 | I32 | 32/0 | 32/0 | 32/0 |
| I8 | I64 | 32/0 | 32/0 | 32/0 |
| I16 | I8 | 32/0 | 32/0 | 32/0 |
| I16 | I16 | 32/0 | 32/0 | 32/0 |
| I16 | I32 | 32/0 | 32/0 | 32/0 |
| I16 | I64 | 32/0 | 32/0 | 32/0 |
| I32 | I8 | 40/0 | 40/0 | 40/0 |
| I32 | I16 | 40/0 | 40/0 | 40/0 |
| I32 | I32 | 40/0 | 40/0 | 40/0 |
| I32 | I64 | 40/0 | 40/0 | 40/0 |
| I64 | I8 | 40/0 | 40/0 | 40/0 |
| I64 | I16 | 40/0 | 40/0 | 40/0 |
| I64 | I32 | 40/0 | 40/0 | 40/0 |
| I64 | I64 | 40/0 | 40/0 | 40/0 |

### `filter` -- pairs

| KeyT | ValueT | dev regs/stack | active_source regs/stack | factory_only regs/stack |
|---|---|---:|---:|---:|
| I8 | I8 | 64/0 | 63/0 | 64/0 |
| I8 | I16 | 64/0 | 63/0 | 64/0 |
| I8 | I32 | 64/0 | 63/0 | 64/0 |
| I8 | I64 | 64/0 | 63/0 | 64/0 |
| I16 | I8 | 56/0 | 60/0 | 56/0 |
| I16 | I16 | 58/0 | 58/0 | 58/0 |
| I16 | I32 | 58/0 | 58/0 | 58/0 |
| I16 | I64 | 58/0 | 56/0 | 58/0 |
| I32 | I8 | 40/0 | 40/0 | 40/0 |
| I32 | I16 | 40/0 | 40/0 | 40/0 |
| I32 | I32 | 40/0 | 40/0 | 40/0 |
| I32 | I64 | 40/0 | 40/8 | 40/0 |
| I64 | I8 | 40/0 | 40/0 | 40/0 |
| I64 | I16 | 40/0 | 40/0 | 40/0 |
| I64 | I32 | 40/0 | 40/0 | 40/0 |
| I64 | I64 | 40/0 | 40/0 | 40/0 |

### `finalize_filter` -- pairs

| KeyT | ValueT | dev regs/stack | active_source regs/stack | factory_only regs/stack |
|---|---|---:|---:|---:|
| I8 | I8 | 116/0 | 118/0 | 110/0 |
| I8 | I16 | 118/0 | 118/0 | 112/0 |
| I8 | I32 | 118/0 | 118/0 | 112/0 |
| I8 | I64 | 118/0 | 118/0 | 112/0 |
| I16 | I8 | 64/0 | 64/0 | 64/0 |
| I16 | I16 | 64/0 | 64/0 | 64/0 |
| I16 | I32 | 64/0 | 64/0 | 64/0 |
| I16 | I64 | 64/0 | 64/0 | 64/0 |
| I32 | I8 | 52/0 | 58/0 | 50/0 |
| I32 | I16 | 52/0 | 55/0 | 48/0 |
| I32 | I32 | 52/0 | 55/0 | 48/0 |
| I32 | I64 | 52/0 | 56/0 | 48/0 |
| I64 | I8 | 54/0 | 58/0 | 54/0 |
| I64 | I16 | 54/0 | 55/0 | 54/0 |
| I64 | I32 | 54/0 | 55/0 | 54/0 |
| I64 | I64 | 54/0 | 55/0 | 54/0 |

### `last_filter` -- pairs

| KeyT | ValueT | dev regs/stack | active_source regs/stack | factory_only regs/stack |
|---|---|---:|---:|---:|
| I8 | I8 | 64/0 | 64/0 | 64/0 |
| I8 | I16 | 64/0 | 64/0 | 64/0 |
| I8 | I32 | 64/0 | 64/0 | 64/0 |
| I8 | I64 | 64/0 | 64/0 | 64/0 |
| I16 | I8 | 54/0 | 60/0 | 54/0 |
| I16 | I16 | 56/0 | 60/0 | 56/0 |
| I16 | I32 | 56/0 | 60/0 | 56/0 |
| I16 | I64 | 56/0 | 62/0 | 56/0 |
| I32 | I8 | 40/0 | 40/0 | 40/0 |
| I32 | I16 | 40/0 | 40/0 | 40/0 |
| I32 | I32 | 40/0 | 40/0 | 40/0 |
| I32 | I64 | 40/0 | 40/8 | 40/0 |
| I64 | I8 | 40/0 | 52/0 | 40/0 |
| I64 | I16 | 40/0 | 50/0 | 40/0 |
| I64 | I32 | 40/0 | 50/0 | 40/0 |
| I64 | I64 | 40/0 | 50/0 | 40/0 |

### `single_cta` -- pairs

| KeyT | ValueT | dev regs/stack | active_source regs/stack | factory_only regs/stack |
|---|---|---:|---:|---:|
| I8 | I8 | 255/136 | 255/136 | 255/136 |
| I8 | I16 | 255/672 | 255/672 | 255/672 |
| I8 | I32 | 255/0 | 255/0 | 255/0 |
| I8 | I64 | 128/0 | 128/0 | 128/0 |
| I16 | I8 | 255/1424 | 255/1424 | 255/1424 |
| I16 | I16 | 255/904 | 255/904 | 255/904 |
| I16 | I32 | 255/128 | 255/128 | 255/128 |
| I16 | I64 | 196/0 | 196/0 | 196/0 |
| I32 | I8 | 255/32 | 255/32 | 255/32 |
| I32 | I16 | 255/56 | 255/56 | 255/56 |
| I32 | I32 | 254/0 | 254/0 | 254/0 |
| I32 | I64 | 196/0 | 196/0 | 196/0 |
| I64 | I8 | 183/0 | 183/0 | 183/0 |
| I64 | I16 | 183/0 | 183/0 | 183/0 |
| I64 | I32 | 183/0 | 183/0 | 183/0 |
| I64 | I64 | 156/0 | 156/0 | 156/0 |

## Keys-only kernels (`keys.base`)

### `initial_histogram` -- keys-only

| KeyT | ValueT | dev regs/stack | active_source regs/stack | factory_only regs/stack |
|---|---|---:|---:|---:|
| I8 | (K only) | 26/0 | 26/0 | 26/0 |
| I16 | (K only) | 28/0 | 28/0 | 28/0 |
| I32 | (K only) | 30/0 | 30/0 | 30/0 |
| I64 | (K only) | 30/0 | 30/0 | 30/0 |
| I128 | (K only) | 30/0 | 30/0 | 30/0 |
| F32 | (K only) | 29/0 | 29/0 | 29/0 |
| F64 | (K only) | 32/0 | 32/0 | 32/0 |

### `finalize_histogram` -- keys-only

| KeyT | ValueT | dev regs/stack | active_source regs/stack | factory_only regs/stack |
|---|---|---:|---:|---:|
| I8 | (K only) | 32/0 | 32/0 | 32/0 |
| I16 | (K only) | 32/0 | 32/0 | 32/0 |
| I32 | (K only) | 40/0 | 40/0 | 40/0 |
| I64 | (K only) | 40/0 | 40/0 | 40/0 |
| I128 | (K only) | 32/0 | 32/0 | 32/0 |
| F32 | (K only) | 40/0 | 40/0 | 40/0 |
| F64 | (K only) | 40/0 | 40/0 | 40/0 |

### `filter` -- keys-only

| KeyT | ValueT | dev regs/stack | active_source regs/stack | factory_only regs/stack |
|---|---|---:|---:|---:|
| I8 | (K only) | 40/0 | 40/0 | 40/0 |
| I16 | (K only) | 32/0 | 40/0 | 40/0 |
| I32 | (K only) | 32/0 | 32/0 | 32/0 |
| I64 | (K only) | 32/0 | 32/0 | 32/0 |
| I128 | (K only) | 39/0 | 39/0 | 39/0 |
| F32 | (K only) | 32/0 | 32/0 | 32/0 |
| F64 | (K only) | 36/0 | 36/0 | 36/0 |

### `finalize_filter` -- keys-only

| KeyT | ValueT | dev regs/stack | active_source regs/stack | factory_only regs/stack |
|---|---|---:|---:|---:|
| I8 | (K only) | 106/0 | 108/0 | 116/0 |
| I16 | (K only) | 64/0 | 64/0 | 64/0 |
| I32 | (K only) | 48/0 | 48/0 | 48/0 |
| I64 | (K only) | 50/0 | 52/0 | 50/0 |
| I128 | (K only) | 50/0 | 50/0 | 50/0 |
| F32 | (K only) | 48/0 | 48/0 | 48/0 |
| F64 | (K only) | 50/0 | 52/0 | 50/0 |

### `last_filter` -- keys-only

| KeyT | ValueT | dev regs/stack | active_source regs/stack | factory_only regs/stack |
|---|---|---:|---:|---:|
| I8 | (K only) | 60/0 | 58/0 | 60/0 |
| I16 | (K only) | 55/0 | 55/0 | 54/0 |
| I32 | (K only) | 32/0 | 32/0 | 31/0 |
| I64 | (K only) | 32/0 | 32/0 | 31/0 |
| I128 | (K only) | 32/0 | 40/0 | 32/0 |
| F32 | (K only) | 32/0 | 32/0 | 31/0 |
| F64 | (K only) | 32/0 | 32/0 | 32/0 |

### `single_cta` -- keys-only

| KeyT | ValueT | dev regs/stack | active_source regs/stack | factory_only regs/stack |
|---|---|---:|---:|---:|
| I8 | (K only) | 255/0 | 255/0 | 255/0 |
| I16 | (K only) | 255/72 | 255/72 | 255/72 |
| I32 | (K only) | 246/0 | 246/0 | 246/0 |
| I64 | (K only) | 128/0 | 128/0 | 128/0 |
| I128 | (K only) | 128/0 | 128/0 | 128/0 |
| F32 | (K only) | 166/0 | 166/0 | 166/0 |
| F64 | (K only) | 128/0 | 128/0 | 128/0 |

