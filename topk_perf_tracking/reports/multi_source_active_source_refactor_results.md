# `multi_source_data_source` refactor: results

Following the design proposal in `proposal_multi_source_active_source_refactor.md`. Implemented + measured two variants on `umb-b200-261` (CTK 13.1.115, B200, sm_100).

## Branches

| branch | tip sha | shape |
|---|---|---|
| `exp/topk-batched-large-segments-regressions` | `e77b23ca23` | dev baseline |
| `exp/topk-multi-source-active-source` | `f90e824e98` | **full refactor**: placement-new'd active source + tagged-union handles + factory-callback ctor + non-copyable |
| `exp/topk-multi-source-factory-only` | `dc13e0a249` | **factory-only variant**: factory-callback ctor + non-copyable, but both children kept alive (no placement-new) |

Agent migration applies to both variants identically (`agent_topk.cuh`,
`agent_batched_topk.cuh`) -- six call sites + the one destroy-then-construct
at the segment boundary in `agent_batched_topk_last_filter::run`.

## Headline: full refactor regresses, factory-only is mostly neutral

### Pairs benchmark (per-(KeyT, ValueT) geo mean ratio vs dev)

| (KeyT, ValueT)  | full refactor | factory-only |
|---|---:|---:|
| `(I64, I32)`    | **1.012x**    | 1.003x |
| `(I64, I16)`    | **1.012x**    | 1.003x |
| `(I64, I8)`     | **1.012x**    | 1.003x |
| `(I64, I64)`    | **1.012x**    | 1.002x |
| `(I32, I8)`     | 1.003x        | 1.002x |
| `(I8,  I32)`    | 1.002x        | 1.003x |
| `(I16, I8)`     | 1.001x        | **1.010x** |
| `(I16, I32)`    | 1.001x        | 0.998x |
| `(I16, I64)`    | 1.001x        | 1.003x |
| `(I8,  I8)`     | 1.000x        | 0.999x |
| `(I16, I16)`    | 1.000x        | 0.999x |
| `(I32, I16)`    | 1.000x        | 1.000x |
| `(I32, I64)`    | 1.000x        | 1.001x |
| `(I32, I32)`    | 0.998x        | 1.000x |
| `(I8,  I64)`    | 0.998x        | 0.999x |
| `(I8,  I16)`    | 0.995x        | 0.995x |

Worst per-cell entry:
- Full refactor: I64 / any V, Elements=2^28, Entropy=0.000 -> **1.065x** (832us -> 1045us baseline equivalent).
- Factory-only:  I16 / I8 worst 1.052x; I16 / others <= 1.021x.

### Keys benchmark (per-KeyT keys-only)

| KeyT     | full refactor | factory-only |
|---|---:|---:|
| `I16`    | 1.006x        | **1.008x** |
| `I64`    | 1.005x        | 1.004x |
| `I128`   | 1.004x        | 1.004x |
| `F32`    | 1.003x        | 0.999x |
| `I32`    | 1.003x        | 1.004x |
| `F64`    | 1.002x        | 1.003x |
| `I8`     | 0.999x        | 1.002x |

Worst per-cell entry:
- Full refactor: I16 / F64 keys-only -> **1.052x**.
- Factory-only:  I16 keys-only Elements=2^28 Entropy=1.000 -> **1.054x**.

## Resource impact (registers / stack / spill / smem)

### Full refactor (`active_source`) — register delta histogram

| delta (regs) | count |
|---:|---:|
| -2 | 2 |
| -1 | 4 |
| 0 | 109 |
| +1 | 3 |
| +2 | 4 |
| +3 | 2 |
| +4 | 5 |
| +6 | 3 |
| +8 | 2 |
| +10 | 3 |
| +12 | 1 |

Worst regressions (full refactor):

| logical | KeyT | ValueT | dev regs | refactor regs | delta |
|---|---|---|---:|---:|---:|
| `last_filter` | I64 | I8 | 40 | **52** | +12 |
| `last_filter` | I64 | I16 | 40 | 50 | +10 |
| `last_filter` | I64 | I32 | 40 | 50 | +10 |
| `last_filter` | I64 | I64 | 40 | 50 | +10 |
| `filter` | I16 | (K only) | 32 | 40 | +8 |
| `last_filter` | I128 | (K only) | 32 | 40 | +8 |
| `finalize_filter` | I32 | I8 | 52 | 58 | +6 |
| `last_filter` | I16 | I8 | 54 | 60 | +6 |
| `last_filter` | I16 | I64 | 56 | 62 | +6 |

Plus new 8 B stack frame / 8 B spills on `filter` and `last_filter` for `I32 / I64`.

### Factory-only variant — register delta histogram

| delta (regs) | count |
|---:|---:|
| -6 | 4 |
| -4 | 3 |
| -2 | 1 |
| -1 | 4 |
| 0 | 124 |
| +8 | 1 |
| +10 | 1 |

Remaining regressions (factory-only):

| logical | KeyT | ValueT | dev regs | refactor regs | delta |
|---|---|---|---:|---:|---:|
| `finalize_filter` | I8 | (K only) | 106 | 116 | +10 |
| `filter` | I16 | (K only) | 32 | 40 | +8 |

Plus improvements on `finalize_filter` for I8 / * (pairs): -6 regs (118 -> 112), and `finalize_filter` for I32 / * (pairs): -2 to -4 regs.

## SASS investigation (full refactor)

The `last_filter` kernel for I64 / I32 (select::min) is the regression locus.
Side-by-side cuobjdump --dump-sass, then per-opcode counts:

| opcode             | dev | full refactor | delta |
|---|---:|---:|---:|
| `BAR.SYNC`         | 0  | 0  | 0     |
| `VOTEU.ANY`        | 14 | 14 | 0     |
| `ATOM*`            | 12 | 12 | 0     |
| `LDCU`             | **19** | **9** | **-10** |
| `LDG`              | 17 | 12 | -5    |
| `STG`              | 24 | 24 | 0     |
| `BSYNC.RECONVERGENT` | 0 | **11** | **+11** |
| `BSSY.RECONVERGENT`  | 0 | **11** | **+11** |
| `IMAD.MOV.U32`     | 14 | **37** | +23   |
| `SEL`              | 12 | **19** | +7    |
| `IADD3`            | 15 | 19 | +4    |
| Total lines        | 1492 | 1572 | +80 |

### Root cause

The dev code emits a UNIFORM-register hoist for the `pick_source_b ?
candidate_buf_offset : input_offset` resolution:

```sass
@!P3 BRA target ;                          // explicit branch
... path-A ...
BRA merge ;
target:
  LDCU UR4, c[0x0][0x3a8] ;                // candidate buffer offset
  IMAD R9, R5, 0x400, R22 ;
  IADD3.X R9, ..., R9, UR4, ... ;
merge:
```

The full-refactor code flattens to predicated execution (no branch), which
needs the same constant in a GENERAL register because it's combined via
predicated `SEL`/`IADD3`:

```sass
@P3 IADD3 R35, ..., R32, 0x1, R23 ;
@!P3 LDC R37, c[0x0][0x3a8] ;              // GP-reg load instead of UR
@!P3 SEL R37, R26, R37, P3 ;               // predicated select
@!P3 IADD3 R4, ..., R37, R4, R32 ;
```

The compiler's choice between branched-merge-with-UR vs predicated-execution
flips because the new code adds runtime `if (pick_source_b)` branches in
`set_tile_base`, `submit_load`, `gather_one`, and the dtor (none of which
existed unconditionally before; the dev shape was `source_a.*; source_b.*;`
unconditional dual-dispatch). Each new branch is uniform-across-warp by
construction (`pick_source_b` is set once at ctor and never changes), but
ptxas can't prove that and falls back to predicated execution.

Predicated `SEL` keeps both operands live in general registers (vs uniform
registers with `LDCU`), which is what drives the +10 register increase on
`last_filter` and the corresponding occupancy drop (3 -> 2 CTAs/SM for I64
keys at 50 regs/thread × 512 threads/CTA = 25600 per CTA vs the 65536/SM
budget).

## What the factory-only variant changes

The factory-only variant keeps the proposal's surface API change (factory-
callback ctor + non-copyable / non-movable) but reverts the *internal*
representation to the OLD shape: `SourceA source_a; SourceB source_b;` plain
members, both constructed via their factory at ctor. `set_tile_base`,
`submit_load`, `gather_one` use the OLD operational patterns:

```cpp
// OLD: unconditional dual-dispatch
void set_tile_base(OffsetT tb) {
  source_a.set_tile_base(tb);
  source_b.set_tile_base(tb);
}
// ternary for gather_one
return pick_source_b ? source_b.gather_one(i) : source_a.gather_one(i);
```

No new runtime branches; ptxas's existing optimization heuristics (LDCU
hoisting, uniform-register propagation, branch+merge) keep firing. The
`I64`-pair regression vanishes (1.012x -> 1.003x).

The trade-off: we drop the proposal's "single active source" smem-saving
target. For today's `<direct, direct>` / `<sync_block_load, direct>`
configurations there is no smem cost (both arms are register-resident anyway),
so this is a no-op until a future `<async_to_shared, X>` config arrives. At
that point the type would need the placement-new shape *and* a way to
preserve LDCU hoisting under the runtime branches — probably an explicit
uniformity hint (`__shfl_sync` broadcast, `makeWarpUniform`, or
`_CCCL_GRID_CONSTANT` on the ctor's `pick_b` arg).

## Remaining factory-only regression

The factory-only variant still shows a small (+8 reg) regression on `filter`
for `I16` keys-only and `finalize_filter` for `I8` keys-only (+10). Runtime
impact: I16 keys-only Elements=2^28 Entropy=1.000 -> **+5.4%** worst case
(462us -> 487us). These are configurations where ptxas's register allocator
struggles regardless of the structural change — both kernels are at the
margin where one extra spilled live range costs a register.

Diagnosing this would require either:
- a focused micro-bisect of the agent diff (the factory ctor introduces
  lambdas; ptxas might be tracking lambda captures less precisely than
  direct-construction);
- or, the proposal's "lift value_state to durable storage" follow-up (the
  per-tile aggregate `value_source_t::TempStorage value_state{};` stack-local
  changes the agent's stack layout and might be pushing the I16 keys-only
  filter over the threshold).

## Recommendation

The full refactor as written **regresses materially** on I64 pairs (~+1.2%
mean, +6.5% worst) due to ptxas losing uniform-register hoisting under the
new runtime branches. The factory-only variant captures most of the
proposal's API benefit (factory-callback ctor, composes with non-copyable
children for future `async_to_shared` work) **without** the SASS regression
on the hot kernels.

Options:
1. **Land the factory-only variant.** Keeps the agent migration done; opens
   the door to non-copyable children later. Costs a small remaining ~1%
   regression on I16-keys-only configurations (worth a separate follow-up).
2. **Land the full refactor and accept the I64 regression.** Needed if we
   want the smem-saving for an imminent async children deployment. The
   regression would have to be paid back by the async savings.
3. **Iterate on the full refactor:** investigate uniformity hints
   (`makeWarpUniform` on `pick_source_b`, or templated specialization for
   common pick_b values) to recover the LDCU hoisting. Open-ended.

I'd lean toward **(1)** — the smem-saving from "single active source" only
pays off for non-trivial children, none of which are in tree today. The
factory-callback API is the durable part of the proposal; the placement-new
mechanic can be added back when the use case materializes and the
uniformity hint story is sorted out.

## Files

Both variants share the agent migrations:
- `cub/cub/agent/agent_topk.cuh` -- `make_value_channel_sources` + the
  early-stop / buffered / last-filter ctor sites.
- `cub/cub/agent/agent_batched_topk.cuh` -- `process_tile_{early_stop,
  buffered, unbuffered}` (filter agent), `make_keys_source_for_segment`
  (last-filter agent), `process_tile` per-tile value-source helper, and the
  destroy-then-construct at the segment boundary.

Variant-specific files:
- `cub/cub/detail/topk/tile_data_source.cuh` -- `multi_source_data_source`
  type definition.

## Snapshots / sweeps captured

- `topk_perf_tracking/snapshots/dev_n261_v3.json` -- dev baseline (n261).
- `topk_perf_tracking/snapshots/active_source.json` -- full refactor.
- `topk_perf_tracking/snapshots/factory_only.json` -- factory-only variant.
- `topk_perf_tracking/bench/sweep_dev_n261_v3_{pairs,keys}.json`.
- `topk_perf_tracking/bench/sweep_active_source_{pairs,keys}.json`.
- `topk_perf_tracking/bench/sweep_factory_only_{pairs,keys}.json`.

Aggregation tool: `topk_perf_tracking/aggregate_sweep_per_kv.py`.
