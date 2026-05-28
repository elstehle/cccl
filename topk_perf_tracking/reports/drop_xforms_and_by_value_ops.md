# BlockPartition / BlockFilter: drop identity transforms + by-value ops

Two experiments stacked on top of the `flat_last` baseline
(`exp/flat-cta-walk-plus-last`).

## Experiment 1 -- drop always-identity transforms

**Branch:** `exp/drop-identity-xforms` / `tmp/perf-eval-drop-xforms`
(`fdbf983018`).

### What changed

Every instantiation of `block_partition_*` and `block_filter_*`
across `agent_topk` and `agent_batched_topk` used `cuda::std::identity`
for the selected/candidate key-out transforms and for the per-channel
value transforms held inside `value_channel_sinks_t` /
`value_channel_sinks_filter_t`. The transforms are removed entirely:

- `value_channel_sinks_t` loses `SelectedValueTransform` /
  `CandidateValueTransform` template params and members
  (writes now go directly through the output iterators).
- `value_channel_sinks_filter_t` loses `SelectedValueTransform`.
- `block_partition_atomics`, `block_partition_staged`,
  `block_partition_shared_mem`, `block_partition_accumulating_candidates`,
  `block_partition_speculative` lose
  `SelectedKeyOutTransformOp` / `CandidateKeyOutTransformOp`
  template parameters, ctor parameters, and reference members.
- `block_filter_atomics`, `block_filter_staged`,
  `block_filter_shared_mem`, `block_filter_accumulating`,
  `block_filter_speculative` lose `SelectedKeyOutTransformOp`.
- The `strategy_to_partition_class` / `strategy_to_filter_class`
  forwarders no longer thread the transforms through.
- The agent-side `key_xform_t` alias and `sel_key_xform` / `cand_key_xform`
  locals are deleted from the ctor call sites.

If a non-trivial transform is ever needed again, the right place to
put it is inside the iterator (e.g. `cuda::transform_output_iterator`),
which is already how the indexed-value gather path threads its
non-trivial transform end-to-end without touching the block
primitive.

LOC: **+116 / -244** (net -128) across 14 files (8 in `cub/cub`,
6 test files).

### Resource impact (PTXAS verbose, sm_100)

**Zero change** across all filter and last_filter kernel instances.
Registers / smem / stack frame / spill stores / spill loads are
byte-identical to `flat_last` for every (KeyT, ValueT) combo.

| | flat_last | drop_xforms | delta |
|---|---|---|---|
| filter, KeyT=int8/int8, regs/stack/spill | 40 / 24 / 28+20 | 40 / 24 / 28+20 | 0 |
| filter, KeyT=int/long, regs/stack/spill | 40 / 0 / 0+0 | 40 / 0 / 0+0 | 0 |
| filter, KeyT=long/long, regs/stack/spill | 40 / 0 / 0+0 | 40 / 0 / 0+0 | 0 |
| last_filter, KeyT=int8, regs | 64 | 64 | 0 |
| last_filter, KeyT=short, regs/stack/spill | 40 / 24 / 68+60 | 40 / 24 / 68+60 | 0 |

Explanation: the references to stateless `cuda::std::identity`
functors were dead weight that ptxas / the C++ inliner had already
eliminated. The struct still nominally held two 8-byte references,
but ptxas/optimization recognized the result of calling the empty
functor as the input itself and dropped the pointer fields too.

### Benchmark impact

I32/I32 sweep (full 51-workload grid, identical axes as the previous
report):

| Entropy | n | drop_xforms / flat_last geomean |
|---|---:|---:|
| 1.000 | 17 | 0.9999x |
| 0.201 | 17 | 1.0002x |
| 0.000 | 17 | 0.9999x |
| **overall** | **51** | **1.0000x** |

Effectively no change (within the noise band of ~1%).

### Verdict

**Pure code-quality win, no perf cost.** Worth landing. The deleted
template parameters & ctor surface make the partition / filter API
notably shorter and clearer.

## Experiment 2 -- by-value ops

**Branch:** `exp/by-value-ops` / `tmp/perf-eval-by-value`
(`5558d89726`), stacks on top of experiment 1.

### What changed

In all six `block_partition_*` / `block_filter_*` classes, the
operator/sink fields previously held by reference are now held by
value:

| field | before | after |
|---|---|---|
| `SelectedReserveOp` | `T& reserve_sel;` | `T reserve_sel;` |
| `CandidateReserveOp` | `T& reserve_cand;` | `T reserve_cand;` |
| `ValueChannelSinksT` | `T& sinks;` | `T sinks;` |
| `IdentifyCandidatesOp` | `T& identify_op;` | `T identify_op;` |
| `CandidateCallbackOp` | `T& callback_op;` | `T callback_op;` |

Same change applies to the ctor parameter declarations
(`T& foo` -> `T foo`). The init lists (`reserve_sel(reserve_selected)`)
are unchanged in syntax -- they just become copy-construction
instead of reference binding. Call sites in the agents don't need
to change.

LOC: +70 / -70 across 6 files.

### Resource impact (PTXAS verbose, sm_100)

**Zero change** vs experiment 1 (and therefore vs `flat_last`).
Every register/smem/stack/spill number is byte-identical.

### Benchmark impact

I32/I32 sweep (51 workloads):

| Entropy | n | by_value / flat_last geomean |
|---|---:|---:|
| 1.000 | 17 | 1.0004x |
| 0.201 | 17 | 0.9996x |
| 0.000 | 17 | 0.9999x |
| **overall** | **51** | **1.0000x** |

Same conclusion -- within noise.

### Why no change?

The optimization the by-value form is supposed to enable -- avoiding
a load on every per-item access of an operator's state -- is
already happening at the ptxas level. The block primitive's body is
fully inlined into the kernel, the call sites of the operators are
visible to the optimizer, and any reference to a small-state /
stateless functor is treated by the inliner as if the functor's
state is directly in scope. The by-reference shape was paying its
abstraction tax in code complexity, not in code generation.

A by-value shape would have shown a real saving if either:
- The block primitive's body was *not* fully inlined into the kernel
  (in which case the reference would force an actual pointer load),
  or
- A captured operator carried enough state that reading it through a
  reference required spilling intermediates while copying it into
  registers wouldn't.

Neither condition holds for the topk pipeline on Blackwell with the
current policy: `__launch_bounds__` plus `_CCCL_FORCEINLINE` on the
hot path keep everything in one big inlined kernel body, and the
operators carry at most an 8-byte pointer of state.

### Verdict

**Also a code-quality win, also no perf change.** Stricter than
experiment 1: nothing structurally breaks (still copies cheap state
into the block primitive), and the by-value form removes one entire
class of dangling-reference / lifetime-of-temporary bugs from the
block-primitive API. Worth landing.

## Recommendation

Land both. Experiment 1 alone is a clear net positive (-128 LOC,
0 perf cost). Experiment 2 on top of it removes another sharp edge
(reference members holding addresses of CTOR-local temporaries) for
zero perf cost. If we want to pick just one, experiment 1 has the
bigger code-cleanup story; experiment 2 is the smaller "API
hygiene" change.

## Artifacts

- `topk_perf_tracking/snapshots/{drop_xforms,by_value}.json` --
  resource snapshots.
- `topk_perf_tracking/bench/{bench,sweep}_{drop_xforms,by_value}_*.json` --
  raw nvbench output for the 12-workload focused set and the 51-workload
  I32/I32 sweep.
- `topk_perf_tracking/raw_logs/{drop_xforms,by_value}__pairs.log` --
  PTXAS verbose build logs.
- Branches: `tmp/perf-eval-drop-xforms`, `tmp/perf-eval-by-value`.
