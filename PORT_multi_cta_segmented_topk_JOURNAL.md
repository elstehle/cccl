# Port journal: multi-CTA (non-cluster) segmented top-k, `exp/batched-topk-agent` -> `main`

Running log of findings, discrepancies, open questions and decisions.
Newest entries appended at the bottom of each section.

---

## 0. Safety / preconditions (done)

| Action | Result |
| --- | --- |
| `backup/batched-topk-agent-pre-port-20260821-221204` | created at `acb52b215d`, pushed to `origin` |
| push `exp/batched-topk-agent` | had no remote; now pushed to `origin` (`acb52b215d`) |
| push `fea/topk-batched` | was 3 ahead; pushed `b6546d93b1..f326b019f6` |

---

## 1. Verification of the brief's claims

Everything numeric in the brief checks out. Recorded here so later
disagreements can be traced.

| Claim | Verdict | Evidence |
| --- | --- | --- |
| merge-base is `39003160b4` | correct | `git merge-base main exp/batched-topk-agent` |
| exp is 15 ahead / 666 behind main | correct | `git rev-list --left-right --count` |
| 1 feature commit `57d05aefd7` + 14 pruning commits | correct | `git log --oneline 39003160b4..exp` |
| additive part = 2764 lines / 10 files | correct, exactly | sum of the 10 per-file insert counts |
| drift figures (dispatch 1319, kernel 414, tuning 323, agent_batched 193, agent_topk 22, dispatch_topk 98) | correct | these are `39003160b4..main` drift, not exp-side sizes |
| bench `topk/{keys,pairs}.cu` drift = 4 each | correct | `git diff --stat 39003160b4..main` |

### 1.1 DISCREPANCY - files in the net diff that the brief does not mention

`git diff --stat 39003160b4..exp -- cub/` lists 23 files; the brief accounts
for 16. The unlisted ones:

| File | exp change | main drift since merge-base | Note |
| --- | --- | --- | --- |
| `cub/test/catch2_test_device_topk_tile_data_source.cu` | +514 (new file) | n/a (absent on main) | unit test for `tile_data_source.cuh`, which *is* in the port list. Genuinely additive; should be ported with its subject. |
| `cub/test/catch2_test_device_segmented_topk_keys.cu` | +259 | **2171** | not a near-clean reapply; see 1.2 |
| `cub/test/catch2_test_device_segmented_topk_pairs.cu` | +180 | **2170** | same |
| `cub/benchmarks/bench/segmented_topk/variable/keys.cu` | +34 | **251** | benchmark knobs for the multi-worker policy |
| `cub/cub/detail/warpspeed/make_warp_uniform.cuh` | +17/-? | **0** | untouched on main -> genuinely near-clean reapply |
| `topk_repro/**` (10 files, 609 lines) | new | n/a | explicitly excluded by the brief |

### 1.2 DISCREPANCY - the tests are a semantic merge, not a reapply

The brief does not classify the test files at all. They are the *third
largest* merge surface after `dispatch_batched_topk.cuh`:

* main's `catch2_test_device_segmented_topk_keys.cu` is 2198 lines / 32 test
  cases (up from ~27 lines at the merge-base).
* Test macro was renamed `C2H_TEST` -> `CUB_TEST` on main. All ported testxp
  cases need mechanical renaming.
* exp adds 3 test cases:
  * "work with large fixed-size segments (multi-CTA all-large)"
  * "work with large variable-size segments (multi-CTA mixed)"
  * "handles boundary large-segment counts (mixed-path edge cases)"
* NAMING COLLISION RISK: main already has cases named
  "run a small multi-CTA segment through the cross-CTA scan" and
  "handle the maximum number of segments through a multi-CTA cluster".
  On main "multi-CTA" means *cluster CTAs*; in the port it means
  *multi-CTA-per-segment baseline*. Ported test names must disambiguate
  (proposal: say "baseline multi-CTA" or "multi-CTA-per-segment").

### 1.3 DISCREPANCY (material) - 3 of the 10 "purely additive" files are extractions, not additions

The brief says of the 10 additive files: "None exist on main." The *files*
indeed do not exist on main. But three of them **define symbols that already
exist on main**, in `namespace detail::topk`, just in different headers:

| New header (exp) | Symbol(s) | Already defined on main at |
| --- | --- | --- |
| `cub/cub/detail/topk/candidate_class.cuh` | `detail::topk::candidate_class` | `cub/cub/agent/agent_topk.cuh:189` |
| `cub/cub/detail/topk/key_prefix_storage.cuh` | `key_prefix_storage_t<K,true/false>`, `calc_start_bit` (2 overloads) | `cub/cub/agent/agent_topk.cuh:62,112,87,98` |
| `cub/cub/device/dispatch/dispatch_topk_identify_candidates.cuh` | `identify_candidates_op_t` (2 partial specs) | `cub/cub/device/dispatch/dispatch_topk.cuh:119,122,155` |

Copying these headers in without removing the originals is an ODR / redefinition
error. This is exactly what exp commits `bd5a41052d` and `915c869562` did: they
**moved** the symbols out of `agent_topk.cuh` / `dispatch_topk.cuh` into the new
headers and included them back. That is also why the brief's "near-clean
reapply" entries `agent_topk.cuh` (-60/+10) and `dispatch_topk.cuh` are
net-negative: they are the deletion half of the extraction.

Mitigating fact: main grew a *third* consumer of these symbols,
`agent_batched_topk_cluster.cuh`, and it already refers to them by their
`detail::topk::`-qualified names, reaching them transitively via
`#include <cub/agent/agent_topk.cuh>` and `#include <.../dispatch_topk.cuh>`.
Since the extraction leaves those two headers including the new ones, the
cluster agent keeps compiling unchanged. Verified by include inspection; to be
confirmed by build.

=> Reclassification: **7 files purely additive (2477 lines), 3 files are a
2-file extraction refactor touching `agent_topk.cuh` + `dispatch_topk.cuh`.**

### 1.4 Confirmed - the additive files carry no params-framework debt

`agent_topk_common.cuh`, `tile_data_source.cuh`, `dispatch_topk_common.cuh`
and `dispatch_topk_identify_candidates.cuh` reference **zero** `params::`
symbols. This matters because the params framework is the single largest
mechanical drift on main (see 2.2). The whole `detail/topk/` block is
params-agnostic, so it ports without translation.

---

## 2. How main's dispatch actually selects a path

### 2.1 The selector

`detail::batched_topk::policy_selector_from_types::operator()(cc)` in
`dispatch_batched_topk.cuh:177-198` returns a `topk_policy{backend, baseline,
cluster}` where `backend` is `topk_algorithm::{baseline, cluster, unsupported}`:

```
deterministic := Determinism != not_guaranteed || TieBreak != unspecified

if (deterministic || !baseline_can_cover)
    backend = cluster_capable(cc) ? cluster : unsupported
else
    beneficial = StaticMaxSegSize >= cluster_beneficial_min_segment_size   // 8*1024
    backend = (cluster_capable(cc) && beneficial) ? cluster : baseline
```

`baseline_can_cover` = `baseline_can_cover_v<...>` =
"some `worker_policy` in the array has `tile_size >= static max segment size`
**and** its instantiated agent's `TempStorage` fits `max_smem_per_block`".
Largest worker tile today is `256*64 = 16384` keys.

`cluster_capable(cc)` = `cc >= 9.0` and `_CCCL_HAS_DYNAMIC_CLUSTER_LAUNCH()`.

Resulting behavior matrix on main:

| static max segment size | SM90+ | pre-SM90 |
| --- | --- | --- |
| <= 8 Ki | baseline | baseline |
| 8 Ki .. 16 Ki | cluster | baseline |
| > 16 Ki | cluster | **unsupported** |
| any, deterministic requested | cluster | **unsupported** |

The `> 16 Ki / pre-SM90 = unsupported` cell is issue #9253.

### 2.2 Structural drift on main that the port must absorb

1. **One kernel symbol, two backends.** `device_batched_topk_kernel`
   (`kernel_batched_topk.cuh:307`) hosts both arms; the arm is picked
   *device-side* by `current_policy<PolicySelector>()`. Backend-specific
   arguments travel in `baseline_kernel_args<...>` / `cluster_kernel_args`
   structs, the unused one default-constructed. The header states this
   honors "CUB's one-kernel-per-arch/problem rule".
2. **`::cuda::args` parameter framework.** The old
   `params::{uniform_param, per_segment_param, static_constant_param,
   uniform_discrete_param}` are gone; the public surface is
   `::cuda::args::{constant, immediate, deferred, deferred_sequence, bounds}`.
   Translation table for the port:
   | exp | main |
   | --- | --- |
   | `params::static_max_value_v<T>` | `::cuda::args::__traits<T>::highest` |
   | `params::static_min_value_v<T>` | `::cuda::args::__traits<T>::lowest` |
   | `T::value_type` | `::cuda::args::__traits<T>::element_type` |
   | `p.get_param(i)` | `params::get_param(p, i)` |
   | `narrow_segment_count_t<T>` | `::cuda::args::__traits<T>::element_type` |
   | `params::is_per_segment_param_v<T>` | `::cuda::args::__traits<T>::is_deferred` (+ `is_single_value`) |
   | (none) | `params::__get_and_clamp_param_to_nonnegative(p, i)` - new, negative sizes clamp to 0 |
3. **Selection direction is compile-time only.** main's `wrap_select_direction`
   accepts *only* `::cuda::args::constant<Dir>` and `static_assert`s on
   anything else. exp's dispatch instead lowers a *runtime* direction with a
   hand-rolled `launch_passes(integral_constant<...>)` host-side branch. The
   port drops that branch entirely: direction arrives already lowered. This
   removes ~15 lines and one host-side `if/else` from exp's dispatch.
4. **Launches go through `KernelLauncherFactory`.** exp calls
   `THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(...).doit(...)`,
   `cudaMemsetAsync`, `cudaGetDevice` + `cudaDeviceGetAttribute`, and
   `MaxSmOccupancy` directly. main routes all of these through
   `launcher_factory`. Verified the factory covers every call the multi-CTA
   dispatch needs: `operator()(grid, block, smem, stream, dependent_launch)`,
   `MemsetAsync`, `MaxSmOccupancy`, `MultiProcessorCount`. So this is a
   mechanical 1:1 rewrite, no capability gap.
5. **`num_segments` must be host-known.** main `static_assert`s
   `is_single_value && !is_deferred` (dispatch_batched_topk.cuh:951). exp
   supported a device-only `num_segments` (that is what its
   `constant_value_op` + sentinel-slot-in-scan machinery is *for*: the
   all-large scan runs over `num_segments + 1` inputs so the sentinel slot
   holds `total_large_tiles`, and `large_segments_count_it` can be a constant
   transform_iterator reading `num_segments.get_param(0)` on-device).
   => OPEN QUESTION Q1 (see section 4).
6. **`policy_selector` renames.** exp `batched_topk_policy` -> main
   `baseline_topk_policy`, wrapped in main's new `topk_policy{backend,
   baseline, cluster}`. exp `policy_selector{key_size}` -> main
   `make_baseline_policy()` (no key_size input, CC-independent).
7. **Ostream/equality boilerplate is now mandatory.** main's
   `dispatch_compute_cap` requires the selector result to be
   `::cuda::std::regular`, so every policy struct carries `operator==`,
   `operator!=` and `operator<<`. exp's `multi_worker_policy` has these, but
   the port must extend them to the 9 new fields (they already are on exp).
8. **`_CCCL_HOSTED()` vs `!_CCCL_COMPILER(NVRTC)`.** exp's tuning header
   guards its `operator<<` with a mix of both (and mismatched closing
   comments). main uses `_CCCL_HOSTED()` uniformly. Clean up while porting.

### 2.3 The producer side already on main, and what is missing

main's `launch_baseline_arm` already:
* computes `any_small_segments` / `only_small_segments` from
  `__traits<SegmentSizeParameterT>::{lowest,highest}` vs the worker tile size;
* allocates `[0]` tile offsets, `[1]` counters, `[2]` large-segment ids (mixed)
  or `[0]` tile offsets, `[1]` scan temp (all-large);
* memsets the counters, launches the worker kernel (which enqueues large
  segments), or runs the transform-scan for the all-large case.

and then **stops**. Nothing consumes the queue. Carried TODO at
`dispatch_batched_topk.cuh:774-778`:

> "the baseline large-segment (multi-CTA) path is WIP. Completing it requires:
> (1) guarding the `num_segments_val * sizeof(...)` byte counts below against
> size_t overflow ...; (2) making the baseline tunable by populating its
> `epilogue` and `multi_worker_per_segment_policy` sub-policies and adding
> matching knobs to the segmented_topk benchmarks, which leave them
> zero-initialized today so baseline sweeps are not yet meaningful."

Both items are exactly what the port supplies. Note (1) is a real bug the port
must fix, not inherit: exp sized allocations off `num_segments_upper_bound`
with the same unguarded multiply.

Also missing vs exp: main's allocation layout stops at 3 slots; exp appends 4
(keys-only) or 6 per-segment multi-CTA slabs (counters, histograms, key buf
A/B, value buf A/B).

### 2.4 What the port adds, by layer

| Layer | Addition |
| --- | --- |
| `agent_batched_topk.cuh` | 3 new agents: `agent_batched_topk_histogram` (325 ln), `..._filter_partition` (819 ln), `..._last_filter` (480 ln). Purely appended after the existing `agent_batched_topk_worker_per_segment`. |
| `kernel_batched_topk.cuh` | 5 new kernel symbols: histogram, finalize_histogram, filter, finalize_filter, last_filter. Plus `find_largest_fitting_smem_policy` + `resolved_worker_per_segment_policy` (the fits-smem fallback main lacks). |
| `tuning_batched_topk.cuh` | `value_materialization_mode` enum; `multi_worker_policy` grows 2 -> 11 fields; `make_baseline_policy()` gains the key-size-dependent `items_per_thread` / `bits_per_pass` computation. |
| `dispatch_batched_topk.cuh` | ~450-line multi-CTA launch block inside the `!only_small_segments` branch: per-segment slab allocation, 2 memsets, radix-pass loop (histogram -> finalize -> [filter -> finalize]* -> last_filter). |

### 2.5 KEY FINDING - main already reserves this design slot

`device_batched_topk.cuh:183-186` caps the statically-known maximum segment
size at 2^21 with this rationale:

> "Only the statically-known *maximum* segment size is constrained: it must not
> exceed 2^21 (about 2 million). Beyond that the streaming cluster backend is
> not competitive; larger segments are future work (**a WIP multi-CTA baseline
> backend**)."

So main's own documented plan is: baseline for small, cluster up to 2^21,
multi-CTA baseline beyond. The port is the intended completion, not a
competing design. This removes most of the ambiguity from the integration
question and is the strongest argument for keeping multi-CTA *inside* the
baseline backend rather than making it a peer `topk_algorithm` enumerator.

### 2.6 Multi-CTA is not a peer of baseline - it is a sub-strategy

Decisive structural point: for a **mixed** batch (small and large segments in
one call) the worker-per-segment kernel and the multi-CTA kernels are not
alternatives - they *cooperate in a single dispatch*. The worker kernel
processes small segments and *enqueues* the large ones; the multi-CTA kernels
then drain that queue. A separate `topk_algorithm::multi_cta` enumerator cannot
express "both at once", and main's `baseline_topk_policy` already carries
`multi_worker_per_segment_policy` as a field, i.e. the slot is already modeled
correctly. Any option that promotes multi-CTA to a backend enumerator has to
re-model the mixed path.

### 2.7 Tests that encode the behavior being changed

* `cub/test/test_device_batched_topk_forced_baseline_oversize_fail.cu` -
  forces the baseline backend with a `2^20` static max segment size and
  `// expected-error {{"cannot cover the static maximum segment size"}}`.
  **The port makes this configuration legal**, so this compile-fail test must
  be retired or repointed (e.g. at a `multi_worker` policy that itself cannot
  fit smem). Direct, unavoidable behavioral consequence.
* `cub/test/test_device_batched_topk_unsupported_arch_fail.cu` - deterministic
  request on pre-SM90. Multi-CTA is **not** deterministic, so this stays valid
  unchanged. Good: it pins the one cell of the matrix the port must not touch.
* `cub/test/test_device_batched_topk_dynamic_cluster_disabled_fail.cu` and
  `..._requirements_fail.cu` - to be reviewed; likely unaffected.

### 2.8 `_CCCL_GRID_CONSTANT` constraint - status

The brief asks to preserve exp's `kernel_batched_topk.cuh:218` decision that
the per-segment reserve-counter pointers stay plain, not `_CCCL_GRID_CONSTANT`
(worth ~10x on contention workloads because grid-constant pointers lower to
generic addressing, defeating warp-aggregation of the scatter `atomicAdd`s).

Verified: main's batched-topk kernels use **no** `_CCCL_GRID_CONSTANT` at all
(the only users anywhere in `cub/` are `dispatch_find_bound_sorted_values.cuh`
and `dispatch_select_if.cuh`). main passes these pointers by value inside
`baseline_kernel_args`, which is equivalent to a plain pointer parameter for
addressing purposes. So the constraint is satisfied by default on main; the
port's job is (a) not to introduce the annotation on the 5 new kernels, and
(b) carry the explanatory comment forward so nobody "tidies" it in later.

---

## 3. Base-branch question

Local `main` is exactly 25 commits behind `upstream/main`, and
`git diff --stat main upstream/main -- '*topk*'` is **empty** - no topk file
differs. So basing the port on local `main` vs `upstream/main` is equivalent
for this work. Proceeding on local `main` unless told otherwise; noted so the
choice is deliberate.

---

## 4. Open questions

**Q1 - device-resident `num_segments`.** exp supports it; main
`static_assert`s it away. exp's sentinel-slot scan design and
`constant_value_op` exist to serve it. Options: (a) port the machinery but
leave it unreachable behind main's assert (dead code, but zero-cost and ready
when main lifts the restriction); (b) simplify it out and re-derive
`total_large_tiles` / `large_segments_count` from host-known
`num_segments`. (b) is less code but discards work and diverges from exp if
main later lifts the assert. Leaning (a)-minus: keep the sentinel-slot layout
(it is also what lets the all-large path read `total_large_tiles` without a
separate counter) but drop `constant_value_op`'s device-read justification and
the `num_segments.max_value` fallback, since `num_segments_upper_bound` is
always exact on main.

**Q2 - kernel symbol count.** main deliberately collapsed to one symbol per
instantiation. The multi-CTA path needs 5 more. Are they acceptable as
separate symbols (they are separate *passes*, not separate *backends*, so the
one-kernel-per-arch rationale arguably does not apply), or should they be
folded behind a pass-selector NTTP on one symbol? Separate symbols is what exp
has and what the single-problem `dispatch_topk.cuh` does; recommending that.

**Q3 - the 2^21 public cap.** Lifting it is the actual user-visible closure of
#9253. Should the port lift it in the same PR, or land the machinery first and
lift the cap in a follow-up? Lifting it changes the public contract and
invalidates the `static_assert` message + docs in `device_batched_topk.cuh`.

**Q4 - benchmark knobs.** main's TODO calls out that the segmented_topk
benchmarks leave the baseline sub-policies zero-initialized, so baseline sweeps
are meaningless. exp's `bench/segmented_topk/variable/keys.cu` change (+34)
plus the `bench/topk/{keys,pairs}.cu` changes are the knob wiring. In scope?

**Q5 - `topk_repro/`** excluded per the brief. Confirm the June B200 reference
numbers are simply dropped rather than regenerated post-port.

---

## 5. Decisions

(recorded as they are made)

* D0 - Do not `git rebase`; port the net diff `39003160b4..exp` onto a fresh
  branch off `main`, in logical commits. (from the brief)
* D1 - Leave `topk_repro/` out. (from the brief)
* D2 - Keep reserve-counter pointers plain, never `_CCCL_GRID_CONSTANT`;
  carry the rationale comment. (from the brief; verified feasible, see 2.8)

---

## 6. Dispatch taxonomy (elstehle's framing) and its mapping to code

### 6.1 The three strategies, as named

| Name | Meaning | Grid shape |
| --- | --- | --- |
| worker-per-segment ("baseline") | at most **one CTA** per segment; conditionally compiles in an escalation step that pushes oversize segments onto a queue | `grid.x = num_segments`, 1 CTA/segment |
| cluster | **one cluster** per segment | `grid = {num_segments, cluster_blocks, 1}`, `cluster = {1, cluster_blocks, 1}`; `segment_id = clusterid.x` (verified: `agent_batched_topk_cluster.cuh:3325`) |
| multi-CTA | **load-balanced multi-CTA** per segment, grid-strided over a global tile space with a per-segment tile-offset table | `grid = min(occupancy * num_sms, total_large_tiles_ub)` |

### 6.2 The scenarios

| # | Scenario | Producer | Consumer(s) |
| --- | --- | --- | --- |
| A | worker-only, no escalation | worker kernel | none |
| B | cluster-only | none (no queue) | cluster |
| C | worker + escalation | worker kernel (or scan, see 6.3) | queue-driven |
| C1 | one queue | " | multi-CTA |
| C2 | two queues (cluster-capable archs only) | " | cluster + multi-CTA, split by a tuning threshold |

### 6.3 ADDITION - C bifurcates on whether any small segments exist

main already implements both halves of the producer side, and they are
structurally different, not a detail:

* **C-mixed** (`any_small_segments && !only_small_segments`): worker kernel
  runs, triages per segment, enqueues via `atomicAdd(&d_counters->
  large_segments_count, 1)` and writes `d_large_segments_ids[queue_idx]` +
  the segment's tile count; last block to retire runs an epilogue scan over
  the queued tile counts. **Queue length is device-resident.**
* **C-all-large** (`!any_small_segments`): worker kernel is skipped entirely;
  a `transform_scan` over the segment sizes synthesizes the tile-offset table,
  the queue is the identity permutation (`counting_iterator` as
  `segment_id_provider`), and **the count is host-known**.

The host-known vs device-resident count is what decides whether a consumer's
grid can be sized on the host. This is the single most consequential
distinction for C2 (see 6.6).

### 6.4 A vs B is already two stacked thresholds on main

* hard capability bound: `baseline_can_cover` (largest worker tile = 16384,
  and the agent's `TempStorage` must fit 48 KiB);
* soft tuning bound: `cluster_beneficial_min_segment_size = 8*1024`, applied
  only where `cluster_capable(cc)`.

So "A if static ub below a threshold" is really "A if the worker can cover it
**and** cluster is either unavailable or not yet beneficial".

### 6.5 Mapping: scenario -> what exists / what the port adds / what is missing

| Scenario | Producer status | Consumer status |
| --- | --- | --- |
| A | complete on main (`allocations_array_size == 1`, dummy alloc) | n/a |
| B | n/a | complete on main (`launch_cluster_arm`) |
| C-all-large | **complete on main** (transform_scan path) | **missing** -> the port supplies it |
| C-mixed | **complete on main** (worker enqueue + epilogue scan) | **missing** -> the port supplies it |
| C1 | = C-all-large + C-mixed producers, unchanged | = the port |
| C2 | needs a 2nd id array + 2nd counter, 3-way triage in the worker agent, 2-way partition in the all-large scan | needs cluster-agent queue indirection **and** a cluster-stride grid; neither exists |

So **C1 is exactly the port**: main has both producers and neither consumer.

### 6.6 C2 blockers (unchanged by the tuning-bounds refinement)

1. **No queue indirection in the cluster agent.** `segment_id = clusterid.x`
   is hardcoded; there is no `d_cluster_segment_ids[queue_idx]` lookup and no
   bound against a device-resident count. The multi-CTA side already solves
   this generically via `segment_id_provider` (raw pointer for mixed,
   `counting_iterator` for all-large). Extending that abstraction to the
   cluster agent is the clean fix but touches Paul's agent.
2. **Grid sizing in C-mixed.** `grid.x` must be host-known; the cluster queue
   length is device-produced. Over-launching `grid.x = num_segments` with
   early-exit is bad here because each cluster reserves its full dynamic-SMEM
   opt-in, so idle clusters burn *occupancy*, not just cycles. A cluster-stride
   loop over the queue is the real answer and does not exist.
   Note: C2 in the **all-large** case has neither problem (host-known count,
   identity mapping) - i.e. C2 is cheapest to build exactly where it is least
   valuable.
3. **Consumer serialization.** Two queues drained on one stream serialize.
   C2 wins only if `T_cluster(subset) + T_multiCTA(subset) <
   min(T_all_cluster, T_all_multiCTA)`. Avoiding it needs 2 streams + events,
   which CUB dispatches avoid.
4. **Launch-count asymmetry.** cluster = **1** launch. multi-CTA =
   `2*num_passes + 1` launches + 2 memsets. With `bits_per_pass = 11`
   (`calc_bits_per_pass`, tuning_topk.cuh:29) that is 3 passes / **7 launches**
   for 32-bit keys and 6 passes / **13 launches** for 64-bit keys. In a
   straddling config C2 pays the multi-CTA launch floor even when its queue
   turns out empty at runtime.
5. **Determinism collapses C2 to B.** Cluster is the only deterministic
   backend, so any determinism / tie-break requirement must route everything
   to the cluster queue. Encode as an invariant.

### 6.7 The tuning-bounds refinement (elstehle's side note) - genuinely changes two of my objections

Side note: *each queue's admitted segment-size range is itself a tuning
policy*, so the size range feeding each consumer is compile-time known (this
applies to the single-queue C1 case too).

Assumed class ordering (**TO CONFIRM, Q7**): worker `[0, worker_tile]`,
cluster `(worker_tile, cluster_hi]`, multi-CTA `(cluster_hi, inf)`. Forced at
the top end by capability - main's own docs say cluster is not competitive
above 2^21 and multi-CTA is the only unbounded strategy.

What this **improves**:

* **Cluster launch right-sizing.** `launch_cluster_arm` currently derives its
  whole launch shape from `max_seg_size = ::cuda::args::__highest_(segment_sizes)`
  - the *global* static bound. With a tuned `cluster_hi` it becomes
  `min(global_highest, cluster_hi)`. That directly answers my earlier
  "routing into a cluster queue does not make the cluster launch adaptive"
  objection: it is not adaptive to runtime queue *content*, but it is
  right-sized to the queue's *declared* range, which is strictly tighter than
  today.
* **Straddling becomes a compile-time test.** If the static segment-size range
  does not cross a class boundary, that class's consumer is provably empty and
  need not be compiled or launched. So blockers 4 (launch floor) and the
  compile-time cost of instantiating both the 3404-line cluster agent and the
  5 multi-CTA kernels narrow to *straddling configurations only*. Note
  `cuda::args::constant<N>` is a single point (never straddles), while an
  un-annotated narrow type (e.g. `uint16_t` -> `[0, 65535]`) does straddle, so
  straddling is common for variable-size batches.

What this does **not** change:

* Blockers 1-3 and 5 above. Size bounds do not bound the queue *length*, so
  grid sizing and indirection remain.
* **Multi-CTA temp storage.** The multi-CTA class is the *top* class, so its
  upper bound *is* the global static max. `candidate_buffer_length` and the
  `N_slabs = num_segments_upper_bound` slab count get no benefit from the
  bound. The existing "flat cap is wasteful" TODO stands.

### 6.8 NEW FINDING - the boundaries have mixed nature, and a tuned gap is silently wrong

Making the class bounds tunable introduces two failure modes that need
compile-time validation, in the spirit of main's existing
`is_valid_cluster_policy`:

1. **Gaps / overlaps.** The classes must tile the size axis monotonically with
   no gap. A `tune`d override producing a gap means segments in that gap are
   processed by *no* consumer - silent wrong results, not an error.
2. **The lowest boundary is a capability limit, not a tuning knob.** The
   worker/queue boundary must be clamped to the *resolved* worker policy's
   tile size. Setting it *above* hands the worker segments it cannot hold
   (smem overflow / wrong results); setting it *below* is merely wasteful
   (escalating segments the worker could have served). So boundary 1 is
   hard-clamped while boundaries above it are freely tunable - worth making
   explicit rather than uniform.

Also concrete: the tile-offset table exists purely for multi-CTA load
balancing. The cluster queue needs no tile offsets, so in C2 the worker
epilogue scan covers only the multi-CTA queue.

### 6.9 Naming

* Mechanism: **escalation** (`escalation_queue`, "the worker escalates
  oversize segments"). Reads correctly in both directions. Alternative framing
  for the *kernel's role* in C: **triage**.
* Incumbent vocabulary on main is `d_large_segments_ids`,
  `large_segments_count`, `d_large_segments_tile_offsets`. Recommend keeping it
  for C1 (renaming is pure churn) and introducing per-strategy names only if
  C2 lands, where the two queues must be distinguished anyway
  (`d_cluster_segment_ids` / `d_multi_cta_segment_ids`).
* **`topk_algorithm::baseline` is already a misnomer on main** and gets worse:
  post-port it means "worker + multi-CTA", and under C2 "worker + multi-CTA +
  cluster". If the selector is being restructured anyway, the enum should name
  *which strategies are compiled in*, not "baseline vs cluster".

---

## 7. Open questions (continued)

**Q6 - Is C2 workload-driven or symmetry-driven?** If symmetry, the
launch-count asymmetry (1 vs 7-13) plus consumer serialization suggests the
honest answer for mixed batches is "one consumer per launch, chosen from the
batch's size *spread* rather than its max" - i.e. C1 with a smarter A/B/C
selector and no second queue.

**Q7 - Class ordering in C2.** Confirm cluster = middle band, multi-CTA = top
band (assumed in 6.7). If reversed, the right-sizing benefit moves to the
multi-CTA slabs instead of the cluster launch shape and the analysis flips.

**Q8 - Must the classes be contiguous size ranges?** The "two ranges" framing
forbids a non-monotonic routing rule (e.g. cluster for a narrow *band* with
multi-CTA on both sides). Contiguous ranges make triage a 2-threshold compare;
allowing non-monotonic rules complicates validation (Q/6.8) considerably.
=> DEFERRED to C2. Not needed for C1 (see D3).

---

## 8. Resolved decisions (2026-08-24 session)

* **D3 - Scope is C1 only; C2 is a follow-up.** With only two classes (worker
  `[0, tile]`, multi-CTA `(tile, inf)`) the routing boundary *is* the
  capability boundary `worker_per_segment_tile_size`. So C1 needs **no tunable
  routing bound**, and the gap/clamp validation of 6.8 plus Q7/Q8 are entirely
  C2 concerns. Constraint on C1's design: keep `segment_id_provider` as the
  consumer-side seam so C2 is dispatch plumbing plus one agent change, not a
  re-architecture.

* **D4 - Selector: fill only the cells that cannot be served today.** My
  original "conservative vs ceiling" split was a mis-partition: there are two
  distinct unserviceable states and I had put one in each question.
  1. *Selector returns `unsupported`* - reachable **only pre-SM90**, because
     `cluster_capable(cc)` is true on SM90+ so that branch never fires there.
  2. *Public entry `static_assert`* - fires on **all** archs, SM90+ included,
     above 2^21.
  Filling (2) on SM90+ is not a reroute of working code; it converts a compile
  error into working code. So "no regression on SM90+" and ">2^21 goes to
  multi-CTA on every arch" are complementary, not contradictory.

  Resulting matrix (C1 fills exactly 3 cells; every working cell untouched):

  | static ub | pre-SM90 today | pre-SM90 after | SM90+ today | SM90+ after |
  | --- | --- | --- | --- | --- |
  | <= 8 Ki | worker | unchanged | worker | unchanged |
  | 8 Ki - 16 Ki | worker | unchanged | cluster | unchanged |
  | 16 Ki - 2^21 | **unsupported** | **C1** | cluster | unchanged |
  | > 2^21 | **compile error** | **C1** | **compile error** | **C1** |
  | deterministic, any size | unsupported | unchanged | cluster | unchanged |

  Determinism stays cluster-only: the multi-CTA path scatters via `atomicAdd`
  and guarantees no ordering.

  Target selector shape (precedence matters - the `>2^21` rule must sit *below*
  the determinism rule, since a deterministic request above 2^21 still has to
  go to cluster, which is uncompetitive there but fully capable):

  ```
  if (deterministic)
      -> cluster if cluster_capable else unsupported            // unchanged
  else if (static_max > cluster_max_competitive_segment_size)   // NEW, 1<<21
      -> baseline (worker + multi-CTA)                          // all archs
  else if (!baseline_can_cover)
      -> cluster if cluster_capable else baseline(+multi-CTA)   // was unsupported
  else
      -> cluster if (cluster_capable && beneficial) else baseline  // unchanged
  ```

* **D5 - New selector constant, paired with the existing one.**
  `cluster_max_competitive_segment_size = 1 << 21` goes next to
  `cluster_beneficial_min_segment_size = 8 * 1024`, forming a symmetric pair
  bounding the cluster's competitive band. Both stay *selector constants*, not
  tunable policy fields, for the reason the existing one already documents:
  tuning the cluster policy must never shift the backend choice. Side benefit:
  2^21 acquires a meaning inside the selector instead of being an unexplained
  API cap.

  VERIFIED: `2^21` occurs in exactly **one** place on main
  (`device_batched_topk.cuh:198`, the entry assert). There is **no** hard 2^21
  assumption anywhere in `agent_batched_topk_cluster.cuh`, so the cluster path
  is *capable* above 2^21, merely uncompetitive.

* **D6 - Entry cap: relax conditionally, do not delete.** FLAGGED TO USER as
  the only public-contract change.
  - deterministic requests: keep `<= 2^21` (preserves today's contract exactly);
  - non-deterministic: replace with a guard on the multi-CTA path's actual
    representability, plus a doc note on temp-storage scaling.

  Sizing the real limits: `large_segment_tile_offset_t` is `uint32_t` over a
  2048-item tile for `int32` keys (`512 threads * 4 ipt`), so the aggregate
  ceiling is ~8.8e12 items - unreachable. The **binding** limit is temp
  storage: `num_segments * (max_seg_size / 128) * sizeof(key) * 4` (double
  buffer x keys+values). For a 4 Mi bound and 1000 segments with `int32` that
  is ~524 MB. This is the existing flat-cap wastefulness (already TODO'd on
  main), amplified by the lifted cap. exp's own comment concedes the offset
  table overflow is "not validated at runtime" - the port should validate it
  rather than inherit the silence.

* **D7 - Build/test: `native` only** (SM 7.0 on this box). SM90+ compile
  coverage deferred to a follow-up, per user.

  CONSEQUENCE TO TRACK: the extraction refactor (D8) moves symbols out of
  `agent_topk.cuh` / `dispatch_topk.cuh`, and
  `agent_batched_topk_cluster.cuh` consumes all three transitively. That agent
  is only instantiated when the selector returns `cluster`, which needs an
  SM90+ target in the compile list. So a `native`-only build compile-checks the
  extraction against every consumer **except its riskiest one**. Accepted risk;
  must be called out in the final report and covered by CI / the follow-up.

* **D8 - Naming: keep main's incumbent `large_segments` vocabulary**
  (`d_large_segments_ids`, `large_segments_count`,
  `d_large_segments_tile_offsets`). No `escalation` rename in this PR; revisit
  only if C2 lands, where the two queues must be distinguished anyway. The
  `topk_algorithm::baseline` misnomer is left alone for now (noted in 6.9).

* **D9 - File scope: agent's judgment, fully written up.** Intended scope:
  all 5 items from the scope question (tile_data_source test, the 3 segmented
  test cases merged into main's file, make_warp_uniform, benchmark knob
  wiring, and repointing `forced_baseline_oversize_fail`). Any deviation gets
  recorded here and in the final report.

### Environment (verified)

| Item | Value |
| --- | --- |
| GPUs | 8x Tesla V100-SXM2-32GB, **compute capability 7.0** (pre-SM90, NOT cluster-capable) |
| CUDA | 12.4 (V12.4.131) - older than the 13.3 in AGENTS.md's example |
| devcontainer | none (`CCCL_BUILD_INFIX` empty) |
| preset | `cub-cpp17` available |

Upside of SM 7.0: the automatic selector routes large segments to C1 **by
default** here, so the new path is exercised without forced-`tune` tests, and
main's large-segment tests that currently skip on this box go live after the
port. Downside: the cluster path (B) cannot be run locally at all.

### ISSUE E1 - no formatter available locally (unresolvable without admin)

`pre-commit`, `clang-format`, `pip` and `apt` are all unavailable and cannot be
installed (no admin rights). CI pins `mirrors-clang-format` **v22.1.5**.
AGENTS.md requires `pre-commit run` before committing and warns CI fails
otherwise.

Mitigation: every line of moved code is a **verbatim copy** from an
already-clang-formatted source (main's inline definitions / exp's headers), so
the bulk is conformant by construction. Only the new file headers and the added
`#include` lines are hand-written, and those follow the surrounding style
(alphabetically sorted includes, same license/`@file` block shape). Residual
risk is low but non-zero and **must be cleared by CI or by the user running
`pre-commit run --files ...` before the PR goes up.**

### ISSUE E2 - infra flake: remote backend outage

Mid-session the SSH remote to `dgx02` dropped and every tool backend returned
"Execution backend unavailable" for ~14 h of wall time. Nothing was lost (no
source file had been modified yet). Recorded only so the gap in the commit
timeline is explicable.

---

## 9. Chunk log

### Chunk 1 - extraction refactor (DONE)

Branch `fea/topk-batched-multi-cta` off `main` @ `0880804593`.

Moved, **verbatim** (diffed against main's inline versions before moving -
bodies were byte-identical to exp's extracted headers, and both were already in
`namespace detail::topk`, so this is a pure move with zero semantic change):

| Symbol(s) | From | To (new file) |
| --- | --- | --- |
| `candidate_class` | `agent_topk.cuh` | `cub/detail/topk/candidate_class.cuh` |
| `key_prefix_storage_t` (fwd + both specs), `calc_start_bit` x2 | `agent_topk.cuh` | `cub/detail/topk/key_prefix_storage.cuh` |
| `identify_candidates_op_t` (fwd + both specs) | `dispatch_topk.cuh` | `cub/device/dispatch/dispatch_topk_identify_candidates.cuh` |

Left in place deliberately: `calc_num_passes`, `set_kth_key_bits`, `Counter`
(exp's headers do not contain them either).

Line deltas: `agent_topk.cuh` -66, `dispatch_topk.cuh` -125. Verified each
moved symbol now has **exactly one** definition site under `cub/cub/`.

**Verification - D7 risk partially retired.** D7 accepted the risk that a
`native`-only build would not compile-check the extraction against
`agent_batched_topk_cluster.cuh` (its riskiest consumer, only instantiated when
the selector returns `cluster`, i.e. SM90+). I closed that cheaply with a
single-TU probe instead of a full build:

* `nvcc -std=c++17 -arch=sm_70` -> clean
* `nvcc -std=c++17 -arch=sm_90` -> clean
* confirmed the cluster arm was *actually emitted*, not `if constexpr`-ed away:
  the sm_90 object is 371,944 B vs 182,680 B for sm_70, `cuobjdump -symbols`
  shows cluster symbols and `device_batched_topk_kernel`.

Probe used `cuda::args::constant<1<<14>` (16384): `baseline_can_cover` is true
(largest worker tile is exactly 16384) *and* `16384 >= cluster_beneficial_min_
segment_size`, so SM90 routes to cluster - exactly the arm needed.

So single-TU `-arch=sm_90` probes are a cheap way to keep SM90 compile coverage
throughout the port despite the native-only build decision. Will reuse per
chunk.

**API-surface finding (not in the brief).** The public entry rejects an empty
environment: `DeviceBatchedTopK::MaxKeys` requires an explicit
`cuda::execution::require(determinism::not_guaranteed, tie_break::unspecified,
output_ordering::unsorted)` because the *default* output ordering is
`stable_sorted`, which is unimplemented. Relevant when writing the new tests
and benchmarks in chunk 7.

Commit: `391d2beaf9`. Diffstat matched exp's exactly on 4/5 files; the 5th
(`dispatch_topk_identify_candidates.cuh`) is 158 vs exp's 160 lines - verified
by `diff` to be **only** two blank lines directly inside the namespace braces,
normalized to main's prevailing style. `candidate_class.cuh` is byte-identical
to exp; `key_prefix_storage.cuh` differs by the same two cosmetic blank lines.

### Chunk 2 - additive building blocks (DONE)

Added verbatim from exp (all compile clean, `-arch=sm_70`, against main's
headers - consistent with the earlier finding that these reference **zero**
`params::` symbols, so the `::cuda::args` migration does not touch them):

`cub/detail/topk/{empty_storage,partition_storage_layout,tile_data_source,`
`block_partition,block_filter}.cuh`, `cub/agent/agent_topk_common.cuh`,
`cub/device/dispatch/dispatch_topk_common.cuh`,
`cub/test/catch2_test_device_topk_tile_data_source.cu`.

#### DEVIATION D10 - `make_warp_uniform.cuh` excluded from the port

The brief listed this as a 17-line "near-clean reapply". Inspection shows it is
not additive at all: exp's pruning commit `7a8c002756` **deletes** two
overloads, `makeWarpUniform(int)` and `makeWarpUniform(::cuda::std::uint64_t)`,
keeping only the `uint32_t` one.

Verified on main: the only caller anywhere in `cub/ thrust/ cudax/` is
`special_registers.cuh:39`, which uses the `uint32_t` overload. So the deletion
is *harmless* - but it is also completely unrelated to the multi-CTA feature
(the ported agents only ever call the `uint32_t` overload, which main already
has). Removing two overloads of a `detail::` utility inside a feature PR is
gratuitous API churn that a reviewer would rightly flag.

=> Reverted this file to main's version. **The port does not include it.**
If the cleanup is wanted it belongs in its own commit.

#### `C2H_TEST` -> `CUB_TEST` is not a rename (main-side convention change)

`C2H_TEST` still exists on main (`c2h/catch2_test_helper.h:622`) but is legacy:
3 files still use it vs 236 using `CUB_TEST`. And `CUB_TEST` is **not** a
drop-in - `cub/test/cub_test_macros.h:12` defines
`CUB_TEST(NAME, TAGS, MEMORY, ...)` with a *mandatory* third argument,
`CUB_SMALL` or `CUB_LARGE` ("Small tests may share a GPU; large tests run
alone. Use CUB_LARGE when unsure."). This convention postdates the exp branch
entirely.

Converted all 7 test cases in `catch2_test_device_topk_tile_data_source.cu` to
`CUB_TEST(..., CUB_SMALL)` and added `#include "cub_test_macros.h"`.
`CUB_SMALL` is correct here: these are foundation unit tests on
64 threads x 4 items. The same conversion is required for the chunk 7 test
cases.

#### ISSUE C2-1 - `using namespace detail::topk;` in a public-ish header

`agent_topk_common.cuh:52` has `using namespace detail::topk;` inside
`namespace detail::batched_topk`. It injects all of `detail::topk` into
`detail::batched_topk` for **every TU** that includes the header. It compiles,
and the comment explains the intent ("bring them into scope so the batched
shared-layer symbols below can refer to them unqualified"), but it is the kind
of thing CCCL style review rejects.
Kept as-is (removing it means re-qualifying names across 626 lines of agent
code plus `block_partition` / `block_filter`, which I have not yet validated).
**Recommended follow-up cleanup.**

#### ISSUE C2-2 - two shadowing duplicate helpers

Consequence of C2-1: two helpers now exist in both namespaces.

| Helper | `detail::topk` (main) | `detail::batched_topk` (new) | Verdict |
| --- | --- | --- | --- |
| `calc_num_passes<int BitsPerPass>(int total_bits)` | `agent_topk.cuh:71` | `agent_topk_common.cuh:94` | **byte-identical duplicate**; the inner one shadows the outer for unqualified lookup from `detail::batched_topk` |
| `set_kth_key_bits` | `agent_topk.cuh:78`, params `<KeyT, BitsPerPass>` | `agent_topk_common.cuh:104`, params **`<BitsPerPass, KeyT>`** | same body, *swapped* template-parameter order |

Both must keep compiling: `detail::topk::set_kth_key_bits` is still used by
`agent_topk.cuh:555` and `agent_batched_topk_cluster.cuh:3093` (the latter
fully qualified, so no ambiguity there). The reordering is deliberate and
documented (only the non-deducible constant needs spelling) - but two
same-named functions with the same body and swapped template parameters, both
reachable unqualified inside `detail::batched_topk`, is a real footgun:
`set_kth_key_bits<8>(...)` and `set_kth_key_bits<KeyT, 8>(...)` silently pick
different functions.

The `calc_num_passes` duplicate is a safe deletion in isolation, **but** exp's
`dispatch_batched_topk.cuh` calls it fully qualified as
`detail::batched_topk::calc_num_passes<bits_per_pass>(total_bits)`, so deleting
it now creates a cross-chunk dependency on a chunk-6 edit. Kept for a faithful,
reviewable port; **recommended follow-up cleanup** (delete the duplicate,
re-point the one qualified call at `detail::topk::`).

#### Chunk 2 commit

`d86c884852`. All 7 building blocks match exp's line counts exactly (626, 329,
516, 76, 89, 718, 123). The test is 524 vs exp's 514: +10 from `CUB_TEST`
argument line-wrapping, +1 include.

### Chunk 3 - tuning policy (DONE)

First chunk requiring a real semantic merge rather than a copy.

Changes to `tuning_batched_topk.cuh`:

* new `value_materialization_mode` enum (`indexed` / `materialized`);
* `multi_worker_policy` grows **2 -> 11** fields, rewritten in main's `//!<`
  Doxygen style rather than exp's `//` block comments (AGENTS.md consistency
  requirement) with the explanatory content preserved;
* new `make_multi_worker_policy(int key_size, ::cuda::compute_capability cc)`;
* `make_baseline_policy` now delegates its multi-worker field to it.

#### DESIGN - resolving the CC-independence invariant

exp computed `items_per_thread` / `bits_per_pass` from *both* `key_size` and
`cc`. main's `make_baseline_policy()` is nullary and CC-independent, and
`policy_selector_from_types` carries a documented invariant plus a
`TODO(bgruber)` explaining that a per-CC baseline policy is hard because
`baseline_can_cover_v` instantiates the agent for `sizeof(TempStorage)` and so
needs the CC as a compile-time constant.

Facts established before touching it:

1. `make_baseline_policy()` has **7 external call sites** (2 compile-fail tests,
   `catch2_test_device_topk_common.cuh` x2, 3 benchmarks) - changing the
   signature would ripple.
2. The worker agent references `multi_worker_per_segment_policy` at only
   `agent_batched_topk.cuh:93-95` and `:215`, purely to compute
   `multi_worker_per_segment_tile_size` for the enqueued tile count. It does
   **not** appear in `TempStorage_`, which depends only on `active_policy`
   (worker) and `epilogue`. main even annotates the field "Number used for
   preprocessing segment-size data, not for tuning".

=> Resolution: give `make_baseline_policy` two **defaulted** parameters
(`key_size = sizeof(uint32_t)`, `cc = {}`). All 7 call sites keep compiling
unchanged and `baseline_can_cover_v` keeps using the CC-independent form, while
`policy_selector_from_types::operator()(cc)` calls
`make_baseline_policy(int{sizeof(KeyT)}, cc)` so host and device resolve the
*same* multi-worker tuning per CC. This matters for correctness, not just
performance: the worker agent's enqueued per-segment tile counts must be
computed with the same tile size the multi-CTA kernels later grid-stride over.

The invariant comment in the selector was rewritten to state precisely what
still holds (the `worker_per_segment_policies` agree; the multi-worker
sub-policy intentionally differs and provably cannot shift coverage).

Default CC is `::cuda::compute_capability{}` == 0.0, which takes the pre-SM90
branch - conservative, and only reached by call sites that ignore the field.

#### ADDITION beyond exp - `is_valid_multi_worker_policy`

exp had no validator for the multi-worker knobs. Added one mirroring main's
existing `is_valid_cluster_policy`, plus a `static_assert` on the default. It
guards the values that would fail confusingly rather than loudly under a `tune`
override: `tiles_per_chunk == 0` (zero-length chunk loop in every multi-CTA
kernel), `bits_per_pass == 0` (zero-bucket histogram), a
non-warp-multiple `threads_per_block`, and non-positive `items_per_thread`
(divide-by-zero in the dispatch's tile math).

#### Minor deviation from exp - policy streaming

exp printed `value_materialization` and `keys_tile_load_kind` as
`static_cast<int>`. Added a proper `operator<<` for `value_materialization_mode`
(matching main's `topk_algorithm` printer). `keys_tile_load_kind` still prints
numerically, with an in-code note that its printer belongs beside the enum in
`cub/detail/topk/tile_data_source.cuh`, which has no host-only streaming
section today. Cosmetic follow-up.

#### Verification

| Check | Result |
| --- | --- |
| `-arch=sm_70` full dispatch instantiation | clean |
| `-arch=sm_90` (cluster arm) | clean |
| `-DCUB_DEBUG_LOG` (instantiates every policy `operator<<`) | clean |
| all 7 `make_baseline_policy()` call-site *shapes* replicated in a probe | clean |
| `baseline_max_covered_segment_size(make_baseline_policy()) == 16384` | holds - worker array provably unchanged |

### Chunk 4 - multi-CTA agents (DONE, `075da6a681`)

Spliced exp lines 389-2132 (1744 lines: `agent_batched_topk_histogram`,
`..._filter_partition`, `..._last_filter`) into main's file between the worker
agent and the namespace close. Result 2132 lines. `batched_topk_counters` needed
**no** change - identical between branches apart from comments.

**The translation surface was far smaller than feared: 8 sites total.** A grep
for `params::*` / `::value_type` / `static_*_value_v` / `narrow_segment_count_t`
across all 1746 new lines returned only two API families. Everything else is
params-framework-agnostic.

#### D11 - segment-size and k reads must use the *clamping* accessor

exp wrote `static_cast<OffsetT>(segment_sizes.get_param(segment_id))` (3 sites)
and the same for `k_param` (1 site). Translated to
`params::__get_and_clamp_param_to_nonnegative(...)`.

This is a **correctness** change, not cosmetics. Three independent reasons:

1. main's worker agent already uses the clamping form for both
   (`agent_batched_topk.cuh:203` and `:224`).
2. The helper's own docstring states its purpose: it clamps "*before* any
   widening/narrowing cast, so a caller that later widens the result cannot
   reinterpret a negative value as a huge unsigned one." exp's code performs
   exactly that widening - `OffsetT` is `uint32_t` / `unsigned long long`.
   A negative segment size would have become a ~4-billion item count.
3. main's *producer* computes per-segment tile counts through
   `segment_size_to_tile_count_op`, which clamps. Non-clamping consumers would
   disagree with the producer's tile offsets for negative sizes.

Negative segment sizes are legal public input on main (documented: "the kernel
clamps any negative runtime size up to 0"), so this is reachable, not
theoretical.

#### D12 - queue index type: follow main's producer, drop exp's narrowing

exp had `narrow_segment_count_t<NumSegmentsParameterT>` - `uint32_t` when the
count provably fit, else 64-bit - explicitly "so `resolve_queue_idx`'s
`UpperBound` + offset-table indexing stay 32-bit". Used at 4 agent sites
(`num_large_segments` member + ctor parameter in the filter and last_filter
agents) and 12 kernel sites.

Not portable as-is, and the reason is decisive rather than stylistic: main's
**producer** side already fixes these types. `baseline_kernel_args::
d_large_segments_ids` and `batched_topk_counters` are typed on
`::cuda::args::__traits<NumSegmentsParameterT>::element_type`. Narrowing only
the consumers would mistype the very queue the worker agent writes. The options
were therefore (a) narrow consumers only = **a bug**, (b) follow main's
producer, (c) also re-type main's worker agent + kernel ABI. (c) modifies
main's tested producer for a pure optimization, which is out of scope for a port
whose premise is "no change to paths main already has".

=> Introduced `queue_segment_count_t<NumSegmentsParameterT>` =
`detail::choose_offset_t<__traits<...>::element_type>`. Deliberately the same
type the worker's enqueue produces (`batched_topk_counters::segment_count_t`),
and unsigned so the `UpperBound` bound and offset-table indexing need no sign
handling.

**Follow-up available, and it is unconditionally safe:** main's public entry
already rejects `num_segments > INT_MAX`
(`dispatch_batched_topk.cuh:1042`, returns `cudaErrorInvalidValue`), so the
count *always* fits 32 bits dynamically. A future commit can narrow
`queue_segment_count_t` to plain `uint32_t` across producer *and* consumer
uniformly, recovering exp's optimization without any conditional logic.

### Chunk 5 - multi-CTA kernels + policy-resolution relaxation (DONE, `06c87346c1`)

Spliced exp kernel lines 263-989 (727 lines, 5 kernel symbols). Translations:

| exp | main | sites |
| --- | --- | --- |
| `batched_topk_policy bp = current_policy<...>()` | `topk_policy bp`, field via `bp.baseline.` | 4 |
| `current_policy<...>().multi_worker_per_segment_policy` | `...().baseline.multi_worker_per_segment_policy` | 8 |
| `batched_topk_policy_selector` | `topk_policy_selector` | 5 |
| `narrow_segment_count_t` | `queue_segment_count_t` (D12) | 12 |
| `.get_param(` on sizes / k | clamping accessor (D11) | 3 |
| `device_segmented_topk_*_kernel` | `device_batched_topk_*_kernel` | 12 |

Verified before renaming that all five kernel names are mutually
non-substring, so `replace_all` could not corrupt them.

#### D13 - kernel naming

Renamed exp's `device_segmented_topk_*_kernel` to `device_batched_topk_*_kernel`.
main renamed the worker kernel the same way since the merge-base
(`device_segmented_topk_kernel` -> `device_batched_topk_kernel`); the file is
`kernel_batched_topk.cuh` and the algorithm is `DeviceBatchedTopK`. Keeping
exp's names would have left two naming schemes in one file.

#### D14 - relaxing worker-policy resolution (the change that makes the path reachable)

This was the one structural gap, and it is easy to miss: main's
`find_smallest_covering_policy_for_getter` carries
`static_assert(selected_index >= 0, "no baseline worker policy covers ...")`.
The multi-CTA path exists precisely for batches where *nothing* covers, so
without this change the baseline arm would fail to compile the moment the
selector routed a large batch to it - the D4 selector change alone would not
have worked.

Added, mirroring main's existing structure rather than importing exp's:

* `find_fitting_policy_index` - index of the **largest** shared-memory fitting
  worker policy (the array is ordered by decreasing tile size, so the first fit
  is the largest). Largest matters: it keeps as many segments as possible on the
  cheap single-CTA path instead of escalating them.
* `worker_policy_index_v` - covering index when one exists, else the fitting
  index. Host and device must agree here, since the worker computes each
  escalated segment's tile count from the resolved multi-worker tile size and
  the multi-CTA kernels grid-stride over exactly those tiles.
* The surviving `static_assert` now fires only when **no** policy fits shared
  memory - a genuine hard error (the smallest policy is 128x2 keys, so it
  requires a key/value type whose block primitives cannot fit even that).

Renamed `find_smallest_covering_policy_{for_getter,device}` ->
`resolve_worker_policy_{for_getter,device}` (8 + 1 call sites) because the
traits no longer always return the smallest *covering* policy. `baseline_can_
cover_v` is retained unchanged - the selector still needs it to distinguish
"worker path alone suffices" from "escalation required".

#### D15 - duplication policy: KEEP (user decision)

A third, larger duplication surfaced in chunk 2's `dispatch_topk_common.cuh`:
it defines its own `calc_mask` and `extract_bin_op_t` in
`detail::batched_topk`, duplicating `detail::topk`'s in `dispatch_topk.cuh`.
Diffing them showed the copies have **already diverged** - exp's pruning
stripped `pass`, `is_descending` and `bit_ordered_type` from the batched copy -
though the `operator()` bodies are still identical.

Full inventory of `detail::topk` <-> `detail::batched_topk` overlap:

| Symbol | Status | Size |
| --- | --- | --- |
| `extract_bin_op_t` | already diverged (batched lost 3 members) | ~70 ln |
| `calc_mask` | duplicate | 6 ln |
| `calc_num_passes<int>(int)` | byte-identical duplicate | 5 ln |
| `set_kth_key_bits` | same body, swapped template params | 16 ln |

Decision (user): **keep the duplication permanently.** The technical
justification, verified: the batched `extract_bin_op_t<...,true>` carries 2
members (`start_bit`, `mask`) against `detail::topk`'s 3, and the functor is
constructed host-side and passed **by value into the kernel parameter area** of
the histogram / finalize kernels. Dropping the unused `pass` is a deliberate
minimization on a hot path, and the two copies serve different consumers with
different evolution pressure (single-problem + cluster agent vs the multi-CTA
kernels). The divergence is intentional, not rot.

#### Verification of chunks 4+5 (done together - see below for why)

Templates are not type-checked until instantiated, so a clean parse of a header
full of agent templates proves almost nothing. Chunk 4 was therefore verified
*with* chunk 5, whose kernels are the instantiation harness the agents were
designed for (user-approved). Method: a probe taking the **address** of each of
the five kernel templates, which ODR-uses the `__global__` template and forces
its body - and transitively all three agents - to be fully instantiated.

| Check | Result |
| --- | --- |
| 5 kernels instantiated, `-arch=sm_70`, keys-only | clean |
| ... and all 5 present as device symbols in the object (`cuobjdump -symbols`) | confirmed - proves real instantiation, not elision |
| 5 kernels instantiated, keys+**values** (`float**`) | clean - exercises the value channel that keys-only `if constexpr`-eliminates |
| 5 kernels instantiated, `-arch=sm_90` | clean |
| full dispatch incl. cluster arm, `-arch=sm_90` | clean - cluster path still intact after the resolver rename |

### Chunk 6 - dispatch wiring + selector + cap (DONE)

The largest chunk, and the one that surfaced the most. Contents:

* new `launch_multi_cta_passes` host arm: per-segment slab allocation, two
  memsets, and the radix-pass launch sequence (histogram -> finalize,
  then [filter -> finalize] per pass, then last-filter), **all through
  `launcher_factory`** (`MemsetAsync`, `MaxSmOccupancy`, `MultiProcessorCount`,
  `operator()(...).doit(...)`) rather than exp's direct `triple_chevron` /
  `cudaMemsetAsync` / `cudaDeviceGetAttribute` calls. Verified the factory
  covers every call needed - no capability gap.
* `per_segment_indexed_out_op` + the indexed/materialized value-channel rewiring.
* allocation layout extended from 3 to 3+4 (keys-only) or 3+6 slots.
* selector change (D4) and the new crossover constant (D5).
* entry cap relaxation (D6).

#### Simplification vs exp: no runtime direction lowering

exp lowered a *runtime* selection direction with a hand-rolled host-side
`launch_passes(integral_constant<...>)` branch over both enumerators. On main
the direction is compile-time only (`wrap_select_direction` accepts only
`cuda::args::constant<Dir>` and maps it to a value-less
`static_discrete_param` whose `get_param` is `constexpr`), so the whole branch
collapses to

```cpp
static constexpr detail::topk::select select_dir =
  ::cuda::std::remove_cv_t<SelectDirectionParameterT>{}.get_param(0);
```

That removes exp's duplicated 300-line launch body (one instantiation per
direction) from the host side.

#### D16 - `total_num_items_guarantee` not threaded into the arm

exp sized `candidate_buffer_length` from
`min(static_max_segment_size, total_num_items_guarantee.max_num_items)`. main's
`launch_baseline_arm` never receives that argument (only
`LargeSegmentTileOffsetT`, derived from its element type), so the port sizes the
buffer from `__traits<SegmentSizeParameterT>::highest` alone. Correct but
looser: a caller declaring a large per-segment bound *and* a small total bound
gets a bigger buffer than exp would have allocated. Recorded as a TODO in the
code next to the existing flat-cap TODO. Not threaded in because it would
change the arm's signature for a degenerate-case optimization.

#### ADDITION beyond exp - size_t overflow guard

main's own WIP TODO asked for this ("guarding the `num_segments_val *
sizeof(...)` byte counts against size_t overflow, safe today only because the
entry bounds num_segments_val to <= INT_MAX"). exp had the same unguarded
multiply. Added a per-segment-bytes product check that returns
`cudaErrorInvalidValue` rather than wrapping into a too-small allocation. The
entry's INT_MAX bound alone is *not* sufficient here because the candidate
buffers also scale with the segment-size bound, so the product is not bounded by
the segment count.

#### The four errors the first real instantiation surfaced

Everything before this chunk compiled while the multi-CTA arm was still
unreachable. Making it reachable exposed four distinct problems - worth
recording because three are pre-existing defects on main, not port artifacts.

**(1) Host-side coverage gate.** `launch_baseline_arm` opened with
`if constexpr (!baseline_can_cover) { ... return cudaErrorNotSupported; } else`,
plus a `static_assert(baseline_can_cover, "...forced baseline backend cannot
cover...")`. That gate replaced the *entire* arm body whenever no worker policy
covered the bound - i.e. exactly the case the multi-CTA path exists for. My
first probe compiled only because it defined
`CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT`, which masked the assert. Removed
both; the surviving hard requirement (some policy fits shared memory) is
asserted by `resolve_worker_policy_for_getter` (D14).

**(2) Device-side twin of the same gate.**
`kernel_batched_topk.cuh:404`,
`static_assert(agent_t::tile_size >= __traits<SegmentSizeParameterT>::highest,
"Block size exceeds maximum segment size supported by SegmentSizeParameterT")`
inside `device_batched_topk_kernel`'s baseline branch. Same reasoning, removed.

**(3) PRE-EXISTING BUG on main: `const` mutable lambda.** The worker epilogue
declared `const auto prefix_callback_op = [...](...) mutable {...}` and passes it
to `BlockScan::ExclusiveSum`, which takes the callback by **non-const
reference** and invokes it. Calling a `mutable` lambda through a const object is
ill-formed, so this could never have compiled - it simply was never
instantiated, because no test on main reaches a `!only_small_segments` baseline
configuration. exp had it as non-const `auto`, which is the working form.
Fixed to `auto` with a comment naming the reason.

**(4) PRE-EXISTING BUG on main: epilogue `BlockScan` hardcoded to `int`.**
`block_scan_epilogue_t = BlockScan<int, ...>` while the matching
`block_load_epilogue_t` / `block_store_epilogue_t` and the thread-local array are
all `segment_size_val_t`. Breaks for any segment-size type wider than `int`;
latent for the same reason as (3). exp had `BlockScan<segment_size_val_t, ...>`.
Fixed to match, with a comment that a tile count is bounded by its segment size
so the wider type is always sufficient.

Together (3) and (4) are good evidence that main's baseline large-segment path
had never been compiled, consistent with its "WIP" TODO.

#### CORRECTION to D12 - the narrowing *is* portable, and was needed

D12 concluded that exp's 32-bit narrowing of the queue count could not be
reproduced because main's producer types the queue on the segment-count
parameter's `element_type`. **That reasoning conflated two distinct types** and
is hereby corrected:

* the queue's **element array** (`baseline_kernel_args::d_large_segments_ids`)
  and `batched_topk_counters` - these are main's kernel ABI and must stay as
  main has them. D12's argument applies here, and only here.
* the queue's **length / index** type - a pure local value type. Narrowing it
  changes no ABI.

The distinction became unavoidable rather than academic: with
`queue_segment_count_t` derived from a 64-bit `element_type`,
`resolve_queue_idx`'s `makeWarpUniform(__shfl_sync(...))` call became
**ambiguous** across the three `makeWarpUniform` overloads (`int`,
`uint32_t`, `uint64_t`) for a `long` argument.

Resolved by making `queue_segment_count_t` an unconditional `uint32_t`, which
(a) recovers exp's optimization - the `UpperBound` search bound and offset-table
indexing stay 32-bit in the per-tile inner loop of every multi-CTA kernel,
(b) resolves the ambiguity naturally, and (c) is *provably* safe: main's entry
rejects `num_segments > INT_MAX` with `cudaErrorInvalidValue`, so the queue can
never be longer than that. It also became a plain alias rather than a template
(14 call sites updated), since it no longer depends on the parameter type.

The `resolve_queue_idx` sites additionally narrow explicitly at the call, with a
comment, because main derives `LargeSegmentTileOffsetT` from the total-items
guarantee (it was `long` in the probe) whereas exp pinned it to `uint32_t`. That
keeps `makeWarpUniform`'s 32-bit overload selected - the one that lowers to a
single CREDUX - without touching main's `baseline_kernel_args` ABI.

**This also retires D10's residual risk.** exp's `make_warp_uniform.cuh`
deletion (which I excluded as unrelated cleanup) turns out to have been exp's
way of dodging this same ambiguity. Narrowing at the call site addresses the
cause instead, so excluding the deletion remains correct *and* the ambiguity is
gone. main keeps all three overloads.

#### Producer/consumer contract fix: the sentinel slot

The multi-CTA kernels read `total_large_tiles` from
`d_large_segments_tile_offsets[num_large_segments]`, but main's worker epilogue
scanned exactly `num_large_segments` entries in place and never wrote that slot.
exp's epilogue loops **one item past** the count, with `BlockLoad`'s
`valid_items` still capped at `num_large_segments` so the out-of-bounds item
loads as the default `0` and only the `BlockStore` extends to the sentinel.
Ported that, widened the allocation to `N + 1`, and gave the all-large path the
matching behaviour by adding a sentinel short-circuit to
`segment_size_to_tile_count_op` and scanning `num_segments + 1` inputs. Both
producers now publish the total identically.

#### D4 / D5 / D6 as implemented

Selector, in precedence order (`dispatch_batched_topk.cuh`):

```
if (deterministic)                                    -> cluster if cluster_capable else unsupported
else if (StaticMaxSegSize > cluster_max_competitive)   -> baseline (multi-CTA), every arch
else if (!baseline_can_cover)                          -> cluster if cluster_capable else baseline
else                                                   -> cluster if (cluster_capable && beneficial) else baseline
```

New constants in `tuning_batched_topk.cuh`:
`cluster_max_competitive_segment_size = 1 << 21` (D5, paired with the existing
`cluster_beneficial_min_segment_size = 8 * 1024`, both with a non-empty-band
static_assert) and `multi_cta_max_supported_segment_size = 1 << 32`.

Entry cap (D6), now determinism-dependent: non-deterministic requests are
bounded by `multi_cta_max_supported_segment_size`, deterministic ones stay at
`cluster_max_competitive_segment_size`. **The 2^32 figure is a judgement call
flagged to the user**: it is a representability bound, and the true aggregate
constraint (total tiles across all segments fitting the tile-offset type) is a
runtime quantity that no static cap can express. In practice the dispatch's
temporary-storage guard rejects the batches that would overflow, since the
per-segment candidate buffers scale with this same bound.

#### Verification

| Check | Result |
| --- | --- |
| selector routing, compile-time `static_assert`s: `!baseline_can_cover`; sm70 large -> baseline; sm90 in-band -> cluster (unchanged) | all hold |
| sm70 / sm90 >2^21 -> baseline on **both** archs | holds |
| multi-CTA arm instantiated: all 5 kernels present twice (keys-only + pairs) in one TU; `launch_multi_cta_passes` in host symbols; object 209 KB -> 994 KB | confirmed |
| D2: no `_CCCL_GRID_CONSTANT` anywhere on the batched top-k path | confirmed |
| full probe matrix (5 probes x sm_70/sm_90) | 10/10 clean |

### Chunk 7 - tests, benchmarks, and the first behavioural evidence (DONE, `6b4ecdb623`)

#### D17 - exp's test additions were superseded, not ported

exp's three keys cases (+259 lines) and pairs cases (+180) are written against
exp's **removed** dispatch API: `segment_size_uniform<...>{}`,
`k_uniform<...>{}`, `num_segments_uniform<>{}`, and - fatally -
`select_direction_uniform{direction}`, a *runtime* selection direction that main
does not support at all (main accepts only `cuda::args::constant<Dir>`). They
also predate the `CUB_TEST(..., CUB_SMALL|CUB_LARGE)` convention. Porting the
text was not an option; they had to be rewritten or replaced.

#### The find that reshaped this chunk: main's existing tests already cover the path

`catch2_test_device_topk_common.cuh::batched_topk_backend_unavailable` decided a
request "needs the cluster backend" if `deterministic || oversize`, where
`oversize` meant "larger than `baseline_max_covered_segment_size`" (16384). On a
pre-SM90 device every large-segment case therefore asserted
`cudaErrorNotSupported` and **skipped**.

After the port only `deterministic` requires the cluster backend. Changing that
one predicate turned main's *existing, already-reviewed* large-segment tests into
live coverage of the new path:

* keys: large fixed-size unaligned (1 Mi x 3, with `-31` / `-4095` tails),
  large variable-size unaligned (a genuinely **mixed** batch - 1 Mi, 1 Mi-31 and
  96 Ki+17 segments alongside 257 and 12 Ki+1 ones, so both the worker and the
  multi-CTA arms run in a single dispatch), signed-32-bit segment size at 1 Mi,
  and a non-contiguous key iterator;
* pairs: four equivalents, which additionally exercise the value channel and the
  `indexed` materialization mode wired in chunk 6.

That is far better than new tests: it is existing correctness checking, applied
to new code.

=> Only two genuinely-uncovered behaviours needed new cases, both added:
1. **segments past 2^21** - unreachable before (rejected at the entry), and the
   only test of the D6 relaxation. Sweeps an exact tile multiple and a trailing
   partial tile, since the partial is handled by the finalize kernels under the
   default `full_tiles_only_*` tuning.
2. **boundary large-segment counts 0 and 1** - pins the sentinel-slot epilogue
   logic, which is exactly where chunk 6 found three defects.

#### D18 - `test_device_batched_topk_forced_baseline_oversize_fail.cu` repointed

Its premise (forcing baseline on an oversize segment is a compile error) is now
false by design. Repointed at what *is* still an error: a segment past the
cluster backend's range **plus** a determinism requirement. Verified it produces
exactly one error, the intended one.

#### PRE-EXISTING BUG on main (5th): spurious `static_assert` kills strict-mode determinism

Repointing that test surfaced a second error, and chasing it found a real defect.
`dispatch_batched_topk.cuh`'s guard against a tuned baseline-for-deterministic
override was written as `static_assert(!deterministic, ...)` inside
`if constexpr (active_policy.backend == baseline) { if constexpr (deterministic)`.
`deterministic` derives only from `dispatch`'s own template parameters, which are
already fixed when the lambda's `operator()` is instantiated - so the condition is
**non-dependent on the lambda's parameter** and is evaluated even when this is the
*discarded* arm. Result: it fires for every deterministic request regardless of
the backend selected.

Confirmed pre-existing by building the identical TU against an unmodified `main`
worktree: both produce it. Invisible in CI only because every CUB top-k test
defines `CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT`, which `#if`-compiles the
assert out - so the *documented strict-mode deterministic path does not compile
on main today*. Note the neighbouring dynamic-cluster guard was written
correctly, with an `active_policy`-dependent condition.

Fixed by making the condition depend on `active_policy`
(`active_policy.backend != topk_algorithm::baseline || !deterministic`).
Verified: a deterministic 2^20 request in strict mode now compiles.

#### Benchmarks - main's TODO item 2

All three tuned-baseline sites (`fixed/keys.cu`, `variable/keys_common.cuh`,
`variable/indexed_common.cuh`) brace-initialized the whole `baseline_topk_policy`
with only the worker array, leaving every `worker_policy::epilogue` **and** the
`multi_worker_per_segment_policy` zero-initialized - a one-bucket histogram
(`bits_per_pass == 0`) and a zero-length chunk loop (`tiles_per_chunk == 0`),
plus a zero-item epilogue. This is precisely main's own TODO ("...leave them
zero-initialized today so baseline sweeps are not yet meaningful").
Changed to start from `make_baseline_policy(sizeof(KeyT))` and override only the
swept worker knobs, rather than inventing 11 untested tuning axes.

#### ISSUE E3 - the preset's CUDA architecture silently overrides `--cmake-options`

The first full test run was **invalid** and it took a while to notice. Passing
`--cmake-options "-DCMAKE_CUDA_ARCHITECTURES=native"` to
`ci/util/build_and_test_targets.sh` left the cache showing
`CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=native` while the actual gencode flags
were `compute_89` and `compute_90`, **virtual-only and both above the V100's
SM 7.0**. So no SASS existed for the device and no PTX could be JIT'd down to it.

The symptom was subtle rather than a hard failure: most cases still "passed"
(they take host-only / empty-batch paths that never launch), and only 4 failed -
with `cudaErrorInvalidDeviceFunction` (98), which is exactly what
`dispatch_compute_cap` returns when the runtime CC matches none of
`__target_compute_capabilities()`.

Fix: reconfigure the existing tree directly with
`cmake -S . -B build/cub-cpp17 -DCMAKE_CUDA_ARCHITECTURES=70`, which yielded
`compute_70` **with `sm_70` SASS** (alongside the virtual 89/90, which usefully
keeps the cluster arm under real compile coverage). Also note the target names
carry no `cpp17` infix and do carry the `%PARAM%` suffixes:
`cub.test.device.segmented_topk_keys.lid_0.types_0`.

**Lesson for anyone repeating this: verify the gencode flags, not the cache
variable.**

#### PRE-EXISTING FAILURE on main (not caused by the port)

4 cases / 64 assertions fail in `keys.lid_0.types_0`:
"clamp k larger than the segment size" for `uint8_t` and `half_t` at
`segment_size = 384`, `k_requested = 385`, with
`cudaErrorInvalidDeviceFunction`. Verified identical on an unmodified `main`
worktree (same 4 cases, same 64 assertions), so it is **pre-existing and
unrelated**. Not investigated further; flagged for the user.

#### Behavioural results (SM 7.0, `cub-cpp17`, arch `70;89-virtual;90-virtual`)

Each variant run on the port and on an unmodified `main` worktree for comparison:

| Variant | main passed | port passed | newly live | failures |
| --- | --- | --- | --- | --- |
| `keys.lid_0.types_0` | 52 (of 71) | **55** (of 73) | +3 = 2 new tests + 1 previously-skipped | 4, identical to main |
| `keys.lid_0.types_1` | 22 (of 52) | **26** (of 52) | +4 large-segment cases | **0** |
| `pairs.lid_0.types_1` | 16 (of 88) | **22** (of 88) | +6 pairs cases (value channel + indexed mode) | **0** |

Net: **13 more passing cases, ~1135 more assertions, zero new failures**, and
the only failures present are main's pre-existing ones. Both new tests pass
(`[multi-cta]` tag: 2 cases, 64 assertions). The five multi-CTA kernels are
present in the test binary (`cuobjdump -symbols`), confirming the path is the one
being exercised rather than eliminated.

This is the first evidence in the whole port that is behavioural rather than
compile-time, and it validates the multi-CTA path against correctness checks that
already existed and were reviewed for the cluster backend.

#### Naming clarification (resolves an earlier worry)

`agent_topk_common.cuh:82` defines its **own** `AgentTopKPolicy` in
`detail::batched_topk`, with a 5th parameter `tile_load_kind KeysTileLoadKind`.
It does **not** collide with main's `detail::topk::agent_topk_policy`
(`agent_topk.cuh:54`, renamed from `AgentTopKPolicy` since the merge-base).
No rename needed; exp's `detail::batched_topk::AgentTopKPolicy<...>` references
in the kernel layer resolve correctly as-is.

---

## 10. PDL experiment (branch `exp/topk-multi-cta-pdl`)

### Measured model of the gap

The batched path issues **9 device ops** per call (2 memsets + histogram +
finalize + [filter + finalize] x (passes-1) + last_filter) against the
single-problem path's **5** (memset + histogram + topk x (passes-1) +
last_filter). For F32, `bits_per_pass = 11` -> 3 passes.

9/5 = 1.80 against a measured 1.69-1.82 slowdown at 2^16, where every kernel is
too small to do meaningful work. The gap is also near-constant in absolute terms
(~18 us at 2^16, ~124 us at 2^28) while the work grows 4096x. Conclusion: at
small sizes the gap is **device-operation count**, not per-kernel efficiency.

### Resources (measured from the SM100 binary, keys-only F32)

| Kernel | regs | smem | blocks/SM | grid |
| --- | --- | --- | --- | --- |
| histogram / filter / last_filter | 32-40 | ~9.2 KiB | 4 (register-limited) | 592 |
| finalize_histogram / finalize_filter | 40-50 | ~9.2 KiB | 3 | **1** (one CTA per segment) |

The pipeline alternates a 592-CTA kernel with a **1-CTA** kernel, i.e. 147 of 148
SMs idle during every finalize.

### Increment 1 - fold the two memsets into one PDL-capable init kernel (`7bc0c2c4a4`)

Also added the late `_CCCL_PDL_GRID_DEPENDENCY_SYNC()` in the histogram agent's
`flush_active_segment()` (the only point it touches the global slabs), mirroring
`agent_radix_sort_histogram`.

Result across 42 configs, batched path: **median -4.03 us, mean -4.01 us**
(min -3.52, max -4.80). Single path control: median -0.01 us.

**Interpretation: the entire saving is the removed device operation, not PDL.**
One device op costs ~4 us on this system, uniformly and independently of problem
size. The PDL overlap contributed nothing measurable, because the init kernel is
~2 us of work - there is nothing long enough to hide behind it.

### What this implies for the remaining PDL work

PDL's benefit per transition is bounded by
`min(launch latency ~4 us, primary kernel duration)`. Applying that:

| Transition | Primary duration | PDL ceiling |
| --- | --- | --- |
| init -> histogram | ~2 us | ~0 (**measured**) |
| histogram -> finalize_hist | long at >=2^20 | ~4 us |
| filter -> finalize_filter (x2) | long at >=2^20 | ~8 us |
| finalize_* -> filter / last_filter | 1-CTA, short | ~0 |

So PDL's total ceiling is **~12 us, and only where the filter/histogram kernels
are long** (>= 2^20). At 2^16 it can recover essentially nothing.

Measured gaps for comparison: 14.9 us at 2^16, 13.0 us at 2^20, 18.7 us at 2^24,
124 us at 2^28 (F32, k=8).

=> PDL covers most of the gap at 2^20-2^24, none of it at 2^16, and ~10% at 2^28.
Fusing the finalize work back into the preceding kernel via last-block election
(what the single-problem path already does through `Counter::finished_block_cnt`)
removes 3 ops = ~12 us at **all** sizes and also eliminates the 1-CTA bubbles.
The 2^28 gap (124 us) is explained by neither and is a separate, real work
difference - likely tuning, since the multi-worker policy has never been swept.
