# Top-k cooperative primitives: swarm exploration plan (draft for review)

Status: all ten decisions in section 12 settled on 2026-10-25, awaiting the go for phase 0. Nothing has been launched.

## 1. What I make of the goal

Find the fastest sub-warp, warp and block level top-k building blocks for key-index pairs across floating key widths (8, 16, 32, 64 bit), input sizes, k, tie-break requirement, and two regimes (latency and throughput). The primary output form is indices only, the keys are never returned. Identify the winner per cell of that matrix. For every approach, document whether it supports deterministic set membership, index tie-break and sorted output inherently, or how it would be extended (index composed into the key, post-sort with a warp or block sort). Deliver a report that updates the GitHub tracking structure: one parent issue per algorithmic approach, sub-issues per follow-up optimization, per type specialization and per determinism feature, in the style of #9747 and #9749.

Interfaces are free. A primitive may take input in registers or shared memory, in blocked, striped or packed arrangement, and may hand results back in any arrangement. Each prototype declares its contract. Fitting the winners into the batched device interface is a later step and only informs the contract (what a worker-per-segment or warp-per-segment agent can feed and consume cheaply).

## 2. Deliverables

1. `WINNERS.md`: the winner matrix with numbers, one table per regime, cells tagged with arch, CUDA version, node and date, plus heatmaps.
2. The knowledge base. `TOPK_KNOWLEDGE_BASE.md` grows continuously during the study, not at the end, as described in section 6. It gains one section per approach (section 9 skeleton: what it is, where the time goes, ideas tried with verdicts, ideas not yet tried, determinism and ordering support, open items, pointers), a measured microarchitecture section for B200 and GB200, an extended codegen pitfalls section, a section of algorithmic findings that hold across approaches, the winner matrix, and a negative results ledger. The rendered page is rebuilt with it.
3. `TECHNIQUES.md`: the working catalogue the agents write to during the study, one entry per generally applicable optimization or intrinsic with measured cost, applicability and SASS evidence. The curator folds it into the knowledge base at every wave boundary, so the catalogue is the inbox and the knowledge base is the record.
4. `GITHUB_ISSUE_TREE.md`: one document with two audiences. Part A is for you: the issue tree as a diagram and a table (parent issue per approach, sub-issues per follow-up optimization, per type specialization and per determinism feature), one line of rationale and evidence per issue, and which existing issues get updated instead of created (#9747 strategy list, #9749 and its siblings). Part B is for an agent with `gh`: an ordered manifest, one block per issue with title, labels, parent reference, dependencies, and the full body in a fenced block, plus the exact creation order and the sub-issue linking step, so the agent needs no judgment beyond running it. Part B is mapped onto the strategy numbers of #9747.
5. Code on a fork branch: the prototypes, the common harness, the benchmark extensions, the node scripts, all under `dsb_experiments/topk_swarm/`.

## 3. Scope I propose (defaults, confirm or change)

| Dimension | Proposed values | Note |
|---|---|---|
| Key types | Focus set: bf16 and fp32. Full evaluation adds fp8 e4m3, fp16 and fp64. | Optimizers iterate on the focus set. CUB already has radix traits for both fp8 types (`util_type.cuh` 1107). e5m2 skipped, same code path as e4m3. |
| Values | int32 index | Pairs throughout as input. |
| Output | Focus set: indices only (arg top-k, keys are never written back). Full evaluation also measures key-index pair output. Keys-only measured as a secondary column where it is free. | Index-only output lets a primitive drop keys after selection, which saves output traffic and registers, and it is what the attention indexer workloads consume. Approaches that resolve indices after a value-only selection, or that carry the index inside the key word, produce this form natively. |
| Scopes | sub-warp (logical warp 4, 8, 16), warp (32), block (256 threads, later 128 and 512) | |
| N per scope | sub-warp 8 to 128, warp 32 to 4096, block 1024 to 32768 | Powers of two for the grid, derived from the LLM workloads in section 3.1. Items per thread 1 to 128. |
| k | 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, bounded by N over 2 | k of 1 is a reduction, not a selection, and is out. k of 2 appears only as one extra column in the final evaluation at sub-warp and warp scope, because grouped MoE routing still uses it. The ratio k over N matters too, since large k favors sorting. |
| Requirements | not_guaranteed unsorted, and prefer_smaller_index unsorted, in the focus set and in the full evaluation | Sorted output is measured once at the very end across all identified solutions that support it (section 9, phase 3b). |
| Inputs | uniform random, tie-heavy, sorted ascending and descending, pivot ties (from `BLOCK_TOPK_LATENCY_SPEC.md`) | special values in correctness only |
| Regimes | latency: single warp or block, chain-slope cycles per call. Throughput: one resident wave, Gelem/s, plus the batched `segmented_topk` benchmarks for approaches wired into a batched agent | methodology of `BLOCK_TOPK_RESULTS.md` section 6 and knowledge base 2.7 |

### 3.1 Workloads the cells stand for

The grid above is chosen so that every current or foreseeable LLM top-k lands on or between grid points. The final evaluation also runs the named cells below as they are, including the non power of two expert counts, so the report can state winners per model rather than per grid point. The algorithm researcher confirms the figures against the model cards before the final run.

| Workload | Where top-k runs | N per call | k | Scope | Regimes |
|---|---|---|---|---|---|
| MoE token-choice routing, current models | one row per token over expert scores: Mixtral 8 experts top 2, DBRX 16 top 4, gpt-oss 32 and 128 top 4, Qwen3 128 top 8, GLM-4.5 160 top 8, DeepSeek-V3 and R1 256 top 8, Kimi K2 384 top 8, Llama 4 128 top 1 | 8 to 384 | 2 to 8 | sub-warp and warp | both, decode is latency bound at small batch, prefill is throughput bound |
| Grouped or node-limited routing (DeepSeek style) | top 2 within each group of 32 experts, then top 4 groups of 8, then top 8 overall | 8 to 32 per group | 2 to 8 | sub-warp | both |
| Future fine-grained MoE and product-key experts | 512 to 4096 routed experts with 8 to 32 active, PEER style product keys with two top-k over 1024 sub-keys | 512 to 4096 | 8 to 32 | warp and block | both |
| Expert-choice routing and mixture-of-depths capacity | experts or layers pick the top fraction of tokens: capacity of one eighth to one half of the tokens | 1024 to 8192 | N over 8 to N over 2 | block | throughput |
| Top-k and top-p sampling, speculative draft trees | per sequence over the vocabulary, 32K to 256K entries, processed as chunks per block then merged: sampling k 20 to 100, draft trees k 4 to 16 per node, beam search k 4 to 8 | 4096 to 32768 per chunk | 4 to 1024 | block | latency at decode, throughput at large batch |
| Sparse attention indexers (DSA top 2048 tokens, NSA top 16 blocks) | per query over context scores: DSA 32K to 128K tokens today, 1M foreseeable, chunked per block then merged; NSA over 512 to 2048 compressed blocks | 1024 to 32768 per chunk | 16 to 4096 | warp and block | both |
| KV cache selection and eviction (Quest, SnapKV, H2O style) | per head over pages or positions with a fixed budget | 1024 to 16384 | 16 to 4096 | block | throughput |

Key representation. The harness hands every approach the raw floating keys. Radix and atomics based approaches twiddle to an order-preserving unsigned integer internally, as they must. Comparison based approaches keep the floating type and compare it natively. An optimizer may still try a twiddled integer variant of a comparison based approach when it expects a gain (integer min and max, `redux.sync`, packed SIMD compares on 16-bit lanes), but it reports that as a variant next to the native one, not as the baseline. Semantics the correctness oracle enforces across both families: minus zero and plus zero are equal under the tie-break requirement (what PR #11835 established for the radix path and what native float comparison gives for free), and focus-set inputs contain no NaN. NaN behavior is documented per approach in its contract rather than normalized.

Interface contract per prototype: input location and arrangement, output location and arrangement, whether keys are materialized on output or indices only, whether output is sorted, temp storage, synchronization it performs and assumes, supported k range, supported N range, type constraints.

## 4. Approach inventory (the exploration queue)

Numbers in brackets refer to the strategy list in #9747. New approaches get new numbers in the issue tree.

### Warp and sub-warp

| Id | Approach | Origin | What its optimizer explores first | Determinism and ordering |
|---|---|---|---|---|
| W1 | Bitonic network top-k [1] | PR #12222 (eager), PR #9281 also has a buffered threshold-compaction variant, `WarpBitonicSort` sub-warp branch `exp/sub-warp-bitonic-sort` | Sub-warp support (MaxK multiple of the logical warp size instead of 32). Packed arrangement for 16-bit and 8-bit keys: 2 or 4 keys per 32-bit register in blocked order inside the register, striped across the logical warp, compared with SIMD min and max. Min and max operator instead of a bool comparator (#12160). The buffered variant for N much larger than k. | Inherent: deterministic, sorted. Index tie-break by comparing (key, index), or for keys of 16 bits or less by packing key and inverted index into one 32-bit word, which makes the tie-break free up to N of 65536. |
| W2 | Redux based iterative extraction [2] | `BLOCK_TOPK_RESULTS.md` (warp round measured at 82 cycles), `hier_extract` | k rounds of `redux.sync` max plus ballot plus elect. For keys of 16 bits or less, pack key and index into the 32-bit word so every round is an exact arg-max with no tie machinery (the `atomic_iter_h` trick at warp scope). Check `redux.sync` f32 min and max on sm_100 for unpacked fp32. Small k only. | Packed form: inherent, sorted, prefer-smaller-index by inverting the index. Value-only form: set membership for distinct values, ties need an index pass. |
| W3 | Warp radix select | `WARP_RADIX_SORT_RESULTS.md` (ballot-ranked LSD) | Digit histograms with `match.any` and ballot counts, k-selection over digits, narrow digits for 8-bit and 16-bit keys (one or two passes). Candidate for larger N per warp and any k. | Set deterministic by construction. Sorted via a final bitonic pass over k. Index tie-break via a last index pass or the warp-redundant rank idea from knowledge base 4.5. |
| W4 | Merge based top-k | `WARP_MERGE_SORT_RESULTS.md`, `SUBWARP_SORT_RESULTS.md` | Sort per-thread runs, merge-path keep-top-k. Expected to lose to W1 at small N, kept to find the crossover at large N and large k. | Inherent, sorted, stable. |
| W5 | Threshold filter with verification (GVR style) | `~/Downloads/topk/GVR_V2_ALGORITHM_REPORT.md` | Guess a threshold from a cheap sample (per-lane maxima sorted by a small bitonic network, or the top-k of a sub-sample), one pass keeps candidates at or above it, finish exactly on the candidates, count to verify, retry lower or fall back to W1 or W3 on overflow. | Exact by counting. Ties at the threshold resolved in the finishing stage. Ordering from the finishing primitive. |

### Block

| Id | Approach | Origin | What its optimizer explores first | Determinism and ordering |
|---|---|---|---|---|
| B1 | AIR radix select, the incumbent | `block_topk_air`, PR #11060, branch `enh/block-topk-striped`, knowledge base 4 | Continue the open items of knowledge base 4.7 on the striped base: raw extractor, `NormalizeSignedZero`, deterministic warp-redundant rank. New: 8-bit keys are a single pass, 16-bit keys two passes, so re-tune digit width per key width. Pairs path with packed small keys. | Set deterministic with the warp-redundant rank (knowledge base 4.5). Sorted via post-sort of k. |
| B2 | Atomics based iterative extraction [4] | `atomic_adaptive`, `atomic_iter_h` in `BLOCK_TOPK_RESULTS.md` | Pairs with index for 16-bit and 8-bit keys by packing key and index into the atomic word. fp32 keys: value rounds plus deterministic index resolution (min-index ticket). Extend from k of 16 to 64, find where rounds stop paying. | Packed form inherent and sorted. fp32 form needs the index pass for ties. |
| B3 | Filtering or stream-compacting radix select | New, cousin of cluster idea A3 (knowledge base 3.4) | After each pass compact the surviving candidates into shared memory or into a dense register set via ballot and prefix, so later passes touch candidates only. Targets large N per block, more passes (fp64), and pairs where moving values is the cost. | Same as B1. |
| B4 | Exact-bound prefilter then small selection | New | Any k elements give a lower bound on the k-th largest: the minimum of that set. Cheap exact bound: the k-th largest of the per-thread maxima, computed by a fast primitive on N equal to the thread count. On random input the expected candidate count is close to k, and the finish runs on candidates only with an exact fallback to B1 when a buffer overflows (sorted or adversarial inputs). Also the sampled, non-exact GVR guess at block scope. | Exact by counting. Finishing primitive decides ordering. |
| B5 | Hierarchical warp then block composition | `hier_extract` and `bitonic_hier` lost at k 16, N 1024 | Each warp runs W1, W2 or W3, then merge the W times k candidates with a bitonic merge or a second warp top-k. Re-measured per cell with packed small keys, where the warp phase gets much cheaper. | Inherits from the warp primitive, merge is deterministic. |
| B6 | Sort when k is large | `HYBRID_BLOCK_SORT_RESULTS.md`, `BlockRadixSort`, `BlockMergeSort` | For k above roughly N over 4, sort and truncate. Find the crossover against B1 per key width. | Inherent, sorted, stable options. |

New ideas from the algorithm researcher enter this table with the same columns. Candidates it will be seeded with: sample sort style pivot selection, bucket select with wide digits for 8-bit keys, k-ary search (measured negative once, at one design point), two-level packed radix for fp64, and threshold reuse across segments in the batched setting.

## 5. Agent roles

| Role | Count | Lifetime | Owns |
|---|---|---|---|
| Lead (this session) | 1 | whole study | plan, queue, wave launches, merge into the knowledge base, final report. Runs a self-paced loop while waves run and reports at wave boundaries. |
| Node broker | 1 | whole study | the node pool: allocation on both clusters, bootstrap, lease registry, re-allocation ahead of expiry, lock files per GPU. |
| Algorithm researcher | 1 | whole study | new idea cards into the queue, each with hypothesis, cells it should win, cost model, sketch. Refreshed with the results board so ideas respond to data. Does not code. |
| CUDA hardware researcher | 1 | whole study | the techniques catalogue: Blackwell and Rubin intrinsics relevant to selection, each verified by a cycle microbenchmark on the node, plus generally applicable code generation lessons. Answers optimizer questions. |
| Evaluation engineer | 1 | whole study | the common harness over all primitives, cells and regimes, correctness oracles, the batched benchmark extensions, the measurement queue, the final same-node evaluation, heatmaps and the winner matrix. |
| Optimizer | 1 per approach, in waves | one wave | a worktree and a prototype directory, iterates on its approach with the methodology, posts generally applicable findings to the catalogue, reads the catalogue at each iteration start, returns a structured result card and a report. |
| Verifier | 1 per result card | short | tries to refute: rebuilds from the worktree, reruns under the lock, checks the anti-DCE sink, SASS for spills and local memory, the correctness oracle, and that the comparison was same-session against the control. |
| Knowledge curator | 1 | whole study, active at every wave boundary and at the end | the knowledge base: folds verified result cards, catalogue entries, idea cards (accepted and rejected) and verifier findings into `TOPK_KNOWLEDGE_BASE.md` with cross references, keeps the ledger and status board current, rebuilds the rendered page, and at the end writes the winners document and the issue tree. Rejects a card whose lessons are not backed by its evidence and sends it back to the optimizer. |

Knowledge sharing is file based, so it survives agent restarts and context compaction: `TECHNIQUES.md` (append only under a lock, one entry per trick with cost, applicability and SASS evidence), `RESULTS_BOARD.md` generated from the result cards, `ideas/queue/` for idea cards. Long-lived agents also message each other and me directly for questions. Optimizers cite the catalogue entries they adopted so we can trace what mattered.

Intrinsics the hardware researcher starts from, to verify rather than assume: `redux.sync` including the f32 min and max forms on sm_100, 3-input integer min and max, SIMD min and max on packed 16-bit and 8-bit lanes, `max.f16x2` and `max.bf16x2` for unnormalized halves, `match.any`, `elect.sync`, `bmsk`, `fns`, `prmt` for packing and unpacking, `ldmatrix` and `stmatrix` for register transposes, `cp.async.bulk` for shared memory fills, `mbarrier` and warp specialization, `setmaxnreg`, and whatever the PTX ISA lists as new for the Rubin targets.

## 6. Knowledge capture

The knowledge base is a primary output, on par with the winner matrix. Three rules make sure it comes out of the study rather than being reconstructed afterwards.

Capture at the source. Every result card has required fields beyond the numbers: low-level findings (instruction costs measured, SASS observations, register and occupancy effects, compiler behavior), algorithmic findings (why the approach wins or loses in which cells, what the cost model turned out to be), negative results (what was tried and did not pay, at which design point), surprises, and open questions. A card without these fields is incomplete and the verifier returns it. Researcher idea cards carry the same discipline, including the ideas the researcher rejected and why.

Fold in at every wave boundary. The knowledge curator reads the new cards, catalogue entries and verifier findings after each wave and writes them into the knowledge base, so a wave 2 optimizer starts from a knowledge base that already contains wave 1. Nothing waits for the end of the study.

Structure in the knowledge base. New or extended sections, all following its section 9 conventions (every number carries hardware, CUDA version, benchmark, date and the file with the full table, no em dashes, no semicolons):

- One top-level section per approach, W1 to W5 and B1 to B6 and any added by the researcher, with the section 9 skeleton.
- Section 2, cross-cutting lessons, gains three subsections: measured microarchitecture facts for B200 and GB200 (instruction and primitive costs in cycles with the microbenchmark file, in the style of `BLOCK_TOPK_RESULTS.md` section 1), codegen pitfalls and how to avoid them (extending the existing per-key and straight-line lessons with what the optimizers hit), and algorithmic findings that hold across approaches (for example when a threshold filter pays, when packing the index into the key pays, where k over N flips the winner to sorting).
- A winners section with the matrix per regime and per arch and pointers to the heatmaps.
- A negative results ledger, one line per idea that did not pay, with the design point and the file, so the next study does not repeat it without a reason.
- Ledger entries per wave and a status board row per branch, as today.

The rendered page is rebuilt from the markdown after each fold and republished to the existing artifact link, so the knowledge base is readable outside this Mac.

## 7. Infrastructure and protocol

Nodes. Target pool: two to four nodes, at least one B200 on computelab (login `ssh computelab-sc-01`) and one GB200 on dlcluster (login `ssh dlcluster`), more when available, with a hard cap of six nodes held at the same time across both clusters. The broker counts pending requests against the cap so a burst of grants cannot overshoot it. The broker allocates detachably with `sbatch` and a hold script (the verified fallback in the memory notes), attaches with `srun --jobid` or ssh, and only uses interactive `crun` or `srun --pty` when you prefer that. Bootstrap per node: computelab nodes run a CUDA devel docker container with a `/work` overlay and a blobless clone. GB200 nodes have no docker daemon and no cmake, so the broker uses pyxis or enroot with a CUDA devel aarch64 image and our cmake tarball, with sources on `/raid`. Each node entry records arch, CUDA version, lease expiry and current holder.

Leases are four hours. Workers checkpoint every result to the Mac right after the run (tar over ssh) and re-queue unfinished runs on the next node. The broker re-requests ahead of expiry so the pool stays at target.

Measurement exclusivity. Compiles run in parallel on the node. Every measurement runs under `flock` on a per-GPU lock file, with the variant and its control inside the same critical section, so the same-session rule holds even when runs interleave. Latency harness runs are short, so queueing is cheap. Throughput runs need the GPU exclusively.

Architecture rule. An optimizer develops on whichever arch it was handed and reports per arch. No cross-arch comparisons enter the matrix. The final evaluation runs every approach on every cell on one B200 node in one session, interleaved with a main build control and a drift canary at start and end, and repeats on one GB200 node when available.

Correctness. A host oracle per cell: set equality for not_guaranteed, exact result for prefer_smaller_index, sorted check, special values (NaN, signed zero, infinities, denormals, fp8 saturation), partial tiles. A prototype that fails the oracle produces no numbers.

Repository. New branch `exp/topk-primitives-swarm` on the fork, prototypes under `dsb_experiments/topk_swarm/<approach>/`, one worktree per optimizer on the Mac, one clone per node. The existing experiment branch `exp/device-side-perf` stays untouched.

## 8. Workload matrix and pruning

The full cross product (5 key types, 3 scopes, 7 to 10 sizes, 12 k values, 2 requirements, 5 inputs, 2 regimes) is far too large to run per approach while iterating. Three tiers:

1. Iteration set, per optimizer: the focus set (bf16 and fp32 keys, int32 indices, index-only output, not_guaranteed unsorted and prefer_smaller_index unsorted), 8 to 12 cells of N and k chosen for its approach (its claimed sweet spot plus two cells just outside it), random input, both regimes.
2. Coarse matrix, evaluation engineer, every approach: the focus set first, then fp8 e4m3, fp16 and fp64, N and k on a logarithmic grid (one point per power of four), both unsorted requirements, random and tie-heavy inputs.
3. Refinement near crossovers: a dense grid where two approaches are within 15 percent, plus the remaining input patterns for the winner and runner-up.
4. Sorted output, once at the end: every identified solution that supports sorted output (inherently or via its documented post-sort) on the focus set, so the cost of ordering is known per approach without steering the search.

## 9. Phases

| Phase | Content | Exit criterion |
|---|---|---|
| 0 Setup | brief finalized from this plan, branch and directories, broker up with one B200 and one GB200, harness skeleton producing cells for two reference primitives (B1 and W1), catalogue seeded from the existing reports | reference numbers reproduce the known results within noise |
| 1 Wave 1 | optimizers for W1, W2, B1, B2, B3, researchers running, verifiers on each card | all five cards verified, board updated, knowledge base folded |
| 2 Wave 2 | W3, W4, W5, B4, B5, B6, new ideas from the researcher, and a go-deeper round for wave 1 survivors with the techniques discovered since | cards verified, knowledge base folded |
| 3 Final evaluation | coarse matrix then refinement on one node, same session, both unsorted requirements | winner matrix with controls and canaries |
| 3b Sorted-output pass | all identified solutions with sorted output where supported, focus set, same node and session as phase 3 | ordering cost per approach documented |
| 4 Reporting | final knowledge base consolidation, winners document, issue tree | you have the documents and the rendered page is current |

Each wave is hours of agent time, bounded mostly by node availability and the measurement queue. I report at every wave boundary and whenever you ask.

## 10. How I would run it in Claude Code

- Long-lived roles (broker, both researchers, evaluation engineer) as named background agents, so they can message each other and me and keep their own context across the study.
- Each optimizer wave as one Workflow run (decision 9): optimize, then verify, then card, as a pipeline so each verifier starts as soon as its optimizer finishes. The workflow is resumable, which matters when a lease expires mid-wave. Concurrency on this Mac is capped at ten agents per workflow (12 cores). Node names live in the registry file, not in prompts, so a node change does not invalidate finished agents on resume.
- I stay in a self-paced loop and wake every 20 to 30 minutes to merge cards, update the queue, rebalance leases and start the next wave. The board files are the source of truth, not my context.
- This session has to stay open for the agents to keep running. Running the terminal under tmux is advisable.
- Cost: around five persistent agents, eleven optimizers, eleven verifiers and a few judges over the study, each with long contexts. Token spend will be large. Say so if you want a ceiling and I will size the waves to it.

## 11. Risks and mitigations

- Node churn (lease expiry, crun outages): detachable sbatch allocation, result checkpointing after every run, re-queue, a pool target instead of a fixed node.
- Contaminated measurements: per-GPU lock with control inside the critical section, verifier rerun, canary.
- Anchoring on prior conclusions: optimizers get the methodology and their approach card, and prior results are labeled as measurements at one design point to be re-measured per cell, not as facts.
- Cross-arch confusion: every number carries its arch, the matrix is per arch, final evaluation on one node.
- Register tier lottery: device-level throughput only on batched kernels for integrated approaches, primitive-level throughput reported with occupancy.
- Context compaction of long-lived agents: board files, re-read on every wake.
- Cluster home quota of 5 GB: everything on node-local storage, mirrored to the Mac.
- Safety classifier stops on long remote command output (seen 2026-10-19): agents keep command output short and log to files.

## 12. Decisions I need from you

1. Decided 2026-10-25: focus set bf16 and fp32 keys with int32 indices, output indices only, for optimizer iteration and the initial evaluation. Full evaluation adds fp8 e4m3, fp16 and fp64 and key-index pair output (my assumption, say if the full set should also stay at bf16 and fp32 or at index-only output).
2. Decided 2026-10-25: requirements are not_guaranteed unsorted and prefer_smaller_index unsorted for both the focus set and the full evaluation. Sorted output is measured only in the final pass (phase 3b). N and k follow the LLM workloads of section 3.1 (MoE routing now and with thousands of experts, grouped routing, sampling and draft trees, sparse attention indexers, KV selection, capacity routing). k starts at 4, k of 2 only as a final-evaluation column at sub-warp and warp scope. Block size 256 in wave 1, 128 and 512 in the final evaluation where the winner is sensitive to it.
3. Decided 2026-10-25: no required twiddle to unsigned. Approaches that handle floating inputs natively keep them, radix and atomics approaches twiddle internally, twiddled variants of comparison methods are optional and reported separately. Signed zero equal under tie-break, no NaN in focus-set inputs.
4. Decided 2026-10-25: branch `exp/topk-primitives-swarm` on the fork (elstehle/cccl), prototypes, harness, scripts and board files under `dsb_experiments/topk_swarm/`.
5. Decided 2026-10-25: at most six nodes held at the same time across both clusters, pending requests included. Login host for computelab is `ssh computelab-sc-01`, for dlcluster `ssh dlcluster`. The broker uses detachable `sbatch` holds and attaches with `srun --jobid` or ssh, releases a node with `scancel` as soon as the queue for it is empty and no wave is pending, and falls back to interactive `crun` or `srun --pty` only when batch submission fails.
6. Decided 2026-10-25: GB200 is a valid development arch, every number carries its arch, final evaluation on one B200 node in one session, second full pass on one GB200 node when available.
7. Decided 2026-10-25: no issue creation by the swarm. The deliverable is `GITHUB_ISSUE_TREE.md` as described in section 2, readable by you as the issue plan and executable by a separate agent with `gh` after your review.
8. Decided 2026-10-25 (delegated to the lead): five optimizers per wave, sized down when fewer than two nodes are held. No hard token or time ceiling. A progress report at every wave boundary is the checkpoint where the study can be stopped or redirected.
9. Decided 2026-10-25 (delegated to the lead): the optimizer waves run as workflows (optimize, verify, card as a pipeline, resumable). The opt-in rests on the user's request for a swarm of sub-agents and on this delegation. Long-lived roles stay named background agents.
10. Decided 2026-10-25 (delegated to the lead): the inventory of section 4 stands. Wave 1 is W1, W2, B1, B2, B3. Wave 2 is W3, W5, B4, B5, B6 plus researcher ideas. W4 (merge based) runs last and only to locate its crossover at large N and k. B4 (exact-bound prefilter) stays in.
