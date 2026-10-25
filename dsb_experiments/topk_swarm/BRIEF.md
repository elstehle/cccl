# Brief for swarm agents

You are one agent in a study of top-k cooperative primitives for CUB. Your launch prompt names your role. This brief is the shared contract. Read it fully before doing anything. Then read, in this order: `PLAN.md` sections 1, 3, 3.1, 4, 6 and 7, the knowledge base `/Users/estehle/Downloads/cccl/TOPK_KNOWLEDGE_BASE.md` sections 2 and 4 (and its section 9 conventions), `reference/BLOCK_TOPK_RESULTS.md` sections 1 and 6, and `reference/DEVICE_SIDE_BENCHMARKING_ISSUE.md`. Role specific reading is listed in your launch prompt.

## 1. Where things are

- Shared worktree on the Mac: `/Users/estehle/Downloads/cccl-topk-swarm`, branch `exp/topk-primitives-swarm` on the fork `origin` (elstehle/cccl), based on upstream main with PR #12222 (`WarpBitonicTopK`) cherry-picked on top. All swarm files live under `dsb_experiments/topk_swarm/`, called `$SWARM` below.
- `$SWARM/PLAN.md`: the approved plan. `$SWARM/BRIEF.md`: this file.
- `$SWARM/approaches/<id>/`: one directory per approach (W1 to W5, B1 to B6), prototypes and the approach report `REPORT.md`.
- `$SWARM/harness/`: the common harness owned by the evaluation engineer. Adapters per approach live next to it under `harness/adapters/`.
- `$SWARM/cards/`: result cards (JSON, schema in `cards/schema.json`) named `<approach>_<variant>_<arch>_<yyyymmdd>_<n>.json`.
- `$SWARM/ideas/queue/`: approach and idea cards waiting for an optimizer. `ideas/rejected/`: ideas the researcher rejected, with the reason.
- `$SWARM/TECHNIQUES.md`: the shared catalogue of generally applicable optimizations and intrinsics with measured costs. Append only, see section 6.
- `$SWARM/RESULTS_BOARD.md`: generated from the cards by `tools/board.py`. Never edit by hand.
- `$SWARM/nodes/README.md`: the node registry protocol. `$SWARM/state/` (gitignored) holds the live registry `state/nodes.json`, fetched results under `state/results/<holder>/`, and logs.
- `$SWARM/tools/`: helper scripts. `tools/lease.py` acquires and releases GPU leases, `tools/append_locked.py` appends to shared markdown files under a lock, `tools/board.py` regenerates the board. The broker writes the node tools, the evaluation engineer writes the board tool. Until a tool exists, do its job by hand with the same semantics and say so in your report.
- Reference material in `$SWARM/reference/`: the July 2026 block top-k study (`BLOCK_TOPK_RESULTS.md`, prototypes `proto_topk*.cu` including the atomics family), the AIR optimization report, the warp merge sort, warp radix sort and sub-warp sort studies, the device-side benchmarking methodology, and the GVR v2 algorithm report.
- Prior knowledge in the Claude memory of the cccl project, readable as files: `/Users/estehle/.claude/projects/-Users-estehle-Downloads-cccl/memory/` (node allocation notes, past study summaries). Treat as background that was true when written.

## 2. Git rules

- Nobody commits or pushes in the shared worktree except the lead. Write your files and say in your report which files you added or changed.
- Optimizers get their own worktree at launch and commit there on their own branch, without gpg (`git -c commit.gpgsign=false commit`), without any AI attribution lines, with short human commit messages.
- Never force-push. Never touch branches other than your own.

## 3. The focus set

Keys bf16 and fp32, values int32 indices, output indices only (the keys are never returned), requirements `not_guaranteed` unsorted and `prefer_smaller_index` unsorted, random input, both regimes. N and k per scope follow `PLAN.md` section 3 and the workloads of section 3.1. k starts at 4 and never exceeds N over 2. Everything outside the focus set waits for the full evaluation unless your launch prompt says otherwise.

Key representation: the harness hands you raw floating keys. Radix and atomics approaches twiddle internally. Comparison approaches compare the floating type natively. A twiddled variant of a comparison method is optional and reported as a separate variant. Minus zero equals plus zero under the tie-break requirement. Focus-set inputs contain no NaN. Document NaN behavior in your contract.

## 4. Methodology (binding)

Latency regime. One warp (`<<<1,32>>>`) or one block (`<<<1,256>>>`), `clock64` cycles, chain-length slope: total cycles over chain lengths 1, 2, 4, 8, 16, least squares slope is the marginal cycles per call, min over at least 24 repetitions. Barrier bracketing for blocks (t0 after a barrier, t1 after the closing barrier). Memory clobber before the end clock, outputs consumed by a `sink()` that the compiler cannot prove dead (see `reference/BLOCK_TOPK_RESULTS.md` section 6 and `nvbench_helper/device_side_benchmark.cuh`). Data-oblivious primitives run an in-place dependency chain with a per-iteration perturbation, data-dependent primitives get fresh random input per call with the chain serialized on the previous output and a generate-only control subtracted.

Throughput regime. Fixed workload of one occupancy wave (`cudaOccupancyMaxActiveBlocksPerMultiprocessor` times SMs) times a fixed call count, element-normalized Gelem/s, best of 5 event timings, occupancy and registers reported next to the number. Approaches wired into a batched agent additionally run the `segmented_topk` nvbench benchmarks with the standard flags (`--device 0 --timeout 30 --stopping-criterion entropy --throttle-threshold 90 --throttle-recovery-delay 0.15 --json`).

Same session A/B. Every measurement of a variant runs together with its control (the incumbent for that scope and cell, or the previous variant) inside the same GPU lock section, on the same node, in the same process launch. Numbers from different sessions or nodes are never compared in a card. A drift canary (a fixed reference kernel) runs at the start and end of every measurement batch.

Correctness before numbers. A host oracle checks every cell before any number is recorded: set equality for `not_guaranteed`, exact result for `prefer_smaller_index`, special values (signed zero, infinities, denormals) where the type has them, partial inputs if the contract allows them. A failing oracle means no number.

Resources. Registers, shared memory and spills from `cuobjdump -res-usage` or `-Xptxas -v`, SASS checked for `STL` and `LDL` (spills) and for `BAR.SYNC` count. Register-tier effects on batched kernels are measured on the device kernel, not inferred from the harness.

Every number carries arch, CUDA version, node, date, commit and the file with the full table. No number without a control. No cross-arch comparison in a card.

## 5. Nodes and measurements

- A node is a B200 (computelab, x86) or a GB200 (dlcluster, aarch64). Both are valid for development. The final evaluation runs on one node in one session. The registry `state/nodes.json` lists the nodes and their GPUs, see `nodes/README.md`.
- Get a GPU with `python3 $SWARM/tools/lease.py acquire --holder <your-name> --arch any` (or `b200`, `gb200`). It prints the node, the GPU index, the ssh target and the on-node root. Release it with `release` as soon as your batch is done. Do not hold a GPU while you think or compile.
- On the node, your working directory is `<root>/work/<your-name>/`. Compiles run outside the lock. Every measurement runs inside `flock <root>/locks/gpu<index>.lock -c '<command>'` with `CUDA_VISIBLE_DEVICES=<index>`, variant and control inside the same lock section.
- Results are JSON files under `<root>/results/<your-name>/`. Fetch them to the Mac into `$SWARM/state/results/<your-name>/` right after every batch (tar over ssh), nodes disappear every four hours.
- Keep remote command output short. Log to files on the node and read the tail. Long outputs have triggered output truncation and classifier stops before.
- Computelab nodes run a CUDA devel docker container with the clone under `/work`. dlcluster GB200 nodes have no docker daemon and no cmake, the broker provides a pyxis or enroot container and a cmake. The 5 GB NFS home is not a work area.

## 6. Knowledge capture (binding)

- Result card: every batch that produces a number ends in a card under `$SWARM/cards/` validated against `cards/schema.json`. The findings fields are required: `low_level` (instruction costs, SASS, registers, occupancy, compiler behavior), `algorithmic` (why it wins or loses where, the cost model that held), `negative` (what did not pay, at which design point), `surprises`, `open_questions`. A card with empty findings is incomplete and comes back.
- Techniques catalogue: when you find something that applies beyond your approach (an intrinsic cost, a codegen pitfall and its fix, a data layout trick), append an entry to `$SWARM/TECHNIQUES.md` right away with `python3 $SWARM/tools/append_locked.py TECHNIQUES.md <entry-file>`. Entry format: a heading `### T<n>: <name>`, then `applies to`, `measured` (number with arch, CUDA, node, date, microbenchmark file), `evidence` (SASS or file), `how to use`, `caveats`, `author`. Read the catalogue at the start of every iteration and cite the entries you adopt in your card (`techniques_adopted`).
- Approach report: `$SWARM/approaches/<id>/REPORT.md` with the knowledge base skeleton: what it is, contract, where the time goes, ideas tried with verdicts, ideas not yet tried, determinism and ordering support, open items, pointers.
- Ideas: a new idea is a card in `ideas/queue/` with the template in `ideas/README.md`. A rejected idea goes to `ideas/rejected/` with the reason, never silently dropped.

## 7. Communication

- Report by finishing your turn with a short structured summary: what you did, files touched, numbers if any (with their controls), what is blocked and on what, what you propose next. The lead continues you with a message when needed, your context is kept.
- Message a named peer with SendMessage when you need something only they have (the broker for nodes, the hardware researcher for an intrinsic question, the evaluation engineer for the harness contract). Keep messages short and specific.
- If you are blocked on a node, do everything that needs no node first, then report.

## 8. Conventions

- No em dashes and no semicolons in prose, in files that may end up in a PR or an issue, which is all of them.
- No AI attribution anywhere. No performance numbers in code comments. Numbers go in cards and reports.
- Prefer the smallest change that answers the question. Re-measure every micro-optimization in its own code structure, codegen does not transfer.
- Prior results in the reference material are measurements at one design point. Re-measure in your cells before you rely on them.
