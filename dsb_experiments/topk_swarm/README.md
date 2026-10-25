# topk_swarm

Multi-agent exploration of sub-warp, warp and block top-k primitives. Start with `PLAN.md` (what and why) and `BRIEF.md` (the contract every agent follows). Layout:

- `PLAN.md`, `BRIEF.md`: plan and brief.
- `approaches/<id>/`: prototypes and `REPORT.md` per approach.
- `harness/`: the common harness and per-approach adapters.
- `cards/`: result cards (JSON) and their schema.
- `ideas/queue/`, `ideas/rejected/`: approach and idea cards.
- `TECHNIQUES.md`: shared catalogue. `RESULTS_BOARD.md`: generated board.
- `nodes/README.md`: node registry protocol. `tools/`: lease, append, board and broker tools.
- `reference/`: prior studies and prototypes this work builds on.
- `state/` (gitignored): live registry, fetched results, logs.
