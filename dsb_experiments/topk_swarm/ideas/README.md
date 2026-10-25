# Idea and approach cards

One markdown file per approach or idea in `queue/`. When an optimizer takes it, the card moves to `approaches/<id>/CARD.md` and the queue entry is replaced by a one-line pointer. Rejected ideas go to `rejected/` with the reason, never deleted. `WORKLOADS.md` holds the workload check behind the cells, `named_cells.json` the machine-readable named cells for the harness, `PRIORITIES.md` the researcher's ranking.

Template (cards stay under 80 lines, numbers from prior reports are measurements at one design point and are labeled as placeholders):

```
# <id>: <name>
status: queued | taken by <agent> | done | rejected
scope: subwarp | warp | block
origin: <papers, PRs, reports, or "new">
hypothesis: <one paragraph: what it does and why it should win>
cells it should win: <key types, N, k, requirement, regime>
cost model: <cycles per call as a formula over N, k, items per thread and the primitive costs, with the placeholders labeled>
focus cells: <three to five focus-set cells the optimizer iterates on>
boundary cells: <two cells just outside the expected sweet spot>
explore first: <ordered list>
determinism and ordering: <inherent / index-in-key / post-pass, with expected cost>
main risk: <one paragraph>
cheapest kill experiment: <what to measure and the threshold that retires the idea>
references: <files and sections>
```

Rejected entry template:

```
# R<n>: <name>
status: rejected (<date>, <agent>)
scope: ...
seed: <where the idea came from>
idea: <one paragraph>
reason: <one paragraph with the evidence>
what survives: <which card carries the usable part, or "nothing">
```
