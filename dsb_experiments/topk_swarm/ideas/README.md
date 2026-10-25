# Idea and approach cards

One markdown file per approach or idea in `queue/`. When an optimizer takes it, the card moves to `approaches/<id>/CARD.md` and the queue entry is replaced by a one-line pointer. Rejected ideas go to `rejected/` with the reason, never deleted.

Template:

```
# <id>: <name>
status: queued | taken by <agent> | done | rejected
scope: subwarp | warp | block
origin: <papers, PRs, reports, or "new">
hypothesis: <one paragraph: what it does and why it should win>
cells it should win: <key types, N, k, requirement, regime>
cost model: <rough per-call cost in terms of rounds, passes, barriers, shuffles>
explore first: <ordered list>
determinism and ordering: <inherent / index-in-key / post-pass, with expected cost>
references: <files and sections>
```
