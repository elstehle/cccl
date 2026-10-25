# Result cards

One JSON file per measurement batch, validated against `schema.json` (`python3 -c "import json,jsonschema; jsonschema.validate(json.load(open(card)), json.load(open('schema.json')))"`, install `jsonschema` with pip if missing). File name `<approach>_<variant>_<arch>_<yyyymmdd>_<n>.json`.

Rules. `commit` is the commit of the adapter source the kernels were built from (the optimizer's
worktree HEAD, passed to `tools/make_card.py --source-commit` or recorded by run.py from
`harness/SOURCE_COMMIT`), the node's CCCL clone commit is the optional `clone_commit`. Every cell has
a control measured in the same lock section. `input_pattern` accepts every harness pattern
(`random`, `tie_heavy`, `sorted_asc`, `sorted_desc`, `pivot_tie`, `all_equal`, `signed_zero`,
`inf_denorm`, `quantized_random`, `relu_quantized`) plus `named_workload`. The findings fields are required and non-trivial. A card is `verification.verified = true` only after a verifier reran it. The board (`RESULTS_BOARD.md`) is regenerated from the cards by `tools/board.py` and shows verified and unverified cards separately.

Harness versions. A card is judged under the harness version it was measured with, because its variant and control shared one lock section and one harness build. When a later harness version changes a control's net number (as the 0.6 no-op control did for shared-memory heavy controls), the verifier records the current-version ratios in the card next to the card's own, and the card stays verified if the conclusions hold under both. Comparisons across cards measured under different harness versions are not made, the final evaluation re-measures every winner under one version in one session.
