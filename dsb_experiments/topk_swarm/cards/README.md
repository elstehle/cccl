# Result cards

One JSON file per measurement batch, validated against `schema.json` (`python3 -c "import json,jsonschema; jsonschema.validate(json.load(open(card)), json.load(open('schema.json')))"`, install `jsonschema` with pip if missing). File name `<approach>_<variant>_<arch>_<yyyymmdd>_<n>.json`.

Rules. Every cell has a control measured in the same lock section. The findings fields are required and non-trivial. A card is `verification.verified = true` only after a verifier reran it. The board (`RESULTS_BOARD.md`) is regenerated from the cards by `tools/board.py` and shows verified and unverified cards separately.
