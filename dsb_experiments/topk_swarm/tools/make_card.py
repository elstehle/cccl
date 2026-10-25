#!/usr/bin/env python3
"""Assemble a result card from harness batch JSON plus a findings file, validate it against
cards/schema.json, and write it to cards/.

  python3 tools/make_card.py --batch state/results/<holder>/<batch>.json [--batch ...] \
      --findings <findings.json> --out cards/<approach>_<variant>_<arch>_<yyyymmdd>_<n>.json

The findings file carries what the harness cannot know (author-written):
  {
    "author": "eval-engineer",
    "approach_id": "B1",            (optional, default: adapter id up to the first underscore)
    "variant": "reference",         (optional, default: adapter id after the first underscore)
    "commit": "...",                (optional, default: repo commit recorded in the batch)
    "contract": {...},              (schema contract object)
    "findings": {...},              (schema findings object, all five lists)
    "techniques_adopted": [], "techniques_contributed": [],
    "report_path": "...", "source_paths": [...],
    "correctness_notes": "...",     (optional, appended to correctness.notes)
    "verification": {...}           (optional)
  }

Cells whose oracle failed, or that have no control value, are dropped from the card and listed in
correctness.notes (no number without a control, a failing oracle means no number). Validation uses
the jsonschema package when installed (pip install jsonschema), otherwise a built-in subset
validator (types, required, enum, pattern, items, minItems).
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWARM = os.path.dirname(HERE)

CELL_KEYS = ["key_type", "value_type", "output_form", "n", "k", "threads", "requirement", "input_pattern",
             "workload_name", "regime", "metric", "value", "control_name", "control_value", "canary_start",
             "canary_end", "resources", "raw_file"]
RES_KEYS = ["registers", "smem_bytes", "spill_bytes", "occupancy_blocks_per_sm", "bar_sync_count"]


def fallback_validate(obj, schema, path="$"):
    """Minimal JSON-schema subset validator. Returns a list of error strings."""
    errors = []
    t = schema.get("type")
    if t:
        ok = {
            "object": isinstance(obj, dict),
            "array": isinstance(obj, list),
            "string": isinstance(obj, str),
            "integer": isinstance(obj, int) and not isinstance(obj, bool),
            "number": isinstance(obj, (int, float)) and not isinstance(obj, bool),
            "boolean": isinstance(obj, bool),
        }.get(t, True)
        if not ok:
            return ["%s: expected %s" % (path, t)]
    if "enum" in schema and obj not in schema["enum"]:
        errors.append("%s: %r not in %r" % (path, obj, schema["enum"]))
    if "pattern" in schema and isinstance(obj, str) and not re.search(schema["pattern"], obj):
        errors.append("%s: %r does not match %s" % (path, obj, schema["pattern"]))
    if isinstance(obj, dict):
        for r in schema.get("required", []):
            if r not in obj:
                errors.append("%s: missing required %s" % (path, r))
        for k, sub in schema.get("properties", {}).items():
            if k in obj:
                errors += fallback_validate(obj[k], sub, path + "." + k)
    if isinstance(obj, list):
        if "minItems" in schema and len(obj) < schema["minItems"]:
            errors.append("%s: fewer than %d items" % (path, schema["minItems"]))
        if "items" in schema:
            for i, it in enumerate(obj):
                errors += fallback_validate(it, schema["items"], "%s[%d]" % (path, i))
    return errors


def validate(card, schema):
    try:
        import jsonschema  # type: ignore
        jsonschema.validate(card, schema)
        return []
    except ImportError:
        return fallback_validate(card, schema)
    except Exception as e:  # jsonschema.ValidationError
        return [str(e).splitlines()[0]]


def arch_of(batch):
    dev = batch.get("node", {}).get("device", {}) or {}
    name = (dev.get("name") or "").upper()
    host = (batch.get("node", {}).get("hostname") or "").lower()
    if "GB200" in name or "gb200" in host:
        return "gb200"
    if "B200" in name or "b200" in host:
        return "b200"
    return None


def rel_to_swarm(path):
    try:
        return os.path.relpath(os.path.abspath(path), SWARM)
    except ValueError:
        return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", action="append", required=True, help="harness batch JSON (repeatable)")
    ap.add_argument("--findings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--schema", default=os.path.join(SWARM, "cards", "schema.json"))
    ap.add_argument("--arch", help="override arch (b200|gb200)")
    ap.add_argument("--allow-missing-control", action="store_true",
                    help="keep cells without a control value (control_value set to the variant value, flagged)")
    args = ap.parse_args()

    schema = json.load(open(args.schema))
    findings = json.load(open(args.findings))
    batches = [json.load(open(b)) for b in args.batch]
    b0 = batches[0]
    adapter_id = b0["adapter"]["id"]
    approach_id = findings.get("approach_id") or adapter_id.split("_")[0]
    variant = findings.get("variant") or (adapter_id.split("_", 1)[1] if "_" in adapter_id else adapter_id)

    arch = args.arch or arch_of(b0)
    if not arch:
        raise SystemExit("cannot infer arch from the batch, pass --arch")

    cells, notes, cases, oracle_all_ok = [], [], 0, True
    counted_runs = set()
    for b, bpath in zip(batches, args.batch):
        if arch_of(b) and arch_of(b) != arch:
            raise SystemExit("batch %s is on %s, card is %s: no cross-arch cards" % (bpath, arch_of(b), arch))
        if b["adapter"]["id"] != adapter_id:
            raise SystemExit("batch %s is for adapter %s, not %s" % (bpath, b["adapter"]["id"], adapter_id))
        for f in b.get("failed", []):
            oracle_all_ok = oracle_all_ok and f.get("reason") != "oracle_failed"
            notes.append("dropped %s (%s)%s" % (f["cell"], f["reason"],
                                               (": " + "; ".join(f.get("failures", [])[:3])) if f.get("failures") else ""))
        for u in b.get("unsupported", []):
            notes.append("unsupported %s" % u["cell"])
        for c in b.get("cells", []):
            raw = c.get("raw", {})
            v = raw.get("variant") or {}
            run_key = (bpath, raw.get("cell_json"))
            if run_key not in counted_runs:
                counted_runs.add(run_key)
                cases += int(v.get("oracle_cases", 0))
            if not v.get("oracle_passed", False):
                oracle_all_ok = False
                notes.append("dropped cell N%d k%d %s %s %s: oracle failed" % (c["n"], c["k"], c["key_type"],
                                                                               c["requirement"], c["regime"]))
                continue
            if "control_value" not in c:
                if args.allow_missing_control:
                    c["control_name"] = "none"
                    c["control_value"] = c["value"]
                    notes.append("cell N%d k%d %s %s %s has no control (flagged)" % (
                        c["n"], c["k"], c["key_type"], c["requirement"], c["regime"]))
                else:
                    notes.append("dropped cell N%d k%d %s %s %s: no control value" % (
                        c["n"], c["k"], c["key_type"], c["requirement"], c["regime"]))
                    continue
            cell = {k: c[k] for k in CELL_KEYS if k in c}
            cell["resources"] = {k: c["resources"][k] for k in RES_KEYS if k in c["resources"]}
            cell["raw_file"] = rel_to_swarm(bpath)
            cells.append(cell)

    if findings.get("correctness_notes"):
        notes.append(findings["correctness_notes"])
    control_ids = sorted({c["control_name"] for c in cells})
    if b0.get("control_is_aa"):
        notes.append("control is an A/A run of the same adapter in the same process (run-to-run noise reference)")

    card = {
        "approach_id": approach_id,
        "variant": variant,
        "scope": b0["adapter"]["scope"],
        "arch": arch,
        "cuda_version": b0["node"].get("nvcc_version", "unknown"),
        "node": b0["node"].get("hostname", "unknown"),
        "date": b0.get("date"),
        "commit": findings.get("commit") or (b0.get("repo", {}).get("commit", "")[:12] or "unknown"),
        "author": findings["author"],
        "contract": findings["contract"],
        "correctness": {"oracle_passed": oracle_all_ok, "cases": cases, "notes": " | ".join(notes)},
        "cells": cells,
        "findings": findings["findings"],
        "techniques_adopted": findings.get("techniques_adopted", []),
        "techniques_contributed": findings.get("techniques_contributed", []),
        "report_path": findings["report_path"],
        "source_paths": findings["source_paths"],
    }
    if findings.get("verification"):
        card["verification"] = findings["verification"]
    card["_controls"] = control_ids
    card["_batches"] = [rel_to_swarm(p) for p in args.batch]
    card["_canary"] = [b.get("canary") for b in batches]

    errors = validate(card, schema)
    if errors:
        for e in errors:
            print("schema error:", e, file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(card, open(args.out, "w"), indent=1)
    print("wrote %s: %d cells, oracle %s, %d cases, controls %s" % (
        args.out, len(cells), "passed" if oracle_all_ok else "FAILED", cases, control_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
