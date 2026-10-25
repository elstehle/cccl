#!/usr/bin/env python3
"""Regenerate RESULTS_BOARD.md from cards/*.json. Verified cards first, then unverified.

  python3 tools/board.py [--cards <dir>] [--out <file>]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWARM = os.path.dirname(HERE)


def load_cards(card_dir):
    cards = []
    for p in sorted(glob.glob(os.path.join(card_dir, "*.json"))):
        if os.path.basename(p) == "schema.json":
            continue
        try:
            c = json.load(open(p))
        except Exception as e:
            print("skipping %s: %s" % (p, e), file=sys.stderr)
            continue
        c["_file"] = os.path.basename(p)
        cards.append(c)
    return cards


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "%.4g" % v
    return str(v)


def ratio(cell):
    v, c = cell.get("value"), cell.get("control_value")
    if not v or not c:
        return ""
    if cell["regime"] == "latency":
        return "%.2fx" % (c / v)  # control cycles over variant cycles: speedup
    return "%.2fx" % (v / c)  # variant throughput over control throughput


def card_rows(card):
    rows = []
    for cell in card.get("cells", []):
        rows.append({
            "approach": card["approach_id"], "variant": card["variant"], "arch": card["arch"],
            "scope": card["scope"], "key": cell["key_type"], "n": cell["n"], "k": cell["k"],
            "threads": cell.get("threads", ""), "req": cell["requirement"], "pattern": cell["input_pattern"],
            "regime": cell["regime"], "metric": cell["metric"], "value": cell["value"],
            "control": cell.get("control_name", ""), "control_value": cell.get("control_value"),
            "ratio": ratio(cell), "regs": cell["resources"].get("registers", ""),
            "spill": cell["resources"].get("spill_bytes", ""), "card": card["_file"],
        })
    return rows


def table(rows):
    if not rows:
        return "(none)\n"
    hdr = ["approach", "variant", "arch", "scope", "key", "N", "k", "req", "pattern", "value", "control",
           "control value", "ratio", "regs", "spill B", "card"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    rows = sorted(rows, key=lambda r: (r["scope"], r["key"], r["n"], r["k"], r["req"], r["approach"], r["variant"]))
    for r in rows:
        out.append("| " + " | ".join([
            r["approach"], r["variant"], r["arch"], r["scope"], r["key"], str(r["n"]), str(r["k"]), r["req"],
            r["pattern"], fmt(r["value"]), r["control"], fmt(r["control_value"]), r["ratio"], fmt(r["regs"]),
            fmt(r["spill"]), r["card"]]) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", default=os.path.join(SWARM, "cards"))
    ap.add_argument("--out", default=os.path.join(SWARM, "RESULTS_BOARD.md"))
    args = ap.parse_args()
    cards = load_cards(args.cards)
    verified = [c for c in cards if c.get("verification", {}).get("verified")]
    unverified = [c for c in cards if not c.get("verification", {}).get("verified")]

    lines = ["# Results board", "",
             "Generated from `cards/*.json` by `tools/board.py`. Do not edit. Verified cards are listed before "
             "unverified ones. Latency values are cycles per call (ratio = control over variant, above 1 is a "
             "win), throughput values are Gelem/s (ratio = variant over control). Every value carries its "
             "control from the same lock section, see the card for canaries, resources and findings.", ""]
    if not cards:
        lines.append("(no cards yet)")
    lines.append("## Cards")
    lines.append("")
    if cards:
        lines.append("| card | approach | variant | scope | arch | node | CUDA | date | author | cells | oracle | verified |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in verified + unverified:
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %d | %s | %s |" % (
                c["_file"], c["approach_id"], c["variant"], c["scope"], c["arch"], c["node"], c["cuda_version"],
                c["date"], c["author"], len(c.get("cells", [])),
                "pass" if c["correctness"]["oracle_passed"] else "FAIL",
                "yes" if c.get("verification", {}).get("verified") else "no"))
        lines.append("")
    for group_name, group in (("Verified", verified), ("Unverified", unverified)):
        for regime in ("latency", "throughput", "batched"):
            rows = [r for c in group for r in card_rows(c) if r["regime"] == regime]
            if not rows:
                continue
            lines.append("## %s cards, %s" % (group_name, regime))
            lines.append("")
            lines.append(table(rows))
    open(args.out, "w").write("\n".join(lines).rstrip("\n") + "\n")
    print("wrote %s: %d cards (%d verified), %d rows" % (
        args.out, len(cards), len(verified), sum(len(c.get("cells", [])) for c in cards)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
