#!/usr/bin/env python3
"""Cross-batch table of the B3 iteration chain: one row per (key type, regime, N, k), the ratio
of every batch (larger is better: control over variant for cycles, variant over control for
Gelem/s) in iteration order, and the final direct ratio against B1_reference.

  python3 -I crosstab.py <results dir>
"""
import glob
import json
import os
import sys

ORDER = ["base1", "pad2", "str3", "dig4", "lean5", "tun6", "fin7"]


def main():
    d = sys.argv[1]
    batches = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        if f.endswith(".build.json"):
            continue
        b = json.load(open(f))
        batches[b["label"]] = b
    table = {}
    for lab in ORDER:
        b = batches.get(lab)
        if not b:
            continue
        for c in b["cells"]:
            if c["requirement"] != "not_guaranteed":
                continue
            key = (c["key_type"], c["regime"], c["n"], c["k"])
            cv = c["control_value"]
            r = cv / c["value"] if c["regime"] == "latency" else c["value"] / cv
            table.setdefault(key, {})[lab] = (r, c["value"], cv, c["resources"].get("registers"), c["resources"].get("spill_bytes"))
    print("type reg      N     k | " + " ".join("%6s" % l for l in ORDER) + " | tuned value  B1 value  regs spill")
    for key in sorted(table):
        row = table[key]
        cells = " ".join(("%6.3f" % row[l][0]) if l in row else "     -" for l in ORDER)
        fin = row.get("fin7")
        tail = ("%11.1f %9.1f %5s %5s" % (fin[1], fin[2], fin[3], fin[4])) if fin else ""
        print("%4s %3s %6d %5d | %s | %s" % (key[0], key[1][:3], key[2], key[3], cells, tail))
    print()
    print("nodes per batch:", {l: batches[l]["node"]["hostname"] for l in ORDER if l in batches})


if __name__ == "__main__":
    main()
