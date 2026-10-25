#!/usr/bin/env python3
"""Compare two harness batch JSONs cell by cell (same adapter, same cells, e.g. a rerun on another
node or a later harness version).

  python3 tools/compare_batches.py <reference batch.json> <other batch.json> [--tolerance 0.03] [--md]

Prints one row per common cell with both values and the ratio other over reference, the cells
outside the tolerance, and the summary (min, max, median ratio per regime). Exit code 1 when any
cell is outside the tolerance, so the call doubles as a check.
"""
import argparse
import json
import statistics
import sys


def cells_of(batch):
    out = {}
    for c in batch.get("cells", []):
        key = (c["key_type"], c["requirement"], c["input_pattern"], c["n"], c["k"], c.get("threads"), c["regime"])
        out[key] = c
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("other")
    ap.add_argument("--tolerance", type=float, default=0.03, help="relative deviation that counts as different")
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()
    a = json.load(open(args.reference))
    b = json.load(open(args.other))
    ca, cb = cells_of(a), cells_of(b)
    common = sorted(set(ca) & set(cb), key=lambda k: (k[6], k[0], k[1], k[3], k[4]))
    print("reference: %s on %s (%s), other: %s on %s (%s), %d common cells, %d only in reference, %d only in other" % (
        a["batch"], a["node"].get("hostname"), a["node"].get("arch"), b["batch"], b["node"].get("hostname"),
        b["node"].get("arch"), len(common), len(set(ca) - set(cb)), len(set(cb) - set(ca))))
    hdr = ["regime", "key", "req", "N", "k", "reference", "other", "ratio", "flag"]
    if args.md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
    ratios = {}
    outside = []
    for key in common:
        va, vb = ca[key]["value"], cb[key]["value"]
        r = vb / va if va else float("nan")
        ratios.setdefault(key[6], []).append(r)
        flag = "" if abs(r - 1) <= args.tolerance else "OUTSIDE"
        if flag:
            outside.append(key)
        row = [key[6], key[0], key[1], str(key[3]), str(key[4]), "%.4g" % va, "%.4g" % vb, "%.3f" % r, flag]
        if args.md:
            print("| " + " | ".join(row) + " |")
        else:
            print("%-10s %-5s %-20s %6s %5s %10s %10s %7s %s" % tuple(row))
    for regime, rs in ratios.items():
        print("%s: ratio other/reference min %.3f median %.3f max %.3f over %d cells" % (
            regime, min(rs), statistics.median(rs), max(rs), len(rs)))
    print("%d of %d cells outside %.0f percent" % (len(outside), len(common), args.tolerance * 100))
    return 1 if outside else 0


if __name__ == "__main__":
    sys.exit(main())
