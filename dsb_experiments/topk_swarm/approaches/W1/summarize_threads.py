#!/usr/bin/env python3
"""Compact table of a harness batch JSON with the thread count kept (sub-warp batches have several
logical warp sizes per N and k, which tools/summarize_batch.py folds into one row).

  python3 approaches/W1/summarize_threads.py <batch.json> [--md]
"""
import json
import sys


def main():
    md = "--md" in sys.argv
    d = json.load(open(sys.argv[1]))
    print("%s | node %s (%s) nvcc %s | commit %s | cells %d failed %d unsupported %d" % (
        d["batch"], d["node"].get("hostname"), d["node"].get("arch"), d["node"].get("nvcc_version"),
        d["repo"]["commit"][:12], len(d["cells"]), len(d["failed"]), len(d["unsupported"])))
    rows = {}
    for c in d["cells"]:
        rows.setdefault((c["key_type"], c["requirement"], c["threads"], c["n"], c["k"]), {})[c["regime"]] = c
    hdr = ["key", "req", "LW", "N", "k", "lat cyc", "ctrl", "speedup", "thr Gelem/s", "ctrl", "ratio", "regs lat",
           "regs thr", "occ", "spill B", "stack B"]
    if md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
    else:
        print("%-5s %-4s %3s %5s %4s | %8s %8s %7s | %10s %8s %6s | %4s %4s %3s %6s %6s" % tuple(
            h if h not in ("not_guaranteed", "prefer_smaller_index") else h[:4] for h in hdr))
    for key in sorted(rows, key=lambda x: (x[0], x[1], x[2], x[3], x[4])):
        L = rows[key].get("latency")
        T = rows[key].get("throughput")
        sp = ("%.2f" % (L["control_value"] / L["value"])) if L and L.get("control_value") and L["value"] else ""
        ra = ("%.2f" % (T["value"] / T["control_value"])) if T and T.get("control_value") else ""
        stack = ""
        if L:
            stack = str(L["resources"].get("stack_bytes", ""))
        vals = [key[0], key[1][:4], key[2], key[3], key[4],
                "%.0f" % L["value"] if L else "", "%.0f" % L["control_value"] if L and "control_value" in L else "", sp,
                "%.1f" % T["value"] if T else "", "%.1f" % T["control_value"] if T and "control_value" in T else "", ra,
                str(L["resources"].get("registers", "")) if L else "", str(T["resources"].get("registers", "")) if T else "",
                str(T["resources"].get("occupancy_blocks_per_sm", "")) if T else "",
                str((L or T)["resources"].get("spill_bytes", "")), stack]
        if md:
            print("| " + " | ".join(str(v) for v in vals) + " |")
        else:
            print("%-5s %-4s %3d %5d %4d | %8s %8s %7s | %10s %8s %6s | %4s %4s %3s %6s %6s" % tuple(vals))
    for f in d["failed"]:
        print("FAILED", f.get("cell"), f.get("reason"))


if __name__ == "__main__":
    main()
