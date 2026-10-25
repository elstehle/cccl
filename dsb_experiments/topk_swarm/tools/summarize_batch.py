#!/usr/bin/env python3
"""Print a compact table of a harness batch JSON (state/results/<holder>/<batch>.json).

  python3 tools/summarize_batch.py <batch.json> [--md]

One row per (key, N, k, requirement): latency value with its control, raw and no-op slopes, fit
quality, throughput value with its control and the harness ceiling, occupancy, registers, spills,
tie multiplicity at rank k when the batch has it. --md prints a markdown table for reports.
"""
import json
import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    md = "--md" in sys.argv
    d = json.load(open(sys.argv[1]))
    node = d["node"]
    print("%s | node %s (%s) nvcc %s | commit %s%s | cells %d failed %d unsupported %d" % (
        d["batch"], node.get("hostname"), node.get("arch"), node.get("nvcc_version"), d["repo"]["commit"][:12],
        " dirty" if d["repo"].get("dirty") else "", len(d["cells"]), len(d["failed"]), len(d["unsupported"])))
    can = d.get("canary", {})
    for tag in ("start", "end"):
        c = can.get(tag, {})
        print("canary %-5s lat %.3f cyc/op  thr %.1f us" % (tag, c.get("lat_cycles_per_op", float("nan")),
                                                           c.get("thr_us", float("nan"))))
    for f in d["failed"]:
        print("FAILED", f)
    rows = {}
    for c in d["cells"]:
        rows.setdefault((c["key_type"], c["n"], c["k"], c["requirement"]), {})[c["regime"]] = c
    hdr = ["key", "N", "k", "req", "lat cyc", "ctrl", "raw", "noop", "r2", "thr Gelem/s", "ctrl", "ceiling", "occ",
           "regs", "spill B", "local", "ties@k"]
    if md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
    else:
        print("%-5s %6s %5s %-20s | %9s %9s %7s %7s %7s | %9s %9s %8s %4s %4s %6s %14s %7s" % tuple(hdr))
    for key in sorted(rows, key=lambda x: (x[0], x[3], x[1], x[2])):
        L = rows[key].get("latency")
        T = rows[key].get("throughput")
        lr = L["raw"]["variant"]["latency"] if L else {}
        tr = T["raw"]["variant"]["throughput"] if T else {}
        any_ = L or T
        ties = any_.get("ties_at_k_mean")
        res = any_["resources"]
        # local memory that is not a register spill: a stack frame (an array NVVM could not promote)
        # shows as stack bytes with STL and LDL in the SASS while spill bytes stay 0
        local = ""
        if res.get("stack_bytes") or res.get("stl_count") or res.get("ldl_count"):
            local = "%dB stl%d ldl%d" % (res.get("stack_bytes", 0), res.get("stl_count", 0), res.get("ldl_count", 0))
            if not res.get("spill_bytes"):
                local = "LOCAL " + local
        vals = [key[0], key[1], key[2], key[3],
                "%.0f" % L["value"] if L else "", "%.0f" % L["control_value"] if L and "control_value" in L else "",
                "%.0f" % lr.get("slope_raw", float("nan")) if L else "", "%.0f" % lr.get("slope_noop", float("nan")) if L else "",
                "%.4f" % lr.get("r2_raw", float("nan")) if L else "",
                "%.1f" % T["value"] if T else "", "%.1f" % T["control_value"] if T and "control_value" in T else "",
                "%.0f" % tr.get("gelem_per_s_noop", float("nan")) if T else "",
                str(T["resources"].get("occupancy_blocks_per_sm", "")) if T else "",
                str(res.get("registers", "")), str(res.get("spill_bytes", "")), local,
                ("%.1f" % ties) if ties is not None else ""]
        if md:
            print("| " + " | ".join(str(v) for v in vals) + " |")
        else:
            print("%-5s %6d %5d %-20s | %9s %9s %7s %7s %7s | %9s %9s %8s %4s %4s %6s %14s %7s" % tuple(vals))
    if d["unsupported"]:
        print("unsupported: %d cells (%s ...)" % (len(d["unsupported"]), d["unsupported"][0]["cell"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
