#!/usr/bin/env python3
"""Compact table of the per-cell JSON files of a harness batch (the log dir on the node or the
merged batch file is not needed): variant and control latency and throughput next to each other.
  python3 celltab.py <dir or files...>"""
import glob
import json
import os
import sys


def num(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return float("nan")
        d = d[p]
    return d


def main():
    files = []
    for a in sys.argv[1:]:
        files += sorted(glob.glob(os.path.join(a, "*.json"))) if os.path.isdir(a) else [a]
    rows = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:  # noqa
            continue
        if "cell" not in d:
            continue
        c = d["cell"]
        v, k = d.get("variant", {}), d.get("control", {})
        rows.append((c["key_type"], c["requirement"][:3], c["n"], c["k"], d.get("runs", 0),
                     "ok" if v.get("oracle_passed") else "FAIL", v.get("oracle_cases", 0),
                     num(v, "latency", "cycles_per_call"), num(k, "latency", "cycles_per_call"),
                     num(v, "latency", "r2_raw"),
                     num(v, "throughput", "gelem_per_s"), num(k, "throughput", "gelem_per_s"),
                     num(v, "latency", "resources", "registers"), num(v, "latency", "resources", "stack_bytes"),
                     num(v, "latency", "resources", "smem_bytes"), v.get("id", "?"), k.get("id", "?")))
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    print("%-4s %-3s %5s %5s %4s %-4s %3s | %9s %9s %6s %6s | %7s %7s %6s | %4s %5s %6s" % (
        "key", "req", "N", "k", "runs", "orc", "cs", "lat", "ctrl", "ratio", "r2", "thr", "ctrl", "ratio", "regs", "stack", "smem"))
    for r in rows:
        lr = r[8] / r[7] if r[7] else float("nan")
        tr = r[10] / r[11] if r[11] else float("nan")
        print("%-4s %-3s %5d %5d %4d %-4s %3d | %9.0f %9.0f %6.2f %6.4f | %7.1f %7.1f %6.2f | %4.0f %5.0f %6.0f" % (
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], lr, r[9], r[10], r[11], tr, r[12], r[13], r[14]))
    if rows:
        print("variant %s control %s" % (rows[0][15], rows[0][16]))


if __name__ == "__main__":
    main()
