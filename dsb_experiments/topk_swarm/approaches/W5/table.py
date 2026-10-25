#!/usr/bin/env python3
"""Compact variant against control table of one or more harness batches, for the report."""
import json
import sys

for path in sys.argv[1:]:
    b = json.load(open(path))
    print(b["batch"], "adapter", b["adapter"], "control", b["control"])
    rows = {}
    for c in b["cells"]:
        key = (c["key_type"], c["n"], c["k"], c["requirement"][:4])
        rows.setdefault(key, {})[c["regime"]] = (c["value"], c["control_value"], c["resources"].get("registers"))
    print(f"{'kt':5}{'N':>6}{'k':>5} req   {'lat':>8}{'ctl':>8}{'gain':>6} | {'thr':>8}{'ctl':>8}{'gain':>6}  regs lat/thr")
    for key in sorted(rows):
        r = rows[key]
        lat = r.get("latency", (None, None, None))
        thr = r.get("throughput", (None, None, None))
        def g(v, c, inv=False):
            if v is None or c is None or not v or not c:
                return "   -"
            return f"{(c / v if not inv else v / c):6.2f}"
        print(f"{key[0]:5}{key[1]:6}{key[2]:5} {key[3]:5}"
              f"{(lat[0] or 0):8.0f}{(lat[1] or 0):8.0f}{g(lat[0], lat[1])} |"
              f"{(thr[0] or 0):8.1f}{(thr[1] or 0):8.1f}{g(thr[0], thr[1], True)}  {lat[2]}/{thr[2]}")
    print()
