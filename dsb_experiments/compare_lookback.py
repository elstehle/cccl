#!/usr/bin/env python3
# Compare two nvbench json outputs (baseline vs aggregate-first) per benchmark state.
import json
import sys


def load(path):
    with open(path) as f:
        d = json.load(f)
    out = {}
    for b in d["benchmarks"]:
        for s in b["states"]:
            if s.get("is_skipped"):
                continue
            key = (b["name"], tuple((a["name"], a["value"]) for a in s["axis_values"]))
            t = None
            for summ in s["summaries"]:
                if summ["tag"] == "nv/cold/time/gpu/mean":
                    t = float(next(x["value"] for x in summ["data"] if x["name"] == "value"))
            out[key] = t
    return out


base, af = load(sys.argv[1]), load(sys.argv[2])
rows = []
for key, tb in base.items():
    ta = af.get(key)
    if ta is None or tb is None:
        continue
    axes = " ".join(f"{n}={v}" for n, v in key[1])
    rows.append((key[0], axes, tb * 1e6, ta * 1e6, (ta - tb) / tb * 100.0))

print(f"{'axes':<60} {'base us':>10} {'af us':>10} {'delta%':>8}")
cur = None
for name, axes, tb, ta, d in rows:
    if name != cur:
        cur = name
        print(f"\n== {name} ==")
    print(f"  {axes:<58} {tb:>10.2f} {ta:>10.2f} {d:>+7.2f}%")
