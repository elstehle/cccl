#!/usr/bin/env python3
"""Aggregate sweep delta per (KeyT, ValueT) pair.

For each (KeyT, ValueT) combination, compute the geometric mean, worst (max
ratio), best (min ratio), median of `candidate/baseline`. Output a markdown
table sorted by mean ratio (worst regressions first).

Usage:
  aggregate_sweep_per_kv.py baseline=path1.json candidate=path2.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def axis_value(a):
    t = a.get("type")
    if t == "string":
        return a.get("value")
    v = a.get("value")
    if v is None:
        return None
    if t == "int64":
        return int(v)
    if t in ("float64", "double"):
        return float(v)
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v


def summary_value(s, tag):
    if not s:
        return None
    for s_ in s:
        if s_.get("tag") == tag:
            for d in s_.get("data", []):
                if d.get("name") == "value":
                    try:
                        return float(d["value"])
                    except (TypeError, ValueError):
                        return None
    return None


def load(path):
    d = json.loads(Path(path).read_text())
    out = {}
    for bm in d.get("benchmarks", []):
        for st in bm.get("states", []):
            params = {a["name"]: axis_value(a) for a in st.get("axis_values", [])}
            key = (
                params.get("KeyT{ct}"),
                params.get("ValueT{ct}"),
                params.get("OffsetT{ct}"),
                params.get("OutOffsetT{ct}"),
                params.get("Elements{io}"),
                params.get("SelectedElements"),
                params.get("Entropy"),
            )
            gpu = summary_value(st.get("summaries", []), "nv/cold/time/gpu/mean")
            if gpu is None:
                continue
            out[key] = gpu * 1e6
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps", nargs=2, help="baseline=path1.json candidate=path2.json")
    args = ap.parse_args()

    bname, bpath = args.sweeps[0].split("=", 1)
    cname, cpath = args.sweeps[1].split("=", 1)
    base = load(bpath)
    cand = load(cpath)
    common = sorted(set(base) & set(cand))

    by_kv = defaultdict(list)
    for k in common:
        KeyT, ValueT, *_ = k
        ratio = cand[k] / base[k]
        by_kv[(KeyT, ValueT)].append(ratio)

    rows = []
    for (KeyT, ValueT), ratios in by_kv.items():
        n = len(ratios)
        geo_mean = math.exp(statistics.mean(math.log(r) for r in ratios))
        median = statistics.median(ratios)
        worst = max(ratios)
        best = min(ratios)
        p90_idx = max(0, int(0.9 * n) - 1)
        sorted_r = sorted(ratios)
        p90 = sorted_r[p90_idx]
        p10 = sorted_r[max(0, int(0.1 * n) - 1)]
        rows.append((KeyT, ValueT, n, geo_mean, median, p10, p90, best, worst))

    rows.sort(key=lambda r: r[3], reverse=True)

    print(f"# Per-(KeyT, ValueT) summary: `{cname}` vs `{bname}` baseline")
    print()
    print(f"- Baseline: `{bpath}`")
    print(f"- Candidate: `{cpath}`")
    print(f"- Sorted by geometric mean ratio (worst regression first).")
    print()
    print("| KeyT | ValueT | n | geo mean | median | p10 | p90 | best | worst |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for KeyT, ValueT, n, gm, med, p10, p90, best, worst in rows:
        print(
            f"| `{KeyT}` | `{ValueT}` | {n} | {gm:.3f}x | {med:.3f}x | {p10:.3f}x | {p90:.3f}x | {best:.3f}x | {worst:.3f}x |"
        )


if __name__ == "__main__":
    sys.exit(main())
