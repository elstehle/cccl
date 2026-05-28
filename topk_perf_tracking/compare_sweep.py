#!/usr/bin/env python3
"""Compare N nvbench sweeps. Joins by (KeyT, ValueT, OffsetT, OutOffsetT,
Elements, SelectedElements, Entropy) and prints absolute GPU times plus
deltas relative to the first ("baseline") sweep.

Usage:
    compare_sweep.py NAME1=path1.json NAME2=path2.json ...
"""

from __future__ import annotations

import argparse
import json
import sys
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
            noise = summary_value(st.get("summaries", []), "nv/cold/time/gpu/stdev/relative")
            if gpu is None:
                continue
            out[key] = {"time_us": gpu * 1e6, "noise_pct": (noise or 0) * 100}
    return out


def fmt_us(t):
    if t is None:
        return "skip"
    if t < 1.0:
        return f"{t:.3f}"
    if t < 1000:
        return f"{t:6.2f}"
    if t < 1000000:
        return f"{t/1000:6.2f}ms"
    return f"{t/1000000:6.2f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps", nargs="+", help="NAME=path.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    names = []
    sweeps = {}
    for arg in args.sweeps:
        name, path = arg.split("=", 1)
        names.append(name)
        sweeps[name] = load(path)

    baseline = names[0]
    common = sorted(set.intersection(*[set(s) for s in sweeps.values()]))

    lines = []
    lines.append(f"# Sweep comparison: baseline = `{baseline}`")
    lines.append("")
    cols = ["KeyT", "ValueT", "Elements", "SelectedElems", "Entropy"]
    for n in names:
        cols.append(f"{n} (us)")
    for n in names[1:]:
        cols.append(f"{n}/{baseline}")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")

    for key in common:
        KeyT, ValueT, _, _, E, S, Ent = key
        row = [KeyT or "", ValueT or "", str(E), str(S), str(Ent)]
        bt = sweeps[baseline][key]["time_us"]
        for n in names:
            row.append(fmt_us(sweeps[n][key]["time_us"]))
        for n in names[1:]:
            t = sweeps[n][key]["time_us"]
            row.append(f"{t/bt:.3f}x")
        lines.append("| " + " | ".join(row) + " |")

    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    sys.exit(main() or 0)
