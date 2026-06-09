#!/usr/bin/env python3
"""Parse an nvbench JSON from the segmented-topk benchmark into a GPU-time table.

Usage: parse_sweep.py <nvbench.json>
Prints rows: SegSize  K  NSeg  Pattern  GPU_us
"""
import json
import sys


def main(path):
    d = json.load(open(path))
    rows = []
    for bench in d["benchmarks"]:
        for s in bench["states"]:
            av = {a["name"]: a["value"] for a in s["axis_values"]}
            gt = None
            for x in s["summaries"]:
                if x["name"] == "GPU Time":
                    for it in x["data"]:
                        if it["name"] == "value":
                            gt = float(it["value"])
            rows.append((
                int(av["MaxSegmentSize{ct}"]),
                int(av["K{ct}"]),
                int(av["NumSegments"]),
                av.get("Pattern", "?"),
                (gt * 1e6) if gt is not None else float("nan"),
            ))
    rows.sort()
    print("%10s %6s %5s %-16s %12s" % ("SegSize", "K", "NSeg", "Pattern", "GPU_us"))
    for r in rows:
        print("%10d %6d %5d %-16s %12.2f" % r)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: parse_sweep.py <nvbench.json>")
    main(sys.argv[1])
