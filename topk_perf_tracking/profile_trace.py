#!/usr/bin/env python3
"""Extract a clean kernel-by-kernel trace of a single TopK dispatch from an
nsys cuda_gpu_trace CSV.

The full trace includes setup work (data generation, fills, RNG) before the
actual TopK call. The TopK pipeline is the contiguous run of kernels in
stream != main (typically the second-to-last stream) after the last setup
kernel. We isolate it by looking for the kernels with topk-shaped names
(DeviceTopK*, device_segmented_topk_*).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


TOPK_KERNEL_PATTERNS = [
    re.compile(r"DeviceTopK(\w*)Kernel"),
    re.compile(r"device_segmented_topk(_\w+)_kernel"),
]


def short_kernel_name(full: str) -> str:
    """Compact a fully-mangled C++ name down to the TopK kernel short tag."""
    for pat in TOPK_KERNEL_PATTERNS:
        m = pat.search(full)
        if m:
            base = m.group(0)
            # Strip duplicate kernel tag if present
            if "DeviceTopK" in base:
                return base
            return base
    return "(unknown)"


def parse(path: Path) -> list[dict]:
    rows = []
    # Skip nsys preamble lines (NOTICE, blank lines) until the CSV header is found.
    lines = path.read_text().splitlines()
    csv_start = None
    for i, line in enumerate(lines):
        if line.startswith("Start (ns),Duration (ns)"):
            csv_start = i
            break
    if csv_start is None:
        raise SystemExit(f"no CSV header found in {path}")
    reader = csv.DictReader(lines[csv_start:])
    for r in reader:
        try:
            dur = int(r.get("Duration (ns)", "0"))
            start = int(r.get("Start (ns)", "0"))
        except (TypeError, ValueError):
            continue
        name = r.get("Name", "")
        rows.append({"start": start, "dur": dur, "name": name, "stream": r.get("Strm", "")})
    rows.sort(key=lambda r: r["start"])
    return rows


def isolate_topk(rows: list[dict]) -> list[dict]:
    out = []
    started = False
    for r in rows:
        is_topk = any(p.search(r["name"]) for p in TOPK_KERNEL_PATTERNS)
        # Pick up memsets in the same stream as topk kernels (they bracket the dispatch).
        if is_topk:
            started = True
            out.append(r)
            continue
        if started and r["name"].startswith("[CUDA"):
            # Trailing memset still counts as part of the dispatch
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--label", default="trace")
    args = ap.parse_args()

    rows = parse(args.trace)
    # The first TopK kernel marks the start of the dispatch. Include any memsets
    # immediately preceding it (within ~1 us).
    topk = []
    first_topk_idx = next(
        (i for i, r in enumerate(rows) if any(p.search(r["name"]) for p in TOPK_KERNEL_PATTERNS)),
        None,
    )
    if first_topk_idx is None:
        raise SystemExit("no TopK kernels in trace")
    # Look backwards for memsets in same/nearby stream that are within 50 us.
    first_topk = rows[first_topk_idx]
    for r in rows[max(0, first_topk_idx - 5):first_topk_idx]:
        if r["name"].startswith("[CUDA") and (first_topk["start"] - r["start"]) < 50_000:
            topk.append(r)
    for r in rows[first_topk_idx:]:
        if any(p.search(r["name"]) for p in TOPK_KERNEL_PATTERNS) or r["name"].startswith("[CUDA"):
            topk.append(r)
        else:
            # Stop at the first non-topk / non-memset kernel
            break

    if not topk:
        raise SystemExit("could not isolate TopK dispatch")

    # Renormalize start times relative to first TopK kernel
    t0 = topk[0]["start"]
    total_dur = sum(r["dur"] for r in topk)
    span = topk[-1]["start"] + topk[-1]["dur"] - topk[0]["start"]

    print(f"# Kernel trace -- `{args.label}`")
    print()
    print(f"- kernels: **{len(topk)}**")
    print(f"- sum of kernel durations: **{total_dur/1000:.2f} us**")
    print(f"- wall-time span (first start -> last end): **{span/1000:.2f} us**")
    print()
    print("| # | start (us, rel) | duration (us) | kernel |")
    print("|---:|---:|---:|---|")
    for i, r in enumerate(topk):
        rel = (r["start"] - t0) / 1000.0
        dur = r["dur"] / 1000.0
        name = short_kernel_name(r["name"]) if not r["name"].startswith("[CUDA") else r["name"]
        print(f"| {i+1} | {rel:.2f} | {dur:.2f} | `{name}` |")


if __name__ == "__main__":
    sys.exit(main() or 0)
