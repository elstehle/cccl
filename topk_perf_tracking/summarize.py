#!/usr/bin/env python3
"""Generate a compact summary report across all logical kernels and metrics.

For each (logical_name, metric), show:
  - baseline min/max
  - target min/max
  - max absolute delta (signed, with the (KeyT, ValueT) where it occurs)
  - net delta (target_sum - baseline_sum)

Usage:
    summarize.py BASELINE TARGET [--out PATH]

Both BASELINE and TARGET are snapshot files (produced by build_snapshot.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


LOGICAL_NAMES = (
    "filter",
    "last_filter",
    "initial_histogram",
    "finalize_filter",
    "finalize_histogram",
    "single_cta",
)

METRICS = ("registers", "smem_bytes", "stack_frame", "spill_stores", "spill_loads")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def index_kernels(snapshot: dict) -> dict:
    """Return mapping (logical_name, key_t, value_t) -> kernel record."""
    table: dict = {}
    for k in snapshot["kernels"]:
        key = (k["logical_name"], k["key_t"], k["value_t"])
        prev = table.get(key)
        if prev is None:
            table[key] = k
        else:
            if k.get("select") == "max" and prev.get("select") != "max":
                table[key] = k
    return table


def fmt(d: int) -> str:
    if d > 0:
        return f"+{d}"
    if d < 0:
        return str(d)
    return "0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("target", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    base = load(args.baseline)
    targ = load(args.target)
    base_idx = index_kernels(base)
    targ_idx = index_kernels(targ)

    base_label = base["metadata"]["label"]
    targ_label = targ["metadata"]["label"]

    lines = []
    lines.append(
        f"# Resource usage summary: `{base_label}` (baseline) -> `{targ_label}` (target)"
    )
    lines.append("")
    lines.append("## Build context")
    lines.append("")
    lines.append("| label | branch | sha | subject |")
    lines.append("|---|---|---|---|")
    for s in (base, targ):
        md = s["metadata"]
        subject = (md.get("commit_subject") or "").replace("|", "\\|")
        if len(subject) > 80:
            subject = subject[:77] + "..."
        lines.append(
            f"| `{md['label']}` | `{md.get('branch','')}` |"
            f" `{md.get('commit_sha_short','')}` | {subject} |"
        )
    lines.append("")
    lines.append(
        "## Per-kernel deltas (target - baseline, summed across all (KeyT, ValueT))"
    )
    lines.append("")
    header = ["kernel"] + list(METRICS) + ["instances (base/target)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for lname in LOGICAL_NAMES:
        base_keys = {k for k in base_idx if k[0] == lname}
        targ_keys = {k for k in targ_idx if k[0] == lname}
        all_keys = base_keys | targ_keys
        if not all_keys:
            continue
        row = [f"`{lname}`"]
        for m in METRICS:
            sum_delta = 0
            for k in all_keys:
                b = base_idx.get(k)
                t = targ_idx.get(k)
                bv = b.get(m) if b else None
                tv = t.get(m) if t else None
                if bv is None or tv is None:
                    continue
                sum_delta += tv - bv
            row.append(fmt(sum_delta))
        row.append(f"{len(base_keys)}/{len(targ_keys)}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(
        "## Per-kernel max-impact (KeyT, ValueT) configurations"
    )
    lines.append("")
    lines.append(
        "For each (logical_name, metric), the (KeyT, ValueT) combination with the largest signed delta."
    )
    lines.append("")
    header = ["kernel", "metric", "KeyT", "ValueT", "baseline", "target", "Δ"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for lname in LOGICAL_NAMES:
        keys_for_lname = sorted({k for k in (base_idx.keys() | targ_idx.keys()) if k[0] == lname})
        if not keys_for_lname:
            continue
        for m in METRICS:
            best = None
            for k in keys_for_lname:
                b = base_idx.get(k)
                t = targ_idx.get(k)
                if not b or not t:
                    continue
                bv = b.get(m)
                tv = t.get(m)
                if bv is None or tv is None:
                    continue
                delta = tv - bv
                if best is None or abs(delta) > abs(best[2]):
                    best = (bv, tv, delta, k[1], k[2])
            if best is None:
                continue
            bv, tv, delta, kt, vt = best
            if delta == 0:
                continue
            lines.append(
                f"| `{lname}` | `{m}` | {kt or '?'} | {vt or '?'} |"
                f" {bv} | {tv} | {fmt(delta)} |"
            )
    lines.append("")
    out = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out)
        print(f"Wrote {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    sys.exit(main() or 0)
