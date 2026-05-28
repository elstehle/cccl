#!/usr/bin/env python3
"""Generate a side-by-side comparison report from two or more snapshots.

Loads snapshots produced by `build_snapshot.py`, deduplicates kernel instances
by (logical_name, key_t, value_t) and prints a Markdown table that highlights
register / smem / stack / spill regressions and improvements between each
snapshot and the first one (the "baseline").

Usage:
    compare_snapshots.py BASELINE [SNAP...] --logical-name filter \
        [--key-t int] [--value-t long] [--out PATH]

If --out is provided, the Markdown report is written to that path instead of
stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_snapshot(path: Path) -> dict:
    return json.loads(path.read_text())


# Dedupe rule: if a kernel appears for both select::min and select::max, the
# resource numbers should be identical (we asserted this empirically). Keep the
# `max` variant since the benchmark uses it.
def index_kernels(snapshot: dict) -> dict:
    """Return mapping (logical_name, key_t, value_t) -> kernel record (max)."""
    table: dict = {}
    for k in snapshot["kernels"]:
        key = (k["logical_name"], k["key_t"], k["value_t"])
        prev = table.get(key)
        if prev is None:
            table[key] = k
        else:
            # Prefer select=max when both are present; otherwise keep first.
            if k.get("select") == "max" and prev.get("select") != "max":
                table[key] = k
    return table


METRICS = [
    ("registers", "regs"),
    ("smem_bytes", "smem"),
    ("stack_frame", "stack"),
    ("spill_stores", "sp.st"),
    ("spill_loads", "sp.ld"),
]


def fmt_delta(d: int) -> str:
    if d > 0:
        return f"+{d}"
    if d < 0:
        return f"{d}"
    return "0"


def fmt_value(v) -> str:
    if v is None:
        return "?"
    return str(v)


def make_markdown(
    snapshots: list[dict],
    logical_name: str,
    key_filter: str | None = None,
    value_filter: str | None = None,
) -> str:
    lines: list[str] = []

    labels = [s["metadata"]["label"] for s in snapshots]
    baseline_label = labels[0]
    other_labels = labels[1:]

    lines.append(f"# Resource usage report: `{logical_name}`")
    lines.append("")
    lines.append("## Build context")
    lines.append("")
    lines.append("| label | branch | sha | subject | target | flags | record count |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in snapshots:
        md = s["metadata"]
        subject = (md.get("commit_subject") or "").replace("|", "\\|")
        if len(subject) > 70:
            subject = subject[:67] + "..."
        flags = (md.get("build_flags") or "").replace("|", "\\|")
        lines.append(
            f"| `{md['label']}` | `{md.get('branch','')}` | `{md.get('commit_sha_short','')}`"
            f" | {subject} | `{md.get('build_target','')}` | `{flags}` | {md.get('record_count','')} |"
        )
    lines.append("")

    indexed = [index_kernels(s) for s in snapshots]

    keys = sorted(
        {k for snap in indexed for k in snap if k[0] == logical_name},
        key=lambda x: ((x[1] or ""), (x[2] or "")),
    )
    if key_filter is not None:
        keys = [k for k in keys if k[1] == key_filter]
    if value_filter is not None:
        keys = [k for k in keys if k[2] == value_filter]

    if not keys:
        lines.append("> No kernels matched the filter.")
        return "\n".join(lines)

    for metric, short in METRICS:
        lines.append(f"## `{metric}` ({short}) per (KeyT, ValueT)")
        lines.append("")
        header = ["KeyT", "ValueT", f"{baseline_label}"]
        for other in other_labels:
            header.append(f"{other}")
            header.append(f"Δ vs {baseline_label}")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for key in keys:
            row = [key[1] or "?", key[2] or "?"]
            base = indexed[0].get(key)
            base_val = None if base is None else base.get(metric)
            row.append(fmt_value(base_val))
            for snap in indexed[1:]:
                rec = snap.get(key)
                val = None if rec is None else rec.get(metric)
                row.append(fmt_value(val))
                if val is None or base_val is None:
                    row.append("—")
                else:
                    row.append(fmt_delta(val - base_val))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshots", nargs="+", type=Path)
    ap.add_argument(
        "--logical-name",
        default="filter",
        help="Logical kernel name to compare (filter / last_filter / initial_histogram / ...).",
    )
    ap.add_argument("--key-t", default=None)
    ap.add_argument("--value-t", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    snapshots = [load_snapshot(p) for p in args.snapshots]
    md = make_markdown(
        snapshots,
        logical_name=args.logical_name,
        key_filter=args.key_t,
        value_filter=args.value_t,
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md + "\n")
        print(f"Wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    sys.exit(main() or 0)
