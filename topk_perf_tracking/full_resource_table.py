#!/usr/bin/env python3
"""Print absolute resource numbers (regs + stack) per (logical_kernel, KeyT,
ValueT) across N snapshots. Separates keys-only kernels from pair kernels.

Usage:
    full_resource_table.py NAME1=snap1.json NAME2=snap2.json ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


KEY_LABEL = {
    "signed char": "I8",
    "short": "I16",
    "int": "I32",
    "long": "I64",
    "__int128": "I128",
    "float": "F32",
    "double": "F64",
}

KEY_ORDER = ["signed char", "short", "int", "long", "__int128", "float", "double"]
VALUE_ORDER_PAIRS = ["signed char", "short", "int", "long"]


def normalise_value(v):
    if v is None or (isinstance(v, str) and "NullType" in v):
        return None
    return v


LOGICAL_ORDER = [
    "initial_histogram",
    "finalize_histogram",
    "filter",
    "finalize_filter",
    "last_filter",
    "single_cta",
]


def load(path):
    return json.load(open(path))["kernels"]


def aggregate(records):
    """Group by (logical, key_t, normalised value_t). Pick worst (max)
    registers and stack across OffsetT / OutOffsetT variants. Only keep
    `select=max` (min/max are byte-identical in dev tracking)."""
    records = [r for r in records if r.get("select", "max") == "max"]
    by_key = defaultdict(list)
    for r in records:
        if r["key_t"] is None:
            continue
        v = normalise_value(r["value_t"])
        by_key[(r["logical_name"], r["key_t"], v)].append(r)
    out = {}
    for key, group in by_key.items():
        regs = max(r["registers"] for r in group if r["registers"] is not None)
        stack = max(r["stack_frame"] for r in group if r["stack_frame"] is not None)
        out[key] = (regs, stack)
    return out


def fmt_pair(regs, stack):
    return f"{regs}/{stack}"


def emit_table(title, snapshots, names, kernel_logical, key_value_pairs):
    # Skip rows where no snapshot has this (kernel, KeyT, ValueT) triple.
    filtered_rows = []
    for kt, vt in key_value_pairs:
        if any((kernel_logical, kt, vt) in snapshots[n] for n in names):
            filtered_rows.append((kt, vt))
    if not filtered_rows:
        return
    print(f"### `{kernel_logical}` -- {title}")
    print()
    header = ["KeyT", "ValueT"] + [f"{n} regs/stack" for n in names]
    sep = ["---", "---"] + ["---:"] * len(names)
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(sep) + "|")
    for kt, vt in filtered_rows:
        kt_lbl = KEY_LABEL.get(kt, kt)
        vt_lbl = KEY_LABEL.get(vt, vt) if vt is not None else "(K only)"
        row = [kt_lbl, vt_lbl]
        for n in names:
            if (kernel_logical, kt, vt) in snapshots[n]:
                regs, stack = snapshots[n][(kernel_logical, kt, vt)]
                row.append(fmt_pair(regs, stack))
            else:
                row.append("-")
        print("| " + " | ".join(row) + " |")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snaps", nargs="+", help="NAME=path.json")
    args = ap.parse_args()

    names = []
    snapshots = {}
    for s in args.snaps:
        n, p = s.split("=", 1)
        names.append(n)
        snapshots[n] = aggregate(load(p))

    # Discover all (logical, K, V) triples covered.
    all_triples = set()
    for s in snapshots.values():
        all_triples.update(s)

    # Split into pair triples and keys-only triples.
    def order_kv_pairs():
        out = []
        for kt in KEY_ORDER:
            for vt in VALUE_ORDER_PAIRS:
                out.append((kt, vt))
        return out

    def order_kv_keys_only():
        return [(kt, None) for kt in KEY_ORDER]

    pair_kv = order_kv_pairs()
    keys_only_kv = order_kv_keys_only()

    print("# Absolute resource numbers per snapshot")
    print()
    print(f"Snapshots: {', '.join(f'`{n}`' for n in names)}.")
    print()
    print("Format `regs/stack` (worst-case across OffsetT/OutOffsetT variants; `select=max` only -- min/max are byte-identical in dev tracking).")
    print()

    print("## Pair kernels (`pairs.base`)")
    print()
    for kernel in LOGICAL_ORDER:
        present = any((kernel, kt, vt) in s for s in snapshots.values() for kt, vt in pair_kv)
        if not present:
            continue
        emit_table("pairs", snapshots, names, kernel, pair_kv)

    print("## Keys-only kernels (`keys.base`)")
    print()
    for kernel in LOGICAL_ORDER:
        present = any((kernel, kt, None) in s for s in snapshots.values() for kt in KEY_ORDER)
        if not present:
            continue
        emit_table("keys-only", snapshots, names, kernel, keys_only_kv)


if __name__ == "__main__":
    sys.exit(main())
