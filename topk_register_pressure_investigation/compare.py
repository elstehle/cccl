#!/usr/bin/env python3
"""Compare batched vs single-problem CUB top-k resource usage.

Reads two `cuobjdump --dump-resource-usage` files -- one built with
`CUB_BENCH_TOPK_USE_BATCHED=1` (batched dispatch) and one with
`CUB_BENCH_TOPK_USE_BATCHED=0` (single-problem `cub::DeviceTopK` dispatch) --
and prints a side-by-side overview table grouped by kernel and type combo.

Usage:
  python3 compare.py keys.raw keys.single.raw                    # keys-only
  python3 compare.py pairs.raw pairs.single.raw --with-values    # pairs

The metrics shown are REG / smem / stack / lmem / cmem (= CONSTANT[0]). For
each metric we print `batched | single | Δ`; Δ is the batched-minus-single
delta (so positive = batched costs more = regression). When the single-problem
side doesn't have a matching kernel (the batched worker-per-segment kernel has
no analog), the single columns are reported as `-`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict

# Reuse parse.py from the same directory.
import parse as P


def collect(raw_path: str) -> dict[tuple[str, str, str], P.Kernel]:
    """Parse one resource-usage dump and return a dict keyed by
    `(short_kernel, key, value)` for the u32/u32 offset combination.

    Mirrors `parse.main`'s default filter (u32/u32) and dedup (collapse
    direction / per-ValueT histogram variants)."""
    with open(raw_path) as fh:
        text = fh.read()
    kernels = P.parse(text)
    kernels = [k for k in kernels if k.offset == "u32" and k.out_offset == "u32"]

    deduped: dict[tuple[str, str, str, str, str], P.Kernel] = {}
    for k in kernels:
        key = (k.kernel, k.key, k.value, k.offset, k.out_offset)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = k
            continue
        if (
            existing.reg != k.reg
            or existing.smem != k.smem
            or existing.stack != k.stack
            or existing.lmem != k.lmem
            or existing.cmem0 != k.cmem0
        ):
            print(
                f"warning: drift across collapsed instantiations of "
                f"{k.kernel}<{k.key}, {k.value}, {k.offset}, {k.out_offset}>; kept first.",
                file=sys.stderr,
            )

    # Re-key by (short_kernel_label, key, value) so we can join batched vs
    # single regardless of which template family they came from. The
    # short-kernel mapping in parse.py uses `"filter   "` etc. for printing;
    # we strip whitespace here.
    out: dict[tuple[str, str, str], P.Kernel] = {}
    for k in deduped.values():
        short = P.SHORT_KERNEL[k.kernel].strip()
        out[(short, k.key, k.value)] = k
    return out


KEY_ORDER = {
    "i8": 1, "u8": 1,
    "i16": 2, "u16": 2,
    "i32": 4, "u32": 4, "f32": 4,
    "i64": 8, "u64": 8, "f64": 8,
    "i128": 16, "u128": 16,
}
KERNEL_ORDER = {"histogram": 0, "filter": 1, "last_filt": 2, "worker": 3}


def fmt_metric(b_val: int | None, s_val: int | None) -> str:
    """Format `batched | single | Δ` for a single metric."""
    b_str = "-" if b_val is None else str(b_val)
    s_str = "-" if s_val is None else str(s_val)
    if b_val is None or s_val is None:
        d_str = "-"
    else:
        d = b_val - s_val
        if d == 0:
            d_str = "0"
        elif d > 0:
            d_str = f"+{d}"
        else:
            d_str = str(d)
    return f"{b_str:>5} {s_str:>5} {d_str:>5}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batched", help="path to batched-dispatch resource-usage dump")
    p.add_argument("single", help="path to single-problem-dispatch resource-usage dump")
    p.add_argument("--with-values", action="store_true", help="show (key, value) rows (pairs benchmark)")
    args = p.parse_args()

    batched = collect(args.batched)
    single = collect(args.single)

    # Build the union of (short_kernel, key, value) keys.
    union = set(batched) | set(single)
    rows = sorted(
        union,
        key=lambda k: (
            KEY_ORDER.get(k[1], 99),
            k[1],
            KEY_ORDER.get(k[2], 0),
            k[2],
            KERNEL_ORDER.get(k[0], 99),
        ),
    )

    # Header.
    if args.with_values:
        type_header = f"{'kernel':<10}  {'key':>5}  {'val':>5}"
        type_width = len(type_header)
    else:
        type_header = f"{'kernel':<10}  {'key':>5}"
        type_width = len(type_header)
    metric_cols = ("REG", "smem", "stack", "lmem", "cmem")
    metric_header = "  ".join(
        f"{('b ' + m):>5} {('s ' + m):>5} {('Δ ' + m):>5}" for m in metric_cols
    )
    print(f"{type_header}  {metric_header}")
    print("-" * (type_width + 2 + len(metric_header)))

    last_group: tuple[str, str] | tuple[str, ...] | None = None
    for k in rows:
        short, key_t, val_t = k
        group = (key_t, val_t) if args.with_values else (key_t,)
        if last_group is not None and group != last_group:
            print()
        last_group = group
        b = batched.get(k)
        s = single.get(k)
        metric_strs = [
            fmt_metric(
                None if b is None else getattr(b, m_attr),
                None if s is None else getattr(s, m_attr),
            )
            for m_attr in ("reg", "smem", "stack", "lmem", "cmem0")
        ]
        if args.with_values:
            print(
                f"{short:<10}  {key_t:>5}  {val_t:>5}  "
                + "  ".join(metric_strs)
            )
        else:
            print(
                f"{short:<10}  {key_t:>5}  "
                + "  ".join(metric_strs)
            )

    print()
    print(f"({len(rows)} kernel x type combos shown; b=batched, s=single, Δ=b-s, +=batched regresses)")


if __name__ == "__main__":
    main()
