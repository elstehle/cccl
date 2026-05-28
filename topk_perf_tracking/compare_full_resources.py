#!/usr/bin/env python3
"""Generate a side-by-side resource comparison between two snapshots.

For each (logical_kernel, key_t, value_t) tuple, aggregate the per-record
metrics (registers, stack, spill stores / loads, smem, barriers) across
all OffsetT / OutOffsetT variants and print:
  - the worst-case (max) value on each side
  - a delta column highlighting regressions / improvements

Designed to compare the single-problem `main` kernels (no `select::min`
instantiation) against the batched `dev` kernels (both selects, plus
`finalize_*` and `single_cta` siblings). For dev, only the `max` select
is shown -- in our tracking `min` and `max` produce byte-identical
resources for every (logical, K, V) we've seen.

Usage:
    compare_full_resources.py main_snapshot.json dev_snapshot.json [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict


KEY_LABEL = {
    "signed char": "I8",
    "short": "I16",
    "int": "I32",
    "long": "I64",
    "__int128": "I128",
    "float": "F32",
    "double": "F64",
}

KEY_ORDER = [
    "signed char",
    "short",
    "int",
    "long",
    "__int128",
    "float",
    "double",
]

VALUE_ORDER = [
    "signed char",
    "short",
    "int",
    "long",
    None,  # keys-only / NullType
]

# Snapshot value_t comes through as `cub::_V_<ver>_SM_<arch>::NullType`
# (varies by branch). Normalise to the keys-only sentinel `None`.
def normalise_value(v):
    if v is None:
        return None
    if "NullType" in v:
        return None
    return v


# Logical kernels in the order we want to print them.
LOGICAL_ORDER_MAIN = ["initial_histogram", "filter", "last_filter"]
LOGICAL_ORDER_DEV = [
    "initial_histogram",
    "finalize_histogram",
    "filter",
    "finalize_filter",
    "last_filter",
    "single_cta",
]


def load_snapshot(path):
    return json.load(open(path))["kernels"]


def aggregate(records, *, only_select="max"):
    """Group records by (logical, key_t, normalised value_t).

    Within each group, return the worst-case (max) and best-case (min)
    of each metric across OffsetT / OutOffsetT variants. If the group
    has only one variant, the two are equal.
    """
    if only_select is not None:
        records = [r for r in records if r["select"] == only_select]
    by_key = defaultdict(list)
    for r in records:
        if r["key_t"] is None:
            continue
        v = normalise_value(r["value_t"])
        by_key[(r["logical_name"], r["key_t"], v)].append(r)

    out = {}
    for key, group in by_key.items():
        regs = [r["registers"] for r in group if r["registers"] is not None]
        stack = [r["stack_frame"] for r in group if r["stack_frame"] is not None]
        sp_st = [r["spill_stores"] for r in group if r["spill_stores"] is not None]
        sp_ld = [r["spill_loads"] for r in group if r["spill_loads"] is not None]
        smem = [r["smem_bytes"] for r in group]
        barr = [r["barriers"] for r in group]
        cmem_total = []
        for r in group:
            cmem_total.append(sum(r.get("cmem", {}).values()))
        out[key] = {
            "n": len(group),
            "regs_max": max(regs) if regs else None,
            "regs_min": min(regs) if regs else None,
            "stack_max": max(stack) if stack else 0,
            "sp_st_max": max(sp_st) if sp_st else 0,
            "sp_ld_max": max(sp_ld) if sp_ld else 0,
            "smem_max": max(smem) if smem else 0,
            "smem_min": min(smem) if smem else 0,
            "barr_max": max(barr) if barr else 0,
            "cmem_max": max(cmem_total) if cmem_total else 0,
        }
    return out


def fmt_range(min_v, max_v):
    if min_v is None and max_v is None:
        return "--"
    if min_v == max_v:
        return f"{max_v}"
    return f"{min_v}..{max_v}"


def fmt_with_delta(main_v, dev_v):
    if main_v is None or dev_v is None:
        m = "--" if main_v is None else str(main_v)
        d = "--" if dev_v is None else str(dev_v)
        return f"{m} -> {d}"
    if main_v == dev_v:
        return f"{main_v}"
    sign = "+" if dev_v > main_v else ""
    return f"{main_v} -> {dev_v} ({sign}{dev_v - main_v})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main_snapshot", type=Path)
    ap.add_argument("dev_snapshot", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    main_recs = load_snapshot(args.main_snapshot)
    dev_recs = load_snapshot(args.dev_snapshot)

    main_agg = aggregate(main_recs, only_select="max")
    dev_agg = aggregate(dev_recs, only_select="max")

    # All keys we'll discuss: union of both, sorted by logical / key / value.
    all_keys = sorted(
        set(main_agg) | set(dev_agg),
        key=lambda k: (
            LOGICAL_ORDER_DEV.index(k[0]) if k[0] in LOGICAL_ORDER_DEV else 99,
            KEY_ORDER.index(k[1]) if k[1] in KEY_ORDER else 99,
            VALUE_ORDER.index(k[2]) if k[2] in VALUE_ORDER else 99,
        ),
    )

    lines = []
    lines.append("# Resource report: `main` vs dev (full type matrix)\n")
    lines.append(
        f"- main:  `{Path(args.main_snapshot).name}` ({len(main_recs)} ptxas records)"
    )
    lines.append(
        f"- dev:   `{Path(args.dev_snapshot).name}` ({len(dev_recs)} ptxas records)"
    )
    lines.append(
        "- Aggregation: per `(logical_kernel, KeyT, ValueT)` group, take the worst-case"
    )
    lines.append(
        "  (max) value across `OffsetT`/`OutOffsetT` variants. Dev's `select::min` and"
    )
    lines.append("  `select::max` are byte-identical here, so only `max` is shown.")
    lines.append("- ValueT `(K only)` denotes the keys-only kernel (`NullType`).\n")

    lines.append(
        "## Side-by-side: `regs / stack / sp_st / sp_ld / smem (bytes)`\n"
    )
    lines.append(
        "Format `MAIN -> DEV (delta)` when they differ; a bare number means equal."
    )
    lines.append(
        "An asterisk (*) marks a logical kernel that exists only on the dev side"
    )
    lines.append("(`finalize_filter`, `finalize_histogram`, `single_cta`).")
    lines.append("")

    # One section per logical kernel
    for ln in LOGICAL_ORDER_DEV:
        sub_keys = [k for k in all_keys if k[0] == ln]
        if not sub_keys:
            continue
        on_main = ln in LOGICAL_ORDER_MAIN
        suffix = "" if on_main else "  (* dev-only)"
        lines.append(f"### `{ln}`{suffix}\n")
        lines.append(
            "| KeyT | ValueT | regs | stack | sp_st | sp_ld | smem (B) |"
        )
        lines.append(
            "|---|---|---|---|---|---|---|"
        )
        for k in sub_keys:
            ka, kt, vt = k
            kt_l = KEY_LABEL.get(kt, kt)
            vt_l = "(K only)" if vt is None else KEY_LABEL.get(vt, vt)

            ma = main_agg.get(k)
            de = dev_agg.get(k)

            def cell(metric, sub_metric=None):
                m = ma[metric] if ma else None
                d = de[metric] if de else None
                if (m is None or m == 0) and (d is None or d == 0):
                    return "0" if (m == 0 and d == 0) else "--"
                return fmt_with_delta(m, d)

            regs = cell("regs_max")
            stack = cell("stack_max")
            sp_st = cell("sp_st_max")
            sp_ld = cell("sp_ld_max")
            smem = cell("smem_max")
            lines.append(
                f"| {kt_l} | {vt_l} | {regs} | {stack} | {sp_st} | {sp_ld} | {smem} |"
            )
        lines.append("")

    # Summary statistics
    lines.append("## Summary statistics\n")
    common_keys = [k for k in all_keys if k in main_agg and k in dev_agg]
    dev_only_keys = [k for k in all_keys if k in dev_agg and k not in main_agg]
    main_only_keys = [k for k in all_keys if k in main_agg and k not in dev_agg]

    def reg_delta(k):
        return (dev_agg[k]["regs_max"] or 0) - (main_agg[k]["regs_max"] or 0)

    common_keys_sorted_by_delta = sorted(common_keys, key=reg_delta, reverse=True)

    lines.append(
        f"- `(logical, K, V)` triples present on both sides: **{len(common_keys)}**"
    )
    lines.append(
        f"- Dev-only triples (no main counterpart -- `finalize_*`, `single_cta`): "
        f"**{len(dev_only_keys)}**"
    )
    lines.append(
        f"- Main-only triples (should be 0): **{len(main_only_keys)}**"
    )

    # Reg-delta histogram across common kernels
    deltas = [reg_delta(k) for k in common_keys]
    if deltas:
        lines.append("")
        lines.append("### Register delta histogram (dev - main, common kernels)\n")
        from collections import Counter

        hist = Counter(deltas)
        lines.append("| delta (regs) | count |")
        lines.append("|---:|---:|")
        for d, c in sorted(hist.items()):
            sign = "+" if d > 0 else ""
            lines.append(f"| {sign}{d} | {c} |")
        lines.append("")

    # Top-N regressions
    lines.append("### Worst register regressions (top 12 dev-over-main)\n")
    lines.append("| logical | KeyT | ValueT | main regs | dev regs | delta |")
    lines.append("|---|---|---|---:|---:|---:|")
    for k in common_keys_sorted_by_delta[:12]:
        ln, kt, vt = k
        kt_l = KEY_LABEL.get(kt, kt)
        vt_l = "(K only)" if vt is None else KEY_LABEL.get(vt, vt)
        m = main_agg[k]["regs_max"]
        d = dev_agg[k]["regs_max"]
        lines.append(f"| `{ln}` | {kt_l} | {vt_l} | {m} | {d} | {d - m:+d} |")
    lines.append("")

    # Stack / spill hotspots on either side
    def has_pressure(agg):
        return [
            k
            for k, v in agg.items()
            if v["stack_max"] > 0 or v["sp_st_max"] > 0 or v["sp_ld_max"] > 0
        ]

    main_pressure = has_pressure(main_agg)
    dev_pressure = has_pressure(dev_agg)
    lines.append("### Stack / spill activity\n")
    lines.append(
        "Triples where any of `stack_frame`, `spill_stores`, `spill_loads` is"
        " non-zero on either side. Dev-only kernels (`single_cta`, `finalize_*`)"
        " are shown when they spill."
    )
    lines.append("")
    lines.append(
        "| logical | KeyT | ValueT | main stack/sp_st/sp_ld | dev stack/sp_st/sp_ld |"
    )
    lines.append("|---|---|---|---|---|")
    pressure_keys = sorted(set(main_pressure) | set(dev_pressure), key=lambda k: (LOGICAL_ORDER_DEV.index(k[0]) if k[0] in LOGICAL_ORDER_DEV else 99, KEY_ORDER.index(k[1]) if k[1] in KEY_ORDER else 99, VALUE_ORDER.index(k[2]) if k[2] in VALUE_ORDER else 99))
    for k in pressure_keys:
        ln, kt, vt = k
        kt_l = KEY_LABEL.get(kt, kt)
        vt_l = "(K only)" if vt is None else KEY_LABEL.get(vt, vt)
        ma = main_agg.get(k)
        de = dev_agg.get(k)

        def fmt_press(a):
            if a is None:
                return "--"
            return f"{a['stack_max']} / {a['sp_st_max']} / {a['sp_ld_max']}"

        lines.append(
            f"| `{ln}` | {kt_l} | {vt_l} | {fmt_press(ma)} | {fmt_press(de)} |"
        )
    lines.append("")

    # Smem deltas on common kernels
    lines.append("### Notable smem differences\n")
    smem_diffs = []
    for k in common_keys:
        m = main_agg[k]["smem_max"]
        d = dev_agg[k]["smem_max"]
        if m != d:
            smem_diffs.append((k, m, d))
    smem_diffs.sort(key=lambda x: x[2] - x[1], reverse=True)
    if smem_diffs:
        lines.append("| logical | KeyT | ValueT | main smem (B) | dev smem (B) | delta (B) |")
        lines.append("|---|---|---|---:|---:|---:|")
        for k, m, d in smem_diffs[:20]:
            ln, kt, vt = k
            kt_l = KEY_LABEL.get(kt, kt)
            vt_l = "(K only)" if vt is None else KEY_LABEL.get(vt, vt)
            lines.append(
                f"| `{ln}` | {kt_l} | {vt_l} | {m} | {d} | {d - m:+d} |"
            )
    else:
        lines.append("(no smem differences across common kernels)")
    lines.append("")

    text = "\n".join(lines)
    if args.out:
        args.out.write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
