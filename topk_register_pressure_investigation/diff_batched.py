#!/usr/bin/env python3
"""Diff two batched resource-usage dumps and print just the rows that moved.

For each (kernel, key, value) present in both dumps, show metrics that
differ between 'before' (-) and 'after' (+). Rows where everything matches
are suppressed."""

import sys
sys.path.insert(0, "/cccl_fork/cccl/build/blackwell-cub-cpp17/topk_baseline")
import compare as C

def main(prev_path, cur_path, with_values: bool):
    prev = C.collect(prev_path)
    cur = C.collect(cur_path)
    union = set(prev) | set(cur)

    rows = sorted(
        union,
        key=lambda k: (
            C.KEY_ORDER.get(k[1], 99),
            k[1],
            C.KEY_ORDER.get(k[2], 0),
            k[2],
            C.KERNEL_ORDER.get(k[0], 99),
        ),
    )

    metrics = ("reg", "smem", "stack", "lmem", "cmem0")
    metric_labels = ("REG", "smem", "stack", "lmem", "cmem")

    if with_values:
        hdr = f"{'kernel':<10}  {'key':>5}  {'val':>5}  "
    else:
        hdr = f"{'kernel':<10}  {'key':>5}  "
    hdr += "  ".join(f"{('prev '+m):>10} {('new '+m):>9} {('Δ '+m):>6}" for m in metric_labels)
    print(hdr)
    print("-" * len(hdr))

    moved = 0
    for k in rows:
        p = prev.get(k)
        c = cur.get(k)
        if p is None or c is None:
            continue
        diffs = []
        for ma in metrics:
            pv = getattr(p, ma)
            cv = getattr(c, ma)
            diffs.append((pv, cv, cv - pv))
        if all(d[2] == 0 for d in diffs):
            continue
        moved += 1
        short, key_t, val_t = k
        parts = []
        for (pv, cv, dv), ml in zip(diffs, metric_labels):
            if dv == 0:
                parts.append(f"{pv:>10} {cv:>9} {'0':>6}")
            elif dv > 0:
                parts.append(f"{pv:>10} {cv:>9} {'+'+str(dv):>6}")
            else:
                parts.append(f"{pv:>10} {cv:>9} {str(dv):>6}")
        prefix = (
            f"{short:<10}  {key_t:>5}  {val_t:>5}  "
            if with_values
            else f"{short:<10}  {key_t:>5}  "
        )
        print(prefix + "  ".join(parts))

    print()
    print(f"({moved} rows moved out of {len(rows)} total)")


if __name__ == "__main__":
    p = sys.argv[1]
    c = sys.argv[2]
    main(p, c, "--with-values" in sys.argv)
