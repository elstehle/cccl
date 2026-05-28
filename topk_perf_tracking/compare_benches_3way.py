#!/usr/bin/env python3
"""Print a 3-way comparison of nvbench JSONs (main vs dev vs flat).

For each workload (joined by (KeyT, ValueT, OffsetT, OutOffsetT, Elements,
SelectedElements, Entropy)) show absolute GPU times for all three columns
plus deltas vs the leftmost ("main") baseline.
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
    if t in ("int64",):
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


def summary_value(s, type_hint=float):
    for d in s.get("data", []):
        if d.get("name") == "value":
            v = d.get("value")
            try:
                return type_hint(v)
            except (TypeError, ValueError):
                return v
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
            gpu_mean = next(
                (s for s in st.get("summaries", []) if s.get("tag") == "nv/cold/time/gpu/mean"),
                None,
            )
            if gpu_mean is None:
                continue
            out[key] = {
                "time_s": summary_value(gpu_mean, float),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main", type=Path)
    ap.add_argument("dev", type=Path)
    ap.add_argument("target", type=Path)
    ap.add_argument("--main-label", default="main")
    ap.add_argument("--dev-label", default="dev")
    ap.add_argument("--target-label", default="flat_walk")
    ap.add_argument("--key-t-filter", default=None)
    ap.add_argument("--value-t-filter", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    m = load(args.main)
    d = load(args.dev)
    t = load(args.target)

    common = sorted(set(m) & set(d) & set(t))
    if args.key_t_filter:
        keep = set(args.key_t_filter.split(","))
        common = [k for k in common if k[0] in keep]
    if args.value_t_filter:
        keep = set(args.value_t_filter.split(","))
        common = [k for k in common if k[1] in keep]

    if not common:
        raise SystemExit("no overlapping workloads")

    lines: list[str] = []
    ml, dl, tl = args.main_label, args.dev_label, args.target_label
    lines.append(f"# nvbench three-way: `{ml}` vs `{dl}` vs `{tl}`")
    lines.append("")
    lines.append(
        f"| KeyT | ValueT | {ml} (us) | {dl} (us) | {tl} (us) | "
        f"{dl}/{ml} | {tl}/{ml} | {tl}/{dl} | gap closed |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    sums = {"m": 0.0, "d": 0.0, "t": 0.0, "n": 0}
    for key in common:
        KeyT, ValueT, *_ = key
        mt = m[key]["time_s"] * 1e6
        dt = d[key]["time_s"] * 1e6
        tt = t[key]["time_s"] * 1e6
        # "gap closed" = how much of the (dev - main) regression we recovered.
        gap_total = dt - mt
        gap_left = tt - mt
        gap_pct = 100.0 * (gap_total - gap_left) / gap_total if abs(gap_total) > 1e-9 else 0.0
        lines.append(
            f"| {KeyT} | {ValueT} | {mt:.2f} | {dt:.2f} | {tt:.2f} | "
            f"{dt/mt:.3f}× | {tt/mt:.3f}× | {tt/dt:.3f}× | "
            f"{gap_pct:.0f}% |"
        )
        sums["m"] += mt
        sums["d"] += dt
        sums["t"] += tt
        sums["n"] += 1

    lines.append("")
    mt, dt_sum, tt = sums["m"] / sums["n"], sums["d"] / sums["n"], sums["t"] / sums["n"]
    lines.append(f"- workloads: {sums['n']}")
    lines.append(f"- mean wall-clock GPU time:")
    lines.append(f"  - `{ml}`: {mt:.2f} us")
    lines.append(f"  - `{dl}`: {dt_sum:.2f} us ({dt_sum/mt:.3f}× vs `{ml}`)")
    lines.append(f"  - `{tl}`: {tt:.2f} us ({tt/mt:.3f}× vs `{ml}`, {tt/dt_sum:.3f}× vs `{dl}`)")
    gap_total = dt_sum - mt
    gap_left = tt - mt
    if abs(gap_total) > 1e-9:
        lines.append(
            f"- overall gap (`{dl}` slowdown vs `{ml}`) closed by `{tl}`: "
            f"**{100*(gap_total-gap_left)/gap_total:.0f}%**"
        )

    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    sys.exit(main() or 0)
