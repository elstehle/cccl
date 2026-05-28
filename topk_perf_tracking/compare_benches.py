#!/usr/bin/env python3
"""Compare two nvbench JSON outputs.

Joins by (KeyT, ValueT, OffsetT, OutOffsetT, Elements, SelectedElements,
Entropy) tuple and prints a per-row table of GPU-mean-time deltas plus
a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def axis_value(a):
    """Pull the numeric or string value out of an axis entry."""
    t = a.get("type")
    if t == "string":
        return a.get("value")
    # Numeric: int64 / float64 / etc -- nvbench encodes as string.
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
    """Extract a single .data[].value out of an nvbench summary entry."""
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
            gpu_noise = next(
                (s for s in st.get("summaries", []) if s.get("tag") == "nv/cold/time/gpu/stdev/relative"),
                None,
            )
            samples = next(
                (s for s in st.get("summaries", []) if s.get("tag") == "nv/cold/sample_size"),
                None,
            )
            if gpu_mean is None:
                continue
            out[key] = {
                "time_s": summary_value(gpu_mean, float),
                "noise_pct": summary_value(gpu_noise, float),
                "samples": summary_value(samples, int),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path, help="reference / baseline JSON")
    ap.add_argument("target", type=Path, help="candidate / target JSON")
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--target-label", default="target")
    ap.add_argument("--key-t-filter", default=None,
                    help="comma-separated list of KeyT values to keep (e.g. I8,I32,I64)")
    ap.add_argument("--value-t-filter", default=None,
                    help="comma-separated list of ValueT values to keep (e.g. I8,I64)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    base = load(args.baseline)
    targ = load(args.target)

    common = sorted(set(base) & set(targ))
    if args.key_t_filter:
        keep = set(args.key_t_filter.split(","))
        common = [k for k in common if k[0] in keep]
    if args.value_t_filter:
        keep = set(args.value_t_filter.split(","))
        common = [k for k in common if k[1] in keep]

    if not common:
        raise SystemExit("no overlapping workloads")

    lines: list[str] = []
    lines.append(
        f"# nvbench: `{args.baseline_label}` vs `{args.target_label}`"
    )
    lines.append("")
    lines.append(
        "| KeyT | ValueT | Elements | SelectedElems | Entropy | "
        f"{args.baseline_label} (us) | noise% | "
        f"{args.target_label} (us) | noise% | Δ (us) | Δ % | speedup |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    deltas_pct: list[float] = []
    for key in common:
        KeyT, ValueT, OffsetT, OutOffsetT, E, S, Ent = key
        b = base[key]
        t = targ[key]
        bt = b["time_s"] * 1e6
        tt = t["time_s"] * 1e6
        du = tt - bt
        dp = (tt - bt) / bt * 100.0
        deltas_pct.append(dp)
        speedup = bt / tt
        sign = "+" if du > 0 else ""
        lines.append(
            f"| {KeyT} | {ValueT} | {E} | {S} | {Ent} | "
            f"{bt:.2f} | {b['noise_pct']*100:.2f} | "
            f"{tt:.2f} | {t['noise_pct']*100:.2f} | "
            f"{sign}{du:.2f} | {sign}{dp:.1f}% | {speedup:.3f}× |"
        )

    lines.append("")
    if deltas_pct:
        avg = sum(deltas_pct) / len(deltas_pct)
        lines.append(f"- workloads compared: {len(deltas_pct)}")
        lines.append(f"- mean Δ time: **{avg:+.2f}%** ({'+'.join(['target slower']) if avg > 0 else 'target faster'})")
        lines.append(f"- best speedup: **{max(b/t for b, t in [(base[k]['time_s'], targ[k]['time_s']) for k in common]):.3f}×**")
        lines.append(f"- worst speedup: **{min(b/t for b, t in [(base[k]['time_s'], targ[k]['time_s']) for k in common]):.3f}×**")

    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    sys.exit(main() or 0)
