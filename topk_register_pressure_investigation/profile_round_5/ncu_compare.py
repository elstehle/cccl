#!/usr/bin/env python3
"""Pair NCU CSV per-kernel metrics with nvbench JSON workload axes.

Usage:
  ncu_compare.py <nvbench_batched.json> <ncu_batched.csv> <nvbench_single.json> <ncu_single.csv>

Both runs must share the same axis configuration (same workload count and
ordering). NCU's kernel-launch IDs match nvbench's workload ordering when the
benchmark is run with `--profile` (one kernel-of-interest per workload).
"""

import csv
import json
import sys
from collections import defaultdict


def load_nvbench_workloads(path):
    """Return [{axis_name: value, ...}, ...] for non-skipped workloads in nvbench's iteration order."""
    with open(path) as fh:
        d = json.load(fh)
    out = []
    for bench in d.get("benchmarks", []):
        for state in bench.get("states", []):
            if state.get("is_skipped", False):
                continue
            axes = {av["name"]: av.get("value", "") for av in state.get("axis_values", [])}
            out.append(axes)
    return out


def load_ncu_metrics(path):
    """Return list-of-dicts indexed by NCU launch ID 0..N-1.

    Each dict: {metric_name: float, "kernel_name": str, "block_size": str, "grid_size": str}.
    NCU emits one CSV row per (launch_id, metric); we group by ID.
    """
    grouped = defaultdict(dict)
    with open(path) as fh:
        # Skip the two ==PROF== preamble lines.
        for line in fh:
            if line.startswith('"ID"'):
                # found header
                fh.seek(0)
                # skip past the preamble
                for skipped in fh:
                    if skipped.startswith('"ID"'):
                        break
                reader = csv.DictReader(fh, fieldnames=skipped.strip().replace('"', "").split(","))
                for row in reader:
                    if not row["ID"].isdigit():
                        continue
                    rid = int(row["ID"])
                    grouped[rid].setdefault("kernel_name", row["Kernel Name"])
                    grouped[rid].setdefault("block_size", row["Block Size"])
                    grouped[rid].setdefault("grid_size", row["Grid Size"])
                    metric = row["Metric Name"]
                    val = row["Metric Value"].replace(",", "")
                    try:
                        grouped[rid][metric] = float(val)
                    except ValueError:
                        grouped[rid][metric] = val
                break
    return [grouped[i] for i in sorted(grouped.keys())]


def fmt_axes(ax):
    sel = ax.get("SelectedElements", "?")
    elem = ax.get("Elements{io}", "?")
    ent = ax.get("Entropy", "?")
    # nvbench json stores int64 power-of-two as a numeric string (e.g. "65536").
    def to_p2(v):
        try:
            return f"2^{int(v).bit_length()-1}"
        except Exception:
            return str(v)
    return f"E={to_p2(elem):>5} k={to_p2(sel):>5} ent={ent}"


def main():
    nvb_b, ncu_b, nvb_s, ncu_s = sys.argv[1:5]
    wb = load_nvbench_workloads(nvb_b)
    ws = load_nvbench_workloads(nvb_s)
    mb = load_ncu_metrics(ncu_b)
    ms = load_ncu_metrics(ncu_s)
    # The single-problem bench runs with USE_BATCHED=0 -> doesn't skip on
    # SelectedElements>=Elements? Actually it does (same skip in keys.cu). But
    # the workload count may also differ if batched vs single iterate axes
    # differently. Trust the alignment we have, but warn on length mismatch.
    if len(wb) != len(ws):
        print(f"warning: nvbench workload count mismatch: batched={len(wb)} single={len(ws)}", file=sys.stderr)
    if len(mb) != len(ms):
        print(f"warning: ncu kernel count mismatch: batched={len(mb)} single={len(ms)}", file=sys.stderr)
    if len(wb) != len(mb):
        print(f"warning: batched nvbench/ncu length mismatch: nvbench={len(wb)} ncu={len(mb)}", file=sys.stderr)

    n = min(len(wb), len(mb), len(ws), len(ms))

    # Header
    hdr = (
        f"{'workload':<28} | "
        f"{'b time us':>9} {'s time us':>9} {'b/s':>5} | "
        f"{'b reg':>5} {'s reg':>5} | "
        f"{'b smem':>6} {'s smem':>6} | "
        f"{'b occ%':>6} {'s occ%':>6} | "
        f"{'b dram%':>7} {'s dram%':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    sums_b_t = 0.0
    sums_s_t = 0.0

    for i in range(n):
        ax_b = wb[i]
        ax_s = ws[i]
        rb = mb[i]
        rs = ms[i]

        # axes should match -- just print one set
        if ax_b != ax_s:
            print(f"warning: axis mismatch at index {i}: batched={ax_b} single={ax_s}", file=sys.stderr)

        bt = rb.get("gpu__time_duration.sum", float("nan")) / 1000.0  # ns -> us
        st = rs.get("gpu__time_duration.sum", float("nan")) / 1000.0
        ratio = bt / st if st > 0 else float("nan")
        sums_b_t += bt
        sums_s_t += st

        breg = int(rb.get("launch__registers_per_thread", 0))
        sreg = int(rs.get("launch__registers_per_thread", 0))
        bsm = int(rb.get("launch__shared_mem_per_block_allocated", 0))
        ssm = int(rs.get("launch__shared_mem_per_block_allocated", 0))
        bocc = rb.get("sm__warps_active.avg.pct_of_peak_sustained_active", float("nan"))
        socc = rs.get("sm__warps_active.avg.pct_of_peak_sustained_active", float("nan"))
        bdr = rb.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", float("nan"))
        sdr = rs.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", float("nan"))

        print(
            f"{fmt_axes(ax_b):<28} | "
            f"{bt:>9.1f} {st:>9.1f} {ratio:>5.2f} | "
            f"{breg:>5d} {sreg:>5d} | "
            f"{bsm:>6d} {ssm:>6d} | "
            f"{bocc:>6.1f} {socc:>6.1f} | "
            f"{bdr:>7.1f} {sdr:>7.1f}"
        )

    print()
    print(f"summary: total batched time {sums_b_t:.1f} us, total single time {sums_s_t:.1f} us, ratio {sums_b_t/sums_s_t if sums_s_t > 0 else float('nan'):.3f}x")


if __name__ == "__main__":
    main()
