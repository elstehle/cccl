#!/usr/bin/env python3
"""Compact table of one harness batch: per cell the variant value, the control value and the
ratio in the direction where larger is better (control/variant for cycles, variant/control for
Gelem/s), plus registers, spills, occupancy, bar.sync count and the tie multiplicity at k.

  python3 -I summarize.py <batch.json> [<batch.json> ...]
"""
import json
import sys


def main():
    for path in sys.argv[1:]:
        b = json.load(open(path))
        m = b
        print("== %s" % path.split("/")[-1])
        print("node %s arch %s nvcc %s commit %s date %s control_req %s" % (
            m["node"]["hostname"], m["node"].get("arch"), m["node"]["nvcc_version"], m["repo"]["commit"][:12],
            m["date"], m.get("control_req")))
        for tag in ("start", "end"):
            c = m.get("canary", {}).get(tag, {})
            print("canary %s: lat %.2f thr_us %.1f" % (tag, c.get("lat_cycles_per_op", -1), c.get("thr_us", -1)))
        print("cells %d failed %d unsupported %d" % (len(m["cells"]), len(m["failed"]), len(m["unsupported"])))
        for f in m["failed"]:
            print("  FAILED", f)
        print("key     n     k req  reg      value    control  ratio regs spill occ bars ties@k")
        rows = sorted(m["cells"], key=lambda c: (c["key_type"], c["regime"], c["n"], c["k"], c["requirement"]))
        for c in rows:
            r = c["resources"]
            cv = c.get("control_value")
            if c["regime"] == "latency":
                ratio = (cv / c["value"]) if cv else float("nan")
            else:
                ratio = (c["value"] / cv) if cv else float("nan")
            ties = c.get("raw", {}).get("variant", {}).get("input_tie_stats", {})
            tk = ties.get("ties_at_k_mean", "") if isinstance(ties, dict) else ""
            print("%4s %6d %5d %-3s %3s %10.1f %10.1f %6.3f %4s %5s %3s %4s %s" % (
                c["key_type"], c["n"], c["k"], c["requirement"][:3], c["regime"][:3], c["value"], cv or 0, ratio,
                r.get("registers"), r.get("spill_bytes"), r.get("occupancy_blocks_per_sm"), r.get("bar_sync_count"), tk))


if __name__ == "__main__":
    main()
