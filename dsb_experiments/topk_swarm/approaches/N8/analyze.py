#!/usr/bin/env python3
"""Cross-block-size tables from one or more harness batch JSON files (N8 block-size study).

  python3 approaches/N8/analyze.py <batch.json> [<batch.json> ...] [--regime latency|throughput]
      [--req not_guaranteed] [--ref-threads 256]

For every (key type, N, k, requirement, regime) the table lists the variant's value at every
measured thread count next to the control's value at every thread count, the resources of each
kernel (registers, spill bytes, stack bytes, STL and LDL counts, occupancy, SASS instructions),
and the ratios against the reference thread count (default 256): variant at T over variant at
the reference, and variant at T over the control at the reference. Latency ratios are control
over variant (larger is better for the variant), throughput ratios are variant over control.
"""
import argparse
import json
from collections import defaultdict


def res_str(r):
    if not r:
        return "-"
    s = "r%s" % r.get("registers")
    if r.get("spill_bytes"):
        s += " sp%s" % r.get("spill_bytes")
    if r.get("stack_bytes"):
        s += " st%s" % r.get("stack_bytes")
    if r.get("stl_count") or r.get("ldl_count"):
        s += " L%s/%s" % (r.get("stl_count"), r.get("ldl_count"))
    s += " o%s" % r.get("occupancy_blocks_per_sm")
    if r.get("sass_instructions"):
        s += " i%s" % r.get("sass_instructions")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="+")
    ap.add_argument("--regime")
    ap.add_argument("--req")
    ap.add_argument("--ref-threads", type=int, default=256)
    args = ap.parse_args()

    rows = defaultdict(dict)  # (key, n, k, req, regime) -> threads -> cell
    variant = control = None
    for path in args.batches:
        b = json.load(open(path))
        variant = b["adapter"]["id"]
        control = b["control"]["id"]
        for c in b.get("cells", []):
            if args.regime and c["regime"] != args.regime:
                continue
            if args.req and c["requirement"] != args.req:
                continue
            key = (c["key_type"], c["n"], c["k"], c["requirement"], c["regime"])
            rows[key][c["threads"]] = c
        for f in b.get("failed", []):
            print("FAILED", f["cell"], f["reason"])
        for u in b.get("unsupported", []):
            print("UNSUPPORTED", u["cell"])
        can = b.get("canary", {})
        print("batch %s canary start %s end %s" % (b["batch"], json.dumps(can.get("start")), json.dumps(can.get("end"))))

    threads = sorted({t for r in rows.values() for t in r})
    print("variant %s, control %s, thread counts %s, reference %d" % (variant, control, threads, args.ref_threads))
    for key in sorted(rows, key=lambda k: (k[4], k[0], k[3], k[1], k[2])):
        kt, n, k, req, regime = key
        r = rows[key]
        lat = regime == "latency"
        ref = r.get(args.ref_threads)
        print("%s %-4s N%-5d k%-4d %s" % (regime[:3], kt, n, k, req[:3]))
        for t in threads:
            c = r.get(t)
            if not c:
                continue
            v = c["value"]
            cv = c.get("control_value")
            line = "  T%-3d var %10.1f %-28s" % (t, v, res_str(c.get("resources")))
            if cv is not None:
                cres = ((c.get("raw") or {}).get("control") or {}).get(regime, {}).get("resources")
                line += " ctl %10.1f %-28s" % (cv, res_str(cres))
                line += " v/c %5.2fx" % ((cv / v) if lat else (v / cv))
            if ref and ref is not c:
                rv = ref["value"]
                line += " vT/vRef %5.2fx" % ((rv / v) if lat else (v / rv))
                rcv = ref.get("control_value")
                if rcv is not None:
                    line += " vT/cRef %5.2fx" % ((rcv / v) if lat else (v / rcv))
            print(line)


if __name__ == "__main__":
    main()
