#!/usr/bin/env python3
"""Winners matrix from phase 3 batch JSONs: one table per scope, regime and requirement with the
best adapter per cell, its margin over the runner-up and over the reference control measured in
the same lock section, and the node and harness version.

  python3 tools/winners.py --results state/results/eval-engineer/phase3/b200 --arch b200 \
      --out WINNERS_b200.md --json state/phase3/winners_b200.json --close state/phase3/close_cells_b200.json

Margins: latency (cycles per call) margin = (other - winner) / winner, throughput (Gelem/s)
margin = (winner - other) / other, in percent. A cell is "close" when the margin over the runner-up
is below --close-margin (default 15 percent); the close cells go to the --close file in the
run.py --cell-list format for the tie_heavy pass. Adapters measured at a non-default thread count
or with sorted runs are labeled `id@T<threads>` and `id@runs<M>`.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

DEFAULT_THREADS = {"block": 256, "warp": 32}


def label_of(cell, adapter_id, scope):
    lab = adapter_id
    t = cell.get("threads")
    if scope in DEFAULT_THREADS and t and t != DEFAULT_THREADS[scope]:
        lab += "@T%d" % t
    elif scope == "subwarp" and t:
        lab += "@T%d" % t
    if cell.get("runs"):
        lab += "@runs%d" % cell["runs"]
    return lab


def pct(x):
    return "%+.1f%%" % (x * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="directory with the phase 3 batch JSONs")
    ap.add_argument("--arch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    ap.add_argument("--close")
    ap.add_argument("--close-margin", type=float, default=0.15)
    ap.add_argument("--pattern", default=None, help="restrict to one input pattern (default: all, one table each)")
    ap.add_argument("--second-reference", action="append", default=[],
                    help="control adapter ids measured as a second control (e.g. B1_det): their values appear as an "
                    "extra reference column and the variant's duplicate measurement from those batches is not a candidate")
    args = ap.parse_args()
    seconds = set(args.second_reference)

    entries = defaultdict(list)  # cell key -> list of candidate dicts
    refs2 = defaultdict(list)  # cell key -> [(second control name, value)] from the second-control batches
    nodes, versions = set(), set()
    for p in sorted(glob.glob(os.path.join(args.results, "*.json"))):
        if p.endswith(".build.json"):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if (d.get("node", {}).get("arch") or "").lower() not in ("", args.arch):
            continue
        scope = d["adapter"]["scope"]
        aid = d["adapter"]["id"]
        nodes.add(d["node"].get("hostname"))
        hv = d.get("harness_version") or (d["cells"][0]["raw"].get("harness_version") if d.get("cells") else None)
        if hv:
            versions.add(hv)
        for c in d.get("cells", []):
            if "control_value" not in c:
                continue
            pattern = c.get("named_pattern") or c["input_pattern"]
            if args.pattern and pattern != args.pattern:
                continue
            key = (scope, c["regime"], c["requirement"], pattern, c["key_type"], c["n"], c["k"], c.get("workload_name") or "")
            if c["control_name"].split("@")[0] in seconds:
                refs2[key].append((c["control_name"], c["control_value"]))
                continue
            entries[key].append({
                "adapter": label_of(c, aid, scope), "value": c["value"], "control": c["control_name"],
                "control_value": c["control_value"], "node": d["node"].get("hostname"), "harness": hv,
                "batch": os.path.basename(p), "regs": c["resources"].get("registers"),
                "spill": c["resources"].get("spill_bytes"), "threads": c.get("threads")})

    rows = []
    close = []
    for key in sorted(entries, key=lambda k: (k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7])):
        scope, regime, req, pattern, kt, n, k, wl = key
        cands = entries[key]
        better = (lambda a, b: a < b) if regime == "latency" else (lambda a, b: a > b)
        cands = sorted(cands, key=lambda e: e["value"], reverse=(regime != "latency"))
        win = cands[0]
        run = cands[1] if len(cands) > 1 else None

        def margin(w, o):
            if o is None or not o or not w:
                return None
            return (o - w) / w if regime == "latency" else (w - o) / o

        m_run = margin(win["value"], run["value"]) if run else None
        m_ref = margin(win["value"], win["control_value"])
        r2 = refs2.get(key, [])
        ref2_name = r2[0][0] if r2 else ""
        ref2_value = (sorted(v for _, v in r2)[len(r2) // 2]) if r2 else None  # median over the batches that carried it
        m_ref2 = margin(win["value"], ref2_value) if ref2_value else None
        row = {"scope": scope, "regime": regime, "requirement": req, "pattern": pattern, "key_type": kt, "n": n, "k": k,
               "workload": wl, "winner": win["adapter"], "value": win["value"], "runner_up": run["adapter"] if run else "",
               "runner_value": run["value"] if run else None, "margin_runner_up": m_run, "reference": win["control"],
               "reference_value": win["control_value"], "margin_reference": m_ref,
               "reference2": ref2_name, "reference2_value": ref2_value, "margin_reference2": m_ref2, "node": win["node"],
               "harness": win["harness"], "candidates": len(cands), "regs": win["regs"], "spill": win["spill"],
               "threads": win["threads"], "all": [{"adapter": e["adapter"], "value": e["value"]} for e in cands]}
        rows.append(row)
        if m_run is not None and m_run < args.close_margin:
            close.append({"scope": scope, "key_type": kt, "threads": win["threads"], "n": n, "k": k, "requirement": req,
                          "winner": win["adapter"], "runner_up": run["adapter"], "margin": m_run})

    # markdown
    lines = ["# Winners matrix, %s" % args.arch, "",
             "Generated by `tools/winners.py` from the phase 3 batches under `%s`. Nodes: %s. Harness: %s. "
             "Latency in cycles per call, throughput in Gelem/s. Margin over the runner-up and over the reference "
             "control measured in the same lock section (latency: how much slower the other is, throughput: how "
             "much faster the winner is). Cells with a margin over the runner-up below %.0f percent are listed "
             "as close and get the tie_heavy pass." % (args.results, ", ".join(sorted(x for x in nodes if x)),
                                                        ", ".join(sorted(versions)), args.close_margin * 100), ""]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["scope"], r["regime"], r["requirement"], r["pattern"])].append(r)
    for g in sorted(groups):
        scope, regime, req, pattern = g
        lines.append("## %s, %s, %s, %s input" % (scope, regime, req, pattern))
        lines.append("")
        has2 = any(r["reference2"] for r in groups[g])
        hdr = ["key", "N", "k", "workload", "winner", "value", "runner-up", "margin", "reference", "margin"]
        if has2:
            hdr += ["second reference", "margin"]
        hdr += ["candidates", "regs", "node"]
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "---|" * len(hdr))
        for r in groups[g]:
            cols = [r["key_type"], str(r["n"]), str(r["k"]), r["workload"], r["winner"], "%.4g" % r["value"],
                    ("%s (%.4g)" % (r["runner_up"], r["runner_value"])) if r["runner_up"] else "",
                    pct(r["margin_runner_up"]) if r["margin_runner_up"] is not None else "",
                    "%s (%.4g)" % (r["reference"], r["reference_value"]),
                    pct(r["margin_reference"]) if r["margin_reference"] is not None else ""]
            if has2:
                cols += [("%s (%.4g)" % (r["reference2"], r["reference2_value"])) if r["reference2"] else "",
                         pct(r["margin_reference2"]) if r["margin_reference2"] is not None else ""]
            cols += [str(r["candidates"]), str(r["regs"]) if r["regs"] is not None else "", str(r["node"])]
            lines.append("| " + " | ".join(cols) + " |")
        lines.append("")
    # summary: wins per adapter per scope and regime
    lines.append("## Wins per adapter")
    lines.append("")
    wins = defaultdict(int)
    for r in rows:
        wins[(r["scope"], r["regime"], r["winner"])] += 1
    lines.append("| scope | regime | adapter | cells won |")
    lines.append("|---|---|---|---|")
    for (scope, regime, adapter), nwin in sorted(wins.items(), key=lambda x: (x[0][0], x[0][1], -x[1])):
        lines.append("| %s | %s | %s | %d |" % (scope, regime, adapter, nwin))
    lines.append("")
    lines.append("%d cells, %d close (margin over the runner-up below %.0f percent)." % (len(rows), len(close), args.close_margin * 100))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        json.dump({"arch": args.arch, "rows": rows}, open(args.json, "w"), indent=1)
    if args.close:
        os.makedirs(os.path.dirname(os.path.abspath(args.close)) or ".", exist_ok=True)
        # one entry per (scope, key, threads, n, k, requirement), the format run.py --cell-list reads
        seen, uniq = set(), []
        for c in close:
            kk = (c["scope"], c["key_type"], c["threads"], c["n"], c["k"], c["requirement"])
            if kk not in seen:
                seen.add(kk)
                uniq.append(c)
        json.dump(uniq, open(args.close, "w"), indent=1)
    print("wrote %s: %d cells, %d close, %d adapters seen" % (args.out, len(rows), len(close),
                                                            len({e["adapter"] for v in entries.values() for e in v})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
