# markdown rows from batch JSONs: python3 tables.py <regime> <req> file...  (prints one table per file)
import json, sys
regime, req = sys.argv[1], sys.argv[2]
for f in sys.argv[3:]:
    b = json.load(open(f))
    name = b["batch"].split("_vs_")[0]
    ctrl = b["control"]["id"] if isinstance(b["control"], dict) else b["control"]
    print("\n%s against %s (%s, %s), batch %s" % (name, ctrl, regime, req, b["batch"]))
    unit = "cycles per call" if regime == "latency" else "Gelem per s"
    print("| cell | %s | %s | ratio | registers | occupancy |" % (name, ctrl))
    print("|---|---|---|---|---|---|")
    for c in sorted(b["cells"], key=lambda c: (c["threads"], c["n"], c["k"])):
        if c["regime"] != regime or not c["requirement"].startswith(req):
            continue
        v, cv, r = c["value"], c["control_value"], c["resources"]
        ratio = (cv / v) if regime == "latency" else (v / cv)
        cell = ("LW %d " % c["threads"] if c["threads"] < 32 else "") + "N %d k %d" % (c["n"], c["k"])
        print("| %s | %.0f | %.0f | %.2fx | %s | %s |" % (cell, v, cv, ratio, r.get("registers"), r.get("occupancy_blocks_per_sm")))
