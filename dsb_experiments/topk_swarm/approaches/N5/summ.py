import json, sys
for f in sys.argv[1:]:
    b = json.load(open(f))
    print("==", b.get("batch"), "node", b.get("node"), "src", b.get("source_commit"))
    can = b.get("canary", {})
    print("   canary", {k: (round(v, 2) if isinstance(v, (int, float)) else v) for k, v in can.items()} if isinstance(can, dict) else can)
    print("   failed", [c.get("name", c) if isinstance(c, dict) else c for c in b.get("failed", [])][:6], "unsupported", len(b.get("unsupported", [])))
    rows = {}
    for c in b["cells"]:
        key = (c["n"], c["k"], c["requirement"][:4])
        r = c["resources"]
        v, cv = c["value"], c.get("control_value")
        rows.setdefault(key, {})[c["regime"]] = (v, cv, r.get("registers"), r.get("occupancy_blocks_per_sm"), r.get("stack_bytes"), r.get("stl_count"), r.get("sass_instructions"))
    for key in sorted(rows):
        lat = rows[key].get("latency"); thr = rows[key].get("throughput")
        s = "   N%4d k%3d %s" % key
        if lat: s += " | lat %6.0f ctrl %6.0f (%.2fx) r%d st%d" % (lat[0], lat[1], lat[1] / lat[0] if lat[0] else 0, lat[2], lat[4] or 0)
        if thr: s += " | thr %6.1f ctrl %6.1f (%.2fx) r%d occ%d sass%d" % (thr[0], thr[1], thr[0] / thr[1] if thr[1] else 0, thr[2], thr[3], thr[6] or 0)
        print(s)
