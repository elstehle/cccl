import json
import statistics
import sys


def load(path):
    d = json.load(open(path))
    out = {}
    for b in d["benchmarks"]:
        for s in b["states"]:
            name = b["name"] + "|" + s["name"]
            t = None
            for summ in s.get("summaries", []):
                if summ.get("tag") == "nv/cold/time/gpu/mean":
                    for x in summ.get("data", []):
                        if x.get("name") == "value":
                            t = float(x["value"])
            if t:
                out[name] = t
    return out


a = load(sys.argv[1])
b = load(sys.argv[2])
ds = sorted((100.0 * (b[k] - a[k]) / a[k], k) for k in a if k in b)
vals = [d for d, _ in ds]
print(f"n={len(vals)}  mean={statistics.mean(vals):+.3f}%  median={statistics.median(vals):+.3f}%")
print(
    "buckets:  <-2%%: %d   -2..-0.5%%: %d   +-0.5%%: %d   0.5..2%%: %d   >2%%: %d"
    % (
        sum(1 for v in vals if v < -2),
        sum(1 for v in vals if -2 <= v < -0.5),
        sum(1 for v in vals if -0.5 <= v <= 0.5),
        sum(1 for v in vals if 0.5 < v <= 2),
        sum(1 for v in vals if v > 2),
    )
)
print("best 6 (PR faster):")
for d, k in ds[:6]:
    print(f"  {d:+.2f}%  {k}")
print("worst 6 (PR slower):")
for d, k in ds[-6:]:
    print(f"  {d:+.2f}%  {k}")
