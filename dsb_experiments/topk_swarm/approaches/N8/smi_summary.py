#!/usr/bin/env python3
"""Summarize an nvidia-smi 100 ms trace (timestamp, clocks.current.sm, clocks.current.memory, power.draw,
temperature.gpu, clocks_event_reasons.active) taken during a measurement batch."""
import csv, statistics as st, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) >= 6]
hdr = [h.strip() for h in rows[0]]
data = rows[1:]
def col(sub):
    i = [k for k, h in enumerate(hdr) if sub in h][0]
    out = []
    for r in data:
        try:
            out.append(float(r[i].split()[0]))
        except ValueError:
            pass
    return out
sm, mem, pw, t = col("current.sm"), col("current.memory"), col("power"), col("temperature")
busy = [x for x in sm if x > 500]
print("%s: %d samples over %.0f s, sm MHz min/median/max %.0f/%.0f/%.0f (idle floor %.0f, %d busy samples median %.0f), mem %.0f, power median %.0f W max %.0f, temp max %.0f C, event reasons %s" % (
    sys.argv[1].split("/")[-1], len(data), 0.1 * len(data), min(sm), st.median(sm), max(sm), min(sm), len(busy),
    st.median(busy) if busy else 0, max(mem), st.median(pw), max(pw), max(t), sorted(set(r[-1].strip() for r in data))))
