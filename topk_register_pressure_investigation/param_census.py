#!/usr/bin/env python3
"""Census of how parameter / segment-uniform values flow through registers.

Compares two SASS files (batched vs single-problem) by counting:
  - LDC / LDC.64 / LDC.128       (load constant -> regular register)
  - LDCU / LDCU.64 / LDCU.128    (load constant -> uniform register)
  - LDG.E / LDG.E.64 / LDG.E.128 (load global through const cache)
  - LD.E / LD.E.64 / LD.E.128    (load global, standard)
  - R2UR / R2UP                  (regular -> uniform / predicate)
  - ULDC / UMOV / S2UR           (uniform-side ops)
  - ATOMG.* / ATOMS.*            (global / shared atomics)
"""

import re
import sys
import collections

patterns = [
    ("LDC",       r"\bLDC(?:\.(?:64|128))?\b"),
    ("LDCU",      r"\bLDCU(?:\.(?:64|128))?\b"),
    ("LDG.E",     r"\bLDG\.E(?:\.(?:64|128|U8|S8|U16|S16))?\b"),
    ("LD.E",      r"\bLD\.E(?:\.(?:64|128|U8|S8|U16|S16))?\b"),
    ("R2UR",      r"\bR2UR\b"),
    ("R2UP",      r"\bR2UP\b"),
    ("ULDC",      r"\bULDC(?:\.(?:64|128))?\b"),
    ("UMOV",      r"\bUMOV\b"),
    ("S2UR",      r"\bS2UR\b"),
    ("S2R",       r"\bS2R\b"),
    ("ATOMG",     r"\bATOMG\."),
    ("ATOMS",     r"\bATOMS\."),
    ("BAR.SYNC",  r"\bBAR\.SYNC\b"),
    ("PRMT",      r"\bPRMT\b"),
    ("IMAD",      r"\bIMAD\."),
    ("LDS",       r"\bLDS(?:\.(?:64|128))?\b"),
    ("STS",       r"\bSTS(?:\.(?:64|128))?\b"),
]

def census(path):
    with open(path) as fh:
        text = fh.read()
    cleaned = re.sub(r"/\*[^*]*\*/", "", text)
    counts = collections.Counter()
    for name, p in patterns:
        counts[name] = len(re.findall(p, cleaned))
    # also count total instruction lines (offset-prefixed)
    counts["total_insts"] = len(re.findall(r"/\*[0-9a-f]{4,8}\*/", text))
    counts["unique_R"] = len({int(x) for x in re.findall(r"\bR(\d+)\b", cleaned)})
    counts["unique_UR"] = len({int(x) for x in re.findall(r"\bUR(\d+)\b", cleaned)})
    return counts

a, b = sys.argv[1], sys.argv[2]
ca = census(a)
cb = census(b)
print(f"{'metric':<14}  {'batched':>10}  {'single':>10}  {'b-s':>8}")
print("-" * 50)
keys = ["total_insts", "unique_R", "unique_UR"] + [n for n, _ in patterns]
for k in keys:
    delta = ca[k] - cb[k]
    print(f"{k:<14}  {ca[k]:>10}  {cb[k]:>10}  {delta:>+8}")
