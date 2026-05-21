#!/usr/bin/env python3
"""Find peak R-register usage with source-line attribution."""

import re
import sys
import collections

with open(sys.argv[1]) as fh:
    lines = fh.readlines()

# Walk; track last-seen source file/line for each instruction.
peak = []
cur_src = None
for i, line in enumerate(lines):
    if "## File " in line:
        m = re.search(r'File "([^"]+)", line (\d+)', line)
        if m:
            cur_src = (m.group(1).split("/")[-1], int(m.group(2)))
        continue
    m = re.search(r"/\*([0-9a-f]+)\*/(.*)$", line.rstrip())
    if not m:
        continue
    off = int(m.group(1), 16)
    body = m.group(2)
    regs = [int(x) for x in re.findall(r"\bR(\d+)\b", body)]
    if not regs:
        continue
    peak.append((max(regs), i, off, body.strip(), cur_src))

peak.sort(key=lambda x: -x[0])
print("== Top 12 R-register peaks ==")
seen = set()
for mx, li, off, body, src in peak:
    if mx in seen:
        continue
    seen.add(mx)
    sstr = f"{src[0]}:{src[1]}" if src else "?"
    print(f"  R{mx:>3}  off {off:#06x}  line {li+1:>5}  {sstr:<40}  {body[:80]}")
    if len(seen) >= 12:
        break

# Source-line attribution: for the top 50 peak instructions, count occurrences per src-line.
print()
print("== Source lines hosting the top-50 absolute peak instructions ==")
counts = collections.Counter()
for mx, li, off, body, src in peak[:50]:
    if src:
        counts[(src[0], src[1])] += 1
for (f, ln), c in counts.most_common():
    print(f"  {c:>3}  {f}:{ln}")
