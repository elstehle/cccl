#!/usr/bin/env python3
"""Analyze SASS dump: find peak register region per window, and uniform/regular census."""

import re
import sys

with open(sys.argv[1]) as fh:
    lines = fh.readlines()


def strip(s):
    return re.sub(r"/\*[^*]*\*/", "", s)


def get_offset(line):
    m = re.search(r"/\*([0-9a-f]+)\*/", line)
    return int(m.group(1), 16) if m else None


# Per-line peak R register written
peak_per_line = []
for i, line in enumerate(lines):
    s = strip(line)
    off = get_offset(line)
    regs = [int(x) for x in re.findall(r"\bR(\d+)\b", s)]
    urs = [int(x) for x in re.findall(r"\bUR(\d+)\b", s)]
    peak_per_line.append((i, off, max(regs) if regs else -1, max(urs) if urs else -1, s.strip()))

# Bucket into 0x800-byte windows and report each window's peak
WIN = 0x800
buckets = {}
for i, off, rmx, urmx, s in peak_per_line:
    if off is None:
        continue
    b = off // WIN
    cur = buckets.get(b, (-1, -1, None, None, None))
    if rmx > cur[0]:
        buckets[b] = (rmx, max(cur[1], urmx), i, off, s)
    elif urmx > cur[1]:
        buckets[b] = (cur[0], urmx, cur[2] or i, cur[3] or off, cur[4] or s)

print(f"== Per-{WIN:#x}-byte window: peak R, peak UR, sample instruction ==")
for b in sorted(buckets):
    rmx, urmx, li, off, s = buckets[b]
    if li is None:
        continue
    rstr = f"R{rmx}" if rmx >= 0 else "-"
    urstr = f"UR{urmx}" if urmx >= 0 else "-"
    print(f"  off {b*WIN:#06x}-{(b+1)*WIN:#06x}  peak {rstr:>5} {urstr:>5}  line {li+1:>5}: {s[:80]}")

# Top-10 absolute peaks
print()
print("== Top instructions by absolute max R reference ==")
seen = set()
for entry in sorted(peak_per_line, key=lambda x: -x[2]):
    i, off, rmx, urmx, s = entry
    if rmx in seen:
        continue
    seen.add(rmx)
    print(f"  R{rmx:>3} at line {i+1:>5} (off {off:#06x}): {s[:100]}")
    if len(seen) >= 10:
        break
