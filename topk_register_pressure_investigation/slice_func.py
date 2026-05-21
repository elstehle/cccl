#!/usr/bin/env python3
"""Slice one .text.SYMBOL function out of an nvdisasm dump."""

import sys
import re

infile, symbol_re, outfile = sys.argv[1], sys.argv[2], sys.argv[3]
sym = re.compile(symbol_re)

with open(infile) as fh:
    lines = fh.readlines()

start = None
for i, line in enumerate(lines):
    if line.startswith(".text.") and sym.search(line):
        start = i
        break
if start is None:
    raise SystemExit(f"symbol matching '{symbol_re}' not found")

end = len(lines)
for i in range(start + 1, len(lines)):
    if lines[i].startswith(".text.") or lines[i].startswith(".section"):
        end = i
        break

with open(outfile, "w") as fh:
    fh.writelines(lines[start:end])
print(f"wrote {end - start} lines to {outfile}", file=sys.stderr)
