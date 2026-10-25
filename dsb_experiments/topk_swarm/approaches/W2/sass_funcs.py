#!/usr/bin/env python3
"""Per-function SASS statistics of a harness cell binary (development aid, run on the node).

  python3 sass_funcs.py <binary> [name-substring ...]

Prints, for every kernel whose demangled name contains all substrings, the instruction count and
the counts of the opcodes that matter for W2: CREDUX, SEL, VIMNMX, LOP3, ISETP, STS, LDS, STL, LDL,
BAR, BRA, VOTE, SHFL. Uses cuobjdump -sass and cu++filt when available.
"""
import re
import subprocess
import sys

OPS = ["CREDUX", "SEL", "VIMNMX", "LOP3", "ISETP", "IMAD", "STS", "LDS", "STL", "LDL", "BAR", "BRA", "VOTE", "SHFL",
       "PRMT", "SHF", "IADD3", "LDG", "FSETP", "WARPSYNC", "BSSY"]


def demangle(name):
    try:
        return subprocess.run(["cu++filt", name], capture_output=True, text=True, timeout=30).stdout.strip() or name
    except Exception:
        return name


def main():
    binary = sys.argv[1]
    subs = sys.argv[2:]
    out = subprocess.run(["cuobjdump", "-sass", binary], capture_output=True, text=True).stdout
    funcs = {}
    cur = None
    for ln in out.splitlines():
        m = re.search(r"Function\s*:\s*(\S+)", ln)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        if cur and re.match(r"\s*/\*[0-9a-f]+\*/", ln):
            funcs[cur].append(ln)
    for name, lines in funcs.items():
        dn = demangle(name)
        short = re.sub(r"\(.*", "", dn)
        if subs and not all(s in dn for s in subs):
            continue
        counts = {op: sum(1 for l in lines if re.search(r"\b%s\b" % op, l.split("/*")[1] if l.count("/*") > 1 else l)) for op in OPS}
        print("%s\n  instr %d  %s" % (short[:160], len(lines), "  ".join("%s %d" % (k, v) for k, v in counts.items() if v)))


if __name__ == "__main__":
    main()
