#!/usr/bin/env python3
"""Identify which registers are alive at the peak liveness instruction(s)
of a kernel and, for each one, report where it was first defined (file:line
plus the inline stack).

Algorithm:
  1. Parse the SASS dump produced by `nvdisasm --print-life-ranges -lrm count`
     for one kernel (via `--cuda-function-index`). For each instruction:
     - Extract its byte offset.
     - Mark the first general-register operand as a DEF (unless the opcode
       is a known store/scatter, in which case it is a USE).
     - Mark every other general-register operand as a USE.
  2. Sweep instructions in order. For each register `R<n>`:
     - Open a new live range at each DEF.
     - Extend the current range at each USE.
     - Close the range when a new DEF occurs.
  3. At any program point, a register is "alive" iff there exists a live
     range `[def, last_use]` covering it.
  4. Find peak alive-set instructions (the live-range computation should
     agree with nvdisasm's reported count to within a couple of registers).
  5. Cross-reference with a separately-produced `--print-line-info` dump to
     attribute each alive register's first DEF to a source line + inline
     stack.

Output: a Markdown table listing the alive registers at the chosen peak
offset, what source line first defined each one, and a rough categorization
hint ("per_segment_state", "keys[]", "scratch", "always-alive prologue").

Caveats:
  - We approximate operand semantics: instructions whose first reg operand
    is *not* a destination (stores, atomics with funky encodings) may be
    miscategorized. We treat opcodes starting with `ST`/`STG`/`STS`/`STL`/
    `RED` as "first reg is a USE".
  - For 64-bit / 128-bit register pairs (`R30.64`, `R8.128`) we expand to
    the implied additional registers.
  - We ignore predicates (`P*`) and uniform registers (`UR*`); the report
    is about general regs which is what drives the ptxas "Used N registers"
    figure.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path


RE_SECTION = re.compile(r"^\s*\.text\._")
RE_INSN = re.compile(
    r"^\s*/\*([0-9a-f]+)\*/\s+(.*?);"  # offset, instruction body
    r"\s*//\s*\|\s*(\d+)?\s*\|"
)

RE_INSN_NOLRM = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(.*?);")
RE_LINEINFO = re.compile(r"^\s*//##\s+File\s+\"([^\"]+)\",\s+line\s+(\d+)")
RE_INLINE = re.compile(
    r"^\s*//##\s+(?:inline_call|inlined_at)\s+\d+,\s+file\s+\"([^\"]+)\",\s+line\s+(\d+)"
)


# Register-token matcher. Captures the base reg name + optional .N width.
RE_R = re.compile(r"\bR(\d+)\b(?:\.(\d+))?")

STORE_OPCODES = (
    "STG",
    "STS",
    "STL",
    "ST.",  # generic ST*
    "RED",
    "ATOMG",
    "ATOMS",
    "ATOM",
    "SUST",
)


def opcode_is_store(insn: str) -> bool:
    op = insn.lstrip().split(None, 1)[0]
    for prefix in STORE_OPCODES:
        if op.startswith(prefix):
            return True
    return False


def opcode_is_branch_or_pure_control(insn: str) -> bool:
    op = insn.lstrip().split(None, 1)[0]
    return op in ("BRA", "EXIT", "BSSY.RECONVERGENT", "BSYNC.RECONVERGENT",
                  "BSYNC", "BSSY", "RET", "JMP", "NOP", "BAR.SYNC",
                  "BAR.SYNC.DEFER_BLOCKING", "WARPSYNC.ALL", "WARPSYNC")


# Opcodes whose destination is a predicate, not a general register. The first
# general register seen in their operand list is therefore a USE, not a DEF.
# Match by prefix on the dot-separated opcode parts so qualified forms like
# `ISETP.GE.U32.AND` are caught.
PREDICATE_DEST_PREFIXES = (
    "ISETP", "FSETP", "DSETP", "HSETP", "BSETP",
    "PLOP3", "PSETP",  # predicate-LUT / pred-set forms
    "VOTE", "VOTEU",   # warp / uniform vote
)


def opcode_is_predicate_dest(opcode: str) -> bool:
    head = opcode.split(".", 1)[0]
    return head in PREDICATE_DEST_PREFIXES


def expand_reg_pair(num: int, width_str: str | None) -> list[int]:
    """`R30.64` -> [30, 31]; `R8.128` -> [8, 9, 10, 11]."""
    if width_str is None:
        return [num]
    width = int(width_str)
    count = width // 32
    return list(range(num, num + count))


def classify_operands(body: str) -> tuple[set[int], set[int]]:
    """Return (defs, uses) sets of general-register numbers."""
    # Trim leading predicate prefix like `@P0` or `@!P2`.
    body = body.strip()
    while body.startswith("@"):
        # Eat the `@(!)?P\w+` and any following whitespace.
        m = re.match(r"@!?P\w+\s*", body)
        if not m:
            break
        body = body[m.end():]

    # Now `body` starts with the opcode.
    opcode = body.split(None, 1)[0]
    rest = body[len(opcode):]

    is_store = opcode_is_store(opcode)
    if opcode_is_branch_or_pure_control(opcode) or opcode_is_predicate_dest(opcode):
        # Treat all reg references in the body as USEs (the destination, if
        # any, is a predicate or barrier register which we don't track in
        # the general-register liveness count).
        uses: set[int] = set()
        for m in RE_R.finditer(rest):
            for r in expand_reg_pair(int(m.group(1)), m.group(2)):
                uses.add(r)
        return set(), uses

    # Collect register tokens in order.
    matches = list(RE_R.finditer(rest))
    defs: set[int] = set()
    uses = set()
    if not matches:
        return defs, uses

    # Handle 64-/128-bit loads where the destination width is encoded in the
    # opcode suffix (e.g. `LD.E.64 R30, ...` writes R30 *and* R31). Pick the
    # widest .NN suffix on the opcode for this purpose.
    dest_width_bits = 32
    parts = opcode.split(".")
    for p in parts:
        if p in ("8", "16", "32", "64", "128"):
            try:
                w = int(p)
                if w > dest_width_bits:
                    dest_width_bits = w
            except ValueError:
                pass

    if is_store:
        # Heuristic: opcodes like `STG.E.U8 desc[UR6][R28.64], R45` — the
        # first general register seen is the address part (R28/R29). The
        # final register is the value being stored. Both are USEs. Atomics
        # may also have a destination; we approximate by saying the first
        # operand IS a DEF only when the mnemonic clearly has a return slot
        # (e.g., `ATOMG.E.ADD.STRONG.GPU PT, R41, desc[UR6][...], R41`).
        # For simplicity, treat *all* general regs in a store as USEs; this
        # tends to overestimate liveness for the value reg by one cycle,
        # which doesn't affect the peak-region analysis.
        for m in matches:
            for r in expand_reg_pair(int(m.group(1)), m.group(2)):
                uses.add(r)
        # ATOMG returns into the first slot.
        if opcode.startswith("ATOMG") or opcode.startswith("ATOMS"):
            first = matches[0]
            for r in expand_reg_pair(int(first.group(1)), first.group(2)):
                defs.add(r)
    else:
        first = matches[0]
        # If the operand has an explicit `.64`/`.128` width, honor it;
        # otherwise apply the opcode-suffix-derived destination width.
        if first.group(2):
            for r in expand_reg_pair(int(first.group(1)), first.group(2)):
                defs.add(r)
        else:
            n = int(first.group(1))
            count = max(1, dest_width_bits // 32)
            for r in range(n, n + count):
                defs.add(r)
        for m in matches[1:]:
            for r in expand_reg_pair(int(m.group(1)), m.group(2)):
                uses.add(r)

    # RZ (R255) is the zero register; ignore it.
    defs.discard(255)
    uses.discard(255)
    return defs, uses


def find_section(lines: list[str], substr: str) -> tuple[int, int]:
    start = end = None
    in_section = False
    for i, line in enumerate(lines):
        if RE_SECTION.match(line):
            if substr in line and not in_section:
                start = i
                in_section = True
            elif in_section:
                end = i
                break
    if start is None:
        raise SystemExit(f"could not find section containing {substr!r}")
    if end is None:
        end = len(lines)
    return start, end


def parse_lrm_section(path: Path, substr: str) -> list[dict]:
    text = path.read_text(errors="replace").splitlines()
    s, e = find_section(text, substr)
    out: list[dict] = []
    for line in text[s:e]:
        m = RE_INSN.match(line)
        if not m:
            continue
        offset_hex, body, alive = m.groups()
        defs, uses = classify_operands(body)
        out.append(
            {
                "offset": int(offset_hex, 16),
                "offset_hex": offset_hex,
                "insn": body.strip(),
                "defs": defs,
                "uses": uses,
                "alive_reported": int(alive) if alive else 0,
            }
        )
    return out


def parse_lineinfo_section(path: Path, substr: str) -> dict[int, list[tuple[str, int]]]:
    text = path.read_text(errors="replace").splitlines()
    s, e = find_section(text, substr)
    mapping: dict[int, list[tuple[str, int]]] = {}
    pending: list[tuple[str, int]] = []
    for line in text[s:e]:
        m = RE_LINEINFO.match(line)
        if m:
            pending = [(m.group(1), int(m.group(2)))]
            continue
        m = RE_INLINE.match(line)
        if m:
            pending.append((m.group(1), int(m.group(2))))
            continue
        m = RE_INSN_NOLRM.match(line)
        if m:
            mapping[int(m.group(1), 16)] = list(pending)
    return mapping


def compute_live_ranges(instrs: list[dict]) -> dict[int, list[tuple[int, int]]]:
    """For each register, list of (first_def_offset, last_use_offset)."""
    ranges: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    current: dict[int, tuple[int, int]] = {}  # reg -> (def_offset, last_use_offset)
    for i in instrs:
        # USEs extend any open range for that register.
        for u in i["uses"]:
            if u in current:
                d, _ = current[u]
                current[u] = (d, i["offset"])
        # DEFs close any open range and start a new one.
        for d in i["defs"]:
            if d in current:
                ranges[d].append(current[d])
            current[d] = (i["offset"], i["offset"])
    for r, (d, lu) in current.items():
        ranges[r].append((d, lu))
    return ranges


def alive_at(offset: int, ranges: dict[int, list[tuple[int, int]]]) -> set[int]:
    out: set[int] = set()
    for r, intervals in ranges.items():
        for d, lu in intervals:
            if d <= offset <= lu:
                out.add(r)
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrm", type=Path, required=True)
    ap.add_argument("--lineinfo", type=Path, required=True)
    ap.add_argument("--kernel-substring", required=True)
    ap.add_argument("--peak-offset", type=lambda x: int(x, 0), default=None,
                    help="Offset (hex 0x... or decimal) at which to compute alive set; "
                         "defaults to the instruction with max reported alive count.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    instrs = parse_lrm_section(args.lrm, args.kernel_substring)
    lineinfo = parse_lineinfo_section(args.lineinfo, args.kernel_substring)

    if not instrs:
        raise SystemExit("no instructions parsed")

    if args.peak_offset is None:
        peak_offset = max(instrs, key=lambda i: i["alive_reported"])["offset"]
    else:
        peak_offset = args.peak_offset

    # Sanity check: how many "alive" regs do we compute vs nvdisasm reports?
    ranges = compute_live_ranges(instrs)
    alive = alive_at(peak_offset, ranges)
    by_offset = {i["offset"]: i for i in instrs}
    nvd_reported = by_offset.get(peak_offset, {}).get("alive_reported", "?")

    out_lines: list[str] = []
    out_lines.append("# Alive registers at peak liveness")
    out_lines.append("")
    out_lines.append(f"- peak offset: `0x{peak_offset:x}`")
    out_lines.append(f"- nvdisasm reported alive (general regs): **{nvd_reported}**")
    out_lines.append(f"- this script computed alive set size: **{len(alive)}**")
    out_lines.append(
        "- discrepancy comes from approximate operand classification "
        "(stores, atomics, branches treat the first reg as USE rather than DEF, "
        "so transient scratch may be missed)."
    )
    out_lines.append("")

    rows: list[tuple[int, int, str, int, str, str]] = []
    for r in sorted(alive):
        # Find the interval containing peak.
        first_def = None
        last_use = None
        for d, lu in ranges[r]:
            if d <= peak_offset <= lu:
                first_def, last_use = d, lu
                break
        if first_def is None:
            continue
        info = lineinfo.get(first_def, [])
        inner = info[-1] if info else ("(no lineinfo)", 0)
        # Innermost frame = last entry (where this offset really came from)
        rows.append((r, first_def, inner[0], inner[1], by_offset[first_def]["insn"], info))

    out_lines.append("## Alive registers (sorted by first-def offset)")
    out_lines.append("")
    out_lines.append("| reg | def offset | last use offset | live span | inner source | def instruction |")
    out_lines.append("|---|---|---|---|---|---|")
    for r, fd, src_file, src_line, insn, _stack in sorted(rows, key=lambda x: x[1]):
        # Recover last_use:
        for d, lu in ranges[r]:
            if d == fd:
                lu_eff = lu
                break
        span = lu_eff - fd
        short_file = src_file.split("/")[-1] if src_file else "?"
        short_insn = insn[:60] if insn else "?"
        out_lines.append(
            f"| R{r} | 0x{fd:x} | 0x{lu_eff:x} | 0x{span:x} | `{short_file}:{src_line}` | `{short_insn}` |"
        )
    out_lines.append("")

    # Roll up by source file:line to expose the pressure source(s).
    counter = collections.Counter()
    for r, fd, sf, sln, _, _ in rows:
        counter[(sf, sln)] += 1
    out_lines.append("## Source line origin of alive registers")
    out_lines.append("")
    out_lines.append("| count | file | line |")
    out_lines.append("|---|---|---|")
    for (sf, sln), n in counter.most_common():
        out_lines.append(f"| {n} | `{sf}` | {sln} |")
    out_lines.append("")

    # Detailed inline stack for the top contributors.
    out_lines.append("## Inline stack for each alive register")
    out_lines.append("")
    for r, fd, sf, sln, insn, stack in sorted(rows, key=lambda x: x[1]):
        out_lines.append(f"### R{r}  def @ 0x{fd:x}  `{insn[:80]}`")
        if not stack:
            out_lines.append("  (no line info)")
        else:
            for path, ln in stack:
                short = path.split("/")[-1]
                out_lines.append(f"  - `{short}:{ln}`")
        out_lines.append("")

    text = "\n".join(out_lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    sys.exit(main() or 0)
