#!/usr/bin/env python3
"""Find peak register liveness in a SASS dump that has nvdisasm life-range
annotations, then cross-reference with a separate line-info dump to map back
to source code.

Inputs:
  --lrm    LRM dump produced by `nvdisasm --print-life-ranges -lrm count
           --cuda-function-index N <cubin>`. Each instruction is annotated
           with `// | regs | preds | uregs | upreds |` trailing columns.
  --lineinfo  Annotated dump produced by `nvdisasm --print-line-info
              --cuda-function-index N <cubin>`. Each instruction is preceded
              by a `//## File "/path/to/foo.cuh", line N` comment.
  --kernel-substring  Substring used to locate the kernel's `.text._<sym>`
                      section. Defaults to "device_segmented_topk_filter_kernel".
  --top N  Number of distinct peak source lines to print.
  --window K  Print +/- K SASS lines around each peak occurrence.

Outputs a Markdown report on stdout (or --out PATH).
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path


RE_TEXT_SECTION = re.compile(r"^\s*\.text\._")
RE_INSN = re.compile(
    r"^\s*/\*([0-9a-f]+)\*/\s+(.*?);"  # offset, instruction body
    r"\s*//\s*\|\s*(\d+)?\s*\|\s*(\d+)?\s*\|\s*(\d+)?\s*\|\s*(\d+)?\s*\|"
)
RE_LABEL = re.compile(r"^\s*\.L_\S+:")

# Line info markers:
#   //## File "...path...", line N
RE_LINEINFO = re.compile(
    r"^\s*//##\s+File\s+\"([^\"]+)\",\s+line\s+(\d+)"
)
#   //## inline_call N, file "...", line M
RE_INLINE = re.compile(
    r"^\s*//##\s+(inline_call|inlined_at)\s+\d+,\s+file\s+\"([^\"]+)\",\s+line\s+(\d+)"
)
# Within lineinfo files, an SASS instruction looks like
#   "/*offset*/   OP ... ;   /* encoding */"
RE_INSN_NOLRM = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(.*?);")


def find_section_offsets(lines: list[str], substr: str) -> tuple[int, int]:
    """Locate the line range that belongs to a single `.text._<substr>` section."""
    start = end = None
    in_section = False
    for i, line in enumerate(lines):
        if RE_TEXT_SECTION.match(line):
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


def parse_lrm(path: Path, kernel_substring: str) -> list[dict]:
    """Return a list of instruction records with life-range info."""
    text = path.read_text(errors="replace").splitlines()
    start, end = find_section_offsets(text, kernel_substring)
    records: list[dict] = []
    for line in text[start:end]:
        m = RE_INSN.match(line)
        if not m:
            continue
        offset_hex, body, r, p, ur, up = m.groups()
        records.append(
            {
                "offset": int(offset_hex, 16),
                "offset_hex": offset_hex,
                "insn": body.strip(),
                "live_regs": int(r) if r else 0,
                "live_preds": int(p) if p else 0,
                "live_uregs": int(ur) if ur else 0,
                "live_upreds": int(up) if up else 0,
            }
        )
    return records


def parse_lineinfo(path: Path, kernel_substring: str) -> dict[int, list[tuple[str, int]]]:
    """Map instruction offset (int) -> list of (file, line) tuples (with inlining)."""
    text = path.read_text(errors="replace").splitlines()
    start, end = find_section_offsets(text, kernel_substring)
    mapping: dict[int, list[tuple[str, int]]] = {}
    pending: list[tuple[str, int]] = []
    for line in text[start:end]:
        m = RE_LINEINFO.match(line)
        if m:
            pending = [(m.group(1), int(m.group(2)))]
            continue
        m = RE_INLINE.match(line)
        if m:
            pending.append((m.group(2), int(m.group(3))))
            continue
        m = RE_INSN_NOLRM.match(line)
        if m:
            offset = int(m.group(1), 16)
            if pending:
                mapping[offset] = list(pending)
    return mapping


def make_report(
    records: list[dict],
    lineinfo: dict[int, list[tuple[str, int]]],
    top: int,
    window: int,
) -> str:
    if not records:
        return "(no records)"

    max_regs = max(r["live_regs"] for r in records)

    # Tag each record with source info (innermost (file, line)).
    for r in records:
        infos = lineinfo.get(r["offset"], [])
        r["source"] = infos[0] if infos else (None, None)
        r["inline_stack"] = infos

    # Find all "peak" instructions (those at or near max).
    peak_threshold = max_regs  # exact-peak only by default
    peak_records = [r for r in records if r["live_regs"] >= peak_threshold]

    # Group by (source_file, source_line) to find which source lines are
    # responsible for peak liveness; sort by frequency.
    line_counter: collections.Counter = collections.Counter()
    for r in peak_records:
        if r["source"] != (None, None):
            line_counter[r["source"]] += 1
        else:
            line_counter[("(no lineinfo)", 0)] += 1

    out: list[str] = []
    out.append(f"# SASS register-liveness analysis")
    out.append("")
    out.append(f"- Total instructions analyzed: **{len(records)}**")
    out.append(f"- Peak live-register count: **{max_regs}**")
    out.append(f"- Instructions at peak: **{len(peak_records)}**")
    out.append(f"- Distinct source lines at peak: **{len(line_counter)}**")
    out.append("")

    out.append(f"## Top {top} source lines contributing to peak liveness")
    out.append("")
    out.append("| count | file | line |")
    out.append("|---|---|---|")
    for (f, ln), n in line_counter.most_common(top):
        out.append(f"| {n} | `{f}` | {ln} |")
    out.append("")

    # Print each peak instruction with a small SASS context window.
    insn_by_offset = {r["offset"]: r for r in records}
    sorted_offsets = sorted(insn_by_offset)
    offset_index = {off: i for i, off in enumerate(sorted_offsets)}

    out.append(f"## SASS context around each peak instruction (±{window})")
    out.append("")
    seen_peaks: set[int] = set()
    for r in peak_records[:top]:
        if r["offset"] in seen_peaks:
            continue
        seen_peaks.add(r["offset"])
        idx = offset_index[r["offset"]]
        lo = max(0, idx - window)
        hi = min(len(sorted_offsets), idx + window + 1)
        f, ln = r["source"]
        out.append(f"### peak @ 0x{r['offset_hex']}  live_regs={r['live_regs']} `{f}:{ln}`")
        if len(r["inline_stack"]) > 1:
            out.append("Inline stack:")
            for sf, sln in r["inline_stack"]:
                out.append(f"  - `{sf}:{sln}`")
        out.append("")
        out.append("```")
        for off in sorted_offsets[lo:hi]:
            rec = insn_by_offset[off]
            mark = "  *" if off == r["offset"] else "   "
            out.append(
                f"{mark} 0x{rec['offset_hex']:>6}  R={rec['live_regs']:>3}  P={rec['live_preds']}  UR={rec['live_uregs']}  UP={rec['live_upreds']}   {rec['insn']}"
            )
        out.append("```")
        out.append("")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrm", type=Path, required=True)
    ap.add_argument("--lineinfo", type=Path, required=True)
    ap.add_argument("--kernel-substring", default="device_segmented_topk_filter_kernel")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    records = parse_lrm(args.lrm, args.kernel_substring)
    lineinfo = parse_lineinfo(args.lineinfo, args.kernel_substring)
    report = make_report(records, lineinfo, args.top, args.window)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
        print(f"Wrote {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    sys.exit(main() or 0)
