#!/usr/bin/env python3
"""Parse `cuobjdump --dump-resource-usage` output for CUB top-k benchmark binaries.

Recognizes both the single-problem top-k kernels and the segmented/batched
top-k kernels that the `CUB_BENCH_TOPK_USE_BATCHED=1` benchmark currently
emits:

  Single-problem (DeviceTopK):
    * DeviceTopKHistogramKernel
    * DeviceTopKFilterKernel
    * DeviceTopKLastFilterKernel

  Segmented / batched (cub::detail::batched_topk):
    * device_segmented_topk_kernel               (worker-per-segment, small)
    * device_segmented_topk_histogram_kernel     (multi-CTA-per-segment, pass 0)
    * device_segmented_topk_filter_kernel        (multi-CTA-per-segment, passes 1..N-1)
    * device_segmented_topk_last_filter_kernel   (multi-CTA-per-segment, last pass)

  In the segmented path, OffsetT / OutOffsetT are fixed to `unsigned int` by
  the dispatch regardless of the benchmark's offset axis (see
  `dispatch_batched_topk.cuh`). Filtering to `u32 u32` is a no-op there.

  For each kernel we demangle the name, pull out (Key, Value, OffsetT,
  OutOffsetT), and print a table of registers / smem / stack / lmem / cmem.
  By default only the 32-bit offset combination
  (OffsetT == OutOffsetT == uint32_t) is reported.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass

# Single-problem top-k kernels.
SINGLE_KERNEL_NAMES = (
    "DeviceTopKHistogramKernel",
    "DeviceTopKFilterKernel",
    "DeviceTopKLastFilterKernel",
)

# Segmented / batched top-k kernels emitted by the `CUB_BENCH_TOPK_USE_BATCHED=1`
# benchmark.
SEGMENTED_KERNEL_NAMES = (
    "device_segmented_topk_kernel",
    "device_segmented_topk_histogram_kernel",
    "device_segmented_topk_filter_kernel",
    "device_segmented_topk_last_filter_kernel",
)

KERNEL_NAMES = SINGLE_KERNEL_NAMES + SEGMENTED_KERNEL_NAMES

# Itanium mangling of the fundamental types we care about.
MANGLE_TO_TYPE = {
    "a": "i8",
    "h": "u8",
    "c": "char",
    "s": "i16",
    "t": "u16",
    "i": "i32",
    "j": "u32",
    "l": "i64",
    "m": "u64",
    "x": "i64",
    "y": "u64",
    "n": "i128",
    "o": "u128",
    "f": "f32",
    "d": "f64",
}


@dataclass
class Kernel:
    kernel: str
    key: str
    value: str
    offset: str
    out_offset: str
    reg: int
    stack: int
    smem: int
    lmem: int
    cmem0: int


def short_type(demangled: str) -> str:
    table = {
        "double": "f64",
        "float": "f32",
        "signed char": "i8",
        "unsigned char": "u8",
        "char": "char",
        "short": "i16",
        "unsigned short": "u16",
        "int": "i32",
        "unsigned int": "u32",
        "long": "i64",
        "unsigned long": "u64",
        "long long": "i64",
        "unsigned long long": "u64",
        "__int128": "i128",
        "unsigned __int128": "u128",
        "cub::NullType": "-",
    }
    return table.get(demangled.strip(), demangled.strip())


def demangle(name: str) -> str:
    out = subprocess.check_output(["c++filt", name]).decode().strip()
    return out


def parse_kernel(demangled: str) -> tuple[str, str, str, str, str] | None:
    """Return (kernel, KeyT, ValueT, OffsetT, OutOffsetT) or None if not topk."""
    # Strip everything up to the kernel name's templated `name<...`. Accepts
    # both `detail::topk::DeviceTopK*Kernel` (single-problem) and
    # `detail::batched_topk::device_segmented_topk*_kernel` (segmented).
    m = re.search(r"detail::(?:topk|batched_topk)::(\w+)<(.+)$", demangled)
    if not m:
        return None
    kernel = m.group(1)
    if kernel not in KERNEL_NAMES:
        return None

    if kernel in SEGMENTED_KERNEL_NAMES:
        return parse_segmented_kernel(kernel, demangled)
    return parse_single_kernel(kernel, demangled)


def split_top_level_commas(s: str) -> list[str]:
    """Split `s` at top-level commas, respecting `<...>` nesting and ignoring
    commas inside parentheses. Doesn't bother with strings/escapes -- the
    inputs are demangled C++ signatures, not source code."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(s):
        if c in "<(":
            depth += 1
        elif c in ">)":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(s[start:i].strip())
            start = i + 1
    parts.append(s[start:].strip())
    return parts


def parse_call_args(demangled: str) -> list[str] | None:
    """Return the depth-0 comma-separated argument list of `void NAME<...>(ARGS)`."""
    args_idx = demangled.find(">(", demangled.find("policy_selector_from_types"))
    if args_idx < 0:
        return None
    open_paren = demangled.find("(", args_idx)
    depth = 0
    end = -1
    for i in range(open_paren, len(demangled)):
        c = demangled[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None
    args_str = demangled[open_paren + 1 : end]
    return split_top_level_commas(args_str)


def parse_single_kernel(kernel: str, demangled: str) -> tuple[str, str, str, str, str] | None:
    """Decode (kernel, key, value, off, out_off) for the single-problem
    DeviceTopK kernels. KeyT is read from `policy_selector_from_types<KEY>` and
    OffsetT / OutOffsetT / ValueT are read out of the kernel's runtime argument
    list (the template parameter list itself doesn't carry them on this path)."""
    key_match = re.search(r"policy_selector_from_types<([^>]+)>", demangled)
    if not key_match:
        return None
    key = short_type(key_match.group(1))

    args = parse_call_args(demangled)
    if args is None:
        return None

    if kernel == "DeviceTopKHistogramKernel":
        #  (d_keys_in,                         // 0
        #   counter<...>*,                     // 1   <- counter<KeyT, OffsetT, OutOffsetT>
        #   OffsetT*, OffsetT, OutOffsetT,     // 2..4
        #   extract_bin_op_t, int, bool)       // 5..7
        offset_arg = args[3]
        out_off_arg = args[4]
        value = "-"
    elif kernel == "DeviceTopKFilterKernel":
        #  (d_keys_in, d_keys_out, d_values_in, d_values_out,   // 0..3
        #   in_key_buf, in_val_buf, out_key_buf, out_val_buf,   // 4..7
        #   counter, histogram*, k, buffer_length,              // 8..11
        #   extract_bin_op, identify_op, int pass, bool)        // 12..15
        offset_arg = args[11]
        out_off_arg = args[10]
        value = parse_value_from_iter(args[3])
    elif kernel == "DeviceTopKLastFilterKernel":
        #  (d_keys_in, d_keys_out, d_values_in, d_values_out,   // 0..3
        #   in_key_buf, in_val_buf, counter, k, identify_op)    // 4..8
        offset_arg = parse_counter_offset(args[6])
        out_off_arg = args[7]
        value = parse_value_from_iter(args[3])
    else:
        return None

    off = parse_simple_type(offset_arg)
    out_off = parse_simple_type(out_off_arg)
    return kernel, key, value, off, out_off


def parse_segmented_kernel(kernel: str, demangled: str) -> tuple[str, str, str, str, str] | None:
    """Decode (kernel, key, value, off, out_off) for the segmented/batched
    top-k kernels. The signature carries everything we need in the template
    parameter list:

      policy_selector_from_types<KeyT, ValueT, SegmentSizeT, MaxK>
        ^- first two args are the user's KeyT and ValueT (NullType -> keys-only)

      OffsetT and OutOffsetT are the last two template parameters of each
      kernel (always `unsigned int` on the batched dispatch path; see
      `dispatch_batched_topk.cuh` lines 348-349). The worker-per-segment
      kernel (`device_segmented_topk_kernel`) doesn't carry them as separate
      template parameters -- its (effective) offsets are wired in through
      `batched_topk_counters<...>`; we report `u32` for it since that's the
      hardcoded internal type."""
    psel_re = re.compile(r"policy_selector_from_types<")
    m = psel_re.search(demangled)
    if not m:
        return None
    # Find the matching `>` for the policy_selector_from_types template.
    depth = 1
    start = m.end()
    end = -1
    for i in range(start, len(demangled)):
        c = demangled[i]
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None
    psel_args = split_top_level_commas(demangled[start:end])
    if len(psel_args) < 2:
        return None
    key = short_type(psel_args[0])
    value = short_type(psel_args[1])

    # For the multi-CTA kernels OffsetT / OutOffsetT are the last two entries
    # of the top-level template parameter list. The kernel symbol is
    #   void detail::batched_topk::<kernel><TEMPLATES>(ARGS)
    # so we just need the matching `>` of the leading `<` after the kernel
    # name and pull the last two depth-0 entries from inside.
    if kernel == "device_segmented_topk_kernel":
        # Worker-per-segment kernel doesn't carry OffsetT / OutOffsetT in its
        # template list; the batched dispatch hardcodes them to uint32_t.
        return kernel, key, value, "u32", "u32"

    name_match = re.search(rf"{re.escape(kernel)}<", demangled)
    if not name_match:
        return None
    depth = 1
    t_start = name_match.end()
    t_end = -1
    for i in range(t_start, len(demangled)):
        c = demangled[i]
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
            if depth == 0:
                t_end = i
                break
    if t_end < 0:
        return None
    top_template_args = split_top_level_commas(demangled[t_start:t_end])
    if len(top_template_args) < 2:
        return None
    off = parse_simple_type(top_template_args[-2])
    out_off = parse_simple_type(top_template_args[-1])

    # The histogram kernel's body is independent of ValueT (it never touches
    # the values), but the template list captures it via
    # `policy_selector_from_types<KeyT, ValueT, ...>` so we still get one
    # instantiation per (KeyT, ValueT). Collapse those into one row per KeyT
    # by reporting `value = "-"` -- the metrics are identical across ValueT
    # for the same KeyT (verified empirically).
    if kernel == "device_segmented_topk_histogram_kernel":
        value = "-"

    return kernel, key, value, off, out_off


def parse_simple_type(arg: str) -> str:
    arg = arg.strip()
    # Strip CV-qualifiers and pointers up to the leaf type
    arg = arg.replace("const", "").strip()
    if arg.endswith("*"):
        arg = arg[:-1].strip()
    return short_type(arg)


def parse_pointer_value(arg: str) -> str:
    arg = arg.strip()
    if arg.endswith("*"):
        arg = arg[:-1].strip()
    arg = arg.replace("const", "").strip()
    return short_type(arg)


def parse_value_from_iter(arg: str) -> str:
    """ValueOutputIteratorT is either `T*` (materialized) or
    `transform_output_iterator<topk_index_gather_op<T const*>, T*>` (indexed).
    Either way, the user's ValueT lives just inside the rightmost `topk_index_gather_op<...>`
    or is the leaf pointer."""
    arg = arg.strip()
    m = re.search(r"topk_index_gather_op<([^>]+?)\s*const\s*\*\s*>", arg)
    if m:
        return short_type(m.group(1).strip())
    # Fall back to materialized form `ValueT*`.
    if arg.endswith("*"):
        return short_type(arg[:-1].strip())
    return short_type(arg)


def parse_counter_offset(arg: str) -> str:
    """Extract OffsetT from counter<KeyT, OffsetT, OutOffsetT>*."""
    open_idx = arg.find("counter<")
    if open_idx < 0:
        return "?"
    start = open_idx + len("counter<")
    depth = 1
    parts = []
    cur = []
    for i in range(start, len(arg)):
        c = arg[i]
        if c == "<":
            depth += 1
            cur.append(c)
        elif c == ">":
            depth -= 1
            if depth == 0:
                parts.append("".join(cur).strip())
                break
            cur.append(c)
        elif c == "," and depth == 1:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
    if len(parts) < 3:
        return "?"
    return parse_simple_type(parts[1])


def parse(dump_text: str) -> list[Kernel]:
    """Walk the resource-usage dump and emit one Kernel per topk kernel."""
    kernels: list[Kernel] = []
    lines = dump_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"\s*Function\s+(_Z\S+):\s*$", line)
        if m and any(k in line for k in KERNEL_NAMES):
            mangled = m.group(1)
            try:
                demangled = demangle(mangled)
            except subprocess.CalledProcessError:
                i += 1
                continue
            parsed = parse_kernel(demangled)
            i += 1
            if i >= len(lines):
                break
            usage = lines[i]
            metrics = re.findall(r"(\w+(?:\[\d+\])?):(\d+)", usage)
            metric_map = {k: int(v) for k, v in metrics}
            if parsed is not None:
                kernel, key, value, off, out_off = parsed
                kernels.append(
                    Kernel(
                        kernel=kernel,
                        key=key,
                        value=value,
                        offset=off,
                        out_offset=out_off,
                        reg=metric_map.get("REG", 0),
                        stack=metric_map.get("STACK", 0),
                        smem=metric_map.get("SHARED", 0),
                        lmem=metric_map.get("LOCAL", 0),
                        cmem0=metric_map.get("CONSTANT[0]", 0),
                    )
                )
        i += 1
    return kernels


SHORT_KERNEL = {
    "DeviceTopKHistogramKernel": "histogram",
    "DeviceTopKFilterKernel": "filter   ",
    "DeviceTopKLastFilterKernel": "last_filt",
    "device_segmented_topk_kernel": "worker   ",
    "device_segmented_topk_histogram_kernel": "histogram",
    "device_segmented_topk_filter_kernel": "filter   ",
    "device_segmented_topk_last_filter_kernel": "last_filt",
}


def print_table(kernels: list[Kernel], show_value: bool, sort_by_key_size: bool) -> None:
    key_order = {
        "i8": 1, "u8": 1,
        "i16": 2, "u16": 2,
        "i32": 4, "u32": 4, "f32": 4,
        "i64": 8, "u64": 8, "f64": 8,
        "i128": 16, "u128": 16,
    }
    kernel_order = {
        "DeviceTopKHistogramKernel": 0,
        "DeviceTopKFilterKernel": 1,
        "DeviceTopKLastFilterKernel": 2,
        "device_segmented_topk_kernel": 0,
        "device_segmented_topk_histogram_kernel": 1,
        "device_segmented_topk_filter_kernel": 2,
        "device_segmented_topk_last_filter_kernel": 3,
    }

    def sort_key(k: Kernel):
        return (
            key_order.get(k.key, 99),
            k.key,
            key_order.get(k.value, 0),
            k.value,
            kernel_order[k.kernel],
        )

    kernels = sorted(kernels, key=sort_key)

    if show_value:
        header = f"{'kernel':<10}  {'key':>5}  {'val':>5}  {'off':>4}  {'oOff':>4}  {'REG':>4}  {'smem':>5}  {'stack':>5}  {'lmem':>5}  {'cmem':>5}"
    else:
        header = f"{'kernel':<10}  {'key':>5}  {'off':>4}  {'oOff':>4}  {'REG':>4}  {'smem':>5}  {'stack':>5}  {'lmem':>5}  {'cmem':>5}"
    print(header)
    print("-" * len(header))
    last_key = None
    for k in kernels:
        cur_key = (k.key, k.value) if show_value else (k.key,)
        if last_key is not None and cur_key != last_key:
            print()
        last_key = cur_key
        if show_value:
            print(
                f"{SHORT_KERNEL[k.kernel]:<10}  {k.key:>5}  {k.value:>5}  {k.offset:>4}  {k.out_offset:>4}  "
                f"{k.reg:>4}  {k.smem:>5}  {k.stack:>5}  {k.lmem:>5}  {k.cmem0:>5}"
            )
        else:
            print(
                f"{SHORT_KERNEL[k.kernel]:<10}  {k.key:>5}  {k.offset:>4}  {k.out_offset:>4}  "
                f"{k.reg:>4}  {k.smem:>5}  {k.stack:>5}  {k.lmem:>5}  {k.cmem0:>5}"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="cuobjdump --dump-resource-usage output file")
    p.add_argument("--with-values", action="store_true", help="show value column (pairs)")
    p.add_argument(
        "--offset-32",
        action="store_true",
        help="only show OffsetT == OutOffsetT == u32 rows",
    )
    p.add_argument(
        "--all-offsets", action="store_true", help="show every offset combination"
    )
    p.add_argument(
        "--filter-key",
        default=None,
        help="only emit kernels with this key type (e.g. i32, f32)",
    )
    args = p.parse_args()

    with open(args.path) as f:
        text = f.read()

    kernels = parse(text)
    if not args.all_offsets:
        kernels = [k for k in kernels if k.offset == "u32" and k.out_offset == "u32"]
    if args.filter_key:
        kernels = [k for k in kernels if k.key == args.filter_key]

    # The segmented dispatch instantiates each multi-CTA kernel for both `max`
    # and `min` selection directions (and the histogram once per ValueT even
    # though its body is ValueT-independent). Resource usage is identical for
    # the dropped dimensions, so collapse to one row per (kernel, key, value,
    # off, out_off). Emit a warning to stderr if any drift is observed.
    deduped: dict[tuple[str, str, str, str, str], Kernel] = {}
    for k in kernels:
        key = (k.kernel, k.key, k.value, k.offset, k.out_offset)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = k
            continue
        if (
            existing.reg != k.reg
            or existing.smem != k.smem
            or existing.stack != k.stack
            or existing.lmem != k.lmem
            or existing.cmem0 != k.cmem0
        ):
            print(
                f"warning: resource-usage drift across collapsed instantiations of "
                f"{k.kernel}<{k.key}, {k.value}, {k.offset}, {k.out_offset}>; "
                f"kept first.",
                file=sys.stderr,
            )
    kernels = list(deduped.values())

    print_table(kernels, show_value=args.with_values, sort_by_key_size=True)
    print(f"\n({len(kernels)} kernels shown)")


if __name__ == "__main__":
    main()
