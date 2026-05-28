#!/usr/bin/env python3
"""Build a snapshot of kernel resource usage from a ptxas verbose build log.

Each snapshot is a JSON document containing:
- metadata (label, branch, sha, datetime, build flags, etc.)
- kernels list (one record per kernel instantiation)

Use comparing snapshots to track register/stack/smem regressions over versions.

Usage:
    build_snapshot.py <log> --label LABEL --branch BRANCH --sha SHA \
        [--commit-subject SUBJECT] [--build-target TARGET] \
        [--build-seconds N] [--out PATH]
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path


# Canonical mapping: ptxas kernel name -> short "logical" name. We use the
# logical name to compare equivalent kernels across the main / dev versions
# (main names its kernels differently from dev).
KERNEL_LOGICAL_NAME = {
    "DeviceTopKKernel": "filter",
    "DeviceTopKHistogramKernel": "initial_histogram",
    "DeviceTopKLastFilterKernel": "last_filter",
    "device_segmented_topk_filter_kernel": "filter",
    "device_segmented_topk_histogram_kernel": "initial_histogram",
    "device_segmented_topk_last_filter_kernel": "last_filter",
    "device_segmented_topk_finalize_filter_kernel": "finalize_filter",
    "device_segmented_topk_finalize_histogram_kernel": "finalize_histogram",
    "device_segmented_topk_kernel": "single_cta",
}


RE_COMPILING = re.compile(
    r"^ptxas info\s*:\s*Compiling entry function '([^']+)' for '([^']+)'\s*$"
)
RE_FN_PROPS = re.compile(r"^ptxas info\s*:\s*Function properties for (\S+)\s*$")
RE_STACK = re.compile(
    r"^\s*(\d+)\s+bytes stack frame,\s*(\d+)\s+bytes spill stores,\s*(\d+)\s+bytes spill loads\s*$"
)
RE_USED = re.compile(r"^ptxas info\s*:\s*Used\s+(\d+)\s+registers")
RE_BARRIERS = re.compile(r"used\s+(\d+)\s+barriers")
RE_SMEM = re.compile(r"(\d+)\s+bytes smem")
RE_CMEM = re.compile(r"(\d+)\s+bytes cmem\[(\d+)\]")
RE_COMPILE_TIME = re.compile(
    r"^ptxas info\s*:\s*Compile time =\s*([\d.]+)\s*ms\s*$"
)


def short_kernel_name(mangled: str) -> str | None:
    for name in KERNEL_LOGICAL_NAME:
        if name in mangled:
            return name
    return None


def parse_ptxas(log_text: str) -> list[dict]:
    records: list[dict] = []
    current: dict | None = None
    awaiting_stack = False

    def flush():
        nonlocal current
        if current is not None:
            records.append(current)
            current = None

    for line in log_text.splitlines():
        m = RE_COMPILING.match(line)
        if m:
            flush()
            mangled = m.group(1)
            current = {
                "mangled": mangled,
                "arch": m.group(2),
                "kernel_name": short_kernel_name(mangled),
                "registers": None,
                "stack_frame": None,
                "spill_stores": None,
                "spill_loads": None,
                "smem_bytes": 0,
                "barriers": 0,
                "cmem": {},
                "compile_time_ms": None,
            }
            awaiting_stack = False
            continue

        m = RE_FN_PROPS.match(line)
        if m and current is not None and m.group(1) == current["mangled"]:
            awaiting_stack = True
            continue

        if awaiting_stack:
            m = RE_STACK.match(line)
            if m:
                current["stack_frame"] = int(m.group(1))
                current["spill_stores"] = int(m.group(2))
                current["spill_loads"] = int(m.group(3))
                awaiting_stack = False
                continue

        m = RE_USED.match(line)
        if m and current is not None:
            current["registers"] = int(m.group(1))
            mb = RE_BARRIERS.search(line)
            if mb:
                current["barriers"] = int(mb.group(1))
            ms = RE_SMEM.search(line)
            if ms:
                current["smem_bytes"] = int(ms.group(1))
            for size_s, idx_s in RE_CMEM.findall(line):
                current["cmem"][idx_s] = int(size_s)
            continue

        m = RE_COMPILE_TIME.match(line)
        if m and current is not None:
            current["compile_time_ms"] = float(m.group(1))
            flush()
            continue

    flush()
    return records


def demangle(mangled_list: list[str]) -> dict[str, str]:
    if not mangled_list:
        return {}
    proc = subprocess.run(
        ["c++filt"],
        input="\n".join(mangled_list),
        text=True,
        capture_output=True,
        check=True,
    )
    return {m: d for m, d in zip(mangled_list, proc.stdout.splitlines())}


_PATTERN_POLICY_DEV = re.compile(
    # dev: policy_selector_from_types<KeyT, ValueT, OffsetT, K_max>
    r"policy_selector_from_types<([^<>,]+),\s*([^<>,]+),"
)
_PATTERN_POLICY_MAIN = re.compile(
    # main: policy_selector_from_types<KeyT> then key/value iterators follow.
    # We pick KeyT from the policy template, then ValueT as the 3rd iterator
    # element type (after the policy `>,` we have `KeyT const*, KeyT*, ValueT const*, ValueT*`).
    # ValueT may be `cub::_V_30...::NullType` for keys-only kernels, so the type-name
    # character class needs to admit `:` and `_` in addition to alnum + space.
    r"policy_selector_from_types<([^<>,]+)>,\s*([\w :]+?) const\*,\s*\2\*,\s*([\w :]+?) const\*"
)
_PATTERN_SELECT = re.compile(r"topk::select\)(\d)")


def extract_types(demangled: str) -> dict:
    """Pull out KeyT, ValueT, and select direction from a demangled signature.

    The dev / main branches share the `policy_selector_from_types` naming, but
    the dev variant takes (KeyT, ValueT, OffsetT, K_max) while main only takes
    (KeyT). For main, ValueT shows up as the 3rd iterator-of-pointer template
    arg of the kernel.
    """
    key_t = None
    value_t = None

    m_dev = _PATTERN_POLICY_DEV.search(demangled)
    if m_dev:
        key_t = m_dev.group(1).strip()
        value_t = m_dev.group(2).strip()
    else:
        m_main = _PATTERN_POLICY_MAIN.search(demangled)
        if m_main:
            key_t = m_main.group(1).strip()
            # group(2) is KeyT (used to assert via backref); group(3) is ValueT.
            value_t = m_main.group(3).strip()

    m_sel = _PATTERN_SELECT.search(demangled)
    select_dir = None
    if m_sel:
        select_dir = "min" if m_sel.group(1) == "0" else "max"

    return {"key_t": key_t, "value_t": value_t, "select": select_dir}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, help="ptxas verbose build log")
    ap.add_argument("--label", required=True, help="snapshot label, e.g. 'main' or 'dev'")
    ap.add_argument("--branch", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--commit-subject", default="")
    ap.add_argument("--build-target", default="cub.bench.topk.pairs.base")
    ap.add_argument("--build-seconds", type=float, default=0.0)
    ap.add_argument("--gpu-arch", default="sm_100")
    ap.add_argument("--ctk-version", default="")
    ap.add_argument("--build-flags", default="")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    log_text = args.log.read_text(errors="replace")
    records = parse_ptxas(log_text)

    # Demangle unique mangled names
    unique_mangled = sorted({r["mangled"] for r in records})
    demangled_map = demangle(unique_mangled)

    # Augment records with derived type info
    enriched = []
    for r in records:
        d = demangled_map.get(r["mangled"], r["mangled"])
        if r["kernel_name"] is None:
            continue
        types = extract_types(d)
        out = {
            "kernel_name": r["kernel_name"],
            "logical_name": KERNEL_LOGICAL_NAME[r["kernel_name"]],
            "key_t": types["key_t"],
            "value_t": types["value_t"],
            "select": types["select"],
            "registers": r["registers"],
            "stack_frame": r["stack_frame"],
            "spill_stores": r["spill_stores"],
            "spill_loads": r["spill_loads"],
            "smem_bytes": r["smem_bytes"],
            "barriers": r["barriers"],
            "cmem": r["cmem"],
            "arch": r["arch"],
            "compile_time_ms": r["compile_time_ms"],
            "demangled": d,
            "mangled": r["mangled"],
        }
        enriched.append(out)

    snapshot = {
        "metadata": {
            "label": args.label,
            "branch": args.branch,
            "commit_sha": args.sha,
            "commit_sha_short": args.sha[:10],
            "commit_subject": args.commit_subject,
            "datetime": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "gpu_arch": args.gpu_arch,
            "ctk_version": args.ctk_version,
            "build_target": args.build_target,
            "build_flags": args.build_flags,
            "build_seconds": args.build_seconds,
            "log_file": str(args.log.name),
            "record_count": len(enriched),
        },
        "kernels": enriched,
    }

    out = args.out
    if out is None:
        out = (
            Path(__file__).resolve().parent
            / "snapshots"
            / f"{args.label}.json"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"Wrote {out} (records: {len(enriched)})")


if __name__ == "__main__":
    sys.exit(main() or 0)
