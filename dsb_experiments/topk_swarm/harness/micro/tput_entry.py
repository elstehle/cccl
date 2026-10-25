#!/usr/bin/env python3
"""Emit the first version of the TECHNIQUES.md pipe throughput entry from the fused rows of intrinsics_micro.cu.

T47 as it stands in TECHNIQUES.md was rewritten by hand after the unfused rerun (alternating max and min rows,
half-rate pipes) and carries opt-N5's codegen note, so this script documents the data path, not the final text.

  python3 tput_entry.py <results-dir> <stamp> [<results-dir> <stamp> ...] > entry.md

Each results dir holds micro_sm_100_<stamp>.jsonl with the "tput ..." rows and sass_sm_100.txt of the
binary that produced them. Numbers carry node, nvcc and date from the JSON.
"""
import json
import re
import sys
from collections import Counter

OPS = [("TpMaxBf16x2", "max.bf16x2 HMNMX2"), ("TpMaxU32", "max.u32 VIMNMX"), ("TpMaxU16x2", "max.u16x2 VIMNMX.U16x2"),
       ("TpSetBf16x2", "set.gt.u32.bf16x2 HSET2"), ("TpLop3", "lop3 LOP3"), ("TpImad", "mad.lo.u32 IMAD"),
       ("TpMaxF32", "max.f32 FMNMX"), ("TpSel", "setp+selp ISETP+SEL"),
       ("TpStageBf16x2", "stage bf16x2: shfl + max + min + sel"), ("TpStageU16x2", "stage u16x2: shfl + max + min + sel")]
TAGS = ["1 warp", "4 warps (128 thr)", "8 warps (256 thr)", "16 warps (512 thr)"]


def sass_per_step(sass, op):
    for fn in re.split(r"\n\s*Function : ", sass)[1:]:
        name = fn.split("\n", 1)[0]
        if "TputI" in name and op + "Li8ELi128EELi2EE" in name:
            ins = [re.sub(r"\s*/\*[0-9a-f]{4,}\*/\s*", "", l).split(";")[0].strip()
                   for l in fn.split("\n") if re.match(r"\s*/\*[0-9a-f]{4,}\*/", l)]
            idx = [i for i, s in enumerate(ins) if "CS2R" in s and "CLOCK" in s]
            body = ins[idx[0] + 1:idx[1]] if len(idx) >= 2 else ins
            c = Counter((s.split()[1] if s.startswith("@") else s.split()[0]).split(".")[0] for s in body if s)
            return (len(body) - 8) / 2.0, c
    return None, None


def main():
    args = sys.argv[1:]
    runs = []
    for d, stamp in zip(args[0::2], args[1::2]):
        rows = {}
        for line in open(f"{d}/micro_sm_100_{stamp}.jsonl"):
            if line.strip():
                j = json.loads(line)
                rows[j["name"]] = j
        sass = open(f"{d}/sass_sm_100.txt").read()
        runs.append((d, stamp, rows, sass))
    d0, stamp0, rows0, sass0 = runs[0]
    any_row = next(iter(rows0.values()))
    prov = f"{any_row['device']} sm_100, {any_row['node']}, nvcc {any_row['nvcc']}, {any_row['date']}, file harness/micro/intrinsics_micro.cu rows \"tput ...\" in {d0}/micro_sm_100_{stamp0}.jsonl"
    lines = []
    for op, label in OPS:
        cyc = [rows0.get(f"tput {label} x8 indep, {t}", {}).get("cycles", float("nan")) for t in TAGS]
        n, c = sass_per_step(sass0, op)
        main_ops = ", ".join(f"{k} x{v}" for k, v in c.most_common(3)) if c else "?"
        lines.append(f"{label}: {cyc[0]:.1f}, {cyc[1]:.1f}, {cyc[2]:.1f}, {cyc[3]:.1f} cycles per step at 1, 4, 8, 16 warps per block ({n:.0f} SASS per step at 128 threads: {main_ops})")
    body = "; ".join(lines).replace("; ", ". ")
    print(f"""### T?: bf16x2 and u16x2 compare-exchange stages cost the same and are dispatch bound at about one warp instruction per cycle per scheduler, min and max run at full rate, the integer add, multiply and logic pipe at half rate
applies to: N5 two-row bf16x2 networks, W1 packed and one-word networks (T42), any issue-bound network at 1 to 4 warps per scheduler, throughput regime
measured: 8 independent accumulators per lane, one operation each per step as inline PTX (consecutive maxes of one accumulator fuse into the three-input VHMNMX, VIMNMX3 and FMNMX3, counted as one instruction), run by 1, 4, 8 and 16 warps per block, which is 0.25, 1, 2 and 4 warps per scheduler. {body}. A bitonic stage at 2 warps per scheduler is 73 instructions in 74 cycles and at 4 warps 146 in 139 for bf16x2, 135 for u16x2, so both forms sit at the dispatch limit of about one warp instruction per cycle per scheduler and differ by 3 percent. The fused min max forms sustain about 0.8 per cycle per scheduler at 4 warps per scheduler (32 per step in 40 cycles), HSET2 about 1 per cycle, IMAD about 0.7 and LOP3 with its IMADs about 0.35, so the integer add, multiply and logic pipe is the half-rate one. {prov}.
evidence: sass_sm_100.txt of that run, kernels `bench<Tput<Tp..., 8, 128>, 2>`: `VHMNMX.BF16_V2` x8, `VIMNMX3.U32` x8, `VIMNMX3.U16x2` x8, `FMNMX3` x8 per two steps with the operand `IMAD`s, `HSET2.BF16_V2.GT.AND` x16, the stage kernels `HMNMX2.BF16_V2` x32 or `VIMNMX.U16x2` x32 plus `SHFL.BFLY` x16 plus 29 `IMAD` and select instructions per two steps.
how to use: Pick the packed bf16x2 or the twiddled u16x2 exchange by instruction count and by the cost of the compare masks (T18: one HSET2 for bf16, an emulation for u16x2), not by pipe rate, both min max forms issue at the dispatch rate. In an issue-bound network every instruction costs one cycle per warp per scheduler, so removing a shuffle, a select or an address IMAD from a stage is worth as much as removing a max, and the IMAD, IADD and LOP3 count matters twice because that pipe is half rate. Fusing nested maxes into VIMNMX3 or VHMNMX halves their instruction count for free (T7).
caveats: Single block per SM, so at most 4 warps per scheduler. The per-accumulator operand adds of this first run share the integer pipe with the measured op and dilute the pure pipe rates, a rerun with one operand per step refines them. SHFL throughput is in T33 (about one per cycle per SM).
author: hw-researcher""")


if __name__ == "__main__":
    main()
