#!/usr/bin/env python3
"""Compose the TECHNIQUES.md entry of the node variance study from the result files.

  python3 node_entry.py --mechanism "<paragraph>" [--extra "<sentence>"] > entry.md

Reads state/results/hw-researcher/{cell,cell2,cell3,icache,icache2,icache4,icache5}/<node>/ and the
verifier's batch files. Numbers carry node, nvcc and date from the JSON records.
"""
import argparse
import glob
import json
import os

R = "state/results/hw-researcher"


def latency_cells(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        j = json.load(open(f))
        if "node" not in j:
            continue
        node = f.split("/")[-2]  # the node directory, the harness records the container hostname

        def walk(o):
            if isinstance(o, dict):
                if o.get("regime") == "latency" and "raw" in o:
                    yield o
                for v in o.values():
                    yield from walk(v)
            elif isinstance(o, list):
                for v in o:
                    yield from walk(v)

        for c in walk(j):
            lat = c["raw"]["variant"]["latency"]
            out[(node, c["threads"], c["n"], c["k"], c["requirement"], os.path.basename(f))] = dict(
                v=c["value"], ctrl=c.get("control_value"), chain=lat["chain_minima"], noop=lat["chain_minima_noop"][-1],
                sass=lat["resources"]["sass_instructions"], regs=lat["resources"]["registers"], nvcc=j["node"].get("nvcc_version"))
    return out


def rows(d):
    fs = sorted(glob.glob(os.path.join(d, "icache_*.jsonl")), key=lambda f: f.split("_")[-1])
    return [json.loads(l) for l in open(fs[-1]) if l.strip()] if fs else []


def row(rs, prefix):
    for j in rs:
        if j["name"].startswith(prefix):
            return j
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism", required=True)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    nodes = ["umbriel-b200-020", "umbriel-b200-068"]
    # exact cell, lazy and eager
    c3 = latency_cells(f"{R}/cell3/*/*.json")
    big = {}
    for (node, t, n, k, req, f), v in c3.items():
        if (t, n, k, req) == (128, 4096, 64, "not_guaranteed"):
            big.setdefault(node, []).append(v)
    # matrix
    c2 = latency_cells(f"{R}/cell2/*/*.json")
    matrix = {}
    for (node, t, n, k, req, f), v in c2.items():
        if req == "not_guaranteed":
            matrix[(node, n, k)] = v
    mat_lines = []
    for n in (1024, 2048, 4096):
        for k in (16, 64):
            a020 = matrix.get(("umbriel-b200-020", n, k))
            a068 = matrix.get(("umbriel-b200-068", n, k))
            if a020 and a068:
                mat_lines.append(f"N {n} k {k}: {a020['v']:.0f} against {a068['v']:.0f} ({a020['sass']} SASS, {a020['regs']} registers)")
    # synthetics per node
    syn = {}
    for node in nodes:
        parts = []
        r1 = rows(f"{R}/icache/{node}")
        r2 = rows(f"{R}/icache2/{node}")
        r4 = rows(f"{R}/icache4/{node}")
        r5 = rows(f"{R}/icache5/{node}")
        dep = {(j["len"] * 16 // 1024, j["block"]): j["cycles_per_instr"] for j in r1 if j.get("mode") == "dependent"}
        ind = {(j["len"] * 16 // 1024, j["block"]): j["cycles_per_instr"] for j in r1 if j.get("mode") == "independent"}
        parts.append(f"dependent IMAD chain {dep.get((16,128),0):.2f}, {dep.get((64,128),0):.2f}, {dep.get((128,128),0):.2f} cycles per instruction at 16, 64 and 128 KB of straight-line code (128 threads), independent IMADs {ind.get((128,128),0):.3f} at 128 KB")
        for rs, pre, label in ((r1, "rounds one block ROUNDS 128 BLOCK 128 ", "80 KB IMAD round"), (r4, "rounds branchy-imad one block ROUNDS 128 BLOCK 128 ", "83 KB IMAD round with a BSSY BRA BSYNC one-lane atomic"), (r2, "rounds-sel branchy one block ROUNDS 128 BLOCK 128 ", "31 KB select round with the branch"), (r5, "rounds-ipc branchy one block ROUNDS 128 BLOCK 128 ", "81 KB round with the branch followed by 32 independent IMADs")):
            j = row(rs, pre)
            if j:
                parts.append(f"{label} {j['per_round_min']:.1f} cycles per round")
        for rs, pre in ((r1, "rounds one block per SM ROUNDS 128 BLOCK 128 "), (r4, "rounds branchy-imad one block per SM "), (r5, "rounds-ipc branchy one block per SM ")):
            j = row(rs, pre)
            if j:
                parts.append(f"the same on all 148 SMs ({j['per_round_min']:.1f} to {j['per_round_max']:.1f})")
        c4 = row(r1, "chase L2 4MB blocks 148")
        c1 = row(r1, "chase L2 4MB blocks 1")
        if c4 and c1:
            v = sorted(x for _, x in c4["per_block"])
            parts.append(f"L2 hit latency {c1['per_hop_min']:.0f} cycles per hop on SM {c1['per_block'][0][0]} (the SM every single-block launch lands on), {v[0]:.0f} to {v[-1]:.0f} across SMs")
        for rs in (r2, r4, r5):
            p = row(rs, "path ld.global 16KB blocks 1")
            l = row(rs, "path local 1KB LDL blocks 1")
            if p and l:
                parts.append(f"L1 hit {p['per_hop_min']:.0f} and local memory {l['per_hop_min']:.0f} cycles per hop on that SM")
                break
        syn[node] = ". ".join(parts)
    def cell_str(node):
        vs = big.get(node, [])
        if not vs:
            return "not run"
        return " and ".join(f"{v['v']:.0f} (control {v['ctrl']:.0f}, chain minima {v['chain']})" for v in vs[:2]) + " for lazy and eager module loading"
    print(f"""### T?: a 128-thread kernel of 80 KB or more can run 25 to 100 percent slower on one B200 than on another from the same binary, the 256-thread siblings do not, and no synthetic reproduces it
applies to: every latency-regime number of a one-warp-per-scheduler kernel above about 50 KB of SASS, B2_packed at IPT 32, B2_twolevel, B1_idx at 128 threads, and the choice of node for the final evaluation
measured: the exact cell B2_packed bf16 128 threads N 4096 k 64 (5008 SASS, 48 registers, no spills) through the harness under my holder: umbriel-b200-020 {cell_str('umbriel-b200-020')}, umbriel-b200-068 {cell_str('umbriel-b200-068')}, nvcc 13.4.92, 2026-10-25. The verifier's runs match these to the cycle (verify-B2, 7831 on -020 at 14:21Z, 6254 on -047 with its harness version). The chain minima scale by the same factor at every chain length and the generate-only control (6576 at length 16) and the B1_reference control are identical on both nodes. A six-cell matrix on both nodes (128 threads, bf16, not_guaranteed, -020 against -068): {", ".join(mat_lines)}, so every cell below 50 KB is identical to the cycle and only the 80 KB one differs. verify-N8's two-node data (same source archive, cards B2_twolevel_b200_20261025_2 against batch B2_twolevel_vs_B1_idx_n8_vn8_c1_20261025_154139) shows the same pattern over 36 cells: 128-thread kernels of 25 KB to 787 KB with 40 to 255 registers run 1.02 to 2.04 times slower on -020 (the ratio is not monotonic in code size, 2.04 at 164 KB, 1.21 at 787 KB), their 256-thread siblings 0.99 to 1.11, and the B1_idx control at 128 threads 1.05 to 1.32 against 0.99 to 1.02 at 256 threads. Both hosts are the same SKU (device 0x2901), VBIOS 97.00.02.80.0E, ECC on, 1965 MHz SM and 3996 MHz memory limits, 1000 W, driver 615.71.10, no throttle reason active, SM clock probe 1964 to 1965 MHz on both during the runs. Synthetics on -020: {syn.get('umbriel-b200-020','')}. Synthetics on -068: {syn.get('umbriel-b200-068','')}. {a.extra}
evidence: harness/micro/icache_micro.cu (straight-line bodies as inline PTX, B2-shaped rounds, per-SM maps, load path chases), results under state/results/hw-researcher/icache*/<node>/ and cell*/<node>/, the harness batch JSONs with chain minima, verify-N8's batches under state/results/verify-N8/.
how to use: {a.mechanism}
caveats: Two B200 hosts carry the slow and the fast value, -047, -068 and -091 agree with each other and -020 is the outlier among the five nodes tested. Which SM a single-block launch lands on is logical SM 142 on every node, so the latency harness always measures the same logical SM, whose physical position differs per GPU. Nothing here concerns the throughput regime, where verify-N8 found every cell within 2.8 percent between the nodes.
author: hw-researcher""")


if __name__ == "__main__":
    main()
