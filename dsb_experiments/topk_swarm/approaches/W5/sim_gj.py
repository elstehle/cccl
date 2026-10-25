#!/usr/bin/env python3
"""W5 bound family (G, J): split the 32 lanes into G groups of 32 / G adjacent lanes, each group
certifies its J largest words, the threshold is the minimum over groups of the group's J-th
largest. G x J >= k certifies at least k items. Candidate count = exact rank of the threshold word.
Also 'exact_cert' = the k-th largest among all G x J certified words (what a k-round W2 over the
certified words would give). Striped lanes, harness random pattern, word space (key desc, idx asc).
"""
import argparse
import numpy as np
from sim_candidates import words, lane_view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--n", default="512,1024,2048,4096")
    ap.add_argument("--k", default="4,8,16,32,64,128")
    ap.add_argument("--keys", default="bf16,fp32")
    ap.add_argument("--combos", default="32x1,16x1,8x1,4x1,32x2,16x2,8x2,4x2,32x4,16x4,8x4,32x8,16x8")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    combos = [tuple(map(int, c.split("x"))) for c in a.combos.split(",")]
    print(f"{'kt':5}{'N':>6}{'k':>5}  {'G x J':8}{'mean':>8}{'p99':>7}{'p999':>7}{'max':>7}"
          f"  ovf D1   ovf D2   ovf D4   ovf D8  ovf D16  maxlane")
    for kt in a.keys.split(","):
        for N in map(int, a.n.split(",")):
            vals = rng.standard_normal((a.trials, N), dtype=np.float32)
            w = words(vals, kt)
            lv = lane_view(w)
            T, L, IPT = lv.shape
            # per-lane top-8 sorted descending
            top8 = -np.sort(-lv.astype(np.int64), axis=2)[:, :, :8]
            for k in map(int, a.k.split(",")):
                if 2 * k > N:
                    continue
                for G, J in combos:
                    if G * J < k or J > 8:
                        continue
                    # smallest certified set: G x J >= k, skip combos that over-certify by more than 2x
                    if G * J > 2 * k and not (G * J == 32 and J == 1):
                        continue
                    lanes_per_group = 32 // G
                    grp = top8.reshape(T, G, lanes_per_group * 8)
                    gs = -np.sort(-grp, axis=2)[:, :, :J]  # group top-J
                    t = gs[:, :, J - 1].min(axis=1)
                    cand = (w >= t[:, None]).sum(axis=1)
                    per_lane = (lv >= t[:, None, None]).sum(axis=2).max(axis=1)
                    ovf = [f"{(cand > 32 * D).mean():8.4f}" for D in (1, 2, 4, 8, 16)]
                    print(f"{kt:5}{N:6}{k:5}  {G:2}x{J:<4}{cand.mean():8.1f}{np.percentile(cand, 99):7.0f}"
                          f"{np.percentile(cand, 99.9):7.0f}{cand.max():7d} " + " ".join(ovf) + f"{per_lane.max():8d}")
                    if G * J >= k and G * J <= 2 * k:
                        allc = gs.reshape(T, G * J)
                        te = -np.partition(-allc, k - 1, axis=1)[:, k - 1]
                        cand = (w >= te[:, None]).sum(axis=1)
                        per_lane = (lv >= te[:, None, None]).sum(axis=2).max(axis=1)
                        ovf = [f"{(cand > 32 * D).mean():8.4f}" for D in (1, 2, 4, 8, 16)]
                        print(f"{'':5}{'':6}{'':5}  {'  kth':8}{cand.mean():8.1f}{np.percentile(cand, 99):7.0f}"
                              f"{np.percentile(cand, 99.9):7.0f}{cand.max():7d} " + " ".join(ovf) + f"{per_lane.max():8d}")
                print()


if __name__ == "__main__":
    main()
