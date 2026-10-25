#!/usr/bin/env python3
"""Host simulation for W5: candidate count after a cheap certified lower bound.

Items are the harness random pattern (unit normal in fp32, optionally rounded to bf16) in a
striped arrangement over 32 lanes (lane t holds items t, t + 32, ...). All comparisons run on
packed words that order by (key descending, index ascending), as the W2 and W5 adapters do, so
ties never inflate a count: the number of candidates is the exact rank of the threshold word.

Bounds simulated (every one certifies at least k items at or above the threshold):
  lane_kth    the k-th largest of the 32 lane maxima (the W5 card's bound, k <= 32)
  grp_J1      min over G groups of 32 / G adjacent lanes of the group maximum, G = next_pow2(k)
              capped at 32 (one xor shuffle tree plus one redux.min), k <= 32
  lane_J2     min over lanes of the lane's second largest (64 certified), k <= 64
  lane_J4     min over lanes of the lane's fourth largest (128 certified), k <= 128
  pair_J4     min over 16 lane pairs of the pair's fourth largest (64 certified), k <= 64
  lane_J2_kth the k-th largest among the 64 words that are top-2 per lane (exact over the
              certified set), k <= 64
Reported per cell: mean, p99, p999 and max of the candidate count, the overflow rate for a
capacity of 32 x D words (D = 2, 4, 8, 16) and the largest per-lane candidate count.
"""
import argparse
import numpy as np


def bf16_round(x):
    b = x.astype(np.float32).view(np.uint32)
    lsb = (b >> 16) & 1
    b = (b + 0x7FFF + lsb) & 0xFFFF0000
    return b.view(np.float32)


def twiddle_f32(x):
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    mask = np.where(u >> 31, np.uint64(0xFFFFFFFF), np.uint64(0x80000000))
    return u ^ mask


def words(vals, kt):
    # trials x N floats to unique words (key desc, index asc)
    T, N = vals.shape
    if kt == "bf16":
        v = bf16_round(vals)
        tw = twiddle_f32(v) >> np.uint64(16)
        inv = (np.uint64(0xFFFF) - np.arange(N, dtype=np.uint64))
        return (tw << np.uint64(16)) | inv[None, :]
    tw = twiddle_f32(vals)
    inv = (np.uint64(N - 1) - np.arange(N, dtype=np.uint64))
    return (tw << np.uint64(16)) | inv[None, :]


def lane_view(w):
    # striped: lane t holds items t, t+32, ... -> trials x 32 x IPT
    T, N = w.shape
    return w.reshape(T, N // 32, 32).transpose(0, 2, 1)


def kth_largest(a, k, axis=-1):
    # k-th largest along axis (k >= 1)
    return -np.partition(-a.astype(np.int64), k - 1, axis=axis).take(k - 1, axis=axis)


def topj_per_lane(lv, j):
    # trials x 32 x IPT -> trials x 32 x j, descending
    s = -np.sort(-lv.astype(np.int64), axis=2)
    return s[:, :, :j]


def thresholds(lv, k):
    T, L, IPT = lv.shape
    out = {}
    lane_max = lv.max(axis=2)  # T x 32
    if k <= 32:
        out["lane_kth"] = kth_largest(lane_max, k, axis=1)
        G = 1
        while G < k:
            G *= 2
        G = min(G, 32)
        grp = lane_max.reshape(T, G, 32 // G).max(axis=2)
        out["grp_J1"] = grp.min(axis=1)
    if k <= 64:
        t2 = topj_per_lane(lv, 2)
        out["lane_J2"] = t2[:, :, 1].min(axis=1)
        out["lane_J2_kth"] = kth_largest(t2.reshape(T, 64), k, axis=1)
        t4 = topj_per_lane(lv, 4)
        pairs = t4.reshape(T, 16, 8)
        p4 = -np.sort(-pairs, axis=2)[:, :, 3]
        out["pair_J4"] = p4.min(axis=1)
    if k <= 128:
        t4 = topj_per_lane(lv, 4)
        out["lane_J4"] = t4[:, :, 3].min(axis=1)
    return out


def run_cell(rng, N, k, kt, trials, pattern):
    if pattern == "random":
        vals = rng.standard_normal((trials, N), dtype=np.float32)
    elif pattern == "sorted_asc":
        vals = np.sort(rng.standard_normal((trials, N), dtype=np.float32), axis=1)
    elif pattern == "sorted_desc":
        vals = -np.sort(-rng.standard_normal((trials, N), dtype=np.float32), axis=1)
    elif pattern == "tie_heavy":
        vals = np.tile((np.arange(N) % 64) / 64.0, (trials, 1)).astype(np.float32)
    elif pattern == "all_equal":
        vals = np.ones((trials, N), dtype=np.float32)
    elif pattern == "uniform":
        vals = rng.random((trials, N), dtype=np.float32)
    else:
        raise ValueError(pattern)
    w = words(vals, kt)
    lv = lane_view(w)
    th = thresholds(lv, k)
    rows = []
    for name, t in th.items():
        cand = (w >= t[:, None]).sum(axis=1)  # exact rank of t
        per_lane = (lv >= t[:, None, None]).sum(axis=2)  # T x 32
        rows.append((name, cand, per_lane.max(axis=1)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pattern", default="random")
    ap.add_argument("--n", default="512,1024,2048,4096")
    ap.add_argument("--k", default="4,8,16,32,64,128")
    ap.add_argument("--keys", default="bf16,fp32")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    print(f"pattern {a.pattern}, trials {a.trials}, striped lanes, word-space thresholds")
    print(f"{'kt':5}{'N':>6}{'k':>5}  {'bound':12}{'mean':>8}{'p99':>7}{'p999':>7}{'max':>7}"
          f"  ovf D2   ovf D4   ovf D8  ovf D16  maxlane")
    for kt in a.keys.split(","):
        for N in map(int, a.n.split(",")):
            for k in map(int, a.k.split(",")):
                if 2 * k > N:
                    continue
                rows = run_cell(rng, N, k, kt, a.trials, a.pattern)
                for name, cand, ml in rows:
                    ovf = [f"{(cand > 32 * D).mean():8.4f}" for D in (2, 4, 8, 16)]
                    print(f"{kt:5}{N:6}{k:5}  {name:12}{cand.mean():8.1f}{np.percentile(cand, 99):7.0f}"
                          f"{np.percentile(cand, 99.9):7.0f}{cand.max():7d} " + " ".join(ovf) + f"{ml.max():8d}")
            print()


if __name__ == "__main__":
    main()
