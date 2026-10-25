#!/usr/bin/env python3
"""Host simulation for the B1 and N6 kill experiments (no GPU).

1. Pass count of radix select with early exit (candidates == k_remaining) on the twiddled key bits for the
   digit schedules of the B1 card: fp32 8x4, 11+11+10, 12+12+8, bf16 8+8.
2. N6: size of the pivot bin after one equi-width pass over the bracket [lo, hi] with 256 bins, hi the block
   max, lo the minimum over threads of the per-thread maximum (a valid lower bound on the k-th largest for
   k <= threads), and the full-range bracket [min, max] for every k. Reports the fraction of rows whose
   pivot bin holds at most 32 items (a warp-sized finish), at most 1 distinct value (a tie class, done by
   tickets), exactly k_remaining items (early exit), and the pivot bin size after a second equi-width pass
   over the pivot bin's sub-range.

Patterns follow harness/topk_harness.cuh make_pattern: random is normal(0, 1) in fp32 rounded to the key
type (numpy's generator, not libstdc++'s, so the rows are statistically equivalent, not bit identical),
sorted inputs have the same value multiset (bin statistics are order independent, only the arrangement
differs), tie_heavy is (j % 64) / 64, and one_outlier is random with one value replaced by 1e30 (fp32) or
the bf16 maximum. Rows: 200 per cell.

  python3 sim_passes.py [--rows 200] [--out sim_passes.md]
"""
import argparse
import numpy as np

THREADS = 256


def to_bf16_bits(x32):
    b = x32.view(np.uint32).astype(np.uint64)
    lsb = (b >> 16) & 1
    rounded = (b + 0x7FFF + lsb) >> 16
    return rounded.astype(np.uint16)


def bf16_to_f32(bits16):
    return (bits16.astype(np.uint32) << 16).view(np.float32)


def round_to_key(v32, key):
    if key == "fp32":
        return v32.astype(np.float32)
    return bf16_to_f32(to_bf16_bits(v32.astype(np.float32)))


def twiddle(v32, key):
    """order-preserving unsigned key, as cub::Traits TwiddleIn (max-direction order = numeric order)"""
    if key == "fp32":
        b = v32.view(np.uint32).astype(np.uint64)
        mask = np.where(b >> 31 == 1, np.uint64(0xFFFFFFFF), np.uint64(0x80000000))
        return (b ^ mask).astype(np.uint32), 32
    b = to_bf16_bits(v32).astype(np.uint32)
    mask = np.where(b >> 15 == 1, np.uint32(0xFFFF), np.uint32(0x8000))
    return (b ^ mask), 16


def make_pattern(name, n, k, rng, key):
    if name in ("random", "sorted"):
        v = rng.standard_normal(n).astype(np.float32)
    elif name == "tie_heavy":
        v = (np.arange(n) % 64).astype(np.float32) / 64.0
    elif name == "one_outlier":
        v = rng.standard_normal(n).astype(np.float32)
        v[rng.integers(n)] = np.float32(1e30) if key == "fp32" else np.float32(3.3e38)
    elif name == "relu_quantized":
        v = rng.standard_normal(n).astype(np.float32)
        v = np.maximum(v, 0)
        r = np.rint(v)
        v = (r + np.rint((v - r) * 32) / 32).astype(np.float32)
    else:
        raise ValueError(name)
    return round_to_key(v, key)


def radix_passes(ukeys, width, k, digits):
    """digits: list of digit widths MSB first (sum == width). Returns the number of passes executed."""
    keys = ukeys.astype(np.uint64)
    cand = np.ones(len(keys), dtype=bool)
    k_rem = k
    hi = width
    passes = 0
    for d in digits:
        lo = hi - d
        digit = (keys >> lo) & ((1 << d) - 1)
        counts = np.bincount(digit[cand].astype(np.int64), minlength=1 << d)
        # from the top bucket down, find the bucket where the cumulative count reaches k_rem
        cum = np.cumsum(counts[::-1])[::-1]  # cum[b] = count of candidates with digit >= b
        above = np.where(cum >= k_rem)[0].max()
        bucket = above
        sel = int(cum[bucket + 1]) if bucket + 1 < (1 << d) else 0
        k_rem -= sel
        cand &= digit == bucket
        passes += 1
        if int(counts[bucket]) == k_rem:
            break
        hi = lo
    return passes


def equiwidth(v, cand, lo, hi, k_rem, bins=256, inside=False):
    """one equi-width pass over candidates with values in [lo, hi]. Returns (cand mask of the pivot bin,
    k_rem after the pass, pivot bin [lo_b, hi_b], selected count above the bin). inside: every candidate
    lies in [lo, hi] by construction (a sub-range pass), so bins are clamped on both sides."""
    v = v.astype(np.float64)
    if hi <= lo:
        return cand & (v == hi), k_rem, (lo, hi), 0
    scale = bins / (hi - lo)
    b = np.floor((v - lo) * scale).astype(np.int64)
    b = np.minimum(b, bins - 1)
    if inside:
        b = np.maximum(b, 0)
    valid = cand & (b >= 0)
    counts = np.bincount(b[valid], minlength=bins)
    cum = np.cumsum(counts[::-1])[::-1]
    piv = np.where(cum >= k_rem)[0].max()
    sel = int(cum[piv + 1]) if piv + 1 < bins else 0
    k_rem -= sel
    return valid & (b == piv), k_rem, (lo + piv / scale, lo + (piv + 1) / scale), sel


def n6_row(v, k):
    """returns dict of statistics of one row for both brackets"""
    n = len(v)
    ipt = n // THREADS
    # striped arrangement: thread t holds items t + i * THREADS
    per_thread_max = v.reshape(ipt, THREADS).max(axis=0)
    hi = float(v.max())
    out = {}
    for name, lo in (("thrmax", float(per_thread_max.min())), ("full", float(v.min()))):
        if name == "thrmax" and k > THREADS:
            continue
        cand = np.ones(n, dtype=bool)
        c1, k1, (blo, bhi), _ = equiwidth(v, cand, lo, hi, k)
        n1 = int(c1.sum())
        d1 = len(np.unique(v[c1]))
        c2, k2, _, _ = equiwidth(v, c1, blo, bhi, k1, inside=True)
        n2 = int(c2.sum())
        d2 = len(np.unique(v[c2]))
        out[name] = dict(n1=n1, d1=d1, k1=k1, n2=n2, d2=d2, k2=k2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(20261025)
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("# Host simulation: radix pass counts and N6 pivot bin sizes (%d rows per cell)" % args.rows)
    emit()
    emit("## 1. Radix select pass count distribution (early exit when the pivot bucket count equals k remaining)")
    emit()
    emit("| key | pattern | N | k | schedule | passes 1 / 2 / 3 / 4 (fraction of rows) | mean |")
    emit("|---|---|---|---|---|---|---|")
    scheds = {"fp32": [("8x4", [8, 8, 8, 8]), ("11+11+10", [11, 11, 10]), ("12+12+8", [12, 12, 8])],
              "bf16": [("8+8", [8, 8]), ("12+4", [12, 4]), ("16", [16])]}
    for key in ("fp32", "bf16"):
        for pat in ("random", "tie_heavy", "relu_quantized"):
            for n in (1024, 32768):
                for k in (16, 64, 256, 2048):
                    if k > n // 2:
                        continue
                    for sname, digits in scheds[key]:
                        hist = np.zeros(5, dtype=np.int64)
                        for r in range(args.rows):
                            v = make_pattern(pat, n, k, rng, key)
                            u, w = twiddle(v, key)
                            p = radix_passes(u, w, k, digits)
                            hist[p] += 1
                        frac = hist[1:] / args.rows
                        mean = float((np.arange(5) * hist).sum() / args.rows)
                        emit("| %s | %s | %d | %d | %s | %s | %.2f |" % (
                            key, pat, n, k, sname, " / ".join("%.2f" % f for f in frac), mean))
    emit()
    emit("## 2. N6 equi-width bins: pivot bin after pass 1 and after pass 2 (256 bins per pass)")
    emit()
    emit("Columns: n1 = items in the pivot bin after pass 1 (mean, p90, max over rows), d1 = distinct values in it (mean), "
         "exit1 = fraction of rows done after pass 1 (n1 == k remaining or d1 == 1), fin32 = fraction with n1 <= 32, "
         "n2 and exit2 the same after a second equi-width pass over the pivot bin's sub-range. Bracket thrmax is "
         "[min of per-thread maxima, max] (valid for k <= 256), full is [min, max].")
    emit()
    emit("| key | pattern | N | k | bracket | n1 mean / p90 / max | d1 mean | exit1 | fin32 | n2 mean / max | d2 mean | exit2 |")
    emit("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in ("fp32", "bf16"):
        for pat in ("random", "tie_heavy", "one_outlier", "relu_quantized"):
            for n in (1024, 4096, 32768):
                for k in (4, 16, 64, 256, 1024, 4096):
                    if k > n // 2 or (n == 4096 and k not in (16, 256, 1024)):
                        continue
                    acc = {}
                    for r in range(args.rows):
                        v = make_pattern(pat, n, k, rng, key)
                        res = n6_row(v, k)
                        for br, st in res.items():
                            acc.setdefault(br, []).append(st)
                    for br, rows in acc.items():
                        n1 = np.array([s["n1"] for s in rows])
                        d1 = np.array([s["d1"] for s in rows])
                        k1 = np.array([s["k1"] for s in rows])
                        n2 = np.array([s["n2"] for s in rows])
                        d2 = np.array([s["d2"] for s in rows])
                        k2 = np.array([s["k2"] for s in rows])
                        exit1 = np.mean((n1 == k1) | (d1 <= 1))
                        exit2 = np.mean((n2 == k2) | (d2 <= 1) | (n1 == k1) | (d1 <= 1))
                        emit("| %s | %s | %d | %d | %s | %.1f / %d / %d | %.1f | %.2f | %.2f | %.1f / %d | %.1f | %.2f |" % (
                            key, pat, n, k, br, n1.mean(), np.percentile(n1, 90), n1.max(), d1.mean(), exit1,
                            np.mean(n1 <= 32), n2.mean(), n2.max(), d2.mean(), exit2))
    if args.out:
        open(args.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
