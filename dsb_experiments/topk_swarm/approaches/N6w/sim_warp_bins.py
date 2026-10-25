#!/usr/bin/env python3
"""Host simulation of the N6 bracket pass at warp scope: 32 equal-width bins over a value bracket,
one pass, then the size of the pivot bin (the bin that holds the k-th largest) decides the finish.

Mirrors the device arithmetic of harness/adapters/N6w_impl.cuh: lane t holds items t*IPT .. t*IPT+IPT-1
(blocked), hi = warp max, lo = min over lanes of the lane's m-th largest (m = ceil(k / 32), exact
lower bound on the k-th largest because 32 m >= k items are at or above it), scale = 32 / (hi - lo),
c = fma(-lo, scale, 2^23), u = bits(fma(v, scale, c)) clamped into [bits(2^23), bits(2^23) + 31],
bin = low 5 bits of u. The map is monotone, so the items in bins above the pivot are the selected
prefix, the pivot bin holds the candidates and r = k - above of them are needed.

Decision after the pass (as in the adapter): exit when n1 == r (take the whole bin), tie class when
the candidates hold one distinct value (rank emission), W2 finish when n1 <= FIN_CAP (compaction and
r redux rounds), else a second equi-width pass over the candidates' exact value range.

Brackets compared: m (exact, m = ceil(k/32) capped at MCAP, else full), m1 (lane maxima only, exact
for k <= 32 and a guess above), full (tile min and max). Patterns: random normal (the harness
pattern, fp32 and bf16-rounded), uniform (fp32), relu_quantized, tie_heavy, one_outlier.
"""
import argparse
import sys

import numpy as np

LANES = 32
TWO23 = np.float32(8388608.0)
FIN_CAP = 64
MCAP = 8


def to_bf16_round(x32):
    """round float32 array to bf16 (round to nearest even) and widen back to float32"""
    b = x32.astype(np.float32).view(np.uint32).astype(np.uint64)
    lsb = (b >> 16) & 1
    b = (b + 0x7FFF + lsb) & 0xFFFF0000
    return b.astype(np.uint32).view(np.float32)


def make_pattern(name, n, k, rng):
    if name == "random":
        return rng.standard_normal(n).astype(np.float32)
    if name == "uniform":
        return rng.random(n, dtype=np.float32)
    if name == "relu_quantized":
        v = rng.standard_normal(n).astype(np.float32)
        v = np.maximum(v, 0)
        return np.round(v * 16) / 16
    if name == "tie_heavy":
        j = np.arange(n)
        return ((j % 64) / 64).astype(np.float32)
    if name == "one_outlier":
        v = rng.standard_normal(n).astype(np.float32)
        v[rng.integers(n)] = np.float32(1e30)
        return v
    raise ValueError(name)


def fma32(a, b, c):
    """fused multiply add in float32 emulated through float64 (product exact, one rounding at the end
    except for rare double rounding of the sum, irrelevant for bin statistics)"""
    return (a.astype(np.float64) * np.float64(b) + np.float64(c)).astype(np.float32)


def bins_of(v, lo, hi):
    """device bin function: returns bin per item (0..31) and whether the bracket was degenerate"""
    lo = np.float32(lo)
    hi = np.float32(hi)
    width = np.float32(hi - lo)
    if not (width > 0) or not np.isfinite(width):
        return np.zeros(len(v), dtype=np.int64), True
    scale = np.float32(np.float32(32.0) / width)
    c = np.float32(np.float64(-lo) * np.float64(scale) + np.float64(TWO23))
    f = fma32(v, scale, c)
    u = f.view(np.int32).astype(np.int64)
    u = np.clip(u, int(TWO23.view(np.int32)), int(TWO23.view(np.int32)) + 31)
    return u & 31, False


def bracket_lo(v, k, mode):
    ipt = len(v) // LANES
    lanes = v.reshape(LANES, ipt)  # blocked: lane t holds items t*IPT ..
    if mode == "full":
        return float(v.min()), 0
    if mode == "m1":
        return float(lanes.max(axis=1).min()), 1
    m = -(-k // LANES)
    if m > MCAP:
        return float(v.min()), 0
    part = np.sort(lanes, axis=1)[:, ::-1][:, m - 1]  # m-th largest per lane
    return float(part.min()), m


def one_pass(v, cand, lo, hi, r):
    """equi-width pass over candidates; returns pivot-candidate mask, r after, n1, above"""
    b, degenerate = bins_of(v, lo, hi)
    counts = np.bincount(b[cand], minlength=32)
    cum = np.cumsum(counts[::-1])[::-1]  # cum[j] = candidates with bin >= j
    piv = int(np.where(cum >= r)[0].max())
    above = int(cum[piv + 1]) if piv + 1 < 32 else 0
    newcand = cand & (b == piv)
    return newcand, r - above, int(counts[piv]), above, degenerate


def row_stats(v, k, mode):
    n = len(v)
    hi = float(v.max())
    lo, m = bracket_lo(v, k, mode)
    cand = np.ones(n, dtype=bool)
    c1, r1, n1, above1, degen = one_pass(v, cand, lo, hi, k)
    vals = v[c1]
    d1 = len(np.unique(vals))
    st = dict(n1=n1, r1=r1, d1=d1, m=m, degen=int(degen))
    st["exit1"] = int(n1 == r1)
    st["tie1"] = int(d1 == 1 and n1 != r1)
    st["fin1"] = int(n1 != r1 and d1 > 1 and n1 <= FIN_CAP)
    st["pass2"] = int(n1 != r1 and d1 > 1 and n1 > FIN_CAP)
    st["bracket_fail"] = int(above1 == 0 and n1 > n // 4)
    # second pass over the candidates' exact value range
    if st["pass2"]:
        c2, r2, n2, _, _ = one_pass(v, c1, float(vals.min()), float(vals.max()), r1)
        d2 = len(np.unique(v[c2]))
        st["n2"] = n2
        st["r2"] = r2
        st["exit2"] = int(n2 == r2)
        st["tie2"] = int(d2 == 1 and n2 != r2)
        st["fin2"] = int(n2 != r2 and d2 > 1 and n2 <= FIN_CAP)
        st["pass3"] = int(n2 != r2 and d2 > 1 and n2 > FIN_CAP)
    return st


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--out", default="-")
    ap.add_argument("--ns", default="256,512,1024,2048,4096")
    ap.add_argument("--ks", default="16,64,128,256,1024")
    ap.add_argument("--patterns", default="random,uniform,relu_quantized,tie_heavy,one_outlier")
    ap.add_argument("--brackets", default="m,m1,full")
    args = ap.parse_args()
    rng = np.random.default_rng(1234)
    out = sys.stdout if args.out == "-" else open(args.out, "w")
    w = out.write
    w("# N6 at warp scope: pivot bin after one 32-bin bracket pass (%d rows per cell, blocked lanes)\n\n" % args.rows)
    w("Columns: m = the per-lane order statistic used for lo (0 = tile min), n1 = pivot bin size after pass 1 "
      "(mean / p90 / max), r1 = items still needed from it (mean), d1 = distinct values in it (mean), "
      "exit1 = fraction of rows with n1 == r1 (no finish), tie1 = one distinct value (rank emission), "
      "fin1 = W2 finish (n1 <= %d), pass2 = second bracket pass needed, fail = bracket collapsed "
      "(pivot is the bottom bin with more than N/4 items), n2 = pivot bin after the second pass over the "
      "candidates' exact range (mean / max over the rows that needed it), done2 = fraction of those rows "
      "finished after pass 2 (exit, tie or W2 finish).\n\n" % FIN_CAP)
    w("| key | pattern | N | k | bracket | m | n1 mean / p90 / max | r1 | d1 | exit1 | tie1 | fin1 | pass2 | fail | n2 mean / max | done2 |\n")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for key in ("bf16", "fp32"):
        for pat in args.patterns.split(","):
            if key == "bf16" and pat == "uniform":
                continue
            for n in map(int, args.ns.split(",")):
                for k in map(int, args.ks.split(",")):
                    if k > n // 2:
                        continue
                    for mode in args.brackets.split(","):
                        rows = []
                        for s in range(args.rows):
                            v = make_pattern(pat, n, k, rng)
                            if key == "bf16":
                                v = to_bf16_round(v)
                            rows.append(row_stats(v, k, mode))
                        n1 = np.array([r["n1"] for r in rows])
                        r1 = np.array([r["r1"] for r in rows])
                        d1 = np.array([r["d1"] for r in rows])
                        p2 = [r for r in rows if r["pass2"]]
                        n2 = np.array([r["n2"] for r in p2]) if p2 else np.array([])
                        done2 = np.mean([r["exit2"] or r["tie2"] or r["fin2"] for r in p2]) if p2 else float("nan")
                        w("| %s | %s | %d | %d | %s | %d | %.1f / %d / %d | %.1f | %.1f | %.2f | %.2f | %.2f | %.2f | %.2f | %s | %s |\n" % (
                            key, pat, n, k, mode, rows[0]["m"], n1.mean(), pct(n1, 90), n1.max(), r1.mean(), d1.mean(),
                            np.mean([r["exit1"] for r in rows]), np.mean([r["tie1"] for r in rows]),
                            np.mean([r["fin1"] for r in rows]), np.mean([r["pass2"] for r in rows]),
                            np.mean([r["bracket_fail"] for r in rows]),
                            ("%.1f / %d" % (n2.mean(), n2.max())) if len(n2) else "-",
                            ("%.2f" % done2) if p2 else "-"))
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
