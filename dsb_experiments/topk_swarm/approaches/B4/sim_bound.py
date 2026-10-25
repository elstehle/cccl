#!/usr/bin/env python3
"""Host simulation for the B4 exact-bound prefilter (no GPU).

For every (key type, pattern, arrangement, N, k) cell it draws rows of the harness patterns
(random is normal(0, 1) in fp32 rounded to the key type, numpy's generator, statistically
equivalent to the harness's mt19937), computes the candidate count C of items at or above the
bound for several bounds, and the pivot-bin statistics of the first finishing pass over the
candidates. 256 threads throughout.

Bounds (lo is the smallest key that is still a candidate, items >= lo are candidates):
  m1    : min over threads of the per-thread maximum (valid for k <= 256)
  gm    : the thread's items split into m = ceil(k / 256) contiguous groups, lo = min over all
          256 m group maxima (valid a priori for every k <= 256 m, one max per item)
  gmh   : the same with m = max(1, ceil(k / 512)), valid only when C >= k (run-time check)
  topm  : exact per-thread m-th largest, lo = min over threads (valid for k <= 256 m)

First finishing pass over the candidates (the "offset digit"): d = tk - lo_tk on the twiddled
u32 keys, W = hi_tk - lo_tk, sh = max(0, bitlen(W) - 8), bin = d >> sh. Reported next to the
float equi-width bin of N6 (floor((v - lo) / (hi - lo) * 256)).

  python3 sim_bound.py [--rows 300] [--out sim_bound.md]
"""
import argparse
import math

import numpy as np

THREADS = 256


def to_bf16_bits(x32):
    b = np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    lsb = (b >> 16) & 1
    rounded = (b + 0x7FFF + lsb) >> 16
    return rounded.astype(np.uint16)


def bf16_to_f32(bits16):
    return (bits16.astype(np.uint32) << 16).view(np.float32)


def round_to_key(v32, key):
    v32 = v32.astype(np.float32) + np.float32(0.0)  # -0.0 onto +0.0, as the primitive does under prefer_smaller_index
    if key == "fp32":
        return v32
    return bf16_to_f32(to_bf16_bits(v32))


def twiddle(v32, key):
    """order-preserving unsigned key (ascending twiddle == ascending value)"""
    if key == "fp32":
        b = np.ascontiguousarray(v32, dtype=np.float32).view(np.uint32)
        mask = np.where(b >> 31 == 1, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
        return (b ^ mask).astype(np.uint32)
    b = to_bf16_bits(v32).astype(np.uint32)
    mask = np.where(b >> 15 == 1, np.uint32(0xFFFF), np.uint32(0x8000))
    return (b ^ mask).astype(np.uint32)


def quantize32(x):
    r = np.rint(x)
    fr = np.rint((x - r) * 32.0)
    return (r + fr / 32.0).astype(np.float32)


def make_rows(name, rows, n, k, rng, key):
    """rows x n matrix of fp32 values in the input (memory) order"""
    if name in ("random", "sorted_asc", "sorted_desc"):
        v = rng.standard_normal((rows, n)).astype(np.float32)
        if name == "sorted_asc":
            v = np.sort(v, axis=1)
        elif name == "sorted_desc":
            v = -np.sort(-v, axis=1)
    elif name == "quantized_random":
        v = quantize32(rng.standard_normal((rows, n)).astype(np.float32))
    elif name == "relu_quantized":
        v = quantize32(np.maximum(rng.standard_normal((rows, n)).astype(np.float32), 0))
    elif name == "tie_heavy":
        v = np.tile(((np.arange(n) % 64) / 64.0).astype(np.float32), (rows, 1))
    elif name == "one_outlier":
        v = rng.standard_normal((rows, n)).astype(np.float32)
        pos = rng.integers(n, size=rows)
        v[np.arange(rows), pos] = np.float32(1e30) if key == "fp32" else np.float32(3.3e38)
    elif name == "pivot_tie":
        gt = max(1, min(k // 4, n // 8))
        v = np.ones((rows, n), dtype=np.float32)
        v[:, n - gt:] = 2.0
    else:
        raise ValueError(name)
    return round_to_key(v, key)


def per_thread_view(v, arrangement):
    """rows x n -> rows x THREADS x IPT (thread t, item i)"""
    rows, n = v.shape
    ipt = n // THREADS
    if arrangement == "striped":
        # item i of thread t is input item i * THREADS + t
        return v.reshape(rows, ipt, THREADS).transpose(0, 2, 1)
    return v.reshape(rows, THREADS, ipt)


def bound_values(pt, k, kind):
    """pt: rows x THREADS x IPT (fp32 values). Returns lo per row (fp32 value)."""
    rows, threads, ipt = pt.shape
    if kind == "m1":
        return pt.max(axis=2).min(axis=1)
    if kind in ("gm", "gmh"):
        m = max(1, math.ceil(k / (threads if kind == "gm" else 2 * threads)))
        m = min(m, ipt)
        g = ipt // m
        used = pt[:, :, : g * m].reshape(rows, threads, m, g)
        return used.max(axis=3).min(axis=(1, 2))
    if kind == "topm":
        m = min(max(1, math.ceil(k / threads)), ipt)
        part = -np.partition(-pt, m - 1, axis=2)[:, :, m - 1]
        return part.min(axis=1)
    raise ValueError(kind)


def pivot_stats(tk_row, v_row, lo_tk, hi_tk, lo_v, hi_v, cand, k):
    """first finishing pass over the candidates of one row. Returns dict with the offset digit
    and the float equi-width digit statistics."""
    out = {}
    c = tk_row[cand].astype(np.uint64)
    # offset digit
    w = int(hi_tk) - int(lo_tk)
    sh = max(0, w.bit_length() - 8)
    d = c - np.uint64(lo_tk)
    b = (d >> np.uint64(sh)).astype(np.int64)
    out["off"] = pivot_of(b, 256, k)
    # second offset digit over the pivot bin
    sh2 = max(0, sh - 8)
    pb = out["off"]["bin"]
    in_pivot = b == pb
    if sh > 0:
        b2 = ((d[in_pivot] >> np.uint64(sh2)) & np.uint64(0xFF)).astype(np.int64)
        st2 = pivot_of(b2, 256, out["off"]["k_rem"])
        out["off"]["n2"] = st2["n1"]
        out["off"]["k_rem2"] = st2["k_rem"]
        out["off"]["d2"] = st2["distinct_dummy"]
    else:
        out["off"]["n2"] = out["off"]["n1"]
        out["off"]["k_rem2"] = out["off"]["k_rem"]
    out["off"]["distinct"] = len(np.unique(c[in_pivot]))
    # float equi-width digit (N6)
    vv = v_row[cand].astype(np.float64)
    if hi_v > lo_v:
        fb = np.floor((vv - lo_v) * (256.0 / (hi_v - lo_v))).astype(np.int64)
        fb = np.clip(fb, 0, 255)
    else:
        fb = np.zeros(len(vv), dtype=np.int64)
    out["ew"] = pivot_of(fb, 256, k)
    return out


def pivot_of(bins, nbins, k):
    counts = np.bincount(bins, minlength=nbins)
    cum = np.cumsum(counts[::-1])[::-1]  # cum[b] = items with bin >= b
    pb = int(np.where(cum >= k)[0].max())
    sel = int(cum[pb + 1]) if pb + 1 < nbins else 0
    return {"bin": pb, "n1": int(counts[pb]), "k_rem": k - sel, "distinct_dummy": 0}


def pct(a, p):
    return float(np.percentile(a, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true", help="random pattern and N 8192, 32768 only")
    args = ap.parse_args()
    rng = np.random.default_rng(20261025)
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("# Host simulation: candidate counts of the exact thread bound and the first finishing pass (%d rows per cell, 256 threads)" % args.rows)
    emit()
    emit("Bounds: m1 = min over threads of the thread maximum (k <= 256), gm = min over 256 m group maxima with "
         "m = ceil(k / 256) (a priori valid), gmh = the same with m = ceil(k / 512) (valid only when C >= k, the "
         "column P(C<k) is its failure rate), topm = min over threads of the exact per-thread m-th largest. "
         "C = items at or above the bound. S = ceil(C / 256) candidate slots per thread. ovf = fraction of rows "
         "with C above the cap. Pivot statistics are for the m1 bound (k <= 256) or gm (k > 256) on the striped "
         "arrangement: n1 = items in the pivot bin after the first offset-digit pass over the candidates "
         "(mean / p90 / max), krem = remaining k inside the pivot bin (mean / max), done1 = fraction of rows "
         "with n1 == krem, le32 = fraction with n1 <= 32, dist = distinct keys in the pivot bin (mean), "
         "n2 = pivot bin after the second offset digit (mean / max), ew1 = the float equi-width pivot bin of N6 "
         "for comparison (mean / max).")
    emit()
    ns = [8192, 32768] if args.quick else [1024, 4096, 8192, 16384, 32768]
    patterns = ["random"] if args.quick else ["random", "sorted_asc", "quantized_random", "relu_quantized",
                                               "tie_heavy", "one_outlier", "pivot_tie"]
    ks = [16, 64, 256, 1024, 2048]

    emit("## 1. Candidate counts per bound")
    emit()
    emit("| key | pattern | arr | N | IPT | k | bound | m | C mean | p50 | p90 | p99 | max | frac mean | S p99 / max | ovf N/4 | ovf N/8 | ovf 2048 | ovf 4096 | P(C<k) |")
    emit("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    pivot_rows = []
    for key in ("fp32", "bf16"):
        for pat in patterns:
            for n in ns:
                ipt = n // THREADS
                rows = args.rows if n <= 16384 else max(100, args.rows // 2)
                if pat in ("tie_heavy", "pivot_tie"):
                    rows = 4
                for k in ks:
                    if k > n // 2:
                        continue
                    v = make_rows(pat, rows, n, k, rng, key)
                    tk = twiddle(v.reshape(-1), key).reshape(rows, n)
                    for arr in (("striped", "blocked") if pat in ("random", "sorted_asc") else ("striped",)):
                        pt = per_thread_view(v, arr)
                        kinds = ["gm", "gmh", "topm"] if k > THREADS else ["m1", "gmh"]
                        for kind in kinds:
                            lo = bound_values(pt, k, kind)
                            m = 1 if kind == "m1" else (max(1, math.ceil(k / (THREADS if kind != "gmh" else 2 * THREADS))))
                            cand = v >= lo[:, None]
                            C = cand.sum(axis=1)
                            S99 = math.ceil(pct(C, 99) / THREADS)
                            Smax = math.ceil(C.max() / THREADS)
                            emit("| %s | %s | %s | %d | %d | %d | %s | %d | %.0f | %.0f | %.0f | %.0f | %d | %.4f | %d / %d | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
                                key, pat, arr, n, ipt, k, kind, m, C.mean(), pct(C, 50), pct(C, 90), pct(C, 99), C.max(),
                                (C / n).mean(), S99, Smax, np.mean(C > n // 4), np.mean(C > n // 8), np.mean(C > 2048),
                                np.mean(C > 4096), np.mean(C < k)))
                            if arr == "striped" and ((kind == "m1" and k <= THREADS) or (kind == "gm" and k > THREADS)) \
                                    and pat not in ("one_outlier", "pivot_tie"):
                                st = []
                                hi = v.max(axis=1)
                                for r in range(rows):
                                    if C[r] < k:
                                        continue
                                    lo_tk = int(twiddle(np.array([lo[r]], dtype=np.float32), key)[0])
                                    hi_tk = int(twiddle(np.array([hi[r]], dtype=np.float32), key)[0])
                                    st.append(pivot_stats(tk[r], v[r], lo_tk, hi_tk, float(lo[r]), float(hi[r]), cand[r], k))
                                if st:
                                    n1 = np.array([s["off"]["n1"] for s in st])
                                    kr = np.array([s["off"]["k_rem"] for s in st])
                                    n2 = np.array([s["off"]["n2"] for s in st])
                                    di = np.array([s["off"]["distinct"] for s in st])
                                    e1 = np.array([s["ew"]["n1"] for s in st])
                                    pivot_rows.append("| %s | %s | %d | %d | %s | %.1f / %d / %d | %.1f / %d | %.2f | %.2f | %.1f | %.1f / %d | %.1f / %d |" % (
                                        key, pat, n, k, kind, n1.mean(), pct(n1, 90), n1.max(), kr.mean(), kr.max(),
                                        np.mean(n1 == kr), np.mean(n1 <= 32), di.mean(), n2.mean(), n2.max(), e1.mean(), e1.max()))
    emit()
    emit("## 2. First finishing pass over the candidates (striped, m1 for k <= 256, gm above)")
    emit()
    emit("| key | pattern | N | k | bound | n1 mean / p90 / max | krem mean / max | done1 | le32 | dist | n2 mean / max | ew1 mean / max |")
    emit("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in pivot_rows:
        emit(r)
    if args.out:
        open(args.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
