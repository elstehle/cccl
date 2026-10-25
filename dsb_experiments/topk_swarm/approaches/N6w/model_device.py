#!/usr/bin/env python3
"""Host model of the N6w device algorithm (N6w_impl.cuh), step for step in float32, used to locate
correctness bugs on the oracle battery patterns without a GPU. Not a performance model.

  python3 model_device.py --n 2048 --k 1024 --pattern inf_denorm --key fp32 --rows 20
"""
import argparse

import numpy as np

FULL = 0xFFFFFFFF
IMG0 = 0x4B000000
TWO23 = np.float32(8388608.0)
FLT_MAX = np.float32(3.4028234663852886e38)
FLT_MIN = np.float32(1.1754943508222875e-38)
ALL_TIES = 0x7FFFFFFF


def f32(x):
    return np.float32(x)


def to_bf16_round(x32):
    b = x32.astype(np.float32).view(np.uint32).astype(np.uint64)
    lsb = (b >> 16) & 1
    b = (b + 0x7FFF + lsb) & 0xFFFF0000
    return b.astype(np.uint32).view(np.float32)


def tw_of(v, bf16):
    u = v.astype(np.float32).view(np.uint32).astype(np.int64)
    s = np.where(u >> 31, 0xFFFFFFFF, 0)
    t = (u ^ (s | 0x80000000)) & FULL
    if bf16:
        t &= 0xFFFF0000
    return t


def untw(t):
    t = int(t)
    s = 0xFFFFFFFF if ((~t) & 0x80000000) else 0
    u = (t ^ (s | 0x80000000)) & FULL
    return np.array([u], dtype=np.uint32).view(np.float32)[0]


def fma32(a, b, c):
    return (a.astype(np.float64) * np.float64(b) + np.float64(c)).astype(np.float32)


def bin_image(v, lo, scale):
    d = (v.astype(np.float32) - lo).astype(np.float32)
    x = fma32(d, scale, TWO23).view(np.int32).astype(np.int64)
    y = np.maximum(x, IMG0)
    return np.minimum(y, IMG0 + 31)


class State:
    def __init__(self, tmin, tmax, r, n):
        self.tmin, self.tmax, self.r, self.n = int(tmin), int(tmax), int(r), int(n)


def locate(counts, r):
    cum = np.cumsum(counts[::-1])[::-1]
    p = int(np.where(cum >= r)[0].max())
    cnt_p = int(counts[p])
    above = int(cum[p]) - cnt_p
    return p, cnt_p, above


def bracket_pass(v, tw, st, all_items, log):
    lo = max(untw(st.tmin), -FLT_MAX)
    hi = min(untw(st.tmax), FLT_MAX)
    lo, hi = f32(lo), f32(hi)
    width = f32(hi - lo)
    with np.errstate(all="ignore"):
        scale = f32(f32(32.0) / width) if width != 0 else f32(np.inf)
    if not (scale > 0 and scale < FLT_MAX):
        scale = FLT_MIN
    x = bin_image(v, lo, scale)
    if all_items:
        cand = np.ones(len(v), dtype=bool)
    else:
        cand = (tw >= st.tmin) & (tw <= st.tmax)
    bins = np.where(cand, x & 31, 0)
    counts = np.bincount(bins[cand], minlength=32)
    p, np_, above = locate(counts, st.r)
    inpiv = cand & (bins == p)
    st.tmin = int(tw[inpiv].min())
    st.tmax = int(tw[inpiv].max())
    st.r -= above
    st.n = np_
    log.append("bracket lo=%g hi=%g scale=%g p=%d n=%d r=%d above=%d" % (lo, hi, scale, p, st.n, st.r, above))


def clz32(x):
    x &= FULL
    return 32 - x.bit_length()


def digit_pass(tw, st, D, KB, log):
    nb = clz32(st.tmin ^ st.tmax)
    d = min(D, KB - nb)
    shift = 32 - nb - d
    dmask = (1 << d) - 1
    cand = (tw >= st.tmin) & (tw <= st.tmax)
    dig = np.where(cand, (tw >> shift) & dmask, 0)
    counts = np.bincount(dig[cand], minlength=1 << d)
    p, np_, above = locate(counts, st.r)
    keep = (~((2 << (shift + d - 1)) - 1)) & FULL
    base = (st.tmin & keep) | (p << shift)
    low = (1 << shift) - 1
    st.tmin = max(st.tmin, base)
    st.tmax = min(st.tmax, base | low)
    st.r -= above
    st.n = np_
    log.append("digit nb=%d d=%d p=%d n=%d r=%d above=%d" % (nb, d, p, st.n, st.r, above))


def emit_threshold(tw, thr, take):
    if take == ALL_TIES:
        return np.where(tw >= thr)[0]
    tie = np.where(tw == thr)[0]  # index order
    sel = list(np.where(tw > thr)[0]) + list(tie[:take])
    return np.array(sorted(sel))


def run_model(v, k, bf16, D=4, maxlev=4, log=None):
    n = len(v)
    ipt = n // 32
    lanes = v.reshape(32, ipt)
    tw = tw_of(v, bf16)
    m_raw = 1 << max(0, int(np.ceil(np.log2(-(-k // 32))))) if k > 32 else 1
    M = m_raw if (m_raw <= 8 and ipt % m_raw == 0) else 0
    if M == 0:
        mth = lanes.min(axis=1)
    else:
        mth = np.sort(lanes, axis=1)[:, ::-1][:, M - 1]
    mx = lanes.max(axis=1)
    st = State(tw_of(mth, bf16).min(), tw_of(mx, bf16).max(), k, n)
    fipt = 2 if ipt <= 16 else 4
    fin_cap = 32 * fipt
    idx_bits = int(np.ceil(np.log2(n)))
    bracket_pass(v, tw, st, True, log)
    level = 1
    while True:
        if st.n == st.r or st.tmin == st.tmax:
            break
        fits = bf16 or ((st.tmax - st.tmin) < ((1 << (32 - idx_bits)) - 1))
        if st.n <= fin_cap and fits:
            above = k - st.r
            up = np.where(tw > st.tmax)[0]
            assert len(up) == above, ("above mismatch", len(up), above)
            inr = np.where((tw >= st.tmin) & (tw <= st.tmax))[0]
            assert len(inr) == st.n, ("n mismatch", len(inr), st.n)
            order = sorted(inr, key=lambda i: (-int(tw[i]), i))
            log.append("w2 finish n=%d r=%d above=%d" % (st.n, st.r, above))
            return np.array(sorted(list(up) + order[: st.r]))
        if level == maxlev:
            break
        bracket_pass(v, tw, st, False, log)
        level += 1
    KB = 16 if bf16 else 32
    while st.n != st.r and clz32(st.tmin ^ st.tmax) < KB:
        digit_pass(tw, st, D, KB, log)
    if st.n == st.r:
        return emit_threshold(tw, max(st.tmin, 1), ALL_TIES)
    return emit_threshold(tw, st.tmin, st.r)


def reference(v, k, bf16):
    tw = tw_of(v, bf16)
    order = sorted(range(len(v)), key=lambda i: (-int(tw[i]), i))
    return np.array(sorted(order[:k]))


def make_input(pattern, n, rng):
    v = rng.standard_normal(n).astype(np.float32)
    if pattern == "inf_denorm":
        specials = [np.inf, np.inf, -np.inf, -np.inf, 1e-39, -1e-39, 1.4e-45, 5.877e-39, 3.0e38, -3.0e38]
        pos = rng.permutation(n)[: len(specials)]
        for p, s in zip(pos, specials):
            v[p] = np.float32(s)
    elif pattern == "signed_zero":
        pass
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--pattern", default="inf_denorm")
    ap.add_argument("--key", default="fp32")
    ap.add_argument("--rows", type=int, default=20)
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    bf16 = args.key == "bf16"
    fails = 0
    for row in range(args.rows):
        v = make_input(args.pattern, args.n, rng)
        if bf16:
            v = to_bf16_round(v)
        log = []
        got = run_model(v, args.k, bf16, log=log)
        want = reference(v, args.k, bf16)
        ok = len(got) == len(want) and np.array_equal(got, want)
        if not ok:
            fails += 1
            print("row %d FAIL: got %d items, want %d" % (row, len(got), len(want)))
            for line in log:
                print("   ", line)
            diff_missing = sorted(set(want) - set(got))[:5]
            diff_extra = sorted(set(got) - set(want))[:5]
            print("    missing", diff_missing, [float(v[i]) for i in diff_missing])
            print("    extra", diff_extra, [float(v[i]) for i in diff_extra])
            if fails >= 3:
                break
    print("rows %d fails %d" % (args.rows, fails))


if __name__ == "__main__":
    main()
