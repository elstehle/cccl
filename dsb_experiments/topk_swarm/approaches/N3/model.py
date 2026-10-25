#!/usr/bin/env python3
"""Sequential host model of block_topk_stream.cuh: the same images (fixed-point value image,
integer image of the twiddled key, index image), the same strict threshold filter with the
validated lower edge of the pivot bin, the same refine exits (exact count, rank finish at most
Threads, tie class by buffer order or by index image), the same overflow compaction. Runs the
harness battery patterns (reimplemented, not bit-identical generators) through both requirements
and checks against an exact oracle. Also prints the buffer and candidate statistics of the
random pattern (entries, pivot set size, refine levels)."""
import argparse
import math
import struct
import sys

import numpy as np

F32 = np.float32
INF = F32(np.inf)
FLT_MAX = F32(3.4028234663852886e38)


def f32(x):
    return F32(x)


def bits(f):
    return struct.unpack("<I", struct.pack("<f", float(f)))[0]


def from_bits(b):
    return F32(struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0])


def twiddle(f):
    f = F32(f) + F32(0.0)
    b = bits(f)
    m = 0xFFFFFFFF if (b >> 31) else 0x80000000
    return (b ^ m) & 0xFFFFFFFF


def untwiddle(t):
    m = 0xFFFFFFFF if (t & 0x80000000) == 0 else 0x80000000
    return from_bits(t ^ m)


def fmaf(a, b, c):
    # single rounding of a*b+c: exact in float64 for float32 a, b (48-bit product) then one rounding
    return F32(np.float64(a) * np.float64(b) + np.float64(c))


def round_bf16(x):
    b = bits(F32(x))
    if (b & 0x7F800000) == 0x7F800000 and (b & 0x007FFFFF):
        return from_bits(b & 0xFFFF0000)
    r = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
    return from_bits(r << 16)


class Params:
    def __init__(self):
        self.kind = 0
        self.lo_c = F32(0)
        self.lo_min = F32(0)
        self.scale = F32(1)
        self.cv = F32(0)
        self.lo_t = 0
        self.sh = 0
        self.p = 0

    def copy(self):
        q = Params()
        q.__dict__.update(self.__dict__)
        return q


class Model:
    def __init__(self, threads, chunk, max_index, cap, nbins, smaller_index):
        self.T = threads
        self.C = chunk
        self.cpt = chunk // threads
        self.N = max_index
        self.cap = cap
        self.NB = nbins
        self.lg_nb = int(math.log2(nbins))
        self.smaller = smaller_index
        self.levels = 0
        self.compactions = 0

    # images --------------------------------------------------------------------------------
    def image_f(self, v, lo_c, scale):
        d = F32(F32(v) - lo_c)
        f = fmaf(d, scale, F32(8388608.0))
        u = bits(f)
        if u & 0x80000000:
            u = u - (1 << 32)
        u = max(u, 0x4B000000)
        u = min(u, 0x4B000000 + self.NB - 1)
        return u - 0x4B000000

    def make_scale(self, width):
        with np.errstate(all="ignore"):
            sc = F32(F32(self.NB - 1) / F32(width))
        if not (sc > 0) or sc == INF:
            sc = F32(1.5e-36) if width == INF else F32(1.0)
        return sc

    def image_of(self, key, idx, q):
        if q.kind == 0:
            return self.image_f(key, q.lo_c, q.scale)
        if q.kind == 1:
            t = twiddle(key)
            d = max(t, q.lo_t) - q.lo_t
            return min(d >> q.sh, self.NB - 1)
        return self.image_f(F32(-idx), q.lo_c, q.scale)

    def classify(self, key, idx, q):
        if q.kind == 2:
            if key > q.cv:
                return 2
            if key < q.cv:
                return 0
        elif key < q.lo_min:
            return 0
        im = self.image_of(key, idx, q)
        return 2 if im > q.p else (1 if im == q.p else 0)

    def lower_edge(self, p, q):
        with np.errstate(all="ignore"):
            t = F32(q.lo_c + F32(F32(F32(p) - F32(0.5)) / q.scale))
        if not (t > q.lo_c) or t == INF or t != t:
            return q.lo_min
        eps = F32(max(abs(t), abs(q.lo_c)) * F32(2.4e-7))
        for _ in range(6):
            if self.image_f(t, q.lo_c, q.scale) < p:
                return t
            t = F32(t - eps)
            eps = F32(eps * 4)
        return t if self.image_f(t, q.lo_c, q.scale) < p else q.lo_min

    # histogram -----------------------------------------------------------------------------
    def scan_hist(self, k, q, set_thr):
        above = self.xa
        for b in range(self.NB - 1, -1, -1):
            h = self.hist[b]
            if above < k <= above + h:
                self.p, self.A, self.P = b, above, h
                if set_thr:
                    self.thr_next = max(self.thr, self.lower_edge(b, q))
                return
            above += h

    def rebin(self, q):
        self.hist = [0] * self.NB
        self.xa = 0
        for key, idx in self.buf:
            if q.kind == 2:
                if key > q.cv:
                    self.xa += 1
                elif key == q.cv:
                    self.hist[self.image_of(key, idx, q)] += 1
            elif not (key < q.lo_min):
                self.hist[self.image_of(key, idx, q)] += 1

    def pivot_minmax(self, q):
        tmn, tmx = 0xFFFFFFFF, 0
        for key, idx in self.buf:
            if self.classify(key, idx, q) == 1:
                t = twiddle(key)
                tmn = min(tmn, t)
                tmx = max(tmx, t)
        return untwiddle(tmn), untwiddle(tmx)

    def resolve(self, k, q):
        while True:
            self.scan_hist(k, q, False)
            q.p = self.p
            A, P = self.A, self.P
            R = k - A
            if P == R:
                return (0, A, R, P)
            if P <= self.T:
                return (2, A, R, P)
            self.levels += 1
            mn, mx = self.pivot_minmax(q)
            if mn == mx:
                if not self.smaller:
                    return (1, A, R, P)
                q.kind = 2
                q.cv = mn
                q.lo_c = F32(-(self.N - 1))
                q.lo_min = q.lo_c
                q.scale = self.make_scale(F32(self.N - 1))
            else:
                q.kind = 1
                q.lo_min = mn
                q.lo_t = twiddle(mn)
                rng = twiddle(mx) - q.lo_t
                nb = rng.bit_length()
                q.sh = max(nb - self.lg_nb, 0)
            self.rebin(q)

    def emit(self, q, res, out):
        mode, A, R, P = res
        apos, ppos = 0, 0
        stage = []
        for key, idx in self.buf:
            c = self.classify(key, idx, q)
            if c == 2:
                out[apos] = idx
                apos += 1
            elif c == 1:
                if mode == 2:
                    stage.append((key, idx))
                elif ppos < R:
                    out[A + ppos] = idx
                ppos += 1
        assert apos == A, (apos, A)
        if mode == 2:
            assert len(stage) == P
            for t, (key, idx) in enumerate(stage):
                rank = 0
                for ko, io in stage:
                    rank += int((ko > key) or (ko == key and io < idx))
                if rank < R:
                    out[A + rank] = idx

    def compact(self, q, limit):
        kept = []
        pk = 0
        for key, idx in self.buf:
            c = self.classify(key, idx, q)
            if c == 2:
                kept.append((key, idx))
            elif c == 1 and pk < limit:
                kept.append((key, idx))
                pk += 1
        self.buf = kept
        self.compactions += 1

    def overflow_compact(self, k):
        q = self.vq.copy()
        res = self.resolve(k, q)
        self.compact(q, res[2] if res[0] == 1 else 1 << 40)
        self.rebin(self.vq)
        self.scan_hist(k, self.vq, True)

    # streaming interface -------------------------------------------------------------------
    def begin(self):
        self.hist = [0] * self.NB
        self.xa = 0
        self.buf = []
        self.thr = F32(-np.inf)
        self.thr_next = self.thr
        self.vq = Params()
        self.p = self.A = self.P = 0

    def insert_chunk(self, chunk_vals, base, k, first):
        T, cpt = self.T, self.cpt
        # thread t holds items i*T + t of the chunk
        v = [[F32(chunk_vals[i * T + t]) for i in range(cpt)] for t in range(T)]
        if first:
            mn = min(twiddle(x) for row in v for x in row)
            mx = max(twiddle(x) for row in v for x in row)
            q = self.vq
            q.lo_min = untwiddle(mn)
            hi = untwiddle(mx)
            q.lo_c = max(q.lo_min, -FLT_MAX)
            hi_c = min(hi, FLT_MAX)
            with np.errstate(all="ignore"):
                q.scale = self.make_scale(F32(hi_c - q.lo_c))
            q.kind = 0
            entries = [None] * self.C
            for t in range(T):
                for i in range(cpt):
                    entries[i * T + t] = (v[t][i], base + i * T + t)
                    self.hist[self.image_f(v[t][i], q.lo_c, q.scale)] += 1
            self.buf = entries
            self.scan_hist(k, q, True)
            self.thr = self.thr_next
            return
        thr = self.thr
        counts = [sum(1 for x in v[t] if x > thr) for t in range(T)]
        tot_c = sum(counts)
        self.scan_hist(k, self.vq, True)  # the owner's threshold from the histogram so far
        if len(self.buf) + tot_c > self.cap:
            self.overflow_compact(k)
        for t in range(T):
            for i in range(cpt):
                if v[t][i] > thr:
                    self.buf.append((v[t][i], base + i * T + t))
                    self.hist[self.image_f(v[t][i], self.vq.lo_c, self.vq.scale)] += 1
        self.thr = self.thr_next

    def finish(self, k, out):
        q = self.vq.copy()
        res = self.resolve(k, q)
        self.emit(q, res, out)
        return res

    def run(self, vals, k):
        self.begin()
        n = len(vals)
        out = [-1] * k
        for j in range(n // self.C):
            self.insert_chunk(vals[j * self.C : (j + 1) * self.C], j * self.C, k, j == 0)
        res = self.finish(k, out)
        return out, res


# patterns ---------------------------------------------------------------------------------------
def quantize32(x):
    r = np.rint(x)
    fr = np.rint((x - r) * 32.0)
    return r + fr / 32.0


def make_pattern(p, n, k, seed):
    rng = np.random.default_rng(seed * 1000003 + 7)
    if p == "random":
        v = rng.standard_normal(n).astype(F32)
    elif p == "quantized_random":
        v = quantize32(rng.standard_normal(n)).astype(F32)
    elif p == "relu_quantized":
        v = quantize32(np.maximum(rng.standard_normal(n), 0.0)).astype(F32)
    elif p == "tie_heavy":
        v = (np.arange(n) % 64 / 64.0).astype(F32)
    elif p in ("sorted_asc", "sorted_desc"):
        v = np.sort(rng.standard_normal(n).astype(F32))
        if p == "sorted_desc":
            v = v[::-1].copy()
    elif p == "pivot_tie":
        gt = max(1, min(k // 4, n // 8))
        v = np.full(n, 1.0, F32)
        v[n - gt :] = 2.0
    elif p == "all_equal":
        v = np.full(n, 1.0, F32)
    elif p.startswith("signed_zero_s"):
        s = int(p[len("signed_zero_s") :])
        v = (-np.abs(rng.standard_normal(n)) - 1.0).astype(F32)
        pos = rng.permutation(n)
        zeros = min(2 * k, n)
        for j in range(zeros):
            neg = {0: (j & 1) != 0, 1: (j % 4) != 0, 2: (j % 4) == 0, 3: True, 4: False}[s % 5]
            v[pos[j]] = F32(-0.0) if neg else F32(0.0)
    elif p == "signed_zero_boundary":
        v = (-np.abs(rng.standard_normal(n)) - 1.0).astype(F32)
        pos = rng.permutation(n)
        positives = max(1, k // 2)
        zeros = min(k, n - positives)
        for j in range(positives):
            v[pos[j]] = abs(v[pos[j]])
        for j in range(zeros):
            v[pos[positives + j]] = F32(0.0) if j % 4 == 0 else F32(-0.0)
    elif p == "pivot_tie_zero":
        gt = max(1, min(k // 4, n // 8))
        v = np.full(n, -1.0, F32)
        v[n - gt :] = 1.0
        pos = rng.permutation(n - gt)
        zeros = min(2 * k, n - gt)
        for j in range(zeros):
            v[pos[j]] = F32(-0.0) if (j & 1) else F32(0.0)
    elif p == "inf_denorm":
        v = rng.standard_normal(n).astype(F32)
        pos = rng.permutation(n)
        specials = [np.inf, np.inf, -np.inf, -np.inf, 1e-39, -1e-39, 1.4e-45, 5.9e-39, 3.0e38, -3.0e38]
        for j, sp in enumerate(specials):
            v[pos[j]] = F32(sp)
    else:
        raise ValueError(p)
    return v


def oracle(vals, k, out, smaller):
    if len(set(out)) != k or min(out) < 0 or max(out) >= len(vals):
        return "indices invalid"
    order = sorted(range(len(vals)), key=lambda i: (-float(vals[i]) if vals[i] == vals[i] else -np.inf, i))
    # -0.0 == +0.0 in float compare, python sort by float treats them equal, fine
    top = order[:k]
    if smaller:
        if sorted(top) != sorted(out):
            return "exact set differs"
        return None
    a = sorted(float(vals[i]) for i in top)
    b = sorted(float(vals[i]) for i in out)
    if a != b:
        return "multiset differs"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--n", type=int, default=32768)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--cap", type=int, default=16384)
    ap.add_argument("--bins", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--patterns", default="all")
    a = ap.parse_args()
    pats = ["random", "quantized_random", "relu_quantized", "tie_heavy", "sorted_asc", "sorted_desc", "pivot_tie",
            "all_equal", "signed_zero_s0", "signed_zero_s1", "signed_zero_s2", "signed_zero_s3", "signed_zero_s4",
            "signed_zero_boundary", "pivot_tie_zero", "inf_denorm"]
    if a.patterns != "all":
        pats = a.patterns.split(",")
    fails = 0
    for kt in ("fp32", "bf16"):
        for smaller in (False, True):
            for p in pats:
                for seed in range(a.seeds):
                    vals = make_pattern(p, a.n, a.k, seed)
                    if kt == "bf16":
                        vals = np.array([round_bf16(x) for x in vals], dtype=F32)
                    m = Model(a.threads, 4096, a.n, a.cap, a.bins, smaller)
                    out, res = m.run(vals, a.k)
                    err = oracle(vals, a.k, out, smaller)
                    tag = "ok " if err is None else "FAIL"
                    if err:
                        fails += 1
                    print("%s %s %-22s smaller=%d seed=%d entries=%5d mode=%d A=%4d R=%4d P=%4d levels=%d compactions=%d %s"
                          % (tag, kt, p, smaller, seed, len(m.buf), res[0], res[1], res[2], res[3], m.levels,
                             m.compactions, err or ""))
    print("failures:", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
