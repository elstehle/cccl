#!/usr/bin/env python3
"""Host model of block_topk_compact.cuh: the same ck mapping, pass structure, capacity branch,
index-ordered compaction and tie ranks, checked against the oracle rules of harness/README.md
section 2 on the battery patterns. Also prints the first-pass pivot bucket size on random normal
input per (N, k) so the capacity choice can be judged before any GPU time.

  python3 -I model.py            check all patterns, both key types, both requirements
  python3 -I model.py --stats    pivot bucket statistics on random input for the B3 cells
"""
import sys
import numpy as np

THREADS = 256
REST = 8
CAP_DIV = 4
FIRST = 8


def schedule(key_bits, first):
    """(shift, bits) per pass: the first digit has `first` bits, later digits 8, the last the rest."""
    out = []
    s = key_bits - first
    out.append((s, first))
    while s > 0:
        b = min(REST, s)
        s -= b
        out.append((s, b))
    return out


def to_bf16_bits(f32):
    b = f32.astype(np.float32).view(np.uint32)
    # round to nearest even as __float2bfloat16 does (inputs carry no NaN)
    lsb = (b >> 16) & 1
    b = b + 0x7FFF + lsb
    return (b >> 16).astype(np.uint32)


def bf16_bits_to_f32(bits16):
    return (bits16.astype(np.uint32) << 16).view(np.float32)


def to_ck(bits, key_bits):
    bits = bits.astype(np.uint32)
    if key_bits == 32:
        f = bits.view(np.float32) + np.float32(0.0)  # -0.0 -> +0.0
        b = f.view(np.uint32)
        m = ((b >> 31) - np.uint32(1)).astype(np.uint32)
        return b ^ (m & np.uint32(0x7FFFFFFF))
    b = bits.copy()
    b[b == 0x8000] = 0
    m = ((b >> 15) - np.uint32(1)).astype(np.uint32)
    return b ^ (m & np.uint32(0x7FFF))


def run_pass(ck, valid, shift, bits, k_rem, prefix, mask):
    sel = valid & ((ck & mask) == prefix)
    digits = (ck[sel] >> shift) & (2**bits - 1)
    hist = np.bincount(digits, minlength=2**bits)
    excl = np.concatenate([[0], np.cumsum(hist)[:-1]])
    incl = excl + hist
    cross = np.nonzero((excl < k_rem) & (incl >= k_rem))[0]
    assert len(cross) == 1, (excl, incl, k_rem)
    b = int(cross[0])
    selected = int(excl[b])
    cands = int(hist[b])
    k_rem -= selected
    prefix |= b << shift
    mask |= (2**bits - 1) << shift
    return cands, k_rem, prefix, mask


def emit(ck, idx, valid, lo, hi_incl, need_ties, total_ties, out_base, k_set, out, tb):
    # ranks in set order (the array order is the index order for both register layouts)
    above = valid & (ck < lo)
    tie = valid & ~above & (ck <= hi_incl)
    sel_total = k_set - need_ties
    assert above.sum() == sel_total, (above.sum(), sel_total)
    assert tie.sum() == total_ties, (tie.sum(), total_ties)
    out[out_base:out_base + sel_total] = idx[above]
    ranks = np.arange(total_ties)
    if tb == "larger":
        ranks = total_ties - 1 - ranks
    take = ranks < need_ties
    tie_idx = idx[tie]
    out[out_base + sel_total + ranks[take]] = tie_idx[take]


def topk_compact(bits, key_bits, k, tb, first=None):
    first = FIRST if first is None else first
    n = len(bits)
    cap = n // CAP_DIV
    cpt = (cap + THREADS - 1) // THREADS
    sched = schedule(key_bits, first)
    key_mask = 2**key_bits - 1
    ck = to_ck(bits, key_bits)
    idx = np.arange(n, dtype=np.int64)
    valid = np.ones(n, dtype=bool)
    out = np.full(k, -1, dtype=np.int64)
    k_rem, prefix, mask = k, 0, 0
    c1, k_rem, prefix, mask = run_pass(ck, valid, sched[0][0], sched[0][1], k_rem, prefix, mask)
    s1 = k - k_rem
    path = "exit1"

    def tie_hi(prefix, mask):
        return prefix | (~mask & key_mask)

    def finish(ck_s, idx_s, valid_s, k_rem, prefix, mask, out_base, k_set):
        cands = 0
        for shift, bits in sched[1:]:
            cands, k_rem, prefix, mask = run_pass(ck_s, valid_s, shift, bits, k_rem, prefix, mask)
            if cands == k_rem:
                break
        emit(ck_s, idx_s, valid_s, prefix, tie_hi(prefix, mask), k_rem, cands, out_base, k_set, out, tb)

    if c1 == k_rem:
        emit(ck, idx, valid, prefix, tie_hi(prefix, mask), k_rem, c1, 0, k, out, tb)
    elif c1 <= cap:
        path = "compact"
        lo, hi = prefix, tie_hi(prefix, mask)
        above = ck < lo
        eq = ~above & (ck <= hi)
        assert above.sum() == s1 and eq.sum() == c1
        out[:s1] = idx[above]  # index order (blocked scan)
        cck = ck[eq]  # compacted in index order
        cidx = idx[eq]
        # register layout: thread t holds positions [t*cpt, (t+1)*cpt), padded to cap
        pad = cap - c1
        cck = np.concatenate([cck, np.zeros(pad, dtype=np.uint32)])
        cidx = np.concatenate([cidx, np.zeros(pad, dtype=np.int64)])
        cvalid = np.arange(cap) < c1
        finish(cck, cidx, cvalid, k_rem, prefix, mask, s1, k - s1)
    else:
        path = "fallback"
        finish(ck, idx, valid, k_rem, prefix, mask, 0, k)
    return out, path, c1


def oracle(keys_f32, k, out, req):
    n = len(keys_f32)
    ok = len(set(out.tolist())) == k and out.min() >= 0 and out.max() < n
    if not ok:
        return False, "indices not distinct or out of range"
    kz = keys_f32.copy()
    kz[kz == 0.0] = 0.0  # -0.0 == +0.0
    order = np.lexsort((np.arange(n), -kz))  # key desc, index asc (-0.0 and +0.0 equal)
    ref = order[:k]
    if req == "prefer_smaller_index":
        if sorted(ref.tolist()) != sorted(out.tolist()):
            return False, "index set differs from the exact reference"
        return True, ""
    got = np.sort(kz[out])
    want = np.sort(kz[ref])
    if not np.array_equal(got, want):
        return False, "multiset of keys differs"
    return True, ""


def patterns(n, k, rng):
    yield "random", rng.standard_normal(n).astype(np.float32)
    yield "sorted_asc", np.sort(rng.standard_normal(n).astype(np.float32))
    yield "sorted_desc", -np.sort(rng.standard_normal(n).astype(np.float32))
    yield "tie_heavy", ((np.arange(n) % 64) / 64.0).astype(np.float32)
    pv = np.ones(n, dtype=np.float32)
    m = max(1, min(k // 4, n // 8))
    pv[n - m:] = 2.0
    yield "pivot_tie", pv
    yield "all_equal", np.full(n, 0.37, dtype=np.float32)
    sz = -np.abs(rng.standard_normal(n)).astype(np.float32) - np.float32(0.5)
    pos = rng.choice(n, size=min(2 * k, n), replace=False)
    sz[pos] = np.where(np.arange(len(pos)) % 2 == 0, np.float32(0.0), np.float32(-0.0))
    yield "signed_zero", sz
    idn = rng.standard_normal(n).astype(np.float32)
    special = np.array([np.inf, -np.inf, np.float32(1e-40), np.float32(-1e-40), np.finfo(np.float32).max,
                        -np.finfo(np.float32).max, np.float32(1e-38)], dtype=np.float32)
    pos = rng.choice(n, size=min(len(special) * 3, n), replace=False)
    idn[pos] = np.tile(special, 3)[:len(pos)]
    yield "inf_denorm", idn
    q = np.round(rng.standard_normal(n) * 4) / 4
    yield "quantized_random", q.astype(np.float32)
    yield "relu_quantized", np.maximum(q, 0).astype(np.float32)
    yield "all_zero_signed", np.where(np.arange(n) % 2 == 0, np.float32(0.0), np.float32(-0.0))


def check():
    rng = np.random.default_rng(7)
    cells = [(1024, 4), (1024, 16), (1024, 512), (2048, 1024), (4096, 32), (4096, 256), (8192, 64), (8192, 1024),
             (16384, 1024), (32768, 256), (32768, 2048), (32768, 16384)]
    fails = 0
    runs = 0
    paths = {}
    for (n, k), first in [(c, f) for c in cells for f in (8, 9, 10)]:
        for name, f32 in patterns(n, k, rng):
            for key_bits, kt in ((32, "fp32"), (16, "bf16")):
                if key_bits == 16:
                    bits = to_bf16_bits(f32)
                    keys = bf16_bits_to_f32(bits)
                else:
                    bits = f32.view(np.uint32)
                    keys = f32
                for req, tb in (("not_guaranteed", "any"), ("prefer_smaller_index", "smaller")):
                    out, path, c1 = topk_compact(bits, key_bits, k, tb, first)
                    ok, why = oracle(keys, k, out, req)
                    runs += 1
                    paths[(first, path)] = paths.get((first, path), 0) + 1
                    if not ok:
                        fails += 1
                        print("FAIL first=%d n=%d k=%d %s %s %s path=%s c1=%d: %s" % (first, n, k, name, kt, req, path, c1, why))
    print("%d runs, %d failures, paths %s" % (runs, fails, paths))
    return fails


def stats():
    rng = np.random.default_rng(11)
    cells = [(1024, 16), (1024, 64), (2048, 16), (2048, 1024), (4096, 32), (4096, 256), (8192, 64), (8192, 1024),
             (16384, 64), (16384, 1024), (32768, 256), (32768, 2048)]
    print("first n k type  c1/N mean  c1/N max  cap/N  exit1  pass2 pivot mean  ties@k mean")
    for first in (8, 9, 10):
      for n, k in cells:
        for key_bits, kt in ((32, "fp32"), (16, "bf16")):
            sched = schedule(key_bits, first)
            c1s, p2s, ex, ties = [], [], 0, []
            for _ in range(16):
                f32 = rng.standard_normal(n).astype(np.float32)
                bits = to_bf16_bits(f32) if key_bits == 16 else f32.view(np.uint32)
                ck = to_ck(bits, key_bits)
                valid = np.ones(n, dtype=bool)
                c1, k_rem, prefix, mask = run_pass(ck, valid, sched[0][0], sched[0][1], k, 0, 0)
                c1s.append(c1 / n)
                if c1 == k_rem:
                    ex += 1
                    p2s.append(0)
                else:
                    c2, k2, _, _ = run_pass(ck, valid, sched[1][0], sched[1][1], k_rem, prefix, mask)
                    p2s.append(c2)
                s = np.sort(ck)
                ties.append(int((s == s[k - 1]).sum()))
            print("%d %5d %5d %s  %.4f  %.4f  %.3f  %2d/16  %7.1f  %5.1f" % (
                first, n, k, kt, np.mean(c1s), np.max(c1s), 1.0 / CAP_DIV, ex, np.mean(p2s), np.mean(ties)))


if __name__ == "__main__":
    if "--stats" in sys.argv:
        stats()
    else:
        sys.exit(1 if check() else 0)
