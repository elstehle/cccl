#!/usr/bin/env python3
"""Host simulation for idea N4: tie class of the k-th truncated fp32 key on the harness's random
pattern (normal(0, 1) in fp32), and how often the W2 tie pass takes its slow path.

For each (N, k, key bits) the fp32 keys are twiddled to order-preserving u32, truncated to the top
`bits` bits and packed with the inverted index. The k rounds extract the k largest packed words.
The class is the set of items whose truncated key equals that of the k-th word. The slow path runs
when a class member has a larger index than the k-th word's item (it was not emitted).

  python3 n4_tie_sim.py [trials]
"""
import sys

import numpy as np


def twiddle(f32):
    u = f32.view(np.uint32)
    mask = ((u.view(np.int32) >> 31).astype(np.uint32)) | np.uint32(0x80000000)
    t = u ^ mask
    return np.where(t == np.uint32(0x7FFFFFFF), np.uint32(0x80000000), t)


def simulate(n, k, bits, trials, rng):
    idx_bits = 32 - bits
    idx_mask = np.uint32((1 << idx_bits) - 1)
    classes, slow, wrong_without_pass = [], 0, 0
    for _ in range(trials):
        v = rng.standard_normal(n).astype(np.float32)
        t = twiddle(v)
        inv = idx_mask ^ np.arange(n, dtype=np.uint32)
        w = (t & ~idx_mask) | inv
        order = np.argsort(-w.astype(np.int64), kind="stable")
        kth = order[k - 1]
        tkey = w[kth] & ~idx_mask
        members = np.nonzero((w & ~idx_mask) == tkey)[0]
        classes.append(len(members))
        if (members > kth).any():
            slow += 1
            # would the packed result be wrong without the pass? compare the exact top-k sets
            exact = set(np.lexsort((np.arange(n), -t.astype(np.int64)))[:k].tolist())
            got = set(order[:k].tolist())
            if exact != got:
                wrong_without_pass += 1
    c = np.array(classes)
    return c.mean(), c.max(), slow / trials, wrong_without_pass / trials


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    rng = np.random.default_rng(12345)
    print("N      k    bits  class_mean  class_max  slow_path_rate  wrong_without_pass")
    for n in (256, 512, 1024, 2048, 4096, 16384, 32768):
        for k in (4, 16, 64):
            if 2 * k > n:
                continue
            bits_list = sorted({32 - int(np.log2(n)), 20, 16})
            for bits in bits_list:
                if bits > 32 - int(np.log2(n)):
                    continue  # the index would not fit
                cm, cx, sr, wr = simulate(n, k, bits, trials, rng)
                print("%-6d %-4d %-5d %-11.3f %-10d %-15.4f %-10.4f" % (n, k, bits, cm, cx, sr, wr))


if __name__ == "__main__":
    main()
