"""Host model of the N10 recovery and of the truncated final merge.
1. recovery: given words (slot-major striped over LW lanes), t = k-th largest, reproduce the
   ballot-rank assignment and compare with the exact (key desc, index asc) top-k.
2. truncation: a bitonic sequence of L words (elementwise max of a descending and an ascending
   sorted tile), half-cleaner stages with stride >= KTop, check positions [0, KTop) hold the KTop
   largest and that their min is the KTop-th largest.
"""
import random

def exact_topk(words, k):
    order = sorted(range(len(words)), key=lambda i: (-words[i], i))
    return sorted(order[:k])

def recover(words, LW, k):
    n = len(words); ipt = n // LW
    t = sorted(words, reverse=True)[k - 1]
    # w0[i] of lane l = words[i*LW + l]
    out = {}
    bgt = []; beq = []
    for i in range(ipt):
        g = 0; e = 0
        for l in range(LW):
            w = words[i * LW + l]
            if w > t: g |= 1 << l
            if w == t: e |= 1 << l
        bgt.append(g); beq.append(e)
    above = sum(bin(x).count('1') for x in bgt)
    assert above < k
    pa = 0; pe = above
    for i in range(ipt):
        for l in range(LW):
            lt = (1 << l) - 1
            sg = pa + bin(bgt[i] & lt).count('1')
            se = pe + bin(beq[i] & lt).count('1')
            w = words[i * LW + l]
            gt = w > t; eq = w == t
            take = gt or (eq and se < k)
            slot = sg if gt else se
            if take:
                assert slot not in out, (slot, out)
                out[slot] = i * LW + l
        pa += bin(bgt[i]).count('1'); pe += bin(beq[i]).count('1')
    assert sorted(out.keys()) == list(range(k)), sorted(out.keys())
    return sorted(out.values())

def half_cleaners(seq, min_stride):
    L = len(seq); s = L // 2
    seq = list(seq)
    while s >= min_stride:
        for p in range(L):
            if p & s == 0:
                a, b = seq[p], seq[p | s]
                seq[p], seq[p | s] = max(a, b), min(a, b)   # descending: lower position keeps max
        s //= 2
    return seq

random.seed(1)
# 1. recovery with heavy ties and all distinct
for trial in range(3000):
    LW = random.choice([4, 8, 16, 32]); ipt = random.choice([1, 2, 4, 8]); n = LW * ipt
    k = random.randint(1, n)
    vals = random.choice([range(0, 3), range(0, 10), range(0, 10**6)])
    words = [random.choice(vals) for _ in range(n)]
    assert recover(words, LW, k) == exact_topk(words, k), (words, LW, k)
# 2. truncation
for trial in range(3000):
    L = random.choice([4, 8, 16, 32, 64, 128])
    vals = random.choice([range(0, 3), range(0, 10**6)])
    a = sorted([random.choice(vals) for _ in range(L)], reverse=True)
    b = sorted([random.choice(vals) for _ in range(L)])
    bit = [max(x, y) for x, y in zip(a, b)]
    ktop = random.choice([2 ** j for j in range(0, L.bit_length()) if 2 ** j <= L])
    res = half_cleaners(bit, ktop)
    top = sorted(a + b, reverse=True)[:ktop]
    assert sorted(res[:ktop], reverse=True) == top, (L, ktop)
    assert min(res[:ktop]) == top[-1]
# also the single-tile case: bitonic = asc-sorted half then desc-sorted half (after sort of halves in opposite directions)
for trial in range(2000):
    L = random.choice([4, 8, 16, 32, 64])
    vals = random.choice([range(0, 3), range(0, 10**6)])
    x = [random.choice(vals) for _ in range(L)]
    # bitonic sort step: first half descending, second half ascending (DescTile true, Size L/2 blocks alternate)
    h = L // 2
    bit = sorted(x[:h], reverse=True) + sorted(x[h:])
    ktop = random.choice([2 ** j for j in range(0, L.bit_length()) if 2 ** j <= L])
    res = half_cleaners(bit, ktop)
    top = sorted(x, reverse=True)[:ktop]
    assert sorted(res[:ktop], reverse=True) == top, (L, ktop)
print("model ok")
