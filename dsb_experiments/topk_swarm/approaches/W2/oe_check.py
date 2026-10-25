import itertools
def oe_merge(lo, hi, r):
    step = r * 2
    if step < hi - lo:
        yield from oe_merge(lo, hi, step)
        yield from oe_merge(lo + r, hi, step)
        for i in range(lo + r, hi - r, step):
            yield (i, i + r)
    else:
        yield (lo, lo + r)
def oe_sort(lo, hi):
    if hi - lo >= 1:
        mid = lo + (hi - lo) // 2
        yield from oe_sort(lo, mid)
        yield from oe_sort(mid + 1, hi)
        yield from oe_merge(lo, hi, 1)
def bitonic_count(n):
    import math
    lg = int(math.log2(n)); return n // 2 * lg * (lg + 1) // 2
for n in (2, 4, 8, 16, 32, 64):
    net = list(oe_sort(0, n - 1))
    ok = True
    if n <= 16:
        for bits in itertools.product((0, 1), repeat=n):
            a = list(bits)
            for i, j in net:
                if a[i] < a[j]: a[i], a[j] = a[j], a[i]   # descending: max at lower index
            if any(a[t] < a[t + 1] for t in range(n - 1)): ok = False; break
    print("n", n, "odd-even comparators", len(net), "bitonic", bitonic_count(n), "0-1 check", ok if n <= 16 else "skipped")
