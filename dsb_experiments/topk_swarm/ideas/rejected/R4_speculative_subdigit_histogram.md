# R4: Speculative sub-digit histogram (two-level histogram in one pass)
status: rejected on the cost model (2026-10-25, algo-researcher)
scope: block
seed: own idea while sharpening B1 and B3
idea: during radix pass 1 also count a 4-bit sub-digit for every 8-bit bucket (4096 counters, 16 KB). After the scan finds the pivot bucket, its 16 sub-bins are already counted, so pass 2 is a 16-bin scan without a sweep over the items, and bf16 finishes 12 bits after one item pass.
reason: the histogram phase is MIO-issue bound and insensitive to contention (`reference/BLOCK_TOPK_AIR_OPT_RESULTS.md` section 6), so two atomics per item double pass 1's histogram cost (about 340 more at N 1024, about 11000 more at N 32768). What it saves is pass 2's histogram sweep, which at N 1024 is the same 340, a wash, and at N 32768 is a sweep that only tests the prefix (the atomics are predicated to the about N over 256 candidates), far cheaper than the extra 32768 atomics. B3's compaction makes pass 2 cheap without touching pass 1, and N6 removes pass 2 by discrimination rather than by speculation.
what survives: nothing. Recorded so the next study does not rebuild it.
