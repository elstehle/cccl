# R10: Block histogram built by per-warp ballots instead of shared atomics
status: rejected on the cost model for 8-bit digits (2026-10-25, algo-researcher)
scope: block
seed: own idea while sharpening B1
idea: each warp counts its share of the items into a register histogram by ballots (W3 style), writes its bins to shared memory once, and every warp sums the 8 partial histograms redundantly before the scan, replacing the MIO-bound atomic histogram.
reason: an 8-bit digit needs 8 ballots per item and 256 bucket counts, 8 buckets per lane each a popcount over an 8-way mask combination per item, about 64 instructions per lane per item at 4 items per lane, above 500 cycles per pass against the 340 cycle atomic histogram. With 4-bit digits the pass is about 440 cycles but there are twice as many passes, so bf16 costs about 1760 against 1500. The approach pays only where no shared memory exists, which is the warp scope, where it is W3.
what survives: nothing at block scope. The warp-redundant read of a small shared table (8 partial histograms of 16 bins) is the pattern of knowledge base 4.5 approach C and stays available for tie counts.
