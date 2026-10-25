# N3: streaming block selection with a running exact threshold (chunked input). Optimizer report (opt-N3, 2026-10-25)

Numbers in this report are B200 (umbriel-b200-068 for the full batch, second node for the latency repeat as stated per table), nvcc 13.4.92, build `-arch=sm_100`, harness 0.7, random normal input (bf16 rounded), latency in cycles per call net of the generate-only control (chain slope), throughput in Gelem/s. Every number sits next to its control from the same lock section, cards `cards/N3_stream_b200_20261025_<n>.json`, raw batches under `state/results/opt-N3/`. Code is on branch `exp/topk-swarm/N3` (worktree `/Users/estehle/Downloads/cccl-topk-swarm-N3`): primitive `approaches/N3/block_topk_stream.cuh`, adapter `harness/adapters/N3_stream.cuh`, cells `harness/cells_n3.json`, host model `approaches/N3/model.py`, findings `approaches/N3/findings_stream.json`.

## 1. What it is

The row of N keys arrives in chunks of 4096 (striped over the block, 16 items per thread at 256 threads, 8 at 512). The primitive keeps three things between chunks: a candidate buffer of (key, index) entries, a histogram of the buffer under a fixed-point image of the key value (1024 equal-width bins over the first chunk's value bracket, T25), and a threshold. The first chunk is stored whole, its histogram is scanned from the top and the lower edge of the bin in which the k-th largest key lies becomes the threshold. Every later chunk pays one compare per item against the threshold, one packed block scan (the per-thread candidate count in the high 16 bits, the thread's histogram bins in the low 16 bits, T29), one predicated store per candidate and one branch-free shared atomic per item slot into the histogram (a private dummy word for the slots that are no candidates, T32 and B4_v4). The scan gives each thread its store offset and, from the histogram prefix, the new pivot bin and threshold, so the threshold rises after every chunk without any merge, re-selection or pass over the buffer. The threshold used for chunk j comes from the histogram of chunks up to j minus 2 (the scan of chunk j minus 1 reads the histogram before that chunk's candidates are stored), one chunk of lag that costs about 1700 extra buffer entries at N 32768 k 2048 and saves one block scan per chunk.

The finish scans the histogram once more, sweeps the buffer twice (a packed count scan for the positions, then the stores), writes the items above the pivot bin to the output directly and resolves the pivot bin: exact count (all of it), at most Threads items (every staged pivot item computes its rank among the staged items by key descending then index ascending, one broadcast shared load per pair, exact under both requirements, no rounds), else a refinement level that re-bins the buffer on the twiddled integer key over the pivot set's own min and max (every level separates distinct keys, so it never stalls on denormals or tiny widths). A tie class (pivot set min equals max) is taken in buffer order under `not_guaranteed` and refined on an index image (1024 bins over the index range, at most Threads items per bin) under `prefer_smaller_index`. When the next chunk's candidates would not fit the buffer (sorted ascending input, a first chunk whose bracket is useless because of infinities), the same refinement runs on the buffer, the buffer is compacted in place blockwise (all loads of a block of 4 x Threads entries before any store, kept entries only move down), the histogram is rebuilt and rescanned.

Why this and not the card's design (init selection, candidate buffer, k-sized radix merges): the card's model put the merges at 3550 cycles each and the init at 5500, about 14k in total against a bar of 16k. The histogram-carried threshold replaces both: the "merge" is one block scan per chunk, and the only pass over the buffer is the finish. Why chunks of 4096 and not 2048: with k at most half a chunk the first chunk's histogram already yields a threshold at or above that chunk's median, with a chunk equal to k it would yield that chunk's minimum and the second chunk would be stored whole, and the scan count doubles.

## 2. Contract

Block scope, 256 or 512 threads, striped input in registers fed as N / 4096 chunks in index order (the adapter copies each chunk out of the register tile, the primitive touches a chunk only inside its `insert_chunk` call), N a multiple of 4096 up to 32768 (16-bit index and count fields), k in [1, 2048] at run time (at most half a chunk), keys fp32 or bf16 (compared as fp32 values, bf16 widened exactly), int32 indices from the striped position. Output: k source indices in the caller's shared array, unspecified order (above items in buffer order, then the pivot items in rank or buffer order). Requirements `not_guaranteed` and `prefer_smaller_index` in one code path (the tie-class exit differs). Temp storage: histogram 4 KB, buffer of 16384 entries (64 KB bf16, 128 KB fp32), stage of Threads entries, scan storage, partials, dummy and control words: about 71 KB (bf16) and 137 KB (fp32) at 256 threads. Synchronization: `__syncthreads` inside (see the card's contract), the caller separates consecutive calls by a barrier. NaN unsupported (never a candidate after the first chunk). -0.0 equals +0.0 everywhere (value compares, the image subtracts the bracket base, the integer image folds the sign of zero, the rank compares values). The host model `model.py` runs the same arithmetic on the battery patterns (320 cases over both key types, both requirements, 256 and 512 threads, the three measured shapes) and found the one logic bug of this build before any GPU time: the threshold's edge validation nudged the edge by ulps of the edge, which vanish in the image's `v - lo` subtraction, so the threshold stalled for whole chunks (fixed with a relative margin).

## 3. Where the time goes

Phase profile with clock64 stamps by thread 0 behind `N3_PROFILE` (`profile_phases.cu`, one block, random normal input, median of 32 calls, umbriel-b200-068, `profiles/profile_random_b200_068.txt`). N 32768 k 2048, cycles.

| phase | fp32 256 threads | fp32 512 threads | bf16 256 | bf16 512 |
|---|---|---|---|---|
| bracket of the first chunk (thread min and max, redux, barrier) | 475 | 929 | 461 | 787 |
| first chunk stores and histogram atomics (16 or 8 slots) | 694 | 716 | 670 | 591 |
| first histogram scan and threshold | 1092 | 1191 | 1035 | 1110 |
| per later chunk: count sweep | 185 to 314 | 215 | 157 to 293 | 253 |
| per later chunk: packed block scan phase (histogram loads, scan, owner's edge) | 895 to 1071 | 881 to 1092 | 871 to 1108 | 813 to 862 |
| per later chunk: store sweep (predicated store, image, one atomic per slot) | 1631 to 1721 | 1100 to 1308 | 1524 to 1729 | 1087 to 1227 |
| seven later chunks together | about 19.6k | about 16.1k | about 18.4k | about 15.1k |
| finish: histogram scan | 690 | 916 | 691 | 905 |
| finish: count sweep over the buffer (rolled, 44 or 22 iterations) | 8079 | 4601 | 8034 | 4694 |
| finish: emit sweep over the buffer (rolled) | 11878 | 7026 | 11909 | 6987 |
| finish: rank of the pivot set (20 to 30 items) | 2294 | 1632 | 2439 | 1814 |
| total (harness number of the same cell) | 45097 (43195) | 33926 (31693) | 44131 (41202) | 32567 (29224) |

Three facts. (1) The stream costs about 2.8k cycles per chunk of 4096 at 256 threads and 2.3k at 512, that is 0.7 and 0.56 cycles per item, as much per item as B3_tuned's whole resident selection (0.56 to 0.63 cycles per item). Sixty percent of it is the store sweep: about 100 cycles per register slot at 256 threads and 140 at 512 for a predicated store, a five-instruction image and one data-dependent shared atomic, against about 15 cycles per slot for B4's compaction sweep without the atomic and the image (T64). Thirty percent is the scan phase at 900 to 1070 cycles, 2.5x a plain cub BlockScan (the four histogram loads and the owner's division are the suspects). (2) The finish is half the call: two rolled sweeps over the 11.1k to 11.3k buffered entries at 184 to 319 cycles per iteration, pure dependent chains with no overlap (T64). (3) 512 threads shrink the sweeps and the finish loops but not the scan phase nor the per-slot atomic cost, which is why the gain over 256 threads is 1.36x to 1.41x and not more.

Resources: 255 registers at fp32 256 threads with 820 to 892 bytes of honest spill (STL 80 to 85, LDL 117 to 124, no hidden local array), 128 at bf16 256 threads with 0 to 56 bytes, 128 at 512 threads (fp32 100 to 776 bytes). Static SASS 24120 to 25848 instructions at N 32768 and 237 BAR.SYNC: the overflow path (resolve, compact, rebin, rescan) is inlined into each of the seven chunk calls and never runs on random input, about 30 barriers execute. Shared memory 88 KB (bf16) and 155 KB (fp32) with the harness buffers, one block per SM in both regimes, like the controls at N 32768.

## 4. Results

Card `cards/N3_stream_b200_20261025_1.json` (umbriel-b200-068, three batches in one outer lock section, variant and control in the same process in each). Latency in cycles per call, bf16 then fp32, random input.

| cell | N3_stream | B3_tuned (256) | N8_b3tuned | B4_v4 | ratio to B3 family | ratio to B4_v4 |
|---|---|---|---|---|---|---|
| N 32768 k 2048 not_guaranteed, 256 threads | 41202 and 43195 | 18387 and 20775 | 18391 and 20771 | 25473 and 25143 | 0.45x and 0.48x | 0.62x and 0.58x |
| N 32768 k 2048 prefer_smaller_index, 256 | 46993 and 46402 | 18382 and 20775 | 18371 and 20773 | 35103 and 28864 | 0.39x and 0.45x | 0.75x and 0.62x |
| N 32768 k 2048 not_guaranteed, 512 threads | 29224 and 31693 | | 16271 and 18051 | 18372 and 19404 | 0.56x and 0.57x | 0.63x and 0.61x |
| N 32768 k 2048 prefer_smaller_index, 512 | 32400 and 35111 | | 16247 and 18061 | 21769 and 20340 | 0.50x and 0.51x | 0.67x and 0.58x |
| N 32768 k 1024 not_guaranteed, 256 | 34163 and 36419 | | 19207 and 20454 | 11838 and 10591 | 0.56x and 0.56x | 0.35x and 0.29x |
| N 32768 k 1024 not_guaranteed, 512 | 24820 and 27426 | | 17084 and 17822 | 12407 and 11865 | 0.69x and 0.65x | 0.50x and 0.43x |
| N 16384 k 2048 not_guaranteed, 256 | 27429 and 26831 | | 10958 and 10413 | 18182 and 19508 | 0.40x and 0.39x | 0.66x and 0.73x |
| N 16384 k 2048 not_guaranteed, 512 | 19663 and 19209 | | 9874 and 10130 | 15101 and 15761 | 0.50x and 0.53x | 0.77x and 0.82x |

Throughput at N 32768 k 2048 (Gelem per s, bf16 and fp32): N3 360 and 214 at 256 threads against B3_tuned 400 and 321 and B4_v4 274 and 294; 298 and 280 at 512 threads against N8_b3tuned 431 and 346 and B4_v4 452 and 437. The only ratios above one in the card (bf16 N 32768 k 2048 1.31x and N 16384 k 2048 1.50x over B4_v4 in throughput at 256 threads) are against B4's weak cells and still 0.60x to 0.90x of B3_tuned.

Verdict against the bar (at least 1.3x under B3_tuned at N 32768 k 2048, at most 16k cycles): failed by 2.1x to 2.7x (43.2k and 41.2k at 256 threads, 31.7k and 29.2k at 512 against a bar of 16k and 14.8k). N3 is retired as a cell candidate per its card's kill criterion. The second-node latency repeat of T58 was not run: the kernels are 256 and 512 threads wide (T58's rule covers 128-thread kernels, 256-thread siblings are within 11 percent across nodes), the result is 2x off the bar, and the card's own instruction on failure is to stop.

### 4.1 Contract note: what "insert chunk, keep k" costs relative to resident selection

Measured on resident data (the adapter feeds the register tile chunk by chunk, so the number is the streaming primitive's cost without the stream's loads), N 32768 k 2048, B200:

- Per chunk of 4096 items: 2.8k cycles at 256 threads, 2.3k at 512 (0.7 and 0.56 cycles per item): one compare per item, one packed block scan (offsets plus the histogram prefix that yields the next threshold), one predicated store per candidate and one shared atomic per item slot. The atomic is 60 percent of it, the scan 30 percent. Candidates stored: k (1 + ln(N over C)) on random order, 34 percent of N here, with the threshold within one bin width of the exact k-th largest seen so far.
- The finish over the buffer of 0.34 N entries: 23k as built (rolled sweeps, T64), about 5k at 256 threads and 3k at 512 with unrolled sweeps (projection from B4's measured sweep rates, not measured).
- In total 2.1x to 2.3x of B3_tuned's resident selection as built, projected 0.8x to 1.1x of B3_tuned at 256 threads and 0.6x to 0.75x of N8_b3tuned at 512 after the two fixes, never below B4_v4's k 1024 cells (10.5k to 12.4k). The resident primitive stays the better choice whenever the row is resident; the streaming primitive is for rows that are not (a DSA indexer's 128K scores per query, a vocabulary in 32 chunks), where its per-chunk cost of about 2.3k to 2.8k cycles per 4096 items (and about 1.4k projected) is the price of never holding the row, against about 7k cycles for one radix histogram pass over 32768 resident items (B1).
- Memory: the buffer bounds the state by the candidate volume, not by N: 16384 entries (64 KB bf16, 128 KB fp32) hold k 2048 at N 32768 with a 45 percent margin on random order, and the in-place compaction handles sorted ascending and infinity-bracket inputs (two compactions per call in the host model, correct and slow). Sorted ascending chunk order floods the buffer every three chunks because the first chunk's bracket never covers the later values; re-bracketing after a compaction would fix the flood, not the per-chunk cost.
- Exactness is free of charge in structure: a strict `v > T` filter keeps a valid top-k by induction under both requirements, and prefer_smaller_index costs 6 to 14 percent here only through code size and register allocation, not through an extra pass.

## 5. Ideas tried, with verdicts

One build, one batch (plus the oracle-only pass and the phase profile). The design decisions inside the build and what the batch says about them:

| decision | verdict |
|---|---|
| histogram-carried threshold instead of init selection plus k-sized merges (the card's design) | works as modeled on the candidate side (34 percent of N buffered, threshold within one bin), but the per-chunk bookkeeping (2.8k) costs more than the merges it replaces would have (the card's 3550 per merge, two merges) |
| chunks of 4096 (k at most half a chunk) | right choice given the design: the first chunk's threshold is its median, 2048 would have stored the second chunk whole and doubled the scans |
| 1024 bins (4 per thread at 256, 2 at 512) | pivot sets of 20 to 30 on random and 106 to 108 on quantized input, always inside the rank finish, no refine level ran on the measured patterns |
| one-chunk lag (one scan per chunk) | about 1700 extra entries at N 32768, cheaper than seven more scans at 900 each |
| branch-free dummy-word atomics in the store sweep (T32, B4_v4) | no help: the atomic unit, not the branch, is the cost (about 100 cycles per slot) |
| rolled finish sweeps to spare registers | the single worst decision: half the call (T64) |
| all-pairs rank finish for pivot sets up to Threads | 1.6k to 2.4k for 20 to 30 items, fine in share, a warp redux finish would be no better |
| 512 threads (T45) | 1.36x to 1.41x over 256, the best shape, still 1.75x to 1.8x behind N8_b3tuned |

## 6. Ideas not yet tried

1. Remove the one-chunk lag with a second histogram scan after chunk 1 only (where the lag costs 1024 entries) or by ticket placement (one aggregated atomic per warp, no block scan for the offsets, index order lost but not needed since the finish ranks by explicit index).
2. Store the bin with the entry (10 bits beside a 16-bit index) so the finish sweeps read one word and no image arithmetic; the fp32 entry then needs 8 bytes anyway, the bf16 entry grows to 8.
3. Re-bracket the stream image after a compaction to the kept entries' range (sorted ascending input then stops flooding the buffer every three chunks).
4. 2048 bins (8 per thread at 256 threads) to halve the pivot set and the rank finish on quantized inputs (106 items per pivot bin on quantized_random).
5. A real streaming kernel: chunks loaded from global memory per `insert_chunk`, which removes the 128-register tile from the kernel and the 255-register pressure of the harness adapter.

## 7. Determinism and ordering support

Set and index deterministic by construction: the items above the pivot bin are an exact set, the pivot bin is resolved by explicit (key, index) rank, by exact count or, under `not_guaranteed`, by buffer order inside a tie class (deterministic for a fixed thread mapping). `prefer_smaller_index` is the same code with the index refine instead of the buffer-order exit. `prefer_larger_index` would flip the rank's index compare and the index image. Output order unspecified, sorted output a post-sort of k.

## 8. Open items

- The two fixes the profile asks for (unrolled finish sweeps with hoisted loads, candidate-only histogram) are not built: the projection says they do not reach the bar, and the card says stop on failure. A real streaming kernel without the register tile is the setting in which they would be worth measuring.
- The T58 second-node latency repeat was skipped (256 and 512-thread kernels, verdict 2x off the bar); the card names its node.
- The harness has no streaming input pattern: a "chunked" arrangement that loads each chunk from global memory inside the adapter's call would measure the streaming contract on non-resident data (asked of the evaluation engineer in the final report).
- The scan phase at 2.5x a plain BlockScan and the 100 cycles per atomic slot are the two numbers that would decide a second design (per-warp histograms, a custom warp-shuffle scan).

## 9. Pointers

- Cards: `cards/N3_stream_b200_20261025_1.json` (full batch, controls N8_b3tuned at 256 and 512 threads, B4_v4 at 256 and 512, B3_tuned at 256), `cards/N3_stream_b200_20261025_2.json` (latency repeat on a second node).
- Raw batches: `state/results/opt-N3/N3_stream_vs_{N8_b3tuned,B4_v4,B3_tuned}_n3_{a,b,c}_*.json` and the oracle-only pass `N3_stream_vs_N8_b3tuned_n3_a_oracle_*.json`.
- Code: `approaches/N3/block_topk_stream.cuh`, `harness/adapters/N3_stream.cuh`, `harness/cells_n3.json`, `approaches/N3/model.py` (`--threads`, `--n`, `--k`, `--patterns`).
- Shipping to a node: `tar czf - -C <worktree>/dsb_experiments/topk_swarm harness approaches/N3 approaches/B1 approaches/B3 approaches/B4 | ssh <node> '<exec_prefix> tar xzf - -C <root>/work/opt-N3'`, then `run.py --cells cells_n3.json --build-only` and `--measure-only` with `--lock <private file>` inside one outer `flock <root>/locks/gpu0.lock` for the three controls.
