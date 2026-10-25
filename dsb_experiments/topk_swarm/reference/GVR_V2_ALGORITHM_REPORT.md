# Self-sampling GVR ("GVR V2"): the algorithm and its kernel specialisations

Scope: the hint-free self-sampling GVR top-k that TRT-LLM ships as its production
indexer top-k (decode: PRs #18410 / #18446, prefill: PR #18702), as vendored in
`kernels/cute_dsl_topk_internal/blackwell/top_k/gvr_topk_decode_self_sampling.py`
(device, 6828 lines) and `gvr_topk_decode_self_sampling_host.py` (host dispatch),
TRT-LLM commit `688bcfbb`. Section 1 places it next to radix select (what
`cub::DeviceBatchedTopK` does), section 2 describes the algorithm independent of
how work is spread over CTAs, section 3 covers the kernel families and the
dispatch between them, section 4 ties both to the measured heatmaps
(`plots/gvr_v2/`, B200, 2026-09-09). Line numbers refer to the vendored files.

Notation: a row (segment) has `n` valid fp32 values, `k` are wanted; `bs` rows per
call. "Rank r" means the r-th largest value.

---

## 1. Two ways to select exactly k out of n

Every exact GPU top-k has to find the value `t` at rank k (the pivot) and then
collect everything above it plus the right number of ties. The two arms compared
here differ in how they find `t`.

**Radix select (CUB).** `t` is determined digit by digit. Pass 1 histograms all
`n` keys on their most significant digit (CUB's `bits_per_pass = 11`, so 2048
buckets, `tuning_batched_topk.cuh:287`), a suffix scan over the buckets locates
the bucket that contains rank k, and only keys with that digit prefix stay
candidates. Pass 2 repeats on the next digit over the candidates, pass 3 pins the
last bits; for 32-bit keys that is `calc_num_passes = 3` (`agent_batched_topk_cluster.cuh:279`).
The pass count is fixed and data-independent, every pass is exact, and the cost
is three histogram passes over a shrinking set: the first over the whole row, the
next ones over the candidates if the row is resident on chip, or over the row
again if it is not (the resident capacity of CUB's cluster kernel on B200 is about
3.6 MB per cluster, see `CUB_BATCHED_TOPK_REPORT.md` section 9).

**Threshold filter with verification (GVR).** Guess a threshold `T` a little
below the pivot, keep only the values `>= T` in one streaming pass (the
"candidates", expected to number a few times k), and finish the selection on
that small set on chip. The guess only steers the cost; correctness comes from
counting. If fewer than k values survive, the guess was too high and a cheaper
retry follows; if far too many survive, the candidate buffer overflows and the
kernel falls back to slower but still exact paths. In the temporal V1 the guess
came from the previous decode step's top-k (`pre_idx`). V2 removes that
dependency: the guess comes from the current row itself, hence "self-sampling".
A hint-free kernel needs no prior state, works on the first decode step and in
disaggregated serving, and cannot be misled by a stale hint (V1 lost 3.6 to 6x on
an invalid hint per the PR #18410 description).

The rest of this document is about how V2 makes the guess, why the result is
exact regardless of the guess, and how the work is laid out on the machine.

---

## 2. The algorithm

### 2.1 Order-preserving integer keys

All comparisons that must be bit-exact run on `fkey(x) = u ^ ((u >> 31) ? 0xFFFFFFFF : 0x80000000)`
with `u` the raw fp32 bits (`fkey_bits`, line 124): a monotone map from floats
(finite and infinite) to `uint32`, so a radix or range argument over keys is an
argument over values. Ties are exact-value ties. A candidate is carried as one
64-bit word `idx << 32 | bits(x)`; when the kernel needs a total order for
tie-breaking it uses `fkey(x) << 32 | idx`, i.e. value first, index second.

### 2.2 Guessing the threshold from a sample (streaming families)

Host and kernel agree on a "ladder" of scalars derived from `(n, k, batch,
R)` alone (`route()` in the host file, mirrored per row by warp 0 of every CTA in
the varlen prologue, device lines 1740 to 1870):

- `aim`: the target number of candidates. `aim = 4k` for k >= 1024 and `2k`
  below when the launch is one wave (`b·R <= 148`), `2k` for a split row raised
  to `3k` (R = 2) or `3.5k` (R > 2) by the split floor, `11k/8` or `3k/2` for
  multi-wave launches; never below `sqrt(6n)`, never above
  half the candidate buffer, never below k. The pivot estimate is therefore
  placed at rank `aim`, two to four times deeper than k. That margin is what
  makes "fewer than k survivors" rare.
- the sample: `SMP` threads each load two adjacent float4 (8 values) at stride
  `SS2` float4-pairs, a deterministic strided sample of `8·SMP` values spread
  over the whole row (P1, line 1881). Its size is `sel ≈ SFAC · n / aim` clamped
  to `[256, n/2]`, with `SFAC` between 16 and 64 depending on k and R (64 or 32 for an
  unsplit row), so the sampled fraction is about `SFAC/aim` (3% of a 64k row for
  k = 512).
- `TGT = aim · 8·SMP / n` and `TGT2 = k · 8·SMP / n`: the ranks, inside the
  sample, that correspond to rank `aim` and rank `k` in the row. Because
  `8·SMP ≈ sel = SFAC·n/aim`, `TGT ≈ SFAC`: the threshold is always roughly the
  32nd or 64th largest sampled value, whatever `n` is. A rank-32 order statistic
  has a relative rank error of order `1/sqrt(32)`, which is what the 2x to 4x
  margin between `aim` and `k` absorbs.

P2 (line 1902) reduces the sample to its min and max, builds a 256-bin uniform
histogram of the sample over `[SMIN, SMAX]` in shared memory, and one warp does a
suffix scan (`scan_cross0`, line 765) that pins three bins at once:

- the bin where the cumulative count from the top crosses `TGT` gives
  `T = SMIN + B·w` (`w` the bin width), the estimate of the value at rank `aim`;
- the bin crossing `TGT2` estimates the value at rank `k`; `HIC`, a few bin
  widths above it, is used only to tighten the upper end of the classification
  window;
- the bin crossing `2·TGT` gives `TSH`, a lower threshold (rank about `2·aim`)
  kept as the first retry rung, or used up front where no retry is possible
  (section 3.4).

If the sample is degenerate (all equal, or the scan never reaches `TGT`), `T`
falls to the sentinel `-3e38`: everything becomes a candidate and the exact
fallbacks of 2.6 take over. Nothing about the sample is ever trusted for
correctness.

Worked example, `n = 65536, k = 512, bs <= 148, R = 1`: `aim = 1024`,
`sel = 32·65536/1024 = 2048`, `SMP = 256` threads sample 2048 values (3.1%),
`TGT = 32`, `TGT2 = 16`. `T` is the lower edge of the sample-histogram bin holding the 32nd largest of
the 2048 samples, an estimate of the 1024th largest of the row; about 1024 candidates are expected from a row of
65536, and k = 512 of them are wanted.

### 2.3 Guessing the bracket without a sample (register-resident families)

When the whole row fits in the registers of the CTA (n <= 16k, section 3.1) no
sample is needed: the bracket is `[min, max]` of the first k values of the row,
already in registers (`hint_free and not deg` arm, device line 3790). This lower
end is not an estimate but a bound: at least k values (the first k) are `>=` it,
so the top-k lies inside the bracket by construction, and the only risk is that
too many values do, which the escape path (2.6) handles. In the degenerate-k
regime (`n <= 3k` or `n <= 4k + 64`, the "DEG" arm) the bracket is the min/max of
the whole row, i.e. a plain histogram of everything.

### 2.4 One streaming pass: classify, stage, bin

P3 (line 2174) streams the row once in float4 batches (`U` float4 per thread per
step, software-prefetched). Each thread builds a bitmask of its values `>= T`;
the warp counts survivors with a ballot/popcount and reserves their slots in the
shared-memory candidate buffer with one atomic per warp. Each survivor is then
re-read as a scalar (the batch registers are not kept alive, to stay under the
register wall), stored as `idx << 32 | bits(x)` at its slot, and counted into a
second 256-bin histogram over the window `[T, hi)`, `hi = min(max(GMAX, T), HIC)`
(`_emitc`, line 1399). Values above the window clamp into the top bin. Survivors
beyond the buffer capacity `SCAP` (16384 or 8192 or 4096 depending on the
family) land in a trash slot; the count keeps running so the overflow is known.

The pass reads the row exactly once from global memory plus one scalar re-read
per survivor, which hits L1/L2. Elements below `T` cost a compare each and are
never touched again.

### 2.5 Verify and resolve the boundary

After the pass, one warp scans the candidate histogram for rank k (line 2426,
`scan_cross0` with target k). It yields the crossing bin `B`, `above` = number of
candidates strictly above bin B, `m` = number inside bin B, and the total.

- **total < k**: the threshold was too high; the top-k is not guaranteed to be
  inside the candidates. Retry the whole pass with a lower rung: first `TSH`,
  then the sentinel floor that admits every value (`NATT = 3` attempts,
  line 2110). Each retry is another full read of the row. Hint-free mode has no
  hint bracket to fall back to, so the last rung is always "everything".
- **total >= k**: accept. The top-k is a subset of the candidates, and the
  crossing bin is exact because the counts are exact. Elements in bins above B
  are all in the top-k; `need = k - above` of the `m` elements in bin B are.

Emission (P5, line 2504) writes every candidate whose bin lies above B directly
to the output; its position is the histogram's suffix count for its bin plus an
atomic cursor within the bin, so the output is unsorted but dense. Candidates in
bin B are moved to a second buffer `ck64` of capacity `CMP` (4096 or 2048 or 1024)
as `fkey << 32 | idx`. If the candidate buffer had overflowed, the emission
instead re-sweeps the row and re-classifies against the same `T` and bins
("collect overflow", line 2562): still exact, one more read.

The crossing bin is then refined (P6, line 2599):

- `need >= m` (in practice `need == m`): the whole bin is taken, nothing to do.
- `m <= 288` (`QUADC_CLUS`): each of the `m` candidates counts how many others
  have a strictly greater 64-bit key; those with fewer than `need` above them
  are written to `out[above + rank]`. Quadratic in `m`, one shared-memory pass,
  and deterministic: equal values are ordered by index.
- otherwise: **key-space narrowing**, a radix select confined to the `m`
  crossing-bin keys. Each level histograms the keys into 256 bins over the current
  key range, scans for the still-needed count, and narrows the range to the
  crossing bin, at most six levels. It ends with a key `ethr` such that all keys
  `> ethr` are selected and exactly `needC` keys `== ethr` are: a warp-ballot
  emit (`_ballot_pair_emit`, line 1439) writes the strict ones and admits the
  first `needC` members of the tie class.

### 2.6 Exact fallbacks

Two situations bypass the normal emission and use narrowing over a larger set
(same procedure as above, up to eight levels):

- **degen A** (line 2758): the crossing bin holds more than `CMP` candidates
  (a heavy tie or a collapsed window). Narrow over all staged candidates instead
  of the crossing bin only.
- **degen B** (line 2911): no valid threshold was found even after the ladder, or
  the buffer overflowed and the row is not otherwise recoverable. Narrow over the
  whole row: a full on-chip radix select on `fkey`, reading the row once per
  level. This is the path that guarantees exactness for any input.

Rows with `n <= k` never enter the body; the epilogue writes identity indices
and pads with -1 (line 3015). `+inf` and `-inf` are ordinary keys; a documented
limitation is that rows containing NaN can select a different set than the CUDA
reference because both order NaNs implementation-specifically.

### 2.7 What is guaranteed and what is only expected

Guaranteed: the output is exactly the set of k largest values (any subset of a
tie class at rank k), because acceptance requires `count(x >= T) >= k`, the
histogram counts over candidates are exact, elements above the crossing bin are
provably in the top-k, and the crossing bin is resolved on exact integer keys.
Which members of a boundary tie class appear is arrival order (a ballot), except
in the quadratic path where it is index order; both are legal under the
harness's set semantics and under CUB's `tie_break::unspecified`.

Expected, i.e. performance only: with a representative sample the row is read
once, about `aim` candidates (2k to 4k) are staged, the crossing bin is small and
resolves in one quadratic or a few narrowing levels. The tail is data dependent:
a bad sample costs one or two extra passes, a heavy tie class at rank k costs
the narrowing over all candidates, and only pathological rows reach whole-row
narrowing. In the real-data sweep no arm hit a failure and the wins were
uniform across the 25 cells, which says the tail is rare on indexer logits.

Cost against radix select in the resident case: one row read plus the sample
against three histogram passes of which the first reads the row; in the
streaming case one row read against three. The on-chip work per row is
`O(aim)` plus the crossing-bin work, against `O(candidates)` per digit pass.

---

## 3. Specialisations: how the work is laid out

The host's pure `route(b, n, npad, k)` picks one of five kernel families from the
batch size, the envelope `n`, the row stride and `k` (`route()` and `route_streaming()` in the host file; the varlen launcher
`_varlen_launcher` reuses them with the per-call envelope, taking the free
route's answer for the register-resident and cluster families and
`route_streaming(force_main=True)` otherwise). Row-count
thresholds are for a 148-SM B200 (`wide = b <= 148`, one wave of CTAs).

| Family | Rows / envelope admitted | Geometry per row | Where the data lives | Algorithm variant |
|---|---|---|---|---|
| `reg` / `regimg` | n <= 16 384 (n4 <= 4096 float4); `regimg` for 2k < n <= 4k, k <= 1024, b <= 148 | one CTA of 256 / 512 / 1024 threads, 1 / 2 / 4 float4 per thread | registers | bracket from the first k values, 1024- or 2048-bin histogram, escape = radix descent over the register batch |
| `reg_clus` | 16k < n <= 128k, k <= 1024, `148 // b >= 2` (cluster of 8 vetoed for b > 15) | cluster of 2 / 4 / 8 CTAs x 1024 threads, 1 to 4 float4 per thread | registers across the cluster | same, histograms merged over DSMEM, crossing bin refined by rank 0 |
| `main`, R = 1 | any n, everything not caught above | one CTA: 1024 threads (b <= 148), 512 (b <= 296), 256 (larger) | streamed from global memory | section 2 as written |
| `main`, SPLIT (R > 1) | b <= 32 with `R = min(148 // b, ceil(n4 / 1024))`, when not admitted to `clus` | R CTAs per row, 1024 threads each | streamed; candidates published to a global-memory slab | one attempt, threshold pre-lowered to TSH, last-arriving CTA finishes the row |
| `clus` | 2 <= R <= 8 and k <= 1024 (b <= 32, or 33..74 with n >= 64k) | cluster of 2 / 4 / 8 CTAs x 1024 threads | streamed; merge over DSMEM | four-float4 sample, cluster-uniform retry ladder, rank 0 refines |

`wide` launches (b <= 148) get the largest CTAs and one CTA per SM; multi-wave
launches trade CTA size for occupancy (`MINB` = 2 or 4 CTAs per SM). The candidate
capacities scale the same way: `SCAP` 16384 for a one-wave single-CTA row, 8192
for split rows and for k > 1024, 4096 otherwise; `CMP` 4096 / 2048 / 1024.

### 3.1 Register-resident single CTA (`reg`, `regimg`)

For rows up to 16k values the whole row is loaded once into registers (`BLK`
threads x `VPT` float4, `VPT` in 1, 2, 4, device line 3660 onward). With the row
on chip the guess is free (the first k values are already in registers, 2.3) and
every later phase is a register sweep: the histogram over the bracket (`NBH` =
1024 bins, 2048 for the `regimg` tuple), the crossing-bin find (`find_cross`, line
722, one warp), the emission of above-bin elements straight from registers, the
quadratic or narrowing refinement of the crossing bin, and the escape. The escape
(comment at line 3884) is a radix descent over the key space of the register
batch that runs when the bracket collapses (`Tv >= GMAX`, an infinite width, or a
NaN) or when the histogram never reaches k; it ranks the full row independently of
the bracket, so nothing here re-reads global memory. The `regimg` tuple's
shared-memory image of the row and the bitmap arm exist for the hinted V1 mode
and are compiled out under `hint_free` (line 3489), so for the harness it is the
`reg` algorithm with 2048 bins. Dispatch picks 256 threads for n <= 1k, 512 for
<= 2k, 1024 (or 512 x 2 float4 when b > 148) for <= 4k, and 1024 x 4 float4 for
<= 16k on one-wave launches.

### 3.2 Clustered register-resident (`reg_clus`)

Rows of 16k to 128k values with k <= 1024 stay in registers by spreading them
over a cluster of `CS` CTAs (`CS · 1024 · VPT · 4 >= n`, `CS` the smallest power
of two that fits, device line 6038). Every CTA redundantly reads the first k
values for the bracket (P0), loads its slice (P1), and histograms it (P4). The
cluster merge (P5, line 6313, `merge_scan0`) has one warp per CTA read all `CS`
histograms through DSMEM (`mapa` + `ld.shared::cluster`) and produce both the
cluster totals and this rank's prefix, so the scan (P6) yields output cursors
that are already cluster-global: each rank writes its above-bin elements directly
to their final positions with local atomics (P7). Crossing-bin candidates are
striped into DSMEM slabs on rank 0 (`CMPC` = 4096 slots, `slot = i & 4095,
rank = i >> 12`), and rank 0 alone finishes the row (P9, line 6410): quadratic
rank if `m <= 96`, key-space narrowing over the striped slabs otherwise, and a
whole-row narrowing safety net when the crossing bin exceeds the cluster buffer.
A final cluster barrier keeps the peers resident until rank 0 has read their
slabs. Admission requires `148 // b >= 2` free SMs per row, and the 8-CTA cluster
is vetoed for b > 15 because eight-wide clusters do not pack into GPCs at that
occupancy; such rows fall through to the streaming families.

### 3.3 Streaming single CTA (`main`, R = 1)

The reference implementation of section 2: one CTA per row, `U` float4 per
thread per step (1 to 8, larger for longer slices), software prefetch one step
ahead, the sample and its ladder computed by warp 0 while the other warps issue
L2 prefetch hints for their own slice, up to three attempts with re-priming of
the prefetch registers between them. This is the only family with no upper bound
on `n`; it is what every row of a multi-wave launch (b > 148) runs, with the CTA
size stepped down (512 threads and two CTAs per SM up to b = 296, 256 threads and
four CTAs per SM beyond) so the launch fills the machine. One detail: for rows
that fit the candidate buffer without sampling (`n <= SCAP` and not the
`small_dense` case) the sampling gate is off, `T` is the sentinel, every value is
staged, and the row is resolved by key-space narrowing over the staged set,
i.e. an on-chip radix select over the whole row after one read.

### 3.4 Streaming multi-CTA split (`main`, SPLIT, R > 1)

For small batches on long rows a single CTA per row cannot fill the GPU, so the
row is cut into `R = min(148 // b, ceil(n4 / 1024))` chunks of `Q` float4 and
each chunk goes to its own CTA (grid `(R, b)`, up to `R = 148` for `b = 1`).
All `R` CTAs compute the same sample and the same `T` (the sample is a function
of the row, not of the chunk), stream their chunk, and publish their candidates
to a per-row slab in the global workspace: an atomic on a per-row offset word
hands out slab positions (`GCAP` = 16384 candidates per row), a coalesced store
writes the packed candidates (or a re-sweep of the chunk if the CTA's buffer
overflowed), a fence and a 64-bit atomic on an arrival word (`count << 32 |
candidates`) tell the last-arriving CTA that it is last (line 2309 onward). That
CTA loads the slab, rebuilds the histogram, scans for k, and runs emission and
refinement alone as in 2.5, reading candidates from the slab if there are more
than fit in shared memory. It also restores the two words to zero, so the slab
needs zeroing only once per device (`default_workspace`, 20,973,568 bytes =
2048 + 160 rows x 16384 x 8; `MAXC = 160` is the cap on rows split at once).
There is no retry in this mode (`NATT = 1`): the row would have to be
re-streamed by all CTAs. Instead the threshold is lowered up front to the
rank-`2·TGT` sample value ("TSH-floor staging", line 2095), which roughly doubles
the staged population and makes `count >= k` very likely; a miss falls to the
degenerate paths on the last CTA.

### 3.5 Cluster streaming (`clus`)

Where the split degree is 2 to 8 and k <= 1024, the split runs inside a thread
block cluster instead of through global memory (device line 4601). The `CS` CTAs
own interleaved chunks of the row; each draws the sample with four float4 per
sample point ("QUAD" geometry) and computes the rung redundantly, stages its
survivors locally, and merges its histogram with the others over DSMEM
(`merge_scan0`, one warp per rank reading all ranks). The verify and the retry
ladder are therefore cluster-uniform and the whole cluster can re-stream the row
on a miss, which the global-slab split cannot. Above-bin elements are emitted by
every rank against cluster-global cursors; crossing-bin candidates go to rank 0's
`ck64c` through DSMEM stores, and rank 0 refines (quadratic up to 288, else
narrowing, else whole-row narrowing). The 8-wide cluster is again vetoed for
b > 15 and rounded down to 4.

### 3.6 Prefill mode

PR #18702 adds a compile-time `prefill` flag to the streaming kernel only
(`GvrMainKernel(prefill=True)`, always R = 1). Row `r` selects over the window
`[ks, ke)` given by two device arrays (`row_starts`, `row_ends`), which ride the
otherwise unused `kv_lens` and `pre_idx` argument slots so the compiled signature
is unchanged. The base is rounded down to a 16-byte boundary and the up to three
"lead" lanes below `ks` are masked positionally in the classification mask, the
sample fold and the two whole-row sweeps rather than replaced by `-inf` (which
would corrupt the sample histogram and form a false tie class); output indices are
written in the local frame `column - ks` with a -1 pad, and windows shorter than k
get the identity. The launcher cache is keyed on the row-count tier (<= 148 /
<= 296 / more, choosing the 1024 / 512 / 256-thread engine), k and a power-of-two
envelope bucket capped at 32768, and launches are slabbed to at most 32768 rows,
so arbitrary q-tile sizes do not grow the cache. The prefill kernel never JIT
compiles under CUDA-graph capture; production warms the six or so engines per k
before serving.

### 3.7 What the real-data sweep exercised

From the `note` column of `plots/gvr_v2/real_*.jsonl` (host `route()` per cell):

| Rows N | bs 1 to 8 | bs 16 | bs 32 | bs 64 | bs 128 | bs 256 |
|---|---|---|---|---|---|---|
| 1k to 2k (flash/pro 4k, 8k) | `reg` 512 / 1024 threads | `reg` | `reg` | `reg` | `reg` | `reg` 512 (x 2 float4 for the 2k row) |
| 4k to 8k (flash/pro 16k, 32k; v32 4k, 8k) | `reg` 1024 x 4 float4 | `reg` | `reg` | `reg` | `reg` | `main` 512 threads |
| 16k to 64k, k <= 1024 (flash/pro 64k to 256k) | `reg_clus` cs 8 | `reg_clus` cs 4 (64k, 128k), `main` split R 9 (256k) | `reg_clus` cs 4, `clus` cs 4 | `reg_clus` cs 2 / `main` R 1 / `clus` cs 2 | `main` 1024, R 1 | `main` 512, R 1 |
| 16k to 160k, k = 2048 (v32 16k to 256k) | `main` split R 5 to 40 | split R 5 to 9 | split R 4 | `main` 1024, R 1 | `main` 1024, R 1 | `main` 512, R 1 |
| 128k to 256k, k <= 1024 (flash/pro 512k, 1024k) | `main` split R 18 to 64 | split R 9 | `clus` cs 4 | `clus` cs 2 | `main` 1024, R 1 | `main` 512, R 1 |

Every family appears on production shapes: the register-resident one owns the
short-row band regardless of batch size, `reg_clus` and `clus` the mid band at
small batches with k <= 1024 (v32 with k = 2048 is excluded from both by the
k gate and uses the split instead), the split the long rows at batch 16 and
below, and the single-CTA streaming kernel everything at batch 128 and above.

---

## 4. Reading the heatmaps with this in mind

- **Small batch, any row length (bs <= 32): GVR V2 leads CUB by 1.3 to 1.4x
  (geometric mean).** This is the pass-count difference of section 2.7: one read
  plus a 3% sample against three histogram passes, both from a resident or
  near-resident working set. CUB's cluster kernel is closest (within 1.15x) at
  batch 16 and below on flash's 512k rows, pro's 256k to 512k rows and v32's 32k
  to 128k rows: the cells where GVR V2 runs its split and has to funnel the slab
  through one last CTA, or a cluster whose rank 0 finishes the row alone.
- **Large batch (bs >= 128) on rows of 32k and up: 1.9 to 3.8x.** CUB's cluster
  kernel serialises waves at one 512-thread CTA per SM and a full shared-memory
  budget (the finding of the CUB report), while GVR V2's dispatch drops to one
  CTA per row at two or four CTAs per SM. This is a dispatch-policy gap, not an
  algorithmic one: the same batch-aware CTA sizing is available to a radix
  select.
- **Rows below 16k at batch 32 and below: 1.1 to 1.8x.** CUB's single-block
  baseline runs its three passes on a row that GVR V2's register-resident kernel
  loads once and never re-reads; the bracket from the first k values costs
  nothing there. The two cells CUB wins (flash 4k at bs 256, 9.6 µs against
  9.6 µs) sit at the launch-latency floor where nothing separates the arms.
- **The prefill arm's 37 cell wins** are 0.1 to 6% margins over the decode arm.
  30 of them are at bs >= 64, where both arms run the same single-CTA streaming
  kernel and the prefill compile is a little leaner (no split machinery, a
  simpler prologue). The seven at bs <= 32 are cells where the decode arm's
  split or cluster variant came out 0.4 to 3% behind one CTA per row, i.e. the
  split bought nothing there. None of this says anything about prefill-shaped
  workloads, which the harness does not contain.

What the sweep cannot show is the data-dependent tail (section 2.7): every cell
holds identical rows of production logits, exactly the distribution the sampling
ladder was tuned on. Bit-clustered synthetic inputs (the harness's `10bit`
distribution puts thousands of exact ties at rank k) push the crossing bin past
`CMP` and into the narrowing fallbacks; the quick verifier's `ties` inputs pass
exactly but were not timed against CUB.

---

## 5. Is the win an artefact of the benchmark data? A falsification run

The heatmaps time identical production rows per cell, the distribution the
sampling ladder was tuned on. Two tools were added to test how much of the win
depends on that (`benchmarks/gvr_v2_path_predictor.py`, which emulates the part
of the kernel that decides its cost, and `benchmarks/bench_topk_sensitivity.py`,
which keeps the heatmap's shapes and timing path and swaps the data), and run on
the same node as the heatmaps (`plots/sensitivity/`). Shapes: flash 64k and
512k, pro 128k, v32 64k and 256k, at batch 8 and 128; arms CUB, GVR-V2 decode,
SGLang; ratio = CUB time / GVR-V2 time, geometric mean over the ten cells.

### 5.1 The median advantage does not depend on the data

| data | ratio | min | max | note |
|---|---|---|---|---|
| real (the heatmap rows) | 1.69 | 1.14 | 2.67 | reference |
| real, columns shuffled | 1.77 | 1.25 | 2.69 | no position structure is exploited |
| real, rounded to bf16 | 1.94 | 1.28 | 3.17 | 65k-level ties cost CUB more than GVR V2 |
| real + 8 sinks at max + 20 | 1.77 | 1.14 | 3.16 | |
| gaussian | 1.83 | 1.30 | 2.90 | |
| uniform | 2.02 | 1.44 | 3.13 | |
| 21bit (shared exponent) | 1.97 | 1.40 | 3.14 | the radix-adversarial pattern |
| 10bit (1024 distinct values) | 1.95 | 1.36 | 3.14 | |
| student-t, 2 degrees of freedom | 1.52 | 0.96 | 2.57 | heavy tails: parity in one cell at bs 128 |
| sorted descending | 1.77 | 1.12 | 3.02 | |
| top-k in one contiguous block | 1.31 | 0.88 | 1.99 | CUB gets faster, GVR V2 refines a dense crossing bin |
| heterogeneous real batch (one model's 7 to 9 rows, own lengths) | 1.66 | 1.09 | 2.50 | vs 2.07 for the longest row repeated |
| 25 distinct real rows one at a time (bs 1) | p50 1.28 | 1.00 | 1.75 | GVR V2 p50 11.7 µs, p99 15.9 µs; never slower than CUB |

On everything smooth, bit-patterned, shuffled, quantised to bf16, sorted or
mixed, GVR V2 keeps a lead of the same size as on the heatmap rows, and the
patterns written to defeat radix kernels widen it. The predictor puts all 25
committed rows on the fast path at every batch size, with one exception that
matches the heatmap: at batch 32 the pro 512k and 1024k rows run the cluster
kernel, whose sample is only 36 blocks of 16 consecutive values there, and
correlated neighbours make its rank-15 estimate miss by an order of magnitude
(the 15th largest sample sits at a value only 356 row entries exceed instead of
about 3400), so the kernel retries at its lower rung: two reads instead of one,
and the cells where CUB came closest (pro 512k at bs 32, ratio 1.006).

### 5.2 Where it does depend on the data: two cliffs

| data | ratio | min | GVR V2 path (predicted, confirmed by the timings) |
|---|---|---|---|
| gaussian + 8 spikes of 1e2 per row | 1.60 | 1.13 | fast path |
| + spikes of 1e3 | 1.31 | 1.00 | 10 to 20% of rows overflow the candidate buffer: one extra read |
| + spikes of 1e4 | 0.53 | 0.14 | rows whose sample or first-k bracket catches a spike fall to whole-row narrowing |
| + spikes of 1e6 ("outlier") | 0.50 | 0.13 | same; 160 to 235 µs per batch against CUB's 24 to 105 |
| 16 distinct values ("q16") | 0.73 | 0.11 | tie class at rank k of n/16 exceeds every buffer: narrowing over all candidates or the whole row |

Both cliffs follow from section 2. The sample histogram and the register-family
bracket are uniform grids of 256 or 1024 bins over `[min, max]` of what was
sampled. A spike more than about a thousand times the bulk's scale away puts the
whole bulk into one bin, the threshold collapses to the sample minimum, every
value becomes a candidate, and the kernel ends in the whole-row narrowing of 2.6,
eight passes by one CTA. Spikes of 100 times the scale leave the bulk a dozen
bins and cost nothing. Massive ties break the other assumption, a small crossing
bin, and are resolved by narrowing over thousands of candidates. CUB's time moves
by about 20% across all of these inputs and never cliffs; radix select has no
estimate to get wrong.

Two properties of the cliff matter for interpretation. It is per row and the
batch runs at its slowest row: in the 1e6 cells two or three of eight rows were
predicted to narrow and the batch took 190 µs, while a batch whose single spiked
row happened not to have a spike in its sample ran at full speed. And the
predictor sees it coming: the per-row emulation, which needs no GPU, flagged
every slow cell before it was timed, so the question "does this workload's data
trigger it" can be answered offline on captured rows.

Realistic triggers are values far outside the logit bulk inside the valid
window (a masking sentinel such as `-1e30` or `finfo.min` that is not excluded
by the length, or genuine extreme scores) and aggressively quantised scores; the
25 captured rows contain neither, and rounding them to bf16 did not create a
problematic tie class.

### 5.3 The whole capture: 13,832 real rows

The raw NFS capture (all 21 layers, all decode steps, three models) was copied
to the node and run through the predictor at batch 1, 32 and 128, and 180 of
its rows were timed one at a time (`plots/sensitivity/capture*`). 900 rows were
dropped as garbage (NaN or magnitudes of 1e34 to 1e38 inside the valid prefix,
all in the first decode steps).

| | value |
|---|---|
| rows predicted off the fast path, all steps | 395 of 41,496 predictions (0.95%), 156 distinct rows |
| of which in decode steps 1 to 3 | 321; the flash 1024k row at step 2 has identical statistics in all 21 layers (same threshold, 41,924 survivors of 262k, a plateau near 1.0), so these are capture artefacts, not per-layer logits |
| rows predicted off the fast path, steps 4 to 18 | 72 of 34,104 (0.21%), 68 of them the moderate "narrow" refinement; no whole-row narrowing |
| streaming rows predicted to retry, steps 4 to 18 | 240 of 15,225, all at batch 32 (218) and 128 (22) on 512k and 1024k rows: the cluster kernel's small sample, section 5.1 |
| candidates staged per k (streaming rows) | p50 2.6, p99 5.7 |
| crossing bin size | p50 16, p99 108 |
| 180 timed rows at batch 1, CUB / GVR-V2 | p50 1.32, p90 1.59, p99 1.83; GVR V2 slower in 2 rows |
| the 2 slow rows | both artefact rows (steps 1 and 2): 305 µs against CUB's 22.8 on the plateau row (predicted whole-row narrowing), 26 µs against 18 on a near-constant 64k row (predicted register-family escape) |
| mixed batches of 8 and 128 distinct rows per model, own lengths | CUB / GVR-V2 1.22 to 2.74, geometric mean 1.69 |

So on rows that are genuine decode logits the predictor finds no cliff event in
34k rows and the timings show none in 180; the 3.5% retry rate of the cluster
kernel at batch 32 is the only systematic cost, and it is what the heatmap's
closest cells already showed. The artefact rows are nevertheless instructive:
they are real buffers a kernel could be handed, they are exactly the plateau
case of 5.2, and the predictor flagged every one of them before it was timed.

### 5.4 Side finding

SGLang's `topk_v2` failed the exactness check on the `uniform` inputs at batch 8
(three cells, values near 1.0 differing in the fifth decimal), on the plateau
rows of the capture (two cells), and, as documented, on the bit-clustered
`21bit` / `10bit` inputs. Its fp16 coarse histogram loses resolution where
values are dense near the top, and `kMaxNumTie` then truncates the candidate
set. CUB and GVR V2 were exact in every measurement of these runs (1,305).
