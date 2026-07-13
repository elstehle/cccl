# `block_topk_air` latency optimizations — per-change ablation & integration guide

Companion to `BLOCK_TOPK_AIR_OPT_RESULTS.md` (§9 has the condensed version). This report is the
integration-facing view: for each change, what the code looks like before/after, what it buys,
what it costs, and which key/value type combinations it applies to.

Measurement: `proto_air_ablate.cu`, NVIDIA B200 (sm_100), CUDA 13.1, node `umb-b200-235`
(baseline reproduces the umb-b200-240 numbers within 0.5%). `AirAblate<KeyT, ValueT, LVL>`
applies the changes **cumulatively in integration order** to a faithful reimplementation of
today's `block_topk_air` (L0). Latency = chain-length-slope cycles/call (single block, min of
24 reps); throughput = fixed workload, 2048 blocks × 33 calls, random input. All 30
instantiations pass set-semantics correctness on all patterns (incl. `neg_zero`, `all_equal`,
`pivot_tie40`) — every level is a correct drop-in.

## Summary table — f32 keys + i32 values (the reference configuration)

| level | change | random | Δ | tie_heavy | pivot_tie40 | sorted | G elem/s | regs | smem B | blk/SM |
|---|---|---|---|---|---|---|---|---|---|---|
| L0 | faithful `block_topk_air` | 2601 | — | 2428 | 4542 | 2534 | 317 | 37 | 4272 | 6 |
| L1 | + fused scan+choose | 2466 | **−135** | 2282 | 4256 | 2392 | 336 | 37 | 4272 | 6 |
| L2 | + double-buffered histograms | 2256 | **−210** | 2075 | 3846 | 2176 | 347 | **32** | 4272 | **8** |
| L3 | + preset scatter counters | 2203 | **−53** | 2023 | 3804 | 2155 | 347 | 32 | 4272 | 8 |
| L4 | + pair scatter (pairs only) | 2154 | **−49** | 1959 | 3751 | 2089 | 366 | 32 | 8368 | 8 |
| L5 | + original-value scatter (fp pairs) | 2137 | **−17** | 1947 | 3734 | 2085 | 383 | 32 | 8368 | 8 |
| L6 | + compile-time unrolled passes | **1817** | **−320** | **1601** | **3221** | **1751** | **440** | 48 | 8368 | 5 |
| L7 | + packed pass state *(diagnostic)* | 1823 | +6 | 1620 | 3264 | 1758 | 421 | 48 | 8368 | 5 |

**Total L0→L6: −30% latency, +39% throughput, zero spills at every level.**
Other configurations, L0→L6 (random): u32+i32 2514→1808 (−28%, 356→452 G elem/s);
f32 keys-only 2525→1813 (−28%); f16+i32 2600→1893 (−27%, worst pattern 2636→1909);
f64+i32 3517→2588 (−26%, worst pattern 8724→6250); f32+i64 → 1856.

## Where the cycles went — per-phase profile, L0 vs L6 (f32+i32)

`./proto_air_ablate prof` stamps `clock64` at every phase boundary of both versions (same
technique as `BLOCK_TOPK_AIR_OPT_RESULTS.md` §1; min across 24 reps × 4 chained calls; totals
agree with the slope measurements within ~2%). A `~0` means the optimizations *eliminated* the
phase:

| stage | L0 random | L6 random | Δ | L0 pivot40 | L6 pivot40 | Δ |
|---|---|---|---|---|---|---|
| prologue (twiddle/init/preset) | 68 | 28 | −40 | 68 | 28 | −40 |
| p0 init / state-update | 29 | 33 | +4 | 29 | 33 | +4 |
| **p0 histogram** | 339 | **30** | **−309** | 339 | **30** | **−309** |
| p0 scan (+choose) | 377 | 397 | +20 | 377 | 382 | +5 |
| p0 choose | 85 | ~0 | −83 | 86 | ~0 | −84 |
| p1 init / state-update | 103 | 74 | −29 | 103 | 74 | −29 |
| **p1 histogram** | 415 | **234** | **−181** | 407 | 234 | −173 |
| p1 scan (+choose) | 377 | 377 | ±0 | 378 | 381 | +3 |
| p1 choose | 89 | ~0 | −81 | 87 | ~0 | −81 |
| p2/p3 (pivot only): histogram / choose each | — | — | — | ~408 / ~87 | ~235 / ~0 | ~−173 / ~−83 |
| post-loop bar + scatter setup | 117 | 89 | −28 | 121 | 74 | −47 |
| scatter | 245 | 213 | −32 | 459 | 442 | −17 |
| gather keys/pairs | 212 | 207 | −5 | 42 | 32 | −10 |
| scatter values | 84 | **6** | **−78** | 84 | **6** | **−78** |
| gather values + out | 105 | 93 | −12 | 101 | 89 | −12 |
| **TOTAL** | **2645** | **1791** | **−854** | **4633** | **3227** | **−1406** |

Reading (random / pivot40):

1. **The largest win by far is the histogram phases: −490 / −828** — more than half of the
   total. Two mechanisms: (a) pass 0's histogram all but vanishes (339→30) because with the
   unrolled, fused structure the compiler software-pipelines the pass-0 atomics into the
   prologue — pass 0 has no filter dependency, so its digit computation and `ATOMS.ADD` issue
   can start immediately; the runtime loop + phase barriers of L0 prevented that hoisting.
   (b) later passes drop ~175 each from the shortened state-broadcast→filter dependency chain
   and the init fold (double-buffering).
2. **Choose elimination: −164 / −331.** The ~87-cyc choose phase per pass goes to ~0, with only
   +0..+20 reappearing on the scan phase — the fusion is nearly free.
3. **Epilogue: −155 / −164**, dominated by the value-scatter trip collapsing (84→6, pair
   scatter), plus setup (−24) and scatter/gather trims.
4. **What does not move: the scan, ~377/pass, in both versions** — now 42% of the optimized
   call. This is the measured floor from §1 (5-step shuffle chain + cross-warp fold + barrier);
   further gains here mean changing the algorithm, not the implementation.

Two structurally different kinds of savings (verified: the 4-pass `pivot_tie40` deltas of the
first kind are ≈2× their 2-pass `random` deltas):

* **per executed pass** — L1 (−135), L2 (−210), L6 (−320): grow with pass count (f64 keys,
  tie floods), shrink for f16 (2 passes max);
* **fixed, epilogue** — L3 (−53), L4 (−49), L5 (−17): independent of pass count and K.

---

## L1 — fused scan + choose

**What:** `compute_bin_offsets` writes the scanned histogram back to shared memory so that
`choose_bucket` can re-read each bin and its predecessor. But after `InclusiveSum`, every thread
already holds its bin's inclusive sum in a register, and the exclusive sum is one subtraction
away. The crossing test can run immediately; the writeback, the choose phase, and one barrier
per pass disappear (~260 shared accesses/pass).

```cpp
// BEFORE (two phases, two barriers, 256-word round trip through smem)
unsigned tb = hist[tid];
block_scan_t(scan_temp).InclusiveSum(tb, tb);
hist[tid] = tb;                                   // writeback
__syncthreads();
const unsigned prev = (tid == 0) ? 0 : hist[tid - 1];   // choose re-reads neighbours
const unsigned cur  = hist[tid];
if (prev < k && cur >= k) { pass_state = {tid, cur - prev, prev}; }
__syncthreads();

// AFTER (one phase, one barrier, all in registers)
const unsigned cnt = hist[tid];
unsigned incl;
block_scan_t(scan_temp).InclusiveSum(cnt, incl);
const unsigned excl = incl - cnt;                 // exclusive = inclusive - own count
if (excl < k && incl >= k) { pass_state = {tid, incl - excl, excl}; }
__syncthreads();
```

**Buys:** −106..−135 cyc (2-pass random), −220..−290 (4-pass patterns). **Costs:** nothing.
**Applies to:** every key type, keys-only and pairs, any RadixBits (with >1 bucket/thread the
same subtraction works per array element). *This is also the PR #9066 sieve's own open TODO.*
**Verdict: unconditional.**

## L2 — double-buffered histograms

**What:** today every pass starts with a zero-init phase plus a barrier. With two histogram
buffers, both are zeroed once up front; while pass *p* histograms into buffer `p & 1`, threads
also zero the *other* buffer — which is safe because its last reader (pass *p−1*'s scan) is
already behind a barrier. The per-pass init phase and its barrier disappear.

```cpp
// BEFORE (every pass)                         // AFTER (prologue, once)
hist[tid] = 0;                                 hist[0][tid] = 0;
__syncthreads();                               hist[1][tid] = 0;
/* histogram ... */                            __syncthreads();

                                               // AFTER (inside each pass's histogram phase)
                                               unsigned* cur = hist[pass & 1];
                                               /* histogram atomics into cur ... */
                                               if (pass > 0 && pass + 1 < num_passes)
                                                 hist[(pass + 1) & 1][tid] = 0;  // free ride
                                               __syncthreads();
```

**Buys:** −190..−215 cyc (and more on 4-pass patterns). **Costs:** none in practice — the extra
1 KB buffer unions under the (larger) exchange stage, so `TempStorage` is unchanged even for
f64 keys; registers *dropped* 37→32 and occupancy *rose* 6→8 blocks/SM in the measurement.
**Applies to:** every type and mode. **Verdict: unconditional.**

## L3 — preset scatter counters + computed tied-base

**What:** the scatter stage currently opens with a setup phase (thread 0 seeds
`selected_offset[0] = 0`, `selected_offset[1] = total_selected`) plus a barrier — and the
post-loop "repurpose smem" barrier ahead of it. Instead, zero both counters in the prologue
(they live *outside* the aliased union, +8 B) and compute the tied-class position as
`total_selected + zero_based_ticket`; `total_selected` is block-uniform in registers. Both
barriers and the setup phase disappear.

```cpp
// BEFORE                                          // AFTER (prologue)
__syncthreads();               // repurpose        if (tid == last) { cntA = 0; cntB = 0; }
if (tid == 0) {                                    // ...ordered by the pass-stage barriers...
  sel_off[0] = 0;
  sel_off[1] = total_selected;                     // AFTER (scatter)
}                                                  const unsigned t =
__syncthreads();               // setup              atomicAdd(cls1 ? &cntB : &cntA, 1u);
const unsigned off =                               const unsigned off =
  atomicAdd(&sel_off[cls1], 1u);                     cls1 ? total_selected + t : t;
```

**Buys:** −23..−53 cyc, fixed. **Costs:** +8 B smem. **Applies to:** every type and mode.
**Verdict: unconditional.**

## L4 — pair scatter (pairs only)

**What:** to keep the exchange buffer at `tile_items * max(sizeof(KeyT), sizeof(ValueT))`,
today's pairs epilogue makes two full trips: scatter keys → barrier → gather keys → barrier →
scatter values → barrier → gather values. Scattering `(key, value)` together and gathering once
removes two barriers and one full item pass; the `scatter_indices` bookkeeping registers also go.

```cpp
// BEFORE: 4 phases, 3 barriers                 // AFTER: 2 phases, 1 barrier
exch.keys[off] = key_i;      /* + idx bookkeeping */
__syncthreads();                                exch.pairs[off] = {key_i, value_i};
keys[i] = exch.keys[bi];                        __syncthreads();
__syncthreads();                                keys[i]   = exch.pairs[bi].key;
exch.values[scatter_idx[i]] = value_i;          values[i] = exch.pairs[bi].value;
__syncthreads();
values[i] = exch.values[bi];
```

**Buys:** −49 (f32+i32) to −87 (u32+i32) cyc + ~20 G elem/s. **Costs:** exchange grows to
`tile_items * sizeof(pair)`: 4.3→8.4 KB for 4B/4B, but **16.6 KB for 16 B pairs** (f64 keys or
i64 values) — where the measured win also shrinks (−32 cyc for f32+i64).
**Applies to:** pairs only (keys-only has no value trip — L4 is a no-op there).
**Verdict: default-on for pair sizes ≤ 8 B; policy-gate 16 B pairs.**

## L5 — original-value scatter (fp pairs only)

**What:** the float path twiddles keys in place, normalizes `-0.0` to `+0.0` (tracking which
items were flipped in a bitvector), and at scatter time untwiddles and restores `-0.0`. Keeping
a register copy of the original keys and scattering *that* deletes the untwiddle and the entire
flip machinery.

```cpp
// BEFORE (prologue)                            // AFTER (prologue)
uk[i] = twiddle(keys[i]);                       KeyT orig[i] = keys[i];   // 4 extra registers
if (uk[i] == tw_minus_zero) {                   uk[i] = twiddle(keys[i]);
  flip |= 1u << i;                              // no normalization, no flip tracking
  uk[i] = tw_plus_zero;
}
// BEFORE (scatter)                             // AFTER (scatter)
exch[off] = (flip >> i & 1) ? -0.0f             exch[off] = orig[i];
                            : untwiddle(uk[i]);
```

**Buys:** −17 cyc + 17 G elem/s for f32+i32 pairs; small for f16/f64 pairs. **Costs / limits:**
* integer keys: **exactly 0** — the compiler already elides the identity un-conversion (the u32
  L4 and L5 binaries measure bit-identically). Skip.
* keys-only: **harmful** — control-measured at +19 registers, occupancy 8→6, −31 G elem/s
  (the register copy outweighs the saved untwiddle). Skip.
* semantics: dropping the ±0 normalization means `-0.0` ranks just below `+0.0` (a consistent
  refinement of the float order — any valid top-k tie-break allows it, and `neg_zero`
  correctness passes) — but it is a *visible tie-break change*; keep the normalization if
  bit-exact parity with today's behavior at a ±0 boundary is required.

**Verdict: fp pairs only; document the ±0 nuance.**

## L6 — compile-time unrolled pass loop

**What:** the pass loop runs with runtime `begin_bit`/`end_bit` arithmetic. When the bit range
is compile-time (the default instantiation covers `[0, sizeof(KeyT)*8)`), fully unrolling turns
every shift, mask, and histogram-buffer selection into immediates, and lets the scheduler
software-pipeline across phase boundaries.

```cpp
// BEFORE                                        // AFTER
#pragma unroll 1                                 #pragma unroll
for (int pass = 0; pass < num_passes; ++pass) {  for (int pass = 0; pass < NPASS; ++pass) {
  const int begin = end_bit - 8 * (pass + 1);      // begin/masks fold to immediates,
  ...                                              // hist[pass & 1] to fixed addresses
```

**Buys:** **the single biggest lever**: −320 cyc random, −513 on 4-pass patterns, +57 G elem/s.
**Costs:** the only change with a real resource bill: +16 registers, occupancy 8→5 blocks/SM
(throughput still rises 27% despite it); code size grows with `NPASS` (f64 = 8 unrolled passes,
64 regs). Runtime bit ranges naturally keep the rolled loop, so this is a specialization, not a
behavior change. **Applies to:** every type/mode with compile-time bits.
**Verdict: integrate for the default instantiation; consider a policy knob if
occupancy-constrained embedders complain.**

## L7 — packed pass state *(evaluated, rejected)*

**What:** pack `(bucket | candidates | selected)` into one 32-bit word so the per-pass state
broadcast is a single shared load instead of three.

```cpp
// considered                                         // kept (faithful)
st = (bucket << 16) | (cands << 5) | sel;             pass_state.bucket / .candidates / .selected
```

**Measured:** +6..+47 cyc and −19 G elem/s here; **~+500 cyc** when ported into PR #9066's
sieve. Three independently-issued (pipelined) shared loads beat one load followed by a
dependent unpack chain feeding the next pass's filter. **Verdict: do not integrate** — and a
general porting lesson: bundled micro-optimizations must be re-measured per structure.

---

## Applicability matrix

| change | f32 pairs | int pairs | f16 pairs | f64 pairs | 8 B values | keys-only |
|---|---|---|---|---|---|---|
| L1 fused scan+choose | ✓ | ✓ | ✓ | ✓✓ (8 passes) | ✓ | ✓ |
| L2 double-buffered hist | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| L3 preset counters | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| L4 pair scatter | ✓ | ✓ | ✓ | gate (16 B pair) | gate (16 B pair) | n/a |
| L5 orig-value scatter | ✓ | skip (no-op) | ✓ (small) | ✓ (small) | ✓ | **skip (harmful)** |
| L6 unrolled passes | ✓✓ | ✓✓ | ✓ | ✓ (code size) | ✓✓ | ✓✓ |
| L7 packed state | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

## Suggested integration order

1. **L1 + L2** — pure wins for every type and mode, and L2 *improves* registers/occupancy.
2. **L3** — small pure win, all types.
3. **L6** — biggest win; +16 regs / −3 occupancy slots; gate only if needed.
4. **L4** — pairs, ≤8 B pair size by default; gate 16 B pairs.
5. **L5** — fp pairs only; mind the ±0 note.
6. **L7** — do not integrate.

Reproduce: `./run_remote.sh proto_air_ablate.cu` (or build with the standard nvcc line) and
`./proto_air_ablate [correct|lat|thr|res]`.
