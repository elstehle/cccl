#!/usr/bin/env python3
"""Generate TECHNIQUES.md entries from the intrinsics microbenchmark results.

  python3 techniques_entries.py --out <dir> --b200 <results-dir> <stamp> --gb200 <results-dir> <stamp>

Reads micro_sm_100_<stamp>.jsonl, micro_sm_100f_<stamp>.jsonl and micro_sm_100_repeat_<stamp>.jsonl
from each node's results directory and writes one entry file per technique (T_<nn>_<slug>.md) with the
`### T?: <name>` heading that tools/append_locked.py numbers on append. Numbers carry arch, CUDA, node,
date and file. Either node may be omitted, the entry then says so.
"""
import argparse
import json
import os
import sys

FILE = "harness/micro/intrinsics_micro.cu"
ALU_OPS = 8


def load(d, stamp):
    runs = {}
    for variant in ("sm_100", "sm_100f", "sm_100_repeat"):
        p = os.path.join(d, f"micro_{variant}_{stamp}.jsonl")
        if not os.path.exists(p):
            continue
        rows = {}
        for line in open(p):
            line = line.strip()
            if line:
                j = json.loads(line)
                rows[j["name"]] = j
        runs[variant] = rows
    return runs


class Node:
    def __init__(self, label, d, stamp):
        self.label = label
        self.runs = load(d, stamp) if d else {}
        self.ok = bool(self.runs)
        any_rows = next(iter(self.runs.values())) if self.runs else {}
        any_row = next(iter(any_rows.values())) if any_rows else {}
        self.node = any_row.get("node", "?")
        self.cuda = any_row.get("nvcc", "?")
        self.date = any_row.get("date", "?")
        self.device = any_row.get("device", "?")

    def row(self, name, variant="sm_100"):
        return self.runs.get(variant, {}).get(name)

    def cyc(self, name, variant="sm_100"):
        r = self.row(name, variant)
        return None if r is None else r["cycles"]

    def rep(self, name):
        return self.cyc(name, "sm_100_repeat")


def fmt(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def both(nodes, name, variant="sm_100", nd=1):
    parts = []
    for n in nodes:
        if not n.ok:
            continue
        c = n.cyc(name, variant)
        r = n.rep(name) if variant == "sm_100" else None
        s = f"{n.label} {fmt(c, nd)}"
        if r is not None and c is not None and abs(r - c) > 0.02 * max(abs(c), 1.0):
            s += f" (repeat {fmt(r, nd)})"
        parts.append(s)
    return ", ".join(parts) if parts else "not measured"


def delta(nodes, pair, base, nd=1):
    parts = []
    for n in nodes:
        if not n.ok:
            continue
        p, b = n.cyc(pair), n.cyc(base)
        if p is None or b is None:
            continue
        parts.append(f"{n.label} {(p - b) / ALU_OPS:.{nd}f}")
    return ", ".join(parts) if parts else "not measured"


def provenance(nodes):
    ps = []
    for n in nodes:
        if n.ok:
            ps.append(f"{n.label} = {n.device} sm_100, {n.node}, nvcc {n.cuda}, {n.date}")
    return ". ".join(ps) + f". Chain-slope method of BRIEF section 4, min of 64 repetitions, file {FILE}, rows named as in the JSON under state/results/hw-researcher/<node>/."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--b200", nargs=2, metavar=("DIR", "STAMP"))
    ap.add_argument("--gb200", nargs=2, metavar=("DIR", "STAMP"))
    a = ap.parse_args()
    nodes = [Node("B200", *(a.b200 or (None, None))), Node("GB200", *(a.gb200 or (None, None)))]
    if not any(n.ok for n in nodes):
        sys.exit("no results")
    os.makedirs(a.out, exist_ok=True)
    prov = provenance(nodes)
    missing = [n.label for n in nodes if not n.ok]
    miss = f" Not measured on {', '.join(missing)} yet." if missing else ""

    def B(name, variant="sm_100", nd=1):
        return both(nodes, name, variant, nd)

    def D(pair, base):
        return delta(nodes, pair, base)

    entries = []

    entries.append(("redux_u32", f"""### T?: redux.sync.max.u32 costs about 22 cycles and compiles to one CREDUX
applies to: W2, B2 and B5 warp phases, any warp max or sum of 32-bit keys, all key widths once twiddled or packed into 32 bits
measured: redux.sync.max.u32 dependent chain {B('redux.sync.max.u32')} cycles per call. For comparison shfl_xor single {B('shfl_xor (single)')}, 5-shuffle max tree u32 {B('warp max 5-shfl tree u32')}, f32 (fmaxf) {B('warp max 5-shfl tree f32 (fmaxf)')}, u64 {B('warp max 5-shfl tree u64')}, ballot {B('ballot_sync')}. {prov}{miss}
evidence: `CREDUX.MAX UR5, R9` then `IMAD.U32 R9, RZ, RZ, UR5` (the result lands in a uniform register and is moved back), sass_sm_100.txt kernel `bench<ReduxU32, 4>`. The July 2026 numbers (22.4 redux, 30 shuffle, 150 and 190 trees) re-verified within one cycle on CUDA 13.4.92.
how to use: One redux replaces a 5-step shuffle tree (about 7 times cheaper) for any 32-bit integer reduction. The result is warp-uniform, so a following `lane == winner` test needs the per-lane value compared against it, see the W2 round entry.
caveats: Integer types only on the sm_100 baseline target. `.add` truncates to 32 bits. The uniform-register destination means the value is not in a per-lane register until the IMAD move, which the compiler inserts.
author: hw-researcher
"""))

    entries.append(("redux_f32", f"""### T?: redux.sync.max.f32 needs an sm_100f build and saves the twiddle round trip
applies to: W2 and B5 warp max of raw fp32 keys, threshold broadcasts, any fp32 warp reduction on B200, GB200, B300 and Rubin
measured: redux.sync.max.f32 {B('redux.sync.max.f32 (sm_100f)', 'sm_100f')} cycles, redux.sync.max.abs.NaN.f32 {B('redux.sync.max.abs.NaN.f32 (sm_100f)', 'sm_100f')} cycles, both on the sm_100f build. The sm_100 fallback twiddle + redux.sync.max.u32 + untwiddle costs {B('redux.sync.max.u32 on twiddled f32 (twiddle+redux+untwiddle)')} cycles. {prov}{miss}
evidence: `CREDUX.MAX.F32 UR5, R6` and `CREDUX.MAXABS.F32.NAN UR5, R6` in sass_sm_100f.txt kernels `bench<ReduxF32, 4>` and `bench<ReduxF32AbsNaN, 4>`. The twiddled path is `SHF.R.S32.HI`, `LOP3.LUT`, `CREDUX.MAX`, `IMAD.U32`, `LEA.HI`, `LOP3.LUT` (sass_sm_100.txt, `bench<ReduxF32ViaTwiddle, 4>`). PTX ISA 9.4: `.f32` type requires sm_100a, supported on sm_100f from PTX ISA 8.8, nothing on plain sm_100.
how to use: Build the primitive for `-arch=sm_100f` (runs on compute capability 10.0, 10.3 and 10.7) and reduce fp32 keys directly with `asm("redux.sync.max.f32 %0, %1, 0xffffffff;")`. Semantics: `+0.0 > -0.0`, a NaN input loses to the other operands unless `.NaN` is given. Keep the twiddled u32 path as the sm_100 fallback and select at compile time on `__CUDA_ARCH_FAMILY_SPECIFIC__` (device pass) plus a host define for the launch.
caveats: The mandated swarm build (`-arch=sm_100`) cannot emit it, so an approach that relies on it reports an sm_100f variant next to the baseline. No CUDA C intrinsic exists, inline PTX only. 18 of the 40 cycles saved per round are the twiddle and untwiddle ALU work, so a design that keeps keys twiddled anyway gains nothing.
author: hw-researcher
"""))

    entries.append(("w2_rounds", f"""### T?: a W2 extraction round costs 89 cycles unpacked and 34 cycles with the index packed into the key
applies to: W2 redux based extraction at warp scope, B5 warp phases, keys of 16 bits or less (packed) and 32-bit keys (unpacked)
measured: unpacked round (redux max, ballot of `v == max`, ffs, winner select) {B('W2 round unpacked: redux + ballot(v==max) + ffs + select')} cycles. Packed round (redux max of `key << 16 | index`, winner by `v == max`) {B('W2 round packed key|index: redux + compare select')} cycles. Components: ballot + ffs + winner select alone {B('ballot + ffs + winner select')}, `__ffs` alone {B('alu __ffs')}, redux alone {B('redux.sync.max.u32')}. Warp max through 32 fire-and-forget shared atomics + syncwarp + LDS {B('warp max atomic32: 32 atomicMax one slot + syncwarp + LDS')}. {prov}{miss}
evidence: unpacked kernel `bench<ReduxBallotRound, 4>`: `CREDUX.MAX`, `VOTE.ANY`, `BREV`, `FLO.U32.SH`, `ISETP` per round. Packed kernel `bench<ReduxPackedRound, 4>`: `CREDUX.MAX`, `ISETP`, `LOP3`, `SHF` per round (sass_sm_100.txt).
how to use: For keys of 16 bits or less, pack `(twiddled_key << 16) | (0xffff - index)` and every round is one redux plus a compare, 2.6 times cheaper than the unpacked round, deterministic and prefer-smaller-index for free. For 32-bit keys the ffs is the expensive part (BREV + FLO, about 40 cycles): resolve the winner with `__ballot_sync(v == m)` and `%lanemask_lt` based rank tests or a second redux of the lane id instead of `__ffs` when the critical path matters.
caveats: The packed form caps the index at 65535 per segment. At warp scope a redux round (22) beats the shared atomic round (49), the opposite of block scope where the atomic256 round wins (see the block max entry). The July figure of 82 cycles for the unpacked round is 89 here because this round includes the winner select.
author: hw-researcher
"""))

    entries.append(("ffs_fns_popc", f"""### T?: __ffs is BREV + FLO at 40 cycles, __popc 23, __fns a 40-instruction emulation at 177 cycles
applies to: any lane election, rank or k-th candidate lookup on the critical path, all scopes
measured: dependent chains, cycles per call: `__ffs` {B('alu __ffs')}, `__popc` (plus one IMAD) {B('alu __popc')}, `__fns(mask, 0, 2)` {B('alu __fns')}, `bmsk.clamp` + iadd {D('alu pair: 8 x (bmsk.clamp + iadd + imad)', 'alu base: 8 x imad u32 per step')} per op in the pair chain. {prov}{miss}
evidence: `bench<FfsChain, 4>`: `BREV R0, R9` / `FLO.U32.SH R0, R0` / `IADD3` per call. `bench<FnsChain, 1>`: 43 instructions per call (`SHF`, `ISETP`, `POPC` x4, `SEL` x4, `SGXT` x4, `PRMT`), no FNS opcode exists in the Blackwell SASS table. `bench<BmskChain, 4>`: `BMSK R0, R0, 0x5` native (sass_sm_100.txt).
how to use: `__popc(ballot & lanemask_lt)` gives a rank in about 23 cycles and needs no FLO. `elect.sync` or a redux of the lane id are cheaper leaders than `__ffs` on a dependent path. Never put `__fns` on a critical path, it costs more than two shuffle trees, a popc based binary search over the mask or a redux of `lane` among candidates is cheaper. `bmsk` is native and builds segment masks for logical warps in one instruction.
caveats: These are dependent latencies at one warp. In throughput code the FLO and POPC pipes overlap with other work and the counts, not the latencies, matter.
author: hw-researcher
"""))

    entries.append(("match_any", f"""### T?: match.any costs 50 cycles plus about 11 cycles per additional distinct value in the warp
applies to: W3 digit histograms, duplicate detection and tie handling with `__match_any_sync`, warp scope
measured: cycles per call with U distinct values among 32 lanes: U=1 {B('match_any 1 class (all equal)')}, 2 {B('match_any 2 classes')}, 4 {B('match_any 4 classes')}, 8 {B('match_any 8 classes')}, 16 {B('match_any 16 classes')}, 32 {B('match_any 32 classes (all distinct)')}. {prov}{miss}
evidence: `MATCH.ANY R6, R9` then `POPC R7, R6` per call, one SASS instruction whose latency grows with the number of distinct values (sass_sm_100.txt, `bench<MatchAnyClasses<32>, 4>`).
how to use: A digit histogram over a warp with `match.any` plus `popc` costs 50 to 390 cycles per digit pass depending on how many digit values are present, so it pays only for narrow digits (2 to 3 bits) or for near-uniform data. For 4-bit or wider digits with random keys expect the all-distinct cost. Ballot-per-bit counting (one `vote.ballot` per digit bit, 27 cycles each) is cheaper for digits of 4 bits or more.
caveats: Measured with a fixed class structure per chain, random keys at 32 lanes are close to the all-distinct case for any digit of 5 bits or more.
author: hw-researcher
"""))

    entries.append(("elect", f"""### T?: elect.sync is one SASS instruction whose latency hides behind a redux, no cheaper than ffs(activemask) but deterministic
applies to: the one-lane work after any warp reduction (shared memory write, atomic, counter), all scopes
measured: redux chain alone {B('redux.sync.max.u32')} cycles, redux chain with an `elect.sync` per step feeding the next input {B('redux chain + elect.sync per step')}, the same with `__ffs(__activemask()) - 1` {B('redux chain + ffs(activemask) leader per step')}. The increment of about 6 cycles is the adds that consume the leader id, not the election. {prov}{miss}
evidence: `ELECT P0, UR8, PT` (sass_sm_100.txt, `bench<ReduxElect, 4>`), the ffs variant is `VOTEU.ANY`, `UBREV`, `UFLO.U32.SH` on the uniform datapath (`bench<ReduxFfsLeader, 4>`).
how to use: `elect.sync` (inline PTX or `cuda::ptx::elect_sync`) is free when issued next to a reduction because it does not depend on the data, and it is deterministic (the same lane every time for the same mask), which `__activemask()` after divergence is not. Prefer it over `threadIdx.x % 32 == 0` only when the leader may be any active lane.
caveats: sm_90 and later only. The predicate output is what matters, the lane id output costs an extra move.
author: hw-researcher
"""))

    entries.append(("alu_latencies", f"""### T?: dependent ALU latencies on sm_100: IMAD and FFMA 4 cycles, IMNMX, FMNMX, PRMT, SIMD and bf16x2 max 6 cycles, the third input is free
applies to: every compare-exchange network (W1, W4, B6), threshold filters (W5, B4), twiddles, packing, all key widths
measured: cycles per dependent operation from pair chains (8 operations interleaved with 8 IMADs, minus the IMAD chain, divided by 8): IMAD base {B('alu base: 8 x imad u32 per step')} per 8 (4.0 per IMAD). max u32 {D('alu pair: 8 x (max u32 + imad)', 'alu base: 8 x imad u32 per step')}, `__vimax3_u32` {D('alu pair: 8 x (__vimax3_u32 + imad)', 'alu base: 8 x imad u32 per step')}, `__vimax3_s32` {D('alu pair: 8 x (__vimax3_s32 + imad)', 'alu base: 8 x imad u32 per step')}, `max(max(v, a), b)` {D('alu pair: 8 x (max(max(v,a),b) u32 + imad)', 'alu base: 8 x imad u32 per step')}, `max(v + a, b)` {D('alu pair: 8 x (max(v+a,b) u32 + imad)', 'alu base: 8 x imad u32 per step')}, prmt {D('alu pair: 8 x (prmt + imad)', 'alu base: 8 x imad u32 per step')}, `__vmaxu2` u16x2 {D('simd pair: 8 x (__vmaxu2 u16x2 + imad)', 'alu base: 8 x imad u32 per step')}, `__vmaxs2` {D('simd pair: 8 x (__vmaxs2 s16x2 + imad)', 'alu base: 8 x imad u32 per step')}, `__vimax3_u16x2` {D('simd pair: 8 x (__vimax3_u16x2 + imad)', 'alu base: 8 x imad u32 per step')}, `__hmax2` bf16x2 {D('simd pair: 8 x (__hmax2 bf16x2 + imad)', 'alu base: 8 x imad u32 per step')}, `max.bf16x2` {D('simd pair: 8 x (max.bf16x2 ptx + imad)', 'alu base: 8 x imad u32 per step')}, `max.f16x2` {D('simd pair: 8 x (max.f16x2 ptx + imad)', 'alu base: 8 x imad u32 per step')}. FFMA base {B('alu base: 8 x ffma f32 per step')} per 8 (4.0 per FFMA), fmaxf {D('alu pair: 8 x (fmaxf + ffma)', 'alu base: 8 x ffma f32 per step')}, `fmaxf(fmaxf(v, a), b)` {D('alu pair: 8 x (fmaxf(fmaxf(v,a),b) + ffma)', 'alu base: 8 x ffma f32 per step')}, `max.f32 d, a, b, c` {D('alu pair: 8 x (max.f32 3-input ptx + ffma)', 'alu base: 8 x ffma f32 per step')}. {prov}{miss}
evidence: SASS per operation (sass_sm_100.txt, `bench<AluPair<...>, 2>`): `VIMNMX.U32` for max, `VIMNMX3.U32 R9, R9, R0, R7, !PT` for `__vimax3_u32` and for `max(max(v,a),b)` (ptxas fuses the nested max), `VIADDMNMX.U32 R9, R9, 0x3, R0, !PT` for `max(v + 3, b)` (add fused into the min max unit), `FMNMX3` for both `fmaxf(fmaxf())` and the PTX three-input form, `VIMNMX.U16x2`, `VIMNMX3.U16x2`, `HMNMX2.BF16_V2` for bf16x2, and `VHMNMX.BF16_V2` when two `__hmax2` are nested. Chains of a single associative operation were reassociated by ptxas (16 maxes became 2 VIMNMX3 of hoisted constants), which is why the pair chain method is needed.
how to use: Count a three-input min or max as one 6-cycle instruction: write compare networks against two candidates as `max(max(a, b), c)` or `__vimax3`, and threshold clamps as `max(v + bias, t)`, ptxas emits the fused forms without inline PTX. Packed 16-bit keys cost the same per instruction as one 32-bit key, so a bf16 or u16x2 network does two keys per 6 cycles. Integer min max is 6 cycles against 4 for IMAD and FFMA, so an index-based select (`(k > t) ? i : j` as IMAD with a predicate) can be cheaper than a max on the index path.
caveats: One warp, dependent issue latency. Throughput and pipe mix are not measured here (knowledge base 2.1 says the pipe mix matters on B200). The `+imad` pair hides fusion opportunities between the measured op and its neighbors in real code.
author: hw-researcher
"""))

    entries.append(("u8x4", f"""### T?: u8x4 SIMD min and max are emulated on sm_100 with 10 instructions, widen 8-bit keys to 16 bits before any compare network
applies to: fp8 keys (e4m3, e5m2) in W1, W4, B5, B6 networks on B200, GB200 and B300
measured: `__vmaxu4` {D('simd pair: 8 x (__vmaxu4 u8x4 + imad)', 'alu base: 8 x imad u32 per step')} cycles per dependent operation against {D('simd pair: 8 x (__vmaxu2 u16x2 + imad)', 'alu base: 8 x imad u32 per step')} for the native `__vmaxu2` u16x2. {prov}{miss}
evidence: `bench<VMaxU4Chain, 1>` in sass_sm_100.txt: `LOP3.LUT` x4, `SHF.R.U32.HI`, `LOP3.LUT`, `IMAD.IADD`, `PRMT`, `LOP3.LUT` per `__vmaxu4` (10 instructions, no VIMNMX). PTX ISA 9.4: `max.u8x4` and `max.s8x4` exist only on sm_107f and sm_120f families, so ptxas has no native form on compute capability 10.0 or 10.3.
how to use: Unpack fp8 keys to u16 lanes (two per register) with `prmt` and run the u16x2 network, 4.7 times cheaper per instruction than the emulated u8x4 and only twice the instruction count of a hypothetical native 4-lane form. On Rubin (sm_107f) the native u8x4 form exists and this trade reverses, re-measure there.
caveats: Emulation sequences differ by operation (`__vminu4`, signed forms, `__vabsdiffu4` not measured). Comparison masks `__vcmpgtu4` are also emulated on sm_100 (not measured).
author: hw-researcher
"""))

    entries.append(("shared_atomics_return", f"""### T?: shared atomicMax with return costs 55 cycles uncontended and serializes at one lane per cycle on one address, u64 is a CAS loop
applies to: B2 atomics rounds, ticket counters, histogram bins in shared memory, block and warp scope
measured: `atomicMax` smem u32 with return: uncontended {B('atomicMax smem u32 with return, uncontended')} cycles, contention 2 {B('atomicMax smem u32 with return, contention 2')}, 8 {B('atomicMax smem u32 with return, contention 8')}, 32 {B('atomicMax smem u32 with return, contention 32')}, 256 {B('atomicMax smem u32 with return, contention 256')}, 8 warps lane 0 {B('atomicMax smem u32 with return, 8 warps lane0')}, 256 threads over 32 banked words {B('atomicMax smem u32 with return, 256thr/32banks')}. u64 with return: uncontended {B('atomicMax smem u64 with return, uncontended')}, contention 32 {B('atomicMax smem u64 with return, contention 32')}, 8 warps lane 0 {B('atomicMax smem u64 with return, 8 warps lane0')}. {prov}{miss}
evidence: u32 is `ATOMS.MAX R0, [UR5], R0` (`bench<AtomUncontended<unsigned>, 1>`). u64 has no native shared max: `LDS.64`, `ISETP.LE.U32.AND`, `ISETP.LE.U32.AND.EX`, `@P0 BRA` (skip when not larger), `ATOMS.CAST.SPIN.64 P0, [R3], R8, R10`, `@!P0 BRA` retry loop with `BSSY` and `BSYNC` (`bench<AtomUncontended<unsigned long long>, 1>`, sass_sm_100.txt). The July uncontended u32 figure of 115 cycles is 55 on CUDA 13.4.92 with the same source, the contended figures (92 to 129 at 2 to 32, 255 at 256) are reproduced within noise.
how to use: Use the returned value only when the algorithm needs it (tickets, min-index resolution) and then keep the contention low (one lane per warp, or banked words). A 256-way same-address burst with return costs 255 cycles, the fire-and-forget form of the next entry costs a tenth of that. Never use 64-bit shared max or min: the CAS loop costs 148 cycles uncontended and 576 at 32-way contention, and the loop branches per retry.
caveats: Contention numbers are self-dependent chains per thread (every thread waits for its own return), the realistic round pattern is in the block max entry.
author: hw-researcher
"""))

    entries.append(("shared_atomics_noret", f"""### T?: fire-and-forget same-address shared atomics issue for free once per warp per round but serialize at 32 to 59 cycles when a warp repeats them back to back
applies to: B2 rounds (one atomic per thread per round), shared memory histograms (many atomics per thread), block scope
measured: pair chains with 8 fire-and-forget `atomicMax` per step interleaved with 8 dependent IMADs, against the IMAD base ({B('alu base: 8 x imad u32 per step')} at 32 threads, {B('alu base: 8 x imad u32 per step (256 thr)')} at 256): 1 lane on the slot and 31 on private words {B('atomics pair: 8 x (atomicMax smem u32 fire-and-forget, 1 on the slot, 31 private + imad)')} (the atomics are hidden entirely), 32 lanes on one slot {B('atomics pair: 8 x (atomicMax smem u32 fire-and-forget, 32 on one slot + imad)')}, 256 threads on one slot {B('atomics pair: 8 x (atomicMax smem u32 fire-and-forget, 256 on one slot + imad)')}. Bursts of 8 independent atomics per step: 32 on one slot {B('atomics burst: 8 independent atomicMax smem u32 fire-and-forget, 32 on one slot + imad')}, 256 on one slot {B('atomics burst: 8 independent atomicMax smem u32 fire-and-forget, 256 on one slot + imad')}, 256 threads on 256 private words {B('atomics burst: 8 independent atomicMax smem u32 fire-and-forget, 256 private words + imad')}. One same-slot atomic per thread per round inside a block round: see the block max entry (74.5 cycles for the whole round including barrier and read back, 0 mismatches in the completion check). {prov}{miss}
evidence: `ATOMS.MAX RZ, [R6], R7` (destination RZ, no return) with the address selected by `SEL`, no branch (`bench<AtomNoRet<unsigned, 256, 256>, 2>`, sass_sm_100.txt). Completion: `bench<BlockMaxAtomic256Check<unsigned>, N>` compares the slot read after `BAR.SYNC` with a shuffle tree max and counted 0 mismatches in 64 repetitions at every chain length on every node measured.
how to use: The B2 pattern (every thread one `atomicMax` into `slot[round]`, barrier, read) is cheap because the slot address is warp-uniform and ptxas aggregates the warp's 32 same-address fire-and-forget atomics into one `CREDUX.MAX` plus one atomic by an elected lane (T26 and T27 by opt-B2b, visible in `bench<BlockMaxAtomic256<unsigned>, 4>` as `CREDUX.MAX`, `VOTEU.ANY`, `UFLO`, `@P0 ATOMS.MAX RZ`), so eight atomics reach the shared memory unit per round and the read after the barrier is correct. Keep that address provably uniform (indexed by a round counter, never by per-thread data): the burst rows above select the address per thread, ptxas cannot aggregate them, and the hardware then serializes same-address lanes at about one per cycle (32 to 59 cycles per warp instruction). A shared memory histogram with data-dependent bins is in that regime, so pre-combine equal keys in registers or spread hits over banked copies, 256 threads over 256 private words issue at about 9 cycles per warp instruction.
caveats: The first version of this entry attributed the cheap round to merging inside the shared memory atomic unit, the SASS shows it is ptxas aggregation at compile time, which only happens for a provably uniform address without a returned value. The u64 fire-and-forget form is the CAS loop of the previous entry and skips the atomic when the value is not larger, so its burst number ({B('atomics pair: 8 x (atomicMax smem u64 fire-and-forget, 256 on one slot + imad)')}) is not comparable.
author: hw-researcher
"""))

    entries.append(("block_max_rounds", f"""### T?: block-wide max round at 256 threads: atomic256 fire-and-forget 74 cycles, classic shuffle tree 290, hybrid with redux 149, u64 atomics 4500
applies to: B2 rounds, B5 merges, any block-wide max or argmax per round, 256 threads
measured: cycles per round: classic (warp shuffle tree, smem partials, barrier, 8-way shuffle) u32 {B('blockmax classic u32 (shfl+smem+bar+8-way shfl)')}, u64 {B('blockmax classic u64 (shfl+smem+bar+8-way shfl)')}, classic with redux for both phases {B('blockmax classic u32 (redux+smem+bar+redux)')}, atomic256 (all threads fire-and-forget atomicMax into slot[n], barrier, read) u32 {B('blockmax atomic256 u32')}, u64 {B('blockmax atomic256 u64')}, atomic256 with return (each thread waits) then barrier {B('blockmax atomic256 u32 with return (each thread waits) + bar')}, hybrid8 (shuffle tree, 8 lane-0 atomics) u32 {B('blockmax hybrid8 u32 (shfl+atomic)')}, u64 {B('blockmax hybrid8 u64 (shfl+atomic)')}, hybrid8 with redux {B('blockmax hybrid8 u32 (redux+atomic)')}. Components: `__syncthreads` at 128 {B('__syncthreads (128 thr)')}, 256 {B('__syncthreads (256 thr)')}, 512 threads {B('__syncthreads (512 thr)')}, STS + LDS round trip {B('STS+LDS round trip')}. {prov}{miss}
evidence: atomic256: `@P0 ATOMS.MAX RZ, [UR5+0x20], R6` / `BAR.SYNC.DEFER_BLOCKING 0x0` / `LDS R9, [UR5+0x20]` per round (sass_sm_100.txt, `bench<BlockMaxAtomic256<unsigned>, 4>`). The u64 round is the `ATOMS.CAST.SPIN.64` loop. Re-verifies the three July 2026 facts on CUDA 13.4.92: (1) the fire-and-forget u32 round costs about 74 cycles of which barrier (28) plus read back (32) are most, (2) the 64-bit packed word is dead at block scope (4500 per round, 368 even through shuffles), (3) pre-reducing with redux before the atomic makes the round slower (149 against 74) because ptxas already aggregates the uniform-address round into one `CREDUX.MAX` plus one elected atomic per warp (T26 and T27), so an explicit redux adds a second 22-cycle dependent step and the lane-0 atomic with a per-warp predicate is no longer aggregated the same way. New: the classic structure with redux for both phases (93) is the best non-atomic round, and the barrier scales 19, 28, 44 cycles at 128, 256, 512 threads.
how to use: For a block-wide max of 32-bit values use the atomic256 round. Reduce the value alone and resolve indices separately (ties are then resolved by a ticket or a second round), never pack value and index into 64 bits for shared atomics. When atomics are unwanted (deterministic index without a second pass), use redux plus smem plus redux (93), not shuffle trees (290).
caveats: Round costs at 1 block in the latency regime. At 128 threads the barrier is 9 cycles cheaper, at 512 threads 16 cycles dearer, the atomic count per round scales accordingly and was not measured at those block sizes.
author: hw-researcher
"""))

    entries.append(("transposes", f"""### T?: register transposes of 16-bit keys: movmatrix 45 cycles per 4 registers, stmatrix plus ldmatrix.trans 63, plain STS plus LDS 83
applies to: W1 and B5 packed 16-bit key arrangements (blocked to striped and back), any 8x8 block transpose of 16-bit values, warp scope
measured: 4 registers per lane transposed per call: 4 STS + syncwarp + 4 LDS {B('transpose 4 words: 4 STS + syncwarp + 4 LDS')} cycles, stmatrix.x4 + syncwarp + ldmatrix.x4.trans {B('transpose 4 words: stmatrix.x4 + syncwarp + ldmatrix.x4.trans')}, 4 x movmatrix.trans in registers {B('transpose 4 words: 4 x movmatrix.trans (registers only)')}, ldmatrix.x4 alone on an address-dependent chain {B('ldmatrix.x4 (address-dependent chain)')}. {prov}{miss}
evidence: `STSM.16.M88.4 [R0], R8` / `LDSM.16.MT88.4 R12, [R0]` (`bench<TransposeStsmLdsm, 1>`), `MOVM.16.MT88 R9, R9` x4 (`bench<TransposeMovm, 1>`), control `STS.128` + 4 x `LDS` (`bench<TransposeStsLds, 1>`), sass_sm_100.txt. PTX ISA 9.4: ldmatrix sm_75+, stmatrix sm_90+, movmatrix sm_75+ (all `.m8n8.b16`).
how to use: To change the arrangement of packed 16-bit keys (two per register) across a warp, `movmatrix.sync.aligned.m8n8.trans.b16` transposes one 8x8 block per register with no shared memory and no barrier, 45 percent cheaper than the shared memory round trip. Use it to go from a lane-major (blocked) to a column-major (striped) layout before a bitonic stage that needs the other orientation. When the data must pass through shared memory anyway, `stmatrix` plus `ldmatrix.trans` saves 20 cycles per 4 registers against scalar `STS` and `LDS`.
caveats: All three work on 16-bit elements in 8x8 blocks, so a 32-bit key needs two passes or a different trick. The ldmatrix forms with `.b8` or `.m16n16` need sm_100a or sm_100f. `movmatrix` needs inline PTX, no `cuda::ptx` wrapper was found in this checkout.
author: hw-researcher
"""))

    entries.append(("smem_fill", f"""### T?: a dependent shared memory fill from L2 costs 460 to 640 cycles whether through LDG plus STS or cp.async.bulk, the bulk path saves instructions and registers rather than time
applies to: B1 to B6 block tile loads, segment fills before block-level selection, 256 threads
measured: pointer-chased fills through a 4 MB footprint (L2 resident, beyond L1), cycles per fill: 4 KB with LDG.128 + STS + barrier {B('fill smem 4KB: LDG.128 + STS + bar (control)')}, 4 KB with cp.async.bulk + mbarrier {B('fill smem 4KB: cp.async.bulk + mbarrier')}, 16 KB with 4 x LDG.128 per thread + STS + barrier {B('fill smem 16KB: LDG.128 x4 + STS + bar (control)')}, 16 KB with cp.async.bulk + mbarrier {B('fill smem 16KB: cp.async.bulk + mbarrier')}. {prov}{miss}
evidence: bulk kernel `bench<FillBulk<4096>, 1>`: `FENCE.VIEW.ASYNC.S`, `SYNCS.ARRIVE.TRANS64 RZ, [UR10+0x1000], R0` (expect_tx), `UBLKCP.S.G [UR10], [UR6], UR5`, `SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR10+0x1000], R0` (sass_sm_100.txt). Control: `LDG.E.128.CONSTANT` x4 then `STS.128` x4 then `BAR.SYNC`.
how to use: Both paths are L2 latency bound (about 450 cycles base plus 10 to 40 cycles per KB), so pick by register and instruction budget, not latency: the bulk copy needs one issuing thread, no staging registers and leaves the other 255 threads free, which matters when the selection kernel is register-tier limited (knowledge base 2.5). For tiles of 16 KB and above the bulk copy is also slightly faster, below 4 KB the mbarrier handshake costs about 30 cycles more than plain loads.
caveats: The chase keeps one fill in flight, a pipelined kernel (next tile's copy issued before this tile's selection) hides most of this latency on either path. Global memory misses to HBM were not measured. The generic `LD.E.128` form (pointer not marked read-only) is what the first version emitted, use `__ldg` or `const __restrict__` to get `LDG.E.128.CONSTANT`.
author: hw-researcher
"""))

    entries.append(("methodology_sink", f"""### T?: a sink that reads %smid inside the timed region hides any chain shorter than about 80 cycles, hoist the read
applies to: every chain-slope microbenchmark and latency harness built on device_side_benchmark.cuh's smid guard
measured: with `if (smid() == -1) store` inside the timed region, a chain of 16 dependent IMADs measured 5 cycles over the 1-IMAD chain (82 against 87 cycles total, GB200, 2026-10-25, superseded run 08:44 and 09:01). With the smid read hoisted before the repetition loop the same chain measures {B('alu base: 8 x imad u32 per step')} cycles per 8 IMADs (4.0 per IMAD, R squared 1.000). {prov}
evidence: `S2R R2, SR_VIRTUALSMID` issued right after `CS2R R4, SR_CLOCKLO` in the superseded SASS, its result consumed by the `ISETP` that predicates the sink store after the chain, so the whole chain executed in the shadow of the S2R latency (sass dumps in state/results/hw-researcher/superseded/).
how to use: Read the special register once before the timed loop and keep only the compare in the timed region, or make the guard depend on the chain value (the July `v == 0xdeadbeef` form). Check the SASS of every latency kernel for `S2R` inside the clock bracket. The same applies to any other long-latency instruction with no dependency on the chain (constant loads, `LDC`), which the compiler may schedule into the timed region.
caveats: Only chains below the S2R latency are affected, the July block top-k numbers (hundreds of cycles per call) are not.
author: hw-researcher
"""))

    entries.append(("methodology_reassoc", f"""### T?: ptxas reassociates chains of one associative operation, measure ALU latencies with pair chains against an IMAD base
applies to: every ALU or SIMD latency measurement and every hand-unrolled compare network whose shape ptxas may rewrite
measured: a chain of 16 dependent `max(v, c_i)` compiled to 2 `VIMNMX3` over hoisted constants and measured 0.4 cycles per op, a chain of `v + c_i` folded to one `VIADD`, a chain of `prmt` composed into one, and an empty `asm volatile("" : "+r"(v))` between steps did not prevent it. Interleaving each operation with one dependent `IMAD` (v = v * 3 + c) keeps the chain and gives {B('alu base: 8 x imad u32 per step')} cycles per 8 IMADs for the base and {B('alu pair: 8 x (max u32 + imad)')} for max plus IMAD, R squared 1.000 on every pair row. {prov}
evidence: `bench<IMaxChain, 4>` in the superseded sass_sm_100.txt: `VIMNMX3.U32` x2 for 4 chained maxes. `bench<AluPair<OpIMax>, 2>` in the final dump: `VIMNMX.U32` x16 and `IMAD` x16 for 16 pairs.
how to use: When a compare network written as nested max and min does not perform as its instruction count predicts, read the SASS, ptxas may have fused pairs into three-input forms or hoisted constant subtrees (which is good) or reassociated a dependency chain you wanted. For microbenchmarks, break associativity with an operation ptxas does not reassociate with the one under test, and subtract its chain.
caveats: The IMAD pair hides fusion of the measured op with neighboring adds (`VIADDMNMX`), measure those shapes on their own as the `max(v + a, b)` row does.
author: hw-researcher
"""))

    entries.append(("gb200_equals_b200", f"""### T?: GB200 and B200 have identical per-SM cycle costs for every warp, ALU, atomic and barrier primitive measured, only the memory fills differ
applies to: all approaches, the choice of development node
measured: every row of harness/micro/intrinsics_micro.cu agrees between B200 ({nodes[0].node if nodes[0].ok else 'pending'}) and GB200 ({nodes[1].node if nodes[1].ok else 'pending'}) within 1 cycle except the shared memory fills (L2 latency differs between the two memory systems): redux {B('redux.sync.max.u32')}, shuffle tree {B('warp max 5-shfl tree u32')}, match 32 classes {B('match_any 32 classes (all distinct)')}, barrier 256 {B('__syncthreads (256 thr)')}, atomic256 round {B('blockmax atomic256 u32')}, 4 KB fill LDG {B('fill smem 4KB: LDG.128 + STS + bar (control)')}, 16 KB fill bulk {B('fill smem 16KB: cp.async.bulk + mbarrier')}. {prov}{miss}
evidence: the three JSON files per node under state/results/hw-researcher/<node>/, the run on each node repeats the sm_100 binary as a drift canary and the repeat agrees to the cycle.
how to use: Latency-regime work can be developed on whichever Blackwell node is free, SM cycle costs transfer. Throughput numbers and anything touching L2 or HBM (fills, spills, global traffic) stay per node and per arch as the brief requires.
caveats: Both are compute capability 10.0 with the same SM. B300 (sm_103) and Rubin (sm_107) are not covered by this statement.
author: hw-researcher
"""))

    def I(name_k, name_1, k):
        parts = []
        for n in nodes:
            if not n.ok:
                continue
            a, b_ = n.cyc(name_k), n.cyc(name_1)
            if a is None or b_ is None:
                continue
            parts.append(f"{n.label} {(a - b_) / (k - 1):.1f}")
        return ", ".join(parts) if parts else "not measured"

    entries.append(("indep_issue", f"""### T?: independent ballots, redux and shuffles pipeline at about 5 to 6 cycles each after the first, not 4
applies to: N1 bit-serial select, W3 ballot histograms, N4 tie passes, B5 and any code that issues several warp collectives on independent data
measured: one step issues K independent collectives on the same input and merges the results. Ballot: K=1 {B('ballot indep: 1 per step + merge')} cycles, K=4 {B('ballot indep: 4 independent per step + merge')}, K=8 {B('ballot indep: 8 independent per step + merge')}, K=16 {B('ballot indep: 16 independent per step + merge')}, so each further ballot in flight costs {I('ballot indep: 8 independent per step + merge', 'ballot indep: 1 per step + merge', 8)} cycles at K=8 and {I('ballot indep: 16 independent per step + merge', 'ballot indep: 1 per step + merge', 16)} at K=16 (including the shift and mask that build its predicate). Redux: K=1 {B('redux indep: 1 per step + merge')}, K=8 {B('redux indep: 8 independent per step + merge')}, {I('redux indep: 8 independent per step + merge', 'redux indep: 1 per step + merge', 8)} per further redux. Shuffle: K=1 {B('shfl indep: 1 per step + merge')}, K=8 {B('shfl indep: 8 independent per step + merge')}, {I('shfl indep: 8 independent per step + merge', 'shfl indep: 1 per step + merge', 8)} per further shuffle. Dependent chains for reference: ballot {B('ballot_sync')}, redux {B('redux.sync.max.u32')}, shuffle {B('shfl_xor (single)')}. {prov}{miss}
evidence: `bench<BallotIndep<8>, 2>`: 16 x `VOTE.ANY` with 16 `SHF.R.U32.HI` and 16 `LOP3.LUT` for the predicates, merged by `IADD3`. `bench<ReduxIndep<8>, 2>`: 16 x `CREDUX.MAX` plus 16 `IMAD.U32` moves. `bench<ShflIndep<8>, 2>`: 16 x `SHFL.BFLY` (sass_sm_100.txt).
how to use: Model a round that issues IPT independent ballots as 27 + 6 (IPT - 1) cycles, not 23 + 4 (IPT - 1): at IPT 8 that is 69 instead of 51, at IPT 16 about 124 instead of 83. Redux and shuffles pipeline at the same rate, so a warp phase that reduces several independent values (N6 bracket, B5 candidates) pays about 6 cycles per extra value. The merge adds (IADD3) are included in these numbers.
caveats: One warp. With several warps per scheduler the VOTE, REDUX and SHFL pipes are shared and the per-op cost in the throughput regime is higher. The predicate construction (2 ALU ops per ballot) is part of the measured increment, a ballot of a ready predicate may be about 2 cycles cheaper.
author: hw-researcher
"""))

    entries.append(("bf16x2_masks", f"""### T?: bf16x2 compares to per-half masks are one 6-cycle HSET2, integer u16x2 and u8x4 compare masks are emulated at 20 to 24 cycles
applies to: N5 two-row bf16x2 networks, W1 packed 16-bit keys, any threshold filter or candidate count over packed keys, all scopes
measured: cycles per dependent operation from pair chains against the IMAD base: `__hgt2_mask` on `__nv_bfloat162` (per-half `0xffff`) {D('simd pair: 8 x (__hgt2_mask bf16x2 per-half mask + imad)', 'alu base: 8 x imad u32 per step')}, PTX `set.gt.u32.bf16x2` {D('simd pair: 8 x (set.gt.u32.bf16x2 ptx + imad)', 'alu base: 8 x imad u32 per step')}, `min.bf16x2` {D('simd pair: 8 x (min.bf16x2 ptx + imad)', 'alu base: 8 x imad u32 per step')}, `max.bf16x2` {D('simd pair: 8 x (max.bf16x2 ptx + imad)', 'alu base: 8 x imad u32 per step')}, `__vcmpgtu2` (u16x2 per-half mask) {D('simd pair: 8 x (__vcmpgtu2 u16x2 per-half mask + imad)', 'alu base: 8 x imad u32 per step')}, `__vcmpgtu4` (u8x4 per-byte mask) {D('simd pair: 8 x (__vcmpgtu4 u8x4 per-byte mask + imad)', 'alu base: 8 x imad u32 per step')}, native `__vmaxu2` u16x2 for reference {D('simd pair: 8 x (__vmaxu2 u16x2 + imad)', 'alu base: 8 x imad u32 per step')}. {prov}{miss}
evidence: `HSET2.BF16_V2.GT.AND R.., R.., R.., PT` for both `__hgt2_mask` and the PTX `set.gt.u32.bf16x2` (`bench<AluPair<OpHgt2MaskBf16>, 2>`, `bench<AluPair<OpSetGtBf16x2Ptx>, 2>`), `HMNMX2.BF16_V2` for `min.bf16x2`. `__vcmpgtu2` is 10 instructions per call (`LOP3.LUT` x6, `SHF.R.U32.HI`, `IMAD.IADD`, `PRMT`), `__vcmpgtu4` 8 instructions (sass_sm_100.txt). PTX ISA 9.4: `set.CmpOp.{{u32,s32,bf16x2}}.bf16x2` requires sm_90, the integer `set.CmpOp.{{u16x2,s16x2,u8x4,s8x4}}` forms exist only on sm_107f.
how to use: Keep bf16 keys as bf16 inside packed networks and filters: compare with `__hgt2_mask` (or `__hge2_mask`, `__hlt2_mask`, `__heq2_mask`), select with `max.bf16x2` and `min.bf16x2`, and count candidates with `__popc(mask)` divided by 16, all native at 6 cycles. Do not twiddle bf16 to u16 pairs for compares: the integer per-half compare mask has no sm_100 instruction and costs 4 times a native compare. Twiddled u16x2 remains fine for pure min and max networks (native `VIMNMX.U16x2`). Semantics of the bf16 forms: `+0.0 > -0.0` for max and min, `__hgt2_mask` is false for NaN operands (`__hgtu2_mask` is the unordered form).
caveats: The predicate-returning `__vibmax_u16x2` (max plus per-half swap predicates) was not measured and may cover the swap decision for value-index pairs without a mask. Per-half mask to per-half select still needs a `LOP3` or `PRMT` based blend (one instruction) which is not in these numbers.
author: hw-researcher
"""))

    entries.append(("w2_packed_list", f"""### T?: the packed W2 round with a per-lane sorted list costs 26 cycles up to IPT 8, 37 at IPT 16 and 89 at IPT 32, predicated shifts only
applies to: W2 redux extraction with buffered per-lane lists, B5 warp phases, N4 truncated packing, warp and sub-warp scope, keys of 16 bits or less packed with the index
measured: cycles per round of a dependent chain (redux max of the heads, head compare, the winner shifts its sorted list by one with predicated selects, next redux reads the new head, the whole list kept live to the end): IPT 1 {B('W2 packed list round IPT 1: redux + head compare + predicated shift')}, IPT 4 {B('W2 packed list round IPT 4: redux + head compare + predicated shift')}, IPT 8 {B('W2 packed list round IPT 8: redux + head compare + predicated shift')}, IPT 16 {B('W2 packed list round IPT 16: redux + head compare + predicated shift')}, IPT 32 {B('W2 packed list round IPT 32: redux + head compare + predicated shift')}. The same round with the shift written as a branch on the winner: IPT 8 {B('W2 packed list round IPT 8: redux + head compare + branchy shift')}, IPT 32 {B('W2 packed list round IPT 32: redux + head compare + branchy shift')}. Registers of the N=16 kernel: 25 at IPT 1, 45 at IPT 8, 64 at IPT 32, no spills. For reference the bare redux is {B('redux.sync.max.u32')} and the unpacked round with ballot and ffs {B('W2 round unpacked: redux + ballot(v==max) + ffs + select')}. {prov}{miss}
evidence: `bench<W2PackedList<8, false>, 2>` (sass_sm_100.txt): per round `CREDUX.MAX`, `ISETP.NE.U32.AND`, then `SEL` of the head followed immediately by the next `CREDUX.MAX`, the remaining 7 `SEL` and a few `IMAD` moves scheduled behind it. At IPT 32 the round has 32 `SEL` plus 17 `IMAD` moves, more than the redux latency can hide. The branchy form compiles to 63 `IMAD.MOV` per round at IPT 32 plus `BSSY`, `BRA`, `BSYNC`, which is why it costs twice the predicated form. An earlier version of this row without the final fold of the list measured 25 cycles at every IPT because the compiler had dropped all shifts but the head's as dead code, so any reproduction must keep the list live.
how to use: Model the packed W2 round as 26 cycles for IPT up to 8, about 26 + 0.7 (IPT - 8) up to 16, and about 2.8 IPT at 32 where the round becomes issue bound. Write the shift as selects (`list[i] = win ? list[i + 1] : list[i]`), never as a branch on the winner. For B5, two warp phases of k rounds each at IPT 8 cost about 52 k cycles against 74 k for one block-wide atomic256 phase, before the merge. Choose IPT 8 (N 256 per warp) as the list depth, and split larger N over more lists or a prefilter rather than deepening the list.
caveats: One warp, unique packed words, every round has exactly one winner. Ties in the key half are resolved by the inverted index half, so the round is exact and prefer-smaller-index without extra work. The list fill and sort are outside the measured round. At IPT 32 the 64 registers would cap occupancy in a batched kernel (knowledge base 2.5), which the latency harness does not show.
author: hw-researcher
"""))

    entries.append(("collective_throughput", f"""### T?: redux, ballot and shuffle throughput limits are per scheduler, about 5.5 cycles per CREDUX, 7 per VOTE and 4 per SHFL, not per SM
applies to: B2 rounds with several redux per warp, B5 and N6 warp phases, W3 and N1 ballot histograms at 128 to 512 threads, any block that issues collectives from all warps at once
measured: the K-independent-per-step chain of T17 run by every warp of a 128, 256 and 512 thread block at once (barrier bracketed, the slowest warp counts). Redux, K=8 per step: 1 warp {B('redux indep: 8 independent per step + merge')} cycles, 4 warps (one per scheduler) {B('redux indep x4 warps: 8 independent per step + merge (128 thr)')}, 8 warps (two per scheduler) {B('redux indep x8 warps: 8 independent per step + merge (256 thr)')}, 16 warps (four per scheduler) {B('redux indep x16 warps: 8 independent per step + merge (512 thr)')}. Redux K=4: 4 warps {B('redux indep x4 warps: 4 independent per step + merge (128 thr)')}, 8 warps {B('redux indep x8 warps: 4 independent per step + merge (256 thr)')}, 16 warps {B('redux indep x16 warps: 4 independent per step + merge (512 thr)')}. Redux K=1 (dependent chain): 4, 8, 16 warps {B('redux indep x4 warps: 1 per step + merge (128 thr)')}, {B('redux indep x8 warps: 1 per step + merge (256 thr)')}, {B('redux indep x16 warps: 1 per step + merge (512 thr)')}. Ballot K=8: 1 warp {B('ballot indep: 8 independent per step + merge')}, 4 warps {B('ballot indep x4 warps: 8 independent per step + merge (128 thr)')}, 8 warps {B('ballot indep x8 warps: 8 independent per step + merge (256 thr)')}, 16 warps {B('ballot indep x16 warps: 8 independent per step + merge (512 thr)')}. Shuffle K=8: 1 warp {B('shfl indep: 8 independent per step + merge')}, 4 warps {B('shfl indep x4 warps: 8 independent per step + merge (128 thr)')}, 8 warps {B('shfl indep x8 warps: 8 independent per step + merge (256 thr)')}, 16 warps {B('shfl indep x16 warps: 8 independent per step + merge (512 thr)')}. {prov}{miss}
evidence: `bench<ReduxIndep<8, 256>, 2>`: 16 `CREDUX.MAX` plus 16 `IMAD.U32` moves per two steps per warp, the same code as the one-warp row (sass_sm_100.txt). Four warps on four schedulers cost exactly what one warp costs (62 against 62 cycles per 8 redux), so the unit is not shared across the SM at that rate. Two warps per scheduler cost 92 per 8 redux each (16 per scheduler per step, 5.75 cycles per CREDUX), four warps per scheduler 165 per 8 (32 per scheduler, 5.2 per CREDUX). The same pattern holds for VOTE (16 per scheduler in 112 cycles, 7.0 each) and for SHFL at a higher rate (32 per scheduler in 127 cycles, 4.0 each, about one SHFL per cycle per SM).
how to use: Budget collectives per scheduler, not per SM: with W warps per scheduler each issuing C redux per round, the throughput floor of the round is about 5.5 W C cycles (7 W C for ballots, 4 W C for shuffles), and it only binds when that exceeds the latency chain of the round. At 256 threads (two warps per scheduler) two redux per warp per round cost 23 cycles of scheduler time, which hides inside a 74 to 93 cycle block round. A dependent redux chain at 21 cycles per step is unaffected by the other warps up to 16 warps per block. When a block round is slower than its latency model, look first for a collective whose latency sits on the critical path (a CREDUX that cannot issue before the barrier and whose result feeds the STS), not for a throughput limit.
caveats: One block per SM in the latency harness. With several resident blocks the per-scheduler budget is shared by all their warps. The per-scheduler versus per-SM distinction rests on the 4-warp row equalling the 1-warp row, which assumes the usual warp-to-scheduler assignment (warp id modulo 4).
author: hw-researcher
"""))

    for i, (slug, text) in enumerate(entries, 1):
        path = os.path.join(a.out, f"T_{i:02d}_{slug}.md")
        with open(path, "w") as f:
            f.write(text)
    print(f"wrote {len(entries)} entries to {a.out}")


if __name__ == "__main__":
    main()
