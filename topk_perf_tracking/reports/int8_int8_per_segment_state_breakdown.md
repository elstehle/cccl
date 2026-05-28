# `per_segment_state_t`: what is actually live in registers (int8/int8 filter, peak)

You asked a fair question: my earlier "24-34 registers for `per_segment_state_t`"
was a static struct-size upper bound, not a measurement. Here is a re-derivation
based on the SASS evidence.

## How I derived it

1. Sat at the peak-liveness instruction (`0x2740`, where nvdisasm reports
   **60 live general regs**) and asked: "which register numbers are alive,
   and where was each one first defined?"
2. Built a register-flow table from the `--print-life-ranges` SASS dump:
   for every R<n>, the offset of its current open live range's first
   DEF and its last USE. DEFs and USEs are recovered from the SASS
   operand list:
   - Stores (`STG`/`STS`/`STL`/`RED`/`ATOM*`): all general regs are USEs;
     atomics get DEF for the result slot.
   - Predicate-defining ops (`ISETP`, `FSETP`, `PLOP3`, `VOTE`, ...): all
     general regs are USEs (the destination is a predicate, not Rn).
   - Width-encoded loads (`LD.E.64`, `LDC.64`, ...): the destination
     covers `R<n>` and `R<n+1>` (and so on for 128-bit).
   - Otherwise the first general-reg operand is the DEF.
3. Cross-referenced each first-DEF offset against the
   `--print-line-info` SASS dump to recover `(file, line)`.
4. Read the source at each line to identify the underlying field.

The tooling is `trace_alive_regs.py` in this directory.

## Coverage caveat (read first)

The analysis identifies **40 of the 60** general registers nvdisasm
reports as live at the peak. The remaining ~20 likely come from:

- `R1` (stack pointer) — never re-defined after offset 0, so it has only
  one DEF event in my linear scan; my "last-use" tracking marks the live
  range as ending at the first instruction even though it's alive forever.
- 64-bit pair partners not always tagged. For example, `IMAD.WIDE.U32`
  writes a 64-bit result (`R<n>:R<n+1>`); my parser keys off the dotted
  opcode suffix and the operand `.NN` token, so subtle SASS encodings
  may slip past it.
- Live registers carried *into* this offset from a different basic
  block whose DEFs land in pieces I close prematurely.

The 40 I do track are individually well-evidenced (each has a concrete
SASS instruction and source line behind it). So treat this as a
**lower bound** on what's in `per_segment_state_t`, and refer to the
ptxas number (64 allocated, 60 live at peak) as ground truth.

## Confirmed `per_segment_state_t` fields in registers at peak

Source lines reference `cub/agent/agent_batched_topk.cuh` unless noted.

| field | source line | regs | size in struct | how I know |
|---|---:|---|---:|---|
| `slab_base` | 1454 | **R3** | 4 B | `LD.E R3, desc[UR6][R24.64]` is the global load at `s.slab_base = d_large_segments_tile_offsets[queue_idx]` |
| `segment_counter` (ptr) | 1468 | **R36:R37** | 8 B | `LDC.64 R36, c[0x0][0x400]` loads the kernel-arg base, then `IMAD.WIDE.U32 R36, R16, 0x280, R36` adds `queue_idx * 0x280` (offset by sizeof(counter)). |
| `segment_histogram` (ptr) | 1469 | **R38:R39** | 8 B | Same shape: `LDC.64 R38, c[0x0][0x408]` + `IMAD.WIDE.U32 R38, R21, 0x4, R38`. The `0x4` stride says the array element is 4 bytes (uint32 histogram entries). |
| `in_key_buf` (ptr) | 1491 | **R6 + partner** | 8 B | `IMAD.X R6, RZ, RZ, R9, P1` is the high-half of a 64-bit pointer addition. Companion low-half register is one I'm not catching cleanly (likely R7-ish but reused for scratch later — see "uncertainty"). |
| `out_key_buf` (ptr) | 1492 | **R9, R10** | 8 B | `SEL R9, R10, RZ, P3` and `SEL R10, R22, RZ, P3` — the conditional `s.will_buffer ? ... : nullptr` selection of the 64-bit pointer halves. |
| `in_val_buf` (ptr) | 1496 | **R32** (+ partner) | 8 B | `SEL R32, R26, RZ, P1` — the `s.load_from_candidates_buffer ? ... : nullptr` selection. |
| `out_val_buf` (ptr) | 1497 | **R34** (+ partner) | 8 B | `SEL R34, R24, RZ, P3` — the `s.will_buffer ? ... : nullptr` selection. |
| `segment_tiles_input` | 1503 | **R12** | 4 B | `VIMNMX.U32 R12, PT, PT, R20, R9, PT` — the `min(a, 1+(a-1)/b)` shape of `cuda::ceil_div`. |

That's **at least 12 registers** for `per_segment_state_t` (rounding up by
4 for the partner-halves I'm probably missing, ~14-16 total).

This is meaningfully less than my earlier "24-34" estimate — the compiler
*does* eliminate or fold a number of struct fields. Specifically, the
following fields do not show up as long-lived registers at this peak:

- `empty`, `early_stop`, `will_buffer`, `load_from_candidates_buffer`:
  promoted to predicates (`P0`-`P3`), each one consumes a predicate slot
  rather than a register. Nvdisasm reports 3 live predicates at peak
  (separate column).
- `pass`, `current_k`, `current_len`, `input_length_actual`,
  `num_full_tiles`, `partial_items`, `queue_segment_end`: these get
  loaded out of the constant-memory kernel args / counter struct on
  demand (the LDC at 0x0d70, 0x0d80, 0x16f0, 0x1700 sequences in the
  prologue) and either consumed immediately or used to compute
  longer-lived derived values (e.g., `segment_tiles_input` survives).
- `d_keys_in`, `d_keys_out`, `d_values_in`, `d_values_out`: dereferenced
  at the BlockLoad entry; the inner pointers are kept alive *via* the
  base address used by the LDG sequence (R28:R29 at the load site),
  not as separate copies.

So the bool flags are in **predicates**, not regs (good). The "always
materialized but unused" value pointers are not as bad as I feared on
the *peak* instruction, but the four pointer fields above (in_key_buf,
out_key_buf, in_val_buf, out_val_buf) all stay alive even though for
this `pass=0`/`first iteration` execution two of them are likely to be
the `nullptr` branch of their respective `SEL`. The compiler still
materializes both halves of the pointer because the predicate is a
runtime value.

## Other persistent state at peak (non-per_segment_state_t)

| bucket | regs | source line | what it is |
|---|---|---:|---|
| `keys[0..15]` from `BlockLoad` | R22, R23, R24, R25, R26, R27, R44, R46, R48, R50, R52, R54, R56, R58, R60 (15 alive; R45 was just consumed) | `block_load.cuh:69` | each int8 key loaded into its own 32-bit register. |
| `first_chunk_start` / global tile count | R0 | `agent_batched_topk.cuh:1872` | `if (first_chunk_start >= *d_total_large_tiles)` plus the running cursor. |
| `chunk_end` (per-stretch loop) | R17 | 1887 | `(chunk_start + chunk_size_v < *d_total_large_tiles) ? ... : *d_total_large_tiles` |
| `local_tile_start` | R21 | 1946 | `chunk_cursor - state.slab_base` |
| `full_tiles_in_stretch` | R19 | 1951 | `max(local_full_end - local_tile_start, 0)` |
| `d_total_large_tiles` (resolved ptr/value) | R30:R31 | `kernel_batched_topk.cuh:639` | the `*large_segments_count_it` value (or its derived sentinel pointer). |
| `threadIdx.x` + derivatives | R20, R43 | 1442 | `S2R R20, SR_TID.X` and `IMAD.SHL.U32 R43, R40, 0x10, RZ` (per-thread tile-output index). |
| `start_bit` (classify) | R11 | `agent_topk_common.cuh:134` | clamped to `max(0, ...)` via VIMNMX |
| derived classify mask | R15, R13 | `dispatch_topk_common.cuh:78`, `agent_batched_topk.cuh:1442` | tied to the LOP3.LUT mask shapes seen at the peak |

## Peak-iteration scratch

| reg | what |
|---|---|
| R47 | `*kth_key_bits` for the current iter (`LDG.E.U8 R47, [R36+0x10]` — the counter struct offset matches `kth_key_bits.bits`). |
| R28 | extracted bits of the current key (`(R13 & 0x7f) | R27`-shape mask combine). |
| R29 | masked-to-8b `*kth_key_bits` for the comparison. |

## Bottom line

Concrete tally at peak (offset 0x2740), with confidence:

| bucket | regs | confidence |
|---|---:|---|
| `keys[0..15]` from `BlockLoad` | **15** | **high** — every reg traced to `block_load.cuh:69` and you can see the 16 consecutive LDG.E.U8 in the SASS |
| `per_segment_state_t` pointers / counters | **~12-16** | **medium-high** — the 7 pointer fields above are concretely identified; the partner halves of three of those pointers are uncertain |
| outer CTA-walk loop state | **~7** | **high** — line attribution maps each to the source loop variables in `run<TilesPerChunk>` |
| `start_bit` / mask classifier constants | **~3** | **medium** — multiple short-lived computations land at lines 78/133/134, exact role of each reg fuzzy |
| current-iter scratch | **~3-5** | **high** — visible directly in the SASS window around the peak |
| stack ptr + 64-bit-pair partners + transients my tracer missed | **~14-20** | **low** — these are the 20-register coverage gap; assume some mix of `R1`, paired high-halves, and short-lived computations |

So the corrected answer is:

- **per_segment_state_t in registers at peak: roughly 12-16, not 24-34.**
  Still meaningful, still the largest *deliberate* pool of persistent
  state — but smaller than the static struct size suggested because the
  bools are predicates and several of the value-channel and counter
  fields are folded.
- **keys[] is comparably big** (15 regs) and shares the spotlight for
  the int8 case in particular.

For the proposal ranking, this argues:

1. **P3 (packing the int8 `keys[]`) and P1 (moving `per_segment_state_t`
   pointers to smem) are comparable in expected win** (12-16 vs 15
   registers each). On int8 specifically, P3 may be the cleaner change
   because it touches a single load algorithm choice; P1 is more
   invasive.
2. The four `SEL`-from-nullptr pointer fields (`in_key_buf`,
   `out_key_buf`, `in_val_buf`, `out_val_buf`) are responsible for ~6-8
   of those persistent regs. Two of them (`in_val_buf`, `out_val_buf`)
   are predicated on `s.will_buffer` / `s.load_from_candidates_buffer`
   — if either condition is statically known false for some segment
   pass (e.g., pass 0 always has `load_from_candidates_buffer = false`),
   those pointer fields can be elided at that pass via templating, and
   we don't even need P1's smem dance for those.
