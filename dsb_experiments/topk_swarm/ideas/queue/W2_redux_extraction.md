# W2: Redux based iterative extraction (warp and sub-warp)
status: queued (wave 1)
scope: warp, subwarp
origin: `reference/BLOCK_TOPK_RESULTS.md` section 1 (warp round 82 cycles) and `hier_extract`, #9747 strategy 2
hypothesis: k rounds of `redux.sync` max plus ballot plus elect. For bf16 pack key and index into the 32-bit word so every round is an exact arg-max with no tie machinery (the `atomic_iter_h` trick at warp scope). Cost is linear in k and almost flat in N, so it owns small k at any N a warp holds, which is exactly the MoE router shape of 2026 (N 256 to 896 experts, k 6 to 16, see `ideas/WORKLOADS.md`).
cells it should win: warp and sub-warp, N 32 to 512, k 4 to 16, bf16 first, prefer_smaller_index free in packed form. Sub-warp with 4 or 8 rows per hardware warp for the small routers (N 8 to 64).

cost model (placeholders July 2026 B200, one design point, re-measure):
- Round: redux 22 + ballot 23 + ffs and elect (ballot plus ffs plus elect measured 71 in total) + local list update of about 2 IPT predicated selects, so round(IPT) about 82 + 2 IPT. Prologue: per-lane sorting network of IPT items, about 40 cycles at IPT 4 and 60 at IPT 8, plus twiddle and pack for bf16 (2 instructions per item).
- Packed bf16 (key in the high half, inverted index in the low half, N at most 65536): cost about 100 + k times (82 + 2 IPT). No epilogue, the k extracted words are the sorted answer, indices by one mask.
- fp32: either `redux.sync` f32 max on sm_100 (hardware researcher confirms availability and cost) with a ballot on equality per round for the index, same round cost, or u32 twiddled value rounds plus one index pass of k ballots at the end (about 25 k). prefer_smaller_index for fp32 needs the ballot over all IPT slots in slot-major order (IPT ballots, pipelined, about 23 + 8 IPT per round). N4 (truncated key plus index in one word) removes the fp32 epilogue for N at most 4096.
- Example: N 256 k 8 bf16 about 100 + 8 times 98 = 884. N 256 k 16 about 1670. N 32 k 8 about 760.

focus cells: 1. warp N 256 k 8 bf16 (DeepSeek V3, Kimi K2 shape), 2. warp N 512 k 16 bf16 and fp32 (Qwen3.5, Kimi K3 shape), 3. sub-warp LW 8 N 128 k 8 bf16 (4 rows per warp, gpt-oss-120b and Qwen3 routing), 4. warp N 128 k 4 fp32, 5. warp N 1024 k 16 bf16 (Kimi K3 896 rounded up).
boundary cells: warp N 256 k 32 (expect N1 or W5 to win), warp N 2048 k 16 (IPT 64, list update dominates, expect W3 or W5).

explore first: (1) packed bf16 arg-max rounds, measure cycles per round against IPT 1, 4, 8, 16. (2) fp32: `redux.sync` f32 if available, else twiddled u32 rounds plus index pass, then N4. (3) sub-warp via masked redux with 2, 4, 8 rows per warp. (4) two candidates per round: a second redux on the lanes that did not win, 45 more cycles for the second item, expected 1.3x fewer cycles per extracted item if the dependency chain allows. (5) where k stops paying against W1, N1 and W5 (the k 16 to 32 band at N 128 to 512).

determinism and ordering: packed form inherent and sorted, prefer_smaller_index by inverting the index. Value-only fp32 form gives set membership for distinct values, ties need the index pass or N4.

main risk: the round is a dependent chain (redux feeds ballot feeds elect feeds the list update) and the 82 cycles were measured at IPT 1 with a trivial list. If the list update at IPT 8 to 32 costs more than 2 IPT, W2 loses its N range to W5 and N1 above N 256. Second risk: bf16 router logits are tie-heavy (bf16 has 8 bits of precision), the packed word resolves ties by index so correctness holds, but the not_guaranteed variant gains nothing from ties.

cheapest kill experiment: packed bf16 round cost at IPT 1, 8 and 32 (N 32, 256, 1024) with k 8, one kernel, slope over k. If round(32) exceeds 160 cycles, W2 is confined to N at most 512.

references: `reference/proto_topk_micro.cu`, `reference/proto_topk.cu` (`atomic_iter_h`), knowledge base 2.7, `ideas/queue/N4_truncated_key_packing.md`
