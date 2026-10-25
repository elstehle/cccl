# W2: Redux based iterative extraction (warp and sub-warp)
status: queued (wave 1)
scope: warp, subwarp
origin: `reference/BLOCK_TOPK_RESULTS.md` section 1 (warp round 82 cycles, value-only keys with ballot and elect) and `hier_extract`, #9747 strategy 2
hypothesis: k rounds of `redux.sync` max. For bf16 pack key and inverted index into the 32-bit word so every word is unique and every round is an exact arg-max: the lane holding the maximum recognizes it by comparing its own head with the redux result, so there is no ballot, no ffs and no elect in the round. The round is the redux latency plus one compare and one select, about 30 cycles, against 74 for a block atomic round. Cost is linear in k and almost flat in N up to the point where the per-lane list shift exhausts issue bandwidth, so W2 owns small and medium k at any N a warp holds, which is the MoE router shape of 2026 (N 256 to 896 experts, k 6 to 22, see `ideas/WORKLOADS.md`).
cells it should win: warp and sub-warp, N 32 to 1024, k 4 to 32, bf16 first, fp32 with N4, prefer_smaller_index free in packed form. Sub-warp with 4 or 8 rows per hardware warp for the small routers (N 8 to 64).

cost model (measured 2026-10-25 by the hardware researcher on B200 and GB200, `TECHNIQUES.md` T2, T4, T17: redux.sync u32 22.1 dependent and 5.7 per further independent redux, dependent ballot 26.7, __ffs 40, __popc 23. July placeholders only where marked):
- Packed round: redux 22 + compare 4 + the head select 2 on the critical path, about 30. The remaining IPT minus 1 list shifts overlap the next round's redux latency while 2 IPT issue slots fit in 30 cycles, so round(IPT) about 30 for IPT at most 8 and about 14 + 2 IPT above (46 at IPT 16, 78 at IPT 32, 270 at IPT 128). This overlap is the assumption to verify first.
- Prologue: twiddle and pack 2 instructions per item, per-lane sorting network of IPT items (about 40 cycles at IPT 4, 60 at IPT 8, July placeholder), about 100 in total.
- Packed bf16 cost about 100 + k round(IPT). N 32 k 8: 340. N 256 k 8: 340. N 256 k 16: 580. N 256 k 32: 1060. N 512 k 16: 836. N 1024 k 16: 1350. N 1024 k 32: 2600. N 4096 k 16: 4420.
- fp32: `redux.sync` f32 exists only on sm_100a and sm_100f builds (22.1 cycles), not on the mandated sm_100, where twiddle plus redux u32 plus untwiddle costs 40 (T2). Value-only rounds need a winner election for ties (ballot on equality 27 plus a redux.min over lane ids 22, not ffs at 40), about 75 + 2 IPT, and prefer_smaller_index is not slot-major. N4 (truncated key plus index in one word, 20 key bits at N 4096) restores the 30-cycle packed round for fp32 with one tie pass of about 150, so fp32 costs bf16 plus 150.
- Against N1 (about 840 + 134 IPT for bf16): W2 wins up to k about 39 at N 256 and k about 46 at N 512. Against W5 (about 1400 + 10 IPT at k 16, 2210 + 10 IPT at k 17 to 64): W2 wins at k 16 up to N 1024 and at k 32 up to N 512, W5 takes k 64 from N 512 and k 16 from N 2048. Against W3 at 4-bit digits (1028 + 104 IPT): W2 wins at k 16 even at N 4096.

focus cells: 1. warp N 256 k 8 bf16 (DeepSeek V3, Kimi K2 shape), 2. warp N 512 k 16 bf16 and fp32 (Qwen3.5, Kimi K3 shape), 3. sub-warp LW 8 N 128 k 8 bf16 (4 rows per warp, gpt-oss-120b and Qwen3 routing), 4. warp N 512 k 32 bf16 (Nemotron 3 Ultra k 22), 5. warp N 1024 k 16 fp32 with N4 (Kimi K3 896 rounded up, PEER, NSA at 64K).
boundary cells: warp N 512 k 64 (expect W5 or N1 to win), warp N 2048 k 16 (IPT 64, list shift exceeds the redux latency, expect W5).

explore first: (1) packed bf16 round at IPT 1, 4, 8, 16, 32 as a slope over k, one kernel, to confirm the 30-cycle round and find where the list shift surfaces. (2) per-lane list structure above IPT 8 (shift list against a sorted run with a head index against a two-level list) since the shift is the only N-dependent term. (3) fp32 through N4, against value-only rounds with the redux.min election for prefer_smaller_index. (4) sub-warp via masked redux with 2, 4, 8 rows per warp. (5) two candidates per round (a second redux over the lanes whose head was not the winner, 22 more cycles for the second item) only if (1) confirms the 30-cycle round, the gain is then about 1.15x.

determinism and ordering: packed form inherent and sorted, prefer_smaller_index by inverting the index. Value-only fp32 form gives set membership for distinct values, ties need the election or N4.

main risk: the 30-cycle round assumes the list shift overlaps the next redux. If the compiler serializes the shift onto the critical path, round(IPT) is about 30 + 2 IPT and the k 32 cells at N 512 and above move to W5. Second risk: register pressure at IPT 32 and above (the whole row plus the list) in the throughput regime.

cheapest kill experiment: explore item 1 at IPT 8 and 32 (N 256 and 1024) with k 8 to 32. If round(8) exceeds 45 cycles or round(32) exceeds 100, W2 is confined to N at most 512 and k at most 16, and B5 (which rests on the same round) closes with it.

references: `reference/proto_topk_micro.cu`, `reference/proto_topk.cu` (`atomic_iter_h`), knowledge base 2.7, `TECHNIQUES.md` T2, T4, T17, `ideas/queue/N4_truncated_key_packing.md`, `ideas/queue/B5_hierarchical.md`
