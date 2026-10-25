# Workloads behind the cells (check of PLAN.md section 3.1, 2026-10-25)

Author: algo-researcher. Figures come from the model configs on Hugging Face (fetched 2026-10-25, raw `config.json` unless noted), from the papers, and from vendor docs. Every number is cited inline. Secondary sources are marked as such. The machine-readable form is `named_cells.json`.

## 1. MoE token-choice routing: per token over expert scores

| Model | Routed experts N | Active k | Shared | Groups | Vocab | Source |
|---|---|---|---|---|---|---|
| Mixtral 8x7B | 8 | 2 | 0 | 1 | 32000 | config `mistralai/Mixtral-8x7B-v0.1` |
| DBRX | 16 | 4 | 0 | 1 | GPT-4 tokenizer | Databricks blog `databricks.com/blog/introducing-dbrx-new-state-art-open-llm` |
| Llama 4 Scout | 16 | 1 | 1 | 1 | 202048 | config via `unsloth/Llama-4-Scout-17B-16E-Instruct` (Meta repo is gated) |
| Llama 4 Maverick | 128 | 1 | 1 | 1 | 202048 | Meta blog `ai.meta.com/blog/llama-4-multimodal-intelligence` ("one of the 128 routed experts" plus a shared expert) |
| gpt-oss-20b | 32 | 4 | 0 | 1 | 201088 | config `openai/gpt-oss-20b` |
| gpt-oss-120b | 128 | 4 | 0 | 1 | 201088 | config `openai/gpt-oss-120b` |
| Qwen3-235B-A22B and 30B-A3B | 128 | 8 | 0 | 1 | 151936 | config `Qwen/Qwen3-235B-A22B` |
| GLM-4.5-Air | 128 | 8 | 1 | 1 | 151552 | config `zai-org/GLM-4.5-Air` |
| GLM-4.5 and 4.6 | 160 | 8 | 1 | 1 | 151552 | config `zai-org/GLM-4.5` |
| Step-3 | 48 | 3 | shared dim 5120 | 1 | 128815 | config `stepfun-ai/step3` |
| ERNIE-4.5-300B-A47B | 64 | 8 | 0 | 1 | 103424 | config `baidu/ERNIE-4.5-300B-A47B-PT` |
| DeepSeek-V3, R1, V3.2-Exp | 256 | 8 | 1 | 8 groups of 32, keep 4 | 129280 | config `deepseek-ai/DeepSeek-V3`, `DeepSeek-V3.2-Exp` (`n_group` 8, `topk_group` 4, `topk_method` noaux_tc, sigmoid) |
| Ling-1T | 256 | 8 | 1 | 8 groups, keep 4 | 157184 | config `inclusionAI/Ling-1T` |
| MiniMax-M2 | 256 | 8 | 0 | 1 | 200064 | config `MiniMaxAI/MiniMax-M2` |
| DeepSeek-V4-Flash (2026) | 256 | 6 | 1 | none (`n_group` absent) | 129280 | config `deepseek-ai/DeepSeek-V4-Flash` |
| DeepSeek-V4-Pro (2026) | 384 | 6 | 1 | none | 129280 | config `deepseek-ai/DeepSeek-V4-Pro`, paper arXiv 2606.19348 (node limit removed, hash routing in early layers) |
| Kimi K2 | 384 | 8 | 1 | 1 | 163840 | config `moonshotai/Kimi-K2-Instruct` |
| MiMo-V2.6-Pro (2026) | 384 | 8 | 0 | 1 | 152576 | config `XiaomiMiMo/MiMo-V2.6-Pro-RL` |
| Qwen3-Next-80B-A3B | 512 | 10 | 1 | 1 | 151936 | config `Qwen/Qwen3-Next-80B-A3B-Instruct` |
| Qwen3.5-397B-A17B (2026) | 512 | 10 | 1 | 1 | 248320 | config `Qwen/Qwen3.5-397B-A17B` |
| Nemotron 3 Ultra (2026) | 512 | 22 | 1 | 1 | 131072 | config `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` |
| LongCat-Flash | 512 + 256 zero-computation | 12 | 0 | 1 | 131072 | config `meituan-longcat/LongCat-Flash-Chat` (`zero_expert_num` 256, so the router selects over 768) |
| Kimi K3 (2026) | 896 | 16 | 2 | 1 | 163840 | config `moonshotai/Kimi-K3` (`num_expert_group` 1) |

What this changes against PLAN.md 3.1:
- The plan's "N 8 to 384, k 2 to 8" is too small on both axes. The 2026 flagships cluster at N 256 to 896 with k 6 to 22. Keep N 8 to 64 for the small routers, add N 512 and 1024 (Kimi K3's 896 pads to 1024) at warp scope with k 8, 16 and 32 (Nemotron's 22 rounds to 32). These are one-row-per-warp cells with 8 to 32 items per lane, not sub-warp cells.
- Llama 4 is top-1 plus a shared expert, an argmax, out of the top-k grid. Mixtral and Grok style top-2 of 8 is the k 2 column of the final evaluation (decision 2), and DBRX and gpt-oss-20b (top 4 of 16 and 32) are the only sub-warp k 4 cells left among current models.
- Grouped routing is receding. It is used by DeepSeek V3, R1, V3.2 and Ling-1T (8 groups of 32 experts, group score is the sum of the two largest sigmoid affinities in the group, keep 4 groups, then top 8 of the remaining 128, `arXiv 2412.19437` section 2.1.2 describes the node-limited form with M 4 nodes and K_r over M equal to 2). DeepSeek V4, Kimi K2 and K3, GLM-4.5, Qwen3.5 and Nemotron 3 have one group. The decomposition is three ordinary cells: sub-warp top 2 of 32 (times 8 groups, one per register of a warp), top 4 of 8 group scores, warp top 8 of 128 with the losing groups masked (see `rejected/R11`).
- Routing scores are usually computed in fp32 (sigmoid or softmax of the gate output) even in bf16 models, so router cells should run both key types with fp32 as the primary and bf16 as the case where the gate output is taken directly.
- Decode at small batch is latency bound (one or a few rows), prefill is throughput bound (thousands of rows per launch with identical N and k), so router cells run both regimes. The throughput regime is where N5 (two rows per lane) applies.

## 2. Future fine-grained MoE and product-key experts

- PEER (`arXiv 2407.04153`): 1024 squared experts, two sub-key sets of 1024 each, h 8 heads, k 16 per head. Retrieval is a top 16 over each 1024 sub-key set, then a top 16 over the 256 candidate sums. Cells: warp N 1024 k 16 (twice) and warp N 256 k 16 per head, 8 heads per token, fp32 scores. The plan's description is confirmed.
- The "512 to 4096 routed experts with 8 to 32 active" row is confirmed at its low end by Qwen3.5, Nemotron 3 Ultra (512, 22), LongCat (768, 12) and Kimi K3 (896, 16). Nothing public above 1024 routed experts as of this check besides PEER style product keys.

## 3. Expert-choice routing and capacity routing: k over N is c over E, not N over 8 to N over 2

- Expert choice (`arXiv 2202.09368`, confirmed through the ar5iv rendering): capacity k = n c over e with n tokens in the batch, capacity factor c in {0.5, 1, 2} (2 is the main setting) and e in {16, 32, 64, 128} experts. So k over N = c over e, which is 1 over 64 (c 1, e 64) to 1 over 8 (c 2, e 16). For a 4096-token group: e 64 c 2 gives k 128, e 128 c 1 gives k 32. Selection cells, k-flat for B1, no sort needed. The score matrix is produced token-major and consumed expert-major, the transpose is the agent's problem (`rejected/R7`).
- Mixture-of-depths (`arXiv 2404.02258`): the best setting is 12.5 percent capacity on every other block, capacities 12.5 to 95 percent were evaluated and smaller was better down to 12.5. k over N = 1 over 8 at the sequence length (N 4096 to 8192 tokens, k 512 to 1024). Block cells, selection regime.
- Correction to PLAN.md 3.1: the "k from N over 8 to N over 2" regime does not come from capacity routing. It comes from (a) the merge step of chunked selections, k over N = 1 over M for M chunk results, M 2 to 8 (section 5 and 6), (b) H2O style eviction at 20 percent of the cache (`arXiv 2306.14048` abstract: 20 percent heavy hitters, balance of recent and heavy-hitter tokens), (c) SnapKV with a 4096 budget on a 16K prompt (1 over 4), and (d) the tiny routers (top 2 of 8, top 4 of 16, 1 over 4). B6 is re-scoped accordingly.

## 4. Top-k and top-p sampling, beam search, draft trees

- Vocabulary sizes: Llama 3 and 3.1 128256 (config via `unsloth/Llama-3.1-8B`, Meta blog says 128K), Qwen3 151936, Qwen3.5 248320, GLM-4.5 151552, DeepSeek 129280, Kimi 163840, gpt-oss 201088, Llama 4 202048, MiniMax-M2 200064, Gemma 3 262208 (`unsloth/gemma-3-27b-it` config, tech report `arXiv 2503.19786` says 262k entries), Mistral v0.1 32000 (legacy). The plan's "32K to 256K" is right at the top end, the 32K end is legacy, 128K to 262K is the current range.
- Sampling k: Qwen3 model card recommends TopK 20 for both thinking and non-thinking modes, Transformers' GenerationConfig default is 50, llama.cpp's default is 40 (not re-verified here). Top-p is implemented by sorting or by a top-k prefilter with a cumulative sum, so k 32 to 128 over the vocabulary is the sampling cell, chunked per block (4 to 8 chunks of 32768, or 32 chunks of 4096 to 8192) with a merge of M k candidates.
- Beam search: Transformers keeps 2 times num_beams candidates per step over num_beams times vocab scores, num_beams 4 to 8 in practice, so k 8 to 16 over 512K to 2M scores, chunked.
- Draft trees: EAGLE-2 (`arXiv 2406.16858` appendix) uses 60, 50 and 48 total draft tokens for 7B, 13B and 70B targets, depth 6, top 10 nodes per expansion. Per expansion step that is 10 rows of top 10 over the vocabulary (chunked), a top 10 of 100 children, and a final rerank top 60 of about 600 nodes. SGLang's defaults are speculative-eagle-topk 4 for Llama and Grok style models and 1 for most others, num-steps 5 or 3, num-draft-tokens 8 (SGLang docs `docs.sglang.ai/advanced_features/speculative_decoding.html`, a secondary snapshot). The plan's "k 4 to 16 per node" holds, add the rerank cell (warp N 1024 k 64) and the children cell (warp N 128 k 16).

## 5. Sparse attention indexers

- DSA (DeepSeek Sparse Attention): V3.2-Exp selects `index_topk` 2048 tokens per query with 64 indexer heads of dimension 128 (config `deepseek-ai/DeepSeek-V3.2-Exp`), contexts 32K to 128K. DeepSeek-V4 (`arXiv 2606.19348`, configs) moves to compressed KV (compression factor m along the sequence) with `index_topk` 1024 for Pro and 512 for Flash, 1M context, so the indexer scores N = context over m entries (about 262144 at 1M with m 4). TensorRT-LLM's `DeepSeekV4SparseAttentionConfig` defaults `index_topk` to 512 and its GVR top-k supports index_topk in {512, 1024, 2048} on SM100 and above with compress_ratio in {1, 4} (`nvidia.github.io/TensorRT-LLM/latest/llm-api/reference/DeepSeekV4SparseAttentionConfig.html`). Correction to PLAN.md 3.1: k is 512, 1024 or 2048 and N per query is 32K to 256K entries. Indexer scores are fp32 (FP8 GEMM with fp32 accumulate). A deterministic variant exists in production (`gvr_deterministic_topk`), so DSA cells run both requirements.
- NSA (`arXiv 2502.11089` section 4.1): compression block 32, stride 16, selection block 64, 16 selected blocks including the fixed 1 initial and 2 local blocks (so 13 chosen by top-k, implementations run top 16 and mask), sliding window 512, pretraining at 8k, adaptation at 32k, evaluation up to 64k. N = context over 64 selection blocks: 512 at 32K, 1024 at 64K, 2048 at 128K, k 16. Warp cells, fp32 scores.
- MoBA (`arXiv 2502.13189`): block 512 with top 3 blocks in the main experiments (N 16 at 8K, 64 at 32K), block 4096 with top 12 at 1M (N 256). Sub-warp and warp cells with router-like shapes.

## 6. KV cache selection and eviction

- Quest (`arXiv 2406.10774`): pages of 16 KV pairs, per-page criticality from channel min and max, token budgets 256 to 4096. N = context over 16 pages (2048 at 32K, 8192 at 128K), k = budget over 16 (16 to 256). Block cells, fp32 scores.
- SnapKV (`arXiv 2404.14469`): observation window 32 (16 and 64 in some runs), max pooling kernel 7 (5 and 13), budgets 1024, 2048, 4096 tokens over prompts of 4K to 32K, top-k per head over pooled attention sums. N 4096 to 32768, k 1024 to 4096, so k over N 1 over 32 to 1 over 4.
- H2O (`arXiv 2306.14048`): 20 percent heavy hitters of the cache, so k = N over 5 over the cache positions, N 2048 to 16384.

## 7. Cells that exceed one warp or block, and how they decompose

Grid limits: sub-warp N at most 128, warp N at most 4096 (128 items per lane), block N at most 32768 (128 items per thread at 256 threads).

| Named workload | N per call | Decomposition | Primitive cells |
|---|---|---|---|
| Sampling over the vocabulary, k 32 to 128 | 128256 to 262208 | M 4 to 8 chunks of 32768 per block, each top k, then one merge of M k | block N 32768 k 64, then warp or block N 256 to 1024 k 64 (M 4 to 8) |
| Same with small chunks | 128K to 262K | 32 to 64 chunks of 4096 | block N 4096 k 64, merge N 2048 to 4096 k 64 |
| DSA V3.2 at 128K, k 2048 | 131072 | 4 chunks of 32768 | block N 32768 k 2048 (k over N 1 over 16), merge N 8192 k 2048 (1 over 4) |
| DSA V4 Pro at 1M compressed by 4, k 1024 | 262144 | 8 chunks of 32768 | block N 32768 k 1024, merge N 8192 k 1024 (1 over 8) |
| DSA V4 Flash, k 512 | 262144 | 8 chunks of 32768 | block N 32768 k 512, merge N 4096 k 512 |
| Beam search candidates | 512K to 2M | chunks of 32768 | block N 32768 k 16, merge N 256 to 1024 k 16 |
| EAGLE-2 expansion rows | 10 rows of 128K | chunks of 32768 | block N 32768 k 16, merge N 64 k 16 |
| Kimi K3 routing | 896 | one warp, pad to 1024 | warp N 1024 k 16 |

The merge cell has k over N = 1 over M, the regime where B1's epilogue and B6's sort compete, and if the chunk results are sorted (W1, N2, B6 outputs) a k-truncated multiway merge (W4 explore 3, B6 explore 2) is O(k log M) against two B1 passes over M k items. The streaming alternative (N3) keeps a running exact threshold across chunks of the same row and never materializes M k candidates, it is listed for the streaming contract only.

## 8. A methodological flag: bf16 keys are tie-heavy by construction

bf16 stores 7 mantissa bits, so each binade has 128 representable values. Router logits and attention scores live in two to four binades, so N 256 to 896 bf16 keys are drawn from a few hundred distinct values and ties at the k-th key are the norm, not a corner case. Consequences for the study: (a) the harness's bf16 "random" input should be drawn from a realistic logit distribution (for example normal with unit variance, rounded to bf16) and the generator should report the tie multiplicity at rank k, random bit patterns understate ties, (b) the gap between not_guaranteed and prefer_smaller_index is material for bf16 and the correctness oracle must exercise large tie classes at the pivot, (c) radix and bit-serial approaches cannot count on early exit for bf16 (W3, N1, B1 cards say so), and (d) the packed key-plus-inverted-index word (W2, B2, N4) resolves ties by index at zero cost, which is a real advantage on this data, not a theoretical one. fp32 scores from a sigmoid or softmax are not tie-heavy but cluster within a few binades, which matters for N4's truncated keys and for radix pass 1 (N6).

## 9. Summary of corrections to PLAN.md section 3.1

1. Routing: N 8 to 896 (not 384), k 1 to 22 (not 2 to 8), the 2026 flagships at N 256 to 896 and k 6 to 16 are warp cells with 8 to 32 items per lane. Llama 4 is k 1 and out. Grouped routing survives only in DeepSeek V3 lineage and Ling-1T.
2. Capacity routing is a selection regime (k over N = c over E, 1 over 64 to 1 over 8, mixture-of-depths 1 over 8). The large k over N regime comes from chunk merges, H2O and SnapKV.
3. DSA top-k is 2048 (V3.2), 1024 (V4 Pro) or 512 (V4 Flash) over 32K to 256K indexer entries, fp32 scores, with a deterministic production variant. NSA is N 512 to 2048 with k 16 (13 free), fp32.
4. Vocabularies are 128K to 262K today, 32K is legacy. Sampling k is 20 to 50 by default, draft trees are top 10 per node with a top 60 rerank, SGLang defaults to top 4.
5. bf16 inputs must be generated from a logit-like distribution and tie density reported, see section 8.
