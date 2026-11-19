# B8 · Inference  ★★ infra core

> training = one-time cost · inference = forever cost · decode is **memory-bound**

---

## 1. Metrics

| metric | meaning |
|---|---|
| TTFT | time to first token (≈ prefill latency) |
| TPOT / ITL | time per output token / inter-token latency |
| latency | total time for one request |
| throughput | tokens/s across all users |
| goodput | throughput that meets latency SLOs |
| cost | $/M tokens |

## 2. Prefill vs decode ★★

```
 prompt: "The cat sat on"            generate: "the" "mat" "." …
 ┌──────────── PREFILL ───────────┐  ┌─── DECODE (1 token / step) ───┐
 all S prompt tokens in parallel    each step: 1 new token per sequence
 big matmuls [S × d]·[d × d]        mat-vec  [1 × d]·[d × d]
 → compute-bound                     → memory-bound (read all weights + KV each step)
 builds KV cache                     appends to KV cache
```

### Why decode is memory-bound
- per step: read all weights (2N bytes) to do 2N·B FLOPs → **arithmetic intensity ≈ B** (batch size)
- H100 ridge ≈ 295 → need batch ≳ ~300 to be compute-bound (ignoring KV reads, which don't amortize)
- batch-1 upper bound speed: `mem_BW / weight_bytes` → 7B bf16: `3.35 TB/s / 14 GB ≈ 240 tok/s`

## 3. KV cache ★★
- attention at step t needs K, V of **all previous tokens** → cache instead of recompute
```
 KV bytes = 2 (K and V) × L × n_kv_heads × d_h × S × B × bytes_per_elem
```

| model | L | n_kv × d_h | per token (bf16) | 4k context |
|---|---|---|---|---|
| Llama 2 7B (MHA) | 32 | 32 × 128 | **512 KB** | 2 GB / seq |
| Llama 3 8B (GQA) | 32 | 8 × 128 | **128 KB** | 0.5 GB / seq |
| Llama 3 70B (GQA) | 80 | 8 × 128 | 320 KB | 1.3 GB / seq |

- KV cache limits **batch size** → limits throughput
- reduce it: GQA/MQA, MLA (low-rank latent), sliding-window/local layers, KV quantization (fp8/int4), shorter context, SSM/linear-attention hybrids

## 4. Batching + memory management

| technique | idea |
|---|---|
| static batching | wait for batch, run until all done → ⚠ GPU idles on finished seqs |
| **continuous batching** | add/remove sequences every decode step (iteration-level scheduling) |
| **PagedAttention** (vLLM) | KV cache in fixed-size blocks like OS virtual-memory pages → no fragmentation, share prefixes |
| prefix caching | reuse KV for shared system prompt / few-shot prefix |
| chunked prefill | split long prompts, mix with decodes → smooth latency |
| disaggregated serving | prefill GPUs and decode GPUs separate (different bottlenecks) |

```
 PagedAttention: logical KV → physical blocks
 seq A: [blk 3][blk 7][blk 1]
 seq B: [blk 3][blk 9]         ← shares prefix block 3
```

## 5. Speculative decoding ★
```
 draft (small, fast)  : proposes k tokens  t₁ t₂ t₃ t₄
 target (big)         : ONE forward pass over all k (parallel, like prefill)
 verify               : accept prefix via rejection sampling; resample first rejected
 → output distribution EXACTLY = target model's
 → speedup ≈ accepted tokens per target pass (2–3× typical)
```
- why it works: decode is memory-bound → verifying k tokens costs ≈ same as 1
- variants: Medusa (extra heads), EAGLE (feature-level draft), n-gram/prompt lookup, MTP heads

## 6. Making the model cheaper

| method | what | trade-off |
|---|---|---|
| **quantization** | weights int8/int4/fp8 (GPTQ, AWQ); W8A8; fp8 KV | fewer bytes → faster decode; small quality loss |
| distillation | train small student on big teacher outputs | needs training |
| pruning | remove weights/heads/layers | quality loss, needs care |
| MoE | fewer active params per token | memory still holds all experts |
| shorter outputs | fewer decode steps | task-dependent |

- weight-only int4 → 4× less bytes → ~up to 4× faster batch-1 decode

## 7. Parallelism at inference
- TP across GPUs in node → split weights → lower latency (each GPU reads fewer bytes)
- PP → more memory, not lower latency
- EP for MoE serving
- replicas (DP) → throughput

## 8. Engines
- **vLLM** (PagedAttention, continuous batching) → used in A5 for RL rollouts
- SGLang (RadixAttention prefix cache), TensorRT-LLM, llama.cpp (CPU/edge, quantized)

## 9. Napkin math
- FLOPs/token ≈ 2N · memory/token step ≈ 2N bytes (bf16) + KV
- max batch ≈ (HBM − weights) / (KV per seq)
  - Llama 3 8B on H100: (80 − 16) GB / 0.5 GB ≈ 128 seqs @ 4k context
