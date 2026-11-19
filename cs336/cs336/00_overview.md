# B0 · CS336 Overview — Language Modeling from Scratch (Stanford)

> build an LLM end-to-end yourself · every component · with an **efficiency** lens

---

## 1. Philosophy
- "understanding via building" → no high-level libs in assignments (write own Linear, attention, AdamW, tokenizer)
- **efficiency is the theme**: given fixed compute + data, get best model
- resources to account for: **compute (FLOPs)**, **memory (bytes)**, **bandwidth (bytes/s)**, **data (tokens)**, **wall-clock**
- three kinds of knowledge:
  - mechanics → how things work (transformer, parallelism)
  - mindset → squeeze the hardware, scale seriously
  - intuitions → which data/arch choices help (partly transfers from small scale, partly not)

## 2. The full pipeline

```
 raw web ──► [ DATA ] ──► [ TOKENIZER ] ──► [ MODEL ARCH ] ──► [ TRAINING ] ──► base model
 (CC HTML)   filter,       BPE, bytes→ids    transformer        AdamW, mixed      │
             dedup, mix                      RoPE, SwiGLU…      precision, GPUs  │
                                                                 parallelism       ▼
                         [ SCALING LAWS ] decide N, D for budget C        [ ALIGNMENT ]
                                                                           SFT → RL (GRPO/DPO)
                                                                                 │
                                    [ EVALUATION ] ◄──── [ INFERENCE ] ◄─────────┘
                                    perplexity,          KV cache, batching,
                                    benchmarks           quantization
```

## 3. Lecture topics → these notes

| unit | topics | notes file |
|---|---|---|
| basics | tokenization; PyTorch + resource accounting | B1, B2 |
| architecture | transformer variants, hyperparams, MoE | B3 |
| training | optimizers, LR schedules, stability | B4 |
| systems | GPUs, kernels, Triton, FlashAttention | B5 |
| systems | parallelism (data, tensor, pipeline, sequence, expert) | B6 |
| scaling | scaling laws, Chinchilla, μP | B7 |
| inference | KV cache, batching, speculative decoding, quantization | B8 |
| data | Common Crawl, filtering, dedup, mixing | B9 |
| alignment | SFT, RLHF, DPO, RL w/ verifiable rewards (GRPO) | B10 |
| evaluation | perplexity, benchmarks, contamination | B11 |

## 4. Assignments ★

| # | name | you implement | infra relevance |
|---|---|---|---|
| A1 | **Basics** | BPE tokenizer · transformer LM (Linear, Embedding, RMSNorm, RoPE, SwiGLU, attention) · cross-entropy · AdamW · LR schedule · training loop · decoding; train on TinyStories + OpenWebText | memory/FLOPs accounting |
| A2 | **Systems** | benchmarking + profiling · mixed precision · activation checkpointing · **FlashAttention-2 in Triton** · **DDP** (naive → bucketed + overlapped) · **optimizer-state sharding** (ZeRO-1) | ★ core infra |
| A3 | **Scaling** | query a training API under FLOPs budget · fit IsoFLOP / scaling laws · predict optimal model size | compute planning |
| A4 | **Data** | HTML → text · language ID · quality / toxicity / PII filters · exact + MinHash dedup · build a training set | data pipelines at scale |
| A5 | **Alignment** | SFT + expert iteration + **GRPO** on math reasoning (Qwen2.5-Math 1.5B) · vLLM rollouts · (optional) DPO / safety | RL infra, inference engines |

## 5. Mindset rules ★
- always do back-of-envelope math **before** running (FLOPs, memory, time)
- measure, don't guess → profile
- small-scale experiments → extrapolate with scaling laws
- data quality often beats architecture tweaks
- what wins at small scale may not at large → be careful

## 6. Resources
- course site: `stanford-cs336.github.io` (lectures as executable Python "trace" files)
- assignments: GitHub `stanford-cs336/assignment1-basics` … `assignment5-alignment`
- companion reading: "How to Scale Your Model" (JAX/TPU scaling book), Horace He "Making Deep Learning Go Brrrr", Megatron-LM, ZeRO, FlashAttention, Chinchilla papers
