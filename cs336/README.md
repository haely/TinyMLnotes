# ML Infra Notes → Stanford CS336 (Language Modeling from Scratch)

> cheat-sheet style · short phrases · read top → bottom · focus: ML infra (GPUs, FLOPs, memory, parallelism)

## Reading order

| # | file | what you get | ~time |
|---|------|--------------|-------|
| **Part A: prereqs** | | | |
| A1 | [prereqs/01_linear_algebra_calculus.md](prereqs/01_linear_algebra_calculus.md) | shapes, matmul, 2mkn FLOPs, gradients, chain rule | 30m |
| A2 | [prereqs/02_probability_stats.md](prereqs/02_probability_stats.md) | distributions, MLE, entropy, cross-entropy, KL, perplexity, error bars | 30m |
| A3 | [prereqs/03_machine_learning.md](prereqs/03_machine_learning.md) | loss, SGD, overfitting, softmax regression | 20m |
| A4 | [prereqs/04_deep_learning_pytorch.md](prereqs/04_deep_learning_pytorch.md) | MLPs, backprop, norms, PyTorch training loop, tensor memory | 40m |
| A5 | [prereqs/05_hardware_systems.md](prereqs/05_hardware_systems.md) | memory hierarchy, GPU/TPU anatomy, FLOPS, roofline, number formats, interconnect | 60m |
| **Part B: CS336** | | | |
| B0 | [cs336/00_overview.md](cs336/00_overview.md) | course map, assignments, big picture | 10m |
| B1 | [cs336/01_tokenization.md](cs336/01_tokenization.md) | UTF-8, BPE algorithm (A1) | 30m |
| B2 | [cs336/02_resource_accounting.md](cs336/02_resource_accounting.md) | 6ND, memory per param, MFU, back-of-envelope math | 40m |
| B3 | [cs336/03_transformer_architecture.md](cs336/03_transformer_architecture.md) | modern LLM block: RMSNorm, RoPE, SwiGLU, GQA, MoE | 45m |
| B4 | [cs336/04_training_optimization.md](cs336/04_training_optimization.md) | AdamW, LR schedules, mixed precision, stability | 30m |
| B5 | [cs336/05_gpus_kernels_triton.md](cs336/05_gpus_kernels_triton.md) | fusion, tiling, FlashAttention, Triton, profiling (A2) | 60m |
| B6 | [cs336/06_parallelism.md](cs336/06_parallelism.md) | collectives, DDP, ZeRO/FSDP, TP, PP, SP, EP (A2) | 60m |
| B7 | [cs336/07_scaling_laws.md](cs336/07_scaling_laws.md) | power laws, Chinchilla, IsoFLOP, μP (A3) | 30m |
| B8 | [cs336/08_inference.md](cs336/08_inference.md) | prefill vs decode, KV cache, batching, speculative, quantization | 40m |
| B9 | [cs336/09_data.md](cs336/09_data.md) | Common Crawl → filter → dedup → mix (A4) | 30m |
| B10 | [cs336/10_alignment_rl.md](cs336/10_alignment_rl.md) | SFT, RLHF, DPO, GRPO (A5) | 40m |
| B11 | [cs336/11_evaluation.md](cs336/11_evaluation.md) | perplexity, benchmarks, pass@k, contamination | 15m |
| ★ | [CHEATSHEET.md](CHEATSHEET.md) | 1-page formulas + numbers to memorize | review |

## Notation used everywhere

| symbol | meaning |
|---|---|
| `N` | # model parameters |
| `D` | # training tokens |
| `C` | training compute (FLOPs) |
| `B` | batch size (sequences) |
| `S` | sequence length (tokens) |
| `d` | model / hidden dim (`d_model`) |
| `d_ff` | MLP hidden dim |
| `L` | # layers |
| `h` | # attention heads; `d_h = d/h` head dim |
| `V` | vocab size |
| `η` | learning rate |
| `θ` | parameters |

## Legend
- **→** leads to / implies
- **⚠** common pitfall
- **★** memorize this
- `[A1]`…`[A5]` → which CS336 assignment uses it
