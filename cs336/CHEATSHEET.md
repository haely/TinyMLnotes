# ★ ML Infra / CS336 One-Page Cheatsheet

## Counting
| what | formula |
|---|---|
| matmul (m×k)(k×n) | **2mkn** FLOPs |
| transformer params | **N ≈ 12·L·d²** (+ V·d embed, + V·d head if untied) |
| forward FLOPs/token | **2N** |
| training FLOPs | **C ≈ 6ND** (2 fwd + 4 bwd) |
| attention extra (train) | ~12·L·S·d per token (matters when S ≳ d) |
| training time | C / (#GPU × peak × MFU) |
| MFU | achieved model FLOP/s ÷ peak; good = 40–60% |

## Memory
| what | bytes |
|---|---|
| train, AdamW mixed precision | **16 / param** (2 w + 2 g + 4 master + 4 m + 4 v) |
| inference weights | 2/param bf16 · 1 int8 · 0.5 int4 |
| activations / layer (bf16) | ≈ S·B·d·(34 + 5hS/d); FlashAttn drops 2nd term |
| KV cache | **2 · L · n_kv · d_h · S · B · bytes** |
| ZeRO-1 / 2 / 3 per GPU | 4Ψ+12Ψ/n · 2Ψ+14Ψ/n · 16Ψ/n |

## Hardware (approx, dense bf16)
| chip | TFLOP/s | HBM | BW | ridge (FLOP/B) |
|---|---|---|---|---|
| A100 | 312 | 80 GB | 2.0 TB/s | ~156 |
| H100 | 989 | 80 GB | 3.35 TB/s | **~295** |
| B200 | ~2,200 | ~180 GB | ~8 TB/s | ~280 |

- hierarchy: registers → SMEM (~228 KB/SM) → L2 (~50 MB) → HBM → NVLink (~450 GB/s/dir) → PCIe (~64 GB/s) → IB (~50 GB/s)
- arithmetic intensity = FLOPs / bytes; < ridge → memory-bound
- elementwise ≈ 0.25 · mat-vec ≈ 1 · n×n matmul ≈ n/3 · decode ≈ batch size

## Number formats
fp32 (1/8/23, 4 B) · fp16 (1/5/10, max 65504, loss scaling) · **bf16** (1/8/7, fp32 range) · fp8 E4M3 / E5M2 · int8 · int4

## Collectives (ring, size M, n GPUs)
- all-reduce ≈ 2M bytes/GPU = reduce-scatter + all-gather
- DP: all-reduce grads · FSDP: AG params fwd+bwd, RS grads (1.5×) · TP: 2 all-reduce/layer fwd (intra-node) · PP: P2P activations, bubble (p−1)/(m+p−1) · EP: 2 all-to-all

## Scaling laws
- `L(N,D) = E + A/N^α + B/D^β` (α≈0.34, β≈0.28)
- Chinchilla: **D ≈ 20N**, N ≈ √(C/120)
- e.g. C=1e21 → N≈2.9B, D≈58B
- overtrain small models for cheap inference · μP → LR transfers across width

## Architecture defaults (Llama-style)
pre-norm · RMSNorm · RoPE · SwiGLU (d_ff ≈ 8/3 d) · GQA · no bias · no dropout · AdamW(0.9, 0.95, wd 0.1) · warmup + cosine/WSD · grad clip 1.0 · z-loss / QK-norm for stability

## Attention
`softmax(QKᵀ/√d_h + mask)·V` · FlashAttention = tiles + online softmax (`m, l` running max/sum) + recompute in bwd → O(S) memory

## Inference
prefill = compute-bound · decode = memory-bound · batch-1 tok/s ≤ BW / weight bytes (7B bf16 H100 ≈ 240) · continuous batching · PagedAttention · speculative decoding (exact) · quantization

## Info theory
CE = −log q(target) · PPL = e^CE · KL(p‖q) = H(p,q) − H(p) · init loss ≈ ln V · SE = √(p(1−p)/n)

## Alignment losses
- RM: `−log σ(r_w − r_l)`
- PPO: `min(ρA, clip(ρ,1±ε)A)` − β·KL
- DPO: `−log σ(β[Δlogratio_w − Δlogratio_l])`
- GRPO: A = (r − mean_group)/std_group, PPO-clip, no critic

## Data
CC WARC → extract → lang ID → Gopher rules → quality clf (fastText) → toxicity/PII → exact + MinHash dedup (P = 1−(1−s^r)^b) → decontaminate → mix
