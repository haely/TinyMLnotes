# B2 · Resource Accounting (FLOPs, Memory, Time)  ★★ infra core

> do this math before every run · "how many GPUs for how long?"

---

## 1. Parameter count of a transformer ★

per layer (MHA, `d_ff = 4d`, or SwiGLU with `d_ff = 8/3·d`):

| part | params |
|---|---|
| attention Q, K, V, O | `4d²` |
| MLP (2 mats × 4d², or 3 mats × 8/3 d²) | `8d²` |
| norms | `~2d` (ignore) |
| **per layer** | **≈ 12d²** |

- total `N ≈ 12·L·d²` + embedding `V·d` (+ LM head `V·d` if untied)
- check: GPT-3 → L=96, d=12288 → 12·96·12288² ≈ **174B** ✓

## 2. FLOPs ★★

### Forward: ≈ 2 FLOPs per param per token
- each weight used in one multiply-add per token → `2N` FLOPs/token

### Backward: ≈ 2× forward
- grad wrt activations (1 matmul) + grad wrt weights (1 matmul) → `4N` FLOPs/token

### Training ★
```
   C ≈ 6 · N · D        (FLOPs = 6 × params × tokens)
        2ND forward + 4ND backward
```
- attention extra: ~`12·L·S·d` FLOPs/token (train) → small unless S ≳ d
- inference: `≈ 2N` FLOPs per generated token

### Example
- 70B model, 15T tokens → `6 · 70e9 · 15e12 = 6.3e24 FLOPs`

## 3. Time ★

```
 time = C / (n_GPUs × peak_FLOP/s × MFU)
```
- **MFU** (model FLOPs utilization) = useful model FLOP/s achieved ÷ hardware peak
  - good: 40–60% · typical small-scale: 20–40%
- **HFU** (hardware FLOPs util) counts recomputation too → ≥ MFU
- example: 6.3e24 on 1024 H100 @ 40% → 6.3e24 / (1024 · 989e12 · 0.4) ≈ 1.6e7 s ≈ **180 days**
- tokens/s per GPU ≈ `peak · MFU / (6N)` → 7B on H100 @ 40%: `989e12·0.4 / 4.2e10 ≈ 9.4k tok/s`

## 4. Memory during training ★★

### Per-parameter bytes (mixed precision + AdamW)

| item | dtype | bytes/param |
|---|---|---|
| weights (compute copy) | bf16 | 2 |
| gradients | bf16 | 2 |
| master weights | fp32 | 4 |
| Adam 1st moment `m` | fp32 | 4 |
| Adam 2nd moment `v` | fp32 | 4 |
| **total** | | **16** |

- 7B → `16 × 7e9 = 112 GB` > one 80 GB H100 → need sharding (ZeRO/FSDP, [B6](06_parallelism.md))
- pure fp32 training: 4+4+4+4 = 16 too (no separate master copy) but slower compute

### Activations (saved for backward)
- approx per layer, bf16, no recompute: `S·B·d·(34 + 5·h·S/d)` bytes (Korthikanti et al.)
- `5hS/d` term = attention score matrix `[B, h, S, S]` → removed by FlashAttention
- grows with batch × seq length → often the real memory limit
- example: L=32, d=4096, S=4096, B=1, FlashAttn → `34·4096·4096·32 ≈ 18 GB` per sequence

### Memory reducers

| technique | saves | cost |
|---|---|---|
| **activation checkpointing** (recompute) | store only layer inputs, recompute rest in backward | ~+33% compute (one extra forward) |
| mixed precision | 2 B vs 4 B activations | numerics care |
| gradient accumulation | smaller micro-batches, sum grads over k steps | more steps per update |
| ZeRO / FSDP sharding | split params/grads/opt state across GPUs | communication |
| offloading | move opt state to CPU RAM | PCIe-bound, slow |
| 8-bit optimizers | Adam states in int8 | slight quality risk |

## 5. Memory at inference
- weights: `2N` bytes (bf16) → 7B ≈ 14 GB; `0.5N` bytes in int4
- + KV cache (see [B8](08_inference.md)) → often bigger than weights at long context / big batch

## 6. Back-of-envelope checklist ★
1. N from `12Ld² + Vd`
2. training C = `6ND`
3. memory: `16N` + activations → how many GPUs just to fit?
4. time: `C / (GPUs · peak · MFU)`
5. inference: `2N` FLOPs/token, `2N` bytes weights → decode speed ≈ `mem_BW / bytes_of_weights`
6. sanity check against known runs (Llama 3 405B: ~3.8e25 FLOPs, ~16k H100s, ~2 months)

## 7. PyTorch accounting tips
- `sum(p.numel() for p in model.parameters())` → N
- `torch.cuda.max_memory_allocated()` → peak memory
- `torch.cuda.memory._record_memory_history()` → memory snapshot viewer
- `x.element_size() * x.numel()` → bytes of a tensor
