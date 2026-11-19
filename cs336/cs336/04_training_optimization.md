# B4 · Training + Optimization  `[A1, A2]`

> loss → optimizer → schedule → precision → stability

---

## 1. Objective
- next-token cross-entropy, averaged over all B·S positions:
  `L = −(1/BS) Σ log softmax(logits)[target]`
- implement stably: `logsumexp(logits) − logits[target]` (subtract max first) ★
- initial loss ≈ `ln V` → sanity check

## 2. Optimizers progression ★

| optimizer | update | state/param | note |
|---|---|---|---|
| SGD | `θ −= η·g` | 0 | noisy, LR-sensitive |
| momentum | `m = βm + g; θ −= η·m` | 1 | smooths direction |
| RMSProp | `v = βv + (1−β)g²; θ −= η·g/√v` | 1 | per-param adaptive step |
| Adam | momentum + RMSProp + bias correction | 2 | default |
| **AdamW** ★ | Adam + **decoupled** weight decay | 2 | LLM standard |

### AdamW (A1 implements this)
```
 m ← β₁m + (1−β₁)g                 1st moment (mean of grads)
 v ← β₂v + (1−β₂)g²                2nd moment (mean of squared grads)
 η_t = η · √(1−β₂ᵗ) / (1−β₁ᵗ)      bias correction (m, v start at 0)
 θ ← θ − η_t · m / (√v + ε)        adaptive step
 θ ← θ − η · λ · θ                 decoupled weight decay (not via gradient)
```
- typical LLM: `β₁=0.9, β₂=0.95, ε=1e-8, λ=0.1`
- memory: m, v in fp32 → +8 bytes/param (see [B2](02_resource_accounting.md))
- newer: **Muon** (orthogonalized momentum for 2-D weight matrices via Newton–Schulz) → used in some recent large runs; Adafactor / Lion / Sophia → lower memory or faster

## 3. Learning-rate schedules ★

```
 η
 │    ___________                    cosine:  warmup → cosine decay → η_min
 │   /           ‾‾‾‾‾‾‾‾\___
 │  /                       ‾‾‾\___
 │ /                               ‾‾─── η_min
 └──────────────────────────────────── step
   warmup

 │    __________________________
 │   /                          \      WSD (warmup–stable–decay):
 │  /                            \     constant LR, short final decay (~10–20%)
 │ /                              \_   → can branch/continue from stable checkpoints
 └──────────────────────────────────── step
```

### Cosine w/ warmup (A1 formula)
- `t < T_w`: `η = (t/T_w)·η_max`
- `T_w ≤ t ≤ T_c`: `η = η_min + ½(1 + cos(π·(t−T_w)/(T_c−T_w)))·(η_max − η_min)`
- `t > T_c`: `η = η_min`
- warmup → avoids early instability (Adam's v estimate is bad at first)

## 4. Batch size
- **global batch** = micro-batch × grad-accum steps × data-parallel GPUs
- often measured in tokens: e.g. 4M tokens/step
- **critical batch size**: beyond it, doubling batch ≠ halving steps → wasted compute; grows as loss drops
- larger batch → can use larger LR (roughly), better GPU efficiency, fewer optimizer steps
- **gradient accumulation**: run k micro-batches, sum grads, step once → big batch w/o big memory

## 5. Mixed precision training ★

```
 fp32 master weights ──cast──► bf16 weights ──► forward (bf16 matmuls, fp32 accum)
        ▲                                             │
        │ optimizer step (fp32)                       ▼
   fp32 grads ◄──cast── bf16 grads ◄────────── backward
```
- `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` → matmuls in bf16, sensitive ops (softmax, norms, loss) in fp32
- bf16: no loss scaling needed (fp32 range)
- fp16: needs **loss scaling** (multiply loss by S so small grads don't underflow, unscale before step; dynamic: shrink S on inf)
- fp8 (H100+): per-tensor/block scaling factors; E4M3 fwd, E5M2 grads → ~2× matmul speed

## 6. Initialization
- small normal (std ~0.02, or `~1/√d`), truncated at ±3σ
- scale residual output projections by `1/√(2L)` → stream variance doesn't grow with depth
- embeddings ~N(0, 1) or small; norms γ = 1

## 7. Stability ★ (loss spikes = expensive at scale)

| symptom | fix |
|---|---|
| loss spikes / divergence | lower LR, longer warmup, grad clipping (norm 1.0) |
| attention logits blow up | QK-norm, z-loss |
| output softmax drift | z-loss `10⁻⁴·(log Z)²` |
| bf16 issues in accumulations | keep reductions fp32 |
| bad data batch | skip batch, rewind to checkpoint |

- monitor: loss, grad norm, LR, per-layer activation/weight norms, tokens/s, MFU

## 8. Training loop infra (A1)
- data: `np.memmap` of token ids → sample random windows of length S+1 → `x = w[:-1]`, `y = w[1:]`
- checkpoint: model + optimizer state + step + RNG → resume exactly
- log: wandb / tensorboard
- eval on held-out val periodically
- decoding: temperature + top-p sampling until `<|endoftext|>`
