# B3 · Transformer Architecture (modern LLM block)  `[A1]`

> decoder-only · causal · "Llama-style" = what A1 builds

---

## 1. Whole model

```
 token ids [B, S]
      │
      ▼
 ┌──────────────┐
 │  Embedding   │  E ∈ ℝ^{V×d}  → x [B, S, d]
 └──────┬───────┘
        │   residual stream x
   ┌────▼────────────────────────────────┐
   │  ┌─────────┐   ┌────────────────┐   │
   │  │ RMSNorm │──►│ Causal MH Attn │──(+)  x = x + Attn(Norm(x))
   │  └─────────┘   │  (+ RoPE on q,k)│   │
   │                └────────────────┘   │   × L layers
   │  ┌─────────┐   ┌────────────────┐   │
   │  │ RMSNorm │──►│   SwiGLU FFN   │──(+)  x = x + FFN(Norm(x))
   │  └─────────┘   └────────────────┘   │
   └────┬────────────────────────────────┘
        ▼
   final RMSNorm
        ▼
   LM head  W ∈ ℝ^{d×V}  → logits [B, S, V] → softmax → P(next token)
```

## 2. Attention ★

```
 Q = xW_Q,  K = xW_K,  V = xW_V            each [B, S, d] → split into h heads [B, h, S, d_h]
 Attn(Q,K,V) = softmax( QKᵀ/√d_h + M ) · V
 M = causal mask: 0 on/below diagonal, −∞ above → token t can't see t+1…
 output: concat heads → · W_O
```

| term | definition |
|---|---|
| query q | "what am I looking for" |
| key k | "what do I contain" |
| value v | "what I pass along if chosen" |
| score `q·k/√d_h` | relevance; `√d_h` keeps variance ~1 |
| causal mask | blocks future tokens → valid next-token training on whole seq in parallel |
| multi-head | h independent attentions on `d_h = d/h` slices → different patterns |

```
 causal mask (S=4)        scores QKᵀ: [S × S] per head → O(S²) memory/compute ⚠
   k₁ k₂ k₃ k₄
 q₁ ✓  ✗  ✗  ✗
 q₂ ✓  ✓  ✗  ✗
 q₃ ✓  ✓  ✓  ✗
 q₄ ✓  ✓  ✓  ✓
```

### MHA → GQA → MQA (KV-cache savers)

| variant | # K/V heads | KV cache | used by |
|---|---|---|---|
| MHA (multi-head) | h | 1× | GPT-2/3 |
| **GQA** (grouped-query) | g (e.g. 8), each shared by h/g query heads | g/h | Llama 2 70B, Llama 3 |
| MQA (multi-query) | 1 | 1/h | PaLM |
| MLA (multi-head latent) | low-rank compressed latent | ≪ | DeepSeek V2/V3 |

## 3. Position: RoPE ★
- attention itself is order-blind → need position info
- old: absolute learned/sinusoidal embeddings added to x
- **RoPE** (rotary): rotate each pair of dims `(q₂ᵢ, q₂ᵢ₊₁)` by angle `m·θᵢ` at position m
- `θᵢ = base^(−2i/d_h)`, base = 10,000 (larger, e.g. 500k, for long context)
- property: `⟨R_m q, R_n k⟩` depends only on `m − n` → **relative** position
- applied to q, k only (not v); no params

```
  pair (x₀, x₁) at position m:
  [x₀']   [cos mθ  −sin mθ] [x₀]
  [x₁'] = [sin mθ   cos mθ] [x₁]
```

## 4. FFN: SwiGLU ★
```
 FFN(x) = W₂ · ( SiLU(W₁x) ⊙ W₃x )         SiLU(z) = z·σ(z)
 W₁, W₃: d → d_ff     W₂: d_ff → d
 d_ff ≈ 8/3 · d  (keeps params = 8d², same as 4d ReLU MLP)  → round to multiple of 64
```
- "GLU" = gated linear unit: one branch gates the other elementwise
- empirically better than ReLU/GELU MLP at same params

## 5. Normalization ★
- **RMSNorm**: `x / √(mean(x²) + ε) · γ` → no mean subtraction, no bias; faster, same quality
- **pre-norm** (norm before sublayer, inside residual) vs post-norm (original 2017, after add)
  - pre-norm → clean residual path → stable deep training → universal now
- extra stability: **QK-norm** (norm q, k before dot product), **z-loss** (`10⁻⁴·(log Z)²` keeps softmax normalizer small), logit soft-capping

## 6. Other modern defaults
- no bias terms in linear layers
- no dropout in pretraining
- weight tying (embedding = LM head) → small models only
- vocab padded to multiple of 64/128 for tensor cores

## 7. Hyperparameter conventions ★

| hyperparam | typical |
|---|---|
| `d_ff / d` | 4 (ReLU/GELU) · 8/3 ≈ 2.67 (SwiGLU) |
| `d_h` | 64–128; `h·d_h = d` |
| aspect ratio `d / L` | ~100–200 |
| vocab | 32k–256k |
| weight decay | 0.1 (AdamW) |
| context | 2k–8k pretrain → extend to 128k+ later |

## 8. Mixture of Experts (MoE) ★

```
            x (one token)
               │
          ┌────▼────┐
          │ router  │  logits = W_r·x → top-k (e.g. k=2 of 64)
          └┬──┬──┬──┘
     g₁    │  │  │   g₂
   ┌───────▼┐ ... ┌▼───────┐
   │expert 1│     │expert 7│   (each expert = FFN)
   └───┬────┘     └───┬────┘
       └──── Σ gᵢ·Eᵢ(x) ──► out
```

| term | definition |
|---|---|
| expert | separate FFN; replaces dense FFN |
| router / gate | small linear layer picking top-k experts per token |
| total vs active params | many experts stored, only k used per token → more params at same FLOPs |
| load balancing loss | aux loss pushing equal usage across experts (else collapse to few) |
| capacity factor | max tokens per expert; overflow → dropped tokens |
| shared experts | always-on experts + many fine-grained routed ones (DeepSeek) |
| expert parallelism | experts on different GPUs → **all-to-all** comm ([B6](06_parallelism.md)) |

- examples: Mixtral 8×7B (2 of 8), DeepSeek-V3 (671B total, ~37B active)
- ⚠ infra pain: routing imbalance, all-to-all bandwidth, training instability

## 9. Shape walkthrough (one layer, A1)

| step | shape |
|---|---|
| input ids | `[B, S]` |
| embed | `[B, S, d]` |
| q, k, v | `[B, h, S, d_h]` |
| scores | `[B, h, S, S]` |
| attn out | `[B, S, d]` |
| FFN hidden | `[B, S, d_ff]` |
| logits | `[B, S, V]` |

## 10. Beyond transformers (awareness)
- sliding-window / local attention → O(S·w)
- linear attention, state-space models (Mamba), hybrids → O(S) memory at inference
- diffusion LMs → parallel generation (research)
