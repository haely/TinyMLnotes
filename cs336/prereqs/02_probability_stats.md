# A2 · Probability + Statistics (CS 109 level)

> goal: LM = probability distribution over next token · loss = cross-entropy · evals need error bars

---

## 1. Basics

| term | definition |
|---|---|
| random variable (RV) | quantity with uncertain value, `X` |
| PMF | `P(X = x)` for discrete X; sums to 1 |
| PDF | density `p(x)` for continuous X; integrates to 1 |
| expectation `E[X]` | probability-weighted average, `Σ x·P(x)` |
| variance `Var(X)` | `E[(X − E[X])²]`, spread |
| std `σ` | `√Var`, same units as X |
| covariance | `E[(X−μₓ)(Y−μᵧ)]`, do X and Y move together |
| correlation | covariance / (σₓσᵧ) ∈ [−1, 1] |
| independent | `P(X,Y) = P(X)P(Y)` |
| joint `P(X,Y)` | prob of both |
| marginal | `P(X) = Σᵧ P(X, y)` |
| conditional | `P(X|Y) = P(X,Y)/P(Y)` |
| Bayes | `P(A|B) = P(B|A)P(A)/P(B)` |

### Chain rule of probability ★ → definition of a language model
```
P(x₁, x₂, …, x_T) = P(x₁) · P(x₂|x₁) · P(x₃|x₁,x₂) ⋯ = Πₜ P(xₜ | x_<ₜ)

 "the cat sat"   →   P(the) · P(cat | the) · P(sat | the cat)
 autoregressive LM = neural net that outputs P(next token | previous tokens)
```

### Useful variance facts
- `Var(aX) = a²Var(X)`
- independent: `Var(X+Y) = Var(X) + Var(Y)`
- dot product of two d-dim vectors with iid unit-variance entries → variance ≈ d
  → attention divides by `√d_h` to keep scores ~unit scale
  → init weights with std ∝ `1/√fan_in` to keep activations ~unit scale

---

## 2. Distributions you'll see

| name | params | what it models | LLM use |
|---|---|---|---|
| Bernoulli | p | coin flip, {0,1} | correct/incorrect on an eval item |
| Categorical | p₁…p_V, Σ=1 | one of V outcomes | **next-token distribution** |
| Gaussian `N(μ, σ²)` | mean, var | bell curve | weight init, noise |
| Uniform | a, b | all equal | random sampling, dropout masks |
| Binomial | n, p | # successes in n trials | eval accuracy counts |

Gaussian: 68% within ±1σ, 95% within ±1.96σ, 99.7% within ±3σ

---

## 3. Big theorems
- **LLN** (law of large numbers): sample mean → true mean as n → ∞
- **CLT** (central limit theorem): sample mean ≈ Gaussian with std `σ/√n`
- **standard error** `SE = σ/√n` → uncertainty of an average
- **95% CI** ≈ `mean ± 1.96·SE`

⚠ eval example: benchmark with n = 1000 questions, accuracy 0.70
→ `SE = √(0.7·0.3/1000) ≈ 0.0145` → 95% CI ≈ **±2.8 pts** → "70.0 vs 71.5" = noise

---

## 4. Estimation

| term | definition |
|---|---|
| likelihood | `P(data | θ)` viewed as function of θ |
| log-likelihood | `Σᵢ log P(xᵢ | θ)`; sums instead of products → stable |
| **MLE** ★ | pick θ maximizing likelihood |
| MLE for LMs | maximize `Σₜ log P_θ(xₜ | x_<ₜ)` ⇔ minimize cross-entropy loss |
| bias (estimator) | E[estimate] − truth |
| variance (estimator) | spread of estimate across datasets |

---

## 5. Information theory ★ (core of LM loss)

| term | formula | meaning |
|---|---|---|
| surprise | `−log p(x)` | rare event → big surprise |
| entropy `H(p)` | `−Σ p(x) log p(x)` | avg surprise under p; uncertainty |
| cross-entropy `H(p, q)` | `−Σ p(x) log q(x)` | avg surprise using model q on data from p |
| KL divergence | `KL(p‖q) = Σ p log(p/q) = H(p,q) − H(p)` | extra cost from using q instead of p; ≥ 0; ⚠ asymmetric |
| perplexity | `exp(avg cross-entropy in nats)` | "effective # of choices" per token |
| bits-per-byte (BPB) | total loss in bits / # bytes of text | tokenizer-independent comparison |

- LM training loss per token = `−log q(true next token)` (p = one-hot target)
- log base: `ln` → nats (PyTorch default) · `log₂` → bits · 1 nat = 1.44 bits
- perplexity examples: uniform over V=50k → PPL 50k (loss ≈ 10.8 nats); good LM ≈ PPL 5–10 (loss ≈ 1.6–2.3)
- KL appears in: RLHF penalty (stay near reference model), distillation

---

## 6. Softmax + sampling

```
logits z ∈ ℝ^V  ──►  softmax  ──►  p ∈ Δ^V (probs, sum to 1)
softmax(z)ᵢ = e^{zᵢ/τ} / Σⱼ e^{zⱼ/τ}          τ = temperature
```

| decoding | rule | effect |
|---|---|---|
| greedy | argmax p | deterministic, repetitive |
| temperature τ | divide logits by τ | τ<1 sharper · τ>1 flatter · τ→0 greedy |
| top-k | keep k most likely, renormalize | cut off long tail |
| top-p (nucleus) | keep smallest set with cumulative prob ≥ p | adaptive tail cut |

- **sigmoid** `σ(z) = 1/(1 + e^{−z})` → 2-class softmax
- **Bradley–Terry** `P(a ≻ b) = σ(r_a − r_b)` → reward models, Chatbot Arena Elo [A5]

---

## 7. Power laws ★ (scaling laws)
- form: `y = a · x^{−α}`
- take logs: `log y = log a − α log x` → **straight line on log-log plot**, slope = −α
- fit: linear regression on (log x, log y)
- loss vs compute / params / data all ≈ power laws → predict big runs from small ones [A3]

```
 log L │•
       │  •
       │     •
       │        •   slope = −α
       │           •
       └──────────────── log C
```
