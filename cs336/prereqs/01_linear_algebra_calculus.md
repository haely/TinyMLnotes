# A1 · Linear Algebra + Calculus (MATH 51 / CME 100 level)

> goal: read any transformer equation · know shapes · count matmul FLOPs · understand backprop

---

## 1. Objects

| term | definition | example in LLMs |
|---|---|---|
| scalar | single number | loss value, learning rate |
| vector | 1-D list of numbers, `x ∈ ℝᵈ` | one token's embedding (d = 4096) |
| matrix | 2-D grid, `W ∈ ℝ^{m×n}` (m rows, n cols) | weight matrix of a linear layer |
| tensor | n-D array | activations `[B, S, d]` |
| shape | list of dim sizes | `(32, 2048, 4096)` |
| dim / axis | one index direction of a tensor | "sum over the last dim" |

⚠ most LLM code bugs = shape bugs → always annotate shapes in comments

---

## 2. Core ops + cost

| op | formula | shapes | FLOPs |
|---|---|---|---|
| elementwise | `z = x + y`, `x ⊙ y` | same shape | n |
| dot product | `x·y = Σᵢ xᵢyᵢ` | (d),(d) → scalar | 2d |
| mat-vec | `y = Wx` | (m×n),(n) → (m) | 2mn |
| **matmul** ★ | `C = AB` | (m×k),(k×n) → (m×n) | **2mkn** |
| transpose | `Aᵀ[i,j] = A[j,i]` | (m×n) → (n×m) | 0 (just a view) |
| outer product | `xyᵀ` | (m),(n) → (m×n) | mn |

- why "2": each multiply paired with one add (multiply-accumulate = MAC = 2 FLOPs)

```
            k                  n                     n
        ┌───────┐          ┌───────┐             ┌───────┐
      m │   A   │   ×    k │   B   │    =      m │   C   │
        └───────┘          └───────┘             └───────┘

   C[i,j] = Σₖ A[i,k] · B[k,j]
   m·n outputs  ×  k multiply-adds each  =  2·m·k·n FLOPs
   inner dims (k) must match · outer dims (m, n) survive
```

### Batched matmul + broadcasting
- batched matmul: `[B, m, k] @ [B, k, n] → [B, m, n]` (independent matmul per batch item)
- linear layer on activations: `[B, S, d] @ [d, d_ff] → [B, S, d_ff]` → FLOPs = `2·B·S·d·d_ff`
- broadcasting: size-1 or missing dims auto-stretched; `[B,S,d] + [d]` → bias added to every token

### einsum (used heavily in CS336)
- notation: letters = dims; letters missing from output = summed over
- `einsum('bsd,df->bsf', x, W)` → linear layer
- `einsum('bhqd,bhkd->bhqk', Q, K)` → attention scores
- tip: `einops.rearrange(x, 'b s (h d) -> b h s d', h=H)` → split heads

---

## 3. Key concepts

| term | definition | why it matters |
|---|---|---|
| linear map | matrix = function that rotates / scales / projects vectors | every layer = linear map + nonlinearity |
| identity `I` | 1s on diagonal, `IA = A` | residual connection ≈ "add identity" |
| inverse `A⁻¹` | `AA⁻¹ = I` | rarely computed in DL (expensive, unstable) |
| rank | # independent rows/cols | low-rank `W ≈ UV` (U: m×r, V: r×n) → LoRA, MLA |
| L2 norm | `‖x‖₂ = √(Σ xᵢ²)` | grad clipping, weight decay, RMSNorm |
| L1 norm | `‖x‖₁ = Σ|xᵢ|` | sparsity |
| Frobenius norm | `‖A‖_F = √(Σᵢⱼ Aᵢⱼ²)` | whole-matrix size |
| spectral norm | largest singular value of A | "max stretch"; μP / Muon intuition |
| orthogonal matrix | `QᵀQ = I`; preserves lengths | RoPE = rotation matrices |
| eigenvector | `Av = λv`: direction only scaled | curvature, stability analysis |
| SVD | `A = UΣVᵀ`; U,V orthogonal, Σ diagonal ≥ 0 | best low-rank approximation |
| cosine similarity | `x·y / (‖x‖‖y‖)` ∈ [-1, 1] | "how aligned" two vectors are |
| projection | dot product = how much x points along y | attention score `q·k` |

---

## 4. Calculus

| term | definition |
|---|---|
| derivative `f'(x)` | slope; rate of change of output wrt input |
| partial `∂f/∂xᵢ` | derivative wrt one input, others fixed |
| gradient `∇f` | vector of all partials; points to steepest increase |
| chain rule ★ | `dz/dx = dz/dy · dy/dx` → backprop = chain rule applied backward |
| Jacobian | matrix of partials for vector→vector function `∂yᵢ/∂xⱼ` |
| Hessian | matrix of 2nd derivatives; curvature; N×N → too big for LLMs |
| Taylor (1st order) | `f(x+δ) ≈ f(x) + ∇f·δ` → small step along `−∇f` lowers f |
| gradient descent | `θ ← θ − η ∇L(θ)` |
| convex | bowl-shaped, one minimum; ⚠ neural nets are NOT convex |

### Computation graph: forward vs backward

```
 forward  →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→
   x ──► [ y = Wx ] ──► [ a = relu(y) ] ──► [ L = loss(a) ]
         ∂L/∂W        ◄──  ∂L/∂y   ◄──── ∂L/∂a  ◄──── 1
 backward ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
 each node: local derivative × upstream gradient (chain rule)
 ⚠ backward needs forward's intermediate values (y, a) → stored = "activation memory"
```

### Matrix-calculus results to memorize ★

| forward | backward |
|---|---|
| `y = Wx` | `∂L/∂x = Wᵀ·(∂L/∂y)` and `∂L/∂W = (∂L/∂y)·xᵀ` |
| `a = relu(y)` | `∂L/∂y = ∂L/∂a ⊙ 1[y > 0]` |
| softmax + cross-entropy | `∂L/∂logits = p − onehot(target)` |
| `z = x + y` (residual) | gradient copied to both branches unchanged |

→ backward of one matmul = **2 matmuls** (one for ∂x, one for ∂W) → backward ≈ 2× forward FLOPs
→ total training ≈ 3× forward = origin of **6ND** rule (see CS336 B2)

---

## 5. Numerics trick: log-sum-exp ★
- problem: `eˣ` overflows for large x (e.g. e¹⁰⁰⁰ = inf)
- fix: `log Σ eˣⁱ = m + log Σ e^{xᵢ − m}`, with `m = max(x)`
- stable softmax: `softmax(x) = softmax(x − max(x))` (same result, no overflow)
- used in: cross-entropy implementation [A1], FlashAttention "online softmax" [A2]

## 6. Quick self-check
- `[4, 128, 512] @ [512, 2048]` → shape? FLOPs? → `[4,128,2048]`, `2·4·128·512·2048 ≈ 1.07 GFLOP`
- why is backward ~2× forward? → each matmul needs grads for both input and weight
