# B7 · Scaling Laws  `[A3]`

> fixed compute budget C → how big a model (N), how many tokens (D)? · predict big runs from small ones

---

## 1. Core idea
- loss falls as a **power law** in compute, params, data (straight lines on log-log)
- fit on cheap small runs → extrapolate to expensive big run
- uses: choose N vs D, choose hyperparams, compare arch/data choices by **slope**, not one point

## 2. Terms

| term | definition |
|---|---|
| scaling law | `L(x) ≈ E + A·x^(−α)` for x ∈ {N, D, C} |
| irreducible loss E | entropy of text itself; can't beat it |
| compute-optimal | lowest loss for given C |
| IsoFLOP curve | many (N, D) pairs with same C = 6ND |
| tokens-per-param | D/N ratio |
| overtraining | D/N ≫ optimal → worse per FLOP but cheaper inference |
| emergence | ability appears suddenly at scale; ⚠ often artifact of discontinuous metric |

## 3. Kaplan (OpenAI, 2020) vs Chinchilla (DeepMind, 2022) ★

| | Kaplan | **Chinchilla** |
|---|---|---|
| finding | scale N faster than D (N ∝ C^0.73) | scale **both equally**: N ∝ C^0.5, D ∝ C^0.5 |
| ratio | few tokens/param (GPT-3: ~1.7) | **D ≈ 20 N** ★ |
| why differ | fixed LR schedule (not tuned to run length), non-embedding params, small models | cosine schedule matched to each run length |

### Chinchilla parametric fit
```
 L(N, D) = E + A / N^α + B / D^β
 E ≈ 1.69, A ≈ 406, B ≈ 411, α ≈ 0.34, β ≈ 0.28
 minimize s.t. C = 6ND  →  N_opt ∝ C^a, D_opt ∝ C^b, a ≈ b ≈ 0.5
```

### Three fitting methods (Chinchilla)
1. fix N, vary D → lower envelope of loss curves
2. **IsoFLOP**: for each C, sweep N → U-shaped curve → take min ★ (A3)
3. fit parametric L(N, D) directly

```
 loss │ C=1e18   C=1e19    C=1e20
      │  ╲  ╱     ╲   ╱      ╲    ╱
      │   ╲╱       ╲ ╱        ╲  ╱      ● = N_opt(C)
      │    ●        ●          ●
      └──────────────────────────────── log N
  then fit: log N_opt vs log C → straight line → extrapolate
```

## 4. Worked example ★
- budget `C = 1e21` FLOPs, Chinchilla ratio D = 20N
- `C = 6·N·20N = 120 N²` → `N = √(1e21/120) ≈ 2.9B params`, `D ≈ 58B tokens`
- rule: **N ≈ √(C/120)**, **D ≈ 20N**

## 5. Beyond Chinchilla
- **inference-aware**: model served billions of times → smaller N, many more tokens (Llama 3 8B: 15T tokens ≈ 1,900 tok/param)
- **data-constrained**: repeating data up to ~4 epochs ≈ as good as fresh; beyond → diminishing (Muennighoff 2023)
- **downstream**: benchmark acc ~ sigmoid of loss → predict via loss → acc mapping
- WSD schedule → one run gives many "end points" → cheaper scaling sweeps

## 6. Hyperparameters at scale ★
- optimal LR **shrinks** as width grows (standard parameterization) → tuning at small scale doesn't transfer
- **μP** (maximal update parameterization): scale init + per-layer LR with width (e.g. Adam LR for hidden matrices ∝ 1/width, output multiplier 1/width) → optimal LR **transfers** from small proxy to big model
- batch size: optimal / critical batch grows with compute (and as loss falls)
- fit scaling laws for LR and batch too (e.g. DeepSeek: `η_opt`, `B_opt` as power laws of C)

## 7. A3 workflow
- given: training API → `(d_model, n_layers, n_heads, batch, lr, train_flops) → final loss`, limited query budget
- goal: predict optimal model size + hyperparams + loss at bigger budget (e.g. 1e19)
1. pick a few compute levels C₁ < C₂ < … (cheap)
2. at each C: sweep N (via d_model, layers) → IsoFLOP curve → N_opt(C)
3. fit `N_opt = k·C^a` in log-log; same for loss `L_opt(C)`
4. extrapolate to target C; pick hyperparams (aspect ratio, LR) from best small runs
5. ⚠ spend query budget wisely: more points near the minimum, fewer extreme sizes

## 8. Pitfalls
- counting params (incl./excl. embeddings) changes exponents at small scale
- LR schedule length must match run length
- small-scale noise → repeat seeds / smooth
- extrapolating too many orders of magnitude
