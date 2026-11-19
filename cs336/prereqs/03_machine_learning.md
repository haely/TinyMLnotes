# A3 · Machine Learning Basics (CS229 / CS221 level)

> goal: vocabulary of training · LM = giant softmax classifier trained with SGD

---

## 1. Setup

| term | definition |
|---|---|
| dataset | examples `{(xᵢ, yᵢ)}` |
| features / input `x` | what model sees |
| label / target `y` | what model should predict |
| model `f_θ` | function with params θ |
| parameters θ | learned numbers (weights, biases) |
| hyperparameters | set by you, not learned: LR, batch size, # layers |
| loss `ℓ(f_θ(x), y)` | how wrong one prediction is |
| empirical risk | avg loss over training set → what we minimize |
| inference | using trained model to predict |

### Learning types
- **supervised**: labeled (x, y)
- **unsupervised**: only x (clustering, density)
- **self-supervised** ★: labels come from data itself → next-token prediction (y = next word)
- **reinforcement learning**: learn from reward signal, not labels → RLHF/GRPO [A5]

---

## 2. Classic models (all reappear in LLMs)

| model | form | loss | in LLMs |
|---|---|---|---|
| linear regression | `ŷ = wᵀx + b` | MSE `(y − ŷ)²` | — |
| logistic regression | `p = σ(wᵀx + b)` | binary CE | reward model head |
| **softmax regression** ★ | `p = softmax(Wx)` | cross-entropy | **LM output layer** (W = V×d) |
| MLP | stacked linear + nonlinearity | any | FFN in transformer |

→ an LLM = deep feature extractor (transformer) + softmax regression on top

---

## 3. Optimization

| term | definition |
|---|---|
| gradient descent (GD) | step against gradient of full-dataset loss |
| SGD | gradient from random mini-batch → noisy but cheap |
| mini-batch | subset of examples per step (batch size B) |
| step / iteration | one parameter update |
| epoch | one pass through dataset; ⚠ LLM pretraining often < 1 epoch |
| learning rate η | step size; most important hyperparameter |
| convergence | loss stops improving |
| momentum | running avg of gradients → smoother direction |

```
 loss
  │  η too big: diverge ↗↘↗ (NaN)
  │
  │╲   η too small: slow ╲________
  │ ╲
  │  ╲___ η good ________________
  └─────────────────────────────── steps
```

---

## 4. Generalization

| term | definition |
|---|---|
| train / val / test split | fit on train · tune on val · report on test once |
| overfitting | train loss ↓ but val loss ↑; memorizing |
| underfitting | both high; model too small / undertrained |
| generalization gap | val loss − train loss |
| regularization | anything reducing overfitting |
| weight decay / L2 | penalty `λ‖θ‖²` → shrink weights |
| dropout | randomly zero activations in training |
| early stopping | stop when val loss rises |
| bias–variance tradeoff | simple model → high bias · complex → high variance |

```
 loss
  │╲            ___ val (overfit region →)
  │ ╲      ___/
  │  ╲____/
  │   ╲_____________ train
  └─────────────────── model capacity / training time
```

⚠ LLM pretraining twist: huge data, ~1 epoch → rarely overfit; main constraint = compute, not generalization gap

---

## 5. Metrics (quick)
- accuracy: % correct
- precision = TP/(TP+FP) · recall = TP/(TP+FN) → used for data-quality classifiers [A4]
- F1 = harmonic mean of precision & recall
- log-loss / perplexity → LM quality

## 6. Workflow
1. baseline small model
2. overfit one batch (sanity check: loss → ~0)
3. scale up; tune LR first
4. track train + val curves
5. ablate one change at a time
