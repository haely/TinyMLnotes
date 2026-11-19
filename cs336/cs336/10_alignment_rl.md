# B10 · Alignment: SFT, RLHF, DPO, GRPO  `[A5]`

> base model = autocomplete · aligned model = follows instructions, reasons, is safe

---

## 1. Pipeline

```
 pretrained base ──► SFT ──► preference tuning / RL ──► chat / reasoning model
                     (imitate)   RLHF (PPO) · DPO · RLVR (GRPO)
```

| term | definition |
|---|---|
| policy π_θ | the LM being trained (state = prompt + tokens so far, action = next token) |
| reference π_ref | frozen copy (usually SFT model) |
| reward r(x, y) | score for response y to prompt x |
| rollout | sampling responses from current policy |
| on-policy | train on samples from current π |
| KL penalty | keep π close to π_ref → avoid reward hacking, preserve fluency |
| reward hacking | policy exploits reward model flaws |

## 2. SFT (supervised fine-tuning) ★
- data: (prompt, ideal response) pairs; chat template with roles
- loss: cross-entropy **only on response tokens** (mask prompt tokens)
- small data works (1k–100k high-quality examples)
- reasoning SFT: train on chain-of-thought traces (distillation from stronger model)
- ⚠ teaches format/style; may teach hallucination if targets contain facts model doesn't know

## 3. RLHF (InstructGPT-style)

### Step 1: reward model (Bradley–Terry)
- human picks preferred `y_w` over `y_l`
- `L_RM = −log σ( r(x, y_w) − r(x, y_l) )`

### Step 2: RL with PPO
```
 maximize  E[ r(x, y) ] − β · KL( π_θ(·|x) ‖ π_ref(·|x) )

 PPO clipped objective:  ρ = π_θ(a|s) / π_old(a|s)
 L = E[ min( ρ·A, clip(ρ, 1−ε, 1+ε)·A ) ]
 A = advantage (how much better than expected) from value/critic model
```
- infra cost: 4 models in memory (policy, ref, reward, value) + generation → heavy

## 4. DPO (Direct Preference Optimization) ★
- skip reward model + RL; closed-form link between optimal policy and reward
```
 L_DPO = −log σ( β·[ log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x) ] )
```
- just a classification-style loss on preference pairs → simple, stable, cheap
- ⚠ offline (fixed data) → less exploration than online RL

## 5. RL with verifiable rewards (RLVR) ★ (reasoning models)
- reward from **checker**, not learned model: math answer matches, code passes tests, format correct
- no reward hacking of a neural RM (but can hack checker/format)
- driver of o1/R1-style "thinking" models

### Policy gradient basics
```
 ∇J(θ) = E_y~π [ ∇ log π_θ(y|x) · (R(y) − b) ]
 b = baseline → lower variance without bias
```

### GRPO (Group Relative Policy Optimization) ★ (A5)
```
 for each prompt q:
   sample G responses o₁…o_G from π_old
   rewards r₁…r_G                          (e.g. 1 if correct else 0)
   advantage Aᵢ = (rᵢ − mean(r)) / (std(r) + ε)    ← group = baseline, no critic!
 loss = −(1/G) Σᵢ (1/|oᵢ|) Σₜ min( ρᵢₜ·Aᵢ, clip(ρᵢₜ, 1−ε, 1+ε)·Aᵢ )  (+ β·KL)
```
- removes value model → memory/infra savings vs PPO
- ⚠ all-correct or all-wrong group → zero advantage → no signal (pick right difficulty)
- ⚠ length normalization / std normalization biases (Dr. GRPO variants fix)

### Expert iteration / rejection-sampling FT
- sample many → keep correct ones → SFT on them → repeat
- simple baseline, competitive

## 6. A5 setup
- model: Qwen2.5-Math-1.5B · data: MATH (+ GSM8K)
- zero-shot baseline with reasoning prompt (`<think> … </think> <answer> … </answer>`)
- reward: format + answer correctness via math grader
- SFT on reasoning traces → expert iteration → **GRPO**
- **vLLM** for fast rollouts; policy on another GPU; sync weights policy → vLLM each step
- explore: on- vs off-policy, group size, LR, length normalization, std normalization, clipping

## 7. RL infra notes ★
- RL loop = **generation (inference) bottleneck** + training
```
 ┌─────────────┐  weights  ┌─────────────┐
 │  trainer    │ ────────► │  inference   │  vLLM / SGLang
 │ (FSDP etc.) │ ◄──────── │  rollouts    │
 └─────────────┘  samples  └─────────────┘
                  + rewards (verifier / RM)
```
- long reasoning outputs → long decode → stragglers (sync vs async RL)
- weight sync cost between trainer and inference engine
- logprob mismatch between engines (bf16 numerics) → importance correction

## 8. Safety (awareness)
- refusals, jailbreak robustness, constitutional AI (AI feedback), red-teaming
