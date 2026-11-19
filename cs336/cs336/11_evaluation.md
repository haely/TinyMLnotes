# B11 · Evaluation

> "what does it mean for a model to be good?" · numbers depend heavily on setup

---

## 1. Kinds of evals

| kind | examples | measures |
|---|---|---|
| perplexity / loss | held-out text, bits-per-byte | raw LM quality; smooth, good for scaling laws |
| multiple-choice | MMLU (57 subjects), MMLU-Pro, HellaSwag, ARC, GPQA | knowledge, reasoning |
| generation + exact match | GSM8K, MATH, AIME | math reasoning |
| code | HumanEval, MBPP, LiveCodeBench, SWE-bench | functional correctness |
| instruction following | IFEval, AlpacaEval, MT-Bench | chat quality |
| human preference | Chatbot Arena (Elo / Bradley–Terry) | overall vibes |
| agents | SWE-bench Verified, terminal / tool-use benchmarks | multi-step tasks |
| safety | refusal, jailbreak, harmful-content evals | risk |
| holistic | HELM (many scenarios × metrics) | broad picture |

## 2. Mechanics ★
- **multiple choice via logprobs**: score each option `log P(option | question)` → pick max (base models)
- **generation**: sample answer → parse → compare (chat/reasoning models)
- few-shot vs zero-shot, prompt template, chain-of-thought → ⚠ change scores by many points
- **pass@k** (code): prob ≥1 of k samples correct; unbiased estimate from n samples, c correct:
  `pass@k = 1 − C(n−c, k) / C(n, k)`
- **LLM-as-judge**: strong model grades outputs → ⚠ biases: length, position, self-preference
- error bars: `SE = √(p(1−p)/n)` → small benchmarks are noisy ([A2 stats](../prereqs/02_probability_stats.md))

## 3. Pitfalls ★
- **contamination**: test data in training set → n-gram overlap checks, fresh/held-out sets
- benchmark saturation → move to harder sets
- Goodhart's law: optimize metric → metric stops meaning quality
- comparing models with different prompts / harnesses
- perplexity not comparable across tokenizers → use bits-per-byte
- cost/latency matters too: accuracy per $ / per FLOP

## 4. Infra view
- evals = large batched inference jobs → use vLLM-style engines
- reproducibility: fix seeds, temperature, prompt, harness version (lm-eval-harness, HELM)
- track evals during training (checkpoint sweeps) → spot regressions early
