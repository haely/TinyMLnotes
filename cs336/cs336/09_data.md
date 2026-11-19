# B9 · Data  `[A4]`

> "data is the most important ingredient" · pretraining data = filtered web + curated sources

---

## 1. Pipeline ★

```
 Common Crawl (WARC: raw HTTP+HTML, petabytes)
      │  HTML → text extraction (WET = CC's own text; better: resiliparse / trafilatura)
      ▼
 language ID (fastText lid) → keep target languages
      ▼
 heuristic filters (Gopher / C4 rules)
      ▼
 model-based quality filter (fastText / small LM classifier)
      ▼
 safety: toxicity / NSFW classifiers · PII masking
      ▼
 dedup: exact (hash) + near-dup (MinHash + LSH)
      ▼
 decontaminate (remove benchmark overlap)
      ▼
 mix with code, books, papers, math, Wikipedia → tokenize → shards
```

## 2. Sources

| source | notes |
|---|---|
| Common Crawl | monthly web snapshots since 2008; noisy; base of almost all datasets |
| Wikipedia | high quality, small |
| books | long-form; ⚠ copyright |
| GitHub / The Stack | code; license filtering |
| arXiv, PubMed | science |
| StackExchange, Reddit | Q&A, dialogue |
| synthetic | LLM-generated / rewritten data (growing share) |

### Known datasets
- C4 (T5), The Pile (diverse 22 sources), RefinedWeb (Falcon, web-only), Dolma (AI2/OLMo), **FineWeb / FineWeb-Edu** (HF), **DCLM** (benchmark for data curation), RedPajama

## 3. Filtering ★

### Heuristic rules (Gopher-style, A4 implements)
- word count 50 – 100,000
- mean word length 3 – 10 chars
- ≤ 30% lines ending with "…"
- ≥ 80% words contain an alphabetic char
- C4-style: lines end in punctuation, drop pages with "lorem ipsum", `{`, bad-word lists

### Model-based quality
- train classifier: positives = high quality (pages cited by Wikipedia, OpenHermes/ELI5 (DCLM), LLM-rated "educational" (FineWeb-Edu)) vs negatives = random CC
- **fastText** (bag of n-grams, linear, super fast on CPU) → score every doc → keep top X%
- ⚠ filter too hard → lose diversity; classifier bias toward one style

### Safety + privacy
- toxicity / NSFW classifiers (e.g. trained on Jigsaw)
- PII masking (A4): regex for emails → `|||EMAIL_ADDRESS|||`, phone → `|||PHONE_NUMBER|||`, IPs → `|||IP_ADDRESS|||`

## 4. Deduplication ★

| type | method |
|---|---|
| exact doc | hash full text |
| exact line | hash each line, drop lines appearing ≥ k times (boilerplate, nav menus) |
| **near-duplicate** | MinHash + LSH on n-gram sets |

### MinHash + LSH
- doc → set of word n-grams (shingles)
- similarity: **Jaccard** `J(A,B) = |A∩B| / |A∪B|`
- MinHash: `P(minhash_h(A) = minhash_h(B)) = J(A,B)` → signature of k hashes approximates J
- LSH: split signature into **b bands × r rows**; candidate pair if any band matches exactly
- `P(candidate) = 1 − (1 − s^r)^b` → S-curve; threshold ≈ `(1/b)^(1/r)`

```
 P(candidate)
  1 │            ______
    │          /
    │         /     ← steep around threshold s* ≈ (1/b)^(1/r)
  0 │________/
    └──────────────── Jaccard s
```
- then verify candidates with true Jaccard, cluster, keep one per cluster
- why dedup: less memorization, better generalization per FLOP, fewer wasted tokens

## 5. Mixing + curriculum
- **data mixture** = weights per domain (web, code, math, books…) → tuned by small-model ablations (or methods like DoReMi, RegMix)
- **mid-training / annealing**: at end of pretraining (LR decay), upweight high-quality data (math, code, instruct) → big benchmark gains
- long-context extension phase: long documents + RoPE base change

## 6. Contamination
- benchmark questions leak into web → inflated scores
- detect: n-gram overlap (e.g. 13-gram) between train and test → remove

## 7. Legal / ethics (awareness)
- copyright, robots.txt / opt-outs, licenses (code), consent, PII
- fair-use questions active in courts

## 8. Infra notes
- scale: TBs–PBs of text → distributed CPU jobs (Spark, Ray, Slurm array, `multiprocessing`)
- embarrassingly parallel per WARC file → sharded processing
- fastText chosen because CPU-cheap at billions of docs
- output: tokenized `uint16/uint32` shards for streaming
