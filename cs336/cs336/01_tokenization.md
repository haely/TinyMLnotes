# B1 · Tokenization  `[A1]`

> text (string) ⇄ list of integers · model only ever sees integers

---

## 1. Terms

| term | definition |
|---|---|
| token | unit the model reads/predicts (a word piece) |
| vocabulary `V` | set of all tokens; size = # rows of embedding matrix |
| encode | string → token ids |
| decode | token ids → string |
| compression ratio | bytes / token; higher = fewer tokens per text = cheaper |
| special token | reserved id, e.g. `<|endoftext|>` separating documents |
| OOV | out-of-vocabulary; ⚠ impossible with byte-level BPE |

## 2. Unicode + UTF-8 (needed for byte-level BPE)
- Unicode: every character → **code point** integer (`ord('a') = 97`, `ord('你') = 20320`)
- ~150k code points → too many/sparse for vocab directly
- UTF-8: code point → **1–4 bytes** (ASCII = 1 byte; most CJK = 3; emoji = 4)
- bytes ∈ 0..255 → **256 base tokens cover every possible string**
- ⚠ decoding arbitrary byte sequences may be invalid UTF-8 → `bytes.decode('utf-8', errors='replace')`

## 3. Options compared

| scheme | vocab | seq length | problem |
|---|---|---|---|
| character | ~150k code points | long | huge sparse vocab, rare chars |
| byte | 256 | very long (1 token/byte) | attention cost ∝ S² → expensive |
| word | unbounded | short | OOV, huge vocab, no morphology |
| **subword (BPE)** ★ | 32k–200k+ | medium (~4 bytes/token English) | sweet spot → everyone uses it |

## 4. BPE (Byte-Pair Encoding) training ★

```
 1. vocab ← 256 byte tokens (+ special tokens)
 2. pre-tokenize corpus (regex split into "words"), count each word
 3. repeat until vocab size reached:
      a. count all adjacent token pairs (weighted by word count)
      b. pick most frequent pair (A, B)   (ties → lexicographically greatest, per A1 spec)
      c. add new token AB to vocab; record merge (A, B)
      d. replace every A B with AB in all words
```

### Worked example (classic)
- word counts: `low:5  lower:2  newest:6  widest:3`
- initial pair counts: `es:9  st:9  we:8  lo:7  ow:7  ne:6  ew:6 …`

| merge # | pair | why |
|---|---|---|
| 1 | `s t` → `st` | tie 9 w/ `es`; `st` lexicographically greater |
| 2 | `e st` → `est` | 9 |
| 3 | `o w` → `ow` | 7 |
| 4 | `l ow` → `low` | 7 |
| 5 | `w est` → `west` | 6 |
| 6 | `n e` → `ne` | 6 |
| … | … | … |

### Encoding new text
- pre-tokenize → bytes → apply merges **in the order learned** (earliest merge first)
- special tokens matched first, never split

### Pre-tokenization (GPT-2 regex)
```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```
- splits on contractions, letters, numbers, punctuation, whitespace
- why: no merges across word boundaries (no `dog.` vs `dog!` duplicates); leading space attaches to word (`" the"`)

## 5. Efficiency tricks (A1)
- pre-tokenize in parallel: chunk file at `<|endoftext|>` boundaries → multiprocessing
- count **unique pre-tokens with counts**, not raw text
- incremental update: after a merge, only update pair counts of words containing the merged pair (don't recount everything)
- store corpus of tokens as `np.uint16` (V < 65,536) → memmap for training

## 6. Practical facts
- English ≈ 4 bytes/token (~0.75 words/token) for GPT-style tokenizers
- vocab trend: GPT-2 50k → Llama 3 128k → bigger for multilingual
- bigger vocab → shorter sequences, but bigger embedding + LM head (`V·d` params, `2·V·d` FLOPs/token for head)
- ⚠ tokenization quirks → arithmetic/spelling errors, poor non-English compression (more tokens = more cost)
- alternatives: byte-level models (ByT5), learned/dynamic patching (BLT) → active research
