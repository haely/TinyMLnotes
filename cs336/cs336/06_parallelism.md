# B6 · Parallelism (multi-GPU, multi-node)  `[A2]`  ★★ infra core

> why: model doesn't fit on 1 GPU (memory) · training too slow on 1 GPU (compute)
> what to split: **data** · **params/grads/optimizer** · **layers** · **within a layer** · **sequence** · **experts**

---

## 1. Collectives recap ★

```
 all-reduce (sum)            all-gather                  reduce-scatter
 r0: a0      → a0+a1+a2+a3   r0: [A]       → [A B C D]    r0: a0..d0 → Σa (shard 0)
 r1: a1      → a0+a1+a2+a3   r1: [B]       → [A B C D]    r1: a1..d1 → Σb (shard 1)
 r2: a2      → …             r2: [C]       → [A B C D]    …
 r3: a3      → …             r3: [D]       → [A B C D]
```

| collective | bytes sent per GPU (ring, size M, n GPUs) |
|---|---|
| all-gather | `(n−1)/n · M` |
| reduce-scatter | `(n−1)/n · M` |
| **all-reduce** = RS + AG | `2(n−1)/n · M ≈ 2M` |
| all-to-all | `(n−1)/n · M` |

- time ≈ bytes / link bandwidth (+ latency per hop)
- `torch.distributed`: `init_process_group("nccl")`, `dist.all_reduce(t)`, `dist.all_gather_into_tensor`, `dist.reduce_scatter_tensor`
- terms: **rank** = process id · **world size** = # processes · **local rank** = GPU id in node

### Ring all-reduce
```
 GPU0 → GPU1 → GPU2 → GPU3 → (back to GPU0)
 phase 1 reduce-scatter: n−1 steps, each passes one chunk, adds own
 phase 2 all-gather:     n−1 steps, pass finished chunks around
 → bandwidth-optimal: per-GPU traffic ~2M regardless of n
```

## 2. Data parallelism (DP / DDP) ★

```
 batch ──split──► GPU0: full model, shard 0 ─┐
                  GPU1: full model, shard 1 ─┼─► all-reduce grads (avg) ─► identical step on each
                  GPU2: full model, shard 2 ─┤
                  GPU3: full model, shard 3 ─┘
```
- comm per step: all-reduce of all grads ≈ `2 × 2N` bytes (bf16)
- ⚠ memory: every GPU holds full 16N bytes → doesn't help fit big models

### DDP versions (A2 builds these)
1. naive: after backward, all-reduce each param separately → many small comms, no overlap
2. flatten: one big all-reduce → fewer launches
3. **overlap**: register grad hooks → start async all-reduce for a param as soon as its grad is ready, while backward continues
4. **bucketed**: group params into ~25 MB buckets → balance # calls vs overlap ★

```
 backward:  [layer L][layer L−1][layer L−2]…[layer 1]
 comm:           [bucket 1 AR][bucket 2 AR] … [last AR]  ← overlapped, only tail exposed
```

## 3. ZeRO / FSDP (sharded data parallel) ★

Ψ = params, N_d = # GPUs, Adam fp32 states K = 12 bytes/param

| stage | shard | memory/GPU | comm vs DDP |
|---|---|---|---|
| DDP baseline | nothing | `16Ψ` | 1× |
| **ZeRO-1** | optimizer state | `4Ψ + 12Ψ/N_d` | 1× (RS grads → local step → AG params) |
| ZeRO-2 | + gradients | `2Ψ + 14Ψ/N_d` | 1× |
| **ZeRO-3 / FSDP** | + parameters | `16Ψ/N_d` | 1.5× (AG in fwd, AG in bwd, RS grads) |

- example: 7B, 8 GPUs → DDP 112 GB ✗ · ZeRO-1 ≈ 28 + 10.5 = 38.5 GB ✓ · ZeRO-3 = 14 GB ✓
- FSDP: all-gather a layer's params just before use, free after → prefetch next layer to overlap
- A2: implement **optimizer state sharding** (ZeRO-1 style): each rank updates its param shard, then broadcast/all-gather

## 4. Tensor parallelism (TP, Megatron) ★

split **inside** a matmul across GPUs

```
 MLP:  Y = GeLU(X·A) · B
   A split by COLUMNS:  [A₁ | A₂]  → GPU i computes GeLU(X·Aᵢ)   (no comm, GeLU is elementwise)
   B split by ROWS:     [B₁ ; B₂]  → GPU i computes GeLU(X·Aᵢ)·Bᵢ = partial Y
   all-reduce partial Y's → Y
 attention: split heads across GPUs (Q,K,V column-split, O row-split) → 1 all-reduce

        X (replicated)
       ╱            ╲
  [X·A₁ → GeLU → ·B₁]   [X·A₂ → GeLU → ·B₂]
       ╲            ╱
        all-reduce → Y
```
- comm: 2 all-reduces/layer forward + 2 backward, size `B·S·d` activations, **every layer**
- ⚠ latency-sensitive, heavy → keep TP **within node** (NVLink), TP ≤ 8 typically
- reduces params, grads, opt state, and some activations per GPU

### Sequence parallelism (SP, with TP)
- norms/dropout regions are replicated under TP → shard them along **sequence dim**
- replace all-reduce with reduce-scatter + all-gather → same comm, less activation memory

## 5. Pipeline parallelism (PP)

split **layers** into stages across GPUs

```
 GPipe, 4 stages, 4 micro-batches (F = forward, B = backward)
 GPU0: F1 F2 F3 F4 .  .  .  .  .  .  B4 B3 B2 B1
 GPU1:    F1 F2 F3 F4 .  .  .  . B4 B3 B2 B1
 GPU2:       F1 F2 F3 F4 .  . B4 B3 B2 B1
 GPU3:          F1 F2 F3 F4 B4 B3 B2 B1
       └─ bubble ─┘ (idle)                 └─ bubble ─┘
 bubble fraction ≈ (p − 1) / (m + p − 1)    p stages, m micro-batches → use m ≫ p
```
- comm: only point-to-point activations between neighbors → cheap → OK **across nodes**
- **1F1B** schedule: interleave fwd/bwd → fewer in-flight activations
- interleaved / zero-bubble schedules → smaller bubble
- ⚠ complex, load balancing between stages

## 6. Context / sequence parallelism for long context
- split the **sequence** across GPUs
- **ring attention**: each GPU keeps its Q block; K/V blocks rotate around ring; online softmax (like FlashAttention across GPUs)
- needed for 128k–1M+ context training

## 7. Expert parallelism (EP, for MoE)
- different experts on different GPUs
- **all-to-all** #1 dispatch tokens to experts' GPUs → compute → all-to-all #2 combine back
- ⚠ bandwidth + load imbalance; often within large NVLink domain

## 8. Putting it together: 3D/4D/5D parallel ★

```
 device mesh example: 512 GPUs = 64 nodes × 8
   TP = 8   (inside node, NVLink)
   PP = 4   (across nodes)
   DP = 16  (FSDP/ZeRO, outermost, across nodes)
   8 × 4 × 16 = 512
```

| strategy | splits | comm | where |
|---|---|---|---|
| DP | batch | grads all-reduce/step | anywhere |
| FSDP | + params/opt | AG/RS per layer | anywhere (needs decent BW) |
| TP | matmuls | all-reduce per layer | **intra-node** |
| PP | layers | P2P activations | inter-node |
| CP/SP | sequence | K/V passing | long context |
| EP | experts | all-to-all | MoE |

### Rule of thumb (course)
1. fit model + activations in memory first: TP up to node size → then PP or FSDP across nodes
2. scale remaining GPUs with DP
3. if global batch too small for DP size → gradient accumulation trade-offs, or more TP/PP
4. always: activation checkpointing, overlap comm with compute

## 9. Napkin math example
- 70B model, bf16 grads (140 GB) DDP all-reduce over IB 50 GB/s/GPU → `2·140/50 ≈ 5.6 s` per step → too slow without overlap/sharding/compression → why FSDP + big batches + overlap matter
