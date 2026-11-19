# B5 · GPUs, Kernels, Triton, FlashAttention  `[A2]`  ★★ infra core

> prereq: [A5 hardware](../prereqs/05_hardware_systems.md) (memory hierarchy, roofline)
> mantra: **compute is cheap, memory movement is expensive**

---

## 1. Why GPUs got fast, memory didn't
- compute FLOP/s grew much faster than HBM bandwidth → gap widens every generation
- → more and more ops become **memory-bound**
- → winning trick = keep data in SRAM/registers, touch HBM once

## 2. Performance toolbox ★

| trick | idea | helps |
|---|---|---|
| low precision | bf16/fp8 → fewer bytes + faster tensor cores | both |
| **operator fusion** | combine many elementwise ops into one kernel | memory-bound ops |
| **tiling** | load block into SMEM, reuse many times | matmul, attention |
| recomputation | recompute cheap values instead of storing/reading | memory + IO |
| coalescing | contiguous accesses per warp | bandwidth |
| avoid divergence | uniform control flow in warp | compute |
| right sizes | dims multiple of 64/128, #blocks ≫ #SMs | tensor cores, waves |
| CUDA graphs / compile | cut Python + launch overhead | overhead-bound |

## 3. Operator fusion

```
 unfused: y = gelu(x·W + b)·s
   HBM ─► [matmul] ─► HBM ─► [add b] ─► HBM ─► [gelu] ─► HBM ─► [mul s] ─► HBM
          4 kernels, 4 round trips

 fused:
   HBM ─► [matmul + add + gelu + mul in registers] ─► HBM
          1 kernel, 1 round trip
```
- like a factory: shipping goods to warehouse (HBM) between each station is the slow part

## 4. Tiled matmul

```
       B (K × N)                 each thread block computes one T×T tile of C
      ┌──┬──┬──┐                 loop over K in chunks:
      │  │▓▓│  │                   load A-tile (T×Tk) + B-tile (Tk×T) → SMEM
      └──┴──┴──┘                   multiply-accumulate in registers
 A (M×K)   C (M × N)              each loaded element reused T times
 ┌─────┐  ┌──┬──┬──┐             → HBM traffic ↓ by ~T → arithmetic intensity ↑
 │▓▓▓▓▓│  │  │██│  │
 └─────┘  └──┴──┴──┘
```

## 5. FlashAttention ★★

### Problem: standard attention
```
 S = QKᵀ   → write [S×S] to HBM
 P = softmax(S) → read S, write P [S×S]
 O = PV    → read P
 memory O(S²), memory-bound; S=8k, h=32 → 8k²·32·2 B ≈ 4 GB per layer per sequence
```

### Fix: tile + online softmax + recompute
```
        K, V blocks (j) ─────────────►
   Q    ┌────┬────┬────┬────┐
 blocks │ ▓▓ │ ▓▓ │    │    │   for each Q block i (in SRAM):
  (i)   ├────┼────┼────┼────┤     for each K/V block j:
   │    │    │    │    │    │       S_ij = Q_i K_jᵀ        (SRAM only)
   ▼    └────┴────┴────┴────┘       update running max m, sum l, output O
                                   O_i = O_i / l → write once to HBM
   never materialize S×S in HBM · causal → skip blocks above diagonal
```

### Online softmax ★ (math)
for each new block `S_j` (row-wise):
```
 m_new = max(m, rowmax(S_j))
 l_new = e^(m − m_new)·l + rowsum(e^(S_j − m_new))
 O_new = e^(m − m_new)·O + e^(S_j − m_new)·V_j
 final: O = O / l ;  also save L = m + log l  (logsumexp) for backward
```
- rescaling by `e^(m − m_new)` fixes earlier partial results when max changes

### Backward
- store only O and L (O(S) memory), **recompute** P blockwise from Q, K, L
- more FLOPs, but far less HBM IO → faster overall
- `D = rowsum(dO ⊙ O)` trick → `dS = P ⊙ (dP − D)`

### Versions
- FA1: tiling + recompute
- **FA2**: parallelize over sequence blocks too, fewer non-matmul FLOPs, better warp work split → **A2 implements forward in Triton**
- FA3: Hopper features (async TMA loads, warp specialization), fp8

## 6. Triton ★

| term | definition |
|---|---|
| Triton | Python DSL (OpenAI) for GPU kernels; program **blocks**, not threads |
| program / `program_id` | one instance = one block of work (≈ CUDA thread block) |
| `tl.arange` | vector of offsets in the block |
| `tl.load / tl.store` | read/write HBM with mask (bounds) |
| `tl.dot` | block matmul on tensor cores |
| `tl.constexpr` | compile-time constant (block sizes) |
| autotune | try configs (block sizes, warps), pick fastest |
| block pointer | `tl.make_block_ptr` → 2-D tile loads with strides |

```python
import triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)                     # which block am I
    offs = pid * BLOCK + tl.arange(0, BLOCK)    # my element indices
    mask = offs < n                             # stay in bounds
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

grid = (triton.cdiv(n, 1024),)
add_kernel[grid](x, y, out, n, BLOCK=1024)
```

- CUDA vs Triton: CUDA = per-thread control, max perf, more work · Triton = per-block, compiler handles coalescing/SMEM · `torch.compile` = auto-generates Triton kernels for fused ops
- wrap kernel in `torch.autograd.Function` (forward + backward) to use in model

## 7. Benchmarking + profiling (A2 part 1) ★
- benchmark: warmup steps → `torch.cuda.synchronize()` → time N steps → mean ± std
- vary: model size, context length, fwd vs fwd+bwd, fp32 vs bf16
- **nsys** timeline: which kernels dominate (usually matmuls `sm90_xmma…` / `gemm`), softmax, elementwise
- NVTX annotate: `with torch.cuda.nvtx.range("attention"):`
- memory profile: `torch.cuda.memory._record_memory_history()` → `pytorch.org/memory_viz`
- observations to expect:
  - attention softmax: few FLOPs, big time → memory-bound
  - bf16 speedup large for matmuls, small for elementwise
  - backward ≈ 2× forward

## 8. Activation checkpointing (A2)
- save only some layer inputs; recompute rest during backward
- per-layer checkpoint: memory `O(L)` → store one tensor per layer; √L-segment trick → `O(√L)`
- `torch.utils.checkpoint.checkpoint(fn, x)`
