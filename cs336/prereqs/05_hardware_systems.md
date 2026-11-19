# A5 · Hardware + Systems Basics (the ML infra core)

> goal: know where data lives, how fast it moves, how fast math happens → predict if code is compute- or memory-bound

---

## 1. Units ★

| term | meaning |
|---|---|
| FLOP | one floating-point op (add or mul) |
| FLOPs | count of ops (amount of work) |
| FLOP/s (FLOPS) | ops per second (speed of hardware) |
| prefixes | K 10³ · M 10⁶ · G 10⁹ · T 10¹² · P 10¹⁵ · E 10¹⁸ |
| bandwidth | bytes/second moved (GB/s, TB/s) |
| latency | time for one thing to finish |
| throughput | things finished per second |
| GB vs GiB | 10⁹ vs 2³⁰ bytes (~7% diff) |

⚠ "FLOPs" (count) vs "FLOPS" (rate) → write FLOP/s to be safe

---

## 2. CPU vs GPU

| | CPU | GPU |
|---|---|---|
| cores | few (8–128), powerful | thousands, simple |
| optimized for | latency, branching logic | throughput, same op on lots of data |
| memory | DDR DRAM, large (100s GB), ~100 GB/s | HBM, smaller (80–192 GB), 2–8 TB/s |
| model | SISD/SIMD | SIMT (single instruction, multiple threads) |

→ matmul = massively parallel → GPUs win by 100×+

---

## 3. Memory hierarchy ★ (H100-ish numbers, approx)

```
            fast, tiny, close to compute
                 ▲
     ┌───────────────────────┐
     │      registers        │  ~256 KB / SM     per-thread, ~1 cycle
     ├───────────────────────┤
     │  L1 / shared mem (SMEM)│  ~228 KB / SM    programmer-managed SRAM, ~10s TB/s total
     ├───────────────────────┤
     │         L2            │  ~50 MB           shared by all SMs
     ├───────────────────────┤
     │   HBM (GPU DRAM)      │  80 GB, ~3.35 TB/s   ← "GPU memory"
     ├───────────────────────┤
     │  NVLink (other GPU)   │  ~450 GB/s per direction
     ├───────────────────────┤
     │  CPU DRAM via PCIe    │  ~64 GB/s (PCIe 5 x16)
     ├───────────────────────┤
     │  network (InfiniBand) │  ~50 GB/s per NIC (400 Gb/s)
     ├───────────────────────┤
     │  disk / object store  │  ~GB/s
     └───────────────────────┘
                 ▼
            slow, huge, far
```

- golden rule: **move data as little as possible**; reuse what's in SRAM
- most "optimization" = reducing HBM reads/writes (fusion, tiling, FlashAttention)

---

## 4. GPU anatomy (NVIDIA)

```
 GPU ─┬─ SM 0 ─┬─ tensor cores (matmul units, bf16/fp8)
      │        ├─ CUDA cores (fp32 scalar math)
      │        ├─ registers + shared memory (SRAM)
      │        └─ warp schedulers
      ├─ SM 1 …
      ├─ …  (H100: 132 SMs · A100: 108)
      ├─ L2 cache (shared)
      └─ HBM stacks (off-chip DRAM)
```

| term | definition |
|---|---|
| SM (streaming multiprocessor) | independent compute unit; like a "core" |
| tensor core | hardware unit doing small matrix multiply-accumulates; ~15× faster than CUDA cores for matmul |
| kernel | function launched on GPU (e.g. one matmul) |
| thread | one execution lane |
| warp | 32 threads executing same instruction in lockstep |
| thread block | group of threads (≤1024) on one SM; share SMEM |
| grid | all blocks of one kernel launch |
| occupancy | how many warps active per SM; hides latency |
| warp divergence | `if` branches split warp → serialized → slow |
| memory coalescing | adjacent threads read adjacent addresses → one big efficient transaction |
| kernel launch overhead | ~µs per launch → many tiny kernels = slow |
| wave quantization | # blocks not a multiple of # SMs → last "wave" underused |

```
 grid
 ├── block(0,0) ──► runs on some SM  ── 32-thread warps ── threads
 ├── block(0,1)
 └── …
```

⚠ tensor cores need dims multiple of 8/16 (bf16) → pad vocab/hidden to multiples of 64/128

---

## 5. TPU (Google) basics

| term | definition |
|---|---|
| TPU | Google's ML ASIC (application-specific chip) |
| MXU | matrix unit = **systolic array** (e.g. 128×128 or 256×256 MACs) |
| systolic array | grid of MAC cells; data pulses through, each cell multiplies + passes partial sums |
| VMEM | on-chip SRAM scratchpad (like SMEM) |
| ICI | inter-chip interconnect; chips wired in 2D/3D torus |
| pod / slice | thousands of TPUs linked by ICI |
| XLA | compiler: whole graph → fused TPU program; needs **static shapes** |
| JAX | NumPy-like Python lib → XLA; `jit`, `pmap`, `shard_map` |

```
 systolic array (weights stay, inputs flow →, partial sums flow ↓)
   x₀ → [w][w][w][w]
   x₁ →  [w][w][w][w]
   x₂ →   [w][w][w][w]
           ↓  ↓  ↓  ↓   outputs
```

| GPU vs TPU | GPU | TPU |
|---|---|---|
| flexibility | general-purpose, CUDA | narrower, compiler-driven |
| scale-out | NVLink in node (8–72), IB between | ICI torus across whole pod |
| programming | PyTorch, CUDA, Triton | JAX/XLA, Pallas |

---

## 6. Peak specs (approx, dense, no sparsity)

| chip | BF16 TFLOP/s | memory | mem BW | notes |
|---|---|---|---|---|
| A100 80GB | 312 | 80 GB HBM2e | ~2.0 TB/s | NVLink 600 GB/s |
| H100 SXM | ~989 | 80 GB HBM3 | ~3.35 TB/s | FP8 ~1979; NVLink 900 GB/s |
| B200 | ~2,200 | ~180–192 GB HBM3e | ~8 TB/s | FP8 ~4.5 P; NVLink 1.8 TB/s |
| TPU v5e | ~197 | 16 GB | ~0.8 TB/s | cost-efficient |
| TPU v5p | ~459 | 95 GB | ~2.8 TB/s | training |
| TPU v6e (Trillium) | ~918 | 32 GB | ~1.6 TB/s | |

⚠ marketing numbers often quote **sparsity** (2×) → halve them
⚠ fp32 non-tensor-core FLOP/s ≪ bf16 tensor core (H100 ~67 vs ~989)

---

## 7. Arithmetic intensity + roofline ★★

- **arithmetic intensity (AI)** = FLOPs / bytes moved from HBM
- **machine balance** = peak FLOP/s ÷ mem BW → H100: 989e12 / 3.35e12 ≈ **~295 FLOPs/byte**
- AI < balance → **memory-bound** (waiting on data)
- AI > balance → **compute-bound** (good: using tensor cores)

```
 attainable
 FLOP/s  │                 ____________________  peak compute
         │               /
         │             /   ← slope = memory bandwidth
         │           /
         │         /  memory-bound | compute-bound
         │       /                 |
         └──────────────────┬──────────────────── arithmetic intensity
                         ridge ≈ 295 (H100)
```

| op (bf16) | FLOPs | bytes | AI | regime |
|---|---|---|---|---|
| elementwise `relu(x)`, n elems | n | 2n read + 2n write | 0.25 | memory |
| mat-vec (m×n)·(n) | 2mn | ≈2mn | ≈1 | memory (→ LLM decode!) |
| matmul n×n×n | 2n³ | 3·2n² | n/3 | compute if n ≳ 900 |
| LayerNorm / softmax | few/elem | 4/elem | ~1 | memory |

→ in transformers: matmuls compute-bound; norms, activations, softmax, residual adds memory-bound → **fuse them**

### Three regimes (Horace He, "Making DL go brrr")
1. **compute-bound** → use tensor cores, lower precision
2. **memory-bound** → fuse ops, avoid round trips to HBM
3. **overhead-bound** → Python/launch overhead; CUDA graphs, `torch.compile`, bigger batches

---

## 8. Number formats ★

```
 format   sign exponent mantissa  bytes  max        notes
 fp32      1     8        23        4    3.4e38    default, precise
 fp16      1     5        10        2    65,504    ⚠ overflows → needs loss scaling
 bf16      1     8         7        2    3.4e38    same range as fp32, low precision → LLM default
 fp8 E4M3  1     4         3        1    448       fwd activations/weights
 fp8 E5M2  1     5         2        1    57,344    gradients
 int8      —     —         —        1    ±127      inference quantization
 int4      —     —         —      0.5    ±7        weight-only quant
```

- exponent bits → **range** · mantissa bits → **precision**
- mixed precision: compute in bf16, keep master weights + optimizer state in fp32
- ⚠ accumulations (sums, softmax denominators, reductions) → do in fp32

---

## 9. Interconnect (multi-GPU)

```
 ┌─────────────── node (8 GPUs) ───────────────┐        ┌──── node ────┐
 │ GPU─GPU─GPU─GPU─GPU─GPU─GPU─GPU             │  IB /  │              │
 │   └──── NVSwitch (all-to-all NVLink) ────┘  │◄─────► │  8 GPUs ...  │
 │ CPU ── PCIe ── GPUs, NICs                   │ RoCE   │              │
 └─────────────────────────────────────────────┘        └──────────────┘
   NVLink ~450–900 GB/s                           ~50 GB/s per GPU NIC
```

| link | where | BW (approx) |
|---|---|---|
| PCIe 5 x16 | CPU ↔ GPU | ~64 GB/s |
| NVLink 4 (H100) | GPU ↔ GPU in node | 900 GB/s total |
| NVSwitch | connects all GPUs in node | full NVLink speed any pair |
| InfiniBand NDR | node ↔ node | 400 Gb/s ≈ 50 GB/s per NIC |
| NVL72 (GB200) | 72 GPUs in one NVLink domain | big TP/EP domains |

→ rule: bandwidth-hungry parallelism (tensor parallel) **inside** a node; lighter (data/pipeline) **across** nodes

### Collective operations (NCCL) — preview of [B6](../cs336/06_parallelism.md)

| collective | before → after |
|---|---|
| broadcast | one rank's data → all ranks |
| reduce | sum from all → one rank |
| **all-reduce** | sum from all → all ranks have the sum |
| **all-gather** | each has a shard → all have full tensor |
| **reduce-scatter** | sum, then each keeps one shard of result |
| all-to-all | each sends distinct piece to each other (MoE routing) |

- **all-reduce = reduce-scatter + all-gather**
- NCCL = NVIDIA's collective comms library; `torch.distributed` wraps it

---

## 10. Profiling basics
- wall-clock timing w/ `torch.cuda.synchronize()` + warmup
- `torch.profiler` → per-kernel time table
- NVIDIA Nsight Systems (`nsys`) → timeline of kernels, comms, CPU gaps
- NVTX ranges → label regions in timeline
- look for: idle GPU gaps (overhead-bound), many small kernels (fuse), comm not overlapped with compute
