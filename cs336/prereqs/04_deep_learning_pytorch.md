# A4 · Deep Learning + PyTorch

> goal: build/train nets in PyTorch · know what lives in memory · ready for CS336 A1 (no `nn.Linear` allowed there!)

---

## 1. Building blocks

| term | definition |
|---|---|
| neuron | `σ(wᵀx + b)`: weighted sum + nonlinearity |
| layer | many neurons in parallel = `σ(Wx + b)` |
| MLP | stack of layers; "feed-forward network" (FFN) |
| activation fn | nonlinearity; without it, stacked linears = one linear |
| logits | raw scores before softmax |
| embedding | lookup table `E ∈ ℝ^{V×d}`; token id → row vector |
| forward pass | compute output from input |
| backward pass | compute gradients via chain rule (backprop) |
| autograd | framework records ops, computes backward automatically |

### Activations

| name | formula | notes |
|---|---|---|
| ReLU | `max(0, x)` | simple, dead neurons possible |
| GELU | `x·Φ(x)` (Φ = Gaussian CDF) | GPT-2/BERT |
| SiLU / Swish | `x·σ(x)` | used inside SwiGLU (Llama) |
| softmax | `eˣⁱ/Σeˣʲ` | output probabilities |

---

## 2. Making deep nets trainable

| trick | what | why |
|---|---|---|
| init (Xavier / Kaiming) | weights ~ N(0, ~1/fan_in) | keep activation variance stable through depth |
| residual connection ★ | `x + f(x)` | gradient highway; enables 100+ layers |
| LayerNorm | normalize each token's vector: subtract mean, divide std, scale γ, shift β | stable activations |
| RMSNorm | `x / RMS(x) · γ`, no mean, no β | cheaper; modern LLM default |
| BatchNorm | normalize across batch | ⚠ not used in LLMs (batch-dependent) |
| dropout | zero random activations, prob p | regularization; ⚠ usually off in LLM pretraining |
| grad clipping | rescale grad if `‖g‖ > c` | prevents loss spikes |

```
 residual stream view (transformer):
   x ─────────────(+)──────────────(+)────► out
        └─► f₁(x) ─┘     └─► f₂(·) ─┘
   each sublayer READS from and ADDS to the stream
```

---

## 3. Sequence modeling, briefly
- RNN: process tokens one by one; hidden state carries memory; ⚠ sequential → slow on GPU, vanishing grads
- attention: every token looks at all previous tokens directly; parallel over sequence ✓
- transformer = attention + MLP + residuals + norms → full detail in [CS336 B3](../cs336/03_transformer_architecture.md)

---

## 4. PyTorch essentials

| concept | what to know |
|---|---|
| `torch.Tensor` | n-D array + `dtype` + `device` + (optional) grad |
| dtype | `float32` (4 B), `bfloat16` (2 B), `float16` (2 B), `int64` (8 B) |
| device | `cpu`, `cuda:0`, `mps`; `x.to(device)` copies |
| `requires_grad=True` | autograd tracks ops on it |
| `loss.backward()` | fills `.grad` of every leaf param |
| `nn.Parameter` | tensor registered as learnable inside `nn.Module` |
| `nn.Module` | container: `__init__` defines params, `forward` computes |
| `torch.no_grad()` | disable tracking (eval, inference) → saves memory |
| `model.train()/eval()` | toggles dropout etc. |
| view vs copy | `view`, `transpose`, slicing = no copy (share storage) |
| stride | step in memory per dim; transpose just swaps strides |
| `.contiguous()` | materialize copy in row-major order (needed after transpose for `view`) |
| `torch.compile` | JIT-fuses ops into faster kernels |

### ★ Tensor memory
- bytes = `numel × bytes_per_element`
- `[32, 2048, 4096]` in bf16 → `268M × 2 B ≈ 537 MB`
- GPU memory at train time = params + grads + optimizer state + **activations** (saved for backward)

### Minimal training loop
```python
model = MyLM().to("cuda")
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

for step, (x, y) in enumerate(loader):          # x, y: [B, S] token ids
    x, y = x.cuda(), y.cuda()
    logits = model(x)                           # [B, S, V]
    loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
    opt.zero_grad(set_to_none=True)             # clear old grads
    loss.backward()                             # backprop
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()                                  # update θ
```

### Custom module (CS336 A1 style: write your own Linear)
```python
class Linear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        std = (2 / (d_in + d_out)) ** 0.5
        self.W = nn.Parameter(torch.nn.init.trunc_normal_(
            torch.empty(d_out, d_in), std=std, a=-3*std, b=3*std))
    def forward(self, x):            # x: [..., d_in]
        return x @ self.W.T          # [..., d_out]
```

### ⚠ GPU timing gotcha
- CUDA ops are **asynchronous**: Python returns before GPU finishes
- correct timing: warm up a few iterations → `torch.cuda.synchronize()` → start timer → run → `synchronize()` → stop

---

## 5. Debugging checklist
- overfit a single batch → loss should → ~0
- check initial loss ≈ `ln(V)` (uniform guess); V=50k → ~10.8
- shapes printed at every stage
- NaN? → lower LR, check for `log(0)`, use stable softmax, check fp16 overflow
- loss flat? → LR too small, grads zero (detached graph), labels shifted wrong
- ⚠ LM targets = inputs shifted by one: `x = tokens[:-1]`, `y = tokens[1:]`
