# ARCHITECTURE_AUDIT.md — ByteCore-DualMemory, Phase 24 code as it exists

**Scope.** Bottom-up description of the code in this checkout, from a scalar
multiply to the training loop. Every formula is transcribed from the
implementation it names. Numbers are labelled **measured** (executed here,
CPU, PyTorch 2.15, float32) or **analytic** (derived, not benchmarked — this
environment has no GPU).

**Files audited.** `models/foundation.py`, `models/memory_model.py`,
`models/encoders.py`, `training/train.py`, `data/sources.py`,
`configs/t4_scaled.yaml`, `pretrain_4h.ipynb`.

**Notation.**

| Symbol | Production value | Meaning |
|---|---|---|
| `B` | 4 (micro-batch) | sequences per step |
| `T` | ≤ 4096 (train window 1024) | bytes per sequence |
| `D` | 1024 | core width (`d_model`) |
| `M` | 8192 | skill/memory width (`memory_dim`) |
| `N` | active slots | memory slots visible to a query |
| `H / H_kv` | 8 / 2 | query / key-value heads (GQA) |
| `F` | 4096 | SwiGLU hidden width |
| `V` | 265 | vocabulary (256 bytes + 9 control ids) |

## 1. Vocabulary (`models/foundation.py`)

Bytes `0..255` are their own ids. Control ids: `BOS=256, EOS=257, PAD=258,
TOOL_SEARCH=259, TOOL_SEARCH_END=260, OBS_START=261, OBS_END=262, EDIT=263,
EDIT_END=264`. `encode_bytes` is a **pure** byte→id map: the literal text
`[TOOL_SEARCH]` stays eight ordinary bytes; only id 259 means a tool call.
Special ids are inserted programmatically (data builders, tool runtime),
never parsed out of text. `decode_ids` drops control ids (or renders their
names with `keep_controls=True`). There is **no regex anywhere in the
runtime** — enforced by `tests/test_tools.py` and `tests/test_repo_hygiene.py`.

## 2. Core block (`TransformerBlock`)

Pre-norm RMSNorm (`x / sqrt(E[x²]+ε) · w`, reduction in float32), then:

```
h ← h + Attention(RMSNorm(h))            # GQA + RoPE, fused SDPA, causal
h ← h + SwiGLU(RMSNorm(h))               # [SiLU(xW_v) ⊙ (xW_b)] W_o
h ← h + GatedSkillAdapter(RMSNorm(h), memory)   # §3, ALL 8 layers
```

Attention: `q: D→D`, `k,v: D→H_kv·d_h = 256`, `o: D→D`; RoPE rotates q/k with
θ=10000; causality via `scaled_dot_product_attention(is_causal=True)` — no
`(B,H,T,T)` materialisation. SwiGLU: `3·D·F = 12,582,912` params/layer.

## 3. The Gated Skill Adapter (`GatedSkillAdapter`) — the Phase 24 core

Exact tensor flow (production widths):

```
hidden      (B, T, 1024)
query       q = W_q · hidden,  W_q ∈ R^{1024×1024}          (B, T, 1024)
keys        K ∈ R^{N×1024}   (or (B, N, 1024) batch-local)
scores      s = q Kᵀ / √1024                                 (B, T, N)
mask        inactive slots → −inf; all-masked rows → weight 0 (never NaN)
weights     α = softmax(s)                                   (B, T, N)
retrieved   r = α · V,  V ∈ R^{N×8192}                       (B, T, 8192)
GLU         g = W_g · r,  W_g ∈ R^{2048×8192}                (B, T, 2048)
split       (I, Φ) = chunk(g, 2)                             (B, T, 1024) each
output      δ = scale · (I ⊙ σ(Φ))                           (B, T, 1024)
injection   h ← h + δ
```

**Why a GLU and not `Linear(8192→1024)`.** A dense projection mixes every
skill dimension into every output channel — concept soup. The GLU gives each
output channel a multiplicative sigmoid gate: `σ(Φ)≈0` vetoes the channel,
`σ(Φ)≈1` passes `I` intact. Selection, not blending. The 8192-dim value is
retrieved **uncompressed**; the only width reduction happens inside the gate.

**`scale` is a learned scalar initialised to 0** (ReZero/LayerScale): at step
0 the bridge is exactly the identity, so enabling it cannot destabilise a
fresh model. `∂L/∂scale = ⟨glu_out, ∂L/δ⟩ ≠ 0`, so it opens from the first
step; projector gradients flow once it is open. Consequence worth stating:
with a **single** visible slot, `softmax([s]) = [1]` is constant and the key
projector receives zero gradient — the trainer therefore merges the
persistent bank into training views so keys face distractors
(`BatchBuilder.pop_batch`).

**Parameter census (measured, meta-device construction):**

| Component | Params |
|---|---|
| Adapter per layer (`W_q` 1,048,576 + `W_g` 16,777,216 + scale 1) | 17,825,793 |
| 8 adapters | 142,606,344 |
| Core without adapters | 123,012,366 |
| **Total** | **265,618,710** |

The analytic count is asserted exactly by
`tests/test_gated_adapter.py::test_injection_in_all_eight_production_layers`.

**Latency (measured, CPU fp32, N=64 slots):** B=1,T=1: 4.18 ms; B=4,T=64:
82.5 ms per adapter call. GPU numbers require the T4 run.

## 4. DualMemory (`models/memory_model.py`)

Two banks, one geometry: **keys `(capacity, 1024)`**, **values
`(capacity, 8192)`**, `used/written_at/modality` bookkeeping. Production
capacity 1024 + 1024 slots → **75.5 MB** of fp32 buffers (measured byte
count: 2 × 1024 × (1024+8192) × 4 = 75,497,472 B).

* `ProceduralSkillBank.write` is `@torch.no_grad()`: tensors are **detached**
  on write; memory is a store, not a gradient update. Eviction is FIFO
  (oldest `written_at`), deterministic and auditable.
* `SkillBankView` is the read interface: global `(N, ·)` for persistent
  banks, batch-local `(B, N, ·)` for freshly projected matrices that still
  carry gradients. `stack_views` concatenates along slots, broadcasting
  global→batched.
* `DualMemory` = episodic events + procedural skills behind one `view()`.
  Persistence: `save/load` with schema version, geometry check, metadata.

## 5. Multimodal ingestion (`models/encoders.py`)

```
raw media ──FrozenEncoder──▶ features (B, E)     [no grad, eval forever]
features ──SkillProjector──▶ value = Linear(E, 8192)   [trainable]
features ──SkillKeyProjector▶ key  = Linear(E, 1024)   [trainable]
(key, value) ──write──▶ ProceduralSkillBank      [detached, inference]
(key, value) ──batch_view──▶ training view       [differentiable]
```

Registry: `resnet50` (E=2048), `vit_b_16` (E=768, production default — its
768-byte codec payload fits a 1024 context), `wav2vec2` (E=768, mean-pooled
over time), video = sampled frames through the image encoder, mean-pooled to
one matrix. `FrozenEncoder` overrides `train()` so the backbone never leaves
eval; forward runs under `no_grad`. **Matrix agnosticism** is tested: two
encoders of different widths feed the same bank + adapter unchanged.

**Uint8 matrix codec** (the "media bytes" of bidirectional training):
`byte = round((tanh(x)+1)·127.5)`; inverse `atanh(byte/127.5 − 1)` clamped to
±3. Deterministic; measured roundtrip error on N(0,1) features: mean 0.0076,
p99 0.074. For |x| ≤ 1.5 the worst-case error is ≤ 0.022 (test-asserted).
This is a lossy representation codec by design — the model predicts
representations, not pixels.

**Bidirectional pairs (`BatchBuilder`):** each multimodal sample yields
* A) *media→text*: `BOS "caption: " caption EOS`, prompt targets masked,
  conditioned on the media matrix;
* B) *text→media*: `BOS "matrix: " caption "\n" codec_bytes EOS`, only codec
  targets unmasked, conditioned on the **text** matrix
  (`TextSkillProjector(mean core embeddings)`).
Both directions use the same byte vocabulary and the same cross-entropy.
Text-only rows get mask=False slots → exact-zero retrieval, no leakage.

## 6. Adaptive loop (`IterativeRefinement`)

```
h_{t+1} = h_t + α_t · Cell(RMSNorm(h_t)),   Cell = Up(SiLU(Down x) ⊙ Branch x)
conf(h) = σ(W_c · RMSNorm(h) + b_c) ∈ (0,1) per position
```

`α` zero-init (identity at start); rank 256 bottleneck (~0.79M params incl.
the confidence head). **Inference:** ≤ 4 passes, break when
`min_positions conf > 0.9`. **Training:** all 4 passes run; auxiliary target
per state `y_t = 1[H(p_t)/log V < 0.35]` where `p_t = softmax(head(norm(h_t)))`,
**detached** — the head learns to *read* certainty, never to manufacture it.
`L = CE + 0.1 · mean_t BCE(conf(h_t), y_t)` over non-PAD positions.

Measured (8-step CPU smoke): CE 5.596 → 5.382, confidence BCE 0.6931 →
0.6798, both directions built (6 media→text + 6 text→media), 6 matrices
ingested, resume bit-exact at step 4/8. Tests force early break (1 step) and
full loop (4 steps) via the head bias, and verify the BCE target equals the
analytic low-entropy fraction to 1e-4.

## 7. True tooling (`generate_with_tools` + `run_search`)

Detection is **by token id**: on `TOOL_SEARCH … TOOL_SEARCH_END` the decoded
query bytes go to `search_fn`; the result returns as `OBS_START bytes OBS_END`
and generation continues (≤ 4 tool calls, observations truncated to 2048 B).
`run_search` executes in a subprocess: `python -I -S` for the deterministic
mock (no site-packages, no env), `python -I` for real `ddgs`; rlimits CPU 5 s,
AS 768 MB, FSIZE 1 MB, NOFILE 64; wall-clock timeout; temp-dir cwd. Timeouts
and failures degrade to parenthesised byte messages, never exceptions.
`EDIT`/`EDIT_END` have **no host handler** — they are trained by the same CE
as every byte (test-asserted: head gradient rows for id 263 are non-zero).
`inject_tool_trajectories` sprinkles token-id trajectories into the real
stream at `tool_trajectory_rate=0.02`: queries are drawn from corpus bytes,
observations come from the actual sandboxed mock search, and the sequences
enter training as `kind="ids"` — no legacy synthetic generator involved.

## 8. Training (`training/train.py`)

One loop: AdamW (β=0.9/0.95, wd 0.1), cosine LR 3e-4 with 500-step warmup,
grad-clip 1.0, micro-batch 4 × accum 8 (effective 32 sequences), gradient
checkpointing on blocks + refinement. Precision `auto`: bf16 on Ampere+,
**fp16 + GradScaler on T4/sm_75** (bf16 autocast raises there), fp32 on CPU —
resolved from `torch.cuda.is_bf16_supported()`, logged, never assumed.
Measured VRAM probe (real forward+backward) shrinks micro-batch 4→2→1 above
12 GB. Checkpoints: atomic tmp+rename, every 30 min **and** every 500 steps,
model + projectors + banks + optimizer + scaler + RNG; frozen encoders are
not persisted (rebuildable); resume recreates projector param groups in
order. Hub upload/download via `huggingface_hub`, failures reported as
`uploaded/downloaded: false` with a reason.

**Analytic T4 VRAM budget** (265.6M params; *not measured here*): fp16 weights
0.53 GB + fp32 master 1.06 GB + fp16 grads 0.53 GB + Adam moments 2.12 GB =
**4.25 GB** optimizer state; checkpointed activation boundaries ≈ 0.067 GB;
one live retrieved `(4,1024,8192)` fp16 tensor ≈ 0.067 GB. Comfortably inside
14 GB; the probe verifies on the real device.

## 9. Data (`data/sources.py`)

Streaming only. Weights 0.40 code / 0.30 language / 0.20 multimodal / 0.10
logic; deterministic seeded sampling; exhausted/failed groups dropped and
weights renormalised (recorded in `failures`). Fallback chains per group;
dead COYO URLs are skipped, embedded MSCOCO-5k and LibriSpeech keep the
multimodal fraction alive. `BytePacker` frames documents `BOS doc EOS` and
packs to fixed-length shifted pairs with carry — no byte dropped.

## 10. Verification (measured)

`python -m pytest tests/ -q` → **129 passed in ~3.3 s** (CPU). Coverage:
exact GLU math at production widths, key-matching fidelity (aligned query
retrieves its value to 1e-3), all-8-layer injection + exact parameter count,
masked/empty retrieval, projector gradient flow, bank FIFO/detach/persistence,
frozen-encoder contract, codec bounds, matrix agnosticism, adaptive
break/cap/targets, token-triggered sandboxed search + EDIT pass-through,
bidirectional batches, accumulation equivalence, checkpoint/resume,
time-based cadence, mixture ratios/fallbacks/packing, notebook validity,
and the cleanup contract itself.

**Known limits.** 4 h ≈ order 10⁸ tokens: a pre-training start, not a
finished model. The codec is lossy (p99 ≈ 0.074 on unit-normal features).
Single-slot views give the key projector zero gradient until distractors
exist (the trainer merges the bank). Real `ddgs` search needs a networked
runtime; training uses the deterministic mock. GPU throughput/VRAM remain
analytic until the notebook runs.
