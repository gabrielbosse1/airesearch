# ARCHITECTURE_AUDIT.md — ByteCore-DualMemory, Phase 22 code as it exists

**Scope.** This document describes the code in this repository at the Phase 23
cleanup commit, bottom-up, from a scalar multiply to the runtime loop. Nothing
here is a recommendation, a roadmap, or an estimate of quality. Every number was
produced by executing the code in this checkout on CPU (PyTorch 2.15, float32),
and every formula is transcribed from the implementation it names.

**Files audited.** `models/foundation.py`, `models/memory_model.py`,
`models/foundation_memory.py`, `models/react.py`, `models/sandbox.py`,
`models/holdout.py`, `models/procedural.py`, `models/contradiction_graph.py`,
`training/phase22_train.py`, `training/phase22_common.py`,
`training/phase22_reward.py`, `data/phase22_sources.py`,
`evaluation/final_agent_eval.py`, `configs/t4_scaled.yaml`.

**Primary configuration audited.** `configs/t4_scaled.yaml → model`
(the Phase 22 production preset). Test configurations use the same classes with
smaller integers; no code path differs.

**Notation.**

| Symbol | Value | Meaning |
|---|---|---|
| `B` | configurable (train default 4) | batch of byte sequences |
| `L` | ≤ `max_seq_len` = 2048 | bytes per sequence |
| `D` | 1024 | `d_model` |
| `H` | 8 | query heads |
| `H_kv` | 2 | key/value heads (GQA) |
| `d_h` | `D/H` = 128 | head width |
| `F` | 4096 | SwiGLU hidden width |
| `R` | 256 | refinement bottleneck rank |
| `N` | 12 | transformer layers |
| `V` | 261 | vocabulary size |
| `eps` | 1e-6 | RMSNorm epsilon (`norm_eps`) |

**Vocabulary is fixed and validated.** `FoundationConfig.__post_init__` raises
unless `vocab_size == 261`: bytes `0..255` are their own ids, then
`BOS=256`, `EOS=257`, `PAD=258`, `REF_START=259`, `REF_END=260`. There is no
tokenizer anywhere in the repository.

---

## Level 1 — Base Primitives (the math and the tensors)

### 1.1 Linear transformation

**Formula.** For weight `W ∈ R^{d_out×d_in}` and optional bias `b ∈ R^{d_out}`:

```
y = x Wᵀ + b            x ∈ R^{B×L×d_in}  →  y ∈ R^{B×L×d_out}
```

**Implementation.** `torch.nn.Linear` everywhere. Initialisation is set once in
`CausalByteTransformer._initialize`:

```python
nn.init.normal_(module.weight, mean=0.0, std=0.02)
if module.bias is not None: nn.init.zeros_(module.bias)
```

**Where it is used, with exact widths (production config):**

| Site | `d_in → d_out` | bias | count |
|---|---|---|---|
| `token_embedding` (`nn.Embedding`) | `261 → 1024` | — | 267,264 |
| `blocks[i].attention.query` | `1024 → 1024` | yes | 1,049,600 |
| `blocks[i].attention.key` | `1024 → 256` | yes | 262,400 |
| `blocks[i].attention.value` | `1024 → 256` | yes | 262,400 |
| `blocks[i].attention.output` | `1024 → 1024` | yes | 1,049,600 |
| `blocks[i].feed_forward.w_value` | `1024 → 4096` | no | 4,194,304 |
| `blocks[i].feed_forward.w_branch` | `1024 → 4096` | no | 4,194,304 |
| `blocks[i].feed_forward.w_out` | `4096 → 1024` | no | 4,194,304 |
| `refinement.down` / `.branch` | `1024 → 256` | no | 262,144 each |
| `refinement.up` | `256 → 1024` | no | 262,144 |
| `head` | `1024 → 261` | yes | 267,525 |

The head is **not** weight-tied to the embedding: both are 267,264-element
tensors and both are trained.

### 1.2 RMSNorm

**Formula.**

```
RMS(x) = sqrt( (1/D) Σ_d x_d² + eps )
RMSNorm(x) = ( x / RMS(x) ) ⊙ w ,      w ∈ R^D, no bias
```

**Implementation** (`models/foundation.py::RMSNorm.forward`), verbatim:

```python
dtype = hidden.dtype
values = hidden.float()
values = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + self.eps)
return (values * self.weight.float()).to(dtype)
```

**Shapes.** `x ∈ R^{B×L×D}` → `R^{B×L×D}`. Parameter count `D = 1024` per norm
site; there are 2 per block (24), 1 in `refinement.norm`, 1 as `norm_final`:
26,624 normalisation parameters in total.

**Numerical detail.** The reduction is always performed in `float32` and the
result cast back to the input dtype, because squaring `float16` activations at
`D = 1024` can overflow before the mean is taken. The test
`test_rmsnorm_is_scale_invariant_and_preserves_dtype` measures that
`RMSNorm(3x) = RMSNorm(x)` up to the `eps` term, and that output dtype equals
input dtype.

**Difference from LayerNorm, stated as arithmetic.** LayerNorm computes
`(x − μ)/sqrt(σ² + eps) ⊙ w + b` over the same axis: it subtracts the mean,
divides by the *centred* standard deviation, and has a bias. RMSNorm does
neither of the first two and has no bias, so it removes only the scale, keeps
`E[x]` free, and has `D` parameters instead of `2D`. `make_norm()` selects the
class from `config.norm_type`; the historical default `"layernorm"` still
constructs `nn.LayerNorm(D, eps=norm_eps)` for checkpoints that predate Phase 22.

### 1.3 SwiGLU feed-forward

**Formula.**

```
h  = x W_vᵀ                     (B, L, F)
g  = x W_bᵀ                     (B, L, F)
a  = SiLU(h) ⊙ g = h ⊙ sigmoid(h) ⊙ g
y  = a W_oᵀ                     (B, L, D)
```

**Implementation**, verbatim:

```python
activated = F.silu(self.w_value(hidden)) * self.w_branch(hidden)
return self.dropout(self.w_out(activated))
```

**Shapes.** `(B,L,1024) → (B,L,4096) → (B,L,4096) → (B,L,1024)`. All three
projections are bias-free (`bias=False`).

**Parameter count per layer.** `3·D·F = 3·1024·4096 = 12,582,912`. A GELU FFN of
the same hidden width would be `2·D·F = 8,388,608`; the third matrix is the
price of the gate. In this model the FFN is 12,582,912 of the 15,208,960
parameters in a block (82.7 %).

**Gating semantics.** The elementwise product is applied *after* the activation,
so the branch `g` is a multiplicative mask on the activated value stream — it
can suppress coordinates but cannot flip their sign, because
`SiLU(h) ∈ (−0.2785, ∞)` and the product inherits the sign of `g` only.

### 1.4 Rotary Position Embeddings (RoPE)

**Formula.** For head width `d_h = 128`, `half = d_h/2 = 64` and
`θ = rope_theta = 10,000`:

```
inv_i = θ^(−2i/d_h) = 1 / θ^(i/half)        i = 0 … half−1
a_{p,i} = p · inv_i
```

For a head vector `x ∈ R^{d_h}` at position `p`, the rotation pairs coordinate
`i` with coordinate `i + half`:

```
(x′_i      )   ( cos a_{p,i}   −sin a_{p,i} ) ( x_i      )
(x′_{i+half}) = ( sin a_{p,i}    cos a_{p,i} ) ( x_{i+half} )
```

which is exactly what `rotate_half` + elementwise multiply encode:

```python
def rotate_half(hidden):
    left, right = hidden.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)

def apply_rotary(hidden, cos, sin):
    return hidden * cos + rotate_half(hidden) * sin
```

with `cos = sin = cat(angles, angles)` so `cos/sin ∈ R^{1×1×L×d_h}`. The angles
are built as:

```python
inverse = 1.0 / (theta ** (torch.arange(0, half) / half))
angles = torch.outer(torch.arange(length), inverse)
```

**Shapes and cost.** Tables are `(1,1,L,d_h)` fp32, recomputed every forward
call, not stored: `2 · L · d_h / 2 = L·d_h = 262,144` floats at `L = 2048`
(≈1 MB for the pair) and **zero parameters**. `q` and `k` are rotated;
`v` is not.

**Property that matters.** Because each pair is a pure rotation, `‖q_p‖` and
`‖k_n‖` are invariant, and
`⟨R(p)q, R(n)k⟩ = ⟨q, R(n−p)k⟩` depends only on `n − p`. The test
`test_rope_dot_product_depends_only_on_the_relative_offset` asserts
`dot(2,7) == dot(9,14) ≠ dot(2,8)`. The `rope` preset therefore needs no
position table at all: `position_embedding` is `None`, whereas the `learned`
preset allocates `max_seq_len × D = 2048 × 1024 = 2,097,152` parameters.

### 1.5 Grouped-Query Attention (GQA)

**Grouping.** `n_heads = 8`, `n_kv_heads = 2` ⇒ `repeats = H / H_kv = 4`. Each
KV head is shared by 4 query heads.

**Projections and shapes.**

```
q = W_q x   → (B, L, 1024) → view(B,L,H,d_h)   → transpose → (B, 8, L, 128)
k = W_k x   → (B, L,  256) → view(B,L,H_kv,d_h)→ transpose → (B, 2, L, 128)
v = W_v x   → (B, L,  256) → view(B,L,H_kv,d_h)→ transpose → (B, 2, L, 128)
```

**Attention.**

```
scores = q kᵀ / sqrt(d_h)                     (B, 8, L, L)   [never materialised]
mask   = j > i   (strictly upper triangle)    → −inf
A      = softmax(scores + mask)
ctx    = A v                                  (B, 8, L, 128)
out    = W_o (ctx → (B, L, 1024))
```

**Implementation.** `F.scaled_dot_product_attention(q, k, v, is_causal=True,
dropout_p=p, enable_gqa=True)` with a `repeat_kv` fallback for torch < 2.5 that
expands `(B,2,L,128) → (B,8,L,128)` by index repetition. The fused kernel is
used so that the `(B,8,L,L)` score tensor is never allocated; the module *also*
registers a boolean upper-triangular buffer `causal_mask` of shape
`(max_seq_len, max_seq_len) = (2048, 2048)`, `persistent=False` (4,194,304
bytes), which is passed to every block but **is not consumed** by the kernel —
causality is enforced by `is_causal=True`. The buffer exists so tests and callers
can assert the property independently of the kernel.

**Parameter budget.** `q: 1024·1024 + 1024 = 1,049,600`;
`k: 256·1024 + 256 = 262,400`; `v: 262,400`; `o: 1,049,600` ⇒ **2,624,000 per
layer**, and `K`+`V` are `2·262,400 = 524,800`, i.e. 25 % of the
`2,099,200` a full 8-head MHA would spend on the same two projections.

**Causality, measured.** `test_attention_is_causal` feeds two sequences that
share the first 4 bytes and differ afterwards; logits at positions `0..3` are
bit-identical (`atol=1e-6`) and position 5 differs.

---

## Level 2 — Building Blocks (arranging the primitives)

### 2.1 The pre-norm SwiGLU block

**Formula**, for block input `x ∈ R^{B×L×D}`:

```
x₁ = x + Attention(RMSNorm₁(x))            (B, L, 1024)
x₂ = x₁ + SwiGLU(RMSNorm₂(x₁))             (B, L, 1024)
```

**Implementation**, verbatim:

```python
def forward(self, hidden, mask=None):
    hidden = hidden + self.attention(self.norm_1(hidden), mask)
    hidden = hidden + self.feed_forward(self.norm_2(hidden))
    return hidden
```

**Shapes through one block (production config, `B=4`, `L=1024`):**

| Stage | Tensor | Elements | fp16 bytes |
|---|---|---|---|
| block input | `(4, 1024, 1024)` | 4,194,304 | 8.4 MB |
| `norm_1` | `(4, 1024, 1024)` | 4,194,304 | 8.4 MB |
| `q` | `(4, 8, 1024, 128)` | 4,194,304 | 8.4 MB |
| `k`, `v` | `(4, 2, 1024, 128)` each | 1,048,576 each | 2.1 MB each |
| attention context | `(4, 1024, 1024)` | 4,194,304 | 8.4 MB |
| `norm_2` | `(4, 1024, 1024)` | 4,194,304 | 8.4 MB |
| `w_value`, `w_branch` | `(4, 1024, 4096)` each | 16,777,216 each | 33.6 MB each |
| `w_out` | `(4, 1024, 1024)` | 4,194,304 | 8.4 MB |

Order is fixed: **norm → attention → residual → norm → FFN → residual**. Norms
are pre-block, not post-block; residual streams carry unnormalised states
throughout. With `gradient_checkpointing: true`, only the block **inputs** are
retained and each block is recomputed in backward via
`torch.utils.checkpoint(block, hidden, self.causal_mask, use_reentrant=False)`;
retained boundary memory is one `(B,L,D)` tensor per block (8.4 MB fp16 at
`B=4,L=1024`), i.e. 12 × 8.4 MB ≈ 100 MB instead of the ≈1.5 GB implied by
storing every tensor listed above (≈122 MB per block).

### 2.2 The iterative refinement loop ("latent thinking")

**Formula.** With `steps = refinement_steps = 2`, `R = 256`, shared parameters:

```
h₀ = blocks(x)
for t in 0 … steps−1:      h_{t+1} = h_t + α_t · Cell(RMSNorm(h_t))
Cell(z) = W_up ( SiLU(W_down z) ⊙ W_branch z )
logits = W_head ( RMSNorm_final(h_steps) )
```

`α ∈ R^{steps}` (`step_scales`) is a free parameter initialised to **zero**, so
at step 0 the loop is exactly the identity map: `h_{t+1} = h_t`. Every pass
re-runs the *same* cell; there is no per-step parameter set and no extra
Transformer layer.

**Implementation**, verbatim:

```python
def cell(self, hidden):
    values = self.norm(hidden)
    return self.up(F.silu(self.down(values)) * self.branch(values))

def forward(self, hidden, *, use_checkpoint=False):
    for index in range(self.steps):
        alpha = self.step_scales[index]
        if use_checkpoint and self.training and torch.is_grad_enabled():
            delta = checkpoint(self.cell, hidden, use_reentrant=False)
        else:
            delta = self.cell(hidden)
        hidden = hidden + alpha * delta
    return hidden
```

**Shapes.** All tensors are `(B, L, 1024)`; the bottleneck is `(B, L, 256)`.
Parameters: `3 · D · R = 786,432` + one RMSNorm (`1024`) + `steps`
(`2`, one scalar per pass) = **787,458**. Holding `D` and `R` fixed, the
parameter count is independent of `steps`; only `step_scales` grows.

**Measured behaviour.** At initialisation the loop returns its input bit-exactly
(`test_iterative_refinement_is_weight_tied_and_identity_at_initialisation`);
with `step_scales = [0.5, 0.25, 0.125]` the output equals the three hand-computed
updates within `1e-6`. Because `α` starts at 0, the loop can only leave the
identity through gradient flow into `step_scales` and the cell.

**Loop count is a constant.** `steps` comes from YAML; nothing in
`models/foundation.py` computes it at runtime. See Level 4, question 2.

### 2.3 The memory projection path

Two distinct mechanisms share the name "memory"; neither is a block of the
Transformer.

**(a) Bytes → model (the only differentiable entry).**
`token_embedding: nn.Embedding(261, 1024)` maps `(B,L)` int64 ids to
`(B,L,1024)` fp32. For `position_type == "learned"` an absolute table of shape
`(2048,1024)` is added; for `"rope"` nothing is added and position enters only
through `apply_rotary` inside attention. `embedding_dropout` (`p=0.0` in the
production config) is applied before block 0.

**(b) Memory store projections (`models/memory_model.py`).** The store keeps
vectors in the same `d_model` space it is configured with (default `d_model=64`,
`capacity=512`). Three independent MLPs, each `Linear(d,d) → SiLU → Linear(d,d)`:

```
k = key_encoder(content_emb)        (d,) → (d,)
v = value_encoder(content_emb)      (d,) → (d,)
q = query_encoder(query_emb)        (d,) → (d,)
```

plus two procedural adapters (`proc_key_adapter`, `proc_trace_adapter`, both
`Linear(d,d)`, both initialised to zero, so they start as no-ops) and
`trace_step_weights: Linear(d,1)` which scores trace steps:

```
w = softmax_over_steps( trace_step_weights(encoded_steps) )     (T,)
proc_v = Σ_t w_t · ( value_encoder(step_t) + proc_trace_adapter(step_t) )   (d,)
```

Shape reconciliation is `_fit_dim`: a `d_in < d` vector is right-zero-padded, a
`d_in > d` vector is average-pooled with `F.adaptive_avg_pool1d`.

**(c) Back-projection into the byte stream: none.** No learned projection maps a
memory vector back into token space. The only path from the store to generated
bytes is `models/foundation_memory.PrefixMemory`, which renders entries as
literal bytes and prepends them to the prompt:

```
[MEM] topic = value\n[MEM] topic = value\nRequest: <query>\nThought:
```

`PrefixMemory.render` truncates each value to `max_entry_bytes = 96` and joins
entries with `\n`; `retrieve` keeps at most `top_k = 3` entries and returns `[]`
if the injected retriever raises (`retrieval must never break generation`). No
gradient passes through this path — it is string concatenation.

---

## Level 3 — Core Networks (the full models)

### 3.1 The generator: `CausalByteTransformer`

**Configuration.** `d_model=1024`, `n_layers=12`, `n_heads=8`, `n_kv_heads=2`,
`d_ff=4096`, `max_seq_len=2048`, `dropout=0.0`, `norm_type=rmsnorm`,
`ffn_type=swiglu`, `position_type=rope`, `refinement_steps=2`,
`refinement_rank=256`, `rope_theta=10000.0`, `norm_eps=1e-6`,
`gradient_checkpointing=true`.

**Parameter count, measured by instantiating the class:**

| Component | Parameters |
|---|---|
| `token_embedding` (261×1024) | 267,264 |
| 12 × `TransformerBlock` (15,208,960 each) | 182,507,520 |
| `refinement` (`IterativeRefinement`) | 787,458 |
| `norm_final` | 1,024 |
| `head` (1024×261 + 261 bias) | 267,525 |
| **Total** | **183,830,791** |

Per block: attention 2,624,000 + SwiGLU 12,582,912 + two RMSNorm 2,048 =
15,208,960. The refinement loop is 0.43 % of the model; the embedding and head
together are 0.29 %.

**Forward pass, tensor by tensor:**

```
ids                        (B, L) int64, 0 ≤ id < 261
token_embedding(ids)       (B, L, 1024) fp32
[+ position table]         (B, L, 1024)  only when position_type == "learned"
embedding_dropout          (B, L, 1024)
for i in 0..11: blocks[i]  (B, L, 1024)      [checkpointed if enabled]
refinement                 (B, L, 1024)      h ← h + α_t·Cell(RMSNorm(h)), twice
norm_final                 (B, L, 1024)
head                       (B, L, 261)       ← logits
```

**Guards.** `forward` raises on rank ≠ 2, on any id outside `[0, 261)`, and on
`L > max_seq_len`. **Cost.** A forward pass is ≈ `2·183.3M = 0.367 GFLOP` per token for the matrix
multiplies, plus the attention products: at `L = 1024` that is
`q kᵀ` and `A v` over `H·d_h` per token, ≈ `4·L·d_h·H = 4.19 MFLOP` per token,
growing linearly in `L`.

**Loss.** `loss(ids)` shifts by one — inputs `ids[:, :-1]`, targets `ids[:, 1:]`
— and applies `F.cross_entropy(..., ignore_index=PAD=258)` over the flattened
`(B·(L−1), 261)` logits. PAD targets contribute nothing; the test compares the
result to a hand-computed cross-entropy restricted to the supervised bytes.

**Generation.** `generate(prefix, max_new_bytes, temperature, stop=(EOS,))` is
plain autoregressive decoding with no cache: each step re-runs the full forward
on `ids[-max_seq_len:]` and takes `argmax` (`temperature = 0`) or samples from
`softmax(logits/temperature)`. It returns only the new ids and stops at `EOS`
(id 257) or the byte budget. Cost is `O(max_new_bytes · L²·d_h·H)` attention
work with no incremental state.

**Checkpoint schema.** `CHECKPOINT_SCHEMA = 2`; saved payload is
`{schema_version, config, state_dict, parameters}` (+ optional extras) and the
returned metadata carries the file's SHA-256. `load_foundation` rejects any file
whose `schema_version ≠ 2` and rebuilds the model from the stored config with
`strict=True`.

### 3.2 The hierarchical memory model

`MemoryModel(d_model=64, capacity=512, n_exemplars=4, max_trace_steps=4)`.
Default split: `epi_cap = 256`, `proc_cap = 128`, `sem_cap = 128`
(`capacity // 2`, `capacity // 4`, remainder). Total trainable parameters at
these defaults: **100,205**.

The banks are `nn.Module`s whose storage is `register_buffer` tensors — they are
part of the state dict and are saved/loaded instrumentally, but writes are
`@torch.no_grad()` storage events, not gradient updates.

#### 3.2.1 Episodic tier — `EpisodicMemoryBank`

| Buffer | Shape | Meaning |
|---|---|---|
| `keys` | `(256, 64)` | key per slot |
| `values` | `(256, 64)` | value per slot |
| `used` | `(256,)` bool | occupancy |
| `written_at`, `last_access`, `access_counts` | `(256,)` int64 | ticks |
| `pred_errors` | `(256,)` float | EMA prediction error |

Parameters: `raw_decay` (1 scalar) → `decay_rate = softplus(raw_decay)`, a
learned, strictly positive time constant multiplier; `time_scale = 50.0`.

Score for query `q ∈ R^{B×d}` against the `M` occupied slots:

```
sim     = q Kᵀ / sqrt(d)                                    (B, M)
age     = (tick − written_at)                                (M,)
logits  = sim − decay_rate · age / time_scale + λ · imp      (B, M)
```

with `λ = importance_weight = 0.35` and `imp` from `ImportanceScorer`.
Note that the decay term uses **write age**, not access age: reading a slot does
not refresh its decay penalty (it only updates `last_access`, which feeds `imp`).

Eviction when full: choose the occupied slot minimising
`imp − decay_rate · (tick − written_at)/time_scale` (`allocate_slot`).

#### 3.2.2 Semantic tier — `SemanticExemplarBank`

Each semantic slot holds up to `n_exemplars = 4` prototype pairs, not one
centroid:

| Buffer | Shape |
|---|---|
| `keys`, `values` | `(128, 4, 64)` |
| `mixture_weights` | `(128, 4)` — sums to 1 over active exemplars |
| `exemplar_counts`, `exemplar_used` | `(128, 4)` |
| `used`, `written_at`, `last_access`, `access_counts`, `pred_errors` | `(128,)` |

Scoring is a mixture likelihood, per slot `i`:

```
s_{i,m}  = q·k_{i,m} / sqrt(d)                              (B, M=slots, E)
c_{i,m}  = s_{i,m} + log π_{i,m}     , −inf where exemplar inactive
logit_i  = logsumexp_m c_{i,m}                              (B, M)
r_{i,m}  = softmax_m c_{i,m}                                (B, M, E)
v_i      = Σ_m r_{i,m} v_{i,m}                              (B, M, d)
logits   = logit_i + λ · imp_i
```

`π` is renormalised from `exemplar_counts` over active exemplars
(`_renormalize_slot`). Writes (`write_or_merge`) compare the new value against
the active exemplars of its concept cluster by cosine similarity: if
`max_m cos(v, v_{i,m}) < novelty_threshold = 0.88` **and** a free exemplar
exists, a new mode is opened; otherwise the closest exemplar is EMA-updated with
`v ← (1−0.25)·v + 0.25·v_new` (and its key likewise) and its count incremented.
`prune_near_duplicate_exemplars` merges exemplars with cosine ≥ 0.985 by
count-weighted averaging.

#### 3.2.3 Procedural tier — `ProceduralSkillBank`

| Buffer | Shape |
|---|---|
| `keys`, `values` | `(128, 64)` |
| `trace_steps` | `(128, 4, 64)` — latent trace, `T ≤ max_trace_steps` |
| `trace_lens` | `(128,)` |
| `structured`, `input_bytes`, `output_bytes`, `explanation_bytes` | `(128,)` bool; `(128,512)`, `(128,512)`, `(128,1024)` uint8 |
| `input_lens`, `output_lens`, `explanation_lens` | `(128,)` |
| `utility_scores` | `(128,)` float |

Scoring:

```
sim    = q Kᵀ / sqrt(d)
logits = sim + λ · imp + 0.15 · tanh(utility)
```

Eviction minimises `imp + 0.2·tanh(utility)`. Byte-exact patterns are stored in
`input_bytes`/`output_bytes`/`explanation_bytes` with their lengths, so replay is
a table lookup of raw bytes (`read_procedure`, `execute_procedure`); the
`structured` flag distinguishes an empty pattern from a legacy unstructured
trace.

#### 3.2.4 The shared scorer — `ImportanceScorer`

```
e = tanh(max(pred_error,0))
f = tanh(log1p(access_count)/3)
r = exp(−max(age,0)/64)
n = tanh(‖v‖/sqrt(d))

drive    = softplus(w_e)·e + softplus(w_f)·f + softplus(w_r)·r + b
imp      = sigmoid( drive + MLP([e,f,r,n]) )
```

with `MLP: Linear(4,16) → SiLU → Linear(16,1)`, both layers initialised to zero,
so `imp` starts as a pure sigmoid of the monotone linear drive
(`w_e=1.2, w_f=1.0, w_r=−0.5, b=−0.9`). Output is in `(0,1)` by construction
(tested).

#### 3.2.5 Retrieval

`retrieve(query_emb, top_k=5, …)` (`@torch.no_grad`):

1. `q = query_encoder(_fit_dim(query_emb))` — one query, `(1, d)`.
2. All three banks' `score_active` are concatenated along slots:
   `logits ∈ R^{1×M}`, `values ∈ R^{1×M×d}`; the per-tier router bias
   `log_softmax(tier_router(q)) + log 3` is added to each tier's columns
   (`tier_router` is zero-initialised, so this starts as a uniform `1×` factor).
3. Optional `modality_filter`, `person_filter`, `metadata_filter` (exact
   primitive equality; at most 16 keys) mask columns to `−inf`.
4. `K = min(top_k, valid)`, `top_logits, top_pos = topk(logits, K)`,
   `w = softmax(top_logits)` — the weights are a softmax over the **selected**
   top-K logits, not over all slots.
5. Returns `entries` (tier, local slot, global slot, weight, importance,
   metadata, `(d,)` value) and `memory_tokens ∈ R^{K×d}` — the raw slot values,
   **not** the weighted sum. `pool=True` returns `Σ_k w_k v_k ∈ R^d` instead.
6. Unless inside `read_only()`, each selected slot's `access_counts += 1` and
   `last_access = session_tick`.

`retrieve_differentiable(query_emb, top_k)` is the batched training-time read:
`A = softmax(logits)` over all `M` candidates, `soft_read = Σ A·v`; the top-K
values are gathered exactly, and

```
mem_tokens = picked_vals + (soft_read − soft_read.detach())
```

is a tangential straight-through estimator: the forward value is the exact
unblended top-K slot vector, while the gradient at `mem_tokens` flows into `A`
and hence into `query_encoder`. Shape `(B, K, d)`; `weights` are
`softmax(top_logits)`; `mask` is all-`True` of shape `(B,K)`.

**How this store connects to the generator: it does not.** No module under
`training/`, `data/`, `evaluation/` or `notebook/` constructs a `MemoryModel` or
calls `retrieve`/`retrieve_differentiable`; the Phase 22 trainer imports only
`models.foundation`. Memory reaches a Phase 22 prompt exclusively as prefix bytes
(Level 2.3), and the retriever that would supply those bytes is an injected
callable — `ScriptedRetriever` is the only implementation shipped, and it is
test-only.

#### 3.2.6 Unused parameters in `MemoryModel`

Constructed, counted and checkpointed, never evaluated in any forward path:

* `context_proj: Linear(2d → d)` — 8,320 parameters at `d=64`; referenced only in
  legacy `load` migration.
* `person_linker: Linear(d → capacity)` — 32,768 parameters at `d=64,
  capacity=512`; referenced only in legacy `load` migration.
* `self.n_heads` — stored, never read.

`contradiction_scorer: TemporalGraphReasoner(d)` **is** used, by `score_graph` /
`explain_contradiction`.

### 3.3 The diffusion core

**Not present in the Phase 22 architecture, and removed from the repository in
Phase 23.** The generator is purely autoregressive; no denoising loop, timestep
embedding, mask-conditioned sampler or continuous byte-embedding space exists on
any code path. The deleted modules were `models/bytecore_diffusion.py`,
`models/diffusion_core.py`, `models/byte_embed.py`, `models/concept_bank.py`,
`models/masking.py`, `models/alignment.py` and `models/adaptive.py`; the
`noise_levels` / `diffusion_timesteps` / adaptive-halting keys are gone from
`configs/`. Historical diffusion results remain only as text/JSON evidence in
`runs/phase5*`, `runs/phase9*`. Every `runs/**/*.pt` checkpoint (46 files) was
deleted in Phase 23: 39 belong to classes that no longer exist, and the 7 Phase
16–18 `foundation.pt` files are loadable but referenced by no surviving code
path and are not the Phase 22 model. All JSON/Markdown/PNG reports are kept.

---

## Level 4 — System Integration (runtime behaviour)

### 4.1 Is there real latent thinking before token emission?

**There is iteration, but it is not token-level deliberation, and at
initialisation it is arithmetically the identity.**

Exact tensor flow for producing logits at one forward call:

```
ids (B,L)
  → embedding (B,L,1024)
  → 12 × block (B,L,1024)                      attention + SwiGLU, causal
  → refinement: 2 passes of h ← h + α_t·Cell(RMSNorm(h))   (B,L,1024)
  → norm_final (B,L,1024)
  → head (B,L,261)
```

Facts about this loop:

1. **It is applied to every position of the whole sequence at once**, not to a
   single "thought" slot. There is no thought token: the vocabulary contains
   only bytes plus `BOS/EOS/PAD/REF_START/REF_END`, and no `THINK` id exists.
2. **No bytes are emitted between passes.** The loop is inside one forward call;
   the model cannot inspect anything it produced.
3. **The number of passes is a config constant** (2). It is not a function of
   the input, of a confidence estimate, or of the position.
4. **Only the pass scales are per-pass** (`α ∈ R²`). The cell is shared, so the
   parameter cost of the loop is `3·D·R` regardless of `steps`.
5. **At `t = 0`, `α = 0`, therefore `h_1 = h_0` exactly.** With untrained
   `step_scales`, the loop contributes nothing; it becomes non-identity only
   after gradients reach the `α` scalars.
6. The deeper of the two effects is the 12-layer stack; the refinement loop
   adds 2 applications of a 1024→256→1024 gated bottleneck, i.e.
   `2 × 3·D·R ≈ 1.57 M` multiply-accumulates per token, versus
   `≈366 M` for the rest of the forward pass.

**Therefore:** there is a weight-tied residual refinement stage before the byte
head, executed a fixed two times, on the entire activation tensor, with learnable
per-pass scales that start at zero. There is no separate latent "thinking"
sequence, no THINK token, no scratchpad, and no inner loop that conditions on
partially decoded output.

### 4.2 How does the model decide how much to think?

**It does not decide. The stopping rule is a constant.**

* `refinement_steps = 2` is read once from YAML into
  `IterativeRefinement.steps`; `forward` executes `for index in
  range(self.steps)` and returns. There is no early-exit branch, no `break`, and
  no per-sample truncation. All `B` sequences always take exactly `steps` passes.
* There is **no confidence head and no halting probability** anywhere in the
  surviving code. `models/epistemic.py` (`ConfidenceHead`, `ThoughtStreamLayer`)
  and `models/adaptive.py` (`configured_threshold`) were deleted in Phase 23;
  `configs/default.yaml` no longer contains an `adaptive_halting` section. The
  earlier "adaptive halting" mechanism was a property of the deleted
  `GeneratorConditioned` model, not of the Phase 22 foundation.
* The only learned quantities governing how much the loop changes the state are
  the `steps` scalars `α_t`. They scale the update; they do not terminate it.
* Byte emission has its own hard stops, both outside the model's control:
  `generate` halts when the argmax/sampled id is `EOS = 257`, or after
  `max_new_bytes` steps, or when the window reaches `max_seq_len`.
* In Phase 3 RL the rollout budget is likewise fixed by the trainer:
  `_sample(prompt, 96)` then, if a search action appeared,
  `_sample(extended, 64)`.

Observed consequence: with `α` initialised to 0, the halting behaviour of a
freshly initialised model is "two no-ops". Whether the trained `α` values are
non-zero in any checkpoint in this repository cannot be verified — no Phase 22
checkpoint exists in the tree (`runs/phase22/checkpoints/` is git-ignored and
empty).

### 4.3 Self-editing and temporal awareness

**The distinction is made by literal bytes in the sequence, and enforced by the
reward function — not by attention masks or loss weights.**

The two blocks, from `training/phase22_reward.py`:

```python
PAST_OPEN  = b"[PAST_ACTION: executed]"
PAST_CLOSE = b"[/PAST_ACTION]"
DRAFT_OPEN = b"[CURRENT_DRAFT: editable]"
DRAFT_CLOSE= b"[/CURRENT_DRAFT]"
```

**Where they enter the sequence.** `build_rl_case(...)` concatenates
`prompt + PAST_OPEN + past_actions + PAST_CLOSE + DRAFT_OPEN + initial_draft +
DRAFT_CLOSE + THOUGHT`. They are ordinary bytes: the model sees them through
`token_embedding` like any other byte, and nothing marks them as special. The
attention pattern over those bytes is the same causal self-attention described in
Level 1.5. **There is no segment embedding, no attention mask, and no loss weight
attached to `PAST_ACTION` or `CURRENT_DRAFT`**; in Phase 2 the per-token loss
weights produced by `BytePacker.feed(trainable_prefix=n)` mask only the first `n`
prompt/header bytes of a distilled document (weight 0.0 there, 1.0 afterwards,
`BOS` weight 0.0, `EOS` weight 1.0).

**How an `[EDIT]` alters the sequence.** An edit is a raw-byte command:

```
[EDIT at=N del=M ins_b64="<base64>"]
```

matched by
`re.compile(rb"\[EDIT at=(\d+) del=(\d+) ins_b64=\"([A-Za-z0-9+/=]*)\"\]")`.
Parsing produces `EditCommand(at, delete, insert, site)`; `site ∈ {draft, past,
outside}` is decided by comparing the match's byte offset against the offsets of
the `PAST_` and `DRAFT_` payloads **that the parser itself located in the
trajectory**:

```
site = "past"    if past_start  ≤ start < past_start  + len(PAST_OPEN)  + len(past)
     = "draft"   if draft_start ≤ start < draft_start + len(DRAFT_OPEN) + len(draft)
     = "outside" otherwise
```

The bytes are then applied host-side, right-to-left so earlier offsets stay
valid:

```python
result[command.at : command.at + command.delete] = command.insert
```

**What the edit does *not* do.** It does not modify any tensor the model holds;
it does not re-enter attention; the model is never re-prompted with the edited
draft inside an episode. `apply_edit_commands` is called by the reward function
to test `initial_draft → expected_draft` byte equality, and by
`evaluation/final_agent_eval.score_case` for the same purpose. An `[EDIT]`
therefore alters (i) the reward, and (ii) the evaluator's reconstruction of the
draft. The "self-editing" is a scoring construct executed by the host.

**Penalties that make the split behavioural rather than decorative**
(`RewardWeights`, exact values): `−0.5` per draft edit, `+0.8` for a verified
1–2-edit fix, `−0.75` per edit beyond 2 (thrashing), `−1.0` per edit whose
`site ≠ "draft"` (temporal violation). `compute_reward` reports
`flags["temporal_violation"] = parsed.illegal_edits > 0`.

**Why the model could learn the distinction.** The only signal that past bytes
are immutable is the reward difference between an edit at a `past` offset and
the same edit at a `draft` offset. With `steps` fixed and no architectural
barrier, a policy that writes an edit command anywhere in the stream receives the
same attention treatment; the split is enforced entirely downstream by the
`site` classifier.

### 4.4 Search vs. remembering: the routing logic

**There is no intent classifier and no learned router in the Phase 22 runtime.**

Three separate, deterministic mechanisms exist:

**(a) Remembering = prefix injection, decided by the caller.**
`PrefixMemory.prefix(query)` calls the injected retriever and renders up to
`top_k = 3` entries as bytes before the request:

```
[MEM] topic = value
Request: <query>
Thought:
```

The decision "use memory or not" is the retriever's: if it returns entries, they
are in the prompt; if it returns `[]` or raises, the prompt is shorter. No
component of `models.foundation` examines the `[MEM]` lines differently from any
other byte. In the Phase 22 trainer, `models/foundation_memory.py` is not
imported at all, so in the three-phase pipeline the answer to "does it remember"
is: only to the extent that the Phase 2 distillation taught the pattern
`[MEM] … = …` inside ordinary text.

**(b) Searching = a regex interception plus a stub observation.**
The trainer samples bytes; if `b"search_ddg(" in trajectory and
b"Observation:" not in trajectory`, it appends a canned observation and samples a
continuation:

```python
result = case.get("search_result") or (case["expected_answer"] + b" (fonte: busca)")
observation = b"Observation: " + result + b"\n"
extended = torch.cat((prompt, first, tool_ids))
second, lp2, n2 = self._sample(extended, 64)
```

No network call is made. The "search tool" is a constant string derived from the
evaluation case, and the default fallback is the expected answer itself. (For the
general-purpose runtime, `models/react.ReactLoop` provides the real interception
— one regex, `Action: tool("arg")`, execution through an injected tool mapping,
`truncate_at_action` discarding everything the model wrote after its action line,
and `format_observation` collapsing the real result to one bounded line — but the
Phase 22 trainer does not use it.)

**(c) The decision rule is the reward, not a gate.**
`compute_reward` implements the choice economically:

```
no search, answer wrong, search_required → −2.0      ("wrong guess without search")
no search, answer right, search_required → −0.5      ("lucky guess")
search + observation used + right        → +1.0 + 1.5
search + observation used + wrong        →  +0.5
search + observation NOT used            → +0.1      ("search ignored")
```

"Used" is decided by token overlap: `_tokens()` extracts lowercase alphanumeric
tokens of length ≥ 4 minus a stop list, and `used_search_result` requires that the
`Answer:` line's token set intersect the `Observation:` token set. The three
search outcomes are ordered `1.5 > 0.5 > 0.1`; `dominance_report()` asserts the
ordering `search_and_use > search_used_but_wrong > search_ignored` and
`efficient edit > thrashing` on fixed synthetic trajectories, not on model
output.

**Therefore.** Routing is not computed by the model: remembering is "the caller
put retrieved bytes in the prompt", searching is "the emitted byte stream
matched `search_ddg(`", and the preference between them is a scalar reward. No
component examines the query to choose a route.

---

## Appendix A — Measured invariants

Executed in this checkout, CPU, float32 unless noted:

| Property | Value | Where asserted |
|---|---|---|
| Production parameter count | 183,830,791 | `test_parameter_budget_of_the_phase22_config_is_exact` |
| Per-block parameters | 15,208,960 | same |
| Refinement parameters | 787,458 | same |
| SwiGLU parameters per layer | 12,582,912 | `test_swiglu_matches_the_closed_form_and_has_no_biases` |
| Attention parameters per layer | 2,624,000 | measured, Level 1.5 |
| `q/k/v` shapes | `(B,8,L,128)`, `(B,2,L,128)`, `(B,2,L,128)` | `test_gqa_matches_explicit_kv_repetition` |
| RoPE tables | `(1,1,L,128)`, 0 parameters | `test_rotary_table_frequencies_follow_theta_to_the_minus_2i_over_d` |
| Refinement at init | exact identity | `test_iterative_refinement_is_weight_tied_and_identity_at_initialisation` |
| Causal mask effect | positions `≤ i` invariant to later bytes | `test_attention_is_causal` |
| PAD contribution to loss | zero | `test_loss_matches_manual_cross_entropy_and_ignores_pad` |
| Gradient-checkpointing parity | grads equal within 1e-5 | `test_gradient_checkpointing_matches_plain_backward` |
| `MemoryModel` default parameters | 100,205 | measured, Level 3.2 |
| Semantic storage | `(128, 4, 64)` keys/values, `π` sums to 1 | `test_semantic_mixture_uses_logsumexp_and_posterior_readout` |
| Memory logits | `sim − decay·age/50 + 0.35·imp` | `test_episodic_logits_are_scaled_dot_product_minus_decay_plus_importance` |
| DIFF read forward value | exact top-K slot vector | `test_differentiable_read_is_exact_top_k_in_the_forward_pass` |
| Full regression suite | 92 passed | `pytest -q` |
| GPU precision / VRAM | **unmeasured** — no CUDA device in this environment | n/a |

## Appendix B — Modules deleted in Phase 23

Deleted because they were unreachable from the Phase 22 entry points and the
classes they define no longer exist in any checkpoint the surviving code can
load:

* `models/`: `generator_conditioned`, `bytecore_diffusion`, `diffusion_core`,
  `byte_embed`, `concept_bank`, `masking`, `alignment`, `adaptive`, `epistemic`,
  `cognitive`, `cognitive_arguments`, `conversation`, `conversation_intents`,
  `conversation_v2`, `conversation_v3`, `dialogue_response`,
  `contextual_response`, `diverse_response`, `chunked_generation`,
  `chunked_decoder`, `episodic_conversation`, `vector_memory`, `tool_use`,
  `edit_router`, `phase14`.
* `data/`: every synthetic corpus builder (`conversation_examples`,
  `conversation_response_v2`, `diverse_dialogue`, `consolidation_examples`,
  `edit_curriculum`, `masked_edits`, `positional_edits`, `structured_edits`,
  `tool_use_examples`, `vague_intent_pairs`, `vector_memory_pairs`,
  `phase11_eval_sets`, `phase12_eval_sets`, `phase13_examples`,
  `phase13_feasibility`, `phase14_data`, `scalable_react_dataset`, `dataset`,
  `generate`) plus `github_to_react`, `mixed_react_dataset`, `phase18_corpus`,
  `github_collector`, `github_raw/`, `holdout_registry`, `ingest_real_traces`,
  `real_edit_pairs.jsonl`.
* `training/`: 26 phase-specific trainers (`train_foundation`, `train.py`,
  `co_train`, `replay`, `rl_editing`, `graph_train`, `cognitive_train`,
  `train_*`, `phase13/14_*`, `consolidate_sessions`, `finalize_replay`,
  `rebind_consolidation`).
* `evaluation/`: 41 historical harnesses including `foundation_eval`,
  `github_eval`, `phase18_eval`, `phase6_*` … `phase13_*`, `copy_probe`,
  `copy_diagnostic`, `tool_eval`, `memory_eval`, `ablation_study`,
  `calibrate_confidence`.
* `debug/` (11 forensic scripts), `demo/` (4 demos), `concept_demo.py`,
  `notebook/` Phase 19 Colab stack (`phone_model`, `preprocess`, `train`,
  `evaluate`, `hf_upload`, `mineflayer_*`, `tools`, `dataset_discovery`,
  `build_notebook`, `test_mineflayer`, `update_docs`, `phase19_colab.ipynb`),
  `configs/phone_100m.yaml`.
* 22 test modules that only exercised the deleted classes, replaced by
  `tests/test_foundation.py`, `tests/test_memory_model.py` and
  `tests/test_react_runtime.py`.

Retained: `models/{foundation, memory_model, foundation_memory, react, sandbox,
holdout, procedural, contradiction_graph}.py`; `data/phase22_sources.py`;
`training/phase22_{common,reward,train}.py`;
`evaluation/{final_agent_eval, phase22_benchmark}.py`;
`notebook/build_phase22_notebooks.py`; `configs/{default,t4_scaled}.yaml`.
