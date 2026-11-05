# ARCHITECTURE_AUDIT.md — ByteCore-DualMemory, Phase 25 code as it exists

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
| `B` | 1 (micro-batch) | sequences per step |
| `T` | 16384 (model cap = train window) | bytes per sequence |
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
θ=10000. The production preset selects **BigBird** (`attention_type: bigbird`);
the legacy dense path still uses `scaled_dot_product_attention(is_causal=True)`.
SwiGLU: `3·D·F = 12,582,912` params/layer. All 8 blocks are checkpointed.

### 2b. Causal BigBird mask (`BigBirdSparseMask`) — mathematical contract

For query row `i`, key column `j`, head `h`, with `L=16384`, `W=256`, `R=32`,
`G=16`, the allowed set is

```
A_i^h = { j : j ≤ i AND [ (i-W+1 ≤ j ≤ i)       # local, inclusive, self included
                         OR j < G              # prefix global columns
                         OR i < G              # prefix global rows
                         OR j ∈ R_i^h ] }      # random past keys
R_i^h ~ sample_without_replacement({G, ..., i-W},
                                   min(R, max(0, i-W-G+1)))
```

The causal intersection `j ≤ i` applies to **global columns too**: the first
`G` rows cannot attend forward. Early rows have fewer than `R` eligible
nonlocal/nonglobal keys; the count is exact only when the eligible set is large
enough. Random sets are seeded per head (`seed + h·1000003`) and reproducible.

`base` is a shared bool tensor `(1,1,L,L)` with **True = ATTEND** (SDPA boolean
convention; the unused legacy dense `causal_mask` used True = FUTURE). Each
head overlay is materialised lazily as `(1,1,L,L)` by scattering `R_i^h` into a
clone of the base. Head attention is checkpointed, so backward recomputes one
overlay at a time rather than retaining `(1,H,L,L)`.

Late-row density is `(W+G+R)/L = 304/16384 ≈ 1.855%`. Storage: base
`L² = 268,435,456 B`; one overlay another `L²`; random coordinates
`H·L·R·8 = 33,554,432 B`. **Limit:** passing a boolean mask to SDPA prunes
connectivity, not asymptotic FLOPs. CUDA forces the efficient-attention backend
(no silent math fallback); CPU tests may use math as a reference. This is not a
block-sparse BigBird kernel and does not provide literally infinite context.
Reducing R/G does not necessarily shrink the dense mask allocation.

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

**Context-to-memory cross-attention (Phase 25, every layer).** In parallel with
the GLU bridge, the adapter now teleports information from the committed memory
timeline across gaps longer than the 16k window:

```
Q = RMSNorm(W_q^x h)          W_q^x ∈ R^{1024×1024}     (B,T,1024)
K = RMSNorm(W_k^x keys)       W_k^x ∈ R^{1024×1024}     (N,1024) or (B,N,1024)
V = W_v^x values              W_v^x ∈ R^{1024×8192}     (N,1024) after projection
heads = softmax(Q_h K_hᵀ / √d_h + slot_bias) V_h        (no causal mask over memory)
δ_x = cross_scale · W_o^x(concat heads)                 (B,T,1024)
h ← h + scale·GLU(retrieved) + δ_x
```

Values are read from the **8192-dim bank**; a shared `8192→1024` projection
before the heads avoids expanding V per head. `cross_scale` is zero-init ReZero,
so the new branch is identity at step 0 and opens by gradient. Inactive rows are
zeroed after SDPA (never an all-masked softmax NaN). Persistent slots are
committed only **after** their producing batch is assembled (`BatchBuilder`
delayed writes), so cross-attention cannot retrieve same-batch futures;
batch-local slots remain row-isolated.

**Parameter census (measured, meta-device construction):**

| Component | Params |
|---|---|
| GLU adapter per layer (`W_q` 1,048,576 + `W_g` 16,777,216 + scale 1) | 17,825,793 |
| Cross-attention per layer (Q/K/O 3,145,728 + V 8,388,608 + norms 2,048 + gate 1) | 11,536,385 |
| Combined adapter per layer | 29,362,178 |
| 8 combined adapters | 234,897,424 |
| Core without adapters (pre-24.4) | 123,012,366 |
| Latent thought buffer + thought-confidence (Phase 24.4, §6b) | 8,407,043 |
| **Total** | **366,316,833** |

The analytic count is asserted exactly by
`tests/test_gated_adapter.py::test_injection_in_all_eight_production_layers`.

The older GLU-only CPU latency figures are historical and do **not** include
the Phase 25 cross-attention branch. GPU latency/VRAM require the T4 run.

## 4. Unified dynamic memory (`models/memory_model.py`)

**One physical bank**, geometry unchanged: keys `(capacity,1024)`, values
`(capacity,8192)`. Production `capacity: 2048`; `DualMemory.episodic` is a
compatibility alias of `DualMemory.procedural`, not a second store. Key/value
buffers are `2048·(1024+8192)·4 = 75,497,472 B`, plus bookkeeping and two
learnable policy vectors (`importance_raw`, `decay_raw`, 2·2048 fp32 params).

For slot `i`, with clock tick `t`, last reinforcement `r_i`, capacity `C`:

```
age_i        a_i = (t - r_i) / C
decay_i      d_i = 0                         if Forever_i
             d_i = softplus(decay_raw_i)     otherwise
importance_i I_i = 2 + sigmoid(importance_raw_i)   if Forever_i
             I_i = sigmoid(importance_raw_i)       otherwise
score_i      S_i = I_i - d_i · a_i
evict        argmin { S_i : used_i AND NOT Forever_i }
```

Short importance lies in `(0,1)`; Forever importance lies in `(2,3)` and has
exactly zero effective decay. Thus every Forever score exceeds every short
score even at infinite age, and eviction additionally excludes Forever indices:
permanence is a **hard guard**, not a hoped-for large score. An ordinary write
to an all-Forever bank raises `MemoryError`, counts the attempt, and records a
policy event; only explicit host curation (`write_forever`/`mark_forever`) pins
or rewrites permanent slots.

Writes remain `@torch.no_grad()` and detach keys/values. Raw policy parameters
are initialised once and **never reset on replacement**, so eviction learning
survives occupant changes. `reinforce(indices)` refreshes age only for slots
actually used; exposing distractors to retrieval does not make the bank fresh.

The auxiliary policy loss uses bounded softplus margins (no `exp(decay·age)`):

```
L_fresh   = mean_short softplus(d_i·a_i - margin)
L_pair    = mean_event softplus(margin + S_victim - S_fresh_retained)
L_stale   = mean_stale softplus(margin - d_victim·a_victim)
L_forever = mean_forever softplus(decay_raw_i)
            + relu(margin + max_short I - min_forever I)
L_policy  = mean(active terms)
```

`L_stale` rewards efficient removal of stale short-term slots; `L_fresh`
discourages premature decay; `L_pair` ranks the evicted victim below fresher
retained slots; `L_forever` penalizes raw decay on permanent slots and any
tier-margin violation (zero in a valid bank by the bounds above). Rejected
Forever-overwrite attempts add a defensive victim-forever event. Events are
transient, detached snapshots consumed each micro-batch; total training loss is
`CE + 0.1·BCE_confidence + 0.01·L_policy`.

`SkillBankView` remains global `(N,·)` or batch-local `(B,N,·)`; `stack_views`
broadcasts/concatenates. Persistence schema **2** stores buffers, policy
parameters, metadata, tick, and Forever flags; old two-bank/FIFO payloads are
rejected rather than silently migrated.

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

### 6b. Latent Thought Matrix (`LatentThoughtBuffer`, Phase 24.4)

A dedicated `(B, S, D)` scratchpad (`S = max_thought_steps = 4`, prod
`D = 1024`) inside the loop — the model "thinks" without emitting bytes:

```
s_t = s_(t-1) + g_buf · CrossAttn(Q = s_(t-1) + e_t, K/V = h)   (buffer reads)
h_T ← h_T + g_ctx · CrossAttn(Q = h_T, K/V = s_(0..t))          (context reads)
conf(h, s̄) = σ(W_c·RMSNorm(h) + W_s·RMSNorm(s̄) + b)             (combined state)
```

* **Bidirectional cross-attention**, both pre-normed, `need_weights=False`
  (fused SDPA). Scores are `(B, heads, S, T)` and `(B, heads, 1, S)` —
  linear in T, ≈0.4% of one T×T self-attention map at production shapes
  (0.5 MB vs 134 MB fp32 at B=4, T=1024, H=8). Measured incremental peak
  RSS of the whole loop with the scratchpad vs without: **+82 MB** (CPU
  fp32, no_grad, B=4/T=1024/D=1024, `ru_maxrss` high-water diff; fp16 on
  the T4 will be smaller). No OOM risk.
* **Causality:** the buffer summarises the whole window, so ONLY the
  frontier position `T` reads it back (its next-byte target may depend on
  the entire window). Positions `< T` are untouched — test-asserted
  byte-for-byte (`test_thought_writeback_is_causal`).
* **Isolation:** the buffer never reaches `head`; the byte head only ever
  sees `norm_final(final context state)` — test-asserted with a forward
  pre-hook (`test_thought_buffer_isolated_from_lm_head`).
* **Asymmetric ReZero (measured, not assumed):** `g_ctx = 0`, slots `0`,
  `W_s = 0` at init ⇒ logits/confidence EXACTLY equal the legacy
  (`max_thought_steps=0`) model. `g_buf = 1`, NOT 0: the first smoke probe
  showed a double-zero init **deadlocks** (zero slots ⇒ zero write-back
  output ⇒ zero gradient on both gates; activation norm stayed 0.0 through
  training). With the read side open, `∂L/∂g_ctx = ⟨∂L/∂h_T, inject⟩ ≠ 0`
  at step 0 and the write side opens from step 1. After the fix, the 4-step
  smoke reports activation norm **3.65**, `g_ctx ≈ 6.6e-4`,
  `‖W_s ≈ 4.9e-3` (`smoke_test_coherence.py`, regression-tested).
* Slots accumulate via `torch.stack` (no in-place writes: earlier slots
  stay version-stable for autograd across loop iterations). Unwritten slots
  (early break) keep their learned init in the exposed `(B, S, D)` telemetry.

Parameter delta: slots 4,096 + 2 cross-MHAs 8,396,800 + 4 pre-norms 4,096 +
2 gates + thought-norm 1,024 + thought-confidence 1,025 = **8,407,043**.
Checkpoint schemas bumped explicitly: foundation v3→v4, trainer v4→v5.

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
grad-clip 1.0, **micro-batch 1 × accum 32**, **seq_len 16384**, gradient
checkpointing strictly on all 8 blocks plus pure thought-step/cell checkpoints.
Precision `auto`: bf16 on Ampere+, **fp16 + GradScaler on T4/sm_75**, fp32 on
CPU — resolved from `torch.cuda.is_bf16_supported()`, logged, never assumed.
Loss = causal-LM CE + confidence BCE + decay-policy loss (§4). Persistent memory
writes are delayed until after batch assembly to prevent same-batch leakage.

**Representative CUDA probe.** `Trainer.probe_vram` measures
`torch.cuda.max_memory_allocated` over a synthetic step with the full unified
bank, differentiable local slots from every projector, confidence path, policy
loss, all-block checkpointing, and worst-case AdamW moment allocation (temporary
lr=0 step; existing resumed moments are preserved, not double-counted). Configured
frozen backbones are loaded so their persistent weights are present. CPU returns
`measured=False`, never a fake peak.

**Autotune / fail-closed.** Above **13.5 GB**, halve micro-batch to 1, then
halve BigBird R toward 0 and G toward 1, rebuilding the mask and reprobing.
Dense boolean-mask allocation may not shrink with R/G; if the measured peak is
still above the ceiling, raise `MemoryError` and refuse to start.

**Simulated T4 accounting (`--simulate-t4`, analytic, NOT measured).** The
production preset estimates **13.073 GB / 13.5 GB** with fp32
params/grads/Adam (16 B/trainable param), rounded-up frozen encoder weights,
fp32 checkpoint boundaries, one live block, one lazy BigBird overlay + possible
fp16 backend bias, pure thought-step checkpoints, streamed confidence logits,
full bank + local slots, and a 0.5 GB allocator/backend reserve. Largest terms:
optimizer residency 6.353 GB, frozen encoders 1.100 GB, block boundaries
0.604 GB, BigBird base/overlay/bias 1.074 GB, refinement 1.191 GB. This is a
conservative shape-based estimate; the CUDA probe is the authority.

Checkpoints: atomic tmp+rename, every 30 min **and** every 500 steps, model +
projectors + unified bank/policy + optimizer + scaler + RNG; trainer schema **6**,
foundation schema **5**, memory schema **2**. Frozen encoders are rebuildable;
resume recreates projector param groups in order. Hub upload/download failures
are reported as `uploaded/downloaded: false` with a reason.

## 9. Data (`data/sources.py`) — PT-BR first, strictly streaming

**Strict streaming (Phase 24.2).** `open_stream` is the single `load_dataset`
call site (AST-enforced by `tests/test_streaming.py`). It (1) forces
`streaming=True` and a concrete split, (2) verifies the result with
`assert_streaming` — a map-style `Dataset`/`DatasetDict` (disk download) or an
`IterableDatasetDict` (unresolved split) raises `TypeError` before any
training — and (3) shuffles with the iterable-safe
`.shuffle(seed=..., buffer_size=10000)`: a seeded RAM-only reservoir
(`SHUFFLE_BUFFER_SIZE`, configurable via `data.shuffle_buffer_size`), never
the map-style `.shuffle(seed=...)` that an `IterableDataset` rejects.
`image_embedded`/`audio` kinds are capped at `MEDIA_SHUFFLE_BUFFER_CAP=1000`
rows (`effective_shuffle_buffer`): the reservoir holds decoded PIL images /
waveforms, so 10k media rows would be 1–2 GB of RAM; `image_url` rows buffer
only url+caption strings and stay at 10k.
`build_mixture` rejects `data.streaming: false`. The trainer loop is
step-based (`while step < max_steps`), never calls `len()` on data, and
rejects sized sample sources; `MixtureSampler` + `BatchBuilder` form the
custom single-process iterable wrapper (DataLoader `num_workers=0`
semantics — no worker subprocesses, no disk prefetch). The notebook's
Zero-Disk cell prints every group's stream type and asserts ≤ 32 MB disk
growth over 10 streamed batches, between the mix audit and the training
launch. Four groups and no fifth bucket:

| Group | Weight | Chain |
|---|---|---|
| `ptbr` | **0.50** | `TucanoBR/GigaVerbo` → `HuggingFaceFW/fineweb-2` (`por_Latn`) |
| `code` | 0.25 | `bigcode/the-stack` (python, javascript) → `the-stack-smol` |
| `multimodal` | 0.15 | COYO (URL images) → MSCOCO-5k (embedded) → LibriSpeech (audio) |
| `english` | 0.10 | `HuggingFaceFW/fineweb` (`sample-10BT`) |

**Quota-exact sampling, not Bernoulli.** The Phase 24 sampler drew
`rng.random()` per item, so the ratio was only correct in expectation (a
100-item window could land at 43%). `MixtureSampler` now allocates every
window of `MIX_WINDOW = 100` draws with the largest-remainder method,

```python
exact   = w_i / Σw · W
counts  = floor(exact);  leftover = W − Σcounts
counts += 1 for the `leftover` largest fractional parts (name tie-break)
tickets = shuffle([group] * counts[group])      # seeded RNG
```

so 0.50/0.25/0.15/0.10 allocates exactly `{ptbr: 50, code: 25, multimodal: 15,
english: 10}` and any multiple of 100 draws carries the target ratio exactly,
while the document order stays stochastic and seed-reproducible
(test-asserted: every consecutive window is exactly on target, and two seeds
produce the same counts but a different order). Exhaustion drops a group,
renormalises the survivors and **rebuilds the current window**, so a dead
chain cannot leave stale tickets belonging to the dead weight
(`failures[group] = "exhausted"`). `build_mixture` rejects unknown weight keys
before opening a single stream.

**Mixing unit.** The weights govern *documents*, the only quantity a sampler
controls; the byte/token share is a different number. `audit_mixture` reports
both, plus a per-group verdict:

```
PT-BR-first data mix audit  |  100 batches x 4 = 400 documents  (0.0 s)
source        target    docs    doc %       bytes   byte %   status
PT-BR          50.0%     200    50.0%      27,156    44.5%   ok
Code           25.0%     100    25.0%      20,967    34.4%   ok
Multimodal     15.0%      60    15.0%       7,740    12.7%   ok
English        10.0%      40    10.0%       5,175     8.5%   ok
total         100.0%     400   100.0%      61,038   100.0%
verdict: PASS -- document share within +/-2.0 pp of target (quota window = 100)
```

(Historical Phase 24 fixture measured on **synthetic streams** through the real
`build_mixture` → `audit_mixture` path, seed 24, `batches=100`, `batch_size=4`,
fake `_open_rows`; document counts are the sampler's real output, byte counts
are fixture-dependent. Production Phase 25 uses `batch_size=1`, so 100 audit
batches draw 100 documents.) The trainer records the realised mix per log step in
`runs/metrics.jsonl["data_mix"]` and the final weights/counts/failures in the
run summary.

**Loss and masking (24.1.3).** All four groups flow through one
`ignore_index=PAD(258)` cross-entropy and one confidence BCE, both means over
**non-PAD tokens**. Measured invariants (CPU, tiny model, `tests/test_train.py`):

* CE equals `F.cross_entropy(logits[valid], targets[valid])` exactly — only the
  mask decides what is learned, not the padding length;
* truncating a row's PAD tail leaves the loss bit-identical (padding is inert),
  for packed text rows and for multimodal rows alike;
* an all-PAD batch used to be `0/0 = NaN` in `F.cross_entropy`; `loss()` now
  returns a graph-carrying **zero loss** with zero gradients (never NaN);
* a batch that is 50% PT-BR is exactly the token-weighted blend of its parts:
  `L_mixed = (n_ptbr·L_ptbr + n_cap·L_cap)/(n_ptbr + n_cap)` for CE, the
  confidence BCE and the total (rel. tol 1e-4) — short, heavily padded caption
  rows cannot dilute the PT-BR gradient.

`BytePacker` frames documents `BOS doc EOS` and packs to fixed-length shifted
pairs with carry — no byte dropped.

## 10. Verification (measured)

`python -m pytest tests/ -q` → **191 passed in ~10 s** (CPU, fully offline:
a session guard replaces the real `datasets.load_dataset` with a loud
failure, and `mock_streaming_hub` serves tiny in-memory `IterableDataset`s).
Coverage: exact GLU math at production widths, key-matching fidelity (aligned
query retrieves its value to 1e-3), all-8-layer injection + exact parameter
count, masked/empty retrieval, projector gradient flow, bank
dynamic-score eviction/Forever guard/detach/persistence, frozen-encoder contract, codec bounds, matrix
agnosticism, adaptive break/cap/targets, token-triggered sandboxed search +
EDIT pass-through, bidirectional batches, accumulation equivalence,
checkpoint/resume, time-based cadence, **quota-exact 50/25/15/10 windows**,
renormalisation, mix-audit verdicts and truncation, **PAD masking/zero-loss
invariants**, token-weighted CE and confidence BCE on a 50% PT-BR batch,
trainer `data_mix` reporting, mixture fallbacks/packing, **BigBird causal
local/random/global reference math, production 16384 mask audit, lazy per-head
overlays, all-8-block checkpointing, cross-attention gradients in every layer,
decay-score/Forever permanence/policy-gradient stability, delayed memory commit,
representative probe view, fail-closed autotune, labelled T4 simulator**,
**strict streaming
(`streaming=True` on every call, `IterableDataset` type verification,
seeded `buffer_size` shuffle, single call site, step-based loop, sized-source
rejection, end-to-end offline quota)**, **latent thought matrix (shape,
slot recurrence, lm-head isolation, frontier-only causality, zero-init
legacy equivalence, combined-state confidence, gate bootstrap against the
double-zero deadlock)**, the coherence smoke script, notebook validity
(audit + Zero-Disk cell precede the run), and the cleanup contract itself.

**Known limits.** 4 h ≈ order 10⁸ tokens: a pre-training start, not a
finished model. The mix is a document quota, so the *byte* share of PT-BR is
~45% rather than 50% (measured on fixtures) — token-weighted sampling would be
a different mechanism and is not implemented. `open-web-math` is no longer in
the mix; there is no fifth bucket. The codec is lossy (p99 ≈ 0.074 on unit-normal features).
Single-slot views give the key projector zero gradient until distractors
exist (the trainer merges the bank). Real `ddgs` search needs a networked
runtime; training uses the deterministic mock. GPU throughput/VRAM remain
analytic until the notebook runs. Boolean-mask SDPA is connectivity-sparse, not
FLOP-sparse; the 13.5 GB ceiling is enforced by the real probe, not by the
simulator alone.
