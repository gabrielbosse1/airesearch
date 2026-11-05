# AGENTS.md — ByteCore-DualMemory operating protocol (Phase 25.2)

Read this file before touching the repository, then read
`ARCHITECTURE_AUDIT.md` (the bottom-up math of the code that actually exists)
and `README.md` (how to run it).

## Mission and invariants

- **Raw bytes only at data boundaries.** UTF-8 text is bytes `0..255`; NUL is
  data; PAD masks the loss. No tokenizer anywhere, ever.
- **Fixed vocabulary.** `VOCAB_SIZE = 265`: bytes `0..255`, then `BOS=256`,
  `EOS=257`, `PAD=258`, `TOOL_SEARCH=259`, `TOOL_SEARCH_END=260`,
  `OBS_START=261`, `OBS_END=262`, `EDIT=263`, `EDIT_END=264`.
  `FoundationConfig` rejects any other `vocab_size`.
- **No host-side text hacks.** Tool detection is by TOKEN ID only. No regex
  parsing of model output, no marker scanning in `encode_bytes`, no `[REF]`
  resolution. `tests/test_tools.py` and `tests/test_repo_hygiene.py` enforce
  this; do not weaken them.
- **`[EDIT]` is ordinary vocabulary.** It is trained by the same
  cross-entropy as every byte and has no runtime handler. Only
  `[TOOL_SEARCH]` executes anything, inside a resource-limited subprocess.
- **The GLU is the V replacement.** The 8192→1024 compression in
  `GatedSkillAdapter` must stay `Linear(8192, 2048)` → split →
  `Information ⊙ sigmoid(Filter)`. Do not "simplify" it to a linear
  projection; the multiplicative gate is the architecture.
- **The Latent Thought Matrix is purely latent (Phase 24.4).** The
  `LatentThoughtBuffer` `(B, max_thought_steps=4, d_model)` lives inside
  `IterativeRefinement`: bidirectional cross-attention (buffer reads context,
  context frontier reads buffer), slot `t` updated from slot `t-1` + context.
  It must NEVER pass through the byte head — only refined CONTEXT states
  reach `head(norm_final(h))`. Write-back touches ONLY the last position:
  the buffer summarises the whole window, so writing it into earlier
  positions would leak future bytes (causality is test-asserted). Init is
  asymmetric ReZero: `buffer_gate=1`, `context_gate=0`, slots/thought-conf
  zero — a double-zero init DEADLOCKS the scratchpad (measured: norms stayed
  0.0; do not "simplify" both gates to zero). The confidence head evaluates
  the combined context+thought state. `max_thought_steps: 0` reproduces the
  legacy architecture exactly. Checkpoint schemas: foundation v6, trainer v7,
  memory v2.
- **Coherence pre-flight gates readiness (Phase 25.2).** `coherence_preflight.py`
  runs exactly 500 optimizer steps on PT-BR/code, snapshots loss/generation at
  0/250/500, and asserts loss decrease, recognizable PT-BR, and valid Python
  syntax. The sandbox run is CPU-feasible scale of the same sliding-window +
  memory cross-attention path; never present it as production-scale fluency.
  The 4-hour Colab run remains the production training event.
- **Sliding-window attention is the T4 path (Phase 25.1).** Production
  `attention_type: sliding_window`, `max_seq_len: 16384`, `window_size: 1024`,
  `sliding_chunk_size: 512`. Allowed edges are exactly `i - W ≤ j ≤ i`
  (self + up to W previous bytes), strictly causal. The former random/global mask
  logic is removed. SDPA receives only chunk-local masks; never allocate an
  `(L,L)` attention mask. CUDA forces memory-efficient attention and the probe
  verifies it. T4/sm_75 cannot use PyTorch Flash/cuDNN SDPA (sm_80+); do not
  call a math fallback "Flash" or claim the 16k row is a dense 16k window.
- **Cross-attention is the long-range hand-off.** Every GatedSkillAdapter keeps
  the original GLU AND adds Q=1024 hidden, K=1024 bank keys, V=8192 bank values
  cross-attention through a zero-init ReZero gate. With no random/global edges,
  committed memory is the only path beyond the sliding window. Persistent
  writes commit only AFTER the current batch is assembled; same-batch future
  leakage is forbidden. Empty/masked rows retrieve zero, never NaN.
- **Memory is a store, not a gradient update.** Bank keys/values are written
  `@torch.no_grad()` and detached. Only reads, batch-local projector views, and
  the small per-slot importance/decay policy parameters are differentiable.
  Keys are 1024-dim, values 8192-dim. ONE physical bank: `DualMemory.episodic`
  aliases `procedural`. Eviction score = importance − softplus(decay_raw)·age;
  Forever slots have exactly zero effective decay, importance above the short
  range, and a hard eviction guard. Policy parameters survive replacement.
  Do not restore FIFO or reset policy on write.
- **Encoders are frozen.** `requires_grad_(False)`, eval mode forever,
  forward under `no_grad`. Only the `SkillProjector`/`SkillKeyProjector`/
  `TextSkillProjector` train. The T4 preset stores frozen inference-only
  backbones in fp16 on CUDA and disables the unused video backbone; trainable
  params/grads/Adam remain fp32. Swapping encoders must never touch the core.
- **One architecture at every scale.** The CPU tests and the T4 run use the
  same classes; only the integers change (`configs/t4_scaled.yaml`).
- **100% streaming, zero disk (Phase 24.2).** `data/sources.open_stream` is
  the ONLY `load_dataset` call site: it forces `streaming=True`, verifies the
  result is an `IterableDataset` (`assert_streaming` raises `TypeError` on a
  disk-backed `Dataset`/`DatasetDict` or an `IterableDatasetDict`), and
  shuffles with `.shuffle(seed=..., buffer_size=10000)` — a RAM-only
  reservoir, capped at `MEDIA_SHUFFLE_BUFFER_CAP=1000` rows for
  `image_embedded`/`audio` kinds (decoded media rows are 100–1000× larger
  than text; 10k of them could OOM the instance). Map-style
  `.shuffle(seed=...)` without a buffer is forbidden on streams.
  `build_mixture` rejects `data.streaming: false`. The trainer loop
  is strictly **step-based** (`while step < max_steps`): no epochs, no
  `len(dataset)`, and sized sample sources are rejected with `TypeError`.
  There is no multi-process DataLoader — `MixtureSampler` + `BatchBuilder`
  are the custom single-process iterable wrapper (`num_workers=0` semantics;
  HF streaming must never run in worker subprocesses). The notebook's
  Zero-Disk cell (right after the mix audit, before training) prints every
  stream's type and asserts ≤ 32 MB disk growth over 10 streamed batches.
  Tests mock `datasets.load_dataset` with tiny in-memory `IterableDataset`s
  (`mock_streaming_hub`); a session guard makes any REAL Hub load raise.
  `tests/test_streaming.py` enforces all of this — do not weaken it.
- **PT-BR is the primary language.** The streaming mix is exactly
  50% `ptbr` / 25% `code` / 15% `multimodal` / 10% `english` documents and it
  is enforced **per 100-draw window** (largest-remainder quota + seeded
  shuffle), not by a per-item `rng.random()` draw. Do not restore the
  Bernoulli sampler, do not add a fifth bucket, and do not "temporarily"
  rebalance toward English. `tests/test_sources.py` asserts every window.
- **The mixing unit is documents.** 50% PT-BR documents is not 50% PT-BR
  bytes; `audit_mixture` prints both shares and the notebook gates on the
  document share. Never quote the byte share as if it were the quota.
- **Masking is one contract for all four groups.** A single
  `ignore_index=PAD(258)` cross-entropy and a single token-weighted confidence
  BCE cover text, code, English and multimodal rows (whose prompts are
  masked). A fully masked batch returns a zero loss, never `0/0` NaN. Do not
  add per-source losses or per-source weights without re-running
  `tests/test_train.py` masking tests.
- **Unmeasured is reported as unmeasured.** This environment has no GPU.
  VRAM/throughput/bf16 behaviour are claims that require the Colab run.
  Never present an analytic estimate as a measurement. `--simulate-t4` is an
  explicit estimate; `Trainer.probe_vram` is the real CUDA authority. The
  production hard ceiling is 13.5 GB and the Phase 25.1 design target is
  <10 GB simulated: batch 1×32, all 8 blocks plus attention/FFN/adapter
  sub-boundaries checkpointed, representative probe (memory/confidence/policy/
  AdamW + efficient-backend verification for self/chunk/cross masks), batch autotune, and fail-closed if
  still over. The 1024-byte window is not silently shrunk.
- **T4 cannot bf16.** `precision: auto` resolves to fp16 + GradScaler on
  sm_75. Never hardcode bf16 autocast on a T4 path.

## Mandatory architectural workflow

For architecture work, report `[THINK]`, `[MAP]`, `[SOLVE]`, `[⚠️ WARN]`,
`[EXECUTE]`, `[UPDATE]`, `[BENCHMARK]`. Deliver complete, runnable PyTorch
code; update docs through file I/O; run the test suite; separate observed
results from goals; never claim a target, speed-up or safety property
without evidence produced in this repository.

## Repository map (Phase 25.1, stripped to the bone)

```
models/foundation.py     byte Transformer + causal sliding-window SDPA + GatedSkillAdapter
                         + memory cross-attention + confidence loop
                         + LatentThoughtBuffer + token-id tools + checkpoint I/O
models/memory_model.py   unified dynamic bank (episodic alias), 1024/8192,
                         learnable decay/importance + Forever guard
models/encoders.py       frozen encoders, skill projectors, uint8 codec
training/train.py        the ONE training script (+ sandboxed search runtime)
data/sources.py          streaming PT-BR-first 50/25/15/10 quota mix
                         + strict-streaming enforcement (open_stream)
                         + mix audit/report + byte packing
configs/t4_scaled.yaml   production preset (8 layers, 16384 rows, W=1024, 8192 memory)
tests/                   199-test regression suite (CPU, ~11 s, fully offline)
pretrain_4h.ipynb        the ONLY notebook (T4, ~4 h, HF checkpoints,
                         mix audit + Zero-Disk verification before training)
smoke_test_coherence.py  CPU smoke + coherence probe (thought-buffer norms)
coherence_preflight.py   Phase 25.2 500-step PT-BR/code coherence pre-flight
README.md  ARCHITECTURE_AUDIT.md  AGENTS.md
requirements.txt  .gitignore  package __init__.py files
```

Everything else was deleted in Phase 24: legacy phase scripts, synthetic
generators, evaluation harnesses, demo notebooks, `runs/` artefacts,
`react.py`/`sandbox.py` regex machinery, the 3-tier memory, holdout guards,
contradiction graphs. `tests/test_repo_hygiene.py` fails if they come back.
Do not add files outside this map without updating the hygiene test and this
document in the same change.

## Commands

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                 # must stay 100% green
python -m training.train --smoke           # CPU plumbing validation
python smoke_test_coherence.py --steps 4   # + thought-buffer activation norms
python -m training.train --config configs/t4_scaled.yaml \
    --hf-repo gabrielbosse9/airesearch --max-minutes 220 --checkpoint-minutes 30
```

## Change discipline

- New model knobs go in `FoundationConfig` (validated, rejected if unknown)
  and in `configs/t4_scaled.yaml`; checkpoint schema bumps are explicit.
- New data sources go in `DEFAULT_SOURCES` with a fallback chain and a
  license-respecting id; the mixture must stay 50/25/15/10
  (`ptbr`/`code`/`multimodal`/`english`) unless the spec changes, and both
  `GROUP_WEIGHTS` and `configs/t4_scaled.yaml` must move together.
- The notebook's mix-audit cell must run **before** `training.train` and must
  assert `report["quotas_ok"]`. A run that starts without a PASS is a bug.
  The Zero-Disk verification cell must sit **between** the audit and the
  training launch and must assert both the `IterableDataset` types and the
  negligible disk growth (`tests/test_notebook.py` enforces the order).
- New dataset-loading code must go through `data/sources.open_stream` (or
  extend it); a second `load_dataset` call site fails
  `tests/test_streaming.py`. Keep the suite offline: mock the Hub with
  `mock_streaming_hub`, never stream from CI.
- Every behavioural claim needs a test. The suite is the contract.
- Checkpoints: atomic writes, resumable, projector-only encoder persistence.
  Never break `load_checkpoint` compatibility without a schema bump.
