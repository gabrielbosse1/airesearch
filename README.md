# ByteCore-DualMemory

A byte-level causal Transformer (**366,316,833 parameters** in its production
preset) with a **16,384-byte causal BigBird window**, an **8192-dim unified
skill/memory space** read through a **Gated Skill Adapter + context-to-memory
cross-attention in every layer**, a confidence-headed **latent thought matrix**,
learnable per-slot **decay/Forever eviction**, and frozen-encoder multimodal
ingestion.
Trained by **one script** on real streamed Hugging Face data, on a single
free Colab T4, from **one notebook**.

**Portuguese (Brazil) is the primary language** (Phase 24.1): 50% of every
training window is PT-BR text, with code, multimodal and English as support
skills. The quota is enforced per 100-draw window and audited in the notebook
before the GPU budget is spent — see
[Data mix (PT-BR first)](#data-mix-pt-br-first).

**100% streaming, zero disk (Phase 24.2):** the corpora (the-stack, GigaVerbo,
fineweb) are far larger than a Colab disk. Every `load_dataset` call is forced
to `streaming=True` and type-checked to return an `IterableDataset`; shuffling
uses a RAM-only `buffer_size=10000` reservoir; the training loop is strictly
step-based; and a **Zero-Disk verification cell** proves negligible disk growth
before any training starts — see [Strict streaming](#strict-streaming-phase-242).

There is no tokenizer: the model reads and writes raw bytes `0..255`. There
is no regex tool parsing: `[TOOL_SEARCH]` is a vocabulary id that triggers a
sandboxed subprocess, and `[EDIT]` is ordinary vocabulary learned by
cross-entropy. The authoritative math is in
[ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md); operating rules are in
[AGENTS.md](AGENTS.md).

## Core ideas

1. **Causal BigBird at 16k.** Each byte attends to a local W=256 window,
   R=32 seeded random past keys per head, and G=16 prefix globals, always
   intersected with `j ≤ i`. A shared `(1,1,L,L)` boolean base plus lazy
   per-head overlays is passed to SDPA. This prunes connectivity, **not**
   asymptotic FLOPs; it is not a block-sparse kernel or infinite context.
2. **Gated Skill Adapter + teleportation hub (1024 ↔ 8192).** The 1024-dim hidden state queries
   1024-dim memory keys (high-fidelity matching), retrieves the 8192-dim skill
   value, and compresses it with a **GLU, not a linear blend**:
   `8192 → 2048 → split → Information ⊙ sigmoid(Filter) → 1024`.
   The sigmoid gate is a sharp bouncer: channels are passed or vetoed, not
   averaged into mush. In parallel, every layer runs context-to-memory
   cross-attention (Q=1024 hidden, K=1024 bank keys, V=8192 bank values,
   zero-init ReZero gate) to bridge gaps beyond the 16k window. Injected
   residually in **all 8 layers**.
3. **Multimodal matrices.** Frozen encoders (ViT/ResNet for images/video,
   Wav2Vec2 for audio) feed trainable `Linear(encoder_dim, 8192)` projectors.
   The core model only ever sees the 8192-dim vector — swap the encoder, keep
   the core. Training is **bidirectional**: caption bytes conditioned on the
   media matrix, and media codec bytes conditioned on a text matrix.
4. **Model-decided adaptive loop.** A weight-tied refinement cell runs up to
   4 passes; a `Linear(1024, 1) → sigmoid` confidence head breaks early above
   0.9 at inference. An auxiliary BCE loss teaches it to recognise low-entropy
   (decided) states.
5. **Latent Thought Matrix (Phase 24.4).** Inside the loop, an isolated
   `(B, 4, 1024)` scratchpad thinks without emitting bytes: bidirectional
   cross-attention lets the buffer read the context and the context frontier
   read the buffer (slot `t` updates from slot `t-1` + context). The
   confidence head evaluates the **combined** state, so an uncertain buffer
   keeps the loop running. The buffer **never** passes through the byte
   head, and only the frontier position reads it back — no future byte
   leaks through the scratchpad (both contracts are test-asserted).
6. **Unified dynamic memory.** One physical 2048-slot bank scores
   `importance − decay·age`; short-term slots decay and are evictable, Forever
   slots have exactly zero effective decay, higher importance, and a hard
   overwrite guard. A policy loss rewards stale eviction and penalizes Forever
   overwrite attempts. Writes are detached and delayed until after the current
   batch, so cross-attention cannot retrieve same-batch futures.

## Repository map

| Path | Contents |
|---|---|
| `models/foundation.py` | Byte Transformer, causal BigBird mask, GatedSkillAdapter + memory cross-attention, confidence loop, LatentThoughtBuffer, token-id tools, checkpoint I/O |
| `models/memory_model.py` | Unified DualMemory alias: one 1024-key/8192-value bank, learnable decay/importance, Forever guard, detached delayed writes |
| `models/encoders.py` | Frozen encoders, SkillProjector / SkillKeyProjector / TextSkillProjector, uint8 matrix codec, ingestion |
| `training/train.py` | The ONE training script: precision, representative VRAM probe/simulator/autotune, bidirectional batches, CE + confidence + policy loss, 30-min HF checkpoints, sandboxed search |
| `data/sources.py` | Streaming PT-BR-first 50/25/15/10 quota mix, strict-streaming enforcement (`open_stream`), mix audit, byte packing |
| `configs/t4_scaled.yaml` | Production preset: 8 layers, 1024-dim, 16384 BigBird context, 8192 memory, batch 1×32, 13.5 GB ceiling |
| `tests/` | 191 tests (fully offline): BigBird causality/mask math, cross-attention, decay/Forever policy, GLU, adaptive loop, thought matrix, tooling, trainer/probe/simulator, quota mix, masking, streaming, notebook, hygiene |
| `pretrain_4h.ipynb` | The ONLY notebook: end-to-end ~4 h T4 pre-training |
| `smoke_test_coherence.py` | CPU smoke + coherence probe: pipeline health, thought-buffer activation norms, greedy continuation |

## Quickstart

```bash
pip install -r requirements.txt

# Full regression suite (CPU, ~10 s, fully offline: the HF Hub is mocked).
python -m pytest tests/ -q

# Full-pipeline plumbing validation on CPU (tiny model, synthetic fixtures).
python -m training.train --smoke

# Smoke + coherence probe: also prints the latent thought-buffer activation
# norms (proof the scratchpad is used) and a greedy byte continuation.
python smoke_test_coherence.py --steps 4

# Analytic T4 VRAM preflight (SIMULATION, not a CUDA measurement).
python -m training.train --simulate-t4 --config configs/t4_scaled.yaml

# Real pre-training (GPU machine; streams HF datasets, checkpoints to the Hub).
python -m training.train --config configs/t4_scaled.yaml \
    --hf-repo gabrielbosse9/airesearch --max-minutes 220 --checkpoint-minutes 30
```

On Colab: open `pretrain_4h.ipynb`, select a T4 runtime, add the `HF_TOKEN`
secret, run all cells. Checkpoints land in `gabrielbosse9/airesearch` every
30 minutes; re-running resumes automatically.

## Data mix (PT-BR first)

| Weight | Group | Sources (first that works wins, then fallbacks) |
|---|---|---|
| **50%** | **ptbr** | `TucanoBR/GigaVerbo` → `HuggingFaceFW/fineweb-2` (`por_Latn`) |
| 25% | code | `bigcode/the-stack` Python + JavaScript → `the-stack-smol` |
| 15% | multimodal | COYO image/caption → MSCOCO-5k embedded → LibriSpeech audio |
| 10% | english | `HuggingFaceFW/fineweb` (`sample-10BT`) |

The sampler is **quota-exact, not on average**: `MixtureSampler` allocates each
window of 100 draws with the largest-remainder method (→ exactly 50/25/15/10)
and shuffles it with the seeded RNG, so any multiple of 100 documents carries
the target ratio while document order stays stochastic. A dead source chain is
dropped and the survivors renormalised (recorded in `failures`, never silent).
`open-web-math` is not part of the PT-BR-first mix; there is no fifth bucket.

`pretrain_4h.ipynb` cell 6 pulls **100 real batches** through the mixture and
prints the realised mix before training starts. Both quantities are shown and
they are not the same thing:

```
PT-BR-first data mix audit  |  100 batches x 1 = 100 documents  (0.0 s)
source        target    docs    doc %       bytes   byte %   status
-----------------------------------------------------------------------
PT-BR          50.0%     200    50.0%      27,156    44.5%   ok
Code           25.0%     100    25.0%      20,967    34.4%   ok
Multimodal     15.0%      60    15.0%       7,740    12.7%   ok
English        10.0%      40    10.0%       5,175     8.5%   ok
-----------------------------------------------------------------------
total         100.0%     400   100.0%      61,038   100.0%
verdict: PASS -- document share within +/-2.0 pp of target (quota window = 100)
```

(Rehearsed here on synthetic streams — the document share is the real,
test-asserted behaviour; byte counts depend on the live datasets.) The weights
govern **documents**; a code file is far longer than an image caption, so the
byte share will never equal 50/25/15/10. Multimodal documents also expand into
two training rows (caption + media codec). Tool/edit trajectories are injected
*on top* at `tool_trajectory_rate=0.02` and are outside the quota by design.
The trainer logs the realised mix in `runs/metrics.jsonl` (`data_mix`) and in
the run summary.

## Strict streaming (Phase 24.2)

The corpora above are **massive** — a wholesale download would fill the Colab
disk and crash the instance. The pipeline therefore guarantees streaming end
to end, and the guarantee is tested, not assumed:

- **One call site.** `data/sources.open_stream` is the only place that calls
  `load_dataset`. It forces `streaming=True` (a tampered spec raises before
  any load) and requires a concrete split.
- **Type verification.** `assert_streaming` checks the returned object is a
  `datasets.IterableDataset`. A disk-backed `Dataset`/`DatasetDict` or an
  unresolved `IterableDatasetDict` raises `TypeError` immediately.
  `build_mixture` also rejects `data.streaming: false` in the config.
- **Iterable-safe shuffling.** Streams are shuffled with
  `.shuffle(seed=..., buffer_size=10000)` — a seeded, RAM-only reservoir
  (`data.shuffle_buffer_size` in the config). Embedded-image/audio fallbacks
  are capped at **1,000 rows** (`MEDIA_SHUFFLE_BUFFER_CAP`): the reservoir
  holds *decoded* media rows, and 10k images/waveforms would be 1–2 GB of RAM
  next to the model. COYO (`image_url`) stays at 10k — it buffers only
  url+caption strings. The map-style `.shuffle(seed=...)` that fails on an
  `IterableDataset` is never used.
- **Step-based training.** The loop is `while step < max_steps` — no epochs,
  no `len(dataset)` (streams have no length). Sized sample sources are
  rejected with `TypeError`. There is no multi-process DataLoader:
  `MixtureSampler` + `BatchBuilder` are a custom single-process iterable
  wrapper (the `num_workers=0` configuration HF recommends for streaming),
  so worker deadlocks and disk prefetch are structurally impossible.
- **Zero-Disk verification cell.** Right after data initialization (and
  before the 4-hour launch), the notebook prints each group's stream type,
  asserts `IterableDataset`, then measures disk usage before/after the first
  **10 streamed batches** and asserts growth ≤ 32 MB (streaming metadata is
  KB-scale; a real download would be GB-scale).
- **Offline tests.** `tests/test_streaming.py` mocks `datasets.load_dataset`
  with tiny in-memory `IterableDataset`s (`mock_streaming_hub`), so the
  enforcement path runs for real with zero network and zero disk; a
  session-wide guard makes any *real* Hub load raise during the suite.

## Honest notes

- **T4 precision:** sm_75 cannot run bf16 autocast. The trainer resolves
  `auto` to **fp16 + GradScaler** there (bf16 on Ampere+), and logs which.
- **Masking is source-agnostic.** Every group trains under the same
  `ignore_index=258` cross-entropy; caption prompts and codec prefixes are
  masked, padding is masked, and a fully masked batch returns a zero loss
  instead of `0/0` NaN. CE and the auxiliary confidence BCE are means over
  non-PAD tokens, so a 50% PT-BR batch is the token-weighted blend of its
  parts — asserted in `tests/test_train.py`.
- **4 hours is a small token budget** for 366M parameters (order 10⁸ tokens).
  Expect falling loss and byte-level structure, not a finished assistant.
- **Media bytes** in the text→media direction are the deterministic uint8
  codec of the encoder representation (tanh-companded), not raw pixels —
  that is the matrix approach: the model predicts representations.
- VRAM figures in the audit are **analytic** unless explicitly marked
  measured; this repository's CI environment has no GPU. The production
  simulator estimates 13.073 GB / 13.5 GB; the real CUDA probe enforces the
  ceiling and fails closed if batch/R/G autotune cannot fit.
- **BigBird masks are connectivity-sparse, not FLOP-sparse.** SDPA still faces
  an L² score problem; lowering R/G may not shrink dense mask allocation.
