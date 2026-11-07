# ByteCore-DualMemory

A byte-level causal Transformer (**366,316,833 parameters** in its production
preset) with a **16,384-byte packed row** and **causal sliding-window attention
(self + up to 1024 previous bytes)**, an **8192-dim unified
skill/memory space** read through a **Gated Skill Adapter + context-to-memory
cross-attention in every layer**, a confidence-headed **latent thought matrix**,
learnable per-slot **decay/Forever eviction**, and frozen-encoder multimodal
ingestion.
Trained by **one script** on real streamed Hugging Face data, on a single
free Colab T4, from **one notebook**.

**Status: Phase 26 evaluation harness implemented.** The user reports the four-hour T4 run is complete, but this sandbox could not establish a Hugging Face TLS connection, so the public checkpoint has not been fetched or evaluated here. No production fluency, code, tool-use, memory-retrieval, or activation claim is inferred from the training duration. The 500-step coherence pre-flight passed separately at CPU sandbox scale; that is not a substitute for production checkpoint evaluation.

**Portuguese (Brazil) is the primary language** (Phase 24.1): 50% of every
training window is PT-BR text, with code, multimodal and English as support
skills. The quota is enforced per 100-draw window and audited in the notebook
before the GPU budget is spent — see
[Data mix (PT-BR first)](#data-mix-pt-br-first).

**100% streaming, zero disk (Phase 24.2):** the corpora (the-stack, GigaVerbo,
fineweb) are far larger than a Colab disk. Every `load_dataset` call is forced
to `streaming=True` and type-checked to return an `IterableDataset`; shuffling
uses a RAM-only `buffer_size=1000` reservoir; the training loop is strictly
step-based; and a **Zero-Disk verification cell** proves negligible disk growth
before any training starts — see [Strict streaming](#strict-streaming-phase-242).

There is no tokenizer: the model reads and writes raw bytes `0..255`. There
is no regex tool parsing: `[TOOL_SEARCH]` is a vocabulary id that triggers a
sandboxed subprocess, and `[EDIT]` is ordinary vocabulary learned by
cross-entropy. The authoritative math is in
[ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md); operating rules are in
[AGENTS.md](AGENTS.md).

## Core ideas

1. **Causal sliding window at 16k rows.** Each byte attends to itself plus up
   to `window_size=1024` previous bytes (`i-1024 ≤ j ≤ i`). Exact chunk-local
   SDPA calls use small masks and never allocate an `(L,L)` tensor. On T4 the
   forced backend is memory-efficient attention; PyTorch Flash/cuDNN SDPA
   require sm_80+. The 16k row is not a 16k dense self-attention window.
2. **Gated Skill Adapter + teleportation hub (1024 ↔ 8192).** The 1024-dim hidden state queries
   1024-dim memory keys (high-fidelity matching), retrieves the 8192-dim skill
   value, and compresses it with a **GLU, not a linear blend**:
   `8192 → 2048 → split → Information ⊙ sigmoid(Filter) → 1024`.
   The sigmoid gate is a sharp bouncer: channels are passed or vetoed, not
   averaged into mush. In parallel, every layer runs context-to-memory
   cross-attention (Q=1024 hidden, K=1024 bank keys, V=8192 bank values,
   zero-init ReZero gate). With random/global edges removed, this committed
   memory hub is the explicit long-range path beyond the 1024-byte local
   window. Injected residually in **all 8 layers**.
3. **Multimodal matrices.** Frozen encoders (ViT/ResNet for images/video,
   Wav2Vec2 for audio) feed trainable `Linear(encoder_dim, 8192)` projectors.
   The core model only ever sees the 8192-dim vector — swap the encoder, keep
   the core. Training is **bidirectional**: caption bytes conditioned on the
   media matrix, and media codec bytes conditioned on a text matrix.
4. **Model-decided adaptive loop.** A weight-tied refinement cell runs up to
   4 passes; a `Linear(1024, 1) → sigmoid` confidence head breaks early above
   0.9 at inference. An auxiliary logits-based BCE loss, reduced in float32
   outside autocast, teaches it to recognise low-entropy (decided) states.
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
| `models/foundation.py` | Byte Transformer, chunk-local causal sliding-window SDPA, GatedSkillAdapter + memory cross-attention, confidence loop, LatentThoughtBuffer, token-id tools, checkpoint I/O |
| `models/memory_model.py` | Unified DualMemory alias: one 1024-key/8192-value bank, learnable decay/importance, Forever guard, detached delayed writes |
| `models/encoders.py` | Frozen encoders, SkillProjector / SkillKeyProjector / TextSkillProjector, uint8 matrix codec, ingestion |
| `training/train.py` | The ONE training script: precision, representative VRAM probe/simulator/autotune, bidirectional batches, CE + confidence + policy loss, 10-min/100-step HF checkpoints + host-RAM/SIGKILL diagnostics, sandboxed search |
| `data/sources.py` | Streaming PT-BR-first 50/25/15/10 quota mix, strict-streaming enforcement (`open_stream`), mix audit, byte packing |
| `configs/t4_scaled.yaml` | Production preset: 8 layers, 1024-dim, 16384 packed rows, 1024 sliding window, 8192 memory, batch 1×32, 13.5 GB ceiling |
| `tests/` | 269 tests (fully offline): previous architecture/training contracts plus checkpoint snoop selection/loading, production shape verification, attention telemetry, notebook instructions, thought/FFN probes, and syntax-only checks |
| `pretrain_4h.ipynb` | The ONLY notebook: end-to-end ~4 h T4 pre-training |
| `evaluation/deep_snoop.py` | Phase 26 safe checkpoint downloader/loader, production shape checks, greedy generation, thought/FFN/sliding-window/memory attention telemetry |
| `runs/final_model_snoop_report.md` | Generated forensic model card; checkpoint weights and raw telemetry are not tracked |
| `smoke_test_coherence.py` | CPU smoke + coherence probe: pipeline health, thought-buffer activation norms, greedy continuation |
| `coherence_preflight.py` | Phase 25.2 500-step PT-BR/code coherence pre-flight with strict loss/text/syntax assertions |

## Quickstart

```bash
pip install -r requirements.txt

# Full regression suite (CPU, ~17 s, fully offline: the HF Hub is mocked).
python -m pytest tests/ -q

# Full-pipeline plumbing validation on CPU (tiny model, synthetic fixtures).
python -m training.train --smoke

# Smoke + coherence probe: also prints the latent thought-buffer activation
# norms (proof the scratchpad is used) and a greedy byte continuation.
python smoke_test_coherence.py --steps 4

# Phase 25.2 coherence pre-flight: exactly 500 steps, PT-BR + code, strict
# assertions (loss decrease, recognizable PT-BR, valid Python syntax).
python coherence_preflight.py

# Phase 26: fetch and inspect the public trained checkpoint (full inference).
# Optional HF_TOKEN is read from the environment; never paste it into source.
python evaluation/deep_snoop.py --repo-id gabrielbosse9/airesearch2 \
    --device auto --max-new-bytes 128
# Or run offline against a previously downloaded checkpoint:
python evaluation/deep_snoop.py --checkpoint /path/to/checkpoint_latest.pt \
    --device cuda --max-new-bytes 128
```

The snoop writes `runs/final_model_snoop_report.md` and an ignored JSON trace.
It uses safe PyTorch checkpoint loading (`weights_only=True`) and memory-maps the
checkpoint where supported; the optimizer tensors can make a training checkpoint
several gigabytes. Generated code is syntax-checked but never executed. Norms
and attention weights document component activity/routing, not semantic proof.

```bash
# Analytic T4 VRAM preflight (SIMULATION, not a CUDA measurement).
python -m training.train --simulate-t4 --config configs/t4_scaled.yaml

# Real pre-training (GPU machine; streams HF datasets, checkpoints to the Hub).
python -m training.train --config configs/t4_scaled.yaml \
    --hf-repo gabrielbosse9/airesearch2 --max-minutes 220 \
    --checkpoint-minutes 10 --upload-minutes 5
```

### Google Colab (recommended)

1. Select **Runtime → Change runtime type → T4 GPU**.
2. In Colab **Secrets**, add `HF_TOKEN` with **write** access to
   `gabrielbosse9/airesearch2`, and enable notebook access for that secret.
   Do not paste the token into notebook code, git, or chat.
3. Open/run the self-contained notebook:
   [pretrain_4h.ipynb on Colab](https://colab.research.google.com/github/gabrielbosse1/airesearch/blob/main/pretrain_4h.ipynb).
   Its first markdown cell contains the operator steps, and it includes an
   optional debug-bundle fetch cell. The notebook clones the repo, verifies HF
   write access with a tiny preflight upload, audits the stream, checks the real
   CUDA VRAM probe, then launches training with an independent parent monitor.

The default remote is **`gabrielbosse9/airesearch2`**. Checkpoints are saved
locally and uploaded every **10 minutes** or **100 optimizer steps**; a normal
finish also uploads `checkpoints/checkpoint_final.pt`. Logs/diagnostics sync
every **5 minutes**, independently of checkpoint cadence; the notebook parent
also heartbeats every **15 seconds** and uploads diagnostics every **60 seconds**
so startup/SIGKILL evidence does not wait for an optimizer step. Upload failures are
retried and then stop the run by default (`fail_on_upload_error: true`) rather
than silently spending GPU time without a recoverable remote checkpoint.

Saved under `runs/colab_latest/` locally and under `runs/<run_id>/<session_id>/`
on the Hub: `console.log`, `metrics.jsonl`, per-step `events.jsonl`, resolved
config, environment/package/GPU manifest, git provenance + code snapshot,
VRAM probe, host-RAM/cgroup snapshots, `startup_stage.json`,
`notebook_monitor.jsonl`, `launch_exit.json`, preflight reports, final PT-BR/code/tool samples, checkpoint hash /
upload manifest, and a small checkpoint-free `debug_bundle.zip`. A cooperative
interrupt/crash after a committed optimizer step attempts an emergency checkpoint
+ bundle upload; a startup failure preserves diagnostics without overwriting a
recoverable checkpoint with step-0 state. A hard Colab
VM kill cannot run final code; recovery is bounded by the last successful
periodic upload. Re-running restores model, projectors, optimizer, scaler,
memory bank, step, counters and RNG state. Live HF iterators/shuffle buffers are
reopened rather than rewound to an exact document cursor; incomplete gradient
accumulation is discarded. This is honest training-state resume, not a claim of
bit-exact stream replay.

### If Colab reports `exit code -9`

`-9` is SIGKILL. The child cannot run Python cleanup after it. The notebook now:

- initializes trainable weights directly on the GPU instead of building a full
  fp32 CPU copy first;
- avoids creating a CUDA context in the long-lived notebook parent;
- reduces shuffle reservoirs to **1,000 text/URL rows** and **100 decoded media
  rows**, and releases preflight streams before launch;
- checks measured system-RAM headroom (`runtime.min_host_available_gb`, default
  2 GB) before large allocations, separately from the 13.5 GB CUDA VRAM ceiling;
- records parent heartbeats and cgroup OOM counters, then writes a local/remote
  `debug_bundle.zip` even when the child is killed.

Run the notebook's **Fetch debug bundle** cell immediately after the failure,
upload that ZIP to Arena, then **Runtime → Restart session** before retrying.
SIGKILL alone does not prove OOM; `launch_exit.json` distinguishes an observed
cgroup `oom_kill` counter increase from an unconfirmed external kill.

For later debugging, download the small bundle from the repo pointed to by
`runs/latest.json`, or in Colab/local Python:

```python
import json
from huggingface_hub import hf_hub_download
repo = "gabrielbosse9/airesearch2"
pointer = json.loads(hf_hub_download(repo, "runs/latest.json", repo_type="model"))
path = hf_hub_download(repo, f"{pointer['remote_run_path']}/debug_bundle.zip",
                       repo_type="model")
print(path)  # upload this ZIP to the sandbox for inspection
```

Checkpoints are roughly multi-GB; the debug ZIP deliberately excludes them.
Hub commit history retains older checkpoint revisions, so monitor repo storage
if you change the cadence or resume many sessions.

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
The trainer logs the realised mix in `runs/colab_latest/metrics.jsonl` (`data_mix`) and in
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
  `.shuffle(seed=..., buffer_size=1000)` — a seeded, RAM-only reservoir
  (`data.shuffle_buffer_size` in the config). Embedded-image/audio fallbacks
  are capped at **100 rows** (`MEDIA_SHUFFLE_BUFFER_CAP`): the reservoir
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

- **T4 precision:** sm_75 has no native bf16. Newer PyTorch may report bf16
  emulation as supported; the trainer rejects emulation for the forced
  memory-efficient SDPA path and resolves `auto` to **fp16 + GradScaler** on T4
  (native bf16 only on Ampere+). The notebook runs a short-lived precision
  preflight and logs the resolved plan before training.
- **Confidence BCE is autocast-safe (Phase 25.6).** CUDA rejects probability
  BCE inside autocast even after a `.float()` cast. Training scores raw
  confidence logits with `binary_cross_entropy_with_logits` in an explicit
  float32, autocast-disabled reduction; inference keeps the sigmoid probability
  API and the detached entropy targets are unchanged. The notebook runs a tiny
  GPU forward/backward preflight before encoder downloads or the production
  VRAM probe. Checkpoint schema and model geometry are unchanged.
- **Audio VRAM probing is bounded (Phase 25.7).** `Trainer.max_audio_samples`
  is derived from `runtime.max_audio_seconds × 16000` and matches `BatchBuilder`.
  The frozen audio encoder probe uses that bound; non-positive audio limits fail
  before training. The encoder-probe helper is regression-tested with mocked
  CUDA accounting, not presented as a sandbox VRAM measurement.
- **HF secrets are normalized before the environment.** Colab secret providers
  may return a wrapped mapping instead of a plain string. The auth cell accepts
  only explicit token/value fields, falls back to hidden input for unsupported
  shapes, and never prints the secret or wrapper before setting `HF_TOKEN`.
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
  simulator estimates 9.822 GB (<10 GB design target) / 13.5 GB hard ceiling;
  the real CUDA probe enforces the ceiling and fails closed if batch autotune
  cannot fit. The window is not silently shrunk.
- **T4 kernel reality:** sm_75 uses memory-efficient SDPA. PyTorch Flash/cuDNN
  SDPA require sm_80+; the probe verifies the efficient backend (self, chunk-local mask, cross mask) before measuring.
- **Long-range hand-off:** self-attention is local. Information older than the
  window must be retrieved from committed memory through the adapter hub.
- **Coherence pre-flight is measured, but sandbox-scale:** `coherence_preflight.py`
  ran 500 optimizer steps on a deterministic offline 1000-example PT-BR/code
  slice using the Phase 25.1 sliding-window + memory cross-attention path
  (`window_size=1024`, 8 layers, CPU-feasible width). It proved learning and
  coherent continuations, not finished production fluency. The 4-hour Colab run
  is still the production-scale training event.
