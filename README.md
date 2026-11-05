# ByteCore-DualMemory

A byte-level causal Transformer (**265,618,710 parameters** in its production
preset) that reads an **8192-dim skill/memory space** through a **Gated Skill
Adapter** in every layer, decides its own "thinking" budget with a confidence
head, and ingests images/audio/video as frozen-encoder skill matrices.
Trained by **one script** on real streamed Hugging Face data, on a single
free Colab T4, from **one notebook**.

There is no tokenizer: the model reads and writes raw bytes `0..255`. There
is no regex tool parsing: `[TOOL_SEARCH]` is a vocabulary id that triggers a
sandboxed subprocess, and `[EDIT]` is ordinary vocabulary learned by
cross-entropy. The authoritative math is in
[ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md); operating rules are in
[AGENTS.md](AGENTS.md).

## The three ideas

1. **Gated Skill Adapter (1024 ↔ 8192).** The 1024-dim hidden state queries
   1024-dim memory keys (high-fidelity matching), retrieves the 8192-dim skill
   value, and compresses it with a **GLU, not a linear blend**:
   `8192 → 2048 → split → Information ⊙ sigmoid(Filter) → 1024`.
   The sigmoid gate is a sharp bouncer: channels are passed or vetoed, not
   averaged into mush. Injected residually in **all 8 layers**.
2. **Multimodal matrices.** Frozen encoders (ViT/ResNet for images/video,
   Wav2Vec2 for audio) feed trainable `Linear(encoder_dim, 8192)` projectors.
   The core model only ever sees the 8192-dim vector — swap the encoder, keep
   the core. Training is **bidirectional**: caption bytes conditioned on the
   media matrix, and media codec bytes conditioned on a text matrix.
3. **Model-decided adaptive loop.** A weight-tied refinement cell runs up to
   4 passes; a `Linear(1024, 1) → sigmoid` confidence head breaks early above
   0.9 at inference. An auxiliary BCE loss teaches it to recognise low-entropy
   (decided) states.

## Repository map

| Path | Contents |
|---|---|
| `models/foundation.py` | Byte Transformer, GatedSkillAdapter, confidence-headed refinement, token-id tool generation, checkpoint I/O |
| `models/memory_model.py` | DualMemory: episodic + procedural banks, 1024-dim keys / 8192-dim values, FIFO, gradient-free writes |
| `models/encoders.py` | Frozen encoders, SkillProjector / SkillKeyProjector / TextSkillProjector, uint8 matrix codec, ingestion |
| `training/train.py` | The ONE training script: precision/VRAM, bidirectional batches, CE + confidence BCE, 30-min HF checkpoints, sandboxed search |
| `data/sources.py` | Streaming 40/30/20/10 mix with fallback chains, byte packing |
| `configs/t4_scaled.yaml` | Production preset: 8 layers, 1024-dim, 4096 context, 8192 memory |
| `tests/` | 127 tests: GLU math, memory, projection, adaptive loop, tooling, trainer, sources, notebook, hygiene |
| `pretrain_4h.ipynb` | The ONLY notebook: end-to-end ~4 h T4 pre-training |

## Quickstart

```bash
pip install -r requirements.txt

# Full regression suite (CPU, ~3 s).
python -m pytest tests/ -q

# Full-pipeline plumbing validation on CPU (tiny model, synthetic fixtures).
python -m training.train --smoke

# Real pre-training (GPU machine; streams HF datasets, checkpoints to the Hub).
python -m training.train --config configs/t4_scaled.yaml \
    --hf-repo gabrielbosse9/airesearch --max-minutes 220 --checkpoint-minutes 30
```

On Colab: open `pretrain_4h.ipynb`, select a T4 runtime, add the `HF_TOKEN`
secret, run all cells. Checkpoints land in `gabrielbosse9/airesearch` every
30 minutes; re-running resumes automatically.

## Data mix (streaming)

| Weight | Group | Sources (first that works wins, then fallbacks) |
|---|---|---|
| 40% | code | `bigcode/the-stack` Python + JavaScript → `the-stack-smol` |
| 30% | language | `TucanoBR/GigaVerbo` (PT-BR) + `HuggingFaceFW/fineweb` (EN) |
| 20% | multimodal | COYO image/caption → MSCOCO-5k embedded → LibriSpeech audio |
| 10% | logic | `open-web-math/open-web-math` |

## Honest notes

- **T4 precision:** sm_75 cannot run bf16 autocast. The trainer resolves
  `auto` to **fp16 + GradScaler** there (bf16 on Ampere+), and logs which.
- **4 hours is a small token budget** for 265M parameters (order 10⁸ tokens).
  Expect falling loss and byte-level structure, not a finished assistant.
- **Media bytes** in the text→media direction are the deterministic uint8
  codec of the encoder representation (tanh-companded), not raw pixels —
  that is the matrix approach: the model predicts representations.
- VRAM figures in the audit are **analytic** unless explicitly marked
  measured; this repository's CI environment has no GPU.
