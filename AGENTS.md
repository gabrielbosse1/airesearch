# AGENTS.md — ByteCore-DualMemory operating protocol (Phase 24)

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
- **Memory is a store, not a gradient update.** Bank writes are
  `@torch.no_grad()` and detach. Only reads (and batch-local projector views)
  are differentiable. Keys are 1024-dim, values 8192-dim, in every bank.
- **Encoders are frozen.** `requires_grad_(False)`, eval mode forever,
  forward under `no_grad`. Only the `SkillProjector`/`SkillKeyProjector`/
  `TextSkillProjector` train. Swapping encoders must never touch the core.
- **One architecture at every scale.** The CPU tests and the T4 run use the
  same classes; only the integers change (`configs/t4_scaled.yaml`).
- **Unmeasured is reported as unmeasured.** This environment has no GPU.
  VRAM/throughput/bf16 behaviour are claims that require the Colab run.
  Never present an analytic estimate as a measurement.
- **T4 cannot bf16.** `precision: auto` resolves to fp16 + GradScaler on
  sm_75. Never hardcode bf16 autocast on a T4 path.

## Mandatory architectural workflow

For architecture work, report `[THINK]`, `[MAP]`, `[SOLVE]`, `[⚠️ WARN]`,
`[EXECUTE]`, `[UPDATE]`, `[BENCHMARK]`. Deliver complete, runnable PyTorch
code; update docs through file I/O; run the test suite; separate observed
results from goals; never claim a target, speed-up or safety property
without evidence produced in this repository.

## Repository map (Phase 24, stripped to the bone)

```
models/foundation.py     byte Transformer + GatedSkillAdapter + confidence
                         loop + token-id tool generation + checkpoint I/O
models/memory_model.py   DualMemory (episodic + procedural), 1024/8192 banks
models/encoders.py       frozen encoders, skill projectors, uint8 codec
training/train.py        the ONE training script (+ sandboxed search runtime)
data/sources.py          streaming 40/30/20/10 mix + byte packing
configs/t4_scaled.yaml   production preset (8 layers, 4096 ctx, 8192 memory)
tests/                   127-test regression suite (CPU, ~3 s)
pretrain_4h.ipynb        the ONLY notebook (T4, ~4 h, HF checkpoints)
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
python -m training.train --config configs/t4_scaled.yaml \
    --hf-repo gabrielbosse9/airesearch --max-minutes 220 --checkpoint-minutes 30
```

## Change discipline

- New model knobs go in `FoundationConfig` (validated, rejected if unknown)
  and in `configs/t4_scaled.yaml`; checkpoint schema bumps are explicit.
- New data sources go in `DEFAULT_SOURCES` with a fallback chain and a
  license-respecting id; the mixture must stay 40/30/20/10 unless the spec
  changes.
- Every behavioural claim needs a test. The suite is the contract.
- Checkpoints: atomic writes, resumable, projector-only encoder persistence.
  Never break `load_checkpoint` compatibility without a schema bump.
