# AGENTS.md — ByteCore-DualMemory operating protocol (Phase 24.2)

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
- **100% streaming, zero disk (Phase 24.2).** `data/sources.open_stream` is
  the ONLY `load_dataset` call site: it forces `streaming=True`, verifies the
  result is an `IterableDataset` (`assert_streaming` raises `TypeError` on a
  disk-backed `Dataset`/`DatasetDict` or an `IterableDatasetDict`), and
  shuffles with `.shuffle(seed=..., buffer_size=10000)` — a RAM-only
  reservoir. Map-style `.shuffle(seed=...)` without a buffer is forbidden on
  streams. `build_mixture` rejects `data.streaming: false`. The trainer loop
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
data/sources.py          streaming PT-BR-first 50/25/15/10 quota mix
                         + strict-streaming enforcement (open_stream)
                         + mix audit/report + byte packing
configs/t4_scaled.yaml   production preset (8 layers, 4096 ctx, 8192 memory)
tests/                   160-test regression suite (CPU, ~6 s, fully offline)
pretrain_4h.ipynb        the ONLY notebook (T4, ~4 h, HF checkpoints,
                         mix audit + Zero-Disk verification before training)
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
