# ByteCore-DualMemory

A byte-level causal Transformer (`183,830,791` parameters in its production
preset) plus an external hierarchical memory store, trained on real streamed
Hugging Face data in three phases on a single Colab T4.

There is no tokenizer: the model reads and writes raw bytes `0..255`. There is
also no diffusion core, no pointer-generator copy head and no control-byte frame
parser — those existed in earlier phases and were deleted. The authoritative
description of what the code does is [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md);
the operating rules are [AGENTS.md](AGENTS.md).

## Repository map

| Path | Contents |
|---|---|
| `models/foundation.py` | RMSNorm, SwiGLU, RoPE, GQA, iterative refinement, the causal byte Transformer, checkpoint I/O |
| `models/memory_model.py` | episodic / semantic-exemplar / procedural store, importance scorer, top-K reads |
| `models/foundation_memory.py` | memory injected as `[MEM] topic = value` prefix bytes |
| `models/react.py`, `models/sandbox.py` | `Thought/Action/Observation` tool use as bytes + sandboxed execution, `[REF]` resolution |
| `models/holdout.py`, `models/procedural.py`, `models/contradiction_graph.py` | provenance guard, byte-exact procedure replay, temporal fact graph |
| `data/phase22_sources.py` | the real Hugging Face dataset registry and streaming converters |
| `training/phase22_common.py` | `BytePacker`, precision resolution, VRAM ladder, checkpoints, Hub I/O |
| `training/phase22_reward.py` | sandbox-RL reward, `[PAST_ACTION]`/`[CURRENT_DRAFT]`, `[EDIT]` commands |
| `training/phase22_train.py` | the three-phase trainer |
| `evaluation/final_agent_eval.py`, `evaluation/phase22_benchmark.py` | held-out probes, parameter/VRAM/inventory report |
| `notebook/build_phase22_notebooks.py` + `01/02/03_*.ipynb` | the T4 pipeline |
| `configs/default.yaml`, `configs/t4_scaled.yaml` | CPU/test preset, production preset |
| `tests/` | 92 regression tests |

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

```bash
python -m pytest -q                              # 92 passed, 1 skipped
python -m evaluation.phase22_benchmark           # writes runs/phase22/*.json
python -m training.phase22_train --self-test     # CPU plumbing smoke test
```

The self-tests run on repository bytes and synthetic trajectories. They validate
plumbing; they produce no model result.

## The model

`CausalByteTransformer` under `configs/t4_scaled.yaml`:

| | |
|---|---|
| depth / width | `n_layers=12`, `d_model=1024` |
| attention | 8 query heads, 2 KV heads (`d_h=128`), causal, fused SDPA |
| FFN | SwiGLU `1024→4096→1024`, bias-free |
| norm | pre-norm RMSNorm, `eps=1e-6` |
| positions | RoPE, `θ=10000`, no learned table |
| refinement | 2 weight-tied passes, rank 256, `α` initialised to 0 |
| vocabulary | 261 (bytes + `BOS/EOS/PAD/REF_START/REF_END`) |
| **parameters** | **183,830,791** |

A forward pass maps `(B,L)` ids to `(B,L,261)` logits through 12 pre-norm blocks,
two refinement passes and a final RMSNorm. `gradient_checkpointing: true` keeps
only block boundaries in memory.

## Memory

`MemoryModel(d_model=64, capacity=512)` keeps three tiers in a shared space —
episodic `(256,64)`, semantic exemplars `(128,4,64)`, procedural `(128,64)` with
`(128,4,64)` latent traces — scored by

```
logit = q·k/√d − decay_rate·write_age/50 + 0.35·importance
```

with a learned, bounded importance scorer and `softmax` over the selected top-K.
Retrieval returns the raw top-K vectors (`(K,d)`) or their weighted pool; a
straight-through estimator provides the training-time gradient. Memory reaches a
prompt as prefix bytes, never as a side-channel tensor.

## Data and training (three phases)

Phase 1 is raw next-byte causal LM over real code (`bigcode/the-stack`), PT-BR
(`TucanoBR/GigaVerbo`, `HuggingFaceFW/fineweb-2 por_Latn`), English
(`fineweb-edu`, DCLM) and math (`open-web-math`). Phase 2 distils conversation
(`no_robots` — non-commercial, `Tucano-SFT`, `distilabel-sft-easy`,
`CodeFeedback`) and real Minecraft trajectories (VLA stage 3, MET-Bench) into
`Thought → Action → Observation → Answer` bytes with prompt masking. Phase 3 runs
sandbox RL with a deterministic search tool and the reward in
`training/phase22_reward.py`.

```bash
python -m training.phase22_train --phase 1 --device cuda   # -> checkpoint_phase1.pt
python -m training.phase22_train --phase 2 --device cuda   # -> checkpoint_phase2.pt
python -m training.phase22_train --phase 3 --device cuda   # -> final_model.pt
```

Every checkpoint is `schema=3`: fp32 master weights, AdamW, GradScaler, RNG
state, the lossless `BytePacker` carry, and the dataset manifest. A missing
predecessor is a hard error, not a silent random restart.

## Phase 22: scaled T4 pipeline (wide, gated architecture + real HF data)

`notebook/build_phase22_notebooks.py` regenerates the three resumable Colab
notebooks; each cell is safe to re-run and pulls its checkpoint back from the Hub
if the VM was recycled. The architecture is RMSNorm + SwiGLU + RoPE + GQA +
weight-tied Iterative Refinement at constant depth. On a Tesla T4 (sm_75)
`torch.autocast("cuda", dtype=torch.bfloat16)` raises
`Current CUDA Device does not support bfloat16`, so `precision: auto` resolves to
bfloat16 on Ampere+ and to float16 with `GradScaler` on T4 — the config records
the intent as `requested_precision: bfloat16`.

VRAM is **unmeasured** here: the analytic budget at `B=4, L=1024` with full
checkpointing is ≈3.97 GB, and `vram_fill_ladder()` exists to probe the real
footprint on the T4 rather than to assert one.

## Phase 21: forensic diagnosis of the Phase 19 collapse (historical)

The Phase 19 notebook model collapsed to a degenerate byte distribution (≈81 %
of the probability mass on byte 32) with no NaN/Inf and healthy gradients. That
diagnosis produced three changes still visible in the code: strict phase ordering,
de-contamination reporting that says `unverified` instead of "clean", and the
removal of the clone-based notebook. The forensic scripts were deleted in Phase
23; the evidence remains in `runs/phase21/`.

## Status, honestly

- CPU-only verification here: 92 tests pass; the parameter count, tensor shapes
  and retrieval algebra in the audit are measured, not estimated.
- No checkpoint from the Phase 22 run is in this repository
  (`runs/phase22/checkpoints/` is git-ignored), so no loss, benchmark or agent
  result is claimed.
- `MemoryModel` is not called by the Phase 22 trainer; memory enters prompts as
  `[MEM] … = …` bytes from an injected retriever.
- The Phase 3 search tool is a deterministic stub, not a web search.
