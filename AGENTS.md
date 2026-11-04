# AGENTS.md — ByteCore-DualMemory operating protocol (Phase 23)

Read this file before touching the repository, then read
`ARCHITECTURE_AUDIT.md` (the bottom-up mathematical description of the code that
actually exists) and `README.md` (how to run it).

## Mission and invariants

- **Raw bytes only at data boundaries.** UTF-8 text is bytes `0..255`; NUL is
  data and masks distinguish padding. There is no tokenizer anywhere in the tree
  and no dependency on `transformers.AutoTokenizer`, `tiktoken` or equivalents.
- **Fixed vocabulary.** `VOCAB_SIZE = 261`: bytes `0..255` are their own ids,
  then `BOS=256`, `EOS=257`, `PAD=258`, `REF_START=259`, `REF_END=260`.
  `FoundationConfig` raises unless `vocab_size == 261`. Control ids are dropped
  on decode; they are never emitted as output bytes except as markers resolved by
  the host.
- **One architecture at every scale.** `CausalByteTransformer` is the same class
  for the CPU unit tests (`configs/default.yaml`, ~0.5M parameters) and the
  production run (`configs/t4_scaled.yaml`, 183,830,791 parameters). Only the
  integers change.
- **External memory is a store, not a gradient update.** Writes are
  `@torch.no_grad()` storage events; only the retrieval readout is
  differentiable (`retrieve_differentiable`, tangential straight-through
  estimator).
- **Unmeasured is reported as unmeasured.** No GPU exists in this environment:
  VRAM, throughput and bfloat16/float16 behaviour are claims that require the
  Colab T4 notebooks. Never present an analytic estimate as a measurement.

## Mandatory architectural workflow

For architecture work, report `[THINK]`, `[MAP]`, `[SOLVE]`, `[⚠️ WARN]`,
`[EXECUTE]`, `[UPDATE]`, `[BENCHMARK]`. Deliver complete, runnable PyTorch code;
update `AGENTS.md` and `QWEN_CHAT_PROMPT.md` through Python file I/O (never by
hand); run the test suite; separate observed results from goals; never claim a
target, a speed-up or a safety property without evidence produced in this
repository.

## Repository map (Phase 23, cleaned)

```
models/
  foundation.py           byte-level causal Transformer (Levels 1-3 of the audit)
  memory_model.py         episodic / semantic-exemplar / procedural store
  foundation_memory.py    memory as `[MEM] topic = value` prefix bytes
  react.py                Thought/Action/Observation interception + [REF] resolution
  sandbox.py              directory-confined, time/size-bounded tool execution
  holdout.py              SHA-256 provenance guard (opt-in, not a security boundary)
  procedural.py           byte-exact procedure replay records
  contradiction_graph.py  temporal fact graph + learned pair scorer
data/
  phase22_sources.py      real Hugging Face ids, licences, streaming converters
training/
  phase22_common.py       BytePacker, precision, VRAM ladder, checkpoints, Hub I/O
  phase22_reward.py       sandbox-RL reward, [PAST_ACTION]/[CURRENT_DRAFT], [EDIT]
  phase22_train.py        the three-phase trainer (one class, `--phase 1|2|3`)
evaluation/
  final_agent_eval.py     four de-contaminated held-out probes
  phase22_benchmark.py    parameter/VRAM/dataset inventory report writer
notebook/
  build_phase22_notebooks.py   regenerates 01/02/03 notebooks
  01_setup_and_pretrain.ipynb  Phase 1 -> checkpoint_phase1.pt (+ fp16 copy)
  02_distill_and_align.ipynb   Phase 2 -> checkpoint_phase2.pt
  03_rl_and_agent.ipynb        Phase 3 -> final_model.pt
configs/
  default.yaml            small CPU/test config (Phase 16 defaults)
  t4_scaled.yaml          the Phase 22 production config + data + VRAM policy
tests/
  test_foundation.py      primitives, blocks, parameter budget, checkpoints
  test_memory_model.py    tiers, retrieval algebra, STE read, guard
  test_react_runtime.py   action grammar, sandbox, references, prefix memory
  test_phase22.py         scaled stack, precision, packer, reward, data, eval
```

Anything not listed above does not exist in this repository. The deletion record
is Appendix B of `ARCHITECTURE_AUDIT.md`.

## Current architecture — verified facts

**Generator (`models/foundation.py`).** `d_model=1024`, `n_layers=12`,
`n_heads=8`, `n_kv_heads=2` (`d_h=128`, 4 query heads per KV head), `d_ff=4096`,
`max_seq_len=2048`, `dropout=0.0`, `norm_type=rmsnorm`, `ffn_type=swiglu`,
`position_type=rope`, `refinement_steps=2`, `refinement_rank=256`,
`norm_eps=1e-6`, `gradient_checkpointing=true`.

| Component | Parameters |
|---|---|
| token embedding 261×1024 | 267,264 |
| 12 blocks × 15,208,960 | 182,507,520 |
| refinement (3·1024·256 + norm + 2 scales) | 787,458 |
| final RMSNorm | 1,024 |
| head 1024×261 (+261 bias) | 267,525 |
| **total** | **183,830,791** |

Per block: attention 2,624,000 + SwiGLU 12,582,912 + two RMSNorm 2,048.

**Forward pass.** `(B,L)` ids → `(B,L,1024)` embeddings → 12 pre-norm blocks
(attention then FFN, each with a residual) → 2 refinement passes
`h ← h + α_t·Cell(RMSNorm(h))` → final RMSNorm → `(B,L,261)` logits. RoPE is
applied to `q`/`k` only; there is no learned position table in the `rope` preset.

**Memory (`models/memory_model.py`).** Three tiers in one `d_model` space:
episodic `(capacity/2, d)`, semantic exemplar slots
`(capacity/4, n_exemplars=4, d)` with a mixture readout
`Σ_m softmax_m(q·k/√d + log π)·v`, procedural `(capacity/4, d)` plus
`(capacity/4, max_trace_steps, d)` latent traces. Scoring is
`q·k/√d − decay_rate·write_age/50 + 0.35·importance`; retrieval keeps top-K and
normalises with `softmax` over the selected logits. Default
`d_model=64, capacity=512` gives 100,205 parameters and 256/128/128 slots.

**Runtime (`models/react.py`, `models/sandbox.py`).** One regex intercepts
`Action: tool("arg")` in the byte stream; everything the model wrote after that
action line is discarded (`truncate_at_action`), the tool runs in the sandbox, and
a real `Observation:` (collapsed to ≤160 bytes, one line) is appended. `[REF]n[/REF]`
arguments are resolved against marked spans of the request, so a tool never
receives an identifier the model could not copy.

## Phase 22: scaled T4 pipeline (wide, gated architecture + real HF data)

`configs/t4_scaled.yaml` is the production preset; `notebook/build_phase22_notebooks.py`
is the only supported way to produce the three Colab notebooks. The stack is
**RMSNorm + SwiGLU + RoPE + GQA + weight-tied Iterative Refinement** at constant
depth (`n_layers=12`) and width (`d_model=1024`):

- **RMSNorm**: `x / sqrt(E[x²] + eps) ⊙ w`, reduction in float32, no mean
  subtraction, no bias; `eps = 1e-6`.
- **SwiGLU**: `W_out(SiLU(x W_value) ⊙ (x W_branch))`, all projections bias-free,
  `3·1024·4096 = 12,582,912` parameters per layer.
- **RoPE**: `θ^(-2i/d_h)` frequencies, pairs `(i, i+64)`, tables built per
  forward pass, zero parameters, no `max_seq_len`-sized table.
- **GQA**: `n_kv_heads=2`, so `K`/`V` projections are 25 % of MHA and
  `F.scaled_dot_product_attention(..., enable_gqa=True)` is used with a
  `repeat_kv` fallback for torch < 2.5.
- **Iterative Refinement**: `h_(t+1) = h_(t) + α_t · Cell(RMSNorm(h_(t)))`
  applied `refinement_steps=2` times before the byte head; the cell is shared
  (`1024 → 256 → 1024`), `α` is initialised to zero so the loop is the identity
  at initialisation, and each pass is activation-checkpointed.

**Precision.** `torch.bfloat16` CUDA autocast is not available on Tesla T4
(Turing, sm_75) and raises `Current CUDA Device does not support bfloat16`. The
config therefore records `requested_precision: bfloat16` and
`precision: auto`; `resolve_precision()` selects bfloat16 on Ampere+ and float16
with `torch.amp.GradScaler` on T4/V100. Both are 2-byte weights. The resolved
precision on this CPU-only environment is `float32`.

**Three phases, three checkpoints.** Phase 1 is raw next-byte causal LM over real
streamed code / PT-BR / English / math sources and writes `checkpoint_phase1.pt`
(plus `checkpoint_phase1_fp16.pt`); Phase 2 distils conversation and real
Minecraft ReAct trajectories with prompt masking and writes
`checkpoint_phase2.pt`; Phase 3 runs sandbox RL and writes `final_model.pt`.
Checkpoints are `schema=3`: fp32 master weights, AdamW moments, GradScaler,
Python/Torch/CUDA RNG state, the lossless `BytePacker` carry and the dataset
manifest. `initialise_from_phase()` makes a missing predecessor a hard error
instead of silently restarting from random weights.

**Data.** `data/phase22_sources.py` is the single registry of real Hub ids
(`bigcode/the-stack`, `TucanoBR/GigaVerbo` / `HuggingFaceFW/fineweb-2 por_Latn`,
`HuggingFaceFW/fineweb-edu` / `mlfoundations/dclm-baseline-1.0`,
`open-web-math/open-web-math`; Phase 2: `HuggingFaceH4/no_robots`
(**CC BY-NC 4.0, non-commercial**), `TucanoBR/Tucano-SFT`,
`argilla/distilabel-sft-easy`, `m-a-p/CodeFeedback-Filtered-Instruction`,
`TESS-Computer/minecraft-vla-stage3`, `vanyacohen/MET-Bench-Minecraft`).
Streaming only, strict UTF-8 (invalid rows counted and skipped, never
replacement-decoded), and missing textual observations are marked
`[source: no textual observation]` rather than invented.

**RL reward (`training/phase22_reward.py`).** `+1.0` correct final answer,
`−0.5` per `[EDIT]`, `+0.8` verified 1–2-edit fix, `−2.0` wrong answer with no
search when search was required, `+1.5` for `Action: search_ddg("…")` **and**
using its `Observation:`; edits outside `[CURRENT_DRAFT: editable]` cost `−1.0`,
each edit beyond two costs `−0.75`.

## Phase 21: forensic diagnosis of the Phase 19 collapse (historical)

Phase 21 established that the Phase 19 notebook model collapsed to a degenerate
byte distribution (≈81 % of the probability mass on byte 32) rather than
suffering a numerical fault: no NaN/Inf, no dead layers, healthy gradients. The
finding drove three Phase 22 decisions that are visible in the code today —
strict phase ordering (`initialise_from_phase`), de-contaminated evaluation
(`evaluation/final_agent_eval.py` reports `unverified` rather than "clean" when no
training shard is available), and the removal of the clone-based notebook in
favour of `notebook/build_phase22_notebooks.py`.

The Phase 21 **scripts** (`debug/*`, `notebook/train.py`, `phone_model.py`,
`configs/phone_100m.yaml`) were deleted in Phase 23 because they trained and
inspected an architecture that no longer exists. The **evidence** is retained:
`runs/phase21/*.json`, `runs/phase21/*.md`, `runs/phase21/*.png`.

## Phase 23: audit and cleanup (this phase)

- `ARCHITECTURE_AUDIT.md` is the normative description of the code: four levels,
  exact formulas, exact tensor shapes, and objective answers to the four runtime
  questions (latent thinking, stopping rule, self-editing/temporal markers,
  search-versus-memory routing).
- 225 tracked files were removed: 25 model modules, 30 data files (25 modules,
  the `real_edit_pairs.jsonl` fixture and the 11 MB `github_raw/` corpus),
  26 trainers, 41 evaluation harnesses, the 15-file `debug/` forensic package,
  4 demos plus `concept_demo.py`, 14 notebook/Phase 19 files,
  `configs/phone_100m.yaml`, 46 `runs/**/*.pt` checkpoints and 22 obsolete test
  modules. All 46 tracked `runs/**/*.pt` checkpoints were deleted
  (39 belong to deleted classes, 7 are Phase 16–18 foundation weights that no
  surviving code path loads); every JSON/Markdown/PNG report is retained.
  The tree went from 421 to 196 tracked files; `runs/` from 80 MB to 41 MB.
- The regression suite is now `tests/test_foundation.py`,
  `tests/test_memory_model.py`, `tests/test_react_runtime.py` and
  `tests/test_phase22.py`: **92 passed**, one skip (the benchmark artefact test
  skips until `evaluation/phase22_benchmark.py` is run).

## Known limitations and non-claims

- **Memory is not wired into Phase 22 training.** No module under `training/`,
  `data/`, `evaluation/` or `notebook/` constructs a `MemoryModel` or calls
  `retrieve`/`retrieve_differentiable`. Memory reaches a prompt only as
  `[MEM] … = …` prefix bytes from an injected retriever; `ScriptedRetriever` is
  the only shipped implementation and it is test-only.
- **No adaptive halting exists.** `refinement_steps` is a constant, `α` starts at
  zero, and there is no confidence head. The old `ConfidenceHead` /
  `configured_threshold` machinery was deleted with `models/epistemic.py` and
  `models/adaptive.py`.
- **The Phase 3 "search tool" is a stub.** No network call is made; the
  observation is a constant taken from the evaluation case (by default the
  expected answer), so the reward can be satisfied by echoing injected text.
- **No GPU measurement exists.** Parameter counts, shapes and CPU parity are
  measured; VRAM utilisation, T4 throughput and bfloat16 behaviour are not.
- **`MemoryModel` carries unused parameters** (`context_proj` 8,320,
  `person_linker` 32,768 at the default shape) that are constructed, saved and
  never evaluated.

## Verification and commands

```bash
python -m pytest -q                                     # 92 passed
python -m data.phase22_sources --self-test               # registry/converters
python -m training.phase22_reward --self-test            # reward ordering
python -m training.phase22_train --self-test             # CPU plumbing smoke test
python -m evaluation.final_agent_eval --self-test        # scripted oracle scorer
python -m evaluation.phase22_benchmark                   # writes runs/phase22/*.json
python -m notebook.build_phase22_notebooks               # regenerates 01/02/03
python -m training.phase22_train --phase 1 --device cuda # Colab T4 only
```

`--self-test` runs on synthetic/repository bytes; it is plumbing validation and
never a training result.

## Documentation protocol

Update `AGENTS.md` and `QWEN_CHAT_PROMPT.md` by writing the file from Python
(`Path.write_text`), then run `python -m pytest -q` — `test_phase22.py` asserts
that both documents contain exactly one `Phase 22` section, one `Phase 21`
section and the required architecture terms. Never hand-edit generated
notebooks; regenerate them from `notebook/build_phase22_notebooks.py`.
